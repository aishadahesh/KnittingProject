"""Real camera mounted on the UR5 gripper.

Simulation Scan Mode renders its camera images, so the picture the user watches
and the picture that lands in the database are the same array by construction.
A real camera has no such guarantee: preview and capture can easily end up
pulling separate frames from the device and disagreeing about what was scanned.

This module removes that possibility. One grabber thread owns the device and
publishes the newest frame; the live preview is a downscaled copy of that frame
and a capture is the full-resolution copy of a frame from that same stream. The
preview is therefore always the capture, scaled -- never a second read.

Backends are probed at runtime. OpenCV drives real USB/UVC hardware when it is
installed; the synthetic backend stands in when nothing is connected so the mode
stays usable, and every frame it produces is flagged ``is_real=False`` and
watermarked so a synthetic frame can never be mistaken for a real capture.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_CAPTURE_SIZE = (1280, 720)
DEFAULT_PREVIEW_WIDTH = 480
# How long capture() will wait for a frame newer than the one it was handed.
FRESH_FRAME_TIMEOUT = 2.0


@dataclass
class CameraFrame:
    """One frame, with everything needed to describe where it came from."""

    image: Image.Image
    timestamp: float
    index: int
    backend: str
    is_real: bool
    size: tuple[int, int] = (0, 0)

    def __post_init__(self):
        if self.size == (0, 0):
            self.size = tuple(self.image.size)

    def metadata(self) -> dict[str, Any]:
        return {
            "camera_backend": self.backend,
            "camera_is_real": bool(self.is_real),
            "camera_frame_index": int(self.index),
            "camera_frame_timestamp": float(self.timestamp),
            "camera_image_width": int(self.size[0]),
            "camera_image_height": int(self.size[1]),
        }


# ============================================================================
# Backends
# ============================================================================

class CameraBackend:
    """Minimal device interface: open, hand back RGB arrays, close."""

    name = "base"
    is_real = False

    def open(self) -> None:
        raise NotImplementedError

    def read(self) -> np.ndarray | None:
        """Returns one HxWx3 uint8 RGB frame, or None if the read failed."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class OpenCVCameraBackend(CameraBackend):
    """USB / UVC camera through OpenCV. The normal path for real hardware."""

    name = "opencv"
    is_real = True

    def __init__(self, device_index: int = 0, size=DEFAULT_CAPTURE_SIZE):
        self.device_index = int(device_index)
        self.requested_size = (int(size[0]), int(size[1]))
        self.actual_size = (0, 0)
        self._capture = None

    def open(self) -> None:
        import cv2  # Imported here so the module loads without OpenCV present.

        # Each backend in turn, so a camera the fast one cannot drive still
        # opens through the default.
        capture = None
        for backend in _probe_backend_flags():
            candidate = cv2.VideoCapture(self.device_index, backend)
            if candidate.isOpened():
                capture = candidate
                break
            candidate.release()
        if capture is None:
            raise RuntimeError(f"Camera device {self.device_index} could not be opened")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.requested_size[0]))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.requested_size[1]))
        # A large driver-side buffer means a read returns a frame from seconds
        # ago, which for a moving robot is a picture of the wrong pose.
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.actual_size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self._capture = capture

    def read(self) -> np.ndarray | None:
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        # OpenCV hands back BGR; everything downstream (PIL, the RGB analysis,
        # the saved PNG) is RGB, so the swap has to happen here or every stored
        # colour measurement would have red and blue transposed.
        return np.ascontiguousarray(frame[:, :, ::-1])

    def close(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

    def describe(self) -> str:
        w, h = self.actual_size if self.actual_size != (0, 0) else self.requested_size
        return f"OpenCV device {self.device_index} ({w}x{h})"


class SyntheticCameraBackend(CameraBackend):
    """Stand-in frames for when no camera is attached.

    Exists so the UR5 workflow -- connection, motion, capture, storage, analysis
    -- can be exercised end to end without hardware. Frames carry a visible
    watermark and ``is_real=False`` travels with them into the database, so
    synthetic captures stay identifiable after the fact.
    """

    name = "synthetic"
    is_real = False

    def __init__(self, size=DEFAULT_CAPTURE_SIZE, seed: int = 0):
        self.size = (int(size[0]), int(size[1]))
        self._rng = np.random.default_rng(int(seed))
        self._frames = 0

    def open(self) -> None:
        self._frames = 0

    def read(self) -> np.ndarray | None:
        w, h = self.size
        self._frames += 1
        # A muted fabric-like field with gentle drift, so the preview visibly
        # updates and the RGB analysis has a plausible non-uniform surface.
        phase = self._frames * 0.05
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        weave = (
            np.sin(xx / 9.0 + phase) * np.cos(yy / 11.0 - phase) * 18.0
            + np.sin((xx + yy) / 23.0) * 10.0
        )
        base = np.array([132.0, 108.0, 96.0], dtype=np.float32)
        frame = np.clip(base[None, None, :] + weave[:, :, None], 0, 255).astype(np.uint8)
        image = Image.fromarray(frame, "RGB")
        draw = ImageDraw.Draw(image)
        draw.text((12, 12), "SYNTHETIC CAMERA - no device connected", fill=(255, 90, 90))
        return np.asarray(image, dtype=np.uint8)

    def close(self) -> None:
        pass

    def describe(self) -> str:
        return f"Synthetic frames ({self.size[0]}x{self.size[1]})"


def opencv_available() -> bool:
    try:
        import cv2  # noqa: F401
    except Exception:
        return False
    return True


def _probe_backend_flags():
    """Capture backends to try, in order.

    The default backend walks several subsystems in turn (FFMPEG, Orbbec's
    obsensor, then MSMF), and each prints its own complaint straight from C++
    when a device is absent. Naming Media Foundation directly on Windows skips
    that walk: measured here it enumerates in 0.14 s against 0.24-0.66 s for the
    default and 0.94-1.20 s for DirectShow, and it avoids the FFMPEG path.

    The default is kept as a fallback so a camera Media Foundation cannot drive
    is still reachable -- probing wants speed, but opening wants tolerance.
    """
    import sys

    import cv2

    if sys.platform.startswith("win"):
        return (cv2.CAP_MSMF, cv2.CAP_ANY)
    return (cv2.CAP_ANY,)


class _SuppressOpenCVLogging:
    """Silences OpenCV's native log while probing for devices.

    Probing absent camera indices makes OpenCV log an error per miss. Those come
    from its C++ layer directly to stderr, so a try/except cannot stop them and
    the only lever is the library's own log level. A failed probe is an expected
    outcome here, not a fault worth reporting.
    """

    def __enter__(self):
        self._previous = None
        try:
            import cv2.utils.logging as cv_logging

            self._logging = cv_logging
            self._previous = cv_logging.getLogLevel()
            cv_logging.setLogLevel(cv_logging.LOG_LEVEL_SILENT)
        except Exception:
            self._logging = None
        return self

    def __exit__(self, *exc_info):
        if self._logging is not None and self._previous is not None:
            try:
                self._logging.setLogLevel(self._previous)
            except Exception:
                pass
        return False


def list_devices(max_devices: int = 5) -> list[dict[str, Any]]:
    """Probe for attached cameras.

    Opening a device is the only reliable way to know it exists, so each index
    is opened and immediately released. Always returns the synthetic entry last
    so there is a selectable option even with nothing plugged in.
    """
    devices: list[dict[str, Any]] = []
    if opencv_available():
        import cv2

        backend = _probe_backend_flags()[0]
        misses = 0
        with _SuppressOpenCVLogging():
            for index in range(max(0, int(max_devices))):
                capture = None
                try:
                    capture = cv2.VideoCapture(index, backend)
                    if capture.isOpened():
                        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        devices.append({
                            "backend": "opencv",
                            "index": index,
                            "label": f"Camera {index} ({width}x{height})",
                            "is_real": True,
                        })
                        misses = 0
                    else:
                        misses += 1
                except Exception:
                    misses += 1
                finally:
                    if capture is not None:
                        try:
                            capture.release()
                        except Exception:
                            pass
                # Indices are handed out in order, so a couple of consecutive
                # gaps means the end of the list. Probing on to the limit only
                # costs the user a wait for devices that are not there.
                if misses >= 2:
                    break
    devices.append({
        "backend": "synthetic",
        "index": -1,
        "label": "Synthetic camera (no hardware)",
        "is_real": False,
    })
    return devices


def make_backend(spec: dict[str, Any] | None, size=DEFAULT_CAPTURE_SIZE) -> CameraBackend:
    spec = spec or {}
    if str(spec.get("backend", "")) == "opencv":
        return OpenCVCameraBackend(int(spec.get("index", 0)), size=size)
    return SyntheticCameraBackend(size=size)


# ============================================================================
# Camera
# ============================================================================

class RobotCamera:
    """The gripper camera: one grab loop, shared by preview and capture."""

    def __init__(self, backend: CameraBackend | None = None, preview_width: int = DEFAULT_PREVIEW_WIDTH):
        self.backend = backend or SyntheticCameraBackend()
        self.preview_width = int(preview_width)
        self.status = "Camera idle"
        self.last_error = ""
        self._frame: CameraFrame | None = None
        self._frame_lock = threading.Lock()
        self._new_frame = threading.Condition(self._frame_lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_index = 0
        self._read_failures = 0

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_real(self) -> bool:
        return bool(getattr(self.backend, "is_real", False))

    def start(self) -> bool:
        if self.running:
            return True
        self._stop.clear()
        try:
            self.backend.open()
        except Exception as exc:
            self.last_error = str(exc)
            self.status = f"Camera failed to open: {exc}"
            return False
        self._thread = threading.Thread(target=self._grab_loop, daemon=True, name="ur5-camera")
        self._thread.start()
        self.status = f"Camera live: {self.backend.describe()}"
        return True

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        try:
            self.backend.close()
        except Exception:
            pass
        with self._new_frame:
            self._new_frame.notify_all()
        self.status = "Camera stopped"

    def _grab_loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self.backend.read()
            except Exception as exc:
                self.last_error = str(exc)
                raw = None
            if raw is None:
                self._read_failures += 1
                if self._read_failures >= 30:
                    self.status = "Camera stopped delivering frames"
                time.sleep(0.03)
                continue
            self._read_failures = 0
            image = Image.fromarray(np.asarray(raw, dtype=np.uint8), "RGB")
            self._frame_index += 1
            frame = CameraFrame(
                image=image,
                timestamp=time.monotonic(),
                index=self._frame_index,
                backend=str(getattr(self.backend, "name", "unknown")),
                is_real=bool(getattr(self.backend, "is_real", False)),
            )
            with self._new_frame:
                self._frame = frame
                self._new_frame.notify_all()
            # Roughly 30 fps. The loop is the only reader of the device, so it
            # sets the rate for preview and capture alike.
            time.sleep(0.02)

    # -- frames ------------------------------------------------------------

    def latest_frame(self) -> CameraFrame | None:
        with self._frame_lock:
            return self._frame

    def wait_for_frame(self, newer_than: float = 0.0, timeout: float = FRESH_FRAME_TIMEOUT) -> CameraFrame | None:
        """Block until a frame captured after ``newer_than`` arrives.

        Capturing at a scan target must not return an image grabbed while the
        arm was still travelling, so the scan runner records the moment the
        robot settled and asks for a frame newer than that.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._new_frame:
            while True:
                frame = self._frame
                if frame is not None and frame.timestamp > float(newer_than):
                    return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return frame
                self._new_frame.wait(timeout=min(remaining, 0.1))

    def capture(self, newer_than: float = 0.0, timeout: float = FRESH_FRAME_TIMEOUT) -> CameraFrame | None:
        """The full-resolution frame to save. Same stream the preview shows."""
        frame = self.wait_for_frame(newer_than=newer_than, timeout=timeout)
        if frame is None:
            return None
        # Copied so the caller owns an image the grab loop will not touch.
        return CameraFrame(
            image=frame.image.copy(),
            timestamp=frame.timestamp,
            index=frame.index,
            backend=frame.backend,
            is_real=frame.is_real,
            size=frame.size,
        )

    def preview_image(self, max_width: int | None = None) -> Image.Image | None:
        """The live view: the newest capture frame, scaled down to fit the UI.

        Deliberately derived from the same frame ``capture()`` would return, so
        what the user sees is what gets saved.
        """
        frame = self.latest_frame()
        if frame is None:
            return None
        width = int(max_width or self.preview_width)
        image = frame.image
        if image.size[0] <= width:
            return image.copy()
        height = max(1, int(round(image.size[1] * width / max(image.size[0], 1))))
        return image.resize((width, height), Image.BILINEAR)

    def describe(self) -> str:
        return self.backend.describe()
