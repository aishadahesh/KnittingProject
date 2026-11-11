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

import math
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_CAPTURE_SIZE = (1280, 720)
DEFAULT_PREVIEW_WIDTH = 480
# How long capture() will wait for a frame newer than the one it was handed.
FRESH_FRAME_TIMEOUT = 2.0
# Between grabs: roughly 30 frames a second, which is also the rate a recorded video is written at.
GRAB_INTERVAL = 0.033
# The automatic exposure and white balance get this many frames to settle after the camera opens,
# then are pinned for the rest of the session (lock_exposure).
EXPOSURE_SETTLE_FRAMES = 45


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

    def set_focus(self, auto: bool, value: float) -> None:
        """Autofocus on, or off at a fixed lens position. A camera with no focus to set ignores it."""

    def open_settings(self) -> None:
        """Show the driver's own settings window - focus, exposure, white balance as the camera offers them."""
        raise RuntimeError(f"{self.describe()} has no settings window")

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
        self.driver = ""
        self._capture = None

    def open(self) -> None:
        import cv2  # Imported here so the module loads without OpenCV present.

        # Each backend in turn, so a camera the fast one cannot drive still
        # opens through the default.
        capture = None
        for backend in _open_backend_flags():
            candidate = cv2.VideoCapture(self.device_index, backend)
            if candidate.isOpened():
                capture = candidate
                break
            candidate.release()
        if capture is None:
            raise RuntimeError(f"Camera device {self.device_index} could not be opened")
        # Compressed frames: uncompressed 1080p and up crawls over USB 2 at a few frames a second.
        capture.set(cv2.CAP_PROP_FOURCC, float(cv2.VideoWriter_fourcc(*"MJPG")))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.requested_size[0]))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.requested_size[1]))
        # A large driver-side buffer means a read returns a frame from seconds
        # ago, which for a moving robot is a picture of the wrong pose.
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as exc:
            logging.warning("robot_camera.open: %s", exc)
        self.actual_size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.driver = capture.getBackendName()
        self._capture = capture
        # Locked once the automatic metering has settled on the first frames (see read()).
        self._frames_read = 0
        self.exposure_locked = {}

    def lock_exposure(self) -> dict[str, bool]:
        """Hold exposure and white balance where they are now, so every picture of a scan is lit the same.

        Left on automatic, the camera re-metered every frame of a scan: brightness varied almost
        twofold and the red/blue balance by 60 % between neighbouring stations of one sweep, so the
        same stitch was white in one picture and blue-grey in the next - and a reconstruction, which
        can give a point only one colour, came out as mush. The current automatic values are read
        and then pinned. Returns which of the two the driver accepted."""
        import cv2

        taken = {"exposure": False, "white_balance": False}
        if self._capture is None:
            return taken
        try:
            exposure = self._capture.get(cv2.CAP_PROP_EXPOSURE)
            # DirectShow takes 0.25 for manual, V4L2 takes 1; both are tried.
            manual = self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) or self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
            if manual and exposure not in (0.0, -1.0):
                self._capture.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
            taken["exposure"] = bool(manual)
        except Exception as exc:
            logging.warning("robot_camera.lock_exposure: %s", exc)
        try:
            temperature = self._capture.get(cv2.CAP_PROP_WB_TEMPERATURE)
            fixed = self._capture.set(cv2.CAP_PROP_AUTO_WB, 0.0)
            if fixed and temperature > 0:
                self._capture.set(cv2.CAP_PROP_WB_TEMPERATURE, float(temperature))
            taken["white_balance"] = bool(fixed)
        except Exception as exc:
            logging.warning("robot_camera.lock_exposure: %s", exc)
        self.exposure_locked = taken
        return taken

    def set_focus(self, auto: bool, value: float) -> None:
        """Through the driver's UVC focus controls, which a close-up needs: most webcam autofocus
        hunts, or gives up, at under 10 cm. Refused loudly when the driver will not take it."""
        import cv2

        if self._capture is None:
            return
        # Some drivers refuse the autofocus switch yet take a lens position, which turns autofocus off
        # by itself - so only a refused lens position means there is no manual focus to be had.
        self._capture.set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if auto else 0.0)
        if not auto and not self._capture.set(cv2.CAP_PROP_FOCUS, float(value)):
            raise RuntimeError(f"this camera ({self.driver or 'its driver'}) takes no manual focus - it may be a fixed-focus camera")

    def open_settings(self) -> None:
        import cv2

        if self._capture is None or not self._capture.set(cv2.CAP_PROP_SETTINGS, 1.0):
            raise RuntimeError(f"the {self.driver or 'camera'} driver offers no settings window - only DirectShow does")

    def read(self) -> np.ndarray | None:
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        self._frames_read += 1
        if self._frames_read == EXPOSURE_SETTLE_FRAMES:
            self.lock_exposure()
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
            except Exception as exc:
                logging.warning("robot_camera.close: %s", exc)
            self._capture = None

    def describe(self) -> str:
        w, h = self.actual_size if self.actual_size != (0, 0) else self.requested_size
        return f"OpenCV device {self.device_index} ({w}x{h}" + (f", {self.driver})" if self.driver else ")")


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


from rendering import _resolve_k_and_d, distortion_map


class SimCameraBackend(CameraBackend):
    """Simulated wrist camera rendering the 3D world from the robot's live 6-DoF pose."""

    name = "sim"
    is_real = False

    def __init__(
        self,
        width: int | tuple[int, int] = DEFAULT_CAPTURE_SIZE[0],
        height: int = DEFAULT_CAPTURE_SIZE[1],
        pose_provider: Callable[..., Any] | None = None,
        scene_provider: Callable[[], dict[str, Any]] | None = None,
        intrinsics: dict[str, float] | tuple[float, ...] | np.ndarray | None = None,
        dist_coeffs: tuple[float, ...] | list[float] | np.ndarray | None = None,
        *,
        size: tuple[int, int] | None = None,
    ):
        if size is not None:
            self.width, self.height = int(size[0]), int(size[1])
        elif isinstance(width, (tuple, list, np.ndarray)):
            self.width, self.height = int(width[0]), int(width[1])
        else:
            self.width, self.height = int(width), int(height)
        self.size = (self.width, self.height)
        self.pose_provider = pose_provider
        self.scene_provider = scene_provider
        self.intrinsics = intrinsics
        self.dist_coeffs = dist_coeffs
        self._gl_pose: np.ndarray | None = None
        self._focus_auto = True
        self._focus_value = 120.0
        self._ctx = None
        self._renderer = None
        self._ctx_thread_id: int | None = None
        self._loaded_mesh_epoch: int = -1
        self._distort_cache: tuple[Any, np.ndarray, tuple[np.ndarray, np.ndarray]] | None = None

    def open(self) -> None:
        pass

    @staticmethod
    def _current_glfw_window():
        import sys
        glfw = sys.modules.get("glfw")
        if glfw is None:
            return None
        prev_rep = getattr(glfw, "ERROR_REPORTING", "raise")
        try:
            glfw.ERROR_REPORTING = "ignore"
            return glfw.get_current_context()
        except Exception:
            return None
        finally:
            glfw.ERROR_REPORTING = prev_rep

    def _release_gl(self) -> None:
        same_thread = self._ctx_thread_id == threading.get_ident()
        prev_win = self._current_glfw_window() if same_thread else None
        if self._renderer is not None:
            if same_thread:
                try:
                    self._renderer.close()
                except Exception as exc:
                    logging.warning("SimCameraBackend.close renderer: %s", exc)
            self._renderer = None
        if self._ctx is not None:
            if same_thread:
                try:
                    self._ctx.release()
                except Exception as exc:
                    logging.warning("SimCameraBackend.close ctx: %s", exc)
            self._ctx = None
        if prev_win is not None:
            try:
                import glfw
                glfw.make_context_current(prev_win)
            except Exception:
                pass
        self._ctx_thread_id = None
        self._loaded_mesh_epoch = -1

    def close(self) -> None:
        self._release_gl()

    def set_focus(self, auto: bool, value: float) -> None:
        self._focus_auto = bool(auto)
        self._focus_value = float(value)

    def update_gl_frame(self, frame_rgb: np.ndarray, pose: list[float] | np.ndarray) -> None:
        self._gl_pose = np.asarray(pose, dtype=float).reshape(-1)[:6].copy()

    def _current_pose_and_target(self) -> tuple[np.ndarray, float, np.ndarray]:
        if self.pose_provider is not None:
            try:
                res = self.pose_provider()
                if isinstance(res, tuple) and len(res) == 3 and hasattr(res[0], "__len__"):
                    pose, hfov, target = res
                    return (
                        np.asarray(pose, dtype=float).reshape(-1)[:6],
                        float(hfov),
                        np.asarray(target, dtype=float).reshape(-1)[:3],
                    )
                return (
                    np.asarray(res, dtype=float).reshape(-1)[:6],
                    55.0,
                    np.array([-0.45, -0.08, 0.0], dtype=float),
                )
            except Exception:
                pass
        return (
            np.array([-0.45, -0.08, 0.25, math.pi, 0.0, 0.0], dtype=float),
            55.0,
            np.array([-0.45, -0.08, 0.0], dtype=float),
        )

    def read(self) -> np.ndarray | None:
        import cv2
        import moderngl
        import rendering

        prev_win = self._current_glfw_window()

        try:
            tid = threading.get_ident()
            if self._ctx is None or self._renderer is None or self._ctx_thread_id != tid:
                if self._ctx is not None or self._renderer is not None:
                    self._release_gl()
                self._ctx = moderngl.create_standalone_context()
                self._renderer = rendering.MeshRenderer(self._ctx, self.width, self.height)
                self._ctx_thread_id = tid
                self._loaded_mesh_epoch = -1
            elif prev_win is not None:
                self._ctx.__enter__()

            pose, hfov, target = self._current_pose_and_target()
            self._gl_pose = pose.copy()
            scene = self.scene_provider() if self.scene_provider is not None else {}
            if not isinstance(scene, dict):
                scene = {}

            joints = scene.get("joints") if "joints" in scene else scene.get("robot_joints")
            grid_span = scene.get("grid_span")
            fabric_center = scene.get("fabric_center", target)
            fabric_size = scene.get("fabric_size", (0.28, 0.19))
            material_uniforms = scene.get("material_uniforms")
            prepared_meshes = scene.get("prepared_meshes")
            colors = scene.get("colors")
            mesh_epoch = scene.get("mesh_epoch")
            intrinsics = scene.get("intrinsics", self.intrinsics)
            dist_coeffs = scene.get("dist_coeffs", self.dist_coeffs)

            if prepared_meshes is not None and (mesh_epoch is None or int(mesh_epoch) != self._loaded_mesh_epoch):
                self._renderer.set_prepared_meshes(prepared_meshes, colors)
                if mesh_epoch is not None:
                    self._loaded_mesh_epoch = int(mesh_epoch)

            K, D = _resolve_k_and_d(intrinsics, dist_coeffs, self.width, self.height, hfov)
            has_distortion = bool(np.any(np.abs(D) > 1e-7))
            if has_distortion:
                cache_key = (
                    self.width,
                    self.height,
                    tuple(round(float(v), 4) for v in K.ravel()),
                    tuple(round(float(v), 6) for v in D.ravel()),
                )
                if self._distort_cache is None or self._distort_cache[0] != cache_key:
                    K_render, remap_xy = distortion_map(K, D, (self.width, self.height))
                    self._distort_cache = (cache_key, K_render, remap_xy)
                else:
                    _, K_render, remap_xy = self._distort_cache
                render_intrinsics = K_render
            else:
                render_intrinsics = K
                remap_xy = None

            rgb = self._renderer.render_wrist_camera(
                pose,
                joints,
                width=self.width,
                height=self.height,
                fabric_center=fabric_center,
                fabric_size=fabric_size,
                material_uniforms=material_uniforms,
                intrinsics=render_intrinsics,
                dist_coeffs=None,
                grid_span=grid_span,
                hfov_deg=hfov,
            )
            if remap_xy is not None:
                rgb = cv2.remap(rgb, remap_xy[0], remap_xy[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            return rgb
        finally:
            if prev_win is not None:
                try:
                    import glfw
                    glfw.make_context_current(prev_win)
                except Exception:
                    pass

    def describe(self) -> str:
        return f"Simulated wrist camera ({self.size[0]}x{self.size[1]})"


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


def _open_backend_flags():
    """Capture backends to open a camera through, in order - which is not the probing order.

    Probing wants speed, so it uses Media Foundation (_probe_backend_flags). Opening wants the
    camera's controls: on Windows, Media Foundation through OpenCV mostly ignores the UVC focus
    controls, so a close-up's manual focus - and Find best focus - silently does nothing there.
    DirectShow passes them through, at the cost of about a second to open, which is paid once.
    """
    import sys

    import cv2

    if sys.platform.startswith("win"):
        return (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
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
            except Exception as exc:
                logging.warning("robot_camera.__exit__: %s", exc)
        return False


def _windows_device_names() -> list[str]:
    """Friendly names (e.g. "2K HD Camera") in DirectShow's enumeration order.

    OpenCV indices normally line up with this order on Windows, which is the only
    way to tell two same-resolution UVC cameras apart in the picker -- "Camera 0"
    and "Camera 1" are otherwise indistinguishable to the user choosing between them.
    """
    import sys

    if not sys.platform.startswith("win"):
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph

        return list(FilterGraph().get_input_devices())
    except Exception:
        return []


def list_devices(max_devices: int = 10) -> list[dict[str, Any]]:
    """Probe for attached cameras.

    Opening a device is the only reliable way to know it exists, so each index
    is opened and immediately released. Always returns the synthetic entry last
    so there is a selectable option even with nothing plugged in.
    """
    devices: list[dict[str, Any]] = []
    if opencv_available():
        import cv2

        names = _windows_device_names()
        # A device the first (fastest) backend can't enumerate may still open
        # on another; falling back here is what catches those before giving
        # up on the index entirely.
        backend_flags = _probe_backend_flags()
        misses = 0
        with _SuppressOpenCVLogging():
            for index in range(max(0, int(max_devices))):
                found = False
                for backend in backend_flags:
                    capture = None
                    try:
                        capture = cv2.VideoCapture(index, backend)
                        if capture.isOpened():
                            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            name = names[index] if index < len(names) else f"Camera {index}"
                            devices.append({
                                "backend": "opencv",
                                "index": index,
                                "label": f"{name} ({width}x{height})",
                                "is_real": True,
                            })
                            found = True
                    except Exception as exc:
                        logging.warning("robot_camera.list_devices: %s", exc)
                    finally:
                        if capture is not None:
                            try:
                                capture.release()
                            except Exception as exc:
                                logging.warning("robot_camera.list_devices: %s", exc)
                    if found:
                        break
                misses = 0 if found else misses + 1
                # Indices are handed out in order, so a run of consecutive gaps
                # means the end of the list. Probing on to the limit only costs
                # the user a wait for devices that are not there.
                if misses >= 4:
                    break
    devices.append({
        "backend": "sim",
        "index": -1,
        "label": "Simulated wrist camera (3D world)",
        "is_real": False,
    })
    devices.append({
        "backend": "synthetic",
        "index": -1,
        "label": "Synthetic camera (no hardware)",
        "is_real": False,
    })
    return devices


def make_backend(
    spec: dict[str, Any] | None,
    size=DEFAULT_CAPTURE_SIZE,
    pose_provider: Callable[..., Any] | None = None,
    scene_provider: Callable[[], dict[str, Any]] | None = None,
    intrinsics: dict[str, float] | tuple[float, ...] | np.ndarray | None = None,
    dist_coeffs: tuple[float, ...] | list[float] | np.ndarray | None = None,
) -> CameraBackend:
    spec = spec or {}
    backend_name = str(spec.get("backend", ""))
    if backend_name == "opencv":
        return OpenCVCameraBackend(int(spec.get("index", 0)), size=size)
    if backend_name == "sim":
        return SimCameraBackend(
            size=size,
            pose_provider=pose_provider,
            scene_provider=scene_provider,
            intrinsics=intrinsics,
            dist_coeffs=dist_coeffs,
        )
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
        # A focus change, or the driver's settings window, waits here for the grab loop - the only
        # thread that touches the device.
        self._pending_focus: tuple[bool, float] | None = None
        self._pending_settings = False

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
        except Exception as exc:
            logging.warning("robot_camera.stop: %s", exc)
        with self._new_frame:
            self._new_frame.notify_all()
        self.status = "Camera stopped"

    def request_focus(self, auto: bool, value: float = 0.0) -> None:
        """Ask for a focus setting; the grab loop applies it before its next read."""
        with self._frame_lock:
            self._pending_focus = (bool(auto), float(value))

    def request_settings(self) -> None:
        """Ask for the driver's settings window; the grab loop opens it."""
        with self._frame_lock:
            self._pending_settings = True

    def _grab_loop(self) -> None:
        try:
            while not self._stop.is_set():
                with self._frame_lock:
                    focus, self._pending_focus = self._pending_focus, None
                    settings, self._pending_settings = self._pending_settings, False
                if settings:
                    try:
                        self.backend.open_settings()
                        self.status = "Camera settings window opened"
                    except Exception as exc:
                        self.last_error = str(exc)
                        self.status = f"No camera settings window: {exc}"
                if focus is not None:
                    try:
                        self.backend.set_focus(*focus)
                        self.status = "Autofocus on" if focus[0] else f"Focus fixed at {focus[1]:.0f}"
                    except Exception as exc:
                        self.last_error = str(exc)
                        self.status = f"Focus not set: {exc}"
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
                # The loop is the only reader of the device, so it sets the rate for preview and capture alike.
                time.sleep(GRAB_INTERVAL)
        finally:
            try:
                self.backend.close()
            except Exception as exc:
                logging.warning("robot_camera._grab_loop close: %s", exc)

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


def sharpness(image: Image.Image, width: int = 640) -> float:
    """How sharp a frame is: the variance of its Laplacian, at a fixed working width.

    A blurred frame has little second-derivative energy, so the variance collapses; a knit
    in focus is all edges. Measured at one width so a threshold means the same thing
    whatever the capture size.
    """
    grey = image.convert("L")
    if grey.size[0] > width:
        grey = grey.resize((width, max(1, int(round(grey.size[1] * width / grey.size[0])))), Image.BILINEAR)
    pixels = np.asarray(grey, dtype=np.float32)
    if pixels.shape[0] < 3 or pixels.shape[1] < 3:
        return 0.0
    laplacian = (pixels[1:-1, :-2] + pixels[1:-1, 2:] + pixels[:-2, 1:-1] + pixels[2:, 1:-1] - 4.0 * pixels[1:-1, 1:-1])
    return float(laplacian.var())


def image_shift(before: Image.Image, after: Image.Image, width: int = 480) -> np.ndarray:
    """How far the picture slid from `before` to `after`, in pixels of `before`, as (x right, y down).

    Phase correlation: the peak of the normalised cross-power spectrum is the shift, found to a
    fraction of a pixel by fitting a parabola through it. Taken at `width` and scaled back, so
    the answer is in the frame's own pixels whatever the capture size.
    """
    def grey(image):
        small = image.convert("L")
        scale = small.size[0] / float(width)
        small = small.resize((width, max(1, int(round(small.size[1] / scale)))), Image.BILINEAR)
        pixels = np.asarray(small, dtype=np.float64)
        pixels -= pixels.mean()
        window = np.outer(np.hanning(pixels.shape[0]), np.hanning(pixels.shape[1]))
        return pixels * window, scale

    a, scale = grey(before)
    b, _ = grey(after)
    spectrum = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    surface = np.real(np.fft.ifft2(spectrum / np.maximum(np.abs(spectrum), 1e-12)))
    row, col = np.unravel_index(int(np.argmax(surface)), surface.shape)

    def refine(line, index):
        left, centre, right = line[index - 1], line[index], line[(index + 1) % len(line)]
        bend = left - 2.0 * centre + right
        return index + (0.5 * (left - right) / bend if abs(bend) > 1e-12 else 0.0)

    dy, dx = refine(surface[:, col], row), refine(surface[row, :], col)
    dy = dy - surface.shape[0] if dy > surface.shape[0] / 2 else dy
    dx = dx - surface.shape[1] if dx > surface.shape[1] / 2 else dx
    return np.array([dx, dy]) * scale
