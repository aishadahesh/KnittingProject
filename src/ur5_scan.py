"""The UR5 scan run: move, photograph, save, measure, store.

This runner executes a ``scan_geometry.CapturePlan`` across both the simulated and
real UR5e robot and camera.

The run lives on its own thread, because a move takes seconds and the UI
has to keep drawing (and its stop button has to keep working) throughout. That
has two consequences the code is shaped around:

  * Nothing here touches the live AppState. The pattern signature and settings
    are snapshotted on the UI thread when the run starts and passed in, so this
    thread never walks state the UI is mutating.
  * Every pass rechecks the safety gate, so a stop, a pause or a protective
    stop raised at the pendant takes effect at the next step rather than after
    the run finishes.
"""

from __future__ import annotations

import logging
import json
import math
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import fabric_detector
import free_capture
import robot_camera
import scan_geometry
import ur5_robot
from persistence import write_json


ROBOT_MODE_REAL = "real_ur5"
# How long to wait for the arm to reach one target before giving up on it.
MOVE_TIMEOUT = 30.0
# Held after arrival so the arm stops ringing before the shutter; a frame taken
# mid-wobble is blurred, and its colour measurement with it.
SETTLE_TIME = 0.35
# Between a blurred frame and the one taken to replace it: long enough for a hunting
# autofocus to move, short enough not to matter over a run.
RETAKE_WAIT = 0.4
VIDEO_POLL = 0.02
VIDEO_WINDOW = 0.3
VIDEO_FILE = "video.mp4"
VIDEO_STILLS = "video_stills.json"
VIDEO_PREFIX = "video_"
RUN_LOG = "run_log.txt"
CONTACT_SHEET = "contact_sheet.jpg"
QUALITY_TABLE = "captures.csv"
SHEET_THUMB_WIDTH = 360
SHEET_COLUMNS = 4
# What the quality table lists, in order: enough to pick the best image of a patch at a glance.
QUALITY_COLUMNS = ("target_index", "capture_view", "row", "col", "camera_height_above_fabric_m", "tilt_from_vertical_deg",
                   "fabric_mm_per_pixel", "fabric_across_frame_mm", "sharpness", "retakes", "fabric_coverage",
                   "fabric_centre_offset", "mostly_background", "blurred", "moved_to_nearest_safe", "path")


class VideoSampler:
    """Picks the pictures out of the camera's stream, live or from a saved video.

    The scan photographs only where it stops, which is a few dozen pictures; a splat and the pose
    solve want hundreds with steady overlap. Every frame fed in joins a window of VIDEO_WINDOW
    seconds, and the sharpest frame of each window is considered. When the run says the arm is
    standing (`still`), the window closes there and its frame is the station's picture, kept
    whatever its sharpness: how sharp a frame can be depends on the yarn under it, a white row
    scoring a fraction of a dark one, so no fixed threshold tells blur from pale cloth. A frame
    taken on the move is kept only if it is nearly as sharp as the last standing one, which showed
    the same cloth - the comparison free_capture makes against the camera's own recent best. Every
    kept frame must show the fabric and have moved on from the recent kept frames. A kept frame is
    a picture and a sidecar in the session folder: the pose solve reads every sidecar there.

    Live, `record` keeps the whole stream as VIDEO_FILE beside the pictures, with the standing
    frames' indices in VIDEO_STILLS, so the same choice can be made again from the file with
    different rules (extract_from_video) without flying the arm again.
    """

    def __init__(self, folder: Path, fabric_check=None, min_coverage: float = 0.0, on_kept=None):
        self.folder = Path(folder)
        self.fabric_check = fabric_check
        self.min_coverage = float(min_coverage)
        self.on_kept = on_kept
        self.kept = 0
        self.rejected = Counter()
        self.error = ""
        self.stills: list[int] = []
        self._recent_features = deque(maxlen=free_capture.RECENT_TO_MATCH)
        self._standing_sharpness = 0.0
        self._window: list = []

    def feed(self, frame, standing: bool = False) -> None:
        """One more frame of the stream, in order; `standing` closes the window on it as a station's picture."""
        self._window.append((robot_camera.sharpness(frame.image, free_capture.BURST_SHARPNESS_WIDTH), frame))
        if standing:
            self.stills.append(frame.index)
        if standing or frame.timestamp - self._window[0][1].timestamp >= VIDEO_WINDOW:
            best = max(self._window, key=lambda entry: entry[0])[1]
            self._window = []
            try:
                self._consider(best, standing)
            except Exception as exc:
                self.error = f"Video frame skipped: {exc}"

    def _consider(self, frame, standing: bool) -> None:
        sharpness = robot_camera.sharpness(frame.image)
        if standing:
            self._standing_sharpness = sharpness
        elif sharpness < max(free_capture.MIN_SHARPNESS, free_capture.BLUR_SHARE * self._standing_sharpness):
            self.rejected["blurred"] += 1
            return
        if self.fabric_check is not None and self.fabric_check(frame.image)[0] < self.min_coverage:
            self.rejected["background"] += 1
            return
        found = free_capture.features(frame.image)
        for earlier in self._recent_features:
            homography, _ = free_capture.match(earlier, found)
            if homography is not None and free_capture.corner_shift(homography, found[2]) < free_capture.DUPLICATE_SHIFT:
                self.rejected["same view"] += 1
                return
        self._recent_features.append(found)
        self.kept += 1
        path = self.folder / f"{VIDEO_PREFIX}{self.kept:04d}.png"
        frame.image.save(path)
        record = {"path": str(path), "video_frame": True, "capture_view": f"video {self.kept}", "standing": standing,
                  "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "sharpness": sharpness, **frame.metadata()}
        write_json(path.with_suffix('.json'), record, indent=2)
        if self.on_kept is not None:
            self.on_kept(self.kept)


class LiveVideo:
    """Feeds the camera's stream to a VideoSampler on its own thread, and records it to a file on the way."""

    def __init__(self, camera, sampler: VideoSampler):
        self.camera = camera
        self.sampler = sampler
        self._still_asked = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ur5-video")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Joined, not abandoned: a thread left inside OpenCV when the app exits aborts the process."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def still(self) -> None:
        self._still_asked.set()

    def _run(self) -> None:
        import cv2

        writer, last_index, fps = None, -1, round(1.0 / self.camera.backend.grab_interval)
        try:
            while not self._stop.is_set():
                frame = self.camera.latest_frame()
                if frame is None or frame.index == last_index:
                    time.sleep(VIDEO_POLL)
                    continue
                last_index = frame.index
                if writer is None:
                    writer = cv2.VideoWriter(str(self.sampler.folder / VIDEO_FILE), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame.image.size)
                writer.write(cv2.cvtColor(np.asarray(frame.image), cv2.COLOR_RGB2BGR))
                standing = self._still_asked.is_set()
                self._still_asked.clear()
                self.sampler.feed(frame, standing)
        finally:
            if writer is not None:
                writer.release()
            write_json(self.sampler.folder / VIDEO_STILLS, self.sampler.stills, indent=None)


def extract_from_video(folder: Path, fabric_check=None, min_coverage: float = 0.0, on_kept=None) -> VideoSampler:
    """Choose the pictures again from the session's recorded video, replacing the ones chosen live."""
    import cv2

    folder = Path(folder)
    stills = set(json.loads((folder / VIDEO_STILLS).read_text())) if (folder / VIDEO_STILLS).is_file() else set()
    for stale in folder.glob(f"{VIDEO_PREFIX}*"):
        stale.unlink()
    sampler = VideoSampler(folder, fabric_check, min_coverage, on_kept)
    video = cv2.VideoCapture(str(folder / VIDEO_FILE))
    if not video.isOpened():
        raise RuntimeError(f"No readable {VIDEO_FILE} in {folder.name}")
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    index = 0
    try:
        while True:
            ok, pixels = video.read()
            if not ok:
                break
            index += 1
            image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB), "RGB")
            sampler.feed(robot_camera.CameraFrame(image, index / fps, index, "video file", True), index in stills)
    finally:
        video.release()
    sampler.stills = sorted(stills)
    return sampler


@dataclass
class ScanProgress:
    """What the UI reads each frame. Plain values, written under the lock."""

    running: bool = False
    paused: bool = False
    finished: bool = False
    target_index: int = 0
    target_count: int = 0
    captured: int = 0
    video_frames: int = 0
    status: str = "Idle"
    error: str = ""
    session_id: str = ""
    output_dir: str = ""
    last_image_path: str = ""

    @property
    def percent(self) -> float:
        if self.target_count <= 0:
            return 0.0
        return 100.0 * min(self.target_index, self.target_count) / self.target_count


class RealScanRunner:
    """Drives one real-robot scan from start to finish."""

    def __init__(
        self,
        robot: ur5_robot.UR5Robot,
        camera,
        plan,
        storage=None,
        *,
        output_root: str | Path = "robot_scans",
        pattern_signature: str = "",
        settings_snapshot: dict[str, Any] | None = None,
        estimated_cell_colors=None,
        lighting_condition: str = "",
        dwell: float = 0.0,
        save_images: bool = True,
        min_sharpness: float = 0.0,
        max_retakes: int = 2,
        fabric_check=None,
        min_coverage: float = 0.8,
        video: bool = False,
    ):
        self.video = bool(video)
        self.video_feed: LiveVideo | None = None
        self.robot = robot
        self.camera = camera
        self.plan = plan
        self.storage = storage
        self.pattern_signature = str(pattern_signature)
        self.settings_snapshot = dict(settings_snapshot or {})
        self.estimated_cell_colors = list(estimated_cell_colors or [])
        self.lighting_condition = str(lighting_condition)
        self.dwell = max(0.0, float(dwell))
        self.save_images = bool(save_images)
        # A frame below `min_sharpness` is retaken up to `max_retakes` times and the sharpest
        # kept. `fabric_check` maps an image to (coverage, mask); a frame under `min_coverage`
        # is kept, but flagged as mostly background and left out of the colour analysis.
        self.min_sharpness = max(0.0, float(min_sharpness))
        self.max_retakes = max(0, int(max_retakes))
        self.fabric_check = fabric_check
        self.min_coverage = float(min_coverage)

        self.session_id = ""
        self.output_dir = Path(output_root)
        self.capture_records: list[dict[str, Any]] = []
        self.analysis_results: dict[str, Any] | None = None

        # Only the real scan stops are visited. approach/retreat/travel points
        # exist to get the arm there safely and are not photographed.
        self.scan_targets = [
            index
            for index, name in enumerate(plan.view_names)
            if scan_geometry.is_scan_view(str(name))
        ]

        self._progress = ScanProgress(target_count=len(self.scan_targets))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- status ------------------------------------------------------------

    @property
    def progress(self) -> ScanProgress:
        with self._lock:
            return ScanProgress(**vars(self._progress))

    def _set(self, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self._progress, key, value)
        # Every status line, timestamped, beside the images: the panel shows only the latest,
        # and a run that stops mid-way has to be able to say afterwards why.
        if self.session_id and ("status" in fields or "error" in fields):
            try:
                with (Path(self.output_dir) / RUN_LOG).open("a", encoding="utf-8") as log:
                    log.write(f"{time.strftime('%H:%M:%S')}  {fields.get('status', '')}"
                              + (f"  [error: {fields['error']}]" if fields.get("error") else "") + "\n")
            except Exception as exc:
                logging.warning("ur5_scan._set: %s", exc)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- lifecycle ---------------------------------------------------------

    def preflight(self) -> tuple[bool, list[str]]:
        """Everything checked before the arm is allowed to move.

        Run as one block so the user gets the full list of what is wrong,
        rather than fixing one problem only to meet the next.
        """
        issues: list[str] = []
        ready, robot_issues = self.robot.readiness()
        if not ready:
            issues.extend(robot_issues)
        # Only when the run would actually take a picture. `save_images` off makes this a
        # motion run - every move, every wait, no shutter - and refusing to start it for
        # want of a camera it never touches is refusing the one thing it can still do.
        if self.save_images and (self.camera is None or not getattr(self.camera, "running", False)):
            issues.append("camera is not running; start the camera preview first, or turn off Save captured images to run the motion alone")
        if not self.scan_targets:
            issues.append("the scan plan has no capture targets")
        if len(self.plan.poses):
            # Each plan kind knows how to judge its own motion, and they judge it
            # differently: a fabric plan bounds its Cartesian steps, a capture plan reports
            # what its joint solve already found.
            safe, path_issues = self.plan.assess()
            if not safe:
                issues.extend(path_issues)
        return len(issues) == 0, issues

    def start(self) -> bool:
        if self.running:
            return False
        ok, issues = self.preflight()
        if not ok:
            self._set(status="Cannot start: " + "; ".join(issues), error="; ".join(issues))
            return False

        self.session_id = time.strftime("ur5_%Y%m%d_%H%M%S")
        self.output_dir = Path(self.output_dir) / self.session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.capture_records = []
        self.analysis_results = None
        self.video_feed = None
        self._stop.clear()

        if self.storage is not None:
            try:
                self.storage.start_robot_session({
                    "session_id": self.session_id,
                    "robot_mode": ROBOT_MODE_REAL,
                    "robot_ip": self.robot.ip,
                    "transport": str(getattr(self.robot.transport, "name", "")),
                    "is_real": bool(self.robot.is_real),
                    "camera_backend": str(getattr(self.camera, "describe", lambda: "")()),
                    "camera_is_real": bool(getattr(self.camera, "is_real", False)),
                    "pattern_signature": self.pattern_signature,
                    "settings": self.settings_snapshot,
                    "status": "running",
                })
            except Exception as exc:
                self._set(error=f"Could not open a database session: {exc}")

        self._set(
            running=True, paused=False, finished=False, target_index=0, captured=0, video_frames=0,
            session_id=self.session_id, output_dir=str(self.output_dir),
            status="Starting scan", error="",
        )
        if self.video and self.save_images:
            self.video_feed = LiveVideo(self.camera, VideoSampler(self.output_dir, self.fabric_check, self.min_coverage,
                                                              lambda kept: self._set(video_frames=kept)))
            self.video_feed.start()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ur5-scan")
        self._thread.start()
        return True

    def pause(self) -> None:
        self.robot.pause()
        self._set(paused=True, status=self.robot.status, error=self.robot.last_error if self.robot.stop_requested else "")

    def resume(self) -> bool:
        if not self.robot.resume():
            self._set(status=self.robot.status)
            return False
        self._set(paused=False, status="Continuing scan")
        return True

    def stop(self) -> None:
        """Ends the run and brings the arm to a controlled halt."""
        self._stop.set()
        try:
            self.robot.transport.stop(2.0)
        except Exception as exc:
            self.robot.stop_requested = True
            self._set(status=f"Stop failed: {exc}", error=str(exc))
            return
        self._set(status="Stopping scan")

    def emergency_stop(self) -> None:
        """Halts the arm immediately and abandons the run."""
        self._stop.set()
        self.robot.emergency_stop()
        self._set(running=False, paused=False, status="EMERGENCY STOP", error="Emergency stop pressed")
        self._finish_session("stopped")

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    # -- the run -----------------------------------------------------------

    def _run(self) -> None:
        try:
            for position, target_index in enumerate(self.scan_targets):
                if self._stop.is_set():
                    self._set(status=f"Scan stopped at {position}/{len(self.scan_targets)}")
                    break
                self._set(target_index=position)

                while self.robot.paused and not self._stop.is_set():
                    self._set(paused=True, status=f"Paused at {position + 1}/{len(self.scan_targets)}")
                    time.sleep(0.1)
                if self._stop.is_set():
                    break
                self._set(paused=False)

                if not self._visit_target(position, target_index):
                    break
            else:
                self._set(status=f"Scan finished | captured {self.progress.captured} images")
                self._set(running=False, finished=True)
                self._finish_session("finished")
                return

            # Ended early: by a stop the user asked for - the Stop button or the emergency stop,
            # which sets _stop - or because a visit failed on its own, which is recorded as a
            # failure with its reason rather than as a run that finished.
            reason = self.progress.error or self.progress.status
            self._set(running=False, finished=True)
            self._finish_session("stopped" if self._stop.is_set() else f"failed: {reason}")
        except Exception as exc:
            self._set(running=False, finished=True, error=str(exc), status=f"Scan failed: {exc}")
            self._finish_session("failed")

    def _visit_target(self, position: int, target_index: int) -> bool:
        """Moves to one target and captures it. False ends the run."""
        pose = np.asarray(self.plan.poses[target_index], dtype=float)
        station = int(self.plan.station_ids[target_index])
        row, col = self.plan.station_cells[station]
        view = str(self.plan.view_names[target_index])

        self._set(status=f"Moving to row {row + 1}, col {col + 1}, {view} ({position + 1}/{len(self.scan_targets)})")
        # A plan that solved its own joints is flown in joints, so the motion is the one
        # Check collision-checked. A fabric plan has only poses, and moves to those.
        configs = getattr(self.plan, "configs", None)
        try:
            if configs is None:
                self.robot.move_to_pose(pose)
                arrived = lambda: self.robot.at_pose(pose)
                shortfall = lambda: f"{self.robot.pose_error(pose)[0]:.3f} m"
                timeout = MOVE_TIMEOUT
            else:
                joints = configs[target_index]
                timeout = 2.0 * self.robot.transport.ramp_for(joints) + 5.0
                self.robot.move_to_joints(joints)
                arrived = lambda: self.robot.at_joints(joints)
                shortfall = lambda: f"{math.degrees(self.robot.joint_error(joints)):.1f} deg"
        except Exception as exc:
            self._set(status=f"Move refused: {exc}", error=str(exc))
            return False

        if not self._wait_until_arrived(arrived, shortfall, timeout):
            return False

        if self.dwell > 0.0:
            time.sleep(self.dwell)
        # The arm reports the target reached the moment it is within tolerance,
        # while the tool is still settling. The shutter waits out that ring.
        time.sleep(SETTLE_TIME)
        settled_at = time.monotonic()
        if self.video_feed is not None:
            self.video_feed.still()

        if not self.save_images:
            self._set(status=f"Visited row {row + 1}, col {col + 1}, {view} (not saving)")
            return True

        frame, sharp, retakes = self._sharpest_frame(settled_at)
        if frame is None:
            self._set(status="No camera frame available; scan stopped", error="camera delivered no frame")
            return False

        record = self._store_capture(frame, pose, target_index, station, row, col, view,
                                     {"sharpness": sharp, "retakes": retakes,
                                      "blurred": bool(self.min_sharpness and sharp < self.min_sharpness)})
        coverage = record.get("fabric_coverage")
        self._set(
            captured=len(self.capture_records),
            last_image_path=str(record.get("path", "")),
            status=(f"Captured row {row + 1}, col {col + 1}, {view} - sharpness {sharp:.0f}"
                    + (f" after {retakes} retake(s)" if retakes else "")
                    + ("" if coverage is None else f", {coverage * 100:.0f}% fabric")
                    + (" - MOSTLY BACKGROUND" if record.get("mostly_background") else "")),
        )
        return True

    def _sharpest_frame(self, since: float):
        """A frame taken after `since`, retaken while it is blurred - the sharpest one kept."""
        best, best_score, taken = None, -1.0, 0
        for taken in range(self.max_retakes + 1):
            frame = self.camera.capture(newer_than=since)
            if frame is None:
                break
            score = robot_camera.sharpness(frame.image)
            if score > best_score:
                best, best_score = frame, score
            if score >= self.min_sharpness or self._stop.is_set():
                break
            time.sleep(RETAKE_WAIT)
            since = time.monotonic()
        return best, float(max(best_score, 0.0)), taken

    def _wait_until_arrived(self, arrived, shortfall, timeout: float) -> bool:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return False
            state = self.robot.state
            # The controller stopping the arm is not a slow move; waiting out the timeout
            # on it is what made a stopped run look stuck.
            if self.robot.stop_requested or state.protective_stopped or state.emergency_stopped:
                kind = "Emergency" if state.emergency_stopped else "Protective" if state.protective_stopped else "Latched"
                joints = ", ".join(f"{math.degrees(value):.0f}" for value in state.joints)
                tcp = ", ".join(f"{value:.3f}" for value in state.tcp_pose)
                self._set(status=f"{kind} stop during move (off by {shortfall()}) - {state.summary()}, joints deg [{joints}], tcp [{tcp}]; the pendant log says why",
                          error="stop latched" if kind == "Latched" else f"{kind.lower()} stop")
                return False
            if arrived():
                return True
            time.sleep(0.05)
        self._set(status=f"Timed out moving to target (off by {shortfall()})", error="move timed out")
        return False

    def _store_capture(self, frame, pose, target_index, station, row, col, view, quality=None) -> dict[str, Any]:
        """Saves the image and writes its metadata and colour measurement."""
        stem = (
            f"ur5_{target_index + 1:04d}_row_{row + 1:02d}_col_{col + 1:02d}"
            f"_station_{station + 1:03d}_{view.replace(' ', '_')}"
        )
        path = self.output_dir / f"{stem}.png"
        try:
            frame.image.save(path)
        except Exception as exc:
            self._set(error=f"Could not save capture: {exc}")

        stats = fabric_detector.fabric_rgb_stats(frame.image)
        actual_pose = [float(v) for v in self.robot.state.tcp_pose]
        record = {
            "row": int(row),
            "col": int(col),
            "station": int(station),
            "target_index": int(target_index),
            "angle": view,
            "path": str(path),
            "rgb": [float(v) for v in np.asarray(stats["rgb"], dtype=np.float32)],
            "fabric_pixel_count": int(stats["pixel_count"]),
            "analysis_total_pixels": int(stats["total_pixels"]),
            "analysis_mask": str(stats["method"]),
            # Real-robot provenance.
            "robot_mode": ROBOT_MODE_REAL,
            "session_id": self.session_id,
            "robot_ip": str(self.robot.ip),
            "robot_transport": str(getattr(self.robot.transport, "name", "")),
            "robot_is_real": bool(self.robot.is_real),
            "target_position": [float(v) for v in pose[:6]],
            "actual_tcp_pose": actual_pose,
            "lighting_condition": self.lighting_condition,
            "capture_mode": "real_camera",
            "speed_limit": float(self.robot.speed_limit),
        }
        record.update(frame.metadata())
        record.update(self.plan.capture_metadata(target_index, frame.image.size[0]))
        record.update(quality or {})
        if self.fabric_check is not None:
            try:
                coverage, mask = self.fabric_check(frame.image)
                mask_path = self.output_dir / f"{stem}_fabric_mask.png"
                Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L").save(mask_path)
                record.update({"fabric_coverage": float(coverage), "fabric_mask": str(mask_path),
                               "mostly_background": bool(coverage < self.min_coverage)})
                if mask.any():
                    rows, cols = np.nonzero(mask)
                    record["fabric_centre_offset"] = [float(cols.mean() / mask.shape[1] - 0.5), float(rows.mean() / mask.shape[0] - 0.5)]
                detected = fabric_detector.annotate(frame.image, mask, self._caption(record))
                detected.save(self.output_dir / f"{stem}_detected.jpg", quality=88)
                record["detected_image"] = str(self.output_dir / f"{stem}_detected.jpg")
            except Exception as exc:
                record["fabric_check_error"] = str(exc)
        try:
            write_json(self.output_dir / f'{stem}.json', record, indent=2, default=str)
        except Exception as exc:
            self._set(error=f"Could not write the capture's details: {exc}")
        self.capture_records.append(record)

        if self.storage is not None:
            try:
                self.storage.record_capture(
                    None,
                    record,
                    capture_settings=self.settings_snapshot,
                    signature=self.pattern_signature,
                )
            except Exception as exc:
                self._set(error=f"Database write failed: {exc}")
        return record

    @staticmethod
    def _caption(record) -> list[str]:
        """The details a capture is judged by, as the lines written onto its detection image."""
        def number(key, fmt):
            value = record.get(key)
            return "-" if value is None else fmt.format(value)
        return [f"{record.get('capture_view', record.get('angle', ''))}  row {int(record.get('row', 0)) + 1} col {int(record.get('col', 0)) + 1}",
                f"height {number('camera_height_above_fabric_m', '{:.3f}')} m  tilt {number('tilt_from_vertical_deg', '{:.0f}')} deg",
                f"{number('fabric_mm_per_pixel', '{:.3f}')} mm/px  sharp {number('sharpness', '{:.0f}')}  fabric {number('fabric_coverage', '{:.0%}')}"]

    def _write_session_summary(self) -> None:
        """A contact sheet of every capture with its details, and the same details as a table."""
        if not self.capture_records or not self.session_id:
            return
        try:
            import csv

            with (self.output_dir / QUALITY_TABLE).open("w", newline="", encoding="utf-8") as table:
                writer = csv.writer(table)
                writer.writerow(QUALITY_COLUMNS)
                for record in self.capture_records:
                    writer.writerow([record.get(column, "") for column in QUALITY_COLUMNS])

            thumbs = []
            for record in self.capture_records:
                source = record.get("detected_image") or record.get("path")
                with Image.open(source) as image:
                    thumb = image.convert("RGB")
                thumb = thumb.resize((SHEET_THUMB_WIDTH, max(1, int(thumb.size[1] * SHEET_THUMB_WIDTH / thumb.size[0]))), Image.BILINEAR)
                if not record.get("detected_image"):
                    thumb = fabric_detector.annotate(thumb, np.ones((1, 1), dtype=bool), self._caption(record))
                thumbs.append(thumb)
            height = max(thumb.size[1] for thumb in thumbs)
            rows = (len(thumbs) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
            sheet = Image.new("RGB", (SHEET_COLUMNS * SHEET_THUMB_WIDTH, rows * height), (12, 12, 16))
            for index, thumb in enumerate(thumbs):
                sheet.paste(thumb, ((index % SHEET_COLUMNS) * SHEET_THUMB_WIDTH, (index // SHEET_COLUMNS) * height))
            sheet.save(self.output_dir / CONTACT_SHEET, quality=85)
        except Exception as exc:
            self._set(error=f"Could not write the session summary: {exc}")

    def _finish_session(self, status: str) -> None:
        if self.video_feed is not None:
            self.video_feed.stop()
            sampler = self.video_feed.sampler
            self._set(status=f"{self.progress.status} | video frames kept {sampler.kept}, rejected {dict(sampler.rejected)}")
        self._write_session_summary()
        if self.storage is None or not self.session_id:
            return
        try:
            self.storage.update_robot_session(
                self.session_id,
                status=status,
                capture_count=len(self.capture_records),
            )
            self.storage.flush_json_index()
        except Exception as exc:
            logging.warning("ur5_scan._finish_session: %s", exc)

    # -- analysis ----------------------------------------------------------

    def analyze(self, save_outputs: bool = True) -> dict[str, Any]:
        """Re-measures every saved image and stores the per-cell RGB result.

        Measuring the file rather than the in-memory frame means the analysis
        describes exactly the image that is in the dataset, PNG round-trip
        included -- the same guarantee the simulation analysis makes.
        """
        if not self.capture_records:
            self.analysis_results = {"cells": [], "summary": "No captured images to analyze."}
            return self.analysis_results

        debug_dir = self.output_dir / "analysis_used_pixels"
        for record in self.capture_records:
            path = Path(str(record.get("path", "")))
            if not path.exists():
                continue
            try:
                debug_path = debug_dir / f"{path.stem}_used_pixels.png" if save_outputs else None
                with Image.open(path) as saved_image:
                    stats = fabric_detector.fabric_rgb_stats(saved_image, debug_path=debug_path)
            except Exception:
                continue
            record["rgb"] = [float(v) for v in np.asarray(stats["rgb"], dtype=np.float32)]
            record["fabric_pixel_count"] = int(stats["pixel_count"])
            record["analysis_total_pixels"] = int(stats["total_pixels"])
            record["analysis_mask"] = str(stats["method"])
            if stats.get("debug_path"):
                record["analysis_used_image"] = str(stats["debug_path"])

        # A frame that is mostly table would average the table's colour into the cloth's.
        result = fabric_detector.summarize_capture_records(
            [record for record in self.capture_records if not record.get("mostly_background")],
            self.estimated_cell_colors,
            pattern_signature=self.pattern_signature,
        )
        result["robot_mode"] = ROBOT_MODE_REAL
        result["session_id"] = self.session_id
        result["robot_ip"] = str(self.robot.ip)
        result["comparisons"] = fabric_detector.compare_cell_colors(result["cells"])

        if save_outputs:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            json_path = self.output_dir / "per_sample_rgb_analysis.json"
            write_json(json_path, result, indent=2)
            self._write_swatches(result["cells"])
            result["json_path"] = str(json_path)
            result["used_pixels_dir"] = str(debug_dir)

        self.analysis_results = result
        if self.storage is not None:
            try:
                self.storage.save_analysis(None, result, signature=self.pattern_signature)
            except Exception as exc:
                logging.warning("ur5_scan.analyze: %s", exc)
        return result

    def _write_swatches(self, cells) -> None:
        """Flat colour tiles per cell and per angle, matching the sim outputs."""
        swatch_w, swatch_h = 72, 54
        for cell in cells:
            row = int(cell["row"])
            col = int(cell["col"])
            color = tuple(int(np.clip(v, 0, 255)) for v in cell["overall_rgb"])
            Image.new("RGB", (swatch_w, swatch_h), color).save(
                self.output_dir / f"avg_rgb_row_{row + 1:02d}_col_{col + 1:02d}.png"
            )
            for angle_result in cell["angles"]:
                angle_name = str(angle_result["angle"]).replace(" ", "_").replace("/", "_")
                angle_color = tuple(int(np.clip(v, 0, 255)) for v in angle_result["rgb"])
                Image.new("RGB", (swatch_w, swatch_h), angle_color).save(
                    self.output_dir / f"avg_rgb_row_{row + 1:02d}_col_{col + 1:02d}_{angle_name}.png"
                )


