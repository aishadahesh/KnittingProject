"""The real UR5 scan run: move, photograph, save, measure, store.

This is the physical counterpart of the loop ``EmbeddedMujocoScanner.update``
runs in simulation, and it visits the same targets from the same
``fabric_scanner`` plan so the two produce comparable datasets. What differs is
what happens at each stop: the simulation renders its camera image, while here a
real camera is asked for a frame taken after the arm actually settled.

The run lives on its own thread, because a real move takes seconds and the UI
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

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import rgb_analysis
import ur5_robot


ROBOT_MODE_REAL = "real_ur5"
# How long to wait for the arm to reach one target before giving up on it.
MOVE_TIMEOUT = 30.0
# Held after arrival so the arm stops ringing before the shutter; a frame taken
# mid-wobble is blurred, and its colour measurement with it.
SETTLE_TIME = 0.35


@dataclass
class ScanProgress:
    """What the UI reads each frame. Plain values, written under the lock."""

    running: bool = False
    paused: bool = False
    finished: bool = False
    target_index: int = 0
    target_count: int = 0
    captured: int = 0
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
    ):
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

        self.session_id = ""
        self.output_dir = Path(output_root)
        self.capture_records: list[dict[str, Any]] = []
        self.analysis_results: dict[str, Any] | None = None

        # Only the real scan stops are visited. approach/retreat/travel points
        # exist to get the arm there safely and are not photographed.
        self.scan_targets = [
            index
            for index, name in enumerate(plan.view_names)
            if _is_scan_view(name)
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
        if self.camera is None or not getattr(self.camera, "running", False):
            issues.append("camera is not running; start the camera preview first")
        if not self.scan_targets:
            issues.append("the scan plan has no capture targets")
        if len(self.plan.poses):
            # The straight line between waypoints is what the controller will
            # actually traverse, so the densified path is what gets checked --
            # a reach violation halfway along a long move is still a collision.
            safe, path_issues = _assess_plan(self.plan)
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
            running=True, paused=False, finished=False, target_index=0, captured=0,
            session_id=self.session_id, output_dir=str(self.output_dir),
            status="Starting scan", error="",
        )
        self._thread = threading.Thread(target=self._run, daemon=True, name="ur5-scan")
        self._thread.start()
        return True

    def pause(self) -> None:
        self.robot.pause()
        self._set(paused=True, status="Paused")

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
        except Exception:
            pass
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
            self._finish_session("finished" if not self._stop.is_set() else "stopped")
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
        try:
            self.robot.move_to_pose(pose)
        except Exception as exc:
            self._set(status=f"Move refused: {exc}", error=str(exc))
            return False

        if not self._wait_until_arrived(pose):
            return False

        if self.dwell > 0.0:
            time.sleep(self.dwell)
        # The arm reports the target reached the moment it is within tolerance,
        # while the tool is still settling. The shutter waits out that ring.
        time.sleep(SETTLE_TIME)
        settled_at = time.monotonic()

        if not self.save_images:
            self._set(status=f"Visited row {row + 1}, col {col + 1}, {view} (not saving)")
            return True

        frame = self.camera.capture(newer_than=settled_at)
        if frame is None:
            self._set(status="No camera frame available; scan stopped", error="camera delivered no frame")
            return False

        record = self._store_capture(frame, pose, target_index, station, row, col, view)
        self._set(
            captured=len(self.capture_records),
            last_image_path=str(record.get("path", "")),
            status=f"Captured row {row + 1}, col {col + 1}, {view}",
        )
        return True

    def _wait_until_arrived(self, pose) -> bool:
        deadline = time.monotonic() + MOVE_TIMEOUT
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return False
            if self.robot.stop_requested:
                self._set(status="Stop latched during move", error="stop latched")
                return False
            if self.robot.at_pose(pose):
                return True
            time.sleep(0.05)
        position_error, _ = self.robot.pose_error(pose)
        self._set(
            status=f"Timed out moving to target (off by {position_error:.3f} m)",
            error="move timed out",
        )
        return False

    def _store_capture(self, frame, pose, target_index, station, row, col, view) -> dict[str, Any]:
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

        stats = rgb_analysis.fabric_rgb_stats(frame.image)
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

    def _finish_session(self, status: str) -> None:
        if self.storage is None or not self.session_id:
            return
        try:
            self.storage.update_robot_session(
                self.session_id,
                status=status,
                capture_count=len(self.capture_records),
            )
            self.storage.flush_json_index()
        except Exception:
            pass

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
                    stats = rgb_analysis.fabric_rgb_stats(saved_image, debug_path=debug_path)
            except Exception:
                continue
            record["rgb"] = [float(v) for v in np.asarray(stats["rgb"], dtype=np.float32)]
            record["fabric_pixel_count"] = int(stats["pixel_count"])
            record["analysis_total_pixels"] = int(stats["total_pixels"])
            record["analysis_mask"] = str(stats["method"])
            if stats.get("debug_path"):
                record["analysis_used_image"] = str(stats["debug_path"])

        result = rgb_analysis.summarize_capture_records(
            self.capture_records,
            self.estimated_cell_colors,
            pattern_signature=self.pattern_signature,
        )
        result["robot_mode"] = ROBOT_MODE_REAL
        result["session_id"] = self.session_id
        result["robot_ip"] = str(self.robot.ip)
        result["comparisons"] = rgb_analysis.compare_cell_colors(result["cells"])

        if save_outputs:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            json_path = self.output_dir / "per_sample_rgb_analysis.json"
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            self._write_swatches(result["cells"])
            result["json_path"] = str(json_path)
            result["used_pixels_dir"] = str(debug_dir)

        self.analysis_results = result
        if self.storage is not None:
            try:
                self.storage.save_analysis(None, result, signature=self.pattern_signature)
            except Exception:
                pass
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


def _is_scan_view(view_name: str) -> bool:
    import fabric_scanner as scanner

    return scanner.is_scan_view(str(view_name))


def _assess_plan(plan) -> tuple[bool, list[str]]:
    import fabric_scanner as scanner

    dense = scanner.densify_plan_for_robot(plan)
    return scanner.assess_plan_safety(dense.mapped_points, max_step=scanner.ROBOT_MAX_CARTESIAN_STEP)
