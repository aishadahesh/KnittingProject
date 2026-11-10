"""A recorded scan's poses, put back on the MuJoCo UR5e: what scripts/replay_session.py shows.

Every picture's sidecar records the camera point the arm was planned to (`target_position`: the
point one tool offset along the flange's +Z, with the flange's attitude) and the subject it was
aimed at. No joint angles are saved, so each picture's arm configuration is solved again here with
the planner's own IK, each solve seeded from the one before - the branch the arm itself stayed on.

The camera drawn is that planned camera point. The lens itself sits a few centimetres off it (see
hand_eye.py), so this shows where the arm was, not a calibrated camera pose.

  load_views     a session's posed pictures (a combined session's come from the scans it lists)
  tool_offset    the tool offset the planner used, from config/params.json
  flange_pose    the flange pose a view's camera point implies
  solve_joints   the arm's joints for every view, chained from Home

No imgui and no viewer here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

import numpy as np
from scipy.spatial.transform import Rotation

import paths

DEFAULT_TOOL_OFFSET = 0.0306


@dataclass
class View:
    image: Path
    session: str
    label: str
    camera: np.ndarray                 # the planned camera point, metres, robot base frame
    rotvec: np.ndarray                 # the flange's attitude
    subject: np.ndarray | None = None  # the point it was aimed at
    footprint: np.ndarray | None = None  # the picture's corners on the fabric (4 x 2), when recorded
    fabric_z: float | None = None
    record: dict = field(default_factory=dict)


def _image_for(sidecar: Path, record: dict) -> Path:
    """The picture beside its sidecar; the stored path is the capturing machine's, so only its name is used."""
    beside = sidecar.with_suffix(".png")
    return beside if beside.is_file() else sidecar.parent / PureWindowsPath(str(record.get("path", ""))).name


def _session_views(folder: Path) -> list[View]:
    views = []
    for sidecar in sorted(folder.glob("*.json")):
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        pose = record.get("target_position") if isinstance(record, dict) else None
        if not pose or len(pose) < 6 or not record.get("path"):
            continue                   # summaries, video frames and hand-held pictures carry no arm pose
        camera = np.asarray(record.get("camera_position") or pose[:3], dtype=float)
        height = record.get("camera_height_above_fabric_m")
        views.append(View(
            image=_image_for(sidecar, record),
            session=folder.name,
            label=str(record.get("capture_view") or record.get("angle") or sidecar.stem),
            camera=camera,
            rotvec=np.asarray(pose[3:6], dtype=float),
            subject=np.asarray(record["capture_subject"], dtype=float) if record.get("capture_subject") else None,
            footprint=np.asarray(record["fabric_footprint"], dtype=float) if record.get("fabric_footprint") else None,
            fabric_z=float(camera[2]) - float(height) if height is not None else None,
            record=record,
        ))
    return views


def load_views(session: Path) -> list[View]:
    """A session's posed pictures in capture order. A combined session (combined.json) stands for the
    scans it lists, found under robot_scans by name."""
    session = Path(session)
    combined = session / "combined.json"
    if combined.is_file():
        entries = json.loads(combined.read_text(encoding="utf-8")).get("sessions", [])
        return [view for entry in entries for view in _session_views(paths.scan_session(entry, session.parent))]
    return _session_views(session)


def tool_offset() -> float:
    """The planner's tool offset (the camera point's distance along the flange's +Z)."""
    try:
        return float(json.loads(paths.PARAMS_JSON.read_text(encoding="utf-8")).get("teleop_tool_offset", DEFAULT_TOOL_OFFSET))
    except (OSError, ValueError):
        return DEFAULT_TOOL_OFFSET


def flange_pose(view: View, offset: float) -> np.ndarray:
    """The flange pose whose camera point, one `offset` along its +Z, is the view's camera."""
    return np.r_[view.camera - Rotation.from_rotvec(view.rotvec).apply([0.0, 0.0, offset]), view.rotvec]


def solve_joints(views: list[View], offset: float, progress=None) -> list[tuple[np.ndarray, str | None]]:
    """The arm's joints for every view, and why not for one that could not be reached (it keeps the
    joints before it). Each solve starts from the last, as the arm moved, so it stays on its branch."""
    import scan_geometry
    import ur5_robot

    q = np.array(ur5_robot.DEFAULT_HOME_JOINTS, dtype=float)
    solved = []
    for index, view in enumerate(views):
        try:
            q, _, _ = scan_geometry.solve_ik(q, flange_pose(view, offset), view.label)
            solved.append((np.array(q, dtype=float), None))
        except scan_geometry.Unreachable as exc:
            solved.append((np.array(q, dtype=float), str(exc)))
        if progress is not None:
            progress(index + 1, len(views))
    return solved
