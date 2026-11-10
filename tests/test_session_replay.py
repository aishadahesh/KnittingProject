"""session_replay: a recorded scan's pictures and poses, and the arm's joints solved back from them."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import session_replay  # noqa: E402


def _capture(folder: Path, name: str, camera, rotvec=(np.pi, 0.0, 0.0), **extra):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.png").write_bytes(b"png")
    record = {"path": rf"C:\Users\someone\robot_scans\{folder.name}\{name}.png", "target_position": [*camera, *rotvec],
              "camera_position": list(camera), "capture_view": name, **extra}
    (folder / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")


def test_views_are_the_posed_pictures_in_order_and_a_combined_scan_s_come_from_its_scans(tmp_path):
    scans = tmp_path / "robot_scans"
    _capture(scans / "ur5_a", "ur5_0002", (0.1, 0.6, 0.1), camera_height_above_fabric_m=0.04)
    _capture(scans / "ur5_a", "ur5_0001", (0.1, 0.6, 0.12))
    (scans / "ur5_a" / "captures_summary.json").write_text(json.dumps({"not": "a capture"}))
    _capture(scans / "ur5_b", "ur5_0001", (0.2, 0.6, 0.1))
    (scans / "ur5_a_rescan").mkdir()
    (scans / "ur5_a_rescan" / "combined.json").write_text(json.dumps({"sessions": [r"D:\elsewhere\ur5_a", "ur5_b"]}))

    views = session_replay.load_views(scans / "ur5_a")
    assert [v.label for v in views] == ["ur5_0001", "ur5_0002"]
    assert views[0].image == scans / "ur5_a" / "ur5_0001.png"
    assert views[1].fabric_z == pytest.approx(0.06)
    combined = session_replay.load_views(scans / "ur5_a_rescan")
    assert [(v.session, v.label) for v in combined] == [("ur5_a", "ur5_0001"), ("ur5_a", "ur5_0002"), ("ur5_b", "ur5_0001")]


def test_the_flange_stands_one_tool_offset_back_along_its_axis():
    view = session_replay.View(image=Path("x.png"), session="s", label="l",
                               camera=np.array([0.1, 0.6, 0.1]), rotvec=np.array([np.pi, 0.0, 0.0]))
    # Rotated half a turn about x, the flange's +Z points down, so the flange sits above the camera.
    assert session_replay.flange_pose(view, 0.03)[:3] == pytest.approx([0.1, 0.6, 0.13])


def test_the_joints_solved_back_put_the_camera_where_it_was_recorded():
    pytest.importorskip("mujoco")
    import scan_geometry
    import ur5_robot

    offset = 0.0306
    home = np.array(ur5_robot.DEFAULT_HOME_JOINTS)
    poses = [home, home + np.radians([5.0, -4.0, 6.0, -3.0, 4.0, 10.0])]
    views = []
    for q in poses:
        camera, rotvec = scan_geometry.camera_pose(q, offset)
        views.append(session_replay.View(image=Path("x.png"), session="s", label="l", camera=camera, rotvec=rotvec))

    solved = session_replay.solve_joints(views, offset)
    assert all(why is None for _, why in solved)
    for (q, _), view in zip(solved, views):
        camera, rotvec = scan_geometry.camera_pose(q, offset)
        assert np.linalg.norm(camera - view.camera) < scan_geometry.UNREACHED_POS
