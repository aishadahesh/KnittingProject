"""UR5 Robot Mode: safety layer, camera, scan runner, storage and splat export.

These run against the dry-run transport and the synthetic camera, so the whole
real-robot workflow is exercised without hardware. The point is that the parts
which would be dangerous or wrong on real hardware -- commanding a stopped arm,
saving a frame from the wrong pose, losing a capture's provenance -- are the
ones pinned down here.
"""

import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

# src/ is put on the path by the project-root conftest.py.
import gaussian_splatting
import rgb_analysis
import robot_camera
import ur5_robot
import ur5_scan
from scanner_storage import ScannerStorage


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def robot():
    bot = ur5_robot.UR5Robot(ur5_robot.DryRunTransport(travel_time=0.02))
    bot.connect("127.0.0.1")
    # Let the poll thread publish one state before anything asks if it is ready.
    time.sleep(0.15)
    yield bot
    bot.disconnect()


@pytest.fixture
def camera():
    cam = robot_camera.RobotCamera(robot_camera.SyntheticCameraBackend(size=(160, 120)))
    cam.start()
    yield cam
    cam.stop()


@pytest.fixture
def plan():
    import fabric_scanner as scanner

    args = SimpleNamespace(
        rows=2, cols=2, number_of_angles=2, width=0.12, length=0.10,
        edge_margin=0.004, square_margin=0.006, surface_wave=0.003,
        view_radius=0.018, angle_lift=0.014, approach_lift=0.040,
        center=[-0.45, -0.08, 0.30], max_span=scanner.DEFAULT_MAX_SPAN.tolist(),
        palette=None, cell_color_sets=None, model_json="", model_curves=None,
        cell_model_curves=None, random_patterns=False,
        pattern_rows=4, pattern_cols=5,
        pattern_repeat_rows=1, pattern_repeat_cols=1,
        pattern_repeat_spacing_x=1.0, pattern_repeat_spacing_y=1.0,
        batch_texture_width=420, batch_texture_height=340,
        pattern_density=0.62, random_seed=1,
        scanner_lighting=None, display_batch_colors=None,
    )
    return scanner.build_plan(args)


# ============================================================================
# Safety
# ============================================================================

def test_stop_latches_and_blocks_motion(robot):
    """A stop must not be clearable by simply asking to move again."""
    target = [-0.45, -0.08, 0.32, np.pi, 0.0, 0.0]
    assert robot.move_to_pose(target) is True

    robot.emergency_stop()
    assert robot.stop_requested is True
    with pytest.raises(RuntimeError, match="stop is latched"):
        robot.move_to_pose(target)
    # Continue must not silently resume through a latched stop either.
    assert robot.resume() is False
    assert robot.stop_requested is True

    robot.reset()
    assert robot.stop_requested is False
    assert robot.move_to_pose(target) is True


def test_pause_blocks_motion_until_resumed(robot):
    target = [-0.45, -0.08, 0.32, np.pi, 0.0, 0.0]
    robot.pause()
    with pytest.raises(RuntimeError, match="paused"):
        robot.move_to_pose(target)
    assert robot.resume() is True
    assert robot.move_to_pose(target) is True


def test_speed_limit_caps_commanded_speed(robot):
    robot.apply_speed_limit(0.25)
    assert robot.limited_speed(1.0) == pytest.approx(0.25)
    # Even asking for more than the arm's ceiling stays capped.
    robot.apply_speed_limit(1.0)
    assert robot.limited_speed(999.0) <= ur5_robot.MAX_SPEED
    assert robot.limited_acceleration(999.0) <= ur5_robot.MAX_ACCELERATION
    # Out-of-range limits are clamped, never applied raw.
    assert robot.apply_speed_limit(50.0) == 1.0
    assert robot.apply_speed_limit(-1.0) == 0.01


def test_readiness_reports_every_blocker():
    bot = ur5_robot.UR5Robot(ur5_robot.DryRunTransport())
    ready, issues = bot.readiness()
    assert ready is False
    assert any("not connected" in issue for issue in issues)


def test_disconnected_robot_refuses_to_move():
    bot = ur5_robot.UR5Robot(ur5_robot.DryRunTransport())
    with pytest.raises(RuntimeError):
        bot.move_to_pose([-0.45, -0.08, 0.32, np.pi, 0.0, 0.0])


def test_pendant_stop_latches_into_the_app(robot):
    """A stop raised on the hardware must stop the app, not just be displayed."""
    robot.transport.protective_stop()
    # Give the poll thread a chance to observe it.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not robot.stop_requested:
        time.sleep(0.02)
    assert robot.stop_requested is True
    with pytest.raises(RuntimeError):
        robot.move_to_pose([-0.45, -0.08, 0.32, np.pi, 0.0, 0.0])


# ============================================================================
# Camera
# ============================================================================

def test_preview_and_capture_come_from_the_same_frame(camera):
    """The saved image must be the previewed frame, not a second read."""
    frame = camera.capture()
    preview = camera.preview_image(80)
    assert frame is not None and preview is not None
    # Preview is the capture scaled: same aspect, smaller.
    assert preview.size[0] == 80
    assert preview.size[1] == pytest.approx(80 * frame.size[1] / frame.size[0], abs=1)


def test_capture_waits_for_a_frame_newer_than_the_move(camera):
    """A capture taken after arriving must not be a frame from during travel."""
    settled_at = time.monotonic()
    frame = camera.capture(newer_than=settled_at)
    assert frame is not None
    assert frame.timestamp > settled_at


def test_synthetic_frames_are_marked_not_real(camera):
    frame = camera.capture()
    assert frame.is_real is False
    assert frame.metadata()["camera_is_real"] is False


def test_camera_reports_failure_to_open():
    class Broken(robot_camera.CameraBackend):
        name = "broken"

        def open(self):
            raise RuntimeError("device busy")

    cam = robot_camera.RobotCamera(Broken())
    assert cam.start() is False
    assert "device busy" in cam.status


# ============================================================================
# Scan runner
# ============================================================================

def test_full_scan_captures_analyses_and_stores(tmp_path, robot, camera, plan):
    storage = ScannerStorage(tmp_path)
    runner = ur5_scan.RealScanRunner(
        robot, camera, plan, storage,
        output_root=tmp_path / "robot_scans",
        pattern_signature="sig-1",
        settings_snapshot={"scanner_rows": 2},
        lighting_condition="Lab LED",
    )
    assert runner.preflight()[0] is True
    assert runner.start() is True
    runner.wait(timeout=120)

    progress = runner.progress
    assert progress.captured == len(runner.scan_targets)
    assert progress.error == ""

    # Every capture is on disk and in the database with its provenance.
    summary = storage.database_summary()
    assert summary["capture_count"] == progress.captured
    stored = summary["captures"][0]
    assert stored["robot_mode"] == "real_ur5"
    assert stored["session_id"] == runner.session_id
    assert stored["lighting_condition"] == "Lab LED"
    assert len(stored["target_position"]) == 6
    assert os.path.exists(stored["image_path"])

    session = storage.robot_sessions()[0]
    assert session["status"] == "finished"
    assert session["capture_count"] == progress.captured
    assert session["is_real"] is False  # dry run, and recorded as such

    result = runner.analyze()
    assert len(result["cells"]) == 4  # a 2x2 grid
    assert all(len(cell["angles"]) == 2 for cell in result["cells"])
    assert result["session_id"] == runner.session_id


def test_scan_only_photographs_scan_views(plan):
    """approach/retreat points get the arm there; they are not captures."""
    runner = ur5_scan.RealScanRunner(None, None, plan)
    names = [plan.view_names[i] for i in runner.scan_targets]
    assert names
    assert not any(name in {"approach", "retreat", "travel"} for name in names)


def test_preflight_refuses_without_a_camera(robot, plan):
    runner = ur5_scan.RealScanRunner(robot, None, plan)
    ok, issues = runner.preflight()
    assert ok is False
    assert any("camera" in issue for issue in issues)
    assert runner.start() is False


def test_preflight_refuses_when_robot_not_ready(camera, plan):
    bot = ur5_robot.UR5Robot(ur5_robot.DryRunTransport())
    runner = ur5_scan.RealScanRunner(bot, camera, plan)
    ok, issues = runner.preflight()
    assert ok is False
    assert any("not connected" in issue for issue in issues)


def test_emergency_stop_ends_a_running_scan(tmp_path, robot, camera, plan):
    storage = ScannerStorage(tmp_path)
    runner = ur5_scan.RealScanRunner(
        robot, camera, plan, storage,
        output_root=tmp_path / "robot_scans",
        dwell=0.15,
    )
    assert runner.start() is True
    time.sleep(0.2)
    runner.emergency_stop()
    runner.wait(timeout=30)

    assert runner.running is False
    assert robot.stop_requested is True
    # The run is recorded as stopped, not quietly filed as a complete scan.
    assert storage.robot_sessions()[0]["status"] == "stopped"


# ============================================================================
# RGB analysis
# ============================================================================

def test_analysis_ignores_background_and_measures_the_fabric():
    """A coloured patch on a dark ground must read as the patch's colour."""
    image = Image.new("RGB", (120, 120), (12, 12, 14))
    image.paste(Image.new("RGB", (60, 60), (200, 80, 60)), (30, 30))
    stats = rgb_analysis.fabric_rgb_stats(image)
    assert stats["rgb"][0] > stats["rgb"][1] > stats["rgb"][2]
    assert stats["rgb"][0] == pytest.approx(200, abs=12)
    assert stats["pixel_count"] < stats["total_pixels"]


def test_analysis_is_the_same_function_the_simulation_uses():
    """Sim and real captures must not diverge on what a colour measurement is."""
    import gui

    image = Image.new("RGB", (80, 80), (90, 140, 70))
    assert gui.EmbeddedMujocoScanner._fabric_rgb_stats(image)["rgb"] == pytest.approx(
        rgb_analysis.fabric_rgb_stats(image)["rgb"]
    )


def test_comparison_ranks_most_different_colours_first():
    cells = [
        {"row": 0, "col": 0, "overall_rgb": [10.0, 10.0, 10.0]},
        {"row": 0, "col": 1, "overall_rgb": [12.0, 11.0, 10.0]},
        {"row": 1, "col": 0, "overall_rgb": [240.0, 230.0, 220.0]},
    ]
    comparisons = rgb_analysis.compare_cell_colors(cells)
    assert len(comparisons) == 3
    assert comparisons[0]["delta_rgb"] > comparisons[-1]["delta_rgb"]


# ============================================================================
# Storage
# ============================================================================

def test_simulation_captures_keep_working_unchanged(tmp_path):
    """The new columns must not disturb the existing Scan Mode write path."""
    storage = ScannerStorage(tmp_path)
    state = SimpleNamespace(_data={"scanner_rows": 2}, get=lambda k, d=None: d)
    storage.record_capture(state, {
        "path": str(tmp_path / "sim.png"),
        "row": 0, "col": 1, "station": 2, "target_index": 3,
        "angle": "angle 0", "rgb": [10.0, 20.0, 30.0],
    })
    capture = storage.database_summary()["captures"][0]
    assert capture["robot_mode"] == "simulation"
    assert capture["session_id"] == ""
    assert capture["average_rgb"] == [10.0, 20.0, 30.0]


def test_migration_preserves_a_pre_existing_database(tmp_path):
    """Opening an old database adds columns without touching its rows."""
    import sqlite3

    db_dir = tmp_path / "scanner_data"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "scanner.db")
    conn.executescript(
        """
        CREATE TABLE captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_signature TEXT, image_path TEXT NOT NULL,
            row_index INTEGER NOT NULL, col_index INTEGER NOT NULL,
            station_index INTEGER NOT NULL, target_index INTEGER NOT NULL,
            angle TEXT NOT NULL, capture_mode TEXT NOT NULL,
            capture_settings_json TEXT NOT NULL, rgb_json TEXT, created_at TEXT NOT NULL
        );
        INSERT INTO captures VALUES
            (1, 'old', 'legacy.png', 0, 0, 0, 0, 'angle 0', 'natural', '{}', '[1,2,3]', '2026-01-01');
        """
    )
    conn.commit()
    conn.close()

    storage = ScannerStorage(tmp_path)
    summary = storage.database_summary()
    assert summary["capture_count"] == 1
    legacy = summary["captures"][0]
    assert legacy["image_path"] == "legacy.png"
    assert legacy["robot_mode"] == "simulation"


def test_splat_output_is_recorded_against_session_and_captures(tmp_path):
    storage = ScannerStorage(tmp_path)
    storage.start_robot_session({"session_id": "s1", "robot_ip": "10.0.0.5", "is_real": True})
    state = SimpleNamespace(_data={}, get=lambda k, d=None: d)
    storage.record_capture(state, {
        "path": "a.png", "row": 0, "col": 0, "station": 0, "target_index": 0,
        "angle": "angle 0", "rgb": [1.0, 2.0, 3.0],
        "robot_mode": "real_ur5", "session_id": "s1",
    })
    storage.set_session_splat_output("s1", "out/model.ply")

    assert storage.robot_sessions()[0]["splat_output_path"] == "out/model.ply"
    assert storage.database_summary()["captures"][0]["splat_output_path"] == "out/model.ply"


# ============================================================================
# App state
# ============================================================================

def test_undo_does_not_orphan_a_live_robot_connection():
    """Undo must not restore a stale controller over a connected robot.

    _clone passes unknown objects through by reference, so without the snapshot
    exclusion an undo would swap the live controller for whichever one the
    snapshot held -- leaving a connected arm with nothing able to stop it.
    """
    from rendering import Camera
    from app_state import AppState
    import gui_ur5

    class StubRenderer:
        def __getattr__(self, name):
            return lambda *a, **k: None

    state = AppState(Camera(), StubRenderer())
    state.project_root = "."
    controller = gui_ur5.UR5ModeController(state, None, None)
    controller.connect("127.0.0.1", "dry_run")
    state.ur5_controller = controller
    try:
        snapshot = state.snapshot_state()
        assert "ur5_controller" not in snapshot
        assert "embedded_scanner" not in snapshot

        state.restore_snapshot(snapshot)
        assert state.get("ur5_controller") is controller
        assert state.get("ur5_controller").robot.connected is True
    finally:
        controller.close()


def test_ur5_settings_persist_but_the_controller_does_not():
    from rendering import Camera
    from app_state import AppState

    state = AppState(Camera(), None)
    saved = [key for key in state.saved_state_keys if key.startswith("ur5_")]
    assert "ur5_robot_ip" in saved
    assert "ur5_speed_limit" in saved
    # Live handles are not settings and must never be written to params.json.
    assert "ur5_controller" not in state.saved_state_keys


def test_ur5_defaults_cannot_command_hardware_on_their_own():
    """Opening the mode must not be able to drive a real arm by itself."""
    from rendering import Camera
    from app_state import AppState

    state = AppState(Camera(), None)
    assert state.get("ur5_transport") == "dry_run"
    assert state.get("ur5_speed_limit") <= 0.25


# ============================================================================
# Gaussian Splatting
# ============================================================================

def test_pose_conversion_round_trips():
    pose = [-0.45, -0.08, 0.30, np.pi, 0.0, 0.2]
    c2w = gaussian_splatting.camera_to_world(pose)
    assert np.allclose(c2w @ np.linalg.inv(c2w), np.eye(4))
    # The camera sits the hand-eye offset away from the tool centre point.
    offset = np.linalg.norm(c2w[:3, 3] - np.array(pose[:3]))
    assert offset == pytest.approx(gaussian_splatting.DEFAULT_CAMERA_OFFSET_M)


def test_export_writes_both_dataset_formats(tmp_path):
    image_path = tmp_path / "cap.png"
    Image.new("RGB", (64, 48), (120, 110, 100)).save(image_path)
    captures = [{
        "path": str(image_path),
        "actual_tcp_pose": [-0.45, -0.08, 0.30, np.pi, 0.0, 0.0],
        "angle": "angle 0", "row": 0, "col": 0,
    }]
    dataset = gaussian_splatting.export_dataset(captures, tmp_path / "ds", session_id="s1")

    assert dataset.image_count == 1
    assert dataset.warnings == []
    assert dataset.transforms_path.exists()
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        assert (dataset.colmap_dir / name).exists()

    import json

    transforms = json.loads(dataset.transforms_path.read_text())
    assert transforms["w"] == 64 and transforms["h"] == 48
    assert len(transforms["frames"]) == 1
    assert np.asarray(transforms["frames"][0]["transform_matrix"]).shape == (4, 4)


def test_export_reports_unusable_captures_rather_than_dropping_them(tmp_path):
    image_path = tmp_path / "cap.png"
    Image.new("RGB", (32, 32), (10, 10, 10)).save(image_path)
    dataset = gaussian_splatting.export_dataset(
        [{"path": "missing.png"}, {"path": str(image_path)}],
        tmp_path / "ds",
    )
    assert dataset.image_count == 0
    assert any("missing image" in w for w in dataset.warnings)
    assert any("no camera pose" in w for w in dataset.warnings)


def test_trainer_refuses_incomplete_configuration(tmp_path):
    dataset = gaussian_splatting.SplatDataset(
        root=tmp_path, image_count=0, transforms_path=tmp_path, colmap_dir=tmp_path
    )
    trainer = gaussian_splatting.SplatTrainer("echo hi", dataset, tmp_path)
    assert trainer.start() is False
    assert "Export a dataset" in trainer.status

    dataset.image_count = 4
    assert gaussian_splatting.SplatTrainer("", dataset, tmp_path).start() is False
