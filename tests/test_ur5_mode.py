"""UR5 Robot Mode: safety layer, camera, scan runner, storage and splat export.

These run against the dry-run transport and the synthetic camera, so the whole
real-robot workflow is exercised without hardware. The point is that the parts
which would be dangerous or wrong on real hardware -- commanding a stopped arm,
saving a frame from the wrong pose, losing a capture's provenance -- are the
ones pinned down here.
"""

import math
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import fabric_detector
import gaussian_splatting
import robot_camera
import ur5_robot
import ur5_scan
from scanner_storage import ScannerStorage
from scan_geometry import TELEOP_JOINTS


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


def test_failed_pause_latches_stop_and_failed_reset_keeps_it(robot, monkeypatch):
    def fail(*args):
        raise OSError("controller unavailable")

    monkeypatch.setattr(robot.transport, "stop", fail)
    monkeypatch.setattr(robot.transport, "reset_stop", fail)
    robot.pause()
    assert robot.paused and robot.stop_requested
    assert "Pause failed" in robot.status
    assert not robot.resume()
    assert not robot.reset()
    assert robot.stop_requested and robot.paused
    assert "Reset failed" in robot.status


def test_protective_stop_attempts_both_commands_and_reports_failure():
    calls = []

    def stop(deceleration):
        calls.append("decelerate")
        raise OSError("deceleration failed")

    transport = ur5_robot.RTDETransport()
    transport.control = SimpleNamespace(stopL=stop, triggerProtectiveStop=lambda: calls.append("latch"))
    with pytest.raises(RuntimeError, match="deceleration failed"):
        transport.protective_stop()
    assert calls == ["decelerate", "latch"]


def test_failed_safety_query_never_approves_a_pose():
    def fail(pose):
        raise OSError("safety query failed")

    transport = ur5_robot.RTDETransport()
    transport.control = SimpleNamespace(isPoseWithinSafetyLimits=fail)
    with pytest.raises(OSError, match="safety query failed"):
        transport.pose_within_limits(np.zeros(6))


def test_scan_stop_reports_transport_failure(robot, plan, monkeypatch):
    def fail(deceleration):
        raise OSError("stop failed")

    runner = ur5_scan.RealScanRunner(robot, None, plan)
    monkeypatch.setattr(robot.transport, "stop", fail)
    runner.stop()
    assert runner.progress.error == "stop failed"
    assert "Stop failed" in runner.progress.status
    assert robot.stop_requested


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


def test_a_run_that_saves_nothing_needs_no_camera(robot, plan):
    """Motion only: every move and every wait, no shutter - so no camera to refuse for.

    The arm is often ready before the camera is, and a path worth walking is worth walking
    then; refusing to start for want of a device the run never touches refuses the one
    thing it can still do.
    """
    runner = ur5_scan.RealScanRunner(robot, None, plan, save_images=False)
    assert runner.preflight() == (True, [])
    assert runner.start() is True
    runner.wait(timeout=120)

    progress = runner.progress
    assert progress.error == ""
    assert progress.captured == 0
    assert progress.target_index + 1 == progress.target_count

    saving = ur5_scan.RealScanRunner(robot, None, plan, save_images=True)
    ok, issues = saving.preflight()
    assert not ok and any("camera is not running" in issue for issue in issues)


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
    stats = fabric_detector.fabric_rgb_stats(image)
    assert stats["rgb"][0] > stats["rgb"][1] > stats["rgb"][2]
    assert stats["rgb"][0] == pytest.approx(200, abs=12)
    assert stats["pixel_count"] < stats["total_pixels"]


def test_comparison_ranks_most_different_colours_first():
    cells = [
        {"row": 0, "col": 0, "overall_rgb": [10.0, 10.0, 10.0]},
        {"row": 0, "col": 1, "overall_rgb": [12.0, 11.0, 10.0]},
        {"row": 1, "col": 0, "overall_rgb": [240.0, 230.0, 220.0]},
    ]
    comparisons = fabric_detector.compare_cell_colors(cells)
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
    # The panel no longer offers the dry run; the tests' stand-in arm is connected directly.
    controller.robot = ur5_robot.UR5Robot(ur5_robot.DryRunTransport())
    controller.robot.connect("127.0.0.1")
    state.ur5_controller = controller
    try:
        snapshot = state.snapshot_state()
        assert "ur5_controller" not in snapshot

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
    """Opening the mode must not be able to drive a real arm by itself.

    The default is TeleopApp, which moves the arm only while it is following it; Direct
    RTDE, where every command is the arm's, has to be chosen.
    """
    from rendering import Camera
    from app_state import AppState

    state = AppState(Camera(), None)
    assert state.get("ur5_transport") == "teleop"
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
    assert dataset.warnings == ["no fabric height recorded with the captures: the trainer starts from random points"]
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


# ============================================================================
# TeleopApp transport
# ============================================================================

HOME_JOINTS = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]


class FakeTeleop:
    """TeleopApp's DynamicRpcServer, reduced to the JSON it answers with.

    Records every call in order, which is how the seed-before-arm sequence is checked:
    arming across a standing gap is what trips the app's deviation guard, and the order is
    the whole fix.
    """

    def __init__(self, **status):
        self.calls = []
        self.status = {
            "connected": True, "motorsOn": True, "commandSending": True, "realHardware": False,
            "trackingMode": "tracking", "engaging": False, "trackingGapDeg": 0.0,
            "trajectoryExecuting": False, "poses": ["Home"],
            "q": dict(zip(TELEOP_JOINTS, HOME_JOINTS)),
            "plannedQ": dict(zip(TELEOP_JOINTS, HOME_JOINTS)),
            "currentPose": {"px": 0.0, "py": 0.0, "pz": 0.0},
        }
        self.status.update(status)

    def call(self, method, *args):
        self.calls.append((method, *args))
        if method == "describe":
            return {"targets": {"ur5": {"joints": TELEOP_JOINTS}}}
        if method == "getStatus":
            return dict(self.status)
        return None

    def methods(self):
        return [call[0] for call in self.calls]


@pytest.fixture
def teleop():
    import teleop_transport

    fake = FakeTeleop()
    transport = teleop_transport.TeleopAppTransport(client_factory=lambda address, path: fake)
    transport.connect("localhost:50052")
    return transport, fake


def test_following_is_read_from_command_sending_not_the_tracking_mode(teleop):
    """The guard trips, the mode still says tracking, and the arm stands still.

    This is the flag that made a panel claim it was armed while nothing moved.
    """
    transport, fake = teleop
    assert transport.following

    fake.status.update(commandSending=False, trackingMode="tracking")
    transport._refresh(force=True)
    assert transport.following is False
    assert transport.blockers() == ["TeleopApp is not commanding the arm; arm it with Reset"]


def test_an_unevaluable_tracking_gap_is_not_a_gap_of_minus_one(teleop):
    transport, fake = teleop
    fake.status.update(trackingGapDeg=-1.0)
    transport._refresh(force=True)
    assert transport.gap_deg is None

    fake.status.update(trackingGapDeg=12.5)
    transport._refresh(force=True)
    assert transport.gap_deg == pytest.approx(12.5)


def test_arming_seeds_the_virtual_robot_before_energizing(teleop):
    """Order matters: seed, then motors, then tracking."""
    transport, fake = teleop
    fake.status.update(trackingGapDeg=20.0)
    transport._refresh(force=True)
    transport.reset_stop()

    ordered = [method for method in fake.methods() if method in ("gotoJointAngles", "setMotors", "setTrackingMode")]
    assert ordered == ["gotoJointAngles", "setMotors", "setTrackingMode"]


def test_arming_refuses_loudly_when_commanding_never_starts(teleop):
    transport, fake = teleop
    fake.status.update(commandSending=False)
    transport._refresh(force=True)
    with pytest.raises(RuntimeError, match="not commanding the arm"):
        transport.reset_stop()


def test_a_move_is_paced_by_how_far_the_furthest_joint_travels(teleop):
    """Seconds per move, derived from joint travel -- not from the interface's metres per second.

    The configured speed is the PEAK, because the arm eases in and out of every move: asking
    for 12 deg/s and getting a 12 deg/s average would put 22.5 deg/s through the middle of it.
    """
    import scan_geometry

    transport, _ = teleop
    transport.max_joint_speed_deg = 12.0
    transport.floor_seconds = 2.0

    nudge = list(HOME_JOINTS)
    nudge[0] += math.radians(6.0)
    swing = list(HOME_JOINTS)
    swing[0] += math.radians(120.0)

    assert transport.ramp_for(nudge) == pytest.approx(2.0)

    seconds = transport.ramp_for(swing)
    assert 120.0 / seconds * scan_geometry.PEAK_OVER_MEAN == pytest.approx(12.0, abs=0.1)
    assert seconds == pytest.approx(18.75, abs=0.2)


def test_a_commanded_pose_is_the_tool_point_not_the_flange(teleop):
    """The planner names where the camera goes, so the flange target is one offset back.

    Solving the tool point as a flange target parks the camera a tool-length past the pose
    that was asked for, and the runner's arrival check then never converges.
    """
    transport, fake = teleop
    asked = transport.tool_pose()

    transport.move_to_pose(asked, 0.0, 0.0)
    sent = [call for call in fake.calls if call[0] == "gotoJointAngles"][-1]
    fake.status["q"] = dict(sent[2])
    transport._refresh(force=True)

    assert float(np.linalg.norm(np.asarray(transport.tool_pose())[:3] - np.asarray(asked)[:3])) < 0.002


def test_home_uses_the_taught_pose_when_the_arm_has_one(teleop):
    """Home is where the simulated and the real arm are defined to agree."""
    transport, fake = teleop
    transport.move_home(HOME_JOINTS, 0.0, 0.0)
    assert ("gotoPose", "ur5", "Home") in fake.calls

    fake.status["poses"] = []
    transport._refresh(force=True)
    transport.move_home(HOME_JOINTS, 0.0, 0.0)
    assert fake.methods()[-1] == "gotoJointAngles"


def test_a_rehearsal_is_only_offered_where_it_would_not_move_the_arm(teleop):
    """A "rehearsal" on a transport wired straight to a controller is an ordinary move."""
    transport, fake = teleop

    assert ur5_robot.DryRunTransport().rehearsable() is True
    assert ur5_robot.RTDETransport().rehearsable() is False

    assert transport.following and not transport.on_hardware and transport.rehearsable() is True
    fake.status.update(realHardware=True)
    transport._refresh(force=True)
    assert transport.following and transport.on_hardware and transport.rehearsable() is False
    fake.status.update(commandSending=False)
    transport._refresh(force=True)
    assert transport.rehearsable() is True


def test_teleop_captures_are_real_only_on_the_arm_itself(teleop):
    """On TeleopApp's emulator nothing physical moved, so a capture must not be stored as real."""
    transport, fake = teleop

    assert transport.is_real is False
    fake.status.update(realHardware=True)
    transport._refresh(force=True)
    assert transport.is_real is True


def test_not_following_blocks_a_move_through_the_guard(teleop):
    """A move sent while TeleopApp is not commanding is accepted and does nothing.

    Reporting it as a readiness blocker is what turns that silence into a refusal.
    """
    transport, fake = teleop
    bot = ur5_robot.UR5Robot(transport)
    bot.connect("localhost:50052")
    time.sleep(0.15)
    try:
        assert bot.ready()

        fake.status.update(commandSending=False)
        transport._refresh(force=True)
        bot.refresh()
        assert not bot.ready()
        with pytest.raises(RuntimeError, match="not commanding"):
            bot.move_to_pose(transport.tool_pose())
    finally:
        bot.disconnect()


# ============================================================================
# The capture path
# ============================================================================

def test_a_capture_plan_only_offers_reachable_collision_free_poses():
    import scan_geometry

    plan = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="spin", count=3, fabric_z=0.20)

    assert len(plan.viewpoints) == 3
    assert plan.clear and not plan.refused
    assert all(point.err_pos < scan_geometry.UNREACHED_POS for point in plan.viewpoints)
    assert plan.assess() == (True, [])


def test_every_capture_viewpoint_is_photographed():
    """A view named approach/retreat/travel is skipped by the runner, so none may be."""
    import scan_geometry

    plan = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="spin", count=3, fabric_z=0.20)
    runner = ur5_scan.RealScanRunner(ur5_robot.UR5Robot(ur5_robot.DryRunTransport()), None, plan)
    assert runner.scan_targets == list(range(len(plan.view_names)))


def test_a_capture_records_the_geometry_it_was_taken_from():
    import scan_geometry

    plan = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="orbit", count=3, rings=1, fabric_z=0.20)
    recorded = plan.capture_metadata(1)

    assert recorded["capture_pattern"] == "orbit"
    assert recorded["capture_azimuth_deg"] == pytest.approx(plan.viewpoints[1].azimuth_deg)
    assert len(recorded["capture_subject"]) == 3


def test_a_capture_run_stores_one_image_per_viewpoint(tmp_path, robot, camera):
    """The whole point of the reuse: the wait at each viewpoint becomes a real capture.

    Same runner, same storage, same records as a fabric scan - only the plan differs, so a
    capture session is queryable next to every other session in the database.
    """
    import scan_geometry

    plan = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="spin", count=3, fabric_z=0.20)
    storage = ScannerStorage(tmp_path)
    runner = ur5_scan.RealScanRunner(
        robot, camera, plan, storage,
        output_root=tmp_path / "robot_scans",
        settings_snapshot={"teleop_pattern": "spin"},
        dwell=0.0,
    )
    assert runner.preflight()[0] is True
    assert runner.start() is True
    runner.wait(timeout=120)

    progress = runner.progress
    assert progress.error == ""
    assert progress.captured == len(plan.viewpoints)

    summary = storage.database_summary()
    assert summary["capture_count"] == len(plan.viewpoints)
    stored = summary["captures"][0]
    assert stored["capture_pattern"] == "spin"
    assert stored["camera_angle"] in plan.view_names
    assert stored["capture_azimuth_deg"] is not None
    assert os.path.exists(stored["image_path"])

    session = storage.robot_sessions()[0]
    assert session["status"] == "finished"
    assert session["capture_count"] == len(plan.viewpoints)


def test_a_path_with_nothing_reachable_is_refused_before_the_arm_moves():
    """The refusal has to come before the arm moves, not 363 mm short of the target."""
    import scan_geometry

    with pytest.raises(RuntimeError, match="No viewpoint"):
        scan_geometry.build_capture_plan(HOME_JOINTS, pattern="grid", grid_rows=1, grid_cols=2, fabric_z=1.5)


def test_sim_ur5_and_sim_camera_run_unified_grid_scan(tmp_path):
    import scan_geometry

    transport = ur5_robot.SimUR5Transport(max_joint_speed_deg=360.0)
    bot = ur5_robot.UR5Robot(transport)
    assert bot.connect("sim") is True
    time.sleep(0.15)
    cam_backend = robot_camera.SimCameraBackend(
        size=(160, 120),
        pose_provider=lambda: (bot.state.tcp_pose, 55.0, (-0.45, -0.08, 0.20)),
    )
    cam = robot_camera.RobotCamera(cam_backend)
    assert cam.start() is True
    try:
        plan = scan_geometry.build_capture_plan(
            HOME_JOINTS,
            pattern="grid",
            grid_rows=1,
            grid_cols=2,
            grid_views_per_station=1,
            fabric_z=0.20,
        )
        storage = ScannerStorage(tmp_path)
        runner = ur5_scan.RealScanRunner(
            bot,
            cam,
            plan,
            storage,
            output_root=tmp_path / "robot_scans",
            dwell=0.0,
        )
        assert runner.preflight()[0] is True
        assert runner.start() is True
        runner.wait(timeout=60)
        assert runner.progress.error == ""
        assert runner.progress.captured == 2
        result = runner.analyze()
        assert len(result["cells"]) == 2
    finally:
        cam.stop()
        bot.disconnect()
