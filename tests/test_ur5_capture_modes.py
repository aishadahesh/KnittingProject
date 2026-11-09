"""Teacher mode and the low grid: the two ways UR5 Robot Mode chooses where to photograph from.

The teacher tests need no MuJoCo - free drive, recording and the saved sets are all robot and
file work. The plan tests solve against the MuJoCo UR5e and skip without it.
"""

import threading
import time

import numpy as np
import pytest

import ur5_robot

HOME_JOINTS = list(ur5_robot.DEFAULT_HOME_JOINTS)


@pytest.fixture
def robot():
    bot = ur5_robot.UR5Robot(ur5_robot.DryRunTransport(travel_time=0.02))
    bot.connect("127.0.0.1")
    time.sleep(0.15)
    yield bot
    bot.disconnect()


@pytest.fixture
def controller(robot, tmp_path, monkeypatch):
    from rendering import Camera
    from app_state import AppState
    import gui_ur5
    import paths

    class StubRenderer:
        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr(paths, "TAUGHT_VIEWS_DIR", tmp_path / "taught_views")
    state = AppState(Camera(), StubRenderer())
    state.project_root = str(tmp_path)
    ctl = gui_ur5.UR5ModeController(state, None, None)
    ctl.robot = robot
    return ctl


# ============================================================================
# Teacher mode
# ============================================================================

def test_free_drive_blocks_every_move_until_held(robot):
    """A move sent to an arm someone is holding is a move into their hands."""
    assert robot.ready()
    robot.free_drive()

    assert robot.free_driving
    assert not robot.ready()
    assert any("free drive" in issue for issue in robot.readiness()[1])
    with pytest.raises(RuntimeError, match="free drive"):
        robot.move_to_joints(HOME_JOINTS)
    assert robot.move_home() is False

    robot.hold()
    assert not robot.free_driving
    assert robot.ready()


def test_free_drive_is_refused_while_a_stop_is_latched(robot):
    robot.emergency_stop()
    with pytest.raises(RuntimeError, match="stop is latched"):
        robot.free_drive()
    assert not robot.free_driving


def test_a_failed_hold_leaves_the_arm_marked_loose(robot, monkeypatch):
    """The flag clears only once the transport says it held - otherwise Run would be offered on a loose arm."""
    robot.free_drive()

    def refuse():
        raise RuntimeError("teach mode would not end")

    monkeypatch.setattr(robot.transport, "hold", refuse)
    with pytest.raises(RuntimeError):
        robot.hold()
    assert robot.free_driving
    assert not robot.ready()


def test_a_recorded_view_is_the_joints_the_arm_stands_in_now(controller):
    posed = np.array(HOME_JOINTS) + np.radians([10.0, -5.0, 3.0, 0.0, 8.0, -12.0])
    controller.free_drive()
    controller.robot.transport._joints = posed
    controller.record_view()

    views = controller.state.get("teleop_taught_views")
    assert len(views) == 1
    assert views[0]["name"] == "view 1"
    assert views[0]["joints"] == pytest.approx(list(posed))
    assert all(isinstance(value, float) for value in views[0]["joints"])


def test_forgetting_a_view_renumbers_the_rest(controller):
    for offset in (0.0, 0.1, 0.2):
        controller.robot.transport._joints = np.array(HOME_JOINTS) + offset
        controller.record_view()
    controller.forget_view(0)

    views = controller.state.get("teleop_taught_views")
    assert [view["name"] for view in views] == ["view 1", "view 2"]
    assert views[0]["joints"][0] == pytest.approx(HOME_JOINTS[0] + 0.1)


def test_a_taught_set_survives_a_round_trip_to_disk(controller):
    import gui_ur5

    for offset in (0.0, 0.05):
        controller.robot.transport._joints = np.array(HOME_JOINTS) + offset
        controller.record_view()
    taught = [dict(view) for view in controller.state.get("teleop_taught_views")]

    controller.save_taught_set("left side / close")
    assert gui_ur5.taught_set_names() == ["left_side_close"]

    controller._set_taught_views([])
    controller.load_taught_set("left_side_close")
    assert controller.state.get("teleop_taught_views") == taught
    assert controller.state.get("teleop_taught_set") == "left_side_close"


def test_changing_the_taught_views_drops_a_checked_taught_plan(controller):
    from types import SimpleNamespace

    controller.plan = SimpleNamespace(pattern="taught")
    controller.record_view()
    assert controller.plan is None


# ============================================================================
# Capture height (Z)
# ============================================================================

def test_a_new_capture_height_is_clamped_and_replans_the_points(controller, monkeypatch):
    import gui_ur5
    from types import SimpleNamespace

    checks = []
    monkeypatch.setattr(controller, "check_in_background", lambda: checks.append(controller.state.get("teleop_camera_height")))
    controller.plan = SimpleNamespace(pattern="grid")
    controller.set_capture_height(0.12)
    assert controller.state.get("teleop_camera_height") == pytest.approx(0.12)
    assert controller.plan is None
    assert checks == [pytest.approx(0.12)]

    controller.set_capture_height(5.0)
    assert controller.state.get("teleop_camera_height") == pytest.approx(gui_ur5.CAPTURE_HEIGHT_MAX)
    controller.set_capture_height(-1.0)
    assert controller.state.get("teleop_camera_height") == pytest.approx(gui_ur5.CAPTURE_HEIGHT_MIN)


def test_a_jog_down_asks_for_the_same_spot_one_step_lower(controller, monkeypatch):
    """The jog pad moves the camera from the panel, through the same solve and check as a capture."""
    pytest.importorskip("mujoco")
    import scan_geometry

    flown = []
    monkeypatch.setattr(controller, "_fly_over", lambda xy, height, stop=None: flown.append((np.asarray(xy, dtype=float), height)))
    controller.robot.transport._joints = np.array(HOME_JOINTS)
    controller.robot.refresh()
    before = controller.camera_height_over_fabric(HOME_JOINTS)
    camera = scan_geometry.camera_position(HOME_JOINTS, 0.0306)
    controller._jog_loop(np.array([0.0, 0.0, -0.02]))

    assert len(flown) == 1
    assert flown[0][0] == pytest.approx(camera[:2], abs=1e-6)
    assert flown[0][1] == pytest.approx(before - 0.02, abs=1e-6)


def test_a_sideways_jog_keeps_the_height_and_moves_one_step(controller, monkeypatch):
    pytest.importorskip("mujoco")
    import scan_geometry

    flown = []
    monkeypatch.setattr(controller, "_fly_over", lambda xy, height, stop=None: flown.append((np.asarray(xy, dtype=float), height)))
    controller.robot.transport._joints = np.array(HOME_JOINTS)
    controller.robot.refresh()
    camera = scan_geometry.camera_position(HOME_JOINTS, 0.0306)
    controller._jog_loop(np.array([0.01, 0.0, 0.0]))

    assert np.linalg.norm(flown[0][0] - camera[:2]) == pytest.approx(0.01, abs=1e-6)
    assert flown[0][1] == pytest.approx(controller.camera_height_over_fabric(HOME_JOINTS), abs=1e-6)


def test_a_corner_under_the_crosshair_lies_on_the_fabric_plane(controller, monkeypatch):
    pytest.importorskip("mujoco")
    import gui_ur5

    monkeypatch.setattr(controller, "check_in_background", lambda: None)
    controller.robot.transport._joints = np.array(HOME_JOINTS)
    controller.set_fabric_gap(0.25)
    plane = gui_ur5._settings_fabric_z(controller.state, None)
    controller.mark_corner_under_crosshair()

    corners = controller.state.get("teleop_fabric_corners")
    assert len(corners) == 1 and corners[0][2] == pytest.approx(plane, abs=1e-9)
    assert controller.plan is None


def test_a_check_asked_for_while_one_runs_follows_it_with_the_latest_settings(controller, monkeypatch):
    import threading

    release, heights = threading.Event(), []

    def build_plan():
        heights.append(controller.state.get("teleop_camera_height"))
        release.wait(2.0)
        return True

    monkeypatch.setattr(controller, "build_plan", build_plan)
    controller.check_in_background()
    for height in (0.08, 0.09, 0.10):
        controller.state.teleop_camera_height = height
        controller.check_in_background()
    release.set()
    deadline = time.monotonic() + 2.0
    while controller.planning and time.monotonic() < deadline:
        time.sleep(0.01)

    assert not controller.planning
    assert len(heights) == 2 and heights[-1] == pytest.approx(0.10)


def test_grid_stations_spread_with_the_capture_height():
    pytest.importorskip("scipy")
    import scan_geometry

    pose = np.r_[0.0, 0.5, 0.40, np.pi, 0.0, 0.0]
    for height in (0.05, 0.10):
        views, _ = scan_geometry.grid_views(pose, 1, 3, height, 0.10, 0.20, 0.0306, None, 55.0, 0.3)
        steps = np.linalg.norm(np.diff([view.position for view in views], axis=0), axis=1)
        assert steps == pytest.approx([scan_geometry.frame_side(height, 55.0) * 0.7] * 2, abs=1e-9)


def test_a_pose_the_arm_reaches_is_solved_even_from_a_start_facing_away():
    """The solve may swing the base and refold the arm, rather than stall on the branch it started on."""
    pytest.importorskip("mujoco")
    import scan_geometry

    mj, model, data, site_id = scan_geometry.load_model()
    fabric_z = scan_geometry.fabric_height(mj, model, data, site_id, 0.20)
    camera = scan_geometry.camera_position(HOME_JOINTS, 0.0306)
    view = scan_geometry.Viewpoint("below", np.r_[camera[:2], fabric_z + 0.06],
                                   scan_geometry.looking_down_along([1.0, 0.0]), elevation_deg=90.0)
    facing_away = np.array(HOME_JOINTS) + np.radians([120.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    solver = (mj, model, data, site_id, fabric_z, 0.0306, None)

    solved = scan_geometry._solve(solver, view, facing_away, leaving=True)
    reached = scan_geometry.camera_point(scan_geometry.seed_from(mj, model, data, site_id, solved.config), 0.0306)
    assert reached == pytest.approx(view.position, abs=scan_geometry.UNREACHED_POS)


def test_a_ruler_gap_moves_the_unmarked_fabric_to_where_it_is(controller, monkeypatch):
    """A guessed plane above the real cloth refuses moves into room that is there; a measured gap fixes it."""
    pytest.importorskip("mujoco")
    import scan_geometry

    monkeypatch.setattr(controller, "check_in_background", lambda: None)
    controller.robot.transport._joints = np.array(HOME_JOINTS)
    controller.set_fabric_gap(0.30)

    import gui_ur5

    camera = scan_geometry.camera_position(HOME_JOINTS, 0.0306)
    assert controller.camera_height_over_fabric(HOME_JOINTS) == pytest.approx(0.30, abs=1e-6)
    assert controller.state.get("teleop_fabric_z") == pytest.approx(camera[2] - 0.30, abs=1e-6)
    assert gui_ur5._capture_settings(controller.state)["fabric_z"] == pytest.approx(camera[2] - 0.30, abs=1e-6)
    assert "teleop_fabric_z" in controller.state.saved_state_keys

    controller.forget_fabric_height()
    assert controller.state.get("teleop_fabric_z") is None
    assert gui_ur5._capture_settings(controller.state)["fabric_z"] is None


def test_a_jog_down_past_the_floor_stops_at_the_floor(controller, monkeypatch):
    """A step that would take the camera under the lowest allowed height goes as low as allowed instead of nowhere."""
    pytest.importorskip("mujoco")
    import gui_ur5
    import scan_geometry

    flown = []
    monkeypatch.setattr(controller, "_fly_over", lambda xy, height, stop=None: flown.append(height))
    controller.robot.transport._joints = np.array(HOME_JOINTS)
    controller.robot.refresh()
    camera = scan_geometry.camera_position(HOME_JOINTS, 0.0306)
    controller.state.teleop_fabric_z = float(camera[2] - 0.071)
    controller._jog_loop(np.array([0.0, 0.0, -0.05]))
    assert flown == [pytest.approx(gui_ur5.CAPTURE_HEIGHT_MIN)]
    assert "lowest allowed" in controller.status

    controller.state.teleop_fabric_z = float(camera[2] - gui_ur5.CAPTURE_HEIGHT_MIN)
    controller._jog_loop(np.array([0.0, 0.0, -0.01]))
    assert len(flown) == 1
    assert "already at the lowest" in controller.status


def test_capturing_from_here_unmarked_takes_this_exact_pose_at_the_typed_gap(controller, monkeypatch):
    """Unmarked, the guessed plane can say 3.5 cm where the ruler says 30; the typed gap wins, and the
    capture height is then exactly where the camera stands."""
    pytest.importorskip("mujoco")

    checks = []
    monkeypatch.setattr(controller, "check_in_background", lambda: checks.append(True))
    controller.robot.transport._joints = np.array(HOME_JOINTS)
    controller.fabric_gap_cm = 30.0
    controller.adopt_camera_height()

    assert controller.state.get("teleop_camera_height") == pytest.approx(0.30, abs=1e-6)
    assert controller.camera_height_over_fabric(HOME_JOINTS) == pytest.approx(0.30, abs=1e-6)
    assert checks and "Capturing from here" in controller.status


def test_a_taught_view_below_the_guessed_plane_becomes_the_capture_height(controller, monkeypatch):
    """The view recorded in the session that found this: 37 cm under Home, so 20 cm under the plane
    guessed from Home - every move toward it was refused as going through the cloth."""
    pytest.importorskip("mujoco")

    monkeypatch.setattr(controller, "check_in_background", lambda: None)
    taught = list(np.radians([-92.6, -62.3, 109.8, -135.6, -82.4, 0.7]))
    assert controller.camera_height_over_fabric(taught) < 0.0
    controller.fabric_gap_cm = 12.0
    controller.adopt_camera_height(taught)

    assert controller.camera_height_over_fabric(taught) == pytest.approx(0.12, abs=1e-6)
    assert controller.state.get("teleop_camera_height") == pytest.approx(0.12, abs=1e-6)


def test_capturing_from_here_unmarked_asks_for_the_gap_first(controller):
    controller.fabric_gap_cm = None
    before = controller.state.get("teleop_camera_height")
    controller.adopt_camera_height()
    assert controller.state.get("teleop_camera_height") == before
    assert "lens is over the cloth" in controller.status


def test_setting_the_height_moves_corners_marked_over_a_guess_onto_it(controller, monkeypatch):
    """Corners marked under the crosshair while the height was a guess lie on the guess; they must not keep holding it."""
    pytest.importorskip("mujoco")
    import gui_ur5
    import scan_geometry

    monkeypatch.setattr(controller, "check_in_background", lambda: None)
    controller.state.teleop_fabric_corners = [[0.0, 0.5, 0.32], [0.1, 0.5, 0.32]]
    controller.robot.transport._joints = np.array(HOME_JOINTS)
    controller.set_fabric_gap(0.30)

    plane = scan_geometry.camera_position(HOME_JOINTS, 0.0306)[2] - 0.30
    assert controller.state.get("teleop_fabric_z") == pytest.approx(plane, abs=1e-6)
    assert [corner[2] for corner in controller.state.get("teleop_fabric_corners")] == [pytest.approx(plane)] * 2
    assert gui_ur5._capture_settings(controller.state)["fabric_z"] == pytest.approx(plane, abs=1e-6)


def test_a_corner_under_the_crosshair_waits_for_the_fabric_height(controller):
    controller.mark_corner_under_crosshair()
    assert controller.state.get("teleop_fabric_corners") == []
    assert "set the fabric height first" in controller.status


def test_measuring_by_parallax_finds_the_fabric_height(controller, monkeypatch):
    """A slide of d moves the picture f*d/h pixels; the measure reads h back from that, with no ruler."""
    pytest.importorskip("mujoco")
    import gaussian_splatting
    import gui_ur5
    import robot_camera
    import scan_geometry
    from PIL import Image

    gap = 0.12
    camera = scan_geometry.camera_position(HOME_JOINTS, 0.0306)
    focal = gaussian_splatting.intrinsics_from_fov(1280, 720, float(controller.state.get("ur5_splat_fov", 55.0)))["fl_x"]
    view = scan_geometry.Viewpoint("here", camera, np.zeros(3))
    monkeypatch.setattr(controller, "check_in_background", lambda: None)
    monkeypatch.setattr(controller, "_fly_over", lambda xy, height, stop=None: view)
    monkeypatch.setattr(controller, "_frame", lambda: Image.new("RGB", (1280, 720)))
    monkeypatch.setattr(robot_camera, "image_shift", lambda before, after: np.array([focal * gui_ur5.MEASURE_STEP / gap, 0.0]))
    controller.robot.transport._joints = np.array(HOME_JOINTS)
    controller.robot.refresh()
    controller._measure_loop()

    assert controller.state.get("teleop_fabric_z") == pytest.approx(camera[2] - gap, abs=1e-6)
    assert controller.camera_height_over_fabric(HOME_JOINTS) == pytest.approx(gap, abs=1e-6)
    assert "measured 12.0 cm" in controller.status


def test_the_capture_height_reaches_the_grid_and_centred_plans(controller, monkeypatch):
    import gui_ur5

    monkeypatch.setattr(controller, "check_in_background", lambda: None)

    controller.set_capture_height(0.11)
    assert gui_ur5._capture_settings(controller.state)["camera_height"] == pytest.approx(0.11)


def test_capturing_from_here_takes_the_height_the_camera_stands_at(controller, monkeypatch):
    pytest.importorskip("mujoco")
    import scan_geometry

    monkeypatch.setattr(controller, "check_in_background", lambda: None)

    camera = scan_geometry.camera_position(HOME_JOINTS, float(controller.state.get("teleop_tool_offset", 0.0306)))
    controller.state.teleop_fabric_z = float(camera[2]) - 0.09
    controller.robot.transport._joints = np.array(HOME_JOINTS)
    controller.adopt_camera_height()

    assert controller.state.get("teleop_camera_height") == pytest.approx(0.09, abs=1e-6)
    assert "Capturing from here" in controller.status


def test_teaching_z_refuses_a_camera_outside_the_offered_range(controller):
    pytest.importorskip("mujoco")
    import scan_geometry

    camera = scan_geometry.camera_position(HOME_JOINTS, float(controller.state.get("teleop_tool_offset", 0.0306)))
    controller.state.teleop_fabric_z = float(camera[2]) - 0.60
    controller.robot.transport._joints = np.array(HOME_JOINTS)
    before = controller.state.get("teleop_camera_height")
    controller.adopt_camera_height()

    assert controller.state.get("teleop_camera_height") == before
    assert "not taken" in controller.status


def test_teacher_settings_persist_with_the_other_ur5_settings():
    from rendering import Camera
    from app_state import AppState

    saved = AppState(Camera(), None).saved_state_keys
    for key in ("teleop_taught_views", "teleop_taught_set", "teleop_grid_rows", "teleop_grid_cols",
                "teleop_camera_height", "teleop_fabric_corners", "teleop_grid_overlap"):
        assert key in saved


def test_teleop_free_drive_and_hold_are_its_tracking_modes():
    pytest.importorskip("mujoco")
    import teleop_transport
    from test_ur5_mode import FakeTeleop

    fake = FakeTeleop()
    transport = teleop_transport.TeleopAppTransport(client_factory=lambda address, path: fake)
    transport.connect("localhost:50052")
    transport.free_drive()
    transport.hold()

    modes = [call[1:] for call in fake.calls if call[0] == "setTrackingMode"]
    assert modes == [("ur5", "free", False), ("ur5", "holding", True)]


# ============================================================================
# The plans
# ============================================================================

def test_the_grid_looks_straight_down_from_the_asked_height():
    pytest.importorskip("mujoco")
    import scan_geometry
    from scipy.spatial.transform import Rotation

    plan = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="grid", grid_rows=2, grid_cols=3, camera_height=0.09, fabric_z=0.20)
    mj, model, data, site_id = scan_geometry.load_model()
    fabric_z = plan.fabric_z

    assert len(plan.viewpoints) + len(plan.skipped) == 6
    assert plan.clear, [point.problems for point in plan.refused]
    for point in plan.viewpoints:
        if not point.adjustment:
            assert point.position[2] - fabric_z == pytest.approx(0.09, abs=1e-6)
        assert Rotation.from_rotvec(point.rotvec).apply([0.0, 0.0, 1.0]) == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)

    flown = [scan_geometry.camera_point(scan_geometry.seed_from(mj, model, data, site_id, q), 0.0306) for q in plan.configs]
    assert np.max(np.abs(np.array(flown) - plan.poses[:, :3])) < scan_geometry.UNREACHED_POS


def test_every_grid_cell_is_its_own_station():
    pytest.importorskip("mujoco")
    import scan_geometry

    plan = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="grid", grid_rows=2, grid_cols=2, fabric_z=0.20)
    assert sorted(plan.station_cells) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for view, station in zip(plan.viewpoints, plan.station_ids):
        assert plan.station_cells[station] == view.cell


def test_a_grid_move_is_one_frame_less_the_overlap_not_a_reconfiguration():
    pytest.importorskip("mujoco")
    import scan_geometry

    plan = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="grid", grid_rows=2, grid_cols=3, camera_height=0.05, grid_overlap=0.3, fabric_z=0.20)
    assert not plan.skipped and not any(point.adjustment for point in plan.viewpoints)
    steps = np.linalg.norm(np.diff(plan.poses[:, :3], axis=0), axis=1)
    assert steps == pytest.approx([scan_geometry.frame_side(0.05, 55.0) * 0.7] * len(steps), abs=1e-6)
    assert max(point.transition_deg for point in plan.viewpoints[1:]) < 30.0


def test_a_sweep_is_the_grid_with_its_stations_closer_together_each_with_a_planned_camera_pose():
    pytest.importorskip("mujoco")
    import scan_geometry

    plan = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="sweep", grid_rows=2, grid_cols=3, camera_height=0.05, sweep_overlap=0.65, fabric_z=0.20)
    assert not plan.skipped and not any(point.adjustment for point in plan.viewpoints)
    steps = np.linalg.norm(np.diff(plan.poses[:, :3], axis=0), axis=1)
    assert steps == pytest.approx([scan_geometry.frame_side(0.05, 55.0) * 0.35] * len(steps), abs=1e-6)
    assert all(point.elevation_deg == 90.0 for point in plan.viewpoints)
    described = plan.capture_metadata(0, 640)
    assert described["camera_position"] == pytest.approx(plan.poses[0, :3]) and described["camera_height_above_fabric_m"] == pytest.approx(0.05)


def test_a_taught_plan_flies_the_recorded_joints_in_order():
    pytest.importorskip("mujoco")
    import scan_geometry

    taught = [{"name": "near", "joints": HOME_JOINTS},
              {"name": "turned", "joints": list(np.array(HOME_JOINTS) + np.radians([8.0, 0.0, 0.0, 0.0, 0.0, 0.0]))}]
    plan = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="taught", taught=taught, fabric_z=0.20)

    assert plan.view_names == ["near", "turned"]
    assert plan.configs == pytest.approx(np.array([entry["joints"] for entry in taught]))
    assert plan.clear


def test_a_taught_plan_with_nothing_taught_refuses():
    pytest.importorskip("mujoco")
    import scan_geometry

    with pytest.raises(RuntimeError, match="No taught views"):
        scan_geometry.build_capture_plan(HOME_JOINTS, pattern="taught", taught=[], fabric_z=0.20)


# ============================================================================
# The fabric plane
# ============================================================================

def test_a_measured_fabric_plane_overrides_the_estimate():
    pytest.importorskip("mujoco")
    import scan_geometry

    assert scan_geometry.fabric_height(None, None, None, None, 0.20, measured=0.123) == 0.123


def test_mark_corner_records_the_camera_point_and_drops_the_plan(controller, monkeypatch):
    import scan_geometry
    from types import SimpleNamespace

    at_home = lambda joints: list(joints) == list(ur5_robot.DEFAULT_HOME_JOINTS)
    monkeypatch.setattr(scan_geometry, "camera_position",
                        lambda joints, tool_offset: np.array([0.0, 0.0, 0.487]) if at_home(joints) else np.array([-0.4, 0.1, 0.141]))
    controller.robot.transport._joints = np.array(HOME_JOINTS) + 0.3
    controller.plan = SimpleNamespace(pattern="grid")
    controller.mark_corner()

    assert controller.state.get("teleop_fabric_corners") == [pytest.approx([-0.4, 0.1, 0.141])]
    assert controller.state.get("teleop_fabric_z") == pytest.approx(0.141)
    assert controller.plan is None

    controller.clear_corners()
    assert controller.state.get("teleop_fabric_corners") == []


def _below_home(drop):
    """The arm with its camera `drop` metres straight below where it is at Home, attitude kept."""
    import scan_geometry

    mj, model, data, site_id = scan_geometry.load_model()
    flange = scan_geometry.seed_from(mj, model, data, site_id, HOME_JOINTS)
    target = np.r_[flange[:3] - [0.0, 0.0, drop], flange[3:6]]
    config, _, _ = scan_geometry.solve_ik(mj, model, data, site_id, target, f"{drop:.2f} below Home")
    return mj, model, data, site_id, config


def test_only_the_move_out_of_where_the_arm_stands_may_start_on_the_fabric():
    """The camera rests on the fabric after Measure fabric. It may lift off - but not stay, and not dig in."""
    pytest.importorskip("mujoco")
    import scan_geometry

    mj, model, data, site_id, low = _below_home(0.30)
    resting = scan_geometry.lowest_point(mj, model, data, site_id, low, 0.0306)
    assert resting < scan_geometry.lowest_point(mj, model, data, site_id, HOME_JOINTS, 0.0306) - 0.10

    _, _, _, _, lifted = _below_home(0.10)
    lift = scan_geometry.Viewpoint("lift", np.zeros(3), np.zeros(3), config=lifted)
    scan_geometry.check_move(mj, model, data, site_id, lift, low, resting, 0.0306, leaving=True)
    assert lift.floor < scan_geometry.FABRIC_CLEARANCE
    assert lift.clear, lift.problems

    planned = scan_geometry.Viewpoint("lift", np.zeros(3), np.zeros(3), config=lifted)
    scan_geometry.check_move(mj, model, data, site_id, planned, low, resting, 0.0306)
    assert planned.floor == scan_geometry.FABRIC_CLEARANCE
    assert not planned.clear

    stay = scan_geometry.Viewpoint("stay", np.zeros(3), np.zeros(3), config=low.copy())
    scan_geometry.check_move(mj, model, data, site_id, stay, low, resting, 0.0306, leaving=True)
    assert not stay.clear
    assert "stands" in "; ".join(stay.problems)

    descent = scan_geometry.Viewpoint("down onto it", np.zeros(3), np.zeros(3), config=low.copy())
    scan_geometry.check_move(mj, model, data, site_id, descent, np.asarray(HOME_JOINTS), resting + 0.01, 0.0306, leaving=True)
    assert descent.floor == scan_geometry.FABRIC_CLEARANCE
    assert not descent.clear
    assert "BELOW the fabric plane" in "; ".join(descent.problems)


def test_no_planned_viewpoint_is_judged_by_where_the_one_before_it_stood():
    """The ratchet: a low view must not lower the bar for the view after it."""
    pytest.importorskip("mujoco")
    import scan_geometry

    _, _, _, _, low = _below_home(0.30)
    plan = scan_geometry.build_capture_plan(low, pattern="orbit", count=5, rings=3, fabric_z=0.20)
    assert all(point.floor == scan_geometry.FABRIC_CLEARANCE for point in plan.viewpoints[1:])
    for point in plan.viewpoints:
        assert point.clear
        assert point.clearance >= point.floor
        assert point.standing >= scan_geometry.FABRIC_CLEARANCE
        assert point.arm_clearance >= scan_geometry.ARM_CLEARANCE
        assert not point.contacts and not point.path_contacts


def test_an_orbit_circles_the_fabric_under_the_camera_not_a_point_inside_the_table():
    pytest.importorskip("mujoco")
    import scan_geometry

    mj, model, data, site_id, low = _below_home(0.30)
    fabric_z = scan_geometry.lowest_point(mj, model, data, site_id, low, 0.0306) - 0.01
    plan = scan_geometry.build_capture_plan(low, pattern="orbit", count=3, rings=1, fabric_z=fabric_z)

    assert plan.subject[2] == pytest.approx(fabric_z)
    assert all(point.position[2] > fabric_z for point in plan.viewpoints)


def test_a_full_orbit_looks_from_every_bearing_around_the_circle():
    """Staggered rings: 3 rings of 6 cover 18 bearings, 20 degrees apart, not 6 bearings three times."""
    pytest.importorskip("mujoco")
    import scan_geometry

    pose = np.array([-0.45, -0.10, 0.40, np.pi, 0.0, 0.0])
    views, _ = scan_geometry.orbit_views(pose, np.array([-0.45, -0.10, 0.20]), 6, 0.20, arc_degrees=360.0, rings=3, max_height=0.19)

    bearings = sorted(round(view.azimuth_deg) % 360 for view in views)
    assert len(set(bearings)) == 18
    assert np.diff(bearings) == pytest.approx([20.0] * 17)


# ============================================================================
# Nearest safe, not all or nothing
# ============================================================================

def test_an_unreachable_view_is_replaced_by_the_nearest_safe_one_or_skipped():
    """One bad view no longer sinks the path: it is moved to the nearest safe spot, or dropped with a reason."""
    pytest.importorskip("mujoco")
    import scan_geometry

    plan = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="orbit", count=6, rings=3, fabric_z=0.20)
    assert plan.viewpoints
    assert plan.clear, [point.problems for point in plan.refused]
    assert len(plan.viewpoints) + len(plan.skipped) == 18
    assert all(reason for _, reason in plan.skipped)
    assert all(point.adjustment == "" or "deg" in point.adjustment or "closer" in point.adjustment for point in plan.viewpoints)


def test_grid_views_out_of_reach_move_up_or_inward_never_down():
    pytest.importorskip("mujoco")
    import scan_geometry

    plan = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="grid", grid_rows=3, grid_cols=3, grid_overlap=0.0, camera_height=0.09, fabric_z=0.20)
    for point in plan.viewpoints:
        assert point.position[2] - plan.fabric_z >= 0.09 - 1e-6
        assert point.adjustment == "" or "higher" in point.adjustment or "nearer the centre" in point.adjustment


def test_a_taught_view_is_never_moved_only_kept_or_skipped():
    pytest.importorskip("mujoco")
    import scan_geometry

    _, _, _, _, low = _below_home(0.30)
    taught = [{"name": "good", "joints": HOME_JOINTS}, {"name": "on the cloth", "joints": list(low)}]
    mj, model, data, site_id = scan_geometry.load_model()
    fabric_z = scan_geometry.lowest_point(mj, model, data, site_id, low, 0.0306) - 0.005

    plan = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="taught", taught=taught, fabric_z=fabric_z)
    assert plan.view_names == ["good"]
    assert [name for name, _ in plan.skipped] == ["on the cloth"]
    assert plan.viewpoints[0].adjustment == ""


def test_mark_corner_refuses_a_reading_taken_at_home_height(controller, monkeypatch):
    """Home looks down at the fabric from above, so a lens at Home's height is not on the fabric."""
    import scan_geometry

    monkeypatch.setattr(scan_geometry, "camera_position", lambda joints, tool_offset: np.array([0.0, 0.0, 0.487]))
    controller.mark_corner()

    assert controller.state.get("teleop_fabric_corners") == []
    assert "not marked" in controller.status

def test_check_refuses_to_plan_from_a_disconnected_arm(controller):
    """A disconnected arm reads as all-zero joints - a pose it is not in - so nothing may be planned from it."""
    controller.robot.disconnect()
    assert controller.build_plan() is False
    assert controller.plan is None
    assert "Connect" in controller.plan_error


# ============================================================================
# A controller found running a program
# ============================================================================

class FakeDashboard:
    """The controller's port-29999 text server, answering the commands the panel sends."""

    def __init__(self, stop_reply="Stopped"):
        import socket

        self.state = "PLAYING"
        self.stop_reply = stop_reply
        self.commands = []
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen()
        self.port = self.server.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                link, _ = self.server.accept()
            except OSError:
                return
            with link:
                link.sendall(b"Connected: Universal Robots Dashboard Server\n")
                command = link.makefile("r").readline().strip()
                self.commands.append(command)
                if command == "stop" and self.stop_reply == "Stopped":
                    self.state = "STOPPED"
                reply = {"stop": self.stop_reply, "programState": f"{self.state} rtde_control",
                         "get loaded program": "Loaded program: /programs/rtde_control.urp"}[command]
                link.sendall((reply + "\n").encode())


@pytest.fixture
def dashboard(monkeypatch):
    fake = FakeDashboard()
    monkeypatch.setattr(ur5_robot, "DASHBOARD_PORT", fake.port)
    yield fake
    fake.server.close()


class FakeScriptPort:
    """The controller's primary interface: whatever program arrives replaces the running one."""

    def __init__(self):
        import socket

        self.received = []
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen()
        self.port = self.server.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                link, _ = self.server.accept()
            except OSError:
                return
            with link:
                self.received.append(link.recv(4096).decode())


def test_a_pendant_program_is_stopped_by_the_dashboard_and_proven_stopped_by_rtde(dashboard):
    reply = ur5_robot.stop_program("127.0.0.1", runtime=lambda ip: 1 if dashboard.state == "STOPPED" else ur5_robot.RUNTIME_STATE_PLAYING)
    assert reply == "Stopped"
    assert dashboard.commands[0] == "stop"


def test_a_stray_script_the_dashboard_cannot_see_is_replaced_by_a_stop(dashboard, monkeypatch):
    """The screenshot case: dashboard STOPPED <unnamed>, RTDE still PLAYING - a script, not a program."""
    script_port = FakeScriptPort()
    monkeypatch.setattr(ur5_robot, "SCRIPT_PORT", script_port.port)
    monkeypatch.setattr(ur5_robot, "STOP_SETTLE_TIMEOUT", 0.5)
    try:
        playing = lambda ip: 1 if script_port.received else ur5_robot.RUNTIME_STATE_PLAYING
        assert "replaced" in ur5_robot.stop_program("127.0.0.1", runtime=playing)
        assert script_port.received == [ur5_robot.STOP_SCRIPT]
        assert "stopj" in ur5_robot.STOP_SCRIPT and ur5_robot.STOP_SCRIPT.strip().endswith("end")
    finally:
        script_port.server.close()


def test_a_script_that_survives_both_stops_is_reported_not_papered_over(dashboard, monkeypatch):
    script_port = FakeScriptPort()
    monkeypatch.setattr(ur5_robot, "SCRIPT_PORT", script_port.port)
    monkeypatch.setattr(ur5_robot, "STOP_SETTLE_TIMEOUT", 0.3)
    try:
        with pytest.raises(RuntimeError, match="still running"):
            ur5_robot.stop_program("127.0.0.1", runtime=lambda ip: ur5_robot.RUNTIME_STATE_PLAYING)
    finally:
        script_port.server.close()


def test_a_dashboard_that_says_stopped_while_rtde_says_playing_is_named_a_network_script():
    left = ur5_robot.ProgramAlreadyRunning("127.0.0.1", "/programs/<unnamed>.urp", "STOPPED <unnamed>")
    assert left.sent_over_network
    assert "script sent over the network" in str(left)

    pendant = ur5_robot.ProgramAlreadyRunning("127.0.0.1", "/programs/scan.urp", "PLAYING scan.urp")
    assert not pendant.sent_over_network
    assert "scan.urp" in str(pendant)


def test_a_busy_controller_names_its_program_and_the_panel_offers_to_stop_it(controller, dashboard, monkeypatch):
    import sys
    import types

    class Receive:
        def __init__(self, ip, *rate):
            pass

        def getRuntimeState(self):
            return ur5_robot.RUNTIME_STATE_PLAYING if dashboard.state == "PLAYING" else 1

        def disconnect(self):
            pass

    monkeypatch.setitem(sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=Receive))
    monkeypatch.setitem(sys.modules, "rtde_control", types.SimpleNamespace(RTDEControlInterface=lambda ip, *rate: (_ for _ in ()).throw(RuntimeError("no arm here"))))
    import gui_ur5

    monkeypatch.setitem(gui_ur5.TRANSPORTS, "rtde", gui_ur5.TRANSPORTS["rtde"]._replace(build=ur5_robot.RTDETransport))
    assert controller.connect("127.0.0.1", "rtde") is False
    assert isinstance(controller.blocking_program, ur5_robot.ProgramAlreadyRunning)
    assert controller.blocking_program.program == "/programs/rtde_control.urp"
    assert "rtde_control.urp" in controller.status
    assert not controller.blocking_program.sent_over_network

    controller.stop_program_and_connect("127.0.0.1", "rtde")
    assert "stop" in dashboard.commands
    assert controller.blocking_program is None
    assert "no arm here" in controller.status


# ============================================================================
# A live link to an arm that is not up
# ============================================================================

def test_teleop_with_its_arm_down_is_linked_says_why_and_can_still_be_planned_from():
    """Connect to TeleopApp succeeds while its UR5e is down: Check must plan, Run must wait, and both must say why."""
    pytest.importorskip("mujoco")
    import teleop_transport
    from test_ur5_mode import FakeTeleop

    fake = FakeTeleop(connected=False)
    transport = teleop_transport.TeleopAppTransport(client_factory=lambda address, path: fake)
    robot = ur5_robot.UR5Robot(transport)
    try:
        assert robot.connect("localhost:50052")
        assert transport.linked()
        assert not robot.connected

        ready, issues = robot.readiness()
        assert not ready
        assert any("its UR5e is not connected" in issue for issue in issues)
    finally:
        robot.disconnect()


def test_a_dropped_rtde_link_is_not_linked():
    transport = ur5_robot.RTDETransport()
    assert not transport.linked()

    class Dead:
        def isConnected(self):
            raise RuntimeError("socket closed")

    transport.receive = Dead()
    assert not transport.linked()


def test_check_names_the_reason_it_has_no_link(controller):
    controller.robot.disconnect()
    controller.robot.state.error = "RTDE receive timed out"
    assert controller.build_plan() is False
    assert "RTDE receive timed out" in controller.plan_error


# ============================================================================
# Find best focus
# ============================================================================

class FocusCamera:
    """A lens whose picture is sharpest at `best`, or the same everywhere when `best` is None."""

    running = True
    status = "Focus fixed"

    def __init__(self, best):
        self.best, self.position, self.requests = best, 0.0, []

    def request_focus(self, auto, value=0.0):
        self.requests.append((auto, value))
        self.position = value

    def capture(self, newer_than=0.0, timeout=1.0):
        from types import SimpleNamespace
        return SimpleNamespace(image=100.0 if self.best is None else 1000.0 - abs(self.position - self.best) * 5.0)


def test_find_focus_locks_the_sharpest_lens_position(controller, monkeypatch):
    import gui_ur5
    import robot_camera

    monkeypatch.setattr(gui_ur5, "FOCUS_SETTLE", 0.0)
    monkeypatch.setattr(robot_camera, "sharpness", lambda image: image)
    controller.camera = FocusCamera(best=97.0)
    controller._focus_loop()

    assert controller.state.get("ur5_camera_autofocus") is False
    assert abs(controller.state.get("ur5_camera_focus") - 97.0) <= gui_ur5.FOCUS_FINE_STEP
    assert controller.camera.requests[-1] == (False, controller.state.get("ur5_camera_focus"))
    assert "locked" in controller.status


def test_find_focus_says_so_when_the_lens_does_not_move(controller, monkeypatch):
    import gui_ur5
    import robot_camera

    monkeypatch.setattr(gui_ur5, "FOCUS_SETTLE", 0.0)
    monkeypatch.setattr(robot_camera, "sharpness", lambda image: image)
    controller.camera = FocusCamera(best=None)
    controller.state.ur5_camera_autofocus = True
    controller._focus_loop()

    assert controller.state.get("ur5_camera_autofocus") is True
    assert controller.camera.requests[-1][0] is True
    assert "did not change" in controller.status
