"""Close, sharp, on-the-fabric captures: the outline, the frame check, focus, blur retakes and MobileSAM.

Everything but the plan tests runs without MuJoCo; the detector is tested against stand-in
ONNX sessions, so the real model files are not needed here.
"""

import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageFilter

import robot_camera
import ur5_robot

HOME_JOINTS = list(ur5_robot.DEFAULT_HOME_JOINTS)


def knit(size=(320, 180), blur=0.0):
    """A striped, stitched stand-in for the cloth: all edges when sharp."""
    rng = np.random.default_rng(3)
    stripes = ((np.arange(size[0])[None, :] // 12) % 3) * 70 + 40
    pixels = np.repeat((stripes + rng.integers(0, 60, (size[1], size[0])))[:, :, None], 3, axis=2).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, "RGB")
    return image.filter(ImageFilter.GaussianBlur(blur)) if blur else image


# ============================================================================
# The fabric outline and what a frame sees of it
# ============================================================================

def square_cloth(width=0.6, depth=0.4, turn_deg=20.0, z=0.10):
    turn = math.radians(turn_deg)
    rotate = np.array([[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]])
    rim = np.array([[0.0, 0.0], [width, 0.0], [width, depth], [0.0, depth]]) @ rotate.T + [-0.7, -0.2]
    return np.c_[rim, np.full(4, z)]


def test_the_outline_is_the_corners_hull_whatever_order_they_were_marked_in():
    import scan_geometry

    corners = square_cloth()
    region = scan_geometry.FabricRegion(corners[[2, 0, 3, 1]])
    assert region.z == pytest.approx(0.10)
    assert len(region.outline) == 4
    signed_area = 0.5 * np.sum(region.outline[:, 0] * np.roll(region.outline[:, 1], -1) - region.outline[:, 1] * np.roll(region.outline[:, 0], -1))
    assert signed_area == pytest.approx(0.24)


def test_a_frame_over_the_middle_is_inside_and_over_a_corner_is_not():
    pytest.importorskip("scipy")
    import scan_geometry

    region = scan_geometry.FabricRegion(square_cloth())
    down = scan_geometry.looking_down(np.array([math.pi, 0.0, 0.0]))
    middle = scan_geometry.footprint(region.centre + [0.0, 0.0, 0.09], down, region.z, 55.0)
    corner = scan_geometry.footprint(np.r_[square_cloth()[0, :2], region.z + 0.09], down, region.z, 55.0)

    assert np.linalg.norm(middle[1] - middle[0]) == pytest.approx(2 * 0.09 * math.tan(math.radians(27.5)), abs=1e-6)
    assert region.overshoot(middle) == 0.0
    assert region.overshoot(corner) == pytest.approx(0.5 * np.linalg.norm(middle[1] - middle[0]) * math.sqrt(2) * math.cos(math.radians(25)), abs=0.02)
    assert region.overshoot(None) == math.inf


def test_a_frame_that_looks_above_the_horizon_has_no_footprint():
    pytest.importorskip("scipy")
    import scan_geometry
    from scipy.spatial.transform import Rotation

    level = Rotation.from_euler("y", 90, degrees=True).as_rotvec()
    assert scan_geometry.footprint(np.array([0.0, 0.0, 0.3]), level, 0.1, 55.0) is None


def test_the_fitted_grid_fills_the_cloth_with_every_frame_on_it():
    pytest.importorskip("scipy")
    import scan_geometry

    region = scan_geometry.FabricRegion(square_cloth())
    pose = np.r_[region.centre + [0.0, 0.0, 0.3], math.pi, 0.0, 0.0]
    views, centre = scan_geometry.grid_views(pose, 3, 3, 0.09, region.z, 0.2, 0.0306, region, 55.0, 0.3)

    side = 2 * 0.09 * math.tan(math.radians(27.5))
    step = side * 0.7
    assert len({view.cell[1] for view in views}) == math.floor((0.6 - side) / step) + 1
    assert len({view.cell[0] for view in views}) == math.floor((0.4 - side) / step) + 1
    for view in views:
        assert view.position[2] == pytest.approx(region.z + 0.09)
        assert region.overshoot(scan_geometry.footprint(view.position, view.rotvec, region.z, 55.0)) <= scan_geometry.FRAME_TOLERANCE
    assert centre == pytest.approx(region.centre)


def test_an_outlined_grid_plan_keeps_every_frame_on_the_cloth():
    pytest.importorskip("mujoco")
    import scan_geometry

    mj, model, data, site_id = scan_geometry.load_model()
    home_camera = scan_geometry.camera_point(scan_geometry.seed_from(mj, model, data, site_id, HOME_JOINTS), 0.0306)
    fabric_z = home_camera[2] - 0.25
    rim = np.array([[-0.15, -0.10], [0.15, -0.10], [0.15, 0.10], [-0.15, 0.10]]) + home_camera[:2]
    corners = np.c_[rim, np.full(4, fabric_z)]

    plan = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="grid", camera_height=0.09, fabric_z=fabric_z,
                                            fabric_outline=corners.tolist(), lens_hfov_deg=55.0, grid_overlap=0.3)
    assert plan.viewpoints
    assert plan.clear, [point.problems for point in plan.refused]
    region = scan_geometry.FabricRegion(corners)
    for point in plan.viewpoints:
        assert point.frame_enforced
        assert point.overshoot <= scan_geometry.FRAME_TOLERANCE
        assert region.overshoot(scan_geometry.footprint(point.position, point.rotvec, fabric_z, 55.0)) <= scan_geometry.FRAME_TOLERANCE


def test_an_orbit_view_that_would_show_the_room_is_refused_when_the_cloth_is_outlined():
    pytest.importorskip("scipy")
    import scan_geometry
    from scipy.spatial.transform import Rotation

    region = scan_geometry.FabricRegion(square_cloth(width=0.2, depth=0.2))
    shallow = scan_geometry.Viewpoint("shallow", region.centre + [0.18, 0.0, 0.05],
                                      Rotation.from_euler("y", -110, degrees=True).as_rotvec())
    scan_geometry._judge_frame(shallow, (region, 55.0), region.z, enforced=True)
    assert shallow.overshoot > scan_geometry.FRAME_TOLERANCE
    assert any("past the fabric" in problem or "horizon" in problem for problem in shallow.problems)


# ============================================================================
# Focus and blur
# ============================================================================

def test_a_blurred_frame_scores_far_below_a_sharp_one():
    assert robot_camera.sharpness(knit()) > 10 * robot_camera.sharpness(knit(blur=3.0))


def test_sharpness_is_measured_at_one_working_width_whatever_the_capture_size():
    large = knit((1280, 720))
    assert robot_camera.sharpness(large) == pytest.approx(robot_camera.sharpness(large.resize((640, 360), Image.BILINEAR)), rel=1e-6)


class FocusBackend(robot_camera.SyntheticCameraBackend):
    def __init__(self):
        super().__init__(size=(64, 48))
        self.focus_calls = []

    def set_focus(self, auto, value):
        self.focus_calls.append((auto, value))


def test_a_focus_request_is_applied_by_the_grab_loop():
    backend = FocusBackend()
    camera = robot_camera.RobotCamera(backend)
    camera.start()
    try:
        camera.request_focus(False, 120.0)
        deadline = time.monotonic() + 2.0
        while not backend.focus_calls and time.monotonic() < deadline:
            time.sleep(0.02)
        assert backend.focus_calls == [(False, 120.0)]
    finally:
        camera.stop()


class ScriptedCamera:
    """Hands out the frames it is given, in order - a camera that is blurred until it is not."""

    running = True
    is_real = False

    def __init__(self, images):
        self.images = list(images)
        self.taken = 0

    def capture(self, newer_than=0.0, timeout=0.0):
        image = self.images[min(self.taken, len(self.images) - 1)]
        self.taken += 1
        return robot_camera.CameraFrame(image=image, timestamp=time.monotonic(), index=self.taken, backend="scripted", is_real=False)

    def describe(self):
        return "scripted"


class OnePointPlan:
    poses = np.zeros((1, 6))
    view_names = ["only"]
    station_ids = [0]
    station_cells = [(0, 0)]

    def assess(self):
        return True, []

    def capture_metadata(self, index, image_width=None):
        return {"capture_pattern": "test"}


def runner_with(camera, tmp_path, **options):
    import ur5_scan

    runner = ur5_scan.RealScanRunner(ur5_robot.UR5Robot(ur5_robot.DryRunTransport()), camera, OnePointPlan(), None,
                                     output_root=tmp_path, **options)
    runner.output_dir = tmp_path
    return runner


def test_a_blurred_capture_is_retaken_and_the_sharpest_kept(tmp_path, monkeypatch):
    import ur5_scan

    monkeypatch.setattr(ur5_scan, "RETAKE_WAIT", 0.0)
    camera = ScriptedCamera([knit(blur=4.0), knit(blur=3.0), knit()])
    frame, score, retakes = runner_with(camera, tmp_path, min_sharpness=150.0)._sharpest_frame(0.0)

    assert retakes == 2
    assert score == pytest.approx(robot_camera.sharpness(knit()))
    assert frame.index == 3


def test_retakes_stop_at_the_limit_and_keep_the_best_of_what_was_taken(tmp_path, monkeypatch):
    import ur5_scan

    monkeypatch.setattr(ur5_scan, "RETAKE_WAIT", 0.0)
    camera = ScriptedCamera([knit(blur=5.0), knit(blur=2.0), knit(blur=6.0), knit()])
    frame, score, retakes = runner_with(camera, tmp_path, min_sharpness=1e9, max_retakes=2)._sharpest_frame(0.0)

    assert camera.taken == 3
    assert frame.index == 2


def test_a_sharp_first_frame_is_not_retaken(tmp_path):
    camera = ScriptedCamera([knit()])
    _, _, retakes = runner_with(camera, tmp_path, min_sharpness=150.0)._sharpest_frame(0.0)
    assert retakes == 0 and camera.taken == 1


# ============================================================================
# Fabric coverage
# ============================================================================

def test_a_capture_that_is_mostly_background_is_flagged_and_its_mask_saved(tmp_path):
    half = np.zeros((180, 320), dtype=bool)
    half[:, :100] = True
    runner = runner_with(ScriptedCamera([knit()]), tmp_path, fabric_check=lambda image: (float(half.mean()), half), min_coverage=0.8)
    frame = ScriptedCamera([knit()]).capture()

    record = runner._store_capture(frame, np.zeros(6), 0, 0, 0, 0, "only", {"sharpness": 900.0})
    assert record["fabric_coverage"] == pytest.approx(half.mean())
    assert record["mostly_background"] is True
    assert record["sharpness"] == 900.0
    assert Image.open(record["fabric_mask"]).size == (320, 180)


def test_a_failing_fabric_check_is_recorded_not_fatal(tmp_path):
    def broken(image):
        raise RuntimeError("model missing")

    runner = runner_with(ScriptedCamera([knit()]), tmp_path, fabric_check=broken)
    record = runner._store_capture(ScriptedCamera([knit()]).capture(), np.zeros(6), 0, 0, 0, 0, "only")
    assert record["fabric_check_error"] == "model missing"
    assert "mostly_background" not in record


def test_the_colour_analysis_leaves_out_mostly_background_frames(tmp_path):
    runner = runner_with(ScriptedCamera([knit()]), tmp_path)
    for row, background in ((0, False), (1, True)):
        path = tmp_path / f"capture_{row}.png"
        knit().save(path)
        runner.capture_records.append({"row": row, "col": 0, "station": 0, "angle": "only", "path": str(path),
                                       "mostly_background": background})
    result = runner.analyze(save_outputs=False)
    assert [int(cell["row"]) for cell in result["cells"]] == [0]


class FakeInput:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class FakeEncoder:
    def __init__(self, rank):
        self.rank = rank
        self.fed = None

    def get_inputs(self):
        return [FakeInput("input_image", [None, None, 3] if self.rank == 3 else [1, 3, 1024, 1024])]

    def run(self, outputs, feeds):
        self.fed = feeds["input_image"]
        return [np.zeros((1, 256, 64, 64), dtype=np.float32)]


class FakeDecoder:
    """Three nested masks, as SAM answers a point: a stripe, a band, the whole cloth."""

    names = ("image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input", "orig_im_size")

    def __init__(self, confidence):
        self.confidence = confidence
        self.fed = None

    def get_inputs(self):
        return [FakeInput(name, []) for name in self.names]

    def run(self, outputs, feeds):
        self.fed = feeds
        height, width = [int(v) for v in feeds["orig_im_size"]]
        masks = np.full((1, 3, height, width), -1.0, dtype=np.float32)
        for index, share in enumerate((0.1, 0.4, 0.9)):
            masks[0, index, :, : int(width * share)] = 1.0
        return [masks, np.array([self.confidence], dtype=np.float32)]


@pytest.mark.parametrize("rank", [3, 4])
def test_the_detector_takes_the_largest_confident_mask_as_the_fabric(rank):
    import fabric_detector

    encoder, decoder = FakeEncoder(rank), FakeDecoder([0.95, 0.9, 0.8])
    share, mask = fabric_detector.coverage(knit((1280, 720)), (encoder, decoder))

    assert share == pytest.approx(0.9, abs=0.01)
    assert mask.shape == (360, 640)
    assert encoder.fed.shape == ((576, 1024, 3) if rank == 3 else (1, 3, 1024, 1024))
    assert decoder.fed["point_coords"][0, 0] == pytest.approx([512.0, 288.0])
    assert list(decoder.fed["point_labels"][0]) == [1.0, -1.0]


def test_the_detector_does_not_trust_a_large_mask_it_is_unsure_of():
    import fabric_detector

    share, _ = fabric_detector.coverage(knit((640, 360)), (FakeEncoder(4), FakeDecoder([0.9, 0.95, 0.3])))
    assert share == pytest.approx(0.4, abs=0.01)


def test_the_fetch_script_pins_both_model_files_by_hash():
    import importlib.util
    import paths

    spec = importlib.util.spec_from_file_location("fetch_mobilesam", paths.SCRIPTS_DIR / "fetch_mobilesam.py")
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    import fabric_detector

    assert {name for name, _, _ in fetch.FILES} == {fabric_detector.ENCODER_FILE, fabric_detector.DECODER_FILE}
    assert all(len(digest) == 64 and size > 1_000_000 for _, size, digest in fetch.FILES)
    assert "/resolve/0d3b403339b4674a82493d5e97964dd78089ddc8" in fetch.REPOSITORY


# ============================================================================
# A run that stops must say why, and the RTDE link must survive a busy laptop
# ============================================================================

class RecordingStorage:
    def __init__(self):
        self.sessions = {}

    def start_robot_session(self, fields):
        self.sessions[fields["session_id"]] = dict(fields)

    def update_robot_session(self, session_id, **fields):
        self.sessions[session_id].update(fields)

    def flush_json_index(self):
        pass

    def record_capture(self, *args, **kwargs):
        pass


def test_a_run_whose_move_fails_is_recorded_as_failed_with_the_reason_and_logged(tmp_path, monkeypatch):
    import ur5_scan

    robot = ur5_robot.UR5Robot(ur5_robot.DryRunTransport(travel_time=0.0))
    robot.connect("127.0.0.1")
    storage = RecordingStorage()
    runner = ur5_scan.RealScanRunner(robot, ScriptedCamera([knit()]), OnePointPlan(), storage, output_root=tmp_path, dwell=0.0)
    assert not runner._stop.is_set()
    monkeypatch.setattr(robot, "move_to_pose", lambda pose: (_ for _ in ()).throw(RuntimeError("RTDE control session dropped")))
    try:
        assert runner.start()
        runner.wait(timeout=10)
    finally:
        robot.disconnect()

    session = storage.sessions[runner.session_id]
    assert session["status"].startswith("failed: ")
    assert "RTDE control session dropped" in session["status"]
    log = (runner.output_dir / ur5_scan.RUN_LOG).read_text()
    assert "Move refused" in log and "RTDE control session dropped" in log


def test_rtde_is_read_at_the_rate_teleop_uses_not_ur_rtdes_500_hz(monkeypatch):
    import sys
    import types

    opened = {}

    class Receive:
        def __init__(self, ip, frequency=-1.0):
            opened["receive"] = frequency

        def getRuntimeState(self):
            return 1

        def disconnect(self):
            pass

    class Control:
        def __init__(self, ip, frequency=-1.0):
            opened["control"] = frequency

        def disconnect(self):
            pass

    monkeypatch.setitem(sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=Receive))
    monkeypatch.setitem(sys.modules, "rtde_control", types.SimpleNamespace(RTDEControlInterface=Control))
    monkeypatch.setitem(sys.modules, "rtde_io", None)
    ur5_robot.RTDETransport().connect("127.0.0.1")
    assert opened == {"receive": ur5_robot.RTDE_FREQUENCY_HZ, "control": ur5_robot.RTDE_FREQUENCY_HZ}
    assert ur5_robot.RTDE_FREQUENCY_HZ <= 125.0


def test_a_dropped_state_stream_is_reopened_and_a_dropped_control_session_blocks_motion():
    transport = ur5_robot.RTDETransport()

    class Receive:
        up = False
        reopened = 0

        def isConnected(self):
            return self.up

        def reconnect(self):
            self.reopened += 1
            self.up = True

        def __getattr__(self, name):
            return lambda: [0.0] * 6 if name.startswith("getActual") else 1

    class Control:
        def isConnected(self):
            return False

    transport.receive, transport.control = Receive(), Control()
    transport.read_state()
    assert transport.receive.reopened == 1
    assert any("control session dropped" in blocker for blocker in transport.blockers())


def test_the_orbit_radius_no_longer_moves_the_estimated_fabric():
    """Subject distance estimates the fabric; the orbit's own radius must not shift it."""
    pytest.importorskip("mujoco")
    import scan_geometry

    near = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="orbit", count=3, rings=1, orbit_radius=0.10)
    far = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="orbit", count=3, rings=1, orbit_radius=0.18)
    assert near.fabric_z == pytest.approx(far.fabric_z)
    for plan, radius in ((near, 0.10), (far, 0.18)):
        for point in plan.viewpoints:
            if not point.adjustment:
                assert np.linalg.norm(point.position - plan.subject) == pytest.approx(radius, abs=1e-3)


def test_an_outlined_orbit_circles_the_fabric_s_middle_not_where_the_camera_looks():
    """The emulator's arm need not stand over the cloth; the orbit must still circle the cloth."""
    pytest.importorskip("mujoco")
    import scan_geometry

    mj, model, data, site_id = scan_geometry.load_model()
    home_camera = scan_geometry.camera_point(scan_geometry.seed_from(mj, model, data, site_id, HOME_JOINTS), 0.0306)
    fabric_z = home_camera[2] - 0.25
    rim = np.array([[-0.20, -0.15], [0.20, -0.15], [0.20, 0.15], [-0.20, 0.15]]) + home_camera[:2] + [0.08, -0.05]
    corners = np.c_[rim, np.full(4, fabric_z)]

    plan = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="orbit", count=4, rings=1, orbit_radius=0.10,
                                            fabric_z=fabric_z, fabric_outline=corners.tolist())
    assert plan.subject == pytest.approx(scan_geometry.FabricRegion(corners).centre)


# ============================================================================
# Centred on the fabric, tilting only as far as the frame stays on it
# ============================================================================

def test_centred_views_aim_at_the_middle_and_never_lose_the_cloth():
    pytest.importorskip("scipy")
    import scan_geometry
    from scipy.spatial.transform import Rotation

    region = scan_geometry.FabricRegion(square_cloth(width=0.40, depth=0.30, turn_deg=0.0, z=0.24))
    pose = np.r_[region.centre + [0.0, 0.0, 0.3], math.pi, 0.0, 0.0]
    views, _ = scan_geometry.centred_views(pose, region.centre, 6, 0.18, 2, 35.0, region, region.z, 55.0)

    assert views[0].name == "straight down" and views[0].elevation_deg == 90.0
    for view in views:
        lens = Rotation.from_rotvec(view.rotvec).apply([0.0, 0.0, 1.0])
        toward = (region.centre - view.position) / np.linalg.norm(region.centre - view.position)
        assert lens == pytest.approx(toward, abs=1e-6)
        assert region.overshoot(scan_geometry.footprint(view.position, view.rotvec, region.z, 55.0)) <= scan_geometry.FRAME_TOLERANCE
        assert 90.0 - view.elevation_deg <= 35.0 + 1e-9


def test_centred_views_tilt_less_where_the_cloth_is_narrow():
    pytest.importorskip("scipy")
    import scan_geometry

    region = scan_geometry.FabricRegion(square_cloth(width=0.60, depth=0.25, turn_deg=0.0, z=0.24))
    pose = np.r_[region.centre + [0.0, 0.0, 0.3], math.pi, 0.0, 0.0]
    views, _ = scan_geometry.centred_views(pose, region.centre, 8, 0.15, 1, 60.0, region, region.z, 55.0)
    tilts = {}
    for view in views[1:]:
        across = abs(math.cos(math.radians(view.azimuth_deg) + math.atan2(pose[1] - region.centre[1], pose[0] - region.centre[0])))
        tilts.setdefault("along" if across > 0.9 else "across" if across < 0.4 else "diagonal", []).append(90.0 - view.elevation_deg)
    assert max(tilts["along"]) > max(tilts["across"])


def test_an_outlined_centred_plan_is_flown_centred_and_on_the_cloth():
    pytest.importorskip("mujoco")
    import scan_geometry

    mj, model, data, site_id = scan_geometry.load_model()
    home_camera = scan_geometry.camera_point(scan_geometry.seed_from(mj, model, data, site_id, HOME_JOINTS), 0.0306)
    fabric_z = home_camera[2] - 0.25
    corners = np.c_[np.array([[-0.2, -0.15], [0.2, -0.15], [0.2, 0.15], [-0.2, 0.15]]) + home_camera[:2], np.full(4, fabric_z)]

    plan = scan_geometry.build_capture_plan(HOME_JOINTS, pattern="centred", count=6, rings=2, camera_height=0.12,
                                            fabric_z=fabric_z, fabric_outline=corners.tolist())
    assert plan.viewpoints and plan.clear
    assert plan.subject == pytest.approx(scan_geometry.FabricRegion(corners).centre)
    for index, point in enumerate(plan.viewpoints):
        assert point.overshoot <= scan_geometry.FRAME_TOLERANCE
        described = plan.capture_metadata(index, 1280)
        assert described["camera_height_above_fabric_m"] > 0.0
        assert described["fabric_mm_per_pixel"] == pytest.approx(described["fabric_across_frame_mm"] / 1280.0)


# ============================================================================
# Every capture described
# ============================================================================

def test_capture_metadata_says_how_high_how_tilted_and_at_what_detail():
    pytest.importorskip("scipy")
    import scan_geometry

    down = scan_geometry.looking_down(np.array([math.pi, 0.0, 0.0]))
    point = scan_geometry.Viewpoint("over", np.array([0.0, 0.5, 0.31]), down)
    plan = scan_geometry.CapturePlan(poses=np.zeros((1, 6)), view_names=["over"], station_ids=[0], station_cells=[(0, 0)],
                                     viewpoints=[point], subject=np.array([0.0, 0.5, 0.24]), pattern="grid", fabric_z=0.24)
    described = plan.capture_metadata(0, 1280)

    assert described["camera_height_above_fabric_m"] == pytest.approx(0.07)
    assert described["tilt_from_vertical_deg"] == pytest.approx(0.0, abs=1e-6)
    across = 2 * 0.07 * math.tan(math.radians(27.5)) * 1000.0
    assert described["fabric_across_frame_mm"] == pytest.approx(across, rel=1e-6)
    assert described["fabric_mm_per_pixel"] == pytest.approx(across / 1280.0)


def test_every_capture_gets_its_details_a_detection_image_and_the_session_a_sheet_and_table(tmp_path):
    import json as json_module
    import ur5_scan

    mask = np.zeros((180, 320), dtype=bool)
    mask[:, 40:280] = True
    runner = runner_with(ScriptedCamera([knit()]), tmp_path, fabric_check=lambda image: (float(mask.mean()), mask))
    runner.session_id = "test_session"
    for index in range(3):
        record = runner._store_capture(ScriptedCamera([knit()]).capture(), np.zeros(6), 0, 0, 0, index, "only", {"sharpness": 800.0 + index})

    details = json_module.loads(Path(record["path"]).with_suffix(".json").read_text())
    assert details["sharpness"] == 802.0
    assert details["fabric_coverage"] == pytest.approx(0.75)
    assert details["fabric_centre_offset"] == pytest.approx([0.0, 0.0], abs=0.01)
    assert Path(details["detected_image"]).exists()

    runner._write_session_summary()
    assert (tmp_path / ur5_scan.CONTACT_SHEET).exists()
    table = (tmp_path / ur5_scan.QUALITY_TABLE).read_text().splitlines()
    assert table[0].split(",")[:2] == ["target_index", "capture_view"]
    assert len(table) == 4


def test_the_detection_image_dims_what_is_not_fabric():
    import fabric_detector

    image = Image.new("RGB", (200, 100), (200, 200, 200))
    mask = np.zeros((10, 20), dtype=bool)
    mask[:, :10] = True
    shown = np.asarray(fabric_detector.annotate(image, mask))
    assert shown[80, 20].mean() > 150
    assert shown[80, 180].mean() < 120
