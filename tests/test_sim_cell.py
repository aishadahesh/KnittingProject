"""TeleopApp's emulator as the simulator: the table and knitted fabric props, and its gripper camera.

Against a stand-in TeleopApp that records the props it is sent and renders nothing but a flat
frame, so no TeleopApp is needed; the MuJoCo-backed parts skip without MuJoCo.
"""

import numpy as np
import pytest
from PIL import Image

import sim_cell
import ur5_robot


class FakeClient:
    """TeleopApp's DynamicRpcServer: props kept by name, and a camera that writes a 4:3 frame."""

    def __init__(self, cameras=("Gripper camera",)):
        self.props = {"workbench": {"shape": "box"}}
        self.cameras = list(cameras)
        self.calls = []

    def call(self, method, *args):
        self.calls.append((method, *args))
        if method == "getProps":
            return dict(self.props)
        if method == "setProp":
            self.props[args[0]] = args[1]
        if method == "removeProp":
            del self.props[args[0]]
        if method == "describe":
            return {"targets": {"ur5": {"cameras": self.cameras}}}
        if method == "captureCameraImage":
            Image.new("RGB", (800, 600), (40, 90, 160)).save(args[2])
            return {"path": args[2], "hFov": 0.9599310886, "aspect": 16.0 / 9.0}
        return None


class FakeTransport:
    def __init__(self, rotation=None, on_hardware=False):
        self.client = FakeClient()
        self.target = "ur5"
        self.tool_offset = 0.0306
        self.on_hardware = on_hardware
        self.rotation = rotation

    def measured_joints(self):
        return list(ur5_robot.DEFAULT_HOME_JOINTS)

    def app_tool_point(self):
        import scan_geometry

        return list(self.rotation @ scan_geometry.camera_position(self.measured_joints(), self.tool_offset))

    def cameras(self):
        return self.client.call("describe")["targets"]["ur5"]["cameras"]

    def linked(self):
        return True


# ============================================================================
# The frame TeleopApp's world is in
# ============================================================================

@pytest.mark.parametrize("name", sorted(sim_cell.FRAME_CANDIDATES))
def test_the_frame_map_is_the_one_teleop_s_tool_point_agrees_with(name):
    pytest.importorskip("mujoco")
    assert sim_cell.frame_map(FakeTransport(sim_cell.FRAME_CANDIDATES[name])) is sim_cell.FRAME_CANDIDATES[name]


def test_a_frame_no_candidate_explains_is_refused_before_anything_is_placed():
    pytest.importorskip("mujoco")
    transport = FakeTransport(np.eye(3))
    with pytest.raises(RuntimeError, match="does not match"):
        sim_cell.build(transport, Image.new("RGB", (400, 300)))
    assert not any(name.startswith(sim_cell.PROP_PREFIX) for name in transport.client.props)


def test_every_candidate_is_a_proper_rotation_with_y_up():
    for rotation in sim_cell.FRAME_CANDIDATES.values():
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        assert rotation @ np.array([0.0, 0.0, 1.0]) == pytest.approx([0.0, 1.0, 0.0])


# ============================================================================
# The cell
# ============================================================================

def layout():
    return {"centre": np.array([-0.14, 0.49, 0.24]), "width": 0.40, "depth": 0.30,
            "corners": [[-0.34, 0.34, 0.24], [0.06, 0.34, 0.24], [0.06, 0.64, 0.24], [-0.34, 0.64, 0.24]]}


def test_the_table_top_sits_just_under_the_fabric_and_its_legs_stand_on_the_floor(tmp_path):
    rotation = sim_cell.FRAME_CANDIDATES["base_link"]
    props = sim_cell.props_for(layout(), rotation, tmp_path / "fabric.obj")
    back = rotation.T

    table = props[f"{sim_cell.PROP_PREFIX}table"]
    table_top = (back @ table["position"])[2] + sim_cell.TABLE_THICKNESS / 2.0
    assert table_top == pytest.approx(0.24 - sim_cell.FABRIC_LIFT)
    assert table["size"][1] == sim_cell.TABLE_THICKNESS

    legs = [spec for name, spec in props.items() if "leg" in name]
    assert len(legs) == 4
    for leg in legs:
        bottom = (back @ leg["position"])[2] - leg["size"][1] / 2.0
        assert bottom == pytest.approx(0.0, abs=1e-9)

    fabric = props[f"{sim_cell.PROP_PREFIX}fabric"]
    assert fabric["shape"] == "mesh" and "color" not in fabric
    assert back @ fabric["position"] == pytest.approx(layout()["centre"])


def test_the_fabric_mesh_is_a_textured_quad_facing_up(tmp_path):
    mesh = sim_cell.write_fabric_mesh(Image.new("RGB", (4000, 3000), (200, 30, 30)), 0.40, 0.30, tmp_path / "cell")
    obj = mesh.read_text()
    assert "mtllib fabric.mtl" in obj and "vn 0 1 0" in obj
    assert "map_Kd fabric.png" in (tmp_path / "cell" / "fabric.mtl").read_text()
    assert max(Image.open(tmp_path / "cell" / "fabric.png").size) == sim_cell.TEXTURE_MAX_SIDE

    corners = [np.array([float(v) for v in line.split()[1:]]) for line in obj.splitlines() if line.startswith("v ")]
    normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    assert normal[1] > 0 and abs(normal[0]) < 1e-12 and abs(normal[2]) < 1e-12
    assert np.ptp([c[0] for c in corners]) == pytest.approx(0.40)
    assert np.ptp([c[2] for c in corners]) == pytest.approx(0.30)


def test_building_replaces_its_own_props_and_leaves_the_rest_and_returns_the_corners(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    import paths

    monkeypatch.setattr(paths, "SIM_CELL_DIR", tmp_path)
    transport = FakeTransport(sim_cell.FRAME_CANDIDATES["base_link"])
    transport.client.props[f"{sim_cell.PROP_PREFIX}old leg"] = {"shape": "box"}

    corners = sim_cell.build(transport, Image.new("RGB", (400, 300)))
    names = set(transport.client.props)
    assert "workbench" in names
    assert f"{sim_cell.PROP_PREFIX}old leg" not in names
    assert {f"{sim_cell.PROP_PREFIX}table", f"{sim_cell.PROP_PREFIX}fabric"} <= names
    assert len(corners) == 4 and len({round(c[2], 9) for c in corners}) == 1
    span = np.ptp(np.array(corners)[:, :2], axis=0)
    assert span == pytest.approx([sim_cell.FABRIC_WIDTH, sim_cell.FABRIC_WIDTH * 0.75])


# ============================================================================
# The gripper camera
# ============================================================================

def test_teleop_s_camera_frames_are_its_renders_resized_to_the_camera_s_aspect(tmp_path, monkeypatch):
    import paths
    import teleop_transport

    monkeypatch.setattr(paths, "SIM_CELL_DIR", tmp_path)
    transport = FakeTransport()
    backend = teleop_transport.TeleopCameraBackend(transport, "Gripper camera", size=(1280, 960), interval=0.0)
    backend.open()
    frame = backend.read()

    assert frame.shape == (720, 1280, 3)
    assert tuple(frame[0, 0]) == (40, 90, 160)
    assert backend.is_real is False
    assert transport.client.calls[-1][:3] == ("captureCameraImage", "ur5", "Gripper camera")


def test_a_camera_teleop_does_not_mount_is_refused_on_open(tmp_path, monkeypatch):
    import paths
    import teleop_transport

    monkeypatch.setattr(paths, "SIM_CELL_DIR", tmp_path)
    transport = FakeTransport()
    transport.client.cameras = []
    with pytest.raises(RuntimeError, match="no camera"):
        teleop_transport.TeleopCameraBackend(transport, "Gripper camera").open()


def test_the_simulated_cell_is_refused_while_teleop_drives_the_real_arm():
    from types import SimpleNamespace
    import gui_ur5

    controller = SimpleNamespace(robot=SimpleNamespace(transport=FakeTransport(on_hardware=True)), status="", plan=None, state=None)
    gui_ur5.UR5ModeController.build_sim_cell(controller, renderer=None)
    assert "refused" in controller.status and "real arm" in controller.status


# ============================================================================
# Finding the fabric with the camera, and centring over it
# ============================================================================

def knit_on_table(share, size=(640, 360)):
    """A frame whose left `share` is knit-like texture and the rest a flat table."""
    rng = np.random.default_rng(1)
    pixels = np.full((size[1], size[0], 3), (115, 84, 56), dtype=np.uint8)
    cut = int(size[0] * share)
    pixels[:, :cut] = rng.integers(0, 255, (size[1], cut, 3), dtype=np.uint8)
    return Image.fromarray(pixels, "RGB")


@pytest.mark.parametrize("share", [0.25, 0.5, 0.75])
def test_the_texture_test_finds_the_knit_and_its_middle(share):
    import fabric_detector

    offset, found = fabric_detector.fabric_offset(knit_on_table(share))
    assert found == pytest.approx(share, abs=0.04)
    assert offset[0] == pytest.approx(share / 2.0 - 0.5, abs=0.02)
    assert offset[1] == pytest.approx(0.0, abs=0.02)


def test_a_frame_with_no_knit_in_it_says_so():
    import fabric_detector

    with pytest.raises(RuntimeError, match="no fabric in view"):
        fabric_detector.fabric_offset(Image.new("RGB", (640, 360), (90, 120, 160)))


def test_centring_finds_the_fabric_on_a_camera_mounted_any_way_round(tmp_path, monkeypatch):
    """The image moves opposite and turned relative to the arm; the probes learn it, the steps converge."""
    import gui_ur5
    from rendering import Camera
    from app_state import AppState

    class StubRenderer:
        def __getattr__(self, name):
            return lambda *a, **k: None

    state = AppState(Camera(), StubRenderer())
    state.project_root = str(tmp_path)
    corners = [[-0.20, 0.35, 0.24], [0.20, 0.35, 0.24], [0.20, 0.65, 0.24], [-0.20, 0.65, 0.24]]
    state.teleop_fabric_corners = corners
    controller = gui_ur5.UR5ModeController(state, None, None)

    true_centre = np.array([0.05, 0.47])
    turn = np.radians(70.0)
    mount = -3.2 * np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]]) @ np.diag([1.0, -1.0])
    here = {"xy": None}
    monkeypatch.setattr(controller, "_fly_over", lambda xy, height, stop=None: here.__setitem__("xy", np.asarray(xy, dtype=float)))
    monkeypatch.setattr(controller, "_look", lambda: (mount @ (true_centre - here["xy"]), 0.6))

    controller._centre_loop()

    assert "Centred on the fabric" in controller.status, controller.status
    assert here["xy"] == pytest.approx(true_centre, abs=0.01)
    moved = np.array(state.get("teleop_fabric_corners"))[:, :2].mean(axis=0)
    assert moved == pytest.approx(true_centre, abs=0.01)


def test_switching_connection_stops_a_camera_read_through_teleop(tmp_path, monkeypatch):
    """Left running, it would keep photographing the simulator during a real-arm run."""
    import gui_ur5
    import paths
    import robot_camera
    import teleop_transport
    from rendering import Camera
    from app_state import AppState

    class StubRenderer:
        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr(paths, "SIM_CELL_DIR", tmp_path)
    state = AppState(Camera(), StubRenderer())
    state.project_root = str(tmp_path)
    controller = gui_ur5.UR5ModeController(state, None, None)
    controller.camera = robot_camera.RobotCamera(teleop_transport.TeleopCameraBackend(FakeTransport(), "Gripper camera", interval=0.0))
    assert controller.camera.start()

    monkeypatch.setitem(gui_ur5.TRANSPORTS, "rtde", gui_ur5.TRANSPORTS["rtde"]._replace(build=ur5_robot.DryRunTransport))
    monkeypatch.setattr(robot_camera, "list_devices", lambda: [])
    try:
        assert controller.connect("127.0.0.1", "rtde")
        assert controller.camera is None
    finally:
        controller.close()
