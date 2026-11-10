"""Straight Home, the captured-images gallery, the splatting preview and the splat viewer.

The preview is tested by a round trip: a known pattern is "photographed" through the very camera
model the export writes, and the stitched map must give the pattern back.
"""

import json
import math

import numpy as np
import pytest
from PIL import Image

import ur5_robot

HOME_JOINTS = list(ur5_robot.DEFAULT_HOME_JOINTS)


# ============================================================================
# Straight
# ============================================================================

def test_the_straight_heading_squares_the_image_with_the_least_turn():
    pytest.importorskip("scipy")
    import scan_geometry
    from scipy.spatial.transform import Rotation

    corners = np.array([[0.0, 0.0, 0.2], [0.6, 0.0, 0.2], [0.6, 0.3, 0.2], [0.0, 0.3, 0.2]])
    region = scan_geometry.FabricRegion(corners)
    for current_deg, expected in ((10.0, [1.0, 0.0]), (80.0, [0.0, 1.0]), (170.0, [-1.0, 0.0]), (-100.0, [0.0, -1.0])):
        rotvec = (Rotation.from_euler("z", current_deg, degrees=True) * Rotation.from_euler("x", 180, degrees=True)).as_rotvec()
        assert scan_geometry.straight_heading(region, rotvec) == pytest.approx(expected)
        assert scan_geometry.straight_heading(None, rotvec) == pytest.approx(expected)


def test_a_grid_is_photographed_square_to_the_cloth():
    pytest.importorskip("scipy")
    import scan_geometry
    from scipy.spatial.transform import Rotation

    turn = math.radians(25.0)
    edge = np.array([math.cos(turn), math.sin(turn)])
    rim = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.3], [0.0, 0.3]]) @ np.array([[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]]).T
    region = scan_geometry.FabricRegion(np.c_[rim, np.full(4, 0.2)])
    pose = np.r_[region.centre + [0.0, 0.0, 0.3], math.pi, 0.0, 0.0]
    views, _ = scan_geometry.grid_views(pose, 1, 1, 0.07, 0.2, 0.2, 0.0306, region, 55.0, 0.3)
    for view in views:
        image_x = Rotation.from_rotvec(view.rotvec).as_matrix()[:2, 0]
        assert abs(abs(float(np.dot(image_x, edge))) - 1.0) < 1e-6 or abs(float(np.dot(image_x, edge))) < 1e-6


def test_straight_home_keeps_home_s_camera_and_points_it_exactly_down():
    pytest.importorskip("mujoco")
    import scan_geometry
    from scipy.spatial.transform import Rotation

    joints = scan_geometry.straight_home(HOME_JOINTS, 0.0306)
    mj, model, data, site_id = scan_geometry.load_model()
    before = scan_geometry.camera_point(scan_geometry.seed_from(mj, model, data, site_id, HOME_JOINTS), 0.0306)
    flange = scan_geometry.seed_from(mj, model, data, site_id, joints)
    after = scan_geometry.camera_point(flange, 0.0306)

    assert after == pytest.approx(before, abs=scan_geometry.UNREACHED_POS)
    lens = Rotation.from_rotvec(flange[3:6]).apply([0.0, 0.0, 1.0])
    assert lens == pytest.approx([0.0, 0.0, -1.0], abs=0.05)
    assert np.max(np.abs(np.asarray(joints) - np.asarray(HOME_JOINTS))) < math.radians(100.0)


# ============================================================================
# The splatting preview, by round trip
# ============================================================================

def pattern(size=400):
    y, x = np.mgrid[0:size, 0:size]
    image = np.stack([(x * 255 // size), (y * 255 // size), ((x // 25 + y // 25) % 2) * 255], axis=-1)
    return image.astype(np.uint8)


def photograph(texture, camera, rotvec, fabric_z, extent, fov, size=(320, 240)):
    """What a pinhole camera at `camera` sees of `texture` laid on the plane over [-extent, extent]^2."""
    import gaussian_splatting
    from scipy.spatial.transform import Rotation

    width, height = size
    k = gaussian_splatting.intrinsics_from_fov(width, height, fov)
    v, u = np.mgrid[0:height, 0:width]
    rays = np.stack([(u - k["cx"]) / k["fl_x"], (v - k["cy"]) / k["fl_y"], np.ones_like(u, dtype=float)], axis=-1)
    world = rays @ Rotation.from_rotvec(rotvec).as_matrix().T
    reach = (fabric_z - camera[2]) / world[..., 2]
    hit = camera[:2] + world[..., :2] * reach[..., None]
    side = texture.shape[0]
    col = np.clip(((hit[..., 0] + extent) / (2 * extent) * side).astype(int), 0, side - 1)
    row = np.clip(((extent - hit[..., 1]) / (2 * extent) * side).astype(int), 0, side - 1)
    return Image.fromarray(texture[row, col], "RGB")


def test_the_preview_gives_back_the_fabric_its_captures_photographed(tmp_path):
    pytest.importorskip("scipy")
    import gaussian_splatting
    import scan_geometry

    texture, fabric_z, extent, fov = pattern(), 0.20, 0.10, 55.0
    records = []
    for index, (x, y) in enumerate(((-0.04, -0.03), (0.04, -0.03), (0.04, 0.03), (-0.04, 0.03))):
        camera = np.array([x, y, fabric_z + 0.09])
        rotvec = scan_geometry.looking_down_along([1.0, 0.0])
        path = tmp_path / f"capture_{index}.png"
        photograph(texture, camera, rotvec, fabric_z, extent, fov).save(path)
        records.append({"path": str(path), "camera_position": camera.tolist(), "target_position": [*camera, *rotvec]})

    outline = np.array([[-0.06, -0.06], [0.06, -0.06], [0.06, 0.06], [-0.06, 0.06]])
    mosaic, covered = gaussian_splatting.preview_mosaic(records, fabric_z, fov, np.c_[outline, np.full(4, fabric_z)])
    assert covered == pytest.approx(1.0, abs=0.02)

    pixels = np.asarray(mosaic, dtype=float)
    (low_x, low_y), (high_x, high_y), metres_per_px = mosaic.info["world_bounds"]
    side = texture.shape[0]
    misses = []
    for x, y in np.random.default_rng(2).uniform(-0.05, 0.05, (60, 2)):
        seen = pixels[int((high_y - y) / metres_per_px), int((x - low_x) / metres_per_px)]
        truth = texture[int((extent - y) / (2 * extent) * side), int((x + extent) / (2 * extent) * side)].astype(float)
        misses.append(np.abs(seen - truth)[:2].max())
    assert np.median(misses) < 12.0


def test_the_export_seeds_the_splat_with_the_photographed_fabric_lying_on_its_plane(tmp_path):
    pytest.importorskip("scipy")
    import gaussian_splatting
    import scan_geometry

    texture, fabric_z, extent, fov = pattern(), 0.20, 0.10, 55.0
    records = []
    for index, (x, y) in enumerate(((-0.04, -0.03), (0.04, -0.03), (0.04, 0.03), (-0.04, 0.03))):
        camera = np.array([x, y, fabric_z + 0.09])
        rotvec = scan_geometry.looking_down_along([1.0, 0.0])
        path = tmp_path / f"capture_{index}.png"
        photograph(texture, camera, rotvec, fabric_z, extent, fov).save(path)
        records.append({"path": str(path), "camera_position": camera.tolist(), "target_position": [*camera, *rotvec],
                        "camera_height_above_fabric_m": 0.09})

    dataset = gaussian_splatting.export_dataset(records, tmp_path / "ds", fov_degrees=fov)
    assert json.loads(dataset.transforms_path.read_text())["ply_file_path"] == gaussian_splatting.SEED_PLY_NAME

    raw = (tmp_path / "ds" / gaussian_splatting.SEED_PLY_NAME).read_bytes()
    body = raw.index(b"end_header\n") + len(b"end_header\n")
    points = np.frombuffer(raw[body:], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    assert 1000 < len(points) <= gaussian_splatting.SEED_MAX_POINTS
    assert np.allclose(points["z"], fabric_z)
    side = texture.shape[0]
    inner = points[(np.abs(points["x"]) < 0.05) & (np.abs(points["y"]) < 0.05)]
    truth = texture[((extent - inner["y"]) / (2 * extent) * side).astype(int), ((inner["x"] + extent) / (2 * extent) * side).astype(int)]
    assert np.median(np.abs(np.stack([inner["r"], inner["g"]], axis=1).astype(float) - truth[:, :2])) < 12.0


def test_captures_without_a_fabric_height_export_no_seed_and_say_so(tmp_path):
    pytest.importorskip("scipy")
    import gaussian_splatting

    path = tmp_path / "capture.png"
    Image.new("RGB", (64, 48), (90, 90, 90)).save(path)
    dataset = gaussian_splatting.export_dataset([{"path": str(path), "target_position": [0.0, 0.4, 0.3, 3.14159, 0.0, 0.0]}], tmp_path / "ds")

    assert "ply_file_path" not in json.loads(dataset.transforms_path.read_text())
    assert not (tmp_path / "ds" / gaussian_splatting.SEED_PLY_NAME).exists()
    assert any("random points" in warning for warning in dataset.warnings)


def test_a_planned_capture_s_pose_is_the_camera_with_nothing_added():
    import gaussian_splatting

    guess = gaussian_splatting.hand_eye_matrix(0.12)
    planned = {"camera_position": [0.1, 0.2, 0.3], "target_position": [0.1, 0.2, 0.3, 3.1, 0.0, 0.0]}
    pose, hand_eye = gaussian_splatting.capture_camera_pose(planned, guess)
    assert pose == pytest.approx([0.1, 0.2, 0.3, 3.1, 0.0, 0.0])
    assert hand_eye == pytest.approx(np.eye(4))

    legacy = {"actual_tcp_pose": [0.0, 0.0, 0.5, 3.1, 0.0, 0.0]}
    pose, hand_eye = gaussian_splatting.capture_camera_pose(legacy, guess)
    assert hand_eye is guess


# ============================================================================
# The splat viewer
# ============================================================================

def write_splat(path, positions, dc, opacity_logit):
    properties = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    header = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(positions)}\n"
              + "".join(f"property float {name}\n" for name in properties) + "end_header\n")
    table = np.zeros(len(positions), dtype=[(name, "<f4") for name in properties])
    table["x"], table["y"], table["z"] = positions.T
    table["f_dc_0"], table["f_dc_1"], table["f_dc_2"] = dc.T
    table["opacity"] = opacity_logit
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        table.tofile(handle)


def test_a_trained_splat_is_read_and_seen_in_its_own_colours(tmp_path):
    import gaussian_splatting

    rng = np.random.default_rng(0)
    positions = rng.uniform(-0.1, 0.1, (5000, 3)).astype(np.float32)
    red = np.tile(np.array([(1.0 - 0.5) / gaussian_splatting.SH_C0, (0.0 - 0.5) / gaussian_splatting.SH_C0, (0.0 - 0.5) / gaussian_splatting.SH_C0]), (5000, 1))
    write_splat(tmp_path / "splat.ply", positions, red, np.full(5000, 4.0))

    splat = gaussian_splatting.load_splat_ply(tmp_path / "splat.ply")
    assert splat["positions"].shape == (5000, 3)
    assert splat["colours"][0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-5)
    assert splat["opacity"][0] == pytest.approx(1.0 / (1.0 + math.exp(-4.0)))

    image = np.asarray(gaussian_splatting.render_splat(splat, 30.0, 45.0, 1.0, (200, 150)))
    lit = image[image.sum(axis=-1) > 100]
    assert len(lit) > 500
    # Soft-edged blobs: the lit pixels include each blob's fading rim, so red averages below full.
    assert lit[:, 0].mean() > 150 and lit[:, 1].mean() < 40


# ============================================================================
# The gallery reads what a run left
# ============================================================================

def test_a_session_s_captures_are_its_sidecars_in_order_or_its_own_images(tmp_path):
    import gui_captures

    with_details = tmp_path / "ur5_a"
    with_details.mkdir()
    for index in (2, 1):
        image = with_details / f"ur5_{index:04d}.png"
        Image.new("RGB", (8, 8)).save(image)
        (with_details / f"ur5_{index:04d}.json").write_text(json.dumps({"path": str(image), "sharpness": index}))
    (with_details / "captures_summary.json").write_text(json.dumps({"not": "a capture"}))
    assert [record["sharpness"] for record in gui_captures.load_session(with_details)] == [1, 2]

    bare = tmp_path / "ur5_b"
    bare.mkdir()
    for name in ("ur5_0001.png", "ur5_0001_fabric_mask.png", "ur5_0001_detected.png", "contact_sheet.png"):
        Image.new("RGB", (8, 8)).save(bare / name)
    assert [record["path"].split("\\")[-1].split("/")[-1] for record in gui_captures.load_session(bare)] == ["ur5_0001.png"]
    assert gui_captures.sessions(tmp_path) == ["ur5_b", "ur5_a"]



def test_a_combined_session_finds_its_scans_under_robot_scans_whoever_wrote_it(tmp_path, monkeypatch):
    import gui_captures
    import paths

    scans = tmp_path / "robot_scans"
    capture = scans / "ur5_a"
    capture.mkdir(parents=True)
    image = capture / "ur5_0001.png"
    Image.new("RGB", (8, 8)).save(image)
    (capture / "ur5_0001.json").write_text(json.dumps({"path": str(image)}))
    combined = scans / "ur5_a_rescan"
    combined.mkdir()
    # One entry as it is written now (a name), one as another machine wrote it (a full path there).
    (combined / "combined.json").write_text(json.dumps({"sessions": ["ur5_a", r"C:\Users\someone\robot_scans\ur5_a"]}))
    monkeypatch.setattr(paths, "ROBOT_SCANS_DIR", scans)
    monkeypatch.setattr(paths.scan_session, "__defaults__", (scans,))

    assert paths.scan_session("C:/Users/someone/projects/robot_scans/ur5_a") == capture
    records = gui_captures.load_session(combined)
    assert [r["combined_from"] for r in records] == ["ur5_a", "ur5_a"]

# ============================================================================
# The camera's image turn, measured from how the picture slides
# ============================================================================

def photograph_turned(texture, camera, rotvec, turn, fabric_z, extent, fov, size=(480, 360)):
    """`photograph`, through a camera whose image is turned by `turn` on the tool."""
    import gaussian_splatting
    import scan_geometry

    width, height = size
    k = gaussian_splatting.intrinsics_from_fov(width, height, fov)
    v, u = np.mgrid[0:height, 0:width]
    rays = np.stack([(u - k["cx"]) / k["fl_x"], (v - k["cy"]) / k["fl_y"], np.ones_like(u, dtype=float)], axis=-1)
    world = rays @ scan_geometry.camera_rotation(rotvec, turn).T
    reach = (fabric_z - camera[2]) / world[..., 2]
    hit = camera[:2] + world[..., :2] * reach[..., None]
    side = texture.shape[0]
    col = np.clip(((hit[..., 0] + extent) / (2 * extent) * side).astype(int), 0, side - 1)
    row = np.clip(((extent - hit[..., 1]) / (2 * extent) * side).astype(int), 0, side - 1)
    return Image.fromarray(texture[row, col], "RGB")


@pytest.mark.parametrize("angle,mirror", [(0.0, False), (23.0, False), (-110.0, False), (35.0, True)])
def test_the_image_turn_is_recovered_from_two_small_probes(angle, mirror):
    pytest.importorskip("scipy")
    import gaussian_splatting
    import robot_camera
    import scan_geometry

    rng = np.random.default_rng(5)
    texture = (rng.random((900, 900, 3)) * 255).astype(np.uint8)
    texture = np.asarray(Image.fromarray(texture).resize((900, 900)).filter(__import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(1.5)))
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    turn = np.array([[c, -s], [s, c]]) @ (np.diag([-1.0, 1.0]) if mirror else np.eye(2))

    fabric_z, height, step, fov = 0.20, 0.10, 0.004, 55.0
    heading = np.array([math.cos(0.4), math.sin(0.4)])
    rotvec = scan_geometry.looking_down_along(heading)
    here = np.array([0.0, 0.0, fabric_z + height])
    shot = lambda offset: photograph_turned(texture, here + offset, rotvec, turn, fabric_z, 0.15, fov)
    before = shot(np.zeros(3))
    slides = [robot_camera.image_shift(before, shot(np.r_[axis, 0.0])) / step for axis in (np.array([step, 0.0]), np.array([0.0, step]))]

    focal = gaussian_splatting.intrinsics_from_fov(480, 360, fov)["fl_x"]
    found, error, scale = scan_geometry.calibrate_image_turn(heading, slides[0], slides[1], height, focal)
    assert found == pytest.approx(turn, abs=0.03)
    assert error < 0.05 and scale == pytest.approx(1.0, abs=0.05)


# ============================================================================
# Responsiveness
# ============================================================================

def test_an_unreachable_pose_is_given_up_on_soon_not_after_every_step(monkeypatch):
    pytest.importorskip("mujoco")
    import fabric_scanner
    import scan_geometry

    steps = {"count": 0}
    real_step = fabric_scanner.step_ik
    monkeypatch.setattr(fabric_scanner, "step_ik", lambda *args: (steps.__setitem__("count", steps["count"] + 1), real_step(*args))[1])
    mj, model, data, site_id = scan_geometry.load_model()
    scan_geometry.seed_from(mj, model, data, site_id, HOME_JOINTS)
    with pytest.raises(scan_geometry.Unreachable):
        scan_geometry.solve_ik(mj, model, data, site_id, np.array([2.0, 2.0, 2.0, math.pi, 0.0, 0.0]), "far away")
    assert steps["count"] < scan_geometry.MAX_IK_ITERS // 2


def test_each_thread_keeps_its_own_model_and_reuses_it():
    pytest.importorskip("mujoco")
    import threading
    import scan_geometry

    mine = scan_geometry.load_model()
    assert scan_geometry.load_model() is mine
    theirs = {}
    worker = threading.Thread(target=lambda: theirs.setdefault("model", scan_geometry.load_model()))
    worker.start()
    worker.join()
    assert theirs["model"] is not mine
    assert theirs["model"][2] is not mine[2]


def test_a_huge_splat_is_sampled_down_for_the_viewer_but_keeps_its_count(tmp_path, monkeypatch):
    import gaussian_splatting

    monkeypatch.setattr(gaussian_splatting, "VIEWER_MAX_POINTS", 1000)
    rng = np.random.default_rng(1)
    write_splat(tmp_path / "big.ply", rng.uniform(-1, 1, (5000, 3)).astype(np.float32), np.zeros((5000, 3)), np.full(5000, 3.0))
    splat = gaussian_splatting.load_splat_ply(tmp_path / "big.ply")
    assert splat["count"] == 5000
    assert len(splat["positions"]) == len(splat["colours"]) == len(splat["opacity"]) == 1000


def test_a_scan_copied_from_another_machine_finds_its_pictures_beside_their_sidecars(tmp_path):
    import gui_captures

    session = tmp_path / "ur5_a"
    session.mkdir()
    Image.new("RGB", (8, 8)).save(session / "ur5_0001.png")
    (session / "ur5_0001.json").write_text(json.dumps({"path": r"C:\Users\someone\robot_scans\ur5_a\ur5_0001.png",
                                                       "detected_image": r"C:\Users\someone\robot_scans\ur5_a\gone.jpg"}))
    record, = gui_captures.load_session(session)
    assert record["path"] == str(session / "ur5_0001.png")
    assert record["detected_image"] == r"C:\Users\someone\robot_scans\ur5_a\gone.jpg"   # not there either: left as written
