"""The camera-mount calibration: fitted from pictures of one dark object, applied at export."""

import json
import math

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

import gaussian_splatting
import hand_eye


# A camera beside and behind the tool centre, turned on its mount - the kind of offset the real
# arm turned out to have (a few centimetres), mild enough for the object to stay in frame.
TRUE_ROTVEC = np.array([0.1, 0.05, -0.6])
TRUE_TRANSLATION = np.array([-0.02, 0.005, -0.03])
TRUE_TARGET = np.array([-0.21, 0.66, 0.056])
AIM_OFFSET = np.array([0.006, -0.004, 0.008])
FOV = 40.0
WIDTH, HEIGHT = 640, 480


def _synthetic_scan(tmp_path, rotvec=TRUE_ROTVEC, translation=TRUE_TRANSLATION, target=TRUE_TARGET, fov=FOV):
    """Captures of a dark disc at `target`, taken with a camera mounted at (rotvec, translation) on
    the tool, from the views a splat scan plans: straight down at two heights and tilted rings."""
    true_hand_eye = hand_eye.hand_eye_from_params(rotvec, translation)
    fx = WIDTH / (2.0 * math.tan(math.radians(fov) / 2.0))
    captures = []
    index = 0
    for depth in (0.05, 0.075):
        views = [(0.0, 0.0, "straight down")]
        views += [(tilt, az, f"az{az} tilt{tilt}") for tilt in (7.0, 13.0, 20.0) for az in range(-180, 180, 30)]
        for tilt, azimuth, view in views:
            # The planner believes the camera is the TCP, so it puts the TCP here looking at `target`;
            # the real camera is the mount's offset away from it, and sees the object off-centre.
            toward = np.array([math.sin(math.radians(tilt)) * math.cos(math.radians(azimuth)),
                               math.sin(math.radians(tilt)) * math.sin(math.radians(azimuth)),
                               math.cos(math.radians(tilt))])
            forward = -toward
            # The arm rolls the tool differently at the two heights (the real scan did, by 24 deg),
            # which is what makes the camera's turn on its mount observable.
            roll = math.radians(0.0 if depth < 0.06 else 24.0)
            right = np.cross([math.sin(roll), math.cos(roll), 0.0], forward)
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            # The scan is aimed a few centimetres from the object, as a scan planned with a wrong
            # mount always is: seen from the same spot in every picture, the object would tell
            # nothing about the mount.
            tcp = np.eye(4)
            tcp[:3, :3], tcp[:3, 3] = np.stack([right, down, forward], axis=1), target + AIM_OFFSET + toward * depth
            camera = tcp @ true_hand_eye
            camera_rotation, camera_centre = camera[:3, :3], camera[:3, 3]
            # The picture the camera would take: the disc projected with the true mount.
            in_camera = camera_rotation.T @ (target - camera_centre)
            if in_camera[2] < 0.01:
                continue
            u = fx * in_camera[0] / in_camera[2] + WIDTH / 2.0
            v = fx * in_camera[1] / in_camera[2] + HEIGHT / 2.0
            radius = fx * 0.006 / in_camera[2]
            if not (radius + 5 < u < WIDTH - radius - 5 and radius + 5 < v < HEIGHT - radius - 5):
                continue
            picture = np.full((HEIGHT, WIDTH), 230, dtype=np.uint8)
            ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
            picture[(xs - u) ** 2 + (ys - v) ** 2 <= radius ** 2] = 20
            index += 1
            path = tmp_path / f"cap_{index:03d}.png"
            Image.fromarray(picture).save(path)
            pose = [*tcp[:3, 3], *Rotation.from_matrix(tcp[:3, :3]).as_rotvec()]
            captures.append({"path": str(path), "actual_tcp_pose": pose, "target_position": pose,
                             "capture_view": view, "camera_image_width": WIDTH, "camera_image_height": HEIGHT})
    return captures, fx


def test_the_mount_is_recovered_from_pictures_of_one_dark_object(tmp_path):
    captures, _ = _synthetic_scan(tmp_path)
    calibration = hand_eye.fit(captures, fov_degrees=FOV)
    assert calibration["rms_px"] < 1.5
    assert np.allclose(calibration["translation"], TRUE_TRANSLATION, atol=0.002)
    assert np.allclose(calibration["rotvec"], TRUE_ROTVEC, atol=0.02)
    assert np.allclose(calibration["target"], TRUE_TARGET, atol=0.002)
    assert calibration["pictures_used"] == len(captures)


def test_the_field_of_view_is_found_when_not_given(tmp_path):
    captures, _ = _synthetic_scan(tmp_path)
    calibration = hand_eye.fit(captures, fov_grid=(30.0, 40.0, 50.0, 60.0))
    assert calibration["fov_degrees"] == pytest.approx(40.0)


def test_pictures_with_the_object_cut_off_or_missing_are_left_out(tmp_path):
    captures, _ = _synthetic_scan(tmp_path)
    blank = tmp_path / "blank.png"
    Image.fromarray(np.full((HEIGHT, WIDTH), 230, dtype=np.uint8)).save(blank)
    clipped = tmp_path / "clipped.png"
    picture = np.full((HEIGHT, WIDTH), 230, dtype=np.uint8)
    picture[:80, :80] = 20
    Image.fromarray(picture).save(clipped)
    extra = [{"path": str(blank), "actual_tcp_pose": captures[0]["actual_tcp_pose"]},
             {"path": str(clipped), "actual_tcp_pose": captures[0]["actual_tcp_pose"]}]
    rows, skipped = hand_eye.observations(captures + extra)
    assert len(rows) == len(captures)
    assert any("no dark object" in reason for reason in skipped)
    assert any("edge of the frame" in reason for reason in skipped)


def test_too_few_pictures_is_refused_plainly(tmp_path):
    captures, _ = _synthetic_scan(tmp_path)
    with pytest.raises(ValueError, match="at least 12"):
        hand_eye.fit(captures[:5], fov_degrees=FOV)


def test_the_calibration_is_saved_and_read_back(tmp_path):
    saved = hand_eye.save_calibration({"rotvec": [0.1, 0.2, 0.3], "translation": [0.01, 0.02, 0.03], "target": [0, 0, 0],
                                       "fov_degrees": 40.0, "rms_px": 2.0, "pictures_used": 30}, tmp_path / "hand_eye.json")
    loaded = hand_eye.load_calibration(saved)
    assert np.allclose(hand_eye.calibration_matrix(loaded), hand_eye.hand_eye_from_params([0.1, 0.2, 0.3], [0.01, 0.02, 0.03]))
    assert "40 deg" in hand_eye.describe(loaded)
    assert hand_eye.load_calibration(tmp_path / "missing.json") is None


def test_a_calibrated_export_places_the_object_where_the_pictures_show_it(tmp_path):
    captures, fx = _synthetic_scan(tmp_path)
    calibration = hand_eye.fit(captures, fov_degrees=FOV)
    dataset = gaussian_splatting.export_dataset(captures, tmp_path / "ds", calibration=calibration, copy_images=False)
    assert dataset.image_count == len(captures) and not dataset.warnings
    transforms = json.loads(dataset.transforms_path.read_text())
    assert transforms["fl_x"] == pytest.approx(fx, rel=1e-6)
    assert "calibration" in transforms["pose_source"]
    assert transforms["ply_file_path"] == gaussian_splatting.SEED_PLY_NAME
    target = np.array(calibration["target"])
    for frame in transforms["frames"]:
        c2w = np.array(frame["transform_matrix"]) @ gaussian_splatting._CV_TO_GL
        in_camera = c2w[:3, :3].T @ (target - c2w[:3, 3])
        u = fx * in_camera[0] / in_camera[2] + WIDTH / 2.0
        v = fx * in_camera[1] / in_camera[2] + HEIGHT / 2.0
        seen, _, _ = hand_eye.find_target(dataset.root / frame["file_path"])
        assert abs(u - seen[0]) < 2.0 and abs(v - seen[1]) < 2.0


def test_the_seed_surrounds_the_calibrated_target(tmp_path):
    target = np.array([0.1, 0.2, 0.3])
    count = gaussian_splatting.write_target_seed(target, tmp_path / "seed.ply", [target + [0, 0, 0.05], target + [0.02, 0, 0.07]])
    assert count > 1000
    splat = gaussian_splatting.load_splat_ply(tmp_path / "seed.ply")
    assert np.allclose(np.median(splat["positions"], axis=0)[:2], target[:2], atol=0.003)
    on_surface = np.isclose(splat["positions"][:, 2], target[2] - 0.004, atol=1e-5)
    assert on_surface.sum() > 1000
    near_target = np.linalg.norm(splat["positions"] - target, axis=1) < 0.01
    assert near_target.sum() > 500 and splat["colours"][near_target].mean() < splat["colours"][on_surface & ~near_target].mean()


def test_a_splat_still_being_written_is_not_read_as_a_small_one(tmp_path):
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 1000\n"
              "property float x\nproperty float y\nproperty float z\nproperty float opacity\nend_header\n").encode("ascii")
    partial = tmp_path / "partial.ply"
    partial.write_bytes(header + np.zeros(10 * 4, dtype="<f4").tobytes())
    with pytest.raises(RuntimeError, match="still being written"):
        gaussian_splatting.load_splat_ply(partial)


def test_a_measured_lens_is_used_as_it_is_and_scales_with_the_picture():
    measured = {"fx": 1200.0, "fy": 1180.0, "cx": 330.0, "cy": 250.0, "k1": -0.1, "width": 640, "height": 480}
    full = hand_eye.intrinsics_for(640, 480, measured)
    assert full["fl_x"] == 1200.0 and full["cx"] == 330.0 and full["camera_model"] == "OPENCV" and full["k1"] == -0.1
    half = hand_eye.intrinsics_for(320, 240, measured)
    assert half["fl_x"] == 600.0 and half["fl_y"] == 590.0 and half["cx"] == 165.0
    on_axis = hand_eye.project(np.array([[0.0, 0.0, 1.0]]), full)
    assert np.allclose(on_axis, [[330.0, 250.0]])
    off_axis = hand_eye.project(np.array([[0.2, 0.0, 1.0]]), full)
    assert 330.0 < off_axis[0, 0] < 330.0 + 0.2 * 1200.0


def test_the_mount_fit_accepts_measured_intrinsics(tmp_path):
    captures, fx = _synthetic_scan(tmp_path)
    measured = {"fx": fx, "fy": fx, "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0, "width": WIDTH, "height": HEIGHT,
                "fov_degrees": FOV}
    calibration = hand_eye.fit(captures, intrinsics=measured)
    assert calibration["lens"].startswith("measured")
    assert calibration["rms_px"] < 1.5
    assert np.allclose(calibration["translation"], TRUE_TRANSLATION, atol=0.002)


def test_a_reported_pose_far_from_the_planned_one_is_not_trusted():
    planned = [-0.06, 0.65, 0.077, 3.139, -0.128, 0.0]
    close = [-0.061, 0.651, 0.076, 3.12, -0.13, 0.002]
    far = [-0.063, 0.649, -0.030, -3.1385, 0.1193, -0.0001]      # 10.7 cm off: another tool-centre setting
    assert hand_eye.arm_pose({"target_position": planned, "actual_tcp_pose": close}) == close
    assert hand_eye.arm_pose({"target_position": planned, "actual_tcp_pose": far}) == planned
    assert hand_eye.arm_pose({"target_position": planned}) == planned
    assert hand_eye.arm_pose({}) is None
