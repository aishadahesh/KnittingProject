"""The per-session mount and lens fit: a wrong mount and lens are recovered from the pictures."""

import math

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import hand_eye
import session_fit

WIDTH, HEIGHT = 640, 480
TRUE_LENS = {"f": 1050.0, "cx": 326.0, "cy": 236.0, "k1": 0.0, "k2": 0.0}
TRUE_MOUNT = hand_eye.hand_eye_from_params([0.20, 0.18, -0.80], [-0.050, -0.006, -0.046])
PLANE_Z = 0.03
PX_PER_MM = 10.0
CLOTH_MM = 160.0


@pytest.fixture(scope="module")
def cloth():
    return make_cloth()


def make_cloth():
    """A cloth texture with detail at every scale (no repeating stitch), CLOTH_MM square."""
    rng = np.random.default_rng(3)
    side = int(CLOTH_MM * PX_PER_MM)
    texture = np.zeros((side, side, 3), np.float32)
    for scale in (3, 9, 27, 81):
        noise = rng.random((side // scale + 2, side // scale + 2, 3)).astype(np.float32)
        texture += cv2.resize(noise, (side, side), interpolation=cv2.INTER_CUBIC)[:side, :side]
    texture = (255 * (texture - texture.min()) / (texture.max() - texture.min())).astype(np.uint8)
    return texture


def render(cloth, camera_to_world):
    """The picture a camera at `camera_to_world` (OpenCV axes) takes of the cloth on the plane z = PLANE_Z,
    centred on the world origin."""
    world_to_camera = np.linalg.inv(camera_to_world)
    s = 1.0 / (PX_PER_MM * 1000.0)
    half = CLOTH_MM / 2000.0
    from_texture = np.array([[s, 0.0, -half], [0.0, s, -half], [0.0, 0.0, 1.0]])
    plane = np.column_stack([world_to_camera[:3, 0], world_to_camera[:3, 1], world_to_camera[:3, 2] * PLANE_Z + world_to_camera[:3, 3]])
    k = np.array([[TRUE_LENS["f"], 0.0, TRUE_LENS["cx"]], [0.0, TRUE_LENS["f"], TRUE_LENS["cy"]], [0.0, 0.0, 1.0]])
    return cv2.warpPerspective(cloth, k @ plane @ from_texture, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderValue=(40, 40, 40))


def look_at(eye, target, roll_deg):
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = Rotation.from_euler("z", roll_deg, degrees=True).apply([1.0, 0.0, 0.0])
    right = right - (right @ forward) * forward
    right /= np.linalg.norm(right)
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = np.stack([right, np.cross(forward, right), forward], axis=1), eye
    return out


def scan(tmp_path, cloth):
    captures, target = [], np.array([0.0, 0.0, PLANE_Z])
    index = 0
    for distance in (0.09, 0.13):
        for tilt in (0.0, 15.0, 30.0):
            for azimuth in (range(0, 360, 45) if tilt else (0,)):
                direction = np.array([math.sin(math.radians(tilt)) * math.cos(math.radians(azimuth)),
                                      math.sin(math.radians(tilt)) * math.sin(math.radians(azimuth)), math.cos(math.radians(tilt))])
                camera = look_at(target + distance * direction, target, azimuth * 0.5)
                arm = camera @ np.linalg.inv(TRUE_MOUNT)
                path = tmp_path / f"ur5_{index:04d}.png"
                cv2.imwrite(str(path), render(cloth, camera))
                pose = [*arm[:3, 3], *Rotation.from_matrix(arm[:3, :3]).as_rotvec()]
                captures.append({"path": str(path), "target_position": pose, "actual_tcp_pose": pose})
                index += 1
    return captures


def test_a_wrong_mount_and_lens_are_fitted_back_from_the_pictures(tmp_path, cloth):
    captures = scan(tmp_path, cloth)
    # The saved calibration is off by a few millimetres and degrees, and its lens by 10 %.
    wrong = TRUE_MOUNT @ hand_eye.hand_eye_from_params([0.02, -0.015, 0.03], [0.004, -0.003, 0.005])
    calibration = {"rotvec": Rotation.from_matrix(wrong[:3, :3]).as_rotvec().tolist(), "translation": wrong[:3, 3].tolist(),
                   "fov_degrees": math.degrees(2 * math.atan(WIDTH / (2 * TRUE_LENS["f"] * 1.1)))}
    fitted = session_fit.fit(captures, calibration, PLANE_Z, mirrored=False, log=lambda *_: None)

    # SIFT locates a point to about half a pixel; the fit cannot beat that.
    assert fitted["rounds"][-1]["median_px"] < 1.0
    assert fitted["lens"]["f"] == pytest.approx(TRUE_LENS["f"], rel=0.01)
    # Every camera where it really was, to a fraction of a millimetre.
    worst = max(np.linalg.norm((hand_eye.tcp_matrix(c["target_position"]) @ fitted["mount"])[:3, 3]
                               - (hand_eye.tcp_matrix(c["target_position"]) @ TRUE_MOUNT)[:3, 3]) for c in captures)
    assert worst < 0.0005

    dataset = session_fit.export_dataset(captures, fitted, tmp_path / "ds")
    assert dataset.image_count == len(captures)
    assert (tmp_path / "ds" / "fabric_seed.ply").stat().st_size > 1000
