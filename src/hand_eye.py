"""Where the camera really sits on the arm, measured from a scan of one small dark object.

A robot scan records the pose the arm was sent to for every picture, and the planner assumes the
camera is *at* that pose - the tool centre point - looking along its axis. On this arm it is not:
the lens sits a few centimetres beside and behind the TCP, tilted. With that wrong, every planned
"look at the subject" view looks a few centimetres away from the subject, and the exported camera
poses disagree with the pictures by hundreds of pixels. A splat trained on them is a cloud of
floaters that reproduces each training picture and nothing in between. (Solving the poses from the
pictures instead, with COLMAP, fails on these scenes for its own reason: a plain sheet and a shiny
object give it nearly no features to match.)

The fix is a hand-eye calibration, and a scan of a single dark object on a plain light background
(a nut on paper) is enough to measure it. Each picture gives one observation: where the object's
centre landed in the image. The unknowns - the object's position in the robot's base frame, the
camera's rotation and translation relative to the tool flange, and optionally the lens's field of
view - are fitted so that the object projects where it was seen, by Levenberg-Marquardt over all
the pictures. The straight-down views at two heights fix the depth, the tilted rings fix the
lateral offset and tilt, and the rotation about the optical axis comes from how the offset turns
with the arm.

The result is saved as config/hand_eye.json and applied to the arm's actual TCP pose when a
dataset is exported (gaussian_splatting.export_dataset). A residual of a few pixels is normal; a
few tens of pixels usually means the object was not alone in frame or the pictures were taken at a
different field of view than fitted.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from paths import CONFIG_DIR
from persistence import write_json

HAND_EYE_JSON = CONFIG_DIR / "hand_eye.json"
INTRINSICS_JSON = CONFIG_DIR / "camera_intrinsics.json"

TARGET_MAX_GREY = 60
TARGET_MIN_PIXELS = 2000
EDGE_MARGIN_PX = 3
FOV_GRID_DEG = (30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0)
TRUST_REPORTED_WITHIN_M = 0.005
TRUST_REPORTED_WITHIN_DEG = 3.0


def load_intrinsics(path: Path = INTRINSICS_JSON) -> dict | None:
    """The measured lens, or None when it has not been measured."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and "fx" in data else None


def intrinsics_for(width: int, height: int, measured: dict) -> dict:
    """The measured lens as nerfstudio's transforms.json keys, scaled to this picture size."""
    sx, sy = width / float(measured["width"]), height / float(measured["height"])
    fx = float(measured["fx"]) * sx
    return {"fl_x": fx, "fl_y": float(measured["fy"]) * sy,
            "cx": float(measured["cx"]) * sx, "cy": float(measured["cy"]) * sy, "w": int(width), "h": int(height),
            "k1": float(measured.get("k1", 0.0)), "k2": float(measured.get("k2", 0.0)), "k3": float(measured.get("k3", 0.0)),
            "p1": float(measured.get("p1", 0.0)), "p2": float(measured.get("p2", 0.0)), "camera_model": "OPENCV",
            "camera_angle_x": 2.0 * float(np.arctan(width / (2.0 * fx)))}


def project(points_cam: np.ndarray, intrinsics: dict) -> np.ndarray:
    """Camera-frame points (x right, y down, z forward) to pixels, through the lens's distortion.
    `intrinsics` holds fl_x, fl_y, cx, cy and optionally k1, k2, k3, p1, p2 (OpenCV's model)."""
    z = np.maximum(points_cam[:, 2], 1e-6)
    x, y = points_cam[:, 0] / z, points_cam[:, 1] / z
    k1, k2, k3 = (float(intrinsics.get(k, 0.0)) for k in ("k1", "k2", "k3"))
    p1, p2 = (float(intrinsics.get(k, 0.0)) for k in ("p1", "p2"))
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack([float(intrinsics["fl_x"]) * xd + float(intrinsics["cx"]), float(intrinsics["fl_y"]) * yd + float(intrinsics["cy"])], axis=1)


def arm_pose(capture: dict):
    """The pose a capture was taken from, in the frame the calibration is measured in: the
    reported TCP pose when it agrees with the planned one, else the planned one. None if neither."""
    planned = capture.get("target_position")
    reported = capture.get("actual_tcp_pose")
    planned = None if planned is None or len(planned) < 6 else [float(v) for v in planned[:6]]
    reported = None if reported is None or len(reported) < 6 else [float(v) for v in reported[:6]]
    if planned is None or reported is None:
        return reported or planned
    apart = float(np.linalg.norm(np.subtract(reported[:3], planned[:3])))
    relative = Rotation.from_rotvec(reported[3:6]).as_matrix() @ Rotation.from_rotvec(planned[3:6]).as_matrix().T
    turned = math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(relative) - 1.0) / 2.0))))
    if apart <= TRUST_REPORTED_WITHIN_M and turned <= TRUST_REPORTED_WITHIN_DEG:
        return reported
    return planned


def hand_eye_from_params(rotvec, translation) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix


def tcp_matrix(pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=float).reshape(-1)[:6]
    return hand_eye_from_params(pose[3:6], pose[:3])


def params_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3]])


_loaded: dict = {}


def load_calibration(path: Path = HAND_EYE_JSON) -> dict | None:
    """The saved calibration, or None when the mount has not been calibrated.

    The panel asks every frame; the file is read again only when it changes on disk."""
    path = Path(path)
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return None
    cached = _loaded.get(str(path))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "rotvec" not in data or "translation" not in data:
        data = None
    _loaded[str(path)] = (stamp, data)
    return data


def calibration_matrix(calibration: dict) -> np.ndarray:
    return hand_eye_from_params(calibration["rotvec"], calibration["translation"])


def find_target(image_path) -> tuple[np.ndarray, float, bool] | None:
    """The dark object's centre in the picture, its width in pixels, and whether the frame cut it.

    None when there is no dark object worth the name."""
    from PIL import Image

    with Image.open(image_path) as image:
        grey = np.asarray(image.convert("L"), dtype=np.uint8)
    ys, xs = np.nonzero(grey < TARGET_MAX_GREY)
    if len(xs) < TARGET_MIN_PIXELS:
        return None
    height, width = grey.shape
    clipped = bool(xs.min() < EDGE_MARGIN_PX or ys.min() < EDGE_MARGIN_PX
                   or xs.max() >= width - EDGE_MARGIN_PX or ys.max() >= height - EDGE_MARGIN_PX)
    size = float(np.percentile(xs, 98) - np.percentile(xs, 2))
    return np.array([xs.mean(), ys.mean()]), size, clipped


def observations(captures) -> tuple[list[dict], list[str]]:
    """One observation per usable capture: the TCP pose and where the object was seen. And why the
    others were left out."""
    rows, skipped = [], []
    for capture in captures:
        path = Path(str(capture.get("path") or ""))
        pose = arm_pose(capture)
        if not path.is_file() or pose is None:
            skipped.append(f"{path.name or '(no picture)'}: no picture or no arm pose")
            continue
        found = find_target(path)
        if found is None:
            skipped.append(f"{path.name}: no dark object in it")
            continue
        centre, size, clipped = found
        if clipped:
            skipped.append(f"{path.name}: the object runs off the edge of the frame")
            continue
        rows.append({"name": path.name, "tcp": tcp_matrix(pose), "uv": centre, "size_px": size,
                     "view": str(capture.get("capture_view", "")),
                     "straight_down": "straight" in str(capture.get("capture_view", ""))})
    return rows, skipped


def _residuals(params, tcps, uvs, fx, width, height):
    """`fx` is a focal length in pixels (an assumed-FOV lens, centred, undistorted) or a dict of
    measured intrinsics (fl_x, fl_y, cx, cy and distortion)."""
    target, rotvec, translation = params[0:3], params[3:6], params[6:9]
    rotation = Rotation.from_rotvec(rotvec).as_matrix()
    camera_rotations = tcps[:, :3, :3] @ rotation
    camera_centres = tcps[:, :3, 3] + tcps[:, :3, :3] @ translation
    in_camera = np.einsum("nji,nj->ni", camera_rotations, target - camera_centres)
    in_camera[:, 2] = np.where(in_camera[:, 2] > 1e-3, in_camera[:, 2], 1e-3)
    if isinstance(fx, dict):
        uv = project(in_camera, fx)
    else:
        uv = project(in_camera, {"fl_x": fx, "fl_y": fx, "cx": width / 2.0, "cy": height / 2.0})
    return np.concatenate([uv[:, 0] - uvs[:, 0], uv[:, 1] - uvs[:, 1]])


def _levenberg_marquardt(params, tcps, uvs, fx, width, height, iterations=200, fixed=()):
    """`fixed` names parameter indices held at their starting value."""
    params = np.array(params, dtype=float)
    damping = 1e-2
    residual = _residuals(params, tcps, uvs, fx, width, height)
    cost = float(residual @ residual)
    for _ in range(iterations):
        jacobian = np.zeros((len(residual), len(params)))
        for index in range(len(params)):
            if index in fixed:
                continue
            step = np.zeros(len(params))
            step[index] = 1e-6
            jacobian[:, index] = (_residuals(params + step, tcps, uvs, fx, width, height) - residual) / 1e-6
        normal, gradient = jacobian.T @ jacobian, jacobian.T @ residual
        improved = False
        while damping < 1e12:
            try:
                delta = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-12), -gradient)
            except np.linalg.LinAlgError:
                break
            trial = params + delta
            trial_residual = _residuals(trial, tcps, uvs, fx, width, height)
            trial_cost = float(trial_residual @ trial_residual)
            if trial_cost < cost:
                params, residual, cost = trial, trial_residual, trial_cost
                damping = max(damping / 3.0, 1e-9)
                improved = True
                break
            damping *= 4.0
        if not improved or np.linalg.norm(delta) < 1e-10:
            break
    return params, residual


ROLL_FIXED = False


def fit(captures, image_size=(640, 480), fov_grid=FOV_GRID_DEG, fov_degrees=None, intrinsics=None, roll_fixed=ROLL_FIXED) -> dict:
    """The calibration from a scan's capture records. With a measured lens (config/camera_intrinsics.json,
    or `intrinsics`) its focal length and distortion are used; else `fov_degrees` pins the field of
    view, or left None it is chosen from the grid. Raises ValueError when there is too little to fit."""
    rows, skipped = observations(captures)
    if len(rows) < 12:
        raise ValueError(f"only {len(rows)} pictures show the object whole; the fit needs at least 12 "
                         f"(straight-down views at two heights and tilted rings)")
    width, height = (int(v) for v in image_size)
    tcps = np.array([row["tcp"] for row in rows])
    uvs = np.array([row["uv"] for row in rows])
    first = next((row for row in rows if row["straight_down"]), rows[0])
    start_target = first["tcp"][:3, 3] + first["tcp"][:3, :3] @ np.array([0.0, 0.0, 0.05])

    measured = load_intrinsics() if intrinsics is None else intrinsics
    results = []
    for fov in ([float(measured["fov_degrees"])] if measured else ([float(fov_degrees)] if fov_degrees else fov_grid)):
        fx = intrinsics_for(width, height, measured) if measured else width / (2.0 * math.tan(math.radians(fov) / 2.0))
        best = None
        for lift in (0.0, 0.03, 0.06):
            for back in (0.0, -0.03, -0.06, 0.03):
                start = np.r_[start_target + [0.0, 0.0, lift], 0.0, 0.0, 0.0, 0.0, 0.0, back]
                params, residual = _levenberg_marquardt(start, tcps, uvs, fx, width, height, fixed=(5,) if roll_fixed else ())
                rms = float(np.sqrt(np.mean(residual ** 2)))
                if best is None or rms < best[0]:
                    best = (rms, params, residual)
        rms, params, residual = best
        ratios = [
            row["size_px"] * float(((row["tcp"] @ hand_eye_from_params(params[3:6], params[6:9]))[:3, :3].T
                                    @ (params[0:3] - (row["tcp"] @ hand_eye_from_params(params[3:6], params[6:9]))[:3, 3]))[2])
            for row in rows
            if float(((row["tcp"] @ hand_eye_from_params(params[3:6], params[6:9]))[:3, :3].T
                      @ (params[0:3] - (row["tcp"] @ hand_eye_from_params(params[3:6], params[6:9]))[:3, 3]))[2]) > 1e-3
            and row["size_px"] > 0
        ]
        ratio_arr = np.array(ratios)
        mismatch = float(np.std(ratio_arr) / np.mean(ratio_arr)) if len(ratio_arr) else 0.0
        results.append((fov, rms, params, residual, mismatch))

    if measured or fov_degrees:
        fov, rms, params, residual, _ = results[0]
    else:
        best_rms = min(result[1] for result in results)
        fov, rms, params, residual, _ = min((result for result in results if result[1] <= best_rms * 1.15),
                                            key=lambda result: result[4])
    per_picture = np.linalg.norm(residual.reshape(2, -1).T, axis=1)
    sessions = sorted({str(c.get("session_id")) for c in captures if c.get("session_id")})
    return {
        "from_sessions": sessions,
        "rotvec": [float(v) for v in params[3:6]],
        "translation": [float(v) for v in params[6:9]],
        "target": [float(v) for v in params[0:3]],
        "fov_degrees": float(fov),
        "lens": "measured (config/camera_intrinsics.json)" if measured else "fitted field of view, no distortion",
        "roll": "held at zero (image axes along the tool's)" if roll_fixed else "fitted",
        "rms_px": float(rms),
        "max_px": float(per_picture.max()),
        "pictures_used": len(rows),
        "skipped": skipped,
        "image_size": [width, height],
        "fitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "worst": [{"name": row["name"], "error_px": float(err)}
                  for row, err in sorted(zip(rows, per_picture), key=lambda pair: -pair[1])[:5]],
    }


def save_calibration(calibration: dict, path: Path = HAND_EYE_JSON) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, calibration, indent=2)
    return path


def describe(calibration: dict) -> str:
    offset = np.asarray(calibration["translation"], dtype=float) * 1000.0
    tilt = math.degrees(np.linalg.norm(np.asarray(calibration["rotvec"], dtype=float)))
    return (f"camera {offset[0]:+.0f}, {offset[1]:+.0f}, {offset[2]:+.0f} mm from the TCP (tool x, y, z), "
            f"turned {tilt:.0f} deg, lens {calibration.get('fov_degrees', 0):.0f} deg; "
            f"fits {calibration.get('pictures_used', 0)} pictures to {calibration.get('rms_px', 0):.1f} px")
