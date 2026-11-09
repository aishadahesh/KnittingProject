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

import lens
from paths import CONFIG_DIR
from persistence import write_json

HAND_EYE_JSON = CONFIG_DIR / "hand_eye.json"

# A pixel darker than this is the object (a dark nut on white paper); a picture needs at least
# this many of them to count, and the blob must stay clear of the frame edge by this many pixels
# or its centre is biased by what was cut off.
TARGET_MAX_GREY = 60
TARGET_MIN_PIXELS = 2000
EDGE_MARGIN_PX = 3
# The field of view is only weakly constrained by a single point, so it is searched on a grid and
# the value whose fit also matches the object's apparent size at the two heights is kept.
FOV_GRID_DEG = (30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0)


# The arm's reported pose is trusted over the planned one only while the two agree: on a day the
# controller carried a different tool-centre setting, the report sat 10 cm from the plan, while the
# arm had in fact gone where it was sent.
TRUST_REPORTED_WITHIN_M = 0.005
TRUST_REPORTED_WITHIN_DEG = 3.0


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


def tcp_matrix(pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=float).reshape(-1)[:6]
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def hand_eye_from_params(rotvec, translation) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix


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
        uv = lens.project(in_camera, fx)
    else:
        uv = lens.project(in_camera, {"fl_x": fx, "fl_y": fx, "cx": width / 2.0, "cy": height / 2.0})
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


# The camera's turn about its own axis can be held at zero (image axes along the tool's) when a
# sweep has shown the picture sliding straight along the camera's x as the arm moves along the
# tool's x. On the mount calibrated here it is fitted: the dome of views pins it (holding it at
# zero made the fit three times worse), and that mount is turned.
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
    # The object's position starts under the first straight-down view, 5 cm along the tool axis.
    first = next((row for row in rows if row["straight_down"]), rows[0])
    start_target = first["tcp"][:3, 3] + first["tcp"][:3, :3] @ np.array([0.0, 0.0, 0.05])

    # A measured lens settles the focal length, centre and distortion; only the mount is fitted.
    measured = lens.load_intrinsics() if intrinsics is None else intrinsics
    results = []
    for fov in ([float(measured["fov_degrees"])] if measured else ([float(fov_degrees)] if fov_degrees else fov_grid)):
        fx = lens.intrinsics_for(width, height, measured) if measured else width / (2.0 * math.tan(math.radians(fov) / 2.0))
        best = None
        for lift in (0.0, 0.03, 0.06):
            for back in (0.0, -0.03, -0.06, 0.03):
                start = np.r_[start_target + [0.0, 0.0, lift], 0.0, 0.0, 0.0, 0.0, 0.0, back]
                params, residual = _levenberg_marquardt(start, tcps, uvs, fx, width, height, fixed=(5,) if roll_fixed else ())
                rms = float(np.sqrt(np.mean(residual ** 2)))
                if best is None or rms < best[0]:
                    best = (rms, params, residual)
        rms, params, residual = best
        results.append((fov, rms, params, residual, _size_mismatch(rows, params, fx, width)))

    if measured or fov_degrees:
        fov, rms, params, residual, _ = results[0]
    else:
        # Among fields of view whose fit is nearly as good as the best, the one whose camera depth
        # agrees with how the object's size changed between heights.
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


def _size_mismatch(rows, params, fx, width) -> float:
    # Only the depths matter here, not the focal length, so a measured lens needs no special case.
    """How far the fitted camera depths disagree with the object's apparent size: the size in
    pixels should scale as 1/depth. Zero when they agree; used to choose the field of view."""
    ratios = []
    for row in rows:
        camera = row["tcp"] @ hand_eye_from_params(params[3:6], params[6:9])
        depth = float((camera[:3, :3].T @ (params[0:3] - camera[:3, 3]))[2])
        if depth > 1e-3 and row["size_px"] > 0:
            ratios.append(row["size_px"] * depth)   # = fx * object width: constant if depths are right
    ratios = np.array(ratios)
    return float(np.std(ratios) / np.mean(ratios)) if len(ratios) else 0.0


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
