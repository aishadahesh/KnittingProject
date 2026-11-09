"""Optional Gaussian Splatting reconstruction from a real UR5 scan.

A scan already produces the two things a splatting trainer needs: images of the
fabric from many angles, and -- because a robot took them -- a known camera pose
for every one. That second part is the useful bit. Normal photogrammetry has to
recover poses with structure-from-motion, which on a flat, repetitively textured
knit is exactly the case SfM handles worst. Robot forward kinematics sidesteps
it entirely.

So this module does not reconstruct anything itself. It converts a scan into the
dataset formats trainers expect (``transforms.json`` and a COLMAP text model),
and can then hand that dataset to whichever trainer the user has installed. The
heavy CUDA training stays external and optional; preparing the dataset does not
require it.

Two things must be right for the exported poses to be worth anything, and both
are surfaced rather than hidden:

  * The hand-eye transform -- where the camera sits relative to the tool flange.
    The default is a plain forward-looking offset, which is a guess. An
    uncalibrated guess yields a misaligned reconstruction, so it is a settable
    parameter and the export records what was used.
  * The intrinsics. Without a calibration file these are derived from an assumed
    field of view, which is likewise approximate.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

import lens
from persistence import write_json


# Camera sits ahead of the tool flange looking along tool +Z. A placeholder for
# a real hand-eye calibration, not a substitute for one.
DEFAULT_CAMERA_OFFSET_M = 0.12
# Measured on this camera (the mount calibration's dome and a sweep's picture-to-picture slide
# agree on it); the earlier 55 was a guess.
DEFAULT_FOV_DEGREES = 38.0
# A reported tool point further than this from the planned one, along the tool axis, is a
# tool-centre setting the planner did not know of; across the axis it is just a miss.
TOOL_CENTRE_MISMATCH_M = 0.005


@dataclass
class SplatDataset:
    """Where an exported dataset landed and what went into it."""

    root: Path
    image_count: int
    transforms_path: Path
    colmap_dir: Path
    session_id: str = ""
    warnings: list[str] = field(default_factory=list)
    pieces: int = 1
    offered: int = 0

def hand_eye_matrix(offset_m: float = DEFAULT_CAMERA_OFFSET_M, rotation_rotvec=None) -> np.ndarray:
    """Tool-flange to camera transform.

    Defaults to a pure translation along the tool's +Z, which is where a
    gripper-mounted camera looking at the work usually sits.
    """
    matrix = np.eye(4, dtype=float)
    if rotation_rotvec is not None:
        matrix[:3, :3] = Rotation.from_rotvec(np.asarray(rotation_rotvec, dtype=float)).as_matrix()
    matrix[:3, 3] = [0.0, 0.0, float(offset_m)]
    return matrix


def pose_to_matrix(pose) -> np.ndarray:
    """UR pose ``[x, y, z, rx, ry, rz]`` to a 4x4 base-to-tool transform.

    The rotation part is an axis-angle (rotation vector), which is how UR
    controllers report and accept orientation.
    """
    pose = np.asarray(pose, dtype=float).reshape(-1)[:6]
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def camera_to_world(pose, hand_eye: np.ndarray | None = None) -> np.ndarray:
    """Camera pose in the robot base frame, in OpenCV axes (x right, y down, z forward)."""
    hand_eye = hand_eye_matrix() if hand_eye is None else np.asarray(hand_eye, dtype=float)
    return pose_to_matrix(pose) @ hand_eye


# OpenCV camera axes to OpenGL/NeRF axes: flip Y and Z. transforms.json is a
# NeRF-convention format, so skipping this flips the reconstruction upside down
# and back to front -- a failure that looks like bad calibration rather than a
# convention mismatch, which is why it is spelled out here.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def measure_image_turn(captures, fov_degrees: float, pairs: int = 12) -> dict | None:
    """The camera's image turn (and mirroring) measured from a grid or sweep scan's own pictures.

    Neighbouring stations are a known step apart in the plane; how far, and which way, the picture
    slides between them says which way the image's axes really point relative to the tool's.
    The planner assumed the tool's axes when it recorded the poses; a scan whose pictures slide the
    other way in y was mirrored, and trained as it was recorded every camera disagreed with its
    picture (12 dB on a cloth). Returns the turn as scan_geometry.calibrate_image_turn gives it,
    with the slides and the picture scale (pixels per mm on the cloth); None if the scan has no
    row/column neighbours or the pictures do not match.
    """
    from PIL import Image

    import scan_geometry

    stations = {}
    for capture in captures:
        if capture.get("row") is None or capture.get("col") is None or capture.get("camera_position") is None:
            continue
        stations.setdefault((int(capture["row"]), int(capture["col"])), capture)
    if len(stations) < 4:
        return None

    def picture(capture):
        path = Path(str(capture.get("path") or ""))
        with Image.open(path) as image:
            image = image.convert("L")
            image = image.resize((image.width // 2, image.height // 2), Image.BILINEAR)
        grey = np.asarray(image, dtype=float)
        return (grey - grey.mean()) / (grey.std() + 1e-9)

    def slide(step):
        """Mean picture shift (full-resolution pixels) over up to `pairs` neighbour pairs one `step` apart."""
        scores, moves, used = {}, [], 0
        for (row, col), capture in stations.items():
            other = stations.get((row + step[0], col + step[1]))
            if other is None:
                continue
            try:
                a, b = picture(capture), picture(other)
            except OSError:
                continue
            moves.append(np.subtract(other["camera_position"][:2], capture["camera_position"][:2]))
            height, width = a.shape
            for dy in range(-80, 81, 4):
                for dx in range(-80, 81, 4):
                    ya, yb, xa, xb = max(0, dy), max(0, -dy), max(0, dx), max(0, -dx)
                    h, w = height - abs(dy), width - abs(dx)
                    if h < 40 or w < 40:
                        continue
                    aa, bb = a[ya:ya + h, xa:xa + w], b[yb:yb + h, xb:xb + w]
                    scores[(dx, dy)] = scores.get((dx, dy), 0.0) + float((aa * bb).mean()) / (aa.std() * bb.std() + 1e-9)
            used += 1
            if used >= pairs:
                break
        if used < 3:
            return None
        (dx, dy), score = max(scores.items(), key=lambda item: item[1])
        return np.array([dx * 2.0, dy * 2.0]), np.mean(moves, axis=0), score / used

    along_col, along_row = slide((0, 1)), slide((1, 0))
    if along_col is None or along_row is None or along_col[2] < 0.15 or along_row[2] < 0.15:
        return None
    # The two steps as the plane sees them; the turn is solved against the tool's heading.
    first = next(iter(stations.values()))
    rotvec = np.asarray(first["target_position"][3:6], dtype=float)
    heading = Rotation.from_rotvec(rotvec).as_matrix() @ np.array([1.0, 0.0, 0.0])
    width = int(first.get("camera_image_width", 640))
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    # Express each slide per metre of the step it was measured over, along that step's direction.
    moves = np.column_stack([along_col[1], along_row[1]])            # 2x2: columns are the steps in the plane
    slides = np.column_stack([along_col[0], along_row[0]])           # 2x2: the slides they caused
    # Slides for unit steps along the base x and y: slides @ inv(moves) @ [x; y]
    per_axis = slides @ np.linalg.inv(moves)
    scale = float(np.linalg.norm(per_axis[:, 0])) / 1000.0           # px per mm
    # calibrate_image_turn wants the slide for +x and +y moves, and the camera height that scale implies.
    height = focal / max(float(np.linalg.norm(per_axis[:, 0])), 1e-9)
    turn, non_orthogonal, _ = scan_geometry.calibrate_image_turn(heading[:2], per_axis[:, 0], per_axis[:, 1], height, focal)
    return {"turn": np.asarray(turn, dtype=float).tolist(), "mirrored": bool(np.linalg.det(turn) < 0.0),
            "non_orthogonal": float(non_orthogonal), "px_per_mm": scale, "height_m": float(height),
            "slide_x": along_col[0].tolist(), "slide_y": along_row[0].tolist(), "step_x_mm": (along_col[1] * 1000).tolist(),
            "step_y_mm": (along_row[1] * 1000).tolist()}


def _normalised_grey(path, scale=2):
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((image.width // scale, image.height // scale), Image.BILINEAR)
    grey = np.asarray(image, dtype=float)
    return (grey - grey.mean()) / (grey.std() + 1e-9)


def _slide_between(a: np.ndarray, b: np.ndarray, prior, window: int = 20, step: int = 2):
    """How far picture b's content sits from a's, in pixels of these arrays, searched round `prior`:
    (dx, dy, correlation). Searching only near the prior keeps a repeating knit from matching a
    stitch over."""
    height, width = a.shape
    best = (-2.0, 0, 0)
    for dy in range(int(prior[1]) - window, int(prior[1]) + window + 1, step):
        for dx in range(int(prior[0]) - window, int(prior[0]) + window + 1, step):
            ya, yb, xa, xb = max(0, dy), max(0, -dy), max(0, dx), max(0, -dx)
            h, w = height - abs(dy), width - abs(dx)
            if h < 30 or w < 30:
                continue
            aa, bb = a[ya:ya + h, xa:xa + w], b[yb:yb + h, xb:xb + w]
            score = float((aa * bb).mean()) / (aa.std() * bb.std() + 1e-9)
            if score > best[0]:
                best = (score, dx, dy)
    return best[1], best[2], best[0]


def register_sweep(captures, fov_degrees: float, min_correlation: float = 0.2) -> dict | None:
    """Camera poses for a straight-down grid or sweep scan of a flat cloth, from the pictures.

    The arm's steps are known but how they map to the picture is not quite - the lens, the camera's
    turn, its height, each a few percent off - and across a 25-station row a few percent is a whole
    stitch, so a splat trained on the arm's poses averages neighbouring stitches into fuzz. The
    pictures settle it: each neighbouring pair is registered (how far the picture slid), the slides
    are stitched into one mosaic by least squares, and the cameras are put where the mosaic says,
    with the arm's steps giving only the overall scale. The pictures are taken as they are, mirrored
    or not: the camera's axes are defined by them.

    Returns per-picture camera placement in a frame whose x and y are the picture's own axes (in
    metres, scaled by the arm) and the derived height, or None if the scan is not a grid or the
    pictures do not register.
    """
    stations = {}
    for capture in captures:
        if capture.get("row") is None or capture.get("col") is None or capture.get("camera_position") is None:
            continue
        stations.setdefault((int(capture["row"]), int(capture["col"])), capture)
    if len(stations) < 6:
        return None
    rough = measure_image_turn(captures, fov_degrees)
    if rough is None:
        return None
    pictures = {}

    def picture(key):
        if key not in pictures:
            pictures[key] = _normalised_grey(Path(str(stations[key].get("path") or "")))
        return pictures[key]

    # How the arm's step maps to a slide, roughly (px per metre, 2x2): the prior for every pair,
    # and the layout the mosaic is held near, so a pair that matched a stitch over cannot drag it.
    steps = np.column_stack([np.asarray(rough["step_x_mm"]) / 1000.0, np.asarray(rough["step_y_mm"]) / 1000.0])
    slides = np.column_stack([rough["slide_x"], rough["slide_y"]])
    slide_per_m = slides @ np.linalg.inv(steps)
    keys = sorted(stations)
    index = {key: i for i, key in enumerate(keys)}
    arm_xy = np.array([np.asarray(stations[key]["camera_position"][:2], dtype=float) for key in keys])
    layout = -(arm_xy - arm_xy[0]) @ slide_per_m.T          # where the arm says each picture's origin lies, in px

    # Every neighbouring pair, registered within a few pixels of what the arm's step predicts.
    edges = []     # (key_a, key_b, slide in full pixels, correlation)
    for key in stations:
        for step in ((0, 1), (1, 0)):
            other = (key[0] + step[0], key[1] + step[1])
            if other not in stations:
                continue
            prior = (layout[index[key]] - layout[index[other]]) / 2.0
            try:
                dx, dy, score = _slide_between(picture(key), picture(other), prior, window=10, step=1)
            except OSError:
                continue
            if score >= min_correlation:
                edges.append((key, other, np.array([dx * 2.0, dy * 2.0]), score))
    if len(edges) < len(stations) - 1:
        return None

    # Least squares: picture j's origin q_j = q_i - slide_ij over all pairs (weighted by their
    # correlation), with a light pull of every picture towards the arm's layout.
    rows, rhs, weights = [], [], []
    for a, b, slide, score in edges:
        row = np.zeros(len(keys))
        row[index[b]] = 1.0
        row[index[a]] = -1.0
        rows.append(row)
        rhs.append(-slide)
        weights.append(score)
    for i in range(len(keys)):
        rows.append(np.eye(len(keys))[i])
        rhs.append(layout[i])
        weights.append(0.02)
    rows.append(np.eye(len(keys))[0])       # anchor the first picture
    rhs.append(np.zeros(2))
    weights.append(1.0)
    rows, rhs, weights = np.array(rows), np.array(rhs), np.array(weights)
    keep = np.ones(len(rows), dtype=bool)
    for _ in range(2):
        system = rows[keep] * np.sqrt(weights[keep])[:, None]
        target = rhs[keep] * np.sqrt(weights[keep])[:, None]
        origins = np.linalg.lstsq(system, target, rcond=None)[0]      # px, per picture: (x, y)
        misfit = np.sqrt(((rows @ origins - rhs) ** 2).sum(axis=1))
        pairs_only = misfit[:len(edges)]
        # A pair that matched a stitch over stands out against the rest; drop it and solve again.
        keep[:len(edges)] = pairs_only <= max(3.0 * np.median(pairs_only), 6.0)
    residual = (rows[:len(edges)][keep[:len(edges)]] @ origins - rhs[:len(edges)][keep[:len(edges)]])
    edges = [edge for edge, kept in zip(edges, keep[:len(edges)]) if kept]
    # Drop pictures no edge reached (an isolated station gets the anchor's answer).
    reached = {a for a, _, _, _ in edges} | {b for _, b, _, _ in edges}

    # Scale: the arm's steps against the mosaic's, over the registered pairs.
    arm, mosaic = [], []
    for a, b, slide, _ in edges:
        arm.append(np.linalg.norm(np.subtract(stations[b]["camera_position"][:2], stations[a]["camera_position"][:2])))
        mosaic.append(np.linalg.norm(origins[index[b]] - origins[index[a]]))
    px_per_m = float(np.median(np.array(mosaic) / np.maximum(np.array(arm), 1e-9)))
    width = int(next(iter(stations.values())).get("camera_image_width", 640))
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    height = focal / px_per_m
    placements = {key: origins[index[key]] / px_per_m for key in keys if key in reached}
    return {"placements": placements, "height_m": float(height), "px_per_mm": px_per_m / 1000.0,
            "pairs": len(edges), "pictures": len(placements), "unplaced": [key for key in keys if key not in reached],
            "residual_px": float(np.sqrt((residual ** 2).sum(axis=1).mean()))}


def image_turn_block(capture: dict, default=None) -> tuple[np.ndarray, bool]:
    """The capture's image turn as a 3x3 on the tool frame, and whether the image is mirrored.

    Recorded with each capture by the plan (scan_geometry.calibrate_image_turn); `default` stands
    in for captures from before it was. A mirrored image has no proper rotation, so the turn is
    returned without its mirror and the caller flips the image instead.
    """
    turn = np.asarray(capture.get("camera_image_turn") or default or [[1.0, 0.0], [0.0, 1.0]], dtype=float)
    block = np.eye(3)
    block[:2, :2] = turn
    mirrored = bool(np.linalg.det(turn) < 0.0)
    if mirrored:
        block = block @ np.diag([-1.0, 1.0, 1.0])
    return block, mirrored


def capture_camera_pose(capture: dict, hand_eye: np.ndarray):
    """The camera's pose for one capture, and the flange-to-camera transform still to apply to it.

    A capture planned by scan_geometry records the camera itself - the point and attitude the
    joints were solved for - so it is used as it is, with nothing added. Only a capture without
    one falls back to the controller's tool pose and the hand-eye guess: adding the guess to a
    pose that is already the camera's put every camera a tool length off.
    """
    planned = capture.get("target_position")
    if capture.get("camera_position") is not None and planned is not None and len(planned) >= 6:
        position = [float(v) for v in capture["camera_position"]]
        # The controller reports where *its* tool point went. When that sits a tool length further
        # along the tool axis than the planned point, the controller carried a tool-centre offset the
        # planner did not know of: the flange - and the camera on it - stood that far back from the
        # planned point. (A sweep at a planned 5 cm came out at 15 cm this way: the pictures slid
        # half as far per step as planned.) The camera is put where it stood.
        reported = capture.get("actual_tcp_pose")
        if reported is not None and len(reported) >= 6:
            axis = Rotation.from_rotvec(np.asarray(planned[3:6], dtype=float)).as_matrix() @ np.array([0.0, 0.0, 1.0])
            apart = np.asarray(reported[:3], dtype=float) - np.asarray(position, dtype=float)
            along = float(apart @ axis)
            if abs(along) > TOOL_CENTRE_MISMATCH_M and np.linalg.norm(apart - along * axis) < TOOL_CENTRE_MISMATCH_M:
                position = [float(v) for v in np.asarray(position, dtype=float) - along * axis]
        return [*position, *[float(v) for v in planned[3:6]]], np.eye(4)
    pose = capture.get("actual_tcp_pose") or planned
    if pose is None or len(np.asarray(pose, dtype=float).reshape(-1)) < 6:
        return None, hand_eye
    return pose, hand_eye


def intrinsics_from_fov(width: int, height: int, fov_degrees: float = DEFAULT_FOV_DEGREES) -> dict[str, float]:
    """Pinhole intrinsics assuming a horizontal field of view."""
    width = max(1, int(width))
    height = max(1, int(height))
    fov = math.radians(float(np.clip(float(fov_degrees), 5.0, 170.0)))
    fx = width / (2.0 * math.tan(fov / 2.0))
    return {
        "fl_x": float(fx),
        "fl_y": float(fx),
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
        "w": int(width),
        "h": int(height),
        "camera_angle_x": float(fov),
    }


SEED_PLY_NAME = "fabric_seed.ply"
SEED_MAX_POINTS = 100_000


def write_fabric_seed(captures, path: Path, fov_degrees: float, image_turn=None) -> int:
    """Starting points for the splat: the fabric mosaic as coloured points on the fabric plane.

    Without them splatfacto scatters random Gaussians through a box metres wide around a cloth a
    hand across, and the few views of a scan never clear them - the result is a shell of floaters.
    The fabric's height comes from the captures themselves (camera height minus height over the
    fabric). Returns how many points were written; 0, and no file, when no capture recorded it.
    """
    heights = [float(c["camera_position"][2]) - float(c["camera_height_above_fabric_m"])
               for c in captures if c.get("camera_position") is not None and c.get("camera_height_above_fabric_m") is not None]
    if not heights:
        return 0
    fabric_z = float(np.median(heights))
    mosaic, weight, low, high, metres_per_px, _ = fabric_mosaic(captures, fabric_z, fov_degrees, None, image_turn)
    rows, cols = np.nonzero(weight > 0)
    stride = max(1, int(np.ceil(np.sqrt(len(rows) / SEED_MAX_POINTS))))
    keep = (rows % stride == 0) & (cols % stride == 0)
    rows, cols = rows[keep], cols[keep]
    table = np.empty(len(rows), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    table["x"], table["y"], table["z"] = low[0] + cols * metres_per_px, high[1] - rows * metres_per_px, fabric_z
    table["red"], table["green"], table["blue"] = mosaic[rows, cols, 0], mosaic[rows, cols, 1], mosaic[rows, cols, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(table)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        table.tofile(handle)
    return len(table)


OBJECT_MAX_GREY = 60
OBJECT_MASK_GROW_PX = 12
OBJECT_MASK_CLOSE_PX = 7


def object_silhouette(image) -> np.ndarray:
    """Where one dark object is in a picture of it on light paper: a boolean mask, grown a little.

    Dark pixels are the object's edges and shadows; a chrome face mirrors the paper and is as light
    as it, so the mask is whatever the dark outline encloses (the paper is what the frame's edge
    reaches; anything else inside is the object), then grown so the edge and its shadow are kept."""
    from PIL import Image, ImageDraw, ImageFilter

    grey = np.asarray(image.convert("L"))
    dark = Image.fromarray(((grey < OBJECT_MAX_GREY) * 255).astype(np.uint8))
    closed = dark.filter(ImageFilter.MaxFilter(OBJECT_MASK_CLOSE_PX)).filter(ImageFilter.MinFilter(OBJECT_MASK_CLOSE_PX))
    width, height = closed.size
    for x, y in [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1),
                 (width // 2, 0), (width // 2, height - 1), (0, height // 2), (width - 1, height // 2)]:
        if closed.getpixel((x, y)) == 0:
            ImageDraw.floodfill(closed, (x, y), 128)
    mask = Image.fromarray(((np.asarray(closed) != 128) * 255).astype(np.uint8))
    return np.asarray(mask.filter(ImageFilter.MaxFilter(OBJECT_MASK_GROW_PX * 2 + 1))) > 0


EXPOSURE_PERCENTILE = 95.0
EXPOSURE_GAIN_LIMITS = (0.5, 2.0)


def exposure_gains(paths) -> dict[str, np.ndarray]:
    """A per-channel gain for each picture that brings its bright tones (the white yarn, present in
    nearly every frame) to the set's median, undoing the camera's automatic exposure and white
    balance. Those changed from frame to frame - brightness by up to 2x, red against blue by 60 % -
    so the same stitch had a different colour in every picture, and a splat, which gives a point one
    colour, could only blur. Measured at a quarter size; the gain is clipped so an odd frame cannot
    be pushed too far."""
    from PIL import Image

    bright = {}
    for path in paths:
        try:
            with Image.open(path) as image:
                small = np.asarray(image.convert("RGB").resize((max(8, image.width // 4), max(6, image.height // 4))), dtype=float)
        except OSError:
            continue
        bright[str(path)] = np.percentile(small.reshape(-1, 3), EXPOSURE_PERCENTILE, axis=0)
    if not bright:
        return {}
    reference = np.median(np.array(list(bright.values())), axis=0)
    return {key: np.clip(reference / np.maximum(value, 1.0), *EXPOSURE_GAIN_LIMITS) for key, value in bright.items()}


def equalised_image(source: Path, gain: np.ndarray, mirrored: bool = False):
    from PIL import Image

    with Image.open(source) as image:
        rgb = image.convert("RGB")
        if mirrored:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        pixels = np.asarray(rgb, dtype=float) * np.asarray(gain, dtype=float)[None, None, :]
        return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def object_only_image(source: Path, target: Path, mirrored: bool = False) -> float:
    """The picture with everything but the object painted black, saved to `target`. Returns the share
    of the frame kept."""
    from PIL import Image

    with Image.open(source) as image:
        rgb = image.convert("RGB")
        if mirrored:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        mask = object_silhouette(rgb)
        pixels = np.asarray(rgb).copy()
        pixels[~mask] = 0
        Image.fromarray(pixels).save(target)
    return float(mask.mean())


def write_target_seed(target: np.ndarray, path: Path, camera_centres, spacing_m: float = 0.0005, surface: bool = True) -> int:
    """Starting points for a splat of one object: a ball of dark points at the calibrated target and
    a light disc of the surface it sits on, out to where the farthest camera could see.

    The target the calibration fitted is the object's visible centre, so the surface is put a little
    under it. Colours are only a start; the trainer repaints them within the first few hundred steps."""
    centres = np.asarray(camera_centres, dtype=float).reshape(-1, 3)
    reach = float(np.max(np.linalg.norm(centres - target, axis=1))) if len(centres) else 0.1
    radius = max(0.02, 0.6 * reach)
    axis = np.arange(-radius, radius + spacing_m, spacing_m)
    xs, ys = np.meshgrid(axis, axis)
    inside = xs ** 2 + ys ** 2 <= radius ** 2
    disc = np.stack([target[0] + xs[inside], target[1] + ys[inside], np.full(inside.sum(), target[2] - 0.004)], axis=1)
    ball = np.random.default_rng(0).normal(size=(3000 if surface else 20000, 3)) * (0.003 if surface else 0.004) + target
    points = np.concatenate([disc, ball]) if surface else ball
    if len(points) > SEED_MAX_POINTS:
        points = points[np.random.default_rng(1).choice(len(points), SEED_MAX_POINTS, replace=False)]
    table = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    table["x"], table["y"], table["z"] = points[:, 0], points[:, 1], points[:, 2]
    dark = np.linalg.norm(points - target, axis=1) < 0.01
    table["red"] = table["green"] = table["blue"] = np.where(dark, 40, 200).astype(np.uint8)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(table)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        table.tofile(handle)
    return len(table)


def export_dataset(
    captures,
    output_dir: str | Path,
    *,
    session_id: str = "",
    hand_eye: np.ndarray | None = None,
    fov_degrees: float = DEFAULT_FOV_DEGREES,
    copy_images: bool = True,
    image_turn=None,
    calibration: dict | None = None,
    object_only: bool = False,
    register_pictures: bool = False,
    equalise_exposure: bool = True,
) -> SplatDataset:
    """Write a splatting-ready dataset from capture records.

    Each capture must carry an image path and the pose it was taken from --
    ``actual_tcp_pose`` where the robot reported one, otherwise the commanded
    ``target_position``. Captures missing either are skipped and reported in
    ``warnings`` rather than silently dropped, since a quietly short dataset
    reconstructs badly for no visible reason.

    With a measured ``calibration`` (hand_eye.fit, saved as config/hand_eye.json) every camera is
    the arm's actual TCP pose with the measured flange-to-camera transform applied, and the lens
    field of view is the calibration's - the planner's idea of where the camera was is ignored,
    because it is exactly what the calibration found wrong. The seed points then surround the
    calibrated target instead of being laid on a fabric plane the planner mis-placed.

    ``object_only`` paints everything but the one dark object black in the exported pictures and
    seeds only the object, and the trainer is told to use a black background: the splat is then the
    object alone, with no paper to fill with Gaussians - which is where most of the floaters were.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    colmap_dir = output_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    colmap_dir.mkdir(parents=True, exist_ok=True)
    # An earlier export's seed must not survive into this one.
    for stale in (SEED_PLY_NAME, "sparse_pc.ply"):
        if (output_dir / stale).exists():
            (output_dir / stale).unlink()

    if calibration is not None:
        import hand_eye as hand_eye_module

        hand_eye = hand_eye_module.calibration_matrix(calibration)
        fov_degrees = float(calibration.get("fov_degrees", fov_degrees))
    else:
        hand_eye = hand_eye_matrix() if hand_eye is None else np.asarray(hand_eye, dtype=float)
    warnings: list[str] = []
    gains = exposure_gains([Path(str(c.get("path") or c.get("image_path") or "")) for c in captures]) if equalise_exposure and not object_only else {}
    if gains:
        spread = np.array(list(gains.values()))
        logging.info("Exposure gains: %.2f-%.2f", spread.min(), spread.max())
    # Without a mount calibration the image's axes are assumed to be the tool's, unless a turn was
    # calibrated. A grid or sweep scan can measure its own: it did here, and found the image mirrored.
    measured_turn = None
    registered = None
    if calibration is None and (image_turn is None or np.allclose(image_turn, np.eye(2))):
        # A grid or sweep over a flat cloth can place its own cameras from the pictures, the arm's
        # steps giving the scale (register_sweep) - off unless asked: on a repeating knit the
        # picture matching is the less reliable of the two (neighbour distances came out 10-27 mm
        # for an 18 mm step), so the arm's steps are kept and only the image turn is measured.
        try:
            registered = register_sweep(captures, fov_degrees) if register_pictures else None
        except Exception as exc:    # a measurement is a bonus, never a reason to fail an export
            warnings.append(f"the pictures could not be registered: {exc}")
        if registered is not None:
            warnings.append(f"cameras placed by registering the pictures ({registered['pictures']} placed from {registered['pairs']} "
                            f"neighbour pairs, {registered['residual_px']:.0f} px misfit; {registered['px_per_mm']:.1f} px/mm, "
                            f"camera {registered['height_m'] * 100:.1f} cm over the cloth)")
            for key in registered["unplaced"]:
                warnings.append(f"not placed: row {key[0]} col {key[1]} (its pictures matched no neighbour)")
        else:
            try:
                measured_turn = measure_image_turn(captures, fov_degrees)
            except Exception as exc:
                warnings.append(f"the image turn could not be measured from the scan: {exc}")
            if measured_turn is not None:
                image_turn = measured_turn["turn"]
                warnings.append("the image turn was measured from the scan's own pictures: "
                                + ("mirrored, " if measured_turn["mirrored"] else "")
                                + f"{measured_turn['px_per_mm']:.1f} px/mm on the cloth, camera {measured_turn['height_m'] * 100:.1f} cm over it")
    frames: list[dict[str, Any]] = []
    colmap_images: list[str] = []
    width = height = 0

    for index, capture in enumerate(captures):
        source = Path(str(capture.get("path") or capture.get("image_path") or ""))
        if not source.exists():
            warnings.append(f"missing image: {source.name or '(no path)'}")
            continue
        placed = None
        if registered is not None:
            key = (int(capture.get("row", -1)), int(capture.get("col", -1)))
            placed = registered["placements"].get(key)
            if placed is None:
                continue
            pose, camera_hand_eye = [float(placed[0]), float(placed[1]), float(registered["height_m"]), math.pi, 0.0, 0.0], np.eye(4)
        elif calibration is not None:
            pose = hand_eye_module.arm_pose(capture)
            camera_hand_eye = hand_eye
            reported = capture.get("actual_tcp_pose")
            if pose is not None and reported is not None and len(reported) >= 6 \
                    and pose != [float(v) for v in reported[:6]] and not any(w.startswith("the arm's reported poses") for w in warnings):
                warnings.append("the arm's reported poses sit far from the planned ones (a different tool-centre setting "
                                "that day?); the planned poses are used")
        else:
            pose, camera_hand_eye = capture_camera_pose(capture, hand_eye)
            if (pose is not None and capture.get("camera_position") is not None
                    and np.linalg.norm(np.asarray(pose[:3]) - np.asarray(capture["camera_position"], dtype=float)) > TOOL_CENTRE_MISMATCH_M
                    and not any(w.startswith("the controller reported") for w in warnings)):
                warnings.append("the controller reported its tool point a tool length past the planned one; the cameras are "
                                "put that far back from the planned points, where the flange stood")
        if pose is None:
            warnings.append(f"no camera pose for {source.name}")
            continue

        if width == 0:
            width = int(capture.get("camera_image_width", 0))
            height = int(capture.get("camera_image_height", 0))
            if width <= 0 or height <= 0:
                from PIL import Image

                with Image.open(source) as probe:
                    width, height = probe.size

        name = source.name
        target = images_dir / name
        # A calibration was fitted on the pictures as they are, so it already holds any turn of the
        # image relative to the tool; applying the planner's turn on top would undo it.
        if registered is not None:
            # The mosaic's axes are the picture's own; a right-handed camera over it needs the
            # picture flipped, which is what the planner's mirror handling does.
            turn, mirrored = image_turn_block({"camera_image_turn": [[1.0, 0.0], [0.0, -1.0]]})
        elif calibration is not None:
            turn, mirrored = np.eye(3), False
        elif measured_turn is not None:
            # Measured from these very pictures: it outranks the turn the plan recorded.
            turn, mirrored = image_turn_block({"camera_image_turn": measured_turn["turn"]})
        else:
            turn, mirrored = image_turn_block(capture, image_turn)
        if object_only:
            kept = object_only_image(source, target, mirrored)
            if kept < 0.002:
                warnings.append(f"no object found in {name}; it trains as a black frame")
            relative = f"images/{name}"
        elif gains:
            equalised_image(source, gains.get(str(source), np.ones(3)), mirrored).save(target)
            relative = f"images/{name}"
        elif copy_images or mirrored:
            if mirrored:
                from PIL import Image

                with Image.open(source) as image:
                    image.transpose(Image.FLIP_LEFT_RIGHT).save(target)
            elif not target.exists():
                shutil.copy2(source, target)
            relative = f"images/{name}"
        else:
            relative = str(source)

        c2w = camera_to_world(pose, camera_hand_eye)
        c2w[:3, :3] = c2w[:3, :3] @ turn
        frames.append({
            "file_path": relative,
            "transform_matrix": (c2w @ _CV_TO_GL).tolist(),
            "colmap_im_id": index + 1,
            "robot_pose": [float(v) for v in np.asarray(pose, dtype=float).reshape(-1)[:6]],
            "angle": str(capture.get("angle", "")),
            "row": int(capture.get("row", 0)),
            "col": int(capture.get("col", 0)),
        })

        # COLMAP stores world-to-camera, the inverse of what transforms.json
        # holds, with the rotation as a wxyz quaternion.
        w2c = np.linalg.inv(c2w)
        quat = Rotation.from_matrix(w2c[:3, :3]).as_quat()  # xyzw
        qw, qx, qy, qz = float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])
        tx, ty, tz = (float(v) for v in w2c[:3, 3])
        colmap_images.append(
            f"{index + 1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.9f} {ty:.9f} {tz:.9f} 1 {name}"
        )

    if not frames:
        warnings.append("no usable captures; nothing was exported")
        width, height = max(width, 1), max(height, 1)

    # A measured lens (scripts/calibrate_lens.py) beats any field-of-view guess: its focal length,
    # centre and distortion go in, and nerfstudio undistorts the pictures itself.
    measured = lens.load_intrinsics() if calibration is not None and calibration.get("lens", "").startswith("measured") else None
    intrinsics = lens.intrinsics_for(width or 1, height or 1, measured) if measured else intrinsics_from_fov(width or 1, height or 1, fov_degrees)
    if not frames:
        seed = 0
    elif registered is not None:
        centres = [np.array(frame["transform_matrix"])[:3, 3] for frame in frames]
        middle = np.mean(centres, axis=0)
        seed = write_target_seed(np.array([middle[0], middle[1], 0.004]), output_dir / SEED_PLY_NAME, centres)
    elif calibration is not None:
        seed = write_target_seed(np.asarray(calibration["target"], dtype=float), output_dir / SEED_PLY_NAME,
                                 [np.array(frame["transform_matrix"])[:3, 3] for frame in frames], surface=not object_only)
    else:
        seed = write_fabric_seed(captures, output_dir / SEED_PLY_NAME, fov_degrees, image_turn)
    transforms = {
        "camera_model": "PINHOLE",
        **intrinsics,
        "session_id": str(session_id),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pose_source": ("robot TCP poses with the measured hand-eye calibration (no structure-from-motion)" if calibration is not None
                        else "cameras registered from the pictures of a straight-down sweep, scaled by the arm's steps" if registered is not None
                        else "robot forward kinematics (no structure-from-motion)"),
        **({"sweep_registration": {k: v for k, v in registered.items() if k != "placements"}} if registered is not None else {}),
        "hand_eye_transform": np.asarray(hand_eye, dtype=float).tolist(),
        **({"hand_eye_calibration": {key: calibration[key] for key in ("rotvec", "translation", "target", "fov_degrees", "rms_px", "fitted_at")
                                     if key in calibration}} if calibration is not None else {}),
        **({"image_turn_measured": measured_turn} if measured_turn is not None else {}),
        "intrinsics_source": ("measured lens (config/camera_intrinsics.json), with distortion" if measured
                              else f"calibrated {float(fov_degrees):.1f} degree horizontal field of view" if calibration is not None
                              else f"assumed {float(fov_degrees):.1f} degree horizontal field of view"),
        **({"ply_file_path": SEED_PLY_NAME} if seed else {}),
        **({"object_only": True, "background": "black"} if object_only else {}),
        **({"exposure_equalised": True} if gains else {}),
        "frames": frames,
    }
    if frames and not seed:
        warnings.append("no fabric height recorded with the captures: the trainer starts from random points")
    transforms_path = output_dir / "transforms.json"
    write_json(transforms_path, transforms, indent=2)

    _write_colmap_model(colmap_dir, colmap_images, intrinsics)

    return SplatDataset(
        root=output_dir,
        image_count=len(frames),
        transforms_path=transforms_path,
        colmap_dir=colmap_dir,
        session_id=str(session_id),
        warnings=warnings,
    )


def _write_colmap_model(colmap_dir: Path, image_lines: list[str], intrinsics: dict[str, float]) -> None:
    """COLMAP text model, for trainers that read that instead of transforms.json."""
    with (colmap_dir / "cameras.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write(
            f"1 PINHOLE {int(intrinsics['w'])} {int(intrinsics['h'])} "
            f"{intrinsics['fl_x']:.6f} {intrinsics['fl_y']:.6f} "
            f"{intrinsics['cx']:.6f} {intrinsics['cy']:.6f}\n"
        )
    with (colmap_dir / "images.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for line in image_lines:
            handle.write(line + "\n")
            # Second line per image is the 2D feature list. Empty: the poses
            # come from the robot, so there are no matched features to record.
            handle.write("\n")
    with (colmap_dir / "points3D.txt").open("w", encoding="utf-8") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")


class SplatTrainer:
    """Runs an external Gaussian Splatting trainer over an exported dataset.

    The command is whatever the user has installed -- the Inria reference
    implementation, nerfstudio's ``ns-train``, gsplat -- given as a template
    with ``{dataset}`` and ``{output}`` placeholders. Nothing about a specific
    trainer is assumed, and it runs in a thread with its log captured so a long
    training does not freeze the UI.
    """

    def __init__(self, command_template: str = "", dataset: SplatDataset | None = None, output_dir: str | Path | None = None):
        self.command_template = str(command_template)
        self.dataset = dataset
        self.output_dir = Path(output_dir) if output_dir else None
        self.status = "Not started"
        self.returncode: int | None = None
        self.log: list[str] = []
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def resolved_command(self) -> str:
        dataset_root = str(self.dataset.root) if self.dataset else ""
        output = str(self.output_dir) if self.output_dir else ""
        return self.command_template.format(dataset=dataset_root, output=output)

    def start(self) -> bool:
        if self.running:
            self.status = "Training already running"
            return False
        if not self.command_template.strip():
            self.status = "No trainer command configured"
            return False
        if self.dataset is None or self.dataset.image_count <= 0:
            self.status = "Export a dataset with images before training"
            return False
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        command = self.resolved_command()
        with self._lock:
            self.log = [f"$ {command}"]
        self.returncode = None
        self.status = "Training started"
        self._thread = threading.Thread(target=self._run, args=(command,), daemon=True, name="gsplat-train")
        self._thread.start()
        return True

    def _run(self, command: str) -> None:
        try:
            # Below normal priority on Windows, inherited by the whole trainer: a reconstruction is a
            # long background job, and at normal priority it starved the app and the desktop.
            self._process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
            )
        except Exception as exc:
            self.status = f"Could not start trainer: {exc}"
            self._append(str(exc))
            return
        assert self._process.stdout is not None
        # The whole output goes to train.log as well: the panel shows only the tail, and a failure's
        # cause is usually a traceback well above the last line.
        log_path = self.output_dir / "train.log" if self.output_dir else None
        with (log_path.open("w", encoding="utf-8") if log_path else open(os.devnull, "w")) as log_file:
            log_file.write(f"$ {command}\n")
            for line in self._process.stdout:
                log_file.write(line)
                log_file.flush()
                self._append(line.rstrip())
        self._process.wait()
        if log_path:
            self._append(f"Full log: {log_path}")
        self.returncode = int(self._process.returncode)
        self.status = (
            "Training finished" if self.returncode == 0 else f"Trainer exited with code {self.returncode}"
        )

    def _append(self, line: str) -> None:
        with self._lock:
            self.log.append(line)
            # Only the tail is ever displayed, and a long training prints tens
            # of thousands of lines.
            if len(self.log) > 400:
                self.log = self.log[-400:]

    def tail(self, count: int = 12) -> list[str]:
        with self._lock:
            return list(self.log[-int(count):])

    def stop(self) -> None:
        """Asks the run to stop and keep what it has: train_splat.ps1 watches for a STOP file in the
        output folder, ends the trainer, and exports the latest checkpoint. A run that does not
        answer in a minute (another trainer, a solve) is ended outright."""
        process = self._process
        if process is None or process.poll() is not None:
            self.status = "Training stopped"
            return
        if self.output_dir is not None and "train_splat.ps1" in self.command_template:
            try:
                (self.output_dir / "STOP").write_text("stop\n", encoding="utf-8")
                self.status = "Stopping - exporting the latest checkpoint..."

                def end_if_stuck(process=process):
                    try:
                        process.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        process.terminate()

                threading.Thread(target=end_if_stuck, daemon=True, name="gsplat-stop").start()
                return
            except OSError as exc:
                logging.warning("gaussian_splatting.stop: %s", exc)
        try:
            process.terminate()
        except Exception as exc:
            logging.warning("gaussian_splatting.stop: %s", exc)
        self.status = "Training stopped"

    def output_artifacts(self) -> list[Path]:
        """Reconstruction files the trainer produced, newest first."""
        if self.output_dir is None or not self.output_dir.exists():
            return []
        # A checkpoint the trainer is replacing can vanish between the walk and the stat.
        stamped = []
        for pattern in ("*.ply", "*.splat", "*.ckpt"):
            for path in self.output_dir.rglob(pattern):
                try:
                    stamped.append((path.stat().st_mtime, path))
                except OSError:
                    continue
        stamped.sort(key=lambda item: item[0], reverse=True)
        return [path for _, path in stamped]


# ============================================================================
# Seeing it before training: every capture laid onto the fabric where its pose says it looked
# ============================================================================

PREVIEW_MAX_SIDE = 1400


def fabric_mosaic(captures, fabric_z: float, fov_degrees: float = DEFAULT_FOV_DEGREES, outline=None, image_turn=None):
    """Each capture projected onto the fabric plane through its recorded pose, blended into one map.

    Returns the blended colours, how much capture landed on each cell (0 where none did), the map's
    world bounds and cell size, and the camera-to-world of each capture used.
    """
    from PIL import Image

    frames = []
    for capture in captures:
        pose, hand_eye = capture_camera_pose(capture, np.eye(4))
        source = Path(str(capture.get("path", "")))
        if pose is None or not source.exists():
            continue
        turn, mirrored = image_turn_block(capture, image_turn)
        with Image.open(source) as image:
            image = image.convert("RGB")
            rgb = np.asarray(image.transpose(Image.FLIP_LEFT_RIGHT) if mirrored else image, dtype=np.float32)
        c2w = camera_to_world(pose, hand_eye)
        c2w[:3, :3] = c2w[:3, :3] @ turn
        frames.append((c2w, rgb))
    if not frames:
        raise RuntimeError("no capture has both an image and a camera pose to lay down")

    height_px, width_px = frames[0][1].shape[:2]
    intrinsics = intrinsics_from_fov(width_px, height_px, fov_degrees)
    spread = math.tan(math.radians(fov_degrees) / 2.0)
    points = []
    for c2w, _ in frames:
        drop = max(float(c2w[2, 3] - fabric_z), 1e-3)
        points.append(c2w[:2, 3] + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * drop * spread * 1.6)
    points = np.vstack(points + ([np.asarray(outline, dtype=float)[:, :2]] if outline is not None and len(outline) else []))
    low, high = points.min(axis=0), points.max(axis=0)
    metres_per_px = float(max(high - low)) / PREVIEW_MAX_SIDE
    cols, rows = int(np.ceil((high[0] - low[0]) / metres_per_px)) + 1, int(np.ceil((high[1] - low[1]) / metres_per_px)) + 1
    xs = low[0] + np.arange(cols) * metres_per_px
    ys = high[1] - np.arange(rows) * metres_per_px
    world = np.stack(np.meshgrid(xs, ys), axis=-1)

    total = np.zeros((rows, cols, 3), dtype=np.float32)
    weight = np.zeros((rows, cols), dtype=np.float32)
    for c2w, rgb in frames:
        rotation, centre = c2w[:3, :3], c2w[:3, 3]
        offset = np.concatenate([world - centre[:2], np.full((rows, cols, 1), fabric_z - centre[2])], axis=-1)
        local = offset @ rotation
        ahead = local[..., 2] > 1e-6
        u = intrinsics["fl_x"] * local[..., 0] / np.where(ahead, local[..., 2], 1.0) + intrinsics["cx"]
        v = intrinsics["fl_y"] * local[..., 1] / np.where(ahead, local[..., 2], 1.0) + intrinsics["cy"]
        inside = ahead & (u >= 0) & (u < width_px - 1) & (v >= 0) & (v < height_px - 1)
        feather = np.clip(np.minimum(np.minimum(u, width_px - 1 - u) / width_px, np.minimum(v, height_px - 1 - v) / height_px) * 6.0, 0.0, 1.0)
        w = np.where(inside, feather + 1e-3, 0.0).astype(np.float32)
        sample = rgb[np.clip(v, 0, height_px - 1).astype(int), np.clip(u, 0, width_px - 1).astype(int)]
        total += sample * w[..., None]
        weight += w

    mosaic = np.where(weight[..., None] > 0, total / np.maximum(weight[..., None], 1e-6), 18.0).clip(0, 255).astype(np.uint8)
    return mosaic, weight, low, high, metres_per_px, [c2w for c2w, _ in frames]


def preview_mosaic(captures, fabric_z: float, fov_degrees: float = DEFAULT_FOV_DEGREES, outline=None, image_turn=None):
    """The fabric mosaic, marked with the outline and where each capture was taken from.

    This is what a trainer will believe: the same poses and the same pinhole the export writes.
    Where the poses are right the stitches line up across overlapping frames; where the camera's
    mount or roll is wrong they visibly do not - found here in seconds rather than after training.
    Returns the mosaic and the share of the outline (when given) that some capture covered.
    """
    from PIL import Image, ImageDraw

    mosaic, weight, low, high, metres_per_px, cameras = fabric_mosaic(captures, fabric_z, fov_degrees, outline, image_turn)
    image = Image.fromarray(mosaic, "RGB")
    image.info["world_bounds"] = (low.tolist(), high.tolist(), metres_per_px)
    draw = ImageDraw.Draw(image)

    def to_px(xy):
        return ((xy[0] - low[0]) / metres_per_px, (high[1] - xy[1]) / metres_per_px)

    covered = None
    if outline is not None and len(outline):
        ring = [to_px(corner) for corner in np.asarray(outline, dtype=float)[:, :2]]
        draw.line(ring + [ring[0]], fill=(255, 255, 255), width=2)
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).polygon(ring, fill=255)
        inside_outline = np.asarray(mask) > 0
        covered = float((weight[inside_outline] > 0).mean()) if inside_outline.any() else None
    for index, c2w in enumerate(cameras):
        x, y = to_px(c2w[:2, 3])
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(255, 200, 60), width=2)
        draw.text((x + 6, y - 6), str(index + 1), fill=(255, 200, 60))
    return image, covered


# ============================================================================
# Seeing it after training: a trained splat, rendered from any side
# ============================================================================

SH_C0 = 0.28209479177387814
# The viewer draws at most this many Gaussians - an even sample, which reads the same at preview
# size and keeps a slider drag instant on a million-Gaussian splat.
VIEWER_MAX_POINTS = 150_000
# A Gaussian is drawn out to 2.5 sigma; one wider than this on screen is a blob at the eye, and skipped.
VIEWER_MAX_RADIUS_PX = 80
VIEWER_DEPTH_LAYERS = 24
# The widest a Gaussian is drawn in the filled quick preview, pixels either side of its middle.
QUICK_FILL_PX = 6
_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4", "int32": "<i4"}


def load_splat_ply(path) -> dict[str, np.ndarray]:
    """A Gaussian-splat PLY (the 3DGS layout nerfstudio and gsplat export) as positions, colours and opacities."""
    with open(path, "rb") as handle:
        header, properties, count = [], [], 0
        while True:
            line = handle.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property"):
                _, kind, name = line.split()[:3]
                properties.append((name, _PLY_TYPES[kind]))
            elif line == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise RuntimeError("only binary little-endian splat PLYs are read")
        # The exporter writes the file while the viewer polls for it: a short read is a file still
        # being written, not a small splat, and showed as a few hundred Gaussians out of 100,000.
        expected = handle.tell() + count * np.dtype(properties).itemsize
        if os.path.getsize(path) < expected:
            raise RuntimeError(f"{Path(path).name} is still being written ({os.path.getsize(path)} of {expected} bytes)")
        table = np.fromfile(handle, dtype=np.dtype(properties), count=count)

    names = table.dtype.names
    positions = np.stack([table["x"], table["y"], table["z"]], axis=1).astype(np.float32)
    if "f_dc_0" in names:
        colours = np.clip(0.5 + SH_C0 * np.stack([table["f_dc_0"], table["f_dc_1"], table["f_dc_2"]], axis=1), 0.0, 1.0)
    else:
        colours = np.stack([table["red"], table["green"], table["blue"]], axis=1) / 255.0
    opacity = 1.0 / (1.0 + np.exp(-table["opacity"])) if "opacity" in names else np.ones(count)
    if "scale_0" in names:
        scales = np.exp(np.stack([table["scale_0"], table["scale_1"], table["scale_2"]], axis=1)).astype(np.float32)
        radius = scales.max(axis=1)
        sigma = np.sort(scales, axis=1)[:, 1:].mean(axis=1)
    else:
        scales = np.zeros((count, 3), dtype=np.float32)
        radius = sigma = np.zeros(count)
    if "rot_0" in names:
        quaternion = np.stack([table["rot_0"], table["rot_1"], table["rot_2"], table["rot_3"]], axis=1).astype(np.float32)  # w x y z
    else:
        quaternion = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1))
    splat = {"positions": positions, "colours": colours.astype(np.float32), "opacity": np.asarray(opacity, dtype=np.float32),
             "radius": np.asarray(radius, dtype=np.float32), "sigma": np.asarray(sigma, dtype=np.float32),
             "scales": scales, "quaternion": quaternion, "count": int(count), "table": table}
    if count > VIEWER_MAX_POINTS:
        keep = np.random.default_rng(0).choice(count, VIEWER_MAX_POINTS, replace=False)
        splat.update({key: splat[key][keep] for key in ("positions", "colours", "opacity", "radius", "sigma", "scales", "quaternion")})
        splat["table"] = table[keep]
    return splat


def _covariances(scales: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Each Gaussian's 3x3 covariance from its axis scales and orientation (w, x, y, z)."""
    q = quaternion / np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-9)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rotation = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)
    rs = rotation * scales[:, None, :]
    return rs @ np.transpose(rs, (0, 2, 1))


# Floaters: Gaussians that do not touch the main body of the object, and ones that cannot be seen
# (near-black against a black background, or nearly transparent). The body is the largest connected
# group of occupied cells on a grid this many cells across the splat's extent.
CLEAN_GRID_CELLS = 48
CLEAN_MIN_OPACITY = 0.05
CLEAN_MIN_LUMINANCE = 0.06


def clean_splat(source, target, object_only: bool = True) -> dict[str, int]:
    """`source` PLY with the floaters removed, written to `target`. Returns what was dropped and why."""
    splat = load_splat_ply(source)
    table = splat["table"]
    positions, colours, opacity = splat["positions"], splat["colours"], splat["opacity"]
    keep = opacity >= CLEAN_MIN_OPACITY
    invisible = int((~keep).sum())
    dark = 0
    if object_only:
        black = colours.mean(axis=1) < CLEAN_MIN_LUMINANCE
        dark = int((black & keep).sum())
        keep &= ~black

    if not keep.any():
        raise RuntimeError("nothing visible would be left")
    # Connected components on a coarse grid; keep the largest.
    low, high = positions[keep].min(axis=0), positions[keep].max(axis=0)
    cell = max(float((high - low).max()) / CLEAN_GRID_CELLS, 1e-9)
    index = np.floor((positions - low) / cell).astype(np.int64)
    shape = tuple(int(v) + 2 for v in np.floor((high - low) / cell).astype(np.int64))
    occupied = np.zeros(shape, dtype=np.int32)
    inside = keep & np.all(index >= 0, axis=1) & np.all(index < np.array(shape), axis=1)
    occupied[tuple(index[inside].T)] = 1
    labels = np.zeros(shape, dtype=np.int32)
    next_label = 0
    for seed in zip(*np.nonzero(occupied)):
        if labels[seed]:
            continue
        next_label += 1
        stack = [seed]
        labels[seed] = next_label
        while stack:
            x, y, z = stack.pop()
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                n = (x + dx, y + dy, z + dz)
                if all(0 <= n[i] < shape[i] for i in range(3)) and occupied[n] and not labels[n]:
                    labels[n] = next_label
                    stack.append(n)
    if next_label:
        sizes = np.bincount(labels[occupied > 0], minlength=next_label + 1)
        main = int(np.argmax(sizes[1:]) + 1)
        attached = inside & (labels[tuple(np.clip(index, 0, np.array(shape) - 1).T)] == main)
        detached = int((keep & ~attached).sum())
        keep &= attached
    else:
        detached = 0

    kept = table[keep]
    kinds = {"<f4": "float", "<f8": "double", "u1": "uchar", "<i4": "int"}
    header = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(kept)}\n"
              + "".join(f"property {kinds.get(kept.dtype[name].str.replace('|', ''), 'float')} {name}\n" for name in kept.dtype.names)
              + "end_header\n")
    with open(target, "wb") as handle:
        handle.write(header.encode("ascii"))
        kept.tofile(handle)
    return {"kept": int(len(kept)), "invisible": invisible, "dark": dark, "detached": detached, "total": int(len(table))}


def orbit_frame(points):
    """The middle an orbit turns round and the extent its distance is set by: the splat's median, and
    the radius holding 95% of it."""
    centre = np.median(points, axis=0)
    return centre, float(np.percentile(np.linalg.norm(points - centre, axis=1), 95)) or 1.0


def orbit_camera(points, azimuth_deg, elevation_deg, zoom, pan=(0.0, 0.0), frame=None):
    """Where the viewer's eye is and which way it looks, orbiting the splat's middle. `pan` slides
    the middle sideways and up, as a share of the splat's extent. `frame` (centre, extent) fixes the
    middle, so two splats drawn with the same one are seen from exactly the same eye."""
    centre, extent = frame if frame is not None else orbit_frame(points)
    centre = np.asarray(centre, dtype=np.float64)
    azimuth, elevation = math.radians(azimuth_deg), math.radians(elevation_deg)
    toward = np.array([math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])
    forward = -toward
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right = right / np.linalg.norm(right) if np.linalg.norm(right) > 1e-6 else np.array([1.0, 0.0, 0.0])
    up = np.cross(right, forward)
    centre = centre + (right * float(pan[0]) + up * float(pan[1])) * extent
    eye = centre + toward * extent * 2.5 / max(float(zoom), 0.05)
    return eye, forward, right, up


def render_splat_quick(splat, azimuth_deg=0.0, elevation_deg=60.0, zoom=1.0, size=(640, 480), background=(18, 18, 24), pan=(0.0, 0.0),
                       frame=None, fill=False):
    """A few-millisecond stand-in while the view is being dragged: every Gaussian a single dot.

    `fill`: every Gaussian instead a dot as wide as it shows on the screen (up to QUICK_FILL_PX), all
    of them drawn - zoomed in, single dots scatter into points with the cloth between them missing."""
    from PIL import Image

    keep = splat["opacity"] >= 0.1
    points, colours = splat["positions"][keep], splat["colours"][keep]
    sigma = splat.get("sigma", splat["radius"])[keep] if fill else None
    if len(points) > 40_000 and not fill:          # enough dots to show the shape at drag speed
        stride = len(points) // 40_000 + 1
        points, colours = points[::stride], colours[::stride]
    width, height = size
    canvas = np.tile(np.asarray(background, dtype=np.uint8), (height, width, 1))
    if len(points) == 0:
        return Image.fromarray(canvas, "RGB")
    eye, forward, right, up = orbit_camera(points, azimuth_deg, elevation_deg, zoom, pan, frame)
    relative = points - eye
    depth = relative @ forward
    ahead = depth > 1e-4
    focal = 0.5 * width / math.tan(math.radians(25.0))
    u = np.round((relative[ahead] @ right) / depth[ahead] * focal + width / 2.0).astype(int)
    v = np.round(-(relative[ahead] @ up) / depth[ahead] * focal + height / 2.0).astype(int)
    order = np.argsort(-depth[ahead])                # far first, so the near dots win
    u, v, rgb = u[order], v[order], (colours[ahead][order] * 255.0).astype(np.uint8)
    if not fill:
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        canvas[v[inside], u[inside]] = rgb[inside]
        return Image.fromarray(canvas, "RGB")
    reach = np.clip(np.round(1.5 * sigma[ahead][order] / depth[ahead][order] * focal), 0, QUICK_FILL_PX)
    # Zoomed in, most dots are off the screen: only the ones that can touch it are drawn.
    near = (u >= -reach) & (u < width + reach) & (v >= -reach) & (v < height + reach)
    u, v, rgb, reach_sq = u[near], v[near], rgb[near], (reach[near] ** 2).astype(int)
    # Widest offsets first, so each dot's own middle is drawn last over its rim; each ring of offsets
    # draws only the dots that reach it (most are small, and are drawn once or a few times).
    offsets = sorted(((dx, dy) for dx in range(-QUICK_FILL_PX, QUICK_FILL_PX + 1) for dy in range(-QUICK_FILL_PX, QUICK_FILL_PX + 1)
                      if dx * dx + dy * dy <= QUICK_FILL_PX * QUICK_FILL_PX), key=lambda o: -(o[0] ** 2 + o[1] ** 2))
    reaching = {}
    for dx, dy in offsets:
        ring = dx * dx + dy * dy
        if ring not in reaching:
            reaching[ring] = np.flatnonzero(reach_sq >= ring)
        index = reaching[ring]
        uu, vv = u[index] + dx, v[index] + dy
        draw = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
        canvas[vv[draw], uu[draw]] = rgb[index[draw]]
    return Image.fromarray(canvas, "RGB")


def render_splat(splat, azimuth_deg=0.0, elevation_deg=60.0, zoom=1.0, size=(640, 480), min_opacity=0.02, background=(18, 18, 24),
                 pan=(0.0, 0.0), frame=None, cancelled=None):
    """The splat seen from an orbit round its middle, rendered the way a splat is meant to be seen:
    every Gaussian a soft, semi-transparent blob, blended back to front.

    A CPU stand-in for the trainer's renderer, close enough to judge a reconstruction by. Each
    Gaussian is drawn round (its mean scale), and the depth range is cut into VIEWER_DEPTH_LAYERS
    slabs: within a slab the blobs are summed, between slabs they are composited over one another -
    the error of summing within a slab is invisible at this size and keeps a slider drag quick.
    `cancelled`, if given, is asked between slabs; when it says yes the render stops and returns None.
    """
    from PIL import Image

    keep = splat["opacity"] >= float(min_opacity)
    points, colours, opacity = splat["positions"][keep], splat["colours"][keep], splat["opacity"][keep]
    sigma = splat.get("sigma", splat["radius"])[keep]
    if len(points) == 0:
        return Image.new("RGB", size, background)
    eye, forward, right, up = orbit_camera(points, azimuth_deg, elevation_deg, zoom, pan, frame)

    relative = points - eye
    depth = relative @ forward
    ahead = depth > 1e-4
    width, height = size
    focal = 0.5 * width / math.tan(math.radians(25.0))
    # Camera axes with y down, as on the screen.
    cam_x, cam_y, cam_z = (relative[ahead] @ right), -(relative[ahead] @ up), depth[ahead]
    u = cam_x / cam_z * focal + width / 2.0
    v = cam_y / cam_z * focal + height / 2.0
    colours, opacity, depth = colours[ahead], opacity[ahead], depth[ahead]

    # Each Gaussian's footprint on the screen: its 3D covariance taken into camera axes and through
    # the projection's Jacobian (EWA splatting), plus a third of a pixel so nothing vanishes between pixels.
    if "scales" in splat and float(splat["scales"].max()) > 0.0:
        world_to_cam = np.stack([right, -up, forward])
        cov = world_to_cam @ _covariances(splat["scales"][keep][ahead], splat["quaternion"][keep][ahead]) @ world_to_cam.T
        jac = np.zeros((len(cam_z), 2, 3))
        jac[:, 0, 0] = focal / cam_z
        jac[:, 0, 2] = -focal * cam_x / cam_z ** 2
        jac[:, 1, 1] = focal / cam_z
        jac[:, 1, 2] = -focal * cam_y / cam_z ** 2
        cov2d = jac @ cov @ np.transpose(jac, (0, 2, 1))
    else:
        s2 = np.clip(sigma[ahead] / cam_z * focal, 0.5, None) ** 2
        cov2d = np.zeros((len(cam_z), 2, 2))
        cov2d[:, 0, 0] = cov2d[:, 1, 1] = s2
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3
    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    largest = 0.5 * (a + c) + np.sqrt(np.maximum(0.25 * (a - c) ** 2 + b * b, 0.0))
    # Close up, a blob can span the frame; it is drawn out to the cap rather than left out, so a
    # zoomed-in view stays a surface and does not fall apart into dots.
    radius_px = np.minimum(np.ceil(2.0 * np.sqrt(largest)).astype(int), VIEWER_MAX_RADIUS_PX)
    too_big = np.zeros(len(radius_px), dtype=bool)
    det = np.maximum(a * c - b * b, 1e-12)
    inv_a, inv_b, inv_c = c / det, -b / det, a / det
    on_screen = (u > -radius_px) & (u < width + radius_px) & (v > -radius_px) & (v < height + radius_px) & ~too_big
    u, v, radius_px, colours, opacity, depth, inv_a, inv_b, inv_c = (
        arr[on_screen] for arr in (u, v, radius_px, colours, opacity, depth, inv_a, inv_b, inv_c))

    canvas = np.tile(np.asarray(background, dtype=np.float64) / 255.0, (height, width, 1))
    if len(u) == 0:
        return Image.fromarray((canvas * 255).astype(np.uint8), "RGB")
    # Far slabs first; each slab is summed, then laid over what is behind it.
    edges = np.quantile(depth, np.linspace(0.0, 1.0, VIEWER_DEPTH_LAYERS + 1))
    slab = np.clip(np.searchsorted(edges, depth, side="right") - 1, 0, VIEWER_DEPTH_LAYERS - 1)
    for layer in range(VIEWER_DEPTH_LAYERS - 1, -1, -1):
        # A viewer that has moved on asks for no more of this picture: it stops between slabs.
        if cancelled is not None and cancelled():
            return None
        members = np.flatnonzero(slab == layer)
        if not len(members):
            continue
        weight_sum = np.zeros(height * width)
        clear_log = np.zeros(height * width)      # log of what still shows through the slab
        colour_sum = np.zeros((height * width, 3))
        for r in np.unique(radius_px[members]):
            bucket = members[radius_px[members] == r]
            offsets = np.arange(-int(r), int(r) + 1)
            du, dv = np.meshgrid(offsets, offsets)
            du, dv = du.ravel(), dv.ravel()
            # Blobs x pixels per blob is kept to a few million elements at a time.
            for this in np.array_split(bucket, max(1, int(np.ceil(len(bucket) * len(du) / 3_000_000)))):
                cx, cy = np.round(u[this]).astype(int), np.round(v[this]).astype(int)
                x = cx[:, None] + du[None, :]
                y = cy[:, None] + dv[None, :]
                dx = x - u[this][:, None]
                dy = y - v[this][:, None]
                power = inv_a[this][:, None] * dx * dx + 2.0 * inv_b[this][:, None] * dx * dy + inv_c[this][:, None] * dy * dy
                weight = opacity[this][:, None] * np.exp(-0.5 * power)
                inside = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (weight > 1e-3)
                index = (y * width + x)[inside]
                w = weight[inside]
                weight_sum += np.bincount(index, weights=w, minlength=height * width)
                clear_log += np.bincount(index, weights=np.log1p(-np.minimum(w, 0.999)), minlength=height * width)
                for channel in range(3):
                    tint = np.broadcast_to(colours[this][:, channel][:, None], weight.shape)[inside]
                    colour_sum[:, channel] += np.bincount(index, weights=w * tint, minlength=height * width)
        alpha = (1.0 - np.exp(clear_log)).reshape(height, width, 1)
        colour = (colour_sum / np.maximum(weight_sum, 1e-9)[:, None]).reshape(height, width, 3)
        canvas = canvas * (1.0 - alpha) + colour * alpha
    return Image.fromarray((np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8), "RGB")
