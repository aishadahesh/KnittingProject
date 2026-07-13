"""
MuJoCo fabric scanner for the UR5e.

This keeps the existing Robot_fabric_scanner.py path planner intact and adds a
MuJoCo viewer based on the UR5-Python simulator idea. It builds a colored fabric
made of grid squares, samples multiple scan locations per square, visits each
location from several angles, and draws the executed TCP path while the robot
moves.

Run after installing the extra simulator deps:
    python mujoco_fabric_scanner.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from knitting_core import build_parametric_control_rows

from Robot_fabric_scanner import (
    DEFAULT_CENTER,
    DEFAULT_MAX_SPAN,
    TOOL_DOWN_ROTVEC,
    assess_plan_safety,
    fit_path_to_workspace,
    generate_urscript,
    points_and_rotvecs_to_poses,
    save_points_csv,
)

MUJOCO_DAMPING = 1e-2
MUJOCO_KP = 0.28
IK_SUBSTEPS = 3
TARGET_TOL = 0.006
TARGET_ROT_TOL = 0.08
MAX_TRAIL_POINTS = 6000
MARKER_RADIUS = 0.006
PATH_WIDTH = 0.003
POSE_AXIS_LENGTH = 0.035
FABRIC_THICKNESS = 0.003
IMAGE_REPEAT_OVERLAP = 1.68
SCAN_PATTERN_FILL = 1.08
CAMERA_MARKER_SIZE = np.array([0.018, 0.012, 0.010])
CAMERA_IMAGE_SIZE = (640, 480)
CAMERA_SAVE_EVERY_STATION = "station"
CAMERA_SAVE_EVERY_VIEW = "view"
CAMERA_CAPTURE_NATURAL = "natural"
CAMERA_CAPTURE_FOCUSED = "focused"
NON_SCAN_VIEW_NAMES = {"approach", "retreat", "travel"}
MODE_SIMULATION = "simulation"
MODE_ROBOT = "robot"
DEFAULT_ROBOT_IP = "132.74.121.230"
DEFAULT_ROBOT_PORT = 30001
ROBOT_MAX_CARTESIAN_STEP = 0.040

SWATCH_PALETTE = [
    (0.86, 0.12, 0.18, 1.0),
    (0.10, 0.39, 0.82, 1.0),
    (0.98, 0.78, 0.16, 1.0),
    (0.12, 0.62, 0.36, 1.0),
    (0.78, 0.18, 0.72, 1.0),
    (0.96, 0.43, 0.16, 1.0),
    (0.12, 0.70, 0.76, 1.0),
    (0.47, 0.34, 0.85, 1.0),
]


@dataclass
class FabricPlan:
    raw_stations: np.ndarray
    mapped_stations: np.ndarray
    raw_points: np.ndarray
    mapped_points: np.ndarray
    poses: np.ndarray
    station_ids: list[int]
    view_names: list[str]
    station_cells: list[tuple[int, int]]
    cell_colors: list[tuple[float, float, float, float]]
    cell_color_sets: list[list[tuple[float, float, float, float]]]
    fabric_origin: np.ndarray
    fabric_size: np.ndarray
    grid_rows: int
    grid_cols: int
    square_width: float
    square_length: float
    model_curves: list[np.ndarray]
    cell_model_curves: list[list[np.ndarray]] | None = None
    pattern_repeat_rows: int = 1
    pattern_repeat_cols: int = 1


def _serpentine_indices(rows: int, cols: int):
    for row in range(rows):
        col_range = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
        for col in col_range:
            yield row, col


def _normalize_palette(palette) -> list[tuple[float, float, float, float]]:
    if palette is None:
        return list(SWATCH_PALETTE)
    if isinstance(palette, np.ndarray):
        palette = palette.tolist()
    if (
        isinstance(palette, (list, tuple))
        and len(palette) >= 3
        and all(isinstance(v, (int, float, np.integer, np.floating)) for v in palette[:3])
    ):
        r, g, b = [float(v) for v in palette[:3]]
        a = float(palette[3]) if len(palette) > 3 else 1.0
        return [(r, g, b, a)]

    colors = []
    for color in palette or []:
        if isinstance(color, np.ndarray):
            color = color.tolist()
        if not isinstance(color, (list, tuple)) or len(color) < 3:
            continue
        r, g, b = [float(v) for v in color[:3]]
        a = float(color[3]) if len(color) > 3 else 1.0
        colors.append((r, g, b, a))
    return colors or list(SWATCH_PALETTE)


def _fallback_model_curves() -> list[np.ndarray]:
    return [
        np.asarray(curve, dtype=np.float32)
        for curve in _knit_cell_curves()
    ]


def _load_saved_model_curves(model_json: str | None) -> list[np.ndarray]:
    if not model_json:
        return _fallback_model_curves()

    path = Path(model_json)
    if not path.exists():
        return _fallback_model_curves()

    try:
        with path.open("r") as handle:
            saved = json.load(handle)

        ctrl_rows = [
            np.asarray(row, dtype=np.float32)
            for row in saved.get("spline_control_rows", [])
            if len(row) > 1
        ]
        if not ctrl_rows:
            return _fallback_model_curves()

        bitmap = np.asarray(saved.get("bitmap", np.ones((len(ctrl_rows), 1))), dtype=np.float32)
        period = np.asarray(saved.get("period_offset", [float(max(1, bitmap.shape[1])), 0.0, 0.0]), dtype=np.float32)

        curves = []
        for row_idx, row in enumerate(ctrl_rows):
            cp = np.asarray(row[:, :2], dtype=np.float32)
            cp_aug = np.vstack((cp, cp[0] + period[:2]))
            seg_lens = np.maximum(np.linalg.norm(np.diff(cp_aug, axis=0), axis=1), 1e-6)
            t = np.concatenate(([0.0], np.cumsum(seg_lens)))
            samples = max(80, len(cp) * 8)
            to = np.linspace(t[0], t[-1], samples, dtype=np.float32)
            detrended = cp_aug - period[:2][None, :] * (t / t[-1])[:, None]
            if len(cp) == 2:
                pts = np.column_stack([np.interp(to, t, detrended[:, i]) for i in range(2)])
            else:
                from scipy.interpolate import CubicSpline
                pts = np.column_stack([CubicSpline(t, detrended[:, i], bc_type="periodic")(to) for i in range(2)])
            pts = pts + period[:2][None, :] * (to / t[-1])[:, None]

            if row_idx < bitmap.shape[0]:
                straight_cols = np.flatnonzero(np.asarray(bitmap[row_idx], dtype=np.float32) <= 0.5)
                if len(straight_cols):
                    samples_per_col = max(1, len(cp) // max(1, bitmap.shape[1]))
                    for col_idx in straight_cols:
                        start = int(col_idx) * samples_per_col
                        end = min((int(col_idx) + 1) * samples_per_col, len(cp_aug) - 1)
                        if end <= start:
                            continue
                        t0, t1 = float(t[start]), float(t[end])
                        span = (to >= t0) & (to <= t1)
                        if np.any(span):
                            alpha = ((to[span] - t0) / max(t1 - t0, 1e-6))[:, None]
                            pts[span] = cp_aug[start] * (1.0 - alpha) + cp_aug[end] * alpha

            curves.append(pts.astype(np.float32))

        if not curves:
            return _fallback_model_curves()

        all_pts = np.vstack(curves)
        min_xy = all_pts.min(axis=0)
        max_xy = all_pts.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1e-6)
        scale = 0.92 / float(max(span[0], span[1]))
        center = (min_xy + max_xy) * 0.5
        normalized = [
            ((curve - center) * scale).astype(np.float32)
            for curve in curves
        ]
        return normalized
    except Exception as exc:
        print(f"[scanner] could not load edited model from {model_json}: {exc}")
        return _fallback_model_curves()


def _cell_color(row: int, col: int, palette=None) -> tuple[float, float, float, float]:
    colors = _normalize_palette(palette)
    return colors[(row * 3 + col * 5 + row * col) % len(colors)]


def _default_cell_color_set(row: int, col: int, palette=None) -> list[tuple[float, float, float, float]]:
    colors = _normalize_palette(palette)
    start = (row * 3 + col * 5 + row * col) % len(colors)
    return [colors[(start + i) % len(colors)] for i in range(4)]


def _normalize_cell_color_sets(raw_sets, rows: int, cols: int, palette=None) -> list[list[tuple[float, float, float, float]]]:
    normalized: list[list[tuple[float, float, float, float]]] = []
    flat_sets = raw_sets if isinstance(raw_sets, list) and len(raw_sets) == rows * cols else None
    for row in range(rows):
        for col in range(cols):
            raw_cell = None
            if flat_sets is not None:
                raw_cell = flat_sets[row * cols + col]
            else:
                try:
                    raw_cell = raw_sets[row][col]
                except Exception:
                    raw_cell = None
            color_set = _normalize_palette(raw_cell)
            if not raw_cell or len(color_set) < 4:
                if color_set:
                    color_set = [color_set[0] for _ in range(4)]
                else:
                    color_set = _default_cell_color_set(row, col, palette)
            normalized.append(color_set[:4])
    return normalized


def _normalize_cell_model_curves(raw_sets, rows: int, cols: int) -> list[list[np.ndarray]] | None:
    if not isinstance(raw_sets, (list, tuple)) or len(raw_sets) != rows * cols:
        return None
    normalized: list[list[np.ndarray]] = []
    for cell in raw_sets:
        curves = []
        if isinstance(cell, (list, tuple)):
            for curve in cell:
                arr = np.asarray(curve, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] >= 2:
                    curves.append(arr[:, :2].astype(np.float32))
        normalized.append(curves)
    return normalized if any(normalized) else None


def _normalize_model_curves(curves: list[np.ndarray]) -> list[np.ndarray]:
    valid = [np.asarray(curve, dtype=np.float32)[:, :2] for curve in curves if len(curve) > 1]
    if not valid:
        return _fallback_model_curves()
    pts = np.vstack(valid)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    center = (min_xy + max_xy) * 0.5
    scale = SCAN_PATTERN_FILL / float(max(span[0], span[1]))
    return [((curve - center) * scale).astype(np.float32) for curve in valid]


def _scanner_param_context(model_json: str | None):
    project_root = Path(__file__).resolve().parent
    with (project_root / "config.json").open("r") as handle:
        config = json.load(handle)
    params = np.asarray([p["initial"] for p in config["knit_parameters"]["parameters"]], dtype=np.float32)
    samples_per_loop = 5
    saved_loop_heights = None
    if model_json and Path(model_json).exists():
        try:
            with Path(model_json).open("r") as handle:
                saved = json.load(handle)
            if "params" in saved:
                if isinstance(saved["params"], dict):
                    loaded_params = params.copy()
                    for name, value in saved["params"].items():
                        index = next(
                            (
                                idx
                                for idx, param in enumerate(config["knit_parameters"]["parameters"])
                                if param["name"] == name
                            ),
                            None,
                        )
                        if index is not None:
                            loaded_params[index] = float(value)
                else:
                    loaded_params = np.asarray(saved["params"], dtype=np.float32)
                if loaded_params.shape == params.shape:
                    params = loaded_params
            if "samples_per_loop" in saved:
                samples_per_loop = int(saved.get("samples_per_loop", samples_per_loop))
            elif isinstance(saved.get("gui_state"), dict) and "samples_per_loop" in saved["gui_state"]:
                samples_per_loop = int(saved["gui_state"].get("samples_per_loop", samples_per_loop))
            raw_loop_heights = saved.get("loop_heights")
            if raw_loop_heights is None and isinstance(saved.get("gui_state"), dict):
                raw_loop_heights = saved["gui_state"].get("loop_heights")
            if raw_loop_heights is not None:
                saved_loop_heights = np.asarray(raw_loop_heights, dtype=np.float32)
        except Exception as exc:
            print(f"[scanner] could not read param context from {model_json}: {exc}")
    pidx = {p["name"]: i for i, p in enumerate(config["knit_parameters"]["parameters"])}
    lh_names = sorted(
        [p["name"] for p in config["knit_parameters"]["parameters"] if p["name"].startswith("loop_height_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    lh_idx = tuple(pidx[name] for name in lh_names)
    return params, pidx, lh_idx, samples_per_loop, saved_loop_heights


def _loop_heights_for_random_bitmap(params, bitmap, lh_idx, saved_loop_heights):
    heights = np.zeros_like(bitmap, dtype=np.float32)
    for row_idx in range(bitmap.shape[0]):
        if lh_idx:
            heights[row_idx, :] = float(params[lh_idx[min(row_idx, len(lh_idx) - 1)]])
        else:
            heights[row_idx, :] = 3.0
    saved = np.asarray(saved_loop_heights, dtype=np.float32) if saved_loop_heights is not None else np.empty((0, 0), dtype=np.float32)
    if saved.ndim == 2 and saved.size:
        keep_rows = min(bitmap.shape[0], saved.shape[0])
        keep_cols = min(bitmap.shape[1], saved.shape[1])
        heights[:keep_rows, :keep_cols] = saved[:keep_rows, :keep_cols]
    return heights * (bitmap > 0.5)


def _random_bitmap_model_curves(
    pattern_rows: int,
    pattern_cols: int,
    seed: int,
    density: float,
    repeat_rows: int = 3,
    repeat_cols: int = 3,
    model_json: str | None = None,
) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    pattern_rows = max(2, int(pattern_rows))
    pattern_cols = max(2, int(pattern_cols))
    density = float(np.clip(density, 0.05, 0.95))
    bitmap = (rng.random((pattern_rows, pattern_cols)) < density).astype(np.float32)
    if not np.any(bitmap > 0.5):
        bitmap[rng.integers(0, pattern_rows), rng.integers(0, pattern_cols)] = 1.0

    params, pidx, lh_idx, samples_per_loop, saved_loop_heights = _scanner_param_context(model_json)
    loop_heights = _loop_heights_for_random_bitmap(params, bitmap, lh_idx, saved_loop_heights)
    curves = build_parametric_control_rows(
        params,
        bitmap,
        pidx,
        lh_idx,
        samples_per_loop,
        loop_heights=loop_heights,
    )
    return _normalize_model_curves(curves)


def _generate_random_cell_model_curves(args: argparse.Namespace) -> list[list[np.ndarray]]:
    rows = max(1, int(args.rows))
    cols = max(1, int(args.cols))
    seed = int(getattr(args, "random_seed", 1))
    pattern_rows = int(getattr(args, "pattern_rows", 4))
    pattern_cols = int(getattr(args, "pattern_cols", 5))
    repeat_rows = int(getattr(args, "pattern_repeat_rows", 3))
    repeat_cols = int(getattr(args, "pattern_repeat_cols", 3))
    density = float(getattr(args, "pattern_density", 0.62))
    return [
        _random_bitmap_model_curves(
            pattern_rows,
            pattern_cols,
            seed + cell * 9973,
            density,
            repeat_rows,
            repeat_cols,
            getattr(args, "model_json", ""),
        )
        for cell in range(rows * cols)
    ]


def build_fabric_grid_stations(
    width: float,
    length: float,
    rows: int,
    cols: int,
    edge_margin: float,
    square_margin: float,
    surface_wave: float,
    palette=None,
    cell_color_sets=None,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[float, float, float, float]], list[list[tuple[float, float, float, float]]]]:
    if min(width, length) <= 0:
        raise ValueError("fabric width and length must be positive")
    if min(rows, cols) <= 0:
        raise ValueError("grid counts must be positive")

    cell_w = width / cols
    cell_l = length / rows
    stations: list[list[float]] = []
    cells: list[tuple[int, int]] = []
    colors: list[tuple[float, float, float, float]] = []
    all_cell_color_sets = _normalize_cell_color_sets(cell_color_sets, rows, cols, palette)

    for row, col in _serpentine_indices(rows, cols):
        x0 = -width / 2 + col * cell_w
        y0 = -length / 2 + row * cell_l
        x = x0 + cell_w * 0.5
        y = y0 + cell_l * 0.5
        z = surface_wave * math.sin(2 * math.pi * (x / width + 0.5))
        z += 0.5 * surface_wave * math.cos(2 * math.pi * (y / length + 0.5))
        stations.append([float(x), float(y), float(z)])
        cells.append((row, col))
        colors.append(all_cell_color_sets[row * cols + col][0])

    return np.asarray(stations, dtype=float), cells, colors, all_cell_color_sets


def expand_stations_to_angle_viewpoints(
    stations: np.ndarray,
    number_of_angles: int,
    view_radius: float,
    angle_lift: float,
) -> tuple[np.ndarray, list[int], list[str], np.ndarray]:
    """Create one fixed scanner position per square with multiple camera rotations."""
    number_of_angles = max(1, int(number_of_angles))
    points: list[np.ndarray] = []
    station_ids: list[int] = []
    view_names: list[str] = []
    rotvecs: list[np.ndarray] = []
    tilt = min(0.60, max(0.0, float(view_radius) / 0.060 * 0.45))

    for station_id, station in enumerate(stations):
        for angle_index in range(number_of_angles):
            theta = 2.0 * math.pi * angle_index / number_of_angles
            angle_deg = 360.0 * angle_index / number_of_angles
            offset = np.array([0.0, 0.0, angle_lift], dtype=float)
            rotvec = np.array(
                [
                    math.pi,
                    tilt * math.sin(theta),
                    tilt * math.cos(theta),
                ],
                dtype=float,
            )
            points.append(station + offset)
            station_ids.append(station_id)
            view_names.append(f"angle {angle_deg:.0f}")
            rotvecs.append(rotvec)
    return np.asarray(points), station_ids, view_names, np.asarray(rotvecs)


def resolve_fabric_grid(args: argparse.Namespace) -> None:
    args.rows = max(1, int(args.rows))
    args.cols = max(1, int(args.cols))
    args.square_width = args.width / max(1, args.cols)
    args.square_length = args.length / max(1, args.rows)


def build_plan(args: argparse.Namespace) -> FabricPlan:
    resolve_fabric_grid(args)
    model_curves = getattr(args, "model_curves", None)
    if model_curves is None:
        model_curves = _load_saved_model_curves(getattr(args, "model_json", ""))
    else:
        model_curves = [np.asarray(curve, dtype=np.float32) for curve in model_curves if len(curve) > 1]
        if not model_curves:
            model_curves = _load_saved_model_curves(getattr(args, "model_json", ""))
    raw_stations, station_cells, station_colors, cell_color_sets = build_fabric_grid_stations(
        width=args.width,
        length=args.length,
        rows=args.rows,
        cols=args.cols,
        edge_margin=args.edge_margin,
        square_margin=args.square_margin,
        surface_wave=args.surface_wave,
        palette=getattr(args, "palette", None),
        cell_color_sets=getattr(args, "cell_color_sets", None),
    )
    cell_model_curves = _normalize_cell_model_curves(
        getattr(args, "cell_model_curves", None),
        args.rows,
        args.cols,
    )
    if cell_model_curves is None and bool(getattr(args, "random_patterns", False)):
        cell_model_curves = _generate_random_cell_model_curves(args)
        if cell_model_curves:
            model_curves = cell_model_curves[0]
    raw_points, station_ids, view_names, rotvecs = expand_stations_to_angle_viewpoints(
        raw_stations,
        number_of_angles=args.number_of_angles,
        view_radius=args.view_radius,
        angle_lift=args.angle_lift,
    )

    lift = np.array([0.0, 0.0, args.approach_lift])
    raw_points = np.vstack([raw_points[0] + lift, raw_points, raw_points[-1] + lift])
    station_ids = [station_ids[0], *station_ids, station_ids[-1]]
    view_names = ["approach", *view_names, "retreat"]
    rotvecs = np.vstack([TOOL_DOWN_ROTVEC, rotvecs, TOOL_DOWN_ROTVEC])

    center = np.asarray(args.center, dtype=float)
    max_span = np.asarray(args.max_span, dtype=float)
    mapped_stations = fit_path_to_workspace(raw_stations, center=center, max_span=max_span)
    mapped_points = fit_path_to_workspace(raw_points, center=center, max_span=max_span)
    poses = points_and_rotvecs_to_poses(mapped_points, rotvecs)

    fabric_xy = mapped_stations[:, :2]
    fabric_origin = np.array([fabric_xy[:, 0].min(), fabric_xy[:, 1].min(), mapped_stations[:, 2].min()])
    fabric_size = np.array([np.ptp(fabric_xy[:, 0]), np.ptp(fabric_xy[:, 1]), 0.0])

    return FabricPlan(
        raw_stations=raw_stations,
        mapped_stations=mapped_stations,
        raw_points=raw_points,
        mapped_points=mapped_points,
        poses=poses,
        station_ids=station_ids,
        view_names=view_names,
        station_cells=station_cells,
        cell_colors=station_colors,
        cell_color_sets=cell_color_sets,
        fabric_origin=fabric_origin,
        fabric_size=fabric_size,
        grid_rows=args.rows,
        grid_cols=args.cols,
        square_width=args.square_width,
        square_length=args.square_length,
        model_curves=model_curves,
        cell_model_curves=cell_model_curves,
        pattern_repeat_rows=max(1, int(getattr(args, "pattern_repeat_rows", 1))),
        pattern_repeat_cols=max(1, int(getattr(args, "pattern_repeat_cols", 1))),
    )


def densify_plan_for_robot(plan: FabricPlan, max_step: float = ROBOT_MAX_CARTESIAN_STEP) -> FabricPlan:
    if max_step <= 0 or len(plan.poses) < 2:
        return plan

    poses: list[np.ndarray] = [plan.poses[0].copy()]
    station_ids: list[int] = [plan.station_ids[0]]
    view_names: list[str] = [plan.view_names[0]]

    for index in range(1, len(plan.poses)):
        start = plan.poses[index - 1]
        end = plan.poses[index]
        distance = float(np.linalg.norm(end[:3] - start[:3]))
        steps = max(1, int(math.ceil(distance / max_step)))

        for step in range(1, steps + 1):
            t = step / steps
            pose = (1.0 - t) * start + t * end
            poses.append(pose)
            station_ids.append(plan.station_ids[index])
            view_names.append(plan.view_names[index] if step == steps else "travel")

    dense_poses = np.asarray(poses, dtype=float)
    return replace(
        plan,
        mapped_points=dense_poses[:, :3].copy(),
        poses=dense_poses,
        station_ids=station_ids,
        view_names=view_names,
    )


def is_scan_view(view_name: str) -> bool:
    return view_name not in NON_SCAN_VIEW_NAMES


def should_save_scan_image(
    args: argparse.Namespace,
    plan: FabricPlan,
    target_index: int,
    saved_targets: set[int],
    saved_stations: set[int],
    save_images: bool | None = None,
) -> bool:
    if save_images is None:
        save_images = bool(args.save_images)
    if not save_images or target_index in saved_targets:
        return False
    view_name = plan.view_names[target_index]
    if not is_scan_view(view_name):
        return False
    station = plan.station_ids[target_index]
    if args.image_every == CAMERA_SAVE_EVERY_STATION:
        return station not in saved_stations
    return True


def rot_from_wxyz(q: np.ndarray) -> Rotation:
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def load_ur5e_model_data():
    import mujoco
    from robot_descriptions import ur5e_mj_description

    model = mujoco.MjModel.from_xml_path(ur5e_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id < 0:
        raise RuntimeError("UR5e MJCF does not contain site 'attachment_site'")

    # Keep the simulated UR5 base frame rotated exactly 180 degrees around Z,
    # then start joint 1 at zero. This rotates the robot placement itself
    # instead of twisting the shoulder joint to fake the direction.
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_id >= 0:
        model.body_quat[base_id] = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    data.qpos[:6] = [0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0]
    mujoco.mj_forward(model, data)
    return mujoco, model, data, site_id


def load_ur5e_scene():
    import mujoco.viewer

    mujoco, model, data, site_id = load_ur5e_model_data()
    handle = mujoco.viewer.launch_passive(model, data)
    return mujoco, model, data, site_id, handle


def get_tcp(mujoco, model, data, site_id) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, data.site_xmat[site_id])
    rotvec = rot_from_wxyz(q).as_rotvec()
    return np.r_[data.site_xpos[site_id].copy(), rotvec]


def step_ik(mujoco, model, data, site_id, target_pose: np.ndarray) -> None:
    cur = get_tcp(mujoco, model, data, site_id)
    err_pos = target_pose[:3] - cur[:3]
    err_rot = (Rotation.from_rotvec(target_pose[3:6]) * Rotation.from_rotvec(cur[3:6]).inv()).as_rotvec()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jac = np.vstack([jacp[:, :6], jacr[:, :6]])
    err = MUJOCO_KP * np.r_[err_pos, 0.35 * err_rot]
    jjt = jac @ jac.T
    dq = jac.T @ np.linalg.solve(jjt + (MUJOCO_DAMPING**2) * np.eye(jjt.shape[0]), err)
    data.qpos[:6] = np.clip(data.qpos[:6] + dq, model.jnt_range[:6, 0], model.jnt_range[:6, 1])
    mujoco.mj_forward(model, data)


def pose_errors(current_pose: np.ndarray, target_pose: np.ndarray) -> tuple[float, float]:
    pos_err = float(np.linalg.norm(current_pose[:3] - target_pose[:3]))
    rot_err = (
        Rotation.from_rotvec(target_pose[3:6])
        * Rotation.from_rotvec(current_pose[3:6]).inv()
    ).magnitude()
    return pos_err, float(rot_err)


def _add_sphere(mujoco, scn, pos, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(pos, dtype=float),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def _add_box(mujoco, scn, pos, size, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_BOX,
        np.asarray(size, dtype=float),
        np.asarray(pos, dtype=float),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def _cell_color_set(plan: FabricPlan, row: int, col: int) -> list[tuple[float, float, float, float]]:
    idx = row * plan.grid_cols + col
    if 0 <= idx < len(plan.cell_color_sets):
        return plan.cell_color_sets[idx]
    return _default_cell_color_set(row, col)


def _cell_model_curves(plan: FabricPlan, row: int, col: int) -> list[np.ndarray]:
    idx = row * plan.grid_cols + col
    if plan.cell_model_curves is not None and 0 <= idx < len(plan.cell_model_curves):
        curves = plan.cell_model_curves[idx]
        if curves:
            return curves
    return plan.model_curves


def _smooth_curve_2d(curve: np.ndarray, samples_per_segment: int = 4, max_points: int = 36) -> np.ndarray:
    pts = np.asarray(curve, dtype=np.float32)[:, :2]
    if pts.shape[0] < 3:
        return pts
    samples_per_segment = max(1, int(samples_per_segment))
    padded = np.vstack([pts[0], pts, pts[-1]])
    smooth = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for step in range(samples_per_segment):
            t = step / float(samples_per_segment)
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            smooth.append(point)
    smooth.append(pts[-1])
    out = np.asarray(smooth, dtype=np.float32)
    if out.shape[0] > max_points:
        keep = np.linspace(0, out.shape[0] - 1, max_points).astype(int)
        out = out[keep]
    return out


def _smoothed_cell_model_curves(plan: FabricPlan, row: int, col: int) -> list[np.ndarray]:
    cache = getattr(plan, "_smoothed_curve_cache", None)
    if cache is None:
        cache = {}
        setattr(plan, "_smoothed_curve_cache", cache)
    key = (int(row), int(col))
    if key not in cache:
        cache[key] = [_smooth_curve_2d(curve) for curve in _cell_model_curves(plan, row, col)]
    return cache[key]


def _knit_cell_curves() -> list[list[tuple[float, float]]]:
    return [
        [(-0.48, 0.16), (-0.33, 0.38), (-0.08, 0.47), (0.18, 0.43), (0.40, 0.24), (0.30, 0.08), (0.06, -0.02), (-0.20, 0.02), (-0.40, 0.14)],
        [(-0.48, -0.02), (-0.26, 0.06), (-0.05, 0.22), (0.18, 0.08), (0.46, -0.02), (0.22, -0.16), (0.02, -0.31), (-0.20, -0.16), (-0.42, -0.03)],
        [(-0.50, -0.18), (-0.24, -0.14), (0.00, -0.10), (0.24, -0.14), (0.50, -0.18)],
        [(-0.42, -0.30), (-0.25, -0.45), (0.02, -0.48), (0.28, -0.42), (0.44, -0.25), (0.27, -0.09), (0.02, 0.00), (-0.22, -0.10), (-0.44, -0.26)],
    ]


def _add_cell_fabric_model(mujoco, scn, center_x: float, center_y: float, z: float, cell_w: float, cell_l: float, colors, highlight=False) -> None:
    _add_box(
        mujoco,
        scn,
        [center_x, center_y, z - FABRIC_THICKNESS * 0.25],
        [cell_w * 0.50, cell_l * 0.50, FABRIC_THICKNESS * 0.30],
        (0.025, 0.030, 0.038, 1.0),
    )
    yarn_width = max(min(cell_w, cell_l) * 0.030, 0.0015)
    z_step = max(FABRIC_THICKNESS * 0.35, yarn_width * 0.45)
    for idx, curve in enumerate(_knit_cell_curves()):
        color = colors[idx % len(colors)]
        rgba = (color[0], color[1], color[2], 1.0)
        pts = [
            np.array([
                center_x + px * cell_w,
                center_y + py * cell_l,
                z + FABRIC_THICKNESS * 0.65 + idx * z_step,
            ])
            for px, py in curve
        ]
        for a, b in zip(pts, pts[1:]):
            _add_segment(mujoco, scn, a, b, rgba, yarn_width)
    if highlight:
        _add_box(
            mujoco,
            scn,
            [center_x, center_y, z + FABRIC_THICKNESS * 0.20],
            [cell_w * 0.50, cell_l * 0.50, FABRIC_THICKNESS * 0.15],
            (0.1, 1.0, 0.3, 0.22),
        )


def _add_model_cell_fabric(mujoco, scn, plan: FabricPlan, row: int, col: int, z: float, highlight=False, simplified=False) -> None:
    cell_w = plan.fabric_size[0] / plan.grid_cols
    cell_l = plan.fabric_size[1] / plan.grid_rows
    center_x = plan.fabric_origin[0] + cell_w * (col + 0.5)
    center_y = plan.fabric_origin[1] + cell_l * (row + 0.5)
    colors = _cell_color_set(plan, row, col)
    _add_box(
        mujoco,
        scn,
        [center_x, center_y, z - FABRIC_THICKNESS * 0.25],
        [cell_w * 0.50, cell_l * 0.50, FABRIC_THICKNESS * 0.25],
        (0.018, 0.022, 0.030, 1.0),
    )
    repeat_rows = max(1, int(plan.pattern_repeat_rows))
    repeat_cols = max(1, int(plan.pattern_repeat_cols))
    tile_w = cell_w / repeat_cols
    tile_l = cell_l / repeat_rows
    yarn_width = max(min(tile_w, tile_l) * 0.050, 0.00075)
    z_step = max(FABRIC_THICKNESS * 0.25, yarn_width * 0.35)
    curves = _smoothed_cell_model_curves(plan, row, col)
    image_repeat_proxy = repeat_rows > 1 or repeat_cols > 1
    x0 = center_x - cell_w * 0.5
    y0 = center_y - cell_l * 0.5
    for tr in range(repeat_rows):
        for tc in range(repeat_cols):
            tile_cx = x0 + (tc + 0.5) * tile_w
            tile_cy = y0 + (tr + 0.5) * tile_l
            for curve_idx, curve in enumerate(curves):
                color = colors[curve_idx % len(colors)]
                rgba = (color[0], color[1], color[2], 1.0)
                pts = [
                    np.array([
                        tile_cx + float(p[0]) * tile_w * IMAGE_REPEAT_OVERLAP,
                        tile_cy + float(p[1]) * tile_l * IMAGE_REPEAT_OVERLAP,
                        z + FABRIC_THICKNESS * 0.65 + (0 if image_repeat_proxy else (curve_idx % len(colors))) * z_step,
                    ])
                    for p in curve
                ]
                for a, b in zip(pts, pts[1:]):
                    _add_segment(mujoco, scn, a, b, rgba, yarn_width)
    if highlight:
        _add_box(
            mujoco,
            scn,
            [center_x, center_y, z + FABRIC_THICKNESS * 0.12],
            [cell_w * 0.50, cell_l * 0.50, FABRIC_THICKNESS * 0.12],
            (0.1, 1.0, 0.3, 0.20),
        )


def _add_color_picker_cell_fabric(mujoco, scn, plan: FabricPlan, row: int, col: int, z: float, highlight=False) -> None:
    cell_w = plan.fabric_size[0] / plan.grid_cols
    cell_l = plan.fabric_size[1] / plan.grid_rows
    center_x = plan.fabric_origin[0] + cell_w * (col + 0.5)
    center_y = plan.fabric_origin[1] + cell_l * (row + 0.5)
    color = _cell_color_set(plan, row, col)[0]
    rgba = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    _add_box(
        mujoco,
        scn,
        [center_x, center_y, z],
        [cell_w * 0.492, cell_l * 0.492, FABRIC_THICKNESS * 0.20],
        rgba,
    )
    border = (0.92, 0.94, 0.98, 0.75)
    corners = [
        np.array([center_x - cell_w * 0.50, center_y - cell_l * 0.50, z + FABRIC_THICKNESS * 0.18]),
        np.array([center_x + cell_w * 0.50, center_y - cell_l * 0.50, z + FABRIC_THICKNESS * 0.18]),
        np.array([center_x + cell_w * 0.50, center_y + cell_l * 0.50, z + FABRIC_THICKNESS * 0.18]),
        np.array([center_x - cell_w * 0.50, center_y + cell_l * 0.50, z + FABRIC_THICKNESS * 0.18]),
    ]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        _add_segment(mujoco, scn, a, b, border, max(min(cell_w, cell_l) * 0.006, 0.0008))
    if highlight:
        _add_box(
            mujoco,
            scn,
            [center_x, center_y, z + FABRIC_THICKNESS * 0.24],
            [cell_w * 0.50, cell_l * 0.50, FABRIC_THICKNESS * 0.10],
            (0.1, 1.0, 0.3, 0.24),
        )


def _add_rendered_fabric_proxy(mujoco, scn, plan: FabricPlan, z: float) -> bool:
    texture = _rendered_texture_for_camera(plan, 0, 0, focused=False)
    if texture is None:
        return False
    tex_w, tex_h = texture.size
    if tex_w <= 1 or tex_h <= 1:
        return False
    sample_cols = min(72, max(18, int(plan.grid_cols * 14)))
    sample_rows = min(54, max(14, int(plan.grid_rows * 12)))
    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)
    sampled = np.asarray(texture.resize((sample_cols, sample_rows), resample), dtype=np.float32) / 255.0
    cell_w = plan.fabric_size[0] / sample_cols
    cell_l = plan.fabric_size[1] / sample_rows
    origin_x = plan.fabric_origin[0]
    origin_y = plan.fabric_origin[1]
    for r in range(sample_rows):
        for c in range(sample_cols):
            color = sampled[r, c]
            if float(color.max()) < 0.035:
                continue
            x = origin_x + (c + 0.5) * cell_w
            y = origin_y + (sample_rows - r - 0.5) * cell_l
            _add_box(
                mujoco,
                scn,
                [x, y, z + FABRIC_THICKNESS * 0.18],
                [cell_w * 0.52, cell_l * 0.52, FABRIC_THICKNESS * 0.07],
                (float(color[0]), float(color[1]), float(color[2]), 1.0),
            )
    return True


def _add_segment(mujoco, scn, a, b, rgba, width=PATH_WIDTH):
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a), np.asarray(b))
    scn.ngeom += 1


def _add_camera_marker(mujoco, scn, tcp_pos: np.ndarray, look_at: np.ndarray) -> None:
    _add_box(mujoco, scn, tcp_pos, CAMERA_MARKER_SIZE, (0.02, 0.02, 0.02, 1.0))
    direction = np.asarray(look_at, dtype=float) - np.asarray(tcp_pos, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm > 1e-9:
        tip = np.asarray(tcp_pos, dtype=float) + direction / norm * 0.055
        _add_segment(mujoco, scn, tcp_pos, tip, (0.05, 0.55, 1.0, 0.95), 0.004)


def _add_pose_axes(mujoco, scn, pose: np.ndarray, scale: float = POSE_AXIS_LENGTH, alpha: float = 0.85) -> None:
    origin = np.asarray(pose[:3], dtype=float)
    axes = Rotation.from_rotvec(pose[3:6]).as_matrix()
    colors = (
        (1.0, 0.15, 0.15, alpha),
        (0.10, 0.85, 0.20, alpha),
        (0.15, 0.45, 1.0, alpha),
    )
    for axis_index, color in enumerate(colors):
        _add_segment(mujoco, scn, origin, origin + axes[:, axis_index] * scale, color, 0.0024)


def _add_rotation_ring(mujoco, scn, center: np.ndarray, radius: float, rgba) -> None:
    points = []
    z = float(center[2]) + 0.010
    for index in range(25):
        theta = 2.0 * math.pi * index / 24
        points.append(np.array([center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta), z]))
    for a, b in zip(points, points[1:]):
        _add_segment(mujoco, scn, a, b, rgba, 0.0015)


def _perspective_coeffs(dst_points, src_points):
    matrix = []
    vector = []
    for (x, y), (u, v) in zip(dst_points, src_points):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        vector.extend([u, v])
    return np.linalg.solve(np.asarray(matrix, dtype=float), np.asarray(vector, dtype=float)).tolist()


def _paste_projected_texture(base: Image.Image, texture: Image.Image, projected_quad) -> bool:
    if any(point is None for point in projected_quad):
        return False
    dst = [(float(point[0]), float(point[1])) for point in projected_quad]
    src = texture.convert("RGB")
    src_w, src_h = src.size
    if src_w <= 1 or src_h <= 1:
        return False
    src_quad = [(0.0, 0.0), (float(src_w), 0.0), (float(src_w), float(src_h)), (0.0, float(src_h))]
    try:
        coeffs = _perspective_coeffs(dst, src_quad)
    except np.linalg.LinAlgError:
        return False
    resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)
    warped = src.transform(base.size, Image.Transform.PERSPECTIVE, coeffs, resample=resample)
    mask_src = Image.new("L", (src_w, src_h), 255)
    mask = mask_src.transform(base.size, Image.Transform.PERSPECTIVE, coeffs, resample=resample)
    base.paste(warped, (0, 0), mask)
    return True


def _trim_rendered_texture(texture: Image.Image) -> Image.Image:
    texture = texture.convert("RGB")
    arr = np.asarray(texture, dtype=np.uint8)
    if arr.size == 0:
        return texture
    edge = np.concatenate([
        arr[:4, :, :].reshape(-1, 3),
        arr[-4:, :, :].reshape(-1, 3),
        arr[:, :4, :].reshape(-1, 3),
        arr[:, -4:, :].reshape(-1, 3),
    ])
    bg = np.median(edge, axis=0)
    diff = np.linalg.norm(arr.astype(np.float32) - bg.astype(np.float32), axis=2)
    mask = (diff > 14.0) | (arr.max(axis=2) > 55)
    ys, xs = np.where(mask)
    if xs.size < 16 or ys.size < 16:
        return texture
    pad = 2
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(texture.size[0], int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(texture.size[1], int(ys.max()) + pad + 1)
    return texture.crop((x0, y0, x1, y1))


def _tile_rendered_texture(texture: Image.Image, repeat_rows: int, repeat_cols: int) -> Image.Image:
    repeat_rows = max(1, int(repeat_rows))
    repeat_cols = max(1, int(repeat_cols))
    tile = _trim_rendered_texture(texture)
    tile_w, tile_h = tile.size
    if tile_w <= 1 or tile_h <= 1:
        return tile
    max_dim = 2048
    scale = min(1.0, max_dim / max(float(tile_w * repeat_cols), 1.0), max_dim / max(float(tile_h * repeat_rows), 1.0))
    if scale < 1.0:
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
        tile = tile.resize((max(1, int(tile_w * scale)), max(1, int(tile_h * scale))), resample)
        tile_w, tile_h = tile.size
    canvas = Image.new("RGB", (tile_w * repeat_cols, tile_h * repeat_rows), (18, 22, 28))
    for r in range(repeat_rows):
        for c in range(repeat_cols):
            canvas.paste(tile, (c * tile_w, r * tile_h))
    return canvas


def _rendered_texture_for_camera(plan: FabricPlan, row: int, col: int, focused: bool) -> Image.Image | None:
    texture = getattr(plan, "rendered_fabric_image", None)
    if texture is None:
        return None
    texture = texture.convert("RGB")
    repeat_rows = max(1, int(plan.pattern_repeat_rows))
    repeat_cols = max(1, int(plan.pattern_repeat_cols))
    if not focused:
        cache_key = ("full", repeat_rows, repeat_cols)
        cache = getattr(plan, "_rendered_texture_cache", None)
        if cache is None:
            cache = {}
            setattr(plan, "_rendered_texture_cache", cache)
        if cache_key not in cache:
            cache[cache_key] = _tile_rendered_texture(texture, repeat_rows, repeat_cols)
        return cache[cache_key]
    rows = max(1, int(plan.grid_rows))
    cols = max(1, int(plan.grid_cols))
    w, h = texture.size
    x0 = int(np.clip(round(col * w / cols), 0, w - 1))
    x1 = int(np.clip(round((col + 1) * w / cols), x0 + 1, w))
    y0 = int(np.clip(round(row * h / rows), 0, h - 1))
    y1 = int(np.clip(round((row + 1) * h / rows), y0 + 1, h))
    cache_key = ("cell", int(row), int(col), repeat_rows, repeat_cols)
    cache = getattr(plan, "_rendered_texture_cache", None)
    if cache is None:
        cache = {}
        setattr(plan, "_rendered_texture_cache", cache)
    if cache_key not in cache:
        cache[cache_key] = _tile_rendered_texture(texture.crop((x0, y0, x1, y1)), repeat_rows, repeat_cols)
    return cache[cache_key]


def draw_scene(
    mujoco,
    handle,
    plan: FabricPlan,
    target_index: int,
    executed: list[np.ndarray],
    camera_enabled: bool = False,
    current_pose: np.ndarray | None = None,
    target_pose: np.ndarray | None = None,
    clear_scene: bool = True,
    simplified: bool = False,
    color_picker_mode: bool = False,
) -> None:
    scn = handle.user_scn
    if clear_scene:
        scn.ngeom = 0

    cell_w = plan.fabric_size[0] / plan.grid_cols
    cell_l = plan.fabric_size[1] / plan.grid_rows
    z = plan.fabric_origin[2] - FABRIC_THICKNESS * 0.5
    fabric_center = plan.fabric_origin + np.array([plan.fabric_size[0] * 0.5, plan.fabric_size[1] * 0.5, 0.0])
    _add_box(
        mujoco,
        scn,
        [fabric_center[0] - 0.12, fabric_center[1], -0.018],
        [0.62, 0.42, 0.012],
        (0.10, 0.105, 0.115, 1.0),
    )
    _add_box(
        mujoco,
        scn,
        [fabric_center[0], fabric_center[1], z - FABRIC_THICKNESS * 0.55],
        [plan.fabric_size[0] * 0.54, plan.fabric_size[1] * 0.54, FABRIC_THICKNESS * 0.16],
        (0.030, 0.036, 0.046, 1.0),
    )
    station_id = plan.station_ids[min(target_index, len(plan.station_ids) - 1)]
    active_cell = plan.station_cells[station_id]
    rendered_proxy = False if color_picker_mode else _add_rendered_fabric_proxy(mujoco, scn, plan, z)
    for row in range(plan.grid_rows):
        for col in range(plan.grid_cols):
            if color_picker_mode:
                _add_color_picker_cell_fabric(mujoco, scn, plan, row, col, z, highlight=(row, col) == active_cell)
            elif not rendered_proxy:
                _add_model_cell_fabric(mujoco, scn, plan, row, col, z, highlight=(row, col) == active_cell, simplified=simplified)
            elif (row, col) == active_cell:
                cell_w = plan.fabric_size[0] / plan.grid_cols
                cell_l = plan.fabric_size[1] / plan.grid_rows
                center_x = plan.fabric_origin[0] + cell_w * (col + 0.5)
                center_y = plan.fabric_origin[1] + cell_l * (row + 0.5)
                _add_box(
                    mujoco,
                    scn,
                    [center_x, center_y, z + FABRIC_THICKNESS * 0.28],
                    [cell_w * 0.50, cell_l * 0.50, FABRIC_THICKNESS * 0.08],
                    (0.1, 1.0, 0.3, 0.18),
                )

    path_step = max(1, int(math.ceil(len(plan.mapped_points) / 80))) if simplified else 1
    for i in range(0, len(plan.mapped_points) - 1, path_step):
        j = min(i + path_step, len(plan.mapped_points) - 1)
        _add_segment(mujoco, scn, plan.mapped_points[i], plan.mapped_points[j], (0.20, 0.24, 0.30, 0.28), 0.0018)

    if target_index > 0:
        progress_step = max(1, int(math.ceil(target_index / 80))) if simplified else 1
        for i in range(0, min(target_index, len(plan.mapped_points) - 1), progress_step):
            j = min(i + progress_step, len(plan.mapped_points) - 1)
            _add_segment(mujoco, scn, plan.mapped_points[i], plan.mapped_points[j], (0.15, 0.85, 0.35, 0.55), 0.0026)

    trail_limit = min(MAX_TRAIL_POINTS, 60 if simplified else MAX_TRAIL_POINTS)
    trail = executed[-trail_limit:]
    if simplified and len(trail) > 40:
        trail = trail[::max(1, int(math.ceil(len(trail) / 40)))]
    for a, b in zip(trail, trail[1:]):
        _add_segment(mujoco, scn, a, b, (1.00, 0.42, 0.08, 0.90), 0.0032)

    target = plan.mapped_points[min(target_index, len(plan.mapped_points) - 1)]
    station = plan.mapped_stations[station_id]
    color = plan.cell_colors[station_id]
    _add_sphere(mujoco, scn, station, MARKER_RADIUS * 1.15, color)
    _add_sphere(mujoco, scn, target, MARKER_RADIUS * 1.45, (0.1, 1.0, 0.3, 1.0))
    _add_rotation_ring(mujoco, scn, station, min(plan.square_width, plan.square_length) * 0.22, (0.1, 1.0, 0.3, 0.65))
    if target_pose is not None:
        _add_pose_axes(mujoco, scn, target_pose, scale=POSE_AXIS_LENGTH * 1.15, alpha=0.55)
    if current_pose is not None:
        _add_pose_axes(mujoco, scn, current_pose, scale=POSE_AXIS_LENGTH, alpha=0.95)
    if trail:
        _add_sphere(mujoco, scn, trail[-1], MARKER_RADIUS * 1.35, (1.0, 0.42, 0.08, 1.0))
        if camera_enabled:
            _add_camera_marker(mujoco, scn, trail[-1], station)

    handle.sync()


def render_camera_image(
    plan: FabricPlan,
    tcp_pos: np.ndarray,
    target_index: int,
    station_id: int,
    view_name: str,
    target_pose: np.ndarray | None = None,
    color_picker_mode: bool = False,
    capture_mode: str = CAMERA_CAPTURE_NATURAL,
    camera_zoom: float = 1.0,
) -> Image.Image:
    width_px, height_px = CAMERA_IMAGE_SIZE
    station = plan.mapped_stations[station_id]
    cell_w = plan.fabric_size[0] / plan.grid_cols
    cell_l = plan.fabric_size[1] / plan.grid_rows
    grid_origin_x = float(plan.mapped_stations[:, 0].min() - cell_w * 0.5)
    grid_origin_y = float(plan.mapped_stations[:, 1].min() - cell_l * 0.5)
    if target_pose is None:
        target_pose = plan.poses[min(target_index, len(plan.poses) - 1)]
    target_pose = np.asarray(target_pose, dtype=float)

    img = Image.new("RGB", (width_px, height_px), (19, 22, 28))

    from PIL import ImageDraw, ImageFilter

    rotation = Rotation.from_rotvec(target_pose[3:6])
    forward = rotation.apply([0.0, 0.0, 1.0])
    forward = forward / max(float(np.linalg.norm(forward)), 1e-8)
    if forward[2] > -0.15:
        forward = station - target_pose[:3]
        forward = forward / max(float(np.linalg.norm(forward)), 1e-8)
    right = rotation.apply([1.0, 0.0, 0.0])
    right = right - forward * float(np.dot(right, forward))
    right = right / max(float(np.linalg.norm(right)), 1e-8)
    up = np.cross(right, forward)
    up = up / max(float(np.linalg.norm(up)), 1e-8)

    active_row, active_col = plan.station_cells[station_id]
    lens_pos = np.asarray(tcp_pos[:3], dtype=float)
    # The tool site is the gripper TCP, while the simulated camera lens sits
    # slightly behind it on the gripper body. This offset gives a real scanner
    # view of the fabric instead of an unusably tiny contact-distance crop.
    focused_capture = str(capture_mode) == CAMERA_CAPTURE_FOCUSED
    rendered_cell_center = np.array(
        [
            grid_origin_x + (active_col + 0.5) * cell_w,
            grid_origin_y + (active_row + 0.5) * cell_l,
            station[2],
        ],
        dtype=float,
    )
    focus = rendered_cell_center + np.array([0.0, 0.0, FABRIC_THICKNESS * 0.15], dtype=float)
    if focused_capture:
        # Focused mode is a batch inspection capture. Use a stable camera
        # centered on the selected batch; the selected scan angle becomes roll
        # around the batch normal. This keeps every station framed consistently.
        angle_deg = 0.0
        if "angle" in str(view_name).lower():
            try:
                angle_deg = float(str(view_name).lower().split("angle", 1)[1].strip().split()[0])
            except (IndexError, ValueError):
                angle_deg = 0.0
        theta = math.radians(angle_deg)
        forward = np.array([0.0, 0.0, -1.0], dtype=float)
        right = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)
        up = np.array([-math.sin(theta), math.cos(theta), 0.0], dtype=float)
        camera_standoff = max(0.105, max(cell_w, cell_l) * 1.35)
        lens_pos = focus - forward * camera_standoff
    else:
        camera_standoff = max(0.120, max(cell_w, cell_l) * 1.55, float(np.linalg.norm(lens_pos - station)) * 1.75)
        lens_pos = lens_pos - forward * camera_standoff
    # Aim the saved camera at the active station while preserving the roll/right
    # axis from the selected scan angle.
    forward = focus - lens_pos
    forward = forward / max(float(np.linalg.norm(forward)), 1e-8)
    right = right - forward * float(np.dot(right, forward))
    right = right / max(float(np.linalg.norm(right)), 1e-8)
    up = np.cross(right, forward)
    up = up / max(float(np.linalg.norm(up)), 1e-8)

    camera_zoom = max(0.20, float(camera_zoom))
    if focused_capture:
        batch_span = math.hypot(cell_w, cell_l)
        fov_y = 2.0 * math.atan((batch_span * 0.58) / max(camera_standoff, 1e-6))
        fov_y = float(np.clip(fov_y, math.radians(36.0), math.radians(68.0)))
    else:
        fov_y = math.radians(74.0)
    fov_y = float(np.clip(fov_y / camera_zoom, math.radians(12.0), math.radians(86.0)))
    focal = (height_px * 0.5) / math.tan(fov_y * 0.5)
    near = 0.004

    def project_camera(point: np.ndarray) -> tuple[int, int, float] | None:
        rel = np.asarray(point, dtype=float) - lens_pos
        depth = float(np.dot(rel, forward))
        if depth <= near:
            return None
        camera_x = float(np.dot(rel, right))
        camera_y = float(np.dot(rel, up))
        x = int(width_px * 0.5 + focal * camera_x / depth)
        y = int(height_px * 0.5 - focal * camera_y / depth)
        return x, y, depth

    def visible_line(points: list[np.ndarray]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for point in points:
            projected = project_camera(point)
            if projected is not None:
                x, y, _ = projected
                out.append((x, y))
        return out

    draw = ImageDraw.Draw(img)
    fabric_z = plan.fabric_origin[2] - FABRIC_THICKNESS * 0.48
    rendered_texture = None if color_picker_mode else _rendered_texture_for_camera(
        plan,
        active_row,
        active_col,
        focused_capture,
    )
    if rendered_texture is not None:
        if focused_capture:
            tex_x0 = grid_origin_x + active_col * cell_w
            tex_y0 = grid_origin_y + active_row * cell_l
            tex_x1 = tex_x0 + cell_w
            tex_y1 = tex_y0 + cell_l
        else:
            tex_x0 = grid_origin_x
            tex_y0 = grid_origin_y
            tex_x1 = grid_origin_x + plan.grid_cols * cell_w
            tex_y1 = grid_origin_y + plan.grid_rows * cell_l
        tex_quad = [
            project_camera(np.array([tex_x0, tex_y0, fabric_z + 0.0003])),
            project_camera(np.array([tex_x1, tex_y0, fabric_z + 0.0003])),
            project_camera(np.array([tex_x1, tex_y1, fabric_z + 0.0003])),
            project_camera(np.array([tex_x0, tex_y1, fabric_z + 0.0003])),
        ]
        _paste_projected_texture(img, rendered_texture, tex_quad)
        draw = ImageDraw.Draw(img)
        if all(p is not None for p in tex_quad):
            quad2d = [(p[0], p[1]) for p in tex_quad if p is not None]
            draw.line(quad2d + [quad2d[0]], fill=(60, 255, 120), width=4)

    render_rows = [active_row] if focused_capture else range(plan.grid_rows)
    render_cols = [active_col] if focused_capture else range(plan.grid_cols)
    if rendered_texture is None:
        for row in render_rows:
            for col in render_cols:
                x0 = grid_origin_x + col * cell_w
                y0 = grid_origin_y + row * cell_l
                x1, y1 = x0 + cell_w, y0 + cell_l
                colors = _cell_color_set(plan, row, col)
                cell_poly = [
                    project_camera(np.array([x0, y0, fabric_z])),
                    project_camera(np.array([x1, y0, fabric_z])),
                    project_camera(np.array([x1, y1, fabric_z])),
                    project_camera(np.array([x0, y1, fabric_z])),
                ]
                if all(p is not None for p in cell_poly):
                    if color_picker_mode:
                        fill_rgba = colors[0]
                        fill = tuple(int(255 * c) for c in fill_rgba[:3])
                    else:
                        fill = (18, 23, 31)
                    draw.polygon([(p[0], p[1]) for p in cell_poly if p is not None], fill=fill)
                depth_to_cell = max(float(np.dot(np.array([x0 + cell_w * 0.5, y0 + cell_l * 0.5, fabric_z]) - lens_pos, forward)), near)
                yarn_px = max(3, min(18, int(focal * max(min(cell_w, cell_l) * 0.030, 0.0014) / depth_to_cell)))
                repeat_rows = max(1, int(plan.pattern_repeat_rows))
                repeat_cols = max(1, int(plan.pattern_repeat_cols))
                tile_w = cell_w / repeat_cols
                tile_l = cell_l / repeat_rows
                if color_picker_mode:
                    fill_rgba = colors[0]
                    fill = tuple(int(255 * c) for c in fill_rgba[:3])
                    for tr in range(repeat_rows):
                        for tc in range(repeat_cols):
                            tx0 = x0 + tc * tile_w
                            ty0 = y0 + tr * tile_l
                            tx1 = tx0 + tile_w
                            ty1 = ty0 + tile_l
                            tile_poly = [
                                project_camera(np.array([tx0, ty0, fabric_z + 0.00035])),
                                project_camera(np.array([tx1, ty0, fabric_z + 0.00035])),
                                project_camera(np.array([tx1, ty1, fabric_z + 0.00035])),
                                project_camera(np.array([tx0, ty1, fabric_z + 0.00035])),
                            ]
                            if all(p is not None for p in tile_poly):
                                draw.polygon([(p[0], p[1]) for p in tile_poly if p is not None], fill=fill)
                else:
                    curves = _smoothed_cell_model_curves(plan, row, col)
                    for tr in range(repeat_rows):
                        for tc in range(repeat_cols):
                            tile_x0 = x0 + tc * tile_w
                            tile_y0 = y0 + tr * tile_l
                            tile_cx = tile_x0 + tile_w * 0.5
                            tile_cy = tile_y0 + tile_l * 0.5
                            for curve_idx, curve in enumerate(curves):
                                rgba = colors[curve_idx % len(colors)]
                                color = tuple(int(255 * c) for c in rgba[:3])
                                shade = tuple(max(0, int(channel * 0.45)) for channel in color)
                                z = fabric_z + curve_idx * max(FABRIC_THICKNESS * 0.08, 0.00045)
                                pts = [
                                    np.array([
                                        tile_cx + float(p[0]) * tile_w * IMAGE_REPEAT_OVERLAP,
                                        tile_cy + float(p[1]) * tile_l * IMAGE_REPEAT_OVERLAP,
                                        z,
                                    ])
                                    for p in curve
                                ]
                                pts2d = visible_line(pts)
                                if len(pts2d) >= 2:
                                    draw.line(pts2d, fill=shade, width=max(1, yarn_px // max(repeat_rows, repeat_cols)) + 2, joint="curve")
                                    draw.line(pts2d, fill=color, width=max(1, yarn_px // max(repeat_rows, repeat_cols)), joint="curve")
                outline = (60, 255, 120) if (row, col) == (active_row, active_col) else (245, 245, 240)
                width = 4 if (row, col) == (active_row, active_col) else 1
                poly = [
                    project_camera(np.array([x0, y0, fabric_z + 0.0002])),
                    project_camera(np.array([x1, y0, fabric_z + 0.0002])),
                    project_camera(np.array([x1, y1, fabric_z + 0.0002])),
                    project_camera(np.array([x0, y1, fabric_z + 0.0002])),
                ]
                if all(p is not None for p in poly):
                    poly2d = [(p[0], p[1]) for p in poly if p is not None]
                    draw.line(poly2d + [poly2d[0]], fill=outline, width=width)

    overlay = Image.new("L", (width_px, height_px), 0)
    mask_draw = ImageDraw.Draw(overlay)
    mask_draw.ellipse([-70, -45, width_px + 70, height_px + 45], fill=255)
    mask = overlay.filter(ImageFilter.GaussianBlur(12))
    dark = Image.new("RGB", (width_px, height_px), (0, 0, 0))
    img = Image.composite(img, dark, mask)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 430, 54], fill=(0, 0, 0))
    draw.text((14, 10), f"UR5 camera | station {station_id + 1} | row {active_row + 1}, col {active_col + 1}", fill=(255, 255, 255))
    draw.text(
        (14, 30),
        f"{view_name} | {capture_mode} | repeats {plan.pattern_repeat_rows}x{plan.pattern_repeat_cols} | tcp z {tcp_pos[2]:.3f} m",
        fill=(210, 230, 255),
    )

    return img


def save_camera_image(
    plan: FabricPlan,
    tcp_pos: np.ndarray,
    output_dir: Path,
    target_index: int,
    station_id: int,
    view_name: str,
    target_pose: np.ndarray | None = None,
    color_picker_mode: bool = False,
    capture_mode: str = CAMERA_CAPTURE_NATURAL,
    camera_zoom: float = 1.0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    img = render_camera_image(
        plan,
        tcp_pos,
        target_index,
        station_id,
        view_name,
        target_pose=target_pose,
        color_picker_mode=color_picker_mode,
        capture_mode=capture_mode,
        camera_zoom=camera_zoom,
    )
    active_row, active_col = plan.station_cells[station_id]
    clean_view = view_name.replace(" ", "_")
    clean_mode = "focused_batch" if str(capture_mode) == CAMERA_CAPTURE_FOCUSED else "natural"
    path = output_dir / f"scan_{target_index + 1:04d}_{clean_mode}_row_{active_row + 1:02d}_col_{active_col + 1:02d}_station_{station_id + 1:03d}_{clean_view}.png"
    img.save(path)
    return path


class SimulationControls:
    def __init__(self, args: argparse.Namespace):
        import tkinter as tk
        from tkinter import ttk

        self.root = tk.Tk()
        self.root.title("Fabric Scanner Controls")
        self.root.geometry("560x420")
        self.root.minsize(500, 360)
        self.root.resizable(True, True)
        self.speed = tk.DoubleVar(value=float(args.speed))
        self.camera_enabled = tk.BooleanVar(value=bool(args.add_camera))
        self.save_images = tk.BooleanVar(value=bool(args.save_images))
        self.status = tk.StringVar(value="Starting")
        self.paused = tk.BooleanVar(value=False)
        self.output_dir = Path(args.image_dir).resolve()
        self.save_now_requested = False
        self.closed = False

        frame = ttk.Frame(self.root, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Simulation speed").pack(anchor="w")
        ttk.Scale(frame, from_=0.05, to=2.0, variable=self.speed, length=300).pack(fill="x", pady=(4, 2))
        ttk.Label(frame, textvariable=self.status).pack(anchor="w", pady=(0, 12))
        ttk.Checkbutton(frame, text="Show camera on grabber", variable=self.camera_enabled).pack(anchor="w", pady=4)
        ttk.Checkbutton(frame, text="Save camera images", variable=self.save_images).pack(anchor="w", pady=4)
        ttk.Label(frame, text=f"Output: {self.output_dir}", wraplength=390).pack(anchor="w", pady=(12, 0))
        image_buttons = ttk.Frame(frame)
        image_buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(image_buttons, text="Save Image Now", command=self.request_save_now).pack(side="left")
        ttk.Button(image_buttons, text="Open Folder", command=self.open_output_folder).pack(side="left", padx=(8, 0))
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(14, 0))
        ttk.Button(buttons, text="Stop Motion", command=self.stop_motion).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(buttons, text="Continue", command=self.continue_motion).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(buttons, text="End Simulation", command=self.close).pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def stop_motion(self) -> None:
        self.paused.set(True)

    def continue_motion(self) -> None:
        self.paused.set(False)

    def request_save_now(self) -> None:
        self.save_now_requested = True

    def consume_save_now(self) -> bool:
        requested = self.save_now_requested
        self.save_now_requested = False
        return requested

    def open_output_folder(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(self.output_dir))

    def close(self) -> None:
        self.closed = True
        self.root.destroy()

    def update(self, target_index: int, total: int, saved_count: int) -> bool:
        if self.closed:
            return False
        state = "paused" if self.paused.get() else "running"
        self.status.set(f"Stop {target_index + 1}/{total} | {state} | speed {self.speed.get():.2f}x | images {saved_count}")
        try:
            self.root.update_idletasks()
            self.root.update()
            return True
        except Exception:
            self.closed = True
            return False


def run_simulation(plan: FabricPlan, args: argparse.Namespace) -> None:
    mujoco, model, data, site_id, handle = load_ur5e_scene()
    controls = None if args.no_run_gui else SimulationControls(args)
    image_dir = Path(args.image_dir).resolve()
    executed: list[np.ndarray] = []
    target_index = 0
    dwell_until = 0.0
    saved_count = 0
    saved_targets: set[int] = set()
    saved_stations: set[int] = set()
    was_paused = False
    print("[mujoco] viewer open. Close the window to stop.")
    print("[legend] grey=planned full path, green=planned completed path, orange=executed TCP trail")

    try:
        while handle.is_running() and target_index < len(plan.poses):
            if controls is not None and not controls.update(target_index, len(plan.poses), saved_count):
                break
            pose = plan.poses[target_index]
            paused = controls.paused.get() if controls is not None else False
            if paused:
                was_paused = True
            elif was_paused:
                dwell_until = 0.0
                was_paused = False
            speed = controls.speed.get() if controls is not None else args.speed
            if not paused:
                for _ in range(max(1, int(IK_SUBSTEPS * max(speed, 0.05)))):
                    step_ik(mujoco, model, data, site_id, pose)
            tcp = get_tcp(mujoco, model, data, site_id)
            if not paused:
                executed.append(tcp[:3].copy())
            camera_enabled = bool(args.add_camera)
            save_images = bool(args.save_images)
            if controls is not None:
                camera_enabled = controls.camera_enabled.get()
                save_images = controls.save_images.get() and not paused
                if controls.save_images.get():
                    image_dir.mkdir(parents=True, exist_ok=True)
                if controls.consume_save_now():
                    station = plan.station_ids[target_index]
                    path = save_camera_image(
                        plan,
                        tcp[:3],
                        image_dir,
                        target_index,
                        station,
                        plan.view_names[target_index],
                        target_pose=pose,
                        capture_mode=args.capture_mode,
                    )
                    saved_count += 1
                    saved_targets.add(target_index)
                    if is_scan_view(plan.view_names[target_index]):
                        saved_stations.add(station)
                    print(f"[camera] saved {path}")
            elif save_images:
                image_dir.mkdir(parents=True, exist_ok=True)
            draw_scene(
                mujoco,
                handle,
                plan,
                target_index,
                executed,
                camera_enabled=camera_enabled,
                current_pose=tcp,
                target_pose=pose,
            )

            err_pos, err_rot = pose_errors(tcp, pose)
            now = time.monotonic()
            if not paused and err_pos < TARGET_TOL and err_rot < TARGET_ROT_TOL:
                if dwell_until == 0.0:
                    dwell_until = now + max(args.dwell, 0.0)
                elif now >= dwell_until:
                    station = plan.station_ids[target_index]
                    should_save = should_save_scan_image(
                        args,
                        plan,
                        target_index,
                        saved_targets,
                        saved_stations,
                        save_images=save_images,
                    )
                    if should_save:
                        saved_targets.add(target_index)
                        saved_stations.add(station)
                        path = save_camera_image(
                            plan,
                            tcp[:3],
                            image_dir,
                            target_index,
                            station,
                            plan.view_names[target_index],
                            target_pose=pose,
                            capture_mode=args.capture_mode,
                        )
                        saved_count += 1
                        print(f"[camera] saved {path}")
                    target_index += 1
                    dwell_until = 0.0
                    if target_index < len(plan.poses):
                        station = plan.station_ids[target_index]
                        cell = plan.station_cells[station]
                        print(f"[scan] {target_index + 1:04d}/{len(plan.poses)} station={station + 1} cell={cell} view={plan.view_names[target_index]}")
            time.sleep(0.01)
    finally:
        handle.close()
        if controls is not None and not controls.closed:
            controls.close()

    print(f"[done] executed trail points: {len(executed)}")
    if saved_count:
        print(f"[done] saved camera images: {saved_count} in {image_dir}")


def show_missing_mujoco_help(project_dir: Path) -> None:
    message = (
        "MuJoCo is not installed in the Python environment that launched this script.\n\n"
        "To run the simulator, install the MuJoCo dependencies:\n"
        f"  cd {project_dir}\n"
        "  python -m pip install -r requirements-mujoco.txt\n\n"
        "To move the real robot without MuJoCo, restart this script and click "
        "'Run Real UR5' instead of 'Start Simulation'."
    )
    print(f"[mujoco] {message}")
    try:
        import tkinter.messagebox as messagebox

        messagebox.showerror("MuJoCo is missing", message)
    except Exception:
        pass


def run_robot_motion(plan: FabricPlan, args: argparse.Namespace) -> None:
    script = generate_urscript(
        plan.poses,
        accel=args.robot_acc,
        vel=args.robot_vel,
        blend_r=0.0,
        dwell=max(args.robot_dwell, 0.0),
        station_ids=plan.station_ids,
        view_names=plan.view_names,
        prog_name="mujoco_fabric_scanner_real",
    )
    if not script.endswith("\n"):
        script += "\n"

    print(f"[robot] sending URScript to UR5 at {args.robot_ip}:{args.robot_port}")
    print("[robot] robot must be in Remote Control mode before sending.")
    print(f"[robot] poses={len(plan.poses)} vel={args.robot_vel:.3f} acc={args.robot_acc:.3f}")
    try:
        with socket.create_connection((args.robot_ip, args.robot_port), timeout=5.0) as sock:
            sock.sendall(script.encode("utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Could not send URScript to {args.robot_ip}:{args.robot_port}: {exc}") from exc
    print("[robot] program sent. Watch the teach pendant / robot motion for completion.")


def run_setup_gui(args: argparse.Namespace):
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("Fabric Scanner Setup")
    root.geometry("1040x680")
    root.minsize(760, 520)
    root.resizable(True, True)
    root.bind("<F11>", lambda _event: root.state("zoomed"))
    root.bind("<Escape>", lambda _event: root.state("normal"))

    values = {
        "execution_mode": tk.StringVar(value=args.execution_mode),
        "width": tk.DoubleVar(value=args.width),
        "length": tk.DoubleVar(value=args.length),
        "rows": tk.IntVar(value=args.rows),
        "cols": tk.IntVar(value=args.cols),
        "number_of_angles": tk.IntVar(value=args.number_of_angles),
        "view_radius": tk.DoubleVar(value=args.view_radius),
        "angle_lift": tk.DoubleVar(value=args.angle_lift),
        "speed": tk.DoubleVar(value=args.speed),
        "add_camera": tk.BooleanVar(value=args.add_camera),
        "save_images": tk.BooleanVar(value=args.save_images),
        "image_every": tk.StringVar(value=args.image_every),
        "robot_ip": tk.StringVar(value=args.robot_ip),
        "robot_port": tk.IntVar(value=args.robot_port),
        "robot_vel": tk.DoubleVar(value=args.robot_vel),
        "robot_acc": tk.DoubleVar(value=args.robot_acc),
        "robot_dwell": tk.DoubleVar(value=args.robot_dwell),
    }
    result = {"args": None}
    summary = tk.StringVar()
    preview_after_id = {"id": None}

    shell = ttk.Frame(root)
    shell.pack(fill="both", expand=True)
    buttons = ttk.Frame(shell, padding=(16, 10))
    buttons.pack(fill="x", side="top")
    frame = ttk.Frame(shell, padding=16)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(0, weight=0)
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(0, weight=1)
    left = ttk.Frame(frame)
    left.grid(row=0, column=0, sticky="nsw", padx=(0, 18))
    right = ttk.Frame(frame)
    right.grid(row=0, column=1, sticky="nsew")
    right.columnconfigure(0, weight=1)
    right.rowconfigure(1, weight=1)
    ttk.Label(buttons, text="Fabric Scanner Setup", font=("Segoe UI", 15, "bold")).pack(side="left")
    ttk.Button(buttons, text="RUN REAL UR5", command=lambda: start(MODE_ROBOT)).pack(side="left", padx=(18, 0))
    ttk.Button(buttons, text="Start Simulation", command=lambda: start(MODE_SIMULATION)).pack(side="left", padx=(8, 0))
    ttk.Label(left, text="Fabric controls", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
    ttk.Label(left, text="Execution target").grid(row=1, column=0, sticky="w", pady=(0, 2))
    mode_combo = ttk.Combobox(
        left,
        state="readonly",
        width=24,
        values=("MuJoCo simulation", "Real UR5 robot motion"),
    )
    mode_combo.grid(row=1, column=1, sticky="e", pady=(0, 2))
    mode_combo.set("Real UR5 robot motion" if args.execution_mode == MODE_ROBOT else "MuJoCo simulation")

    def sync_execution_mode(_event=None) -> None:
        values["execution_mode"].set(MODE_ROBOT if mode_combo.get().startswith("Real UR5") else MODE_SIMULATION)

    mode_combo.bind("<<ComboboxSelected>>", sync_execution_mode)
    ttk.Label(
        left,
        text="Use Real UR5 robot motion to send the generated scan path to the controller.",
        foreground="#6b7280",
        wraplength=330,
    ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))

    slider_specs = [
        ("Full fabric width", "width", 0.10, 0.80, 0.01, "m"),
        ("Full fabric length", "length", 0.10, 0.80, 0.01, "m"),
        ("Rows", "rows", 1, 12, 1, ""),
        ("Columns", "cols", 1, 12, 1, ""),
        ("Number of angles", "number_of_angles", 1, 12, 1, ""),
        ("Angle tilt strength", "view_radius", 0.0, 0.060, 0.001, ""),
        ("Angled view lift", "angle_lift", 0.0, 0.060, 0.001, "m"),
        ("Initial speed", "speed", 0.05, 2.0, 0.05, "x"),
        ("Robot velocity", "robot_vel", 0.005, 0.080, 0.005, "m/s"),
        ("Robot acceleration", "robot_acc", 0.020, 0.250, 0.010, "m/s2"),
        ("Robot dwell", "robot_dwell", 0.0, 2.0, 0.05, "s"),
    ]
    value_labels = {}

    def format_value(key: str, unit: str) -> str:
        value = values[key].get()
        if key in {"rows", "cols", "number_of_angles"}:
            return f"{int(round(value))}{unit}"
        return f"{float(value):.3f}{unit}"

    def snap_value(key: str, step: float) -> None:
        value = values[key].get()
        if step >= 1:
            values[key].set(int(round(value)))
        else:
            values[key].set(round(float(value) / step) * step)

    for index, (label, key, low, high, step, unit) in enumerate(slider_specs, start=1):
        row = index * 2 + 1
        ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", pady=(5, 0))
        value_labels[key] = ttk.Label(left, width=9, anchor="e")
        value_labels[key].grid(row=row, column=1, sticky="e", pady=(5, 0))
        scale = ttk.Scale(left, from_=low, to=high, variable=values[key], length=300, command=lambda _v, k=key, s=step: snap_value(k, s))
        scale.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(0, 3))

    ttk.Label(left, text="Robot IP").grid(row=28, column=0, sticky="w", pady=(12, 4))
    ttk.Entry(left, textvariable=values["robot_ip"], width=18).grid(row=28, column=1, sticky="e", pady=(12, 4))
    ttk.Label(left, text="Robot port").grid(row=29, column=0, sticky="w", pady=(2, 4))
    ttk.Entry(left, textvariable=values["robot_port"], width=18).grid(row=29, column=1, sticky="e", pady=(2, 4))
    ttk.Checkbutton(left, text="Add camera on grabber", variable=values["add_camera"]).grid(row=30, column=0, columnspan=2, sticky="w", pady=(12, 4))
    ttk.Checkbutton(left, text="Save scanner images", variable=values["save_images"]).grid(row=31, column=0, columnspan=2, sticky="w", pady=4)
    ttk.Label(left, text="Image saving mode").grid(row=32, column=0, sticky="w", pady=(2, 4))
    image_mode_combo = ttk.Combobox(
        left,
        state="readonly",
        width=18,
        values=("Every angle view", "One per station"),
    )
    image_mode_combo.grid(row=32, column=1, sticky="e", pady=(2, 4))
    image_mode_combo.set("Every angle view" if args.image_every == CAMERA_SAVE_EVERY_VIEW else "One per station")

    def sync_image_every(_event=None) -> None:
        values["image_every"].set(
            CAMERA_SAVE_EVERY_VIEW
            if image_mode_combo.get().startswith("Every")
            else CAMERA_SAVE_EVERY_STATION
        )

    image_mode_combo.bind("<<ComboboxSelected>>", sync_image_every)
    ttk.Label(left, textvariable=summary, foreground="#355c7d", wraplength=330).grid(row=33, column=0, columnspan=2, sticky="w", pady=(14, 8))

    ttk.Label(right, text="Live fabric preview", font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))
    canvas = tk.Canvas(right, bg="#f7f7f2", highlightthickness=1, highlightbackground="#c8c8c0")
    canvas.grid(row=1, column=0, sticky="nsew")

    def current_grid():
        width = max(1e-6, float(values["width"].get()))
        length = max(1e-6, float(values["length"].get()))
        rows_count = max(1, int(values["rows"].get()))
        cols = max(1, int(values["cols"].get()))
        square_width = width / cols
        square_length = length / rows_count
        number_of_angles = max(1, int(values["number_of_angles"].get()))
        return width, length, square_width, square_length, rows_count, cols, number_of_angles

    def draw_preview() -> None:
        canvas.delete("all")
        try:
            width, length, square_width, square_length, rows_count, cols, number_of_angles = current_grid()
            canvas_w = max(240, canvas.winfo_width())
            canvas_h = max(240, canvas.winfo_height())
            pad = 36
            scale = min((canvas_w - 2 * pad) / width, (canvas_h - 2 * pad) / length)
            ox = (canvas_w - width * scale) / 2
            oy = (canvas_h - length * scale) / 2
            for row in range(rows_count):
                for col in range(cols):
                    x0 = ox + col * square_width * scale
                    y0 = oy + row * square_length * scale
                    x1 = x0 + square_width * scale
                    y1 = y0 + square_length * scale
                    color = "#%02x%02x%02x" % tuple(int(255 * c) for c in _cell_color(row, col)[:3])
                    canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#ffffff", width=2)
                    px = (x0 + x1) * 0.5
                    py = (y0 + y1) * 0.5
                    ray = min(x1 - x0, y1 - y0) * 0.28
                    for angle_index in range(number_of_angles):
                        theta = 2.0 * math.pi * angle_index / number_of_angles
                        canvas.create_line(px, py, px + ray * math.cos(theta), py - ray * math.sin(theta), fill="#111827", width=1)
                    canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#111827", outline="#ffffff")
            canvas.create_rectangle(ox, oy, ox + width * scale, oy + length * scale, outline="#1f2937", width=3)
            stations = rows_count * cols
            summary.set(
                f"Full fabric: {width:.3f} x {length:.3f} m | "
                f"Grid: {rows_count} x {cols} | "
                f"Square: {square_width:.3f} x {square_length:.3f} m | "
                f"Scan stations: {stations} | Angles: {number_of_angles} | Views: {stations * number_of_angles + 2}"
            )
            for label, key, _low, _high, _step, unit in slider_specs:
                value_labels[key].configure(text=format_value(key, unit))
        except Exception:
            canvas.create_text(235, 235, text="Enter positive numbers to preview fabric", fill="#6b7280", font=("Segoe UI", 12))
            summary.set("Enter positive sizes to preview the grid.")
        preview_after_id["id"] = root.after(200, draw_preview)

    def start(force_mode: str | None = None) -> None:
        if force_mode is not None:
            values["execution_mode"].set(force_mode)
            mode_combo.set("Real UR5 robot motion" if force_mode == MODE_ROBOT else "MuJoCo simulation")
        else:
            sync_execution_mode()
        try:
            args.width = float(values["width"].get())
            args.length = float(values["length"].get())
            args.rows = int(values["rows"].get())
            args.cols = int(values["cols"].get())
            args.square_width = args.width / max(1, args.cols)
            args.square_length = args.length / max(1, args.rows)
            args.number_of_angles = int(values["number_of_angles"].get())
            args.view_radius = float(values["view_radius"].get())
            args.angle_lift = float(values["angle_lift"].get())
            args.speed = float(values["speed"].get())
            args.add_camera = bool(values["add_camera"].get())
            args.save_images = bool(values["save_images"].get())
            sync_image_every()
            args.image_every = values["image_every"].get()
            args.execution_mode = values["execution_mode"].get()
            args.robot_ip = values["robot_ip"].get().strip()
            args.robot_port = int(values["robot_port"].get())
            args.robot_vel = float(values["robot_vel"].get())
            args.robot_acc = float(values["robot_acc"].get())
            args.robot_dwell = float(values["robot_dwell"].get())
            if min(args.width, args.length, args.rows, args.cols, args.number_of_angles) <= 0:
                raise ValueError
            if args.execution_mode == MODE_ROBOT and (not args.robot_ip or args.robot_port <= 0):
                raise ValueError
        except Exception:
            messagebox.showerror("Check setup", "Use positive numeric values for fabric size, square size, and scan counts.")
            return
        if preview_after_id["id"] is not None:
            root.after_cancel(preview_after_id["id"])
            preview_after_id["id"] = None
        result["args"] = args
        root.destroy()

    def cancel() -> None:
        if preview_after_id["id"] is not None:
            root.after_cancel(preview_after_id["id"])
            preview_after_id["id"] = None
        result["args"] = None
        root.destroy()

    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right", padx=(8, 0))
    root.protocol("WM_DELETE_WINDOW", cancel)
    draw_preview()
    root.mainloop()
    return result["args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo UR5e fabric-grid scanner with executed path trail")
    parser.add_argument("--execution-mode", choices=[MODE_SIMULATION, MODE_ROBOT], default=MODE_SIMULATION, help="Run MuJoCo simulation or execute on the real UR5 robot")
    parser.add_argument("--rows", type=int, default=4, help="Number of fabric swatch rows")
    parser.add_argument("--cols", type=int, default=5, help="Number of fabric swatch columns")
    parser.add_argument("--number-of-angles", type=int, default=6, help="Scanner angles around each square center; step is 360 / N degrees")
    parser.add_argument("--width", type=float, default=0.34, help="Fabric width in local metres")
    parser.add_argument("--length", type=float, default=0.24, help="Fabric length in local metres")
    parser.add_argument("--edge-margin", type=float, default=0.004, help="Outer fabric margin before sampling")
    parser.add_argument("--square-margin", type=float, default=0.006, help="Margin inside each colored square")
    parser.add_argument("--surface-wave", type=float, default=0.003, help="Small Z wave to mimic fabric")
    parser.add_argument("--view-radius", type=float, default=0.018, help="Tilt strength for angled views; scanner XY position stays fixed at the square center")
    parser.add_argument("--angle-lift", type=float, default=0.014, help="Z lift for angled views")
    parser.add_argument("--approach-lift", type=float, default=0.040, help="Lift for approach/retreat points")
    parser.add_argument("--center", nargs=3, type=float, default=DEFAULT_CENTER.tolist(), metavar=("X", "Y", "Z"))
    parser.add_argument("--max-span", nargs=3, type=float, default=DEFAULT_MAX_SPAN.tolist(), metavar=("X", "Y", "Z"))
    parser.add_argument("--speed", type=float, default=0.35, help="Viewer animation speed multiplier")
    parser.add_argument("--dwell", type=float, default=0.20, help="Pause at each scan view before capturing/advancing, in seconds")
    parser.add_argument("--no-setup-gui", action="store_true", help="Skip the setup window and use CLI/default values")
    parser.add_argument("--no-run-gui", action="store_true", help="Skip the live speed/camera control window")
    parser.add_argument("--add-camera", action="store_true", help="Show a camera marker mounted at the grabber/TCP")
    parser.add_argument("--save-images", action="store_true", help="Save simulated scanner camera images")
    parser.add_argument("--image-dir", default="scanner_images", help="Directory for saved camera images")
    parser.add_argument("--image-every", choices=[CAMERA_SAVE_EVERY_STATION, CAMERA_SAVE_EVERY_VIEW], default=CAMERA_SAVE_EVERY_VIEW, help="Save one image per station or every angle view")
    parser.add_argument("--capture-mode", choices=[CAMERA_CAPTURE_NATURAL, CAMERA_CAPTURE_FOCUSED], default=CAMERA_CAPTURE_NATURAL, help="Natural robot camera images or focused single-batch images")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="Real UR5 robot IP address for URScript mode")
    parser.add_argument("--robot-port", type=int, default=DEFAULT_ROBOT_PORT, help="Real UR5 primary interface port for URScript mode")
    parser.add_argument("--robot-vel", type=float, default=0.015, help="Real robot Cartesian velocity for moveL (m/s)")
    parser.add_argument("--robot-acc", type=float, default=0.08, help="Real robot Cartesian acceleration for moveL (m/s^2)")
    parser.add_argument("--robot-dwell", type=float, default=0.20, help="Real robot dwell time at each scan pose (s)")
    parser.add_argument("--allow-unsafe-real", action="store_true", help="Allow real robot execution even if the path safety check reports issues")
    parser.add_argument("--export-script", action="store_true", help="Also export a URScript path")
    parser.add_argument("--save-points", action="store_true", help="Save mapped scan points as CSV")
    parser.add_argument("--output", default="mujoco_fabric_scanner.script", help="URScript output filename")
    parser.add_argument("--csv-output", default="mujoco_fabric_scanner_points.csv", help="CSV output filename")
    parser.add_argument("--model-json", default="", help="Saved app params.json containing the edited knitted model")
    parser.add_argument("--palette-json", default="", help="JSON list of RGB/RGBA colors for fabric mini-squares")
    parser.add_argument("--cell-colors-json", default="", help="JSON rows x cols x 4 x RGB/RGBA colors for each mini-fabric")
    parser.add_argument("--cell-model-curves-json", default="", help="JSON rows*cols list of random model curves for each mini-fabric")
    parser.add_argument("--random-patterns", action="store_true", help="Generate random bitmap-based fabric patterns instead of scanning the saved edited model")
    parser.add_argument("--pattern-rows", type=int, default=4, help="Rows in each random bitmap pattern")
    parser.add_argument("--pattern-cols", type=int, default=5, help="Columns in each random bitmap pattern")
    parser.add_argument("--pattern-repeat-rows", type=int, default=3, help="How many times to repeat each random bitmap vertically inside one fabric sample")
    parser.add_argument("--pattern-repeat-cols", type=int, default=3, help="How many times to repeat each random bitmap horizontally inside one fabric sample")
    parser.add_argument("--pattern-density", type=float, default=0.62, help="Probability that a random bitmap stitch is active")
    parser.add_argument("--random-seed", type=int, default=1, help="Seed for reproducible random pattern generation")
    args = parser.parse_args()
    if args.palette_json:
        try:
            args.palette = json.loads(args.palette_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --palette-json: {exc}") from exc
    else:
        args.palette = None
    if args.cell_colors_json:
        try:
            args.cell_color_sets = json.loads(args.cell_colors_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --cell-colors-json: {exc}") from exc
    else:
        args.cell_color_sets = None
    if args.cell_model_curves_json:
        try:
            args.cell_model_curves = json.loads(args.cell_model_curves_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --cell-model-curves-json: {exc}") from exc
    else:
        args.cell_model_curves = None
    return args


def confirm_simulation_capture_options(args: argparse.Namespace, plan: FabricPlan) -> bool:
    import tkinter as tk
    from tkinter import ttk

    scan_view_count = sum(1 for name in plan.view_names if is_scan_view(name))
    station_count = len(plan.mapped_stations)
    expected_images = scan_view_count if args.image_every == CAMERA_SAVE_EVERY_VIEW else station_count

    root = tk.Tk()
    root.title("Confirm Scanner Capture")
    root.geometry("520x360")
    root.resizable(False, False)

    add_camera = tk.BooleanVar(value=bool(args.add_camera))
    save_images = tk.BooleanVar(value=bool(args.save_images))
    image_every = tk.StringVar(value=args.image_every)
    result = {"start": False}
    summary = tk.StringVar()

    frame = ttk.Frame(root, padding=18)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Before Starting Simulation", font=("Segoe UI", 14, "bold")).pack(anchor="w")
    ttk.Label(
        frame,
        text=(
            "Choose the scanner camera workflow before the robot path starts. "
            "Angle images are captured at the same scan views used by the path."
        ),
        wraplength=470,
        foreground="#4b5563",
    ).pack(anchor="w", pady=(6, 14))
    ttk.Checkbutton(frame, text="Show camera on grabber", variable=add_camera).pack(anchor="w", pady=4)
    ttk.Checkbutton(frame, text="Save scanner images", variable=save_images).pack(anchor="w", pady=4)
    ttk.Label(frame, text="Save images").pack(anchor="w", pady=(12, 2))
    mode = ttk.Combobox(frame, state="readonly", values=("Every angle view", "One per station"), width=22)
    mode.pack(anchor="w")
    mode.set("Every angle view" if args.image_every == CAMERA_SAVE_EVERY_VIEW else "One per station")

    def sync_mode(_event=None) -> None:
        image_every.set(CAMERA_SAVE_EVERY_VIEW if mode.get().startswith("Every") else CAMERA_SAVE_EVERY_STATION)
        image_count = scan_view_count if image_every.get() == CAMERA_SAVE_EVERY_VIEW else station_count
        enabled_text = "will be saved" if save_images.get() else "will not be saved"
        summary.set(
            f"Stations: {station_count} | Angles/station: {args.number_of_angles} | "
            f"Expected images: {image_count if save_images.get() else 0} ({enabled_text})\n"
            f"Output folder: {Path(args.image_dir).resolve()}"
        )

    mode.bind("<<ComboboxSelected>>", sync_mode)
    add_camera.trace_add("write", lambda *_: sync_mode())
    save_images.trace_add("write", lambda *_: sync_mode())
    ttk.Label(frame, textvariable=summary, wraplength=470, foreground="#355c7d").pack(anchor="w", pady=(14, 12))

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", side="bottom")

    def start() -> None:
        args.add_camera = bool(add_camera.get())
        args.save_images = bool(save_images.get())
        args.image_every = image_every.get()
        result["start"] = True
        root.destroy()

    def cancel() -> None:
        result["start"] = False
        root.destroy()

    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right", padx=(8, 0))
    ttk.Button(buttons, text="Start Path", command=start).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", cancel)
    sync_mode()
    root.mainloop()
    return bool(result["start"])


def main() -> None:
    args = parse_args()
    if not args.no_setup_gui:
        args = run_setup_gui(args)
        if args is None:
            print("Canceled.")
            return
    project_dir = Path(__file__).resolve().parent
    args.image_dir = str((project_dir / args.image_dir).resolve())
    plan = build_plan(args)
    if args.execution_mode in {MODE_SIMULATION, MODE_ROBOT}:
        original_pose_count = len(plan.poses)
        plan = densify_plan_for_robot(plan)
        if len(plan.poses) != original_pose_count:
            print(f"[path] densified path: {original_pose_count} -> {len(plan.poses)} poses")

    print("=" * 64)
    print("  UR5e MuJoCo Fabric Scanner")
    print("=" * 64)
    print(f"fabric grid: {args.rows} x {args.cols}")
    print(f"fabric size: {args.width:.3f} x {args.length:.3f} m")
    print(f"square size: {plan.square_width:.3f} x {plan.square_length:.3f} m")
    print(f"scan stations: {len(plan.mapped_stations)}")
    print(f"angles per station: {args.number_of_angles} ({360.0 / max(1, args.number_of_angles):.1f} deg step)")
    print(f"scan views: {len(plan.mapped_points)}")
    print(f"mode: {args.execution_mode}")
    if args.add_camera:
        print(f"camera: enabled | save images: {args.save_images} | output: {args.image_dir}")

    safe, issues = assess_plan_safety(plan.mapped_points, max_step=0.08)
    print(f"safety: {'PASS' if safe else 'CHECK'}")
    for issue in issues:
        print(f"  - {issue}")
    if args.execution_mode == MODE_ROBOT and not safe and not args.allow_unsafe_real:
        print("[robot] refusing to run on real robot because the safety check failed")
        return

    if args.save_points:
        csv_path = project_dir / args.csv_output
        save_points_csv(csv_path, plan.mapped_points)
        print(f"csv: {csv_path}")

    if args.export_script:
        script = generate_urscript(
            plan.poses,
            accel=0.10,
            vel=0.015,
            blend_r=0.0,
            dwell=max(args.dwell, 0.0),
            station_ids=plan.station_ids,
            view_names=plan.view_names,
            prog_name="mujoco_fabric_scanner",
        )
        script_path = project_dir / args.output
        script_path.write_text(script + "\n", encoding="utf-8")
        print(f"urscript: {script_path}")

    if args.execution_mode == MODE_SIMULATION:
        if not args.no_setup_gui:
            if not confirm_simulation_capture_options(args, plan):
                print("[mujoco] canceled before starting simulation path")
                return
            image_scope = "every angle view" if args.image_every == CAMERA_SAVE_EVERY_VIEW else "one per station"
            print(f"camera: {'enabled' if args.add_camera else 'hidden'} | save images: {args.save_images} ({image_scope}) | output: {args.image_dir}")
        try:
            run_simulation(plan, args)
        except ModuleNotFoundError as exc:
            if exc.name == "mujoco":
                show_missing_mujoco_help(project_dir)
                return
            raise
    else:
        if not args.no_setup_gui:
            import tkinter.messagebox as messagebox

            ok = messagebox.askyesno(
                "Run Real UR5 Robot",
                f"This will connect to {args.robot_ip}:{args.robot_port} and move the real UR5 through {len(plan.poses)} Cartesian poses.\n\n"
                "Confirm the robot is in Remote Control mode, the workspace is clear, and the emergency stop is reachable.",
            )
            if not ok:
                print("[robot] canceled before connection")
                return
        run_robot_motion(plan, args)


if __name__ == "__main__":
    main()
