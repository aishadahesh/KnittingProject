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
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from Robot_fabric_scanner import (
    DEFAULT_CENTER,
    DEFAULT_MAX_SPAN,
    TOOL_DOWN_ROTVEC,
    assess_plan_safety,
    expand_stations_to_viewpoints,
    fit_path_to_workspace,
    generate_urscript,
    points_and_rotvecs_to_poses,
    save_points_csv,
)

MUJOCO_DAMPING = 1e-2
MUJOCO_KP = 0.28
IK_SUBSTEPS = 3
TARGET_TOL = 0.006
MAX_TRAIL_POINTS = 6000
MARKER_RADIUS = 0.006
PATH_WIDTH = 0.003
FABRIC_THICKNESS = 0.003
CAMERA_MARKER_SIZE = np.array([0.018, 0.012, 0.010])
CAMERA_IMAGE_SIZE = (640, 480)
CAMERA_SAVE_EVERY_STATION = "station"
CAMERA_SAVE_EVERY_VIEW = "view"

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
    fabric_origin: np.ndarray
    fabric_size: np.ndarray
    grid_rows: int
    grid_cols: int
    square_width: float
    square_length: float


def _serpentine_indices(rows: int, cols: int):
    for row in range(rows):
        col_range = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
        for col in col_range:
            yield row, col


def _cell_color(row: int, col: int) -> tuple[float, float, float, float]:
    return SWATCH_PALETTE[(row * 3 + col * 5 + row * col) % len(SWATCH_PALETTE)]


def build_fabric_grid_stations(
    width: float,
    length: float,
    rows: int,
    cols: int,
    samples_x: int,
    samples_y: int,
    edge_margin: float,
    square_margin: float,
    surface_wave: float,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[float, float, float, float]]]:
    if min(width, length) <= 0:
        raise ValueError("fabric width and length must be positive")
    if min(rows, cols, samples_x, samples_y) <= 0:
        raise ValueError("grid and sample counts must be positive")

    cell_w = width / cols
    cell_l = length / rows
    stations: list[list[float]] = []
    cells: list[tuple[int, int]] = []
    colors: list[tuple[float, float, float, float]] = []

    for row, col in _serpentine_indices(rows, cols):
        x0 = -width / 2 + col * cell_w
        y0 = -length / 2 + row * cell_l
        inner_x0 = x0 + max(edge_margin, square_margin)
        inner_x1 = x0 + cell_w - max(edge_margin, square_margin)
        inner_y0 = y0 + max(edge_margin, square_margin)
        inner_y1 = y0 + cell_l - max(edge_margin, square_margin)
        if inner_x1 <= inner_x0:
            inner_x0, inner_x1 = x0 + cell_w * 0.35, x0 + cell_w * 0.65
        if inner_y1 <= inner_y0:
            inner_y0, inner_y1 = y0 + cell_l * 0.35, y0 + cell_l * 0.65

        xs = np.linspace(inner_x0, inner_x1, samples_x)
        ys = np.linspace(inner_y0, inner_y1, samples_y)
        local_points = [(x, y) for y in ys for x in xs]
        if (row + col) % 2:
            local_points.reverse()
        for x, y in local_points:
            z = surface_wave * math.sin(2 * math.pi * (x / width + 0.5))
            z += 0.5 * surface_wave * math.cos(2 * math.pi * (y / length + 0.5))
            stations.append([float(x), float(y), float(z)])
            cells.append((row, col))
            colors.append(_cell_color(row, col))

    return np.asarray(stations, dtype=float), cells, colors


def resolve_fabric_grid(args: argparse.Namespace) -> None:
    args.rows = max(1, int(args.rows))
    args.cols = max(1, int(args.cols))
    args.square_width = args.width / max(1, args.cols)
    args.square_length = args.length / max(1, args.rows)


def build_plan(args: argparse.Namespace) -> FabricPlan:
    resolve_fabric_grid(args)
    raw_stations, station_cells, station_colors = build_fabric_grid_stations(
        width=args.width,
        length=args.length,
        rows=args.rows,
        cols=args.cols,
        samples_x=args.samples_x,
        samples_y=args.samples_y,
        edge_margin=args.edge_margin,
        square_margin=args.square_margin,
        surface_wave=args.surface_wave,
    )
    raw_points, station_ids, view_names, rotvecs = expand_stations_to_viewpoints(
        raw_stations,
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
        fabric_origin=fabric_origin,
        fabric_size=fabric_size,
        grid_rows=args.rows,
        grid_cols=args.cols,
        square_width=args.square_width,
        square_length=args.square_length,
    )


def rot_from_wxyz(q: np.ndarray) -> Rotation:
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def load_ur5e_scene():
    import mujoco
    import mujoco.viewer
    from robot_descriptions import ur5e_mj_description

    model = mujoco.MjModel.from_xml_path(ur5e_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id < 0:
        raise RuntimeError("UR5e MJCF does not contain site 'attachment_site'")

    data.qpos[:6] = [0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0]
    mujoco.mj_forward(model, data)
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


def draw_scene(
    mujoco,
    handle,
    plan: FabricPlan,
    target_index: int,
    executed: list[np.ndarray],
    camera_enabled: bool = False,
) -> None:
    scn = handle.user_scn
    scn.ngeom = 0

    cell_w = plan.fabric_size[0] / plan.grid_cols
    cell_l = plan.fabric_size[1] / plan.grid_rows
    z = plan.fabric_origin[2] - FABRIC_THICKNESS * 0.5
    for row in range(plan.grid_rows):
        for col in range(plan.grid_cols):
            x = plan.fabric_origin[0] + cell_w * (col + 0.5)
            y = plan.fabric_origin[1] + cell_l * (row + 0.5)
            rgba = _cell_color(row, col)
            _add_box(mujoco, scn, [x, y, z], [cell_w * 0.48, cell_l * 0.48, FABRIC_THICKNESS], rgba)

    for i in range(len(plan.mapped_points) - 1):
        _add_segment(mujoco, scn, plan.mapped_points[i], plan.mapped_points[i + 1], (0.20, 0.24, 0.30, 0.28), 0.0018)

    if target_index > 0:
        for i in range(min(target_index, len(plan.mapped_points) - 1)):
            _add_segment(mujoco, scn, plan.mapped_points[i], plan.mapped_points[i + 1], (0.15, 0.85, 0.35, 0.55), 0.0026)

    trail = executed[-MAX_TRAIL_POINTS:]
    for a, b in zip(trail, trail[1:]):
        _add_segment(mujoco, scn, a, b, (1.00, 0.42, 0.08, 0.90), 0.0032)

    target = plan.mapped_points[min(target_index, len(plan.mapped_points) - 1)]
    station_id = plan.station_ids[min(target_index, len(plan.station_ids) - 1)]
    station = plan.mapped_stations[station_id]
    color = plan.cell_colors[station_id]
    _add_sphere(mujoco, scn, station, MARKER_RADIUS * 1.15, color)
    _add_sphere(mujoco, scn, target, MARKER_RADIUS * 1.45, (0.1, 1.0, 0.3, 1.0))
    if trail:
        _add_sphere(mujoco, scn, trail[-1], MARKER_RADIUS * 1.35, (1.0, 0.42, 0.08, 1.0))
        if camera_enabled:
            _add_camera_marker(mujoco, scn, trail[-1], station)

    handle.sync()


def save_camera_image(
    plan: FabricPlan,
    tcp_pos: np.ndarray,
    output_dir: Path,
    target_index: int,
    station_id: int,
    view_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    width_px, height_px = CAMERA_IMAGE_SIZE
    station = plan.mapped_stations[station_id]
    target = plan.mapped_points[target_index]
    cell_w = plan.fabric_size[0] / plan.grid_cols
    cell_l = plan.fabric_size[1] / plan.grid_rows
    view_span = max(cell_w * 2.4, cell_l * 2.4, 0.035)
    center = station[:2] + 0.35 * (target[:2] - station[:2])
    x_min, x_max = center[0] - view_span / 2, center[0] + view_span / 2
    y_min, y_max = center[1] - view_span / 2, center[1] + view_span / 2

    img = Image.new("RGB", (width_px, height_px), (19, 22, 28))

    def project_xy(point: np.ndarray) -> tuple[int, int]:
        x = int((point[0] - x_min) / max(1e-9, x_max - x_min) * (width_px - 1))
        y = int((y_max - point[1]) / max(1e-9, y_max - y_min) * (height_px - 1))
        return x, y

    from PIL import ImageDraw, ImageFilter

    draw = ImageDraw.Draw(img)
    for row in range(plan.grid_rows):
        for col in range(plan.grid_cols):
            x0 = plan.fabric_origin[0] + col * cell_w
            y0 = plan.fabric_origin[1] + row * cell_l
            x1, y1 = x0 + cell_w, y0 + cell_l
            if x1 < x_min or x0 > x_max or y1 < y_min or y0 > y_max:
                continue
            rgba = _cell_color(row, col)
            color = tuple(int(255 * c) for c in rgba[:3])
            p0 = project_xy(np.array([x0, y0]))
            p1 = project_xy(np.array([x1, y1]))
            rect = [min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])]
            draw.rectangle(rect, fill=color, outline=(245, 245, 240), width=2)

    for sid, point in enumerate(plan.mapped_stations):
        if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max:
            px, py = project_xy(point[:2])
            r = 5 if sid == station_id else 3
            fill = (255, 255, 255) if sid == station_id else (30, 30, 30)
            draw.ellipse([px - r, py - r, px + r, py + r], fill=fill, outline=(0, 0, 0))

    tx, ty = project_xy(target[:2])
    draw.line([tx - 16, ty, tx + 16, ty], fill=(0, 255, 80), width=3)
    draw.line([tx, ty - 16, tx, ty + 16], fill=(0, 255, 80), width=3)
    draw.ellipse([tx - 8, ty - 8, tx + 8, ty + 8], outline=(0, 255, 80), width=3)

    overlay = Image.new("L", (width_px, height_px), 0)
    mask_draw = ImageDraw.Draw(overlay)
    mask_draw.ellipse([-90, -60, width_px + 90, height_px + 60], fill=255)
    mask = overlay.filter(ImageFilter.GaussianBlur(12))
    dark = Image.new("RGB", (width_px, height_px), (0, 0, 0))
    img = Image.composite(img, dark, mask)
    draw = ImageDraw.Draw(img)
    draw.text((14, 12), f"station {station_id + 1} | {view_name}", fill=(255, 255, 255))
    draw.text((14, 32), f"tcp z {tcp_pos[2]:.3f} m", fill=(210, 230, 255))

    clean_view = view_name.replace(" ", "_")
    path = output_dir / f"scan_{target_index + 1:04d}_station_{station_id + 1:03d}_{clean_view}.png"
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
                    )
                    saved_count += 1
                    saved_targets.add(target_index)
                    print(f"[camera] saved {path}")
            elif save_images:
                image_dir.mkdir(parents=True, exist_ok=True)
            draw_scene(mujoco, handle, plan, target_index, executed, camera_enabled=camera_enabled)

            err = np.linalg.norm(tcp[:3] - pose[:3])
            now = time.monotonic()
            if not paused and err < TARGET_TOL:
                if dwell_until == 0.0:
                    dwell_until = now + max(args.dwell, 0.0)
                elif now >= dwell_until:
                    station = plan.station_ids[target_index]
                    should_save = save_images and target_index not in saved_targets
                    if args.image_every == CAMERA_SAVE_EVERY_STATION:
                        should_save = should_save and plan.view_names[target_index] == "top"
                    if should_save:
                        saved_targets.add(target_index)
                        path = save_camera_image(
                            plan,
                            tcp[:3],
                            image_dir,
                            target_index,
                            station,
                            plan.view_names[target_index],
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


def run_setup_gui(args: argparse.Namespace):
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("Fabric Scanner Setup")
    root.geometry("980x720")
    root.minsize(820, 560)
    root.resizable(True, True)
    root.bind("<F11>", lambda _event: root.state("zoomed"))
    root.bind("<Escape>", lambda _event: root.state("normal"))

    values = {
        "width": tk.DoubleVar(value=args.width),
        "length": tk.DoubleVar(value=args.length),
        "rows": tk.IntVar(value=args.rows),
        "cols": tk.IntVar(value=args.cols),
        "samples_x": tk.IntVar(value=args.samples_x),
        "samples_y": tk.IntVar(value=args.samples_y),
        "view_radius": tk.DoubleVar(value=args.view_radius),
        "angle_lift": tk.DoubleVar(value=args.angle_lift),
        "speed": tk.DoubleVar(value=args.speed),
        "add_camera": tk.BooleanVar(value=args.add_camera),
        "save_images": tk.BooleanVar(value=args.save_images),
    }
    result = {"args": None}
    summary = tk.StringVar()

    shell = ttk.Frame(root)
    shell.pack(fill="both", expand=True)
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
    ttk.Label(left, text="Fabric Scanner Setup", font=("Segoe UI", 15, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

    slider_specs = [
        ("Full fabric width", "width", 0.10, 0.80, 0.01, "m"),
        ("Full fabric length", "length", 0.10, 0.80, 0.01, "m"),
        ("Rows", "rows", 1, 12, 1, ""),
        ("Columns", "cols", 1, 12, 1, ""),
        ("Scan locations X", "samples_x", 1, 5, 1, ""),
        ("Scan locations Y", "samples_y", 1, 5, 1, ""),
        ("Angled view radius", "view_radius", 0.0, 0.060, 0.001, "m"),
        ("Angled view lift", "angle_lift", 0.0, 0.060, 0.001, "m"),
        ("Initial speed", "speed", 0.05, 2.0, 0.05, "x"),
    ]
    value_labels = {}

    def format_value(key: str, unit: str) -> str:
        value = values[key].get()
        if key in {"rows", "cols", "samples_x", "samples_y"}:
            return f"{int(round(value))}{unit}"
        return f"{float(value):.3f}{unit}"

    def snap_value(key: str, step: float) -> None:
        value = values[key].get()
        if step >= 1:
            values[key].set(int(round(value)))
        else:
            values[key].set(round(float(value) / step) * step)

    for index, (label, key, low, high, step, unit) in enumerate(slider_specs, start=1):
        row = index * 2 - 1
        ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", pady=(5, 0))
        value_labels[key] = ttk.Label(left, width=9, anchor="e")
        value_labels[key].grid(row=row, column=1, sticky="e", pady=(5, 0))
        scale = ttk.Scale(left, from_=low, to=high, variable=values[key], length=300, command=lambda _v, k=key, s=step: snap_value(k, s))
        scale.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(0, 3))

    ttk.Checkbutton(left, text="Add camera on grabber", variable=values["add_camera"]).grid(row=20, column=0, columnspan=2, sticky="w", pady=(12, 4))
    ttk.Checkbutton(left, text="Save scanner images", variable=values["save_images"]).grid(row=21, column=0, columnspan=2, sticky="w", pady=4)
    ttk.Label(left, textvariable=summary, foreground="#355c7d", wraplength=330).grid(row=22, column=0, columnspan=2, sticky="w", pady=(14, 8))

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
        samples_x = max(1, int(values["samples_x"].get()))
        samples_y = max(1, int(values["samples_y"].get()))
        return width, length, square_width, square_length, rows_count, cols, samples_x, samples_y

    def draw_preview() -> None:
        canvas.delete("all")
        try:
            width, length, square_width, square_length, rows_count, cols, samples_x, samples_y = current_grid()
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
                    for sy in range(samples_y):
                        for sx in range(samples_x):
                            px = x0 + (sx + 0.5) * (x1 - x0) / samples_x
                            py = y0 + (sy + 0.5) * (y1 - y0) / samples_y
                            canvas.create_oval(px - 3, py - 3, px + 3, py + 3, fill="#111827", outline="#ffffff")
            canvas.create_rectangle(ox, oy, ox + width * scale, oy + length * scale, outline="#1f2937", width=3)
            stations = rows_count * cols * samples_x * samples_y
            summary.set(
                f"Full fabric: {width:.3f} x {length:.3f} m | "
                f"Grid: {rows_count} x {cols} | "
                f"Square: {square_width:.3f} x {square_length:.3f} m | "
                f"Scan stations: {stations} | Views: {stations * 5 + 2}"
            )
            for label, key, _low, _high, _step, unit in slider_specs:
                value_labels[key].configure(text=format_value(key, unit))
        except Exception:
            canvas.create_text(235, 235, text="Enter positive numbers to preview fabric", fill="#6b7280", font=("Segoe UI", 12))
            summary.set("Enter positive sizes to preview the grid.")
        root.after(200, draw_preview)

    def start() -> None:
        try:
            args.width = float(values["width"].get())
            args.length = float(values["length"].get())
            args.rows = int(values["rows"].get())
            args.cols = int(values["cols"].get())
            args.square_width = args.width / max(1, args.cols)
            args.square_length = args.length / max(1, args.rows)
            args.samples_x = int(values["samples_x"].get())
            args.samples_y = int(values["samples_y"].get())
            args.view_radius = float(values["view_radius"].get())
            args.angle_lift = float(values["angle_lift"].get())
            args.speed = float(values["speed"].get())
            args.add_camera = bool(values["add_camera"].get())
            args.save_images = bool(values["save_images"].get())
            if min(args.width, args.length, args.rows, args.cols, args.samples_x, args.samples_y) <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Check setup", "Use positive numeric values for fabric size, square size, and scan counts.")
            return
        result["args"] = args
        root.destroy()

    def cancel() -> None:
        result["args"] = None
        root.destroy()

    buttons = ttk.Frame(shell, padding=(16, 10))
    buttons.pack(fill="x", side="bottom")
    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right", padx=(8, 0))
    ttk.Button(buttons, text="Next", command=start).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", cancel)
    draw_preview()
    root.mainloop()
    return result["args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo UR5e fabric-grid scanner with executed path trail")
    parser.add_argument("--rows", type=int, default=4, help="Number of fabric swatch rows")
    parser.add_argument("--cols", type=int, default=5, help="Number of fabric swatch columns")
    parser.add_argument("--samples-x", type=int, default=2, help="Scan locations per square along X")
    parser.add_argument("--samples-y", type=int, default=2, help="Scan locations per square along Y")
    parser.add_argument("--width", type=float, default=0.34, help="Fabric width in local metres")
    parser.add_argument("--length", type=float, default=0.24, help="Fabric length in local metres")
    parser.add_argument("--edge-margin", type=float, default=0.004, help="Outer fabric margin before sampling")
    parser.add_argument("--square-margin", type=float, default=0.006, help="Margin inside each colored square")
    parser.add_argument("--surface-wave", type=float, default=0.003, help="Small Z wave to mimic fabric")
    parser.add_argument("--view-radius", type=float, default=0.018, help="XY offset for angled views")
    parser.add_argument("--angle-lift", type=float, default=0.014, help="Z lift for angled views")
    parser.add_argument("--approach-lift", type=float, default=0.040, help="Lift for approach/retreat points")
    parser.add_argument("--center", nargs=3, type=float, default=DEFAULT_CENTER.tolist(), metavar=("X", "Y", "Z"))
    parser.add_argument("--max-span", nargs=3, type=float, default=DEFAULT_MAX_SPAN.tolist(), metavar=("X", "Y", "Z"))
    parser.add_argument("--speed", type=float, default=0.35, help="Viewer animation speed multiplier")
    parser.add_argument("--dwell", type=float, default=0.03, help="Pause at each scan view in seconds")
    parser.add_argument("--no-setup-gui", action="store_true", help="Skip the setup window and use CLI/default values")
    parser.add_argument("--no-run-gui", action="store_true", help="Skip the live speed/camera control window")
    parser.add_argument("--add-camera", action="store_true", help="Show a camera marker mounted at the grabber/TCP")
    parser.add_argument("--save-images", action="store_true", help="Save simulated scanner camera images")
    parser.add_argument("--image-dir", default="scanner_images", help="Directory for saved camera images")
    parser.add_argument("--image-every", choices=[CAMERA_SAVE_EVERY_STATION, CAMERA_SAVE_EVERY_VIEW], default=CAMERA_SAVE_EVERY_STATION, help="Save one image per station or every view")
    parser.add_argument("--export-script", action="store_true", help="Also export a URScript path")
    parser.add_argument("--save-points", action="store_true", help="Save mapped scan points as CSV")
    parser.add_argument("--output", default="mujoco_fabric_scanner.script", help="URScript output filename")
    parser.add_argument("--csv-output", default="mujoco_fabric_scanner_points.csv", help="CSV output filename")
    return parser.parse_args()


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

    print("=" * 64)
    print("  UR5e MuJoCo Fabric Scanner")
    print("=" * 64)
    print(f"fabric grid: {args.rows} x {args.cols}")
    print(f"fabric size: {args.width:.3f} x {args.length:.3f} m")
    print(f"square size: {plan.square_width:.3f} x {plan.square_length:.3f} m")
    print(f"scan stations: {len(plan.mapped_stations)}")
    print(f"scan views: {len(plan.mapped_points)}")
    if args.add_camera:
        print(f"camera: enabled | save images: {args.save_images} | output: {args.image_dir}")

    safe, issues = assess_plan_safety(plan.mapped_points, max_step=0.08)
    print(f"safety: {'PASS' if safe else 'CHECK'}")
    for issue in issues:
        print(f"  - {issue}")

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

    run_simulation(plan, args)


if __name__ == "__main__":
    main()

