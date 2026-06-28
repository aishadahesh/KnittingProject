"""
Robot_fabric_scanner.py
=======================

Demo path planner for a UR5 fabric-scanning motion.

The script builds a serpentine scan path over a rectangular fabric area, maps it
into a conservative UR5 workspace, and opens an animated GUI preview. It can
also write/send a Cartesian URScript program for URSim. URSim/PolyScope solves
IK for the generated movel poses.

Typical demo:
    python Robot_fabric_scanner.py

Send to URSim after checking the preview/safety report:
    python Robot_fabric_scanner.py --export-script --send --ip 192.168.86.129 --port 30001
"""

from __future__ import annotations

import argparse
import math
import socket
import time
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_CENTER = np.array([0.45, 0.00, 0.30], dtype=float)
DEFAULT_MAX_SPAN = np.array([0.30, 0.22, 0.04], dtype=float)
TOOL_DOWN_ROTVEC = np.array([math.pi, 0.0, 0.0], dtype=float)
UR5_REACH_MAX = 0.850
UR5_REACH_MIN = 0.180
UR5_MIN_XY_RADIUS = 0.150
UR5_D1 = 0.089159
UR5_UPPER_ARM = 0.42500
UR5_FOREARM = 0.39225
UR5_WRIST_TOOL = 0.12


def build_fabric_scan_stations(
    width: float,
    length: float,
    line_spacing: float,
    points_per_line: int,
    edge_margin: float,
    surface_wave: float,
) -> np.ndarray:
    """Create raw local scan stations in metres.

    The stations are centred around the local origin and ordered in a serpentine
    sequence. Z includes a small sinusoidal wave so the preview resembles a
    cloth surface instead of a perfectly flat plate.
    """
    if width <= 0 or length <= 0:
        raise ValueError("width and length must be positive")
    if line_spacing <= 0:
        raise ValueError("line_spacing must be positive")
    if edge_margin < 0:
        raise ValueError("edge_margin cannot be negative")

    scan_width = max(width - 2.0 * edge_margin, line_spacing)
    scan_length = max(length - 2.0 * edge_margin, line_spacing)

    y_values = np.arange(-scan_length / 2.0, scan_length / 2.0 + 1e-9, line_spacing)
    if len(y_values) < 2:
        y_values = np.array([-scan_length / 2.0, scan_length / 2.0])

    x_forward = np.linspace(-scan_width / 2.0, scan_width / 2.0, points_per_line)
    points: list[list[float]] = []

    for row, y in enumerate(y_values):
        xs = x_forward if row % 2 == 0 else x_forward[::-1]
        for x in xs:
            z = surface_wave * math.sin(2.0 * math.pi * (x / scan_width + 0.5))
            z += 0.5 * surface_wave * math.cos(2.0 * math.pi * (y / scan_length + 0.5))
            points.append([float(x), float(y), float(z)])

    return np.asarray(points, dtype=float)


def expand_stations_to_viewpoints(
    stations: np.ndarray,
    view_radius: float,
    angle_lift: float,
) -> tuple[np.ndarray, list[int], list[str], np.ndarray]:
    """Expand each fabric station into several scanner viewpoints.

    Each station gets a straight-down stop plus four angled stops. The angled
    stops move slightly around the station and lift the TCP, imitating a camera
    or scanner inspecting the same fabric patch from multiple directions.
    """
    views = [
        ("top", np.array([0.0, 0.0, 0.0]), np.array([math.pi, 0.0, 0.0])),
        ("left angle", np.array([-view_radius, 0.0, angle_lift]), np.array([math.pi, 0.0, -0.35])),
        ("right angle", np.array([view_radius, 0.0, angle_lift]), np.array([math.pi, 0.0, 0.35])),
        ("front angle", np.array([0.0, view_radius, angle_lift]), np.array([math.pi, 0.35, 0.0])),
        ("back angle", np.array([0.0, -view_radius, angle_lift]), np.array([math.pi, -0.35, 0.0])),
    ]
    points: list[np.ndarray] = []
    station_ids: list[int] = []
    view_names: list[str] = []
    rotvecs: list[np.ndarray] = []
    for station_id, station in enumerate(stations):
        for view_name, offset, rotvec in views:
            points.append(station + offset)
            station_ids.append(station_id)
            view_names.append(view_name)
            rotvecs.append(rotvec)
    return np.asarray(points), station_ids, view_names, np.asarray(rotvecs)


def add_approach_and_retreat(path: np.ndarray, lift: float) -> np.ndarray:
    """Add short vertical approach/retreat points for simulator readability."""
    if lift <= 0:
        return path
    start = path[0].copy()
    end = path[-1].copy()
    start[2] += lift
    end[2] += lift
    return np.vstack([start, path, end])


def fit_path_to_workspace(points: np.ndarray, center: np.ndarray, max_span: np.ndarray) -> np.ndarray:
    """Scale and translate local path points into the UR5 base frame."""
    points = np.asarray(points, dtype=float)
    span = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-9)
    scale = float(np.min(max_span / span))
    return (points - points.mean(axis=0)) * scale + center


def points_to_poses(points: np.ndarray, rotvec: np.ndarray = TOOL_DOWN_ROTVEC) -> np.ndarray:
    """Convert XYZ points to URScript p[x,y,z,rx,ry,rz] poses."""
    rot = np.repeat(rotvec.reshape(1, 3), len(points), axis=0)
    return np.hstack([points, rot])


def points_and_rotvecs_to_poses(points: np.ndarray, rotvecs: np.ndarray) -> np.ndarray:
    """Convert XYZ points plus per-stop orientations to URScript poses."""
    return np.hstack([points, rotvecs])


def save_points_csv(path: Path, points: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        import csv

        writer = csv.writer(fh)
        writer.writerow(["x", "y", "z"])
        writer.writerows(points.tolist())


def _svg_polyline(points: np.ndarray, width: int, height: int, pad: int) -> str:
    xy = points[:, :2].astype(float)
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    span = np.maximum(mx - mn, 1e-9)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])
    px = pad + (xy[:, 0] - mn[0]) * scale
    py = height - pad - (xy[:, 1] - mn[1]) * scale
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(px, py))


def save_preview_svg(preview_path: Path, raw_pts: np.ndarray, mapped_pts: np.ndarray) -> None:
    """Save a dependency-free top-down SVG preview."""
    width, height, pad = 980, 560, 46
    raw_line = _svg_polyline(raw_pts, width // 2 - 20, height - 80, pad)
    mapped_line = _svg_polyline(mapped_pts, width // 2 - 20, height - 80, pad)
    mapped_shifted = " ".join(
        f"{float(x.split(',')[0]) + width / 2:.1f},{x.split(',')[1]}" for x in mapped_line.split()
    )
    text = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f7f7f2"/>
  <text x="46" y="32" font-family="Arial" font-size="20" font-weight="700">Local fabric raster</text>
  <text x="{width // 2 + 46}" y="32" font-family="Arial" font-size="20" font-weight="700">Mapped UR5 base XY path</text>
  <rect x="30" y="52" width="{width // 2 - 60}" height="{height - 88}" fill="white" stroke="#d0d0c8"/>
  <rect x="{width // 2 + 30}" y="52" width="{width // 2 - 60}" height="{height - 88}" fill="white" stroke="#d0d0c8"/>
  <polyline points="{raw_line}" fill="none" stroke="#2563eb" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>
  <polyline points="{mapped_shifted}" fill="none" stroke="#e26d2f" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>
  <circle cx="{raw_line.split()[0].split(',')[0]}" cy="{raw_line.split()[0].split(',')[1]}" r="5" fill="#22863a"/>
  <circle cx="{raw_line.split()[-1].split(',')[0]}" cy="{raw_line.split()[-1].split(',')[1]}" r="5" fill="#d73a49"/>
  <circle cx="{mapped_shifted.split()[0].split(',')[0]}" cy="{mapped_shifted.split()[0].split(',')[1]}" r="5" fill="#22863a"/>
  <circle cx="{mapped_shifted.split()[-1].split(',')[0]}" cy="{mapped_shifted.split()[-1].split(',')[1]}" r="5" fill="#d73a49"/>
  <text x="46" y="{height - 24}" font-family="Arial" font-size="13" fill="#555">green=start, red=end, serpentine scan rows</text>
  <text x="{width // 2 + 46}" y="{height - 24}" font-family="Arial" font-size="13" fill="#555">coordinates are metres in UR base frame after workspace fitting</text>
</svg>
"""
    preview_path.write_text(text, encoding="utf-8")


def generate_urscript(
    poses: np.ndarray,
    accel: float,
    vel: float,
    blend_r: float,
    dwell: float,
    station_ids: list[int] | None = None,
    view_names: list[str] | None = None,
    prog_name: str = "fabric_scanner_demo",
) -> str:
    lines = [
        "# URScript - generated by Robot_fabric_scanner.py",
        "# Cartesian fabric scan demo for UR5 / URSim",
        f"# Waypoints : {len(poses)}",
        f"# Accel     : {accel}  Vel: {vel}  Blend: {blend_r}",
        "",
        f"def {prog_name}():",
        "  set_tcp(p[0.000000, 0.000000, 0.120000, 0.000000, 0.000000, 0.000000])",
        "  # Move to the first scan pose slowly.",
        f"  movej(p[{', '.join(f'{v:.6f}' for v in poses[0])}], a=0.15, v=0.05, r=0)",
        "",
        "  # Fabric scan stops. URSim solves IK for each Cartesian pose.",
    ]
    for i, pose in enumerate(poses):
        if station_ids is not None and view_names is not None:
            lines.append(f"  # station {station_ids[i] + 1}: {view_names[i]}")
        pose_str = ", ".join(f"{v:.6f}" for v in pose)
        lines.append(f"  movel(p[{pose_str}], a={accel}, v={vel}, r={blend_r})")
        lines.append(f"  sleep({dwell})")
    lines += ["end", "", f"{prog_name}()"]
    return "\n".join(lines)


def assess_plan_safety(points: np.ndarray, max_step: float = 0.035) -> tuple[bool, list[str]]:
    issues: list[str] = []
    reach = np.linalg.norm(points, axis=1)
    xy_radius = np.linalg.norm(points[:, :2], axis=1)
    if float(np.max(reach)) > UR5_REACH_MAX:
        issues.append(f"max reach is {float(np.max(reach)):.3f} m, above UR5 reach {UR5_REACH_MAX:.3f} m")
    if float(np.min(reach)) < UR5_REACH_MIN:
        issues.append(f"min reach is {float(np.min(reach)):.3f} m, inside inner reach {UR5_REACH_MIN:.3f} m")
    if float(np.min(xy_radius)) < UR5_MIN_XY_RADIUS:
        issues.append(f"min XY radius is {float(np.min(xy_radius)):.3f} m, too close to base column")
    if float(np.min(points[:, 2])) < 0.05:
        issues.append(f"min Z is {float(np.min(points[:, 2])):.3f} m, too close to the table/base")
    if len(points) > 1:
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if float(np.max(steps)) > max_step:
            issues.append(f"max Cartesian step is {float(np.max(steps)):.3f} m, above limit {max_step:.3f} m")
    return len(issues) == 0, issues


def send_to_ursim(script: str, ip: str, port: int, timeout: float = 10.0) -> None:
    if not script.endswith("\n"):
        script += "\n"
    with socket.create_connection((ip, port), timeout=timeout) as sock:
        sock.sendall(script.encode("utf-8"))
        time.sleep(0.2)


class FabricScannerGui:
    """Small Tkinter GUI for visualizing the scan before URSim."""

    def __init__(
        self,
        raw_pts: np.ndarray,
        mapped_pts: np.ndarray,
        raw_stations: np.ndarray,
        mapped_stations: np.ndarray,
        station_ids: list[int],
        view_names: list[str],
        safe: bool,
        issues: list[str],
    ):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.raw_pts = raw_pts
        self.mapped_pts = mapped_pts
        self.raw_stations = raw_stations
        self.mapped_stations = mapped_stations
        self.station_ids = station_ids
        self.view_names = view_names
        self.safe = safe
        self.issues = issues
        self.index = 0
        self.running = False
        self.after_id = None

        self.root = tk.Tk()
        self.root.title("UR5 Fabric Scanner Path Demo")
        self.root.geometry("1120x760")
        self.root.minsize(980, 660)

        self.canvas = tk.Canvas(self.root, bg="#f7f7f2", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        controls = ttk.Frame(self.root, padding=(12, 8))
        controls.pack(fill="x")

        self.play_btn = ttk.Button(controls, text="Play", command=self.toggle)
        self.play_btn.pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Reset", command=self.reset).pack(side="left", padx=(0, 16))

        ttk.Label(controls, text="Speed").pack(side="left")
        self.speed = tk.DoubleVar(value=1.0)
        ttk.Scale(controls, from_=0.25, to=4.0, variable=self.speed, length=180).pack(side="left", padx=8)

        self.status_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=18)

        self.canvas.bind("<Configure>", lambda _event: self.draw())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def run(self) -> None:
        self.draw()
        self.root.mainloop()

    def close(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        self.root.destroy()

    def toggle(self) -> None:
        self.running = not self.running
        self.play_btn.configure(text="Pause" if self.running else "Play")
        if self.running:
            self.animate()

    def reset(self) -> None:
        self.running = False
        self.play_btn.configure(text="Play")
        self.index = 0
        self.draw()

    def animate(self) -> None:
        if not self.running:
            return
        self.index = (self.index + 1) % len(self.mapped_pts)
        self.draw()
        delay = max(35, int(360 / float(self.speed.get())))
        self.after_id = self.root.after(delay, self.animate)

    def _projector(self, pts: np.ndarray, box: tuple[int, int, int, int]):
        x0, y0, x1, y1 = box
        xy = pts[:, :2]
        mn = xy.min(axis=0)
        mx = xy.max(axis=0)
        span = np.maximum(mx - mn, 1e-9)
        pad = 34
        scale = min((x1 - x0 - 2 * pad) / span[0], (y1 - y0 - 2 * pad) / span[1])

        def project(point: np.ndarray) -> tuple[float, float]:
            px = x0 + pad + (point[0] - mn[0]) * scale
            py = y1 - pad - (point[1] - mn[1]) * scale
            return float(px), float(py)

        return project

    def _side_projector(self, pts: np.ndarray, box: tuple[int, int, int, int]):
        x0, y0, x1, y1 = box
        radii = np.linalg.norm(pts[:, :2], axis=1)
        rz = np.column_stack([radii, pts[:, 2]])
        rz = np.vstack([rz, [0.0, 0.0], [0.0, UR5_D1], [UR5_REACH_MAX, 0.0], [UR5_REACH_MAX, 0.55]])
        mn = rz.min(axis=0)
        mx = rz.max(axis=0)
        span = np.maximum(mx - mn, 1e-9)
        pad = 46
        scale = min((x1 - x0 - 2 * pad) / span[0], (y1 - y0 - 2 * pad) / span[1])

        def project(point: np.ndarray) -> tuple[float, float]:
            r = float(np.linalg.norm(point[:2]))
            z = float(point[2])
            px = x0 + pad + (r - mn[0]) * scale
            py = y1 - pad - (z - mn[1]) * scale
            return float(px), float(py)

        return project

    def _draw_polyline(self, pts: np.ndarray, project, color: str, width: int, upto: int | None = None) -> None:
        if upto is not None:
            pts = pts[: max(2, upto + 1)]
        coords: list[float] = []
        for point in pts:
            coords.extend(project(point))
        if len(coords) >= 4:
            self.canvas.create_line(*coords, fill=color, width=width, smooth=False, capstyle=self.tk.ROUND)

    def _draw_station_dots(self, stations: np.ndarray, project, current_station: int) -> None:
        for station_index, station in enumerate(stations):
            x, y = project(station)
            fill = "#f97316" if station_index == current_station else "#ffffff"
            outline = "#c2410c" if station_index == current_station else "#8a94a6"
            r = 7 if station_index == current_station else 4
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline, width=2)

    def _draw_robot(self, tcp: np.ndarray, box: tuple[int, int, int, int]) -> None:
        project = self._side_projector(self.mapped_pts, box)
        tcp_r = float(np.linalg.norm(tcp[:2]))
        tcp_z = float(tcp[2])
        shoulder = np.array([0.0, 0.0, UR5_D1])
        wrist = np.array([tcp_r, 0.0, tcp_z + UR5_WRIST_TOOL])

        dx = float(wrist[0])
        dz = float(wrist[2] - UR5_D1)
        d = min(max(math.hypot(dx, dz), 0.08), UR5_UPPER_ARM + UR5_FOREARM - 1e-3)
        theta = math.atan2(dz, dx)
        cos_phi = (UR5_UPPER_ARM**2 + d**2 - UR5_FOREARM**2) / (2 * UR5_UPPER_ARM * d)
        phi = math.acos(float(np.clip(cos_phi, -1.0, 1.0)))
        shoulder_angle = theta + phi
        elbow = np.array([
            UR5_UPPER_ARM * math.cos(shoulder_angle),
            0.0,
            UR5_D1 + UR5_UPPER_ARM * math.sin(shoulder_angle),
        ])
        tcp_side = np.array([tcp_r, 0.0, tcp_z])

        base_xy = project(np.array([0.0, 0.0, 0.0]))
        shoulder_xy = project(shoulder)
        elbow_xy = project(elbow)
        wrist_xy = project(wrist)
        tcp_xy = project(tcp_side)

        x0, y0, x1, y1 = box
        table_y = project(np.array([0.0, 0.0, float(np.min(self.mapped_stations[:, 2])) - 0.006]))[1]
        self.canvas.create_line(x0 + 36, table_y, x1 - 28, table_y, fill="#d8c7ad", width=3)

        self.canvas.create_rectangle(base_xy[0] - 24, base_xy[1] - 10, base_xy[0] + 24, base_xy[1] + 10, fill="#30343b", outline="")
        self.canvas.create_line(*base_xy, *shoulder_xy, fill="#30343b", width=18, capstyle=self.tk.ROUND)
        self.canvas.create_line(*shoulder_xy, *elbow_xy, fill="#215a93", width=15, capstyle=self.tk.ROUND)
        self.canvas.create_line(*elbow_xy, *wrist_xy, fill="#2b6cb0", width=13, capstyle=self.tk.ROUND)
        self.canvas.create_line(*wrist_xy, *tcp_xy, fill="#4a5568", width=8, capstyle=self.tk.ROUND)

        for label, xy, radius in [
            ("J1", base_xy, 11),
            ("J2", shoulder_xy, 12),
            ("J3", elbow_xy, 11),
            ("J4-J6", wrist_xy, 10),
        ]:
            x, y = xy
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="#ffffff", outline="#1a365d", width=3)
            self.canvas.create_text(x, y - radius - 12, text=label, font=("Segoe UI", 9), fill="#1a365d")

        self.canvas.create_oval(tcp_xy[0] - 7, tcp_xy[1] - 7, tcp_xy[0] + 7, tcp_xy[1] + 7, fill="#f97316", outline="#9a3412", width=2)
        self.canvas.create_polygon(
            tcp_xy[0] - 18,
            tcp_xy[1] + 30,
            tcp_xy[0] + 18,
            tcp_xy[1] + 30,
            tcp_xy[0],
            tcp_xy[1],
            fill="",
            outline="#f97316",
            width=2,
        )

    def draw(self) -> None:
        self.canvas.delete("all")
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        left = (28, 58, w // 2 - 18, h - 34)
        right = (w // 2 + 18, 58, w - 28, h - 34)

        self.canvas.create_text(28, 24, anchor="w", text="Fabric scanner path", font=("Segoe UI", 20, "bold"), fill="#1f2937")
        status = "Safety: PASS" if self.safe else "Safety: CHECK"
        status_color = "#1f7a3f" if self.safe else "#a15c00"
        self.canvas.create_text(w - 28, 27, anchor="e", text=status, font=("Segoe UI", 13, "bold"), fill=status_color)

        for box, title in [(left, "Top view: fabric scan stations"), (right, "UR5 side view: base, shoulder, elbow, wrist, TCP")]:
            x0, y0, x1, y1 = box
            self.canvas.create_rectangle(x0, y0, x1, y1, fill="#ffffff", outline="#d6d3c9", width=1)
            self.canvas.create_text(x0 + 16, y0 + 18, anchor="w", text=title, font=("Segoe UI", 13, "bold"), fill="#333333")

        raw_project = self._projector(self.raw_pts, left)
        mapped_project = self._side_projector(self.mapped_pts, right)
        current_station = self.station_ids[self.index]
        self._draw_polyline(self.raw_stations, raw_project, "#b9c3d6", 2)
        self._draw_station_dots(self.raw_stations, raw_project, current_station)
        self._draw_station_dots(self.mapped_stations, mapped_project, current_station)
        self._draw_polyline(self.raw_pts, raw_project, "#2563eb", 3, self.index)
        self._draw_polyline(self.mapped_pts, mapped_project, "#e26d2f", 3, self.index)

        raw_tcp = raw_project(self.raw_pts[min(self.index, len(self.raw_pts) - 1)])
        tcp = self.mapped_pts[self.index]
        mapped_tcp = mapped_project(tcp)
        for x, y, fill in [(raw_tcp[0], raw_tcp[1], "#2563eb"), (mapped_tcp[0], mapped_tcp[1], "#e26d2f")]:
            self.canvas.create_oval(x - 8, y - 8, x + 8, y + 8, fill=fill, outline="#ffffff", width=2)

        self._draw_robot(tcp, right)
        self.canvas.create_text(
            right[0] + 16,
            right[1] + 44,
            anchor="w",
            text=f"Station {current_station + 1} | {self.view_names[self.index]} | TCP p[{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}]",
            font=("Consolas", 11),
            fill="#444444",
        )
        msg = f"Stop {self.index + 1}/{len(self.mapped_pts)} | station {current_station + 1}/{len(self.mapped_stations)} | {self.view_names[self.index]}"
        if self.issues:
            msg += " | " + "; ".join(self.issues[:2])
        self.status_var.set(msg)


def print_safety_report(safe: bool, issues: Iterable[str]) -> None:
    print("\n[4] Safety gate")
    if safe:
        print("    PASS: path is within the configured limits")
        return
    print("    FAIL: not sending unless --send --allow-unsafe is used")
    for issue in issues:
        print(f"    - {issue}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a UR5 fabric-scanner demo path and URSim script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--width", type=float, default=0.34, help="Raw fabric width before workspace fitting (m)")
    parser.add_argument("--length", type=float, default=0.24, help="Raw fabric length before workspace fitting (m)")
    parser.add_argument("--line-spacing", type=float, default=0.045, help="Distance between scan rows (m)")
    parser.add_argument("--points-per-line", type=int, default=9, help="Scan stations on each row")
    parser.add_argument("--edge-margin", type=float, default=0.015, help="Margin left unscanned at fabric edge (m)")
    parser.add_argument("--surface-wave", type=float, default=0.004, help="Small local Z wave for cloth-like preview (m)")
    parser.add_argument("--approach-lift", type=float, default=0.035, help="Vertical lift before/after scan (m)")
    parser.add_argument("--view-radius", type=float, default=0.018, help="Local offset for angled views at each station (m)")
    parser.add_argument("--angle-lift", type=float, default=0.012, help="Local Z lift for angled views at each station (m)")
    parser.add_argument("--dwell", type=float, default=0.25, help="Stop time at each scan pose in generated URScript (s)")
    parser.add_argument("--n-points", type=int, default=0, help="Deprecated; stops are generated from stations and views")
    parser.add_argument("--tcp-z", type=float, default=0.12, help="Tool Z offset (m)")
    parser.add_argument("--center", nargs=3, type=float, default=DEFAULT_CENTER.tolist(), metavar=("X", "Y", "Z"))
    parser.add_argument("--max-span", nargs=3, type=float, default=DEFAULT_MAX_SPAN.tolist(), metavar=("X", "Y", "Z"))
    parser.add_argument("--accel", type=float, default=0.25)
    parser.add_argument("--vel", type=float, default=0.04)
    parser.add_argument("--blend-r", type=float, default=0.001)
    parser.add_argument("--output", default="fabric_scanner_demo.script", help="Generated URScript file")
    parser.add_argument("--csv-output", default="fabric_scanner_demo_points.csv", help="Generated raw path CSV")
    parser.add_argument("--preview-output", default="fabric_scanner_demo_preview.svg", help="Generated preview image")
    parser.add_argument("--ip", default="192.168.86.129", help="URSim IP address")
    parser.add_argument("--port", type=int, default=30001, help="URSim primary interface port")
    parser.add_argument("--send", action="store_true", help="Send script to URSim after planning")
    parser.add_argument("--export-script", action="store_true", help="Save the generated URScript file")
    parser.add_argument("--save-points", action="store_true", help="Save the raw path CSV")
    parser.add_argument("--save-preview", action="store_true", help="Save the SVG preview")
    parser.add_argument("--no-gui", action="store_true", help="Do not open the Tkinter GUI")
    parser.add_argument("--allow-unsafe", action="store_true", help="Send even if safety gate reports issues")
    parser.add_argument("--max-step", type=float, default=0.060, help="Max Cartesian distance between scan stops (m)")
    parser.add_argument("--dry-run", action="store_true", help="Alias for the default no-send behaviour")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent

    print("=" * 64)
    print("  UR5 Fabric Scanner Demo")
    print("=" * 64)

    print("\n[1] Building fabric scan stations")
    raw_stations = build_fabric_scan_stations(
        width=args.width,
        length=args.length,
        line_spacing=args.line_spacing,
        points_per_line=max(2, args.points_per_line),
        edge_margin=args.edge_margin,
        surface_wave=args.surface_wave,
    )
    raw_pts, station_ids, view_names, rotvecs = expand_stations_to_viewpoints(
        raw_stations,
        view_radius=args.view_radius,
        angle_lift=args.angle_lift,
    )
    raw_pts = add_approach_and_retreat(raw_pts, args.approach_lift)
    station_ids = [station_ids[0], *station_ids, station_ids[-1]]
    view_names = ["approach", *view_names, "retreat"]
    rotvecs = np.vstack([rotvecs[0], rotvecs, rotvecs[-1]])
    print(f"    Fabric stations: {len(raw_stations)}")
    print(f"    Scanner stops:   {len(raw_pts)}")
    if args.save_points:
        csv_path = project_dir / args.csv_output
        save_points_csv(csv_path, raw_pts)
        print(f"    Saved CSV:  {csv_path}")

    print(f"\n[2] Mapping stops into UR5 workspace")
    mapped_stations = fit_path_to_workspace(
        raw_stations,
        center=np.asarray(args.center, dtype=float),
        max_span=np.asarray(args.max_span, dtype=float),
    )
    mapped_pts = fit_path_to_workspace(
        raw_pts,
        center=np.asarray(args.center, dtype=float),
        max_span=np.asarray(args.max_span, dtype=float),
    )
    tcp_poses = points_and_rotvecs_to_poses(mapped_pts, rotvecs)

    print("\n[3] Preparing GUI/URSim demo data")
    if args.save_preview:
        preview_path = project_dir / args.preview_output
        save_preview_svg(preview_path, raw_pts, mapped_pts)
        print(f"    Preview:   {preview_path}")

    script = generate_urscript(
        tcp_poses,
        accel=args.accel,
        vel=args.vel,
        blend_r=0.0 if args.dwell > 0 else args.blend_r,
        dwell=args.dwell,
        station_ids=station_ids,
        view_names=view_names,
        prog_name="fabric_scanner_demo",
    )
    if args.export_script or args.send:
        script_path = project_dir / args.output
        script_path.write_text(script + "\n", encoding="utf-8")
        print(f"    URScript:  {script_path}")

    safe, issues = assess_plan_safety(
        mapped_pts,
        max_step=args.max_step,
    )
    print_safety_report(safe, issues)

    if args.send:
        if safe or args.allow_unsafe:
            print(f"\n[5] Sending to URSim at {args.ip}:{args.port}")
            send_to_ursim(script, ip=args.ip, port=args.port)
        else:
            print("\n[5] Skipping URSim send because the safety gate failed")
    else:
        print("\n[5] No send requested. Run with --export-script --send for URSim.")

    if not args.no_gui:
        print("\n[6] Opening GUI demo")
        FabricScannerGui(
            raw_pts,
            mapped_pts,
            raw_stations,
            mapped_stations,
            station_ids,
            view_names,
            safe,
            issues,
        ).run()

    print("\nDone.")


if __name__ == "__main__":
    main()
