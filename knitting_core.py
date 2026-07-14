import ctypes
import glob
import os
import sys
import json
import numpy as np
from scipy.interpolate import CubicSpline

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _preload_linux_nvidia_libs():
    if not sys.platform.startswith("linux"):
        return
    for p in sys.path:
        if not p.endswith("site-packages"):
            continue
        root = os.path.join(p, "nvidia")
        if not os.path.isdir(root):
            continue
        for so in sorted(glob.glob(os.path.join(root, "**/*.so*"), recursive=True)):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
        return


_preload_linux_nvidia_libs()

def compute_bitmap_scale_factors(bitmap):
    """Return active spans; zero bitmap cells remain inactive."""
    bitmap_array = np.asarray(bitmap, dtype=np.float32) > 0.5
    n_rows, n_cols = bitmap_array.shape
    scale_factors = np.zeros((n_rows, n_cols), dtype=np.float32)

    for col_idx in range(n_cols):
        active_rows = np.flatnonzero(bitmap_array[:, col_idx])
        for active_index, row_idx in enumerate(active_rows):
            next_row = active_rows[active_index + 1] if active_index + 1 < len(active_rows) else n_rows
            scale_factors[row_idx, col_idx] = float(next_row - row_idx)

    return scale_factors


def _height_grid_from_params(params, bitmap, lh_idx):
    params = np.asarray(params, dtype=np.float32)
    bitmap = np.asarray(bitmap, dtype=np.float32)
    row_heights = np.asarray(params[np.array(lh_idx, dtype=np.int32)], dtype=np.float32)
    if row_heights.size == 0:
        return np.zeros_like(bitmap, dtype=np.float32)
    grid = np.zeros_like(bitmap, dtype=np.float32)
    for row_idx in range(bitmap.shape[0]):
        grid[row_idx, :] = float(row_heights[min(row_idx, row_heights.size - 1)])
    return grid


def build_parametric_control_rows(params, bitmap, pidx, lh_idx, spl=5, loop_heights=None):
    """Build stitch control rows with per-bitmap-cell loop heights."""
    p = np.asarray(params, dtype=np.float32)
    bitmap_array = np.asarray(bitmap, dtype=np.float32)
    stitch_bulge = float(p[pidx["stitch_bulge"]])
    stitch_z = float(p[pidx["stitch_z"]])
    dy = float(p[pidx["dy"]])
    if loop_heights is None:
        height_grid = _height_grid_from_params(p, bitmap_array, lh_idx)
    else:
        height_grid = np.asarray(loop_heights, dtype=np.float32)
        if height_grid.shape != bitmap_array.shape:
            fallback = _height_grid_from_params(p, bitmap_array, lh_idx)
            fixed = fallback.copy()
            h_rows = min(fixed.shape[0], height_grid.shape[0])
            h_cols = min(fixed.shape[1], height_grid.shape[1])
            fixed[:h_rows, :h_cols] = height_grid[:h_rows, :h_cols]
            height_grid = fixed
    scale_factors = compute_bitmap_scale_factors(bitmap_array)
    n_rows, n_cols = scale_factors.shape
    base_t_values = np.linspace(0.0, 2.0 * np.pi, int(spl), endpoint=False, dtype=np.float32)
    rows = []

    for row_idx in range(n_rows):
        row_points = []
        col_indices = range(n_cols) if row_idx % 2 == 0 else range(n_cols - 1, -1, -1)
        t_values = base_t_values if row_idx % 2 == 0 else base_t_values[::-1]

        for col_idx in col_indices:
            has_loop = 1.0 if scale_factors[row_idx, col_idx] > 0.0 else 0.0
            loop_height = float(height_grid[row_idx, col_idx]) if has_loop else 0.0
            for t in t_values:
                x = col_idx + (stitch_bulge * np.sin(2.0 * t) if has_loop else 0.0) + t / (2.0 * np.pi)
                y = row_idx * dy - loop_height * (np.cos(t) - 1.0) / 2.0
                z = has_loop * stitch_z * (np.cos(2.0 * t) - 1.0) / 2.0
                row_points.append([x, y, z])

        row_points.append([float(n_cols if row_idx % 2 == 0 else 0.0), row_idx * dy, 0.0])
        rows.append(np.array(row_points, dtype=float))

    return rows

def compute_knitting_faces(seg, vl):
    if not vl:
        return []
    n = vl[0][1]
    i, j = np.meshgrid(np.arange(n - 1), np.arange(seg), indexing="ij")
    faces = np.stack((
        i * seg + j,
        i * seg + (j + 1) % seg,
        (i + 1) * seg + (j + 1) % seg,
        (i + 1) * seg + j
    ), axis=-1).reshape(-1, 4)
    return [faces] * len(vl)


def build_surface_fiber_meshes(
    base_vl,
    segments,
    enabled,
    count,
    radius,
    radius_scale,
    lift,
    surface_arc,
    randomness,
    twist,
):
    if not enabled:
        return list(base_vl), [{"row": row_idx} for row_idx in range(len(base_vl))]

    fiber_radius = max(radius * radius_scale, 1e-5)
    lift_val = max(lift, 0.0)
    surface_arc_val = float(np.clip(surface_arc, 0.05, 1.0))
    randomness_val = float(np.clip(randomness, 0.0, 1.0))

    out_vl = []
    meta = []

    for row_idx, (verts, n_points) in enumerate(base_vl):
        verts = np.asarray(verts, dtype=np.float32)
        n_points = int(n_points)
        if n_points < 2 or len(verts) != n_points * segments:
            continue

        rings = verts.reshape(n_points, segments, 3)
        centers = rings.mean(axis=1)
        top_idx = int(np.argmax((rings - centers[:, None, :])[:, :, 2].mean(axis=0)))
        offsets = (
            np.zeros(1, dtype=np.float32)
            if count == 1
            else np.linspace(-0.5, 0.5, count, dtype=np.float32) * surface_arc_val * float(segments)
        )

        for fiber_idx, offset in enumerate(offsets):
            rng = np.random.default_rng(row_idx * 1009 + fiber_idx * 9173)
            phase_jitter = rng.normal(0.0, 0.35 * randomness_val)
            lift_jitter = rng.normal(0.0, 0.20 * randomness_val)
            radius_jitter = float(np.clip(1.0 + rng.normal(0.0, 0.18 * randomness_val), 0.55, 1.45))
            local_radius = max(fiber_radius * radius_jitter, 1e-5)

            sample_idx = np.mod(
                top_idx + offset + phase_jitter + twist * np.linspace(0.0, 1.0, n_points, dtype=np.float32) * segments,
                float(segments),
            )
            lo_float = np.floor(sample_idx)
            lo = lo_float.astype(np.int32) % segments
            hi = (lo + 1) % segments
            frac = (sample_idx - lo_float).astype(np.float32)
            surface = rings[np.arange(n_points), lo] * (1.0 - frac[:, None]) + rings[np.arange(n_points), hi] * frac[:, None]
            radial = surface - centers
            surface_radius = np.linalg.norm(radial, axis=1, keepdims=True)
            radial /= surface_radius + 1e-8
            center_radius = np.maximum(surface_radius - local_radius + local_radius * (lift_val + lift_jitter), local_radius)
            line = centers + radial * center_radius

            tangent = np.gradient(line, axis=0)
            tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8
            side = np.cross(tangent, radial)
            bad = np.linalg.norm(side, axis=1) < 1e-6
            if np.any(bad):
                side[bad] = np.cross(tangent[bad], [1.0, 0.0, 0.0])
            side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-8
            normal = np.cross(side, tangent)
            normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8

            angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=np.float32)
            offsets_ring = (
                normal[:, None, :] * np.cos(angles)[None, :, None]
                + side[:, None, :] * np.sin(angles)[None, :, None]
            ) * local_radius
            out_vl.append(((line[:, None, :] + offsets_ring).reshape(-1, 3).astype(np.float32), n_points))
            meta.append({'row': row_idx})

    return out_vl, meta



def build_spline_mesh(
    ctrl_rows,
    params,
    config,
    pidx,
    period_offset,
    radius_ctrl_rows=None,
):
    p = np.asarray(params)
    rad, rat = p[pidx["radius"]], p[pidx["ellipse_ratio"]]
    seg, res = config["knit_parameters"]["segments"], config["knit_parameters"]["loop_res"]
    
    if isinstance(period_offset, (int, float, np.integer, np.floating)):
        D = np.array([float(period_offset), 0.0, 0.0], dtype=float)
        bitmap_width = float(period_offset)
    else:
        D = np.asarray(period_offset, dtype=float)
        bitmap_width = float(np.linalg.norm(D))
        
    nout = max(3, res * int(round(bitmap_width)) + 1)
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ca, sa = np.cos(a)[None, :, None], np.sin(a)[None, :, None]
    out = []
    for row_idx, r in enumerate(ctrl_rows):
        cp = np.asarray(r, dtype=float)
        if len(cp) == 0:
            continue
        if len(cp) <= 1:
            pts = np.repeat(cp, nout, axis=0)
            ctrl_sample_idx = np.zeros(nout, dtype=float)
        else:
            cp_aug = np.concatenate((cp, (cp[0] + D)[None, :]), axis=0)
            t = np.concatenate(([0.0], np.cumsum(np.maximum(np.linalg.norm(np.diff(cp_aug, axis=0), axis=1), 1e-6))))
            to = np.linspace(t[0], t[-1], nout)
            cp_detrended = cp_aug - D[None, :] * (t / t[-1])[:, None]
            if len(cp) == 2:
                pts_detrended = np.column_stack([np.interp(to, t, cp_detrended[:, i]) for i in range(3)])
            else:
                pts_detrended = np.column_stack([CubicSpline(t, cp_detrended[:, i], bc_type="periodic")(to) for i in range(3)])
            pts = pts_detrended + D[None, :] * (to / t[-1])[:, None]
            ctrl_sample_idx = np.linspace(0.0, len(cp), nout, dtype=float)

        if radius_ctrl_rows is not None and row_idx < len(radius_ctrl_rows):
            radius_cp = np.asarray(radius_ctrl_rows[row_idx], dtype=float)
            if radius_cp.shape[0] == len(cp):
                if len(cp) <= 1:
                    radius_line = np.full(nout, float(radius_cp[0]) if len(radius_cp) else float(rad), dtype=float)
                else:
                    radius_cp_aug = np.append(radius_cp, radius_cp[0])
                    radius_line = np.interp(ctrl_sample_idx, np.arange(len(radius_cp_aug), dtype=float), radius_cp_aug)
            else:
                radius_line = np.full(nout, float(rad), dtype=float)
        else:
            radius_line = np.full(nout, float(rad), dtype=float)
        radius_line = np.maximum(radius_line, 1e-6)


        if len(cp) <= 1:
            T = np.gradient(pts, axis=0)
        else:
            T = np.zeros_like(pts)
            T[1:-1] = (pts[2:] - pts[:-2]) / 2.0
            T[0] = (pts[1] - (pts[-2] - D)) / 2.0
            T[-1] = ((pts[1] + D) - pts[-2]) / 2.0
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-8
        U = np.cross(T, [0, 0, 1])
        b = np.linalg.norm(U, axis=1) < 1e-6
        U[b] = np.cross(T[b], [1, 0, 0])
        U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
        V = np.cross(T, U)
        rline = radius_line[:, None, None]
        out.append(((pts[:, None, :] + U[:, None, :] * ca * rline * rat + V[:, None, :] * sa * rline).reshape(-1, 3), nout))
    return out
