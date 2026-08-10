# %% PRELOAD & IMPORTS ──────────────────────────────────────────────────────────────
import ctypes
import glob
import os
import sys
import jax
import jax.numpy as jnp
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

# %% PARAMETRIC GENERATION & GEOMETRY ─────────────────────────────────────────────────
@jax.jit
def _scale_factors_jax(bitmap):
    a = bitmap > 0.5
    rows = bitmap.shape[0]

    def step(nxt, i):
        m = a[i]
        s = jnp.where(m, nxt - i, 0)
        return jnp.where(m, i, nxt), s

    init = jnp.full((bitmap.shape[1],), rows, dtype=jnp.int32)
    _, rev = jax.lax.scan(step, init, jnp.arange(rows - 1, -1, -1, dtype=jnp.int32))
    return jnp.flip(rev.astype(jnp.float32), axis=0)


@jax.jit
def eval_curve(t, hl, lh, sb, sz):
    x = sb * jnp.sin(2 * t) + t / (2 * jnp.pi)
    y = lh * (-(jnp.cos(t) - 1) / 2)
    z = sz * (jnp.cos(2 * t) - 1) / 2 * hl
    x = jnp.where(hl == 0.0, t / (2 * jnp.pi), x)
    return jnp.stack((x, y, z), axis=-1)




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


def build_parametric_control_rows(params, bitmap, pidx, lh_idx, spl=5):
    p = np.asarray(params, dtype=np.float32)
    idx = pidx
    bulge, stz, dy = float(p[idx["stitch_bulge"]]), float(p[idx["stitch_z"]]), float(p[idx["dy"]])
    lut = np.concatenate((np.zeros(1), p[np.array(lh_idx)]))
    sf = np.asarray(_scale_factors_jax(jnp.asarray(bitmap))).astype(np.int32)
    rows, cols = sf.shape
    x_pitch = 1.0
    base_t = np.linspace(0.0, 2 * np.pi, spl, endpoint=False, dtype=np.float32)
    t = np.tile(base_t, cols)
    xoff = np.repeat(np.arange(cols, dtype=np.float32), spl)
    s = np.repeat(sf, spl, axis=1)
    has = (s > 0).astype(np.float32)
    h = lut[s]
    c = np.array(eval_curve(
        jnp.asarray(t[None, :]), jnp.asarray(has), jnp.asarray(h), bulge, stz
    ), dtype=np.float32, copy=True)
    c[:, :, 0] = (c[:, :, 0] + xoff[None, :]) * x_pitch
    c[:, :, 1] += np.arange(rows, dtype=np.float32)[:, None] * dy

    return [c[r].astype(float) for r in range(rows)]



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


def build_display_meshes_precise(
    verts_list, faces_list, meta, radius, bitmap_size, period_offset_x, period_offset_y, display_copies, segments, ctrl_rows
):
    if not verts_list:
        return [], [], []
    row_count = max(1, int(bitmap_size[0]))

    seg = int(segments)
    x_tiles = list(range(-int(display_copies[0]), int(display_copies[0]) + 1))
    y_tiles = list(range(-int(display_copies[1]), int(display_copies[1]) + 1))

    display_vl, display_fl, display_meta = [], [], []
    for y_tile in y_tiles:
        y_translation = y_tile * period_offset_y
        for part_idx, ((verts, n_points), _faces, part_meta) in enumerate(zip(verts_list, faces_list, meta)):
            rings = np.asarray(verts, dtype=np.float32).reshape(int(n_points), seg, 3)
            base_faces = compute_knitting_faces(seg, [(rings.reshape(-1, 3), int(n_points))])[0]
            
            stitched_rings = []
            stitched_faces = []
            for tile_i, x_tile in enumerate(x_tiles):
                translated = rings + x_tile * period_offset_x[None, None, :]
                stitched_rings.append(translated)
                
                tile_faces = base_faces + tile_i * int(n_points) * seg
                stitched_faces.append(tile_faces)
                
            stitched = np.concatenate(stitched_rings, axis=0) + y_translation[None, None, :]
            stitched_n_points = int(stitched.shape[0])
            display_vl.append((stitched.reshape(-1, 3), stitched_n_points))
            
            combined_faces = np.concatenate(stitched_faces, axis=0)
            display_fl.append(combined_faces)
            copied_meta = dict(part_meta)
            copied_meta['row'] = int(copied_meta.get('row', 0)) + y_tile * row_count
            copied_meta['base_row'] = int(part_meta.get('row', 0))
            copied_meta['tile_x'] = 0
            copied_meta['tile_y'] = y_tile
            copied_meta['stitched_x_copies'] = len(x_tiles)
            display_meta.append(copied_meta)
    return display_vl, display_fl, display_meta


# %% SPLINE CENTERLINE REPRESENTATION ──────────────────────────────────────────────────
def eval_centerline(cp, D, nout, t=None, to=None):
    cp = np.asarray(cp, dtype=float)
    if len(cp) <= 1:
        return np.repeat(cp, nout, axis=0)
    cp_aug = np.concatenate((cp, (cp[0] + D)[None, :]), axis=0)
    if t is None or to is None:
        t = np.concatenate(([0.0], np.cumsum(np.maximum(np.linalg.norm(np.diff(cp_aug, axis=0), axis=1), 1e-6))))
        to = np.linspace(t[0], t[-1], nout)
    cp_detrended = cp_aug - D[None, :] * (t / t[-1])[:, None]
    if len(cp) == 2:
        pts_detrended = np.column_stack([np.interp(to, t, cp_detrended[:, i]) for i in range(3)])
    else:
        pts_detrended = np.column_stack([CubicSpline(t, cp_detrended[:, i], bc_type="periodic")(to) for i in range(3)])
    return pts_detrended + D[None, :] * (to / t[-1])[:, None]


def evaluate_centerlines(ctrl_rows, period_offset_x, config):
    seg, res = config["knit_parameters"]["segments"], config["knit_parameters"]["loop_res"]
    
    D = np.asarray(period_offset_x, dtype=float)
    bitmap_width = float(np.linalg.norm(D))
        
    nout = res * int(round(bitmap_width)) + 1
    
    V_list = []
    for r in ctrl_rows:
        cp = np.asarray(r, dtype=float)
        pts = eval_centerline(cp, D, nout)
        V_list.append(pts)
        
    if not V_list:
        return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32), D, nout
        
    V = np.vstack(V_list)
    num_ctrl_rows = len(ctrl_rows)
    
    edges_list = []
    row_offset = 0
    for r in range(num_ctrl_rows):
        row_edges = np.array([[i, i+1] for i in range(nout - 1)], dtype=np.int32).reshape(-1, 2) + row_offset
        edges_list.append(row_edges)
        row_offset += nout
    edges = np.vstack(edges_list) if edges_list else np.empty((0, 2), dtype=np.int32)
    
    return V, edges, D, nout


def build_row_spline_jacobian(cp, D, nout):
    cp = np.asarray(cp, dtype=float)
    num_ctrl = len(cp)
    if num_ctrl <= 1:
        return np.ones((nout, 1))
    cp_aug = np.concatenate((cp, (cp[0] + D)[None, :]), axis=0)
    t = np.concatenate(([0.0], np.cumsum(np.maximum(np.linalg.norm(np.diff(cp_aug, axis=0), axis=1), 1e-6))))
    to = np.linspace(t[0], t[-1], nout)
    cols = []
    for k in range(num_ctrl):
        cp_dummy = np.zeros((num_ctrl, 3))
        cp_dummy[k, 0] = 1.0
        pts = eval_centerline(cp_dummy, np.zeros(3), nout, t=t, to=to)
        cols.append(pts[:, 0])
    return np.column_stack(cols)


def build_spline_mesh(ctrl_rows, params, config, pidx, period_offset_x, radius_ctrl_rows=None):
    p = np.asarray(params)
    rad, rat = p[pidx["radius"]], p[pidx["ellipse_ratio"]]
    seg, res = config["knit_parameters"]["segments"], config["knit_parameters"]["loop_res"]
    
    V, edges, D, nout = evaluate_centerlines(ctrl_rows, period_offset_x, config)
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ca, sa = np.cos(a)[None, :, None], np.sin(a)[None, :, None]
    out = []
    for row_idx, r in enumerate(ctrl_rows):
        cp = np.asarray(r, dtype=float)
        pts = V[row_idx * nout : (row_idx + 1) * nout]
        ctrl_sample_idx = np.linspace(0.0, len(cp), nout, dtype=float) if len(cp) > 1 else np.zeros(nout, dtype=float)

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
        V_vec = np.cross(T, U)
        rline = radius_line[:, None, None]
        out.append(((pts[:, None, :] + U[:, None, :] * ca * rline * rat + V_vec[:, None, :] * sa * rline).reshape(-1, 3), nout))
    return out
