import ctypes
import glob
import os
import sys
import json
from functools import partial
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


@jax.jit
def eval_curve_derivative(t, hl, lh, sb, sz):
    dx = 2 * sb * jnp.cos(2 * t) + 1 / (2 * jnp.pi)
    dy = 0.5 * jnp.sin(t) * lh
    dz = -sz * jnp.sin(2 * t) * hl
    dx = jnp.where(hl == 0.0, 1 / (2 * jnp.pi), dx)
    return jnp.stack((dx, dy, dz), axis=-1)


@jax.jit
def compute_orthonormal_frame(tan):
    t = tan / (jnp.linalg.norm(tan, axis=-1, keepdims=True) + 1e-8)
    u = jnp.cross(t, jnp.array([0.0, 0.0, 1.0]))
    u = jnp.where(jnp.linalg.norm(u, axis=-1, keepdims=True) < 1e-6, jnp.cross(t, jnp.array([1.0, 0.0, 0.0])), u)
    u = u / (jnp.linalg.norm(u, axis=-1, keepdims=True) + 1e-8)
    return u, jnp.cross(t, u)


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def compute_knitting_vertices_jit(params, bitmap, loop_res, seg, indices, lh_idx):
    bulge_idx, stz_idx, dy_idx, rad_idx, rat_idx = indices
    bulge, stz = params[bulge_idx], params[stz_idx]
    dy, rad, rat = params[dy_idx], params[rad_idx], params[rat_idx]
    x_pitch = jnp.maximum(1.0, 2.8 * rad)
    row_heights = params[jnp.array(lh_idx)]

    rows, loops = bitmap.shape
    t = jnp.linspace(0.0, 2 * jnp.pi * loops, loop_res * loops + 1)
    a = jnp.linspace(0, 2 * jnp.pi, seg, endpoint=False)
    ca, sa = jnp.cos(a)[None, :, None], jnp.sin(a)[None, :, None]

    def row_fn(i, bitmap_row):
        row_height = jnp.take(row_heights, jnp.minimum(i, len(lh_idx) - 1))
        active = bitmap_row > 0.5
        has = jnp.append(jnp.repeat(active, loop_res), active[-1]).astype(jnp.float32)
        h = has * row_height
        p = eval_curve(t, has, h, bulge, stz).at[:, 1].add(i * dy)
        d = eval_curve_derivative(t, has, h, bulge, stz)
        p = p.at[:, 0].multiply(x_pitch)
        d = d.at[:, 0].multiply(x_pitch)
        u, v = compute_orthonormal_frame(d)
        off = u[:, None, :] * ca * rad * rat + v[:, None, :] * sa * rad
        return (p[:, None, :] + off).reshape(-1, 3)

    return jax.vmap(row_fn)(jnp.arange(rows), bitmap)


def compute_knitting_vertices(params, bitmap, config, pidx, lh_idx):
    seg = config["knit_parameters"]["segments"]
    res = config["knit_parameters"]["loop_res"]
    indices = (pidx["stitch_bulge"], pidx["stitch_z"], pidx["dy"], pidx["radius"], pidx["ellipse_ratio"])
    v = compute_knitting_vertices_jit(
        jnp.asarray(params, dtype=jnp.float32),
        jnp.asarray(bitmap, dtype=jnp.float32),
        res,
        seg,
        indices,
        lh_idx
    )
    return [(np.asarray(r), len(r) // seg) for r in v]


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
    x_pitch = max(1.0, 2.8 * float(p[idx["radius"]]))
    row_heights = p[np.array(lh_idx)]
    rows, cols = bitmap.shape
    base_t = np.linspace(0.0, 2 * np.pi, spl, endpoint=False, dtype=np.float32)
    t = np.tile(base_t, cols)
    xoff = np.repeat(np.arange(cols, dtype=np.float32), spl)
    active = np.asarray(bitmap > 0.5, dtype=np.float32)
    has = np.repeat(active, spl, axis=1)
    row_height = row_heights[np.minimum(np.arange(rows), len(row_heights) - 1)]
    h = has * np.repeat(row_height[:, None], has.shape[1], axis=1)
    c = np.array(eval_curve(
        jnp.asarray(t[None, :]), jnp.asarray(has), jnp.asarray(h), bulge, stz
    ), dtype=np.float32, copy=True)
    c[:, :, 0] = (c[:, :, 0] + xoff[None, :]) * x_pitch
    c[:, :, 1] += np.arange(rows, dtype=np.float32)[:, None] * dy

    end = np.zeros((rows, 1, 3), dtype=np.float32)
    end[:, 0, 0] = float(cols) * x_pitch
    end[:, 0, 1] = np.arange(rows, dtype=np.float32) * dy
    return [np.concatenate((c[r], end[r]), axis=0).astype(float) for r in range(rows)]


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


def build_spline_mesh(ctrl_rows, params, config, pidx, bitmap_width, radius_ctrl_rows=None):
    p = np.asarray(params)
    rad, rat = p[pidx["radius"]], p[pidx["ellipse_ratio"]]
    seg, res = config["knit_parameters"]["segments"], config["knit_parameters"]["loop_res"]
    nout = res * bitmap_width + 1
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ca, sa = np.cos(a)[None, :, None], np.sin(a)[None, :, None]
    out = []
    for row_idx, r in enumerate(ctrl_rows):
        cp = np.asarray(r, dtype=float)
        if len(cp) <= 1:
            pts = np.repeat(cp, nout, axis=0)
            ctrl_sample_idx = np.zeros(nout, dtype=float)
        else:
            t = np.concatenate(([0.0], np.cumsum(np.maximum(np.linalg.norm(np.diff(cp, axis=0), axis=1), 1e-6))))
            to = np.linspace(t[0], t[-1], nout)
            pts = np.column_stack([np.interp(to, t, cp[:, i]) for i in range(3)]) if len(cp) == 2 else np.column_stack([CubicSpline(t, cp[:, i], bc_type="natural")(to) for i in range(3)])
            ctrl_sample_idx = np.linspace(0.0, len(cp) - 1, nout, dtype=float)

        if radius_ctrl_rows is not None and row_idx < len(radius_ctrl_rows):
            radius_cp = np.asarray(radius_ctrl_rows[row_idx], dtype=float)
            if radius_cp.shape[0] == len(cp):
                if len(cp) <= 1:
                    radius_line = np.full(nout, float(radius_cp[0]) if len(radius_cp) else float(rad), dtype=float)
                else:
                    radius_line = np.interp(ctrl_sample_idx, np.arange(len(radius_cp), dtype=float), radius_cp)
            else:
                radius_line = np.full(nout, float(rad), dtype=float)
        else:
            radius_line = np.full(nout, float(rad), dtype=float)
        radius_line = np.maximum(radius_line, 1e-6)

        T = np.gradient(pts, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-8
        U = np.cross(T, [0, 0, 1])
        b = np.linalg.norm(U, axis=1) < 1e-6
        U[b] = np.cross(T[b], [1, 0, 0])
        U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
        V = np.cross(T, U)
        rline = radius_line[:, None, None]
        out.append(((pts[:, None, :] + U[:, None, :] * ca * rline * rat + V[:, None, :] * sa * rline).reshape(-1, 3), nout))
    return out
