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
    lut = jnp.concatenate((jnp.zeros(1), params[jnp.array(lh_idx)]))
    scale = _scale_factors_jax(bitmap)

    rows, loops = bitmap.shape
    t = jnp.linspace(0.0, 2 * jnp.pi * loops, loop_res * loops + 1)
    a = jnp.linspace(0, 2 * jnp.pi, seg, endpoint=False)
    ca, sa = jnp.cos(a)[None, :, None], jnp.sin(a)[None, :, None]

    def row_fn(i, srow):
        ls = jnp.append(jnp.repeat(srow, loop_res), 1.0).astype(jnp.int32)
        h = jnp.take(lut, ls)
        has = (ls > 0).astype(jnp.float32)
        p = eval_curve(t, has, h, bulge, stz).at[:, 1].add(i * dy)
        d = eval_curve_derivative(t, has, h, bulge, stz)
        u, v = compute_orthonormal_frame(d)
        off = u[:, None, :] * ca * rad * rat + v[:, None, :] * sa * rad
        return (p[:, None, :] + off).reshape(-1, 3)

    return jax.vmap(row_fn)(jnp.arange(rows), scale)


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
    lut = np.concatenate((np.zeros(1), p[np.array(lh_idx)]))
    sf = np.asarray(_scale_factors_jax(jnp.asarray(bitmap))).astype(np.int32)
    rows, cols = sf.shape
    base_t = np.linspace(0.0, 2 * np.pi, spl, endpoint=False, dtype=np.float32)
    t = np.tile(base_t, cols)
    xoff = np.repeat(np.arange(cols, dtype=np.float32), spl)
    s = np.repeat(sf, spl, axis=1)
    has = (s > 0).astype(np.float32)
    h = lut[s]
    c = np.array(eval_curve(
        jnp.asarray(t[None, :]), jnp.asarray(has), jnp.asarray(h), bulge, stz
    ), dtype=np.float32, copy=True)
    c[:, :, 0] += xoff[None, :]
    c[:, :, 1] += np.arange(rows, dtype=np.float32)[:, None] * dy

    end = np.zeros((rows, 1, 3), dtype=np.float32)
    end[:, 0, 0] = float(cols)
    end[:, 0, 1] = np.arange(rows, dtype=np.float32) * dy
    return [np.concatenate((c[r], end[r]), axis=0).astype(float) for r in range(rows)]


def save_combined_obj(mesh_data_list, base_filename="knitting_model"):
    path = f"{base_filename}_combined.obj"
    off = 0
    with open(path, "w") as h:
        h.write("# Knitting Model\n")
        for i, (v, _, f, _) in enumerate(mesh_data_list):
            h.write(f"o mesh_{i}\n")
            np.savetxt(h, v, fmt="v %.6f %.6f %.6f")
            np.savetxt(h, f + off + 1, fmt="f %d %d %d %d")
            off += len(v)


def save_per_loop_objs(mesh_data_list, base_filename, loop_res, segments):
    """Save each stitch loop as a separate OBJ file."""
    loop_vertex_count = (loop_res + 1) * segments
    loop_faces = compute_knitting_faces(segments, [(np.empty((loop_vertex_count, 3)), loop_res + 1)])[0]
    loop_specs = [
        (
            row_idx,
            loop_idx,
            verts[loop_idx * loop_res * segments:loop_idx * loop_res * segments + loop_vertex_count],
            f"{base_filename}_r{row_idx:02d}_l{loop_idx:02d}.obj",
        )
        for row_idx, (verts, _, _, n_points) in enumerate(mesh_data_list)
        for loop_idx in range((n_points - 1) // loop_res)
    ]

    obj_info = []
    for row_idx, loop_idx, loop_verts, path in loop_specs:
        with open(path, "w") as handle:
            np.savetxt(handle, loop_verts, fmt="v %.6f %.6f %.6f")
            np.savetxt(handle, loop_faces + 1, fmt="f %d %d %d %d")
        obj_info.append((row_idx, loop_idx, path))
    return obj_info


def build_spline_mesh(ctrl_rows, params, config, pidx, bitmap_width):
    p = np.asarray(params)
    rad, rat = p[pidx["radius"]], p[pidx["ellipse_ratio"]]
    seg, res = config["knit_parameters"]["segments"], config["knit_parameters"]["loop_res"]
    nout = res * bitmap_width + 1
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ca, sa = np.cos(a)[None, :, None], np.sin(a)[None, :, None]
    out = []
    for r in ctrl_rows:
        cp = np.asarray(r, dtype=float)
        if len(cp) <= 1:
            pts = np.repeat(cp, nout, axis=0)
        else:
            t = np.concatenate(([0.0], np.cumsum(np.maximum(np.linalg.norm(np.diff(cp, axis=0), axis=1), 1e-6))))
            to = np.linspace(t[0], t[-1], nout)
            pts = np.column_stack([np.interp(to, t, cp[:, i]) for i in range(3)]) if len(cp) == 2 else np.column_stack([CubicSpline(t, cp[:, i], bc_type="natural")(to) for i in range(3)])
        T = np.gradient(pts, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-8
        U = np.cross(T, [0, 0, 1])
        b = np.linalg.norm(U, axis=1) < 1e-6
        U[b] = np.cross(T[b], [1, 0, 0])
        U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
        V = np.cross(T, U)
        out.append(((pts[:, None, :] + U[:, None, :] * ca * rad * rat + V[:, None, :] * sa * rad).reshape(-1, 3), nout))
    return out
