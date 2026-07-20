# %% PRELOAD & IMPORTS ──────────────────────────────────────────────────────────────
import ctypes
import glob
import os
import sys
import json
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
import scipy.sparse
import ipctk

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


# %% YARN SIMULATION PIPELINE ─────────────────────────────────────────────────────────
# DEBUG: last V and edges from eval_energy — read by the renderer overlay
_debug_sim_geometry = None  # (V: ndarray(N,3), edges: ndarray(E,2))
_debug_collisions = None    # collision positions: ndarray(C,3)
class Objective:
    def __init__(self, edges, L0_array, period_offset_x, period_offset_y, nout, dhat):
        self.edges = edges
        self.L0_array = L0_array
        self.period_offset_x = period_offset_x
        self.period_offset_y = period_offset_y
        self.nout = nout
        self.dhat = dhat

    def value(self, V: np.ndarray) -> float:
        """Returns the scalar energy value of the objective at centerline positions V (M x 3)."""
        raise NotImplementedError

    def gradient(self, V: np.ndarray) -> np.ndarray:
        """Analytical gradient (defaults to numerical_gradient if not overridden)."""
        return self.numerical_gradient(V)

    def hessian(self, V: np.ndarray) -> scipy.sparse.csr_matrix:
        """Analytical Hessian (defaults to numerical_hessian if not overridden)."""
        return self.numerical_hessian(V)

    def numerical_gradient(self, V: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Computes the numerical gradient via central differences of value(V) with respect to V.
        
        Returns:
            np.ndarray of shape (M, 3)
        """
        V = np.asarray(V, dtype=float)
        M = V.shape[0]
        N_vars = M * 3
        flat_V = V.flatten()
        grad = np.zeros(N_vars)
        for i in range(N_vars):
            V_plus = flat_V.copy()
            V_plus[i] += eps
            val_plus = self.value(V_plus.reshape(M, 3))
            
            V_minus = flat_V.copy()
            V_minus[i] -= eps
            val_minus = self.value(V_minus.reshape(M, 3))
            
            grad[i] = (val_plus - val_minus) / (2.0 * eps)
        return grad.reshape(M, 3)

    def numerical_hessian(self, V: np.ndarray, eps: float = 1e-5) -> scipy.sparse.csr_matrix:
        """Computes the numerical Hessian via central differences of self.gradient(V) with respect to V.
        
        Returns:
            scipy.sparse.csr_matrix of shape (3*M, 3*M)
        """
        import scipy.sparse
        V = np.asarray(V, dtype=float)
        M = V.shape[0]
        N_vars = M * 3
        flat_V = V.flatten()
        
        H_dense = np.zeros((N_vars, N_vars))
        for j in range(N_vars):
            V_plus = flat_V.copy()
            V_plus[j] += eps
            g_plus = self.gradient(V_plus.reshape(M, 3)).flatten()
            
            V_minus = flat_V.copy()
            V_minus[j] -= eps
            g_minus = self.gradient(V_minus.reshape(M, 3)).flatten()
            
            H_dense[:, j] = (g_plus - g_minus) / (2.0 * eps)
            
        return scipy.sparse.csr_matrix(H_dense)

    def check_gradient(self, V: np.ndarray, eps: float = 1e-5) -> tuple[float, float]:
        g_anal = self.gradient(V)
        g_num = self.numerical_gradient(V, eps)
        diff = np.abs(g_anal - g_num)
        return float(np.max(diff)), float(np.mean(diff))

    def check_hessian(self, V: np.ndarray, eps: float = 1e-5) -> tuple[float, float]:
        import scipy.sparse
        H_anal = self.hessian(V).toarray()
        H_num = self.numerical_hessian(V, eps).toarray()
        diff = np.abs(H_anal - H_num)
        return float(np.max(diff)), float(np.mean(diff))


class StretchObjective(Objective):
    def __init__(self, edges, L0_array, period_offset_x, period_offset_y, nout, dhat, project_psd=True):
        super().__init__(edges, L0_array, period_offset_x, period_offset_y, nout, dhat)
        self.project_psd = project_psd

    def value(self, V: np.ndarray) -> float:
        V = np.asarray(V, dtype=float)
        v0_pts = V[self.edges[:, 0]]
        v1_pts = V[self.edges[:, 1]]
        lengths = np.linalg.norm(v1_pts - v0_pts, axis=1)
        return float(0.5 * np.sum((lengths - self.L0_array) ** 2))

    def gradient(self, V: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        V = np.asarray(V, dtype=float)
        grad = np.zeros_like(V)
        v0 = self.edges[:, 0]
        v1 = self.edges[:, 1]
        diff = V[v1] - V[v0]
        lengths = np.linalg.norm(diff, axis=1)
        
        valid = lengths > 1e-8
        u = np.zeros_like(diff)
        u[valid] = diff[valid] / lengths[valid][:, None]
        
        g = (lengths - self.L0_array)[:, None] * u
        np.add.at(grad, v1, g)
        np.add.at(grad, v0, -g)
        return grad

    def hessian(self, V: np.ndarray, eps: float = 1e-5) -> scipy.sparse.csr_matrix:
        V = np.asarray(V, dtype=float)
        M = V.shape[0]
        v0 = self.edges[:, 0]
        v1 = self.edges[:, 1]
        diff = V[v1] - V[v0]
        lengths = np.linalg.norm(diff, axis=1)
        
        rows = []
        cols = []
        vals = []
        
        for idx, edge in enumerate(self.edges):
            u_idx = edge[0]
            w_idx = edge[1]
            l = lengths[idx]
            if l < 1e-8:
                continue
            u = diff[idx] / l
            uuT = np.outer(u, u)
            lam = (l - self.L0_array[idx]) / l
            if self.project_psd:
                lam = max(0.0, lam)
            H_local = uuT + lam * (np.eye(3) - uuT)
            
            for r in range(3):
                  for c in range(3):
                      val = H_local[r, c]
                      rows.append(3 * u_idx + r); cols.append(3 * u_idx + c); vals.append(val)
                      rows.append(3 * w_idx + r); cols.append(3 * w_idx + c); vals.append(val)
                      rows.append(3 * u_idx + r); cols.append(3 * w_idx + c); vals.append(-val)
                      rows.append(3 * w_idx + r); cols.append(3 * u_idx + c); vals.append(-val)
        return scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(3*M, 3*M)).tocsr()


class BendObjective(Objective):
    def value(self, V: np.ndarray) -> float:
        V = np.asarray(V, dtype=float)
        M = V.shape[0]
        num_ctrl_rows = M // self.nout
        e_b = 0.0
        for r in range(num_ctrl_rows):
            start = r * self.nout
            end = (r + 1) * self.nout
            v_prev = V[start : end - 2]
            v_curr = V[start + 1 : end - 1]
            v_next = V[start + 2 : end]
            e_b += 0.5 * np.sum((v_prev - 2.0 * v_curr + v_next)**2)
            
            bound_prev = V[end - 2] - self.period_offset_x
            bound_curr = V[start]
            bound_next = V[start + 1]
            e_b += 0.5 * np.sum((bound_prev - 2.0 * bound_curr + bound_next)**2)
        return float(e_b)

    def gradient(self, V: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        V = np.asarray(V, dtype=float)
        M = V.shape[0]
        grad_V = np.zeros_like(V)
        num_ctrl_rows = M // self.nout
        for r in range(num_ctrl_rows):
            start = r * self.nout
            end = (r + 1) * self.nout
            for i in range(start, end - 1):
                if i == start:
                    v_prev_pt = V[end - 2] - self.period_offset_x
                    v_curr = i
                    v_next = i + 1
                    lap = v_prev_pt - 2.0 * V[v_curr] + V[v_next]
                    v_prev_idx = end - 2
                else:
                    v_prev_idx = i - 1
                    v_curr = i
                    v_next = i + 1
                    lap = V[v_prev_idx] - 2.0 * V[v_curr] + V[v_next]
                
                grad_V[v_prev_idx] += lap
                grad_V[v_curr] -= 2.0 * lap
                grad_V[v_next] += lap
        return grad_V

    def hessian(self, V: np.ndarray, eps: float = 1e-5) -> scipy.sparse.csr_matrix:
        V = np.asarray(V, dtype=float)
        M = V.shape[0]
        rows = []
        cols = []
        vals = []
        num_ctrl_rows = M // self.nout
        for r in range(num_ctrl_rows):
            start = r * self.nout
            end = (r + 1) * self.nout
            for i in range(start, end - 1):
                if i == start:
                    v_prev_idx = end - 2
                    v_curr = i
                    v_next = i + 1
                else:
                    v_prev_idx = i - 1
                    v_curr = i
                    v_next = i + 1
                 
                stencil = [(v_prev_idx, 1.0), (v_curr, -2.0), (v_next, 1.0)]
                for idx1, c1 in stencil:
                    for idx2, c2 in stencil:
                        val = c1 * c2
                        for d in range(3):
                            rows.append(3 * idx1 + d)
                            cols.append(3 * idx2 + d)
                            vals.append(val)
        return scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(3*M, 3*M)).tocsr()


class CollisionObjective(Objective):
    def __init__(self, edges, L0_array, period_offset_x, period_offset_y, nout, dhat, psd_projection=None):
        super().__init__(edges, L0_array, period_offset_x, period_offset_y, nout, dhat)
        import ipctk
        self.psd_projection = psd_projection if psd_projection is not None else ipctk.PSDProjectionMethod.CLAMP

    def value(self, V: np.ndarray) -> float:
        import ipctk
        V = np.asarray(V, dtype=float)
        M = V.shape[0]
        c_x_range = [-1, 0, 1] if np.linalg.norm(self.period_offset_x) > 1e-6 else [0]
        c_y_range = [-1, 0, 1] if np.linalg.norm(self.period_offset_y) > 1e-6 else [0]
        grid = [(c_x, c_y) for c_y in c_y_range for c_x in c_x_range]
        offsets = [c_x * self.period_offset_x + c_y * self.period_offset_y for c_x, c_y in grid]
        num_copies = len(offsets)
        V_tiled = np.vstack([V + offset[None, :] for offset in offsets])
        edges_tiled = np.vstack([self.edges + c * M for c in range(num_copies)])
        mesh_tiled = ipctk.CollisionMesh(V_tiled, edges_tiled, np.empty((0, 3), dtype=np.int32))
        mesh_tiled.can_collide = build_can_collide(M, self.edges, self.period_offset_x, self.period_offset_y)
        
        # Stash full tiled geometry for the debug overlay
        global _debug_sim_geometry, _debug_collisions
        _debug_sim_geometry = (V_tiled, edges_tiled)
        
        collisions = ipctk.NormalCollisions()
        collisions.build(mesh_tiled, V_tiled, self.dhat)
        
        # Extract collision positions
        collision_pts = []
        try:
            for ev in getattr(collisions, "ev_collisions", []):
                v_idx = int(getattr(ev, "vertex_id", -1))
                if 0 <= v_idx < len(V_tiled):
                    collision_pts.append(V_tiled[v_idx])
            for ee in getattr(collisions, "ee_collisions", []):
                e0 = int(getattr(ee, "edge0_id", -1))
                e1 = int(getattr(ee, "edge1_id", -1))
                if 0 <= e0 < len(edges_tiled) and 0 <= e1 < len(edges_tiled):
                    v0_0 = V_tiled[edges_tiled[e0, 0]]
                    v0_1 = V_tiled[edges_tiled[e0, 1]]
                    v1_0 = V_tiled[edges_tiled[e1, 0]]
                    v1_1 = V_tiled[edges_tiled[e1, 1]]
                    pt = 0.25 * (v0_0 + v0_1 + v1_0 + v1_1)
                    collision_pts.append(pt)
            for vv in getattr(collisions, "vv_collisions", []):
                v0 = int(getattr(vv, "vertex0_id", -1))
                v1 = int(getattr(vv, "vertex1_id", -1))
                if 0 <= v0 < len(V_tiled) and 0 <= v1 < len(V_tiled):
                    pt = 0.5 * (V_tiled[v0] + V_tiled[v1])
                    collision_pts.append(pt)
        except Exception as e:
            print(f"[Sim Debug] Error extracting collision positions: {e}")

        if collision_pts:
            _debug_collisions = np.vstack(collision_pts)
        else:
            _debug_collisions = np.empty((0, 3), dtype=np.float32)

        barrier = ipctk.BarrierPotential(self.dhat, 1.0)
        return float(barrier(collisions, mesh_tiled, V_tiled))

    def gradient(self, V: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        import ipctk
        V = np.asarray(V, dtype=float)
        M = V.shape[0]
        c_x_range = [-1, 0, 1] if np.linalg.norm(self.period_offset_x) > 1e-6 else [0]
        c_y_range = [-1, 0, 1] if np.linalg.norm(self.period_offset_y) > 1e-6 else [0]
        offsets = [c_x * self.period_offset_x + c_y * self.period_offset_y for c_y in c_y_range for c_x in c_x_range]
        num_copies = len(offsets)
        V_tiled = np.vstack([V + offset[None, :] for offset in offsets])
        edges_tiled = np.vstack([self.edges + c * M for c in range(num_copies)])
        mesh_tiled = ipctk.CollisionMesh(V_tiled, edges_tiled, np.empty((0, 3), dtype=np.int32))
        mesh_tiled.can_collide = build_can_collide(M, self.edges, self.period_offset_x, self.period_offset_y)
        collisions = ipctk.NormalCollisions()
        collisions.build(mesh_tiled, V_tiled, self.dhat)
        barrier = ipctk.BarrierPotential(self.dhat, 1.0)
        barrier_grad = barrier.gradient(collisions, mesh_tiled, V_tiled)
        grad_V = np.zeros((M, 3))
        barrier_grad_reshaped = barrier_grad.reshape(num_copies, M, 3)
        for c in range(num_copies):
            grad_V += barrier_grad_reshaped[c]
        return grad_V

    def hessian(self, V: np.ndarray, eps: float = 1e-5) -> scipy.sparse.csr_matrix:
        import ipctk
        V = np.asarray(V, dtype=float)
        M = V.shape[0]
        c_x_range = [-1, 0, 1] if np.linalg.norm(self.period_offset_x) > 1e-6 else [0]
        c_y_range = [-1, 0, 1] if np.linalg.norm(self.period_offset_y) > 1e-6 else [0]
        offsets = [c_x * self.period_offset_x + c_y * self.period_offset_y for c_y in c_y_range for c_x in c_x_range]
        num_copies = len(offsets)
        V_tiled = np.vstack([V + offset[None, :] for offset in offsets])
        edges_tiled = np.vstack([self.edges + c * M for c in range(num_copies)])
        mesh_tiled = ipctk.CollisionMesh(V_tiled, edges_tiled, np.empty((0, 3), dtype=np.int32))
        mesh_tiled.can_collide = build_can_collide(M, self.edges, self.period_offset_x, self.period_offset_y)
        collisions = ipctk.NormalCollisions()
        collisions.build(mesh_tiled, V_tiled, self.dhat)
        barrier = ipctk.BarrierPotential(self.dhat, 1.0)
        barrier_hess = barrier.hessian(collisions, mesh_tiled, V_tiled, self.psd_projection)
        barrier_hess_coo = barrier_hess.tocoo()
        folded_rows = barrier_hess_coo.row % (3 * M)
        folded_cols = barrier_hess_coo.col % (3 * M)
        folded_vals = barrier_hess_coo.data
        return scipy.sparse.coo_matrix((folded_vals, (folded_rows, folded_cols)), shape=(3*M, 3*M)).tocsr()


def run_simulation_step(ctrl_rows, period_offset_x, period_offset_y, config, J_cached, L0_array, k_s, k_b, k_c, dhat):
    import scipy.sparse.linalg
    res = config["knit_parameters"]["loop_res"]
    bitmap_width = float(np.linalg.norm(period_offset_x))
    nout = res * int(round(bitmap_width)) + 1
    
    flat_P = np.concatenate(ctrl_rows).astype(float)
    num_ctrl_rows = len(ctrl_rows)
    N = len(flat_P) // 3
    
    V, edges, D, nout = evaluate_centerlines(ctrl_rows, period_offset_x, config)
    M = len(V)
    
    kwargs = {
        "edges": edges,
        "L0_array": L0_array,
        "period_offset_x": period_offset_x,
        "period_offset_y": period_offset_y,
        "nout": nout,
        "dhat": dhat
    }
    
    stretch = StretchObjective(**kwargs)
    bend = BendObjective(**kwargs)
    
    g_V = k_s * stretch.gradient(V) + k_b * bend.gradient(V)
    H_V = k_s * stretch.hessian(V) + k_b * bend.hessian(V)
    
    if k_c > 0.0:
        import ipctk
        collision = CollisionObjective(**kwargs, psd_projection=ipctk.PSDProjectionMethod.CLAMP)
        g_V += k_c * collision.gradient(V)
        H_V += k_c * collision.hessian(V)
        
    g_V_flat = g_V.flatten()
    
    # Starting energy from eval_energy (el + bending + col) for consistent line search comparison
    e_el_s, e_b_s, e_col_s = eval_energy(flat_P, ctrl_rows, period_offset_x, period_offset_y, config, L0_array, k_s, k_b, k_c, dhat)
    e = e_el_s + e_b_s + e_col_s

    # 2. Project to control point space using the cached Jacobian matrix J_cached
    g_P = J_cached.T @ g_V_flat
    H_P = J_cached.T @ H_V @ J_cached
    
    # Regularize to guarantee positive definiteness
    H_P_dense = H_P.toarray()
    H_pd = H_P_dense + 1e-6 * np.eye(H_P_dense.shape[0])
    
    # 3. Solve for search direction
    try:
        delta_P = np.linalg.solve(H_pd, -g_P)
    except np.linalg.LinAlgError:
        print("[Sim Step] Solver FAILED: Hessian is singular.")
        return ctrl_rows  # skip step if singular
    
    # Debug prints for optimization step
    g_norm = np.linalg.norm(g_P)
    dot_prod = np.dot(g_P, delta_P)
    print(f"[Sim Step] E_start: {e:.6e} (Elastic: {e_el_s:.6e}, Bending: {e_b_s:.6e}, Barrier: {e_col_s:.6e}) | ||g_P||: {g_norm:.6e} | g_P^T delta_P: {dot_prod:.6e}")

    # 4. CCD Step Size Calculation on 3x3 tiled copies
    delta_P_reshaped = delta_P.reshape(-1, 3)
    P_cand = flat_P.reshape(-1, 3) + delta_P_reshaped
    
    # Map P_cand back to V_cand
    ctrl_rows_cand = []
    ctrl_offset = 0
    for r in range(num_ctrl_rows):
        n_c = len(ctrl_rows[r])
        ctrl_rows_cand.append(P_cand[ctrl_offset:ctrl_offset+n_c])
        ctrl_offset += n_c
    V_cand, _, _, _ = evaluate_centerlines(ctrl_rows_cand, period_offset_x, config)
    
    # Build an open-interval tiled mesh for CCD.
    # eval_centerline is periodic: V[row_end - 1] == V[row_start] + period_offset_x exactly.
    # If we tile the full closed rows, adjacent copies share coincident endpoints, giving
    # initial distance=0 → alpha_max=0. Strip the redundant last vertex of each row before tiling.
    c_x_range = [-1, 0, 1] if np.linalg.norm(period_offset_x) > 1e-6 else [0]
    c_y_range = [-1, 0, 1] if np.linalg.norm(period_offset_y) > 1e-6 else [0]
    offsets = []
    for c_y in c_y_range:
        for c_x in c_x_range:
            offsets.append(c_x * period_offset_x + c_y * period_offset_y)
    num_copies = len(offsets)

    nout_open = nout - 1  # drop the redundant periodic endpoint from each row
    M_open = nout_open * num_ctrl_rows
    keep = np.ones(M, dtype=bool)
    for r in range(num_ctrl_rows):
        keep[(r + 1) * nout - 1] = False   # mask out last vertex of each row
    V_open      = V[keep]
    V_cand_open = V_cand[keep]

    # Rebuild row edges without the now-absent endpoint
    edges_open_list = []
    for r in range(num_ctrl_rows):
        base = r * nout_open
        edges_open_list.append(np.array([[base + i, base + i + 1] for i in range(nout_open - 1)], dtype=np.int32))
    edges_open = np.vstack(edges_open_list)

    V_tiled_0 = np.vstack([V_open      + offset[None, :] for offset in offsets])
    V_tiled_1 = np.vstack([V_cand_open + offset[None, :] for offset in offsets])

    edges_tiled = np.vstack([edges_open + c * M_open for c in range(num_copies)])
    mesh_tiled  = ipctk.CollisionMesh(V_tiled_0, edges_tiled, np.empty((0, 3), dtype=np.int32))
    mesh_tiled.can_collide = build_can_collide(M_open, edges_open, period_offset_x, period_offset_y)

    alpha_max = ipctk.compute_collision_free_stepsize(mesh_tiled, V_tiled_0, V_tiled_1)
    print(f"[Sim Step] alpha_max (CCD): {alpha_max:.6f}")

    # 5. Backtracking Line Search — delegates to eval_energy for correct, consistent energy
    alpha = alpha_max
    tau = 0.5
    success = False
    
    for search_iter in range(10):
        P_new = flat_P.reshape(-1, 3) + alpha * delta_P_reshaped
        new_ctrl_rows = []
        ctrl_offset = 0
        for r in range(num_ctrl_rows):
            n_c = len(ctrl_rows[r])
            new_ctrl_rows.append(P_new[ctrl_offset:ctrl_offset+n_c])
            ctrl_offset += n_c
        
        e_el_new, e_b_new, e_col_new = eval_energy(P_new.flatten(), new_ctrl_rows, period_offset_x, period_offset_y, config, L0_array, k_s, k_b, k_c, dhat)
        e_new = e_el_new + e_b_new + e_col_new
        print(f"  [Line Search] Iter {search_iter}: alpha = {alpha:.6e} | E_cand = {e_new:.6e} (Elastic: {e_el_new:.6e}, Bending: {e_b_new:.6e}, Barrier: {e_col_new:.6e})")

        if e_new < e:
            success = True
            ctrl_rows = new_ctrl_rows
            print(f"[Sim Step] SUCCESS: alpha = {alpha:.6e} | E_final: {e_new:.6e} (decreased by {e - e_new:.6e})")
            break
        alpha *= tau
        
    if not success:
        print(f"[Sim Step] FAILED: line search could not find energy decrease.")

    return ctrl_rows


def eval_energy(flat_P, ctrl_rows, period_offset_x, period_offset_y, config, L0_array, k_s, k_b, k_c, dhat, filter_collisions=True):
    num_ctrl_rows = len(ctrl_rows)
    ctrl_offset = 0
    perturbed_ctrl_rows = []
    flat_P_reshaped = flat_P.reshape(-1, 3)
    for r in range(num_ctrl_rows):
        n_c = len(ctrl_rows[r])
        perturbed_ctrl_rows.append(flat_P_reshaped[ctrl_offset:ctrl_offset+n_c])
        ctrl_offset += n_c
        
    V, edges, _, nout = evaluate_centerlines(perturbed_ctrl_rows, period_offset_x, config)
    M = len(V)

    kwargs = {
        "edges": edges,
        "L0_array": L0_array,
        "period_offset_x": period_offset_x,
        "period_offset_y": period_offset_y,
        "nout": nout,
        "dhat": dhat
    }
    
    stretch = StretchObjective(**kwargs)
    bend = BendObjective(**kwargs)
    
    e_el = stretch.value(V)
    e_b = bend.value(V)
    e_col = 0.0
    
    if k_c > 0.0:
        collision = CollisionObjective(**kwargs)
        e_col = collision.value(V)
        
    return e_el, e_b, e_col


# %% HEADLESS FINITE DIFFERENCE VERIFICATION ──────────────────────────────────────────
def check_gradients_and_hessians_fd(ctrl_rows, period_offset_x, period_offset_y, config, J_cached, k_s, k_b, k_c, dhat, eps=1e-4):
    V, edges, D, nout = evaluate_centerlines(ctrl_rows, period_offset_x, config)
    
    import ipctk
    kwargs = {
        "edges": edges,
        "L0_array": np.full(len(edges), D[0] / (nout - 1), dtype=float),
        "period_offset_x": period_offset_x,
        "period_offset_y": period_offset_y,
        "nout": nout,
        "dhat": dhat
    }
    
    # We check Stretch, Bend, and Collision objectives directly in physical space V
    stretch = StretchObjective(**kwargs, project_psd=False)
    bend = BendObjective(**kwargs)
    collision = CollisionObjective(**kwargs, psd_projection=ipctk.PSDProjectionMethod.NONE)
    
    result_lines = []
    for name, obj in [("stretch", stretch), ("bend", bend), ("collision", collision)]:
        g_anal = obj.gradient(V).flatten()
        g_num = obj.numerical_gradient(V, eps).flatten()
        H_anal = obj.hessian(V).toarray()
        H_num = obj.numerical_hessian(V, eps).toarray()
        
        g_diff = np.abs(g_anal - g_num)
        H_diff = np.abs(H_anal - H_num)
        
        lines = [f"[{name.upper()} Check]"]
        lines.append(f"  {'Idx':<5} | {'Analytical':<14} | {'Numerical':<14} | {'Difference':<14}")
        lines.append("  " + "-" * 57)
        for idx in range(len(g_anal)):
            lines.append(f"  {idx:<5} | {g_anal[idx]:>14.6f} | {g_num[idx]:>14.6f} | {g_diff[idx]:>14.6f}")
        lines.append(f"  Hessian difference: max={np.max(H_diff):.6e}, mean={np.mean(H_diff):.6e}")
        result_lines.append("\n".join(lines))
        
    result_str = "\n".join(result_lines)
    print(result_str)
    return result_str


def build_can_collide(M, edges, period_offset_x, period_offset_y):
    import ipctk
    num_rows = M - len(edges)
    nout = M // num_rows

    c_x_range = [-1, 0, 1] if np.linalg.norm(period_offset_x) > 1e-6 else [0]
    c_y_range = [-1, 0, 1] if np.linalg.norm(period_offset_y) > 1e-6 else [0]
    grid = [(c_x, c_y) for c_y in c_y_range for c_x in c_x_range]

    # Compute absolute x-position along the periodic chain for every vertex in the tiled mesh.
    # abs_x = c_x * (nout - 1) + i  — this is continuous across x-copy boundaries:
    #   the periodic endpoint (i == nout-1) in cx=k maps to the same abs_x as i=0 in cx=k+1,
    #   so adjacent pairs across the boundary are naturally caught by |abs_x1 - abs_x2| <= 1.
    by_row_cy = {}
    for v in range(len(grid) * M):
        c = v // M
        u = v % M
        c_x, c_y = grid[c]
        r = u // nout
        i = u % nout
        abs_x = c_x * (nout - 1) + i
        by_row_cy.setdefault((r, c_y), []).append((abs_x, v))

    explicit_values = {}
    for items in by_row_cy.values():
        n = len(items)
        for idx1 in range(n):
            ax1, v1 = items[idx1]
            for idx2 in range(idx1 + 1, n):
                ax2, v2 = items[idx2]
                if abs(ax1 - ax2) <= 4:
                    explicit_values[(min(v1, v2), max(v1, v2))] = False

    return ipctk.make_sparse_filter(explicit_values, True)
