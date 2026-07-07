import ctypes
import glob
import os
import sys
import json
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
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


def build_spline_mesh(ctrl_rows, params, config, pidx, period_offset, radius_ctrl_rows=None):
    p = np.asarray(params)
    rad, rat = p[pidx["radius"]], p[pidx["ellipse_ratio"]]
    seg, res = config["knit_parameters"]["segments"], config["knit_parameters"]["loop_res"]
    
    if isinstance(period_offset, (int, float, np.integer, np.floating)):
        D = np.array([float(period_offset), 0.0, 0.0], dtype=float)
        bitmap_width = float(period_offset)
    else:
        D = np.asarray(period_offset, dtype=float)
        bitmap_width = float(np.linalg.norm(D))
        
    nout = res * int(round(bitmap_width)) + 1
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ca, sa = np.cos(a)[None, :, None], np.sin(a)[None, :, None]
    out = []
    for row_idx, r in enumerate(ctrl_rows):
        cp = np.asarray(r, dtype=float)
        pts = eval_centerline(cp, D, nout)
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
        V = np.cross(T, U)
        rline = radius_line[:, None, None]
        out.append(((pts[:, None, :] + U[:, None, :] * ca * rline * rat + V[:, None, :] * sa * rline).reshape(-1, 3), nout))
    return out


def compute_elastic_forces_and_hessian(V, edges, L0, k_s, k_b, project_psd=True):
    import scipy.sparse
    M = len(V)
    grad_V = np.zeros_like(V)
    energy = 0.0

    # Triplet list for building sparse Hessian
    rows = []
    cols = []
    vals = []

    # 1. Stretch potential
    for edge in edges:
        v0, v1 = edge
        diff = V[v1] - V[v0]
        l = np.linalg.norm(diff)
        if l < 1e-8:
            continue
        energy += 0.5 * k_s * (l - L0)**2
        u = diff / l
        g = k_s * (l - L0) * u
        grad_V[v1] += g
        grad_V[v0] -= g

        # Projected local 3x3 Hessian
        lam = k_s * (l - L0) / l
        if project_psd:
            lam = max(0.0, lam)
        uuT = np.outer(u, u)
        H_local = k_s * uuT + lam * (np.eye(3) - uuT)

        for r in range(3):
            for c in range(3):
                val = H_local[r, c]
                # Add to sparse coordinates
                # v0-v0
                rows.append(3 * v0 + r); cols.append(3 * v0 + c); vals.append(val)
                # v1-v1
                rows.append(3 * v1 + r); cols.append(3 * v1 + c); vals.append(val)
                # v0-v1
                rows.append(3 * v0 + r); cols.append(3 * v1 + c); vals.append(-val)
                # v1-v0
                rows.append(3 * v1 + r); cols.append(3 * v0 + c); vals.append(-val)

    # 2. Bending potential (wrapped periodically)
    # Triplets: (V_{i-1}, V_i, V_{i+1})
    for i in range(M):
        # Wrapping for closed periodic loop
        v_prev = (i - 1) % M
        v_curr = i
        v_next = (i + 1) % M

        lap = V[v_prev] - 2.0 * V[v_curr] + V[v_next]
        energy += 0.5 * k_b * np.sum(lap**2)

        g = k_b * lap
        grad_V[v_prev] += g
        grad_V[v_curr] -= 2.0 * g
        grad_V[v_next] += g

        # Bending Hessian is k_b * A A^T (always PSD)
        # A = [1, -2, 1]^T \otimes I_3
        stencil = [(v_prev, 1.0), (v_curr, -2.0), (v_next, 1.0)]
        for idx1, c1 in stencil:
            for idx2, c2 in stencil:
                val = k_b * c1 * c2
                for d in range(3):
                    rows.append(3 * idx1 + d)
                    cols.append(3 * idx2 + d)
                    vals.append(val)

    H_V = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(3*M, 3*M)).tocsr()
    return energy, grad_V.flatten(), H_V


def compute_collision_forces_and_hessian(V, D, edges, dhat, k_c, psd_projection=None):
    import ipctk
    import scipy.sparse
    if psd_projection is None:
        psd_projection = ipctk.PSDProjectionMethod.CLAMP
    M = len(V)
    
    # 1. Tile vertices dynamically to 3x3 copies
    # Copies indices: c_x, c_y from -1 to 1
    offsets = []
    for c_x in [-1, 0, 1]:
        for c_y in [-1, 0, 1]:
            offsets.append(c_x * np.array([D[0], 0.0, 0.0]) + c_y * np.array([0.0, D[1], 0.0]))
    
    V_tiled = []
    for offset in offsets:
        V_tiled.append(V + offset[None, :])
    V_tiled = np.vstack(V_tiled)  # Shape (9*M, 3)

    # 2. Build edges for the tiled mesh
    edges_tiled = []
    for c in range(9):
        edges_tiled.append(edges + c * M)
    edges_tiled = np.vstack(edges_tiled)

    faces_tiled = np.empty((0, 3), dtype=np.int32)
    mesh_tiled = ipctk.CollisionMesh(V_tiled, edges_tiled, faces_tiled)
    mesh_tiled.can_collide = build_can_collide(M, edges, D)

    # 3. IPC collisions build
    collisions = ipctk.NormalCollisions()
    collisions.build(mesh_tiled, V_tiled, dhat)

    barrier = ipctk.BarrierPotential(dhat)
    barrier_E = barrier(collisions, mesh_tiled, V_tiled) * k_c
    barrier_grad = barrier.gradient(collisions, mesh_tiled, V_tiled) * k_c
    barrier_hess = barrier.hessian(collisions, mesh_tiled, V_tiled, psd_projection) * k_c

    # 4. Fold gradient down to primary unit cell
    grad_V = np.zeros((M, 3))
    barrier_grad_reshaped = barrier_grad.reshape(9, M, 3)
      # Sum across the 9 tile copies
    for c in range(9):
        grad_V += barrier_grad_reshaped[c]

    # 5. Fold sparse Hessian down to primary unit cell (3M x 3M)
    barrier_hess_coo = barrier_hess.tocoo()
    folded_rows = barrier_hess_coo.row % (3 * M)
    folded_cols = barrier_hess_coo.col % (3 * M)
    folded_vals = barrier_hess_coo.data

    H_V = scipy.sparse.coo_matrix(
        (folded_vals, (folded_rows, folded_cols)), shape=(3*M, 3*M)
    ).tocsr()

    print(f"[Sim] Collisions: {len(collisions)} | Barrier Energy: {barrier_E:.6e}")
    return barrier_E, grad_V.flatten(), H_V


def run_simulation_step(ctrl_rows, D, config, J_cached, k_s, k_b, k_c, dhat):
    import scipy.sparse.linalg
    res = config["knit_parameters"]["loop_res"]
    bitmap_width = float(D[0])
    nout = res * int(round(bitmap_width)) + 1
    
    # Assemble P
    flat_P = np.concatenate(ctrl_rows).astype(float)
    num_ctrl_rows = len(ctrl_rows)
    N = len(flat_P) // 3
    
    # Evaluate centerline vertices V
    V_list = []
    for row_idx, cp in enumerate(ctrl_rows):
        pts = eval_centerline(cp, D, nout)
        V_list.append(pts)
    V = np.vstack(V_list)
    M = len(V)
    
    # Setup edges topology (local segments per row)
    edges_list = []
    row_offset = 0
    for r in range(num_ctrl_rows):
        row_edges = np.array([[i, i+1] for i in range(nout - 1)], dtype=np.int32) + row_offset
        edges_list.append(row_edges)
        row_offset += nout
    edges = np.vstack(edges_list)
    
    L0 = bitmap_width / (nout - 1)
    
    # 1. Compute energies, gradients, and Hessians on unit cell
    e_el, g_el, H_el = compute_elastic_forces_and_hessian(V, edges, L0, k_s, k_b)
    e_col, g_col, H_col = compute_collision_forces_and_hessian(V, D, edges, dhat, k_c)
    
    e = e_el + e_col
    g_V = g_el + g_col
    H_V = H_el + H_col
    
    # 2. Project to control point space using the cached Jacobian matrix J_cached
    g_P = J_cached.T @ g_V
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
    print(f"[Sim Step] E_start: {e:.6e} (Elastic: {e_el:.6e}, Barrier: {e_col:.6e}) | ||g_P||: {g_norm:.6e} | g_P^T delta_P: {dot_prod:.6e}")

    # 4. CCD Step Size Calculation on 3x3 tiled copies
    delta_P_reshaped = delta_P.reshape(-1, 3)
    P_cand = flat_P.reshape(-1, 3) + delta_P_reshaped
    
    # Map P_cand back to V_cand
    V_cand_list = []
    ctrl_offset = 0
    for r in range(num_ctrl_rows):
        n_c = len(ctrl_rows[r])
        cp_cand = P_cand[ctrl_offset:ctrl_offset+n_c]
        pts_cand = eval_centerline(cp_cand, D, nout)
        V_cand_list.append(pts_cand)
        ctrl_offset += n_c
    V_cand = np.vstack(V_cand_list)
    
    # Replicate for CCD step size check
    offsets = []
    for c_x in [-1, 0, 1]:
        for c_y in [-1, 0, 1]:
            offsets.append(c_x * np.array([D[0], 0.0, 0.0]) + c_y * np.array([0.0, D[1], 0.0]))
    
    V_tiled_0 = np.vstack([V + offset[None, :] for offset in offsets])
    V_tiled_1 = np.vstack([V_cand + offset[None, :] for offset in offsets])
    
    edges_tiled = np.vstack([edges + c * M for c in range(9)])
    mesh_tiled = ipctk.CollisionMesh(V_tiled_0, edges_tiled, np.empty((0, 3), dtype=np.int32))
    mesh_tiled.can_collide = build_can_collide(M, edges, D)
    
    alpha_max = ipctk.compute_collision_free_stepsize(mesh_tiled, V_tiled_0, V_tiled_1)
    print(f"[Sim Step] alpha_max (CCD): {alpha_max:.6f}")

    # 5. Backtracking Line Search
    alpha = alpha_max
    tau = 0.5
    success = False
    
    for search_iter in range(10):
        P_new = flat_P.reshape(-1, 3) + alpha * delta_P_reshaped
        # Evaluate new energy
        V_new_list = []
        ctrl_offset = 0
        new_ctrl_rows = []
        for r in range(num_ctrl_rows):
            n_c = len(ctrl_rows[r])
            cp_new = P_new[ctrl_offset:ctrl_offset+n_c]
            new_ctrl_rows.append(cp_new)
            pts_new = eval_centerline(cp_new, D, nout)
            V_new_list.append(pts_new)
            ctrl_offset += n_c
        V_new = np.vstack(V_new_list)
        
        # Elastic energy
        e_el_new = 0.0
        for edge in edges:
            v0, v1 = edge
            e_el_new += 0.5 * k_s * (np.linalg.norm(V_new[v1] - V_new[v0]) - L0)**2
        for i in range(M):
            v_prev = (i - 1) % M
            v_curr = i
            v_next = (i + 1) % M
            e_el_new += 0.5 * k_b * np.sum((V_new[v_prev] - 2.0 * V_new[v_curr] + V_new[v_next])**2)
            
        # Barrier energy
        V_tiled_new = np.vstack([V_new + offset[None, :] for offset in offsets])
        collisions_new = ipctk.NormalCollisions()
        collisions_new.build(mesh_tiled, V_tiled_new, dhat)
        barrier_new = ipctk.BarrierPotential(dhat)
        e_col_new = barrier_new(collisions_new, mesh_tiled, V_tiled_new) * k_c
        
        e_new = e_el_new + e_col_new
        print(f"  [Line Search] Iter {search_iter}: alpha = {alpha:.6e} | E_cand = {e_new:.6e} (Elastic: {e_el_new:.6e}, Barrier: {e_col_new:.6e})")

        if e_new < e:
            success = True
            ctrl_rows = new_ctrl_rows
            print(f"[Sim Step] SUCCESS: alpha = {alpha:.6e} | E_final: {e_new:.6e} (decreased by {e - e_new:.6e})")
            break
        alpha *= tau
        
    if not success:
        print(f"[Sim Step] FAILED: line search could not find energy decrease.")

    return ctrl_rows


def eval_energy(flat_P, ctrl_rows, D, config, k_s, k_b, k_c, dhat):
    num_ctrl_rows = len(ctrl_rows)
    ctrl_offset = 0
    perturbed_ctrl_rows = []
    flat_P_reshaped = flat_P.reshape(-1, 3)
    for r in range(num_ctrl_rows):
        n_c = len(ctrl_rows[r])
        perturbed_ctrl_rows.append(flat_P_reshaped[ctrl_offset:ctrl_offset+n_c])
        ctrl_offset += n_c
        
    res = config["knit_parameters"]["loop_res"]
    bitmap_width = float(D[0])
    nout = res * int(round(bitmap_width)) + 1
    
    V_list = []
    for cp in perturbed_ctrl_rows:
        pts = eval_centerline(cp, D, nout)
        V_list.append(pts)
    V = np.vstack(V_list)
    M = len(V)
    
    edges_list = []
    row_offset = 0
    for r in range(num_ctrl_rows):
        row_edges = np.array([[i, i+1] for i in range(nout - 1)], dtype=np.int32) + row_offset
        edges_list.append(row_edges)
        row_offset += nout
    edges = np.vstack(edges_list)
    
    L0 = bitmap_width / (nout - 1)
    
    e_el = 0.0
    for edge in edges:
        v0, v1 = edge
        e_el += 0.5 * k_s * (np.linalg.norm(V[v1] - V[v0]) - L0)**2
    for i in range(M):
        v_prev = (i - 1) % M
        v_curr = i
        v_next = (i + 1) % M
        e_el += 0.5 * k_b * np.sum((V[v_prev] - 2.0 * V[v_curr] + V[v_next])**2)
        
    offsets = []
    for c_x in [-1, 0, 1]:
        for c_y in [-1, 0, 1]:
            offsets.append(c_x * np.array([D[0], 0.0, 0.0]) + c_y * np.array([0.0, D[1], 0.0]))
    V_tiled = np.vstack([V + offset[None, :] for offset in offsets])
    edges_tiled = np.vstack([edges + c * M for c in range(9)])
    mesh_tiled = ipctk.CollisionMesh(V_tiled, edges_tiled, np.empty((0, 3), dtype=np.int32))
    mesh_tiled.can_collide = build_can_collide(M, edges, D)
    
    collisions = ipctk.NormalCollisions()
    collisions.build(mesh_tiled, V_tiled, dhat)
    barrier = ipctk.BarrierPotential(dhat)
    e_col = barrier(collisions, mesh_tiled, V_tiled) * k_c
    
    return e_el + e_col


def check_gradients_and_hessians_fd(ctrl_rows, D, config, J_cached, k_s, k_b, k_c, dhat, eps=1e-5):
    flat_P = np.concatenate(ctrl_rows).astype(float).flatten()
    N = flat_P.size
    
    res = config["knit_parameters"]["loop_res"]
    bitmap_width = float(D[0])
    nout = res * int(round(bitmap_width)) + 1
    
    V_list = []
    for row_idx, cp in enumerate(ctrl_rows):
        pts = eval_centerline(cp, D, nout)
        V_list.append(pts)
    V = np.vstack(V_list)
    M = len(V)
    
    edges_list = []
    row_offset = 0
    for r in range(len(ctrl_rows)):
        row_edges = np.array([[i, i+1] for i in range(nout - 1)], dtype=np.int32) + row_offset
        edges_list.append(row_edges)
        row_offset += nout
    edges = np.vstack(edges_list)
    
    L0 = bitmap_width / (nout - 1)
    
    # 1. Analytical Evaluation
    import ipctk
    e_el, g_el, H_el = compute_elastic_forces_and_hessian(V, edges, L0, k_s, k_b, project_psd=False)
    e_col, g_col, H_col = compute_collision_forces_and_hessian(V, D, edges, dhat, k_c, psd_projection=ipctk.PSDProjectionMethod.NONE)
    
    g_V = g_el + g_col
    H_V = H_el + H_col
    
    g_P_anal = J_cached.T @ g_V
    H_P_anal = (J_cached.T @ H_V @ J_cached).toarray()
    
    def get_gradient_at(P_val):
        P_val_reshaped = P_val.reshape(-1, 3)
        perturbed_rows = []
        ctrl_offset = 0
        for r in range(len(ctrl_rows)):
            n_c = len(ctrl_rows[r])
            perturbed_rows.append(P_val_reshaped[ctrl_offset:ctrl_offset+n_c])
            ctrl_offset += n_c
            
        V_new_list = []
        for cp in perturbed_rows:
            pts = eval_centerline(cp, D, nout)
            V_new_list.append(pts)
        V_new = np.vstack(V_new_list)
        
        _, g_el_new, _ = compute_elastic_forces_and_hessian(V_new, edges, L0, k_s, k_b)
        _, g_col_new, _ = compute_collision_forces_and_hessian(V_new, D, edges, dhat, k_c)
        
        g_V_new = g_el_new + g_col_new
        return J_cached.T @ g_V_new

    # 2. Numerical Gradient (central difference)
    g_num = np.zeros(N)
    for i in range(N):
        P_plus = flat_P.copy()
        P_plus[i] += eps
        E_plus = eval_energy(P_plus, ctrl_rows, D, config, k_s, k_b, k_c, dhat)
        
        P_minus = flat_P.copy()
        P_minus[i] -= eps
        E_minus = eval_energy(P_minus, ctrl_rows, D, config, k_s, k_b, k_c, dhat)
        
        if np.isinf(E_plus) or np.isinf(E_minus):
            g_num[i] = 0.0
        else:
            g_num[i] = (E_plus - E_minus) / (2.0 * eps)
        
    # 3. Numerical Hessian (central difference of gradient)
    H_num = np.zeros((N, N))
    for j in range(N):
        P_plus = flat_P.copy()
        P_plus[j] += eps
        g_plus = get_gradient_at(P_plus)
        
        P_minus = flat_P.copy()
        P_minus[j] -= eps
        g_minus = get_gradient_at(P_minus)
        
        H_num[:, j] = (g_plus - g_minus) / (2.0 * eps)
        
    # Comparisons
    grad_diff = np.abs(g_P_anal - g_num)
    max_grad_diff = np.max(grad_diff)
    mean_grad_diff = np.mean(grad_diff)
    
    hess_diff = np.abs(H_P_anal - H_num)
    max_hess_diff = np.max(hess_diff)
    mean_hess_diff = np.mean(hess_diff)
    
    result_str = (
        f"[FD Check] Gradient difference: max={max_grad_diff:.6e}, mean={mean_grad_diff:.6e}\n"
        f"[FD Check] Hessian difference: max={max_hess_diff:.6e}, mean={mean_hess_diff:.6e}"
    )
    print(result_str)
    return result_str


def build_can_collide(M, edges, D):
    import ipctk
    num_rows = M - len(edges)
    nout = M // num_rows
    
    row_starts = [r * nout for r in range(num_rows + 1)]
    
    grid = []
    for c_x in [-1, 0, 1]:
        for c_y in [-1, 0, 1]:
            grid.append((c_x, c_y))
            
    def get_lattice_coords(v):
        c = v // M
        u = v % M
        c_x, c_y = grid[c]
        r = u // nout
        i = u % nout
        if i == nout - 1:
            return (r + c_y, 0, c_x + 1)
        else:
            return (r + c_y, i, c_x)
            
    keys = [get_lattice_coords(v) for v in range(9 * M)]
    explicit_values = {}
    for v1 in range(9 * M):
        k1 = keys[v1]
        r1, i1, px1 = k1
        for v2 in range(v1 + 1, 9 * M):
            k2 = keys[v2]
            r2, i2, px2 = k2
            if k1 == k2:
                explicit_values[(v1, v2)] = False
            elif r1 == r2 and px1 == px2 and abs(i1 - i2) <= 1:
                explicit_values[(v1, v2)] = False
                
    return ipctk.SparseCanCollide(explicit_values, True)
