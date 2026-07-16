import time
import json
import numpy as np
import scipy.sparse
import ipctk

# Import new optimized objectives and step runner
from yarn_simulation import (
    Objective,
    StretchObjective,
    BendObjective,
    CollisionObjective,
    build_can_collide,
    run_simulation_step,
    eval_energy
)
from knitting_core import evaluate_centerlines, build_row_spline_jacobian

# ──────────────────────────────────────────────────────────────────────────────
# LEGACY (OLD) OBJECTIVE IMPLEMENTATIONS
# ──────────────────────────────────────────────────────────────────────────────

class OldStretchObjective(Objective):
    def __init__(self, edges, L0_array, period_offset_x, period_offset_y, nout, dhat, project_psd=True):
        super().__init__(edges, L0_array, period_offset_x, period_offset_y, nout, dhat)
        self.project_psd = project_psd

    def update(self, V: np.ndarray):
        super().update(V)

    def value(self) -> float:
        V = self.V
        v0_pts = V[self.edges[:, 0]]
        v1_pts = V[self.edges[:, 1]]
        lengths = np.linalg.norm(v1_pts - v0_pts, axis=1)
        return float(0.5 * np.sum((lengths - self.L0_array) ** 2))

    def gradient(self) -> np.ndarray:
        V = self.V
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

    def hessian(self) -> scipy.sparse.csr_matrix:
        V = self.V
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


class OldBendObjective(Objective):
    def update(self, V: np.ndarray):
        super().update(V)

    def value(self) -> float:
        V = self.V
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

    def gradient(self) -> np.ndarray:
        V = self.V
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

    def hessian(self) -> scipy.sparse.csr_matrix:
        V = self.V
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


class OldCollisionObjective(Objective):
    def __init__(self, edges, L0_array, period_offset_x, period_offset_y, nout, dhat, psd_projection=None):
        super().__init__(edges, L0_array, period_offset_x, period_offset_y, nout, dhat)
        self.psd_projection = psd_projection if psd_projection is not None else ipctk.PSDProjectionMethod.CLAMP

    def update(self, V: np.ndarray):
        super().update(V)

    def value(self) -> float:
        V = self.V
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
        collisions = ipctk.NormalCollisions()
        collisions.build(mesh_tiled, V_tiled, self.dhat)
        barrier = ipctk.BarrierPotential(self.dhat)
        return float(barrier(collisions, mesh_tiled, V_tiled))

    def gradient(self) -> np.ndarray:
        V = self.V
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
        barrier = ipctk.BarrierPotential(self.dhat)
        barrier_grad = barrier.gradient(collisions, mesh_tiled, V_tiled)
        grad_V = np.zeros((M, 3))
        barrier_grad_reshaped = barrier_grad.reshape(num_copies, M, 3)
        for c in range(num_copies):
            grad_V += barrier_grad_reshaped[c]
        return grad_V

    def hessian(self) -> scipy.sparse.csr_matrix:
        V = self.V
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
        barrier = ipctk.BarrierPotential(self.dhat)
        barrier_hess = barrier.hessian(collisions, mesh_tiled, V_tiled, self.psd_projection)
        barrier_hess_coo = barrier_hess.tocoo()
        folded_rows = barrier_hess_coo.row % (3 * M)
        folded_cols = barrier_hess_coo.col % (3 * M)
        folded_vals = barrier_hess_coo.data
        return scipy.sparse.coo_matrix((folded_vals, (folded_rows, folded_cols)), shape=(3*M, 3*M)).tocsr()


def old_eval_energy(flat_P, ctrl_rows, period_offset_x, period_offset_y, config, L0_array, dhat):
    num_ctrl_rows = len(ctrl_rows)
    ctrl_offset = 0
    perturbed_ctrl_rows = []
    flat_P_reshaped = flat_P.reshape(-1, 3)
    for r in range(num_ctrl_rows):
        n_c = len(ctrl_rows[r])
        perturbed_ctrl_rows.append(flat_P_reshaped[ctrl_offset:ctrl_offset+n_c])
        ctrl_offset += n_c
        
    V, edges, _, nout = evaluate_centerlines(perturbed_ctrl_rows, period_offset_x, config)

    kwargs = {
        "edges": edges,
        "L0_array": L0_array,
        "period_offset_x": period_offset_x,
        "period_offset_y": period_offset_y,
        "nout": nout,
        "dhat": dhat
    }
    
    stretch = OldStretchObjective(**kwargs)
    bend = OldBendObjective(**kwargs)
    collision = OldCollisionObjective(**kwargs)
    
    stretch.update(V)
    bend.update(V)
    collision.update(V)
    
    e_el = stretch.value()
    e_b = bend.value()
    e_col = collision.value()
        
    return e_el, e_b, e_col


def old_run_simulation_step(ctrl_rows, period_offset_x, period_offset_y, config, J_cached, L0_array, k_s, k_b, k_c, dhat):
    res = config["knit_parameters"]["loop_res"]
    bitmap_width = float(np.linalg.norm(period_offset_x))
    nout = res * int(round(bitmap_width)) + 1
    
    flat_P = np.concatenate(ctrl_rows).astype(float)
    num_ctrl_rows = len(ctrl_rows)
    
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
    
    stretch = OldStretchObjective(**kwargs)
    bend = OldBendObjective(**kwargs)
    collision = OldCollisionObjective(**kwargs, psd_projection=ipctk.PSDProjectionMethod.CLAMP)

    stretch.update(V)
    bend.update(V)
    collision.update(V)

    e_el_s, e_b_s, e_col_s = old_eval_energy(flat_P, ctrl_rows, period_offset_x, period_offset_y, config, L0_array, dhat)
    e = e_el_s + e_b_s + e_col_s

    # 2. Project to control point space using the cached Jacobian matrix J_cached
    g_V = k_s * stretch.gradient() + k_b * bend.gradient() + k_c * collision.gradient()
    H_V = k_s * stretch.hessian() + k_b * bend.hessian() + k_c * collision.hessian()
    
    g_V_flat = g_V.flatten()

    g_P = J_cached.T @ g_V_flat
    H_P = J_cached.T @ H_V @ J_cached
    
    # Regularize to guarantee positive definiteness
    H_P_dense = H_P.toarray()
    H_pd = H_P_dense + 1e-6 * np.eye(H_P_dense.shape[0])
    
    # 3. Solve for search direction
    try:
        delta_P = np.linalg.solve(H_pd, -g_P)
    except np.linalg.LinAlgError:
        return ctrl_rows  # skip step if singular
    
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

    # 5. Backtracking Line Search — delegates to old_eval_energy for correct, consistent energy
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
        
        e_el_new, e_b_new, e_col_new = old_eval_energy(P_new.flatten(), new_ctrl_rows, period_offset_x, period_offset_y, config, L0_array, dhat)
        e_new = e_el_new + e_b_new + e_col_new

        if e_new < e:
            success = True
            ctrl_rows = new_ctrl_rows
            break
        alpha *= tau
        
    return ctrl_rows


# ──────────────────────────────────────────────────────────────────────────────
# BENCHMARK RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmarks():
    print("Loading params.json and config.json...")
    with open("params.json", "r") as f:
        params_data = json.load(f)
    with open("config.json", "r") as f:
        config = json.load(f)

    ctrl_rows = [np.array(row, dtype=float) for row in params_data["spline_control_rows"]]
    period_offset_x = np.array(params_data["period_offset_x"], dtype=float)
    period_offset_y = np.array(params_data["period_offset_y"], dtype=float)
    dhat = 0.1

    print("Evaluating initial geometries...")
    V, edges, _, nout = evaluate_centerlines(ctrl_rows, period_offset_x, config)
    v0_pts = V[edges[:, 0]]
    v1_pts = V[edges[:, 1]]
    L0_array = np.linalg.norm(v1_pts - v0_pts, axis=1)

    print(f"Mesh size: {V.shape[0]} vertices, {edges.shape[0]} edges, {nout} samples per row.")

    kwargs = {
        "edges": edges,
        "L0_array": L0_array,
        "period_offset_x": period_offset_x,
        "period_offset_y": period_offset_y,
        "nout": nout,
        "dhat": dhat
    }

    # Instantiate objectives
    stretch_old = OldStretchObjective(**kwargs)
    stretch_new = StretchObjective(**kwargs)

    bend_old = OldBendObjective(**kwargs)
    bend_new = BendObjective(**kwargs)

    collision_old = OldCollisionObjective(**kwargs)
    collision_new = CollisionObjective(**kwargs)

    # Helper function to benchmark a call
    def benchmark_call(obj, method_name, V_in, runs=100):
        obj.update(V_in)
        func = getattr(obj, method_name)
        func()
        t0 = time.perf_counter()
        for _ in range(runs):
            obj.update(V_in)
            func()
        return (time.perf_counter() - t0) / runs

    print("\nBenchmarking StretchObjective (100 runs)...")
    t_val_stretch_old = benchmark_call(stretch_old, "value", V, 100)
    t_val_stretch_new = benchmark_call(stretch_new, "value", V, 100)
    t_grad_stretch_old = benchmark_call(stretch_old, "gradient", V, 100)
    t_grad_stretch_new = benchmark_call(stretch_new, "gradient", V, 100)
    t_hess_stretch_old = benchmark_call(stretch_old, "hessian", V, 100)
    t_hess_stretch_new = benchmark_call(stretch_new, "hessian", V, 100)

    print("Benchmarking BendObjective (100 runs)...")
    t_val_bend_old = benchmark_call(bend_old, "value", V, 100)
    t_val_bend_new = benchmark_call(bend_new, "value", V, 100)
    t_grad_bend_old = benchmark_call(bend_old, "gradient", V, 100)
    t_grad_bend_new = benchmark_call(bend_new, "gradient", V, 100)
    t_hess_bend_old = benchmark_call(bend_old, "hessian", V, 100)
    t_hess_bend_new = benchmark_call(bend_new, "hessian", V, 100)

    print("Benchmarking CollisionObjective (10 runs)...")
    t_val_col_old = benchmark_call(collision_old, "value", V, 10)
    t_val_col_new = benchmark_call(collision_new, "value", V, 10)
    t_grad_col_old = benchmark_call(collision_old, "gradient", V, 10)
    t_grad_col_new = benchmark_call(collision_new, "gradient", V, 10)
    t_hess_col_old = benchmark_call(collision_old, "hessian", V, 10)
    t_hess_col_new = benchmark_call(collision_new, "hessian", V, 10)

    # ──────────────────────────────────────────────────────────────────────────
    # FULL SIMULATION BENCHMARK
    # ──────────────────────────────────────────────────────────────────────────
    print("\nBenchmarking Full Optimization solver to convergence (10 iterations)...")
    loop_res = config["knit_parameters"]["loop_res"]
    
    def rebuild_jacobian(rows, px, res):
        J_blocks = []
        bitmap_width = float(np.linalg.norm(px))
        nout_cp = res * int(round(bitmap_width)) + 1
        for cp in rows:
            J_r = build_row_spline_jacobian(cp, px, nout_cp)
            J_blocks.append(J_r)
        J_base = scipy.sparse.block_diag(J_blocks, format="csr")
        return scipy.sparse.kron(J_base, scipy.sparse.identity(3), format="csr")

    J_cached = rebuild_jacobian(ctrl_rows, period_offset_x, loop_res)
    K_S, K_B, K_C = 1000.0, 10.0, 1.0

    # Old full solver
    ctrl_rows_old = [cp.copy() for cp in ctrl_rows]
    t0_old = time.perf_counter()
    for step in range(10):
        ctrl_rows_old = old_run_simulation_step(
            ctrl_rows_old, period_offset_x, period_offset_y, config, J_cached, L0_array, K_S, K_B, K_C, dhat
        )
    t_full_old = time.perf_counter() - t0_old

    # New full solver
    ctrl_rows_new = [cp.copy() for cp in ctrl_rows]
    t0_new = time.perf_counter()
    for step in range(10):
        ctrl_rows_new = run_simulation_step(
            ctrl_rows_new, period_offset_x, period_offset_y, config, J_cached, L0_array, K_S, K_B, K_C, dhat
        )
    t_full_new = time.perf_counter() - t0_new

    # Print results table
    print("\n" + "="*80)
    print("                      BENCHMARK RESULTS (Average Runtime)")
    print("="*80)
    print(f"{'Objective / Call':<30} | {'Old Code (s)':<14} | {'New Code (s)':<14} | {'Speedup':<10}")
    print("-"*80)
    
    def print_row(label, old, new):
        speedup = old / new if new > 0 else float('inf')
        print(f"{label:<30} | {old:>14.6f} | {new:>14.6f} | {speedup:>9.2f}x")

    print_row("Stretch Value", t_val_stretch_old, t_val_stretch_new)
    print_row("Stretch Gradient", t_grad_stretch_old, t_grad_stretch_new)
    print_row("Stretch Hessian", t_hess_stretch_old, t_hess_stretch_new)
    print("-" * 80)
    print_row("Bend Value", t_val_bend_old, t_val_bend_new)
    print_row("Bend Gradient", t_grad_bend_old, t_grad_bend_new)
    print_row("Bend Hessian", t_hess_bend_old, t_hess_bend_new)
    print("-" * 80)
    print_row("Collision Value", t_val_col_old, t_val_col_new)
    print_row("Collision Gradient", t_grad_col_old, t_grad_col_new)
    print_row("Collision Hessian", t_hess_col_old, t_hess_col_new)
    print("="*80)
    print_row("Full Optimization (10 steps)", t_full_old, t_full_new)
    print("="*80)

if __name__ == "__main__":
    run_benchmarks()
