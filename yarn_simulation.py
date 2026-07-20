import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import ipctk

from knitting_core import evaluate_centerlines

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
        self.V = None

    def update(self, V: np.ndarray):
        """Updates the internal centerline positions and precomputes/caches any dependent structures."""
        self.V = np.asarray(V, dtype=float)

    def value(self) -> float:
        """Returns the scalar energy value of the objective."""
        raise NotImplementedError

    def gradient(self) -> np.ndarray:
        """Analytical gradient."""
        return self.numerical_gradient()

    def hessian(self) -> scipy.sparse.csr_matrix:
        """Analytical Hessian."""
        return self.numerical_hessian()

    def numerical_gradient(self, eps: float = 1e-5) -> np.ndarray:
        """Computes the numerical gradient via central differences."""
        V_orig = self.V.copy()
        M = V_orig.shape[0]
        N_vars = M * 3
        flat_V = V_orig.flatten()
        grad = np.zeros(N_vars)
        for i in range(N_vars):
            V_plus = flat_V.copy()
            V_plus[i] += eps
            self.update(V_plus.reshape(M, 3))
            val_plus = self.value()
            
            V_minus = flat_V.copy()
            V_minus[i] -= eps
            self.update(V_minus.reshape(M, 3))
            val_minus = self.value()
            
            grad[i] = (val_plus - val_minus) / (2.0 * eps)
        
        self.update(V_orig)
        return grad.reshape(M, 3)

    def numerical_hessian(self, eps: float = 1e-5) -> scipy.sparse.csr_matrix:
        """Computes the numerical Hessian via central differences of gradient."""
        V_orig = self.V.copy()
        M = V_orig.shape[0]
        N_vars = M * 3
        flat_V = V_orig.flatten()
        
        H_dense = np.zeros((N_vars, N_vars))
        for j in range(N_vars):
            V_plus = flat_V.copy()
            V_plus[j] += eps
            self.update(V_plus.reshape(M, 3))
            g_plus = self.gradient().flatten()
            
            V_minus = flat_V.copy()
            V_minus[j] -= eps
            self.update(V_minus.reshape(M, 3))
            g_minus = self.gradient().flatten()
            
            H_dense[:, j] = (g_plus - g_minus) / (2.0 * eps)
            
        self.update(V_orig)
        return scipy.sparse.csr_matrix(H_dense)

    def check_gradient(self, eps: float = 1e-5) -> tuple[float, float]:
        g_anal = self.gradient()
        g_num = self.numerical_gradient(eps)
        diff = np.abs(g_anal - g_num)
        return float(np.max(diff)), float(np.mean(diff))

    def check_hessian(self, eps: float = 1e-5) -> tuple[float, float]:
        H_anal = self.hessian().toarray()
        H_num = self.numerical_hessian(eps).toarray()
        diff = np.abs(H_anal - H_num)
        return float(np.max(diff)), float(np.mean(diff))


class TotalObjective(Objective):
    def __init__(self, objective_weight_pairs: list[tuple[Objective, float]]):
        self.objective_weight_pairs = objective_weight_pairs
        self.V = None

    def update(self, V: np.ndarray):
        self.V = np.asarray(V, dtype=float)
        for obj, _ in self.objective_weight_pairs:
            obj.update(self.V)

    def value(self) -> float:
        return sum(w * obj.value() for obj, w in self.objective_weight_pairs)

    def gradient(self) -> np.ndarray:
        grad = np.zeros_like(self.V)
        for obj, w in self.objective_weight_pairs:
            grad += w * obj.gradient()
        return grad

    def hessian(self) -> scipy.sparse.csr_matrix:
        total_H = None
        for obj, w in self.objective_weight_pairs:
            h = w * obj.hessian()
            if total_H is None:
                total_H = h
            else:
                total_H = total_H + h
        return total_H


class StretchObjective(Objective):
    def __init__(self, edges, L0_array, period_offset_x, period_offset_y, nout, dhat, project_psd=True):
        super().__init__(edges, L0_array, period_offset_x, period_offset_y, nout, dhat)
        self.project_psd = project_psd
        self.diff = None
        self.lengths = None

    def update(self, V: np.ndarray):
        super().update(V)
        v0 = self.edges[:, 0]
        v1 = self.edges[:, 1]
        self.diff = self.V[v1] - self.V[v0]
        self.lengths = np.linalg.norm(self.diff, axis=1)

    def value(self) -> float:
        return float(0.5 * np.sum((self.lengths - self.L0_array) ** 2))

    def gradient(self) -> np.ndarray:
        grad = np.zeros_like(self.V)
        v0 = self.edges[:, 0]
        v1 = self.edges[:, 1]
        
        valid = self.lengths > 1e-8
        u = np.zeros_like(self.diff)
        u[valid] = self.diff[valid] / self.lengths[valid][:, None]
        
        g = (self.lengths - self.L0_array)[:, None] * u
        np.add.at(grad, v1, g)
        np.add.at(grad, v0, -g)
        return grad

    def hessian(self) -> scipy.sparse.csr_matrix:
        M = self.V.shape[0]
        v0 = self.edges[:, 0]
        v1 = self.edges[:, 1]
        valid = self.lengths > 1e-8
        E = len(self.edges)
        
        u = np.zeros((E, 3))
        u[valid] = self.diff[valid] / self.lengths[valid][:, None]
        
        uuT = u[:, :, None] * u[:, None, :]  # shape (E, 3, 3)
        
        lam = np.zeros(E)
        lam[valid] = (self.lengths[valid] - self.L0_array[valid]) / self.lengths[valid]
        if self.project_psd:
            lam = np.maximum(0.0, lam)
            
        H_local = uuT + lam[:, None, None] * (np.eye(3)[None, :, :] - uuT)  # shape (E, 3, 3)
        
        # Broadcast block row/col indices
        r_idx = np.arange(3)
        rows_u = np.broadcast_to((3 * v0)[:, None, None] + r_idx[None, :, None], (E, 3, 3))
        cols_u = np.broadcast_to((3 * v0)[:, None, None] + r_idx[None, None, :], (E, 3, 3))
        rows_w = np.broadcast_to((3 * v1)[:, None, None] + r_idx[None, :, None], (E, 3, 3))
        cols_w = np.broadcast_to((3 * v1)[:, None, None] + r_idx[None, None, :], (E, 3, 3))
        
        I = np.concatenate([
            rows_u.ravel(),
            rows_w.ravel(),
            rows_u.ravel(),
            rows_w.ravel()
        ])
        J = np.concatenate([
            cols_u.ravel(),
            cols_w.ravel(),
            cols_w.ravel(),
            cols_u.ravel()
        ])
        V_coeffs = np.concatenate([
            H_local.ravel(),
            H_local.ravel(),
            -H_local.ravel(),
            -H_local.ravel()
        ])
        
        return scipy.sparse.coo_matrix((V_coeffs, (I, J)), shape=(3*M, 3*M)).tocsr()


class BendObjective(Objective):
    def __init__(self, edges, L0_array, period_offset_x, period_offset_y, nout, dhat):
        super().__init__(edges, L0_array, period_offset_x, period_offset_y, nout, dhat)
        self._cached_hessian = self._compute_constant_hessian()
        self.lap_internal = None
        self.lap_bound = None

    def _compute_constant_hessian(self) -> scipy.sparse.csr_matrix:
        num_ctrl_rows = len(self.edges) // (self.nout - 1)
        M = num_ctrl_rows * self.nout
        rows = []
        cols = []
        vals = []
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

    def update(self, V: np.ndarray):
        super().update(V)
        M = self.V.shape[0]
        num_ctrl_rows = M // self.nout
        V_reshaped = self.V.reshape(num_ctrl_rows, self.nout, 3)
        
        # Internal stencils
        self.lap_internal = V_reshaped[:, :-2] - 2.0 * V_reshaped[:, 1:-1] + V_reshaped[:, 2:]
        # Boundary stencils
        self.lap_bound = (V_reshaped[:, -2] - self.period_offset_x[None, :]) - 2.0 * V_reshaped[:, 0] + V_reshaped[:, 1]

    def value(self) -> float:
        return float(0.5 * np.sum(self.lap_internal**2) + 0.5 * np.sum(self.lap_bound**2))

    def gradient(self) -> np.ndarray:
        M = self.V.shape[0]
        num_ctrl_rows = M // self.nout
        
        grad_V = np.zeros((num_ctrl_rows, self.nout, 3))
        # Accumulate internal contributions
        grad_V[:, :-2] += self.lap_internal
        grad_V[:, 1:-1] -= 2.0 * self.lap_internal
        grad_V[:, 2:] += self.lap_internal
        # Accumulate boundary contributions
        grad_V[:, -2] += self.lap_bound
        grad_V[:, 0] -= 2.0 * self.lap_bound
        grad_V[:, 1] += self.lap_bound
        
        return grad_V.reshape(M, 3)

    def hessian(self) -> scipy.sparse.csr_matrix:
        return self._cached_hessian


class CollisionObjective(Objective):
    def __init__(self, edges, L0_array, period_offset_x, period_offset_y, nout, dhat, psd_projection=None):
        super().__init__(edges, L0_array, period_offset_x, period_offset_y, nout, dhat)
        self.psd_projection = psd_projection if psd_projection is not None else ipctk.PSDProjectionMethod.ABS
        
        # Precompute constants and cache indices
        num_ctrl_rows = len(self.edges) // (self.nout - 1)
        self.M = num_ctrl_rows * self.nout
        
        c_x_range = [-1, 0, 1] if np.linalg.norm(self.period_offset_x) > 1e-6 else [0]
        c_y_range = [-1, 0, 1] if np.linalg.norm(self.period_offset_y) > 1e-6 else [0]
        self.grid = [(c_x, c_y) for c_y in c_y_range for c_x in c_x_range]
        self.offsets = [c_x * self.period_offset_x + c_y * self.period_offset_y for c_x, c_y in self.grid]
        self.num_copies = len(self.offsets)
        
        # Cache tiled edges and can_collide matrix
        self.edges_tiled = np.vstack([self.edges + c * self.M for c in range(self.num_copies)])
        self.can_collide = build_can_collide(self.M, self.edges, self.period_offset_x, self.period_offset_y)
        
        self.V_tiled = None
        self.mesh_tiled = None
        self.collisions = None

    def update(self, V: np.ndarray):
        super().update(V)
        self.V_tiled = np.vstack([self.V + offset[None, :] for offset in self.offsets])
        self.mesh_tiled = ipctk.CollisionMesh(self.V_tiled, self.edges_tiled, np.empty((0, 3), dtype=np.int32))
        self.mesh_tiled.can_collide = self.can_collide
        
        self.collisions = ipctk.NormalCollisions()
        self.collisions.build(self.mesh_tiled, self.V_tiled, self.dhat)
        
        # Stash full tiled geometry for the debug overlay
        global _debug_sim_geometry, _debug_collisions
        _debug_sim_geometry = (self.V_tiled, self.edges_tiled)
        
        # Extract collision positions
        collision_pts = []
        try:
            for ev in getattr(self.collisions, "ev_collisions", []):
                v_idx = int(getattr(ev, "vertex_id", -1))
                if 0 <= v_idx < len(self.V_tiled):
                    collision_pts.append(self.V_tiled[v_idx])
            for ee in getattr(self.collisions, "ee_collisions", []):
                e0 = int(getattr(ee, "edge0_id", -1))
                e1 = int(getattr(ee, "edge1_id", -1))
                if 0 <= e0 < len(self.edges_tiled) and 0 <= e1 < len(self.edges_tiled):
                    v0_0 = self.V_tiled[self.edges_tiled[e0, 0]]
                    v0_1 = self.V_tiled[self.edges_tiled[e0, 1]]
                    v1_0 = self.V_tiled[self.edges_tiled[e1, 0]]
                    v1_1 = self.V_tiled[self.edges_tiled[e1, 1]]
                    pt = 0.25 * (v0_0 + v0_1 + v1_0 + v1_1)
                    collision_pts.append(pt)
            for vv in getattr(self.collisions, "vv_collisions", []):
                v0 = int(getattr(vv, "vertex0_id", -1))
                v1 = int(getattr(vv, "vertex1_id", -1))
                if 0 <= v0 < len(self.V_tiled) and 0 <= v1 < len(self.V_tiled):
                    pt = 0.5 * (self.V_tiled[v0] + self.V_tiled[v1])
                    collision_pts.append(pt)
        except Exception as e:
            print(f"[Sim Debug] Error extracting collision positions: {e}")

        if collision_pts:
             _debug_collisions = np.vstack(collision_pts)
        else:
             _debug_collisions = np.empty((0, 3), dtype=np.float32)

    def value(self) -> float:
        barrier = ipctk.BarrierPotential(self.dhat, 1.0)
        return float(barrier(self.collisions, self.mesh_tiled, self.V_tiled))

    def gradient(self) -> np.ndarray:
        barrier = ipctk.BarrierPotential(self.dhat, 1.0)
        barrier_grad = barrier.gradient(self.collisions, self.mesh_tiled, self.V_tiled)
        barrier_grad_reshaped = barrier_grad.reshape(self.num_copies, self.M, 3)
        return np.sum(barrier_grad_reshaped, axis=0)

    def hessian(self) -> scipy.sparse.csr_matrix:
        barrier = ipctk.BarrierPotential(self.dhat, 1.0)
        barrier_hess = barrier.hessian(self.collisions, self.mesh_tiled, self.V_tiled, self.psd_projection)
        barrier_hess_coo = barrier_hess.tocoo()
        folded_rows = barrier_hess_coo.row % (3 * self.M)
        folded_cols = barrier_hess_coo.col % (3 * self.M)
        folded_vals = barrier_hess_coo.data
        return scipy.sparse.coo_matrix((folded_vals, (folded_rows, folded_cols)), shape=(3*self.M, 3*self.M)).tocsr()


_ccd_cache = {}


def get_ccd_topology(nout, num_ctrl_rows, period_offset_x, period_offset_y, M):
    key = (nout, num_ctrl_rows, float(np.linalg.norm(period_offset_x)), float(np.linalg.norm(period_offset_y)))
    if key in _ccd_cache:
        return _ccd_cache[key]
    
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

    # Rebuild row edges without the now-absent endpoint
    edges_open_list = []
    for r in range(num_ctrl_rows):
        base = r * nout_open
        edges_open_list.append(np.array([[base + i, base + i + 1] for i in range(nout_open - 1)], dtype=np.int32))
    edges_open = np.vstack(edges_open_list) if edges_open_list else np.empty((0, 2), dtype=np.int32)

    edges_tiled = np.vstack([edges_open + c * M_open for c in range(num_copies)])
    can_collide = build_can_collide(M_open, edges_open, period_offset_x, period_offset_y)
    
    res = {
        "offsets": offsets,
        "keep": keep,
        "edges_tiled": edges_tiled,
        "can_collide": can_collide
    }
    _ccd_cache[key] = res
    return res


def run_simulation_step(ctrl_rows, period_offset_x, period_offset_y, config, J_cached, L0_array, k_s, k_b, k_c, dhat):
    import time
    t_start = time.perf_counter()
    print("HI!!!!!!")
    debug = True
    res = config["knit_parameters"]["loop_res"]
    bitmap_width = float(np.linalg.norm(period_offset_x))
    nout = res * int(round(bitmap_width)) + 1
    
    flat_P = np.concatenate(ctrl_rows).astype(float)
    num_ctrl_rows = len(ctrl_rows)
    
    t0 = time.perf_counter()
    V, edges, D, nout = evaluate_centerlines(ctrl_rows, period_offset_x, config)
    M = len(V)
    t_geom = time.perf_counter() - t0
    
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
    collision = CollisionObjective(**kwargs, psd_projection=ipctk.PSDProjectionMethod.CLAMP)

    total_obj = TotalObjective([(stretch, k_s), (bend, k_b), (collision, k_c)])
    total_obj.update(V)

    e_el_s, e_b_s, e_col_s = eval_energy(flat_P, ctrl_rows, period_offset_x, period_offset_y, config, L0_array, dhat)
    e = total_obj.value()

    # Project to control point space using the cached Jacobian matrix J_cached
    t0 = time.perf_counter()
    g_V = total_obj.gradient()
    H_V = total_obj.hessian()
    t_objectives = time.perf_counter() - t0

    t0 = time.perf_counter()
    g_V_flat = g_V.flatten()
    g_P = J_cached.T @ g_V_flat
    H_P = J_cached.T @ H_V @ J_cached
    
    # Regularize to guarantee positive definiteness
    H_P_dense = H_P.toarray()
    H_pd = H_P_dense + 1e-6 * np.eye(H_P_dense.shape[0])
    
    delta_P = np.linalg.solve(H_pd, -g_P)
    t_solve = time.perf_counter() - t0
    
    # Check for descent direction
    dot_prod = np.dot(g_P, delta_P)
    if dot_prod >= 0.0:
        raise ValueError(f"Solver computed an ascent direction: g_P^T delta_P = {dot_prod:.6e} >= 0. The search direction must be a descent direction.")
    
    # Debug prints for optimization step
    g_norm = np.linalg.norm(g_P)
    if debug:
        print(f"[Sim Step] E_start: {e:.6e} (Elastic: {e_el_s * k_s:.6e}, Bending: {e_b_s * k_b:.6e}, Barrier: {e_col_s * k_c:.6e}) | ||g_P||: {g_norm:.6e} | g_P^T delta_P: {dot_prod:.6e}")

    # 4. CCD Step Size Calculation on 3x3 tiled copies
    t0 = time.perf_counter()
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
    ccd_topo = get_ccd_topology(nout, num_ctrl_rows, period_offset_x, period_offset_y, M)
    offsets = ccd_topo["offsets"]
    keep = ccd_topo["keep"]
    edges_tiled = ccd_topo["edges_tiled"]
    can_collide_ccd = ccd_topo["can_collide"]

    V_open      = V[keep]
    V_cand_open = V_cand[keep]

    V_tiled_0 = np.vstack([V_open      + offset[None, :] for offset in offsets])
    V_tiled_1 = np.vstack([V_cand_open + offset[None, :] for offset in offsets])

    mesh_tiled  = ipctk.CollisionMesh(V_tiled_0, edges_tiled, np.empty((0, 3), dtype=np.int32))
    mesh_tiled.can_collide = can_collide_ccd

    alpha_max = ipctk.compute_collision_free_stepsize(mesh_tiled, V_tiled_0, V_tiled_1)
    t_ccd = time.perf_counter() - t0
    if debug:
        print(f"[Sim Step] alpha_max (CCD): {alpha_max:.6f}")

    # 5. Backtracking Line Search
    t0 = time.perf_counter()
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
        
        e_el_new, e_b_new, e_col_new = eval_energy(P_new.flatten(), new_ctrl_rows, period_offset_x, period_offset_y, config, L0_array, dhat)
        e_new = k_s * e_el_new + k_b * e_b_new + k_c * e_col_new
        if debug:
            print(f"  [Line Search] Iter {search_iter}: alpha = {alpha:.6e} | E_cand = {e_new:.6e} (Elastic: {k_s * e_el_new:.6e}, Bending: {k_b * e_b_new:.6e}, Barrier: {k_c * e_col_new:.6e})")

        if e_new < e:
            success = True
            ctrl_rows = new_ctrl_rows
            if debug:
                print(f"[Sim Step] SUCCESS: alpha = {alpha:.6e} | E_final: {e_new:.6e} (decreased by {e - e_new:.6e})")
            break
        alpha *= tau
        
    if not success:
        if debug:
            print(f"[Sim Step] FAILED: line search could not find energy decrease.")
            
    t_ls = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start
    print(f"[Sim Timing] Geom: {t_geom:.4f}s | Objectives: {t_objectives:.4f}s | Solve: {t_solve:.4f}s | CCD: {t_ccd:.4f}s | LineSearch: {t_ls:.4f}s | Total: {t_total:.4f}s")

    return ctrl_rows


def eval_energy(flat_P, ctrl_rows, period_offset_x, period_offset_y, config, L0_array, dhat, filter_collisions=True):
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
    collision = CollisionObjective(**kwargs)
    
    stretch.update(V)
    bend.update(V)
    collision.update(V)
    
    e_el = stretch.value()
    e_b = bend.value()
    e_col = collision.value()
        
    return e_el, e_b, e_col


# %% HEADLESS FINITE DIFFERENCE VERIFICATION ──────────────────────────────────────────
def check_gradients_and_hessians_fd(ctrl_rows, period_offset_x, period_offset_y, config, J_cached, k_s, k_b, k_c, dhat, eps=1e-4):
    V, edges, D, nout = evaluate_centerlines(ctrl_rows, period_offset_x, config)
    
    kwargs = {
        "edges": edges,
        "L0_array": np.full(len(edges), D[0] / (nout - 1), dtype=float),
        "period_offset_x": period_offset_x,
        "period_offset_y": period_offset_y,
        "nout": nout,
        "dhat": dhat
    }
    
    stretch = StretchObjective(**kwargs, project_psd=False)
    bend = BendObjective(**kwargs)
    collision = CollisionObjective(**kwargs, psd_projection=ipctk.PSDProjectionMethod.NONE)
    
    result_lines = []
    for name, obj in [("stretch", stretch), ("bend", bend), ("collision", collision)]:
        obj.update(V)
        g_anal = obj.gradient().flatten()
        g_num = obj.numerical_gradient(eps).flatten()
        H_anal = obj.hessian().toarray()
        H_num = obj.numerical_hessian(eps).toarray()
        
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


_can_collide_cache = {}


def build_can_collide(M, edges, period_offset_x, period_offset_y):
    key = (M, len(edges), float(np.linalg.norm(period_offset_x)), float(np.linalg.norm(period_offset_y)))
    if key in _can_collide_cache:
        return _can_collide_cache[key]

    num_rows = M - len(edges)
    nout = M // num_rows

    c_x_range = [-1, 0, 1] if np.linalg.norm(period_offset_x) > 1e-6 else [0]
    c_y_range = [-1, 0, 1] if np.linalg.norm(period_offset_y) > 1e-6 else [0]
    grid = [(c_x, c_y) for c_y in c_y_range for c_x in c_x_range]

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

    res = ipctk.make_sparse_filter(explicit_values, True)
    _can_collide_cache[key] = res
    return res
