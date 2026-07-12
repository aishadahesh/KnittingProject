# Yarn Simulation Pipeline Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the physical yarn simulation pipeline in `knitting_core.py` to use a clean, decoupled OOP Objective class hierarchy with identical inputs (`V` matrix) and a generic numerical finite difference validator in the base class, eliminating code duplication and fixing the FD check mismatch.

**Architecture:** 
1. Define a base `Objective` class in [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py) that implements default numerical finite-difference calculation of gradients and Hessians with respect to physical centerline positions `V` (M x 3).
2. Inherit three stateless concrete classes (`StretchObjective`, `BendObjective`, and `CollisionObjective`) implementing their value, gradient, and Hessian analytically.
3. Integrate these objectives in the solver pipeline in `run_simulation_step` and `eval_energy` by evaluating them on physical positions `V` and projecting their aggregated forces/Hessians back to control point space `P` using the cached Jacobian matrix `J_cached`.

**Tech Stack:** Python, NumPy, SciPy (Sparse matrices), IPCTK (Incremental Potential Contact Toolkit).

## Global Constraints
- **Stateless Core:** `knitting_core.py` must remain stateless; all objectives must store only constant parameters (stiffness scaled outside, connectivity, offsets) and their computation functions must be pure.
- **Uniform Signatures:** All methods (`value`, `gradient`, `hessian`) across all objectives must share the exact same signature accepting only `V` (matrix of positions) and optional `eps` parameter.
- **Direct Vertex Perturbations:** The finite difference checking must perturb the vertex coordinates `V` directly, avoiding spline parameterization/chord-length recalculation mismatches.

---

### Task 1: Implement the `Objective` Base Class

**Files:**
- Modify: [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py)

**Interfaces:**
- Produces: `Objective` base class with default numerical finite-difference methods `gradient` and `hessian`, and comparison helper methods `check_gradient` and `check_hessian`.

- [ ] **Step 1: Write base class definition**
  Define `Objective` class in `knitting_core.py` with standard constructor:
  ```python
  class Objective:
      def __init__(self, edges, L0_array, period_offset_x, period_offset_y, nout, dhat):
          self.edges = edges
          self.L0_array = L0_array
          self.period_offset_x = period_offset_x
          self.period_offset_y = period_offset_y
          self.nout = nout
          self.dhat = dhat

      def value(self, V: np.ndarray) -> float:
          raise NotImplementedError

      def gradient(self, V: np.ndarray, eps: float = 1e-5) -> np.ndarray:
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

      def hessian(self, V: np.ndarray, eps: float = 1e-5) -> scipy.sparse.csr_matrix:
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
          g_num = Objective.gradient(self, V, eps)
          diff = np.abs(g_anal - g_num)
          return float(np.max(diff)), float(np.mean(diff))

      def check_hessian(self, V: np.ndarray, eps: float = 1e-5) -> tuple[float, float]:
          H_anal = self.hessian(V).toarray()
          H_num = Objective.hessian(self, V, eps).toarray()
          diff = np.abs(H_anal - H_num)
          return float(np.max(diff)), float(np.mean(diff))
  ```

- [ ] **Step 2: Commit base class implementation**

---

### Task 2: Implement the `StretchObjective` Subclass

**Files:**
- Modify: [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py)

**Interfaces:**
- Consumes: `Objective` base class
- Produces: `StretchObjective` with analytical `value`, `gradient`, and `hessian` implementations.

- [ ] **Step 1: Write subclass definition**
  Define `StretchObjective` with optional `project_psd` parameter:
  ```python
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
  ```

- [ ] **Step 2: Commit subclass implementation**

---

### Task 3: Implement the `BendObjective` Subclass

**Files:**
- Modify: [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py)

**Interfaces:**
- Consumes: `Objective` base class
- Produces: `BendObjective` with analytical `value`, `gradient`, and `hessian` implementations.

- [ ] **Step 1: Write subclass definition**
  ```python
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
  ```

- [ ] **Step 2: Commit subclass implementation**

---

### Task 4: Implement the `CollisionObjective` Subclass

**Files:**
- Modify: [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py)

**Interfaces:**
- Consumes: `Objective` base class
- Produces: `CollisionObjective` with analytical `value`, `gradient`, and `hessian` implementations.

- [ ] **Step 1: Write subclass definition**
  Include `psd_projection` parameter (default `ipctk.PSDProjectionMethod.CLAMP`):
  ```python
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
          offsets = [c_x * self.period_offset_x + c_y * self.period_offset_y for c_y in c_y_range for c_x in c_x_range]
          num_copies = len(offsets)
          V_tiled = np.vstack([V + offset[None, :] for offset in offsets])
          edges_tiled = np.vstack([self.edges + c * M for c in range(num_copies)])
          mesh_tiled = ipctk.CollisionMesh(V_tiled, edges_tiled, np.empty((0, 3), dtype=np.int32))
          mesh_tiled.can_collide = build_can_collide(M, self.edges, self.period_offset_x, self.period_offset_y)
          collisions = ipctk.NormalCollisions()
          collisions.build(mesh_tiled, V_tiled, self.dhat)
          barrier = ipctk.BarrierPotential(self.dhat)
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
          barrier = ipctk.BarrierPotential(self.dhat)
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
          barrier = ipctk.BarrierPotential(self.dhat)
          barrier_hess = barrier.hessian(collisions, mesh_tiled, V_tiled, self.psd_projection)
          barrier_hess_coo = barrier_hess.tocoo()
          folded_rows = barrier_hess_coo.row % (3 * M)
          folded_cols = barrier_hess_coo.col % (3 * M)
          folded_vals = barrier_hess_coo.data
          return scipy.sparse.coo_matrix((folded_vals, (folded_rows, folded_cols)), shape=(3*M, 3*M)).tocsr()
  ```

- [ ] **Step 2: Commit subclass implementation**

---

### Task 5: Refactor Simulation Pipeline

**Files:**
- Modify: [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py)

**Interfaces:**
- Consumes: `StretchObjective`, `BendObjective`, `CollisionObjective`
- Produces: Refactored `run_simulation_step`, `eval_energy`, and `check_gradients_and_hessians_fd` using the new classes.
- Removes deprecated `compute_elastic_forces_and_hessian` and `compute_collision_forces_and_hessian` to reduce structural density.

- [ ] **Step 1: Rewrite pipeline helper integration**
  Update `eval_energy` and `run_simulation_step` to instantiate objectives and combine weights `k_s, k_b, k_c`:
  ```python
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
  ```

  Update `run_simulation_step`:
  ```python
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
      
      e_el_s, e_b_s, e_col_s = eval_energy(flat_P, ctrl_rows, period_offset_x, period_offset_y, config, L0_array, k_s, k_b, k_c, dhat)
      e = e_el_s + e_b_s + e_col_s

      g_P = J_cached.T @ g_V_flat
      H_P = J_cached.T @ H_V @ J_cached
      
      H_P_dense = H_P.toarray()
      H_pd = H_P_dense + 1e-6 * np.eye(H_P_dense.shape[0])
      
      try:
          delta_P = np.linalg.solve(H_pd, -g_P)
      except np.linalg.LinAlgError:
          print("[Sim Step] Solver FAILED: Hessian is singular.")
          return ctrl_rows
          
      # [CCD and backtracking lines search remains exactly identical, but uses the updated eval_energy and V_cand mapping]
      ...
  ```

  Update `check_gradients_and_hessians_fd`:
  ```python
  def check_gradients_and_hessians_fd(ctrl_rows, period_offset_x, period_offset_y, config, J_cached, k_s, k_b, k_c, dhat, eps=1e-8):
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
      
      rep_s = "Stretch: " + str(stretch.check_gradient(V, eps)) + " " + str(stretch.check_hessian(V, eps))
      rep_b = "Bend: " + str(bend.check_gradient(V, eps)) + " " + str(bend.check_hessian(V, eps))
      rep_c = "Collision: " + str(collision.check_gradient(V, eps)) + " " + str(collision.check_hessian(V, eps))
      
      return "\n".join([rep_s, rep_b, rep_c])
  ```

- [ ] **Step 2: Commit pipeline changes**

---

### Task 6: Update Sim Unit Tests

**Files:**
- Modify: [tests/test_knitting_core_sim.py](file:///home/roip/projects/KnittingProject/tests/test_knitting_core_sim.py)

**Interfaces:**
- Consumes: Refactored API in `knitting_core.py`

- [ ] **Step 1: Rewrite sim unit tests**
  Update `test_elastic_forces_and_hessian`, `test_collision_forces_and_hessian`, and `test_check_gradients_and_hessians_fd` to invoke the class-based objectives and their FD checkers.
  
- [ ] **Step 2: Verify tests**
  (Verify they pass/fail and work seamlessly).
