# Yarn Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a frame-by-frame, interactive Projected Newton physical simulation for the periodic yarn using the IPC Toolkit (ipctk), running asynchronously in a background thread and fully integrated into the imgui sidebar.

**Architecture:** 
1. Cache B-spline evaluation Jacobians $J_r$ at initialization or whenever user-edits occur.
2. Formulate elastic (stretch/bend) potentials analytically on the primary unit cell.
3. Compute barrier potential on a 3x3 tiled copies mesh for periodic boundary conditions, folding the gradient/Hessian down to the unit cell.
4. Perform a local-PSD-projected Newton step with Continuous Collision Detection (CCD) backtracking line search in a background thread.

**Tech Stack:** ipctk, scipy.sparse, numpy, Dear ImGui

## Global Constraints
* knitting_core.py MUST remain a stateless, purely functional module.
* Centralized State: All application state (locks, parameters, thread control flags) must reside in AppState.
* Pure Spline Geometry Pipeline: Use `eval_centerline` as the canonical spline evaluation model.
* Zero-Tolerance for "Schema/String Mapping": Map properties dynamically when binding state variables.

---

### Task 1: Core Spline Centerline Evaluator & Cacheable Jacobian

**Files:**
- Modify: [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py)
- Create: [tests/test_knitting_core_sim.py](file:///home/roip/projects/KnittingProject/tests/test_knitting_core_sim.py)

**Interfaces:**
- Produces: `eval_centerline(cp, D, nout, t=None, to=None)`
- Produces: `build_row_spline_jacobian(cp, D, nout)`

- [ ] **Step 1: Implement `eval_centerline` and `build_row_spline_jacobian`**
  Modify [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py) to add the helper functions and refactor `build_spline_mesh` to reuse `eval_centerline`.
  ```python
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
  ```

- [ ] **Step 2: Write tests for Spline Centerline Evaluation and Jacobian**
  Create [tests/test_knitting_core_sim.py](file:///home/roip/projects/KnittingProject/tests/test_knitting_core_sim.py):
  ```python
  import numpy as np
  from knitting_core import eval_centerline, build_row_spline_jacobian

  def test_spline_centerline_and_jacobian():
      cp = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.1], [2.0, -0.2, 0.3]])
      D = np.array([3.0, 0.0, 0.0])
      nout = 20
      pts = eval_centerline(cp, D, nout)
      assert pts.shape == (nout, 3)
      J = build_row_spline_jacobian(cp, D, nout)
      assert J.shape == (nout, len(cp))
      # Verify linearity: J @ Px + bx == Vx
      cp_aug = np.concatenate((cp, (cp[0] + D)[None, :]), axis=0)
      t = np.concatenate(([0.0], np.cumsum(np.maximum(np.linalg.norm(np.diff(cp_aug, axis=0), axis=1), 1e-6))))
      to = np.linspace(t[0], t[-1], nout)
      pts_zero_offset = eval_centerline(cp, np.zeros(3), nout, t=t, to=to)
      for dim in range(3):
          V_dim = pts_zero_offset[:, dim]
          P_dim = cp[:, dim]
          assert np.allclose(J @ P_dim, V_dim, atol=1e-5)
  ```

- [ ] **Step 3: Run the test to verify it passes**
  Run: `uv run pytest tests/test_knitting_core_sim.py -v`
  Expected: PASS

- [ ] **Step 4: Commit**
  ```bash
  git add knitting_core.py tests/test_knitting_core_sim.py
  git commit -m "feat: add spline centerline and cached Jacobian builder"
  ```

---

### Task 2: Analytical PSD Stretch & Bend Energies, Gradients & Hessians

**Files:**
- Modify: [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py)
- Modify: [tests/test_knitting_core_sim.py](file:///home/roip/projects/KnittingProject/tests/test_knitting_core_sim.py)

**Interfaces:**
- Consumes: None
- Produces: `compute_elastic_forces_and_hessian(V, edges, L0, k_s, k_b)` returning `energy`, `grad_V` (flat, size $3M$), and `H_V` (sparse `coo_matrix` or `csr_matrix`, size $3M \times 3M$).

- [ ] **Step 1: Implement analytical stretch/bend energies, gradients and local PSD projections**
  Add `compute_elastic_forces_and_hessian` to [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py).
  ```python
  import scipy.sparse

  def compute_elastic_forces_and_hessian(V, edges, L0, k_s, k_b):
      M = len(V)
      V_flat = V.reshape(-1)
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
          lam = max(0.0, k_s * (l - L0) / l)
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
  ```

- [ ] **Step 2: Add unit tests for `compute_elastic_forces_and_hessian`**
  Modify [tests/test_knitting_core_sim.py](file:///home/roip/projects/KnittingProject/tests/test_knitting_core_sim.py) to add:
  ```python
  from knitting_core import compute_elastic_forces_and_hessian

  def test_elastic_forces_and_hessian():
      V = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, -0.1], [2.0, -0.05, 0.05]], dtype=float)
      edges = np.array([[0, 1], [1, 2]])
      L0 = 1.0
      k_s, k_b = 100.0, 10.0
      e, g, H = compute_elastic_forces_and_hessian(V, edges, L0, k_s, k_b)
      assert e > 0
      assert g.shape == (9,)
      assert H.shape == (9, 9)
      # Check symmetry
      assert np.allclose(H.toarray(), H.toarray().T)
  ```

- [ ] **Step 3: Run the test to verify it passes**
  Run: `uv run pytest tests/test_knitting_core_sim.py -v`
  Expected: PASS

- [ ] **Step 4: Commit**
  ```bash
  git add knitting_core.py tests/test_knitting_core_sim.py
  git commit -m "feat: implement analytical stretch and bend potentials with PSD projection"
  ```

---

### Task 3: Tiled Mesh Collision, Barrier Potential, and Folding

**Files:**
- Modify: [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py)
- Modify: [tests/test_knitting_core_sim.py](file:///home/roip/projects/KnittingProject/tests/test_knitting_core_sim.py)

**Interfaces:**
- Consumes: `ipctk` classes
- Produces: `compute_collision_forces_and_hessian(V, D, edges, dhat, k_c)` returning `energy`, `grad_V` (flat, size $3M$), and `H_V` (sparse `coo_matrix` or `csr_matrix`, size $3M \times 3M$).

- [ ] **Step 1: Implement barrier potential evaluation on 3x3 tiled mesh with gradient/Hessian folding**
  Add `compute_collision_forces_and_hessian` to [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py).
  ```python
  import ipctk

  def compute_collision_forces_and_hessian(V, D, edges, dhat, k_c):
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

      # 3. IPC collisions build
      collisions = ipctk.NormalCollisions()
      collisions.build(mesh_tiled, V_tiled, dhat)

      barrier = ipctk.BarrierPotential(dhat)
      barrier_E = barrier(collisions, mesh_tiled, V_tiled) * k_c
      barrier_grad = barrier.gradient(collisions, mesh_tiled, V_tiled) * k_c
      barrier_hess = barrier.hessian(collisions, mesh_tiled, V_tiled, ipctk.PSDProjectionMethod.CLAMP) * k_c

      # 4. Fold gradient down to primary unit cell
      grad_V = np.zeros((M, 3))
      barrier_grad_reshaped = barrier_grad.reshape(9, M, 3)
      for c in range(9):
          grad_V += barrier_grad_reshaped[c]

      # 5. Fold sparse Hessian down to primary unit cell (3M x 3M)
      # Iterate over non-zero elements of the 27M x 27M barrier Hessian and sum them to 3M x 3M
      barrier_hess_coo = barrier_hess.tocoo()
      folded_rows = barrier_hess_coo.row % (3 * M)
      folded_cols = barrier_hess_coo.col % (3 * M)
      folded_vals = barrier_hess_coo.data

      H_V = scipy.sparse.coo_matrix(
          (folded_vals, (folded_rows, folded_cols)), shape=(3*M, 3*M)
      ).tocsr()

      return barrier_E, grad_V.flatten(), H_V
  ```

- [ ] **Step 2: Add unit tests for `compute_collision_forces_and_hessian`**
  Modify [tests/test_knitting_core_sim.py](file:///home/roip/projects/KnittingProject/tests/test_knitting_core_sim.py) to add:
  ```python
  from knitting_core import compute_collision_forces_and_hessian

  def test_collision_forces_and_hessian():
      V = np.array([[0.0, 0.0, 0.0], [0.01, 0.01, 0.0], [1.0, 0.0, 0.0]], dtype=float)
      D = np.array([1.5, 1.5, 0.0])
      edges = np.array([[0, 1], [1, 2]])
      dhat = 0.05
      k_c = 10.0
      e, g, H = compute_collision_forces_and_hessian(V, D, edges, dhat, k_c)
      assert e > 0  # since V[0] and V[1] are very close (< dhat)
      assert g.shape == (9,)
      assert H.shape == (9, 9)
  ```

- [ ] **Step 3: Run the test to verify it passes**
  Run: `uv run pytest tests/test_knitting_core_sim.py -v`
  Expected: PASS

- [ ] **Step 4: Commit**
  ```bash
  git add knitting_core.py tests/test_knitting_core_sim.py
  git commit -m "feat: implement tiled mesh collision barrier and gradient/Hessian folding"
  ```

---

### Task 4: Projected Newton Solver & CCD Line Search

**Files:**
- Modify: [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py)
- Modify: [tests/test_knitting_core_sim.py](file:///home/roip/projects/KnittingProject/tests/test_knitting_core_sim.py)

**Interfaces:**
- Consumes: `eval_centerline`, `build_row_spline_jacobian`
- Produces: `run_simulation_step(ctrl_rows, D, config, J_cached, k_s, k_b, k_c, dhat)` returning updated `ctrl_rows` list.

- [ ] **Step 1: Implement `run_simulation_step` with Newton solver and CCD**
  Add `run_simulation_step` to [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py).
  ```python
  import scipy.sparse.linalg

  def run_simulation_step(ctrl_rows, D, config, J_cached, k_s, k_b, k_c, dhat):
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
      # J_cached is J_base tensor-producted with I_3
      g_P = J_cached.T @ g_V
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
      
      alpha_max = ipctk.compute_collision_free_stepsize(mesh_tiled, V_tiled_0, V_tiled_1)
      
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
          
          if (e_el_new + e_col_new) < e:
              success = True
              ctrl_rows = new_ctrl_rows
              break
          alpha *= tau
          
      return ctrl_rows
  ```

- [ ] **Step 2: Add test case for full simulation step optimization**
  Modify [tests/test_knitting_core_sim.py](file:///home/roip/projects/KnittingProject/tests/test_knitting_core_sim.py) to add:
  ```python
  from knitting_core import run_simulation_step

  def test_run_simulation_step():
      ctrl_rows = [np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.0, 0.0]])]
      D = np.array([3.0, 3.0, 0.0])
      config = {"knit_parameters": {"loop_res": 5, "segments": 4}}
      
      # Build cached Jacobian
      J_base = build_row_spline_jacobian(ctrl_rows[0], D, 16) # loop_res*3 + 1 = 16
      J_cached = scipy.sparse.kron(J_base, scipy.sparse.identity(3), format="csr")
      
      updated = run_simulation_step(ctrl_rows, D, config, J_cached, k_s=100.0, k_b=10.0, k_c=1.0, dhat=0.05)
      assert len(updated) == 1
      assert updated[0].shape == (3, 3)
  ```

- [ ] **Step 3: Run the tests to verify it passes**
  Run: `uv run pytest tests/test_knitting_core_sim.py -v`
  Expected: PASS

- [ ] **Step 4: Commit**
  ```bash
  git add knitting_core.py tests/test_knitting_core_sim.py
  git commit -m "feat: implement Projected Newton step with CCD line search"
  ```

---

### Task 5: Background Threading & GUI Integration

**Files:**
- Modify: [app_state.py](file:///home/roip/projects/KnittingProject/app_state.py)
- Modify: [gui.py](file:///home/roip/projects/KnittingProject/gui.py)
- Modify: [app.py](file:///home/roip/projects/KnittingProject/app.py)

**Interfaces:**
- Consumes: `run_simulation_step`
- Produces: Sidebar "Yarn Simulation" UI controls, mutex lock protections.

- [ ] **Step 1: Add simulation parameters, cache lock, and background runner loop to `AppState`**
  Modify [app_state.py](file:///home/roip/projects/KnittingProject/app_state.py).
  Import `threading` and our new functions. Add initialization inside `__init__`:
  ```python
  # (Near the start of app_state.py)
  import threading
  from knitting_core import run_simulation_step, build_row_spline_jacobian
  import scipy.sparse
  ```
  Inside `__init__` in `AppState`:
  ```python
          # Yarn physical simulation state
          self.sim_active = False
          self.sim_lock = threading.Lock()
          self.sim_k_s = 1000.0
          self.sim_k_b = 10.0
          self.sim_k_c = 1.0
          self.sim_dhat = 0.02
          self.sim_needs_jacobian_rebuild = True
          self.J_cached = None
  ```
  Add methods to update/rebuild the cached Jacobian when needed:
  ```python
      def rebuild_cached_jacobian(self):
          if not self.ctrl_rows:
              return
          J_blocks = []
          res = self.config["knit_parameters"]["loop_res"]
          bitmap_width = float(self.period_offset[0])
          nout = res * int(round(bitmap_width)) + 1
          for cp in self.ctrl_rows:
              J_r = build_row_spline_jacobian(cp, self.period_offset, nout)
              J_blocks.append(J_r)
          J_base = scipy.sparse.block_diag(J_blocks, format="csr")
          self.J_cached = scipy.sparse.kron(J_base, scipy.sparse.identity(3), format="csr")
          self.sim_needs_jacobian_rebuild = False
  ```
  Acquire the lock in methods modifying control points:
  * In `move_ctrl_pt(self, flat_idx, pos)`: wrap the execution block in `with self.sim_lock:` and set `self.sim_needs_jacobian_rebuild = True`.
  * In `rebuild_spline_from_params(self)`: wrap in `with self.sim_lock:` and set `self.sim_needs_jacobian_rebuild = True`.
  * In `nudge_spline_from_params(self)`: wrap in `with self.sim_lock:` and set `self.sim_needs_jacobian_rebuild = True`.

- [ ] **Step 2: Add sidebar controls for Yarn Simulation**
  Modify [gui.py](file:///home/roip/projects/KnittingProject/gui.py).
  Add collapsing header UI:
  ```python
      # Under draw_sidebar in gui.py, add after the copies configuration sliders:
      if imgui.collapsing_header("Yarn Simulation##sim_header"):
          changed_active, active = imgui.checkbox("Run Simulation##run_sim", state.sim_active)
          if changed_active:
              state.sim_active = active
          
          # Sliders for parameters
          changed_ks, val_ks = imgui.slider_float("Stretch Stiffness##ks", state.sim_k_s, 0.0, 5000.0, "%.1f")
          if changed_ks: state.sim_k_s = val_ks
          
          changed_kb, val_kb = imgui.slider_float("Bending Stiffness##kb", state.sim_k_b, 0.0, 500.0, "%.1f")
          if changed_kb: state.sim_k_b = val_kb
          
          changed_kc, val_kc = imgui.slider_float("Collision Stiffness##kc", state.sim_k_c, 0.0, 100.0, "%.1f")
          if changed_kc: state.sim_k_c = val_kc
          
          changed_dhat, val_dhat = imgui.slider_float("Yarn Thickness##dhat", state.sim_dhat, 0.005, 0.1, "%.3f")
          if changed_dhat: state.sim_dhat = val_dhat
          
          if imgui.button("Reset to Rest State##reset_rest"):
              state.sim_active = False
              state.rebuild_spline_from_params()
  ```

- [ ] **Step 3: Start the Background Simulation Thread**
  Modify [app.py](file:///home/roip/projects/KnittingProject/app.py).
  Add helper function and spawn thread in `main`:
  ```python
  # Under imports
  import time
  import threading

  def start_simulation_thread(state):
      def run_loop():
          while True:
              if state.sim_active:
                  with state.sim_lock:
                      if state.sim_needs_jacobian_rebuild:
                          state.rebuild_cached_jacobian()
                      ctrl_rows = [cp.copy() for cp in state.ctrl_rows]
                      D = state.period_offset.copy()
                      config = state.config.copy()
                      J_cached = state.J_cached
                      ks = state.sim_k_s
                      kb = state.sim_k_b
                      kc = state.sim_k_c
                      dhat = state.sim_dhat
                  
                  new_ctrl_rows = run_simulation_step(ctrl_rows, D, config, J_cached, ks, kb, kc, dhat)
                  
                  with state.sim_lock:
                      # Only apply if user didn't modify it during computation
                      if not state.sim_needs_jacobian_rebuild:
                          state.ctrl_rows = new_ctrl_rows
                          state.flat_pts = np.concatenate(new_ctrl_rows).astype(np.float32) if new_ctrl_rows else np.empty((0, 3), np.float32)
                          state.sim_needs_jacobian_rebuild = True # rebuild mesh next frame
                          
              time.sleep(0.01)

      thread = threading.Thread(target=run_loop, daemon=True)
      thread.start()
  ```
  Spawn the thread in `main()` after `state = AppState(...)`:
  ```python
      start_simulation_thread(state)
  ```
  Also modify `main` loop to trigger mesh rebuild if the background thread has modified control points:
  ```python
          # In app.py's while loop, check if a mesh rebuild is needed:
          with state.sim_lock:
              if state.sim_needs_jacobian_rebuild and state.sim_active:
                  state.rebuild_spline_mesh(preserve_model_placement=True)
                  state.sim_needs_jacobian_rebuild = False
  ```

- [ ] **Step 4: Run the app and verify all UI controls and simulation operate smoothly**
  Run: `uv run python app.py`
  Expected: App launches, Yarn Simulation header shows, checking "Run Simulation" relaxes yarn in real-time, dragging control points works smoothly under locks.

- [ ] **Step 5: Commit**
  ```bash
  git add app.py app_state.py gui.py
  git commit -m "feat: integrate yarn simulation into app state, background thread, and GUI sidebar"
  ```
