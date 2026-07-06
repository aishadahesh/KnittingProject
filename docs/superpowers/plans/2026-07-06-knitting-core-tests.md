# Unit Tests for Knitting Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a comprehensive unit test suite in `tests/test_knitting_core.py` using `pytest` to test the JAX and NumPy core logic in `knitting_core.py` and verify it is a headless core.

**Architecture:** A stateless pytest module with fixtures for config/params loading, running purely on JAX CPU backend.

**Tech Stack:** `pytest`, `jax`, `numpy`, `scipy`.

## Global Constraints
- Target platform is CPU only for tests (`jax.config.update("jax_platform_name", "cpu")`).
- Zero UI dependencies or AppState references inside `knitting_core.py` (checked by import validation).

---

### Task 1: Environment and Pytest Setup

**Files:**
- Create: `tests/test_knitting_core.py`
- Modify: `.gitignore` (if needed, to ensure standard cache directories are ignored)

**Interfaces:**
- Consumes: None
- Produces: Base test file with imports and backend settings.

- [ ] **Step 1: Write the environment setup in test file**
  Create `tests/test_knitting_core.py` with:
  ```python
  import os
  import sys
  
  # Force JAX to use CPU backend
  import jax
  jax.config.update("jax_platform_name", "cpu")
  os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
  ```
- [ ] **Step 2: Install pytest via uv**
  Run: `uv pip install pytest`
- [ ] **Step 3: Run the empty test file**
  Run: `.venv/bin/pytest tests/test_knitting_core.py -v`
  Expected: Success or "collected 0 items" response (no failures).
- [ ] **Step 4: Commit setup**
  Run:
  ```bash
  git add tests/test_knitting_core.py
  git commit -m "test: set up test suite environment and configure JAX for CPU"
  ```

---

### Task 2: Headless Isolation Check

**Files:**
- Modify: `tests/test_knitting_core.py`

**Interfaces:**
- Consumes: `knitting_core`
- Produces: `test_core_module_imports()`

- [ ] **Step 1: Write the failing isolation test**
  Add the test method in `tests/test_knitting_core.py`:
  ```python
  def test_core_module_imports():
      # Verify that importing knitting_core does not pull in GUI-related libraries
      import knitting_core
      assert "moderngl" not in sys.modules
      assert "imgui_bundle" not in sys.modules
      assert "gui" not in sys.modules
      assert "app_state" not in sys.modules
  ```
- [ ] **Step 2: Run test to verify it fails/passes**
  Run: `.venv/bin/pytest tests/test_knitting_core.py::test_core_module_imports -v`
  Expected: PASS (as long as knitting_core doesn't import these at module load time).
- [ ] **Step 3: Commit**
  Run:
  ```bash
  git add tests/test_knitting_core.py
  git commit -m "test: implement headless isolation check"
  ```

---

### Task 3: Configuration & Data Fixtures

**Files:**
- Modify: `tests/test_knitting_core.py`

**Interfaces:**
- Consumes: `config.json`, `params.json`
- Produces: pytest fixtures `config_fixture` and `params_fixture`

- [ ] **Step 1: Add pytest fixtures**
  Add imports and fixtures in `tests/test_knitting_core.py`:
  ```python
  import json
  import pytest
  import numpy as np
  
  @pytest.fixture
  def config_fixture():
      with open("config.json", "r") as f:
          return json.load(f)
  
  @pytest.fixture
  def params_fixture():
      with open("params.json", "r") as f:
          return json.load(f)
  
  def test_fixtures_load(config_fixture, params_fixture):
      assert "knit_parameters" in config_fixture
      assert "params" in params_fixture
      assert "bitmap" in params_fixture
  ```
- [ ] **Step 2: Run the fixture test**
  Run: `.venv/bin/pytest tests/test_knitting_core.py::test_fixtures_load -v`
  Expected: PASS
- [ ] **Step 3: Commit**
  Run:
  ```bash
  git add tests/test_knitting_core.py
  git commit -m "test: add configuration and parameter fixtures"
  ```

---

### Task 4: Math Utilities and Orthonormal Frame Checks

**Files:**
- Modify: `tests/test_knitting_core.py`

**Interfaces:**
- Consumes: `knitting_core._scale_factors_jax`, `knitting_core.eval_curve`, `knitting_core.eval_curve_derivative`, `knitting_core.compute_orthonormal_frame`
- Produces: `test_scale_factors_jax()`, `test_eval_curve_and_derivative()`, `test_compute_orthonormal_frame()`

- [ ] **Step 1: Write utility tests**
  Add the following test functions in `tests/test_knitting_core.py`:
  ```python
  import jax.numpy as jnp
  from knitting_core import (
      _scale_factors_jax, eval_curve, eval_curve_derivative, compute_orthonormal_frame
  )
  
  def test_scale_factors_jax():
      bitmap = jnp.array([[1.0, 0.0], [1.0, 1.0]], dtype=jnp.float32)
      factors = _scale_factors_jax(bitmap)
      assert factors.shape == (2, 2)
      assert np.all(factors >= 0.0)
  
  def test_eval_curve_and_derivative():
      t = jnp.linspace(0.0, 2.0 * jnp.pi, 10)
      hl = jnp.ones(10)
      lh = 1.5
      sb = 0.25
      sz = 0.1
      
      pos = eval_curve(t, hl, lh, sb, sz)
      der = eval_curve_derivative(t, hl, lh, sb, sz)
      
      assert pos.shape == (10, 3)
      assert der.shape == (10, 3)
  
  def test_compute_orthonormal_frame():
      tangent = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
      u, v = compute_orthonormal_frame(tangent)
      assert u.shape == (2, 3)
      assert v.shape == (2, 3)
      
      # Assert orthonormality
      for i in range(2):
          t_val = tangent[i]
          u_val = u[i]
          v_val = v[i]
          assert np.abs(np.dot(t_val, u_val)) < 1e-5
          assert np.abs(np.dot(t_val, v_val)) < 1e-5
          assert np.abs(np.dot(u_val, v_val)) < 1e-5
          assert np.abs(np.linalg.norm(u_val) - 1.0) < 1e-5
          assert np.abs(np.linalg.norm(v_val) - 1.0) < 1e-5
  ```
- [ ] **Step 2: Run mathematical utility tests**
  Run: `.venv/bin/pytest tests/test_knitting_core.py -k "jax or curve or orthonormal" -v`
  Expected: PASS
- [ ] **Step 3: Commit**
  Run:
  ```bash
  git add tests/test_knitting_core.py
  git commit -m "test: add test cases for math and orthonormal frame utilities"
  ```

---

### Task 5: Vertices, Faces, and Splines Verification

**Files:**
- Modify: `tests/test_knitting_core.py`

**Interfaces:**
- Consumes: `knitting_core.compute_knitting_vertices`, `knitting_core.compute_knitting_faces`, `knitting_core.build_parametric_control_rows`, `knitting_core.build_spline_mesh`
- Produces: `test_mesh_generation()`, `test_spline_construction()`

- [ ] **Step 1: Write geometry construction tests**
  Add the test functions in `tests/test_knitting_core.py`:
  ```python
  from knitting_core import (
      compute_knitting_vertices, compute_knitting_faces,
      build_parametric_control_rows, build_spline_mesh
  )
  
  def test_mesh_generation(config_fixture, params_fixture):
      params_dict = params_fixture["params"]
      # Reconstruct the list of params using configuration order
      config_params = config_fixture["knit_parameters"]["parameters"]
      params_list = [params_dict[p["name"]] for p in config_params]
      
      pidx = {p["name"]: i for i, p in enumerate(config_params)}
      lh_params = sorted(
          [p["name"] for p in config_params if p["name"].startswith("loop_height_")],
          key=lambda name: int(name.split("_")[-1])
      )
      lh_idx = tuple(pidx[name] for name in lh_params)
      bitmap = np.asarray(params_fixture["bitmap"], dtype=np.float32)
      
      # 1. Compute vertices
      vl = compute_knitting_vertices(params_list, bitmap, config_fixture, pidx, lh_idx)
      assert isinstance(vl, list)
      assert len(vl) == bitmap.shape[0]
      for verts, pt_count in vl:
          assert isinstance(verts, np.ndarray)
          assert verts.ndim == 2
          assert verts.shape[1] == 3
          assert isinstance(pt_count, int)
          
      # 2. Compute faces
      seg = config_fixture["knit_parameters"]["segments"]
      fl = compute_knitting_faces(seg, vl)
      assert len(fl) == len(vl)
      for faces in fl:
          assert isinstance(faces, np.ndarray)
          assert faces.ndim == 2
          assert faces.shape[1] == 4 # quads
  
  def test_spline_construction(config_fixture, params_fixture):
      params_dict = params_fixture["params"]
      config_params = config_fixture["knit_parameters"]["parameters"]
      params_list = [params_dict[p["name"]] for p in config_params]
      
      pidx = {p["name"]: i for i, p in enumerate(config_params)}
      lh_params = sorted(
          [p["name"] for p in config_params if p["name"].startswith("loop_height_")],
          key=lambda name: int(name.split("_")[-1])
      )
      lh_idx = tuple(pidx[name] for name in lh_params)
      bitmap = np.asarray(params_fixture["bitmap"], dtype=np.float32)
      
      # Build parametric control rows
      ctrl_rows = build_parametric_control_rows(params_list, bitmap, pidx, lh_idx)
      assert len(ctrl_rows) == bitmap.shape[0]
      for row in ctrl_rows:
          assert isinstance(row, np.ndarray)
          assert row.shape[1] == 3
          
      # Build spline mesh
      spline_mesh = build_spline_mesh(ctrl_rows, params_list, config_fixture, pidx, bitmap.shape[1])
      assert len(spline_mesh) == len(ctrl_rows)
      for pts, nout in spline_mesh:
          assert pts.shape[1] == 3
          assert isinstance(nout, int)
  ```
- [ ] **Step 2: Run mesh and spline tests**
  Run: `.venv/bin/pytest tests/test_knitting_core.py -k "mesh or spline" -v`
  Expected: PASS
- [ ] **Step 3: Commit**
  Run:
  ```bash
  git add tests/test_knitting_core.py
  git commit -m "test: implement validation for vertices, faces, and spline construction"
  ```

---

### Task 6: Fiber Geometry Assembly

**Files:**
- Modify: `tests/test_knitting_core.py`

**Interfaces:**
- Consumes: `knitting_core.build_surface_fiber_meshes`
- Produces: `test_fiber_meshes()`

- [ ] **Step 1: Write fiber mesh generation tests**
  Add the test functions in `tests/test_knitting_core.py`:
  ```python
  from knitting_core import build_surface_fiber_meshes
  
  def test_fiber_meshes():
      # Mock base vertices
      # List of (verts, count)
      base_vl = [(np.arange(60).reshape(10, 6).astype(np.float32)[:, :3], 10)]
      segments = 6
      
      # 1. Disabled case
      out_vl, meta = build_surface_fiber_meshes(
          base_vl=base_vl,
          segments=segments,
          enabled=False,
          count=3,
          radius=0.1,
          radius_scale=1.0,
          lift=0.0,
          surface_arc=0.5,
          randomness=0.0,
          twist=0.0
      )
      assert len(out_vl) == len(base_vl)
      assert meta[0]["row"] == 0
      
      # 2. Enabled case
      out_vl, meta = build_surface_fiber_meshes(
          base_vl=base_vl,
          segments=segments,
          enabled=True,
          count=3,
          radius=0.1,
          radius_scale=1.0,
          lift=0.0,
          surface_arc=0.5,
          randomness=0.0,
          twist=0.0
      )
      # We expect count=3 fiber meshes produced
      assert len(out_vl) == 3
      assert len(meta) == 3
      for entry in meta:
          assert entry["row"] == 0
  ```
- [ ] **Step 2: Run fiber mesh tests**
  Run: `.venv/bin/pytest tests/test_knitting_core.py -k "fiber" -v`
  Expected: PASS
- [ ] **Step 3: Run the full test suite**
  Run: `.venv/bin/pytest tests/test_knitting_core.py -v`
  Expected: PASS (all tests pass)
- [ ] **Step 4: Commit**
  Run:
  ```bash
  git add tests/test_knitting_core.py
  git commit -m "test: add verification for fiber mesh assembly"
  ```
