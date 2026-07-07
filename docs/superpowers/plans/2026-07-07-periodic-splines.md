# Periodic Splines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement periodic boundary conditions for the spline-based knitting reconstruction to enable seamless tiling.

**Architecture:** Remove the redundant endpoint from stored data, dynamically augment control points and radius profiles, and fit periodic cubic splines on detrended coordinates (all three channels) to enforce C2 continuity at boundaries.

**Tech Stack:** Python, SciPy (CubicSpline), NumPy, Pytest.

## Global Constraints
- `knitting_core.py` MUST remain a stateless, purely functional module. Do not introduce classes that hold state.
- Centralized State: All application state must reside in `app_state.py` within the `AppState` class.
- No Backward Compatibility Branching: Do not implement branching or validation wrappers in Python code to support deprecated data formats; migrate data files manually.

---

### Task 1: Manual Parameter Data Migration

**Files:**
- Modify: `params.json`

**Interfaces:**
- Consumes: None
- Produces: Updated JSON with 10-point control and radius rows instead of 11.

- [ ] **Step 1: Run migration command**
  Run: `python -c "import json; d = json.load(open('params.json')); d['spline_control_rows'] = [r[:-1] for r in d['spline_control_rows']]; d['spline_radius_rows'] = [r[:-1] for r in d['spline_radius_rows']]; json.dump(d, open('params.json', 'w'), indent=2)"`
- [ ] **Step 2: Verify lengths of the updated parameter file**
  Run: `python -c "import json; d = json.load(open('params.json')); print([len(r) for r in d['spline_control_rows']]); print([len(r) for r in d['spline_radius_rows']])"`
  Expected: Output `[10, 10, 10, 10]` for both lists.
- [ ] **Step 3: Commit migration**
  Run: `git commit -am "data: migrate params.json to 10 control points"`

---

### Task 2: Core Spline Pipeline (knitting_core.py)

**Files:**
- Modify: `knitting_core.py`
- Test: `tests/test_knitting_core.py`

**Interfaces:**
- Consumes: `params.json` (10-point rows)
- Produces: Updated `build_parametric_control_rows` (returns shape `(cols * spl, 3)`) and `build_spline_mesh` (periodic fitting using detrending/retrending).

- [ ] **Step 1: Write the failing tests**
  Add a test to `tests/test_knitting_core.py` to assert C2 boundary continuity:
  ```python
  def test_periodic_spline_continuity(config_fixture, params_fixture):
      from knitting_core import build_parametric_control_rows, build_spline_mesh
      params_dict = params_fixture["params"]
      config_params = config_fixture["knit_parameters"]["parameters"]
      params_list = [params_dict.get(p["name"], p["initial"]) for p in config_params]
      pidx = {p["name"]: i for i, p in enumerate(config_params)}
      lh_params = sorted(
          [p["name"] for p in config_params if p["name"].startswith("loop_height_")],
          key=lambda name: int(name.split("_")[-1])
      )
      lh_idx = tuple(pidx[name] for name in lh_params)
      bitmap = np.asarray(params_fixture["bitmap"], dtype=np.float32)
      
      # 1. Verify control point lengths are cols * spl (10)
      ctrl_rows = build_parametric_control_rows(params_list, bitmap, pidx, lh_idx)
      assert ctrl_rows[0].shape[0] == 10
      
      # 2. Build mesh and check continuity of evaluated spline points
      spline_mesh = build_spline_mesh(ctrl_rows, params_list, config_fixture, pidx, bitmap.shape[1])
      for pts, nout in spline_mesh:
          # Spacing in path length parameter
          dt = 1.0 / (nout - 1)
          
          # First derivatives at start and end must match within tolerance
          # (For x-axis, there is a trend of +bitmap.shape[1] = 2.0 across the spline)
          dx_start = pts[1] - pts[0]
          dx_end = pts[-1] - pts[-2]
          np.testing.assert_allclose(dx_start[1:], dx_end[1:], atol=1e-4) # Y and Z derivatives
          np.testing.assert_allclose(dx_start[0], dx_end[0], atol=1e-4) # X derivative
          
          # Second derivatives must match
          d2x_start = pts[2] - 2 * pts[1] + pts[0]
          d2x_end = pts[-1] - 2 * pts[-2] + pts[-3]
          np.testing.assert_allclose(d2x_start, d2x_end, atol=1e-4)
  ```
- [ ] **Step 2: Run test to verify it fails**
  Run: `uv run pytest tests/test_knitting_core.py::test_periodic_spline_continuity`
  Expected: FAIL (assertion error or shape mismatch)
- [ ] **Step 3: Modify `build_parametric_control_rows`**
  In `knitting_core.py`:
  Replace the end concatenation lines:
  ```python
      end = np.zeros((rows, 1, 3), dtype=np.float32)
      end[:, 0, 0] = float(cols) * x_pitch
      end[:, 0, 1] = np.arange(rows, dtype=np.float32) * dy
      return [np.concatenate((c[r], end[r]), axis=0).astype(float) for r in range(rows)]
  ```
  With:
  ```python
      return [c[r].astype(float) for r in range(rows)]
  ```
- [ ] **Step 4: Modify `build_spline_mesh`**
  In `knitting_core.py`:
  Replace the spline fitting block inside `build_spline_mesh` to implement dynamic augmentation, detrending, and periodic CubicSpline evaluation. Update radius profile augmentation to append the first radius value.
- [ ] **Step 5: Run tests to verify they pass**
  Run: `uv run pytest`
  Expected: All tests pass (including `test_periodic_spline_continuity` and existing spline/fiber tests).
- [ ] **Step 6: Commit changes**
  Run: `git commit -am "feat: implement periodic cubic spline interpolation with detrending"`

---

### Task 3: State & GUI Synchronization (app_state.py)

**Files:**
- Modify: `app_state.py`

**Interfaces:**
- Consumes: `knitting_core.py` changes
- Produces: Simpler `move_ctrl_pt`, direct X period translation calculation.

- [ ] **Step 1: Update `_display_copy_x_period`**
  In `app_state.py`:
  Simplify `_display_copy_x_period` to return the bitmap width column count directly:
  ```python
      def _display_copy_x_period(self, verts_list, radius):
          return max(float(self.bitmap_size[1]), radius)
  ```
- [ ] **Step 2: Verify existing tests**
  Run: `uv run pytest`
  Expected: PASS
- [ ] **Step 3: Commit state changes**
  Run: `git commit -am "feat: simplify app_state copy width and remove legacy end point synchronization"`

---
