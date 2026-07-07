# Periodic Splines with Interactive Virtual Control Points Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement interactive periodic boundary conditions for spline-based knitting reconstruction.

**Architecture:** Introduce `self.period_offset` to the state, represent it in the 3D viewport using green/cyan "virtual" control points, allow interactive dragging of these virtual points to change the offset, and scale it proportionally with bounding box scaling.

**Tech Stack:** Python, SciPy, NumPy, OpenGL/ModernGL, ImGui.

## Global Constraints
- `knitting_core.py` MUST remain a stateless, purely functional module. Do not introduce classes that hold state.
- Centralized State: All application state must reside in `app_state.py` within the `AppState` class.
- No Backward Compatibility Branching: Do not implement branching or validation wrappers in Python code to support legacy data formats; migrate data files manually.

---

### Task 1: Core Spline Pipeline Update (knitting_core.py)

**Files:**
- Modify: `knitting_core.py`
- Modify: `tests/test_knitting_core.py`

- [ ] **Step 1: Update `build_spline_mesh` signature and logic**
  In `knitting_core.py`:
  Modify `build_spline_mesh` to accept `period_offset` (which can be a scalar width or a 3D vector):
  ```python
  def build_spline_mesh(ctrl_rows, params, config, pidx, period_offset, radius_ctrl_rows=None):
      p = np.asarray(params)
      rad, rat = p[pidx["radius"]], p[pidx["ellipse_ratio"]]
      seg, res = config["knit_parameters"]["segments"], config["knit_parameters"]["loop_res"]
      
      # Handle both scalar width (for tests/compatibility) and 3D offset vector
      if isinstance(period_offset, (int, float, np.integer, np.floating)):
          D = np.array([float(period_offset), 0.0, 0.0], dtype=float)
          bitmap_width = float(period_offset)
      else:
          D = np.asarray(period_offset, dtype=float)
          bitmap_width = float(np.linalg.norm(D)) # Use length for resolution sizing
          
      nout = res * int(round(bitmap_width)) + 1
  ```
- [ ] **Step 2: Update test case**
  In `tests/test_knitting_core.py`:
  Update `test_periodic_spline_continuity` to pass `[float(bitmap.shape[1]), 0.0, 0.0]` as the `period_offset` argument:
  ```python
      # 2. Build mesh and check periodic boundary alignment
      spline_mesh = build_spline_mesh(ctrl_rows, params_list, config_fixture, pidx, [float(bitmap.shape[1]), 0.0, 0.0])
  ```
- [ ] **Step 3: Run pytest**
  Run: `uv run pytest`
  Expected: PASS
- [ ] **Step 4: Commit changes**
  Run: `git commit -am "feat: update build_spline_mesh to accept 3D period_offset vector"`

---

### Task 2: State Integration (app_state.py)

**Files:**
- Modify: `app_state.py`

- [ ] **Step 1: Add `self.period_offset` to AppState**
  In `app_state.py`:
  Initialize `self.period_offset` in `AppState.__init__` or `reset` / `load_params`:
  ```python
      # In load_params:
      self.period_offset = np.array(
          loaded_data.get('period_offset', [float(self.bitmap_size[1]), 0.0, 0.0]),
          dtype=np.float32
      )
  ```
  Ensure it is saved in `save_params`:
  ```python
      'period_offset': self.period_offset.tolist(),
  ```
- [ ] **Step 2: Implement `flat_pts_all` property**
  In `app_state.py`:
  ```python
      @property
      def flat_pts_all(self):
          if not self.ctrl_rows:
              return np.empty((0, 3), dtype=np.float32)
          virtual_pts = np.array([row[0] + self.period_offset for row in self.ctrl_rows], dtype=np.float32)
          return np.concatenate((self.flat_pts, virtual_pts), axis=0)
  ```
- [ ] **Step 3: Update `move_ctrl_pt`**
  In `app_state.py`:
  ```python
      def move_ctrl_pt(self, flat_idx, pos):
          n_real = len(self.flat_pts)
          if flat_idx >= n_real:
              row_idx = flat_idx - n_real
              if 0 <= row_idx < len(self.ctrl_rows):
                  self.period_offset = pos - self.ctrl_rows[row_idx][0]
                  self.rebuild_spline_mesh()
          else:
              r = np.searchsorted(self._row_starts, flat_idx, side="right") - 1
              if 0 <= r < len(self.ctrl_rows):
                  self.ctrl_rows[r][flat_idx - self._row_starts[r]] = pos
                  self._rebuild_spline_points()
  ```
- [ ] **Step 4: Update copy offsets**
  In `app_state.py`:
  ```python
      def _display_copy_x_period(self, verts_list, radius):
          return max(float(self.period_offset[0]), radius)
  
      def _display_copy_y_period(self, verts_list, radius):
          return float(self.period_offset[1])
  
      def _display_copy_z_period(self, verts_list, depth_gap):
          return float(self.period_offset[2])
  ```
- [ ] **Step 5: Pass `period_offset` to `build_spline_mesh`**
  In `AppState.rebuild_spline_mesh`:
  ```python
          vl = build_spline_mesh(
              self.ctrl_rows,
              self.params,
              self.config,
              self._pidx,
              self.period_offset,
              radius_ctrl_rows=radius_profiles,
          )
  ```
- [ ] **Step 6: Commit state changes**
  Run: `git commit -am "feat: add period_offset and flat_pts_all to AppState"`

---

### Task 3: Shaders Update (rendering.py)

**Files:**
- Modify: `rendering.py`

- [ ] **Step 1: Update shaders to accept `n_real`**
  In `rendering.py`:
  Update `PT_VERT` to:
  ```glsl
  #version 330
  in  vec3 in_pos;
  uniform mat4 mvp;
  uniform int  hover_idx;
  uniform int  selected_idx;
  uniform int  n_real;
  flat out int state;   // 0=normal 1=hover 2=selected
  flat out int is_virtual;
  void main() {
      gl_Position = mvp * vec4(in_pos, 1.0);
      int vid = gl_VertexID;
      state = (vid == selected_idx) ? 2 : (vid == hover_idx ? 1 : 0);
      is_virtual = (vid >= n_real) ? 1 : 0;
      gl_PointSize = (state > 0) ? 16.0 : 10.0;
  }
  ```
  Update `PT_FRAG` to:
  ```glsl
  #version 330
  flat in int state;
  flat in int is_virtual;
  out vec4 f_color;
  void main() {
      vec2 c = gl_PointCoord * 2.0 - 1.0;
      if (dot(c, c) > 1.0) discard;
      if      (state == 2) f_color = vec4(1.0, 1.0, 0.0, 1.0);  // drag  → yellow
      else if (state == 1) f_color = vec4(1.0, 0.5, 0.0, 1.0);  // hover → orange
      else if (is_virtual == 1) f_color = vec4(0.0, 0.8, 1.0, 0.9); // virtual -> cyan
      else                 f_color = vec4(1.0, 1.0, 1.0, 0.9);  // normal → white
  }
  ```
- [ ] **Step 2: Update render uniforms**
  In `rendering.py` inside `MeshRenderer.render`:
  ```python
          if self.pt_vao:
              self.ctx.disable(moderngl.DEPTH_TEST)
              self.pt_prog['mvp'].write(mvp.T.tobytes())
              self.pt_prog['hover_idx'].value    = hover_idx
              self.pt_prog['selected_idx'].value = selected_idx
              if 'n_real' in self.pt_prog:
                  self.pt_prog['n_real'].value   = n_real_pts if n_real_pts >= 0 else self.n_pts
  ```
  Also update `MeshRenderer.render` signature to accept `n_real_pts=-1`.
- [ ] **Step 3: Commit shader changes**
  Run: `git commit -am "feat: update control point shaders to color virtual points in cyan"`

---

### Task 4: GUI Integration (gui.py)

**Files:**
- Modify: `gui.py`

- [ ] **Step 1: Include virtual points in visible index list**
  In `gui.py` around line 640:
  ```python
      visible_ctrl_indices = np.empty((0,), dtype=np.int32)
      visible_ctrl_index_map = {}
      if state.mode == 'spline':
          n_real = len(state.flat_pts)
          visible_chunks = []
          for row_idx, row in enumerate(state.ctrl_rows):
              if not state.row_visible[row_idx]:
                  continue
              start = state._row_starts[row_idx]
              end = start + len(row)
              visible_chunks.append(np.arange(start, end, dtype=np.int32))
              # Append the virtual control point index for this row
              visible_chunks.append(np.array([n_real + row_idx], dtype=np.int32))
              
          if visible_chunks:
              visible_ctrl_indices = np.concatenate(visible_chunks)
              visible_ctrl_pts = state.flat_pts_all[visible_ctrl_indices]
          else:
              visible_ctrl_pts = np.empty((0, 3), dtype=np.float32)
              
          renderer.set_ctrl_pts(visible_ctrl_pts)
          visible_ctrl_index_map = {
              int(flat_idx): int(local_idx)
              for local_idx, flat_idx in enumerate(visible_ctrl_indices.tolist())
          }
          render_hover_idx = visible_ctrl_index_map.get(int(state.hover_idx), -1)
          render_selected_idx = visible_ctrl_index_map.get(int(state.selected_idx), -1)
  ```
- [ ] **Step 2: Pass `n_real` to renderer.render**
  In `gui.py` around line 710:
  ```python
      renderer.render(
          mvp, mv,
          state.get_material_uniforms(),
          render_hover_idx, render_selected_idx,
          hover_mesh_idx=state.hover_mesh_idx,
          selected_mesh_idx=state.selected_mesh_idx,
          visible_rows=state.row_visible,
          bg_tex      = ref_tex if state.show_ref_bg else None,
          bg_alpha    = state.ref_bg_alpha,
          bg_uniforms = bg_uniforms,
          camera      = state.camera,
          n_real_pts  = sum(len(row) for r_idx, row in enumerate(state.ctrl_rows) if state.row_visible[r_idx])
      )
  ```
- [ ] **Step 3: Update control point references in viewport mouse/gizmo interactions**
  In `gui.py`:
  Replace `state.flat_pts` with `state.flat_pts_all` in:
  - Line 851 (for gizmo starting position)
  - Line 1267 (for keyboard nudge position)
  - Line 1287 (for hover calculation)
- [ ] **Step 4: Update bbox drag to scale `period_offset`**
  In `gui.py`:
  - When starting bbox drag (around line 919):
    ```python
                state.bbox_start_period_offset = np.array(state.period_offset, dtype=np.float32).copy()
    ```
  - When performing bbox drag update (around line 981):
    ```python
                        state.period_offset = (state.bbox_start_period_offset * scale_vec).astype(np.float32)
    ```
  - When ending/cancelling bbox drag, set `state.bbox_start_period_offset = None`.
- [ ] **Step 5: Verify implementation**
  Launch: `uv run python app.py`
  Verify that cyan virtual control points are drawn at the end of each row, dragging them updates the period offset, and scaling the bbox scales the period offset proportionally.
- [ ] **Step 6: Commit changes**
  Run: `git commit -am "feat: integrate virtual control points and period offset dragging into the GUI"`

---
