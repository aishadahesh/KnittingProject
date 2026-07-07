# Design Spec: Periodic Splines for Seamless Tiling

## Goal Description
To make the spline-based knitting reconstruction periodic so that tiled copies of the mesh connect seamlessly without visual seams or tangent discontinuities. We will implement periodic boundary conditions (`bc_type='periodic'`) for cubic splines in the core geometry pipeline.

To maintain a Single Source of Truth (SSOT) and minimize state complexity, the duplicate boundary end point will be removed from the saved/edited control points dataset (`ctrl_rows`). Instead, this endpoint will be dynamically generated and appended during the spline mesh evaluation phase.

To allow users to adjust the tiling period and connection offset interactively, we will introduce a **configurable period offset vector** $\vec{D}$ in the state. This vector will be represented as a set of **"virtual" control points** in the 3D viewport (rendered at $cp[0] + \vec{D}$). Dragging any of these virtual control points will adjust the period offset dynamically. Additionally, scaling the bounding box will scale the period offset proportionally.

## Proposed Changes

### 1. Core Spline Pipeline

#### [MODIFY] [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py)
* **`build_parametric_control_rows`**:
  * Return rows of length `cols * spl` containing only the true independent control points.
* **`build_spline_mesh`**:
  * Take a 3D `period_offset` parameter instead of a scalar `bitmap_width`.
  * Augment the control points array `cp` of length $N$ by appending:
    $$cp_{\text{aug}}[N] = cp[0] + \text{period\_offset}$$
  * Calculate cumulative path length $t$ over `cp_aug`.
  * Detrend all three coordinates simultaneously:
    $$cp_{\text{detrended}} = cp_{\text{aug}} - \text{period\_offset} \otimes \frac{t}{t[-1]}$$
  * Fit periodic cubic splines on each coordinate of $cp_{\text{detrended}}$.
  * Evaluate the spline and retrend.
  * Compute the tangents periodically by wrapping around boundaries using central differences, ensuring the normals and binormals match exactly at the boundary.
  * Augment `radius_ctrl_rows` by appending the first radius value to the end of the profile row:
    $$radius\_profile_{\text{aug}} = \text{append}(radius\_profile, radius\_profile[0])$$

---

### 2. Centralized State and GUI

#### [MODIFY] [app_state.py](file:///home/roip/projects/KnittingProject/app_state.py)
* **`self.period_offset` State**:
  * Add a 3D float array `self.period_offset` to `AppState` (initially `[float(self.bitmap_size[1]), 0.0, 0.0]`).
  * Save and load `period_offset` in JSON parameters.
* **`flat_pts_all` Property**:
  * Expose an augmented list of flat control points:
    $$\text{flat\_pts\_all} = \text{concat}(\text{flat\_pts}, \text{virtual\_pts})$$
    where the virtual control point for row $r$ is `ctrl_rows[r][0] + period_offset`.
* **Simplify & Update `move_ctrl_pt`**:
  * If `flat_idx` references a virtual point index (i.e. $\ge len(flat\_pts)$), calculate the new `period_offset`:
    $$\text{period\_offset} = \text{pos} - \text{ctrl\_rows}[r][0]$$
  * Otherwise, move the real control point and rebuild `flat_pts`.
* **Update Bbox Drag Handling**:
  * Save `state.bbox_start_period_offset` when dragging starts.
  * Update `state.period_offset` on drag update:
    $$\text{period\_offset} = \text{bbox\_start\_period\_offset} \times \text{scale\_vec}$$
* **Tiling Translation offset**:
  * Update `_display_copy_x_period` to return exactly the x-coordinate of the period offset: `max(float(self.period_offset[0]), radius)`.
  * Update `_display_copy_y_period` and `_display_copy_z_period` to use the components of `self.period_offset`.

#### [MODIFY] [gui.py](file:///home/roip/projects/KnittingProject/gui.py)
* **Virtual Control Points Visibility**:
  * Include the virtual control points in the visible index list:
    $$\text{all\_visible\_indices} = \text{concat}(\text{real\_visible\_indices}, \text{virtual\_indices})$$
  * When rendering and picking control points, use `state.flat_pts_all` instead of `state.flat_pts`.
* **Pass Virtual Point Count to Shaders**:
  * Pass the number of real control points to the OpenGL shaders so virtual points can be rendered with a distinct cyan color.

#### [MODIFY] [rendering.py](file:///home/roip/projects/KnittingProject/rendering.py)
* **Highlight Virtual Control Points**:
  * Update `PT_VERT` and `PT_FRAG` to accept uniform `int n_real`. Any point with vertex ID $\ge n\_real$ will be rendered in cyan (`vec4(0.0, 0.8, 1.0, 0.9)`).

---

## Legacy Data Migration (Manual)
Strip the 26th control point from all rows in `params.json`.

## Verification Plan

### Automated Tests
Run `pytest` to verify spline evaluation and mesh generation. We will add a new test case `test_periodic_spline_continuity` in `tests/test_knitting_core.py` to assert that:
* Spline start and end positions match exactly (with period translation).
* Spline start and end first derivatives match exactly.
* Spline start and end second derivatives match exactly.

### Manual Verification
1. Launch `uv run python app.py`.
2. Select a virtual control point (cyan) in the 3D viewport and drag it. Verify that the tiling offset changes seamlessly.
3. Drag the edges of the bounding box. Verify that the virtual control points scale proportionally.

