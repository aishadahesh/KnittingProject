# Design Spec: Periodic Splines for Seamless Tiling

## Goal Description
To make the spline-based knitting reconstruction periodic so that tiled copies of the mesh connect seamlessly without visual seams or tangent discontinuities. We will implement periodic boundary conditions (`bc_type='periodic'`) for cubic splines in the core geometry pipeline. 

To maintain a Single Source of Truth (SSOT) and minimize state complexity, the duplicate boundary end point will be removed from the saved/edited control points dataset (`ctrl_rows`). Instead, this endpoint will be dynamically generated and appended during the spline mesh evaluation phase.

## Proposed Changes

### 1. Core Spline Pipeline

#### [MODIFY] [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py)
* **`build_parametric_control_rows`**:
  * Stop concatenating the legacy `end` point to the control rows. The function will return rows of length `cols * spl` containing only the true independent control points.
* **`build_spline_mesh`**:
  * Compute a canonical period offset vector:
    $$\vec{D} = \begin{bmatrix}\text{bitmap\_width} \\ 0 \\ 0\end{bmatrix}$$
  * Augment the control points array `cp` of length $N$ by appending:
    $$cp_{\text{aug}}[N] = cp[0] + \vec{D}$$
  * Calculate cumulative path length $t$ over `cp_aug`.
  * Detrend all three coordinates simultaneously:
    $$cp_{\text{detrended}} = cp_{\text{aug}} - \vec{D} \otimes \frac{t}{t[-1]}$$
  * Fit periodic cubic splines (`CubicSpline(..., bc_type='periodic')`) on each coordinate of $cp_{\text{detrended}}$.
  * Evaluate the spline on $to = \text{linspace}(0, t[-1], nout)$ and retrend the coordinates by adding back $\vec{D} \otimes \frac{to}{t[-1]}$.
  * Augment `radius_ctrl_rows` by appending the first radius value to the end of the profile row:
    $$radius\_profile_{\text{aug}} = \text{append}(radius\_profile, radius\_profile[0])$$
    and interpolate over the control sample index using the augmented list.

---

### 2. Centralized State and GUI

#### [MODIFY] [app_state.py](file:///home/roip/projects/KnittingProject/app_state.py)
* **Remove Duplicate End Point representation**:
  * `self.ctrl_rows` and `self.spline_radius_rows` will store elements of length `cols * spl` (no end point).
* **Tiling Translation offset**:
  * Update `_display_copy_x_period` to return exactly `max(float(self.bitmap_size[1]), radius)`.
* **Simplify `move_ctrl_pt`**:
  * Keep the mapping 1-to-1. No custom constraint synchronization between boundary points is needed because the end point is dynamically computed from the start point `cp[0]`.

---

## Legacy Data Migration (Manual)
Since we are following the **No Backward Compatibility Branching** rule, we will manually update `params.json` to strip the redundant 26th control point (`[5.0, row_idx * dy, 0.0]`) from each row of `spline_control_rows` and its corresponding entry in `spline_radius_rows` instead of writing fallback logic in Python.

## Verification Plan

### Automated Tests
Run `pytest` to verify spline evaluation and mesh generation. We will add a new test case `test_periodic_spline_continuity` in `tests/test_knitting_core.py` to assert that:
* Spline start and end positions match exactly (with period translation).
* Spline start and end first derivatives match exactly.
* Spline start and end second derivatives match exactly.

### Manual Verification
1. Launch `uv run python app.py`.
2. Enable 3D spline rendering and tiled copies/tessellation.
3. Visually verify that the connection between tiled copies of the knitted mesh is completely smooth and continuous.
