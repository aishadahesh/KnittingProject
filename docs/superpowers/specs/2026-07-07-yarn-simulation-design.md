# Design Spec: Interactive Yarn Physical Simulation via IPC Toolkit

## Goal Description
To add a simplistic interactive physical simulation to the yarn using the IPC Toolkit (ipctk). The simulation must be adapted to periodic boundary conditions and cubic splines, run frame-by-frame asynchronously in a background thread to maintain high responsiveness in the GUI, and allow interactive tweaking of parameters (stiffness values, yarn radius).

## Proposed Architecture

### 1. Spline Geometry & Analytical/Cached Jacobian
For each row $r$, the evaluated centerline spline points $V_r \in \mathbb{R}^{3M_r}$ are a linear function of the row's control points $P_r \in \mathbb{R}^{3N_r}$ for a fixed spline parameterization $t_r$ and evaluation knots $to_r$:
$$V_r = J_r P_r + b_r$$
* The parameterization $t_r$ depends on the control points but changes only when the user manually edits the shape. Thus, we will build and cache the B-spline interpolation Jacobian $J_r$ once at initialization (when starting the simulation or when the control points are edited).
* The global centerline Jacobian $J \in \mathbb{R}^{3M \times 3N}$ is built as a sparse block-diagonal matrix of $J_r \otimes I_{3\times3}$.

### 2. Tiling & Boundary Conditions for Collisions
To detect collisions across the unit cell boundary, the centerline vertices $V$ are replicated to a 3x3 tiled grid using the period offset vector $\vec{D}$:
$$V_{\text{tiled}, c} = V + c_x \vec{D}_x + c_y \vec{D}_y \quad \text{for } c_x, c_y \in \{-1, 0, 1\}$$
* Tiled space is used strictly for building the collision mesh and evaluating the barrier potential.
* The topology (edges) of the tiled mesh is constructed by replicating the centerline segments across the 9 tile copies.

### 3. Energy, Gradient, and Hessian
* **Stretch & Bend Energies**: Computed strictly on the primary unit cell $V$:
  * **Stretch**: $E_{\text{stretch}} = \sum_{\text{edges}} \frac{1}{2} k_s (l_i - L_0)^2$. We compute the analytical gradient and Hessian per segment. To guarantee PSD, we analytically project the local $3 \times 3$ stretch Hessian:
    $$H_{\text{edge}}^{\text{psd}} = k_s u u^T + \max\left(0, \frac{k_s(l - L_0)}{l}\right)(I - u u^T)$$
  * **Bend**: $E_{\text{bend}} = \sum_{\text{triplets}} \frac{1}{2} k_b \|V_{prev} - 2 V_i + V_{next}\|^2$. Indices are wrapped periodically around the primary unit cell boundaries. The local triplet Hessian is PSD by construction ($k_b A A^T$).
* **Collision Barrier**: Evaluated on the tiled mesh $V_{\text{tiled}}$:
  * Energy $E_{\text{barrier}}$ and gradient $g_{V_{\text{tiled}}, \text{barrier}}$ are computed using `ipctk.BarrierPotential(dhat)`.
  * The sparse tiled Hessian $H_{V_{\text{tiled}}, \text{barrier}}$ is evaluated with local PSD projection:
    $$\text{barrier.hessian}(..., \text{project\_hessian\_to\_psd}=\text{ipctk.PSDProjectionMethod.CLAMP})$$

### 4. Folding to Unit Cell & Projection
Instead of doing projections in the $9N$ tiled space, we fold the gradients and Hessians down to the unit cell before applying the spline Jacobian $J$:
* **Gradient Folding**: Sum the barrier gradients of the 9 copies into a single primary cell gradient $g_{V, \text{barrier}} \in \mathbb{R}^{3M}$:
  $$g_{V, \text{barrier}} = \sum_{c=0}^8 g_{V_{\text{tiled}}, \text{barrier}}[c]$$
  $$g_V = g_{V, \text{stretch}} + g_{V, \text{bend}} + g_{V, \text{barrier}}$$
  $$g_P = J^T g_V$$
* **Hessian Folding**: Fold the tiled sparse barrier Hessian blocks $H_{c_1, c_2}$ into a standard primary cell Hessian $H_{V, \text{barrier}} \in \mathbb{R}^{3M \times 3M}$ by summing their contributions:
  $$H_{V, \text{barrier}} = \sum_{c_1=0}^8 \sum_{c_2=0}^8 H_{V_{\text{tiled}}, \text{barrier}}[c_1, c_2]$$
  $$H_V = H_{V, \text{stretch}} + H_{V, \text{bend}} + H_{V, \text{barrier}}$$
  $$H_P = J^T H_V J$$

### 5. Projected Newton Step & CCD Line Search
* Since local stencils are projected prior to assembly, $H_P \in \mathbb{R}^{3N \times 3N}$ is PSD by construction. We apply a diagonal regularization to ensure positive definiteness:
  $$H_{\text{pd}} = H_P + 10^{-6} I$$
* Solve $H_{\text{pd}} \Delta P = -g_P$ to obtain search direction $\Delta P$.
* CCD step size: compute $\alpha_{\text{max}} \in [0, 1]$ using:
  $$\alpha_{\text{max}} = \text{ipctk.compute\_collision\_free\_stepsize}(\text{mesh}_{\text{tiled}}, V_{\text{tiled}}(P), V_{\text{tiled}}(P + \Delta P))$$
* Backtrack step size $\alpha = \alpha_{\text{max}}$ by factor of $0.5$ until energy decreases:
  $$E(P + \alpha \Delta P) < E(P)$$
* Update: $P \leftarrow P + \alpha \Delta P$.

### 6. Asynchronous Background Thread & GUI Integration
* A background simulation thread executes Newton steps continuously in a loop when `state.sim_active` is enabled.
* Access to `state.ctrl_rows` is protected via a `state.sim_lock` mutex to avoid race conditions during user viewport edits.
* Simulation parameters ($k_s, k_b, k_c, d_{hat}$) and simulation control buttons/toggles are integrated into the GUI sidebar.

## Verification Plan

### Automated Verification
* Add unit test `test_simulation_step` in `tests/test_knitting_core.py` to:
  * Check that `eval_centerline` evaluates correctly.
  * Check that the Jacobian $J$ maps perfectly to spline evaluation (i.e. $J P + b == V$).
  * Verify that the local stretch and bend Hessians are correctly assembled and PSD.
  * Run a single step of the Projected Newton solver and verify that the energy decreases and no collisions occur.

### Manual Verification
* Run `uv run python app.py`.
* Toggle "Run Simulation" in the sidebar, edit the stiffness parameters, and verify in the viewport that the yarn relaxes smoothly.
* Interactively drag a control point while the simulation is running, and verify that the yarn dynamically relaxes to the new constraint.
