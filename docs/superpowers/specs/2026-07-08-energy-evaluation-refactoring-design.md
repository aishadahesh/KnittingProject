# Energy Evaluation Refactoring & Simulation Visualization

This document outlines the design for improving the simulation energy evaluation (`eval_energy`), fixing vectorization bugs, adding robust collision testing, and exposing energy visualizers to the user.

## 1. Architecture & Refactoring (`knitting_core.py`)

### DRY `V` Creation
Currently, the generation of centerline vertices (`V`) and edges is duplicated between the physical simulation (`eval_energy`) and the mesh construction (`build_spline_mesh`). 
We will introduce a single source of truth for generating centerlines:
`evaluate_centerlines(ctrl_rows, D, config) -> (V, edges)`
Both the rendering pipeline and the energy evaluation functions will call this method to ensure structural consistency across the application.

### `L0` Source of Truth
The nominal rest length (`L0`) for the elastic energy objective is currently computed dynamically as a uniform straight-line distance `bitmap_width / (nout - 1)`. Because the control points define a Cubic Spline, the actual Euclidean arc-lengths of the generated segments are not uniform. 
We will compute an array of `L0` rest-lengths corresponding to the exact 3D Euclidean distances of the segments when the mesh is first initialized. This `L0` array will be stored in `app_state.py` and passed into `eval_energy` as a constant source of truth.

### Vectorized Loops & Bending Bug Fix
The Python `for` loops used to compute elastic (`e_el`) and bending (`e_b`) energies are slow. We will replace them with fast, vectorized NumPy operations. 
Additionally, we will fix a bug in the bending term where the modulo operator (`% M`) incorrectly wrapped the evaluation across different spline rows.

## 2. Testing Collision (`test_knitting_core_sim.py`)

To ensure the collision physics objective is well-behaved, we will implement headless unit tests focusing on interacting splines.
- **Can-Collide Filtering**: The `build_can_collide` filter matrix will be ignored during testing (and optionally removed from `eval_energy` pending testing results) to simplify debugging.
- **Intersecting Yarn Test**: We will create a test case with two straight yarns positioned in an 'X' shape so that they explicitly intersect. The test will evaluate the collision energy, perturb one of the yarns away in the Z-axis, and assert that the collision energy strictly decreases as the distance between the centerlines increases.

## 3. Visualization (`gui.py` & `rendering.py`)

To aid in debugging the physics solver, we will expose the energy state and forces directly to the user interface.

- **UI Metrics Panel**: A real-time text readout will be added to the ImGui control panel, displaying the scalar values for the Total Energy, Elastic Energy (`e_el`), Bending Energy (`e_b`), and Collision Energy (`e_col`).
- **Force Vectors**: The analytical gradient function will be used to calculate the gradient (force) vector at each vertex. The 3D renderer will be updated to draw these forces as arrows/lines originating from the vertices, pointing in the direction the solver intends to move the geometry.
