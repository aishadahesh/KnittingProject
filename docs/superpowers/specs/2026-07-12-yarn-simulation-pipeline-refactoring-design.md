# Yarn Simulation Pipeline Refactoring Design Spec

This document details the refactoring design to improve the correctness of the physical simulation solver, verify numerical derivatives (gradient and Hessian) using a base class for Objectives, and eliminate code duplication.

## 1. Architectural Changes

We decouple the physics equations (energy potentials, gradients, Hessians) from the spline parameterization/interpolation mapping.

- **`P`** represents the spline control points matrix (shape $N \times 3$).
- **`V`** represents the sampled centerline vertices matrix (shape $M \times 3$).
- **`J_cached`** maps the control points variables to centerline coordinates.

All physical objectives will be defined on the physical centerline coordinates `V` and will not have any knowledge of control points or splines.

---

## 2. Objective Class Hierarchy

We introduce the `Objective` base class and concrete subclasses:

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
        # Default: Computes the numerical gradient via central differences of value(V)
        # Returns shape (M, 3)
        ...

    def hessian(self, V: np.ndarray, eps: float = 1e-5) -> scipy.sparse.csr_matrix:
        # Default: Computes the numerical Hessian via central differences of gradient(V)
        # Returns shape (3M, 3M)
        ...
```

### Concrete Objectives
1. **`StretchObjective`**
   - **`value(V)`**: $\sum_{e \in \text{edges}} \frac{1}{2} (\|V_{v_1} - V_{v_0}\| - L_{0,e})^2$
   - **`gradient(V)`** and **`hessian(V)`**: Analytical implementations computed with unit stiffness.

2. **`BendObjective`**
   - **`value(V)`**: $\sum_{r=0}^{\text{rows}-1} \sum_{i=\text{start}}^{\text{end}-2} \frac{1}{2} \|V_{v_{\text{prev}}} - 2 V_{v_{\text{curr}}} + V_{v_{\text{next}}}\|^2$
   - Boundary condition handles periodic wrap-around offset `period_offset_x`.
   - **`gradient(V)`** and **`hessian(V)`**: Analytical implementations computed with unit stiffness.

3. **`CollisionObjective`**
   - **`value(V)`**, **`gradient(V)`**, and **`hessian(V)`**: Evaluate standard IPC barriers via `ipctk` with unit stiffness.

---

## 3. Pipeline Integration

In `run_simulation_step` and `eval_energy`:

1. Sample centerline positions `V` from control points `P` (via `evaluate_centerlines`).
2. Construct the objectives with uniform parameters (`edges`, `L0_array`, etc.).
3. Evaluate value, gradient, and Hessian in physical space `V`:
   $$\nabla E(V) = k_s \nabla E_{\text{stretch}}(V) + k_b \nabla E_{\text{bend}}(V) + k_c \nabla E_{\text{collision}}(V)$$
   $$H(V) = k_s H_{\text{stretch}}(V) + k_b H_{\text{bend}}(V) + k_c H_{\text{collision}}(V)$$
4. Map gradient and Hessian to control point space `P` using `J_cached`:
   $$g_P = J_{\text{cached}}^T \text{flat}(\nabla E(V))$$
   $$H_P = J_{\text{cached}}^T H(V) J_{\text{cached}}$$

---

## 4. Verification Plan

### Automated Verification
- We will implement unit tests in `tests/test_knitting_core_sim.py` that invoke `check_gradient` and `check_hessian` on the objectives using perturbed centerline coordinates `V` directly.
- The default finite-difference solver in the base class `Objective` acts as the reference implementation.

### Manual Verification
- Run the GUI Verification tool (clicking "FD Verify Derivatives") to verify all components pass the tolerance checks successfully.
