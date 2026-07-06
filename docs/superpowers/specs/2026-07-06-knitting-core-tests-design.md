# Design Specification: Unit Tests for Knitting Core

This document outlines the design for unit tests covering [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py). The test suite aims to verify numerical and geometry outputs of the core computation functions, ensure JAX JIT execution behaves correctly on CPU backend, and validate that `knitting_core.py` remains the pure, headless core of the application without UI/state dependencies.

## 1. Architectural Goals

1. **Pure Functional Verification**: Validate that all core mathematical operations inside [knitting_core.py](file:///home/roip/projects/KnittingProject/knitting_core.py) behave correctly as pure, stateless functions.
2. **Headless Boundary Verification**: Verify that the core module does not import or depend on UI frameworks (e.g., `moderngl`, `imgui_bundle`, `glfw`) or GUI state handlers ([app_state.py](file:///home/roip/projects/KnittingProject/app_state.py)).
3. **Environment Independence**: Ensure tests run reliably on CPU-only environments (e.g., standard CI pipelines, local CPU-only virtual environments) by explicitly configuring JAX.

---

## 2. Test Setup & Configuration

- **Framework**: `pytest`.
- **Backend Configuration**: Set environment variable `XLA_PYTHON_CLIENT_PREALLOCATE=false` and configure JAX to target the CPU platform:
  ```python
  import jax
  jax.config.update("jax_platform_name", "cpu")
  ```
- **Fixtures**:
  - `config`: Loads parameters and settings from [config.json](file:///home/roip/projects/KnittingProject/config.json).
  - `params_data`: Loads default parameters, bitmap, and expected row properties from [params.json](file:///home/roip/projects/KnittingProject/params.json).

---

## 3. Test Cases

We will implement the following tests in `tests/test_knitting_core.py`:

### 3.1. Headless Isolation Check
- **`test_core_module_imports`**:
  - Asserts that importing `knitting_core` does not trigger imports of UI modules (`moderngl`, `imgui_bundle`, `glfw`) or application components (`app_state`, `gui`).
  - Asserts that all core functions accept pure parameters (lists, dictionaries, or numpy/jax arrays) and return python/numpy/jax primitives (no stateful objects or UI references).

### 3.2. JAX Math Utilities
- **`test_scale_factors_jax`**:
  - Verifies scale factor computation given standard binary bitmap patterns.
  - Verifies shapes and output ranges.
- **`test_eval_curve_and_derivative`**:
  - Evaluates both [eval_curve](file:///home/roip/projects/KnittingProject/knitting_core.py#L50) and [eval_curve_derivative](file:///home/roip/projects/KnittingProject/knitting_core.py#L59) across a range of parameters.
  - Verifies output shape is `(..., 3)`.
- **`test_compute_orthonormal_frame`**:
  - Evaluates orthonormal frames on positive, negative, and edge-case tangents.
  - Asserts that output frames $u$ and $v$ are perpendicular to the tangent $t$ ($\langle t, u \rangle \approx 0, \langle t, v \rangle \approx 0$) and to each other ($\langle u, v \rangle \approx 0$).

### 3.3. Knitting Mesh and Vertices Generation
- **`test_compute_knitting_vertices_and_faces`**:
  - Executes [compute_knitting_vertices](file:///home/roip/projects/KnittingProject/knitting_core.py#L102) using config/parameter fixtures.
  - Verifies the structure of returned vertices is a list of `(vertices, points_per_segment)` pairs.
  - Feeds output into [compute_knitting_faces](file:///home/roip/projects/KnittingProject/knitting_core.py#L117) and verifies returned faces list matches the vertex lengths and shapes (faces are 4-sided quadrilaterals).

### 3.4. Spline Construction
- **`test_build_parametric_control_rows`**:
  - Builds control rows and validates shape matches rows/columns in parameter/bitmap specs.
- **`test_build_spline_mesh`**:
  - Builds spline meshes from control rows and checks dimensions.

### 3.5. Fiber Mesh Generation
- **`test_build_surface_fiber_meshes`**:
  - With `enabled=False`: asserts output meshes match input vertex structures.
  - With `enabled=True`: asserts output meshes have the correct shapes/dimensions and metadata maps row indices properly.

---

## 4. Verification & Running Tests

- Run tests using:
  ```bash
  .venv/bin/pytest tests/test_knitting_core.py
  ```
