# Implementation Plan - Refactor and Unify Knitting Model Optimizer

## Phase 1: Configuration & Foundation
- [ ] Task: Create `config.json` and implement centralized loading.
    - [ ] Identify and extract all hardcoded constants from `optimize_knitting.py`.
    - [ ] Create `config.json` with logical sections (Geometry, Rendering, Optimization, UI).
    - [ ] Implement `load_config()` and replace hardcoded references in the script.
- [ ] Task: Logical Reorganization of Script Structure.
    - [ ] Reorder imports and global setup.
    - [ ] Move Geometry Engine functions (JAX) to the top section.
    - [ ] Move Rendering Pipeline and Loss functions to the middle section.
    - [ ] Organize the Optimization loop and UI App at the bottom.

## Phase 2: Unified UI Implementation
- [ ] Task: Design the Unified UI State Machine.
    - [ ] Define a single state object to manage parameters, splines, and current view mode.
- [ ] Task: Merge Spline and Parameter Editors.
    - [ ] Consolidate key event handlers into a single unified listener.
    - [ ] Update the drawing logic to overlay spline controls and parameter feedback in one window.
- [ ] Task: Integrate Optimization & Rendering View.
    - [ ] Ensure the unified window can switch between or simultaneously show model previews and optimization progress.

## Phase 3: Consolidation & Cleanup
- [ ] Task: Apply DRY and Flatten Code.
    - [ ] Consolidate single-use functions.
    - [ ] Remove deep nesting in the main optimization and UI loops.
- [ ] Task: Verification and Final Polish.
    - [ ] Verify that the differentiable pipeline still computes gradients correctly after refactoring.
    - [ ] Ensure all features (Spline edit, Param edit, Optimize) are accessible from the new unified UI.

- [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)
