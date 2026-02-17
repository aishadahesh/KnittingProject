# Implementation Plan - Refactor and Unify Knitting Model Optimizer

## Phase 1: Configuration & Foundation
- [x] Task: Create `config.json` and implement centralized loading.
    - [x] Identify and extract all hardcoded constants from `optimize_knitting.py`.
    - [x] Create `config.json` with logical sections (Geometry, Rendering, Optimization, UI).
    - [x] Implement `load_config()` and replace hardcoded references in the script.
- [x] Task: Logical Reorganization of Script Structure.
    - [x] Reorder imports and global setup.
    - [x] Move Geometry Engine functions (JAX) to the top section.
    - [x] Move Rendering Pipeline and Loss functions to the middle section.
    - [x] Organize the Optimization loop and UI App at the bottom.

## Phase 2: Unified UI Implementation
- [x] Task: Design the Unified UI State Machine.
    - [x] Define a single state object to manage parameters, splines, and current view mode.
- [x] Task: Merge Spline and Parameter Editors.
    - [x] Consolidate key event handlers into a single unified listener.
    - [x] Update the drawing logic to overlay spline controls and parameter feedback in one window.
- [x] Task: Integrate Optimization - [ ] Task: Integrate Optimization & Rendering View. Rendering View.
    - [x] Ensure the unified window can switch between or simultaneously show model previews and optimization progress.

## Phase 3: Consolidation & Cleanup
- [x] Task: Apply DRY and Flatten Code.
    - [x] Consolidate single-use functions.
    - [x] Remove deep nesting in the main optimization and UI loops.
- [x] Task: Verification and Final Polish.
    - [x] Verify that the differentiable pipeline still computes gradients correctly after refactoring.
    - [x] Ensure all features (Spline edit, Param edit, Optimize) are accessible from the new unified UI.

- [x] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)
