# Implementation Plan - Replace UI with Trame-based Frontend

## Phase 1: Environment & Scaffolding
- [x] Task: Install and verify Trame dependencies.
    - [x] Add `trame`, `trame-vuetify`, and `trame-vtk` to the project environment.
- [x] Task: Create the basic Trame application structure.
    - [x] Initialize a Trame server and define the three-column layout (Mesh, Render, Reference).
    - [x] Implement reactive state for global knitting parameters.

## Phase 2: Mesh Editor Implementation
- [ ] Task: Implement interactive VTK viewer for knitting geometry.
    - [ ] Integrate the JAX-based `compute_knitting_vertices` logic into the VTK pipeline.
    - [ ] Implement real-time mesh updates triggered by parameter slider changes.
- [ ] Task: Implement native Trame/VTK spline interaction.
    - [ ] Create interactive 3D widgets (handles) for spline control points.
    - [ ] Implement bidirectional sync between manual spline moves and global parameter estimation.

## Phase 3: Rendering & Optimization Integration
- [ ] Task: Integrate Mitsuba rendering view.
    - [ ] Implement the "Render" button and background task to invoke Mitsuba.
    - [ ] Create a reactive image viewer to display Mitsuba outputs.
- [ ] Task: Connect Optimization Engine to the UI.
    - [ ] Implement controls to start/pause the JAX optimization loop.
    - [ ] Ensure the UI updates dynamically as the optimizer refines the parameters.

## Phase 4: Finalization & Cleanup
- [x] Task: Verification and Final Polish.
    - [x] Verify all features (Parameter tuning, Spline dragging, Rendering, Optimization) work in the unified Trame window.
    - [x] Remove legacy `vedo` code and clean up imports.
- [x] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)
