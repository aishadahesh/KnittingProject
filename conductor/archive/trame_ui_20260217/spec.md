# Track Specification: Replace UI with Trame-based Frontend

## Overview
This track involves replacing the current `vedo`-based GUI with a modern, reactive frontend powered by **Trame** from Kitware. The goal is to leverage Trame's robust integration with VTK and its reactive state management to create a more stable and interactive reconstruction tool.

## Functional Requirements
1.  **Unified Three-Column Interface:**
    - **Display 1: Mesh Editor:** An interactive VTK-based view for geometry manipulation.
    - **Display 2: Mitsuba Render:** A dedicated image viewer showing the latest render output.
    - **Display 3: Reference Image:** A static viewer for the target reconstruction image.
2.  **Native Trame Interaction:**
    - **Spline Editing:** Implement interactive 3D widgets (VTK handle/point widgets) for direct manipulation of control points within the Mesh Editor.
    - **Parameter Tuning:** A dedicated sidebar containing reactive sliders and input fields for all global knitting parameters defined in `config.json`.
3.  **Rendering Workflow:**
    - Explicit **"Render" Button** to trigger a high-quality Mitsuba render on demand.
    - Status indicators to show when a render or optimization is in progress.
4.  **Optimization Integration:**
    - Trigger and monitor the JAX-based optimization loop directly from the Trame UI.
    - Real-time updates of parameters and mesh previews as the optimizer converges.

## Non-Functional Requirements
- **Responsiveness:** UI state changes (parameter sliders) should update the Mesh Editor view near-instantaneously using reactive bindings.
- **Stability:** Replace the version-sensitive `vedo` interactor logic with Trame's more robust client-server model.

## Acceptance Criteria
- A single Trame application can be launched from `optimize_knitting.py` (or a dedicated entry point).
- Users can switch between "Parameter Mode" and "Spline Mode" seamlessly.
- Manual edits to spline control points are reflected in global parameter estimates.
- The "Render" command correctly invokes Mitsuba and displays the result in the middle column.
