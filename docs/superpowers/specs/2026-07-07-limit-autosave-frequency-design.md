# Design Spec: Defer Autosaving During Continuous UI Dragging

## Goal Description
To prevent disk performance issues, file-locking, micro-stutters, and excessive git status changes caused by writing `params.json` to the filesystem during continuous operations like dragging spline control points, scaling the bounding box, or dragging/moving the model. 

Currently, `state.maybe_autosave()` is called in the main GUI rendering loop. If the `autosave_interval_sec` (1.0 second) has elapsed, it performs a blocking write of the parameter configuration. If a user holds and drags a control point, this write happens multiple times mid-drag.

We will modify `maybe_autosave()` to skip saving if any active user manipulation flags are set to `True` (e.g. `gizmo_edit_active`, `spline_grab_active`, `radius_grab_active`, `model_drag_undo_active`, `spline_keyboard_edit_active`, `radius_keyboard_edit_active`, or `bbox_active_handle >= 0`). 

Once the interaction finishes and all flags return to `False`, the subsequent frame immediately writes the final state because the elapsed time will still be greater than the autosave interval.

## Proposed Changes

### Centralized State and Autosaving Logic

#### [MODIFY] [app_state.py](file:///home/roip/projects/KnittingProject/app_state.py)
* Update `maybe_autosave(self)`:
  * Check if the user is in the middle of a continuous edit session.
  * We will inspect:
    * `self.get('gizmo_edit_active', False)`
    * `self.get('spline_grab_active', False)`
    * `self.get('radius_grab_active', False)`
    * `self.get('model_drag_undo_active', False)`
    * `self.get('spline_keyboard_edit_active', False)`
    * `self.get('radius_keyboard_edit_active', False)`
    * `int(self.get('bbox_active_handle', -1)) >= 0`
  * If any of these conditions are true, return early without saving.

## Verification Plan

### Manual Verification
1. Run the application using `uv run python app.py`.
2. Grab a control point and move it around continuously using the ImGui gizmo for 5-10 seconds.
3. Observe that no "Saved -> params.json" status message appears, and no disk writes happen while dragging is active.
4. Release the control point.
5. Verify that the "Saved -> params.json" status message appears immediately after releasing the drag, and that `params.json` contains the updated control point coordinate.
6. Verify keyboard control point editing ('G' then drag, WASD edit keys, 'R' radius edit, bounding box resize) all defer autosaves until completion.
