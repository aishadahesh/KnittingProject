# Limit Autosave Frequency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Defer autosaving of parameters during continuous mouse and keyboard dragging interactions to limit disk write frequency and prevent micro-stutters.

**Architecture:** We will modify `AppState.maybe_autosave` in `app_state.py` to check active user interaction flags (`gizmo_edit_active`, `spline_grab_active`, `radius_grab_active`, `model_drag_undo_active`, `spline_keyboard_edit_active`, `radius_keyboard_edit_active`, or `bbox_active_handle >= 0`) and return early without saving if any are active. We will also add a corresponding unit test in `tests/test_autosave.py`.

**Tech Stack:** Python, pytest, NumPy.

## Global Constraints
- None

---

### Task 1: Write failing test for deferred autosaving

**Files:**
- Create: `tests/test_autosave.py`

**Interfaces:**
- Consumes: `app_state.AppState`

- [ ] **Step 1: Create the test file with a failing test**

```python
import os
import sys
import time
import pytest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force JAX to use CPU backend
import jax
jax.config.update("jax_platform_name", "cpu")

from app_state import AppState

class DummyCamera:
    def __init__(self):
        self.dist = 10.0
        self.az = 0.0
        self.el = 0.0
        self.target = [0.0, 0.0, 0.0]

def test_autosave_deferred_during_dragging(tmp_path):
    test_params_file = tmp_path / "test_params.json"
    
    state = AppState(camera=DummyCamera(), renderer=None)
    state.save_path = str(test_params_file)
    state.load_path = str(test_params_file)
    state.autosave_enabled = True
    state.autosave_interval_sec = 0.05
    state.autosave_last_time = time.monotonic() - 1.0
    
    # By default, not dragging, so it should save
    assert not test_params_file.exists()
    state.maybe_autosave()
    assert test_params_file.exists()
    
    # Delete the file and verify autosave is skipped during active drags
    test_params_file.unlink()
    
    interaction_flags = [
        ('gizmo_edit_active', True),
        ('spline_grab_active', True),
        ('radius_grab_active', True),
        ('model_drag_undo_active', True),
        ('spline_keyboard_edit_active', True),
        ('radius_keyboard_edit_active', True),
    ]
    
    for flag, value in interaction_flags:
        state.autosave_last_time = time.monotonic() - 1.0
        setattr(state, flag, value)
        state.maybe_autosave()
        assert not test_params_file.exists(), f"Autosave should be skipped when {flag} is active"
        setattr(state, flag, False)  # Reset
        
    # Test bbox_active_handle >= 0
    state.autosave_last_time = time.monotonic() - 1.0
    state.bbox_active_handle = 1
    state.maybe_autosave()
    assert not test_params_file.exists(), "Autosave should be skipped when bbox_active_handle >= 0"
    
    # Reset bbox_active_handle and verify it autosaves
    state.bbox_active_handle = -1
    state.maybe_autosave()
    assert test_params_file.exists(), "Autosave should succeed once interaction ends"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_autosave.py -v`
Expected: FAIL with assertion error (since active flags are not yet checked, it will write the file anyway).

- [ ] **Step 3: Commit**

```bash
git add tests/test_autosave.py
git commit -m "test: add test for deferred autosaving"
```

---

### Task 2: Implement deferred autosaving checks

**Files:**
- Modify: `app_state.py`

**Interfaces:**
- Produces: Updated `maybe_autosave()` method that checks user interaction flags before writing.

- [ ] **Step 1: Modify `maybe_autosave` in `app_state.py`**

In [app_state.py](file:///home/roip/projects/KnittingProject/app_state.py), update the `maybe_autosave` method:

```python
    def maybe_autosave(self):
        if not self.autosave_enabled:
            return
        
        # Defer autosaving during active user interactions to prevent micro-stutters
        if (self.get('gizmo_edit_active', False) or
            self.get('spline_grab_active', False) or
            self.get('radius_grab_active', False) or
            self.get('model_drag_undo_active', False) or
            self.get('spline_keyboard_edit_active', False) or
            self.get('radius_keyboard_edit_active', False) or
            int(self.get('bbox_active_handle', -1)) >= 0):
            return

        now = time.monotonic()
        if now - float(self.autosave_last_time) < float(self.autosave_interval_sec):
            return
        target_path = self.save_path or os.path.join(self.project_root, 'params.json')
        self.save_params(target_path, silent=True)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_autosave.py -v`
Expected: PASS

- [ ] **Step 3: Run all unit tests to make sure no regressions**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add app_state.py
git commit -m "feat: defer autosaving during active user interactions"
```
