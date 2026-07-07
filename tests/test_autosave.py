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
