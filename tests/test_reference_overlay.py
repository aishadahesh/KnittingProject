"""The reference image belongs to Edit Mode, and modes must not leak into it.

The reference photograph exists so the model can be aligned against it, which is
an editing task. It was appearing behind Puzzle Mode's fabric because the only
gate was "not a scanner mode".

The fix is a display gate rather than a write to `show_ref_bg`: the checkbox
stays the user's stored preference, so leaving Edit Mode hides the overlay and
returning restores it. These tests pin that -- particularly that nothing ever
writes the preference, which is what would lose it across a mode switch.

No GL needed; the gate is a pure function of state.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from app_state import AppState  # noqa: E402
from rendering import Camera  # noqa: E402

MODES = ("edit", "scan", "puzzle", "database", "ur5")


class StubRenderer:
    mesh_pick_data = None

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture
def state(tmp_path):
    app_state = AppState(Camera(), StubRenderer())
    app_state.save_path = str(tmp_path / "params.json")
    app_state.load_path = str(tmp_path / "params.json")
    app_state.autosave_enabled = False
    app_state.rebuild_spline_from_params()
    app_state.capture_initial_state()
    return app_state


@pytest.fixture
def overlay_active():
    import gui

    return gui._reference_overlay_active


def test_overlay_draws_in_edit_mode_only(state, overlay_active):
    state.show_ref_bg = True
    drawn = {}
    for mode in MODES:
        state.app_mode = mode
        drawn[mode] = overlay_active(state)
    assert drawn["edit"] is True
    assert not any(drawn[m] for m in MODES if m != "edit"), (
        f"reference image leaked into {[m for m in MODES if m != 'edit' and drawn[m]]}"
    )


def test_overlay_stays_off_when_the_checkbox_is_off(state, overlay_active):
    state.show_ref_bg = False
    for mode in MODES:
        state.app_mode = mode
        assert overlay_active(state) is False


def test_leaving_and_returning_to_edit_restores_the_overlay(state, overlay_active):
    """Requirements 4, 5 and 6 together: hide on leave, restore on return."""
    state.app_mode = "edit"
    state.show_ref_bg = True
    assert overlay_active(state) is True

    state.app_mode = "puzzle"
    assert overlay_active(state) is False
    # The preference itself must survive -- this is what "do not permanently
    # lose the checkbox state" means.
    assert state.show_ref_bg is True

    state.app_mode = "edit"
    assert overlay_active(state) is True


def test_a_disabled_overlay_is_still_disabled_on_return(state, overlay_active):
    state.app_mode = "edit"
    state.show_ref_bg = False
    state.app_mode = "puzzle"
    state.app_mode = "edit"
    assert overlay_active(state) is False


# ---------------------------------------------------------------------------
# Resetting and undoing must not move the user between modes
# ---------------------------------------------------------------------------

def test_reset_does_not_change_the_mode(state):
    """Reset restores the model, not where the user is standing.

    It used to restore app_mode out of the snapshot, which moved the user
    without going through _set_app_mode -- so the mode they left was never torn
    down and the one they landed in was never set up.
    """
    state.app_mode = "puzzle"
    state.reset_to_initial()
    assert state.app_mode == "puzzle"


def test_undo_does_not_change_the_mode(state):
    state.app_mode = "puzzle"
    state.push_undo("something")
    state.app_mode = "puzzle"
    state.undo_last()
    assert state.app_mode == "puzzle"


def test_reset_in_puzzle_mode_does_not_bring_back_the_overlay(state, overlay_active):
    """The reported symptom, end to end."""
    state.app_mode = "puzzle"
    state.show_ref_bg = False
    state.reset_to_initial()
    assert state.app_mode == "puzzle"
    assert overlay_active(state) is False


def test_app_mode_is_not_snapshotted(state):
    assert "app_mode" not in state.snapshot_state()
