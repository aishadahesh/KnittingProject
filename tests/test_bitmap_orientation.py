"""The stitch grid must read the same way up as the fabric it describes.

Rows are built at `y = row_idx * dy`, so index 0 is the lowest row knitted. The
editor drew index 0 first, which imgui lays out at the top -- so the grid was a
mirror image of the model. Clearing the third square from the top opened a gap
second from the top, and the stitch that grew to fill it was not the one under
the square that had been clicked.

These tests pin both halves: the fabric's own convention, and that the grid is
drawn to match it. Bottom-up is also how a knitting chart is read.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from app_state import AppState  # noqa: E402
from rendering import Camera  # noqa: E402


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
    return app_state


def _bases(state):
    return [float(row[:, 1].min()) for row in state.ctrl_rows]


def _spans(state):
    return [float(row[:, 1].max() - row[:, 1].min()) for row in state.ctrl_rows]


# ---------------------------------------------------------------------------
# The fabric's convention, which the grid has to match
# ---------------------------------------------------------------------------

def test_row_zero_is_the_bottom_of_the_fabric(state):
    bases = _bases(state)
    assert bases == sorted(bases), "row index must increase upward"
    assert bases[0] == pytest.approx(min(bases))


def test_clearing_a_row_grows_the_row_below_it(state):
    """The stitch under a gap elongates to stand in for the missing one."""
    state.bitmap[:] = 1.0
    state.on_bitmap_change()
    natural = _spans(state)
    bases = _bases(state)

    cleared = 2
    state.bitmap[cleared, :] = 0.0
    state.on_bitmap_change()
    spans = _spans(state)

    assert spans[cleared] == pytest.approx(0.0, abs=1e-6), "a cleared row has no loop"
    below = cleared - 1
    # It should finish exactly where the row it replaces would have finished.
    would_have_ended = bases[cleared] + natural[cleared]
    assert bases[below] + spans[below] == pytest.approx(would_have_ended, rel=1e-4)
    # And nothing else moves.
    for other in range(len(spans)):
        if other not in (cleared, below):
            assert spans[other] == pytest.approx(natural[other], rel=1e-6)


# ---------------------------------------------------------------------------
# The grid is drawn the same way up
# ---------------------------------------------------------------------------

def _drawn_row_order(state, monkeypatch):
    """Row indices in the order the editor emits them, top of the grid first."""
    import gui
    from imgui_bundle import imgui

    seen = []
    real_button = imgui.button

    def spy(label, *args, **kwargs):
        marker = "##bm_"
        if isinstance(label, str) and marker in label:
            # id is "##bm{suffix}_{row}_{col}"
            row = int(label.split(marker, 1)[1].split("_")[0])
            if row not in seen:
                seen.append(row)
        return False

    monkeypatch.setattr(imgui, "button", spy)
    monkeypatch.setattr(imgui, "small_button", lambda *a, **k: False)
    monkeypatch.setattr(imgui, "slider_int", lambda label, v, *a, **k: (False, v))
    monkeypatch.setattr(imgui, "text_disabled", lambda *a, **k: None)
    monkeypatch.setattr(imgui, "same_line", lambda *a, **k: None)
    monkeypatch.setattr(imgui, "push_style_var", lambda *a, **k: None)
    monkeypatch.setattr(imgui, "pop_style_var", lambda *a, **k: None)
    monkeypatch.setattr(imgui, "push_style_color", lambda *a, **k: None)
    monkeypatch.setattr(imgui, "pop_style_color", lambda *a, **k: None)

    gui._draw_bitmap_editor(state)
    imgui.button = real_button
    return seen


def test_the_grid_is_drawn_bottom_up(state, monkeypatch):
    """Index 0 must be emitted last, so it lands at the bottom of the grid."""
    order = _drawn_row_order(state, monkeypatch)
    n_rows = int(state.bitmap.shape[0])
    assert order == list(reversed(range(n_rows))), (
        "the stitch grid must be drawn bottom-up so it matches the fabric; "
        f"got {order}"
    )


def test_the_square_clicked_matches_the_row_that_grows(state, monkeypatch):
    """The reported bug, stated as the user experiences it.

    Clicking the third square from the top of the grid must clear the third row
    from the top of the fabric, and the row visually beneath it is the one that
    grows.
    """
    # Four rows so "third from the top" still has a row beneath it to grow.
    state.on_bitmap_resize(4, 2)
    state.bitmap[:] = 1.0
    state.on_bitmap_change()

    order = _drawn_row_order(state, monkeypatch)
    n_rows = int(state.bitmap.shape[0])
    assert n_rows == 4

    third_from_top = order[2]
    state.bitmap[third_from_top, :] = 0.0
    state.on_bitmap_change()

    bases = _bases(state)
    spans = _spans(state)
    # Third from the top of the fabric, counted by height.
    by_height = sorted(range(n_rows), key=lambda i: -bases[i])
    assert by_height[2] == third_from_top, "the cleared row is not where it was clicked"

    grew = by_height[3]
    assert spans[third_from_top] == pytest.approx(0.0, abs=1e-6)
    assert spans[grew] > spans[by_height[0]], "the row beneath the gap should have grown"
