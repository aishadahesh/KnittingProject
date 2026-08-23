"""A pattern must give the fabric something to hang from.

A cleared cell is normally stood in for by the stitch below it, which elongates
across the gap; the topmost active cell stretches to the top of the fabric. Row 0
is the one place that cannot work, because there is nothing below it -- so a
cleared bottom cell is never covered and the rows above hook into nothing.

The first test derives that rule from `compute_bitmap_scale_factors` rather than
restating it, so if the coverage behaviour ever changes, the rule is re-checked
against it instead of quietly going stale.

No GL needed.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from app_state import AppState  # noqa: E402
from knitting_core import (  # noqa: E402
    anchor_bitmap,
    bitmap_is_anchored,
    compute_bitmap_scale_factors,
    unanchored_columns,
)
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


def _uncovered(column):
    """Cells no stitch reaches, derived from the real span function."""
    bitmap = np.asarray(column, dtype=np.float32).reshape(-1, 1)
    spans = compute_bitmap_scale_factors(bitmap).reshape(-1).astype(int)
    covered = set()
    for row, active in enumerate(column):
        if active:
            covered.update(range(row, min(row + spans[row], len(column))))
    return sorted(set(range(len(column))) - covered)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def test_only_a_cleared_bottom_cell_goes_uncovered():
    """Derived from compute_bitmap_scale_factors over every four-cell column."""
    for column in itertools.product([0, 1], repeat=4):
        legal = not _uncovered(list(column))
        assert legal == bool(column[0]), (
            f"column {list(column)} uncovered={_uncovered(list(column))}"
        )


@pytest.mark.parametrize("column,expected", [
    ([1, 1, 0, 1], []),
    ([1, 0, 0, 1], []),
    ([1, 1, 1, 0], []),
    ([1, 0, 1, 0], []),
    ([0, 1, 1, 1], [0]),
    ([0, 0, 1, 1], [0, 1]),
    ([0, 0, 0, 0], [0, 1, 2, 3]),
])
def test_coverage_of_named_column_shapes(column, expected):
    assert _uncovered(column) == expected


def test_unanchored_columns_finds_the_cleared_bottom_cells():
    bitmap = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    assert unanchored_columns(bitmap).tolist() == [1]
    assert unanchored_columns(np.ones((3, 4), dtype=np.float32)).tolist() == []
    assert bitmap_is_anchored(np.ones((3, 4), dtype=np.float32))
    assert not bitmap_is_anchored(bitmap)


def test_anchor_bitmap_leaves_its_input_alone():
    bitmap = np.array([[0, 1], [1, 1]], dtype=np.float32)
    original = bitmap.copy()
    repaired, _ = anchor_bitmap(bitmap)
    assert np.array_equal(bitmap, original)
    assert repaired is not bitmap


def test_anchor_bitmap_changes_only_the_bottom_row():
    bitmap = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float32)
    repaired, columns = anchor_bitmap(bitmap)
    assert columns.tolist() == [0, 2]
    assert repaired[0].tolist() == [1.0, 1.0, 1.0]
    assert np.array_equal(repaired[1:], bitmap[1:])


def test_anchor_bitmap_is_a_no_op_on_a_legal_pattern():
    bitmap = np.array([[1, 1], [0, 1], [1, 0]], dtype=np.float32)
    repaired, columns = anchor_bitmap(bitmap)
    assert columns.size == 0
    assert np.array_equal(repaired, bitmap)
    again, columns_again = anchor_bitmap(repaired)
    assert np.array_equal(again, repaired) and columns_again.size == 0


def test_a_repaired_pattern_leaves_nothing_uncovered():
    """The bridge: the repair satisfies the rule, not just its implementation."""
    for flat in itertools.product([0, 1], repeat=6):
        bitmap = np.asarray(flat, dtype=np.float32).reshape(3, 2)
        repaired, _ = anchor_bitmap(bitmap)
        for col in range(repaired.shape[1]):
            assert _uncovered(repaired[:, col].tolist()) == []


# ---------------------------------------------------------------------------
# The editor refuses the edit
# ---------------------------------------------------------------------------

def _click(state, monkeypatch, row, col, id_suffix=""):
    """Draw the editor with one square reporting a click. Returns its verdict."""
    import gui
    from imgui_bundle import imgui

    target = f"##bm{id_suffix}_{row}_{col}"
    monkeypatch.setattr(imgui, "button", lambda label, *a, **k: label == target)
    monkeypatch.setattr(imgui, "small_button", lambda *a, **k: False)
    monkeypatch.setattr(imgui, "slider_int", lambda label, v, *a, **k: (False, v))
    monkeypatch.setattr(imgui, "text_disabled", lambda *a, **k: None)
    monkeypatch.setattr(imgui, "same_line", lambda *a, **k: None)
    monkeypatch.setattr(imgui, "is_item_hovered", lambda *a, **k: False)
    monkeypatch.setattr(imgui, "set_tooltip", lambda *a, **k: None)
    monkeypatch.setattr(imgui, "push_style_var", lambda *a, **k: None)
    monkeypatch.setattr(imgui, "pop_style_var", lambda *a, **k: None)
    monkeypatch.setattr(imgui, "push_style_color", lambda *a, **k: None)
    monkeypatch.setattr(imgui, "pop_style_color", lambda *a, **k: None)
    return gui._draw_bitmap_editor(state, id_suffix)


def test_clicking_a_bottom_cell_leaves_the_stitch_there(state, monkeypatch):
    state.bitmap[:] = 1.0
    state.on_bitmap_change()
    undo_before = len(state.undo_stack)

    for col in range(int(state.bitmap.shape[1])):
        changed = _click(state, monkeypatch, 0, col)
        assert changed is False, f"clicking bottom cell {col} should do nothing"
        assert float(state.bitmap[0, col]) == 1.0

    assert np.all(state.bitmap[0] > 0.5)
    # A refused click must not leave an undo entry that appears to do nothing.
    assert len(state.undo_stack) == undo_before


def test_the_puzzle_editor_refuses_too(state, monkeypatch):
    """Both call sites share the guard."""
    state.bitmap[:] = 1.0
    state.on_bitmap_change()
    assert _click(state, monkeypatch, 0, 0, id_suffix="_puzzle") is False
    assert float(state.bitmap[0, 0]) == 1.0


def test_rows_above_the_bottom_still_toggle(state, monkeypatch):
    state.bitmap[:] = 1.0
    state.on_bitmap_change()
    for row in range(1, int(state.bitmap.shape[0])):
        undo_before = len(state.undo_stack)
        assert _click(state, monkeypatch, row, 0) is True
        assert float(state.bitmap[row, 0]) == 0.0
        assert len(state.undo_stack) == undo_before + 1


def test_a_cleared_bottom_cell_can_be_switched_back_on(state, monkeypatch):
    """So a pattern that arrived cleared is fixable by clicking it."""
    state.bitmap[0, 0] = 0.0
    assert _click(state, monkeypatch, 0, 0) is True
    assert float(state.bitmap[0, 0]) == 1.0


# ---------------------------------------------------------------------------
# The ways a bad pattern can arrive
# ---------------------------------------------------------------------------

def test_on_bitmap_change_backstops_a_direct_write(state):
    state.bitmap[0, 0] = 0.0
    state.on_bitmap_change()
    assert float(state.bitmap[0, 0]) == 1.0


def test_resize_does_not_carry_a_violation_through(state):
    state.bitmap[0, :] = 0.0
    state.on_bitmap_resize(5, 3)
    assert np.all(state.bitmap[0] > 0.5)


def test_loading_a_broken_pattern_repairs_the_geometry_too(state, tmp_path):
    """The regression guard for the reconciliation.

    Repairing only the bitmap leaves the saved control rows carrying the flat
    bottom row, so the pattern would read as anchored while the fabric still
    showed the bar.
    """
    state.on_bitmap_resize(4, 2)
    state.bitmap[:] = 1.0
    state.on_bitmap_change()
    state.save_params(state.save_path, silent=True)

    saved = json.loads(Path(state.save_path).read_text(encoding="utf-8"))
    saved["bitmap"] = [[0, 0], [1, 1], [1, 1], [1, 1]]
    # The degenerate row a cleared bottom cell produces, as a real file holds it.
    rows = [np.asarray(r, dtype=float) for r in saved["spline_control_rows"]]
    rows[0][:, 1] = 0.0
    saved["spline_control_rows"] = [r.tolist() for r in rows]
    Path(state.save_path).write_text(json.dumps(saved), encoding="utf-8")

    above_before = [float(r[:, 1].max() - r[:, 1].min()) for r in rows[1:]]
    state.load_params(state.save_path)

    assert np.all(state.bitmap[0] > 0.5)
    assert "bottom row" in state.status_msg
    spans = [float(r[:, 1].max() - r[:, 1].min()) for r in state.ctrl_rows]
    assert spans[0] > 1e-6, "the repaired bottom row must have a real loop"
    assert spans[1:] == pytest.approx(above_before, rel=1e-4), (
        "the rows above must keep the geometry they were saved with"
    )


def test_loading_a_legal_pattern_says_nothing_about_repair(state):
    state.bitmap[:] = 1.0
    state.on_bitmap_change()
    state.save_params(state.save_path, silent=True)
    state.load_params(state.save_path)
    assert "bottom row" not in state.status_msg


def test_restoring_a_broken_snapshot_repairs_it(state):
    snapshot = state.snapshot_state()
    snapshot["bitmap"] = np.asarray(snapshot["bitmap"], dtype=np.float32).copy()
    snapshot["bitmap"][0, :] = 0.0
    anchored = state.restore_snapshot(snapshot)
    assert np.all(state.bitmap[0] > 0.5)
    assert len(anchored) > 0


def test_a_clean_undo_message_is_unchanged(state):
    """No note when nothing was repaired -- otherwise every undo would nag."""
    state.bitmap[:] = 1.0
    state.on_bitmap_change()
    state.push_undo("Pattern")
    state.bitmap[2, 0] = 0.0
    state.on_bitmap_change()
    state.undo_last()
    assert state.status_msg == "Undid last change"


def test_reset_keeps_the_bottom_row(state):
    state.bitmap[:] = 1.0
    state.on_bitmap_change()
    state.capture_initial_state()
    state.bitmap[0, :] = 0.0
    state.reset_to_initial()
    assert np.all(state.bitmap[0] > 0.5)


# ---------------------------------------------------------------------------
# The deliberate non-change
# ---------------------------------------------------------------------------

def test_scan_random_patterns_are_left_unconstrained(state):
    """Scan Mode's generator is deliberately NOT held to this rule.

    Its render path substitutes an all-active bitmap and carries the pattern in
    loop heights instead, so no scanned image is affected. Constraining it would
    change the scan configuration signature and stop duplicate detection
    recognising the captures already in the database.
    """
    patterns = [state._scanner_random_bitmap(i) for i in range(200)]
    assert any(float(bm[0].min()) < 0.5 for bm in patterns)
