"""The scan pattern grid is a setting, and the scanned fabric follows it.

`scanner_pattern_rows` / `scanner_pattern_cols` were saved, fed the tile-cache
key and counted toward the scan configuration signature, but nothing read them:
both the generator and the plan geometry took their size from the scan template's
bitmap and only fell back to the settings when the template had none, which never
happened. So the grid was welded to the template's 4x2 and resizing the model
changed nothing.

The row count is not cosmetic -- the loop-height grid has one entry per stitch,
so the pattern's rows are the scanned fabric's rows. These tests pin that the two
move together and that the heights still land exactly.

No GL needed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from app_state import AppState  # noqa: E402
from knitting_core import build_parametric_control_rows  # noqa: E402
from rendering import Camera  # noqa: E402
from scanner_core import _scanner_pattern_dimensions  # noqa: E402
from scanner_storage import scan_config_signature  # noqa: E402

PEAK = 0.904508  # crest reached at five samples per loop


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


def _template_shape(state):
    bitmap = np.asarray(state._scanner_template().get("bitmap", np.empty((0, 0))), dtype=np.float32)
    return (int(bitmap.shape[0]), int(bitmap.shape[1]))


# ---------------------------------------------------------------------------
# The setting is honoured
# ---------------------------------------------------------------------------

def test_defaults_still_give_the_template_size(state):
    """Untouched settings must not change what anyone was already scanning."""
    assert state._scanner_pattern_size() == _template_shape(state)
    assert state._scanner_random_bitmap(0).shape == _template_shape(state)


@pytest.mark.parametrize("rows,cols", [(3, 2), (5, 3), (2, 4)])
def test_the_generator_uses_the_setting(state, rows, cols):
    state.scanner_pattern_rows = rows
    state.scanner_pattern_cols = cols
    assert state._scanner_pattern_size() == (rows, cols)
    for cell in range(4):
        assert state._scanner_random_bitmap(cell).shape == (rows, cols)


def test_the_plan_geometry_uses_the_same_setting(state):
    """_scanner_pattern_dimensions feeds the tile-cache key, so it has to agree."""
    state.scanner_pattern_rows = 3
    state.scanner_pattern_cols = 4
    assert _scanner_pattern_dimensions(state) == (3, 4)
    assert _scanner_pattern_dimensions(state) == state._scanner_pattern_size()


def test_the_size_is_clamped_to_what_the_model_can_describe(state):
    """Rows beyond the loop-height parameters would all reuse the last one."""
    max_rows = int(state.config["knit_parameters"]["bitmap_rows"])
    state.scanner_pattern_rows = 99
    state.scanner_pattern_cols = 99
    assert state._scanner_pattern_size() == (max_rows, int(state.SCANNER_PATTERN_MAX_COLS))
    state.scanner_pattern_rows = 0
    state.scanner_pattern_cols = 1
    assert state._scanner_pattern_size() == (2, 2)


# ---------------------------------------------------------------------------
# The fabric follows, and the pattern still lands on it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows,cols", [(3, 2), (5, 3), (2, 4)])
def test_the_fabric_follows_the_pattern_size(state, rows, cols):
    state.scanner_pattern_rows = rows
    state.scanner_pattern_cols = cols
    params = state._scanner_template_params()
    bitmap = state._scanner_random_bitmap(0)
    heights = state._scanner_loop_heights_for_bitmap(bitmap)
    fabric = build_parametric_control_rows(
        params, np.ones_like(bitmap), state._pidx, state._lh_idx,
        state.samples_per_loop, loop_heights=heights,
    )
    assert len(fabric) == rows


@pytest.mark.parametrize("rows,cols", [(3, 2), (5, 3), (2, 4)])
def test_every_stitch_still_lands_on_its_height(state, rows, cols):
    """The check that proved the pattern was applied correctly, at any size."""
    state.scanner_pattern_rows = rows
    state.scanner_pattern_cols = cols
    params = state._scanner_template_params()
    stitch_width = float(params[state._pidx["stitch_width"]])
    dy = float(params[state._pidx["dy"]])
    row_heights = [float(params[i]) for i in state._lh_idx]
    scale = float(state.SCANNER_INACTIVE_LOOP_HEIGHT_SCALE)

    for cell in range(6):
        bitmap = state._scanner_random_bitmap(cell)
        heights = state._scanner_loop_heights_for_bitmap(bitmap)
        fabric = build_parametric_control_rows(
            params, np.ones_like(bitmap), state._pidx, state._lh_idx,
            state.samples_per_loop, loop_heights=heights,
        )
        for r in range(rows):
            row = np.asarray(fabric[r])
            base = r * dy
            for c in range(cols):
                inside = (row[:, 0] >= c * stitch_width - 1e-6) & (row[:, 0] < (c + 1) * stitch_width - 1e-6)
                if not inside.any():
                    continue
                crest = float(row[inside, 1].max() - base)
                natural = row_heights[min(r, len(row_heights) - 1)]
                expected = natural * (1.0 if bitmap[r, c] > 0.5 else scale) * PEAK
                assert crest == pytest.approx(expected, abs=0.02), (
                    f"cell {cell} R{r + 1} C{c + 1} at pattern {rows}x{cols}"
                )


# ---------------------------------------------------------------------------
# The template's tuned rows only fit its own size
# ---------------------------------------------------------------------------

def test_the_tuned_fabric_is_used_only_at_the_template_size(state):
    """Otherwise the fallback to a parametric build must be an explicit refusal."""
    template_rows, template_cols = _template_shape(state)
    state.params = state._scanner_template_params()

    state.scanner_pattern_rows = template_rows
    state.scanner_pattern_cols = template_cols
    assert state.apply_scanner_template_base() is True

    for rows, cols in ((template_rows - 1, template_cols), (template_rows, template_cols + 1)):
        state.scanner_pattern_rows = rows
        state.scanner_pattern_cols = cols
        assert state.apply_scanner_template_base() is False, (
            f"the template's {template_rows}x{template_cols} rows cannot describe a {rows}x{cols} fabric"
        )


# ---------------------------------------------------------------------------
# Two sizes are two different scans
# ---------------------------------------------------------------------------

def test_pattern_size_changes_the_scan_configuration(state):
    """So duplicate detection treats a resized pattern as new data, not a repeat."""
    from scanner_storage import ScannerStorage

    snapshot = ScannerStorage.scanner_state_snapshot(ScannerStorage.__new__(ScannerStorage), state)
    before = scan_config_signature(snapshot)
    state.scanner_pattern_rows = 3
    after = scan_config_signature(
        ScannerStorage.scanner_state_snapshot(ScannerStorage.__new__(ScannerStorage), state)
    )
    assert before != after
