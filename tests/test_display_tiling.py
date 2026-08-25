"""The vertical tiling invariant: stacked copies must join at the row pitch.

A copy above is more rows of the same fabric, so the gap where one copy ends and
the next begins has to be exactly the gap between rows inside a copy. Anything
else shows up as the copies overlapping (gap too small) or the fabric splitting
into visible bands (gap too large).

The period was previously read off the `dy` parameter. That is correct only
while the control rows still match the parameter they were built from; once a
model is scaled or a point dragged, `dy` describes a fabric that is no longer
there, and the copies overlapped by nearly a whole row. These tests measure
geometry so the two cannot drift apart again.

No GL needed -- AppState builds fine against a stub renderer, so this runs
everywhere the panel harness skips.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from app_state import AppState  # noqa: E402
from knitting_core import build_parametric_control_rows, row_base_pitch  # noqa: E402
from rendering import Camera  # noqa: E402


Y_SCALE = 1.86268  # the scale the real saved model carries, versus its `dy`


class StubRenderer:
    """Accepts every renderer call; these tests are about geometry, not GL."""

    mesh_pick_data = None

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture
def state(tmp_path):
    app_state = AppState(Camera(), StubRenderer())
    # Redirected before anything can autosave, so a test run can never write
    # over the real config/params.json.
    app_state.save_path = str(tmp_path / "params.json")
    app_state.load_path = str(tmp_path / "params.json")
    app_state.autosave_enabled = False
    app_state.rebuild_spline_from_params()
    return app_state


def _bases(state):
    """Each row's base Y, sorted. The lattice the fabric actually repeats on."""
    return sorted(float(row[:, 1].min()) for row in state.ctrl_rows if len(row))


def _pitch(bases):
    return float(np.median(np.abs(np.diff(bases))))


def _join_gap(state, y_period):
    """The gap between one copy's last row and the next copy's first row."""
    bases = _bases(state)
    return (bases[0] + y_period) - bases[-1]


def _scale_y(state, factor):
    """Scale the model vertically, leaving `params` untouched.

    This is the state the bug needed: real geometry on one pitch, `dy` still
    claiming another. Reproduced here rather than loaded from a fixture file so
    the test does not depend on whatever model happens to be saved.
    """
    state.ctrl_rows = [
        np.column_stack((row[:, 0], row[:, 1] * factor, row[:, 2])).astype(np.float32)
        for row in state.ctrl_rows
    ]
    state._rebuild_spline_points()
    state.sync_period_offset_y_to_row_count()


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------

def test_parametric_model_tiles_seamlessly(state):
    bases = _bases(state)
    pitch = _pitch(bases)
    y_period = state.display_copy_periods()[1]

    # Pins the identity the whole fix rests on: a row's base is row_idx * dy,
    # because the loop-height term is zero at t = 0. If the row formula in
    # knitting_core changes, measuring the pitch from base Y stops being valid
    # and this is the test that should say so.
    assert pitch == pytest.approx(abs(float(state.params[state._pidx["dy"]])), rel=1e-6)

    assert y_period == pytest.approx(pitch * len(state.ctrl_rows), rel=1e-6)
    assert _join_gap(state, y_period) == pytest.approx(pitch, rel=1e-6)


def test_y_scaled_model_tiles_seamlessly(state):
    """The regression case: geometry scaled, `dy` left stale."""
    _scale_y(state, Y_SCALE)
    dy = abs(float(state.params[state._pidx["dy"]]))
    pitch = _pitch(_bases(state))
    y_period = state.display_copy_periods()[1]

    assert pitch == pytest.approx(dy * Y_SCALE, rel=1e-5)
    assert _join_gap(state, y_period) == pytest.approx(pitch, rel=1e-6)
    # The negative half matters as much as the positive: it is what stops the
    # old `dy * n_rows` formula ever passing here again.
    assert y_period != pytest.approx(dy * len(state.ctrl_rows), rel=1e-3)


def test_one_dragged_control_point_does_not_move_the_pitch(state):
    """Why the pitch is a median and not a fitted slope."""
    _scale_y(state, Y_SCALE)
    pitch_before = _pitch(_bases(state))

    state.ctrl_rows[1][2, 1] -= 0.4 * pitch_before
    state._rebuild_spline_points()

    y_period = state.display_copy_periods()[1]
    assert _join_gap(state, y_period) == pytest.approx(pitch_before, rel=1e-3)


@pytest.mark.parametrize("scale", [1.0, Y_SCALE])
def test_period_offset_y_matches_the_display_period(state, scale):
    """The simulation and the display must agree on how tall the fabric is.

    They disagreed by a factor of nearly two, which is how the overlap arose:
    two derivations of one quantity, only one of them measured.
    """
    if scale != 1.0:
        _scale_y(state, scale)
    state.sync_period_offset_y_to_row_count()

    offset = np.asarray(state.period_offset_y, dtype=float)
    assert offset[1] == pytest.approx(state.display_copy_periods()[1], rel=1e-6)
    assert offset[0] == 0.0 and offset[2] == 0.0


def test_one_row_model_falls_back_to_the_parametric_pitch(state):
    """A single row has no step to measure; `dy` is the right answer there."""
    state.ctrl_rows = state.ctrl_rows[:1]
    state._rebuild_spline_points()

    dy = abs(float(state.params[state._pidx["dy"]]))
    assert state.display_copy_periods()[1] == pytest.approx(dy, rel=1e-6)


# ---------------------------------------------------------------------------
# The measurement itself
# ---------------------------------------------------------------------------

def test_row_base_pitch_recovers_dy_from_a_parametric_build(state):
    rows = build_parametric_control_rows(
        state.params, state.bitmap, state._pidx, state._lh_idx, state.samples_per_loop
    )
    rows = rows[0] if isinstance(rows, tuple) else rows
    dy = abs(float(state.params[state._pidx["dy"]]))
    assert row_base_pitch(rows) == pytest.approx(dy, rel=1e-6)


def test_row_base_pitch_reports_nothing_to_measure():
    assert row_base_pitch([]) is None
    assert row_base_pitch([np.zeros((4, 3), dtype=np.float32)]) is None


def test_row_base_pitch_ignores_loop_height(state):
    """Loop height stretches a row upward; it must not move the lattice.

    Row centres and the mesh bounding box both shift with loop height, which is
    why neither can be used to measure the period.
    """
    rows = [np.asarray(row, dtype=np.float32).copy() for row in state.ctrl_rows]
    before = row_base_pitch(rows)
    # Push the tallest points of every row further up, leaving the bases alone.
    for row in rows:
        crest = row[:, 1] > row[:, 1].min() + 1e-6
        row[crest, 1] += 1.5
    assert row_base_pitch(rows) == pytest.approx(before, rel=1e-9)


# ---------------------------------------------------------------------------
# Scan and puzzle tiling agree with the display
# ---------------------------------------------------------------------------

def test_scan_template_reproduces_its_saved_fabric(state, tmp_path, monkeypatch):
    """Scan Mode must scan the tuned fabric, not a rebuild of its parameters.

    The template's parameters do not describe the rows saved beside them -- the
    real file has a 1.106 pitch with near-uniform loops, while its parameters
    say dy 0.594 with loop heights spanning 4.7:1. Rebuilt from those, rows
    tower many times their own pitch and pass through the rows above, which is
    what captures were showing.
    """
    import json

    import paths

    # A template whose saved rows sit on a different pitch from its `dy`,
    # which is the condition the real file is in.
    rows = build_parametric_control_rows(
        state.params, state.bitmap, state._pidx, state._lh_idx, state.samples_per_loop
    )
    rows = rows[0] if isinstance(rows, tuple) else rows
    scaled = [
        np.column_stack((r[:, 0], r[:, 1] * Y_SCALE, r[:, 2])).astype(np.float32)
        for r in rows
    ]
    saved_pitch = row_base_pitch(scaled)

    template_path = tmp_path / "initial_params.json"
    template_path.write_text(json.dumps({
        "params": {
            p["name"]: float(state.params[i])
            for i, p in enumerate(state.config["knit_parameters"]["parameters"])
        },
        "bitmap": np.asarray(state.bitmap, dtype=float).tolist(),
        "spline_control_rows": [r.tolist() for r in scaled],
    }), encoding="utf-8")
    monkeypatch.setattr(paths, "INITIAL_PARAMS_JSON", template_path)
    # _scanner_template memoises on the file's mtime; drop the cached entry.
    object.__setattr__(state, "_scanner_template_cache", None)

    # The saved rows are only usable at the size they describe, and the scan
    # pattern size is a setting now -- point it at this template's fabric, the
    # way the defaults point at the real one.
    state.scanner_pattern_rows, state.scanner_pattern_cols = (
        int(state.bitmap.shape[0]), int(state.bitmap.shape[1]),
    )

    state.params = state._scanner_template_params()
    assert state.apply_scanner_template_base() is True
    state.nudge_spline_from_params(rebuild_mesh=False)

    assert row_base_pitch(state.ctrl_rows) == pytest.approx(saved_pitch, rel=1e-5)
    # And emphatically not the pitch a parametric rebuild would have produced.
    dy = abs(float(state.params[state._pidx["dy"]]))
    assert row_base_pitch(state.ctrl_rows) != pytest.approx(dy, rel=1e-3)


def test_scan_template_falls_back_when_it_has_no_saved_rows(state, tmp_path, monkeypatch):
    """A template with only parameters keeps the previous behaviour."""
    import json

    import paths

    template_path = tmp_path / "initial_params.json"
    template_path.write_text(json.dumps({
        "params": {
            p["name"]: float(state.params[i])
            for i, p in enumerate(state.config["knit_parameters"]["parameters"])
        },
    }), encoding="utf-8")
    monkeypatch.setattr(paths, "INITIAL_PARAMS_JSON", template_path)
    object.__setattr__(state, "_scanner_template_cache", None)

    assert state.apply_scanner_template_base() is False


def test_puzzle_tiles_carry_no_depth_shift(state):
    """Y copies translate in Y alone, in every one of the three tiling sites.

    scanner_core used to push each Y copy back by a whole depth period while
    the rendered mesh did not, so the pixel vectors driving the seamless-tile
    glue described a fabric that was never drawn.
    """
    from scanner_core import _puzzle_tiled_control_points

    state.display_copies = np.array([1, 1], dtype=np.int32)
    tiled = _puzzle_tiled_control_points(state)
    assert tiled, "expected tiled control rows"

    y_period = state.display_copy_periods()[1]
    sources = {int(row_idx): row for row_idx, row in enumerate(state.ctrl_rows)}

    offsets = set()
    for row, row_idx in tiled:
        source = sources[int(row_idx)]
        assert np.allclose(row[:, 2], source[:, 2], atol=1e-6), "Y copies must not shift in Z"
        offsets.add(round(float(row[0, 1] - source[0, 1]), 4))

    expected = {round(k * y_period, 4) for k in (-1, 0, 1)}
    assert offsets == expected

class CaptureRenderer(StubRenderer):
    """A stub that also hands back a readable colour buffer, like the real one."""

    def __init__(self, vp_w, vp_h, rgba_bytes):
        self.vp_w = int(vp_w)
        self.vp_h = int(vp_h)
        self.color_tex = _ColorTex(rgba_bytes)


class _ColorTex:
    def __init__(self, rgba_bytes):
        self._rgba_bytes = rgba_bytes

    def read(self):
        return self._rgba_bytes


def _lattice_pixel(px, py, period_w, period_h):
    """A synthetic fabric: any pixel is fixed by its phase within one repeat."""
    u = np.asarray(px) % period_w
    v = np.asarray(py) % period_h
    return np.stack([
        (u * 7 + v * 3) % 256,
        (u * 3 + v * 11) % 256,
        (u * 13 + v * 5) % 256,
    ], axis=-1).astype(np.uint8)


def test_puzzle_glue_tile_is_exactly_one_repeat_period(state):
    """The glued tile must be one period tall, not one whole model tall.

    Cropping the model's full projected height instead -- every row of the
    central copy -- still produces a tile, but it is several periods plus the
    top and bottom loop overhang. Stamped edge to edge that repeats partial
    rows, which is the banding the glue exists to rule out.
    """
    from scanner_core import _puzzle_build_seamless_tile, _puzzle_period_pixel_vectors

    vp_w, vp_h = 960, 720
    state.camera.dist = 120.0
    probe = StubRenderer()
    probe.vp_w, probe.vp_h = vp_w, vp_h
    periods = _puzzle_period_pixel_vectors(state, probe)
    period_w = int(round(abs(float(periods["x_step"][0]))))
    period_h = int(round(abs(float(periods["y_step"][1]))))
    assert 4 < period_w < vp_w and 4 < period_h < vp_h

    # A viewport painted with a fabric that repeats on exactly that lattice.
    ys, xs = np.mgrid[0:vp_h, 0:vp_w]
    visible = _lattice_pixel(xs, ys, period_w, period_h)
    rgba = np.dstack([np.flipud(visible), np.full((vp_h, vp_w, 1), 255, np.uint8)])
    renderer = CaptureRenderer(vp_w, vp_h, rgba.tobytes())

    tile, glued, info = _puzzle_build_seamless_tile(state, renderer, 3, 4)

    assert (info["tile_w"], info["tile_h"]) == (period_w, period_h)
    assert tile.size == (period_w, period_h)

    # The glue continues the fabric: every pixel of the canvas is the pixel the
    # underlying lattice would have had, extended past the captured viewport.
    x0, y0 = info["tile_box"][0], info["tile_box"][1]
    out = np.asarray(glued, dtype=np.uint8)
    gys, gxs = np.mgrid[0:out.shape[0], 0:out.shape[1]]
    expected = _lattice_pixel(gxs + x0, gys + y0, period_w, period_h)
    assert np.array_equal(out, expected)
