"""Each swatch cell in the shared 3D world owns its own independent pattern state."""

import numpy as np
import pytest

from app_state import AppState
from rendering import Camera


class StubRenderer:
    mesh_pick_data = None

    def __init__(self, state=None):
        self.state = state
        self.last_meta = []
        self.mesh_meta = []
        self.renderer_seen_during_upload = []
        self.vp_w = 320
        self.vp_h = 240

    def prepare_meshes(self, _verts, _faces):
        return None

    def set_meshes(self, verts, _faces, **kwargs):
        self.last_meta = kwargs.get("meta", []) or []
        self.mesh_meta = [dict(item) for item in self.last_meta]
        self.mesh_pick_data = [
            (np.asarray(item[0], dtype=np.float32), int(self.last_meta[idx].get("row", idx)))
            for idx, item in enumerate(verts)
        ]
        if self.state is not None:
            self.renderer_seen_during_upload.append(self.state.renderer)

    def resize(self, width, height):
        self.vp_w = int(width)
        self.vp_h = int(height)

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


def _change_cell(state, row, col):
    state.bitmap[row, col] = 0.0
    state.on_bitmap_change()
    return np.asarray(state.bitmap, dtype=np.float32).copy()


def test_swatch_cells_preserve_independent_patterns_across_edit_and_scan(state):
    import gui

    state.select_swatch(0, 0)
    pattern_00 = _change_cell(state, 1, 0)
    state.ctrl_rows[1][0, 2] += 0.25
    state._rebuild_spline_points()
    state.rebuild_spline_mesh()
    rows_00 = [row.copy() for row in state.ctrl_rows]

    state.select_swatch(0, 1)
    pattern_01 = _change_cell(state, 1, 1)
    state.ctrl_rows[1][0, 2] -= 0.35
    state._rebuild_spline_points()
    state.rebuild_spline_mesh()
    rows_01 = [row.copy() for row in state.ctrl_rows]

    gui._set_app_mode(state, "scan")
    assert state.scanner_preview_grid_enabled is True
    assert any("scanner_cell" in item for item in state.renderer.last_meta)

    gui._set_app_mode(state, "edit")
    assert state.scanner_preview_grid_enabled is False
    np.testing.assert_array_equal(state.bitmap, pattern_01)
    np.testing.assert_allclose(state.ctrl_rows[1], rows_01[1])

    state.select_swatch(0, 0)
    np.testing.assert_array_equal(state.bitmap, pattern_00)
    np.testing.assert_allclose(state.ctrl_rows[1], rows_00[1])


def test_private_scan_render_never_owns_the_main_renderer(state):
    from scanner_core import _scan_measure_pattern_frame

    main_renderer = state.renderer
    private_renderer = StubRenderer(state)

    _scan_measure_pattern_frame(state, private_renderer, state.bitmap.shape)

    assert state.renderer is main_renderer
    assert private_renderer.renderer_seen_during_upload
    assert all(renderer is main_renderer for renderer in private_renderer.renderer_seen_during_upload)


def test_late_scan_cleanup_cannot_restore_over_mode_switch(state):
    import gui
    from scanner_core import _scan_batch

    gui._set_app_mode(state, "scan")
    batch = _scan_batch(state)
    batch.__enter__()
    gui._set_app_mode(state, "edit")
    expected_bitmap = np.asarray(state.bitmap, dtype=np.float32).copy()
    expected_rows = [np.asarray(row, dtype=np.float32).copy() for row in state.ctrl_rows]
    batch.__exit__(None, None, None)

    np.testing.assert_array_equal(state.bitmap, expected_bitmap)
    for actual, expected in zip(state.ctrl_rows, expected_rows):
        np.testing.assert_allclose(actual, expected)
    assert state.app_mode == "edit"
