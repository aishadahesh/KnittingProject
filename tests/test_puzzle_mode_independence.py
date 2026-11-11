"""Each swatch cell in the shared 3D world owns its own independent pattern state."""

import numpy as np
import pytest

from app_state import AppState
from rendering import Camera
from scanner_core import _scanner_pattern_repeats, ensure_swatch_unit_tile_images


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
            if idx >= len(self.last_meta) or not bool(self.last_meta[idx].get("neighbor_swatch", False))
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

    state.save_active_swatch()
    state.app_mode = "scan"
    state.scanner_preview_grid_enabled = True
    state.rebuild_spline_mesh(preserve_model_placement=False)
    assert state.scanner_preview_grid_enabled is True
    assert any("scanner_cell" in item for item in state.renderer.last_meta)

    state.app_mode = "edit"
    state.scanner_preview_grid_enabled = False
    state.rebuild_spline_mesh(preserve_model_placement=False)
    assert state.scanner_preview_grid_enabled is False
    np.testing.assert_array_equal(state.bitmap, pattern_01)
    np.testing.assert_allclose(state.ctrl_rows[1], rows_01[1])

    state.select_swatch(0, 0)
    np.testing.assert_array_equal(state.bitmap, pattern_00)
    np.testing.assert_allclose(state.ctrl_rows[1], rows_00[1])


def test_in_place_swatch_slot_offsets_and_ctrl_pt_editing(state):
    state.scanner_rows = 2
    state.scanner_cols = 2
    state.select_swatch(1, 1)
    epoch_before = state.mesh_epoch
    state.rebuild_spline_mesh()
    assert state.mesh_epoch == epoch_before + 1
    assert state.last_prepared_meshes is not None

    slot_11 = state.swatch_slot_offset(1, 1)
    assert np.linalg.norm(slot_11[:2]) > 1.0
    pts_before = state.flat_pts.copy()
    delta = np.array([0.4, -0.3, 0.2], dtype=np.float32)
    state.move_ctrl_pt(1, pts_before[1] + delta)
    np.testing.assert_allclose(state.flat_pts[1], pts_before[1] + delta, atol=1e-5)
    np.testing.assert_allclose(state.flat_pts[0], pts_before[0], atol=1e-5)
    np.testing.assert_allclose(state.flat_pts[2:], pts_before[2:], atol=1e-5)



def test_swatch_cells_preserve_session_splat_and_fit_state_and_neighbor_swatches(state):
    state.scanner_rows = 2
    state.scanner_cols = 2
    state.select_swatch(0, 0)
    state.swatch_session_id = "20260417_120000"
    state.swatch_splat_path = "robot_scans/20260417_120000/splat/splat_clean.ply"
    state.swatch_fitted_model = {"fabric": "knit", "parameters": {"stitch_width": 1.1}, "bitmap": state.bitmap.copy()}
    state.save_active_swatch()

    state.select_swatch(1, 1)
    assert state.swatch_session_id == ""
    assert state.swatch_splat_path == ""
    assert state.swatch_fitted_model is None
    state.swatch_session_id = "20260417_130000"
    state.save_active_swatch()

    state.select_swatch(0, 0)
    assert state.swatch_session_id == "20260417_120000"
    assert state.swatch_splat_path == "robot_scans/20260417_120000/splat/splat_clean.ply"
    assert state.swatch_fitted_model["parameters"]["stitch_width"] == 1.1

    active_pick_count = len(state.renderer.mesh_pick_data)
    state.world_show_neighbor_swatches = True
    state.rebuild_spline_mesh(preserve_model_placement=True)
    assert not any(item.get("neighbor_swatch") for item in state.renderer.last_meta)
    assert len(state.renderer.mesh_pick_data) == active_pick_count


def test_swatch_unit_tile_images_and_3x3_unit_cell_builder(state):
    state.scanner_rows = 2
    state.scanner_cols = 2
    state.select_swatch(0, 0)
    assert _scanner_pattern_repeats(state) == (24, 24)

    tiles_1, epoch_1 = ensure_swatch_unit_tile_images(state)
    assert len(tiles_1) == 4
    assert epoch_1 >= 1
    for tile in tiles_1:
        assert isinstance(tile, np.ndarray)
        assert tile.dtype == np.uint8
        assert tile.ndim == 3 and tile.shape[2] == 3
        assert float(tile.std()) > 5.0

    tiles_2, epoch_2 = ensure_swatch_unit_tile_images(state)
    assert epoch_2 == epoch_1
    assert tiles_2[0] is tiles_1[0]

    state.bitmap[1, 0] = 0.0
    state.on_bitmap_change()
    tiles_3, epoch_3 = ensure_swatch_unit_tile_images(state)
    assert epoch_3 == epoch_1 + 1
    assert not np.array_equal(tiles_3[0], tiles_1[0])
    assert tiles_3[3] is tiles_1[3]

    prepared, colors, center_xy, xp, yp = state.build_swatch_unit_cell_data(1, 1)
    assert len(prepared) == 9 * len(state.ctrl_rows)
    assert len(colors) == len(prepared)
    assert len(center_xy) == 2
    assert xp > 0.5 and yp > 0.5

