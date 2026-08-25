"""Puzzle Mode owns a model instead of borrowing the last visited mode's one."""

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


def test_puzzle_pattern_survives_edit_and_scan_without_leaking(state):
    import gui

    baseline = np.asarray(state.bitmap, dtype=np.float32).copy()
    puzzle_baseline_rows = [
        np.asarray(row, dtype=np.float32).copy()
        for row in state.puzzle_model_snapshot["ctrl_rows"]
    ]
    edit_pattern = _change_cell(state, 1, 0)
    state.ctrl_rows[1][0, 2] += 0.25
    state._rebuild_spline_points()
    state.rebuild_spline_mesh()
    edit_rows = [row.copy() for row in state.ctrl_rows]
    state.push_undo("Edit workspace")
    edit_undo_count = len(state.undo_stack)

    gui._set_app_mode(state, "puzzle")
    np.testing.assert_array_equal(state.bitmap, baseline)
    np.testing.assert_allclose(state.ctrl_rows[1], puzzle_baseline_rows[1])
    # An all-active Puzzle pattern has one height per row, not Scan's random
    # short/tall height per cell.
    assert all(np.allclose(row, row[0]) for row in np.asarray(state.loop_heights))
    assert not np.any(state.loop_height_overrides)
    assert state.undo_stack == []

    puzzle_pattern = _change_cell(state, 1, 1)
    state.ctrl_rows[1][0, 2] -= 0.35
    state._rebuild_spline_points()
    state.rebuild_spline_mesh()
    puzzle_rows = [row.copy() for row in state.ctrl_rows]
    state.model_t = np.asarray(state.model_t, dtype=np.float32) + np.array([2.5, -1.25, 0.4], dtype=np.float32)
    state.model_scale = np.array([1.3, 0.8, 1.1], dtype=np.float32)
    state.camera.az = 0.37
    state.camera.el = -0.21
    puzzle_model_t = state.model_t.copy()
    puzzle_model_scale = state.model_scale.copy()
    puzzle_camera = (state.camera.az, state.camera.el)
    state.push_undo("Puzzle workspace")
    puzzle_undo_count = len(state.undo_stack)

    gui._set_app_mode(state, "scan")
    np.testing.assert_array_equal(state.bitmap, edit_pattern)
    np.testing.assert_allclose(state.ctrl_rows[1], edit_rows[1])
    assert len(state.undo_stack) == edit_undo_count
    assert state.scanner_preview_grid_enabled is True
    assert any("scanner_cell" in item for item in state.renderer.last_meta)
    # A Scan-only placement change must not become Puzzle's placement.
    state.model_t = np.asarray(state.model_t, dtype=np.float32) + np.array([20.0, 10.0, 0.0], dtype=np.float32)

    gui._set_app_mode(state, "puzzle")
    np.testing.assert_array_equal(state.bitmap, puzzle_pattern)
    np.testing.assert_allclose(state.ctrl_rows[1], puzzle_rows[1])
    np.testing.assert_allclose(state.model_t, puzzle_model_t)
    np.testing.assert_allclose(state.model_scale, puzzle_model_scale)
    assert (state.camera.az, state.camera.el) == pytest.approx(puzzle_camera)
    assert len(state.undo_stack) == puzzle_undo_count
    assert state.scanner_preview_grid_enabled is False
    assert not any("scanner_cell" in item for item in state.renderer.last_meta)

    gui._set_app_mode(state, "edit")
    np.testing.assert_array_equal(state.bitmap, edit_pattern)
    np.testing.assert_allclose(state.ctrl_rows[1], edit_rows[1])
    assert len(state.undo_stack) == edit_undo_count


def test_puzzle_does_not_overwrite_the_main_autosave(state, monkeypatch):
    calls = []
    monkeypatch.setattr(state, "save_params", lambda *args, **kwargs: calls.append(args))
    state.autosave_enabled = True
    state.autosave_last_time = 0.0
    state.app_mode = "puzzle"

    state.maybe_autosave()

    assert calls == []


def test_reset_initial_in_puzzle_uses_the_clean_puzzle_baseline(state):
    import gui

    gui._set_app_mode(state, "puzzle")
    _change_cell(state, 1, 0)

    state.reset_to_initial()

    np.testing.assert_array_equal(state.bitmap, state.puzzle_initial_snapshot["bitmap"])
    assert all(np.allclose(row, row[0]) for row in np.asarray(state.loop_heights))
    assert not np.any(state.loop_height_overrides)
    assert state.app_mode == "puzzle"


def test_private_scan_render_never_owns_the_main_renderer(state):
    from scanner_core import _scan_measure_pattern_frame

    main_renderer = state.renderer
    private_renderer = StubRenderer(state)

    _scan_measure_pattern_frame(state, private_renderer, state.bitmap.shape)

    assert state.renderer is main_renderer
    assert private_renderer.renderer_seen_during_upload
    assert all(renderer is main_renderer for renderer in private_renderer.renderer_seen_during_upload)


def test_puzzle_uses_dedicated_gpu_scene(tmp_path):
    import gui

    main_renderer = StubRenderer()
    puzzle_renderer = StubRenderer()
    state = AppState(Camera(), main_renderer, puzzle_renderer=puzzle_renderer)
    main_renderer.state = state
    puzzle_renderer.state = state
    state.save_path = str(tmp_path / "params.json")
    state.load_path = str(tmp_path / "params.json")
    state.autosave_enabled = False
    state.rebuild_spline_from_params()
    state.capture_initial_state()

    gui._set_app_mode(state, "scan")
    assert state.active_scene_renderer() is main_renderer
    assert any("scanner_cell" in item for item in main_renderer.last_meta)

    gui._set_app_mode(state, "puzzle")
    assert state.active_scene_renderer() is puzzle_renderer
    assert puzzle_renderer.last_meta
    assert not any("scanner_cell" in item for item in puzzle_renderer.last_meta)
    # The main Scan buffers may still exist, but Puzzle never displays them.
    assert any("scanner_cell" in item for item in main_renderer.last_meta)


def test_late_scan_cleanup_cannot_restore_over_puzzle(state):
    import gui
    from scanner_core import _scan_batch

    puzzle_renderer = StubRenderer(state)
    state.puzzle_renderer = puzzle_renderer
    gui._set_app_mode(state, "scan")

    # Model the dangerous ordering directly: a Scan batch owns a temporary
    # snapshot, the user enters Puzzle, then the old Scan cleanup completes.
    batch = _scan_batch(state)
    batch.__enter__()
    gui._set_app_mode(state, "puzzle")
    expected_bitmap = np.asarray(state.bitmap, dtype=np.float32).copy()
    expected_rows = [np.asarray(row, dtype=np.float32).copy() for row in state.ctrl_rows]
    batch.__exit__(None, None, None)

    np.testing.assert_array_equal(state.bitmap, expected_bitmap)
    for actual, expected in zip(state.ctrl_rows, expected_rows):
        np.testing.assert_allclose(actual, expected)
    assert state.app_mode == "puzzle"
    assert not any("scanner_cell" in item for item in puzzle_renderer.last_meta)
