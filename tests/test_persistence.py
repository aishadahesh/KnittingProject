import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app_state import AppState
from persistence import json_ready, write_json
from rendering import Camera
from scanner_storage import ScannerStorage


@pytest.fixture
def state(tmp_path):
    renderer = SimpleNamespace(
        prepare_meshes=lambda *args: [],
        set_meshes=lambda *args, **kwargs: None,
        set_ctrl_pts=lambda *args: None,
    )
    state = AppState(Camera(), renderer)
    state.autosave_enabled = False
    state.save_path = str(tmp_path / 'model.json')
    state.rebuild_spline_from_params()
    return state


def test_json_serialization_is_shared_and_does_not_mutate_input(tmp_path):
    value = {'array': np.array([1, 2]), 'number': np.float32(0.5), 'path': tmp_path}
    target = tmp_path / 'data.json'
    write_json(target, value)
    assert json.loads(target.read_text()) == json_ready(value)
    assert isinstance(value['array'], np.ndarray)


@pytest.mark.parametrize('operation', ['write_text', 'replace'])
def test_failed_write_preserves_previous_file_and_removes_temporary(tmp_path, monkeypatch, operation):
    target = tmp_path / 'data.json'
    write_json(target, {'version': 1})
    original = target.read_bytes()

    def fail(*args, **kwargs):
        raise OSError('disk unavailable')

    monkeypatch.setattr(Path, operation, fail)
    with pytest.raises(OSError, match='disk unavailable'):
        write_json(target, {'version': 2})
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


def test_invalid_json_value_cannot_replace_previous_file(tmp_path):
    target = tmp_path / 'data.json'
    write_json(target, {'value': 1})
    with pytest.raises(ValueError):
        write_json(target, {'value': np.nan})
    assert json.loads(target.read_text()) == {'value': 1}


@pytest.mark.parametrize('field,value', [
    ('bitmap', [1, 0]),
    ('spline_control_rows', [[[1, 2]]]),
    ('spline_radius_rows', [[-1]]),
    ('gui_state', {'model_scale': [1]}),
    ('gui_state', {'camera_target': [np.nan, 0, 0]}),
    ('format_version', 1),
])
def test_failed_load_keeps_model_camera_and_live_resources(state, field, value):
    state.save_params(state.save_path)
    source = Path(state.save_path)
    data = json.loads(source.read_text())
    data['params']['radius'] = 0.1
    data[field] = value
    source.write_text(json.dumps(data))
    before = json_ready(state.snapshot_state())
    renderer = state.renderer
    camera = state.camera

    assert not state.load_params(source)

    after = json_ready(state.snapshot_state())
    before.pop('status_msg')
    after.pop('status_msg')
    assert after == before
    assert state.renderer is renderer
    assert state.camera is camera
    assert state.status_msg.startswith('Load error:')


def test_load_rebuilds_dimensions_and_preserves_active_mode(state):
    state.save_params(state.save_path)
    source = Path(state.save_path)
    data = json.loads(source.read_text())
    data['gui_state']['bitmap_size'] = [99, 99]
    data['gui_state']['app_mode'] = 'database'
    source.write_text(json.dumps(data))

    assert state.load_params(source)
    np.testing.assert_array_equal(state.bitmap_size, state.bitmap.shape)
    assert state.app_mode == 'edit'
    assert state.load_params(source, restore_mode=True)
    assert state.app_mode == 'database'


@pytest.mark.parametrize('shape', [(7, 3), (2, 4)])
@pytest.mark.parametrize('saved_size', [None, [99, 99]])
def test_loading_a_different_bitmap_size_restores_all_row_state(state, shape, saved_size):
    state.on_bitmap_resize(*shape)
    state.save_params(state.save_path)
    source = Path(state.save_path)
    data = json.loads(source.read_text())
    data['gui_state']['row_visible'] = [False, True, False]
    if saved_size is not None:
        data['gui_state']['bitmap_size'] = saved_size
    source.write_text(json.dumps(data))
    state.on_bitmap_resize(3, 5)

    assert state.load_params(source), state.status_msg

    np.testing.assert_array_equal(state.bitmap_size, shape)
    assert len(state.ctrl_rows) == len(state.row_colors) == len(state.row_visible) == shape[0]
    assert state.loop_heights.shape == state.loop_height_overrides.shape == shape
    np.testing.assert_array_equal(state.row_visible[:min(3, shape[0])], [False, True, False][:shape[0]])
    assert state.row_visible[3:].all()
    for actual, saved in zip(state.ctrl_rows, data['spline_control_rows']):
        np.testing.assert_allclose(actual, saved, rtol=1e-7, atol=1e-7)


def test_reading_display_periods_does_not_replace_radius_state(state):
    radii = state.spline_radius_rows
    before = json_ready(state.snapshot_state())
    state.display_copy_periods()
    assert state.spline_radius_rows is radii
    assert json_ready(state.snapshot_state()) == before


def test_scanner_restore_does_not_apply_part_of_a_corrupt_snapshot(state, tmp_path):
    storage = ScannerStorage(tmp_path)
    previous = state.scanner_rows
    with storage._connect() as connection:
        connection.execute(
            'INSERT INTO scanner_state VALUES (1, ?, ?)',
            (json.dumps({'scanner_rows': 99, 'scanner_angles': ['invalid']}), 'now'),
        )
    assert not storage.restore_scanner_state(state)
    assert state.scanner_rows == previous
    assert 'Restore error:' in state.scanner_status


def test_scanner_restore_cannot_replace_resources_or_editor_geometry(state, tmp_path):
    storage = ScannerStorage(tmp_path)
    state.scanner_storage = storage
    bitmap = state.bitmap
    with storage._connect() as connection:
        connection.execute(
            'INSERT INTO scanner_state VALUES (1, ?, ?)',
            (json.dumps({'scanner_storage': None, 'bitmap': [], 'scanner_unknown': 1, 'scanner_rows': 2}), 'now'),
        )
    assert storage.restore_scanner_state(state)
    assert state.scanner_storage is storage
    assert state.bitmap is bitmap
    assert state.scanner_rows == 2
    assert 'scanner_unknown' not in state._data
