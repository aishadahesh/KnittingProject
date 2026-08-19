"""Scanning the same configuration twice should be a deliberate choice.

A scan takes minutes and writes hundreds of images. The user's own database had
2492 captures across only twenty distinct configurations, one of them scanned
twenty-four separate times -- so repeating a scan by accident was the normal
case, not an edge one.

The check hinges on what counts as "the same scan". These tests pin both halves
of that: every setting that changes a pixel or a measured colour must make the
configuration different, and every setting that only changes timing or which
widget is selected must not. Getting the first half wrong silently loses data
the user wanted; getting the second wrong makes the prompt appear constantly and
trains them to click through it.

No GL needed.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from app_state import AppState  # noqa: E402
from rendering import Camera  # noqa: E402
from scanner_storage import (  # noqa: E402
    SCAN_CONFIG_KEYS,
    ScannerStorage,
    scan_config_signature,
)

# Changing any of these changes the images, or the RGB measured from them.
OUTPUT_AFFECTING = [
    ("scanner_rows", 7),
    ("scanner_cols", 9),
    ("scanner_angles", 8),
    ("scanner_pattern_density", 0.9),
    ("scanner_random_seed", 4242),
    ("scanner_pattern_repeat_rows", 5),
    ("scanner_repeat_spacing_x", 1.3),
    ("scanner_color_variants", [[0.1, 0.2, 0.3]]),
    ("scanner_lighting_enabled", False),
    ("scanner_light_azimuth", -20.0),
    ("scanner_light_sun_intensity", 0.9),
    ("scanner_capture_mode", "focused"),
    ("scanner_camera_zoom", 2.0),
    ("scanner_image_every", "station"),
    ("scanner_execution_mode", "robot"),
]

# Changing any of these changes nothing about the resulting images.
OUTPUT_NEUTRAL = [
    ("scanner_speed", 4.0),
    ("scanner_dwell", 1.0),
    ("scanner_add_camera", False),
    ("scanner_selected_cell", [2.0, 2.0]),
    ("scanner_single_row", 3),
    ("scanner_robot_ip", "10.0.0.9"),
    ("ui_theme", "light"),
]


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
    return app_state


@pytest.fixture
def storage(tmp_path, state):
    store = ScannerStorage(tmp_path)
    object.__setattr__(state, "scanner_storage", store)
    return store


def _record_run(storage, state, run="run_A", count=3):
    for index in range(count):
        storage.record_capture(state, {
            "path": f"scanner_images/{run}/img_{index}.png",
            "row": 0, "col": 0, "station": 0, "target_index": index,
            "angle": "angle 0", "rgb": [1.0, 2.0, 3.0],
        })


# ---------------------------------------------------------------------------
# Finding a previous run
# ---------------------------------------------------------------------------

def test_nothing_is_a_duplicate_in_an_empty_database(state, storage):
    import gui

    assert gui._scan_duplicate_runs(state) is None


def test_a_completed_run_is_found_again(state, storage):
    import gui

    _record_run(storage, state, run="run_A", count=3)
    found = gui._scan_duplicate_runs(state)
    assert found is not None
    assert [(r["scan_run"], r["capture_count"]) for r in found] == [("run_A", 3)]


def test_multiple_runs_of_one_configuration_are_reported_separately(state, storage):
    import gui

    _record_run(storage, state, run="run_A", count=2)
    _record_run(storage, state, run="run_B", count=5)
    found = gui._scan_duplicate_runs(state)
    assert {r["scan_run"] for r in found} == {"run_A", "run_B"}
    assert sum(r["capture_count"] for r in found) == 7


@pytest.mark.parametrize("key,value", OUTPUT_AFFECTING, ids=[k for k, _ in OUTPUT_AFFECTING])
def test_changing_what_the_images_look_like_is_not_a_duplicate(state, storage, key, value):
    import gui

    _record_run(storage, state)
    assert gui._scan_duplicate_runs(state), "the run just recorded should match itself"
    state[key] = value
    assert gui._scan_duplicate_runs(state) is None, (
        f"{key} changes the scanned images, so it must not count as the same scan"
    )


@pytest.mark.parametrize("key,value", OUTPUT_NEUTRAL, ids=[k for k, _ in OUTPUT_NEUTRAL])
def test_changing_something_cosmetic_is_still_a_duplicate(state, storage, key, value):
    import gui

    _record_run(storage, state)
    state[key] = value
    assert gui._scan_duplicate_runs(state), (
        f"{key} does not change a single pixel, so the scan is still the same one"
    )


def test_a_failed_check_never_blocks_scanning(state, monkeypatch):
    """The check exists to save a redundant run, not to be able to stop a wanted one."""
    import gui

    class Exploding:
        def scan_config_signature_for_state(self, _state):
            raise RuntimeError("database unavailable")

    object.__setattr__(state, "scanner_storage", Exploding())
    assert gui._scan_duplicate_runs(state) is None


# ---------------------------------------------------------------------------
# The signature itself
# ---------------------------------------------------------------------------

def test_signature_is_stable_across_float_noise():
    base = {key: 1 for key in SCAN_CONFIG_KEYS}
    base["scanner_pattern_density"] = 0.62
    noisy = dict(base, scanner_pattern_density=0.62 + 1e-9)
    assert scan_config_signature(base) == scan_config_signature(noisy)


def test_signature_ignores_keys_outside_the_configuration():
    base = {key: 1 for key in SCAN_CONFIG_KEYS}
    assert scan_config_signature(base) == scan_config_signature(dict(base, something_else=99))


def test_signature_is_order_independent():
    forward = {key: index for index, key in enumerate(SCAN_CONFIG_KEYS)}
    backward = dict(reversed(list(forward.items())))
    assert scan_config_signature(forward) == scan_config_signature(backward)


# ---------------------------------------------------------------------------
# Existing databases
# ---------------------------------------------------------------------------

def test_scans_recorded_before_this_feature_are_still_matched(tmp_path, state):
    """A database full of old runs must be usable by the check immediately.

    Every capture already stored the settings it was taken under, so the digest
    can be derived for them rather than the check only working for runs made
    from here on.
    """
    db_dir = tmp_path / "scanner_data"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "scanner.db")
    conn.executescript(
        """
        CREATE TABLE captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_signature TEXT, image_path TEXT NOT NULL,
            row_index INTEGER NOT NULL, col_index INTEGER NOT NULL,
            station_index INTEGER NOT NULL, target_index INTEGER NOT NULL,
            angle TEXT NOT NULL, capture_mode TEXT NOT NULL,
            capture_settings_json TEXT NOT NULL, rgb_json TEXT, created_at TEXT NOT NULL
        );
        """
    )
    settings = ScannerStorage.scanner_state_snapshot(ScannerStorage.__new__(ScannerStorage), state)
    conn.execute(
        "INSERT INTO captures VALUES (1,'old','scanner_images/run_old/a.png',0,0,0,0,"
        "'angle 0','natural',?, '[1,2,3]','2026-01-01')",
        (json.dumps(settings),),
    )
    conn.commit()
    conn.close()

    storage = ScannerStorage(tmp_path)
    object.__setattr__(state, "scanner_storage", storage)

    import gui

    found = gui._scan_duplicate_runs(state)
    assert found is not None, "an existing run must be recognised after migration"
    assert found[0]["scan_run"] == "run_old"


# ---------------------------------------------------------------------------
# Rescan must not overwrite the run it matched
# ---------------------------------------------------------------------------

def test_a_rescan_gets_its_own_run_folder(tmp_path):
    from embedded_scanner import _unique_scan_run_id

    first = _unique_scan_run_id(tmp_path)
    (tmp_path / first).mkdir(parents=True)
    second = _unique_scan_run_id(tmp_path)

    assert second != first, "a rescan started in the same second must not reuse the folder"
    assert not (tmp_path / second).exists()
