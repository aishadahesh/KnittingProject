"""Scanner logic, with no UI attached.

Scan Mode and Capture Mode share the fabric palette and per-cell colours, the
pattern grid geometry, and the scanner database state. None of it draws anything.

Nothing in this module may import imgui; a test enforces it.
"""

import json
import logging
import time

import numpy as np

from scan_geometry import clamp_capture_size


def _default_scanner_palette_from_model(state):
    if bool(state.use_row_colors):
        source = state.row_colors
    else:
        source = [state.single_model_color]
    count = max(1, int(state.bitmap_size[0]))
    palette = []
    for i in range(count):
        color = source[i % len(source)]
        palette.append([float(color[0]), float(color[1]), float(color[2]), 1.0])
    return palette


def _as_rgba(color, fallback):
    if isinstance(color, np.ndarray):
        color = color.tolist()
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        return list(fallback)
    return [
        float(color[0]),
        float(color[1]),
        float(color[2]),
        float(color[3]) if len(color) > 3 else 1.0,
    ]


def _ensure_scanner_shared_colors(state, count=None):
    fallback_palette = _default_scanner_palette_from_model(state)
    raw = state.get('scanner_color_variants', [])
    if count is None:
        count = len(raw) if isinstance(raw, list) and raw else len(fallback_palette)
    count = int(np.clip(int(count), 1, 12))

    colors = []
    for i in range(count):
        fallback = fallback_palette[i % len(fallback_palette)]
        if isinstance(raw, list) and i < len(raw):
            colors.append(_as_rgba(raw[i], fallback))
        else:
            colors.append(list(fallback))
    state.scanner_color_variants = colors
    return colors


def _scanner_base_palette(state):
    return [list(color) for color in _ensure_scanner_shared_colors(state)]


def _scanner_shared_cell_color_sets(state):
    rows = max(1, int(state.scanner_rows))
    cols = max(1, int(state.scanner_cols))
    base = _scanner_base_palette(state)
    return [[list(color) for color in base] for _ in range(rows * cols)]


def _scanner_estimated_cell_colors(state):
    """Estimate perceived color per Scan Mode sample from bitmap visibility.

    Active bitmap cells are treated as visible yarn. Inactive cells are treated
    as hidden/behind yarn and receive a small weight so the estimate remains
    stable without pretending hidden stitches dominate the final color.
    """
    rows = max(1, int(state.scanner_rows))
    cols = max(1, int(state.scanner_cols))
    palette = np.asarray([color[:3] for color in _scanner_base_palette(state)], dtype=np.float32)
    if palette.size == 0:
        palette = np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32)
    hidden_weight = 0.10
    raw_overrides = state.get('scanner_estimated_color_overrides', [])
    estimates = []
    for cell_index in range(rows * cols):
        try:
            bitmap = np.asarray(state._scanner_random_bitmap(cell_index), dtype=np.float32)
        except Exception:
            pattern_rows, pattern_cols = _scanner_pattern_dimensions(state)
            bitmap = np.ones((pattern_rows, pattern_cols), dtype=np.float32)
        if bitmap.ndim != 2 or bitmap.size == 0:
            bitmap = np.ones((1, 1), dtype=np.float32)
        row_colors = palette[np.arange(bitmap.shape[0]) % len(palette)]
        weights = np.where(bitmap > 0.5, 1.0, hidden_weight).astype(np.float32)
        weighted = weights[:, :, None] * row_colors[:, None, :]
        total_weight = max(float(weights.sum()), 1e-6)
        rgb = np.clip(weighted.sum(axis=(0, 1)) / total_weight, 0.0, 1.0)
        overridden = False
        if isinstance(raw_overrides, list) and cell_index < len(raw_overrides):
            override = raw_overrides[cell_index]
            if isinstance(override, (list, tuple, np.ndarray)) and len(override) >= 3:
                try:
                    rgb = np.clip(np.asarray(override[:3], dtype=np.float32), 0.0, 1.0)
                    overridden = True
                except Exception as exc:
                    logging.warning("scanner_core._scanner_estimated_cell_colors: %s", exc)
        active_ratio = float(np.mean(bitmap > 0.5))
        estimates.append({
            "row": int(cell_index // cols),
            "col": int(cell_index % cols),
            "rgb": [float(v * 255.0) for v in rgb],
            "active_ratio": active_ratio,
            "overridden": overridden,
            "bitmap": bitmap.astype(int).tolist(),
        })
    return estimates


def _scanner_measured_cell_colors(state):
    ctl = state.get('ur5_controller')
    result = getattr(ctl, "analysis", None) if ctl is not None else None
    if not result or not result.get("cells"):
        return None
    rows = max(1, int(state.scanner_rows))
    cols = max(1, int(state.scanner_cols))
    colors = [[42.0, 42.0, 42.0] for _ in range(rows * cols)]
    for cell in result.get("cells", []):
        row = int(cell.get("row", 0))
        col = int(cell.get("col", 0))
        if 0 <= row < rows and 0 <= col < cols:
            colors[row * cols + col] = [float(v) for v in cell.get("overall_rgb", [42.0, 42.0, 42.0])[:3]]
    return colors


def _scanner_display_batch_colors(state):
    measured = _scanner_measured_cell_colors(state)
    if measured is not None:
        return measured, "actual"
    estimates = _scanner_estimated_cell_colors(state)
    colors = [[float(v) for v in item.get("rgb", [42.0, 42.0, 42.0])[:3]] for item in estimates]
    return colors, "estimated"


def _scanner_storage(state):
    storage = getattr(state, "scanner_storage", None)
    if storage is not None:
        return storage
    try:
        from scanner_storage import ScannerStorage
        storage = ScannerStorage(state.project_root)
        object.__setattr__(state, "scanner_storage", storage)
        return storage
    except Exception:
        return None


def _scanner_pattern_database_payload(state):
    rows = max(1, int(state.scanner_rows))
    cols = max(1, int(state.scanner_cols))
    patterns = []
    for cell_index in range(rows * cols):
        try:
            bitmap = np.asarray(state._scanner_random_bitmap(cell_index), dtype=np.float32)
        except Exception:
            pattern_rows, pattern_cols = _scanner_pattern_dimensions(state)
            bitmap = np.ones((pattern_rows, pattern_cols), dtype=np.float32)
        patterns.append({
            "row": int(cell_index // cols),
            "col": int(cell_index % cols),
            "bitmap": bitmap.astype(int).tolist(),
        })
    return patterns


def _persist_scanner_state(state, patterns=None, estimates=None):
    storage = _scanner_storage(state)
    if storage is None:
        return None
    try:
        if patterns is not None or estimates is not None:
            return storage.save_pattern_set(
                state,
                patterns if patterns is not None else _scanner_pattern_database_payload(state),
                estimates if estimates is not None else _scanner_estimated_cell_colors(state),
            )
        storage.save_scanner_state(state)
    except Exception as exc:
        state.scanner_status = f"Could not save scanner database state: {exc}"
    return None


def _scanner_pattern_dimensions(state):
    """The scan pattern grid, from the one place that decides it."""
    if hasattr(state, '_scanner_pattern_size'):
        return state._scanner_pattern_size()
    return (
        max(2, int(state.get('scanner_pattern_rows', max(2, int(state.bitmap_size[0]))))),
        max(2, int(state.get('scanner_pattern_cols', max(2, int(state.bitmap_size[1]))))),
    )


def _scanner_capture_image_size(state, key='scanner_capture_width'):
    return clamp_capture_size(state.get(key, 1024))


def _scanner_lighting_settings(state):
    return {
        "enabled": 1.0 if bool(state.get('scanner_lighting_enabled', True)) else 0.0,
        "azimuth": float(state.get('scanner_light_azimuth', -35.0)),
        "elevation": float(state.get('scanner_light_elevation', 48.0)),
        "sun_intensity": float(state.get('scanner_light_sun_intensity', 0.68)),
        "shadow": float(state.get('scanner_light_shadow', 0.20)),
        "sheen": float(state.get('scanner_light_sheen', 0.025)),
    }


def _scanner_material_uniforms(state):
    """Base material uniforms with Scan Mode's lighting settings applied -- the
    same real-3D-shader lighting used by the live viewport, reused so automated
    scan-pattern captures are lit consistently with what the user configured."""
    material_uniforms = dict(state.get_material_uniforms())
    lighting = _scanner_lighting_settings(state)
    if float(lighting.get("enabled", 1.0)) < 0.5:
        material_uniforms.update({
            "light_color": (1.0, 1.0, 1.0),
            "light_dir": (0.0, 0.0, 1.0),
            "light_intensity": 1.0,
            "ao_strength": 0.0,
            "texture_gloss_strength": 0.0,
            "texture_center_shadow": 0.0,
            "texture_groove_darkness": 0.0,
        })
    else:
        sun = float(lighting.get("sun_intensity", 0.68))
        shadow = float(lighting.get("shadow", 0.20))
        sheen = float(lighting.get("sheen", 0.025))
        az = np.deg2rad(float(lighting.get("azimuth", -35.0)))
        el = np.deg2rad(float(lighting.get("elevation", 48.0)))
        light_dir = (
            float(np.cos(az) * np.cos(el)),
            float(np.sin(az) * np.cos(el)),
            float(np.sin(el)),
        )
        material_uniforms.update({
            "light_color": (1.0, 0.96, 0.88),
            "light_dir": light_dir,
            "light_intensity": float(np.clip(0.65 + sun * 0.55, 0.30, 1.35)),
            "ao_strength": float(np.clip(0.22 + shadow * 1.35, 0.0, 0.85)),
            "ao_radius": float(np.clip(0.08 + shadow * 0.35, 0.02, 0.35)),
            "texture_gloss_strength": float(np.clip(0.08 + sheen * 3.8, 0.0, 0.55)),
            "texture_center_shadow": float(np.clip(0.12 + shadow * 0.55, 0.0, 0.55)),
            "texture_groove_darkness": float(np.clip(0.06 + shadow * 0.45, 0.0, 0.45)),
        })
    return material_uniforms


def _clamp_scanner_selected_cell(state):
    rows = max(1, int(state.scanner_rows))
    cols = max(1, int(state.scanner_cols))
    selected = np.asarray(state.get('scanner_selected_cell', [0, 0]), dtype=np.int32).reshape(-1)
    selected_r = int(selected[0]) if selected.size > 0 else 0
    selected_c = int(selected[1]) if selected.size > 1 else 0
    state.scanner_selected_cell = [
        int(np.clip(selected_r, 0, rows - 1)),
        int(np.clip(selected_c, 0, cols - 1)),
    ]
    return tuple(state.scanner_selected_cell)


def _set_scanner_estimated_batch_color(state, row: int, col: int, rgb) -> None:
    rows = max(1, int(state.scanner_rows))
    cols = max(1, int(state.scanner_cols))
    index = int(np.clip(row, 0, rows - 1)) * cols + int(np.clip(col, 0, cols - 1))
    overrides = state.get('scanner_estimated_color_overrides', [])
    if not isinstance(overrides, list):
        overrides = []
    while len(overrides) < rows * cols:
        overrides.append(None)
    rgba = _as_rgba(rgb, [0.5, 0.5, 0.5, 1.0])
    overrides[index] = [float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0]
    state.scanner_estimated_color_overrides = overrides[:rows * cols]


def _maybe_persist_scanner_state(state, force=False):
    storage = _scanner_storage(state)
    if storage is None:
        return
    now = time.monotonic()
    if not force and now - float(state.__dict__.get("_scanner_db_last_save", 0.0)) < 2.0:
        return
    try:
        snapshot = storage.scanner_state_snapshot(state)
        signature = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        if force or signature != state.__dict__.get("_scanner_db_last_signature"):
            storage.save_scanner_state(state)
            object.__setattr__(state, "_scanner_db_last_signature", signature)
            object.__setattr__(state, "_scanner_db_last_save", now)
    except Exception as exc:
        logging.warning("scanner_core._maybe_persist_scanner_state: %s", exc)
