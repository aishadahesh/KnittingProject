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
        active = bitmap > 0.5
        row_weights = np.where(active, 1.0, hidden_weight).sum(axis=1)
        rgb = np.clip(row_weights @ row_colors / max(float(row_weights.sum()), 1e-6), 0.0, 1.0)
        overridden = False
        if isinstance(raw_overrides, list) and cell_index < len(raw_overrides):
            override = raw_overrides[cell_index]
            if isinstance(override, (list, tuple, np.ndarray)) and len(override) >= 3:
                try:
                    rgb = np.clip(np.asarray(override[:3], dtype=np.float32), 0.0, 1.0)
                    overridden = True
                except Exception as exc:
                    logging.warning("scanner_core._scanner_estimated_cell_colors: %s", exc)
        active_ratio = float(active.mean())
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


def _scanner_display_batch_colors(state, estimates=None):
    measured = _scanner_measured_cell_colors(state)
    if measured is not None:
        return measured, "actual"
    estimates = estimates if estimates is not None else _scanner_estimated_cell_colors(state)
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


def _scanner_pattern_repeats(state) -> tuple[int, int]:
    if hasattr(state, 'swatch_repeat_counts'):
        return state.swatch_repeat_counts()
    rep_r = int(np.clip(int(state.get('scanner_pattern_repeat_rows', 24)), 1, 128))
    rep_c = int(np.clip(int(state.get('scanner_pattern_repeat_cols', 24)), 1, 128))
    return rep_r, rep_c


def _FallbackUnitTile(state, r: int, c: int) -> np.ndarray:
    rows = max(1, int(state.get('scanner_rows', 1)))
    cols = max(1, int(state.get('scanner_cols', 1)))
    sel_r = int(np.clip(int(state.get('scanner_selected_row', 0)), 0, rows - 1))
    sel_c = int(np.clip(int(state.get('scanner_selected_col', 0)), 0, cols - 1))
    cell_index = int(r) * cols + int(c)
    cells = state.get('swatch_cells') or {}
    entry = cells.get(f"{int(r)},{int(c)}")

    if (int(r), int(c)) == (sel_r, sel_c):
        bitmap = np.asarray(state.get('bitmap', np.ones((4, 4), dtype=np.float32)), dtype=np.float32)
        raw_colors = state.get('row_colors') or _scanner_base_palette(state)
    elif isinstance(entry, dict):
        bitmap = np.asarray(entry.get('bitmap', state._scanner_random_bitmap(cell_index)), dtype=np.float32)
        raw_colors = entry.get('row_colors') or _scanner_base_palette(state)
    else:
        bitmap = np.asarray(state._scanner_random_bitmap(cell_index), dtype=np.float32)
        raw_colors = _scanner_base_palette(state)

    if bitmap.ndim != 2 or bitmap.size == 0:
        bitmap = np.ones((4, 4), dtype=np.float32)
    p_rows, p_cols = bitmap.shape
    palette = np.asarray([_as_rgba(col, [0.7, 0.7, 0.7, 1.0])[:3] for col in (raw_colors or [[0.7, 0.7, 0.7]])], dtype=np.float32)
    if palette.size == 0:
        palette = np.asarray([[0.7, 0.7, 0.7]], dtype=np.float32)

    h, w = 96, 96
    yy = np.linspace(0.0, float(p_rows), h, endpoint=False, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, float(p_cols), w, endpoint=False, dtype=np.float32)[None, :]
    row_idx = np.floor(yy).astype(np.int32) % p_rows
    col_idx = np.floor(xx).astype(np.int32) % p_cols
    u = xx - np.floor(xx)
    v = yy - np.floor(yy)

    leg_x = np.where(u < 0.5, 0.28 + 0.18 * v, 0.72 - 0.18 * v)
    dist = np.abs(u - leg_x)
    tube = np.clip(1.0 - (dist / 0.21) ** 2, 0.0, 1.0)
    arch = np.clip(1.0 - ((u - 0.5) / 0.34) ** 2 - ((v - 0.82) / 0.24) ** 2, 0.0, 1.0)
    yarn_mask = np.maximum(tube, 0.85 * arch)
    twist = 0.09 * np.sin((u * 14.0 + np.where(u < 0.5, v, -v) * 10.0) * np.pi)
    ridge = np.where(yarn_mask > 0.02, 0.38 + 0.58 * np.sqrt(yarn_mask) + twist, 0.18)

    active = bitmap[row_idx, col_idx] > 0.5
    eff_row = np.where(active | (v < 0.48), row_idx, (row_idx - 1) % p_rows)
    base_rgb = palette[eff_row % len(palette)]
    inactive_dim = np.where(active, 1.0, 0.78)[:, :, None]
    shaded = np.clip(base_rgb * ridge[:, :, None] * inactive_dim, 0.0, 1.0)
    return np.ascontiguousarray((shaded * 255.0 + 0.5).astype(np.uint8))


def _swatch_tile_cache_key(state, r: int, c: int, sel_r: int, sel_c: int, cols: int, lighting_sig: tuple) -> tuple:
    palette_sig = tuple(tuple(round(float(x), 4) for x in col[:3]) for col in _scanner_base_palette(state))
    if (r, c) == (sel_r, sel_c):
        active_cols = tuple(tuple(round(float(x), 4) for x in col[:3]) for col in state.active_colors())
        row_cols = tuple(tuple(round(float(x), 4) for x in col[:3]) for col in (state.get('row_colors') or []))
        bm_bytes = np.asarray(state.get('bitmap', []), dtype=np.float32).tobytes()
        return (
            "active",
            int(getattr(state, 'mesh_epoch', 0)),
            active_cols,
            row_cols,
            palette_sig,
            bm_bytes,
            lighting_sig,
        )
    cells = state.get('swatch_cells') or {}
    entry = cells.get(f"{r},{c}")
    if isinstance(entry, dict):
        bm_bytes = np.asarray(entry.get('bitmap', []), dtype=np.float32).tobytes()
        ctrl_bytes = b"".join(np.asarray(cr, dtype=np.float32).tobytes() for cr in (entry.get('ctrl_rows') or []))
        params_tuple = tuple(round(float(x), 5) for x in (entry.get('params') or []))
        rc_tuple = tuple(tuple(round(float(x), 4) for x in col[:3]) for col in (entry.get('row_colors') or []))
        return ("saved", bm_bytes, ctrl_bytes, params_tuple, rc_tuple, palette_sig, lighting_sig)
    return (
        "random",
        int(state.get('scanner_random_seed', 1)),
        round(float(state.get('scanner_pattern_density', 0.62)), 4),
        _scanner_pattern_dimensions(state),
        cols,
        palette_sig,
        lighting_sig,
    )


def ensure_swatch_unit_tile_images(state, renderer=None) -> tuple[list[np.ndarray], int]:
    cache = state.__dict__.get('_swatch_tile_cache')
    if not isinstance(cache, dict):
        cache = {}
        object.__setattr__(state, '_swatch_tile_cache', cache)
    tiles_epoch = int(state.__dict__.get('_swatch_tiles_epoch', 0))

    rows = max(1, int(state.get('scanner_rows', 1)))
    cols = max(1, int(state.get('scanner_cols', 1)))
    sel_r = int(np.clip(int(state.get('scanner_selected_row', 0)), 0, rows - 1))
    sel_c = int(np.clip(int(state.get('scanner_selected_col', 0)), 0, cols - 1))
    lighting = _scanner_lighting_settings(state)
    lighting_sig = tuple(sorted((k, round(float(v), 4)) for k, v in lighting.items()))

    can_gl = renderer is not None and hasattr(renderer, 'render_unit_cell_tile')
    max_gl_per_call = 4
    gl_rendered_count = 0
    regenerated = False
    mat_uniforms = _scanner_material_uniforms(state) if can_gl else None

    coords = [(sel_r, sel_c)] + [
        (r, c) for r in range(rows) for c in range(cols) if (r, c) != (sel_r, sel_c)
    ]
    for r, c in coords:
        key = _swatch_tile_cache_key(state, r, c, sel_r, sel_c, cols, lighting_sig)
        cached = cache.get((r, c))
        if cached is not None and len(cached) == 3:
            cached_key, cached_is_gl, _cached_img = cached
            if cached_key == key and (cached_is_gl or not can_gl):
                continue

        tile_rgb = None
        is_gl = False
        if can_gl and gl_rendered_count < max_gl_per_call:
            try:
                prepared, colors, center_xy, xp, yp = state.build_swatch_unit_cell_data(r, c)
                rendered = renderer.render_unit_cell_tile(
                    prepared, colors, center_xy, xp, yp, mat_uniforms
                )
                if (
                    isinstance(rendered, np.ndarray)
                    and rendered.ndim == 3
                    and rendered.shape[2] == 3
                    and rendered.size > 0
                ):
                    tile_rgb = np.ascontiguousarray(rendered, dtype=np.uint8)
                    is_gl = True
                    gl_rendered_count += 1
            except Exception as exc:
                logging.warning("scanner_core.ensure_swatch_unit_tile_images: %s", exc)

        if tile_rgb is None:
            if cached is not None and len(cached) == 3 and cached[0] == key:
                continue
            tile_rgb = _FallbackUnitTile(state, r, c)
            is_gl = False

        cache[(r, c)] = (key, is_gl, tile_rgb)
        regenerated = True

    tile_images = [cache[(r, c)][2] for r in range(rows) for c in range(cols)]
    grid_shape = (rows, cols)
    if state.__dict__.get('_swatch_tiles_grid_shape') != grid_shape:
        object.__setattr__(state, '_swatch_tiles_grid_shape', grid_shape)
        regenerated = True

    if regenerated or tiles_epoch == 0:
        tiles_epoch += 1
        object.__setattr__(state, '_swatch_tiles_epoch', tiles_epoch)
    return tile_images, tiles_epoch


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
