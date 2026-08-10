"""Scanner/puzzle logic shared by the GUI, the embedded simulator, and the database UI.

Extracted verbatim from gui.py. Everything here is pure state/geometry/image work
with no imgui dependency, so it can be imported by headless callers and by any
module that must not depend on the GUI layer.
"""
import copy

import numpy as np
from PIL import Image

from knitting_core import build_parametric_control_rows, build_spline_mesh


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
                except Exception:
                    pass
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
    embedded = state.get('embedded_scanner')
    result = getattr(embedded, "analysis_results", None) if embedded is not None else None
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


def _scanner_batch_colors_for_simulator(state):
    colors, _stage = _scanner_display_batch_colors(state)
    return [
        [
            float(np.clip(color[0] / 255.0, 0.0, 1.0)),
            float(np.clip(color[1] / 255.0, 0.0, 1.0)),
            float(np.clip(color[2] / 255.0, 0.0, 1.0)),
            1.0,
        ]
        for color in colors
    ]


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
    template_bitmap = None
    if hasattr(state, '_scanner_template'):
        try:
            template_bitmap = np.asarray(state._scanner_template().get('bitmap', None), dtype=np.float32)
        except Exception:
            template_bitmap = None
    if template_bitmap is not None and template_bitmap.ndim == 2 and template_bitmap.size:
        return max(2, int(template_bitmap.shape[0])), max(2, int(template_bitmap.shape[1]))
    return (
        max(2, int(state.get('scanner_pattern_rows', max(2, int(state.bitmap_size[0]))))),
        max(2, int(state.get('scanner_pattern_cols', max(2, int(state.bitmap_size[1]))))),
    )


def _scanner_pattern_repeats(state):
    repeat_rows = int(np.clip(int(state.get('scanner_pattern_repeat_rows', 3)), 1, 64))
    repeat_cols = int(np.clip(int(state.get('scanner_pattern_repeat_cols', 3)), 1, 64))
    return repeat_rows, repeat_cols


def _scanner_repeat_spacing(state):
    spacing_x = float(np.clip(float(state.get('scanner_repeat_spacing_x', 1.0)), 0.55, 1.45))
    spacing_y = float(np.clip(float(state.get('scanner_repeat_spacing_y', 1.0)), 0.55, 1.45))
    return spacing_x, spacing_y


def _scanner_batch_texture_size(state):
    width = int(np.clip(int(state.get('scanner_batch_texture_width', 420)), 160, 2048))
    height = int(round(width * 340.0 / 420.0))
    height = int(np.clip(height, 120, 1660))
    return width, height


def _scanner_capture_image_size(state, key='scanner_capture_width'):
    width = int(np.clip(int(state.get(key, 1024)), 320, 4096))
    height = int(round(width * 0.75))
    height = int(np.clip(height, 240, 3072))
    return width, height


def _puzzle_capture_rect(state):
    rect = np.asarray(state.get('puzzle_capture_rect', [0.08, 0.08, 0.84, 0.84]), dtype=np.float32).reshape(-1)
    if rect.size < 4:
        rect = np.array([0.08, 0.08, 0.84, 0.84], dtype=np.float32)
    x = float(np.clip(rect[0], 0.0, 0.95))
    y = float(np.clip(rect[1], 0.0, 0.95))
    w = float(np.clip(rect[2], 0.05, 1.0 - x))
    h = float(np.clip(rect[3], 0.05, 1.0 - y))
    return [x, y, w, h]


def _puzzle_project_points(pts, mvp, vp_w, vp_h):
    """Projects world-space points through mvp into (top-down) pixel coordinates."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    homog = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    clip = homog @ mvp.T
    w = np.where(np.abs(clip[:, 3]) > 1e-8, clip[:, 3], 1.0)
    ndc = clip[:, :3] / w[:, None]
    px = (ndc[:, 0] + 1.0) * 0.5 * vp_w
    py = (1.0 - ndc[:, 1]) * 0.5 * vp_h
    return np.stack([px, py], axis=1)


def _puzzle_tiled_control_points(state):
    """Rebuilds the real per-stitch spline control points, tiled across the same
    X/Y display copies baked into the rendered mesh, in world (pre-model-matrix) space."""
    if not state.ctrl_rows:
        return []
    state._ensure_spline_radius_rows()
    radius = max(float(state.params[state._pidx['radius']]), 1e-6)
    radius_profiles = [np.asarray(row, dtype=np.float32) for row in state.spline_radius_rows]
    base_vl = build_spline_mesh(
        state.ctrl_rows,
        state.params,
        state.config,
        state._pidx,
        np.asarray(state.period_offset_x, dtype=np.float32),
        radius_ctrl_rows=radius_profiles,
    )
    x_period = state._display_copy_x_period(base_vl, radius)
    y_period = state._display_copy_y_period(base_vl, radius)
    depth_gap = max(radius * 2.4, 1e-6)
    z_period = state._display_copy_z_period(base_vl, depth_gap)
    copies_x = int(state.display_copies[0])
    copies_y = int(state.display_copies[1])

    tiled = []
    for row_idx, row in enumerate(state.ctrl_rows):
        row = np.asarray(row, dtype=np.float32)
        if row.shape[0] == 0:
            continue
        for y_tile in range(-copies_y, copies_y + 1):
            y_shift = np.array([0.0, y_tile * y_period, -y_tile * z_period], dtype=np.float32)
            for x_tile in range(-copies_x, copies_x + 1):
                shift = y_shift + np.array([x_tile * x_period, 0.0, 0.0], dtype=np.float32)
                tiled.append((row + shift[None, :], row_idx))
    return tiled


def _puzzle_period_pixel_vectors(state, renderer):
    """Computes the exact pixel-space translation for one X repeat and one Y repeat,
    using the same world-space periods and camera/model matrix as the mesh tiling.
    Because the camera is orthographic, a constant world-space offset always maps
    to a constant pixel-space offset, independent of position."""
    if not state.ctrl_rows:
        return None
    vp_w = int(getattr(renderer, "vp_w", 0))
    vp_h = int(getattr(renderer, "vp_h", 0))
    if vp_w < 2 or vp_h < 2:
        return None

    state._ensure_spline_radius_rows()
    radius = max(float(state.params[state._pidx['radius']]), 1e-6)
    radius_profiles = [np.asarray(row, dtype=np.float32) for row in state.spline_radius_rows]
    base_vl = build_spline_mesh(
        state.ctrl_rows,
        state.params,
        state.config,
        state._pidx,
        np.asarray(state.period_offset_x, dtype=np.float32),
        radius_ctrl_rows=radius_profiles,
    )
    x_period = state._display_copy_x_period(base_vl, radius)
    y_period = state._display_copy_y_period(base_vl, radius)
    depth_gap = max(radius * 2.4, 1e-6)
    z_period = state._display_copy_z_period(base_vl, depth_gap)

    model_mat = state.current_model_matrix()
    mvp = (state.camera.mvp(vp_w, vp_h) @ model_mat).astype(np.float32)
    ref_pts = np.array([
        [0.0, 0.0, 0.0],
        [x_period, 0.0, 0.0],
        [0.0, y_period, -z_period],
    ], dtype=np.float32)
    pix = _puzzle_project_points(ref_pts, mvp, vp_w, vp_h)
    return {
        "x_step": pix[1] - pix[0],
        "y_step": pix[2] - pix[0],
        "x_period": float(x_period),
        "y_period": float(y_period),
        "z_period": float(z_period),
        "vp_w": vp_w,
        "vp_h": vp_h,
    }


def _puzzle_build_seamless_tile(state, renderer, cols, rows, crop_rect=None):
    """Extracts one exact repeat-period tile from the live render and glues `cols` x
    `rows` copies of it edge-to-edge. Because the tile size equals the true geometric
    repeat period in pixels, adjacent copies connect without search-based alignment.

    `crop_rect`, if given, is an (rx, ry, rw, rh) fraction whose (rx, ry) anchors the
    tile's top-left corner (rw/rh are unused -- the tile is always exactly one period
    wide/tall). Defaults to Puzzle Mode's manual `_puzzle_capture_rect`, but Scan Mode's
    automated capture passes its own auto-detected tight anchor instead."""
    vp_w = int(getattr(renderer, "vp_w", 0))
    vp_h = int(getattr(renderer, "vp_h", 0))
    if vp_w < 2 or vp_h < 2 or getattr(renderer, "color_tex", None) is None:
        return None, None, {}

    periods = _puzzle_period_pixel_vectors(state, renderer)
    if periods is None:
        return None, None, {}

    x_step = periods["x_step"]
    y_step = periods["y_step"]
    tile_w = max(4, int(round(abs(float(x_step[0])))))
    tile_h = max(4, int(round(abs(float(y_step[1])))))
    tile_w = min(tile_w, vp_w)
    tile_h = min(tile_h, vp_h)

    raw = renderer.color_tex.read()
    full_image = Image.frombytes("RGBA", (vp_w, vp_h), raw).convert("RGB")
    full_image = full_image.transpose(Image.FLIP_TOP_BOTTOM)

    rx, ry, _rw, _rh = crop_rect if crop_rect is not None else _puzzle_capture_rect(state)
    x0 = int(np.clip(round(rx * vp_w), 0, vp_w - tile_w))
    y0 = int(np.clip(round(ry * vp_h), 0, vp_h - tile_h))
    tile_box = (x0, y0, x0 + tile_w, y0 + tile_h)
    tile = full_image.crop(tile_box)

    cols = max(1, int(cols))
    rows = max(1, int(rows))
    # Guard against exceeding typical GL max texture size (and runaway memory use)
    # when the user pushes the copy count high.
    max_canvas_dim = 8192
    if tile_w * cols > max_canvas_dim:
        cols = max(1, max_canvas_dim // tile_w)
    if tile_h * rows > max_canvas_dim:
        rows = max(1, max_canvas_dim // tile_h)

    canvas = Image.new("RGB", (tile_w * cols, tile_h * rows), (18, 23, 31))
    for row_i in range(rows):
        for col_i in range(cols):
            canvas.paste(tile, (col_i * tile_w, row_i * tile_h))

    info = {
        "tile_box": tile_box,
        "tile_w": tile_w,
        "tile_h": tile_h,
        "cols": cols,
        "rows": rows,
        "skew_x_px": float(x_step[1]),
        "skew_y_px": float(y_step[0]),
        "x_period_world": periods["x_period"],
        "y_period_world": periods["y_period"],
    }
    return tile, canvas, info


_SCAN_TILE_SNAPSHOT_FIELDS = (
    'params', 'bitmap', 'bitmap_size', 'loop_heights',
    'row_colors', 'use_row_colors', 'display_copies',
    'scanner_preview_grid_enabled', 'mesh_center', 'model_t',
    # Fully-derived spline state, snapshotted and restored verbatim (not
    # regenerated from bitmap+params) so any manually-edited control points on
    # the live model survive round-tripping through a temporary scan pattern.
    # period_offset_y is included because rebuild_spline_from_params() recomputes
    # it from the row count, and this function calls that against a temporary
    # scan pattern whose row count differs from the user's model.
    'ctrl_rows', 'period_offset_x', 'period_offset_y', 'spline_radius_rows', 'param_ref_radius',
    'flat_pts', '_row_starts', 'param_ref_ctrl_rows',
)


def _scan_snapshot_state(state):
    snap = {key: copy.deepcopy(state.get(key)) for key in _SCAN_TILE_SNAPSHOT_FIELDS}
    snap['camera'] = {
        'target': np.array(state.camera.target, dtype=np.float32).copy(),
        'dist': float(state.camera.dist),
        'az': float(state.camera.az),
        'el': float(state.camera.el),
        'fov_deg': float(state.camera.fov_deg),
    }
    return snap


def _scan_restore_state(state, snap):
    for key in _SCAN_TILE_SNAPSHOT_FIELDS:
        setattr(state, key, snap[key])
    cam = snap['camera']
    state.camera.target = cam['target']
    state.camera.dist = cam['dist']
    state.camera.az = cam['az']
    state.camera.el = cam['el']
    state.camera.fov_deg = cam['fov_deg']


def _scan_render_tiled_pattern_image(state, renderer, bitmap, loop_heights, colors, repeat_cols, repeat_rows, copies=1, target_w=480, camera_az_deg=0.0, camera_el_deg=0.0, zoom=1.0):
    """Renders one Scan Mode pattern with the real 3D pipeline and tiles it using
    Puzzle Mode's exact-period capture/glue logic -- automated per pattern, with no
    Puzzle Mode UI shown. Temporarily takes over the shared live model/camera/renderer
    (duplicate -> auto-frame -> capture -> exact-period crop -> glue), then restores
    everything so the user's own edited model/3D View is left exactly as it was.

    `camera_az_deg`/`camera_el_deg` let a caller request a genuinely different
    viewing angle (used for per-capture-angle scan renders): rotating azimuth
    about the world Y axis (the fabric's own row axis) reveals different yarn
    surface facets under the scene's fixed light direction -- a real
    diffuse-shading difference, not just a reprojection of the same pixels.
    This does introduce some Y-tiling skew when repeat_rows > 1 (the Y-repeat
    offset used to avoid z-fighting between duplicated rows has a small Z
    component, which becomes a real parallax shift once az != 0) -- callers
    that vary az per capture should keep az modest and/or accept a mild
    seam between internal repeat copies as a worthwhile trade for genuinely
    different per-angle shading. `zoom` scales the auto-fit camera distance
    (>1 = closer/more zoomed in)."""
    snap = _scan_snapshot_state(state)
    # state.rebuild_spline_mesh() always uploads the mesh to `state.renderer`,
    # so the renderer we draw with has to *be* state.renderer for the duration.
    # Otherwise a caller that passes its own renderer (EmbeddedMujocoScanner's
    # dedicated _pattern_renderer) draws a scene that was never given any
    # meshes: the render is just the clear color, and the capture comes back a
    # flat dark frame. Binding it here (rather than forcing callers to pass the
    # main renderer) keeps the dedicated-renderer isolation that avoids fighting
    # the live "3D View" viewport over resize().
    prev_renderer = state.renderer
    state.renderer = renderer
    try:
        state.params = state._scanner_template_params()
        state.bitmap = np.asarray(bitmap, dtype=np.float32)
        state.bitmap_size = np.array(state.bitmap.shape, dtype=np.int32)
        state.loop_heights = np.asarray(loop_heights, dtype=np.float32)
        state.row_colors = [list(np.asarray(c, dtype=np.float32)[:3]) for c in colors] if colors else state.row_colors
        state.use_row_colors = True
        copies = max(1, int(copies))
        state.display_copies = np.array([copies, copies], dtype=np.int32)
        state.scanner_preview_grid_enabled = False

        state.rebuild_spline_from_params()
        state.rebuild_spline_mesh(preserve_model_placement=False)

        mesh_verts = [
            np.asarray(v, dtype=np.float32)
            for v, _row_idx in getattr(renderer, "mesh_pick_data", [])
            if len(v)
        ]
        if not mesh_verts:
            # Nothing was uploaded to draw: rendering anyway would hand back a
            # flat clear-color frame that looks like a real (but black) capture.
            # Report failure instead so callers use their fallback imagery.
            return None
        state.camera.az = float(np.radians(camera_az_deg))
        state.camera.el = float(np.radians(camera_el_deg))
        all_v = np.vstack(mesh_verts)
        bounds_min = all_v.min(axis=0)
        bounds_max = all_v.max(axis=0)
        half_w = max(float(bounds_max[0] - bounds_min[0]) * 0.5 * 1.15, 1e-3)
        half_h = max(float(bounds_max[1] - bounds_min[1]) * 0.5 * 1.15, 1e-3)

        target_h = max(240, min(960, int(round(target_w * (half_h / max(half_w, 1e-6))))))
        aspect = float(target_w) / float(target_h)
        half_fov = np.radians(max(1.0, float(state.camera.fov_deg)) * 0.5)
        tan_half_fov = max(np.tan(half_fov), 1e-6)
        dist_for_h = half_h / tan_half_fov
        dist_for_w = half_w / (tan_half_fov * aspect)
        state.camera.dist = max(dist_for_h, dist_for_w, 1e-3) / max(0.1, float(zoom))

        renderer.resize(target_w, target_h)
        model_mat = state.current_model_matrix()
        mvp = (state.camera.mvp(target_w, target_h) @ model_mat).astype(np.float32)
        mv = (state.camera.mv(target_w, target_h) @ model_mat).astype(np.float32)
        material_uniforms = _scanner_material_uniforms(state)
        renderer.render(mvp, mv, material_uniforms)

        # Zoom/crop so no unnecessary background is visible: derive the crop
        # anchor from the actual projected content bounding box, instead of a
        # manually-tuned fraction (there is no user available to tune one here).
        crop_rect = None
        tiled_points = _puzzle_tiled_control_points(state)
        if tiled_points:
            all_proj = []
            for verts, _row_idx in tiled_points:
                all_proj.append(_puzzle_project_points(verts, mvp, target_w, target_h))
            all_proj = np.vstack(all_proj)
            x_min, y_min = all_proj.min(axis=0)
            x_max, y_max = all_proj.max(axis=0)
            margin_x = (x_max - x_min) * 0.04
            margin_y = (y_max - y_min) * 0.04
            rx = float(np.clip((x_min - margin_x) / target_w, 0.0, 0.95))
            ry = float(np.clip((y_min - margin_y) / target_h, 0.0, 0.95))
            rw = float(np.clip((x_max + margin_x) / target_w - rx, 0.05, 1.0 - rx))
            rh = float(np.clip((y_max + margin_y) / target_h - ry, 0.05, 1.0 - ry))
            crop_rect = [rx, ry, rw, rh]

        _tile, canvas, _info = _puzzle_build_seamless_tile(
            state, renderer, repeat_cols, repeat_rows, crop_rect=crop_rect,
        )
        return canvas
    finally:
        # Restore the real renderer first, so the rebuild below re-uploads the
        # user's own model to the viewport renderer (and not to a temporary one).
        state.renderer = prev_renderer
        _scan_restore_state(state, snap)
        state.rebuild_spline_mesh(preserve_model_placement=True)


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


def _normalize_scanner_curves(curves):
    valid = [np.asarray(curve, dtype=np.float32)[:, :2] for curve in curves if len(curve) > 1]
    if not valid:
        return []
    pts = np.vstack(valid)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    center = (min_xy + max_xy) * 0.5
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = 1.08 / float(max(span[0], span[1]))
    return [((curve - center) * scale).astype(np.float32) for curve in valid]


def _random_bitmap_model_curves(state, pattern_rows, pattern_cols, seed, density, repeat_rows=3, repeat_cols=3):
    rng = np.random.default_rng(int(seed))
    density = float(np.clip(density, 0.05, 0.95))
    bitmap = (rng.random((pattern_rows, pattern_cols)) < density).astype(np.float32)
    if not np.any(bitmap > 0.5):
        bitmap[rng.integers(0, pattern_rows), rng.integers(0, pattern_cols)] = 1.0

    if hasattr(state, '_scanner_template'):
        try:
            template = state._scanner_template()
            params = np.asarray(template.get('params', state.params), dtype=np.float32).copy()
            source = np.asarray(template.get('loop_heights', np.empty((0, 0))), dtype=np.float32)
        except Exception:
            params = np.asarray(state.params, dtype=np.float32)
            source = np.empty((0, 0), dtype=np.float32)
    else:
        params = np.asarray(state.params, dtype=np.float32)
        source = np.empty((0, 0), dtype=np.float32)

    heights = np.zeros_like(bitmap, dtype=np.float32)
    for row_idx in range(pattern_rows):
        if getattr(state, '_lh_idx', ()):
            idx = state._lh_idx[min(row_idx, len(state._lh_idx) - 1)]
            heights[row_idx, :] = float(params[idx])
        else:
            heights[row_idx, :] = 3.0
    if source.ndim == 2 and source.size:
        keep_rows = min(pattern_rows, source.shape[0])
        keep_cols = min(pattern_cols, source.shape[1])
        heights[:keep_rows, :keep_cols] = source[:keep_rows, :keep_cols]

    ctrl_rows = build_parametric_control_rows(
        params,
        bitmap,
        state._pidx,
        state._lh_idx,
        state.samples_per_loop,
        loop_heights=heights * (bitmap > 0.5),
    )
    return _normalize_scanner_curves(ctrl_rows)


def _generate_scanner_random_patterns(state):
    rows = max(1, int(state.scanner_rows))
    cols = max(1, int(state.scanner_cols))
    pattern_rows, pattern_cols = _scanner_pattern_dimensions(state)
    repeat_rows, repeat_cols = _scanner_pattern_repeats(state)
    seed = int(state.get('scanner_random_seed', 1))
    density = float(state.get('scanner_pattern_density', 0.62))
    return [
        _random_bitmap_model_curves(state, pattern_rows, pattern_cols, seed + cell * 9973, density, repeat_rows, repeat_cols)
        for cell in range(rows * cols)
    ]
