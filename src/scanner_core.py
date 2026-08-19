"""Scanner and puzzle logic, with no UI attached.

Scan Mode, Puzzle Mode and UR5 Robot Mode all need the same things: the fabric
palette and per-cell colours, the pattern grid geometry, the tile renderer that
turns one knitted pattern into a seamless repeat, and the on-disk caches around
them. None of it draws anything.

Keeping it out of gui.py is what lets UR5 Mode use it. gui_ur5 previously
imported gui from inside three functions to reach eleven of these helpers --
a cycle it had to dodge at call time because gui imports gui_ur5. With them
here, that import goes away.

Nothing in this module may import imgui; a test enforces it. The names are
re-imported into gui.py, so call sites there are unchanged.
"""

import contextlib
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fabric_scanner import clamp_capture_size
from knitting_core import build_parametric_control_rows
from rendering import pil_to_texture


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
    return clamp_capture_size(state.get(key, 1024))


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
    x_period, y_period, z_period = state.display_copy_periods()
    copies_x = int(state.display_copies[0])
    copies_y = int(state.display_copies[1])

    tiled = []
    for row_idx, row in enumerate(state.ctrl_rows):
        row = np.asarray(row, dtype=np.float32)
        if row.shape[0] == 0:
            continue
        for y_tile in range(-copies_y, copies_y + 1):
            y_shift = np.array([0.0, y_tile * y_period, 0.0], dtype=np.float32)
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

    x_period, y_period, z_period = state.display_copy_periods()

    model_mat = state.current_model_matrix()
    mvp = (state.camera.mvp(vp_w, vp_h) @ model_mat).astype(np.float32)
    ref_pts = np.array([
        [0.0, 0.0, 0.0],
        [x_period, 0.0, 0.0],
        [0.0, y_period, 0.0],
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


def _puzzle_build_seamless_tile(state, renderer, cols, rows, crop_rect=None, periods=None, build_canvas=True):
    """Extracts one exact repeat-period tile from the live render and glues `cols` x
    `rows` copies of it edge-to-edge. Because the tile size equals the true geometric
    repeat period in pixels, adjacent copies connect without search-based alignment.

    `crop_rect`, if given, is an (rx, ry, rw, rh) fraction whose (rx, ry) anchors the
    tile's top-left corner (rw/rh are unused -- the tile is always exactly one period
    wide/tall). Defaults to Puzzle Mode's manual `_puzzle_capture_rect`, but Scan Mode's
    automated capture passes its own auto-detected tight anchor instead.

    `build_canvas=False` returns None in place of the glued image and leaves the
    caller to hold a TiledFabricTexture built from `tile` and info's cols/rows.
    Puzzle Mode wants the real canvas to display; Scan Mode does not, and at 64
    repeats that canvas is 165 MB of exact repetition per cell."""
    vp_w = int(getattr(renderer, "vp_w", 0))
    vp_h = int(getattr(renderer, "vp_h", 0))
    if vp_w < 2 or vp_h < 2 or getattr(renderer, "color_tex", None) is None:
        return None, None, {}

    # `periods` may be supplied so a batch of renders all use one repeat
    # period and therefore come out the same pixel size; otherwise it is
    # measured from the geometry currently in `state`.
    if periods is None:
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

    if build_canvas:
        canvas = Image.new("RGB", (tile_w * cols, tile_h * rows), (18, 23, 31))
        for row_i in range(rows):
            for col_i in range(cols):
                canvas.paste(tile, (col_i * tile_w, row_i * tile_h))
    else:
        # The caller intends to hold a TiledFabricTexture instead. At high
        # repeat counts this canvas is hundreds of MB of exact repetition, so
        # not building it is the entire saving.
        canvas = None

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


class TiledFabricTexture:
    """One seamless period tile plus the repeat counts that tile it.

    _puzzle_build_seamless_tile builds its glued image by stamping `tile`
    cols x rows times edge to edge, so the result is exact integer repetition.
    Measured on a real scan cell: all 1088 blocks of an 8160x6720 canvas were
    byte-identical to the 480x105 tile they came from -- 151 KB of actual
    content stored as 165 MB. Twenty cells of that is 3.9 GB, which is what put
    the app into swap and made the whole machine stall.

    Keeping the tile and stamping on demand is lossless: `size` reports exactly
    what the glued image measured, and rasterize() with no cap reproduces it
    byte for byte. Consumers that only need a smaller version -- a grid-preview
    slot, or a camera quad a few hundred pixels across -- ask for that size and
    get a properly filtered image, rather than an aliased bicubic downsample of
    a giant one.

    Quacks like a PIL image for `size`, `mode` and `convert` so that any caller
    that was handed the glued image still works.
    """

    __slots__ = ("tile", "cols", "rows", "_cache")

    # Rasterisations at or above this are not cached; the whole point is to
    # avoid holding large glued images alive.
    _CACHE_MAX_DIM = 2600
    _CACHE_ENTRIES = 2

    def __init__(self, tile, cols, rows):
        self.tile = tile if tile.mode == "RGB" else tile.convert("RGB")
        self.cols = max(1, int(cols))
        self.rows = max(1, int(rows))
        self._cache = {}

    @property
    def mode(self):
        return "RGB"

    @property
    def size(self):
        tile_w, tile_h = self.tile.size
        return (tile_w * self.cols, tile_h * self.rows)

    def _build(self, max_dim):
        tile_w, tile_h = self.tile.size
        full_w, full_h = self.size
        paste_tile, paste_w, paste_h = self.tile, tile_w, tile_h
        if max_dim is not None and max(full_w, full_h) > int(max_dim):
            shrink = min(int(max_dim) / full_w, int(max_dim) / full_h)
            paste_w = max(2, int(round(tile_w * shrink)))
            paste_h = max(2, int(round(tile_h * shrink)))
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
            paste_tile = self.tile.resize((paste_w, paste_h), resample)

        canvas = Image.new("RGB", (paste_w * self.cols, paste_h * self.rows))
        for row in range(self.rows):
            for col in range(self.cols):
                canvas.paste(paste_tile, (col * paste_w, row * paste_h))

        if (paste_w, paste_h) != (tile_w, tile_h):
            # Rounding the tile's width and height to whole pixels
            # independently skews the fabric's aspect ratio, so square it back
            # up against the full-resolution geometry. Cheap: by this point the
            # canvas is small.
            target_h = max(1, int(round(canvas.size[0] * full_h / max(full_w, 1))))
            if target_h != canvas.size[1]:
                resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
                canvas = canvas.resize((canvas.size[0], target_h), resample)
        return canvas

    def rasterize(self, max_dim=None):
        """Stamps the tile out, capped to `max_dim` on its longest side.

        `max_dim=None` reproduces the full glued image exactly.
        """
        full_w, full_h = self.size
        if max_dim is not None and max(full_w, full_h) <= int(max_dim):
            max_dim = None
        key = None if max_dim is None else int(max_dim)
        if key is not None and key in self._cache:
            return self._cache[key]
        canvas = self._build(max_dim)
        if key is not None and key <= self._CACHE_MAX_DIM:
            if len(self._cache) >= self._CACHE_ENTRIES:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = canvas
        return canvas

    def convert(self, mode="RGB"):
        image = self.rasterize()
        return image if mode == "RGB" else image.convert(mode)


@contextlib.contextmanager
def _scan_batch(state):
    """Collapses a run of scan renders down to a single model restore.

    Every _scan_render_tiled_pattern_image call borrows the live model, swaps in
    a scan pattern, and then puts the user's model back so the 3D View is never
    left showing a scan pattern. Across a batch that restore is dead work: the
    next cell overwrites the mesh it just re-uploaded. Twelve cells paid for
    twelve restores where one suffices.

    The snapshot is taken here, before any cell has touched the state, so what
    gets restored is still the user's own model. The per-cell inputs
    (_scanner_random_bitmap, _scanner_loop_heights_for_bitmap) are derived from
    the cached scanner template rather than live state, so leaving a previous
    cell's pattern in place between iterations does not affect them.
    """
    depth = int(state.__dict__.get('_scan_batch_depth', 0))
    if depth == 0:
        object.__setattr__(state, '_scan_batch_snapshot', _scan_snapshot_state(state))
    object.__setattr__(state, '_scan_batch_depth', depth + 1)
    try:
        yield
    finally:
        remaining = int(state.__dict__.get('_scan_batch_depth', 1)) - 1
        object.__setattr__(state, '_scan_batch_depth', remaining)
        if remaining <= 0:
            state.__dict__.pop('_scan_batch_depth', None)
            snapshot = state.__dict__.pop('_scan_batch_snapshot', None)
            if snapshot is not None:
                _scan_restore_state(state, snapshot)
                state.rebuild_spline_mesh(preserve_model_placement=True)


def _scan_batch_active(state):
    return int(state.__dict__.get('_scan_batch_depth', 0)) > 0


def _scan_autoframe(state, renderer, target_w, camera_az_deg, camera_el_deg, zoom):
    """Fits the camera to the geometry currently uploaded to `renderer` and
    returns the framing as a dict of target_h / camera_dist / crop_rect.

    Assumes the mesh has already been rebuilt. crop_rect is normalised against
    (target_w, target_h), so reusing a frame across renders keeps their output
    identical in size.
    """
    mesh_verts = [
        np.asarray(v, dtype=np.float32)
        for v, _row_idx in getattr(renderer, "mesh_pick_data", [])
        if len(v)
    ]
    if not mesh_verts:
        return None

    state.camera.az = float(np.radians(camera_az_deg))
    state.camera.el = float(np.radians(camera_el_deg))
    all_v = np.vstack(mesh_verts)
    # Fit to the geometry as it is actually drawn. mesh_pick_data holds raw
    # model-space vertices, but the render multiplies them by
    # current_model_matrix(), which carries the model's scale. Fitting the
    # unscaled bounds framed a scaled model as if it were unscaled: with a
    # scale of ~4.5 in X the camera ended up 4.5x too close, so less than one
    # repeat period fitted across the frame. _puzzle_build_seamless_tile then
    # clamped its period crop to the viewport width, producing a tile that was
    # a fraction of a period -- which is why Scan Mode's duplicates did not line
    # up while Puzzle Mode's, framed by hand, always did.
    model_mat = state.current_model_matrix()
    homog = np.concatenate(
        [all_v, np.ones((len(all_v), 1), dtype=np.float32)], axis=1
    ).astype(np.float32)
    drawn_v = (homog @ model_mat.T)[:, :3]
    bounds_min = drawn_v.min(axis=0)
    bounds_max = drawn_v.max(axis=0)
    half_w = max(float(bounds_max[0] - bounds_min[0]) * 0.5 * 1.15, 1e-3)
    half_h = max(float(bounds_max[1] - bounds_min[1]) * 0.5 * 1.15, 1e-3)

    target_h = max(240, min(960, int(round(target_w * (half_h / max(half_w, 1e-6))))))
    aspect = float(target_w) / float(target_h)
    half_fov = np.radians(max(1.0, float(state.camera.fov_deg)) * 0.5)
    tan_half_fov = max(np.tan(half_fov), 1e-6)
    dist_for_h = half_h / tan_half_fov
    dist_for_w = half_w / (tan_half_fov * aspect)
    camera_dist = max(dist_for_h, dist_for_w, 1e-3) / max(0.1, float(zoom))
    state.camera.dist = camera_dist

    # Anchor the exact-period crop inside the fabric.
    #
    # This used to anchor at the projected bounding box's minimum corner, minus
    # a margin -- i.e. just outside the geometry. The period crop therefore
    # started on the fabric's top-left boundary, where the edge loops open into
    # empty space instead of interlocking with a neighbour. Stamping that out
    # 64x64 reproduced the boundary every repeat, which is what made Scan Mode's
    # duplicates read as separate motifs while Puzzle Mode, whose capture rect
    # is an inset the user places over the middle of the fabric, tiled cleanly.
    #
    # The centre is interior for any framing that holds more than about two
    # periods (the fit above gives 3.5 across and ~5 down), and
    # _puzzle_build_seamless_tile clamps the box back inside the viewport if a
    # period would overhang. rw/rh stay as the tight content extent; only rx/ry
    # are read when cropping a period.
    crop_rect = None
    mvp = (state.camera.mvp(target_w, target_h) @ model_mat).astype(np.float32)
    tiled_points = _puzzle_tiled_control_points(state)
    if tiled_points:
        all_proj = np.vstack([
            _puzzle_project_points(verts, mvp, target_w, target_h)
            for verts, _row_idx in tiled_points
        ])
        x_min, y_min = all_proj.min(axis=0)
        x_max, y_max = all_proj.max(axis=0)
        margin_x = (x_max - x_min) * 0.04
        margin_y = (y_max - y_min) * 0.04
        rx = float(np.clip((x_min + x_max) * 0.5 / target_w, 0.0, 0.95))
        ry = float(np.clip((y_min + y_max) * 0.5 / target_h, 0.0, 0.95))
        rw = float(np.clip((x_max + margin_x) / target_w - rx, 0.05, 1.0 - rx))
        rh = float(np.clip((y_max + margin_y) / target_h - ry, 0.05, 1.0 - ry))
        crop_rect = [rx, ry, rw, rh]

    return {'target_h': target_h, 'camera_dist': camera_dist, 'crop_rect': crop_rect}


def _scan_measure_pattern_frame(state, renderer, bitmap_shape, target_w=480, camera_az_deg=0.0, camera_el_deg=0.0, zoom=1.0):
    """Measures one framing for a whole grid of scan patterns.

    Uses an all-loops-active pattern of the given shape, so the frame describes
    the full fabric square rather than whichever loops a particular random
    bitmap happened to switch on. Feed the result to every cell's
    _scan_render_tiled_pattern_image call to get a uniform grid.
    """
    rows, cols = (int(v) for v in bitmap_shape)
    full_bitmap = np.ones((max(1, rows), max(1, cols)), dtype=np.float32)
    snap = _scan_snapshot_state(state)
    prev_renderer = state.renderer
    state.renderer = renderer
    try:
        state.params = state._scanner_template_params()
        state.bitmap = full_bitmap
        state.bitmap_size = np.array(full_bitmap.shape, dtype=np.int32)
        state.loop_heights = state._scanner_loop_heights_for_bitmap(full_bitmap)
        state.display_copies = np.array([1, 1], dtype=np.int32)
        state.scanner_preview_grid_enabled = False
        # rebuild_mesh=False: the very next line rebuilds and uploads the same
        # mesh, with the placement flag this pass actually wants.
        state.rebuild_spline_from_params(rebuild_mesh=False)
        state.rebuild_spline_mesh(preserve_model_placement=False)
        frame = _scan_autoframe(state, renderer, target_w, camera_az_deg, camera_el_deg, zoom)
        if frame is None:
            return None
        # The tile's pixel size is the repeat period, and that is measured from
        # the mesh bounds -- which also shrink when loops are switched off. So
        # the period has to be measured here too, on the fully active pattern,
        # or cells would still come out at different sizes despite sharing a
        # crop. Needs the viewport at its final size first.
        renderer.resize(target_w, int(frame['target_h']))
        frame['periods'] = _puzzle_period_pixel_vectors(state, renderer)
        return frame
    except Exception:
        return None
    finally:
        state.renderer = prev_renderer
        # Inside a _scan_batch the restore is deferred to the end of the batch.
        if not _scan_batch_active(state):
            _scan_restore_state(state, snap)
            state.rebuild_spline_mesh(preserve_model_placement=True)


def _scan_capture_one_azimuth(state, renderer, repeat_cols, repeat_rows, target_w,
                              camera_az_deg, camera_el_deg, zoom, frame):
    """Renders the mesh currently uploaded to `renderer` from one azimuth.

    Split out of _scan_render_tiled_pattern_image so a caller can build the
    pattern's geometry once and then sweep the camera over several angles: the
    mesh is identical for every angle of a scan cell, only the camera moves.
    Assumes the caller has already borrowed the state and uploaded the mesh.
    """
    if frame is None:
        frame = _scan_autoframe(state, renderer, target_w, camera_az_deg, camera_el_deg, zoom)
        if frame is None:
            # Nothing was uploaded to draw: rendering anyway would hand back a
            # flat clear-color frame that looks like a real (but black) capture.
            # Report failure so callers use their fallback imagery.
            return None
    else:
        # A shared frame still needs the camera pointed the same way it was when
        # the frame was measured, or the crop would not line up.
        state.camera.az = float(np.radians(camera_az_deg))
        state.camera.el = float(np.radians(camera_el_deg))
        state.camera.dist = float(frame['camera_dist'])
    target_h = int(frame['target_h'])
    crop_rect = frame.get('crop_rect')
    shared_periods = frame.get('periods')

    renderer.resize(target_w, target_h)
    model_mat = state.current_model_matrix()
    mvp = (state.camera.mvp(target_w, target_h) @ model_mat).astype(np.float32)
    mv = (state.camera.mv(target_w, target_h) @ model_mat).astype(np.float32)
    material_uniforms = _scanner_material_uniforms(state)
    renderer.render(mvp, mv, material_uniforms)

    tile, _canvas, info = _puzzle_build_seamless_tile(
        state, renderer, repeat_cols, repeat_rows, crop_rect=crop_rect,
        periods=shared_periods, build_canvas=False,
    )
    if tile is None:
        return None
    texture = TiledFabricTexture(tile, info["cols"], info["rows"])
    if not _tile_is_usable(texture, target_w, tile=tile):
        # The exact-period crop only works while the fabric is seen close to
        # face-on. Viewed edge-on the X period projects to almost nothing and the
        # crop degenerates into a sliver, which glues into a smear rather than
        # fabric. Report failure so the caller falls back to a tile that was
        # built at an angle where the period is measurable.
        return None
    return texture


def _scan_render_tiled_pattern_images(state, renderer, bitmap, loop_heights, colors,
                                      repeat_cols, repeat_rows, camera_az_degs,
                                      copies=1, target_w=480, camera_el_deg=0.0,
                                      zoom=1.0, frame=None):
    """Renders one scan pattern from several azimuths, building its mesh once.

    A scan visits every angle of a cell consecutively (74 targets, 11 cell
    changes on a 3x4 grid), and the yarn geometry is identical for all of them
    -- only the camera azimuth differs. Rendering them one call at a time meant
    a full rebuild_spline_from_params + rebuild_spline_mesh + GPU upload per
    angle, which measured 30% of the entire scan loop. Sweeping the camera over
    the angles inside a single borrow does the same renders off one build.

    Returns {azimuth: TiledFabricTexture or None}.
    """
    snap = _scan_snapshot_state(state)
    prev_renderer = state.renderer
    state.renderer = renderer
    results = {}
    try:
        state.params = state._scanner_template_params()
        # Every stitch exists; the pattern is carried by loop height alone.
        # build_parametric_control_rows drops a loop outright wherever the
        # bitmap is zero -- has_loop forces loop_height to 0 there, whatever
        # height it was given -- and at the usual density that emptied whole
        # columns, so tiled copies stopped touching and a 64x64 repeat read as a
        # grid of separate motifs. An all-active bitmap keeps the yarn
        # continuous across every repeat, the way Puzzle Mode renders the same
        # model, while _scanner_loop_heights_for_bitmap still distinguishes the
        # cells as tall and short loops.
        pattern_bitmap = np.asarray(bitmap, dtype=np.float32)
        state.bitmap = np.ones_like(pattern_bitmap)
        state.bitmap_size = np.array(pattern_bitmap.shape, dtype=np.int32)
        state.loop_heights = np.asarray(loop_heights, dtype=np.float32)
        state.row_colors = [list(np.asarray(c, dtype=np.float32)[:3]) for c in colors] if colors else state.row_colors
        state.use_row_colors = True
        copies = max(1, int(copies))
        state.display_copies = np.array([copies, copies], dtype=np.int32)
        state.scanner_preview_grid_enabled = False

        # rebuild_mesh=False: the very next line rebuilds and uploads the same
        # mesh, with the placement flag this pass actually wants.
        state.rebuild_spline_from_params(rebuild_mesh=False)
        state.rebuild_spline_mesh(preserve_model_placement=False)
        # Control-point markers belong to the editing viewport, not to captured
        # fabric. rebuild_spline_mesh uploads them to whichever renderer it is
        # handed, and the period crop samples the middle of the fabric, which is
        # exactly where they sit -- so they would be stamped into every repeat.
        renderer.set_ctrl_pts([])

        for az in camera_az_degs:
            results[float(az)] = _scan_capture_one_azimuth(
                state, renderer, repeat_cols, repeat_rows, target_w,
                float(az), camera_el_deg, zoom, frame,
            )
        return results
    finally:
        state.renderer = prev_renderer
        if not _scan_batch_active(state):
            _scan_restore_state(state, snap)
            state.rebuild_spline_mesh(preserve_model_placement=True)


def _scan_render_tiled_pattern_image(state, renderer, bitmap, loop_heights, colors, repeat_cols, repeat_rows, copies=1, target_w=480, camera_az_deg=0.0, camera_el_deg=0.0, zoom=1.0, frame=None):
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
    (>1 = closer/more zoomed in).

    `frame` overrides the automatic framing with one measured elsewhere. Auto-
    framing sizes the image from the geometry actually present, and a scan
    pattern's random bitmap switches loops off, which physically shortens the
    fabric -- so cells with fewer active loops came out shorter than their
    neighbours and the grid composite went ragged. Passing a shared frame (see
    _scan_measure_pattern_frame) makes every cell the same pixel size, which is
    also the physically honest reading: every scanned square is the same piece
    of fabric, just with a different pattern printed on it."""
    return _scan_render_tiled_pattern_images(
        state, renderer, bitmap, loop_heights, colors, repeat_cols, repeat_rows,
        camera_az_degs=(float(camera_az_deg),), copies=copies, target_w=target_w,
        camera_el_deg=camera_el_deg, zoom=zoom, frame=frame,
    ).get(float(camera_az_deg))


# A usable glued tile has to be wide enough to actually be fabric and to carry
# some structure. Both failure modes show up together at oblique azimuths: the
# crop collapses to a sliver and/or the result is a flat single colour.
#
# The width test is relative to the width that was asked for, which separates
# the cases far more cleanly than any absolute pixel count: a healthy tile came
# back at ~86% of target_w (276/320, 414/480) while a collapsed one was ~4%
# (12/320). An absolute floor would have to sit somewhere in between and would
# misjudge small models.
_MIN_TILE_WIDTH_FRACTION = 0.15


_MIN_TILE_SIDE_PX = 24


_MIN_TILE_STDDEV = 1.0


def _tile_is_usable(canvas, target_w, tile=None):
    """Same predicate as before; `tile` only makes the colour test cheaper.

    The glued canvas is exactly cols x rows edge-to-edge copies of `tile` and
    nothing else, so the two have the same standard deviation. Passing the tile
    measures ~136x105 px instead of ~8160x6720 -- 831 ms of float32 .std() per
    cell, more than the render it was checking. The size test still reads the
    canvas dimensions, so what the function decides is unchanged.
    """
    if canvas is None:
        return False
    width, height = canvas.size
    if width < max(_MIN_TILE_SIDE_PX, _MIN_TILE_WIDTH_FRACTION * float(target_w)):
        return False
    if height < _MIN_TILE_SIDE_PX:
        return False
    measured = canvas if tile is None else tile
    return float(np.asarray(measured.convert("RGB"), dtype=np.float32).std()) >= _MIN_TILE_STDDEV


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
    """Scan Mode's pattern curves, normalized by fabric_scanner's own helper.

    This was a second copy of that function; the fill constant it hardcoded was
    already fabric_scanner.SCAN_PATTERN_FILL to the digit.
    """
    import fabric_scanner as scanner

    return scanner.normalize_model_curves(curves, fallback=[])


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
    _refresh_embedded_scanner_display_colors(state)


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
    except Exception:
        pass


def _refresh_embedded_scanner_display_colors(state):
    embedded = state.get('embedded_scanner')
    if embedded is None or getattr(embedded, "plan", None) is None:
        return
    embedded.plan.display_batch_colors = _scanner_batch_colors_for_simulator(state)
    if hasattr(embedded, "_render_frame"):
        embedded._render_frame()


def _puzzle_target_copies(state):
    copies_x = int(np.clip(int(state.get('puzzle_copies_x', 5)), 1, 9))
    copies_y = int(np.clip(int(state.get('puzzle_copies_y', 5)), 1, 9))
    if copies_x % 2 == 0:
        copies_x += 1
    if copies_y % 2 == 0:
        copies_y += 1
    return copies_x, copies_y


def _puzzle_apply_geometry_copies(state, preserve=True):
    copies_x, copies_y = _puzzle_target_copies(state)
    state.display_copies = np.array([(copies_x - 1) // 2, (copies_y - 1) // 2], dtype=np.int32)
    state.scanner_preview_grid_enabled = False
    state.rebuild_spline_mesh(preserve_model_placement=preserve)


def _set_puzzle_capture_rect(state, rect):
    x, y, w, h = [float(v) for v in rect]
    x = float(np.clip(x, 0.0, 0.95))
    y = float(np.clip(y, 0.0, 0.95))
    w = float(np.clip(w, 0.05, 1.0 - x))
    h = float(np.clip(h, 0.05, 1.0 - y))
    state.puzzle_capture_rect = [x, y, w, h]


def _puzzle_capture_geometry_image(state, renderer):
    """Reads back the actual rendered 3D viewport (same pixels shown behind the
    'Puzzle capture area' frame) and overlays real spline control points projected
    with the same camera/model matrices used to render that frame."""
    vp_w = int(getattr(renderer, "vp_w", 0))
    vp_h = int(getattr(renderer, "vp_h", 0))
    if vp_w < 2 or vp_h < 2 or getattr(renderer, "color_tex", None) is None:
        return None, [], []

    raw = renderer.color_tex.read()
    full_image = Image.frombytes("RGBA", (vp_w, vp_h), raw).convert("RGB")
    full_image = full_image.transpose(Image.FLIP_TOP_BOTTOM)

    model_mat = state.current_model_matrix()
    mvp = (state.camera.mvp(vp_w, vp_h) @ model_mat).astype(np.float32)

    projected_points = []
    for verts, row_idx in _puzzle_tiled_control_points(state):
        pix = _puzzle_project_points(verts, mvp, vp_w, vp_h)
        for px, py in pix:
            projected_points.append({"x": int(round(float(px))), "y": int(round(float(py))), "row": int(row_idx)})

    landmarks = []
    if projected_points:
        xs = np.array([p["x"] for p in projected_points], dtype=np.float32)
        ys = np.array([p["y"] for p in projected_points], dtype=np.float32)
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        for x in (x_min, x_max):
            for y in np.linspace(y_min, y_max, 9):
                landmarks.append({"x": int(round(x)), "y": int(round(float(y))), "kind": "x-edge"})
        for y in (y_min, y_max):
            for x in np.linspace(x_min, x_max, 9):
                landmarks.append({"x": int(round(float(x))), "y": int(round(y)), "kind": "y-edge"})

    rx, ry, rw, rh = _puzzle_capture_rect(state)
    crop_box = (
        int(round(rx * vp_w)),
        int(round(ry * vp_h)),
        int(round((rx + rw) * vp_w)),
        int(round((ry + rh) * vp_h)),
    )
    crop_box = (
        int(np.clip(crop_box[0], 0, vp_w - 1)),
        int(np.clip(crop_box[1], 0, vp_h - 1)),
        int(np.clip(crop_box[2], crop_box[0] + 1, vp_w)),
        int(np.clip(crop_box[3], crop_box[1] + 1, vp_h)),
    )
    cropped = full_image.crop(crop_box)
    x0, y0, x1, y1 = crop_box

    def adjust_points(items):
        adjusted = []
        for item in items:
            px = int(item.get("x", -1))
            py = int(item.get("y", -1))
            if x0 <= px < x1 and y0 <= py < y1:
                out = dict(item)
                out["x"] = px - x0
                out["y"] = py - y0
                adjusted.append(out)
        return adjusted

    return cropped, adjust_points(projected_points), adjust_points(landmarks)


def _puzzle_detect_colors(image, count=6):
    if image is None:
        return []
    count = max(1, int(count))
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).reshape(-1, 3)
    bg = np.array([18, 23, 31], dtype=np.int16)
    mask = np.linalg.norm(arr.astype(np.int16) - bg[None, :], axis=1) > 24.0
    pixels = arr[mask]
    if pixels.size == 0:
        return []
    quantized = (pixels // 16) * 16
    unique, counts = np.unique(quantized, axis=0, return_counts=True)
    order = np.argsort(-counts)[:count]
    return [
        {"rgb": [int(v) for v in unique[i].tolist()], "count": int(counts[i])}
        for i in order
    ]


def _upload_puzzle_texture(state, renderer, slot):
    """Cache one of Puzzle Mode's preview images as a GL texture.

    `slot` names the pair of state keys -- 'capture' or 'glued'. These were two
    functions identical but for that word. The cache key is the image's identity
    and size, so a rebuilt image uploads and an unchanged one does not.
    """
    image = state.get(f'puzzle_{slot}_image', None)
    if image is None:
        return None
    texture = state.get(f'puzzle_{slot}_texture', None)
    key = (id(image), image.size)
    if texture is not None and state.get(f'puzzle_{slot}_texture_key') == key:
        return texture
    if texture is not None:
        try:
            texture.release()
        except Exception:
            pass
    texture = pil_to_texture(renderer.ctx, image)
    state[f'puzzle_{slot}_texture'] = texture
    state[f'puzzle_{slot}_texture_key'] = key
    return texture


# Scan tiles survive between sessions on disk. Entering Scan Mode re-renders
# every cell from scratch otherwise -- ~4 s at a 3x4 grid, paid on every launch
# even when nothing about the pattern changed.
_TILE_CACHE_DIR_NAME = "tile_cache"


_TILE_CACHE_VERSION = 2


_TILE_CACHE_KEEP = 3


def _scanner_tile_cache_digest(state):
    """Identity of a persisted set of scan tiles, or None if it cannot be formed.

    Deliberately stricter than the in-memory cache key. A tile held in memory
    dies with the process, so keying it loosely only risks a stale preview for
    one session; a tile on disk outlives the settings that produced it, and
    serving one rendered under different lighting, material or template settings
    would put wrong pixels into a scan. So this also covers the material
    uniforms and the scanner template the geometry is built from, and carries a
    version so a change to the render pipeline invalidates everything written by
    an older build.
    """
    try:
        template = state._scanner_template()
        uniforms = _scanner_material_uniforms(state)
        payload = {
            "version": _TILE_CACHE_VERSION,
            "layout": json.dumps(_scanner_tiled_layout_cache_key(state), default=str, sort_keys=True),
            "uniforms": json.dumps(
                {str(k): (round(float(v), 6) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v))
                 for k, v in uniforms.items()},
                sort_keys=True,
            ),
            "spacing": [round(float(v), 6) for v in _scanner_repeat_spacing(state)],
            "batch_texture": [int(v) for v in _scanner_batch_texture_size(state)],
            "params": np.asarray(template.get("params"), dtype=np.float32).round(6).tolist(),
            "bitmap": np.asarray(template.get("bitmap"), dtype=np.float32).tolist(),
            "loop_heights": np.asarray(
                template.get("loop_heights", np.empty((0, 0))), dtype=np.float32).round(6).tolist(),
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        return None
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _scanner_tile_cache_root(state):
    return Path(state.project_root) / "scanner_data" / _TILE_CACHE_DIR_NAME


def _scanner_load_cached_tiles(state, digest, rows, cols):
    """Returns (preview, per_cell) from disk, or None to render fresh."""
    folder = _scanner_tile_cache_root(state) / digest
    manifest_path = folder / "manifest.json"
    try:
        if not manifest_path.exists():
            return None
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if int(manifest.get("version", -1)) != _TILE_CACHE_VERSION:
            return None
        if manifest.get("digest") != digest:
            return None
        cells = manifest.get("cells", [])
        if len(cells) != rows * cols:
            return None
        per_cell = []
        for index, entry in enumerate(cells):
            tile_path = folder / f"cell_{index:03d}.png"
            if not tile_path.exists():
                return None
            with Image.open(tile_path) as handle:
                tile = handle.convert("RGB")
            per_cell.append(TiledFabricTexture(tile, int(entry["cols"]), int(entry["rows"])))
        preview_path = folder / "preview.png"
        if not preview_path.exists():
            return None
        with Image.open(preview_path) as handle:
            preview = handle.convert("RGB")
        folder.touch(exist_ok=True)
        return preview, per_cell
    except Exception:
        # A damaged or half-written cache must never break Scan Mode; fall back
        # to rendering, which is what happened before this cache existed.
        return None


def _scanner_store_cached_tiles(state, digest, preview, per_cell):
    if not digest or preview is None or not per_cell:
        return
    # Only a full set of real tiles is worth keeping. A failed cell falls back to
    # a flat placeholder image, and persisting that would hand the same
    # placeholder back on every future launch.
    if any(not isinstance(cell, TiledFabricTexture) for cell in per_cell):
        return
    root = _scanner_tile_cache_root(state)
    folder = root / digest
    try:
        folder.mkdir(parents=True, exist_ok=True)
        entries = []
        for index, cell in enumerate(per_cell):
            cell.tile.save(folder / f"cell_{index:03d}.png")
            entries.append({"cols": int(cell.cols), "rows": int(cell.rows)})
        preview.save(folder / "preview.png")
        with (folder / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump({"version": _TILE_CACHE_VERSION, "digest": digest,
                       "cells": entries}, handle, indent=2)
        # Keep a few recent settings' worth and drop the rest.
        folders = sorted(
            (p for p in root.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        for stale in folders[_TILE_CACHE_KEEP:]:
            shutil.rmtree(stale, ignore_errors=True)
    except Exception:
        pass


def _scanner_tiled_layout_cache_key(state):
    rows = max(1, int(state.scanner_rows))
    cols = max(1, int(state.scanner_cols))
    repeat_rows, repeat_cols = _scanner_pattern_repeats(state)
    pattern_rows, pattern_cols = _scanner_pattern_dimensions(state)
    lighting_settings = _scanner_lighting_settings(state)
    lighting_key = tuple((key, round(float(lighting_settings[key]), 4)) for key in sorted(lighting_settings))
    return (
        rows,
        cols,
        repeat_rows,
        repeat_cols,
        pattern_rows,
        pattern_cols,
        round(float(state.get('scanner_pattern_density', 0.62)), 4),
        int(state.get('scanner_random_seed', 1)),
        json.dumps(state.get('scanner_color_variants', []), sort_keys=True),
        lighting_key,
    )


def _scanner_generate_tiled_layout(state, renderer):
    """Builds Scan Mode's per-pattern fabric imagery automatically, using the same
    real-3D-render + exact-period capture/glue pipeline validated in Puzzle Mode
    (duplicate the real model -> auto-frame -> capture -> exact-period crop -> glue
    by Image Repeat X/Y), once per grid cell. Cached like the old preview was, so
    this only re-renders when a relevant setting actually changes."""
    try:
        # Normalizes scanner_color_variants (e.g. RGB -> RGBA) as a side effect the
        # first time it runs each session -- must happen before computing the cache
        # key, or the key would change out from under an already-stored cache entry.
        cell_sets = _scanner_shared_cell_color_sets(state)
        cache_key = _scanner_tiled_layout_cache_key(state)
        cached_key = state.__dict__.get("_scanner_tiled_layout_key")
        if cached_key == cache_key:
            cached = state.__dict__.get("_scanner_tiled_layout_result")
            if cached is not None:
                full_image, per_cell = cached
                return full_image.copy(), list(per_cell)

        rows = max(1, int(state.scanner_rows))
        cols = max(1, int(state.scanner_cols))
        repeat_rows, repeat_cols = _scanner_pattern_repeats(state)
    except Exception:
        return None, []

    # Tiles written by an earlier session with identical settings, if any. The
    # digest covers every input the render depends on, so a hit is the same
    # imagery this function would produce.
    digest = _scanner_tile_cache_digest(state)
    restored = _scanner_load_cached_tiles(state, digest, rows, cols) if digest else None
    if restored is not None:
        full_image, per_cell_images = restored
        object.__setattr__(state, "_scanner_tiled_layout_key", cache_key)
        object.__setattr__(state, "_scanner_tiled_layout_result", (full_image.copy(), list(per_cell_images)))
        return full_image, per_cell_images

    # Measure the framing once, from a pattern with every loop active, and reuse
    # it for all cells. Auto-framing per cell sized each tile to the loops that
    # cell's random bitmap happened to switch on, so cells came out at different
    # heights and the composite below tiled them into a ragged staircase with
    # gaps. Every scanned square is the same piece of fabric, so they belong in
    # identically sized frames.
    # One borrow of the live model for the whole batch: the framing probe and
    # every cell render against it, and the user's own model is put back once at
    # the end instead of after each cell.
    per_cell_images = []
    with _scan_batch(state):
        shared_frame = None
        try:
            probe_bitmap = state._scanner_random_bitmap(0)
            shared_frame = _scan_measure_pattern_frame(state, renderer, probe_bitmap.shape)
        except Exception:
            shared_frame = None

        for cell_index in range(rows * cols):
            try:
                bitmap = state._scanner_random_bitmap(cell_index)
                loop_heights = state._scanner_loop_heights_for_bitmap(bitmap)
                colors = cell_sets[cell_index % len(cell_sets)] if cell_sets else None
                tile_image = _scan_render_tiled_pattern_image(
                    state, renderer, bitmap, loop_heights, colors, repeat_cols, repeat_rows,
                    frame=shared_frame,
                )
            except Exception:
                tile_image = None
            if tile_image is None:
                tile_image = Image.new("RGB", (64, 64), (18, 23, 31))
            per_cell_images.append(tile_image)

    cell_w = max((img.size[0] for img in per_cell_images), default=64)
    cell_h = max((img.size[1] for img in per_cell_images), default=64)

    # This composite is only ever used as a preview texture capped at max_dim,
    # so it is assembled at that size directly. Building it at full cell
    # resolution first meant allocating and filling a 32640x20160 canvas (658 MP,
    # ~2 GB) and LANCZOS-downscaling all of it to 1400 px -- 3.5 s of resize plus
    # 1.2 s of allocation to produce a 1.2 MP image. Scaling each cell into its
    # slot is the same filter over the same pixels, without the intermediate.
    max_dim = 1400
    scale = min(
        1.0,
        max_dim / max(float(cols * cell_w), 1.0),
        max_dim / max(float(rows * cell_h), 1.0),
    )
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)

    def _scaled(value):
        return max(1, int(round(value * scale))) if scale < 1.0 else max(1, int(value))

    def _cell_slot_image(source):
        """The cell at slot size, whether it is an image or a lazy tiled texture."""
        target = (_scaled(source.size[0]), _scaled(source.size[1]))
        rasterize = getattr(source, "rasterize", None)
        if rasterize is not None:
            # Stamping straight to slot size is not the same picture: at 64
            # repeats across 288 px each tile lands on ~4.5 px, so rounding it
            # to whole pixels shifts every repeat slightly, and each one gets
            # filtered in isolation instead of across its seams. Stamp to a
            # bounded intermediate first and let the final LANCZOS do the
            # cross-boundary filtering, exactly as downscaling the full glued
            # image used to.
            intermediate = int(np.clip(max(target) * 8, 256, 3072))
            image = rasterize(intermediate)
            return image if image.size == target else image.resize(target, resample)
        return source.resize(target, resample) if scale < 1.0 else source

    slot_w, slot_h = _scaled(cell_w), _scaled(cell_h)
    full_image = Image.new("RGB", (max(1, cols * slot_w), max(1, rows * slot_h)), (18, 23, 31))
    # The 1 px border used to be drawn on the full-resolution canvas and then
    # downscaled by `scale`, so it only ever contributed about that fraction of
    # a pixel's colour -- a hairline, not a line. Drawing it into an overlay at
    # the matching alpha keeps the preview looking the way it does today; a
    # solid 1 px line here would be roughly 23x more prominent than before.
    border_overlay = Image.new("RGBA", full_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(border_overlay)
    border = (38, 124, 137, max(1, int(round(255 * scale))) if scale < 1.0 else 255)
    for row in range(rows):
        for col in range(cols):
            cell_index = row * cols + col
            cell_img = _cell_slot_image(per_cell_images[cell_index])
            # Any cell that came back a different size (a failed render falling
            # back to the placeholder) is centred in its slot rather than pasted
            # at the corner, so it cannot masquerade as fabric that stops halfway.
            full_image.paste(
                cell_img,
                (col * slot_w + (slot_w - cell_img.size[0]) // 2,
                 row * slot_h + (slot_h - cell_img.size[1]) // 2),
            )
            # Border follows the cell slot, not the pasted image, so the grid
            # reads as a regular grid even if one cell had to fall back.
            draw.rectangle(
                [col * slot_w, row * slot_h,
                 col * slot_w + slot_w - 1, row * slot_h + slot_h - 1],
                outline=border,
                width=1,
            )
    full_image = Image.alpha_composite(full_image.convert("RGBA"), border_overlay).convert("RGB")

    object.__setattr__(state, "_scanner_tiled_layout_key", cache_key)
    object.__setattr__(state, "_scanner_tiled_layout_result", (full_image.copy(), list(per_cell_images)))
    # Written only on a fresh render, so what lands on disk always matches the
    # settings hashed into `digest`.
    _scanner_store_cached_tiles(state, digest, full_image, per_cell_images)
    return full_image, per_cell_images
