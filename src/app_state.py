import functools
import logging
import os
import json
import threading
import time
import numpy as np

import crochet_core
import paths
from persistence import json_ready, state_value, write_json

from knitting_core import (
    anchor_bitmap, compute_knitting_faces, row_base_pitch,
    build_parametric_control_rows, build_spline_mesh, build_surface_fiber_meshes
)

# %% APP STATE ─────────────────────────────────────────────────────────────────
# %% APP STATE ─────────────────────────────────────────────────────────────────

# Appended to a status message when a pattern had to be repaired on the way in.
# Short because status_msg is a single unwrapped line with no timeout.
ANCHOR_NOTE = 'bottom row filled in - a stitch there had nothing to hang from'


def _with_anchor_note(message, anchored_columns):
    """Append the repair note only when a repair actually happened.

    Keeps a clean undo or load reading exactly as it always did, rather than
    mentioning a rule the user never ran into.
    """
    if anchored_columns is None or len(anchored_columns) == 0:
        return message
    return f'{message}; {ANCHOR_NOTE}'


@functools.lru_cache(maxsize=4096)
def _seeded_pattern(rows: int, cols: int, density: float, seed: int) -> np.ndarray:
    """Random scan bitmap with at least one active cell; memoised because Scan Mode reads every cell each frame."""
    rng = np.random.default_rng(seed)
    bitmap = (rng.random((rows, cols)) < density).astype(np.float32)
    if not np.any(bitmap > 0.5):
        bitmap[rng.integers(0, rows), rng.integers(0, cols)] = 1.0
    return bitmap


# Inset of a scan sample inside its grid tile, in yarn radii per side. The tile
# bounds and the grid spacing must agree on this, or the samples overlap.
SCANNER_TILE_INSET_X = 0.10
SCANNER_TILE_INSET_Y = 0.20

class AppState:
    # How tall a scan-pattern loop is where the cell's bitmap is off, as a
    # fraction of the active height. Not zero: a zero-height loop is no stitch
    # at all, which breaks the fabric into disconnected motifs once it is tiled.
    # Low enough that cells still look meaningfully different from one another.
    SCANNER_INACTIVE_LOOP_HEIGHT_SCALE = 0.45

    @staticmethod
    def _clone(v):
        if isinstance(v, np.ndarray):
            return v.copy()
        if isinstance(v, list):
            return [AppState._clone(x) for x in v]
        if isinstance(v, dict):
            return {k: AppState._clone(x) for k, x in v.items()}
        return v

    def __init__(self, camera, renderer, orbit_camera=None, orbit_renderer=None):
        self.camera = camera
        self.renderer = renderer
        # Optional second view. rebuild_spline_mesh uploads to state.renderer
        # only, so the orbit renderer has to be fed explicitly there -- without
        # that it draws an empty scene, which is the same failure mode as a
        # scan capture rendered through a renderer that was never given meshes.
        self.orbit_camera = orbit_camera
        self.orbit_renderer = orbit_renderer

        # Project root and configurations. Locations come from paths.py so this
        # module does not carry its own idea of where the tree is laid out.
        project_root = str(paths.PROJECT_ROOT)
        super().__setattr__('project_root', project_root)
        super().__setattr__('resolve_project_path', lambda p: str(paths.resolve(p)))

        with open(paths.CONFIG_JSON, "r") as f:
            config_data = json.load(f)
        super().__setattr__('config', config_data)
        super().__setattr__('_scanner_template_cache', None)
        super().__setattr__('_cached_canonical_tile_bounds', None)
        super().__setattr__('_active_swatch_render_offset', np.zeros(3, dtype=np.float32))
        super().__setattr__('mesh_epoch', 0)
        super().__setattr__('last_prepared_meshes', None)
        super().__setattr__('_swatch_tile_cache', {})
        super().__setattr__('_swatch_tiles_epoch', 0)

        pidx = {p["name"]: i for i, p in enumerate(config_data["knit_parameters"]["parameters"])}
        super().__setattr__('_pidx', pidx)

        lh_params = sorted(
            [p["name"] for p in config_data["knit_parameters"]["parameters"] if p["name"].startswith("loop_height_")],
            key=lambda name: int(name.split("_")[-1])
        )
        lh_idx = tuple(pidx[name] for name in lh_params)
        super().__setattr__('_lh_idx', lh_idx)

        # Load schema config
        schema_path = str(paths.STATE_SCHEMA_JSON)
        with open(schema_path, 'r') as handle:
            schema = json.load(handle)

        app_config = schema['app_config']

        # 1. Dynamically load config attributes driven by a cast mapping
        _SCHEMA_CASTS = {
            'workflow_stages': lambda v: tuple((item['title'], item['subtitle']) for item in v),
            'texture_control_groups': tuple,
            'texture_preset_buttons': tuple,
            'texture_presets': dict,
            'texture_param_keys': tuple,
            'material_uniform_aliases': dict,
            'saved_state_keys': tuple,
            'camera_attributes': tuple,
        }
        for attr, cast_fn in _SCHEMA_CASTS.items():
            super().__setattr__(attr, cast_fn(app_config[attr]))

        # Auto-coerce flat numeric lists to np.float32 arrays, leaving other structures intact
        def _coerce(val):
            if isinstance(val, list) and val and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in val):
                return np.array(val, dtype=np.float32)
            return val

        # 2. Extract and coerce static defaults from schema
        state_defaults = {}
        for section, keys_dict in schema['state_defaults'].items():
            for key, val in keys_dict.items():
                state_defaults[key] = AppState._clone(_coerce(val))

        # 3. Add static defaults that need special empty array dimensions
        state_defaults['flat_pts'] = np.empty((0, 3), dtype=np.float32)

        # 4. Integrate texture presets defaults (clear + soft_yarn chain)
        preset_clear = self.texture_presets['clear']
        preset_soft = self.texture_presets['soft_yarn']
        for k, v in preset_clear.items():
            if k != 'render_texture_color':
                state_defaults[k] = AppState._clone(_coerce(v))
        for k, v in preset_soft.items():
            state_defaults[k] = AppState._clone(_coerce(v))

        initial_pattern_rows = 4
        initial_pattern_cols = int(config_data['knit_parameters']['bitmap_loops'])
        try:
            with open(paths.INITIAL_PARAMS_JSON, 'r') as f:
                initial_data = json.load(f)
            initial_bitmap = np.asarray(initial_data.get('bitmap', []), dtype=np.float32)
            if initial_bitmap.ndim == 2 and initial_bitmap.size:
                initial_pattern_rows, initial_pattern_cols = [int(v) for v in initial_bitmap.shape]
        except Exception as exc:
            logging.warning("app_state.__init__: %s", exc)

        # 5. Overlay computed defaults depending on config.json or runtime pathing
        computed_defaults = {
            'save_path': str(paths.PARAMS_JSON),
            'load_path': str(paths.PARAMS_JSON),
            'params': [p['initial'] for p in config_data['knit_parameters']['parameters']],
            'bitmap': np.ones((3, config_data['knit_parameters']['bitmap_loops']), dtype=np.float32),
            'bitmap_size': np.array([3, config_data['knit_parameters']['bitmap_loops']], dtype=np.int32),
            'display_copies': np.array([0, 0], dtype=np.int32),
            'crochet_params': crochet_core.CrochetParams().as_dict(),
            'crochet_stitches': np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32),
            'scanner_pattern_rows': initial_pattern_rows,
            'scanner_pattern_cols': initial_pattern_cols,
            'loop_heights': np.empty((0, 0), dtype=np.float32),
            'loop_height_overrides': np.zeros((0, 0), dtype=bool),
            'mesh_center': np.zeros(3, dtype=np.float32),
            'row_colors': [
                list(config_data['knit_parameters']['yarn_colors'][i % len(config_data['knit_parameters']['yarn_colors'])])
                for i in range(3)
            ],
            'row_visible': np.ones(3, dtype=bool),
            'period_offset_x': np.array([float(config_data['knit_parameters']['bitmap_loops']), 0.0, 0.0], dtype=np.float32),
            'period_offset_y': np.array(
                [0.0, 3.0 * (float(config_data['knit_parameters']['parameters'][pidx['dy']]['initial']) if 'dy' in pidx else 1.0), 0.0],
                dtype=np.float32,
            ),
            'sim_L0': np.array([]),
        }
        for k, v in computed_defaults.items():
            state_defaults[k] = AppState._clone(_coerce(v))

        # 6. Direct assignments for types that _coerce would get wrong
        state_defaults['_row_starts'] = [0]
        state_defaults['swatch_cells'] = {}
        state_defaults['swatch_session_id'] = ""
        state_defaults['swatch_splat_path'] = ""
        state_defaults['swatch_fitted_model'] = None

        super().__setattr__('_state_defaults', state_defaults)
        self._data = {k: self._clone(v) for k, v in state_defaults.items()}
        super().__setattr__('scanner_process', None)
        super().__setattr__('scanner_status', 'Scanner idle')
        super().__setattr__('scanner_started_at', 0.0)
        # Guards every cross-thread read/write of the simulation's view of the
        # model (ctrl_rows, flat_pts, the cached Jacobian and the energy
        # readouts). Not part of _data: it must never be cloned or snapshotted.
        super().__setattr__('sim_lock', threading.Lock())
        super().__setattr__('J_cached', None)

    # ── DICTIONARY INTERFACE ──────────────────────────────────────────────────

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)

    # ── ATTRIBUTE ROUTING ─────────────────────────────────────────────────────

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"'AppState' object has no attribute '{name}'") from None

    # Live objects rather than state: they must bypass _data so they are never
    # cloned, snapshotted, undone or written to params.json.
    _DIRECT_ATTRS = (
        'camera', 'renderer',
        'orbit_camera', 'orbit_renderer', '_data',
    )

    def __setattr__(self, name, value):
        if name in self._DIRECT_ATTRS or name in self.__dict__:
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    # ── GET UNIFORM MAPS ──────────────────────────────────────────────────────

    def get_material_uniforms(self):
        """Returns standard material properties as a dict mapped to shader uniform names."""
        uniforms = {k: self._data[k] for k in self.texture_param_keys}
        uniforms.update({
            uniform_name: self._data[state_key]
            for state_key, uniform_name in self.material_uniform_aliases.items()
        })
        return uniforms

    # ── MESH GENERATION HELPERS ───────────────────────────────────────────────

    def _sync_row_colors(self, n_rows):
        palette = self.config['knit_parameters']['yarn_colors']
        self.row_colors = self.row_colors[:n_rows] + [
            list(palette[i % len(palette)]) for i in range(len(self.row_colors), n_rows)
        ]

    def _sync_row_visibility(self, n_rows):
        old = np.asarray(self.row_visible, dtype=bool)
        new_vis = np.ones(n_rows, dtype=bool)
        new_vis[:min(old.shape[0], n_rows)] = old[:min(old.shape[0], n_rows)]
        self.row_visible = new_vis

    # ── SCANNER TEMPLATE AND RANDOM PATTERN MESHES ───────────────────────────
    # Scan Mode uses initial_params.json as the stable fabric template. These
    # helpers randomize only the bitmap, then build meshes for preview/display.

    def _scanner_template(self):
        """Read initial_params.json once; the app never writes it, so the cache stays valid."""
        cached = self._scanner_template_cache
        if cached is not None:
            return cached
        path = str(paths.INITIAL_PARAMS_JSON)

        params = np.array(
            [p['initial'] for p in self.config['knit_parameters']['parameters']],
            dtype=np.float32,
        )
        bitmap = np.ones(
            (max(1, int(self.get('scanner_pattern_rows', 4))), max(1, int(self.get('scanner_pattern_cols', self.config['knit_parameters']['bitmap_loops'])))),
            dtype=np.float32,
        )
        loop_heights = np.empty((0, 0), dtype=np.float32)
        gui_state = {}
        ctrl_rows = []
        radius_rows = []
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            p_dict = data.get('params', {})
            for i, pd in enumerate(self.config['knit_parameters']['parameters']):
                if pd["name"] in p_dict:
                    lo, hi = pd["range"]
                    params[i] = float(np.clip(p_dict[pd["name"]], lo, hi))
            loaded_bitmap = np.asarray(data.get('bitmap', bitmap), dtype=np.float32)
            if loaded_bitmap.ndim == 2 and loaded_bitmap.size:
                bitmap = loaded_bitmap
            gui_state = data.get('gui_state', {}) if isinstance(data.get('gui_state', {}), dict) else {}
            loaded_heights = np.asarray(gui_state.get('loop_heights', data.get('loop_heights', loop_heights)), dtype=np.float32)
            if loaded_heights.ndim == 2 and loaded_heights.size:
                loop_heights = loaded_heights
            # The tuned fabric itself, not just the parameters it was tuned
            # from. Rebuilding from the parameters alone does not reproduce it:
            # the saved rows sit on a 1.106 pitch with near-uniform loops, while
            # the parameters say dy 0.594 with loop heights spanning 4.7:1, so
            # the rebuilt rows tower over their own pitch and pass through the
            # rows above. Scan Mode showed that instead of the real model.
            ctrl_rows = [
                np.asarray(row, dtype=np.float32)
                for row in data.get('spline_control_rows', [])
                if len(row) > 1
            ]
            radius_rows = [
                np.asarray(row, dtype=np.float32)
                for row in data.get('spline_radius_rows', [])
            ]
        except Exception as exc:
            logging.warning("app_state._scanner_template: %s", exc)

        template = {
            'params': params,
            'bitmap': bitmap,
            'loop_heights': loop_heights,
            'gui_state': gui_state,
            'ctrl_rows': ctrl_rows,
            'radius_rows': radius_rows,
        }
        super().__setattr__('_scanner_template_cache', template)
        super().__setattr__('_cached_canonical_tile_bounds', None)
        return template

    def _scanner_template_params(self):
        return np.asarray(self._scanner_template()['params'], dtype=np.float32).copy()

    def _canonical_tile_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        cached = getattr(self, '_cached_canonical_tile_bounds', None)
        if cached is not None:
            return cached
        template = self._scanner_template()
        params = np.asarray(template['params'], dtype=np.float32)
        bitmap = np.asarray(template.get('bitmap', np.ones((4, 4), dtype=np.float32)), dtype=np.float32)
        if bitmap.ndim != 2 or not bitmap.size:
            bitmap = np.ones(self._scanner_pattern_size(), dtype=np.float32)
        ctrl_rows = [np.asarray(row, dtype=np.float32) for row in (template.get('ctrl_rows') or []) if len(row) > 1]
        if not ctrl_rows:
            ctrl_rows = build_parametric_control_rows(
                params, np.ones_like(bitmap), self._pidx, self._lh_idx, self.samples_per_loop
            )
        stitch_width = float(params[self._pidx['stitch_width']])
        period_offset_x = np.array([float(bitmap.shape[1]) * stitch_width, 0.0, 0.0], dtype=np.float32)
        radius = max(float(params[self._pidx['radius']]), 1e-6)
        radius_profiles = [np.full(len(row), radius, dtype=np.float32) for row in ctrl_rows]
        vl = build_spline_mesh(
            ctrl_rows, params, self.config, self._pidx, period_offset_x, radius_ctrl_rows=radius_profiles
        )
        bounds = self._display_mesh_bounds(vl)
        if bounds is None:
            lo = np.array([-2.0, -2.0, -0.5], dtype=np.float32)
            hi = np.array([2.0, 2.0, 0.5], dtype=np.float32)
        else:
            min_v, max_v = bounds
            center = np.array([0.5 * (min_v[0] + max_v[0]), 0.5 * (min_v[1] + max_v[1]), 0.0], dtype=np.float32)
            lo = (min_v - center).astype(np.float32)
            hi = (max_v - center).astype(np.float32)
        result = (lo, hi)
        super().__setattr__('_cached_canonical_tile_bounds', result)
        return result

    def _unit_cell_periods(self) -> tuple[float, float, float]:
        lo, hi = self._canonical_tile_bounds()
        canon_params = self._scanner_template()['params']
        radius = max(float(canon_params[self._pidx['radius']]), 1e-6)
        cell_w = max(float(hi[0] - lo[0]) - 2.0 * SCANNER_TILE_INSET_X * radius, radius)
        cell_h = max(float(hi[1] - lo[1]) - 2.0 * SCANNER_TILE_INSET_Y * radius, radius)
        layout_pattern = str(self.get('scanner_layout_pattern', 'grid'))
        cell_row_shift = 0.5 * cell_w if layout_pattern in ('staggered', 'hex') else 0.0
        return cell_w, cell_h, cell_row_shift

    def unit_cell_span(self) -> float:
        lo, hi = self._canonical_tile_bounds()
        return float(max(hi[0] - lo[0], hi[1] - lo[1], 2.0))

    def swatch_repeat_counts(self) -> tuple[int, int]:
        rep_r = min(max(int(self.get('scanner_pattern_repeat_rows', 24)), 1), 128)
        rep_c = min(max(int(self.get('scanner_pattern_repeat_cols', 24)), 1), 128)
        return rep_r, rep_c

    def _swatch_grid_periods(self) -> tuple[float, float, float]:
        cell_w, cell_h, cell_shift = self._unit_cell_periods()
        rep_r, rep_c = self.swatch_repeat_counts()
        return float(cell_w * rep_c), float(cell_h * rep_r), float(cell_shift * rep_c)

    def swatch_slot_offsets(self) -> np.ndarray:
        """Slot centres in model units, shape (scanner_rows, scanner_cols, 3)."""
        rows = max(1, int(self.get('scanner_rows', 1)))
        cols = max(1, int(self.get('scanner_cols', 1)))
        x_period, y_period, row_shift = self._swatch_grid_periods()
        r = np.arange(rows, dtype=np.float64)[:, None]
        c = np.arange(cols, dtype=np.float64)[None, :]
        shift_x = np.where(r % 2 == 1, row_shift, 0.0) - (0.5 * row_shift if rows > 1 else 0.0)
        x = (c - 0.5 * (cols - 1)) * x_period + shift_x
        y = np.broadcast_to((r - 0.5 * (rows - 1)) * y_period, x.shape)
        return np.stack([x, y, np.zeros_like(x)], axis=-1).astype(np.float32)

    def swatch_slot_offset(self, row: int, col: int) -> np.ndarray:
        return self.swatch_slot_offsets()[int(row), int(col)]

    def all_swatch_slot_rects(self) -> list[tuple[float, float, float, float]]:
        slot_w, slot_h, _ = self._swatch_grid_periods()
        xy = self.swatch_slot_offsets()[..., :2].reshape(-1, 2).astype(np.float64)
        half = 0.47 * np.array([slot_w, slot_h])
        return [tuple(rect) for rect in np.hstack([xy - half, xy + half]).tolist()]

    def canonical_grid_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        tile_lo, tile_hi = self._canonical_tile_bounds()
        slot_w, slot_h, _ = self._swatch_grid_periods()
        xy = self.swatch_slot_offsets()[..., :2].reshape(-1, 2)
        half = 0.5 * np.array([slot_w, slot_h])
        lo = np.array([*(xy.min(axis=0) - half), tile_lo[2]], dtype=np.float32)
        hi = np.array([*(xy.max(axis=0) + half), tile_hi[2]], dtype=np.float32)
        return lo, hi

    def canonical_grid_span(self) -> float:
        lo, hi = self.canonical_grid_bounds()
        return float(max(hi[0] - lo[0], hi[1] - lo[1], 2.0))

    def _compute_active_swatch_render_offset(self, verts_list=None) -> np.ndarray:
        rows = max(1, int(self.get('scanner_rows', 1)))
        cols = max(1, int(self.get('scanner_cols', 1)))
        sel_r = int(np.clip(int(self.get('scanner_selected_row', 0)), 0, rows - 1))
        sel_c = int(np.clip(int(self.get('scanner_selected_col', 0)), 0, cols - 1))
        active_offset = self.swatch_slot_offset(sel_r, sel_c)
        ref_rows = self.get('param_ref_ctrl_rows') or self.ctrl_rows
        bounds = self._ctrl_rows_bounds(ref_rows)
        if bounds is None and verts_list is not None:
            bounds = self._display_mesh_bounds(verts_list)
        if bounds is not None:
            min_v, max_v = bounds
            local_xy_center = np.array([0.5 * (min_v[0] + max_v[0]), 0.5 * (min_v[1] + max_v[1]), 0.0], dtype=np.float32)
        else:
            local_xy_center = np.zeros(3, dtype=np.float32)
        offset = (active_offset - local_xy_center).astype(np.float32)
        self._active_swatch_render_offset = offset
        return offset

    def apply_scanner_template_base(self):
        """Install the scan template's own tuned fabric as the working model.

        Scan Mode used to rebuild its fabric from the template's parameters,
        which do not describe the fabric saved beside them -- so every capture
        showed geometry the user had never seen. This installs the saved rows
        instead, and sets param_ref_ctrl_rows to the parametric build those rows
        correspond to, so a following nudge_spline_from_params applies only the
        per-cell pattern change on top of the real fabric rather than replacing
        it.

        Returns False when the template carries no saved rows, or when they do
        not match the shape the current pattern needs; the caller then falls
        back to the parametric rebuild, which is the previous behaviour.
        """
        template = self._scanner_template()
        rows = template.get('ctrl_rows') or []
        if not rows:
            return False

        # The saved rows describe a fabric of the template's own size. At any
        # other pattern size they cannot carry it, so say so here rather than
        # installing them and leaving nudge_spline_from_params' shape guard to
        # discard them further down -- the caller's parametric fallback is then
        # a decision instead of an accident.
        pattern_rows, pattern_cols = self._scanner_pattern_size()
        template_bitmap = np.asarray(template.get('bitmap', np.empty((0, 0))), dtype=np.float32)
        if len(rows) != pattern_rows:
            return False
        if template_bitmap.ndim == 2 and template_bitmap.size and int(template_bitmap.shape[1]) != pattern_cols:
            return False

        reference = build_parametric_control_rows(
            np.asarray(template['params'], dtype=np.float32),
            np.asarray(template['bitmap'], dtype=np.float32),
            self._pidx,
            self._lh_idx,
            self.samples_per_loop,
            loop_heights=(
                np.asarray(template['loop_heights'], dtype=np.float32)
                if np.asarray(template['loop_heights']).size else None
            ),
        )
        if len(reference) != len(rows) or any(r.shape != c.shape for r, c in zip(reference, rows)):
            return False

        self.ctrl_rows = [np.asarray(row, dtype=np.float32).copy() for row in rows]
        saved_radius = template.get('radius_rows') or []
        if len(saved_radius) == len(self.ctrl_rows):
            self.spline_radius_rows = [np.asarray(row, dtype=np.float32).copy() for row in saved_radius]
        else:
            self.spline_radius_rows = []
        self._ensure_spline_radius_rows()
        self.param_ref_ctrl_rows = [np.asarray(row, dtype=np.float32).copy() for row in reference]
        self.param_ref_radius = max(float(self.params[self._pidx['radius']]), 1e-6)
        self._rebuild_spline_points()
        return True

    def _scanner_default_loop_height_for_row(self, params, row_idx):
        if not self._lh_idx:
            return 0.0
        idx = self._lh_idx[min(int(row_idx), len(self._lh_idx) - 1)]
        return float(params[idx])

    # Beyond this the extra rows would all reuse the last loop_height parameter,
    # since that is how many the model defines.
    SCANNER_PATTERN_MAX_COLS = 8

    def _scanner_pattern_size(self):
        """The scan pattern grid: rows and columns, as the user set them.

        The single source of truth. Both the generator and the plan geometry
        used to take their size from the scan template's bitmap instead, falling
        back to these settings only when the template had none -- which never
        happens, so the size was welded to the template and neither the settings
        nor the model could move it.

        The defaults already come from the template (AppState.__init__ reads
        initial_params.json's bitmap shape), so honouring them here changes
        nothing until the user moves the sliders.
        """
        max_rows = max(2, int(self.config['knit_parameters']['bitmap_rows']))
        rows = int(self.get('scanner_pattern_rows', max(2, int(self.bitmap_size[0]))))
        cols = int(self.get('scanner_pattern_cols', max(2, int(self.bitmap_size[1]))))
        return min(max(rows, 2), max_rows), min(max(cols, 2), int(self.SCANNER_PATTERN_MAX_COLS))

    SWATCH_STATE_KEYS = (
        'bitmap',
        'bitmap_size',
        'params',
        'loop_heights',
        'loop_height_overrides',
        'ctrl_rows',
        'spline_radius_rows',
        'row_colors',
        'use_row_colors',
        'period_offset_x',
        'period_offset_y',
        'fabric_type',
        'crochet_params',
        'crochet_stitches',
        'swatch_session_id',
        'swatch_splat_path',
        'swatch_fitted_model',
    )

    def _swatch_key(self, row: int, col: int) -> str:
        return f"{int(row)},{int(col)}"

    def _cell_index_to_row_col(self, cell_index: int) -> tuple[int, int]:
        cols = max(1, int(self.get('scanner_cols', 1)))
        idx = max(0, int(cell_index))
        return idx // cols, idx % cols

    def save_active_swatch(self) -> None:
        cells = self._data.setdefault('swatch_cells', {})
        row = int(self.get('scanner_selected_row', 0))
        col = int(self.get('scanner_selected_col', 0))
        key = self._swatch_key(row, col)
        entry = {}
        for k in self.SWATCH_STATE_KEYS:
            if k in self._data:
                entry[k] = self._clone(self._data[k])
        cells[key] = entry

    def select_swatch(self, row: int, col: int, rebuild: bool = True) -> None:
        cells = self._data.setdefault('swatch_cells', {})
        cur_row = int(self.get('scanner_selected_row', 0))
        cur_col = int(self.get('scanner_selected_col', 0))
        if (cur_row, cur_col) != (int(row), int(col)) or self._swatch_key(cur_row, cur_col) in cells:
            self.save_active_swatch()
        self.scanner_selected_row = int(row)
        self.scanner_selected_col = int(col)
        key = self._swatch_key(row, col)
        self.swatch_session_id = ""
        self.swatch_splat_path = ""
        self.swatch_fitted_model = None
        if key in cells:
            entry = cells[key]
            for k, v in entry.items():
                self._data[k] = self._clone(v)
            self.param_ref_ctrl_rows = [r.copy() for r in self._fresh_rebuild_rows()]
            self._rebuild_spline_points()
        else:
            cols = max(1, int(self.get('scanner_cols', 1)))
            cell_index = int(row) * cols + int(col)
            pattern_bitmap = self._scanner_random_bitmap(cell_index)
            loop_heights = self._scanner_loop_heights_for_bitmap(pattern_bitmap)
            params = self._scanner_template_params()
            self.bitmap = pattern_bitmap.copy()
            self.bitmap_size = np.array(pattern_bitmap.shape, dtype=np.int32)
            self.params = [float(v) for v in params]
            self.loop_heights = loop_heights.copy()
            self.loop_height_overrides = np.ones_like(pattern_bitmap, dtype=bool)
            self._sync_row_colors(int(pattern_bitmap.shape[0]))
            self._sync_row_visibility(int(pattern_bitmap.shape[0]))
            self.rebuild_spline_from_params(rebuild_mesh=False)
            self.save_active_swatch()
        if rebuild:
            self.rebuild_spline_mesh(preserve_model_placement=True)

    def _scanner_random_bitmap(self, cell_index):
        r, c = self._cell_index_to_row_col(cell_index)
        cells = self.get('swatch_cells') or {}
        entry = cells.get(self._swatch_key(r, c))
        if isinstance(entry, dict) and 'bitmap' in entry:
            bm = np.asarray(entry['bitmap'], dtype=np.float32)
            if bm.ndim == 2 and bm.size:
                return bm.copy()
        pattern_rows, pattern_cols = self._scanner_pattern_size()
        density = min(max(float(self.get('scanner_pattern_density', 0.62)), 0.05), 0.95)
        seed = int(self.get('scanner_random_seed', 1)) + int(cell_index) * 9973
        return _seeded_pattern(pattern_rows, pattern_cols, density, seed).copy()

    def _scanner_loop_heights_for_bitmap(self, bitmap):
        bitmap = np.asarray(bitmap, dtype=np.float32)
        template = self._scanner_template()
        params = np.asarray(template['params'], dtype=np.float32)
        heights = np.zeros_like(bitmap, dtype=np.float32)
        source = np.asarray(template.get('loop_heights', np.empty((0, 0))), dtype=np.float32)
        for row_idx in range(bitmap.shape[0]):
            default_height = self._scanner_default_loop_height_for_row(params, row_idx)
            heights[row_idx, :] = default_height
        if source.ndim == 2 and source.size:
            keep_rows = min(bitmap.shape[0], source.shape[0])
            keep_cols = min(bitmap.shape[1], source.shape[1])
            heights[:keep_rows, :keep_cols] = source[:keep_rows, :keep_cols]
        # Cells the pattern switches off are shortened, not removed. Multiplying
        # by the bitmap collapsed them to zero height, which deleted the stitch
        # outright: at the usual density that emptied whole columns, so tiled
        # copies stopped touching and a 64x64 repeat read as a grid of separate
        # motifs instead of one piece of fabric. Keeping a shorter loop leaves
        # the yarn continuous across every repeat -- the way Puzzle Mode renders
        # the same model -- while still giving each scan cell a distinct pattern.
        active = np.asarray(bitmap > 0.5, dtype=np.float32)
        scale = active + (1.0 - active) * float(self.SCANNER_INACTIVE_LOOP_HEIGHT_SCALE)
        return heights * scale

    def build_swatch_unit_cell_data(self, row: int, col: int):
        from rendering import MeshRenderer
        scanner_rows = max(1, int(self.get('scanner_rows', 1)))
        scanner_cols = max(1, int(self.get('scanner_cols', 1)))
        sel_r = int(np.clip(int(self.get('scanner_selected_row', 0)), 0, scanner_rows - 1))
        sel_c = int(np.clip(int(self.get('scanner_selected_col', 0)), 0, scanner_cols - 1))
        r = int(np.clip(int(row), 0, scanner_rows - 1))
        c = int(np.clip(int(col), 0, scanner_cols - 1))
        cell_index = r * scanner_cols + c
        cells = self.get('swatch_cells') or {}
        entry = cells.get(self._swatch_key(r, c))

        if (r, c) == (sel_r, sel_c) and self.ctrl_rows:
            ctrl_rows = [np.asarray(row_pts, dtype=np.float32) for row_pts in self.ctrl_rows if len(row_pts) > 1]
            params = np.asarray(self.params, dtype=np.float32)
            period_offset_x = np.asarray(self.period_offset_x, dtype=np.float32)
            radius = max(float(params[self._pidx['radius']]), 1e-6)
            saved_radii = self.get('spline_radius_rows') or []
            if len(saved_radii) == len(ctrl_rows):
                radius_profiles = [np.asarray(rad, dtype=np.float32) for rad in saved_radii]
            else:
                radius_profiles = [np.full(len(row_pts), radius, dtype=np.float32) for row_pts in ctrl_rows]
            base_colors = list(self.row_colors or self.get('scanner_color_variants') or [self.single_model_color])
        elif isinstance(entry, dict) and entry.get('ctrl_rows'):
            ctrl_rows = [np.asarray(row_pts, dtype=np.float32) for row_pts in entry['ctrl_rows'] if len(row_pts) > 1]
            params = np.asarray(entry.get('params', self._scanner_template_params()), dtype=np.float32)
            bm = np.asarray(entry.get('bitmap', self._scanner_random_bitmap(cell_index)), dtype=np.float32)
            stitch_width = float(params[self._pidx['stitch_width']])
            period_offset_x = np.asarray(
                entry.get('period_offset_x', [float(bm.shape[1]) * stitch_width, 0.0, 0.0]),
                dtype=np.float32,
            )
            radius = max(float(params[self._pidx['radius']]), 1e-6)
            saved_radii = entry.get('spline_radius_rows') or []
            if len(saved_radii) == len(ctrl_rows):
                radius_profiles = [np.asarray(rad, dtype=np.float32) for rad in saved_radii]
            else:
                radius_profiles = [np.full(len(row_pts), radius, dtype=np.float32) for row_pts in ctrl_rows]
            base_colors = list(entry.get('row_colors') or self.get('scanner_color_variants') or self.row_colors or [self.single_model_color])
        else:
            pattern_bitmap = self._scanner_random_bitmap(cell_index)
            loop_heights = self._scanner_loop_heights_for_bitmap(pattern_bitmap)
            params = self._scanner_template_params()
            bitmap = np.ones_like(pattern_bitmap)
            ctrl_rows = build_parametric_control_rows(
                params,
                bitmap,
                self._pidx,
                self._lh_idx,
                self.samples_per_loop,
                loop_heights=loop_heights,
            )
            stitch_width = float(params[self._pidx['stitch_width']])
            period_offset_x = np.array([float(bitmap.shape[1]) * stitch_width, 0.0, 0.0], dtype=np.float32)
            radius = max(float(params[self._pidx['radius']]), 1e-6)
            radius_profiles = [np.full(len(row_pts), radius, dtype=np.float32) for row_pts in ctrl_rows]
            base_colors = list(self.get('scanner_color_variants') or self.row_colors or [self.single_model_color])

        seg = min(int(self.config['knit_parameters']['segments']), 12)
        loop_res = min(int(self.config['knit_parameters']['loop_res']), 12)
        tile_config = {
            **self.config,
            'knit_parameters': {
                **self.config['knit_parameters'],
                'segments': seg,
                'loop_res': loop_res,
            },
        }
        vl = build_spline_mesh(
            ctrl_rows,
            params,
            tile_config,
            self._pidx,
            period_offset_x,
            radius_ctrl_rows=radius_profiles,
        )
        bounds = self._display_mesh_bounds(vl)
        if bounds is not None:
            min_v, max_v = bounds
            cx = 0.5 * float(min_v[0] + max_v[0])
            cy = 0.5 * float(min_v[1] + max_v[1])
        else:
            cx, cy = 0.0, 0.0
        x_period = self._display_copy_x_period(vl, radius)
        pitch = row_base_pitch(ctrl_rows)
        if pitch is None:
            dy_idx = self._pidx.get('dy')
            pitch = abs(float(params[dy_idx])) if dy_idx is not None else 0.0
        if pitch and pitch > 1e-6:
            y_period = max(float(pitch * len(ctrl_rows)), radius)
        elif bounds is not None:
            y_period = max(float(bounds[1][1] - bounds[0][1]), radius)
        else:
            y_period = radius

        offsets = [(dx * x_period, dy * y_period, 0.0) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        prepared_meshes = MeshRenderer.prepare_meshes(vl, compute_knitting_faces(seg, vl), offsets)
        row_colors = [[float(x) for x in base_colors[row % len(base_colors)][:3]] for row in range(len(vl))]
        return prepared_meshes, row_colors * len(offsets), (float(cx), float(cy)), float(x_period), float(y_period)

    def build_display_meshes_precise(self, verts_list, faces_list, meta=None):
        if not verts_list:
            return [], [], []
        if meta is None:
            meta = [{'row': i, 'base_row': i} for i in range(len(verts_list))]
        radius = max(float(self.params[self._pidx['radius']]), 1e-6)
        seg = int(self.config['knit_parameters']['segments'])
        scanner_preview = bool(self.get('scanner_preview_grid_enabled', False))
        scanner_rows = max(1, int(self.get('scanner_rows', 1)))
        scanner_cols = max(1, int(self.get('scanner_cols', 1)))
        sel_r = int(np.clip(int(self.get('scanner_selected_row', 0)), 0, scanner_rows - 1))
        sel_c = int(np.clip(int(self.get('scanner_selected_col', 0)), 0, scanner_cols - 1))
        cell_index = sel_r * scanner_cols + sel_c

        x_period = self._display_copy_x_period(verts_list, radius)
        y_period = self._display_copy_y_period(verts_list, radius)
        if scanner_preview:
            x_tiles = [0]
            y_tiles = [0]
        else:
            x_tiles = list(range(-int(self.display_copies[0]), int(self.display_copies[0]) + 1))
            y_tiles = list(range(-int(self.display_copies[1]), int(self.display_copies[1]) + 1))
        active_render_offset = self._compute_active_swatch_render_offset(verts_list)

        display_vl, display_fl, display_meta = [], [], []
        for y_tile in y_tiles:
            y_translation = np.array([0.0, y_tile * y_period, 0.0], dtype=np.float32)
            for (verts, n_points), _faces, part_meta in zip(verts_list, faces_list, meta):
                rings = (np.asarray(verts, dtype=np.float32) + active_render_offset).reshape(int(n_points), seg, 3)
                row_direction = 1.0
                if int(n_points) > 1:
                    row_direction = float(rings[-1, :, 0].mean() - rings[0, :, 0].mean())
                ordered_tiles = x_tiles if row_direction >= 0.0 else list(reversed(x_tiles))
                stitched_rings = []
                for tile_i, x_tile in enumerate(ordered_tiles):
                    translated = rings + np.array([x_tile * x_period, 0.0, 0.0], dtype=np.float32)
                    if tile_i > 0 and len(translated) > 1:
                        translated = translated[1:]
                    stitched_rings.append(translated)
                stitched = np.concatenate(stitched_rings, axis=0) + y_translation
                stitched_n_points = int(stitched.shape[0])
                stitched_verts = stitched.reshape(-1, 3)
                display_vl.append((stitched_verts, stitched_n_points))
                display_fl.extend(compute_knitting_faces(seg, [(stitched_verts, stitched_n_points)]))
                copied_meta = dict(part_meta)
                base_row = int(part_meta.get('row', 0))
                copied_meta['row'] = base_row
                copied_meta['base_row'] = base_row
                copied_meta['tile_x'] = sel_c
                copied_meta['tile_y'] = y_tile
                copied_meta['scanner_cell'] = cell_index
                copied_meta['stitched_x_copies'] = len(x_tiles)
                copied_meta['neighbor_swatch'] = False
                display_meta.append(copied_meta)
        return display_vl, display_fl, display_meta

    def _display_copy_z_period(self, verts_list, depth_gap):
        base_bounds = np.vstack([np.asarray(verts, dtype=np.float32) for verts, _ in verts_list])
        z_span = float(base_bounds[:, 2].max() - base_bounds[:, 2].min())
        return max(z_span + float(depth_gap), float(depth_gap))

    def _display_mesh_bounds(self, verts_list):
        valid = [np.asarray(verts, dtype=np.float32) for verts, _ in verts_list if len(verts)]
        if not valid:
            return None
        pts = np.vstack(valid)
        return pts.min(axis=0), pts.max(axis=0)

    def _display_copy_x_period(self, verts_list, radius):
        bounds = self._display_mesh_bounds(verts_list)
        if bounds is not None:
            min_v, max_v = bounds
            mesh_width = float(max_v[0] - min_v[0])
            if mesh_width > radius:
                return max(mesh_width - radius * 0.6, radius)
        period = np.asarray(self.period_offset_x, dtype=np.float32).reshape(-1) if hasattr(self, 'period_offset_x') else np.array([])
        if period.size > 0 and abs(float(period[0])) > max(float(radius), 1e-6):
            return abs(float(period[0]))
        if self.ctrl_rows:
            x_min = min(float(np.min(row[:, 0])) for row in self.ctrl_rows if len(row))
            x_max = max(float(np.max(row[:, 0])) for row in self.ctrl_rows if len(row))
            return max(x_max - x_min, radius)
        if not verts_list:
            return radius
        base_bounds = np.vstack([np.asarray(verts, dtype=np.float32) for verts, _ in verts_list])
        return max(float(base_bounds[:, 0].max() - base_bounds[:, 0].min()) - radius * 2.0, radius)

    def _display_copy_y_period(self, verts_list, radius):
        # A copy above is simply more rows of the same fabric, so it belongs
        # one whole row lattice above: pitch times row count. Measured from the
        # rows rather than read from `dy` -- an edited or rescaled model no
        # longer matches the parameter it was built from, and `dy * n_rows`
        # then tiles copies at a fraction of the fabric's real height and
        # overlaps them.
        #
        # Neither the row centres nor the mesh bounding box work here: loop
        # height moves both, so a tall row shifts them away from the lattice
        # the fabric actually repeats on.
        period = self._row_lattice_period()
        if period is not None:
            return max(period, radius)
        # No control rows to measure. One bounding box up is the least-wrong
        # guess left.
        bounds = self._display_mesh_bounds(verts_list)
        if bounds is None:
            return radius
        return max(float(bounds[1][1] - bounds[0][1]), radius)

    def prepare_display_meshes(self, vl, fl):
        vl, meta = build_surface_fiber_meshes(
            vl,
            segments=int(self.config['knit_parameters']['segments']),
            enabled=self.fiber_geometry_enabled,
            count=max(1, int(round(float(self.fiber_geometry_count)))),
            radius=float(self.params[self._pidx['radius']]),
            radius_scale=float(self.fiber_geometry_radius_scale),
            lift=float(self.fiber_geometry_lift),
            surface_arc=float(self.fiber_geometry_surface_arc),
            randomness=float(self.fiber_geometry_randomness),
            twist=float(self.fiber_geometry_twist),
        )
        fl = compute_knitting_faces(self.config['knit_parameters']['segments'], vl)
        return self.build_display_meshes_precise(vl, fl, meta)

    def active_colors(self):
        base_colors = self.row_colors if self.use_row_colors else [self.single_model_color]
        if bool(self.get('scanner_preview_grid_enabled', False)):
            rows = max(1, int(self.get('scanner_rows', 1)))
            cols = max(1, int(self.get('scanner_cols', 1)))
            template_bitmap = np.asarray(self._scanner_template().get('bitmap', np.empty((0, 0))), dtype=np.float32)
            base_rows = int(template_bitmap.shape[0]) if template_bitmap.ndim == 2 and template_bitmap.shape[0] > 0 else max(1, int(self.bitmap_size[0]))
            scanner_palette = self.get('scanner_color_variants', [])
            if isinstance(scanner_palette, list) and scanner_palette:
                base_colors = scanner_palette
            colors = []
            for _cell in range(rows * cols):
                for base_row in range(base_rows):
                    src = base_colors[base_row % len(base_colors)]
                    colors.append([float(src[0]), float(src[1]), float(src[2])])
            return colors or base_colors
        return base_colors


    def display_copy_periods(self):
        """World-space X, Y and Z spacing between display copies of the model.

        Both scan and puzzle tiling needed these and each rebuilt the spline
        mesh to derive them, so a single call site was paying for two identical
        mesh builds. The periods live here because every input to them --
        control rows, params, radius profiles -- is state.
        """
        radius = max(float(self.params[self._pidx['radius']]), 1e-6)
        radius_profiles = [np.asarray(row, dtype=np.float32) for row in self.spline_radius_rows]
        base_vl = build_spline_mesh(
            self.ctrl_rows,
            self.params,
            self.config,
            self._pidx,
            np.asarray(self.period_offset_x, dtype=np.float32),
            radius_ctrl_rows=radius_profiles,
        )
        depth_gap = max(radius * 2.4, 1e-6)
        return (
            self._display_copy_x_period(base_vl, radius),
            self._display_copy_y_period(base_vl, radius),
            self._display_copy_z_period(base_vl, depth_gap),
        )

    def _ensure_spline_radius_rows(self):
        base_radius = max(float(self.params[self._pidx['radius']]), 1e-6)
        rows = self.get('spline_radius_rows')
        if (
            not rows
            or len(rows) != len(self.ctrl_rows)
            or any(np.asarray(rad).shape[0] != len(row) for rad, row in zip(rows, self.ctrl_rows))
        ):
            self.spline_radius_rows = [
                np.full(len(row), base_radius, dtype=np.float32)
                for row in self.ctrl_rows
            ]

    def set_local_radius(self, flat_idx, value, start_rows=None):
        self._ensure_spline_radius_rows()
        row_idx = np.searchsorted(self._row_starts, flat_idx, side="right") - 1
        if not (0 <= row_idx < len(self.spline_radius_rows)):
            return

        local_idx = int(flat_idx - self._row_starts[row_idx])
        base_rows = start_rows if start_rows is not None else self.spline_radius_rows
        if not base_rows or row_idx >= len(base_rows):
            base_rows = self.spline_radius_rows

        radius_idx = self._pidx['radius']
        lo, hi = self.config['knit_parameters']['parameters'][radius_idx]["range"]
        target = float(np.clip(value, lo, hi))
        base = np.asarray(base_rows[row_idx], dtype=np.float32)
        if base.shape[0] != len(self.ctrl_rows[row_idx]):
            base = np.asarray(self.spline_radius_rows[row_idx], dtype=np.float32)

        indices = np.arange(len(base), dtype=np.float32)
        influence = max(1.5, min(4.0, len(base) * 0.12))
        weights = np.exp(-0.5 * ((indices - float(local_idx)) / influence) ** 2).astype(np.float32)
        updated = base * (1.0 - weights) + target * weights
        self.spline_radius_rows[row_idx] = updated.astype(np.float32)

    def rebuild_cached_jacobian(self):
        """Caches d(centreline samples)/d(control points) for the whole model.

        The solver works in control-point space but its energies are defined on
        the sampled centreline, so it needs this to map gradients back. It only
        depends on the control-point layout and the X period, so it is rebuilt
        when those change rather than every step.

        Callers must already hold sim_lock.
        """
        import scipy.sparse
        from knitting_core import build_row_spline_jacobian, _centerline_sample_count

        if not self.ctrl_rows:
            super().__setattr__('J_cached', None)
            self.sim_needs_jacobian_rebuild = False
            return
        # Shared helper so the Jacobian is built at the same sample count the
        # mesh and evaluate_centerlines use; a mismatch would leave the solver
        # differentiating geometry other than what is drawn.
        nout = _centerline_sample_count(self.period_offset_x, self.config)
        J_blocks = [build_row_spline_jacobian(cp, self.period_offset_x, nout) for cp in self.ctrl_rows]
        J_base = scipy.sparse.block_diag(J_blocks, format="csr")
        super().__setattr__('J_cached', scipy.sparse.kron(J_base, scipy.sparse.identity(3), format="csr"))
        self.sim_needs_jacobian_rebuild = False

    def _refresh_sim_rest_lengths(self):
        """Rest edge lengths the stretch energy is measured against. Recomputed
        with the mesh so the current shape is treated as the relaxed state."""
        from knitting_core import evaluate_centerlines
        V, edges, _, _ = evaluate_centerlines(self.ctrl_rows, self.period_offset_x, self.config)
        if len(edges) > 0:
            self.sim_L0 = np.linalg.norm(V[edges[:, 1]] - V[edges[:, 0]], axis=1)
        else:
            self.sim_L0 = np.array([])

    def active_scene_renderer(self):
        """Renderer owned by the 3D world viewport."""
        return self.renderer

    def rebuild_spline_mesh(self, preserve_model_placement=True, target_renderer=None):
        """Build the current model and upload it to the requested renderer."""
        old_center = np.asarray(self.mesh_center, dtype=np.float32).copy()
        old_model_t = np.asarray(self.model_t, dtype=np.float32).copy()
        if self.crochet_active():
            self._rebuild_crochet_mesh(target_renderer)
            if preserve_model_placement:
                self.mesh_center, self.model_t = old_center, old_model_t
            return
        self._ensure_spline_radius_rows()
        radius_profiles = [np.asarray(row, dtype=np.float32) for row in self.spline_radius_rows]
        vl = build_spline_mesh(
            self.ctrl_rows,
            self.params,
            self.config,
            self._pidx,
            np.asarray(self.period_offset_x, dtype=np.float32),
            radius_ctrl_rows=radius_profiles,
        )
        fl = compute_knitting_faces(self.config['knit_parameters']['segments'], vl)
        display_vl, display_fl, meta = self.prepare_display_meshes(vl, fl)
        colors = self.active_colors()
        destination = self.active_scene_renderer() if target_renderer is None else target_renderer
        prepared = destination.prepare_meshes(display_vl, display_fl)
        self.mesh_epoch = int(getattr(self, 'mesh_epoch', 0)) + 1
        self.last_prepared_meshes = (prepared, colors)
        destination.set_meshes(display_vl, display_fl, colors=colors, meta=meta, prepared=prepared)
        self._rebuild_spline_points()
        destination.set_ctrl_pts(self.flat_pts)
        if destination is self.renderer and self.orbit_renderer is not None and self.orbit_renderer is not self.renderer:
            self.orbit_renderer.set_meshes(display_vl, display_fl, colors=colors, meta=meta, prepared=prepared)
        if preserve_model_placement:
            self.mesh_center = old_center
            self.model_t = old_model_t
        else:
            self._recompute_center(display_vl)

    def crochet_active(self):
        """Edit Mode is showing a crochet fabric (the other modes always work on the knit model)."""
        return str(self.get('fabric_type', 'knit')) == 'crochet' and str(self.get('app_mode', 'edit')) == 'edit'

    def crochet_parameters(self):
        return crochet_core.CrochetParams(**{**crochet_core.CrochetParams().as_dict(), **dict(self.get('crochet_params', {}))})

    def _rebuild_crochet_mesh(self, target_renderer=None):
        """The crochet fabric from its equations (crochet_core), repeated display_copies times each way,
        as Edit Mode's meshes: one tube per post leg, wrap, V top and chain, coloured by row."""
        stitches = np.asarray(self.get('crochet_stitches'), dtype=np.float32)
        if stitches.ndim != 2 or stitches.size == 0:
            stitches = np.ones((1, 1), dtype=np.float32)
        segments = int(self.config['knit_parameters']['segments'])
        meshes, rows = crochet_core.ring_meshes(stitches, self.crochet_parameters(), segments,
                                                copies=tuple(int(c) for c in self.display_copies))
        bounds = self._display_mesh_bounds(meshes)
        if bounds is not None:
            min_v, max_v = bounds
            center = np.array([0.5 * (min_v[0] + max_v[0]), 0.5 * (min_v[1] + max_v[1]), 0.0], dtype=np.float32)
            rows_cnt = max(1, int(self.get('scanner_rows', 1)))
            cols_cnt = max(1, int(self.get('scanner_cols', 1)))
            sel_r = int(np.clip(int(self.get('scanner_selected_row', 0)), 0, rows_cnt - 1))
            sel_c = int(np.clip(int(self.get('scanner_selected_col', 0)), 0, cols_cnt - 1))
            shift = self.swatch_slot_offset(sel_r, sel_c) - center
            meshes = [((np.asarray(v, dtype=np.float32) + shift).astype(np.float32), int(n)) for v, n in meshes]
        faces = [compute_knitting_faces(segments, [mesh])[0] for mesh in meshes]
        meta = [{'row': row, 'base_row': row} for row in rows]
        destination = self.active_scene_renderer() if target_renderer is None else target_renderer
        colors = self.active_colors()
        prepared = destination.prepare_meshes(meshes, faces)
        self.mesh_epoch = int(getattr(self, 'mesh_epoch', 0)) + 1
        self.last_prepared_meshes = (prepared, colors)
        destination.set_meshes(meshes, faces, colors=colors, meta=meta, prepared=prepared)
        destination.set_ctrl_pts(np.empty((0, 3), dtype=np.float32))
        if destination is self.renderer and self.orbit_renderer is not None and self.orbit_renderer is not self.renderer:
            self.orbit_renderer.set_meshes(meshes, faces, colors=colors, meta=meta, prepared=prepared)
        self._recompute_center(meshes)

    def _recompute_center(self, _display_vl=None):
        self.mesh_center = np.zeros(3, dtype=np.float32)
        self.model_t = np.zeros(3, dtype=np.float32)

    def _rebuild_spline_points(self):
        self._row_starts = np.concatenate(([0], np.cumsum([len(r) for r in self.ctrl_rows]))).tolist()
        if self.ctrl_rows:
            offset = self._compute_active_swatch_render_offset()
            self.flat_pts = (np.concatenate(self.ctrl_rows).astype(np.float32) + offset[None, :]).astype(np.float32)
        else:
            self.flat_pts = np.empty((0, 3), np.float32)

    def _period_offset_from_ctrl_rows(self):
        periods = []
        cols = max(1, int(self.bitmap_size[1])) if hasattr(self, 'bitmap_size') else 1
        for row in self.ctrl_rows:
            row = np.asarray(row, dtype=np.float32)
            if len(row) >= 2:
                samples_per_col = max(1, len(row) // cols)
                if cols > 1 and samples_per_col < len(row):
                    shifted = row[samples_per_col:] - row[:-samples_per_col]
                    x_steps = shifted[:, 0]
                    x_steps = x_steps[x_steps > 1e-6]
                    if len(x_steps):
                        periods.append(float(np.median(x_steps)) * cols)
                        continue
                span = float(row[:, 0].max() - row[:, 0].min())
                if span > 1e-6:
                    periods.append(span)
        if not periods:
            return np.array([float(self.bitmap_size[1]), 0.0, 0.0], dtype=np.float32)
        return np.array([float(np.median(periods)), 0.0, 0.0], dtype=np.float32)

    def sync_period_offset_to_model_width(self):
        self.period_offset_x = self._period_offset_from_ctrl_rows()

    def _row_lattice_period(self):
        """One whole fabric height: the row pitch times the number of rows.

        The single definition of the vertical period. Both the yarn
        simulation's periodic wrap and the display-copy tiling ask for it here,
        so the two cannot drift apart -- which is exactly how the copies came
        to overlap: one derivation trusted `dy` while the geometry had moved on.

        Returns None when there are no rows to measure at all.
        """
        n_rows = len(self.ctrl_rows)
        if n_rows == 0:
            return None
        pitch = row_base_pitch(self.ctrl_rows)
        if pitch is None:
            # A single row has no step to measure. One parametric pitch up is
            # the right answer for a one-row fabric.
            dy_idx = self._pidx.get('dy')
            pitch = abs(float(self.params[dy_idx])) if dy_idx is not None else 0.0
        return pitch * n_rows if pitch > 1e-6 else None

    def sync_period_offset_y_to_row_count(self):
        """Y period spans the whole stack of rows, so a vertically tiled copy
        lands exactly one fabric-height above.

        Measured from the control rows, the same way the X period is, so a
        rescaled or hand-edited model reports its real height instead of the
        `dy` it was first built from. Only the yarn simulation reads this;
        display tiling reaches the same measurement through
        _display_copy_y_period.
        """
        period = self._row_lattice_period()
        if period is None:
            dy_idx = self._pidx.get('dy')
            dy_val = abs(float(self.params[dy_idx])) if dy_idx is not None else 1.0
            period = dy_val * max(1, len(self.ctrl_rows))
        self.period_offset_y = np.array([0.0, period, 0.0], dtype=np.float32)

    @property
    def flat_pts_all(self):
        if not self.ctrl_rows:
            return np.empty((0, 3), dtype=np.float32)
        offset = self._compute_active_swatch_render_offset()
        virtual_pts_x = np.array([row[0] + self.period_offset_x + offset for row in self.ctrl_rows], dtype=np.float32)
        virtual_pts_y = np.array([row[0] + self.period_offset_y + offset for row in self.ctrl_rows], dtype=np.float32)
        return np.concatenate((self.flat_pts, virtual_pts_x, virtual_pts_y), axis=0)

    def move_ctrl_pt(self, flat_idx, pos):
        local_pos = np.asarray(pos, dtype=np.float32) - self._compute_active_swatch_render_offset()
        n_real = len(self.flat_pts)
        if flat_idx >= n_real:
            n_rows = len(self.ctrl_rows)
            virtual_idx = flat_idx - n_real
            row_idx = virtual_idx % n_rows if n_rows else 0
            if not (0 <= row_idx < n_rows):
                return
            if virtual_idx < n_rows:
                self.period_offset_x = local_pos - self.ctrl_rows[row_idx][0]
                self.rebuild_spline_mesh()
            else:
                self.period_offset_y = local_pos - self.ctrl_rows[row_idx][0]
            with self.sim_lock:
                self.sim_needs_jacobian_rebuild = True
        else:
            r = np.searchsorted(self._row_starts, flat_idx, side="right") - 1
            if 0 <= r < len(self.ctrl_rows):
                self.ctrl_rows[r][flat_idx - self._row_starts[r]] = local_pos
                self._rebuild_spline_points()

    def center_model_on_view(self):
        self.mesh_center = np.zeros(3, dtype=np.float32)
        self.model_t = np.zeros(3, dtype=np.float32)

    def _default_loop_height_for_row(self, row_idx):
        if not self._lh_idx:
            return 0.0
        idx = self._lh_idx[min(int(row_idx), len(self._lh_idx) - 1)]
        return float(self.params[idx])

    def _loop_height_overrides(self, rows, cols):
        """Boolean grid marking cells whose height the user set by hand."""
        stored = self.get('loop_height_overrides', None)
        grid = np.zeros((rows, cols), dtype=bool)
        if stored is not None:
            stored = np.asarray(stored, dtype=bool)
            if stored.ndim == 2:
                keep_rows = min(rows, stored.shape[0])
                keep_cols = min(cols, stored.shape[1])
                if keep_rows > 0 and keep_cols > 0:
                    grid[:keep_rows, :keep_cols] = stored[:keep_rows, :keep_cols]
        return grid

    def _sync_loop_heights(self):
        """Rebuilds the per-cell height grid from the row parameters.

        Only cells the user has actually dragged keep their own value. This grid
        used to win outright over the parameters, so whatever it happened to
        hold first became permanent: it starts life as a block of 3.0 that is
        unrelated to loop_height_1..n, and those 3.0s then survived every later
        parameter change. A model whose parameters read 1.06/1.84/4.28/4.96
        would still be built from 3.0/3.0/3.0/4.96 -- only the row added last,
        which had no stale entry to inherit, ever picked its parameter up. That
        is why one row stood out against the rest.
        """
        rows, cols = int(self.bitmap_size[0]), int(self.bitmap_size[1])
        existing = np.asarray(self.get('loop_heights', np.empty((0, 0))), dtype=np.float32)
        overrides = self._loop_height_overrides(rows, cols)
        synced = np.zeros((rows, cols), dtype=np.float32)
        for row_idx in range(rows):
            synced[row_idx, :] = self._default_loop_height_for_row(row_idx)
        if existing.ndim == 2:
            keep_rows = min(rows, existing.shape[0])
            keep_cols = min(cols, existing.shape[1])
            if keep_rows > 0 and keep_cols > 0:
                kept = overrides[:keep_rows, :keep_cols]
                synced[:keep_rows, :keep_cols] = np.where(
                    kept, existing[:keep_rows, :keep_cols], synced[:keep_rows, :keep_cols]
                )
        self.loop_heights = synced
        self.loop_height_overrides = overrides
        return synced

    def set_loop_height_cell(self, row_idx, col_idx, value):
        self._sync_loop_heights()
        row_idx = int(np.clip(row_idx, 0, int(self.bitmap_size[0]) - 1))
        col_idx = int(np.clip(col_idx, 0, int(self.bitmap_size[1]) - 1))
        lo, hi = 0.0, 6.0
        if self._lh_idx:
            pd = self.config['knit_parameters']['parameters'][self._lh_idx[min(row_idx, len(self._lh_idx) - 1)]]
            lo, hi = float(pd['range'][0]), float(pd['range'][1])
        self.loop_heights[row_idx, col_idx] = float(np.clip(value, lo, hi))
        # From here this cell is the user's, and no longer follows its row.
        overrides = np.asarray(self.loop_height_overrides, dtype=bool).copy()
        overrides[row_idx, col_idx] = True
        self.loop_height_overrides = overrides

    def clear_loop_height_overrides(self):
        """Hands every cell back to its row parameter."""
        rows, cols = int(self.bitmap_size[0]), int(self.bitmap_size[1])
        self.loop_height_overrides = np.zeros((rows, cols), dtype=bool)
        return self._sync_loop_heights()

    def _fresh_rebuild_rows(self):
        self._sync_loop_heights()
        return build_parametric_control_rows(
            self.params,
            self.bitmap,
            self._pidx,
            self._lh_idx,
            self.samples_per_loop,
            loop_heights=self.loop_heights,
        )

    def rebuild_spline_from_params(self, rebuild_mesh=True):
        """Rebuilds the control rows from the current parameters.

        `rebuild_mesh=False` skips the trailing mesh build and GPU upload. It is
        for callers that follow this immediately with their own
        rebuild_spline_mesh call and so were paying for two full builds where
        the second discards the first -- the scan pattern pipeline does exactly
        that, once per cell. Safe because rebuild_spline_mesh is a pure function
        of the control rows and params set above, and the only state the skipped
        call writes is mesh_center/model_t, which with
        preserve_model_placement=True it just writes back unchanged.
        """
        self.ctrl_rows = self._fresh_rebuild_rows()
        self.sync_period_offset_to_model_width()
        self.sync_period_offset_y_to_row_count()
        # Rebuilding from parameters defines a new relaxed shape, so this is one
        # of the two moments rest lengths are (re)captured -- the other is when
        # the simulation is switched on. Deliberately NOT done on every
        # rebuild_spline_mesh: that runs after each solver step, and refreshing
        # rest lengths there would keep resetting the rest state to the deformed
        # shape, leaving the stretch energy permanently near zero.
        self._refresh_sim_rest_lengths()
        self.sim_needs_jacobian_rebuild = True
        base_radius = max(float(self.params[self._pidx['radius']]), 1e-6)
        self.spline_radius_rows = [
            np.full(len(row), base_radius, dtype=np.float32)
            for row in self.ctrl_rows
        ]
        self.param_ref_radius = base_radius
        self.param_ref_ctrl_rows = [row.copy() for row in self.ctrl_rows]
        self._rebuild_spline_points()
        if rebuild_mesh:
            self.rebuild_spline_mesh(preserve_model_placement=True)

    def _ctrl_rows_bounds(self, rows):
        valid = [np.asarray(row, dtype=np.float32) for row in rows if len(row)]
        if not valid:
            return None
        pts = np.vstack(valid)
        return pts.min(axis=0), pts.max(axis=0)

    def _fit_ctrl_rows_to_bounds(self, rows, target_bounds):
        source_bounds = self._ctrl_rows_bounds(rows)
        if source_bounds is None or target_bounds is None:
            return rows
        src_min, src_max = source_bounds
        dst_min, dst_max = target_bounds
        src_center = (src_min + src_max) * 0.5
        dst_center = (dst_min + dst_max) * 0.5
        src_span = np.maximum(src_max - src_min, 1e-6)
        dst_span = np.maximum(dst_max - dst_min, 1e-6)
        scale = np.array([dst_span[0] / src_span[0], dst_span[1] / src_span[1], 1.0], dtype=np.float32)
        return [
            (dst_center + (np.asarray(row, dtype=np.float32) - src_center) * scale).astype(np.float32)
            for row in rows
        ]

    def nudge_spline_from_params(self, preserve_bounds=False, rebuild_mesh=True):
        """Apply the parametric change on top of the live rows, keeping edits.

        `rebuild_mesh=False` skips the trailing mesh build and upload, for
        callers that follow this with their own rebuild_spline_mesh -- the scan
        pattern pipeline does exactly that, once per cell, and would otherwise
        pay for two full builds where the second discards the first. Same
        contract as rebuild_spline_from_params.
        """
        old_bounds = self._ctrl_rows_bounds(self.ctrl_rows) if preserve_bounds else None
        target_rows = self._fresh_rebuild_rows()
        ref_rows = self.get('param_ref_ctrl_rows')
        current_radius = max(float(self.params[self._pidx['radius']]), 1e-6)
        ref_radius = getattr(self, 'param_ref_radius', current_radius)
        if not ref_rows or len(self.ctrl_rows) != len(target_rows) or any(c.shape != t.shape for c, t in zip(self.ctrl_rows, target_rows)):
            self.ctrl_rows = [row.copy() for row in target_rows]
            self.spline_radius_rows = [
                np.full(len(row), current_radius, dtype=np.float32)
                for row in self.ctrl_rows
            ]
        else:
            nudged_rows = []
            for current_row, ref_row, target_row in zip(self.ctrl_rows, ref_rows, target_rows):
                nudged_rows.append(current_row + (target_row - ref_row))
            self.ctrl_rows = nudged_rows
            self._ensure_spline_radius_rows()
            if abs(current_radius - ref_radius) > 1e-7:
                ratio = current_radius / ref_radius
                for r_idx, row in enumerate(self.spline_radius_rows):
                    self.spline_radius_rows[r_idx] = row * ratio

        if preserve_bounds:
            self.ctrl_rows = self._fit_ctrl_rows_to_bounds(self.ctrl_rows, old_bounds)

        self.param_ref_radius = current_radius
        self.param_ref_ctrl_rows = [row.copy() for row in target_rows]
        self._rebuild_spline_points()
        self.sync_period_offset_to_model_width()
        # Y as well as X: nudged rows can change the fabric's height, and the
        # display and the simulation both read the Y period back.
        self.sync_period_offset_y_to_row_count()
        if rebuild_mesh:
            self.rebuild_spline_mesh(preserve_model_placement=True)

    def current_model_matrix(self):
        scale = np.asarray(self.model_scale, dtype=np.float32).reshape(-1)
        if scale.size == 1:
            scale = np.repeat(scale, 3)
        if scale.size < 3:
            scale = np.pad(scale, (0, 3 - scale.size), constant_values=1.0)
        scale = np.maximum(scale[:3], 1e-4)
        model_s = np.eye(4, dtype=np.float32)
        model_s[0, 0] = scale[0]
        model_s[1, 1] = scale[1]
        model_s[2, 2] = scale[2]
        tneg, tpos, tuser = np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)
        tneg[:3, 3] = -self.mesh_center
        tpos[:3, 3] = self.mesh_center
        tuser[:3, 3] = self.model_t
        return (tuser @ tpos @ model_s @ tneg).astype(np.float32)

    # ── UNDO / REDO ───────────────────────────────────────────────────────────

    # Live handles: threads, GL textures, database and robot connections. They
    # are objects the app owns right now, not values describing the model, and
    # _clone passes them through by reference -- so an undo would write a stale
    # handle back over the current one. For the UR5 controller that would orphan
    # a live robot connection mid-scan, leaving nothing able to stop the arm.
    _SNAPSHOT_EXCLUDE = frozenset({
        'render_tex', 'render_result', 'undo_stack',
        'ur5_controller', 'scanner_storage', 'scanner_process',
        'workspace_model_snapshot', 'workspace_undo_stack',
        'single_capture_job', 'single_capture_job_seq',
        'mode_epoch', 'mesh_epoch', 'last_prepared_meshes',
        '_cached_canonical_tile_bounds', '_active_swatch_render_offset',
        '_swatch_tile_cache', '_swatch_tiles_epoch',
        'app_mode',
    })

    def snapshot_state(self):
        exclude = self._SNAPSHOT_EXCLUDE
        snap = {k: self._clone(v) for k, v in self._data.items() if k not in exclude}
        for attr in self.camera_attributes:
            snap[f'camera_{attr}'] = self._clone(getattr(self.camera, attr))
        snap['ctrl_rows'] = [row.copy() for row in self.ctrl_rows]
        snap['spline_radius_rows'] = [row.copy() for row in self.spline_radius_rows]
        return snap

    def push_undo(self, label):
        snap = self.snapshot_state()
        snap['label'] = label
        self.undo_stack.append(snap)
        if len(self.undo_stack) > self.max_undo:
            del self.undo_stack[0]

    def restore_snapshot(self, snap, target_renderer=None):
        """Restore a snapshot. Returns the bottom-row columns it had to repair.

        The return value exists because undo_last and reset_to_initial both set
        status_msg *after* calling this, so a message written in here would be
        clobbered. They append a note instead, and only when this reports
        something -- which is what keeps a clean undo's message unchanged.
        """
        for k, v in snap.items():
            if not k.startswith('camera_') and k not in ('ctrl_rows', 'spline_radius_rows', 'label'):
                self._data[k] = self._clone(v)
        for attr in self.camera_attributes:
            setattr(self.camera, attr, self._clone(snap[f'camera_{attr}']))
        self.camera.fov_deg = self.view_fov

        pre_anchor_bitmap = np.asarray(self.bitmap, dtype=np.float32).copy()
        anchored_columns = self._anchor_bitmap()

        self.ctrl_rows = [row.copy() for row in snap['ctrl_rows']]
        self.spline_radius_rows = [row.copy() for row in snap.get('spline_radius_rows', [])]
        self._ensure_spline_radius_rows()
        self.param_ref_radius = max(float(self.params[self._pidx['radius']]), 1e-6)
        self.param_ref_ctrl_rows = self._fresh_rebuild_rows()
        self._rebuild_spline_points()

        if anchored_columns.size:
            # The snapshot's control rows carry the flat bottom row, same as a
            # loaded file does.
            self._reconcile_rows_after_anchor(pre_anchor_bitmap)
        self.rebuild_spline_mesh(target_renderer=target_renderer)
        return anchored_columns

    def undo_last(self):
        if not self.undo_stack:
            return
        anchored = self.restore_snapshot(self.undo_stack.pop())
        self.status_msg = _with_anchor_note('Undid last change', anchored)

    def capture_initial_state(self):
        super().__setattr__('_initial_snapshot', self.snapshot_state())

    def reset(self, unit_model=False):
        """Unified reset entry point, matching feat/simulation's call style.

        Deliberately a wrapper rather than a merged implementation: on this
        branch reset_to_initial restores the snapshot captured from
        initial_params.json -- which is also Scan Mode's fabric template --
        whereas feat/simulation's reset() falls back to bare state defaults.
        Folding them together would quietly drop that behaviour, so both
        methods keep their own semantics and this just dispatches.
        """
        if unit_model:
            self.reset_to_unit_model()
        else:
            self.reset_to_initial()

    def reset_to_initial(self):
        initial_snapshot = getattr(self, '_initial_snapshot', None)
        if initial_snapshot is not None:
            self.push_undo("Reset all")
            anchored = self.restore_snapshot(initial_snapshot)
            self.status_msg = _with_anchor_note('Reset to saved initial model', anchored)
            return

        self.push_undo("Reset all")
        defaults = self._state_defaults

        self.params = [p['initial'] for p in self.config['knit_parameters']['parameters']]
        self.bitmap_size = defaults['bitmap_size'].copy()
        self.bitmap = defaults['bitmap'].copy()
        self.samples_per_loop = 5
        self.display_copies = defaults['display_copies'].copy()
        self.mode = defaults['mode']
        self.workflow_step = 0

        reset_keys = (
            'selected_idx', 'hover_idx', 'hover_mesh_idx', 'selected_mesh_idx',
            'model_rot', 'model_scale', 'model_t', 'mesh_center',
            'use_row_colors', 'single_model_color', 'row_colors', 'row_visible',
            'fiber_geometry_enabled', 'fiber_geometry_count',
            'fiber_geometry_radius_scale', 'fiber_geometry_lift',
            'fiber_geometry_surface_arc', 'fiber_geometry_randomness',
            'fiber_geometry_twist', 'spline_keyboard_step',
            'spline_grab_active', 'radius_grab_active',
            'spline_keyboard_edit_active', 'radius_keyboard_edit_active',
            'period_offset_x', 'period_offset_y', 'app_mode', 'scanner_layout_pattern', 'loop_heights', 'loop_height_overrides',
        )
        for key in reset_keys:
            if key in defaults:
                self._data[key] = self._clone(defaults[key])
        for key in ('render_texture_color', *self.texture_param_keys):
            if key in defaults:
                self._data[key] = self._clone(defaults[key])

        self._sync_row_colors(int(self.bitmap_size[0]))
        self._sync_row_visibility(int(self.bitmap_size[0]))
        self._sync_loop_heights()
        self.spline_radius_rows = []
        self.param_ref_ctrl_rows = []
        self.rebuild_spline_from_params()
        self.center_model_on_view()
        self.status_msg = 'Reset to initial parameters'

    def reset_to_unit_model(self):
        self.push_undo("Reset unit model")
        defaults = self._state_defaults

        self.params = [p['initial'] for p in self.config['knit_parameters']['parameters']]
        radius_idx = self._pidx['radius']
        radius_range = self.config['knit_parameters']['parameters'][radius_idx]['range']
        self.params[radius_idx] = float(np.clip(0.5, radius_range[0], radius_range[1]))
        self.bitmap_size = defaults['bitmap_size'].copy()
        self.bitmap = np.ones(tuple(int(v) for v in self.bitmap_size), dtype=np.float32)
        self.samples_per_loop = 5
        self.display_copies = defaults['display_copies'].copy()
        self.mode = defaults['mode']
        self.workflow_step = 0

        for key in (
            'selected_idx', 'hover_idx', 'hover_mesh_idx', 'selected_mesh_idx',
            'model_rot', 'model_scale', 'model_t', 'mesh_center',
            'use_row_colors', 'single_model_color', 'row_colors', 'row_visible',
            'fiber_geometry_enabled', 'fiber_geometry_count',
            'fiber_geometry_radius_scale', 'fiber_geometry_lift',
            'fiber_geometry_surface_arc', 'fiber_geometry_randomness',
            'fiber_geometry_twist', 'spline_grab_active', 'radius_grab_active',
            'spline_keyboard_edit_active', 'radius_keyboard_edit_active',
            'period_offset_x', 'period_offset_y', 'app_mode', 'scanner_layout_pattern', 'loop_heights', 'loop_height_overrides',
        ):
            if key in defaults:
                self._data[key] = self._clone(defaults[key])
        for key in ('render_texture_color', *self.texture_param_keys):
            if key in defaults:
                self._data[key] = self._clone(defaults[key])
        self._sync_row_colors(int(self.bitmap_size[0]))
        self._sync_row_visibility(int(self.bitmap_size[0]))
        self.row_visible[:] = True
        self.spline_radius_rows = []
        self.param_ref_ctrl_rows = []
        self.rebuild_spline_from_params()
        self.center_model_on_view()
        self.status_msg = 'Reset initial model: pattern=1, samples=5, radius=0.5, multi-fiber off'

    def apply_texture_preset(self, preset_name, undo_label="Texture preset"):
        if preset_name in self.texture_presets:
            self.push_undo(undo_label)
            for k, val in self.texture_presets[preset_name].items():
                setattr(self, k, np.array(val, dtype=np.float32) if isinstance(val, list) else val)

    # ── BITMAP / WORKFLOW UPDATES ─────────────────────────────────────────────

    def _anchor_bitmap(self):
        """Force a stitch into every bottom-row cell, and report what was filled.

        The one place that assigns a repaired bitmap. Sets no status message --
        callers compose their own, because the two that go through
        restore_snapshot write theirs afterwards and would clobber it.
        """
        repaired, columns = anchor_bitmap(self.bitmap)
        if columns.size:
            self.bitmap = repaired
        return columns

    def _reconcile_rows_after_anchor(self, pre_bitmap):
        """Give the newly-anchored bottom row its loop, leaving the rest alone.

        Repairing the bitmap is not enough on the load path: the saved control
        rows are installed verbatim afterwards, and they carry the flat,
        zero-height bottom row the cleared cell produced -- so the pattern would
        read as anchored while the fabric still showed the bar.

        Nudging from a reference built with the *pre-repair* bitmap confines the
        change to the cells that actually changed: only row 0 differs between
        the two parametric builds, so the user's tuned geometry above it is
        carried through untouched.
        """
        saved = self.bitmap
        self.bitmap = np.asarray(pre_bitmap, dtype=np.float32)
        try:
            # _fresh_rebuild_rows -> _sync_loop_heights reads bitmap_size, never
            # bitmap, and the repair cannot change the shape, so this swap is safe.
            reference = [row.copy() for row in self._fresh_rebuild_rows()]
        finally:
            self.bitmap = saved
        self.param_ref_ctrl_rows = reference
        self.nudge_spline_from_params(rebuild_mesh=False)

    def on_bitmap_change(self):
        # Silent backstop. The editor refuses the edit that would break this, so
        # in practice nothing is left to repair here -- but any future caller
        # writing straight to state.bitmap still cannot produce a fabric that
        # hangs from nothing.
        self._anchor_bitmap()
        self._sync_loop_heights()
        self.rebuild_spline_from_params()

    def on_bitmap_resize(self, new_rows, new_cols):
        new_rows = max(1, min(int(new_rows), int(self.config['knit_parameters']['bitmap_rows'])))
        new_cols = max(1, min(int(new_cols), 32))
        old = self.bitmap
        new_bm = np.ones((new_rows, new_cols), dtype=np.float32)
        new_bm[:min(old.shape[0], new_rows), :min(old.shape[1], new_cols)] = old[:min(old.shape[0], new_rows), :min(old.shape[1], new_cols)]
        self.bitmap = new_bm
        self.bitmap_size = np.array([new_rows, new_cols], dtype=np.int32)
        # The old bottom row is copied straight into the new one, so a resize can
        # carry an existing violation through even though it never creates one.
        self._anchor_bitmap()
        self._sync_row_colors(new_rows)
        self._sync_row_visibility(new_rows)
        self._sync_loop_heights()
        self.rebuild_spline_from_params()

    def apply_knit_pattern(self, parameters, bitmap):
        """Apply fitted knit loop parameters (mm, by name) and stitch bitmap atomically."""
        ranges = {p['name']: p['range'] for p in self.config['knit_parameters']['parameters']}
        params = list(self.params)
        for name, value in parameters.items():
            if name in self._pidx:
                low, high = ranges[name]
                params[self._pidx[name]] = float(np.clip(value, low, high))
        self.params = params
        max_rows = int(self.config['knit_parameters']['bitmap_rows'])
        bitmap = np.asarray(bitmap, dtype=np.float32)[:max_rows, :32].copy()
        self.bitmap = bitmap
        self.bitmap_size = np.array(bitmap.shape, dtype=np.int32)
        self._sync_row_colors(bitmap.shape[0])
        self._sync_row_visibility(bitmap.shape[0])
        self.clear_loop_height_overrides()
        self.on_bitmap_change()

    def loop_height_span(self, name):
        if name.startswith("loop_height_"):
            try:
                return int(name[12:])
            except ValueError:
                pass
        return None

    # ── PARAMETER SERIALIZATION ───────────────────────────────────────────────

    def save_params(self, path, silent=False):
        params_to_save = {
            name: float(self.params[i])
            for i, name in enumerate(p["name"] for p in self.config['knit_parameters']['parameters'])
            if (span := self.loop_height_span(name)) is None or span <= self.bitmap_size[0]
        }
        gui_state = {k: json_ready(self._data[k]) for k in self.saved_state_keys if k in self._data}
        for attr in self.camera_attributes:
            gui_state[f'camera_{attr}'] = json_ready(getattr(self.camera, attr))

        data = {
            'format_version': 2,
            'params': params_to_save,
            'bitmap': self.bitmap.tolist(),
            'period_offset_x': self.period_offset_x.tolist(),
            'period_offset_y': self.period_offset_y.tolist(),
            'spline_control_rows': [row.tolist() for row in self.ctrl_rows],
            'spline_radius_rows': [row.tolist() for row in self.spline_radius_rows],
            'gui_state': gui_state,
        }
        try:
            write_json(path, data)
            self.save_path  = path
            self.autosave_last_time = time.monotonic()
            if not silent:
                self.status_msg = f'Saved → {os.path.basename(path)}'
        except Exception as e:
            self.status_msg = f'Save error: {e}'

    # Interactions during which a write would both stutter the frame and capture
    # a half-finished edit. Autosave waits for the user to let go.
    _AUTOSAVE_DEFER_FLAGS = (
        'gizmo_edit_active',
        'spline_grab_active',
        'radius_grab_active',
        'model_drag_undo_active',
        'spline_keyboard_edit_active',
        'radius_keyboard_edit_active',
    )

    def autosave_deferred(self):
        """True while an edit is mid-gesture and autosaving should hold off."""
        if any(bool(self.get(flag, False)) for flag in self._AUTOSAVE_DEFER_FLAGS):
            return True
        return int(self.get('bbox_active_handle', -1)) >= 0

    def maybe_autosave(self):
        if not self.autosave_enabled:
            return
        if self.autosave_deferred():
            return
        now = time.monotonic()
        if now - float(self.autosave_last_time) < float(self.autosave_interval_sec):
            return
        target_path = self.save_path or str(paths.PARAMS_JSON)
        # The solver rewrites ctrl_rows/flat_pts from another thread, so hold
        # sim_lock while serialising to avoid writing a file that mixes geometry
        # from two different steps. It only ever holds the lock briefly (the
        # solve itself runs outside it), so this does not stall the frame.
        if bool(self.get('sim_active', False)):
            with self.sim_lock:
                self.save_params(target_path, silent=True)
        else:
            self.save_params(target_path, silent=True)

    def load_params(self, path, *, restore_mode=False, rebuild_mesh=True):
        with self.sim_lock:
            previous = self._data
            camera = {attr: self._clone(getattr(self.camera, attr)) for attr in self.camera_attributes}
            self._data = self._clone(previous)
            uploading = False
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
                if data.get('format_version') != 2:
                    raise ValueError('Expected model format_version 2')
                definitions = self.config['knit_parameters']['parameters']
                self.params = [float(np.clip(data['params'].get(p['name'], p['initial']), *p['range']))
                               for p in definitions]
                self.bitmap = np.asarray(data['bitmap'], dtype=np.float32)
                if self.bitmap.ndim != 2 or not self.bitmap.size or not np.isfinite(self.bitmap).all():
                    raise ValueError('bitmap must be a finite, nonempty 2D array')
                if not np.isfinite(self.params).all():
                    raise ValueError('parameters must be finite')
                self.bitmap_size = np.array(self.bitmap.shape, dtype=np.int32)
                pre_anchor_bitmap = self.bitmap.copy()
                anchored_columns = self._anchor_bitmap()
                gui_state = data['gui_state']
                for key in self.saved_state_keys:
                    if key not in gui_state or key == 'bitmap_size' or (key == 'app_mode' and not restore_mode):
                        continue
                    default = self._state_defaults.get(key)
                    value = state_value(gui_state[key], default)
                    if isinstance(default, np.ndarray):
                        if default.ndim == 1 and default.size and value.shape != default.shape and key != 'row_visible':
                            raise ValueError(f'{key} must have shape {default.shape}')
                    self._data[key] = value
                if int(self._data.get('scanner_pattern_repeat_rows', 24)) <= 4:
                    self._data['scanner_pattern_repeat_rows'] = 24
                if int(self._data.get('scanner_pattern_repeat_cols', 24)) <= 4:
                    self._data['scanner_pattern_repeat_cols'] = 24
                for attr in self.camera_attributes:
                    key = f'camera_{attr}'
                    if key in gui_state:
                        value = np.asarray(gui_state[key], dtype=float)
                        if value.shape != np.shape(camera[attr]) or not np.isfinite(value).all():
                            raise ValueError(f'{key} has invalid shape or values')
                        setattr(self.camera, attr, value if value.ndim else float(value))
                self._sync_loop_heights()
                self._sync_row_colors(self.bitmap_size[0])
                self._sync_row_visibility(self.bitmap_size[0])
                self.mode = 'spline'
                self.camera.fov_deg = self.view_fov
                self.rebuild_spline_from_params(rebuild_mesh=False)
                rows = [np.asarray(row, dtype=np.float32) for row in data['spline_control_rows']]
                if rows:
                    if len(rows) != self.bitmap.shape[0] or any(
                        row.ndim != 2 or row.shape[1] != 3 or len(row) < 2 or not np.isfinite(row).all()
                        for row in rows
                    ):
                        raise ValueError('spline rows must match bitmap rows and contain finite 3D points')
                    radii = [np.asarray(row, dtype=np.float32) for row in data['spline_radius_rows']]
                    if len(radii) != len(rows) or any(
                        radius.shape != (len(row),) or not np.isfinite(radius).all() or np.any(radius <= 0)
                        for row, radius in zip(rows, radii)
                    ):
                        raise ValueError('spline radii must be positive and match control points')
                    self.ctrl_rows = rows
                    self.spline_radius_rows = radii
                    self._rebuild_spline_points()
                    self.sync_period_offset_to_model_width()
                    self.sync_period_offset_y_to_row_count()
                    if anchored_columns.size:
                        self._reconcile_rows_after_anchor(pre_anchor_bitmap)
                    else:
                        self.param_ref_ctrl_rows = self._fresh_rebuild_rows()
                if rebuild_mesh:
                    uploading = True
                    self.rebuild_spline_mesh(preserve_model_placement=False)
                self.load_path = path
                self.status_msg = _with_anchor_note(f'Loaded {os.path.basename(path)}', anchored_columns)
                return True
            except Exception as exc:
                self._data = previous
                for attr, value in camera.items():
                    setattr(self.camera, attr, value)
                if uploading:
                    self.rebuild_spline_mesh()
                self.status_msg = f'Load error: {exc}'
                return False
