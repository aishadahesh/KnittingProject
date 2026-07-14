import os
import json
import threading
import time
import numpy as np
import jax.numpy as jnp
from PIL import Image

from knitting_core import (
    compute_knitting_faces,
    build_parametric_control_rows, build_spline_mesh, build_surface_fiber_meshes
)

# %% APP STATE ─────────────────────────────────────────────────────────────────
# %% APP STATE ─────────────────────────────────────────────────────────────────

class AppState:
    @staticmethod
    def _json_ready(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [AppState._json_ready(v) for v in value]
        if isinstance(value, dict):
            return {k: AppState._json_ready(v) for k, v in value.items()}
        return value

    @staticmethod
    def _clone(v):
        if isinstance(v, np.ndarray):
            return v.copy()
        if isinstance(v, list):
            return [AppState._clone(x) for x in v]
        if isinstance(v, dict):
            return {k: AppState._clone(x) for k, x in v.items()}
        return v

    def __init__(self, camera, renderer):
        self.camera = camera
        self.renderer = renderer

        # Project root and configurations
        project_root = os.path.dirname(os.path.abspath(__file__))
        super().__setattr__('project_root', project_root)
        super().__setattr__('resolve_project_path', lambda p: p if os.path.isabs(p) else os.path.join(project_root, p))

        with open(os.path.join(project_root, "config.json"), "r") as f:
            config_data = json.load(f)
        super().__setattr__('config', config_data)
        super().__setattr__('_scanner_template_cache', None)
        super().__setattr__('_scanner_template_mtime', None)

        pidx = {p["name"]: i for i, p in enumerate(config_data["knit_parameters"]["parameters"])}
        super().__setattr__('_pidx', pidx)

        lh_params = sorted(
            [p["name"] for p in config_data["knit_parameters"]["parameters"] if p["name"].startswith("loop_height_")],
            key=lambda name: int(name.split("_")[-1])
        )
        lh_idx = tuple(pidx[name] for name in lh_params)
        super().__setattr__('_lh_idx', lh_idx)

        # Load schema config
        schema_path = os.path.join(project_root, 'state_schema.json')
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
        extra_saved_keys = (
            'app_mode', 'loop_heights', 'scanner_layout_pattern', 'scanner_color_mode', 'ui_theme',
            'scanner_random_seed', 'scanner_pattern_density', 'scanner_pattern_rows', 'scanner_pattern_cols',
            'scanner_pattern_repeat_rows', 'scanner_pattern_repeat_cols', 'scanner_capture_mode',
            'scanner_camera_workflow', 'scanner_single_row', 'scanner_single_col', 'scanner_single_angle', 'scanner_camera_zoom',
        )
        super().__setattr__('saved_state_keys', tuple(dict.fromkeys((*self.saved_state_keys, *extra_saved_keys))))

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
            with open(os.path.join(project_root, 'initial_params.json'), 'r') as f:
                initial_data = json.load(f)
            initial_bitmap = np.asarray(initial_data.get('bitmap', []), dtype=np.float32)
            if initial_bitmap.ndim == 2 and initial_bitmap.size:
                initial_pattern_rows, initial_pattern_cols = [int(v) for v in initial_bitmap.shape]
        except Exception:
            pass

        # 5. Overlay computed defaults depending on config.json or runtime pathing
        computed_defaults = {
            'save_path': os.path.join(project_root, 'params.json'),
            'load_path': os.path.join(project_root, 'params.json'),
            'params': [p['initial'] for p in config_data['knit_parameters']['parameters']],
            'bitmap': np.ones((3, config_data['knit_parameters']['bitmap_loops']), dtype=np.float32),
            'bitmap_size': np.array([3, config_data['knit_parameters']['bitmap_loops']], dtype=np.int32),
            'display_copies': np.array([0, 0], dtype=np.int32),
            'app_mode': 'edit',
            'scanner_layout_pattern': 'grid',
            'scanner_random_seed': 1,
            'scanner_pattern_density': 0.62,
            'scanner_pattern_rows': initial_pattern_rows,
            'scanner_pattern_cols': initial_pattern_cols,
            'scanner_pattern_repeat_rows': 3,
            'scanner_pattern_repeat_cols': 3,
            'scanner_capture_mode': 'natural',
            'scanner_camera_workflow': 'path',
            'scanner_single_row': 1,
            'scanner_single_col': 1,
            'scanner_single_angle': 1,
            'scanner_camera_zoom': 1.0,
            'ui_theme': 'dark',
            'loop_heights': np.full((3, config_data['knit_parameters']['bitmap_loops']), 3.0, dtype=np.float32),
            'mesh_center': np.zeros(3, dtype=np.float32),
            'row_colors': [
                list(config_data['knit_parameters']['yarn_colors'][i % len(config_data['knit_parameters']['yarn_colors'])])
                for i in range(3)
            ],
            'row_visible': np.ones(3, dtype=bool),
            'mi_cam_dist_mult': float(config_data['rendering']['camera_dist_mult']),
            'mi_cam_fov': float(config_data['rendering']['camera_fov']),
            'period_offset': np.array([float(config_data['knit_parameters']['bitmap_loops']), 0.0, 0.0], dtype=np.float32),
        }
        for k, v in computed_defaults.items():
            state_defaults[k] = AppState._clone(_coerce(v))

        # 6. Direct assignments for types that _coerce would get wrong
        state_defaults['_row_starts'] = [0]

        super().__setattr__('_state_defaults', state_defaults)
        self._data = {k: self._clone(v) for k, v in state_defaults.items()}
        super().__setattr__('scanner_process', None)
        super().__setattr__('scanner_status', 'Scanner idle')
        super().__setattr__('scanner_started_at', 0.0)

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

    def __setattr__(self, name, value):
        if name in ('camera', 'optimizer', 'renderer', '_data') or name in self.__dict__:
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

    def _scanner_template(self):
        path = os.path.join(self.project_root, 'initial_params.json')
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
        cached = getattr(self, '_scanner_template_cache', None)
        if cached is not None and getattr(self, '_scanner_template_mtime', None) == mtime:
            return cached

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
        except Exception:
            pass

        template = {
            'params': params,
            'bitmap': bitmap,
            'loop_heights': loop_heights,
            'gui_state': gui_state,
        }
        super().__setattr__('_scanner_template_cache', template)
        super().__setattr__('_scanner_template_mtime', mtime)
        return template

    def _scanner_template_params(self):
        return np.asarray(self._scanner_template()['params'], dtype=np.float32).copy()

    def _scanner_default_loop_height_for_row(self, params, row_idx):
        if not self._lh_idx:
            return 0.0
        idx = self._lh_idx[min(int(row_idx), len(self._lh_idx) - 1)]
        return float(params[idx])

    def _scanner_random_bitmap(self, cell_index):
        template_bitmap = np.asarray(self._scanner_template().get('bitmap', np.ones((2, 2))), dtype=np.float32)
        if template_bitmap.ndim == 2 and template_bitmap.size:
            pattern_rows, pattern_cols = [max(2, int(v)) for v in template_bitmap.shape]
        else:
            pattern_rows = max(2, int(self.get('scanner_pattern_rows', max(2, int(self.bitmap_size[0])))))
            pattern_cols = max(2, int(self.get('scanner_pattern_cols', max(2, int(self.bitmap_size[1])))))
        density = float(np.clip(float(self.get('scanner_pattern_density', 0.62)), 0.05, 0.95))
        seed = int(self.get('scanner_random_seed', 1)) + int(cell_index) * 9973
        rng = np.random.default_rng(seed)
        bitmap = (rng.random((pattern_rows, pattern_cols)) < density).astype(np.float32)
        if not np.any(bitmap > 0.5):
            bitmap[rng.integers(0, pattern_rows), rng.integers(0, pattern_cols)] = 1.0
        return bitmap

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
        return heights * (bitmap > 0.5)

    def _fit_vl_to_bounds(self, vl, target_bounds):
        if not vl:
            return vl
        src_bounds = self._display_mesh_bounds(vl)
        if src_bounds is None or target_bounds is None:
            return vl
        src_min, src_max = src_bounds
        dst_min, dst_max = target_bounds
        src_center = (src_min + src_max) * 0.5
        dst_center = (dst_min + dst_max) * 0.5
        src_span = np.maximum(src_max - src_min, 1e-6)
        dst_span = np.maximum(dst_max - dst_min, 1e-6)
        scale = float(min(dst_span[0] / src_span[0], dst_span[1] / src_span[1]))
        fitted = []
        for verts, n_points in vl:
            verts = (np.asarray(verts, dtype=np.float32) - src_center) * scale + dst_center
            fitted.append((verts.astype(np.float32), int(n_points)))
        return fitted

    def _scanner_pattern_meshes(self, cell_index, target_bounds):
        bitmap = self._scanner_random_bitmap(cell_index)
        loop_heights = self._scanner_loop_heights_for_bitmap(bitmap)
        params = self._scanner_template_params()
        ctrl_rows = build_parametric_control_rows(
            params,
            bitmap,
            self._pidx,
            self._lh_idx,
            self.samples_per_loop,
            loop_heights=loop_heights,
        )
        period_offset = np.array([float(bitmap.shape[1]), 0.0, 0.0], dtype=np.float32)
        radius = max(float(params[self._pidx['radius']]), 1e-6)
        radius_profiles = [np.full(len(row), radius, dtype=np.float32) for row in ctrl_rows]
        vl = build_spline_mesh(
            ctrl_rows,
            params,
            self.config,
            self._pidx,
            period_offset,
            radius_ctrl_rows=radius_profiles,
        )
        vl, meta = build_surface_fiber_meshes(
            vl,
            segments=int(self.config['knit_parameters']['segments']),
            enabled=self.fiber_geometry_enabled,
            count=max(1, int(round(float(self.fiber_geometry_count)))),
            radius=radius,
            radius_scale=float(self.fiber_geometry_radius_scale),
            lift=float(self.fiber_geometry_lift),
            surface_arc=float(self.fiber_geometry_surface_arc),
            randomness=float(self.fiber_geometry_randomness),
            twist=float(self.fiber_geometry_twist),
        )
        return self._fit_vl_to_bounds(vl, target_bounds), meta

    def build_display_meshes_precise(self, verts_list, faces_list, meta):
        if not verts_list:
            return [], [], []
        radius = max(float(self.params[self._pidx['radius']]), 1e-6)
        row_count = max(1, int(self.bitmap_size[0]))
        x_period = self._display_copy_x_period(verts_list, radius)
        y_period = self._display_copy_y_period(verts_list, radius)
        depth_gap = max(float(radius) * 2.4, 1e-6)
        z_period = self._display_copy_z_period(verts_list, depth_gap)
        scanner_preview = bool(self.get('scanner_preview_grid_enabled', False))
        if scanner_preview:
            bounds = self._display_mesh_bounds(verts_list)
            if bounds is not None:
                min_v, max_v = bounds
                model_w = float(max_v[0] - min_v[0])
                model_h = float(max_v[1] - min_v[1])
                x_period = max(model_w - radius * 2.25, radius)
                y_period = max(model_h - radius * 7.25, radius)
        
        seg = int(self.config['knit_parameters']['segments'])
        if scanner_preview:
            x_tiles = list(range(max(1, int(self.get('scanner_preview_cols', 1)))))
            y_tiles = list(range(max(1, int(self.get('scanner_preview_rows', 1)))))
        else:
            x_tiles = list(range(-int(self.display_copies[0]), int(self.display_copies[0]) + 1))
            y_tiles = list(range(-int(self.display_copies[1]), int(self.display_copies[1]) + 1))

        display_vl, display_fl, display_meta = [], [], []
        if scanner_preview:
            scanner_cols = max(1, int(self.get('scanner_preview_cols', 1)))
            scanner_rows = max(1, int(self.get('scanner_preview_rows', 1)))
            layout_pattern = str(self.get('scanner_layout_pattern', 'grid'))
            base_bounds = self._display_mesh_bounds(verts_list)
            for y_tile in range(scanner_rows):
                y_translation = np.array([0.0, y_tile * y_period, -y_tile * z_period], dtype=np.float32)
                row_offset = 0.5 * x_period if layout_pattern == 'staggered' and (y_tile % 2 == 1) else 0.0
                for x_tile in range(scanner_cols):
                    cell_index = y_tile * scanner_cols + x_tile
                    x_translation = np.array([x_tile * x_period + row_offset, 0.0, 0.0], dtype=np.float32)
                    if base_bounds is not None:
                        tile_bounds = (
                            base_bounds[0] + np.array([radius * 0.10, radius * 0.20, 0.0], dtype=np.float32),
                            base_bounds[1] - np.array([radius * 0.10, radius * 0.20, 0.0], dtype=np.float32),
                        )
                        cell_parts, cell_meta = self._scanner_pattern_meshes(cell_index, tile_bounds)
                        for curve_idx, ((verts, n_points), part_meta) in enumerate(zip(cell_parts, cell_meta)):
                            base_row = int(part_meta.get('row', curve_idx)) % row_count
                            translated = np.asarray(verts, dtype=np.float32) + x_translation + y_translation
                            display_vl.append((translated.astype(np.float32), int(n_points)))
                            display_fl.extend(compute_knitting_faces(seg, [(translated, int(n_points))]))
                            display_meta.append({
                                'row': cell_index * row_count + base_row,
                                'base_row': base_row,
                                'tile_x': x_tile,
                                'tile_y': y_tile,
                                'scanner_cell': cell_index,
                            })
                        continue

                    for (verts, n_points), _faces, part_meta in zip(verts_list, faces_list, meta):
                        base_row = int(part_meta.get('row', 0))
                        translated = np.asarray(verts, dtype=np.float32) + x_translation + y_translation
                        display_vl.append((translated.astype(np.float32), int(n_points)))
                        display_fl.extend(compute_knitting_faces(seg, [(translated, int(n_points))]))
                        copied_meta = dict(part_meta)
                        copied_meta['row'] = cell_index * row_count + base_row
                        copied_meta['base_row'] = base_row
                        copied_meta['tile_x'] = x_tile
                        copied_meta['tile_y'] = y_tile
                        copied_meta['scanner_cell'] = cell_index
                        display_meta.append(copied_meta)
            return display_vl, display_fl, display_meta

        for y_tile in y_tiles:
            y_translation = np.array([0.0, y_tile * y_period, -y_tile * z_period], dtype=np.float32)
            for (verts, n_points), _faces, part_meta in zip(verts_list, faces_list, meta):
                rings = np.asarray(verts, dtype=np.float32).reshape(int(n_points), seg, 3)
                stitched_rings = []
                for tile_i, x_tile in enumerate(x_tiles):
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
                copied_meta['tile_x'] = 0
                copied_meta['tile_y'] = y_tile
                copied_meta['stitched_x_copies'] = len(x_tiles)
                display_meta.append(copied_meta)
        return display_vl, display_fl, display_meta

    def _x_copy_period_for_ctrl_row(self, row, radius):
        row = np.asarray(row, dtype=np.float32)
        if len(row) > 1:
            period = row[-1] - row[0]
            if abs(float(period[0])) > max(float(radius), 1e-6) * 0.25:
                return np.array([float(period[0]), 0.0, 0.0], dtype=np.float32)
        return np.array([self._display_copy_x_period([], radius), 0.0, 0.0], dtype=np.float32)

    def _extended_x_copy_rows(self, radius_profiles):
        copies_x = int(self.display_copies[0])
        if copies_x <= 0 or not self.ctrl_rows:
            return self.ctrl_rows, radius_profiles, 0

        radius = max(float(self.params[self._pidx['radius']]), 1e-6)
        x_tiles = list(range(-copies_x, copies_x + 1))
        extended_rows = []
        extended_radius_rows = []
        for row_idx, row in enumerate(self.ctrl_rows):
            row = np.asarray(row, dtype=np.float32)
            if len(row) == 0:
                extended_rows.append(row)
                extended_radius_rows.append(np.asarray(radius_profiles[row_idx], dtype=np.float32))
                continue
            row_parts = []
            rad_parts = []
            radius_row = np.asarray(radius_profiles[row_idx], dtype=np.float32)
            period = self._x_copy_period_for_ctrl_row(row, radius)
            for tile_i, tile in enumerate(x_tiles):
                row_tile = row + tile * period
                rad_tile = radius_row
                if tile_i > 0 and len(row_tile) > 1:
                    row_tile = row_tile[1:]
                    rad_tile = rad_tile[1:]
                row_parts.append(row_tile)
                rad_parts.append(rad_tile)
            extended_rows.append(np.concatenate(row_parts, axis=0).astype(np.float32))
            extended_radius_rows.append(np.concatenate(rad_parts, axis=0).astype(np.float32))
        return extended_rows, extended_radius_rows, copies_x

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
        period = np.asarray(self.period_offset, dtype=np.float32).reshape(-1) if hasattr(self, 'period_offset') else np.array([])
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

    def _display_copy_x_period_for_rings(self, rings, fallback_period, radius):
        centers = np.asarray(rings, dtype=np.float32).mean(axis=1)
        if len(centers) < 2:
            return np.array([fallback_period, 0.0, 0.0], dtype=np.float32)

        period = centers[-1] - centers[0]
        if abs(float(period[0])) < max(float(radius), 1e-6) * 0.25:
            period = np.array([fallback_period, 0.0, 0.0], dtype=np.float32)
        else:
            # X copies should continue the curve horizontally; keep Y/Z fixed so
            # edited or twisted rows do not drift upward/downward between tiles.
            period = np.array([period[0], 0.0, 0.0], dtype=np.float32)
        return period.astype(np.float32)

    def _display_copy_y_period(self, verts_list, radius):
        if self.ctrl_rows:
            row_centers = np.array([
                float(np.mean(row[:, 1]))
                for row in self.ctrl_rows
                if len(row)
            ], dtype=np.float32)
            if len(row_centers) > 1:
                row_pitch = abs(float(np.median(np.diff(np.sort(row_centers)))))
                if row_pitch > 1e-6:
                    return max(row_pitch * max(1, len(row_centers) - 1), radius)

        bounds = self._display_mesh_bounds(verts_list)
        if bounds is None:
            return radius
        min_v, max_v = bounds
        model_height = float(max_v[1] - min_v[1])
        return max(model_height * 0.58, radius)

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
            rows = max(1, int(self.get('scanner_preview_rows', 1)))
            cols = max(1, int(self.get('scanner_preview_cols', 1)))
            base_rows = max(1, int(self.bitmap_size[0]))
            colors = []
            for _cell in range(rows * cols):
                for base_row in range(base_rows):
                    src = base_colors[base_row % len(base_colors)]
                    colors.append([float(src[0]), float(src[1]), float(src[2])])
            return colors or base_colors
        return base_colors


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

    def rebuild_spline_mesh(self, preserve_model_placement=True):
        old_center = np.asarray(self.mesh_center, dtype=np.float32).copy()
        old_model_t = np.asarray(self.model_t, dtype=np.float32).copy()
        self._ensure_spline_radius_rows()
        radius_profiles = [np.asarray(row, dtype=np.float32) for row in self.spline_radius_rows]
        vl = build_spline_mesh(
            self.ctrl_rows,
            self.params,
            self.config,
            self._pidx,
            np.asarray(self.period_offset, dtype=np.float32),
            radius_ctrl_rows=radius_profiles,
        )
        fl = compute_knitting_faces(self.config['knit_parameters']['segments'], vl)
        display_vl, display_fl, meta = self.prepare_display_meshes(vl, fl)
        self.renderer.set_meshes(display_vl, display_fl, colors=self.active_colors(), meta=meta)
        self.renderer.set_ctrl_pts(self.flat_pts)
        if preserve_model_placement:
            self.mesh_center = old_center
            self.model_t = old_model_t
        else:
            self._recompute_center(display_vl)

    def _recompute_center(self, display_vl):
        if not display_vl:
            self.mesh_center = np.zeros(3, dtype=np.float32)
        else:
            all_v = np.vstack([v for v, _ in display_vl])
            self.mesh_center = (all_v.min(axis=0) + all_v.max(axis=0)) / 2.0
        self.model_t = np.asarray(self.camera.target, dtype=np.float32) - np.asarray(self.mesh_center, dtype=np.float32)

    def _rebuild_spline_points(self):
        self._row_starts = np.concatenate(([0], np.cumsum([len(r) for r in self.ctrl_rows]))).tolist()
        self.flat_pts = np.concatenate(self.ctrl_rows).astype(np.float32) if self.ctrl_rows else np.empty((0, 3), np.float32)

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
        self.period_offset = self._period_offset_from_ctrl_rows()

    @property
    def flat_pts_all(self):
        if not self.ctrl_rows:
            return np.empty((0, 3), dtype=np.float32)
        virtual_pts = np.array([row[0] + self.period_offset for row in self.ctrl_rows], dtype=np.float32)
        return np.concatenate((self.flat_pts, virtual_pts), axis=0)

    def move_ctrl_pt(self, flat_idx, pos):
        n_real = len(self.flat_pts)
        if flat_idx >= n_real:
            row_idx = flat_idx - n_real
            if 0 <= row_idx < len(self.ctrl_rows):
                self.period_offset = pos - self.ctrl_rows[row_idx][0]
                self.rebuild_spline_mesh()
        else:
            r = np.searchsorted(self._row_starts, flat_idx, side="right") - 1
            if 0 <= r < len(self.ctrl_rows):
                self.ctrl_rows[r][flat_idx - self._row_starts[r]] = pos
                self._rebuild_spline_points()

    def center_model_on_view(self):
        self.model_t = np.asarray(self.camera.target, dtype=np.float32) - np.asarray(self.mesh_center, dtype=np.float32)

    def _default_loop_height_for_row(self, row_idx):
        if not self._lh_idx:
            return 0.0
        idx = self._lh_idx[min(int(row_idx), len(self._lh_idx) - 1)]
        return float(self.params[idx])

    def _sync_loop_heights(self):
        rows, cols = int(self.bitmap_size[0]), int(self.bitmap_size[1])
        existing = np.asarray(self.get('loop_heights', np.empty((0, 0))), dtype=np.float32)
        synced = np.zeros((rows, cols), dtype=np.float32)
        for row_idx in range(rows):
            synced[row_idx, :] = self._default_loop_height_for_row(row_idx)
        if existing.ndim == 2:
            keep_rows = min(rows, existing.shape[0])
            keep_cols = min(cols, existing.shape[1])
            if keep_rows > 0 and keep_cols > 0:
                synced[:keep_rows, :keep_cols] = existing[:keep_rows, :keep_cols]
        self.loop_heights = synced
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

    def rebuild_spline_from_params(self):
        self.ctrl_rows = self._fresh_rebuild_rows()
        self.sync_period_offset_to_model_width()
        base_radius = max(float(self.params[self._pidx['radius']]), 1e-6)
        self.spline_radius_rows = [
            np.full(len(row), base_radius, dtype=np.float32)
            for row in self.ctrl_rows
        ]
        self.param_ref_radius = base_radius
        self._rebuild_spline_points()
        self.param_ref_ctrl_rows = [row.copy() for row in self.ctrl_rows]
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

    def nudge_spline_from_params(self, preserve_bounds=False):
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
        self._rebuild_spline_points()
        self.param_ref_ctrl_rows = [row.copy() for row in target_rows]
        self.sync_period_offset_to_model_width()
        self.rebuild_spline_mesh(preserve_model_placement=True)

    def debug_compare_to_fresh_rebuild(self):
        fresh_rows = self._fresh_rebuild_rows()
        curr_rows = self.ctrl_rows
        if len(curr_rows) != len(fresh_rows):
            return {
                'ok': False,
                'reason': 'row_count_mismatch',
                'curr_rows': len(curr_rows),
                'fresh_rows': len(fresh_rows),
            }
        all_d = []
        for curr, fresh in zip(curr_rows, fresh_rows):
            if curr.shape != fresh.shape:
                return {
                    'ok': False,
                    'reason': 'row_shape_mismatch',
                    'curr_shape': tuple(curr.shape),
                    'fresh_shape': tuple(fresh.shape),
                }
            d = np.linalg.norm(curr - fresh, axis=1)
            all_d.append(d)
        all_d = np.concatenate(all_d) if all_d else np.zeros((0,), dtype=np.float32)
        if all_d.size == 0:
            return {'ok': True, 'mean': 0.0, 'max': 0.0, 'p95': 0.0}
        return {
            'ok': True,
            'mean': float(np.mean(all_d)),
            'max': float(np.max(all_d)),
            'p95': float(np.percentile(all_d, 95.0)),
        }

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

    def snapshot_state(self):
        exclude = {'render_tex', 'render_result', 'undo_stack'}
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

    def restore_snapshot(self, snap):
        for k, v in snap.items():
            if not k.startswith('camera_') and k not in ('ctrl_rows', 'spline_radius_rows', 'label'):
                self._data[k] = self._clone(v)
        for attr in self.camera_attributes:
            setattr(self.camera, attr, self._clone(snap[f'camera_{attr}']))
        self.camera.fov_deg = self.view_fov

        self.ctrl_rows = [row.copy() for row in snap['ctrl_rows']]
        self.spline_radius_rows = [row.copy() for row in snap.get('spline_radius_rows', [])]
        self._ensure_spline_radius_rows()
        self.param_ref_radius = max(float(self.params[self._pidx['radius']]), 1e-6)
        self._rebuild_spline_points()
        self.param_ref_ctrl_rows = self._fresh_rebuild_rows()

        self.rebuild_spline_mesh()

    def undo_last(self):
        if not self.undo_stack:
            return
        self.restore_snapshot(self.undo_stack.pop())
        self.status_msg = 'Undid last change'

    def capture_initial_state(self):
        super().__setattr__('_initial_snapshot', self.snapshot_state())

    def reset_to_initial(self):
        initial_snapshot = getattr(self, '_initial_snapshot', None)
        if initial_snapshot is not None:
            self.push_undo("Reset all")
            self.restore_snapshot(initial_snapshot)
            self.status_msg = 'Reset to saved initial model'
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
            'period_offset', 'app_mode', 'scanner_layout_pattern', 'loop_heights',
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
            'period_offset', 'app_mode', 'scanner_layout_pattern', 'loop_heights',
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

    def on_bitmap_change(self):
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
        self._sync_row_colors(new_rows)
        self._sync_row_visibility(new_rows)
        self._sync_loop_heights()
        self.rebuild_spline_from_params()

    def loop_height_span(self, name):
        if name.startswith("loop_height_"):
            try:
                return int(name[12:])
            except ValueError:
                pass
        return None

    def fit_loop_heights_to_rows(self):
        dy = float(self.params[self._pidx['dy']])
        for span in range(1, self.bitmap_size[0] + 1):
            name = f"loop_height_{span}"
            if name in self._pidx:
                idx = self._pidx[name]
                lo, hi = self.config['knit_parameters']['parameters'][idx]["range"]
                self.params[idx] = float(np.clip(span * dy, lo, hi))
        self.on_bitmap_change()

    # ── PARAMETER SERIALIZATION ───────────────────────────────────────────────

    def save_params(self, path, silent=False):
        params_to_save = {
            name: float(self.params[i])
            for i, name in enumerate(p["name"] for p in self.config['knit_parameters']['parameters'])
            if (span := self.loop_height_span(name)) is None or span <= self.bitmap_size[0]
        }
        gui_state = {k: AppState._json_ready(self._data[k]) for k in self.saved_state_keys if k in self._data}
        for attr in self.camera_attributes:
            gui_state[f'camera_{attr}'] = AppState._json_ready(getattr(self.camera, attr))

        data = {
            'format_version': 2,
            'params': params_to_save,
            'bitmap': self.bitmap.tolist(),
            'period_offset': self.period_offset.tolist(),
            'spline_control_rows': [row.tolist() for row in self.ctrl_rows],
            'spline_radius_rows': [row.tolist() for row in self.spline_radius_rows],
            'gui_state': gui_state,
        }
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            self.save_path  = path
            self.autosave_last_time = time.monotonic()
            if not silent:
                self.status_msg = f'Saved → {os.path.basename(path)}'
        except Exception as e:
            self.status_msg = f'Save error: {e}'

    def maybe_autosave(self):
        if not self.autosave_enabled:
            return
        now = time.monotonic()
        if now - float(self.autosave_last_time) < float(self.autosave_interval_sec):
            return
        target_path = self.save_path or os.path.join(self.project_root, 'params.json')
        self.save_params(target_path, silent=True)

    def load_params(self, path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            p_dict = data.get('params', {})
            for i, pd in enumerate(self.config['knit_parameters']['parameters']):
                if pd["name"] in p_dict:
                    lo, hi = pd["range"]
                    self.params[i] = float(np.clip(p_dict[pd["name"]], lo, hi))

            if 'bitmap' in data:
                self.bitmap = np.array(data['bitmap'], dtype=np.float32)
                self.bitmap_size = np.array(self.bitmap.shape, dtype=np.int32)

            loaded_spline_rows = data.get('spline_control_rows')
            loaded_radius_rows = data.get('spline_radius_rows')
            gui_state = data.get('gui_state', {})
            get_val = lambda k, d=None: gui_state.get(k, data.get(k, d))

            for key in self.saved_state_keys:
                val = get_val(key)
                if val is not None:
                    default_val = self._state_defaults.get(key)
                    if isinstance(default_val, np.ndarray):
                        arr = np.array(val, dtype=default_val.dtype)
                        if key == 'model_scale' and arr.size == 1:
                            arr = np.array([float(arr.item()), float(arr.item()), float(arr.item())], dtype=default_val.dtype)
                        self._data[key] = arr
                    else:
                        self._data[key] = val

            for attr in self.camera_attributes:
                field = f'camera_{attr}'
                val = get_val(field)
                if val is not None:
                    setattr(self.camera, attr, np.array(val, dtype=float) if isinstance(val, list) else float(val))

            self._sync_loop_heights()

            row_vis = get_val('row_visible')
            if row_vis is not None:
                self.row_visible = np.array(row_vis, dtype=bool)
            elif 'loop_visible' in data:
                self.row_visible = np.any(np.array(data['loop_visible'], dtype=bool), axis=1)

            self._sync_row_colors(self.bitmap_size[0])
            self._sync_row_visibility(self.bitmap_size[0])
            self.mode = 'spline'
            self.camera.fov_deg = self.view_fov
            self.load_path = path
            self.status_msg = f'Loaded ← {os.path.basename(path)}'
            self.period_offset = np.array(data.get('period_offset', [float(self.bitmap_size[1]), 0.0, 0.0]), dtype=np.float32)
            self.rebuild_spline_from_params()
            if loaded_spline_rows:
                self.ctrl_rows = [np.array(row, dtype=np.float32) for row in loaded_spline_rows]
                if loaded_radius_rows and len(loaded_radius_rows) == len(self.ctrl_rows):
                    self.spline_radius_rows = [
                        np.array(row, dtype=np.float32)
                        for row in loaded_radius_rows
                    ]
                else:
                    self.spline_radius_rows = []
                self._ensure_spline_radius_rows()
                self._rebuild_spline_points()
                self.sync_period_offset_to_model_width()
                self.param_ref_ctrl_rows = [row.copy() for row in self._fresh_rebuild_rows()]
                self.rebuild_spline_mesh(preserve_model_placement=False)
        except Exception as e:
            self.status_msg = f"Load error: {e}"
