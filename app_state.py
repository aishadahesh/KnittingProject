import os
import json
import threading
import time
import numpy as np
import jax.numpy as jnp
from PIL import Image

from knitting_core import (
    compute_knitting_vertices, compute_knitting_faces,
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

        super().__setattr__('workflow_stages', tuple(
            (item['title'], item['subtitle'])
            for item in schema['workflow_stages']
        ))
        super().__setattr__('texture_control_groups', tuple(schema['texture_control_groups']))
        super().__setattr__('texture_preset_buttons', tuple(schema['texture_preset_buttons']))
        super().__setattr__('texture_presets', dict(schema['texture_presets']))
        super().__setattr__('texture_param_keys', tuple(schema['texture_param_keys']))
        super().__setattr__('material_uniform_aliases', dict(schema['material_uniform_aliases']))
        super().__setattr__('saved_state_keys', tuple(schema['saved_state_keys']))
        super().__setattr__('camera_attributes', ('dist', 'az', 'el', 'target'))

        overlay_defaults = dict(schema.get('overlay_defaults', {}))

        default_state_config = {
            'ui': {
                'workflow_step': 0,
                'mode': 'spline',
                'hover_idx': -1,
                'selected_idx': -1,
                'status_msg': '',
                'save_path': os.path.join(project_root, 'params.json'),
                'load_path': os.path.join(project_root, 'params.json'),
                'autosave_enabled': True,
                'autosave_interval_sec': 1.0,
                'autosave_last_time': 0.0,
                'undo_stack': [],
                'max_undo': 40,
            },
            'geometry': {
                'params': [p['initial'] for p in config_data['knit_parameters']['parameters']],
                'bitmap': np.ones((3, config_data['knit_parameters']['bitmap_loops']), dtype=np.float32),
                'bitmap_size': np.array([3, config_data['knit_parameters']['bitmap_loops']], dtype=np.int32),
                'samples_per_loop': 5,
                'display_copies': np.array([0, 0], dtype=np.int32),
                'mesh_center': np.zeros(3, dtype=np.float32),
                'ctrl_rows': [],
                'flat_pts': np.empty((0, 3), np.float32),
                '_row_starts': [0],
                'auto_fix_spline_endpoints': True,
                'spline_point_drag_active': False,
                'spline_keyboard_edit_active': False,
                'spline_keyboard_step': 0.025,
                'fiber_geometry_enabled': False,
                'fiber_geometry_count': 4,
                'fiber_geometry_radius_scale': 0.18,
                'fiber_geometry_lift': 0.0,
                'fiber_geometry_surface_arc': 0.55,
                'fiber_geometry_randomness': 0.18,
                'fiber_geometry_twist': 0.0,
            },
            'viewport': {
                'mouse_in_vp': False,
                'vp_origin': np.array([0.0, 0.0], dtype=np.float32),
                'vp_scale': 1.0,
                'viewport_zoom': 1.0,
                'viewport_pan': np.array([0.0, 0.0], dtype=np.float32),
                'hover_mesh_idx': -1,
                'selected_mesh_idx': -1,
                'view_fov': 45.0,
                'model_rot': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'model_scale': np.array([1.0, 1.0, 1.0], dtype=np.float32),
                'model_rot_dragging': False,
                'model_t': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'model_drag_undo_active': False,
                'gizmo_edit_active': False,
            },
            'overlay': {
                'show_ref_bg': bool(overlay_defaults.get('show_ref_bg', False)),
                'ref_bg_alpha': float(overlay_defaults.get('ref_bg_alpha', 0.5)),
                'ref_bg_scale': np.array(overlay_defaults.get('ref_bg_scale', [1.0, 1.0]), dtype=np.float32),
                'ref_bg_lock_dimensions': bool(overlay_defaults.get('ref_bg_lock_dimensions', True)),
                'ref_bg_lock_zoom': bool(overlay_defaults.get('ref_bg_lock_zoom', False)),
                'ref_bg_rotation': float(overlay_defaults.get('ref_bg_rotation', 0.0)),
                'ref_bg_offset': np.array(overlay_defaults.get('ref_bg_offset', [0.0, 0.0]), dtype=np.float32),
            },
            'material': {
                'model_alpha': 1.0,
                'single_model_color': np.array([0.85, 0.12, 0.10], dtype=np.float32),
                'use_row_colors': False,
                'row_colors': [
                    list(config_data['knit_parameters']['yarn_colors'][i % len(config_data['knit_parameters']['yarn_colors'])])
                    for i in range(3)
                ],
                'row_visible': np.ones(3, dtype=bool),
                'render_texture_color': np.array([0.8, 0.4, 0.3], dtype=np.float32),
                **{k: v for k, v in self.texture_presets['clear'].items() if k != 'render_texture_color'},
                **self.texture_presets['soft_yarn'],
            },
            'rendering': {
                'render_light_color': np.array([1.0, 1.0, 1.0], dtype=np.float32),
                'render_light_intensity': 0.9,
                'mi_cam_dist_mult': float(config_data['rendering']['camera_dist_mult']),
                'mi_cam_fov': float(config_data['rendering']['camera_fov']),
            }
        }

        state_defaults = {
            key: value
            for section in default_state_config.values()
            for key, value in section.items()
        }
        super().__setattr__('_state_defaults', state_defaults)

        self._data = {k: self._clone(v) for k, v in state_defaults.items()}

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

    def build_display_meshes_precise(self, verts_list, faces_list, meta):
        if not verts_list:
            return [], [], []
        x_period = max(float(self.bitmap_size[1]), 1e-6)
        y_period = max(float(self.bitmap_size[0]) * abs(float(self.params[self._pidx['dy']])), 1e-6)
        display_vl, display_fl, display_meta = [], [], []
        for y_tile in range(-int(self.display_copies[1]), int(self.display_copies[1]) + 1):
            for x_tile in range(-int(self.display_copies[0]), int(self.display_copies[0]) + 1):
                translation = np.array([x_tile * x_period, y_tile * y_period, 0.0], dtype=np.float32)
                for (verts, n_points), faces, part_meta in zip(verts_list, faces_list, meta):
                    display_vl.append((np.asarray(verts, dtype=np.float32) + translation, n_points))
                    display_fl.append(faces)
                    display_meta.append(part_meta)
        return display_vl, display_fl, display_meta

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
        return self.row_colors if self.use_row_colors else [self.single_model_color]

    def rebuild_param_mesh(self):
        vl = compute_knitting_vertices(self.params, self.bitmap, self.config, self._pidx, self._lh_idx)
        fl = compute_knitting_faces(self.config['knit_parameters']['segments'], vl)
        display_vl, display_fl, meta = self.prepare_display_meshes(vl, fl)
        self.renderer.set_meshes(display_vl, display_fl, colors=self.active_colors(), meta=meta)
        self.renderer.set_ctrl_pts([])
        self._recompute_center(display_vl)

    def rebuild_spline_mesh(self):
        vl = build_spline_mesh(self.ctrl_rows, self.params, self.config, self._pidx, self.bitmap_size[1])
        fl = compute_knitting_faces(self.config['knit_parameters']['segments'], vl)
        display_vl, display_fl, meta = self.prepare_display_meshes(vl, fl)
        self.renderer.set_meshes(display_vl, display_fl, colors=self.active_colors(), meta=meta)
        self.renderer.set_ctrl_pts(self.flat_pts)
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

    def move_ctrl_pt(self, flat_idx, pos):
        r = np.searchsorted(self._row_starts, flat_idx, side="right") - 1
        if 0 <= r < len(self.ctrl_rows):
            self.ctrl_rows[r][flat_idx - self._row_starts[r]] = pos
            self._rebuild_spline_points()

    def center_model_on_view(self):
        self.model_t = np.asarray(self.camera.target, dtype=np.float32) - np.asarray(self.mesh_center, dtype=np.float32)

    def _fresh_rebuild_rows(self):
        return build_parametric_control_rows(self.params, self.bitmap, self._pidx, self._lh_idx, self.samples_per_loop)

    def rebuild_spline_from_params(self):
        self.ctrl_rows = self._fresh_rebuild_rows()
        self._rebuild_spline_points()
        self.param_ref_ctrl_rows = [row.copy() for row in self.ctrl_rows]
        self.rebuild_spline_mesh()

    def nudge_spline_from_params(self):
        target_rows = self._fresh_rebuild_rows()
        ref_rows = self.get('param_ref_ctrl_rows')
        if not ref_rows or len(self.ctrl_rows) != len(target_rows) or any(c.shape != t.shape for c, t in zip(self.ctrl_rows, target_rows)):
            self.ctrl_rows = [row.copy() for row in target_rows]
        else:
            nudged_rows = []
            for current_row, ref_row, target_row in zip(self.ctrl_rows, ref_rows, target_rows):
                nudged_rows.append(current_row + (target_row - ref_row))
            self.ctrl_rows = nudged_rows
        self._rebuild_spline_points()
        self.param_ref_ctrl_rows = [row.copy() for row in target_rows]
        self.rebuild_spline_mesh()

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
        from rendering import rotation_matrix_xyz
        model_rot = rotation_matrix_xyz(*self.model_rot)
        model_s = np.eye(4, dtype=np.float32)
        model_s[0, 0] = 1.0
        model_s[1, 1] = 1.0
        model_s[2, 2] = 1.0
        tneg, tpos, tuser = np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)
        tneg[:3, 3] = -self.mesh_center
        tpos[:3, 3] = self.mesh_center
        tuser[:3, 3] = self.model_t
        return (tuser @ tpos @ model_rot @ model_s @ tneg).astype(np.float32)

    # ── UNDO / REDO ───────────────────────────────────────────────────────────

    def snapshot_state(self):
        exclude = {'render_tex', 'render_result', 'undo_stack'}
        snap = {k: self._clone(v) for k, v in self._data.items() if k not in exclude}
        for attr in self.camera_attributes:
            snap[f'camera_{attr}'] = self._clone(getattr(self.camera, attr))
        snap['ctrl_rows'] = [row.copy() for row in self.ctrl_rows]
        return snap

    def push_undo(self, label):
        snap = self.snapshot_state()
        snap['label'] = label
        self.undo_stack.append(snap)
        if len(self.undo_stack) > self.max_undo:
            del self.undo_stack[0]

    def restore_snapshot(self, snap):
        for k, v in snap.items():
            if not k.startswith('camera_') and k not in ('ctrl_rows', 'label'):
                self._data[k] = self._clone(v)
        for attr in self.camera_attributes:
            setattr(self.camera, attr, self._clone(snap[f'camera_{attr}']))
        self.camera.fov_deg = self.view_fov

        self.ctrl_rows = [row.copy() for row in snap['ctrl_rows']]
        self._rebuild_spline_points()
        self.param_ref_ctrl_rows = self._fresh_rebuild_rows()

        self.rebuild_spline_mesh()

    def undo_last(self):
        if not self.undo_stack:
            return
        self.restore_snapshot(self.undo_stack.pop())
        self.status_msg = 'Undid last change'

    def apply_texture_preset(self, preset_name, undo_label="Texture preset"):
        if preset_name in self.texture_presets:
            self.push_undo(undo_label)
            for k, val in self.texture_presets[preset_name].items():
                setattr(self, k, np.array(val, dtype=np.float32) if isinstance(val, list) else val)

    # ── BITMAP / WORKFLOW UPDATES ─────────────────────────────────────────────

    def on_bitmap_change(self):
        self.rebuild_spline_from_params()

    def on_bitmap_resize(self, new_rows, new_cols):
        new_rows = max(1, min(int(new_rows), int(self.config['knit_parameters']['bitmap_rows'])))
        new_cols = max(1, min(int(new_cols), 16))
        old = self.bitmap
        new_bm = np.ones((new_rows, new_cols), dtype=np.float32)
        new_bm[:min(old.shape[0], new_rows), :min(old.shape[1], new_cols)] = old[:min(old.shape[0], new_rows), :min(old.shape[1], new_cols)]
        self.bitmap = new_bm
        self.bitmap_size = np.array([new_rows, new_cols], dtype=np.int32)
        self._sync_row_colors(new_rows)
        self._sync_row_visibility(new_rows)
        self.on_bitmap_change()

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
            'spline_control_rows': [row.tolist() for row in self.ctrl_rows],
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

            self.rebuild_spline_from_params()
            if loaded_spline_rows:
                self.ctrl_rows = [np.array(row, dtype=np.float32) for row in loaded_spline_rows]
                self._rebuild_spline_points()
                self.param_ref_ctrl_rows = [row.copy() for row in self._fresh_rebuild_rows()]
                self.rebuild_spline_mesh()
        except Exception as e:
            self.status_msg = f"Load error: {e}"
