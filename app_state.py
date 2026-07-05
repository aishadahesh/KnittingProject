import os
import json
import threading
import time
import numpy as np
import jax.numpy as jnp
from PIL import Image

from knitting_core import (
    CONFIG,
    LOOP_HEIGHT_PARAM_INDICES, compute_knitting_vertices, compute_knitting_faces,
    geometry_param_index, geometry_param_range, geometry_parameter_names,
    initial_geometry_params, save_combined_obj, run_optimization_loop, PROJECT_ROOT
)

# %% STATE CONSTRAINTS ─────────────────────────────────────────────────────────

def _load_state_schema(filename='state_schema.json'):
    schema_path = os.path.join(PROJECT_ROOT, filename)
    with open(schema_path, 'r') as handle:
        return json.load(handle)


STATE_SCHEMA = _load_state_schema()

WORKFLOW_STAGES = tuple(
    (item['title'], item['subtitle'])
    for item in STATE_SCHEMA['workflow_stages']
)

TEXTURE_CONTROL_GROUPS = tuple(STATE_SCHEMA['texture_control_groups'])
TEXTURE_PRESET_BUTTONS = tuple(STATE_SCHEMA['texture_preset_buttons'])
TEXTURE_PRESETS = dict(STATE_SCHEMA['texture_presets'])

TEXTURE_PARAM_KEYS = tuple(STATE_SCHEMA['texture_param_keys'])
MATERIAL_UNIFORM_ALIASES = dict(STATE_SCHEMA['material_uniform_aliases'])
SAVED_STATE_KEYS = tuple(STATE_SCHEMA['saved_state_keys'])
OVERLAY_DEFAULTS = dict(STATE_SCHEMA.get('overlay_defaults', {}))

CAMERA_ATTRIBUTES = ('dist', 'az', 'el', 'target')

DEFAULT_STATE_CONFIG = {
    # ── Workflow / UI State ───────────────────────────────────────────────────
    'ui': {
        'workflow_step': 0,
        'mode': 'parameter',
        'hover_idx': -1,
        'selected_idx': -1,
        'status_msg': '',
        'save_path': os.path.join(PROJECT_ROOT, 'params.json'),
        'load_path': os.path.join(PROJECT_ROOT, 'params.json'),
        'autosave_enabled': True,
        'autosave_interval_sec': 1.0,
        'autosave_last_time': 0.0,
        'undo_stack': [],
        'max_undo': 40,
    },
    # ── Geometry & Knitting Parameters ────────────────────────────────────────
    'geometry': {
        'params': initial_geometry_params(),
        'bitmap': np.ones((3, CONFIG['geometry']['bitmap_loops']), dtype=np.float32),
        'bitmap_size': np.array([3, CONFIG['geometry']['bitmap_loops']], dtype=np.int32),
        'samples_per_loop': 5,
        'display_copies': np.array([0, 0], dtype=np.int32),
        'mesh_center': np.zeros(3, dtype=np.float32),
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
    # ── Viewport, Camera & Interaction ────────────────────────────────────────
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
        'model_scale': 1.0,
        'model_rot_dragging': False,
        'model_t': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        'model_drag_undo_active': False,
        'gizmo_edit_active': False,
    },
    # ── Reference Overlay ─────────────────────────────────────────────────────
    'overlay': {
        'show_ref_bg': bool(OVERLAY_DEFAULTS.get('show_ref_bg', False)),
        'ref_bg_alpha': float(OVERLAY_DEFAULTS.get('ref_bg_alpha', 0.5)),
        'ref_bg_scale': np.array(OVERLAY_DEFAULTS.get('ref_bg_scale', [1.0, 1.0]), dtype=np.float32),
        'ref_bg_lock_dimensions': bool(OVERLAY_DEFAULTS.get('ref_bg_lock_dimensions', True)),
        'ref_bg_lock_zoom': bool(OVERLAY_DEFAULTS.get('ref_bg_lock_zoom', False)),
        'ref_bg_rotation': float(OVERLAY_DEFAULTS.get('ref_bg_rotation', 0.0)),
        'ref_bg_offset': np.array(OVERLAY_DEFAULTS.get('ref_bg_offset', [0.0, 0.0]), dtype=np.float32),
    },
    # ── Material & Appearance ─────────────────────────────────────────────────
    'material': {
        'model_alpha': 1.0,
        'single_model_color': np.array([0.85, 0.12, 0.10], dtype=np.float32),
        'use_row_colors': False,
        'row_colors': [
            list(CONFIG['ui']['yarn_colors'][i % len(CONFIG['ui']['yarn_colors'])])
            for i in range(3)
        ],
        'row_visible': np.ones(3, dtype=bool),
        'render_texture_color': np.array([0.8, 0.4, 0.3], dtype=np.float32),
        # Texture parameters populated directly from clear base + soft_yarn preset
        **{k: v for k, v in TEXTURE_PRESETS['clear'].items() if k != 'render_texture_color'},
        **TEXTURE_PRESETS['soft_yarn'],
    },
    # ── Mitsuba Rendering & Optimization ──────────────────────────────────────
    'mitsuba': {
        'is_rendering': False,
        'is_optimizing': False,
        'render_result': None,
        'pending_tex': False,
        'render_tex': None,
        'render_light_color': np.array([1.0, 1.0, 1.0], dtype=np.float32),
        'render_light_intensity': 0.9,
        'mi_cam_dist_mult': float(CONFIG['rendering']['camera_dist_mult']),
        'mi_cam_fov': float(CONFIG['rendering']['camera_fov']),
    }
}

def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    return value

def _clone(v):
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, list):
        return [_clone(x) for x in v]
    if isinstance(v, dict):
        return {k: _clone(x) for k, x in v.items()}
    return v


STATE_DEFAULTS = {
    key: value
    for section in DEFAULT_STATE_CONFIG.values()
    for key, value in section.items()
}


# %% APP STATE ─────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self, camera, spline, optimizer, renderer):
        self.camera = camera
        self.spline = spline
        self.optimizer = optimizer
        self.renderer = renderer
        self._data = {k: _clone(v) for k, v in STATE_DEFAULTS.items()}

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
        if name in ('camera', 'spline', 'optimizer', 'renderer', '_data'):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    # ── GET UNIFORM MAPS ──────────────────────────────────────────────────────

    def get_material_uniforms(self):
        """Returns standard material properties as a dict mapped to shader uniform names."""
        uniforms = {k: self._data[k] for k in TEXTURE_PARAM_KEYS}
        uniforms.update({
            uniform_name: self._data[state_key]
            for state_key, uniform_name in MATERIAL_UNIFORM_ALIASES.items()
        })
        return uniforms

    # ── MESH GENERATION HELPERS ───────────────────────────────────────────────

    def _sync_row_colors(self, n_rows):
        palette = CONFIG['ui']['yarn_colors']
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
        y_period = max(float(self.bitmap_size[0]) * abs(float(self.params[geometry_param_index('dy')])), 1e-6)
        display_vl, display_fl, display_meta = [], [], []
        for y_tile in range(-int(self.display_copies[1]), int(self.display_copies[1]) + 1):
            for x_tile in range(-int(self.display_copies[0]), int(self.display_copies[0]) + 1):
                translation = np.array([x_tile * x_period, y_tile * y_period, 0.0], dtype=np.float32)
                for (verts, n_points), faces, part_meta in zip(verts_list, faces_list, meta):
                    display_vl.append((np.asarray(verts, dtype=np.float32) + translation, n_points))
                    display_fl.append(faces)
                    display_meta.append(part_meta)
        return display_vl, display_fl, display_meta

    def _build_surface_fiber_meshes(self, base_vl):
        if not self.fiber_geometry_enabled:
            return list(base_vl), [{'row': row_idx} for row_idx in range(len(base_vl))]

        seg = int(CONFIG['geometry']['segments'])
        count = max(1, int(round(float(self.fiber_geometry_count))))
        radius = float(self.params[geometry_param_index('radius')])
        fiber_radius = max(radius * float(self.fiber_geometry_radius_scale), 1e-5)
        lift = max(float(self.fiber_geometry_lift), 0.0)
        surface_arc = float(np.clip(self.fiber_geometry_surface_arc, 0.05, 1.0))
        randomness = float(np.clip(self.fiber_geometry_randomness, 0.0, 1.0))
        twist = float(self.fiber_geometry_twist)

        out_vl = []
        meta = []

        for row_idx, (verts, n_points) in enumerate(base_vl):
            verts = np.asarray(verts, dtype=np.float32)
            n_points = int(n_points)
            if n_points < 2 or len(verts) != n_points * seg:
                continue

            rings = verts.reshape(n_points, seg, 3)
            centers = rings.mean(axis=1)
            top_idx = int(np.argmax((rings - centers[:, None, :])[:, :, 2].mean(axis=0)))
            offsets = (
                np.zeros(1, dtype=np.float32)
                if count == 1
                else np.linspace(-0.5, 0.5, count, dtype=np.float32) * surface_arc * float(seg)
            )

            for fiber_idx, offset in enumerate(offsets):
                rng = np.random.default_rng(row_idx * 1009 + fiber_idx * 9173)
                phase_jitter = rng.normal(0.0, 0.35 * randomness)
                lift_jitter = rng.normal(0.0, 0.20 * randomness)
                radius_jitter = float(np.clip(1.0 + rng.normal(0.0, 0.18 * randomness), 0.55, 1.45))
                local_radius = max(fiber_radius * radius_jitter, 1e-5)
                sample_idx = np.mod(
                    top_idx + offset + phase_jitter + twist * np.linspace(0.0, 1.0, n_points, dtype=np.float32) * seg,
                    float(seg),
                )
                lo_float = np.floor(sample_idx)
                lo = lo_float.astype(np.int32) % seg
                hi = (lo + 1) % seg
                frac = (sample_idx - lo_float).astype(np.float32)
                surface = rings[np.arange(n_points), lo] * (1.0 - frac[:, None]) + rings[np.arange(n_points), hi] * frac[:, None]
                radial = surface - centers
                surface_radius = np.linalg.norm(radial, axis=1, keepdims=True)
                radial /= surface_radius + 1e-8
                center_radius = np.maximum(surface_radius - local_radius + local_radius * (lift + lift_jitter), local_radius)
                line = centers + radial * center_radius

                tangent = np.gradient(line, axis=0)
                tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8
                side = np.cross(tangent, radial)
                bad = np.linalg.norm(side, axis=1) < 1e-6
                if np.any(bad):
                    side[bad] = np.cross(tangent[bad], [1.0, 0.0, 0.0])
                side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-8
                normal = np.cross(side, tangent)
                normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8

                angles = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False, dtype=np.float32)
                offsets_ring = (
                    normal[:, None, :] * np.cos(angles)[None, :, None]
                    + side[:, None, :] * np.sin(angles)[None, :, None]
                ) * local_radius
                out_vl.append(((line[:, None, :] + offsets_ring).reshape(-1, 3).astype(np.float32), n_points))
                meta.append({'row': row_idx})

        return out_vl, meta

    def prepare_display_meshes(self, vl, fl):
        vl, meta = self._build_surface_fiber_meshes(vl)
        fl = compute_knitting_faces(CONFIG['geometry']['segments'], vl)
        return self.build_display_meshes_precise(vl, fl, meta)

    def active_colors(self):
        return self.row_colors if self.use_row_colors else [self.single_model_color]

    def rebuild_param_mesh(self):
        vl = compute_knitting_vertices(self.params, self.bitmap)
        fl = compute_knitting_faces(CONFIG['geometry']['segments'], vl)
        display_vl, display_fl, meta = self.prepare_display_meshes(vl, fl)
        self.renderer.set_meshes(display_vl, display_fl, colors=self.active_colors(), meta=meta)
        self.renderer.set_ctrl_pts([])
        self._recompute_center(display_vl)

    def rebuild_spline_mesh(self):
        if self.auto_fix_spline_endpoints:
            self.spline.fix_endpoints(self.params)
        vl = self.spline.build_mesh(self.params)
        fl = compute_knitting_faces(CONFIG['geometry']['segments'], vl)
        display_vl, display_fl, meta = self.prepare_display_meshes(vl, fl)
        self.renderer.set_meshes(display_vl, display_fl, colors=self.active_colors(), meta=meta)
        self.renderer.set_ctrl_pts(self.spline.flat_pts)
        self._recompute_center(display_vl)

    def _recompute_center(self, display_vl):
        if not display_vl:
            return
        all_pts = np.concatenate([v for v, _ in display_vl], axis=0)
        self.mesh_center = ((all_pts.min(axis=0) + all_pts.max(axis=0)) * 0.5).astype(np.float32)

    def center_model_on_view(self):
        self.model_t = np.asarray(self.camera.target, dtype=np.float32) - np.asarray(self.mesh_center, dtype=np.float32)

    def current_model_matrix(self):
        from rendering import rotation_matrix_xyz
        model_rot = rotation_matrix_xyz(*self.model_rot)
        model_s = np.eye(4, dtype=np.float32)
        model_s[0, 0] = float(self.model_scale)
        model_s[1, 1] = float(self.model_scale)
        model_s[2, 2] = float(self.model_scale)
        tneg, tpos, tuser = np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)
        tneg[:3, 3] = -self.mesh_center
        tpos[:3, 3] = self.mesh_center
        tuser[:3, 3] = self.model_t
        return (tuser @ tpos @ model_rot @ model_s @ tneg).astype(np.float32)

    # ── UNDO / REDO ───────────────────────────────────────────────────────────

    def snapshot_state(self):
        exclude = {'render_tex', 'render_result', 'undo_stack'}
        snap = {k: _clone(v) for k, v in self._data.items() if k not in exclude}
        for attr in CAMERA_ATTRIBUTES:
            snap[f'camera_{attr}'] = _clone(getattr(self.camera, attr))
        snap['ctrl_rows'] = [row.copy() for row in self.spline.ctrl_rows]
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
                self._data[k] = _clone(v)
        for attr in CAMERA_ATTRIBUTES:
            setattr(self.camera, attr, _clone(snap[f'camera_{attr}']))
        self.camera.fov_deg = self.view_fov

        self.spline.bitmap = self.bitmap
        self.spline.samples_per_loop = self.samples_per_loop
        self.spline.ctrl_rows = [row.copy() for row in snap['ctrl_rows']]
        self.spline._rebuild()

        self.optimizer.bitmap = jnp.array(self.bitmap)
        if self.mode == 'parameter':
            self.renderer.set_ctrl_pts([])
            self.rebuild_param_mesh()
        else:
            self.rebuild_spline_mesh()

    def undo_last(self):
        if not self.undo_stack:
            return
        self.restore_snapshot(self.undo_stack.pop())
        self.status_msg = 'Undid last change'

    def apply_texture_preset(self, preset_name, undo_label="Texture preset"):
        if preset_name in TEXTURE_PRESETS:
            self.push_undo(undo_label)
            for k, val in TEXTURE_PRESETS[preset_name].items():
                setattr(self, k, np.array(val, dtype=np.float32) if isinstance(val, list) else val)

    # ── BITMAP / WORKFLOW UPDATES ─────────────────────────────────────────────

    def on_bitmap_change(self):
        self.spline.bitmap = self.bitmap
        self.optimizer.bitmap = jnp.array(self.bitmap)
        if self.mode == 'parameter':
            self.rebuild_param_mesh()
        else:
            self.spline.init_from_params(self.params)
            self.rebuild_spline_mesh()

    def on_bitmap_resize(self, new_rows, new_cols):
        new_rows = max(1, min(int(new_rows), int(CONFIG['geometry']['bitmap_rows'])))
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
        dy = float(self.params[geometry_param_index('dy')])
        for span in range(1, self.bitmap_size[0] + 1):
            name = f"loop_height_{span}"
            try:
                idx = geometry_param_index(name)
            except KeyError:
                continue
            else:
                lo, hi = geometry_param_range(idx)
                self.params[idx] = float(np.clip(span * dy, lo, hi))
        self.on_bitmap_change()

    # ── PARAMETER SERIALIZATION ───────────────────────────────────────────────

    def save_params(self, path, silent=False):
        params_to_save = {
            name: float(self.params[i])
            for i, name in enumerate(geometry_parameter_names())
            if (span := self.loop_height_span(name)) is None or span <= self.bitmap_size[0]
        }
        gui_state = {k: json_ready(self._data[k]) for k in SAVED_STATE_KEYS if k in self._data}
        for attr in CAMERA_ATTRIBUTES:
            gui_state[f'camera_{attr}'] = json_ready(getattr(self.camera, attr))

        data = {
            'format_version': 2,
            'params': params_to_save,
            'bitmap': self.bitmap.tolist(),
            'spline_control_rows': [row.tolist() for row in self.spline.ctrl_rows],
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
        target_path = self.save_path or os.path.join(PROJECT_ROOT, 'params.json')
        self.save_params(target_path, silent=True)

    def load_params(self, path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            p_dict = data.get('params', {})
            for i, name in enumerate(geometry_parameter_names()):
                if name in p_dict:
                    lo, hi = geometry_param_range(i)
                    self.params[i] = float(np.clip(p_dict[name], lo, hi))

            if 'bitmap' in data:
                self.bitmap = np.array(data['bitmap'], dtype=np.float32)
                self.bitmap_size = np.array(self.bitmap.shape, dtype=np.int32)

            loaded_spline_rows = data.get('spline_control_rows')
            gui_state = data.get('gui_state', {})
            get_val = lambda k, d=None: gui_state.get(k, data.get(k, d))

            for key in SAVED_STATE_KEYS:
                val = get_val(key)
                if val is not None:
                    default_val = STATE_DEFAULTS.get(key)
                    if isinstance(default_val, np.ndarray):
                        self._data[key] = np.array(val, dtype=default_val.dtype)
                    else:
                        self._data[key] = val



            for attr in CAMERA_ATTRIBUTES:
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
            self.camera.fov_deg = self.view_fov
            self.load_path = path
            self.status_msg = f'Loaded ← {os.path.basename(path)}'
            self.spline.samples_per_loop = self.samples_per_loop

            self.on_bitmap_change()
            if loaded_spline_rows:
                self.spline.ctrl_rows = [np.array(row, dtype=np.float32) for row in loaded_spline_rows]
                self.spline._rebuild()
                if self.mode == 'spline':
                    self.rebuild_spline_mesh()
        except Exception as e:
            self.status_msg = f'Load error: {e}'

    # ── BACKGROUND THREAD TRIGGERS ────────────────────────────────────────────

    def start_background_render(self):
        if not self.is_rendering:
            self.is_rendering = True
            threading.Thread(target=self._bg_render_job, daemon=True).start()

    def _bg_render_job(self):
        try:
            import mitsuba as mi
            vl = compute_knitting_vertices(self.params, self.bitmap)
            fl = compute_knitting_faces(CONFIG['geometry']['segments'], vl)
            path = os.path.join(CONFIG['rendering']['output_dir'], "meshes", "imgui_preview")
            save_combined_obj([(v, [], f, n) for (v, n), f in zip(vl, fl)], path)
            scene = self.optimizer.get_scene_dict(path + "_combined.obj", self.params, camera_params=(self.mi_cam_dist_mult, self.mi_cam_fov))
            scene['mesh']['bsdf']['reflectance']['value'] = [float(v) for v in self.render_texture_color]
            scene['emitter']['radiance']['value'] = [float(self.render_light_intensity) * float(v) for v in self.render_light_color]
            arr = (np.clip(np.array(mi.render(mi.load_dict(scene), spp=128)), 0, 1) * 255).astype(np.uint8)
            self.render_result = Image.fromarray(arr)
            self.pending_tex   = True
        except Exception as e:
            print(f"Render error: {e}")
        finally:
            self.is_rendering = False

    def start_background_optimize(self):
        if not self.is_optimizing:
            self.is_optimizing = True
            threading.Thread(target=self._bg_optimize_job, daemon=True).start()

    def _bg_optimize_job(self):
        try:
            new_params, _ = run_optimization_loop(self.optimizer, self.params)
            self.params[:] = [float(v) for v in new_params]
            self.rebuild_param_mesh()
        except Exception as e:
            print(f"Optimize error: {e}")
        finally:
            self.is_optimizing = False
