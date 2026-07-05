# %% IMPORTS AND RUNTIME SETUP
import os
import sys
import glob
import ctypes

# ── Dynamic library path setup for GPU/CUDA ──────────────────────────────────
# Prevents JAX from hogging VRAM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Preloads venv-packaged nvidia shared libraries to resolve symbols globally on Linux
if sys.platform.startswith("linux"):
    for path in sys.path:
        if path.endswith("site-packages") and os.path.exists(os.path.join(path, "nvidia")):
            for so_path in sorted(glob.glob(os.path.join(path, "nvidia/**/*.so*"), recursive=True)):
                try:
                    ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass
            break

import json
import numpy as np
from PIL import Image
import mitsuba as mi
import drjit as dr
import jax.numpy as jnp
import jax 
import optax
from functools import partial
from scipy.interpolate import CubicSpline


# %% CONFIGS

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_project_path(path):
    """Resolves a config path relative to the project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def load_config(config_path="config.json"):
    """Loads project configuration from a JSON file."""
    with open(resolve_project_path(config_path), 'r') as f:
        return json.load(f)

CONFIG = load_config()
REFERENCE_IMAGE_PATH = resolve_project_path(CONFIG['ui']['reference_image'])


# Set Mitsuba variant from configuration.
if not mi.variant():
    for _variant_key in ('mitsuba_variant', 'mitsuba_variant_fallback', 'mitsuba_variant_fallback2'):
        _variant = CONFIG['rendering'].get(_variant_key)
        if not _variant:
            continue
        try:
            mi.set_variant(_variant)
            break
        except Exception:
            pass


# Initialize output directories.
OUTPUT_DIR = resolve_project_path(CONFIG['rendering']['output_dir'])
for sub_dir in ["meshes", "renders"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub_dir), exist_ok=True)


# %% PARAMETER REGISTRY

def load_geometry_parameters():
    """Loads geometry parameter definitions from the required array-of-structs config."""
    geometry_cfg = CONFIG['geometry']
    raw_parameters = geometry_cfg.get('parameters')
    if not isinstance(raw_parameters, list) or not raw_parameters:
        raise ValueError("CONFIG['geometry']['parameters'] must be a non-empty list.")

    required_keys = {'name', 'initial', 'range', 'delta'}
    for index, param in enumerate(raw_parameters):
        if not isinstance(param, dict):
            raise ValueError(f"CONFIG['geometry']['parameters'][{index}] must be an object.")

        missing_keys = required_keys - param.keys()
        if missing_keys:
            missing_list = ', '.join(sorted(missing_keys))
            raise ValueError(
                f"CONFIG['geometry']['parameters'][{index}] is missing keys: {missing_list}."
            )

    loop_height_names = [
        f"loop_height_{scale_factor}"
        for scale_factor in range(1, int(geometry_cfg['bitmap_rows']) + 1)
    ]

    params_by_name = {param['name']: param for param in raw_parameters}
    default_param = params_by_name.get('default_loop_height')
    parameters = [dict(param) for param in raw_parameters if param['name'] != 'default_loop_height']
    existing_names = {param['name'] for param in parameters}

    if default_param is not None:
        lo, hi = default_param['range']
        base = float(default_param['initial'])
        for scale_factor, name in enumerate(loop_height_names, start=1):
            if name in existing_names:
                continue
            parameters.append({
                'name': name,
                'initial': float(np.clip(base * scale_factor, lo, hi)),
                'range': list(default_param['range']),
                'delta': default_param['delta'],
            })
            existing_names.add(name)

    missing_loop_height_params = [
        name for name in loop_height_names
        if name not in existing_names
    ]
    if missing_loop_height_params:
        missing_list = ', '.join(missing_loop_height_params)
        raise ValueError(f"Missing loop height parameters: {missing_list}.")

    return parameters


GEOMETRY_PARAMETERS = tuple(load_geometry_parameters())


def geometry_parameter_names():
    """Returns geometry parameter names in vector order."""
    return tuple(param['name'] for param in GEOMETRY_PARAMETERS)


def initial_geometry_params():
    """Returns a mutable positional vector initialized from config."""
    return [param['initial'] for param in GEOMETRY_PARAMETERS]


def geometry_param_range(index_or_name):
    """Returns the configured range for a parameter index or name."""
    if isinstance(index_or_name, str):
        index_or_name = geometry_param_index(index_or_name)
    return GEOMETRY_PARAMETERS[int(index_or_name)]['range']


def geometry_param_index(name):
    """Returns the positional vector index for a configured parameter name."""
    for index, param in enumerate(GEOMETRY_PARAMETERS):
        if param['name'] == name:
            return index
    raise KeyError(f"Unknown geometry parameter: {name}")


def geometry_param_bounds_jax():
    """Returns lower/upper JAX arrays directly from config."""
    ranges = jnp.array([param['range'] for param in GEOMETRY_PARAMETERS], dtype=jnp.float32)
    return ranges[:, 0], ranges[:, 1]

LOOP_HEIGHT_PARAM_NAMES = tuple(
    f"loop_height_{scale_factor}"
    for scale_factor in range(1, int(CONFIG['geometry']['bitmap_rows']) + 1)
)
LOOP_HEIGHT_PARAM_INDICES = tuple(geometry_param_index(name) for name in LOOP_HEIGHT_PARAM_NAMES)
LOOP_HEIGHT_PARAM_INDEX_ARRAY = jnp.array(LOOP_HEIGHT_PARAM_INDICES, dtype=jnp.int32)

def get_param_value(params, name):
    """Returns a geometry parameter by name from a positional parameter vector."""
    return params[geometry_param_index(name)]

def get_loop_height_lookup(params):
    """Builds a lookup table where index == discrete bitmap scale factor."""
    loop_heights = [0.0]
    loop_heights.extend(float(params[index]) for index in LOOP_HEIGHT_PARAM_INDICES)
    return np.array(loop_heights, dtype=np.float32)

def get_loop_height_lookup_jax(params):
    """JAX version of the discrete loop height lookup table."""
    return jnp.concatenate(
        (jnp.array([0.0], dtype=jnp.float32), jnp.take(params, LOOP_HEIGHT_PARAM_INDEX_ARRAY)),
        axis=0,
    )


# %% DISPLAY FUNCTIONS

def get_display_periods(params, bitmap):
    """Returns the translational display periods for tiled unit-cell previews."""
    bitmap_array = np.asarray(bitmap)
    x_period = float(bitmap_array.shape[1])
    y_period = float(bitmap_array.shape[0]) * float(get_param_value(params, 'dy'))
    return x_period, y_period

def build_display_meshes(verts_list, faces_list, params, bitmap, horizontal_copies, vertical_copies):
    """Expands the base unit-cell meshes into display-only tiled duplicates."""
    horizontal_copies = max(0, int(horizontal_copies))
    vertical_copies = max(0, int(vertical_copies))

    if horizontal_copies == 0 and vertical_copies == 0:
        row_indices = list(range(len(verts_list)))
        return verts_list, faces_list, row_indices

    x_period, y_period = get_display_periods(params, bitmap)
    x_tiles = np.arange(-horizontal_copies, horizontal_copies + 1, dtype=np.float32)
    y_tiles = np.arange(-vertical_copies, vertical_copies + 1, dtype=np.float32)
    tile_x, tile_y = np.meshgrid(x_tiles, y_tiles, indexing='xy')
    translations = np.column_stack((
        tile_x.ravel() * x_period,
        tile_y.ravel() * y_period,
        np.zeros(tile_x.size, dtype=np.float32),
    )).astype(np.float32)

    base_vertices = [np.asarray(verts, dtype=np.float32) for verts, _ in verts_list]
    n_points = [n for _, n in verts_list]
    translated_groups = [
        verts[None, :, :] + translations[:, None, :]
        for verts in base_vertices
    ]
    display_verts_list = [
        (translated_groups[row_index][tile_index], n_points[row_index])
        for tile_index in range(len(translations))
        for row_index in range(len(base_vertices))
    ]
    display_faces_list = list(faces_list) * len(translations)
    row_indices = np.tile(np.arange(len(verts_list), dtype=int), len(translations)).tolist()

    return display_verts_list, display_faces_list, row_indices


# %% EQUATIONS OF MODEL

def compute_bitmap_scale_factors(bitmap):
    """Computes per-stitch vertical span counts directly from the bitmap pattern."""
    bitmap_array = np.asarray(bitmap, dtype=np.float32) > 0.5
    n_rows, n_cols = bitmap_array.shape
    row_indices = np.arange(n_rows, dtype=np.int32)
    future_active = (row_indices[:, None, None] < row_indices[None, :, None]) & bitmap_array[None, :, :]
    has_next = np.any(future_active, axis=1)
    next_row = np.argmax(future_active.astype(np.int32), axis=1)
    distance = np.where(has_next, next_row - row_indices[:, None], n_rows - row_indices[:, None])
    return np.where(bitmap_array, distance, 0).astype(np.float32)

def build_parametric_control_rows(params, bitmap, samples_per_loop=5):
    """Builds spline control rows from the same shared parameterization as the mesh path."""
    stitch_bulge = float(get_param_value(params, 'stitch_bulge'))
    stitch_z = float(get_param_value(params, 'stitch_z'))
    dy = float(get_param_value(params, 'dy'))
    loop_height_lookup = get_loop_height_lookup(params)
    scale_factors = compute_bitmap_scale_factors(bitmap)
    n_rows, n_cols = scale_factors.shape
    samples_per_loop = max(2, int(samples_per_loop))

    row_indices = np.arange(n_rows, dtype=np.float32)
    forward_cols = np.arange(n_cols, dtype=np.float32)
    reverse_cols = forward_cols[::-1]
    serpentine_cols = np.where(
        (row_indices[:, None].astype(np.int32) % 2) == 0,
        forward_cols[None, :],
        reverse_cols[None, :],
    )
    ordered_scales = np.take_along_axis(
        scale_factors.astype(np.int32),
        serpentine_cols.astype(np.int32),
        axis=1,
    )

    base_t = np.linspace(0.0, 2.0 * np.pi, samples_per_loop, endpoint=False, dtype=np.float32)
    t_values = np.where(
        (row_indices[:, None].astype(np.int32) % 2) == 0,
        base_t[None, :],
        base_t[::-1][None, :],
    )

    t_grid = np.broadcast_to(t_values[:, None, :], (n_rows, n_cols, samples_per_loop))
    has_loop = (ordered_scales[:, :, None] > 0).astype(np.float32)
    loop_height = loop_height_lookup[ordered_scales][:, :, None]

    x = (
        serpentine_cols[:, :, None]
        + np.where(has_loop > 0.0, stitch_bulge * np.sin(2.0 * t_grid), 0.0)
        + t_grid / (2.0 * np.pi)
    )
    y = row_indices[:, None, None] * dy - loop_height * (np.cos(t_grid) - 1.0) / 2.0
    z = has_loop * stitch_z * (np.cos(2.0 * t_grid) - 1.0) / 2.0

    path_points = np.stack((x, y, z), axis=-1).reshape(n_rows, n_cols * samples_per_loop, 3)
    endpoints = np.column_stack((
        np.where((row_indices.astype(np.int32) % 2) == 0, float(n_cols), 0.0),
        row_indices * dy,
        np.zeros(n_rows, dtype=np.float32),
    ))[:, None, :]
    rows = np.concatenate((path_points, endpoints), axis=1)
    return [row.astype(float) for row in rows]

# %% JAX GEOMETRY ENGINE

def eval_curve_batch(t, has_loop, loop_height, stitch_bulge, stitch_z):
    """Vectorized evaluation of the knitting curve geometry."""
    x = stitch_bulge * jnp.sin(2 * t) + t / (2 * jnp.pi)
    y = loop_height * (-(jnp.cos(t) - 1) / 2)
    z = stitch_z * (jnp.cos(2 * t) - 1) / 2 * has_loop
    x = jnp.where(has_loop == 0.0, t / (2 * jnp.pi), x)
    return jnp.stack((jnp.asarray(x), jnp.asarray(y), jnp.asarray(z)), axis=-1)

def eval_curve_derivative_batch(t, has_loop, loop_height, stitch_bulge, stitch_z):
    """Vectorized evaluation of the knitting curve derivatives."""
    d_x = 2 * stitch_bulge * jnp.cos(2 * t) + 1 / (2 * jnp.pi)
    d_y = 0.5 * jnp.sin(t) * loop_height
    d_z = -stitch_z * jnp.sin(2 * t) * has_loop
    d_x = jnp.where(has_loop == 0.0, 1 / (2 * jnp.pi), d_x)
    return jnp.stack((jnp.asarray(d_x), jnp.asarray(d_y), jnp.asarray(d_z)), axis=-1)

def compute_orthonormal_frame_batch(tangent):
    """Computes orthonormal frames along the curve for tube generation."""
    tangent = tangent / (jnp.linalg.norm(tangent, axis=-1, keepdims=True) + 1e-8)
    ref = jnp.array([0.0, 0.0, 1.0])
    normal_u = jnp.cross(tangent, ref)
    u_norm = jnp.linalg.norm(normal_u, axis=-1, keepdims=True)
    
    # Handle parallel cases to avoid gimbal lock
    alt_ref = jnp.array([1.0, 0.0, 0.0])
    normal_u = jnp.where(u_norm < 1e-6, jnp.cross(tangent, alt_ref), normal_u)
    
    normal_u = normal_u / (jnp.linalg.norm(normal_u, axis=-1, keepdims=True) + 1e-8)
    normal_v = jnp.cross(tangent, normal_u)
    return normal_u, normal_v


def compute_bitmap_scale_factors_jax(bitmap):
    """Vectorized JAX equivalent of compute_bitmap_scale_factors."""
    active = bitmap > 0.5
    n_rows = active.shape[0]
    row_indices = jnp.arange(n_rows, dtype=jnp.int32)
    future_active = (row_indices[:, None, None] < row_indices[None, :, None]) & active[None, :, :]
    has_next = jnp.any(future_active, axis=1)
    next_row = jnp.argmax(future_active.astype(jnp.int32), axis=1)
    distance = jnp.where(has_next, next_row - row_indices[:, None], n_rows - row_indices[:, None])
    return jnp.where(active, distance, 0).astype(jnp.float32)


@partial(jax.jit, static_argnums=(2, 3))
def compute_knitting_vertices_jit(geometry_params, bitmap, loop_res, segments):
    """JIT-compiled function to generate all mesh vertices for the pattern."""
    stitch_bulge = geometry_params[geometry_param_index('stitch_bulge')]
    stitch_z = geometry_params[geometry_param_index('stitch_z')]
    dy = geometry_params[geometry_param_index('dy')]
    radius = geometry_params[geometry_param_index('radius')]
    ellipse_ratio = geometry_params[geometry_param_index('ellipse_ratio')]
    loop_height_lookup = get_loop_height_lookup_jax(geometry_params)
    scale_factor = compute_bitmap_scale_factors_jax(bitmap)
    
    n_rows, n_loops = bitmap.shape
    t_vals = jnp.linspace(0.0, 2 * jnp.pi * n_loops, loop_res * n_loops + 1)
    
    def process_row(row_idx, row_scales):
        loop_scales = jnp.append(jnp.repeat(row_scales, loop_res), 1.0).astype(jnp.int32)
        loop_heights = jnp.take(loop_height_lookup, loop_scales)
        has_loops = (loop_scales > 0).astype(jnp.float32)
        pos = eval_curve_batch(t_vals, has_loops, loop_heights, stitch_bulge, stitch_z)
        pos = pos.at[:, 1].add(row_idx * dy)
        d_pos = eval_curve_derivative_batch(t_vals, has_loops, loop_heights, stitch_bulge, stitch_z)
        u_frame, v_frame = compute_orthonormal_frame_batch(d_pos)
        
        angles = jnp.linspace(0, 2 * jnp.pi, segments, endpoint=False)
        offsets = (u_frame[:, None, :] * jnp.cos(angles)[None, :, None] * 
                   radius * ellipse_ratio + 
                   v_frame[:, None, :] * jnp.sin(angles)[None, :, None] * radius)
        
        return (pos[:, None, :] + offsets).reshape(-1, 3)

    return jax.vmap(process_row)(jnp.arange(n_rows), scale_factor)

def compute_knitting_vertices(geometry_params, bitmap):
    """High-level wrapper for JIT vertex computation."""
    res = CONFIG['geometry']['loop_res']
    seg = CONFIG['geometry']['segments']
    vertices = compute_knitting_vertices_jit(
        jnp.array(geometry_params), jnp.array(bitmap), res, seg
    )
    return [(v, len(v) // seg) for v in vertices]


# %% MESH TOPOLOGY

def compute_knitting_faces(segments, verts_list):
    """Computes mesh faces based on vertex counts for each row."""
    return [build_tube_faces(n_points, segments) for _, n_points in verts_list]


def build_tube_faces(n_points, segments):
    """Build quad faces for a tube sampled as n_points rings with segments vertices each."""
    i_grid, j_grid = np.meshgrid(np.arange(n_points - 1), np.arange(segments), indexing='ij')
    v0 = i_grid * segments + j_grid
    v1 = i_grid * segments + (j_grid + 1) % segments
    v2 = (i_grid + 1) * segments + (j_grid + 1) % segments
    v3 = (i_grid + 1) * segments + j_grid
    return np.stack([v0, v1, v2, v3], axis=-1).reshape(-1, 4)

def compute_geometry_jacobian(geometry_params, bitmap):
    """Computes the Jacobian of vertex positions using JAX autodiff."""
    res, seg = CONFIG['geometry']['loop_res'], CONFIG['geometry']['segments']
    def get_all_vertices(params):
        verts = compute_knitting_vertices_jit(params, bitmap, res, seg)
        return verts.flatten()
    jacobian = jax.jacfwd(get_all_vertices)(geometry_params)
    return jacobian.reshape(-1, 3, len(geometry_params))

# %% MESH IO

def save_combined_obj(mesh_data_list, base_filename="knitting_model"):
    """Saves all mesh parts into a single combined OBJ file."""
    combined_filename = f"{base_filename}_combined.obj"
    vertex_offset = 0
    with open(combined_filename, 'w') as f:
        f.write("# Knitting Model\n")
        for i, (verts, _, faces, _) in enumerate(mesh_data_list):
            f.write(f"o mesh_{i}\n")
            for v in verts: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                idx_str = ' '.join([str(int(idx) + vertex_offset + 1) for idx in face])
                f.write(f"f {idx_str}\n")
            vertex_offset += len(verts)

def save_per_loop_objs(mesh_data_list, base_filename, loop_res, segments):
    """Saves each stitch loop as a separate OBJ for pattern analysis."""
    loop_vertex_count = (loop_res + 1) * segments
    loop_faces = build_tube_faces(loop_res + 1, segments)
    loop_specs = [
        (
            row_idx,
            loop_idx,
            verts[loop_idx * loop_res * segments:loop_idx * loop_res * segments + loop_vertex_count],
            f"{base_filename}_r{row_idx:02d}_l{loop_idx:02d}.obj",
        )
        for row_idx, (verts, _, _, n_points) in enumerate(mesh_data_list)
        for loop_idx in range((n_points - 1) // loop_res)
    ]

    obj_info = []
    for row_idx, loop_idx, loop_verts, path in loop_specs:
        vertex_lines = ''.join(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in loop_verts)
        face_lines = ''.join(f"f {' '.join(str(int(x) + 1) for x in face)}\n" for face in loop_faces)
        with open(path, 'w') as f:
            f.write(vertex_lines)
            f.write(face_lines)
        obj_info.append((row_idx, loop_idx, path))
    return obj_info

# %% OPTIMIZATION

def get_loop_color(row_idx, loop_idx):
    """Returns discrete colors for loops from config.json."""
    palette = CONFIG['ui']['yarn_colors']
    row_pattern = row_idx % 3
    if row_pattern == 0: return palette[0]
    if row_pattern == 1: return palette[3 % len(palette)] if loop_idx % 2 == 0 else palette[2 % len(palette)]
    return palette[1 % len(palette)]

class KnittingOptimizer:
    """Handles the differentiable rendering and gradient descent loop."""
    def __init__(self, reference_img, bitmap):
        self.bitmap = bitmap
        self.iteration = 0
        opt_cfg = CONFIG['optimization']
        self.optimizer = optax.adam(opt_cfg['learning_rate'])
        self.opt_state = None
        self.ref_array = np.array(reference_img).astype(np.float32) / 255.0
        self.res_height, self.res_width = self.ref_array.shape[:2]
        self.loss_weights = opt_cfg['loss_weights']
        self.loss_mask = mi.TensorXf(self._build_mask(opt_cfg['loss_center_crop']))
        self.ref_tensor = mi.TensorXf(self.ref_array)
        self.loss_history, self.param_history = [], []
        rc = CONFIG['rendering']
        self.camera_params = (rc['camera_dist_mult'], rc['camera_fov'])
        self.row_colors = [tuple(c) for c in CONFIG['ui']['yarn_colors']]
        self.row_colors = [self.row_colors[i % len(self.row_colors)] 
                           for i in range(bitmap.shape[0])]

    def _build_mask(self, crop):
        cw, ch = crop
        mask = np.zeros((self.res_height, self.res_width, 3), dtype=np.float32)
        x0, x1 = int((1 - cw) * 0.5 * self.res_width), int((1 + cw) * 0.5 * self.res_width)
        y0, y1 = int((1 - ch) * 0.5 * self.res_height), int((1 + ch) * 0.5 * self.res_height)
        mask[max(0, y0):min(self.res_height, y1), 
             max(0, x0):min(self.res_width, x1), :] = 1.0
        return mask

    def get_scene_dict(self, obj_path, params, camera_params=None):
        """Generates a Mitsuba scene dictionary for the current model state."""
        dm, fov = camera_params if camera_params else self.camera_params
        verts_list = compute_knitting_vertices(params, self.bitmap)
        all_verts = jnp.concatenate([v for v, _ in verts_list])
        vmin, vmax = np.min(all_verts, axis=0), np.max(all_verts, axis=0)
        center = (vmin + vmax) * 0.5
        dist = max((vmax[1]-vmin[1])/np.tan(np.deg2rad(fov)*0.5), 
                   (vmax[0]-vmin[0])/np.tan(np.deg2rad(fov)*0.5 * 
                   self.res_width/self.res_height)) * 0.5 * dm + (vmax[2]-vmin[2])*0.2
        return {
            "type": "scene", "integrator": {"type": "prb", "max_depth": 2},
            "sensor": {
                "type": "perspective", "fov": fov,
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=[float(center[0]), float(center[1]), float(vmax[2]+dist)], 
                    target=[float(center[0]), float(center[1]), float(center[2])], 
                    up=[0, 1, 0]
                ),
                "film": {"type": "hdrfilm", "width": self.res_width, 
                         "height": self.res_height, "pixel_format": "rgb"},
            },
            "emitter": {"type": "constant", "radiance": {"type": "rgb", "value": [0.9, 0.9, 0.9]}},
            "mesh": { "type": "obj", "filename": obj_path, 
                      "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.8, 0.4, 0.3]}} }
        }

    def step(self, params):
        """Performs a single optimization step using differentiable rendering."""
        self.iteration += 1
        params_np = np.array(params)
        verts_list = compute_knitting_vertices(params_np, self.bitmap)
        faces_list = compute_knitting_faces(CONFIG['geometry']['segments'], verts_list)
        mesh_data = [(v, [], f, n) for (v, n), f in zip(verts_list, faces_list)]
        
        base_path = os.path.join(OUTPUT_DIR, "meshes", f"epoch_{self.iteration:03d}")
        save_combined_obj(mesh_data, base_path)
        scene = mi.load_dict(self.get_scene_dict(base_path + "_combined.obj", params_np))
        params_scene = mi.traverse(scene)
        vertex_key = [k for k in params_scene.keys() if 'vertex_positions' in k]
        
        if vertex_key:
            vpos = params_scene[vertex_key[0]]
            dr.enable_grad(vpos)
            params_scene.update()
            img = mi.render(scene, params=params_scene, spp=CONFIG['rendering']['spp_optimization'])
            diff = (img - self.ref_tensor) * self.loss_mask
            loss_dr = dr.sum(dr.sqr(diff)) / (dr.sum(self.loss_mask) + 1e-8)
            dr.backward(loss_dr)
            jacobian = compute_geometry_jacobian(params_np, self.bitmap)
            vertex_grads_np = np.array(dr.grad(vpos)).reshape(-1, 3)
            grads = np.array([np.sum(vertex_grads_np * jacobian[:, :, i]) for i in range(len(params_np))])
            current_loss = float(dr.ravel(loss_dr)[0])
        else:
            grads, current_loss = np.zeros(len(params_np)), 0.0
            
        if self.opt_state is None:
            self.opt_state = self.optimizer.init(jnp.array(params))
        updates, self.opt_state = self.optimizer.update(jnp.array(grads), self.opt_state)
        lower_bounds, upper_bounds = geometry_param_bounds_jax()
        new_params = jnp.clip(
            optax.apply_updates(jnp.array(params), updates),
            lower_bounds,
            upper_bounds,
        )
        self.loss_history.append(current_loss)
        self.param_history.append(new_params)
        print(f"Epoch {self.iteration:02d} | Loss: {current_loss:.6f}")
        return new_params

def run_optimization_loop(optimizer, params):
    """Executes the optimization loop based on configuration."""
    cfg = CONFIG['optimization']
    best_l, best_p, count = float('inf'), params, 0
    for _ in range(cfg['max_epochs']):
        params = optimizer.step(params)
        l = optimizer.loss_history[-1]
        if l < best_l - 1e-6: best_l, best_p, count = l, params, 0
        elif (count := count + 1) >= cfg['patience']: break
    return best_p, best_l

# %% SPLINE EDITING AND CONTROL-POINT MESHES

def _interp_spline_rows(ctrl_rows, n_out):
    """Interpolates equal-length spline control rows in one vectorized call."""
    rows = np.asarray(ctrl_rows, dtype=float)
    if rows.ndim != 3 or rows.shape[1] < 2:
        return np.empty((0, n_out, 3), dtype=np.float32)

    t = np.linspace(0.0, 1.0, rows.shape[1])
    t_out = np.linspace(0.0, 1.0, n_out)
    if rows.shape[1] == 2:
        alpha = t_out[None, :, None]
        return ((1.0 - alpha) * rows[:, :1, :] + alpha * rows[:, 1:2, :]).astype(np.float32)

    return CubicSpline(t, rows, axis=1, bc_type="natural")(t_out).astype(np.float32)


def build_tube_vertices_from_paths(paths, radius, ellipse_ratio, segments):
    """Vectorized tube mesh construction for paths shaped (rows, points, 3)."""
    paths = np.asarray(paths, dtype=np.float32)
    if paths.size == 0:
        return np.empty((0, 0, 3), dtype=np.float32)

    tangent = np.gradient(paths, axis=1)
    tangent /= np.linalg.norm(tangent, axis=2, keepdims=True) + 1e-8
    u_frame = np.cross(tangent, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    bad = np.linalg.norm(u_frame, axis=2) < 1e-6
    u_frame[bad] = np.cross(tangent[bad], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    u_frame /= np.linalg.norm(u_frame, axis=2, keepdims=True) + 1e-8
    v_frame = np.cross(tangent, u_frame)

    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=np.float32)
    offsets = (
        u_frame[:, :, None, :] * np.cos(angles)[None, None, :, None] * radius * ellipse_ratio
        + v_frame[:, :, None, :] * np.sin(angles)[None, None, :, None] * radius
    )
    return (paths[:, :, None, :] + offsets).reshape(paths.shape[0], -1, 3)


class SplineManager:
    def __init__(self, bitmap, config, samples_per_loop=5):
        self.bitmap          = bitmap
        self.config          = config
        self.samples_per_loop = samples_per_loop
        self.ctrl_rows  = []                          # list of (N,3) arrays
        self.flat_pts   = np.empty((0, 3), np.float32)
        self._row_starts = [0]

    def init_from_params(self, params):
        self.ctrl_rows = build_parametric_control_rows(
            params, self.bitmap, self.samples_per_loop)
        self._rebuild()

    def _rebuild(self):
        self._row_starts = [0]
        for row in self.ctrl_rows:
            self._row_starts.append(self._row_starts[-1] + len(row))
        self.flat_pts = (
            np.concatenate(self.ctrl_rows).astype(np.float32)
            if self.ctrl_rows else np.empty((0, 3), np.float32)
        )

    def move(self, flat_idx, pos):
        row_idx = int(np.searchsorted(self._row_starts, flat_idx, side='right') - 1)
        if 0 <= row_idx < len(self.ctrl_rows):
            local_idx = flat_idx - self._row_starts[row_idx]
            if 0 <= local_idx < len(self.ctrl_rows[row_idx]):
                self.ctrl_rows[row_idx][local_idx] = pos
        self._rebuild()

    def fix_endpoints(self, params):
        """Locks row endpoints to tile boundaries so spline edits still duplicate cleanly."""
        if not self.ctrl_rows:
            return
        width = float(self.bitmap.shape[1])
        dy = float(params[geometry_param_index('dy')])
        for row_idx, row in enumerate(self.ctrl_rows):
            if len(row) < 2:
                continue
            if row[0, 0] <= row[-1, 0]:
                left_idx, right_idx = 0, -1
            else:
                left_idx, right_idx = -1, 0
            baseline_y = float(row_idx) * dy
            row[left_idx] = np.array([0.0, baseline_y, 0.0], dtype=row.dtype)
            row[right_idx] = np.array([width, baseline_y, 0.0], dtype=row.dtype)
        self._rebuild()

    def build_mesh(self, params):
        radius = params[geometry_param_index('radius')]
        ratio  = params[geometry_param_index('ellipse_ratio')]
        seg    = self.config['geometry']['segments']
        res    = self.config['geometry']['loop_res']
        n_out  = res * self.bitmap.shape[1] + 1
        paths = _interp_spline_rows(self.ctrl_rows, n_out)
        vertices = build_tube_vertices_from_paths(paths, radius, ratio, seg)
        return list(zip(vertices, np.full(len(vertices), n_out, dtype=int)))

