# %% IMPORTS
import os
import json
import numpy as np
from PIL import Image
import mitsuba as mi
import drjit as dr
import jax.numpy as jnp
import jax 
import optax
from functools import partial

# %% CONFIGURATION LOADING

def load_config(config_path="config.json"):
    """Loads project configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

CONFIG = load_config()

# Set Mitsuba variant from configuration
if not mi.variant():
    try:
        mi.set_variant(CONFIG['rendering']['mitsuba_variant'])
    except Exception:
        mi.set_variant(CONFIG['rendering']['mitsuba_variant_fallback'])

# Initialize output directories
OUTPUT_DIR = CONFIG['rendering']['output_dir']
for sub_dir in ["meshes", "renders"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub_dir), exist_ok=True)

# %% GEOMETRY ENGINE (JAX)

@jax.jit
def eval_curve_batch(t, scale, stitch_bulge, stitch_z):
    """Vectorized evaluation of the knitting curve geometry."""
    x = stitch_bulge * jnp.sin(2 * t) + t / (2 * jnp.pi)
    y = -(jnp.cos(t) - 1) / 2
    z = stitch_z * (jnp.cos(2 * t) - 1) / 2
    x = jnp.where(scale == 0, t / (2 * jnp.pi), x)
    return jnp.stack([x, y * scale, z * scale], axis=-1)

@jax.jit
def eval_curve_derivative_batch(t, scale, stitch_bulge, stitch_z):
    """Vectorized evaluation of the knitting curve derivatives."""
    dx = 2 * stitch_bulge * jnp.cos(2 * t) + 1 / (2 * jnp.pi)
    dy = 0.5 * jnp.sin(t) * scale
    dz = -stitch_z * jnp.sin(2 * t) * scale
    dx = jnp.where(scale == 0, 1 / (2 * jnp.pi), dx)
    return jnp.stack([dx, dy, dz], axis=-1)

@jax.jit
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

@partial(jax.jit, static_argnums=(2, 3))
def compute_knitting_vertices_jit(geometry_params, bitmap, loop_res, segments):
    """JIT-compiled function to generate all mesh vertices for the pattern."""
    (stitch_bulge, stitch_z, loop_height, dy, radius, 
     _, _, _, _, ellipse_ratio) = geometry_params
    
    def count_consecutive_zeros(row):
        n = len(row)
        indices = jnp.arange(n)
        mask = (indices[:, None] < indices[None, :])
        masked_row = jnp.where(mask, row[None, :], 999)
        nonzero_mask = (masked_row != 0) & mask
        first_nonzero = jnp.argmax(nonzero_mask.astype(jnp.int32), axis=1)
        counts = jnp.where(jnp.any(nonzero_mask, axis=1), 
                           first_nonzero - indices, n - indices - 1)
        return jnp.where(row == 1, counts + 1, 0)
    
    scale_factor = jax.vmap(count_consecutive_zeros)(bitmap)
    scale_factor = jnp.where((scale_factor <= 1), scale_factor, 
                             1 + dy * (scale_factor - 1))
    
    n_rows, n_loops = bitmap.shape
    t_vals = jnp.linspace(0.0, 2 * jnp.pi * n_loops, loop_res * n_loops + 1)
    
    def process_row(row_idx, row_scales):
        x_scale = jnp.append(jnp.repeat(row_scales, loop_res), 1.0)
        pos = eval_curve_batch(t_vals, x_scale, stitch_bulge, stitch_z)
        pos = pos.at[:, 1].add(row_idx * dy)
        d_pos = eval_curve_derivative_batch(t_vals, x_scale, stitch_bulge, stitch_z)
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

def compute_knitting_faces(segments, verts_list):
    """Computes mesh faces based on vertex counts for each row."""
    faces_list = []
    for _, n_points in verts_list:
        i_grid, j_grid = np.meshgrid(np.arange(n_points - 1), 
                                     np.arange(segments), indexing='ij')
        v0 = i_grid * segments + j_grid
        v1 = i_grid * segments + (j_grid + 1) % segments
        v2 = (i_grid + 1) * segments + (j_grid + 1) % segments
        v3 = (i_grid + 1) * segments + j_grid
        faces_list.append(np.stack([v0, v1, v2, v3], axis=-1).reshape(-1, 4))
    return faces_list

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
    obj_info = []
    for row_idx, (verts, _, _, n_points) in enumerate(mesh_data_list):
        n_loops = (n_points - 1) // loop_res
        for loop_idx in range(n_loops):
            v_start = loop_idx * loop_res * segments
            v_end = (loop_idx + 1) * loop_res * segments + segments
            l_verts = verts[v_start:v_end]
            n_l_pts = (v_end - v_start) // segments
            i_g, j_g = np.meshgrid(np.arange(n_l_pts - 1), 
                                   np.arange(segments), indexing='ij')
            v0, v1 = i_g * segments + j_g, i_g * segments + (j_g + 1) % segments
            v2, v3 = ((i_g + 1) * segments + (j_g + 1) % segments, 
                      (i_g + 1) * segments + j_g)
            l_faces = np.stack([v0, v1, v2, v3], axis=-1).reshape(-1, 4)
            path = f"{base_filename}_r{row_idx:02d}_l{loop_idx:02d}.obj"
            with open(path, 'w') as f:
                for v in l_verts: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
                for fa in l_faces: 
                    f.write(f"f {' '.join([str(int(x)+1) for x in fa])}")
            obj_info.append((row_idx, loop_idx, path))
    return obj_info

# %% OPTIMIZATION ENGINE

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
            "type": "scene", "integrator": {"type": "path", "max_depth": 2},
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
        new_params = jnp.clip(optax.apply_updates(jnp.array(params), updates), 
                              jnp.array([r[0] for r in CONFIG['geometry']['param_ranges']]), 
                              jnp.array([r[1] for r in CONFIG['geometry']['param_ranges']]))
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
