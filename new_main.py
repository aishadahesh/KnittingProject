#%% Imports
import numpy as np
import bmesh

# Try to import JAX (only works outside Blender)
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    print("JAX not available - using manual derivatives instead")


#%% Frame functions

def eval_curve(t, scale, stitch_bulge=0.30, stitch_z=-0.4):
    t = np.asarray(t, dtype=float)
    scale = np.asarray(scale, dtype=float)
    
    # Base curve - create proper knitting loop progression
    x = stitch_bulge * np.sin(2*t) + t/(2*np.pi)  # Horizontal stitch pattern
    y = -(np.cos(t) - 1)/2  # Creates the loop shape (U-shaped)
    z = stitch_z * (np.cos(2*t) - 1)/2  # Depth variation for stitch structure
    
    # Apply scale - only where scale is not zero (for dropped stitches)
    x = np.where(scale == 0, t/(2*np.pi), x)
    y = y * scale
    z = z * scale
    
    return np.column_stack((x, y, z))


# JAX version of eval_curve for automatic differentiation
def eval_curve_jax(t_val, scale_val, stitch_bulge=0.30, stitch_z=-0.4):

    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is not available")
    
    # Base curve
    x = stitch_bulge * jnp.sin(2*t_val) + t_val/(2*jnp.pi)
    y = -(jnp.cos(t_val) - 1)/2
    z = stitch_z * (jnp.cos(2*t_val) - 1)/2
    
    # Apply scale
    x = jnp.where(scale_val == 0, t_val/(2*jnp.pi), x)
    y = y * scale_val
    z = z * scale_val
    
    return jnp.array([x, y, z])

#   Compute first derivative. Uses JAX autodiff if available, otherwise falls back to manual derivatives.
def eval_curve_derivative(t, scale, stitch_bulge=0.30, stitch_z=-0.4):

    t = np.asarray(t, dtype=float)
    scale = np.asarray(scale, dtype=float)
    
    if JAX_AVAILABLE:
        # Use JAX automatic differentiation
        def curve_at_t(t_val, scale_val):
            return eval_curve_jax(t_val, scale_val, stitch_bulge, stitch_z)
        
        vectorized_grad = jax.vmap(
            lambda t_val, scale_val: jax.jacfwd(curve_at_t, argnums=0)(t_val, scale_val)
        )
        
        derivatives = vectorized_grad(jnp.array(t), jnp.array(scale))
        return np.array(derivatives)
    else:
        # Manual derivatives 
        dx = 2*stitch_bulge*np.cos(2*t) + 1/(2*np.pi)
        dy = 0.5*np.sin(t)*scale
        dz = -stitch_z*np.sin(2*t)*scale
        
        dx = np.where(scale == 0, 1/(2*np.pi), dx)
        
        return np.column_stack((dx, dy, dz))


# Compute second derivative. Uses JAX autodiff if available, otherwise falls back to manual derivatives.
def eval_curve_second_derivative(t, scale, stitch_bulge=0.30, stitch_z=-0.4):

    t = np.asarray(t, dtype=float)
    scale = np.asarray(scale, dtype=float)
    
    if JAX_AVAILABLE:
        # Use JAX automatic differentiation
        def curve_at_t(t_val, scale_val):
            return eval_curve_jax(t_val, scale_val, stitch_bulge, stitch_z)
        
        vectorized_second_grad = jax.vmap(
            lambda t_val, scale_val: jax.jacfwd(
                jax.jacfwd(curve_at_t, argnums=0), 
                argnums=0
            )(t_val, scale_val)
        )
        
        second_derivatives = vectorized_second_grad(jnp.array(t), jnp.array(scale))
        return np.array(second_derivatives)
    else:
        # Manual second derivatives 
        d2x = -4*stitch_bulge*np.sin(2*t)
        d2y = 0.5*np.cos(t)*scale
        d2z = -2*stitch_z*np.cos(2*t)*scale
        
        d2x = np.where(scale == 0, 0.0, d2x)
        
        return np.column_stack((d2x, d2y, d2z))


def compute_frenet_frame(dp, ddp):
    dp = np.asarray(dp, dtype=float)
    ddp = np.asarray(ddp, dtype=float)

    # Tangent: normalize dp
    dp_norm = np.linalg.norm(dp, axis=1, keepdims=True)
    T = dp / dp_norm

    dp_dot_ddp = np.sum(dp * ddp, axis=1, keepdims=True)
    numerator = ddp * (dp_norm**2) - dp * dp_dot_ddp
    denom = dp_norm**3
    dT_ds = numerator / denom

    dT_norm = np.linalg.norm(dT_ds, axis=1, keepdims=True)
    N = np.zeros_like(dT_ds)
    mask = dT_norm[:, 0] > 1e-14
    N[mask] = dT_ds[mask] / dT_norm[mask]

    # Binormal: cross product
    B = np.cross(T, N)

    return T, N, B


def compute_orthonormal_frame(T):
    # Normalize T
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    
    # Pick a global up vector
    up = np.array([0, 0, 1], dtype=float)

    U = np.zeros_like(T)
    V = np.zeros_like(T)

    for i in range(len(T)):
        # If T is too parallel to up, pick another up vector
        if abs(np.dot(T[i], up)) > 0.99:
            up_vec = np.array([0, 1, 0], dtype=float)
        else:
            up_vec = up

        # Compute U axis (perpendicular to T and up)
        U[i] = np.cross(up_vec, T[i])
        U[i] /= np.linalg.norm(U[i]) + 1e-8

        # Compute V axis (perpendicular to T and U)
        V[i] = np.cross(T[i], U[i])
        V[i] /= np.linalg.norm(V[i]) + 1e-8

    return T, U, V

#%% uv mapping functions

def apply_uv_mapping(obj, segments, n_points):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)

    uv_layer = bm.loops.layers.uv.new("UVMap")

    for face in bm.faces:
        for loop in face.loops:
            v_idx = loop.vert.index
            ring = v_idx % segments      # position around circumference
            row = v_idx // segments      # position along length

            u = ring / segments
            v = row / (n_points - 1)
            loop[uv_layer].uv = (u, v)

    bm.to_mesh(me)
    bm.free()

def scale_uv_map(obj, u_scale=1.0, v_scale=1.0):
    """Scales the UV map of the given object by u_scale and v_scale."""
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)

    uv_layer = bm.loops.layers.uv.active
    if uv_layer is None:
        print("No UV map found!")
        bm.free()
        return

    for face in bm.faces:
        for loop in face.loops:
            u, v = loop[uv_layer].uv
            loop[uv_layer].uv = (u * u_scale, v * v_scale)

    bm.to_mesh(me)
    bm.free()




#%% mesh data

# Returns: tuple: (vertices, edges, faces) as lists
def build_mesh(points, U, V, radius, segments=16):
    
    verts = []
    faces = []

    # Generate vertices for each circle
    for i in range(len(points)):
        for j in range(segments):
            angle = 2 * np.pi * j / segments
            offset = (np.cos(angle) * U[i]  + np.sin(angle) * V[i]) * radius  # * (1 + 0.5 * np.sin(2 * angle))
            verts.append(points[i] + offset)

    # Connect vertices between circles
    for i in range(len(points) - 1):
        for j in range(segments):
            v0 = i * segments + j
            v1 = i * segments + (j + 1) % segments
            v2 = (i + 1) * segments + (j + 1) % segments
            v3 = (i + 1) * segments + j
            faces.append([v0, v1, v2, v3])
    
    edges = []  # No explicit edges needed for this mesh type
    
    return verts, edges, faces

# List of tuples: [(verts, edges, faces, n_points), ...] for each row
def generate_mesh_data(created_objs, yarn_radius=0.12, segments=8):
    mesh_data_list = []
    for i, (p, U, V, scale) in enumerate(created_objs):
        verts, edges, faces = build_mesh(p, U, V, radius=yarn_radius, segments=segments)
        mesh_data_list.append((verts, edges, faces, len(p)))
    
    return mesh_data_list

def save_into_obj_files(mesh_data_list, base_filename="knitting_model"):
    # Option 1: Save as a single combined OBJ file
    combined_filename = f"{base_filename}_combined.obj"
    vertex_offset = 0
    
    with open(combined_filename, 'w') as f:
        f.write(f"# Knitting Model - Combined Mesh\n")
        f.write(f"# Generated with {len(mesh_data_list)} mesh parts\n\n")
        
        for i, (verts, edges, faces, n_points) in enumerate(mesh_data_list):
            f.write(f"# Mesh part {i+1}\n")
            f.write(f"o knittingMesh_{i}\n\n")
            
            # Write vertices
            for vert in verts:
                # Convert numpy array to list if needed
                if hasattr(vert, 'tolist'):
                    v = vert.tolist()
                else:
                    v = vert
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            f.write("\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                # Adjust indices by vertex_offset and add 1 for OBJ format
                face_indices = [str(idx + vertex_offset + 1) for idx in face]
                f.write(f"f {' '.join(face_indices)}\n")
            
            f.write("\n")
            vertex_offset += len(verts)
    
    print(f"Saved combined mesh to: {combined_filename}")

#%% Build mesh in blender

# Returns: Blender object containing the created mesh
def build_mesh_in_blender(verts, edges, faces, name="knittingMesh"):
    mesh_data = bpy.data.meshes.new(name)
    mesh_data.from_pydata(verts, edges, faces)
    mesh_data.update()

    obj = bpy.data.objects.new(name + "Object", mesh_data)
    bpy.context.collection.objects.link(obj)
    return obj

def add_duplicate_index(obj, value):
    mesh = obj.data
    if not mesh or obj.type != 'MESH':
        print(f"{obj.name} is not a valid mesh object.")
        return
    if "duplicate_index" not in mesh.attributes:
        mesh.attributes.new(name="duplicate_index", type='FLOAT', domain='POINT')
    attr = mesh.attributes["duplicate_index"].data
    for i in range(len(attr)):
        attr[i].value = value

#  Returns: List of created Blender objects
def create_model_in_blender(mesh_data_list, segments=8):
    created_objects = []
    for i, (verts, edges, faces, n_points) in enumerate(mesh_data_list):
        obj_yarn = build_mesh_in_blender(verts, edges, faces, name=f"knittingMesh_{i}")
        
        if obj_yarn:
            apply_uv_mapping(obj_yarn, segments=segments, n_points=n_points)
            add_duplicate_index(obj_yarn, i)
            created_objects.append(obj_yarn)
    
    return created_objects

# merge all created objects into one mesh
def join_objects(objects, new_name="MergedLoops"):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    merged_obj = bpy.context.view_layer.objects.active
    merged_obj.name = new_name

    return merged_obj

def join_loop(objects, new_name="MergedLoops"):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    merged_obj = bpy.context.view_layer.objects.active
    merged_obj.name = new_name
    return merged_obj


#%% knitting loop functions

def count_consecutive_zeros_after(A):
    A = np.asarray(A)
    n = len(A)
    result = np.zeros(n, dtype=int)
    
    for i in range(n):
        if A[i] == 1:
            remaining = A[i+1:] if i+1 < n else np.array([])
            if len(remaining) > 0:
                nonzero_positions = np.where(remaining != 0)[0]
                if len(nonzero_positions) > 0:
                    consecutive_zeros = nonzero_positions[0]
                else:
                    consecutive_zeros = len(remaining)
                result[i] = consecutive_zeros + 1
            else:
                result[i] = 1
    return result

def convert_bitmap_to_scales_factors(matrix):
    return np.apply_along_axis(count_consecutive_zeros_after, axis=0, arr=matrix)

def create_curve(loop_res, n_loops, x_scale, tx = 0.0):
    # Restored original proven knitting parameters
    stitch_bulge = 0.26   # Standard knitting bulge
    stitch_z = -0.48      # Depth variation for knitting texture
    
    t = np.linspace(0, 2 * np.pi * n_loops, loop_res * n_loops + 1, endpoint=True)
    x_scale = np.repeat(x_scale, loop_res)
    x_scale = np.append(x_scale, 1)  # Add 1 to the end of the vector
    
    # Use the original numpy-based eval_curve (not the JAX version)
    p = eval_curve(t, x_scale, stitch_bulge=stitch_bulge, stitch_z=stitch_z)
    p[:,1] += tx
    dp = eval_curve_derivative(t, x_scale, stitch_bulge=stitch_bulge, stitch_z=stitch_z)
    ddp = eval_curve_second_derivative(t, x_scale, stitch_bulge=stitch_bulge, stitch_z=stitch_z)

    return t, p, dp, ddp


def knitting_loop_main(scale_factor):
    n_loops = scale_factor.shape[1]
    print(f"Number of loops: {n_loops}")
    n_rows = scale_factor.shape[0]
    print(f"Number of rows: {n_rows}")
    loop_res = 32    # Reduced resolution for cleaner geometry

    created_objects = []
    
    for i in range(len(scale_factor)):
        scale = scale_factor[i]
        t, p, dp, ddp = create_curve(loop_res, n_loops, scale, i * dy)
        T = dp / (np.linalg.norm(dp, axis=1, keepdims=True) + 1e-8)
        T, U, V = compute_orthonormal_frame(T)
        # T, U, V = compute_frenet_frame(dp, ddp)
        created_objects.append([p, U, V, scale])
    
    return created_objects


    
#%% main

# Run with default bitmap and colors, no GUI
print("Running in non-GUI mode with default bitmap and colors.")
bitmap = [
    [1,1,1],
    [1,1,1],
    [1,1,1],
]
bitmap = np.array(bitmap)
colors = [
    (1,0,0,1),
    (0,1,0,1),
    (0,0,1,1),
]
bitmap = bitmap[::-1]
colors = colors[::-1]

dy = 0.35  # Reduced spacing between rows for better interlocking

scale_factor = convert_bitmap_to_scales_factors(bitmap)
scale_factor = np.where((scale_factor <= 1), scale_factor, 1 + dy * (scale_factor - 1))

# Generate knitting geometry data 
created_objs = knitting_loop_main(bitmap)   

# Generate mesh data from geometry 
mesh_data_list = generate_mesh_data(created_objs, yarn_radius=0.12, segments=8)

Blender = False
if Blender:
    import bpy
    del_obj = bpy.context.active_object
    if del_obj:
        bpy.data.objects.remove(del_obj, do_unlink=True)
    # Create Blender objects from mesh data
    created_objects = create_model_in_blender(mesh_data_list, segments=8)
    if created_objects:
        merged_obj = join_objects(created_objects)
        scale_uv_map(merged_obj, u_scale=1.0, v_scale=1.0)
        bpy.context.view_layer.objects.active = merged_obj
        bpy.ops.object.shade_smooth()
else:
    save_into_obj_files(mesh_data_list, "knitting_model")
