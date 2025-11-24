#%% imports
import bpy
import numpy as np
import sys
# sys.path.append(r"C:\projects\KnittingProject")     # Roi
sys.path.append(r"C:\Users\Aisha\KnittingProject")   # Aisha    
import pick_colors_gui
import render_images_gui
import obj_to_mesh
import coloring
import rendering
import bmesh
import math
import mathutils
import random

#%% frame functions

def eval_curve(t, scale, stitch_count=8, stitch_bulge=0.30, stitch_height=1.2, stitch_z=-0.4, yarn_radius=0.25):
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

def eval_curve_derivative(t, scale, stitch_count=8, stitch_bulge=0.30, stitch_height=1.2, stitch_z=-0.4, yarn_radius=0.25):
    t = np.asarray(t, dtype=float)
    scale = np.asarray(scale, dtype=float)
    
    # Derivatives of the base curve
    dx = 2*stitch_bulge*np.cos(2*t) + 1/(2*np.pi)
    dy = 0.5*np.sin(t)*scale
    dz = -stitch_z*np.sin(2*t)*scale
    
    # Handle dropped stitches (scale == 0)
    dx = np.where(scale == 0, 1/(2*np.pi), dx)
    
    return np.column_stack((dx, dy, dz))

def eval_curve_second_derivative(t, scale, stitch_count=8, stitch_bulge=0.30, stitch_height=1.2, stitch_z=-0.4, yarn_radius=0.25):
    t = np.asarray(t, dtype=float)
    scale = np.asarray(scale, dtype=float)
    
    # Second derivatives of the base curve
    d2x = -4*stitch_bulge*np.sin(2*t)
    d2y = 0.5*np.cos(t)*scale
    d2z = -2*stitch_z*np.cos(2*t)*scale
    
    # Handle dropped stitches (scale == 0)
    d2x = np.where(scale == 0, 0.0, d2x)
    
    return np.column_stack((d2x, d2y, d2z))



def compute_frenet_frame(t, p, dp, ddp):
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

# Function to create an arrow mesh
def create_arrow(name, length=0.3):
    bpy.ops.mesh.primitive_cone_add(vertices=6, radius1=0.02, depth=0.1)
    arrow = bpy.context.object
    arrow.name = name
    bpy.ops.mesh.primitive_cylinder_add(radius=0.005, depth=length)
    shaft = bpy.context.object
    shaft.name = name + "_shaft"
    shaft.location.z += length/2
    arrow.parent = shaft
    bpy.context.view_layer.objects.active = shaft
    shaft.select_set(True)
    arrow.select_set(True)
    bpy.ops.object.join()
    return shaft

# Visualize T, N, B vectors as arrows at points
def visualize_in_blender(points, T, U, V, scale=0.3, step=10):
    for obj in bpy.data.objects:
        if obj.name.startswith("Frenet_"):
            bpy.data.objects.remove(obj, do_unlink=True)

    for i in range(0, len(points), step):
        p = points[i]
        
        # Tangent arrow (blue)
        t_arrow = create_arrow(f"Frenet_T_{i}", length=scale)
        t_arrow.location = p
        t_arrow.rotation_mode = 'QUATERNION'
        t_arrow.rotation_quaternion = direction_to_quaternion(T[i])
        t_arrow.active_material = get_material("Blue", (0,0,1,1))
        t_arrow.name = f"Frenet_T_{i}"

        # U arrow (green)
        n_arrow = create_arrow(f"Frenet_N_{i}", length=scale)
        n_arrow.location = p
        n_arrow.rotation_mode = 'QUATERNION'
        n_arrow.rotation_quaternion = direction_to_quaternion(U[i])
        n_arrow.active_material = get_material("Green", (0,1,0,1))
        n_arrow.name = f"Frenet_U_{i}"

        # V arrow (red)
        b_arrow = create_arrow(f"Frenet_V_{i}", length=scale)
        b_arrow.location = p
        b_arrow.rotation_mode = 'QUATERNION'
        b_arrow.rotation_quaternion = direction_to_quaternion(V[i])
        b_arrow.active_material = get_material("Red", (1,0,0,1))
        b_arrow.name = f"Frenet_V_{i}"

# Helper to convert vector direction to quaternion rotation
def direction_to_quaternion(direction, up=(0,0,1)):
    direction = mathutils.Vector(direction).normalized()
    up_vec = mathutils.Vector(up)
    axis = up_vec.cross(direction)
    if axis.length < 1e-6:
        if up_vec.dot(direction) > 0:
            return mathutils.Quaternion()  
        else:
            axis = up_vec.orthogonal()
            return mathutils.Quaternion(axis, 3.14159)
    angle = up_vec.angle(direction)
    return mathutils.Quaternion(axis, angle)

# Helper to get/create a material with given name and color
def get_material(name, rgba):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.diffuse_color = rgba
    return mat

def create_circles_along_curve(points, U, V, radius=0.1, num_segments=32):
    all_circles = []
    for i in range(len(points)):
        center = points[i]
        u = U[i]
        v = V[i]
        circle_points = []
        for j in range(num_segments):
            angle = (2 * np.pi * j) / num_segments
            offset = radius * (math.cos(angle) * u + math.sin(angle) * v)
            circle_point = center + offset
            circle_points.append(circle_point)
        all_circles.append(np.array(circle_points))
    return all_circles

def plot_circles_in_blender(all_circles):
    for circle_points in all_circles:
        mesh = bpy.data.meshes.new("CircleMesh")
        obj = bpy.data.objects.new("Circle", mesh)
        bpy.context.collection.objects.link(obj)
        verts = [tuple(p) for p in circle_points]
        edges = [(i, (i+1)%len(verts)) for i in range(len(verts))] 
        mesh.from_pydata(verts, edges, [])
        mesh.update()

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


def build_mesh(points, U, V, radius, segments=16):
    verts = []
    faces = []

    # Generate vertices for each circle
    for i in range(len(points)):
        # Support both constant radius and variable radius
        current_radius = radius[i] if isinstance(radius, (list, np.ndarray)) else radius
        for j in range(segments):
            angle = 2 * np.pi * j / segments
            offset = (np.cos(angle) * U[i]  + np.sin(angle) * V[i]) * current_radius  # * (1 + 0.5 * np.sin(2 * angle))
            verts.append(points[i] + offset)

    # Connect vertices between circles
    for i in range(len(points) - 1):
        for j in range(segments):
            v0 = i * segments + j
            v1 = i * segments + (j + 1) % segments
            v2 = (i + 1) * segments + (j + 1) % segments
            v3 = (i + 1) * segments + j
            faces.append([v0, v1, v2, v3])
            
    # Create mesh in Blender
    edges = [] # [(i - 1,   i) for i in range(1, len(points))]
    mesh_data = bpy.data.meshes.new("knittingMesh")
    mesh_data.from_pydata(verts, edges, faces)
    mesh_data.update()

    obj = bpy.data.objects.new("knittingObject", mesh_data)
    bpy.context.collection.objects.link(obj)
    return obj


def build_mesh_edges_only(points, U, V, radius, segments=16):

    created_objects = []
    num_points = len(points)
    
    # Group vertices by their position in the cross-section (which "strand" they belong to)
    for strand_idx in range(segments):
        verts = []
        edges = []
        
        # Collect all vertices for this strand along the entire curve
        for i in range(num_points):
            current_radius = radius[i] if isinstance(radius, (list, np.ndarray)) else radius
            angle = 2 * np.pi * strand_idx / segments
            offset = (np.cos(angle) * U[i] + np.sin(angle) * V[i]) * current_radius
            verts.append(points[i] + offset)
        
        # Create edges connecting consecutive points along this strand
        for i in range(len(verts) - 1):
            edges.append([i, i + 1])
        
        # Create mesh for this strand
        faces = []  # No faces, just edges
        mesh_data = bpy.data.meshes.new(f"Loop_Strand_{strand_idx}")
        mesh_data.from_pydata(verts, edges, faces)
        mesh_data.update()
        
        obj = bpy.data.objects.new(f"Loop_Strand_{strand_idx}", mesh_data)
        bpy.context.collection.objects.link(obj)
        created_objects.append(obj)
    
    return created_objects


def convert_edges_to_mesh_with_profile(obj, profile_radius=0.01, profile_segments=8):

    # Make the object active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Convert to curve first
    bpy.ops.object.convert(target='CURVE')
    
    # Set curve properties for circular bevel
    curve_data = obj.data
    curve_data.bevel_depth = profile_radius
    curve_data.bevel_resolution = profile_segments
    curve_data.use_fill_caps = True
    
    # Convert back to mesh to finalize
    bpy.ops.object.convert(target='MESH')
    
    obj.select_set(False)
    
    return obj


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
    stitch_count = 5 
    stitch_bulge = 0.26   
    stitch_height = 1.2  
    stitch_z = -0.48      
    yarn_radius = 0.19
    
    t = np.linspace(0, 2 * np.pi * n_loops, loop_res * n_loops + 1, endpoint=True)
    n = len(t)
    x_scale = np.repeat(x_scale, loop_res)
    x_scale = np.append(x_scale, 1)  # Add 1 to the end of the vector
    
    p = eval_curve(t, x_scale, stitch_count=stitch_count, stitch_bulge=stitch_bulge, 
                   stitch_height=stitch_height, stitch_z=stitch_z, yarn_radius=yarn_radius)
    p[:,1] += tx
    
    dp = eval_curve_derivative(t, x_scale, stitch_count=stitch_count, stitch_bulge=stitch_bulge,
                              stitch_height=stitch_height, stitch_z=stitch_z, yarn_radius=yarn_radius)
    ddp = eval_curve_second_derivative(t, x_scale, stitch_count=stitch_count, stitch_bulge=stitch_bulge,
                                      stitch_height=stitch_height, stitch_z=stitch_z, yarn_radius=yarn_radius)

    return t, p, dp, ddp

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


def generate_fibres(p, U, V, n_fibers, scale, angle, n_random = 0):
    # generate n_fibers around the main curve
    # p: points along the main curve
    # U, V: orthonormal frame along the curve
    # n_fibers: number of fibers to generate
    # scale: vector of scale factors along the curve
    # angle: twist angle along the curve
    # returns: list of fiber_points
    n = p.shape[0]

    # create a template
    # points on a circle in the UV plane
    a = 2 * np.pi * np.arange(n_fibers) / n_fibers
    pattern = np.stack((np.cos(a), np.sin(a)), axis=1)

    fibers = []
    for i in range(n_fibers):
        fiber_points = []
        for j in range(n):
            r = scale[j]
            a = angle[j]
            cp = r * pattern[i,:]
            cp_rotated = (cp[0] * math.cos(a) - cp[1] * math.sin(a),
                          cp[0] * math.sin(a) + cp[1] * math.cos(a))
            offset = cp_rotated[0] * U[j] + cp_rotated[1] * V[j]
            fiber_point = p[j] + offset
            fiber_points.append(fiber_point)
        fiber_points = np.array(fiber_points)
        fibers.append(fiber_points)
    
    # add some random fibers
    for _ in range(n_random):
        fiber_points = []
        offset_strength = 0.08 * random.uniform(0.8, 1.2) 
        offset_angle = random.uniform(0, 2 * np.pi)
        twist_phase = random.uniform(0, 2 * np.pi)
        twist_strength = 0.04 * random.uniform(0.8, 1.2)
        for j in range(n):
            r = scale[j]
            a = angle[j]
            cp = r * pattern[0,:]  
            cp_rotated = (cp[0] * math.cos(a) - cp[1] * math.sin(a),
                          cp[0] * math.sin(a) + cp[1] * math.cos(a))
            offset = cp_rotated[0] * U[j] + cp_rotated[1] * V[j]
            smooth_offset = offset_strength * (math.cos(offset_angle) * U[j] + math.sin(offset_angle) * V[j])
            twist = twist_strength * math.sin(j / n * 2 * np.pi + twist_phase)
            twist_offset = twist * U[j]
            fiber_point = p[j] + offset + smooth_offset + twist_offset
            fiber_points.append(fiber_point)
        fiber_points = np.array(fiber_points)
        fibers.append(fiber_points)
    
    return fibers

def generate_knitting_radius(base_radius, num_points, loop_res, row_index, scale, n_loops):

    variable_radius = []
    
    # Create realistic knitting parameters
    interlocking_factor = 0.8  # How much loops interlock with adjacent rows
    randomness = 0.15  # Natural yarn variation
    
    for j in range(num_points):
        # Current loop and position within loop
        loop_index = j // loop_res
        loop_position = (j % loop_res) / loop_res
         
        # Base yarn thickness varies naturally
        base_variation = 1.0 + randomness * np.sin(j * 0.37 + row_index * 2.1) * np.cos(j * 0.19)
        
        # Compression effects from neighboring loops
        # Loops compress each other where they touch
        neighbor_compression = 1.0
        
        # Horizontal compression from adjacent loops in same row
        if loop_index > 0 and loop_index < n_loops - 1:
            # More compression in middle of row
            side_compression = 0.85 + 0.15 * np.abs(np.sin(loop_position * np.pi))
            neighbor_compression *= side_compression
        
        # Vertical compression from rows above/below
        if row_index > 0:
            # Alternate rows nest into each other - offset pattern
            if row_index % 2 == 0:
                vertical_offset = loop_position + 0.5
            else:
                vertical_offset = loop_position
            
            # Create interlocking compression pattern
            interlock_pattern = 1.0 - 0.3 * np.exp(-8 * (vertical_offset % 1 - 0.5)**2)
            neighbor_compression *= interlocking_factor * interlock_pattern + (1 - interlocking_factor)
        
        # Loop shape affects compression - more squeezed at "waist"
        loop_shape_factor = 1.0
        if 0.2 < loop_position < 0.8:
            # Main body of loop - varies with position
            if 0.35 < loop_position < 0.65:
                # Crown of loop - less compressed
                loop_shape_factor = 1.2
            else:
                # Sides of loop - more compressed  
                loop_shape_factor = 0.9
        else:
            # Connection areas - heavily compressed
            connection_intensity = min(loop_position / 0.2, (1.0 - loop_position) / 0.2)
            loop_shape_factor = 0.6 + 0.3 * connection_intensity
        
        # Dropped stitch effects
        stitch_index = min(loop_index, len(scale) - 1) if len(scale) > 0 else 0
        stitch_scale = scale[stitch_index] if len(scale) > 0 else 1.0
        drop_factor = 0.4 + 0.6 * stitch_scale  # Thinner for dropped stitches
        
        # Combine all factors
        radius_multiplier = base_variation * neighbor_compression * loop_shape_factor * drop_factor
            
        # Apply the radius
        current_radius = base_radius * radius_multiplier
        
        # Ensure minimum thickness for mesh stability
        current_radius = max(current_radius, base_radius * 0.3)
        current_radius = min(current_radius, base_radius * 1.8)  # Cap maximum thickness
        
        variable_radius.append(current_radius)
    
    return variable_radius

def knitting_loop_main(map):
    dy = 0.35  # Reduced spacing between rows for better interlocking

    scale_factor = convert_bitmap_to_scales_factors(map)
    scale_factor = np.where((scale_factor <= 1), scale_factor, 1 + dy * (scale_factor - 1))

    n_loops = map.shape[1]
    print(f"Number of loops: {n_loops}")
    n_rows = map.shape[0]
    print(f"Number of rows: {n_rows}")
    loop_res = 128    # Reduced resolution for cleaner geometry
    num_points = loop_res * n_loops

    del_obj = bpy.context.active_object
    if del_obj:
        bpy.data.objects.remove(del_obj, do_unlink=True)

    created_objects = []
    for i in range(len(scale_factor)):
        scale = scale_factor[i]
        t, p, dp, ddp = create_curve(loop_res, n_loops, scale, i * dy)
        T = dp / (np.linalg.norm(dp, axis=1, keepdims=True) + 1e-8)
        T, U, V = compute_orthonormal_frame(T)

        # Create variable yarn radius along the curve
        base_radius = 0.15  # Base yarn thickness for realistic knitting
        variable_radius = generate_knitting_radius(
            base_radius=base_radius,
            num_points=len(p),
            loop_res=loop_res,
            row_index=i,
            scale=scale,
            n_loops=n_loops
        )
        
        # Create yarn mesh with variable radius and higher resolution for detail
        # obj_yarn = build_mesh(p, U, V, radius=variable_radius, segments=12)
        loop_objects = build_mesh_edges_only(p, U, V, radius=variable_radius, segments=35)
        
        # Convert each loop strand to mesh with circular profile
        for obj_yarn in loop_objects:
            # Convert edges to mesh with circular profile
            convert_edges_to_mesh_with_profile(obj_yarn, profile_radius=0.01, profile_segments=8)
            
            apply_uv_mapping(obj_yarn, segments=12, n_points=len(p))
            add_duplicate_index(obj_yarn, i)
            created_objects.append(obj_yarn)

    # # Join all rows into a single mesh
    # if created_objects:
    #     merged_obj = join_objects(created_objects)
    #     scale_uv_map(merged_obj, u_scale=1.0, v_scale=1.0)
        
    #     # Add smoothing
    #     bpy.context.view_layer.objects.active = merged_obj
    #     bpy.ops.object.shade_smooth()
        
    #     # Add subdivision surface for smoother knitting appearance
    #     subsurf_mod = merged_obj.modifiers.new(name="SubSurf", type='SUBSURF')
    #     subsurf_mod.levels = 2


    # merged_obj = join_objects(created_objects)
    # scale_uv_map(merged_obj, u_scale=1.0, v_scale=34.0)
    # cloth_mod = merged_obj.modifiers.new(name="Cloth", type='CLOTH')
    # cloth_mod.point_cache.frame_end = 100
    # solidify_mod = merged_obj.modifiers.new(name="Solidify", type='SOLIDIFY')
    # solidify_mod.thickness = 0.02

    ### smooth object
    bpy.ops.object.shade_smooth()


    
#%% main

def main():
    app = QApplication(sys.argv)
    color_window = pick_colors_gui.ColorPickerApp(on_see_result=None)
    render_window = render_images_gui.RenderImagesApp(None, None, other_window=color_window)
    color_window.move(100, 100)
    render_window.move(1100, 100)

    color_window.show()
    render_window.show()

    # Define callback after windows created
    def on_colors_updated(bitmap, colors):
        bitmap = bitmap[::-1]
        colors = colors[::-1]

        knitting_loop_main(bitmap)
        obj_to_mesh.add_geo()
        coloring.set_colors(colors, "input_")

        obj = bpy.context.active_object
        if obj:
            cam = bpy.data.objects.get("Camera")
            if cam:
                cam.location = (8, -4, 4)

            rendering.render_model(obj)
            render_window.obj = obj
            render_window.render_callback = lambda o: rendering.render_more_combinations(o, colors)
            render_window.refresh_render()
        else:
            print("No object to render")

    # Assign the real callback to the color picker window
    color_window.on_see_result = on_colors_updated

    app.exec()

Gui = False
if Gui:
    from PyQt6.QtWidgets import QApplication
    main()
else:
    # Run with default bitmap and colors, no GUI
    print("Running in non-GUI mode with default bitmap and colors.")
    # bitmap = [
    #     [1, 1, 1, 1, 1],
    #     [1, 1, 0, 1, 1],
    #     [1, 0, 0, 0, 1],
    #     [1, 0, 0, 0, 1],
    #     [1, 1, 1, 1, 1],
    # ]
    bitmap = [
        [1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1],
        # [1, 1, 1, 1, 1],
        # [1, 1, 1, 1, 1],
        # [1, 1, 1, 1, 1],
        # [1, 1, 1, 1, 1],
        # [1, 1, 1, 1, 1],
        # [1, 1, 1, 1, 1],
        # [1, 1, 1, 1, 1],
        # [1, 1, 1, 1, 1],
    ]
    bitmap = np.array(bitmap)
    # colors = [
    #     (1,0,0,1),
    #     (0,1,0,1),
    #     (0,0,1,1),
    #     (1,1,0,1),
    #     (1,0,1,1),
    # ]
    colors = [
        (1,0,0,1),
        (0,1,0,1),
        (0,0,1,1),
    ]
    bitmap = bitmap[::-1]
    colors = colors[::-1]
    knitting_loop_main(bitmap)  # Using knitted variation for realistic fabric
    # obj_to_mesh.add_geo()
    # coloring.set_colors(colors, "input_")
    