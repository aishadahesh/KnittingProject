#%% imports
import bpy
import numpy as np
import sys
sys.path.append(r"C:\projects\KnittingProject")     
# import pick_colors_gui
# import render_images_gui
import obj_to_mesh
# import coloring
import rendering
# import knitting_loop


#%% frame functions
import math
import bpy
import numpy as np
import mathutils
import bpy


def eval_curve(t, scale, ax=0.25, az=-0.2):
    t = np.asarray(t, dtype=float)
    scale = np.asarray(scale, dtype=float)
    x = ax * np.sin(2*t) + t/(2*np.pi)
    y = -(np.cos(t) - 1)/2
    z = az * (np.cos(2*t) - 1)/2
    x = np.where(scale == 0, t/(2*np.pi), x)
    return np.column_stack((x, y*scale, z*scale))

def eval_curve_derivative(t, scale, ax=0.25, az=-0.2):
    t = np.asarray(t, dtype=float)
    scale = np.asarray(scale, dtype=float)
    dx = 2*ax*np.cos(2*t) + 1/(2*np.pi)
    dy = 0.5*np.sin(t)*scale
    dz = -az*np.sin(2*t)*scale
    dx = np.where(scale == 0, 1/(2*np.pi), dx)
    return np.column_stack((dx, dy, dz))

def eval_curve_second_derivative(t, scale, ax=0.25, az=-0.2):
    t = np.asarray(t, dtype=float)
    scale = np.asarray(scale, dtype=float)
    d2x = -4*ax*np.sin(2*t)
    d2y = 0.5*np.cos(t)*scale
    d2z = -2*az*np.cos(2*t)*scale
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
            
    # Create mesh in Blender
    edges = [] # [(i - 1,   i) for i in range(1, len(points))]
    mesh_data = bpy.data.meshes.new("knittingMesh")
    mesh_data.from_pydata(verts, edges, faces)
    mesh_data.update()

    obj = bpy.data.objects.new("knittingObject", mesh_data)
    bpy.context.collection.objects.link(obj)
    return obj



#%% knitting loop functions
import bpy
import numpy as np
# import frame_functions
import mathutils

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
    # t_values = 2 * np.pi * np.arange(loop_res * n_loops) / loop_res
    t = np.linspace(0, 2 * np.pi * n_loops, loop_res * n_loops + 1, endpoint=True)
    n = len(t)
    x_scale = np.repeat(x_scale, loop_res)
    x_scale = np.append(x_scale, 1)  # Add 1 to the end of the vector
    p = eval_curve(t, x_scale)
    p[:,1] += tx
    dp = eval_curve_derivative(t, x_scale)
    ddp = eval_curve_second_derivative(t, x_scale)

    return t,p, dp, ddp

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


def generate_fibres(p, U, V, n_fibers, scale, angle):
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
    
    return fibers

def knitting_loop_main(map):
    dy = 0.55

    scale_factor = convert_bitmap_to_scales_factors(map)
    scale_factor = np.where((scale_factor <= 1), scale_factor, 1 + dy * (scale_factor - 1))

    n_loops = map.shape[1]
    print(f"Number of loops: {n_loops}")
    n_rows = map.shape[0]
    print(f"Number of rows: {n_rows}")
    loop_res = 128    # loop resolution
    num_points = loop_res * n_loops

    del_obj = bpy.context.active_object
    if del_obj:
        bpy.data.objects.remove(del_obj, do_unlink=True)

    created_objects = []
    for i in range(len(scale_factor)):
        scale = scale_factor[i]
        t, p, dp, ddp = create_curve(loop_res, n_loops, scale, i * dy)
        T, U, V = compute_frenet_frame(t, p, dp, ddp)
        # T, U, V = compute_orthonormal_frame(dpoints)
        # obj_fiber = build_mesh(p, U, V, radius=0.05)
        # created_objects.append(obj_fiber)

        yarn_radius = 0.08 + 0.05*np.sin(np.linspace(0, 8 * np.pi * n_loops, num_points + 1, endpoint=True))
        angle = np.linspace(0, 1 * np.pi * n_loops, num_points + 1, endpoint=True)  # twist angle along the yarn
        fiber_radius = 0.01
        n_fibers = 32

        fibers = generate_fibres(p, U, V, n_fibers=n_fibers, scale=yarn_radius, angle=angle)
        for p in fibers:
            # build mesh for each fiber with smaller tube radius
            obj_fiber = build_mesh(p, U, V, radius=fiber_radius)
            created_objects.append(obj_fiber)
            add_duplicate_index(obj_fiber, i)



    join_objects(created_objects)

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
        #obj_to_mesh.add_geo()
        #coloring.set_colors(colors, "input_")

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
    bitmap = [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    bitmap = np.array(bitmap)
    colors = [
        (1,0,0,1),
        (0,1,0,1),
        (0,0,1,1),
        (1,1,0,1),
    ]
    bitmap = bitmap[::-1]
    colors = colors[::-1]
    knitting_loop_main(bitmap)
    # obj_to_mesh.add_geo()
    # coloring.set_colors(colors, "input_")
    
    
