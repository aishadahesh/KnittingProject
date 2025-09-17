#%% imports
import bpy
import numpy as np
from PyQt6.QtWidgets import QApplication
import sys
sys.path.append(r"C:\Users\Aisha\KnittingProject")     
import pick_colors_gui
import render_images_gui
import obj_to_mesh
import coloring
import rendering
# import knitting_loop


#%% frame functions
import math
import bpy
import numpy as np
import mathutils
import bpy


def eval_curve(t, scale, ax=0.25, az=-0.2):
    x = ax * np.sin(2 * t) +  t / (2 * np.pi)
    y = - (np.cos(t) - 1)/2
    z = az * (np.cos(2 * t) - 1)/2
    x = np.where(scale==0, t / (2 * np.pi), x) # when scale is zero, use a linear function, otherwise the line will overlap itself
    return (x, y*scale, z*scale)

def eval_curve_derivative(t, scale, ax=0.25, az=-0.2):
    dx = 2 * ax * np.cos(2 * t) + 1 / (2 * np.pi)
    dy = 0.5 * np.sin(t) * scale
    dz = -az * np.sin(2 * t) * scale
    dx = np.where(scale == 0, 1 / (2 * np.pi), dx)
    return dx, dy, dz


def eval_curve_second_derivative(t, scale, ax=0.25, az=-0.2):
    d2x = -4 * ax * np.sin(2 * t)
    d2y = 0.5 * np.cos(t) * scale
    d2z = -2 * az * np.cos(2 * t) * scale
    d2x = np.where(scale == 0, 0, d2x)
    return d2x, d2y, d2z


# Compute the Frenet frame (T, N, B) for a parametric curve
def compute_frenet_frame(points, dpoints, ddpoints):
    d1 = np.array(dpoints)  # First derivative (tangent)
    d2 = np.array(ddpoints)  # Second derivative (curvature)
    # Tangent vector (normalized first derivative)
    T = d1 / (np.linalg.norm(d1, axis=-1, keepdims=True) + 1e-8)
    # Normal vector (normalized derivative of T)
    dT = d2 - (np.sum(d2 * T, axis=-1, keepdims=True)) * T
    N = dT / (np.linalg.norm(dT, axis=-1, keepdims=True) + 1e-8)
    # Binormal vector (cross product of T and N)
    B = np.cross(T, N)
    B = B / (np.linalg.norm(B, axis=-1, keepdims=True) + 1e-8)
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

def build_mesh(points, U, V, radius=0.115, segments=16):
    verts = []
    faces = []

    # Generate vertices for each circle
    for i in range(len(points)):
        for j in range(segments):
            angle = 2 * np.pi * j / segments
            offset = np.cos(angle) * U[i] * radius + np.sin(angle) * V[i] * radius
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
    edges = [(i - 1, i) for i in range(1, len(points))]
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
    t_values = np.linspace(0, 2 * np.pi * n_loops, loop_res * n_loops + 1, endpoint=True)
    num_points = len(t_values)
    x_scale = np.repeat(x_scale, loop_res)
    x_scale = np.append(x_scale, 1)  # Add 1 to the end of the vector
    x, y, z = eval_curve(t_values, x_scale)
    dx, dy, dz = eval_curve_derivative(t_values, x_scale)
    ddx, ddy, ddz = eval_curve_derivative(t_values, x_scale)
    points = [(x[i], y[i] + tx, z[i]) for i in range(num_points)]
    dpoints = [(dx[i], dy[i], dz[i]) for i in range(num_points)]
    ddpoints = [(ddx[i], ddy[i], ddz[i]) for i in range(num_points)]
    
    return t_values,points, dpoints, ddpoints

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


def create_twisted(points, U, V, t_values, radius=0.01, omega=6.0, n_fibers=4, phi0=0.0):
    """
    Returns list of fibers: each fiber is (points_list, U_list, V_list)
    - points: list of 3D positions (same length as points)
    - U,V: rotated local frames for that fiber
    t_values: array of parameter t per sample - used for theta = omega * t + phi
    """
    # convert U,V to numpy arrays shape (n,3)
    U = np.asarray(U)
    V = np.asarray(V)
    pts_np = np.asarray(points)  # shape (n,3)
    n = pts_np.shape[0]

    # if U/V are shape (3,n) transpose if needed
    if U.shape[0] == 3 and U.shape[1] == n:
        U = U.T
    if V.shape[0] == 3 and V.shape[1] == n:
        V = V.T

    # normalize 
    U = U / np.linalg.norm(U, axis=1)[:, None]
    V = V / np.linalg.norm(V, axis=1)[:, None]

    fibers = []
    dphi = 2.0 * np.pi / float(n_fibers)
    for k in range(n_fibers):
        phi = phi0 + k * dphi
        fiber_pts = []
        fiber_U = []
        fiber_V = []
        for i in range(n):
            t = t_values[i]
            theta = omega * t + phi  # use real param t
            cu = np.cos(theta)
            su = np.sin(theta)
            u = U[i]
            v = V[i]
            offset = radius[i] * (cu * u + su * v)
            p = pts_np[i] + offset

            fiber_pts.append((float(p[0]), float(p[1]), float(p[2])))
            fiber_U.append(u)
            fiber_V.append(v)
        fibers.append((fiber_pts, fiber_U, fiber_V))
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
        t_values, points, dpoints, ddpoints = create_curve(loop_res, n_loops, scale, i * dy)
        
        # Convert points to np.array for calculation
        T, U, V = compute_frenet_frame(points, dpoints, ddpoints)

        # Ensure U and V are shape (n,3) numpy arrays
        U = np.asarray(U)
        V = np.asarray(V)
        if U.shape[0] == 3 and U.shape[1] == len(points):
            U = U.T
        if V.shape[0] == 3 and V.shape[1] == len(points):
            V = V.T

        # --- Main core yarn: build a core mesh (optional) ---
        core_radius = 0.1  # try larger core if you want a thicker yarn
        # obj_core = frame_functions.build_mesh(points, U, V, radius=core_radius)
        # created_objects.append(obj_core)
        # add_duplicate_index(obj_core, i)

        # --- Twisted fibers around core ---
        fiber_radius = 0.0  # much smaller than core
        n_fibers = 1
        omega = 6.0  # try 4..8

        pts_np = np.asarray(points)  # shape (n,3)
        n = pts_np.shape[0]
        # Make radius a sine function along the fiber
        t = np.linspace(0, 0, n)
        base_radius = core_radius
        radius = base_radius + t
        print(f"Fiber radius range: {radius.min()} to {radius.max()}")
        fibers = create_twisted(points, U, V, t_values, radius=radius, omega=omega, n_fibers=n_fibers)
        for f_points, f_U, f_V in fibers:
            # build mesh for each fiber with smaller tube radius
            obj_fiber = build_mesh(f_points, f_U, f_V, radius=fiber_radius)
            created_objects.append(obj_fiber)
            add_duplicate_index(obj_fiber, i)



    #join_objects(created_objects)

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
    #obj_to_mesh.add_geo()
    #coloring.set_colors(colors, "input_")
    
    
