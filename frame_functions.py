import math
import bpy
import numpy as np
import mathutils
import bpy


def eval_curve_derivative(t, scale, ax=0.25, az=-0.2):
    dx = 2 * ax * np.cos(2 * t) + 1 / (2 * np.pi)
    dy = 0.5 * np.sin(t) * scale
    dz = -az * np.sin(2 * t) * scale
    dx = np.where(scale == 0, 1 / (2 * np.pi), dx)
    return dx, dy, dz

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
