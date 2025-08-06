import os
import bpy
import numpy as np
import sys
sys.path.append(r"C:\Users\Aisha\KnittingProject")     
import pick_colors
import render_images
import itertools
import obj_to_mesh
import coloring
import rendering


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

def eval_curve(t, scale, ax=0.25, az=-0.2):
    x = ax * np.sin(2 * t) +  t / (2 * np.pi)
    y = - (np.cos(t) - 1)/2
    z = az * (np.cos(2 * t) - 1)/2
    x = np.where(scale==0, t / (2 * np.pi), x) # when scale is zero, use a linear function, otherwise the line will overlap itself
    return (x, y*scale, z*scale)

def create_curve(loop_res, n_loops, x_scale, tx = 0.0):
    # t_values = 2 * np.pi * np.arange(loop_res * n_loops) / loop_res
    t_values = np.linspace(0, 2 * np.pi * n_loops, loop_res * n_loops + 1, endpoint=True)
    num_points = len(t_values)
    x_scale = np.repeat(x_scale, loop_res)
    x_scale = np.append(x_scale, 1)  # Add 1 to the end of the vector
    x, y, z = eval_curve(t_values, x_scale)
    points = [(x[i], y[i] + tx, z[i]) for i in range(num_points)]
    
    return points

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


def main(map):
    dy = 0.55

    scale_factor = convert_bitmap_to_scales_factors(map)
    scale_factor = np.where((scale_factor <= 1), scale_factor, 1 + dy * (scale_factor - 1))

    n_loops = map.shape[1]
    print(f"Number of loops: {n_loops}")
    n_rows = map.shape[0]
    print(f"Number of rows: {n_rows}")
    loop_res = 32    # loop resolution
    num_points = loop_res * n_loops

    del_obj = bpy.context.active_object
    if del_obj:
        bpy.data.objects.remove(del_obj, do_unlink=True)

    created_objects = []
    for i in range(len(scale_factor)):
        scale = scale_factor[i]
        points = create_curve(loop_res, n_loops, scale, i * dy)
        
        edges = [(i - 1, i) for i in range(1, len(points))]

        mesh_data = bpy.data.meshes.new("knittingMesh")
        mesh_data.from_pydata(points, edges, [])
        mesh_data.update()

        obj = bpy.data.objects.new("knittingObject", mesh_data)
        bpy.context.collection.objects.link(obj)

        created_objects.append(obj)
        add_duplicate_index(obj, i)

    # Select all created objects and join them
    join_objects(created_objects)



pick_colors.run_color_app()
map = np.load("bitmap.npy")
map=map[::-1]
colors = np.load("colors.npy")
colors = colors[::-1]
main(map)

# convert object to mesh
obj_to_mesh.convert_obj_to_mesh()

# add colors to mesh
coloring.set_colors(colors)

# render images
obj = bpy.context.active_object
if obj:
    camera = bpy.data.objects.get("Camera")
    if not camera:
        print("No camera found in the scene. Please add a camera.")
    camera.location = (8, -4, 4)

    rendering.render_model(obj)
    render_images.run_rendering_app(obj, lambda o: rendering.render_more_combinations(o, colors))
else:
    print("No active object to render.")
    