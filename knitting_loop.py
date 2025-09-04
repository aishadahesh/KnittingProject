import bpy
import numpy as np
import frame_functions
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
    dx, dy, dz = frame_functions.eval_curve_derivative(t_values, x_scale)
    points = [(x[i], y[i] + tx, z[i]) for i in range(num_points)]
    
    return points, dx, dy, dz

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

offsets = [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05], [-0.05, 0, 0], [0, -0.05, 0], [0, 0, -0.05], 
           [0.025, 0, 0], [0, 0.025, 0], [0, 0, 0.025], [-0.025, 0, 0], [0, -0.025, 0], [0, 0, -0.025],
           [0.075, 0, 0], [0, 0.075, 0], [0, 0, 0.075], [-0.075, 0, 0], [0, -0.075, 0], [0, 0, -0.075],
           [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01], [-0.01, 0, 0], [0, -0.01, 0], [0, 0, -0.01],
           [0.015, 0, 0], [0, 0.015, 0], [0, 0, 0.015], [-0.015, 0, 0], [0, -0.015, 0], [0, 0, -0.015],
           [0.09, 0, 0], [0, 0.09, 0], [0, 0, 0.09], [-0.09, 0, 0], [0, -0.09, 0], [0, 0, -0.09],
           [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [-0.1, 0, 0], [0, -0.1, 0], [0, 0, -0.1],
           [0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.03], [-0.03, 0, 0], [0, -0.03, 0], [0, 0, -0.03],
           [0.08, 0, 0], [0, 0.08, 0], [0, 0, 0.08], [-0.08, 0, 0], [0, -0.08, 0], [0, 0, -0.08]]

def duplicate_loop(obj, objs_list=None):
    global offsets
    duplicates = []
    base_loc = obj.location.copy()

    for i in range(len(offsets)):
        # Create a new object sharing the same mesh data
        dup = obj.copy()
        dup.data = obj.data.copy()  
        dup.location = base_loc + mathutils.Vector((
            offsets[i][0],
            offsets[i][1],
            offsets[i][2]
        ))
        dup.name = f"{obj.name}_dup_{i}"  
        bpy.context.collection.objects.link(dup)
        duplicates.append(dup)
        if objs_list is not None:
            objs_list.append(dup)

    return duplicates

def join_loop(objects, new_name="MergedLoops"):
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
        points, dx_, dy_, dz_ = create_curve(loop_res, n_loops, scale, i * dy)
        
        # Convert points to np.array for calculation
        points_np = np.array([
            (p[0], p[1], p[2]) for p in points
        ])
        T = np.column_stack((dx_, dy_, dz_))
        T, U, V = frame_functions.compute_orthonormal_frame(T)
        # frame_functions.visualize_in_blender(points_np, T, U, V, scale=0.3, step=5)
        # all_circles = frame_functions.create_circles_along_curve(points_np, U, V, radius=0.05)
        # frame_functions.plot_circles_in_blender(all_circles)
        obj = frame_functions.build_mesh(points, U, V, radius=0.01)

        # edges = [(i - 1, i) for i in range(1, len(points))]

        # mesh_data = bpy.data.meshes.new("knittingMesh")
        # mesh_data.from_pydata(points, edges, [])
        # mesh_data.update()

        # obj = bpy.data.objects.new("knittingObject", mesh_data)
        # bpy.context.collection.objects.link(obj)

        created_objects.append(obj)
        add_duplicate_index(obj, i)

    global offsets
    merged_loops = []  # store each loop's merged object

    for i in range(n_loops):
        duplicates = duplicate_loop(created_objects[i])
        merged = join_loop([created_objects[i]] + duplicates, f"MergedLoop{i}")
        merged_loops.append(merged)
        
    join_objects(merged_loops)

    