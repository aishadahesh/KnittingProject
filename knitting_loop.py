import random
import bpy
import math
import collections
import bmesh

val1 = 6.283  # change the closeness between two curves (frequency)
val2 = 2     # affect number of twists in loop
val3 = 0.110  # the wave height of the loop in Y direction.
val4 = 0.040    # controls how far the loop "swings" on X.
val5 = -0.030 # controls vertical variation in Z 
clamp_factor = True  


def convert_map_to_input(map):
    # map = [[0,0,1], [0,0,1],[1,1,1]]
    output = [row.copy() for row in map]
    rows = len(map)
    cols = len(map[0])
    for col in range(cols):
        zero = 0
        prev= map[0][col]
        for row in range(rows):
            if map[row][col]==1:
                if prev == 0:
                    output[row][col] = zero
                    break
            elif map[row][col] == 0:
                zero+=1
    return output
    #return [[0,0,1], [0,0,1], [2,2,1]]

def compute_offset(loop_index, loop_factor, group_input,scale_x=1, val1=6.283, val2=2, val3=0.110, val4=0.04, val5=-0.030):
    
    cos1 = math.cos(loop_factor * val1)
    sin1 = math.sin(loop_factor * val1 * val2)

    x = cos1 * val3 - val3      # control the waves of loop
    z = math.cos(loop_factor * val1 * val2) * val5  

    add1 = max(0.0, min(1.0, loop_index + group_input))
    y = (1 - add1) * loop_factor + add1 * (sin1 * val4) 
    
    return (x*scale_x, y, z)

num_points = 200    # controls the length of the loop
points_per_loop = 32    # controls the closeness of the curves
map = [[0,0,1], [0,0,1],[1,1,1]]
input = convert_map_to_input(map)
def create_loop(x_transform = 0.0, y_transform=0.0, dup_index=0):
    verts = []
    edges = []

    for i in range(num_points):
        x = 0 
        y = i*0.005
        z = 0

        loop_index = i // points_per_loop
        loop_factor = (i % points_per_loop) / points_per_loop
        group_input = 2  
        group_index = loop_index % len(input)
        dup_col_index = (dup_index+2) % len(input[0])
        scale = input[dup_col_index][group_index]

        if scale!=0:
            dx, dy, dz = compute_offset(loop_index, loop_factor, group_input, scale)
        else:
            dx, dy, dz = 0, 0, 0

        pos = (x + dx + x_transform, y + dy + y_transform, z + dz)
        verts.append(pos)

        if i > 0:
            edges.append((i - 1, i))

    mesh_data = bpy.data.meshes.new("knittingMesh")
    mesh_data.from_pydata(verts, edges, [])
    mesh_data.update()

    obj = bpy.data.objects.new("knittingObject", mesh_data)
    bpy.context.collection.objects.link(obj)
    return obj


print("knitting loop object created.")


print("Creating loops...")

y_transform = -0.960143     # transform loop by y and merge 2 loops in the same row
x_transform = -0.120    # distance on x between loops
dup = 3

created_objects = []
for i in range(dup):
    obj1 = create_loop(i * x_transform, 0,i)
    obj2 = create_loop(i * x_transform, y_transform,i)
    created_objects.extend([obj1, obj2])

# Now merge all created objects into one mesh

def join_objects(objects, new_name="MergedLoops"):
    # Deselect all objects first
    bpy.ops.object.select_all(action='DESELECT')

    # Select all objects to join
    for obj in objects:
        obj.select_set(True)

    # Set the active object (the one that will remain after join)
    bpy.context.view_layer.objects.active = objects[0]

    # Join selected objects into the active object
    bpy.ops.object.join()

    # Rename the joined object
    merged_obj = bpy.context.view_layer.objects.active
    merged_obj.name = new_name

    return merged_obj


# Select all created objects and join them
merged_obj = join_objects(created_objects)

print(f"All loops merged into object: {merged_obj.name}")



#################### CURVE TO MESH ####################

# Convert merged mesh to curve
bpy.context.view_layer.objects.active = merged_obj
merged_obj.select_set(True)
bpy.ops.object.convert(target='CURVE')
curve_obj = bpy.context.view_layer.objects.active
curve_obj.name = "MergedLoopCurve"

# Create a smooth profile circle
bpy.ops.curve.primitive_bezier_circle_add(radius=0.025)     # radius controls the thickness of the loop
profile_curve_obj = bpy.context.active_object
profile_curve_obj.name = "ProfileCircle"

# Assign as bevel object and increase resolution
curve_obj.data.bevel_mode = 'OBJECT'
curve_obj.data.bevel_object = profile_curve_obj
curve_obj.data.resolution_u = 24
curve_obj.data.bevel_resolution = 16

# Convert to mesh 
bpy.ops.object.select_all(action='DESELECT')
curve_obj.select_set(True)
bpy.context.view_layer.objects.active = curve_obj
bpy.ops.object.mode_set(mode='OBJECT')  # Ensure Object mode
bpy.ops.object.convert(target='MESH')
final_obj = bpy.context.view_layer.objects.active

# Shade smooth and enable auto smooth
bpy.ops.object.shade_smooth()

# Remove the profile curve
bpy.data.objects.remove(profile_curve_obj, do_unlink=True)

