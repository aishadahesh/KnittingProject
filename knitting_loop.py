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
                    output[row][col] = zero+1
                    break
            elif map[row][col] == 0:
                zero+=1
    print("OUTPUT")
    print(output)
    print("OUTPUT[::-1]")
    print(output[::-1])
    return output[::-1]

    #return [[0,0,1], [0,0,1], [2,2,1]]

def compute_offset(loop_index, loop_factor, group_input,scale_x=1, val1=6.283, val2=2, val3=0.110, val4=0.04, val5=-0.030):
    
    cos1 = math.cos(loop_factor * val1)
    sin1 = math.sin(loop_factor * val1 * val2)

    x = scale_x* cos1 * val3 - val3      # control the waves of loop
    z = math.cos(loop_factor * val1 * val2) * val5  

    add1 = max(0.0, min(1.0, loop_index + group_input))
    y = (1 - add1) * loop_factor + add1 * (sin1 * val4) 
    
    return (x, y, z)

points_per_loop = 32    # controls the closeness of the curves
num_of_loops = 7
num_points = points_per_loop * num_of_loops    # controls the length of the loop

map = [[0,0,0,1], 
       [0,0,1,1],
       [0,1,1,1],
       [1,1,1,1]]
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
        group_index = loop_index % len(input[0])
        dup_col_index = (dup_index) % len(input)
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

def add_duplicate_index(obj, value):
    mesh = obj.data

    # Only add if object has data and is a mesh
    if not mesh or obj.type != 'MESH':
        print(f"{obj.name} is not a valid mesh object.")
        return

    # Create the attribute if not exists
    if "duplicate_index" not in mesh.attributes:
        mesh.attributes.new(name="duplicate_index", type='FLOAT', domain='POINT')

    attr = mesh.attributes["duplicate_index"].data

    # Set value for all points
    for i in range(len(attr)):
        attr[i].value = value

y_transform = -0.960143     # transform loop by y and merge 2 loops in the same row
x_transform = -0.120    # distance on x between loops
dup = 4


named_attr_dup = 0
created_objects = []
for i in range(dup):
    obj1 = create_loop(i * x_transform, 0,i)
    obj2 = create_loop(i * x_transform, y_transform,i)
    created_objects.extend([obj1, obj2])
    named_attr_dup = i
    # named attribute duplicate
    add_duplicate_index(obj1, i)
    add_duplicate_index(obj2, i)

    
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



########################  CONVERT OBJECT TO MESH ########################

group_name = "AutoMeshToCurve"
if group_name in bpy.data.node_groups:
    node_group = bpy.data.node_groups[group_name]
else:
    node_group = bpy.data.node_groups.new(name=group_name, type='GeometryNodeTree')

    # Clear interface and nodes
    node_group.interface.clear()
    node_group.nodes.clear()

    # Add interface sockets
    node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    # Create nodes
    nodes = node_group.nodes
    links = node_group.links

    input_node = nodes.new("NodeGroupInput")
    output_node = nodes.new("NodeGroupOutput")
    mesh_to_curve = nodes.new("GeometryNodeMeshToCurve")
    curve_to_mesh = nodes.new("GeometryNodeCurveToMesh")
    curve_circle = nodes.new("GeometryNodeCurvePrimitiveCircle")

    curve_to_mesh.label= "CurveToMesh_"
    # Arrange nodes
    input_node.location = (-600, 0)
    mesh_to_curve.location = (-300, 0)
    curve_circle.location = (-300, -200)
    curve_to_mesh.location = (0, 0)
    output_node.location = (900, 0)

    # Make links
    links.new(mesh_to_curve.inputs["Mesh"], input_node.outputs["Geometry"])
    links.new(curve_to_mesh.inputs["Curve"], mesh_to_curve.outputs["Curve"])
    links.new(curve_to_mesh.inputs["Profile Curve"], curve_circle.outputs["Curve"])
    links.new(output_node.inputs["Geometry"], curve_to_mesh.outputs["Mesh"])

    # Set parameters
    curve_circle.inputs["Radius"].default_value = 0.02
    curve_circle.inputs["Resolution"].default_value = 16

print(f"✅ Node group '{group_name}' is ready.")


obj_name = "MergedLoops"
obj = bpy.data.objects.get(obj_name)

if obj:
    # Remove previous Geo Node modifiers (optional)
    for mod in obj.modifiers:
        if mod.type == 'NODES' and mod.node_group and mod.node_group.name == group_name:
            obj.modifiers.remove(mod)

    # Add new modifier
    mod = obj.modifiers.new(name="AutoCurveMod", type='NODES')
    mod.node_group = node_group
    print(f"✅ Modifier added to object: {obj.name}")
else:
    print(f"❌ Object '{obj_name}' not found.")




############################ COLORING ############################

# Define RGB colors with alpha
colors = [
    (0.0, 0.0, 1.0, 1.0),  # Blue
    (1.0, 0.0, 0.0, 1.0),  # Red
    (0.0, 1.0, 0.0, 1.0),  # Green
]

materials = []
for i, rgba in enumerate(colors):
    mat_name = f"Material_{i}"

    # Check if the material already exists
    if mat_name in bpy.data.materials:
        material = bpy.data.materials[mat_name]
    else:
        material = bpy.data.materials.new(name=mat_name)
        material.use_nodes = True

    # Update the material's base color
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba

    materials.append(material)

# Apply to object
obj = bpy.context.active_object 
obj.data.materials.clear()
for material in materials:
    obj.data.materials.append(material)


# Ensure the object has a Geometry Nodes modifier
geo_nodes_mod = None
for mod in obj.modifiers:
    if mod.type == 'NODES':
        geo_nodes_mod = mod
        break

if not geo_nodes_mod:
    print("No Geometry Nodes modifier found on the object.")

# Access the Geometry Nodes tree
node_tree = geo_nodes_mod.node_group
if not node_tree:
    print("No node group found in the Geometry Nodes modifier.")

for i, material in enumerate(materials):
        set_material_node = node_tree.nodes.new(type="GeometryNodeSetMaterial")
        set_material_node.location = (600, 100 * i)  # Position the node
        print(f"Added a Set Material node at location {set_material_node.location}.")
        # Assign the material to the Set Material node
        set_material_node.inputs[2].default_value = material
        print(f"Assigned material {material.name} to the Set Material node.")

# Find the Curve to Mesh node
curve_to_mesh_node = None
for node in node_tree.nodes:
    if node.label == "CurveToMesh_":
        curve_to_mesh_node = node
        break

if not curve_to_mesh_node:
    print("No Curve to Mesh node found in the Geometry Nodes tree.")

# Add a Join Geometry node to combine all outputs
join_geometry_node = node_tree.nodes.new(type="GeometryNodeJoinGeometry")
join_geometry_node.location = (200, 400)  # Position the node
print("Added a Join Geometry node.")

floored_modulo_node = node_tree.nodes.new(type="ShaderNodeMath")
floored_modulo_node.operation = "FLOORED_MODULO"
floored_modulo_node.location = (230, 200)  # Position the node
print("Added a FLOORED MODULO node.")


# Link each Set Material node to the Join Geometry node
j = 0
for i, node in enumerate(node_tree.nodes):
    if node.type == "SET_MATERIAL":
        compare_node = node_tree.nodes.new(type="ShaderNodeMath")  # Add a Function Compare node
        compare_node.operation = "COMPARE"  # Set the operation to COMPARE
        compare_node.location = (node.location.x-300 , node.location.y-200) # Position the node
        compare_node.inputs[1].default_value = j  # Set the value to compare
        j += 1
        compare_node.inputs[2].default_value = 0  # Set the threshold value

        print(f"Added a Function Compare node at location {compare_node.location}.")

        node_tree.links.new(curve_to_mesh_node.outputs[0], node.inputs[0])  # Link Curve to Mesh to Set Material
        node_tree.links.new(node.outputs[0], join_geometry_node.inputs[0])  # Link Set Material to Join Geometry
        node_tree.links.new(compare_node.outputs[0], node.inputs[1])  # Link Set Material to Function Compare
        node_tree.links.new(floored_modulo_node.outputs[0], compare_node.inputs[0])  # Link FLOORED MODULO to Function Compare

        print(f"Linked Set Material node {i} to Join Geometry node.")


named_attribute_node = node_tree.nodes.new(type="GeometryNodeInputNamedAttribute")
named_attribute_node.location = (600, 500)  # Position the node
named_attribute_node.data_type = "INT"
named_attribute_node.inputs[0].default_value = "duplicate_index"
print("Added a Geometry Input Named Attribute node.")

int_node = node_tree.nodes.new(type="FunctionNodeInputInt")
int_node.integer = j
int_node.location = (600, 300)  # Position the node
print("Added a int node.")

node_tree.links.new(named_attribute_node.outputs[0], floored_modulo_node.inputs[0])  # Link Geometry Input Named Attribute to FLOORED MODULO
node_tree.links.new(int_node.outputs[0], floored_modulo_node.inputs[1])  # Link int to FLOORED MODULO


# Link Join Geometry output to Group Output
group_output_node = None
for node in node_tree.nodes:
    if node.type == "GROUP_OUTPUT":
        group_output_node = node
        break

if group_output_node:
    node_tree.links.new(join_geometry_node.outputs[0], group_output_node.inputs[0])
    print("Linked Join Geometry node to Group Output.")
else:
    print("No Group Output node found in the Geometry Nodes tree.")



