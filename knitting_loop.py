import bpy
import numpy as np
import sys

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

dy = 0.55

# map = np.array([[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
#                 [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
#                 [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
#                 [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])

map = np.array([[1,1,1,1],
                [0,1,0,1],
                [0,0,1,1],
                [1,1,0,1]])

scale_factor = convert_bitmap_to_scales_factors(map)
scale_factor = np.where((scale_factor <= 1), scale_factor, 1 + dy * (scale_factor - 1))

n_loops = map.shape[1]
print(f"Number of loops: {n_loops}")
n_rows = map.shape[0]
print(f"Number of rows: {n_rows}")
loop_res = 32    # loop resolution
num_points = loop_res * n_loops

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
merged_obj = join_objects(created_objects)


########################  CONVERT OBJECT TO MESH ########################

group_name = "AutoMeshToCurve"
if group_name in bpy.data.node_groups:
    node_group = bpy.data.node_groups[group_name]
else:
    node_group = bpy.data.node_groups.new(name=group_name, type='GeometryNodeTree')

    node_group.interface.clear()
    node_group.nodes.clear()

    node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    nodes = node_group.nodes
    links = node_group.links

    input_node = nodes.new("NodeGroupInput")
    output_node = nodes.new("NodeGroupOutput")
    mesh_to_curve = nodes.new("GeometryNodeMeshToCurve")
    curve_to_mesh = nodes.new("GeometryNodeCurveToMesh")
    curve_circle = nodes.new("GeometryNodeCurvePrimitiveCircle")

    curve_to_mesh.label= "CurveToMesh_"

    input_node.location = (-600, 0)
    mesh_to_curve.location = (-300, 0)
    curve_circle.location = (-300, -200)
    curve_to_mesh.location = (0, 0)
    output_node.location = (900, 0)

    links.new(mesh_to_curve.inputs["Mesh"], input_node.outputs["Geometry"])
    links.new(curve_to_mesh.inputs["Curve"], mesh_to_curve.outputs["Curve"])
    links.new(curve_to_mesh.inputs["Profile Curve"], curve_circle.outputs["Curve"])
    links.new(output_node.inputs["Geometry"], curve_to_mesh.outputs["Mesh"])

    curve_circle.inputs["Radius"].default_value = 0.13  # controls thickness
    curve_circle.inputs["Resolution"].default_value = 16


obj_name = "MergedLoops"
obj = bpy.data.objects.get(obj_name)

if obj:
    for mod in obj.modifiers:
        if mod.type == 'NODES' and mod.node_group and mod.node_group.name == group_name:
            obj.modifiers.remove(mod)
    mod = obj.modifiers.new(name="AutoCurveMod", type='NODES')
    mod.node_group = node_group
else:
    print(f"Object '{obj_name}' not found.")



############################ COLORING ############################

# Define RGB colors with alpha
colors = [
    (0.0, 0.0, 1.0, 1.0),  # Blue
    (1.0, 0.0, 0.0, 1.0),  # Red
    (0.0, 1.0, 0.0, 1.0),  # Green
    (0.5, 0.5, 0.0, 0.0)    # Yellow
]

materials = []
for i, rgba in enumerate(colors):
    mat_name = f"Material_{i}"
    if mat_name in bpy.data.materials:
        material = bpy.data.materials[mat_name]
    else:
        material = bpy.data.materials.new(name=mat_name)
        material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
    materials.append(material)

obj = bpy.context.active_object 
obj.data.materials.clear()
for material in materials:
    obj.data.materials.append(material)

geo_nodes_mod = None
for mod in obj.modifiers:
    if mod.type == 'NODES':
        geo_nodes_mod = mod
        break

node_tree = geo_nodes_mod.node_group
for i, material in enumerate(materials):
        set_material_node = node_tree.nodes.new(type="GeometryNodeSetMaterial")
        set_material_node.location = (500, 100 * i)  
        set_material_node.inputs[2].default_value = material

curve_to_mesh_node = None
for node in node_tree.nodes:
    if node.label == "CurveToMesh_":
        curve_to_mesh_node = node
        break

join_geometry_node = node_tree.nodes.new(type="GeometryNodeJoinGeometry")
join_geometry_node.location = (680, 100)  

floored_modulo_node = node_tree.nodes.new(type="ShaderNodeMath")
floored_modulo_node.operation = "FLOORED_MODULO"
floored_modulo_node.location = (20, 200)  

j = 0
for i, node in enumerate(node_tree.nodes):
    if node.type == "SET_MATERIAL":
        compare_node = node_tree.nodes.new(type="ShaderNodeMath") 
        compare_node.operation = "COMPARE"  
        compare_node.location = (node.location.x-300 , node.location.y-300) 
        compare_node.inputs[1].default_value = j  
        j += 1
        compare_node.inputs[2].default_value = 0  

        node_tree.links.new(curve_to_mesh_node.outputs[0], node.inputs[0])  
        node_tree.links.new(node.outputs[0], join_geometry_node.inputs[0])  
        node_tree.links.new(compare_node.outputs[0], node.inputs[1]) 
        node_tree.links.new(floored_modulo_node.outputs[0], compare_node.inputs[0]) 

named_attribute_node = node_tree.nodes.new(type="GeometryNodeInputNamedAttribute")
named_attribute_node.location = (-200, 250)  
named_attribute_node.data_type = "INT"
named_attribute_node.inputs[0].default_value = "duplicate_index"

int_node = node_tree.nodes.new(type="FunctionNodeInputInt")
int_node.integer = j
int_node.location = (-250, 100)  

node_tree.links.new(named_attribute_node.outputs[0], floored_modulo_node.inputs[0])  
node_tree.links.new(int_node.outputs[0], floored_modulo_node.inputs[1]) 

group_output_node = None
for node in node_tree.nodes:
    if node.type == "GROUP_OUTPUT":
        group_output_node = node
        break

if group_output_node:
    node_tree.links.new(join_geometry_node.outputs[0], group_output_node.inputs[0])
