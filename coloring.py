import bpy

def set_colors(colors):
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


def update_materials(obj, color_combo):
    if not obj:
        print("No object passed to update_materials.")
        return

    obj.data.materials.clear()
    for i, color in enumerate(color_combo):
        mat_name = f"Material_{i}"
        material = bpy.data.materials.get(mat_name) or bpy.data.materials.new(name=mat_name)
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = color
        obj.data.materials.append(material)