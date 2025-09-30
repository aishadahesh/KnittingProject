import bpy

def create_yarn_material(color, mat_name="YarnMaterial"):
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in nodes:
        nodes.remove(n)

    # Nodes
    output = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping1 = nodes.new(type="ShaderNodeMapping")
    mapping2 = nodes.new(type="ShaderNodeMapping")
    wave1 = nodes.new(type="ShaderNodeTexWave")
    wave2 = nodes.new(type="ShaderNodeTexWave")
    color_ramp1 = nodes.new(type="ShaderNodeValToRGB")
    color_ramp2 = nodes.new(type="ShaderNodeValToRGB")
    mix1 = nodes.new(type="ShaderNodeMixRGB")
    multiply = nodes.new(type="ShaderNodeMixRGB")
    mix2 = nodes.new(type="ShaderNodeMixRGB")
    noise = nodes.new(type="ShaderNodeTexNoise")

    # Node locations
    tex_coord.location = (-1200, 0)
    mapping1.location = (-1000, 0)
    mapping2.location = (-1000, -200)
    wave1.location = (-800, 0)
    wave2.location = (-800, -200)
    color_ramp1.location = (-600, 200)
    color_ramp2.location = (-600, -200)
    mix1.location = (-400, 100)
    multiply.location = (-200, 100)
    mix2.location = (0, 100)
    bsdf.location = (200, 100)
    output.location = (400, 100)
    noise.location = (-800, -400)

    # Set blend types
    multiply.blend_type = 'MULTIPLY'
    mix1.blend_type = 'MIX'
    mix2.blend_type = 'MIX'

    # Set default values 
    wave1.inputs["Scale"].default_value = 5.0
    wave2.inputs["Scale"].default_value = 7.5
    wave2.bands_direction = 'Y'
    noise.inputs["Scale"].default_value = 5.0
    color_ramp1.color_ramp.elements[0].position = 0.218
    color_ramp2.color_ramp.elements[0].position = 0.295
    mix1.inputs["Fac"].default_value = 0.2
    multiply.inputs["Color1"].default_value = color
    mix2.inputs["Fac"].default_value = 0.2

    # Connections
    links.new(tex_coord.outputs["UV"], mapping1.inputs["Vector"])
    links.new(tex_coord.outputs["UV"], mapping2.inputs["Vector"])
    links.new(tex_coord.outputs["UV"], noise.inputs["Vector"])
    links.new(mapping1.outputs["Vector"], wave1.inputs["Vector"])
    links.new(mapping2.outputs["Vector"], wave2.inputs["Vector"])
    links.new(wave1.outputs["Fac"], color_ramp1.inputs["Fac"])
    links.new(wave2.outputs["Fac"], color_ramp2.inputs["Fac"])
    links.new(color_ramp1.outputs["Color"], mix1.inputs["Color1"])
    links.new(color_ramp2.outputs["Color"], mix1.inputs["Color2"])
    links.new(mix1.outputs["Color"], multiply.inputs["Fac"])
    links.new(multiply.outputs["Color"], mix2.inputs["Color1"])
    links.new(noise.outputs["Color"], mix2.inputs["Color2"])
    links.new(mix2.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    return mat

def set_colors(colors, connect_node_label = "CurveToMesh_", obj_name = "MergedLoops"):
    materials = []
    for i, rgba in enumerate(colors):
        mat_name = f"Material_{i}"
        if mat_name in bpy.data.materials:
            material = bpy.data.materials[mat_name]
        else:
            material = bpy.data.materials.new(name=mat_name)
            material.use_nodes = True
        material = create_yarn_material(rgba, mat_name)
        materials.append(material)

    obj = bpy.data.objects.get(obj_name)
    if obj.data.materials:
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

    connect_node = None
    for node in node_tree.nodes:
        if node.label == connect_node_label:
            connect_node = node
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

            node_tree.links.new(connect_node.outputs[0], node.inputs[0])  
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