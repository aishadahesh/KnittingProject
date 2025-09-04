import bpy

def convert_obj_to_mesh():
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

def add_geo(obj_name = "MergedLoops"):
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

        input_node.location = (-600, 0)
        input_node.label = "input_"
        output_node.location = (900, 0)

    obj = bpy.data.objects.get(obj_name)

    if obj:
        for mod in obj.modifiers:
            if mod.type == 'NODES' and mod.node_group and mod.node_group.name == group_name:
                obj.modifiers.remove(mod)
        mod = obj.modifiers.new(name="AutoCurveMod", type='NODES')
        mod.node_group = node_group
    else:
        print(f"Object '{obj_name}' not found.")   