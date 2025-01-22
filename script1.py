import bpy
import os  

def forward(colors):
    # Get the active object (knitting object)
    knitting_obj = bpy.context.active_object
    if not knitting_obj:
        print("No active object found. Please select the knitting object.")
        return

    # Add materials for each color
    materials = []
    for color in colors:
        name = f"Material_{color}"
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        bsdf = material.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = color  # Set color (RGBA)
        materials.append(material)

    # Assign materials to the knitting object
    for material in materials:
        if material.name not in [mat.name for mat in knitting_obj.data.materials]:
            knitting_obj.data.materials.append(material)
        print(f"Added material to {knitting_obj.name}: {material.name}")

    # Update the Geometry Nodes modifier
    set_geometry_node_materials(knitting_obj, materials)

    # Render images from different angles
    render_from_angles(knitting_obj)

# Function to set the materials in the Geometry Nodes tree
def set_geometry_node_materials(knitting_obj, materials):
    
    # Ensure the object has a Geometry Nodes modifier
    geo_nodes_mod = None
    for mod in knitting_obj.modifiers:
        if mod.type == 'NODES':
            geo_nodes_mod = mod
            break

    if not geo_nodes_mod:
        print("No Geometry Nodes modifier found on the object.")
        return

    # Access the Geometry Nodes tree
    node_tree = geo_nodes_mod.node_group
    if not node_tree:
        print("No node group found in the Geometry Nodes modifier.")
        return

    # Add a new Set Material node for each color
    for i, material in enumerate(materials):
        set_material_node = node_tree.nodes.new(type="GeometryNodeSetMaterial")
        set_material_node.location = (5800, 199 + 200 * i)  # Position the node
        print(f"Added a Set Material node at location {set_material_node.location}.")


        # Assign the material to the Set Material node
        set_material_node.inputs[2].default_value = material
        print(f"Assigned material {material.name} to the Set Material node.")

    # Find the Curve to Mesh node
    curve_to_mesh_node = None
    for node in node_tree.nodes:
        if node.label == "CurveToMesh_MaterialsPoint":
            curve_to_mesh_node = node
            break

    if not curve_to_mesh_node:
        print("No Curve to Mesh node found in the Geometry Nodes tree.")
        return

    # Add a Join Geometry node to combine all outputs
    join_geometry_node = node_tree.nodes.new(type="GeometryNodeJoinGeometry")
    join_geometry_node.location = (6000, 0)  # Position the node
    print("Added a Join Geometry node.")

    floored_modulo_node = node_tree.nodes.new(type="ShaderNodeMath")
    floored_modulo_node.operation = "FLOORED_MODULO"
    floored_modulo_node.location = (5600, 0)  # Position the node
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
    named_attribute_node.location = (5600, 50)  # Position the node
    named_attribute_node.data_type = "INT"
    named_attribute_node.inputs[0].default_value = "duplicate_index"
    print("Added a Geometry Input Named Attribute node.")

    int_node = node_tree.nodes.new(type="FunctionNodeInputInt")
    int_node.integer = j
    floored_modulo_node.location = (5600, 50)  # Position the node
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


    # print("Debug: Node types in the Geometry Nodes tree:")
    # for node in node_tree.nodes:
    #     print(f"Node name: {node.name}, Node type: {node.type}")



def render_from_angles(obj):
    # Ensure the output folder exists in the project directory
    script_dir = os.path.dirname(bpy.data.filepath)  # Get the directory of the current Blender file
    output_folder = os.path.join(script_dir, "renders")  # Create a "renders" folder in the project directory
    
    # Create the folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set the Blender render filepath
    bpy.context.scene.render.filepath = os.path.join(output_folder, "render_1.png")
    
    # Get the camera in the scene
    camera = bpy.data.objects.get("Camera")
    if not camera:
        print("No camera found in the scene. Please add a camera.")
        return

    # Render and save the image
    bpy.ops.render.render(write_still=True)
    print(f"Rendered image saved to {bpy.context.scene.render.filepath}")


def main():
    colors = []
    print("Enter the colors you want to add to the knitting simulation.")
    print("Enter colors as tuples (R, G, B, A) with values between 0 and 1.")
    print("Enter 'done' when you are finished.")

    while True:
        color = input("Enter a color (e.g., (1, 0, 0, 1) for red): ")
        if color.lower() == "done":
            break
        try:
            parsed_color = tuple(map(float, color.strip("()").split(",")))
            if len(parsed_color) == 4 and all(0 <= c <= 1 for c in parsed_color):
                colors.append(parsed_color)
            else:
                print("Invalid color format. Please enter (R, G, B, A) with values between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a tuple like (1, 0, 0, 1).")

    forward(colors)

main()
