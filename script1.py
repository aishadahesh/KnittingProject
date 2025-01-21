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

    # Add a new Set Material node
    # Assumes there are only one material
    set_material_node = node_tree.nodes.new(type="GeometryNodeSetMaterial")
    set_material_node.location = (5800, 199)  # Position the node
    print(f"Added a Set Material node at location {set_material_node.location}.")

    # Assign the first material from the list (assuming a single color case)
    if len(materials) > 0:
        set_material_node.inputs[2].default_value = materials[0]  
        print(f"Assigned material {materials[0].name} to the Set Material node.")
    else:
        print("No materials found to assign to the Set Material node.")
        return

    # Find the Curve to Mesh node
    curve_to_mesh_node = None
    for node in node_tree.nodes:
        if node.label == "CurveToMesh_MaterialsPoint":
            curve_to_mesh_node = node
            break

    if not curve_to_mesh_node:
        print("No Curve to Mesh node found in the Geometry Nodes tree.")
        return

    # Link the Curve to Mesh output to the Set Material input
    node_tree.links.new(curve_to_mesh_node.outputs[0], set_material_node.inputs[0])
    print("Linked the Curve to Mesh node to the Set Material node.")

    group_output_node = None

    # Find the Group Output node
    for node in node_tree.nodes:
        if node.type == "GROUP_OUTPUT":
            group_output_node = node
            break

    if not group_output_node:
        print("No group output node found in the Geometry Nodes tree.")
        return

    # Link the Set Material output to the Group Output node input
    node_tree.links.new(set_material_node.outputs[0], group_output_node.inputs[0])
    print("Linked the Set Material node to the Group Output node.")


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
