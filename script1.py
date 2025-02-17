import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"

import bpy
import math
import mathutils
import itertools
from PyQt6.QtWidgets import (
    QApplication, QColorDialog, QPushButton, QVBoxLayout, QWidget, QLabel, QHBoxLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PIL import Image


class ColorPickerApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Knitting Color Picker")
        self.setGeometry(100, 100, 500, 450)  
        
        self.selected_colors = [(1.0, 1.0, 1.0, 1.0)] * 3  # Default colors (White)
        self.render_path = os.path.join(os.path.dirname(bpy.data.filepath), "results", "live_result.png")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title Label
        title_label = QLabel("Select 3 Colors for Knitting")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label)

        # Color selection buttons
        self.color_labels = []
        self.color_buttons = []

        for i in range(3):
            color_layout = QHBoxLayout()

            color_label = QLabel(f"Color {i+1}: #FFFFFF")  # Default to white
            color_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            color_label.setStyleSheet("background-color: white; padding: 5px; border: 1px solid black;")

            self.color_labels.append(color_label)

            color_button = QPushButton("Pick Color")
            color_button.clicked.connect(lambda checked, index=i: self.pick_color(index))
            self.color_buttons.append(color_button)

            color_layout.addWidget(color_label)
            color_layout.addWidget(color_button)
            layout.addLayout(color_layout)

        # Render button
        self.render_button = QPushButton("Render")
        self.render_button.clicked.connect(self.update_render)
        self.render_button.setStyleSheet("font-size: 12px; font-weight: bold; background-color: #4CAF50; color: white; padding: 5px; border-radius: 5px;")
        layout.addWidget(self.render_button)

        # Image Preview
        self.image_label = QLabel("Rendered Image Will Appear Here")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray; padding: 5px;")
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        # Timer to check for image updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_for_render)

    def pick_color(self, index):
        color = QColorDialog.getColor()
        if color.isValid():
            rgba = (color.redF(), color.greenF(), color.blueF(), 1.0)
            self.selected_colors[index] = rgba
            hex_color = f"#{color.red():02X}{color.green():02X}{color.blue():02X}"
            self.color_labels[index].setText(f"Color {index+1}: {hex_color}")
            self.color_labels[index].setStyleSheet(f"background-color: {hex_color}; padding: 5px; border: 1px solid black;")

    def update_render(self):
        """Triggers the rendering process and starts checking for the image."""
        print("Updating render with colors:", self.selected_colors)
        forward(self.selected_colors)

        # Start the timer to check if the render is done
        self.timer.start(500)  # Check every 0.5 seconds

    def check_for_render(self):
        """Checks if the rendered image is ready, then displays it."""
        if os.path.exists(self.render_path):
            print("Render complete. Displaying image...")
            self.display_rendered_image()
            self.timer.stop()  # Stop checking once the image is found

    def display_rendered_image(self):
        """Loads and displays the rendered image inside the PyQt window."""
        if not os.path.exists(self.render_path):
            print("Error: Rendered image not found!")
            self.image_label.setText("Error: Rendered image not found!")
            return

        # Load and set the image
        pixmap = QPixmap(self.render_path)
        self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))



def forward(colors):
    """Assigns colors to materials and updates Geometry Nodes."""
    knitting_obj = bpy.context.active_object
    if not knitting_obj:
        print("No active object found. Please select the knitting object.")
        return

    materials = []
    for i, color in enumerate(colors):
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
            bsdf.inputs["Base Color"].default_value = color
        materials.append(material)

    # Clear old materials before adding new ones
    knitting_obj.data.materials.clear()

    # Assign updated materials
    for material in materials:
        knitting_obj.data.materials.append(material)

    # Update the Geometry Nodes modifier
    set_geometry_node_materials(knitting_obj, materials)

    update_materials(knitting_obj, materials, colors)

    render_model(knitting_obj)


# function to update the materials for more colors collaborations  
def update_materials(knitting_obj, materials, colors):
    # Generate all unique permutations of colors
    unique_combinations = list(itertools.permutations(colors, len(materials)))

    # Iterate through each unique combination and update materials
    for combination_index, combination in enumerate(unique_combinations):
        for i, material in enumerate(materials):
            material.use_nodes = True
            bsdf = material.node_tree.nodes["Principled BSDF"]
            bsdf.inputs["Base Color"].default_value = combination[i]
            print(f"Updated material {material.name} with color {combination[i]}.")

        # Render the scene for this color combination
        # print(f"Rendering for combination {combination_index + 1}/{len(unique_combinations)}: {combination}")
        # render_from_angles(knitting_obj)


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


collab = 0

# render multiple images from different angles
def render_from_angles(obj):
    global collab

    collab += 1
    index = 0
    # Ensure the output folder exists in the project directory
    script_dir = os.path.dirname(bpy.data.filepath)  # Get the directory of the current Blender file
    output_folder = os.path.join(script_dir, "renders")  # Create a "renders" folder in the project directory

    # Create the folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if collab == 1:
        # delete the content of renders folder
        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))

    # Get the camera in the scene
    camera = bpy.data.objects.get("Camera")
    if not camera:
        print("No camera found in the scene. Please add a camera.")
        return

    # Save the original camera location and rotation
    original_location = camera.location.copy()
    original_rotation = camera.rotation_euler.copy()

    # Define camera locations and rotations
    camera_locations = [(-3, -1, 8), (-3.5, -2, 9.5)]
    camera_rotations = [
        (math.radians(18), math.radians(-5), math.radians(270)),
        (math.radians(18), math.radians(6), math.radians(230))
    ]

    # Render and save the image
    for i, location in enumerate(camera_locations):
        for j, rotation in enumerate(camera_rotations):
            camera.location = location
            camera.rotation_euler = mathutils.Euler(rotation, 'XYZ')
            
            # Set the Blender render filepath
            bpy.context.scene.render.filepath = os.path.join(output_folder, f"collab{collab}_render_{index+1}.png")
            index += 1
            bpy.ops.render.render(write_still=True)
            print(f"Rendered image saved to {bpy.context.scene.render.filepath}")

    # Restore the original camera location and rotation
    camera.location = original_location
    camera.rotation_euler = original_rotation


def duplicate_image_grid():
    global result
    script_dir = os.path.dirname(bpy.data.filepath)  # Get the directory of the current Blender file
    output_folder = os.path.join(script_dir, "results")  # Output folder for images

    # Define the original image path (latest render)
    original_image_path = os.path.join(output_folder, f"result{result}_render.png")
    
    # Ensure the original image exists before proceeding
    if not os.path.exists(original_image_path):
        print(f"Error: Image {original_image_path} not found!")
        return

    # Open the original image
    original_image = Image.open(original_image_path)

    # Get dimensions of the original image
    width, height = original_image.size

    # Create a new blank image (3x3 grid)
    grid_image = Image.new('RGB', (width * 3, height * 3))

    # Paste the original image 9 times in a 3x3 grid
    for row in range(3):
        for col in range(3):
            grid_image.paste(original_image, (col * width, row * height))
            # Rotate the image by 180 degrees if row is even
            # img_to_paste = original_image.rotate(180) if row % 2 == 0 else original_image
            # grid_image.paste(img_to_paste, (col * width, row * height))

    # Save the new grid image
    grid_image_path = os.path.join(output_folder, f"result{result}_grid.png")
    grid_image.save(grid_image_path)
    print(f"Grid image saved to {grid_image_path}")


result = 0
def render_model(obj):
    global result
    # Ensure the output folder exists in the project directory
    script_dir = os.path.dirname(bpy.data.filepath)  # Get the directory of the current Blender file
    output_folder = os.path.join(script_dir, "results")  # Create a "results" folder in the project directory

    # Create the folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    bpy.context.scene.render.filepath = os.path.join(output_folder, f"result{result+1}_render.png")
    result += 1
    bpy.ops.render.render(write_still=True)
    print(f"Rendered image saved to {bpy.context.scene.render.filepath}")
    duplicate_image_grid()

    bpy.context.scene.render.filepath = os.path.join(output_folder, "live_result.png")
    bpy.ops.render.render(write_still=True)
    print(f"Rendered image saved to {bpy.context.scene.render.filepath}")



def main():
    """Launches the PyQt UI for color selection."""
    app = QApplication([])
    window = ColorPickerApp()
    window.show()
    app.exec()

main()
