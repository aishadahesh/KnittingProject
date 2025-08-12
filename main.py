import bpy
import numpy as np
import sys
sys.path.append(r"C:\Users\Aisha\KnittingProject")     
import pick_colors_gui
import render_images_gui
import obj_to_mesh
import coloring
import rendering
import knitting_loop

pick_colors_gui.run_color_app()
map = np.load("bitmap.npy")
map=map[::-1]
colors = np.load("colors.npy")
colors = colors[::-1]
knitting_loop.main(map)

# # convert object to mesh
# obj_to_mesh.convert_obj_to_mesh()

# # add colors to mesh
# coloring.set_colors(colors)

obj_to_mesh.add_geo()
coloring.set_colors(colors, "input_")

# render images
obj = bpy.context.active_object
if obj:
    camera = bpy.data.objects.get("Camera")
    if not camera:
        print("No camera found in the scene. Please add a camera.")
    camera.location = (8, -4, 4)

    rendering.render_model(obj)
    render_images_gui.run_rendering_app(obj, lambda o: rendering.render_more_combinations(o, colors))
else:
    print("No active object to render.")