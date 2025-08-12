import bpy
import numpy as np
from PyQt6.QtWidgets import QApplication
import sys
sys.path.append(r"C:\Users\Aisha\KnittingProject")     
import pick_colors_gui
import render_images_gui
import obj_to_mesh
import coloring
import rendering
import knitting_loop


def main():
    app = QApplication(sys.argv)
    color_window = pick_colors_gui.ColorPickerApp(on_see_result=None)
    render_window = render_images_gui.RenderImagesApp(None, None, other_window=color_window)
    color_window.move(100, 100)
    render_window.move(1100, 100)

    color_window.show()
    render_window.show()

    # Define callback after windows created
    def on_colors_updated(bitmap, colors):
        bitmap = bitmap[::-1]
        colors = colors[::-1]

        knitting_loop.main(bitmap)
        obj_to_mesh.add_geo()
        coloring.set_colors(colors, "input_")

        obj = bpy.context.active_object
        if obj:
            cam = bpy.data.objects.get("Camera")
            if cam:
                cam.location = (8, -4, 4)

            rendering.render_model(obj)
            render_window.obj = obj
            render_window.render_callback = lambda o: rendering.render_more_combinations(o, colors)
            render_window.refresh_render()
        else:
            print("No object to render")

    # Assign the real callback to the color picker window
    color_window.on_see_result = on_colors_updated

    app.exec()


main()
