import itertools
import os
import bpy
import coloring

def render_model(obj=None, render_path=None):
    script_dir = os.path.dirname(__file__)
    output_folder = os.path.join(script_dir, "images")
    os.makedirs(output_folder, exist_ok=True)

    if render_path is None:
        render_path = os.path.join(output_folder, "result.png")

    bpy.context.scene.render.filepath = render_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered image saved to {render_path}")


def render_more_combinations(obj, colors):
    script_dir = os.path.dirname(__file__)
    output_folder = os.path.join(script_dir, "images")
    os.makedirs(output_folder, exist_ok=True)
    
    for file in os.listdir(output_folder):
                os.remove(os.path.join(output_folder, file))

    permutations = list(itertools.permutations(colors, len(colors)))
    selected_combos = permutations[1:4]  # Pick 3 combos for demo

    for i, combo in enumerate(selected_combos):
        coloring.update_materials(obj, combo)
        render_path = os.path.join(output_folder, f"combo_{i+1}.png")
        render_model(obj, render_path)
