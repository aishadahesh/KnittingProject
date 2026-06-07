# %% IMPORTS
import os
import numpy as np
from PIL import Image
import mitsuba as mi
import jax.numpy as jnp
from functools import partial

# Import core reconstruction logic
from knitting_core import (
    CONFIG,
    INITIAL_PARAMS,
    PARAM_DELTAS,
    PARAM_NAMES,
    PARAM_RANGES,
    REFERENCE_IMAGE_PATH,
    compute_knitting_vertices,
    compute_knitting_faces,
    save_combined_obj,
    KnittingOptimizer,
    run_optimization_loop,
    get_loop_color
)

# Conditional imports for UI components
try:
    from vedo import Plotter, Mesh, Text2D, Sphere, Plane, Spline
except ImportError:
    print("vedo not installed. Standalone Interactive App will be unavailable.")

# %% LEGACY INTERACTIVE UI (VEDO)

class InteractiveModelEditor:
    """Legacy interactive editor using the Vedo library."""
    def __init__(self, optimizer, initial_params):
        self.optimizer = optimizer
        self.params = np.array(initial_params)
        self.bitmap = optimizer.bitmap
        self.plotter = Plotter(title="Knitting Model Editor (Legacy)", size=(1200, 900))
        self.mesh_actors = []
        self._update_display()
        
        self.plotter.add_callback('key_press', self._on_key)
        print("\n" + "="*50)
        print(" LEGACY EDITOR CONTROLS:")
        print(" [O] - Trigger Optimization Loop")
        print(" [R] - Perform High-Quality Render")
        print(" [Arrow Keys] - Adjust Selected Parameter")
        print(" [0-9] - Select Parameter to Adjust")
        print(" [F] - Finish and Close")
        print("="*50 + "\n")

    def _update_display(self):
        verts_list = compute_knitting_vertices(self.params, self.bitmap)
        faces_list = compute_knitting_faces(CONFIG['geometry']['segments'], verts_list)
        
        for actor in self.mesh_actors: self.plotter.remove(actor)
        self.mesh_actors = []
        
        for i, (verts, _) in enumerate(verts_list):
            m = Mesh([np.array(verts), faces_list[i]])
            m.color(get_loop_color(i, 0))
            self.mesh_actors.append(m)
        self.plotter.add(self.mesh_actors)

    def _on_key(self, event):
        k = event.keypress
        
        if k == 'o':
            print("Starting optimization from legacy UI...")
            self.params, _ = run_optimization_loop(self.optimizer, self.params)
            self._update_display()
        elif k == 'f':
            self.plotter.close()
        elif k in [str(i) for i in range(len(PARAM_NAMES))]:
            self.selected_idx = int(k)
            print(f"Selected: {PARAM_NAMES[self.selected_idx]}")
        elif k == 'Up':
            self.params[self.selected_idx] = np.clip(
                self.params[self.selected_idx] + PARAM_DELTAS[self.selected_idx],
                PARAM_RANGES[self.selected_idx][0], PARAM_RANGES[self.selected_idx][1])
            self._update_display()
        elif k == 'Down':
            self.params[self.selected_idx] = np.clip(
                self.params[self.selected_idx] - PARAM_DELTAS[self.selected_idx],
                PARAM_RANGES[self.selected_idx][0], PARAM_RANGES[self.selected_idx][1])
            self._update_display()

# %% STANDALONE EXECUTION

if __name__ == "__main__":
    print("=" * 80)
    print("KNITTING RECONSTRUCTION - STANDALONE MODE")
    print("=" * 80)
    
    ref_img = Image.open(REFERENCE_IMAGE_PATH).convert("RGB")
    bitmap = jnp.ones((CONFIG['geometry']['bitmap_rows'], CONFIG['geometry']['bitmap_loops']))
    optimizer = KnittingOptimizer(ref_img, bitmap)
    
    try:
        editor = InteractiveModelEditor(optimizer, INITIAL_PARAMS)
        editor.plotter.show(interactive=True)
    except NameError:
        print("\n[ERROR] Vedo is not installed. Standalone mode requires 'vedo'.")
        print("Please run the modern UI instead:")
        print("    python trame_app.py\n")
