import numpy as np
from knitting_core import compute_knitting_faces

def save_combined_obj(mesh_data_list, base_filename="knitting_model"):
    path = f"{base_filename}_combined.obj"
    off = 0
    with open(path, "w") as h:
        h.write("# Knitting Model\n")
        for i, (v, _, f, _) in enumerate(mesh_data_list):
            h.write(f"o mesh_{i}\n")
            np.savetxt(h, v, fmt="v %.6f %.6f %.6f")
            np.savetxt(h, f + off + 1, fmt="f %d %d %d %d")
            off += len(v)


def save_per_loop_objs(mesh_data_list, base_filename, loop_res, segments):
    """Save each stitch loop as a separate OBJ file."""
    loop_vertex_count = (loop_res + 1) * segments
    loop_faces = compute_knitting_faces(segments, [(np.empty((loop_vertex_count, 3)), loop_res + 1)])[0]
    loop_specs = [
        (
            row_idx,
            loop_idx,
            verts[loop_idx * loop_res * segments:loop_idx * loop_res * segments + loop_vertex_count],
            f"{base_filename}_r{row_idx:02d}_l{loop_idx:02d}.obj",
        )
        for row_idx, (verts, _, _, n_points) in enumerate(mesh_data_list)
        for loop_idx in range((n_points - 1) // loop_res)
    ]

    obj_info = []
    for row_idx, loop_idx, loop_verts, path in loop_specs:
        with open(path, "w") as handle:
            np.savetxt(handle, loop_verts, fmt="v %.6f %.6f %.6f")
            np.savetxt(handle, loop_faces + 1, fmt="f %d %d %d %d")
        obj_info.append((row_idx, loop_idx, path))
    return obj_info
