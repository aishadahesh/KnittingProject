"""A coarse 3D model of one dark object from an exported splat dataset, on the CPU, in a minute.

    python scripts\\visual_hull.py --dataset robot_scans\\<session>\\splat_dataset [--output <file.ply>]

Where the splat trainer needs a GPU and fifteen minutes, this needs only the dataset's poses: it is
a check that the poses agree with the pictures, with a model as the by-product. Every picture's
dark pixels (the object against light paper) are its silhouette; a voxel survives only if every
picture that can see it puts it inside the silhouette - the visual hull. Surviving surface voxels
are coloured from the pictures that look at them face-on. Right poses carve a nut; wrong poses
carve nothing at all, because no voxel satisfies silhouettes that disagree.

The grid sits round the object: the calibrated target when the dataset has one, otherwise the
point the cameras' axes converge on. Writes a PLY (x y z red green blue) the app's splat viewer
and any point-cloud viewer open, and prints how many voxels survived and how big the object is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

DARK_MAX_GREY = 60
# Pose error tolerance: the silhouette is grown this much before carving, so a camera a few pixels
# off does not shave the object. Closing fills the shiny streaks inside a chrome part.
SILHOUETTE_GROW_PX = 9
SILHOUETTE_CLOSE_PX = 7
VOXEL_M = 0.00025
HALF_WIDTH_M = 0.016
BELOW_M, ABOVE_M = 0.010, 0.010
MIN_VIEWS = 8
SLAB_AREA_FRACTION = 0.8


def silhouette(path: Path, object_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The object's mask (grown for tolerance) and the picture's colours.

    An object-only export has already painted everything but the object black, so there the mask is
    simply what is not black."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        grey = np.asarray(rgb.convert("L"))
    if object_only:
        mask = np.asarray(rgb).max(axis=2) > 0
        grown = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(SILHOUETTE_GROW_PX * 2 + 1))
        return np.asarray(grown) > 0, np.asarray(rgb)
    dark = Image.fromarray(((grey < DARK_MAX_GREY) * 255).astype(np.uint8))
    closed = dark.filter(ImageFilter.MaxFilter(SILHOUETTE_CLOSE_PX)).filter(ImageFilter.MinFilter(SILHOUETTE_CLOSE_PX))
    # A chrome part mirrors the white paper, so its faces come out as light as the background and
    # only its edges are dark: fill whatever the dark outline encloses. Paper is what the frame's
    # edge reaches; anything else inside the outline is the object.
    filled = closed.copy()
    width, height = filled.size
    for x, y in [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1),
                 (width // 2, 0), (width // 2, height - 1), (0, height // 2), (width - 1, height // 2)]:
        if filled.getpixel((x, y)) == 0:
            ImageDraw.floodfill(filled, (x, y), 128)
    mask = np.asarray(filled) != 128
    grown = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(SILHOUETTE_GROW_PX * 2 + 1))
    return np.asarray(grown) > 0, np.asarray(rgb)


def load_dataset(dataset: Path):
    transforms = json.loads((dataset / "transforms.json").read_text(encoding="utf-8"))
    fx, fy, cx, cy = (float(transforms[key]) for key in ("fl_x", "fl_y", "cx", "cy"))
    width, height = int(transforms["w"]), int(transforms["h"])
    views = []
    for frame in transforms["frames"]:
        c2w = np.array(frame["transform_matrix"], dtype=float) @ np.diag([1.0, -1.0, -1.0, 1.0])  # back to OpenCV axes
        picture = dataset / frame["file_path"]
        if not picture.is_file():
            picture = Path(frame["file_path"])
        if not picture.is_file():
            continue
        mask, colours = silhouette(picture, bool(transforms.get("object_only")))
        if not mask.any():
            continue
        views.append({"w2c": np.linalg.inv(c2w), "centre": c2w[:3, 3], "axis": c2w[:3, 2], "mask": mask, "rgb": colours})
    centre = transforms.get("hand_eye_calibration", {}).get("target")
    if centre is None:
        # Where the optical axes come closest to meeting.
        normal = np.zeros((3, 3))
        rhs = np.zeros(3)
        for view in views:
            projector = np.eye(3) - np.outer(view["axis"], view["axis"])
            normal += projector
            rhs += projector @ view["centre"]
        centre = np.linalg.solve(normal, rhs)
    return views, (fx, fy, cx, cy, width, height), np.asarray(centre, dtype=float)


def carve(views, intrinsics, centre):
    fx, fy, cx, cy, width, height = intrinsics
    xs = np.arange(-HALF_WIDTH_M, HALF_WIDTH_M, VOXEL_M)
    zs = np.arange(-BELOW_M, ABOVE_M, VOXEL_M)
    grid = np.stack(np.meshgrid(xs, xs, zs, indexing="ij"), axis=-1).reshape(-1, 3) + centre
    alive = np.ones(len(grid), dtype=bool)
    seen = np.zeros(len(grid), dtype=np.int32)
    for view in views:
        w2c = view["w2c"]
        cam = grid[alive] @ w2c[:3, :3].T + w2c[:3, 3]
        depth = cam[:, 2]
        u = np.round(fx * cam[:, 0] / np.maximum(depth, 1e-6) + cx).astype(int)
        v = np.round(fy * cam[:, 1] / np.maximum(depth, 1e-6) + cy).astype(int)
        inside = (depth > 1e-3) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        keep = np.ones(inside.shape, dtype=bool)
        keep[inside] = view["mask"][v[inside], u[inside]]
        indices = np.flatnonzero(alive)
        seen[indices[inside]] += 1
        alive[indices[~keep]] = False
    alive &= seen >= MIN_VIEWS
    shape = (len(xs), len(xs), len(zs))
    alive = alive.reshape(shape)
    # Views from at most 20 degrees off vertical cannot see a top or a bottom face edge-on, so the
    # hull runs on above and below the object as a tapering spike. Between the real top and bottom
    # the hull has the object's full width; where it has narrowed clearly, it is spike, and cut.
    area = alive.sum(axis=(0, 1))
    full = np.flatnonzero(area >= SLAB_AREA_FRACTION * area.max())
    if len(full):
        alive[:, :, :full.min()] = False
        alive[:, :, full.max() + 1:] = False
    return grid, alive, shape


def surface_points(grid, alive, shape):
    """Voxels with an empty neighbour, and their outward normals."""
    padded = np.pad(alive, 1)
    neighbours = sum(np.roll(padded, shift, axis)[1:-1, 1:-1, 1:-1].astype(np.int8)
                     for axis in range(3) for shift in (-1, 1))
    surface = alive & (neighbours < 6)
    # The normal points from the full side to the empty side.
    normal = np.zeros(shape + (3,))
    for axis in range(3):
        normal[..., axis] = (np.roll(padded, -1, axis)[1:-1, 1:-1, 1:-1].astype(float)
                             - np.roll(padded, 1, axis)[1:-1, 1:-1, 1:-1].astype(float)) * -1.0
    points = grid.reshape(shape + (3,))[surface]
    normals = normal[surface]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(lengths > 0, normals / np.maximum(lengths, 1e-9), np.array([0.0, 0.0, 1.0]))
    return points, normals


def colour(points, normals, views, intrinsics):
    fx, fy, cx, cy, width, height = intrinsics
    samples = np.zeros((len(points), 3))
    weights = np.zeros(len(points))
    for view in views:
        toward = view["centre"] - points
        toward /= np.linalg.norm(toward, axis=1, keepdims=True)
        facing = np.einsum("ij,ij->i", toward, normals)
        w2c = view["w2c"]
        cam = points @ w2c[:3, :3].T + w2c[:3, 3]
        depth = np.maximum(cam[:, 2], 1e-6)
        u = np.round(fx * cam[:, 0] / depth + cx).astype(int)
        v = np.round(fy * cam[:, 1] / depth + cy).astype(int)
        ok = (facing > 0.5) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        samples[ok] += view["rgb"][v[ok], u[ok]] * facing[ok, None]
        weights[ok] += facing[ok]
    colours = np.where(weights[:, None] > 0, samples / np.maximum(weights[:, None], 1e-9), 90.0)
    return np.clip(colours, 0, 255).astype(np.uint8)


def write_ply(path: Path, points, colours) -> None:
    table = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    table["x"], table["y"], table["z"] = points[:, 0], points[:, 1], points[:, 2]
    table["red"], table["green"], table["blue"] = colours[:, 0], colours[:, 1], colours[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(table)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        table.tofile(handle)


def write_mesh(path: Path, alive: np.ndarray, grid: np.ndarray, shape, colours_at=None) -> int:
    """The hull as a mesh of its exposed voxel faces - OBJ (with per-vertex colour) or STL by the
    file's extension. Watertight, so it slices for printing; blocky at the voxel size (0.25 mm)."""
    origin = grid.reshape(shape + (3,))[0, 0, 0]
    padded = np.pad(alive, 1)
    faces = []   # (voxel index triple, axis, direction)
    for axis in range(3):
        for direction in (-1, 1):
            neighbour = np.roll(padded, -direction, axis)[1:-1, 1:-1, 1:-1]
            exposed = alive & ~neighbour
            for index in np.argwhere(exposed):
                faces.append((index, axis, direction))
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    face_corners = {(0, -1): [0, 4, 7, 3], (0, 1): [1, 2, 6, 5], (1, -1): [0, 1, 5, 4], (1, 1): [3, 7, 6, 2],
                    (2, -1): [0, 3, 2, 1], (2, 1): [4, 5, 6, 7]}
    quads = []
    for index, axis, direction in faces:
        base = origin + (np.asarray(index, dtype=float) - 0.5) * VOXEL_M
        quads.append(base + corners[face_corners[(axis, direction)]] * VOXEL_M)
    quads = np.array(quads) if quads else np.zeros((0, 4, 3))
    if path.suffix.lower() == ".stl":
        triangles = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]) if len(quads) else np.zeros((0, 3, 3))
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        record = np.zeros(len(triangles), dtype=[("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        record["normal"], record["v"] = normals * 1000.0, triangles * 1000.0    # STL in millimetres
        with path.open("wb") as handle:
            handle.write(b"visual hull".ljust(80, b"\0"))
            handle.write(np.uint32(len(triangles)).tobytes())
            record.tofile(handle)
        return len(triangles)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# visual hull, millimetres\n")
        for quad in quads:
            for vertex in quad:
                colour = ""
                if colours_at is not None:
                    colour = " " + " ".join(f"{c:.3f}" for c in colours_at(vertex))
                handle.write(f"v {vertex[0] * 1000:.3f} {vertex[1] * 1000:.3f} {vertex[2] * 1000:.3f}{colour}\n")
        for index in range(len(quads)):
            base = 4 * index + 1
            handle.write(f"f {base} {base + 1} {base + 2} {base + 3}\n")
    return len(quads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--mesh", default="", help="also write a mesh: a .obj (coloured) or .stl (millimetres) path")
    args = parser.parse_args()
    dataset = Path(args.dataset)
    output = Path(args.output) if args.output else dataset.parent / "splat_output" / "visual_hull.ply"
    views, intrinsics, centre = load_dataset(dataset)
    print(f"{len(views)} pictures with a dark object; grid centred on {np.round(centre * 1000, 1)} mm")
    grid, alive, shape = carve(views, intrinsics, centre)
    if not alive.any():
        raise SystemExit("no voxel survived every silhouette: the poses do not agree with the pictures")
    points, normals = surface_points(grid, alive, shape)
    colours = colour(points, normals, views, intrinsics)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_ply(output, points, colours)
    extent = (points.max(axis=0) - points.min(axis=0)) * 1000.0
    print(f"{int(alive.sum()):,} voxels survived, {len(points):,} on the surface; "
          f"object {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} mm")
    print(f"Written to {output}")
    if args.mesh:
        mesh = Path(args.mesh)
        by_voxel = {tuple(np.round((point - points.min(axis=0)) / VOXEL_M).astype(int)): colour / 255.0
                    for point, colour in zip(points, colours)}
        low = points.min(axis=0)

        def colour_at(vertex):
            key = tuple(np.round((vertex - low) / VOXEL_M).astype(int))
            for dx in (0, -1, 1):
                for dy in (0, -1, 1):
                    for dz in (0, -1, 1):
                        found = by_voxel.get((key[0] + dx, key[1] + dy, key[2] + dz))
                        if found is not None:
                            return found
            return np.array([0.4, 0.4, 0.4])

        faces = write_mesh(mesh, alive, grid, shape, colour_at if mesh.suffix.lower() == ".obj" else None)
        print(f"Mesh with {faces:,} faces written to {mesh}")


if __name__ == "__main__":
    main()
