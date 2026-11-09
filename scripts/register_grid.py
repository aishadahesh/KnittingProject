"""Camera poses for a straight-down grid scan, from the arm's steps and two numbers fitted to the pictures.

    uv run python scripts\\register_grid.py ur5_20261018_152616

A grid scan keeps one tool attitude for every picture (ur5_20261018_152616: within 0.01 deg) and one
height, so whatever the camera mount is - however wrong the calibration, whatever the pendant's
tool-centre setting, however the camera flexes with attitude - it is the same for every picture, and
the camera moves exactly as the arm does (to a few hundredths of a millimetre). Between two pictures
the cloth then slides across the image by the arm's step, times a scale (pixels per millimetre),
turned by one angle and possibly mirrored. Those are the only unknowns, and every neighbouring pair
votes on them together: a stitch-sized wrong shift can fool one pair, never a hundred at once.

Each pair's phase-correlation surface (every shift, how well it fits) is computed once; the scale,
turn and mirroring are then searched against all of them, and the winner's margin over the best
distinct rival is reported. The dataset written has every camera at the arm's position, all with the
one fitted attitude, at the height the scale implies for the lens; the seed is the cloth plane,
coloured from the pictures. With only straight-down views a splat learns the cloth's look from above,
not its relief: that needs tilted views.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import gaussian_splatting  # noqa: E402
import gui_captures  # noqa: E402
from persistence import write_json

HALF = 4                     # pictures correlated at a quarter size (enough: every pair votes)
MAX_PAIR_STEP_MM = 22.0     # direct neighbours: a 37 mm two-step does not overlap a 25 mm frame
SCALES = np.geomspace(2.0, 40.0, 260)        # px per mm on the cloth, full size
TURNS = np.radians(np.arange(0.0, 360.0, 1.0))


def load(paths):
    from PIL import Image

    out = []
    for path in paths:
        with Image.open(path) as image:
            grey = np.asarray(image.convert("L").resize((image.width // HALF, image.height // HALF), Image.BILINEAR), dtype=np.float32)
        grey = grey - grey.mean()
        out.append(grey / (grey.std() + 1e-6))
    return out


MIN_OVERLAP = 0.15


def surfaces(images, pairs):
    """Normalised cross-correlation of each pair over every shift (zero-padded, so up to a whole
    frame), over the part the two pictures share: how well picture j matches picture i moved by
    (dx, dy), indexed [pair, dy + H, dx + W]. Not windowed phase correlation: a window fades the
    picture edges, which is exactly where a 30 % overlap is. Shifts sharing under MIN_OVERLAP score 0."""
    h, w = images[0].shape
    size = (2 * h, 2 * w)
    ones = np.ones((h, w), np.float32)

    def spectrum(a):
        return np.fft.rfft2(a, s=size)

    def correlate(fa, fb):
        return np.fft.fftshift(np.fft.irfft2(fb * np.conj(fa), s=size))

    values, squares = [spectrum(i) for i in images], [spectrum(i * i) for i in images]
    mask = spectrum(ones)
    overlap = correlate(mask, mask)
    out = np.zeros((len(pairs), *size), np.float32)
    usable = overlap > MIN_OVERLAP * h * w
    for k, (i, j) in enumerate(pairs):
        numerator = correlate(values[i], values[j])
        energy = correlate(squares[i], mask) * correlate(mask, squares[j])
        out[k][usable] = (numerator[usable] / np.sqrt(np.maximum(energy[usable], 1e-6)))
    return out


def score(maps, steps_mm, scale, turn, mirror):
    """Summed correlation at the shifts this (scale, turn, mirror) predicts, for every pair."""
    c, s = math.cos(turn), math.sin(turn)
    a = np.array([[c, -s], [s, c]]) * (np.array([[-1.0], [1.0]]) if mirror else 1.0)
    shifts = steps_mm @ a.T * (scale / HALF)
    _, hh, ww = maps.shape
    x, y = shifts[:, 0] + ww / 2, shifts[:, 1] + hh / 2
    inside = (x >= 0) & (x < ww - 1) & (y >= 0) & (y < hh - 1)
    k = np.nonzero(inside)[0]
    x0, y0 = np.floor(x[k]).astype(int), np.floor(y[k]).astype(int)
    fx, fy = x[k] - x0, y[k] - y0
    value = (maps[k, y0, x0] * (1 - fx) * (1 - fy) + maps[k, y0, x0 + 1] * fx * (1 - fy)
             + maps[k, y0 + 1, x0] * (1 - fx) * fy + maps[k, y0 + 1, x0 + 1] * fx * fy)
    return float(value.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session")
    parser.add_argument("--output", default="splat_dataset_grid")
    parser.add_argument("--focal", type=float, default=1015.0, help="lens focal length, px at 640 wide (today's COLMAP solve)")
    parser.add_argument("--k1", type=float, default=0.055)
    args = parser.parse_args()

    folder = ROOT / "robot_scans" / args.session
    records = [r for r in gui_captures.load_session(folder) if r.get("actual_tcp_pose") and Path(str(r.get("path", ""))).is_file()]
    poses = np.array([r["actual_tcp_pose"][:6] for r in records], dtype=float)
    positions_mm = poses[:, :2] * 1000.0
    pairs = [(i, j) for i in range(len(records)) for j in range(i + 1, len(records))
             if 1.0 < np.linalg.norm(positions_mm[j] - positions_mm[i]) < MAX_PAIR_STEP_MM]
    steps = np.array([positions_mm[j] - positions_mm[i] for i, j in pairs])
    print(f"{args.session}: {len(records)} pictures, {len(pairs)} pairs within {MAX_PAIR_STEP_MM:.0f} mm")
    started = time.monotonic()
    images = load([r["path"] for r in records])
    maps = surfaces(images, pairs)
    print(f"correlation surfaces in {time.monotonic() - started:.0f} s; searching scale x turn x mirror")

    results = []
    for mirror in (False, True):
        for scale in SCALES:
            for turn in TURNS:
                results.append((score(maps, steps, scale, turn, mirror), scale, turn, mirror))
    results.sort(key=lambda r: r[0], reverse=True)
    best = results[0]
    # The best rival that is a genuinely different answer (another scale or turn, not a neighbour of the best).
    rival = next((r for r in results[1:] if r[3] != best[3] or abs(math.log(r[1] / best[1])) > 0.05
                  or abs((r[2] - best[2] + math.pi) % (2 * math.pi) - math.pi) > math.radians(4)), None)
    # Refine around the best.
    fine = [(score(maps, steps, s, t, best[3]), s, t, best[3])
            for s in best[1] * np.linspace(0.97, 1.03, 61) for t in best[2] + np.radians(np.linspace(-1.5, 1.5, 61))]
    total, scale, turn, mirror = max(fine)
    margin = total / rival[0] if rival else float("inf")
    print(f"best: {scale:.3f} px/mm, image turned {math.degrees(turn) % 360:.2f} deg, mirrored={mirror}; "
          f"score {total:.1f} vs best distinct rival {rival[0]:.1f} ({rival[1]:.2f} px/mm, {math.degrees(rival[2]) % 360:.0f} deg, "
          f"mirrored={rival[3]}) - margin x{margin:.2f}")

    # How many pairs individually agree: the surface peaks within 2 px of the predicted shift.
    c, s = math.cos(turn), math.sin(turn)
    a = np.array([[c, -s], [s, c]]) * (np.array([[-1.0], [1.0]]) if mirror else 1.0)
    predicted = steps @ a.T * (scale / HALF)
    _, hh, ww = maps.shape
    agree = 0
    for k, shift in enumerate(predicted):
        x, y = int(round(shift[0] + ww / 2)), int(round(shift[1] + hh / 2))
        if 2 <= x < ww - 2 and 2 <= y < hh - 2:
            peak = np.unravel_index(np.argmax(maps[k]), maps[k].shape)
            agree += abs(peak[1] - x) <= 2 and abs(peak[0] - y) <= 2
    print(f"pairs whose own strongest shift is the predicted one: {agree} of {len(pairs)}")

    # Cameras: content slides by scale * A * step when the camera moves by step, so the camera's image
    # axes in the plane are -A (rows); a mirrored picture is flipped so that the camera is right-handed.
    width, height = 640, 480
    axes = -a
    if mirror:
        axes = np.diag([-1.0, 1.0]) @ axes
    x_axis, y_axis = np.r_[axes[0], 0.0], np.r_[axes[1], 0.0]
    z_axis = np.cross(x_axis, y_axis)
    if z_axis[2] > 0:
        raise SystemExit("the fitted turn makes a camera looking up - the fit is wrong")
    camera_height = args.focal / scale / 1000.0      # metres over the cloth
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    output = folder / args.output
    if (output / "images").exists():
        shutil.rmtree(output / "images")
    (output / "images").mkdir(parents=True)
    (output / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    gains = gaussian_splatting.exposure_gains([Path(r["path"]) for r in records])
    origin = positions_mm.mean(axis=0) / 1000.0
    frames, colmap_images, centres = [], [], []
    for index, record in enumerate(records):
        source = Path(record["path"])
        gaussian_splatting.equalised_image(source, gains.get(str(source), np.ones(3)), mirror).save(output / "images" / source.name)
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = rotation, np.r_[poses[index, :2] - origin, camera_height]
        centres.append(c2w)
        frames.append({"file_path": f"images/{source.name}", "transform_matrix": (c2w @ gaussian_splatting._CV_TO_GL).tolist(),
                       "colmap_im_id": index + 1, "row": int(record.get("row", 0)), "col": int(record.get("col", 0))})
        w2c = np.linalg.inv(c2w)
        from scipy.spatial.transform import Rotation

        q = Rotation.from_matrix(w2c[:3, :3]).as_quat()
        colmap_images.append(f"{index + 1} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {w2c[0, 3]:.9f} {w2c[1, 3]:.9f} {w2c[2, 3]:.9f} 1 {source.name}")

    # Seed: the cloth plane (z = 0) over the scanned area, each point the median colour the pictures show.
    from PIL import Image

    colour_images = [np.asarray(Image.open(output / "images" / Path(r["path"]).name).convert("RGB")) for r in records]
    xs = np.arange((positions_mm[:, 0] - origin[0] * 1000).min() - 25, (positions_mm[:, 0] - origin[0] * 1000).max() + 25, 0.5) / 1000
    ys = np.arange((positions_mm[:, 1] - origin[1] * 1000).min() - 25, (positions_mm[:, 1] - origin[1] * 1000).max() + 25, 0.5) / 1000
    grid = np.array([[x, y, 0.0] for y in ys for x in xs])
    sums, counts = np.zeros((len(grid), 3)), np.zeros(len(grid))
    for c2w, image in zip(centres, colour_images):
        cam = (grid - c2w[:3, 3]) @ c2w[:3, :3]
        u = args.focal * cam[:, 0] / cam[:, 2] + width / 2
        v = args.focal * cam[:, 1] / cam[:, 2] + height / 2
        inside = (cam[:, 2] > 0) & (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
        sums[inside] += image[v[inside].astype(int), u[inside].astype(int)]
        counts[inside] += 1
    seen = counts > 0
    points, colours = grid[seen], (sums[seen] / counts[seen, None]).astype(np.uint8)
    if len(points) > gaussian_splatting.SEED_MAX_POINTS:
        keep = np.random.default_rng(0).choice(len(points), gaussian_splatting.SEED_MAX_POINTS, replace=False)
        points, colours = points[keep], colours[keep]
    table = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    table["x"], table["y"], table["z"] = points.T
    table["red"], table["green"], table["blue"] = colours.T
    with (output / gaussian_splatting.SEED_PLY_NAME).open("wb") as handle:
        handle.write(("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(table)}\n"
                      "property float x\nproperty float y\nproperty float z\n"
                      "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii"))
        table.tofile(handle)

    intrinsics = {"fl_x": args.focal, "fl_y": args.focal, "cx": width / 2, "cy": height / 2, "w": width, "h": height,
                  "k1": args.k1, "k2": 0.0, "p1": 0.0, "p2": 0.0, "camera_model": "OPENCV",
                  "camera_angle_x": 2 * math.atan(width / (2 * args.focal))}
    transforms = {**intrinsics, "session_id": args.session, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "pose_source": (f"arm steps with one camera attitude and scale fitted to {len(pairs)} picture pairs "
                                  f"({scale:.3f} px/mm, turn {math.degrees(turn) % 360:.2f} deg, mirrored={mirror}, margin x{margin:.2f})"),
                  "grid_registration": {"px_per_mm": scale, "turn_deg": math.degrees(turn) % 360, "mirrored": mirror,
                                        "pairs": len(pairs), "pairs_agreeing": agree, "margin": margin, "camera_height_m": camera_height},
                  **({"mirrored": True} if mirror else {}), "ply_file_path": gaussian_splatting.SEED_PLY_NAME,
                  "exposure_equalised": bool(gains), "frames": frames}
    write_json(output / 'transforms.json', transforms, indent=2)
    gaussian_splatting._write_colmap_model(output / "sparse" / "0", colmap_images, intrinsics)
    print(f"camera {camera_height * 100:.1f} cm over the cloth (for a {args.focal:.0f} px lens); "
          f"{len(frames)} pictures and {len(points)} seed points written to {output}")


if __name__ == "__main__":
    main()
