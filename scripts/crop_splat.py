"""Crop a trained splat to the cloth: drop the background, the huge soft blobs and the floaters.

    uv run python scripts\\crop_splat.py robot_scans\\<session>\\<splat_output> robot_scans\\<session>\\<splat_dataset>

A splat reproduces what many pictures saw; what few saw - the table past the cloth's edge, the
background of the steepest views - it fills with large, soft, half-transparent Gaussians that look
like fog and blobs from anywhere but the pictures' own viewpoints. This keeps a Gaussian only if
  * at least --min-views of the training cameras saw it (inside their frame, in front of them),
  * it lies within a slab round the cloth's plane (fitted to the well-seen, opaque Gaussians),
  * it is not a huge smear (its largest axis under --max-size of the cameras' distance), and
  * it is not nearly transparent.
Writes export\\splat_clean.ply next to export\\splat.ply, every property kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def read_ply(path: Path):
    kinds = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4"}
    with path.open("rb") as handle:
        header, properties, count = [], [], 0
        while True:
            line = handle.readline().decode("ascii").strip()
            header.append(line)
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property"):
                _, kind, name = line.split()[:3]
                properties.append((name, kinds[kind]))
            elif line == "end_header":
                break
        return header, np.fromfile(handle, dtype=np.dtype(properties), count=count)


def write_ply(path: Path, header, table) -> None:
    lines = [f"element vertex {len(table)}" if line.startswith("element vertex") else line for line in header]
    with path.open("wb") as handle:
        handle.write(("\n".join(lines) + "\n").encode("ascii"))
        table.tofile(handle)


def cameras(run: Path, dataset: Path):
    """The training cameras (OpenCV axes) and intrinsics, in the dataset's frame and in nerfstudio's
    normalised one - whichever the export uses."""
    transforms = json.loads((dataset / "transforms.json").read_text())
    parser = json.loads(next(run.rglob("dataparser_transforms.json")).read_text())
    to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    raw = np.array([np.array(f["transform_matrix"]) @ to_cv for f in transforms["frames"]])
    move = np.eye(4)
    move[:3, :] = np.array(parser["transform"])
    # nerfstudio folds a COLMAP export's applied_transform (an axis swap already in transforms.json)
    # into the transform it saves, so that is undone first.
    if "applied_transform" in transforms:
        applied = np.eye(4)
        applied[:3, :] = np.array(transforms["applied_transform"])
        move = move @ np.linalg.inv(applied)
    normalised = np.einsum("ij,njk->nik", move, raw)
    normalised[:, :3, 3] *= float(parser["scale"])
    lens = {k: float(transforms[k]) for k in ("fl_x", "fl_y", "cx", "cy", "w", "h")}
    return {"dataset": raw, "normalised": normalised}, lens


def views_of(points, c2ws, lens, chunk=200_000):
    """How many cameras see each point inside their frame."""
    counts = np.zeros(len(points), np.int32)
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk]
        for c2w in c2ws:
            cam = (p - c2w[:3, 3]) @ c2w[:3, :3]
            z = cam[:, 2]
            ok = z > 1e-6
            u = lens["fl_x"] * cam[:, 0] / np.where(ok, z, 1) + lens["cx"]
            v = lens["fl_y"] * cam[:, 1] / np.where(ok, z, 1) + lens["cy"]
            counts[start:start + chunk] += ok & (u >= 0) & (u < lens["w"]) & (v >= 0) & (v < lens["h"])
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("dataset", type=Path, help="the dataset it was trained on (holds transforms.json)")
    parser.add_argument("--min-views", type=int, default=15)
    parser.add_argument("--behind", type=float, default=0.10, help="how far behind the cloth plane is kept, as a share of the cameras' distance")
    parser.add_argument("--front", type=float, default=0.15, help="how far in front of it (toward the cameras) is kept")
    parser.add_argument("--max-size", type=float, default=0.02, help="largest Gaussian axis kept, as a share of the cameras' distance")
    parser.add_argument("--min-opacity", type=float, default=0.05)
    parser.add_argument("--speck-radius", type=float, default=4.0,
                        help="a Gaussian with under 12 kept neighbours within this many typical spacings is a speck. "
                             "A fine-detail splat packs the cloth tighter: use 8 there")
    parser.add_argument("--keep-radius", type=float, default=0.0,
                        help="keep only what lies within this distance of the patch's middle, along the cloth, as a "
                             "share of the cameras' distance (e.g. 0.4) - 0 keeps everything")
    parser.add_argument("--layer", default="",
                        help="LOW,HIGH: keep this height band round the plane of the well-seen Gaussians, as shares of "
                             "the cameras' distance (+ toward the cameras) - e.g. -0.5,-0.12 for a cloth lying below the board")
    parser.add_argument("--drop-board", action="store_true",
                        help="also drop the ChArUco board beside the cloth (found by its black squares against the lace's blue holes)")
    parser.add_argument("--one-piece", action="store_true",
                        help="keep only the cloth's own connected piece (as --drop-board does last) without dropping the "
                             "board's squares - for a board masked out in training, beside a dark band the square finder would take")
    parser.add_argument("--drop-blue", action="store_true",
                        help="also drop the deep-blue surface the cloth lies on, leaving the cloth alone")
    parser.add_argument("--checkpoint", action="store_true",
                        help="also write the cropped splat as a new training run nerfstudio's viewer opens (run with .venv-splat's python)")
    args = parser.parse_args()

    header, table = read_ply(args.run / "export" / "splat.ply")
    points = np.stack([table["x"], table["y"], table["z"]], axis=1).astype(np.float64)
    frames, lens = cameras(args.run, args.dataset)
    sample = points[np.random.default_rng(0).choice(len(points), min(20000, len(points)), replace=False)]
    # nerfstudio exports the checkpoint's means as they are - its normalised frame (checked against
    # the checkpoint in viewer_copy). Guessing the frame from which cameras see the Gaussians tied on
    # a dome (28 views either way) and picked the wrong one.
    frame = "normalised"
    del sample
    c2ws = frames[frame]
    log_scales = np.stack([table["scale_0"], table["scale_1"], table["scale_2"]], axis=1)
    colour = np.stack([table["f_dc_0"], table["f_dc_1"], table["f_dc_2"]], axis=1)
    keep = crop_mask(points, table["opacity"], log_scales, c2ws, lens, args, f"export, {frame} frame", colour)
    target = args.run / "export" / "splat_clean.ply"
    write_ply(target, header, table[keep])
    print(f"kept {int(keep.sum())} ({keep.mean() * 100:.0f}%) -> {target}")
    if args.checkpoint:
        viewer_copy(args, frames, lens, points, frame)


SH_C0 = 0.28209479177387814
SPECK_MIN_NEIGHBOURS = 12


def deep_blue(colour_dc) -> np.ndarray:
    """Gaussians coloured like the deep blue surface the cloth lies on: blue well above red and green,
    neither near-black (the cloth's dark stripes) nor pale (its light-blue band)."""
    rgb = np.clip(0.5 + SH_C0 * np.asarray(colour_dc, dtype=np.float64).reshape(-1, 3), 0.0, 1.0)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return (b > 0.25) & (b - np.maximum(r, g) > 0.18) & (np.maximum(r, g) < 0.45)


def board_side(points, centre, normal, colour_dc, trusted) -> np.ndarray:
    """What lies on the board's side of the cloth's edge. The board is in the cloth's plane, so neither
    depth nor distance separates them; colour shows where the edge runs - the board's black squares on
    one side, the lace's blue holes (the blue surface showing through) on the other. The line between
    the two groups across the plane is fitted (Fisher's direction), and everything past the cloth's
    nearest few percent towards the board goes: the board, and the white paper strip at its edge."""
    rgb = np.clip(0.5 + SH_C0 * np.asarray(colour_dc, dtype=np.float64).reshape(-1, 3), 0.0, 1.0)
    black = trusted & (rgb.max(axis=1) < 0.22) & (np.ptp(rgb, axis=1) < 0.08)
    blue = trusted & deep_blue(colour_dc)
    if black.sum() < 50 or blue.sum() < 50:
        print(f"  (board not separated: {int(black.sum())} black and {int(blue.sum())} blue Gaussians)")
        return np.zeros(len(points), bool)
    across = np.linalg.svd(np.eye(3) - np.outer(normal, normal))[0][:, :2]      # two axes in the plane
    flat = (points - centre) @ across
    mean_board, mean_cloth = flat[black].mean(axis=0), flat[blue].mean(axis=0)
    spread = np.cov(flat[black].T) + np.cov(flat[blue].T)
    direction = np.linalg.solve(spread + 1e-12 * np.eye(2), mean_cloth - mean_board)
    direction /= np.linalg.norm(direction)
    along = flat @ direction
    edge = np.percentile(along[blue], 3)                # the cloth's edge, from its side
    print(f"  board/cloth line fitted to {int(black.sum())} black and {int(blue.sum())} blue Gaussians")
    return along < edge


def board_squares(points, centre, normal, reach, colour_dc, keep) -> np.ndarray:
    """The board as what is left of it after the other rules: its black squares packed close together,
    where the lace has only a few dark specks scattered over it. Over a grid on the cloth's plane, the
    cells where black makes up a good share of what is kept, joined into patches; a patch as big as a
    few squares is the board, and it goes with a margin round it (the squares' white rims)."""
    from scipy import ndimage

    rgb = np.clip(0.5 + SH_C0 * np.asarray(colour_dc, dtype=np.float64).reshape(-1, 3), 0.0, 1.0)
    black = keep & (rgb.max(axis=1) < 0.22) & (np.ptp(rgb, axis=1) < 0.08)
    across = np.linalg.svd(np.eye(3) - np.outer(normal, normal))[0][:, :2]
    flat = (points - centre) @ across
    cell = BOARD_CELL * reach
    low = flat[keep].min(axis=0)
    ij = np.floor((flat - low) / cell).astype(int)
    shape = tuple(np.maximum(ij[keep].max(axis=0) + 1, 1))
    inside = keep & (ij >= 0).all(axis=1) & (ij < shape).all(axis=1)
    count = np.zeros(shape)
    dark = np.zeros(shape)
    np.add.at(count, tuple(ij[inside].T), 1)
    np.add.at(dark, tuple(ij[black & inside].T), 1)
    share = ndimage.uniform_filter(dark, 3) / np.maximum(ndimage.uniform_filter(count, 3), 1e-9)
    label, n = ndimage.label(share > BOARD_BLACK_SHARE)
    if not n:
        return np.zeros(len(points), bool)
    sizes = ndimage.sum(np.ones(shape), label, range(1, n + 1))
    board = np.isin(label, 1 + np.flatnonzero(sizes >= BOARD_MIN_CELLS))
    board = ndimage.binary_closing(board, iterations=2)
    board = ndimage.binary_dilation(board, iterations=2)
    print(f"  board squares: {int(board.sum())} grid cells of {cell:.4f} units")
    hit = (ij >= 0).all(axis=1) & (ij < shape).all(axis=1)
    out = np.zeros(len(points), bool)
    out[hit] = board[tuple(ij[hit].T)]
    return out


def largest_piece(points, centre, normal, reach, keep) -> np.ndarray:
    """The biggest connected piece of what is kept, over a grid on the cloth's plane: the cloth. With the
    board's black squares gone, its white squares and the paper strip beside it lie apart from it."""
    from scipy import ndimage

    across = np.linalg.svd(np.eye(3) - np.outer(normal, normal))[0][:, :2]
    flat = (points - centre) @ across
    cell = BOARD_CELL * reach
    low = flat[keep].min(axis=0)
    ij = np.floor((flat - low) / cell).astype(int)
    shape = tuple(np.maximum(ij[keep].max(axis=0) + 1, 1))
    filled = np.zeros(shape, bool)
    filled[tuple(ij[keep].T)] = True
    # Thinned first, so a thin strip touching the cloth's edge does not join the board to it.
    solid = ndimage.binary_closing(filled, iterations=1) | filled
    core = ndimage.binary_erosion(solid, iterations=PIECE_THIN)
    label, n = ndimage.label(core)
    if n <= 1:
        return keep.copy()
    sizes = ndimage.sum(core, label, range(1, n + 1))
    piece = ndimage.binary_dilation(label == 1 + int(np.argmax(sizes)), iterations=PIECE_THIN + 1) & solid
    hit = (ij >= 0).all(axis=1) & (ij < shape).all(axis=1)
    out = np.zeros(len(points), bool)
    out[hit] = piece[tuple(ij[hit].T)]
    return out & keep


# The board's patch finder: a grid cell this share of the cameras' distance, black making up this
# share of what is kept round it, a patch this many cells at least (a few of the board's squares).
BOARD_CELL = 0.015
BOARD_BLACK_SHARE = 0.25
BOARD_MIN_CELLS = 40
PIECE_THIN = 3                  # grid cells a bridge between the cloth and the board may be thick


def crop_mask(points, opacity_logit, log_scales, c2ws, lens, args, label, colour_dc=None) -> np.ndarray:
    """Which Gaussians to keep, by the four rules in the module docstring."""
    points = np.asarray(points, dtype=np.float64)
    seen = views_of(points, c2ws, lens)
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(opacity_logit, dtype=np.float64).reshape(-1)))
    size = np.exp(np.asarray(log_scales, dtype=np.float64)).max(axis=1)
    # The cloth's plane: a dome's cameras look, on average, straight at it, so its normal is their mean
    # viewing direction, and it lies at the median depth of the well-seen, opaque Gaussians along that.
    # (A free plane fit locked onto background clumps, 40 deg off the cloth.)
    normal = -np.mean(c2ws[:, :3, 2], axis=0)
    normal /= np.linalg.norm(normal)
    solid = (seen >= args.min_views) & (opacity > 0.5)
    centre = np.median(points[solid], axis=0)
    centre = centre + normal * (np.median((points[solid] - centre) @ normal))
    reach = float(np.median(np.abs((c2ws[:, :3, 3] - centre) @ normal)))
    depth = (points - centre) @ normal / reach          # + toward the cameras, - behind the cloth
    # --layer LOW,HIGH replaces the slab: the board can stand higher than the cloth (on 29 Oct the lace
    # lay ~3 cm below the board's face), and the plane through the well-seen Gaussians is the board's.
    low, high = (float(v) for v in args.layer.split(",")) if args.layer else (-args.behind, args.front)
    rules = {"seen by too few cameras": seen < args.min_views,
             "off the cloth's layer": (depth < low) | (depth > high),
             "huge smear": size > args.max_size * reach,
             "nearly transparent": opacity < args.min_opacity}
    if args.keep_radius:
        # The board lies in the cloth's plane, so the slab keeps it; a patch scan wants only the patch:
        # what lies farther than this from the middle of what was looked at, along the plane, goes.
        offset = points - centre
        sideways = np.linalg.norm(offset - np.outer(offset @ normal, normal), axis=1) / reach
        rules["outside the patch"] = sideways > args.keep_radius
    if args.drop_blue and colour_dc is not None:
        rules["the deep-blue surface under the cloth"] = deep_blue(colour_dc)
    if args.drop_board and colour_dc is not None:
        rules["the board beside the cloth"] = board_side(points, centre, normal, colour_dc, (seen >= args.min_views) & (opacity > 0.5))
    keep = np.isfinite(points).all(axis=1)
    print(f"{len(points)} Gaussians ({label}); cameras {reach:.3f} units from the cloth plane")
    for reason, drop in rules.items():
        print(f"  {reason}: {int((drop & keep).sum())} dropped")
        keep &= ~drop
    if args.drop_board and colour_dc is not None:
        # The fitted line above can miss (no blue left to fit it to, once the blue surface has gone):
        # what is left of the board is found by its squares.
        drop = board_squares(points, centre, normal, reach, colour_dc, keep)
        print(f"  the board's squares: {int((drop & keep).sum())} dropped")
        keep &= ~drop
    # Last, specks: what is left of a removed surface is scattered, the cloth is dense. A Gaussian with
    # few kept neighbours within a few typical spacings is a speck.
    from scipy.spatial import cKDTree

    kept = np.nonzero(keep)[0]
    tree = cKDTree(points[kept])
    spacing = float(np.median(tree.query(points[kept], k=2)[0][:, 1]))
    neighbours = np.array([len(n) for n in tree.query_ball_point(points[kept], args.speck_radius * spacing)]) - 1
    specks = kept[neighbours < SPECK_MIN_NEIGHBOURS]
    print(f"  isolated speck: {len(specks)} dropped")
    keep[specks] = False
    if (args.drop_board and colour_dc is not None) or args.one_piece:
        # Last, once the specks no longer bridge the gap between them: only the cloth's own piece stays.
        drop = keep & ~largest_piece(points, centre, normal, reach, keep)
        print(f"  apart from the cloth (the board's white squares, the paper strip): {int(drop.sum())} dropped")
        keep &= ~drop
    return keep


def viewer_copy(args, frames, lens, ply_points, frame) -> None:
    """The training run copied as a new run whose checkpoint holds only the kept Gaussians, so
    nerfstudio's viewer (which opens checkpoints, not PLYs) shows the cropped splat."""
    import shutil

    import torch

    config = next(args.run.rglob("config.yml"))
    checkpoint = sorted((config.parent / "nerfstudio_models").glob("step-*.ckpt"))[-1]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    params = {k: v for k, v in state["pipeline"].items() if "gauss_params" in k}
    # The export writes the checkpoint's means as they are; checked, not assumed.
    chosen = params["_model.gauss_params.means"].numpy().astype(np.float64)
    from scipy.spatial import cKDTree

    finite = np.isfinite(ply_points).all(axis=1)
    probe = chosen[np.isfinite(chosen).all(axis=1)][:5000]
    gap, _ = cKDTree(ply_points[finite]).query(probe)
    if np.median(gap) > 1e-4 * float(np.ptp(ply_points[finite], axis=0).max()):
        raise SystemExit(f"the checkpoint's Gaussians do not line up with the export (median gap {np.median(gap):.4g}); not writing a viewer copy")
    keep = crop_mask(chosen, params["_model.gauss_params.opacities"].numpy(), params["_model.gauss_params.scales"].numpy(),
                     frames[frame], lens, args, "checkpoint", params["_model.gauss_params.features_dc"].numpy())
    mask = torch.from_numpy(keep)
    pruned = {"step": state["step"], "pipeline": {k: (v[mask] if "gauss_params" in k else v) for k, v in state["pipeline"].items()}}
    text = config.read_text(encoding="utf-8")
    old = next(line for line in text.splitlines() if line.startswith("timestamp:"))
    stamp = old.split(":", 1)[1].strip().strip("'\"") + "_cloth"
    target = config.parent.parent / stamp
    if target.exists():
        shutil.rmtree(target)
    (target / "nerfstudio_models").mkdir(parents=True)
    (target / "config.yml").write_text(text.replace(old, f"timestamp: {stamp}"), encoding="utf-8")
    for extra in config.parent.glob("dataparser_transforms.json"):
        shutil.copy2(extra, target / extra.name)
    torch.save(pruned, target / "nerfstudio_models" / checkpoint.name)
    print(f"viewer copy with {int(keep.sum())} Gaussians: {target}")


if __name__ == "__main__":
    main()
