"""The fabric's colours, read from a model's photos: which yarns it is made of, how much of each, and
the colour of each - for a cropped splat and the pictures it was trained on.

    .venv\\Scripts\\python.exe scripts\\fabric_colours.py <splat_output> <dataset> [--colours N]

1. Every Gaussian of export/splat_clean.ply gets its colour from the PHOTOS (photo_colours): from each
   training camera (up to PHOTO_VIEWS) the Gaussian in front at every pixel is found (splat_gl's id
   render: solid cores, depth-tested), and the photo's pixel goes to it. A Gaussian's colour is the
   median over the views that saw it of its mean there (linear light) - robust to a view where it was
   misplaced, glared or shaded. Unlit, clipped and board-masked pixels give nothing. The splat's own
   colours are not used: they are the trainer's approximation (a dark band is drawn as near-black fog).
2. The Gaussians are grouped into yarns (group_yarns) by the colour of their 3D neighbourhood - a
   yarn's lit and shaded sides stay together - with weighted k-means, then groups merged while closer
   than MERGE_DELTA_E (or, with --colours N, until N are left), and groups of one hue that differ only
   in lightness merged as one yarn in more and less light. The deep-blue table seen through the
   openings is labelled as such, and so is a group that is only a blend of a yarn and the table (the
   openings' rims, mark_edges).
3. Each yarn's colour is its lit side: the mean of its Gaussians' photo colours between the 50th and
   90th brightness percentile, unclipped; its shade the darkest quarter. The board's paper, in the same
   photos, gives a corrected colour too (neutral, at paper's reflectance).

Writes <splat_output>/fabric_colours.json (the groups), fabric_colours_labels.npy (each Gaussian's
group, in splat_clean.ply's order), fabric_photo_colours.npy (each Gaussian's photo colour, NaN where
unseen) and fabric_colours.png (a swatch strip).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import gaussian_splatting  # noqa: E402
from fabric_palette import hex_of, linear_to_srgb, srgb_to_linear  # noqa: E402
from persistence import write_json

MAX_COLOURS = 8               # groups the k-means starts from, before merging
MERGE_DELTA_E = 12.0          # neighbourhood colours (CIELAB) closer than this are one yarn
NEIGHBOURS = 24               # how many nearest seen Gaussians make a Gaussian's neighbourhood
SAME_YARN_AB, SAME_YARN_L = 7.0, 35.0   # lit colours this close in a*, b* and L* are one yarn in more or less light
CLIPPED = 0.90                # a channel this bright (sRGB 0-1) is at the camera's top: colour lost
SATURATED = 0.985             # a photo pixel with a channel this bright says nothing reliable
UNLIT_LIGHTNESS = 4.0         # a photo pixel this dark (CIELAB L*) is where the light did not reach
PHOTO_VIEWS = 40              # training photos the colours are read from (spread over the dome)
MIN_PIXELS = 2                # pixels a Gaussian needs in a view for that view to count
MIN_VIEWS = 3                 # views a Gaussian needs to have a photo colour
TABLE_CHROMA, TABLE_HUE = 25.0, (235.0, 315.0)   # strongly blue: the table under the openings
EDGE_DELTA_E, EDGE_MIX, EDGE_MAIN_SHARE = 8.0, (0.03, 0.95), 0.15   # a blend of a yarn and the table (mark_edges)
MIN_SHARE = 0.02              # a group covering less of the cloth than this is left out of the report
MIN_OPACITY = 0.3
WHITE_REFLECTANCE = 0.85      # the board's white paper: how much light matte printer paper sends back
BLACK_REFLECTANCE = 0.04      # and its printed black (laser toner on matte paper)


# ----------------------------------------------------------------------------------------- colour maths

_RGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb):
    xyz = srgb_to_linear(rgb) @ _RGB_TO_XYZ.T / _WHITE_D65
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def describe(lab) -> str:
    """A plain name for a colour: light/dark, and the hue when it has one."""
    lightness, a, b = (float(v) for v in lab)
    chroma = float(np.hypot(a, b))
    tone = "very light" if lightness > 85 else "light" if lightness > 65 else "mid" if lightness > 40 else "dark" if lightness > 20 else "very dark"
    if chroma < 6:
        return ("white" if lightness >= 85 else "light grey" if lightness > 65 else "mid grey" if lightness > 40
                else "dark grey" if lightness > 25 else "charcoal" if lightness > 12 else "black")
    hue = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0
    names = ((20, "pink"), (60, "red-orange"), (100, "yellow"), (160, "green"), (220, "cyan"), (290, "blue"), (340, "purple"), (360, "pink"))
    name = next(n for limit, n in names if hue < limit)
    return f"{tone} {name}" + (" (greyish)" if chroma < 18 else "")


# ----------------------------------------------------------------------------------------- photo colours

def deep_blue(rgb) -> np.ndarray:
    """The deep-blue table the cloth lies on (crop_splat's rule): blue well above red and green, not pale."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return (b > 0.25) & (b - np.maximum(r, g) > 0.18) & (np.maximum(r, g) < 0.45)


def photo_colours(splat, cameras, lens, photos, masks):
    """Each Gaussian's colour as the photos show it (module docstring, 1): (n, 3) sRGB, NaN where fewer
    than MIN_VIEWS views saw it, and how many views did."""
    import moderngl
    from PIL import Image

    import splat_gl

    renderer = splat_gl.SplatRenderer(moderngl.create_standalone_context())
    frame = gaussian_splatting.orbit_frame(splat["positions"])
    size = (int(lens["w"]), int(lens["h"]))
    count = len(splat["positions"])
    chosen = np.linspace(0, len(cameras) - 1, min(PHOTO_VIEWS, len(cameras))).astype(int)
    per_view = np.full((len(chosen), count, 3), np.nan, dtype=np.float32)
    for row, index in enumerate(chosen):
        c2w = cameras[index]
        view = (c2w[:3, 3], c2w[:3, 2], c2w[:3, 0], -c2w[:3, 1], lens["fl_x"])
        ids = renderer.gaussian_ids("cloth", splat, view, size, frame).reshape(-1)
        photo = np.asarray(Image.open(photos[index]).convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255.0
        good = (ids >= 0) & (photo.max(axis=1) < SATURATED) & (srgb_to_lab(photo)[:, 0] >= UNLIT_LIGHTNESS)
        if masks[index] is not None and Path(masks[index]).is_file():
            good &= np.asarray(Image.open(masks[index]).convert("L")).reshape(-1) > 127
        lin = srgb_to_linear(photo[good])
        owner = ids[good]
        hits = np.bincount(owner, minlength=count)
        sums = np.stack([np.bincount(owner, weights=lin[:, c], minlength=count) for c in range(3)], axis=1)
        seen = hits >= MIN_PIXELS
        per_view[row, seen] = (sums[seen] / hits[seen, None]).astype(np.float32)
    views = np.sum(~np.isnan(per_view[..., 0]), axis=0)
    colour = np.full((count, 3), np.nan)
    enough = views >= MIN_VIEWS
    colour[enough] = linear_to_srgb(np.nanmedian(per_view[:, enough], axis=0))
    return colour, views


def weighted_kmeans(points, weight, k, iterations=60, seed=0):
    rng = np.random.default_rng(seed)
    # k-means++ seeding, weighted
    centres = [points[rng.choice(len(points), p=weight / weight.sum())]]
    for _ in range(1, k):
        d2 = np.min([((points - c) ** 2).sum(1) for c in centres], axis=0) * weight
        centres.append(points[rng.choice(len(points), p=d2 / d2.sum())])
    centres = np.array(centres)
    for _ in range(iterations):
        label = np.argmin(((points[:, None, :] - centres[None]) ** 2).sum(2), axis=1)
        new = np.array([np.average(points[label == j], axis=0, weights=weight[label == j]) if (label == j).any() else centres[j]
                        for j in range(k)])
        if np.allclose(new, centres, atol=1e-4):
            break
        centres = new
    return centres, label


def lit_side(rgb):
    """A set of colours' lit side - the mean (linear light) of the unclipped ones between the 50th and
    90th brightness percentile - and its shade, the mean of the darkest quarter."""
    lin = srgb_to_linear(rgb)
    luma = lin @ np.array([0.2126, 0.7152, 0.0722])
    honest = rgb.max(axis=1) < CLIPPED
    if honest.sum() < 10:
        honest = np.ones(len(rgb), dtype=bool)
    lo, hi = np.percentile(luma[honest], [50, 90])
    lit = honest & (luma >= lo) & (luma <= hi)
    shade = luma <= np.percentile(luma, 25)
    return linear_to_srgb(lin[lit].mean(axis=0)), linear_to_srgb(lin[shade].mean(axis=0))


def is_table(lab) -> bool:
    chroma, hue = float(np.hypot(lab[1], lab[2])), (np.degrees(np.arctan2(lab[2], lab[1])) + 360.0) % 360.0
    return chroma > TABLE_CHROMA and TABLE_HUE[0] <= hue <= TABLE_HUE[1]


def group_yarns(positions, photo, views, weight, wanted: int = 0):
    """Every Gaussian's group (module docstring, 2): labels (n,), each Gaussian the group of its nearest
    seen one when it was not seen itself."""
    from scipy.spatial import cKDTree

    seen = np.flatnonzero(~np.isnan(photo[:, 0]))
    lab = srgb_to_lab(photo[seen])
    tree = cKDTree(positions[seen])
    _, near = tree.query(positions[seen], k=min(NEIGHBOURS, len(seen)))
    trust = views[seen][near].astype(np.float64)
    context = (lab[near] * trust[..., None]).sum(1) / trust.sum(1)[:, None]
    w = weight[seen]
    _, label = weighted_kmeans(context, w, min(MAX_COLOURS, len(seen)))
    groups = [np.flatnonzero(label == j) for j in np.unique(label)]

    def lit_lab(g):
        return srgb_to_lab(lit_side(photo[seen[g]])[0])

    def merge(i, j):
        groups[i] = np.concatenate([groups[i], groups[j]])
        del groups[j]

    while len(groups) > 1:
        means = [np.average(context[g], axis=0, weights=w[g]) for g in groups]
        distance, i, j = min((np.linalg.norm(means[i] - means[j]), i, j) for i in range(len(groups)) for j in range(i + 1, len(groups)))
        if distance >= MERGE_DELTA_E and not (wanted and len(groups) > wanted):
            break
        merge(i, j)
    # One yarn in more and less light: one hue, lightness apart by less than a white and a grey are.
    while not wanted or len(groups) > wanted:
        labs = [lit_lab(g) for g in groups]
        pairs = [(i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))
                 if not is_table(labs[i]) and not is_table(labs[j])
                 and np.linalg.norm(labs[i][1:] - labs[j][1:]) < SAME_YARN_AB and abs(labs[i][0] - labs[j][0]) < SAME_YARN_L]
        if not pairs:
            break
        merge(*pairs[0])
    labels = np.full(len(positions), -1)
    for k, g in enumerate(groups):
        labels[seen[g]] = k
    unseen = np.flatnonzero(labels < 0)
    if len(unseen):
        _, nearest = tree.query(positions[unseen])
        labels[unseen] = labels[seen[nearest]]
    return labels


def mark_edges(groups) -> None:
    """A group whose colour is a blend of a yarn and the table - the rims of the openings, where both
    fall in one pixel - is no yarn of its own: its role becomes "edge" (recoloured with the fabric,
    not listed as a yarn). A blend: some mix (linear light) of a main yarn and a table group comes
    within EDGE_DELTA_E of it, neither all of one nor of the other."""
    main = [g for g in groups if g["role"] == "yarn" and g["share"] >= EDGE_MAIN_SHARE]
    tables = [g for g in groups if g["role"] == "table"]
    for g in groups:
        if g["role"] != "yarn" or g in main or not tables:
            continue
        c = srgb_to_linear(g["rgb"])
        for y in main:
            for t in tables:
                a, b = srgb_to_linear(y["rgb"]), srgb_to_linear(t["rgb"])
                mix = float(np.clip(np.dot(c - b, a - b) / max(np.dot(a - b, a - b), 1e-9), 0.0, 1.0))
                blend = linear_to_srgb(mix * a + (1 - mix) * b)
                if EDGE_MIX[0] < mix < EDGE_MIX[1] and np.linalg.norm(srgb_to_lab(blend) - g["lab"]) < EDGE_DELTA_E:
                    g["role"], g["blend_of_yarn"] = "edge", round(mix, 2)
                    break
            if g["role"] == "edge":
                break


def describe_groups(labels, photo, views, weight):
    """Each group: its role, share of the cloth (by area), colour (lit side) and shade, how many
    Gaussians and their median views; the yarns first, the most of the cloth first. Returns the groups
    and labels renumbered in that order."""
    total = weight.sum()
    groups = []
    for k in np.unique(labels):
        mine = labels == k
        known = mine & ~np.isnan(photo[:, 0])
        rgb, shade = lit_side(photo[known])
        lab = srgb_to_lab(rgb)
        groups.append({"old": int(k), "role": "table" if is_table(lab) else "yarn", "share": float(weight[mine].sum() / total),
                       "rgb": rgb, "lab": lab, "shade_rgb": shade, "gaussians": int(mine.sum()),
                       "seen_gaussians": int(known.sum()), "median_views": float(np.median(views[known])),
                       "lightness_range": [float(v) for v in np.percentile(srgb_to_lab(photo[known])[:, 0], [10, 90])]})
    mark_edges(groups)
    groups.sort(key=lambda g: (["yarn", "edge", "table"].index(g["role"]), -g["share"]))
    renumbered = np.full(len(labels), -1)
    for new, g in enumerate(groups):
        renumbered[labels == g.pop("old")] = new
    return groups, renumbered


# ----------------------------------------------------------------------------------------- the board, as a reference

def board_reference(dataset: Path, limit: int = 40):
    """The board paper's white and black as the camera saw them (median sRGB over the pictures that show
    it), from the pictures the splat was trained on; None when no picture shows enough of it."""
    import cv2
    import charuco

    spec = charuco.load_board_spec()
    whites, blacks, used = [], [], 0
    for picture in sorted((dataset / "images").glob("*.png")):
        image = cv2.imread(str(picture))
        paper, markers = charuco.board_paper_mask(image, spec)
        if paper.mean() < 0.05:
            continue
        # Away from the paper's edges: the blur there mixes in the cloth and the table.
        inner = cv2.erode(paper.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        rgb = image[..., ::-1].astype(np.float64) / 255.0
        value = rgb.max(axis=2)
        white, black = inner & (value > np.percentile(value[inner], 70)), inner & (value < np.percentile(value[inner], 25))
        if white.sum() < 500 or black.sum() < 500:
            continue
        whites.append(np.median(rgb[white], axis=0))
        blacks.append(np.median(rgb[black], axis=0))
        used += 1
        if used >= limit:
            break
    if not used:
        return None
    return {"white": np.median(whites, axis=0), "black": np.median(blacks, axis=0), "pictures": used}


def corrected(rgb, reference):
    """The camera's colour mapped so the board's paper comes out as paper: per channel, in linear light,
    the paper's black to BLACK_REFLECTANCE and its white to WHITE_REFLECTANCE - neutral, and at the
    reflectance real paper has."""
    if reference is None:
        return None
    lo, hi = srgb_to_linear(reference["black"]), srgb_to_linear(reference["white"])
    t_lo, t_hi = BLACK_REFLECTANCE, WHITE_REFLECTANCE
    lin = (srgb_to_linear(rgb) - lo) / np.maximum(hi - lo, 1e-6) * (t_hi - t_lo) + t_lo
    return linear_to_srgb(lin)


# ----------------------------------------------------------------------------------------- report

def swatches(colours, path: Path, key: str) -> None:
    from PIL import Image, ImageDraw

    width, height = 900, 140
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    x = 0
    for colour in colours:
        w = max(int(round(colour["share"] * width)), 1)
        fill = tuple(int(round(v * 255)) for v in np.clip(colour[key], 0, 1))
        draw.rectangle([x, 0, x + w, height - 30], fill=fill)
        draw.text((x + 4, height - 26), f"{colour['hex_' + key]}  {colour['share'] * 100:.0f}%", fill=(0, 0, 0))
        x += w
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="the splat_output folder (export/splat_clean.ply inside)")
    parser.add_argument("dataset", type=Path, help="the dataset it was trained on (its images show the board)")
    parser.add_argument("--colours", type=int, default=0, help="how many colours to find (0: as many as differ by MERGE_DELTA_E)")
    args = parser.parse_args()

    import crop_splat

    ply = args.run / "export" / "splat_clean.ply"
    splat = gaussian_splatting.load_splat_ply(ply)
    if splat["count"] != len(splat["positions"]):
        raise SystemExit(f"{ply.name} has {splat['count']} Gaussians, more than are read whole ({len(splat['positions'])})")
    frames, lens = crop_splat.cameras(args.run, args.dataset)
    transforms = json.loads((args.dataset / "transforms.json").read_text(encoding="utf-8"))
    photos = [args.dataset / f["file_path"] for f in transforms["frames"]]
    masks = [args.dataset / f["mask_path"] if f.get("mask_path") else None for f in transforms["frames"]]
    photo, views = photo_colours(splat, frames["normalised"], lens, photos, masks)
    scales = np.sort(np.asarray(splat["scales"], dtype=np.float64), axis=1)
    weight = np.asarray(splat["opacity"], dtype=np.float64) * scales[:, 1] * scales[:, 2]
    labels = group_yarns(np.asarray(splat["positions"], dtype=np.float64), photo, views, weight, args.colours)
    colours, labels = describe_groups(labels, photo, views, weight)
    np.save(args.run / "fabric_colours_labels.npy", labels.astype(np.int8))
    np.save(args.run / "fabric_photo_colours.npy", photo.astype(np.float32))
    reference = board_reference(args.dataset)
    for colour in colours:
        colour["hex_rgb"] = hex_of(colour["rgb"])
        colour["name"] = describe(colour["lab"])
        fixed = corrected(colour["rgb"], reference)
        if fixed is not None:
            colour["corrected"] = fixed
            colour["hex_corrected"] = hex_of(fixed)
            colour["corrected_lab"] = srgb_to_lab(fixed)
            colour["corrected_name"] = describe(colour["corrected_lab"])

    def plain(value):
        if isinstance(value, np.ndarray):
            return [round(float(v), 4) for v in value]
        if isinstance(value, float):
            return round(value, 4)
        return value

    seen = int((~np.isnan(photo[:, 0])).sum())
    report = {
        "splat": str(ply), "gaussians": {"total": int(len(photo)), "with_photo_colour": seen},
        "photo_views": int(min(PHOTO_VIEWS, len(photos))), "colours_asked": args.colours,
        "board_reference": None if reference is None else {k: plain(v) for k, v in reference.items()},
        "colours": [{k: plain(v) for k, v in c.items()} for c in colours],
        "method": "each Gaussian coloured from the training photos (front-most at each pixel, median over views); grouped by "
                  "neighbourhood colour (%d nearest), merged under %.0f delta E%s; each group's colour its lit side, shade its darkest "
                  "quarter; 'corrected' maps the board paper's white to reflectance %.2f and black to %.2f"
                  % (NEIGHBOURS, MERGE_DELTA_E, f" down to {args.colours} colours" if args.colours else "", WHITE_REFLECTANCE, BLACK_REFLECTANCE),
    }
    write_json(args.run / 'fabric_colours.json', report, indent=1)
    yarns = [c for c in colours if c["role"] == "yarn"]
    strip = [dict(c, share=c["share"] / sum(y["share"] for y in yarns)) for c in yarns]
    swatches(strip, args.run / "fabric_colours.png", "corrected" if reference is not None else "rgb")
    print(f"{seen} of {len(photo)} Gaussians coloured from {report['photo_views']} photos (median {np.median(views[views > 0]):.0f} views each)")
    if reference is not None:
        print(f"board reference from {reference['pictures']} pictures: paper white {hex_of(reference['white'])}, black {hex_of(reference['black'])}")
    for c in colours:
        line = f"  {c['share'] * 100:4.1f}%  {c['role']:5s} {c['hex_rgb']} ({c['name']})" + (f" [{c['blend_of_yarn']:.0%} yarn]" if "blend_of_yarn" in c else "")
        if "hex_corrected" in c:
            line += f" -> corrected {c['hex_corrected']}"
        print(line + f"   shade {hex_of(c['shade_rgb'])}   {c['gaussians']} Gaussians, {c['median_views']:.0f} views")


if __name__ == "__main__":
    main()
