"""The fabric's colours for a model (scripts/fabric_colours.py's result), and recolouring a splat by them.

fabric_colours.py writes, beside a model's export folder:
  fabric_colours.json         each group of the cloth: its role (yarn or table), share, the colour the
                              camera saw ("rgb"), the board-corrected one ("corrected") and its shade;
  fabric_colours_labels.npy   which group each Gaussian of export/splat_clean.ply is, in the file's order.

The fabric is given one colour, or several spread over it by yarn or in stripes (spread), with
recolour_all: every yarn Gaussian takes its colour, keeping only its brightness, so light, shade and
texture stay. The fabric's own colour (fabric_colour) is measured on
the splat the same way, so a colour taken from one model and put on another comes out the same.
Colours are saved by name in config/fabric_palettes/ (save_colour, list_palettes).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from paths import CONFIG_DIR, PROJECT_ROOT, SCRIPTS_DIR
from persistence import write_json

PALETTES_DIR = CONFIG_DIR / "fabric_palettes"
COLOURS_FILE, LABELS_FILE = "fabric_colours.json", "fabric_colours_labels.npy"
MAX_GAIN = 8.0


def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin):
    lin = np.clip(np.asarray(lin, dtype=np.float64), 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def hex_of(rgb) -> str:
    return "#" + "".join(f"{int(round(float(np.clip(v, 0, 1)) * 255)):02x}" for v in rgb)


def load(output: Path, gaussians: int) -> dict | None:
    """The fabric's groups for a model's output folder, with each Gaussian's group when the labels match
    the splat drawn (`gaussians` of them); None when the colours were never extracted."""
    path = Path(output) / COLOURS_FILE
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    groups = []
    for index, colour in enumerate(report.get("colours", [])):
        measured = np.asarray(colour.get("corrected", colour["rgb"]), dtype=np.float64)
        groups.append({"index": index, "role": colour.get("role", "yarn"), "share": float(colour.get("share", 0.0)),
                       "blend": float(colour.get("blend_of_yarn", 1.0)),
                       "name": colour.get("corrected_name", colour.get("name", "")),
                       "measured": measured, "camera": np.asarray(colour["rgb"], dtype=np.float64),
                       "shade": np.asarray(colour.get("shade_rgb", colour["rgb"]), dtype=np.float64)})
    labels = None
    label_path = Path(output) / LABELS_FILE
    if label_path.is_file():
        labels = np.load(label_path)
        if len(labels) != gaussians:          # the splat drawn is not the one the colours were read from
            labels = None
    return {"groups": groups, "labels": labels}


YARN_MAP_COLOURS = {"edge": (1.0, 0.55, 0.1), "table": (0.1, 0.3, 1.0)}


def yarn_map(fabric: dict, count: int) -> np.ndarray:
    """Each Gaussian painted flat in its group's colour - a yarn its own, the openings' rims orange,
    the table blue, the rest grey - to see how the fabric was divided."""
    paint = np.full((count, 3), 0.3, dtype=np.float32)
    for group in fabric["groups"]:
        paint[fabric["labels"] == group["index"]] = YARN_MAP_COLOURS.get(group["role"], group["measured"])
    return paint


def yarns(fabric: dict | None) -> list[dict]:
    """The fabric's yarns (not the table), the most of the cloth first."""
    if not fabric:
        return []
    return sorted((g for g in fabric["groups"] if g["role"] == "yarn"), key=lambda g: -g["share"])


_LUMA = np.array([0.2126, 0.7152, 0.0722])


LIT_WINDOW = (50.0, 90.0)       # a yarn's lit side: between these brightness percentiles of its Gaussians
CLIPPED = 0.90                  # a channel this bright is at the camera's top: its colour is lost


def lit_colour(colours: np.ndarray):
    """The colour of a set of Gaussians' lit side - the mean (linear light) of those between the
    LIT_WINDOW brightness percentiles, leaving out clipped ones - and its brightness (luminance)."""
    lin = srgb_to_linear(colours)
    honest = lin[np.asarray(colours).max(axis=1) < CLIPPED] if (np.asarray(colours).max(axis=1) < CLIPPED).sum() >= 20 else lin
    luma = honest @ _LUMA
    lo, hi = np.percentile(luma, LIT_WINDOW)
    mean = honest[(luma >= lo) & (luma <= hi)].mean(axis=0)
    return linear_to_srgb(mean), float(mean @ _LUMA)


def fabric_colour(fabric: dict | None, colours: np.ndarray):
    """The fabric's one colour, as the scan shows it: the lit colour of its main yarn's Gaussians in
    `colours` (the splat's own, as trained). Measured on the splat itself, so a colour taken from one
    model and put on another (recolour_all, measured the same way) comes out the same. None when the
    yarns are not known."""
    main = yarns(fabric)
    if not main or fabric["labels"] is None or not (fabric["labels"] == main[0]["index"]).any():
        return None
    return lit_colour(np.asarray(colours)[fabric["labels"] == main[0]["index"]])[0]


def yarn_colours(fabric: dict | None, colours: np.ndarray) -> list:
    """Every yarn's colour as the scan shows it, the most of the cloth first (fabric_colour, for each)."""
    if not fabric or fabric["labels"] is None:
        return []
    found = []
    for group in yarns(fabric):
        mine = fabric["labels"] == group["index"]
        if mine.sum() >= 20:
            found.append(lit_colour(np.asarray(colours)[mine])[0])
    return found


def parse_sequence(text: str, count: int) -> list[int]:
    """The order colours are spread in: "1,2" or "1 1 2" - colour numbers from 1 - as indices from 0;
    numbers past `count` are left out, and an empty or unreadable order is 1, 2, ... count."""
    picked = [int(t) - 1 for t in text.replace(",", " ").split() if t.isdigit() and 1 <= int(t) <= count]
    return picked or list(range(count))


def long_axis_deg(positions: np.ndarray, labels: np.ndarray, groups: list[dict]) -> float:
    """The direction the fabric is longest in, across its plane (x, y of a laid-flat splat), in degrees."""
    yarn = np.isin(labels, [g["index"] for g in groups if g["role"] == "yarn"])
    xy = np.asarray(positions, dtype=np.float64)[yarn if yarn.sum() >= 3 else slice(None), :2]
    _, _, axes = np.linalg.svd(xy - xy.mean(axis=0), full_matrices=False)
    return float(np.degrees(np.arctan2(axes[0][1], axes[0][0])) % 180.0)


SPREADS = ("By yarn", "In stripes")
BY_YARN, STRIPES = range(len(SPREADS))


def spread(fabric: dict, positions: np.ndarray, palette: list, sequence: list[int], how: int,
           stripe_mm: float = 5.0, angle_deg: float = 0.0, mm_per_unit: float = 1.0) -> np.ndarray:
    """Which colour each Gaussian takes ((n, 3) sRGB) when `palette`'s colours are spread over the
    fabric in `sequence` (indices into the palette, repeated):
      BY_YARN     the fabric's yarns, most of the cloth first: yarn 1 takes the sequence's first colour,
                  yarn 2 the second, and so on round the sequence;
      STRIPES     bands `stripe_mm` wide across the fabric's plane, one after another in the direction
                  `angle_deg` (x, y of a laid-flat splat): band 1 the first colour, band 2 the second...
    The openings' rims and the table take their nearest yarn's (recolour_all moves the rims as far as
    they are yarn and leaves the table)."""
    from scipy.spatial import cKDTree

    positions = np.asarray(positions, dtype=np.float64)
    labels = fabric["labels"]
    palette = np.asarray(palette, dtype=np.float64).reshape(-1, 3)
    order = [palette[k] for k in sequence] or [palette[0]]
    targets = np.tile(order[0], (len(positions), 1))
    yarn_list = yarns(fabric)
    yarn = np.isin(labels, [g["index"] for g in yarn_list])
    if how == BY_YARN:
        for rank, group in enumerate(yarn_list):
            targets[labels == group["index"]] = order[rank % len(order)]
    else:
        angle = np.radians(angle_deg)
        along = positions[:, :2] @ np.array([np.cos(angle), np.sin(angle)]) * float(mm_per_unit)
        start = along[yarn].min() if yarn.any() else along.min()
        band = np.floor((along - start) / max(float(stripe_mm), 1e-3)).astype(int)
        targets = np.asarray(order)[np.mod(band, len(order))]
    if yarn.any() and (~yarn).any():
        _, nearest = cKDTree(positions[yarn]).query(positions[~yarn])
        targets[~yarn] = targets[np.flatnonzero(yarn)[nearest]]
    return targets


def recolour_all(colours: np.ndarray, labels: np.ndarray, groups: list[dict], target) -> np.ndarray:
    """Every yarn Gaussian of the fabric in `target` - one colour (sRGB 0-1), or one per Gaussian ((n, 3),
    from spread) - its hue and saturation
    replaced, its brightness kept relative to its yarn's lit brightness (lit_colour) - so the lit side
    comes out `target` and shading, glare and texture stay. The fabric is one yarn now: every other
    yarn takes the main yarn's light and shade - a Gaussian's brightness rank in its own yarn is given
    the main yarn's brightness at that rank (a dark band is mostly near-black fog with a few lighter
    Gaussians; in one light colour it would be specks on black). The openings' rims move as far as
    they are yarn; the table is left as it is."""
    colours = np.asarray(colours)
    out = srgb_to_linear(colours).copy()
    target = np.broadcast_to(srgb_to_linear(np.asarray(target, dtype=np.float64)), out.shape)
    yarn_groups = sorted((g for g in groups if g["role"] == "yarn" and (labels == g["index"]).sum() >= 20), key=lambda g: -g["share"])
    if not yarn_groups:
        return colours.astype(np.float32)

    def relative(group):
        mine = labels == group["index"]
        _, reference = lit_colour(colours[mine])
        return mine, (out[mine] @ _LUMA) / max(reference, 1e-5)

    main_mask, main_brightness = relative(yarn_groups[0])
    ladder = np.sort(main_brightness)
    for group in yarn_groups:
        mine, brightness = (main_mask, main_brightness) if group is yarn_groups[0] else relative(group)
        if group is not yarn_groups[0]:
            rank = (np.argsort(np.argsort(brightness)) + 0.5) / len(brightness)
            brightness = np.interp(rank, (np.arange(len(ladder)) + 0.5) / len(ladder), ladder)
        out[mine] = target[mine] * brightness[:, None]
    # The openings' rims ("edge": yarn and table in one pixel) move toward the colour as far as they are
    # yarn - their brightness read against the main yarn's lit side.
    _, main_reference = lit_colour(colours[main_mask])
    for group in groups:
        mine = labels == group["index"]
        if group["role"] != "edge" or not mine.any():
            continue
        coloured = target[mine] * ((out[mine] @ _LUMA) / max(main_reference, 1e-5))[:, None]
        out[mine] += group["blend"] * (coloured - out[mine])
    return linear_to_srgb(out).astype(np.float32)


def extract(output: Path, dataset: Path, colours: int = 0) -> subprocess.Popen:
    """Runs fabric_colours.py for a model (about a minute) as its own process - `colours` of them to find,
    0 for as many as differ - and returns it to be polled."""
    return subprocess.Popen([sys.executable, str(SCRIPTS_DIR / "fabric_colours.py"), str(output), str(dataset),
                             "--colours", str(int(colours))],
                            cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def save_colour(name: str, colour, source: str, folder: Path = PALETTES_DIR) -> Path:
    """A fabric colour (sRGB 0-1) - or several, (k, 3) - saved under `name`, to put on another model."""
    safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in name).strip() or "colour"
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{safe}.json"
    entries = [{"rgb": [round(float(v), 4) for v in c], "hex": hex_of(c)} for c in np.asarray(colour, dtype=np.float64).reshape(-1, 3)]
    write_json(path, {'name': safe, 'source': source, 'colours': entries}, indent=1)
    return path


def list_palettes(folder: Path = PALETTES_DIR) -> list[dict]:
    """The saved palettes: {"name", "path", "source", "colours": [{"rgb", "hex", "name", "share"}]}."""
    palettes = []
    for path in sorted(Path(folder).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            palettes.append({"name": data.get("name", path.stem), "path": path, "source": data.get("source", ""),
                             "colours": data["colours"]})
        except (OSError, ValueError, KeyError):
            continue
    return palettes
