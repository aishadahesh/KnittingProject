"""Model Mode: a trained splat of the cloth, its yarn as equations, and the yarn rebuilt from them.

Left, the controls: which model, its scale, the equation fit and what it found. Right, the pictures,
all from one shared orbit camera (drag to turn, right-drag to move, wheel to zoom):
  * Scan - the splat as trained (the cropped cloth),
  * Equations - yarn tubes built from the strand equations alone, white or coloured by how far
    each point lies from the scan (the heat map),
  * Side by side - the two at once, turning together,
  * Overlay - the equation yarn laid over the scan, as transparent as the slider says.
The equation yarn is either the traced strands (only what the pictures showed clearly), or the
complete repeating loops: the cloth's repeat found in the scan and filled in as Edit Mode's knit
(lace_lattice.py), which "Open in Edit Mode" hands over to be edited there.
"Fabric colour" shows the fabric's colour, measured on the splat (its yarn's lit side; which Gaussians
are yarn comes from scripts/fabric_colours.py), and puts one colour on the whole fabric - picked, taken
from another model, or saved earlier - keeping its light and shade (fabric_palette.py); the equation
yarn is drawn in it too.
The pictures are drawn on the GPU (splat_gl.py) every frame, so they follow the mouse. "Spin" turns
them by themselves, "Wiggle" rocks them a little either way (near yarn slides over far yarn),
"Draw" shows the soft splat or its single Gaussians (ellipses, centres, outlines in the soft splat,
or each one's longest axis as a 3D line, alone or on the soft splat), "Edge-on" looks along the cloth, and "Colour by height"
paints each Gaussian by how far it stands above the cloth's bow - the ways to see that the model is
really 3D. Where the GPU
renderer cannot start, the CPU one (gaussian_splatting.py) draws them, sharpening while the view rests.
The geometry is strand_equations.py and lace_lattice.py; the splat loading is gaussian_splatting.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from imgui_bundle import imgui
from PIL import Image

import crochet_core
import crochet_fit
import fabric_palette
import gaussian_splatting
import yarn_tracks
import lace_lattice
import paths
import splat_gl
import strand_equations
from rendering import draw_fitted_texture, upload_rgb_texture

COLOR_OK = (0.25, 0.80, 0.40, 1.0)
COLOR_WARN = (0.95, 0.75, 0.20, 1.0)
COLOR_MUTED = (0.65, 0.65, 0.65, 1.0)
RENDER_SIZE = (560, 420)
SPIN_DEG_PER_S = 25.0
YARNS_FILE, YARNS_JSON = "fabric_yarns.npy", "fabric_yarns.json"      # scripts/fabric_yarns.py's
# "Wiggle" rocks the view this many degrees either way, this many times a second: the parallax of
# near yarn sliding over far yarn is the plainest sign the model is 3D.
WIGGLE_DEG, WIGGLE_HZ = 8.0, 0.8
BACKGROUND = (18, 18, 24)
SHOWS = ("Scan", "Equations", "Side by side", "Overlay")
COLOURINGS = ("Fabric colour", "Distance to scan (heat map)")
SOURCES = ("Traced strands", "Knit repeating loops", "Crochet model")
SELECTED_COLOUR = (1.0, 0.55, 0.10)
CONTROLS_WIDTH = 410.0


def find_models(root: Path = paths.ROBOT_SCANS_DIR) -> list[dict]:
    """Every trained splat under robot_scans: its PLY (the cropped one when there is one), the training
    run and dataset it came from (for the cameras), its scale and its held-out PSNR."""
    models = []
    for export in sorted(Path(root).glob("*/splat_output*/export")):
        ply = export / "splat_clean.ply"
        if not ply.is_file():
            ply = export / "splat.ply"
        if not ply.is_file():
            continue
        output, session = export.parent, export.parent.parent
        runs = [p for p in output.glob("*/splatfacto/*")
                if (p / "dataparser_transforms.json").is_file() and not p.name.endswith("_cloth")]
        run = max(runs, key=lambda p: p.stat().st_mtime) if runs else None
        dataset = session / run.parent.parent.name if run is not None else None
        try:
            psnr = float(json.loads((output / "eval.json").read_text(encoding="utf-8"))["results"]["psnr"])
        except (OSError, ValueError, KeyError, TypeError):
            psnr = None
        models.append({
            "name": f"{session.name} / {output.name}" + ("" if ply.name == "splat_clean.ply" else " (uncropped)"),
            "ply": ply, "output": output, "run": run,
            "dataset": dataset if dataset is not None and (dataset / "transforms.json").is_file() else None,
            "mm_per_unit": strand_equations.load_scale(output), "psnr": psnr,
        })
    return models


def _view(state) -> SimpleNamespace:
    view = state.get("model_view")
    if view is None:
        view = SimpleNamespace(
            models=None, index=-1, splat=None, loader=None, frame=None, mm_per_unit=1.0,
            camera=(0.0, 70.0, 1.0, (0.0, 0.0)), dragging=False, pictures={},
            tolerance=0.05, max_terms=10, extraction=None, extractor=None, progress="", status="",
            selected=-1, show=2, colouring=1, heat_mm=0.5, overlay=0.6, tube_splat=None, tube_key=None,
            periodic=None, periodic_job=None, source=0, counts=(3, 6),
            crochet=None, crochet_job=None, crochet_counts=[3, 3], crochet_wraps=1, crochet_along_holes=True, crochet_auto=False,
            fabric=None, fabric_measured=None, fabric_target=None, fabric_base=None, fabric_version=0, fabric_job=None,
            recolour_scan=True, fabric_source=0, palettes=None, palette_pick=0, palette_name="", yarn_map=False, colours_to_find=0,
            multi=None, yarns=None, yarns_info=None, yarns_job=None, yarn_colours={}, show_yarns=False, yarn_tool=0,
            yarn_pen=None, yarn_order="1,2", yarn_flip=False, brush_yarn=0, brush_seeds=[], yarn_dirs=None, yarn_dirs_job=None,
            yarns_edited=False,
            gl=None, gl_failed=False, gl_used=set(), spin=False, wiggle=False, wiggle_offset=0.0, by_height=False, plane=None, style=splat_gl.SOFT,
        )
        state.model_view = view
    return view


# ----------------------------------------------------------------------------------------- loading

def _select_model(view, index: int) -> None:
    model = view.models[index]
    view.index, view.splat, view.frame, view.extraction, view.selected = index, None, None, None, -1
    view.periodic, view.crochet, view.source = None, None, 0
    view.tube_splat, view.tube_key, view.pictures, view.plane = None, None, {}, None
    view.fabric, view.fabric_measured, view.fabric_target, view.fabric_base, view.multi = None, None, None, None, None
    view.yarns, view.yarns_info, view.yarn_colours, view.yarn_dirs, view.yarns_edited = None, None, {}, None, False
    view.palette_name = model["name"].split(" / ")[0]
    view.mm_per_unit = model["mm_per_unit"] or 1.0
    holder = {}

    def load():
        try:
            splat = gaussian_splatting.load_splat_ply(model["ply"])
            normal = None
            if model["run"] is not None and model["dataset"] is not None:
                centres, looking = strand_equations.camera_centres_and_axes(model["run"], model["dataset"])
                normal = -looking.mean(axis=0)
                if model["mm_per_unit"] is None:
                    holder["guess"] = strand_equations.guess_scale(splat["positions"], centres)
            # Shown and measured lying flat: the orbit then turns round the cloth's own up.
            holder["splat"] = strand_equations.lay_flat(splat, normal)
        except Exception as exc:          # shown in the panel, not raised into the frame
            holder["error"] = str(exc)

    thread = threading.Thread(target=load, daemon=True, name="model-load")
    thread.start()
    view.loader = (thread, holder)


def _poll_loader(view) -> None:
    if view.loader is None or view.loader[0].is_alive():
        return
    holder = view.loader[1]
    view.loader = None
    if "error" in holder:
        view.status = f"Could not read the model: {holder['error']}"
        return
    view.splat = holder["splat"]
    if "guess" in holder:
        view.mm_per_unit = holder["guess"]
    view.frame = gaussian_splatting.orbit_frame(view.splat["positions"][view.splat["opacity"] >= 0.1])
    view.plane = splat_gl.cloth_plane(view.splat["positions"][view.splat["opacity"] >= 0.1])
    view.fabric_base = np.asarray(view.splat["colours"], dtype=np.float32).copy()
    _load_fabric(view)


# ----------------------------------------------------------------------------------------- fabric colours

def _load_fabric(view) -> None:
    """The selected model's fabric (fabric_palette) and its one colour, measured on the splat itself."""
    view.fabric = fabric_palette.load(view.models[view.index]["output"], len(view.fabric_base)) if view.fabric_base is not None else None
    view.fabric_measured = fabric_palette.fabric_colour(view.fabric, view.fabric_base) if view.fabric else None
    view.fabric_target, view.multi = None, None
    _load_yarns(view)
    _apply_fabric(view)


def _load_yarns(view) -> None:
    """The model's yarns (scripts/fabric_yarns.py: each Gaussian's yarn) when they were found for this splat."""
    output = view.models[view.index]["output"]
    path, info = output / YARNS_FILE, output / YARNS_JSON
    view.yarns, view.yarns_info = None, None
    if path.is_file() and view.fabric_base is not None:
        yarns = np.load(path)
        if len(yarns) == len(view.fabric_base):
            view.yarns = yarns.astype(np.int64)
            view.yarns_info = json.loads(info.read_text(encoding="utf-8")) if info.is_file() else {}
    view.yarn_colours, view.yarn_dirs, view.yarns_edited = {}, None, False


def _yarn_numbers(view) -> list[int]:
    numbers = sorted(int(k) for k in np.unique(view.yarns) if k >= 0) if view.yarns is not None else []
    return numbers[::-1] if view.yarn_flip else numbers


def _main_yarn_colour(view):
    """The colour the equation yarn is drawn in: the fabric's colour as chosen or measured, else plain white."""
    if view.yarn_colours:
        return tuple(float(v) for v in next(iter(view.yarn_colours.values())))
    colour = (view.multi["colours"][0] if view.multi else
              view.fabric_target if view.fabric_target is not None else view.fabric_measured)
    return (0.92, 0.92, 0.88) if colour is None else tuple(float(v) for v in colour)


def _apply_fabric(view) -> None:
    """The scan's colours with the whole fabric in the chosen colour (or as trained, unticked or unchosen)."""
    if view.splat is None or view.fabric_base is None:
        return
    colours = view.fabric_base
    if view.show_yarns and view.yarns is not None:
        colours = _yarns_painted(view)
    elif (view.recolour_scan and view.yarn_colours and view.yarns is not None and view.fabric is not None
            and view.fabric["labels"] is not None):
        # Each coloured yarn's Gaussians take its colour, light and shade kept; the rest stay as scanned.
        targets = np.asarray(view.fabric_base, dtype=np.float64).copy()
        coloured = np.zeros(len(targets), dtype=bool)
        for number, colour in view.yarn_colours.items():
            mine = view.yarns == number
            targets[mine] = colour
            coloured |= mine
        recoloured = fabric_palette.recolour_all(view.fabric_base, view.fabric["labels"], view.fabric["groups"], targets)
        colours = np.where(coloured[:, None], recoloured, view.fabric_base).astype(np.float32)
    elif view.yarn_map and view.fabric is not None and view.fabric["labels"] is not None:
        colours = fabric_palette.yarn_map(view.fabric, len(view.fabric_base))
    elif view.recolour_scan and view.multi and view.fabric is not None and view.fabric["labels"] is not None:
        m = view.multi
        targets = fabric_palette.spread(view.fabric, view.splat["positions"], m["colours"],
                                        fabric_palette.parse_sequence(m["sequence"], len(m["colours"])), m["how"],
                                        m["stripe_mm"], m["angle"], view.mm_per_unit)
        colours = fabric_palette.recolour_all(view.fabric_base, view.fabric["labels"], view.fabric["groups"], targets)
    elif (view.recolour_scan and view.fabric_target is not None and view.fabric is not None
            and view.fabric["labels"] is not None):
        colours = fabric_palette.recolour_all(view.fabric_base, view.fabric["labels"], view.fabric["groups"], view.fabric_target)
    view.splat["colours"] = colours
    view.fabric_version += 1
    view.pictures = {}          # the CPU pictures are drawn again; the GPU ones are keyed by the version


def _poll_fabric_job(view) -> None:
    if view.fabric_job is None or view.fabric_job.poll() is None:
        return
    job, view.fabric_job = view.fabric_job, None
    output = job.stdout.read() if job.stdout else ""
    if job.returncode == 0:
        _load_fabric(view)
        view.status = "Fabric colours extracted."
    else:
        view.status = "Could not extract the fabric colours: " + (output.strip().splitlines() or ["(no message)"])[-1]


def _colours_of_model(model) -> list:
    """Another model's yarn colours, the most of its cloth first, measured on its own splat as this
    one's are; empty when it has none."""
    try:
        splat = gaussian_splatting.load_splat_ply(model["ply"])
    except Exception:
        return []
    fabric = fabric_palette.load(model["output"], len(splat["colours"]))
    return fabric_palette.yarn_colours(fabric, splat["colours"]) if fabric else []


def _take_colours(view, colours, source: str) -> None:
    """Put `colours` on this model: one is the fabric's colour; several are spread - by yarn when this
    fabric has as many yarns, else in stripes across it."""
    colours = [np.asarray(c, dtype=np.float64) for c in colours]
    if len(colours) == 1:
        view.fabric_target, view.multi = colours[0], None
        view.status = f"Fabric coloured {fabric_palette.hex_of(colours[0])}, from {source}."
    else:
        enough_yarns = len(fabric_palette.yarns(view.fabric)) >= len(colours)
        view.multi = {"colours": colours, "sequence": ",".join(str(k + 1) for k in range(len(colours))),
                      "how": fabric_palette.BY_YARN if enough_yarns else fabric_palette.STRIPES, "stripe_mm": 5.0,
                      "angle": fabric_palette.long_axis_deg(view.splat["positions"], view.fabric["labels"], view.fabric["groups"])}
        view.status = (f"{len(colours)} colours from {source}, spread " + ("by yarn." if enough_yarns else
                       "in stripes (this fabric has fewer yarns than colours)."))
    _apply_fabric(view)


def _draw_multi(view) -> None:
    """Several colours: each with its picker, how they are spread (by yarn, or in stripes of a width and
    a direction) and in what order."""
    m = view.multi
    imgui.separator()
    imgui.text(f"{len(m['colours'])} colours")
    changed_any = False
    remove = None
    for k, colour in enumerate(m["colours"]):
        changed, picked = imgui.color_edit3(f"Colour {k + 1}##multi_{k}", tuple(float(v) for v in colour), imgui.ColorEditFlags_.no_inputs)
        if changed:
            m["colours"][k] = np.asarray(picked[:3], dtype=np.float64)
            changed_any = True
        imgui.same_line()
        imgui.text(fabric_palette.hex_of(m["colours"][k]))
        if len(m["colours"]) > 1:
            imgui.same_line()
            if imgui.small_button(f"x##multi_remove_{k}"):
                remove = k
    if remove is not None:
        del m["colours"][remove]
        changed_any = True
    if imgui.small_button("Add colour##multi_add"):
        m["colours"].append(m["colours"][-1].copy())
        changed_any = True
    imgui.same_line()
    if imgui.small_button("Back to one colour##multi_clear"):
        view.fabric_target, view.multi = m["colours"][0], None
        _apply_fabric(view)
        return
    for i, name in enumerate(fabric_palette.SPREADS):
        if i:
            imgui.same_line()
        if imgui.radio_button(f"{name}##multi_how{i}", m["how"] == i):
            m["how"] = i
            changed_any = True
    if m["how"] == fabric_palette.BY_YARN:
        found = len(fabric_palette.yarns(view.fabric))
        imgui.text_colored(COLOR_MUTED if found > 1 else COLOR_WARN,
                           f"Yarn 1 takes the order's first colour, yarn 2 the second... ({found} yarn{'s' if found != 1 else ''} found here"
                           + ("; use stripes to show more than one colour)" if found < 2 else ")"))
    else:
        imgui.set_next_item_width(160)
        changed, m["stripe_mm"] = imgui.slider_float("stripe width (mm)##multi_width", float(m["stripe_mm"]), 0.5, 30.0, "%.1f")
        changed_any |= changed
        imgui.set_next_item_width(160)
        changed, m["angle"] = imgui.slider_float("direction (deg)##multi_angle", float(m["angle"]), 0.0, 180.0, "%.0f")
        changed_any |= changed
        imgui.same_line()
        if imgui.small_button("Turn 90##multi_turn"):
            m["angle"] = (float(m["angle"]) + 90.0) % 180.0
            changed_any = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Stripes across the fabric, or along it.")
    imgui.set_next_item_width(160)
    changed, m["sequence"] = imgui.input_text("order##multi_order", m["sequence"])
    changed_any |= changed
    if imgui.is_item_hovered():
        imgui.set_tooltip("The colour numbers in the order they follow one another, repeated: 1,2 alternates;\n"
                          "1,1,2 is two of colour 1, then one of colour 2.")
    if changed_any:
        _apply_fabric(view)


def _draw_fabric_colours(view) -> None:
    """The fabric's one colour: as the scan shows it, a picker for another, another model's, saved ones."""
    model = view.models[view.index] if view.models and view.index >= 0 else None
    if model is None or view.splat is None:
        imgui.text_colored(COLOR_MUTED, "Load a model first.")
        return
    if view.fabric_job is not None:
        imgui.text_colored(COLOR_MUTED, "Reading the yarns from the photos (about a minute)...")
        return
    if view.fabric is None or view.fabric_measured is None:
        imgui.text_wrapped("Not extracted for this model yet: which Gaussians are the fabric's yarn (not the table "
                           "under it), and the fabric's colour.")
        if model["dataset"] is None or not model["ply"].name.endswith("_clean.ply"):
            imgui.text_colored(COLOR_WARN, "Needs the cropped splat (splat_clean.ply) and its training pictures.")
        elif imgui.button("Extract the fabric colour##fabric_extract", (-1, 0)):
            view.fabric_job = fabric_palette.extract(model["output"], model["dataset"], view.colours_to_find)
        return
    _draw_extracted(view, model)
    measured = view.fabric_measured
    imgui.color_button("##fabric_measured", (*[float(v) for v in measured], 1.0), 0, (40, 22))
    imgui.same_line()
    imgui.text(f"Measured {fabric_palette.hex_of(measured)}  (the yarn's lit side, as the scan shows it)")
    shown = view.fabric_target if view.fabric_target is not None else measured
    if not view.multi:
        changed, picked = imgui.color_edit3("New colour##fabric_target", tuple(float(v) for v in shown))
        if changed:
            view.fabric_target = np.asarray(picked[:3], dtype=np.float64)
            _apply_fabric(view)
    if imgui.is_item_hovered():
        imgui.set_tooltip("The whole fabric takes this colour; its light and shade are kept (the table under it is not changed).")
    if imgui.small_button("Reset to measured##fabric_reset"):
        view.fabric_target, view.multi = None, None
        _apply_fabric(view)
    imgui.same_line()
    changed, view.recolour_scan = imgui.checkbox("Recolour the scan##fabric_recolour", bool(view.recolour_scan))
    if changed:
        _apply_fabric(view)
    if view.fabric["labels"] is None:
        imgui.text_colored(COLOR_WARN, "The colours were read from another splat file: extract again to recolour this one.")
        if imgui.small_button("Extract again##fabric_again") and model["dataset"] is not None:
            view.fabric_job = fabric_palette.extract(model["output"], model["dataset"], view.colours_to_find)
    # Another model's colour.
    others = [m for m in view.models if m is not model and (m["output"] / fabric_palette.COLOURS_FILE).is_file()]
    if others:
        view.fabric_source = min(int(view.fabric_source), len(others) - 1)
        imgui.set_next_item_width(-150)
        _, view.fabric_source = imgui.combo("##fabric_source", view.fabric_source, [m["name"] for m in others])
        imgui.same_line()
        if imgui.button("Take its colours##fabric_take", (-1, 0)):
            found = _colours_of_model(others[view.fabric_source])
            if not found:
                view.status = "That model's colours could not be read (extract them again there)."
            else:
                _take_colours(view, found, others[view.fabric_source]["name"])
        if imgui.is_item_hovered():
            imgui.set_tooltip("All the other model's yarn colours, measured the same way: put on this one, they look the same.\n"
                              "Several are spread by yarn or in stripes - choose below.")
    if view.multi:
        _draw_multi(view)
    _draw_saved_colours(view, shown)
    imgui.text_colored(COLOR_MUTED, "The equation yarn is drawn in this colour too.")
    _draw_yarns(view, model)


def _yarns_painted(view) -> np.ndarray:
    """Each yarn in its own flat colour (the one it is given, else a distinct hue), the rest dark."""
    import colorsys

    numbers = _yarn_numbers(view)
    paint = np.full((len(view.yarns), 3), 0.12, dtype=np.float32)
    for k, number in enumerate(sorted(numbers)):
        hue = (k * 0.61803) % 1.0
        paint[view.yarns == number] = view.yarn_colours.get(number, colorsys.hsv_to_rgb(hue, 0.8, 0.95))
    return paint


def _poll_yarns_job(view) -> None:
    if view.yarns_job is not None and view.yarns_job.poll() is not None:
        job, view.yarns_job = view.yarns_job, None
        message = (job.stdout.read() if job.stdout else "").strip().splitlines()
        if job.returncode == 0:
            _load_yarns(view)
            _apply_fabric(view)
            view.status = "Yarns found: " + (message[-2] if len(message) >= 2 else "")
        else:
            view.status = "Could not find the yarns: " + (message[-1] if message else "(no message)")
    if view.yarn_dirs_job is not None and not view.yarn_dirs_job[0].is_alive():
        view.yarn_dirs = view.yarn_dirs_job[1].get("dirs")
        view.yarn_dirs_job = None


def _draw_yarns(view, model) -> None:
    """The fabric's yarns: found from the fitted model (or the Gaussians' directions), shown each in a
    colour, coloured in order or one by one with a click, and corrected with a brush."""
    imgui.separator()
    imgui.text("Yarns")
    fit_path = model["output"] / "stitch_fit" / "stitch_model.json"
    if view.yarns_job is not None:
        imgui.text_colored(COLOR_MUTED, "Finding the yarns...")
        return
    label = "from the fitted stitches" if fit_path.is_file() else "from the Gaussians' directions"
    if imgui.small_button(("Find the yarns again" if view.yarns is not None else "Find the yarns") + f" ({label})##yarns_find"):
        view.yarns_job = subprocess.Popen([sys.executable, str(paths.PROJECT_ROOT / "scripts" / "fabric_yarns.py"), str(model["output"])],
                                          cwd=str(paths.PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if imgui.is_item_hovered():
        imgui.set_tooltip("From the fitted stitches: each yarn Gaussian takes its nearest fitted curve; a row's curves are one yarn.\n"
                          "Without a photo fit, from the Gaussians alone (less reliable: yarns come out in patches).")
    if view.yarns is None:
        imgui.text_colored(COLOR_MUTED, "Not found yet for this model.")
        return
    info = view.yarns_info or {}
    numbers = _yarn_numbers(view)
    how = (f"from the fit, median {info['median_distance_mm']:.2f} mm from the fitted curves" if info.get("source") == "fit"
           else "from the Gaussians' directions")
    imgui.text(f"{len(numbers)} yarns ({how})")
    changed, view.show_yarns = imgui.checkbox("Show the yarns##yarns_show", bool(view.show_yarns))
    if changed:
        _apply_fabric(view)
    # Colouring in order: the colours shown above (several taken, or the one), round and round.
    palette = view.multi["colours"] if view.multi else [view.fabric_target if view.fabric_target is not None else view.fabric_measured]
    imgui.set_next_item_width(90)
    _, view.yarn_order = imgui.input_text("order##yarn_order", view.yarn_order)
    imgui.same_line()
    _, view.yarn_flip = imgui.checkbox("Flip##yarn_flip", bool(view.yarn_flip))
    imgui.same_line()
    if imgui.small_button("Colour the yarns in order##yarn_in_order"):
        order = fabric_palette.parse_sequence(view.yarn_order, len(palette))
        view.yarn_colours = {number: np.asarray(palette[order[k % len(order)]], dtype=np.float64) for k, number in enumerate(_yarn_numbers(view))}
        _apply_fabric(view)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Yarn 1 takes the order's first colour, yarn 2 the second... with the colours above\n"
                          "(the several taken from a model, or the one). Flip starts from the other side.")
    imgui.same_line()
    if imgui.small_button("Clear##yarn_clear"):
        view.yarn_colours = {}
        _apply_fabric(view)
    # Tools on the picture.
    for i, name in enumerate(("Turn the view", "Click a yarn to colour it", "Brush a yarn")):
        if i:
            imgui.same_line()
        if imgui.radio_button(f"{name}##yarn_tool{i}", view.yarn_tool == i):
            view.yarn_tool = i
    if view.yarn_tool:
        if view.yarn_pen is None:
            view.yarn_pen = np.asarray(palette[0], dtype=np.float64)
        changed, picked = imgui.color_edit3("colour to give##yarn_pen", tuple(float(v) for v in view.yarn_pen), imgui.ColorEditFlags_.no_inputs)
        if changed:
            view.yarn_pen = np.asarray(picked[:3], dtype=np.float64)
        imgui.same_line()
        imgui.text_colored(COLOR_MUTED, "on the Scan picture; right-drag still moves the view")
    if view.yarn_tool == 2:
        imgui.set_next_item_width(90)
        _, view.brush_yarn = imgui.input_int("yarn to brush into##brush_yarn", int(view.brush_yarn))
        if view.yarn_dirs is None and view.yarn_dirs_job is None:
            holder = {}
            positions = np.asarray(view.splat["positions"], dtype=np.float64)
            splat, mm = view.splat, float(view.mm_per_unit)

            def work():
                axes = yarn_tracks.long_axes(splat["scales"], splat["quaternion"])
                sc = np.sort(np.asarray(splat["scales"], dtype=np.float64), axis=1)
                weight = np.asarray(splat["opacity"], dtype=np.float64) * sc[:, 1] * sc[:, 2]
                radius = float((view.yarns_info or {}).get("radius_mm", yarn_tracks.YARN_RADIUS_MM)) / mm
                holder["dirs"] = (*yarn_tracks.directions(positions, axes, weight, yarn_tracks.NEIGHBOURHOOD_RADII * radius), radius)

            thread = threading.Thread(target=work, daemon=True, name="yarn-directions")
            thread.start()
            view.yarn_dirs_job = (thread, holder)
        if view.yarn_dirs is None:
            imgui.text_colored(COLOR_MUTED, "Reading the yarn directions for the brush...")
    if view.yarns_edited:
        if imgui.small_button("Save the yarn edits##yarns_save"):
            np.save(model["output"] / YARNS_FILE, view.yarns.astype(np.int32))
            view.yarns_edited = False
            view.status = "Yarn edits saved."


def _draw_extracted(view, model) -> None:
    """The colours found: each yarn's swatch, colour and share; the rims and the table summed up; how many
    to find, and extracting again; the yarn map, to see the division on the scan."""
    groups = view.fabric["groups"]
    yarn_total = sum(g["share"] for g in groups if g["role"] == "yarn") or 1.0
    for g in fabric_palette.yarns(view.fabric):
        imgui.color_button(f"##found_{g['index']}", (*[float(v) for v in g["measured"]], 1.0), 0, (22, 22))
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"{g['name']}: {fabric_palette.hex_of(g['measured'])} (board-corrected), "
                              f"{fabric_palette.hex_of(g['camera'])} as the camera saw it; in shade {fabric_palette.hex_of(g['shade'])}")
        imgui.same_line()
        imgui.text(f"{fabric_palette.hex_of(g['measured'])}  {g['share'] / yarn_total * 100:.0f}% of the yarn  {g['name']}")
    others = [g for g in groups if g["role"] != "yarn"]
    if others:
        edge = sum(g["share"] for g in others if g["role"] == "edge")
        table = sum(g["share"] for g in others if g["role"] == "table")
        imgui.text_colored(COLOR_MUTED, f"Also: the openings' rims {edge * 100:.0f}%, the table under them {table * 100:.0f}% (not yarn)")
    imgui.set_next_item_width(120)
    _, view.colours_to_find = imgui.slider_int("##colours_to_find", int(view.colours_to_find), 0, 6,
                                               "Auto" if int(view.colours_to_find) == 0 else "%d colours")
    if imgui.is_item_hovered():
        imgui.set_tooltip("How many colours to look for. Auto: as many as really differ (more than about 12 delta E).")
    imgui.same_line()
    if imgui.small_button("Extract again##fabric_reextract") and model["dataset"] is not None:
        view.fabric_job = fabric_palette.extract(model["output"], model["dataset"], view.colours_to_find)
    imgui.same_line()
    changed, view.yarn_map = imgui.checkbox("Show the yarn map##fabric_map", bool(view.yarn_map))
    if changed:
        _apply_fabric(view)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Each Gaussian painted flat in its group's colour: each yarn its own, the openings' rims orange,\n"
                          "the table blue - to check how the fabric was divided.")
    imgui.separator()


def _draw_saved_colours(view, shown) -> None:
    """Save the colour shown under a name, or put a saved one on this model (fabric_palette)."""
    imgui.separator()
    imgui.set_next_item_width(-150)
    _, view.palette_name = imgui.input_text("##palette_name", view.palette_name)
    imgui.same_line()
    if imgui.button("Save colours##palette_save", (-1, 0)):
        to_save = view.multi["colours"] if view.multi else [shown]
        path = fabric_palette.save_colour(view.palette_name, to_save, view.models[view.index]["name"])
        view.palettes = None
        view.status = f"Colour saved: {path}"
    if view.palettes is None:
        view.palettes = fabric_palette.list_palettes()
    if not view.palettes:
        imgui.text_colored(COLOR_MUTED, "No colours saved yet.")
        return
    view.palette_pick = min(int(view.palette_pick), len(view.palettes) - 1)
    saved = view.palettes[view.palette_pick]
    imgui.color_button("##palette_swatch", (*saved["colours"][0]["rgb"], 1.0), 0, (22, 22))
    imgui.same_line()
    imgui.set_next_item_width(-150)
    _, view.palette_pick = imgui.combo("##palette_pick", view.palette_pick,
                                       [f"{p['name']}  " + " ".join(c["hex"] for c in p["colours"]) for p in view.palettes])
    imgui.same_line()
    if imgui.button("Use these##palette_apply", (-1, 0)):
        saved = view.palettes[view.palette_pick]
        _take_colours(view, [c["rgb"] for c in saved["colours"]], f"saved \"{saved['name']}\"")


def _start_extraction(view) -> None:
    splat = view.splat
    holder = {}
    tolerance, max_terms, mm_per_unit = float(view.tolerance), int(view.max_terms), float(view.mm_per_unit)

    def run():
        try:
            # The splat was laid flat when loaded: the cloth's up is z.
            holder["result"] = strand_equations.extract(
                splat, mm_per_unit, (0.0, 0.0, 1.0), tolerance, max_terms, progress=lambda message: setattr(view, "progress", message))
        except Exception as exc:
            holder["error"] = str(exc)

    view.progress = "Starting"
    view.extractor = (threading.Thread(target=run, daemon=True, name="model-equations"), holder)
    view.extractor[0].start()


def _poll_extractor(view) -> None:
    if view.extractor is None or view.extractor[0].is_alive():
        return
    holder = view.extractor[1]
    view.extractor = None
    if "error" in holder:
        view.status = f"Equations not extracted: {holder['error']}"
        return
    view.extraction, view.selected, view.tube_key = holder["result"], -1, None
    view.periodic, view.crochet, view.source = None, None, 0
    summary = view.extraction.summary()
    if not summary["strands"]:
        view.status = ("No strands found. Check the scale (mm per splat unit): the yarn must come out "
                       "about 0.5-1.5 mm thick for the strand finder to see it.")
        return
    view.status = (f"{summary['strands']} strands as equations ({summary['coefficients']:,} numbers, "
                   f"{summary['fit_rms_mm']:.3f} mm from the traced centrelines)")
    if view.show == 0:
        view.show = 2


def _start_periodic_fit(view, counts=None) -> None:
    extraction, holder = view.extraction, {}

    def run():
        try:
            holder["result"] = lace_lattice.fit(extraction, counts, progress=lambda message: setattr(view, "progress", message))
        except Exception as exc:
            holder["error"] = str(exc)

    view.progress = "Starting"
    view.periodic_job = (threading.Thread(target=run, daemon=True, name="model-periodic"), holder)
    view.periodic_job[0].start()


def _poll_periodic(view) -> None:
    if view.periodic_job is None or view.periodic_job[0].is_alive():
        return
    holder = view.periodic_job[1]
    view.periodic_job = None
    if "error" in holder:
        view.status = f"No repeating loops fitted: {holder['error']}"
        return
    view.periodic, view.source, view.tube_key = holder["result"], 1, None
    fit = view.periodic
    view.counts = (fit.n_cols, fit.n_rows)
    view.status = (f"The cloth repeats every {fit.n_rows} rows x {fit.n_cols} stitches "
                   f"({fit.row_pitch * fit.n_rows:.1f} x {fit.stitch_width * fit.n_cols:.1f} mm); complete loops built over it")
    if view.show == 0:
        view.show = 2


def _equation_model(view):
    """What the equation pictures show: the chosen model when it has been fitted, else the traced strands."""
    if view.source == 2 and view.crochet is not None:
        return view.crochet
    if view.source == 1 and view.periodic is not None:
        return view.periodic
    return view.extraction


def _start_crochet_fit(view) -> None:
    extraction, holder = view.extraction, {}
    known = {} if view.crochet_auto else {"stitches": int(view.crochet_counts[0]), "rows": int(view.crochet_counts[1]),
                                          "wraps": int(view.crochet_wraps), "rows_along_holes": bool(view.crochet_along_holes)}

    def run():
        try:
            holder["result"] = crochet_fit.fit(extraction, progress=lambda message: setattr(view, "progress", message), **known)
        except Exception as exc:
            holder["error"] = str(exc)

    view.progress = "Starting"
    view.crochet_job = (threading.Thread(target=run, daemon=True, name="model-crochet"), holder)
    view.crochet_job[0].start()


def _poll_crochet(view) -> None:
    if view.crochet_job is None or view.crochet_job[0].is_alive():
        return
    holder = view.crochet_job[1]
    view.crochet_job = None
    if "error" in holder:
        view.status = f"No crochet model: {holder['error']}"
        return
    view.crochet, view.source, view.tube_key = holder["result"], 2, None
    fit = view.crochet
    rows, stitches = fit.cells.shape
    view.crochet_counts, view.crochet_wraps = [stitches, rows], fit.params.wraps
    view.status = (f"{fit.stitch_name}: a stitch every {fit.params.stitch_width:.2f} mm, rows {fit.params.row_height:.2f} mm; "
                   f"repeat {rows} rows x {stitches} stitches, rebuilt over the cloth")
    if view.show == 0:
        view.show = 2


def _draw_crochet(view, state) -> None:
    """The crochet model: what can be counted on the lace, the fit, what it found, and the hand-over."""
    imgui.text_wrapped("Count on the real lace what the pictures cannot show clearly; the rest is measured.")
    _, view.crochet_auto = imgui.checkbox("Search for the counts myself (less sure)##crochet_auto", bool(view.crochet_auto))
    if not view.crochet_auto:
        imgui.set_next_item_width(120)
        _, counts = imgui.input_int2("stitches, rows per repeat##crochet_counts", [int(c) for c in view.crochet_counts])
        view.crochet_counts = [int(np.clip(counts[0], 1, 32)), int(np.clip(counts[1], 1, 16))]
        if imgui.is_item_hovered():
            imgui.set_tooltip("From one opening to the next along a row: how many stitches (posts and chains);\n"
                              "from one row of openings to the next: how many rows.")
        imgui.set_next_item_width(120)
        _, wraps = imgui.slider_int("yarn-overs (1 dc, 2 tr)##crochet_wraps_fit", int(view.crochet_wraps), 0, 4)
        view.crochet_wraps = int(wraps)
        _, view.crochet_along_holes = imgui.checkbox("Rows run along the rows of openings##crochet_along", bool(view.crochet_along_holes))
    busy = view.crochet_job is not None or view.extraction is None
    if busy:
        imgui.begin_disabled()
    if imgui.button("Fit crochet model##crochet_fit", (-1, 0)):
        _start_crochet_fit(view)
    if busy:
        imgui.end_disabled()
    if view.extraction is None:
        imgui.text_colored(COLOR_MUTED, "Extract the loop equations first (it maps the yarn).")
    if view.crochet_job is not None:
        imgui.text_colored(COLOR_MUTED, view.progress + "...")
    fit = view.crochet
    if fit is None:
        return
    p = fit.params
    imgui.text(f"{fit.stitch_name}: stitch {p.stitch_width:.2f} mm, row {p.row_height:.2f} mm, yarn r {p.yarn_radius:.2f} mm")
    imgui.text(f"Rows at {fit.angle_deg:.0f} deg; repeat {fit.cells.shape[0]} rows x {fit.cells.shape[1]} stitches")
    _draw_bitmap(fit.cells)
    imgui.same_line()
    imgui.begin_group()
    for line in crochet_core.equations(p):
        imgui.text_wrapped(line)
    imgui.end_group()
    _draw_comparison(fit, "model_compare_crochet")
    imgui.push_style_color(imgui.Col_.button, (0.22, 0.48, 0.78, 1.0))
    if imgui.button("Open crochet in Edit Mode##crochet_to_edit", (-1, 0)):
        state.model_to_edit = {"fabric": "crochet", "parameters": p.as_dict(), "stitches": fit.cells.copy()}
    imgui.pop_style_color()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Edit Mode switches to its crochet fabric with these numbers and this stitch grid (Undo there goes back).")


def _tube_splat(view):
    """The rebuilt yarn as a splat-shaped dict, coloured as asked, rebuilt only when that changes."""
    result = _equation_model(view)
    if result is None or not len(result.tube.get("vertices", ())):
        return None
    selected = view.selected if result is view.extraction else -1
    key = (id(result), view.colouring, round(view.heat_mm, 3), selected, _main_yarn_colour(view))
    if view.tube_key != key:
        count = len(result.tube["vertices"])
        if view.colouring == 1 and "error_mm" in result.tube:
            colours = strand_equations.heat_colours(result.tube["error_mm"], view.heat_mm)
            # The underside was never photographed: grey, not held against the scan.
            colours[~result.tube["seen"]] = (0.35, 0.35, 0.38)
        else:
            colours = np.tile(_main_yarn_colour(view), (count, 1))
        if 0 <= selected < len(result.strands):
            colours[result.tube["strand"] == selected] = SELECTED_COLOUR
        view.tube_splat, view.tube_key = strand_equations.tube_as_splat(result, colours), key
    return view.tube_splat


# ----------------------------------------------------------------------------------------- pictures

# A picture sharpens in three stages: a filled-dot preview at once (stage 0, drawn at half size, a few
# hundredths of a second, so turning and zooming stay smooth), then the proper soft-blob render at half
# size (stage 1, about half a second), then at full size (stage 2, two to six seconds zoomed in).
PREVIEW_STAGE, HALF_STAGE, FULL_STAGE = 0, 1, 2
REST_BEFORE_SHARPENING_S = 0.12


def _render(view, which: str, stage: int, camera=None, cancelled=None):
    azimuth, elevation, zoom, pan = camera or view.camera
    full = stage == FULL_STAGE
    size = RENDER_SIZE if full else (RENDER_SIZE[0] // 2, RENDER_SIZE[1] // 2)
    options = dict(size=size, background=BACKGROUND, pan=pan, frame=view.frame)
    if stage == PREVIEW_STAGE:
        def draw(splat):
            return gaussian_splatting.render_splat_quick(splat, azimuth, elevation, zoom, fill=True, **options)
    else:
        def draw(splat):
            return gaussian_splatting.render_splat(splat, azimuth, elevation, zoom, cancelled=cancelled, **options)
    if which == "scan":
        image = draw(view.splat)
    else:
        tubes = _tube_splat(view)
        if tubes is None:
            return Image.new("RGB", RENDER_SIZE, BACKGROUND)
        image = draw(tubes)
        if which == "overlay" and image is not None:
            scan_image = draw(view.splat)
            if scan_image is None:
                return None
            scan, over = np.asarray(scan_image, dtype=np.float32), np.asarray(image, dtype=np.float32)
            covered = (np.abs(over - np.array(BACKGROUND)).max(axis=2) > 6)[..., None]
            alpha = float(view.overlay)
            image = Image.fromarray(np.where(covered, scan * (1.0 - alpha) + over * alpha, scan).clip(0, 255).astype(np.uint8), "RGB")
    if image is None:
        return None
    return image if full else image.resize(RENDER_SIZE, Image.BILINEAR)


def _picture(view, ctx, which: str):
    """One picture's texture, sharpened stage by stage (above) while the camera rests. A sharpening
    render still going when the camera moves is cancelled, so the next view never waits for it."""
    slot = view.pictures.setdefault(which, SimpleNamespace(texture=None, size=RENDER_SIZE, shown=None, stage=PREVIEW_STAGE,
                                                           worker=None, result=None, rested=0.0))
    wanted = (view.camera, view.tube_key, round(view.overlay, 2) if which == "overlay" else None)
    now = time.monotonic()
    if slot.texture is None or slot.shown != wanted:
        image = _render(view, which, PREVIEW_STAGE)
        slot.texture = upload_rgb_texture(ctx, slot.texture, np.asarray(image))
        slot.size, slot.shown, slot.rested, slot.stage = image.size, wanted, now, PREVIEW_STAGE
    if slot.worker is not None and not slot.worker.is_alive():
        result, slot.worker = slot.result, None
        if result is not None and result[0] == slot.shown and result[2] is not None:
            slot.texture = upload_rgb_texture(ctx, slot.texture, np.asarray(result[2]))
            slot.stage = result[1]
    if slot.worker is None and slot.stage < FULL_STAGE and now - slot.rested > REST_BEFORE_SHARPENING_S:
        stage, camera = slot.stage + 1, view.camera

        def work(wanted=wanted, stage=stage, camera=camera):
            try:
                image = _render(view, which, stage, camera, cancelled=lambda: slot.shown != wanted)
            except Exception as exc:          # shown in the panel; the preview stays up
                view.status, image = f"Could not render the {which} picture: {exc}", None
            slot.result = (wanted, stage, image)

        slot.result = None
        slot.worker = threading.Thread(target=work, daemon=True, name=f"model-render-{which}")
        slot.worker.start()
    return slot


def _model_caption(view) -> str:
    model = _equation_model(view)
    if model is view.crochet:
        return "Crochet model, from the stitch equations"
    if model is view.periodic:
        return "Knit repeating loops"
    return "Rebuilt from the traced equations"


def _gpu(view, ctx):
    """The GPU splat renderer, made on first use; None where this GL cannot run it (the CPU draws)."""
    if view.gl is None and not view.gl_failed:
        try:
            view.gl = splat_gl.SplatRenderer(ctx)
        except Exception as exc:
            view.gl_failed = True
            view.status = f"Drawing the pictures on the CPU: the GPU renderer could not start ({exc})"
    return view.gl


def _gpu_picture(view, gl, which: str, size):
    """One picture drawn on the GPU, again only when the camera, the yarn or the size has changed."""
    def colouring(splat):
        return (lambda: splat_gl.height_colours(splat["positions"], view.plane)) if view.by_height else None

    layers = []
    if which in ("scan", "overlay"):
        layers.append((("scan", view.index, view.by_height, view.fabric_version), view.splat, colouring(view.splat), 1.0))
    if which in ("equations", "overlay"):
        tubes = _tube_splat(view)
        if tubes is not None:
            layers.append((("tubes", view.tube_key, view.by_height), tubes, colouring(tubes),
                           1.0 if which == "equations" else float(view.overlay)))
    slot = view.pictures.setdefault(which, SimpleNamespace(texture=None, shown=None))
    wanted = (view.camera, size, view.style, tuple((key, strength) for key, _, _, strength in layers))
    if slot.shown != wanted or getattr(slot, "gpu", None) is None:
        slot.gpu = gl.render(layers, view.camera, view.frame, size, BACKGROUND, target=which, style=view.style)
        slot.shown = wanted
    view.gl_used.update(key for key, *_ in layers)
    return slot.gpu


def _draw_picture(view, ctx, which: str, width: float, caption: str):
    imgui.begin_group()
    gl = _gpu(view, ctx)
    if gl is not None:
        size = (int(width), int(min(width * 0.75, 620.0)))
        try:
            texture = _gpu_picture(view, gl, which, size)
        except Exception as exc:          # the CPU pictures from here on
            view.gl, view.gl_failed, view.pictures = None, True, {}
            view.status = f"Drawing the pictures on the CPU: the GPU renderer failed ({exc})"
        else:
            imgui.text_colored(COLOR_MUTED, caption)
            rect = draw_fitted_texture(texture.glo, size[0], size[1], width, size[1], flip_y=True)
            imgui.end_group()
            return rect
    slot = _picture(view, ctx, which)
    imgui.text_colored(COLOR_MUTED, caption)
    if slot.stage < FULL_STAGE:
        imgui.same_line()
        imgui.text_colored(COLOR_WARN, "  sharpening..." if slot.stage == PREVIEW_STAGE else "  sharpening (half size shown)...")
    w, h = slot.size
    rect = draw_fitted_texture(slot.texture.glo, w, h, width, min(width * h / w, 620.0), flip_y=True)
    imgui.end_group()
    return rect


def _pick(view, rect, points) -> np.ndarray:
    """The Gaussians in front at screen `points` (x, y) on the Scan picture `rect`, as drawn from the
    current camera (splat_gl's id render). Returns their indices (repeats removed)."""
    x0, y0, w, h = rect
    size = (int(round(w)), int(round(h)))
    azimuth, elevation, zoom, pan = view.camera
    eye, forward, right, up = gaussian_splatting.orbit_camera(None, azimuth, elevation, zoom, pan, view.frame)
    focal = 0.5 * size[0] / np.tan(np.radians(splat_gl.HALF_FOV_DEG))
    ids = view.gl.gaussian_ids(("ids", view.index), view.splat, (eye, forward, right, up, focal), size, view.frame)
    view.gl_used.add(("ids", view.index))
    found = []
    for x, y in points:
        px, py = int(x - x0), int(y - y0)
        window = ids[max(py - 3, 0):py + 4, max(px - 3, 0):px + 4]
        found.extend(window[window >= 0].ravel().tolist())
    return np.unique(np.asarray(found, dtype=int))


def _yarn_tool(view, rect, io) -> bool:
    """Clicking a yarn or brushing along one, on the Scan picture; True when the mouse was used for it."""
    if not view.yarn_tool or view.yarns is None or view.gl is None or rect is None:
        return False
    mx, my = float(io.mouse_pos.x), float(io.mouse_pos.y)
    inside = rect[0] <= mx <= rect[0] + rect[2] and rect[1] <= my <= rect[1] + rect[3] and imgui.is_window_hovered()
    if view.yarn_tool == 1:
        if inside and imgui.is_mouse_clicked(imgui.MouseButton_.left):
            hit = _pick(view, rect, [(mx, my)])
            numbers = view.yarns[hit][view.yarns[hit] >= 0]
            if len(numbers):
                number = int(np.bincount(numbers).argmax())
                view.yarn_colours[number] = np.asarray(view.yarn_pen, dtype=np.float64)
                _apply_fabric(view)
                view.status = f"Yarn {number} coloured {fabric_palette.hex_of(view.yarn_pen)}."
        return inside and imgui.is_mouse_down(imgui.MouseButton_.left)
    # The brush: the stroke's points while the button is down, the yarn grown from them on release.
    if inside and imgui.is_mouse_down(imgui.MouseButton_.left):
        view.brush_seeds.extend(_pick(view, rect, [(mx, my)]).tolist())
        return True
    if view.brush_seeds and not imgui.is_mouse_down(imgui.MouseButton_.left):
        if view.yarn_dirs is not None:
            dirs, strength, radius = view.yarn_dirs
            chosen = yarn_tracks.brush(view.splat["positions"], dirs, strength, view.brush_seeds, radius)
            chosen &= view.yarns >= 0                      # yarn only: the table stays out
            view.yarns = view.yarns.copy()
            view.yarns[chosen] = int(view.brush_yarn)
            view.yarns_edited = True
            if view.yarn_pen is not None:
                view.yarn_colours[int(view.brush_yarn)] = np.asarray(view.yarn_pen, dtype=np.float64)
            _apply_fabric(view)
            view.status = f"{int(chosen.sum())} Gaussians brushed into yarn {int(view.brush_yarn)}."
        view.brush_seeds = []
    return False


def _handle_mouse(view, rects) -> None:
    io = imgui.get_io()
    if view.show in (0, 2) and rects and _yarn_tool(view, rects[0], io):
        return
    mx, my = float(io.mouse_pos.x), float(io.mouse_pos.y)
    over = next((r for r in rects if r is not None and r[0] <= mx <= r[0] + r[2] and r[1] <= my <= r[1] + r[3]), None)
    over = over if imgui.is_window_hovered() else None
    azimuth, elevation, zoom, pan = view.camera
    left, right = imgui.is_mouse_down(imgui.MouseButton_.left), imgui.is_mouse_down(imgui.MouseButton_.right)
    if (over is not None or view.dragging) and (left or right):
        dx, dy = float(io.mouse_delta.x), float(io.mouse_delta.y)
        size = over or next((r for r in rects if r is not None), (0, 0, 480.0, 360.0))
        if left:
            azimuth = ((azimuth - dx * 0.4 + 180.0) % 360.0) - 180.0
            elevation = float(np.clip(elevation + dy * 0.4, -89.0, 89.0))
        else:
            pan = (pan[0] - dx / max(size[2], 1.0) * 2.0 / zoom, pan[1] + dy / max(size[3], 1.0) * 2.0 / zoom)
        view.dragging = True
    elif view.dragging:
        view.dragging = False
    if over is not None and float(io.mouse_wheel) != 0.0:
        zoom = float(np.clip(zoom * np.exp(float(io.mouse_wheel) * 0.2), 0.2, 20.0))
    view.camera = (azimuth, elevation, zoom, pan)


# ----------------------------------------------------------------------------------------- panel

def _draw_model_choice(view) -> None:
    if view.models is None:
        view.models = find_models()
    names = [m["name"] for m in view.models]
    if not names:
        imgui.text_colored(COLOR_WARN, "No trained splats under robot_scans (train one in UR5 Robot Mode).")
        return
    if view.index < 0:
        # The measured, cropped model with the best held-out score.
        best = max(range(len(view.models)), key=lambda i: (view.models[i]["mm_per_unit"] is not None,
                                                          view.models[i]["ply"].name == "splat_clean.ply",
                                                          view.models[i]["psnr"] or 0.0))
        _select_model(view, best)
    imgui.set_next_item_width(-70)
    changed, index = imgui.combo("##model_pick", view.index, names)
    if changed and index != view.index:
        _select_model(view, index)
    imgui.same_line()
    if imgui.button("Refresh##model_refresh"):
        current = view.models[view.index]["ply"] if view.index >= 0 else None
        view.models = find_models()
        view.index = next((i for i, m in enumerate(view.models) if m["ply"] == current), -1)
    model = view.models[view.index]
    if view.loader is not None:
        imgui.text_colored(COLOR_MUTED, f"Reading {model['ply'].name}...")
    elif view.splat is not None:
        imgui.text(f"{view.splat['count']:,} Gaussians" + (f", held-out PSNR {model['psnr']:.1f} dB" if model["psnr"] else ""))
    imgui.set_next_item_width(140)
    changed, value = imgui.input_float("mm per splat unit##model_scale", float(view.mm_per_unit), 0.5, 5.0, "%.2f")
    if changed and value > 0:
        view.mm_per_unit = float(value)
    if imgui.is_item_hovered():
        imgui.set_tooltip("The model's real size. Read from model_scale.json beside the model (set from the ChArUco board);\n"
                          "every length and equation below is in mm through it.")
    if model["mm_per_unit"] is None:
        imgui.text_colored(COLOR_WARN, "No model_scale.json: scale guessed (cameras taken to be 9 cm away).")
        imgui.same_line()
        if imgui.small_button("Save scale##model_scale_save"):
            (model["output"] / strand_equations.SCALE_FILE).write_text(
                json.dumps({"mm_per_unit": view.mm_per_unit, "source": "entered in Model Mode"}, indent=1), encoding="utf-8")
            model["mm_per_unit"] = view.mm_per_unit


def _draw_extraction(view) -> None:
    imgui.set_next_item_width(160)
    _, view.tolerance = imgui.slider_float("Fit tolerance (mm)##model_tol", float(view.tolerance), 0.01, 0.2, "%.3f")
    if imgui.is_item_hovered():
        imgui.set_tooltip("How close each equation must follow its traced strand. Smaller: more terms per equation.")
    imgui.set_next_item_width(160)
    _, view.max_terms = imgui.slider_int("Most terms##model_terms", int(view.max_terms), 1, 16)
    busy = view.extractor is not None or view.splat is None
    if busy:
        imgui.begin_disabled()
    if imgui.button("Extract loop equations##model_extract", (-1, 0)):
        _start_extraction(view)
    if busy:
        imgui.end_disabled()
    if view.extractor is not None:
        imgui.text_colored(COLOR_MUTED, view.progress + "...")
    result = view.extraction
    if result is None:
        imgui.text_wrapped("Traces the yarn's strands in the model, writes each as an equation (x(t), y(t), z(t) in mm), "
                           "and rebuilds the yarn from the equations alone.")
        return
    summary = result.summary()
    imgui.text(f"{summary['strands']} strands, {summary['total_length_mm']:.0f} mm of yarn, radius {summary['median_radius_mm']:.2f} mm")
    imgui.text(f"{summary['coefficients']:,} numbers; median {summary['median_terms']:.0f} sine terms per coordinate")
    if imgui.button("Save equations (JSON)##model_save", ((imgui.get_content_region_avail().x - 8) / 2, 0)):
        target = strand_equations.save(result, view.models[view.index]["output"] / "strand_equations.json")
        view.status = f"Equations saved to {target}"
    imgui.same_line()
    if imgui.button("Save rebuilt yarn (PLY)##model_mesh", (-1, 0)):
        target = strand_equations.save_mesh(result, view.models[view.index]["output"] / "rebuilt_from_equations.ply")
        view.status = f"Rebuilt yarn saved to {target}"


def _draw_comparison(result, table_id: str = "model_compare") -> None:
    if result is None or not result.comparison:
        return
    c = result.comparison
    if imgui.begin_table(table_id, 3, imgui.TableFlags_.borders_inner_h):
        imgui.table_setup_column("")
        imgui.table_setup_column("mean")
        imgui.table_setup_column("95%")
        imgui.table_headers_row()
        for label, mean, p95, tip in (
                ("Rebuilt yarn -> scan", c["rebuilt_to_scan_mean_mm"], c["rebuilt_to_scan_p95_mm"],
                 "How far the equation yarn's upper half lies from the scanned surface: is what the equations make really there?"),
                ("Scan -> rebuilt yarn", c["scan_to_rebuilt_mean_mm"], c["scan_to_rebuilt_p95_mm"],
                 "How far the scanned yarn surface lies from the equation yarn: how much of the scan do the equations explain?")):
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(label)
            if imgui.is_item_hovered():
                imgui.set_tooltip(tip)
            imgui.table_next_column()
            imgui.text(f"{mean:.2f} mm")
            imgui.table_next_column()
            imgui.text(f"{p95:.2f} mm")
        imgui.end_table()
    imgui.text(f"{c['scan_within_0.2mm'] * 100:.0f}% of the scanned yarn surface is within 0.2 mm of the equation yarn")


def _draw_strands(view) -> None:
    result = view.extraction
    if result is None:
        return
    imgui.text_colored(COLOR_MUTED, "Strands, longest first (click to pick; it shows in orange)")
    order = sorted(range(len(result.strands)), key=lambda i: -result.strands[i].length_mm)
    if imgui.begin_child("model_strands", (0, 150), imgui.ChildFlags_.borders):
        for i in order:
            s = result.strands[i]
            label = f"#{i:<4d} {s.length_mm:5.2f} mm  r {s.radius_mm:.2f}  {s.terms:2d} terms  {s.rms_mm:.3f} mm" + ("  loop" if s.closed else "")
            if imgui.selectable(f"{label}##strand{i}", view.selected == i)[0]:
                view.selected = -1 if view.selected == i else i
    imgui.end_child()
    if 0 <= view.selected < len(result.strands):
        text = "\n".join(result.strands[view.selected].equations())
        imgui.input_text_multiline("##model_equation", text, (-1, 70), imgui.InputTextFlags_.read_only)
        if imgui.small_button("Copy##model_eq_copy"):
            imgui.set_clipboard_text(text)
        imgui.same_line()
        imgui.text_colored(COLOR_MUTED, "t from 0 to 1 along the strand; mm; z up")


def _draw_bitmap(bitmap: np.ndarray) -> None:
    """The repeat's tile as Edit Mode shows it: top row first, a filled square a worked stitch."""
    size, gap = 14.0, 2.0
    origin = imgui.get_cursor_screen_pos()
    draw = imgui.get_window_draw_list()
    rows, cols = bitmap.shape
    for r in range(rows):
        for c in range(cols):
            x0 = origin.x + c * (size + gap)
            y0 = origin.y + (rows - 1 - r) * (size + gap)
            on = bitmap[r, c] > 0.5
            colour = imgui.get_color_u32((0.85, 0.85, 0.80, 1.0) if on else (0.20, 0.20, 0.24, 1.0))
            draw.add_rect_filled((x0, y0), (x0 + size, y0 + size), colour)
    imgui.dummy((cols * (size + gap), rows * (size + gap)))


def _draw_periodic(view, state) -> None:
    """The complete repeating loops: fit them, what was found, and the hand-over to Edit Mode."""
    busy = view.periodic_job is not None or view.extraction is None
    if busy:
        imgui.begin_disabled()
    if imgui.button("Fit complete repeating loops##model_periodic", (-1, 0)):
        _start_periodic_fit(view)
    if busy:
        imgui.end_disabled()
    if view.extraction is None:
        imgui.text_colored(COLOR_MUTED, "Extract the loop equations first.")
    if view.periodic_job is not None:
        imgui.text_colored(COLOR_MUTED, view.progress + "...")
    fit = view.periodic
    if fit is None:
        imgui.text_wrapped("The traced strands are only what the pictures showed clearly. A cloth repeats: this finds "
                           "its repeat (stitch, row, openings) and fills in every loop from it, as Edit Mode builds them.")
        return
    imgui.text(f"Repeat: {fit.n_rows} rows x {fit.n_cols} stitches, the rows at {fit.angle_deg:.0f} deg")
    imgui.text(f"Stitch {fit.stitch_width:.2f} mm wide, rows {fit.row_pitch:.2f} mm apart")
    imgui.text(f"Loop {fit.loop_height:.2f} mm tall, bulge {fit.stitch_bulge:.2f}, depth {fit.stitch_z:+.2f} mm, yarn r {fit.radius:.2f} mm")
    # The split of the repeat into stitches and rows can be read wrong (half a stitch looks like a
    # stitch): set it here and fit again.
    imgui.set_next_item_width(90)
    _, cols = imgui.input_int("stitches##model_cols", int(view.counts[0]))
    imgui.same_line()
    imgui.set_next_item_width(90)
    _, rows = imgui.input_int("rows per repeat##model_rows", int(view.counts[1]))
    view.counts = (int(np.clip(cols, 1, 32)), int(np.clip(rows, 1, 8)))
    if view.counts != (fit.n_cols, fit.n_rows):
        imgui.same_line()
        if view.periodic_job is None and imgui.small_button("Refit##model_refit"):
            _start_periodic_fit(view, view.counts)
    _draw_bitmap(fit.bitmap)
    imgui.same_line()
    imgui.begin_group()
    for line in fit.equations():
        imgui.text(line)
    imgui.text_colored(COLOR_MUTED, "one stitch, t from 0 to 2π; mm")
    imgui.end_group()
    _draw_comparison(fit, "model_compare_periodic")
    imgui.push_style_color(imgui.Col_.button, (0.22, 0.48, 0.78, 1.0))
    if imgui.button("Open in Edit Mode##model_to_edit", (-1, 0)):
        # Edit Mode's own state is gui.py's to change; it picks this up after the panel is drawn.
        state.model_to_edit = {"parameters": fit.edit_parameters(), "bitmap": fit.bitmap.copy()}
    imgui.pop_style_color()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Replaces Edit Mode's pattern and loop parameters with these (Undo there brings the old ones back).")


def _draw_view_options(view) -> None:
    if view.periodic is not None or view.crochet is not None:
        imgui.text("Equation yarn:")
        for i, name in enumerate(SOURCES):
            if (i == 1 and view.periodic is None) or (i == 2 and view.crochet is None):
                continue
            imgui.same_line()
            if imgui.radio_button(f"{name}##model_source{i}", view.source == i):
                view.source = i
    for i, name in enumerate(SHOWS):
        if i:
            imgui.same_line()
        if imgui.radio_button(f"{name}##model_show{i}", view.show == i):
            view.show = i
    if view.extraction is None and view.show:
        imgui.same_line()
        imgui.text_colored(COLOR_MUTED, "(extract the equations first)")
    if view.show in (1, 2, 3):
        imgui.set_next_item_width(230)
        _, view.colouring = imgui.combo("Equation yarn colour##model_colour", int(view.colouring), list(COLOURINGS))
        if view.colouring == 1:
            imgui.same_line()
            imgui.set_next_item_width(120)
            _, view.heat_mm = imgui.slider_float("red at (mm)##model_heat", float(view.heat_mm), 0.1, 2.0, "%.2f")
            for label, colour in (("on the scan", (0.15, 0.35, 0.95, 1.0)), (f"{view.heat_mm / 3:.2f} mm", (0.15, 0.80, 0.45, 1.0)),
                                  (f"{2 * view.heat_mm / 3:.2f} mm", (0.95, 0.85, 0.20, 1.0)), (f">= {view.heat_mm:.2f} mm", (0.90, 0.20, 0.15, 1.0)),
                                  ("underside (never seen)", (0.35, 0.35, 0.38, 1.0))):
                imgui.text_colored(colour, "■")
                imgui.same_line()
                imgui.text(label)
                imgui.same_line()
            imgui.new_line()
    if view.show == 3:
        imgui.set_next_item_width(200)
        _, view.overlay = imgui.slider_float("Equation yarn opacity##model_overlay", float(view.overlay), 0.0, 1.0, "%.2f")
    imgui.text_colored(COLOR_MUTED, "Drag to turn, right-drag to move, wheel to zoom")
    imgui.same_line()
    if imgui.small_button("Reset view##model_reset"):
        view.camera = (0.0, 70.0, 1.0, (0.0, 0.0))
    imgui.same_line()
    if imgui.small_button("Top##model_top"):
        view.camera = (view.camera[0], 89.0, view.camera[2], view.camera[3])
    imgui.same_line()
    if imgui.small_button("Edge-on##model_edge"):
        view.camera = (view.camera[0], 3.0, view.camera[2], view.camera[3])
    if imgui.is_item_hovered():
        imgui.set_tooltip("Looks along the cloth: a flat picture would vanish to a line, the model shows its thickness.")
    imgui.same_line()
    _, view.spin = imgui.checkbox("Spin##model_spin", bool(view.spin))
    imgui.same_line()
    _, view.wiggle = imgui.checkbox("Wiggle##model_wiggle", bool(view.wiggle))
    if imgui.is_item_hovered():
        imgui.set_tooltip("Rocks the view a few degrees either way: in a 3D model the near yarn slides\n"
                          "over the far yarn and the table; a flat picture would only skew as one piece.")
    imgui.same_line()
    _, view.by_height = imgui.checkbox("Colour by height##model_height", bool(view.by_height))
    if view.gl is not None:
        imgui.same_line()
        imgui.set_next_item_width(150)
        _, view.style = imgui.combo("Draw##model_style", int(view.style), list(splat_gl.STYLES))
        if imgui.is_item_hovered():
            imgui.set_tooltip("Soft splat: the picture as trained.\n"
                              "Gaussian ellipses: every Gaussian solid out to one standard deviation, rimmed -\n"
                              "its size, stretch and turn. Gaussian centres: only their middles, as dots.\n"
                              "Soft splat + Gaussian outlines: the picture with each Gaussian's outline in it;\n"
                              "turn it and the outlines change shape as flattened 3D ellipsoids do.\n"
                              "Gaussian axes: each Gaussian's longest axis as a 3D line - red along the cloth's x,\n"
                              "green along its y, blue standing up out of it.\n"
                              "Soft splat + Gaussian axes: those axes on the model; the ones behind the cloth stay hidden.")
    if view.plane is not None:
        relief_mm = (view.plane[4] - view.plane[3]) * float(view.mm_per_unit)
        imgui.text_colored(COLOR_MUTED, f"Relief above the cloth's bow: {relief_mm:.2f} mm (2nd to 98th percentile of "
                                        f"{len(view.splat['positions']):,} Gaussians); a flat picture would be 0.")
    if imgui.is_item_hovered():
        imgui.set_tooltip("Each Gaussian coloured by how far it stands above the cloth's smooth bow:\n"
                          "blue low, red high. The yarn's crossings and the openings' rims show as relief.")


def draw_model_panel(state, renderer) -> None:
    """Model Mode's whole window: controls on the left, the pictures on the right."""
    view = _view(state)
    _poll_loader(view)
    _poll_extractor(view)
    _poll_periodic(view)
    _poll_crochet(view)
    _poll_fabric_job(view)
    _poll_yarns_job(view)
    if imgui.begin_child("model_controls", (CONTROLS_WIDTH, 0)):
        if imgui.collapsing_header("Model##model_section", imgui.TreeNodeFlags_.default_open):
            _draw_model_choice(view)
        if imgui.collapsing_header("Fabric colour##model_fabric_section", imgui.TreeNodeFlags_.default_open):
            _draw_fabric_colours(view)
        if imgui.collapsing_header("Loop equations##model_eq_section", imgui.TreeNodeFlags_.default_open):
            _draw_extraction(view)
            _draw_comparison(view.extraction)
            _draw_strands(view)
        if imgui.collapsing_header("Crochet model (posts and chains)##model_crochet_section", imgui.TreeNodeFlags_.default_open):
            _draw_crochet(view, state)
        if imgui.collapsing_header("Knit repeating loops##model_periodic_section"):
            _draw_periodic(view, state)
        if view.status:
            imgui.separator()
            imgui.text_wrapped(view.status)
    imgui.end_child()
    imgui.same_line()
    if imgui.begin_child("model_pictures", (0, 0)):
        _draw_view_options(view)
        if view.spin and not view.dragging:
            azimuth, elevation, zoom, pan = view.camera
            azimuth = ((azimuth + SPIN_DEG_PER_S * float(imgui.get_io().delta_time) + 180.0) % 360.0) - 180.0
            view.camera = (azimuth, elevation, zoom, pan)
        if view.wiggle and not view.dragging:
            offset = WIGGLE_DEG * float(np.sin(2.0 * np.pi * WIGGLE_HZ * time.monotonic()))
            azimuth, elevation, zoom, pan = view.camera
            view.camera = (azimuth + offset - view.wiggle_offset, elevation, zoom, pan)
            view.wiggle_offset = offset
        if view.splat is None or view.frame is None:
            imgui.text_colored(COLOR_MUTED, "Loading the model..." if view.loader is not None else "No model loaded.")
        else:
            ctx = renderer.ctx
            avail = max(200.0, imgui.get_content_region_avail().x)
            have_tubes = view.extraction is not None
            if view.show == 2 and have_tubes:
                half = (avail - imgui.get_style().item_spacing.x) * 0.5
                rects = [_draw_picture(view, ctx, "scan", half, "Scan")]
                imgui.same_line()
                rects.append(_draw_picture(view, ctx, "equations", half, _model_caption(view)))
            elif view.show == 1 and have_tubes:
                rects = [_draw_picture(view, ctx, "equations", avail, _model_caption(view))]
            elif view.show == 3 and have_tubes:
                rects = [_draw_picture(view, ctx, "overlay", avail, "Equation yarn over the scan")]
            else:
                rects = [_draw_picture(view, ctx, "scan", avail, "Scan")]
            _handle_mouse(view, rects)
    imgui.end_child()
    if view.gl is not None:               # splats no picture drew this frame are let go
        view.gl.forget(keep=view.gl_used)
        view.gl_used = set()
