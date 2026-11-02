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
The geometry is strand_equations.py and lace_lattice.py; the splat loading and CPU renderer are
gaussian_splatting.py.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from imgui_bundle import imgui
from PIL import Image

import crochet_core
import crochet_fit
import gaussian_splatting
import lace_lattice
import paths
import strand_equations
from rendering import draw_fitted_texture, upload_rgb_texture

COLOR_OK = (0.25, 0.80, 0.40, 1.0)
COLOR_WARN = (0.95, 0.75, 0.20, 1.0)
COLOR_MUTED = (0.65, 0.65, 0.65, 1.0)
RENDER_SIZE = (560, 420)
BACKGROUND = (18, 18, 24)
SHOWS = ("Scan", "Equations", "Side by side", "Overlay")
COLOURINGS = ("Yarn white", "Distance to scan (heat map)")
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
        )
        state.model_view = view
    return view


# ----------------------------------------------------------------------------------------- loading

def _select_model(view, index: int) -> None:
    model = view.models[index]
    view.index, view.splat, view.frame, view.extraction, view.selected = index, None, None, None, -1
    view.periodic, view.crochet, view.source = None, None, 0
    view.tube_splat, view.tube_key, view.pictures = None, None, {}
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
    key = (id(result), view.colouring, round(view.heat_mm, 3), selected)
    if view.tube_key != key:
        count = len(result.tube["vertices"])
        if view.colouring == 1 and "error_mm" in result.tube:
            colours = strand_equations.heat_colours(result.tube["error_mm"], view.heat_mm)
            # The underside was never photographed: grey, not held against the scan.
            colours[~result.tube["seen"]] = (0.35, 0.35, 0.38)
        else:
            colours = np.tile((0.92, 0.92, 0.88), (count, 1))
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


def _draw_picture(view, ctx, which: str, width: float, caption: str):
    imgui.begin_group()
    slot = _picture(view, ctx, which)
    imgui.text_colored(COLOR_MUTED, caption)
    if slot.stage < FULL_STAGE:
        imgui.same_line()
        imgui.text_colored(COLOR_WARN, "  sharpening..." if slot.stage == PREVIEW_STAGE else "  sharpening (half size shown)...")
    w, h = slot.size
    rect = draw_fitted_texture(slot.texture.glo, w, h, width, min(width * h / w, 620.0), flip_y=True)
    imgui.end_group()
    return rect


def _handle_mouse(view, rects) -> None:
    io = imgui.get_io()
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


def draw_model_panel(state, renderer) -> None:
    """Model Mode's whole window: controls on the left, the pictures on the right."""
    view = _view(state)
    _poll_loader(view)
    _poll_extractor(view)
    _poll_periodic(view)
    _poll_crochet(view)
    if imgui.begin_child("model_controls", (CONTROLS_WIDTH, 0)):
        if imgui.collapsing_header("Model##model_section", imgui.TreeNodeFlags_.default_open):
            _draw_model_choice(view)
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
