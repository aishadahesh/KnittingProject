"""Each yarn of a model's fabric as its own set of Gaussians, saved beside it.

    .venv\\Scripts\\python.exe scripts\\fabric_yarns.py <splat_output> [--from fit|directions]

Two ways to find them:
  fit         from the fitted crochet model (src/fit_yarns.py: each yarn Gaussian takes its nearest
              fitted curve, a row's curves being one yarn) - the default when the model has a photo fit
              (stitch_fit/stitch_model.json), or one per part of a cloth worked in two structures
              (stitch_fit_region_<group>/, fit_stitch_model.py --region): each part's Gaussians take
              their own fit's curves, and the parts' yarns are numbered one after the other;
  directions  from the Gaussians alone (src/yarn_tracks.py: linked where close and pointing alike).

Needs the fabric colours first (scripts/fabric_colours.py): its groups say which Gaussians are yarn
(and the openings' rims) rather than the table, and its photo colours give each yarn's colour.
Writes <splat_output>/fabric_yarns.npy (each Gaussian's yarn, -1 for none, in splat_clean.ply's
order) and fabric_yarns.json (per yarn: Gaussians, span in mm, middle, direction, photo colour,
colour group; and the yarn radius found).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import fabric_palette  # noqa: E402
import gaussian_splatting  # noqa: E402
import fit_yarns  # noqa: E402
import strand_equations  # noqa: E402
import yarn_tracks  # noqa: E402

YARNS_FILE, YARNS_JSON = "fabric_yarns.npy", "fabric_yarns.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="the splat_output folder (export/splat_clean.ply, fabric_colours.json inside)")
    parser.add_argument("--from", dest="source", choices=("fit", "directions"), default=None,
                        help="how to find the yarns (default: fit when the model has stitch_fit/stitch_model.json)")
    args = parser.parse_args()
    fit_path = args.run / "stitch_fit" / "stitch_model.json"
    regions = sorted(args.run.glob("stitch_fit_region_*/stitch_model.json"), key=lambda p: int(p.parent.name.rsplit("_", 1)[1]))
    fits = regions or ([fit_path] if fit_path.is_file() else [])
    source = args.source or ("fit" if fits else "directions")
    if source == "fit" and not fits:
        raise SystemExit("no photo fit (stitch_fit/ or stitch_fit_region_*/) for --from fit")

    splat = gaussian_splatting.load_splat_ply(args.run / "export" / "splat_clean.ply")
    fabric = fabric_palette.load(args.run, len(splat["positions"]))
    if fabric is None or fabric["labels"] is None:
        raise SystemExit("extract the fabric colours first (scripts/fabric_colours.py)")
    mm = strand_equations.load_scale(args.run)
    if mm is None:
        raise SystemExit("no model_scale.json: the yarn's size in mm is needed")
    yarn_groups = [g["index"] for g in fabric["groups"] if g["role"] in ("yarn", "edge")]
    candidates = np.isin(fabric["labels"], yarn_groups) & (np.asarray(splat["opacity"]) >= 0.3)
    if source == "fit":
        yarn = np.full(len(candidates), -1)
        curve = np.full(len(candidates), -1)
        distance_mm = np.full(len(candidates), np.inf)
        roles = [g["role"] for g in sorted(fabric["groups"], key=lambda g: g["index"])]
        count, radii, parts = 0, [], []
        for path in fits:
            fit = json.loads(path.read_text(encoding="utf-8"))
            prepared = fit_yarns.fit_frame(path)
            if prepared is None:                       # saved before fits kept their frame: the whole cloth's
                import gui_model
                import scan_fit

                model = next(m for m in gui_model.find_models() if m["output"].resolve() == args.run.resolve())
                model["mm_per_unit"] = mm
                prepared = scan_fit.prepare(model)
            mine = candidates.copy()
            if "region" in fit:
                mine &= fit_yarns.region_members(splat["positions"], fabric["labels"], roles, fit["region"]["group"])
            curves = fit_yarns.fitted_curves(path)
            y, c, d = fit_yarns.yarn_of(splat["positions"], curves, prepared, mine)
            yarn[mine], curve[mine], distance_mm[mine] = y[mine] + count, c[mine] + sum(parts), d[mine]
            count = int(yarn.max()) + 1
            parts.append(len(curves))
            radii.append(float(fit["yarn_radius"]))
            log_part = fit.get("region", {}).get("name", "the whole cloth")
            print(f"{log_part}: {int(mine.sum())} yarn Gaussians onto {len(curves)} curves, median {np.median(d[mine]):.2f} mm")
        candidates = yarn >= 0
        radius_mm = float(np.mean(radii))
        extra = {"source": "fit", "fits": [p.parent.name for p in fits], "curves": int(sum(parts)),
                 "median_distance_mm": round(float(np.median(distance_mm[candidates])), 3),
                 "within_1mm": round(float((distance_mm[candidates] <= 1.0).mean()), 3),
                 "within_2mm": round(float((distance_mm[candidates] <= 2.0).mean()), 3)}
        np.save(args.run / "fabric_yarn_curves.npy", curve.astype(np.int32))
        found = {"radius": radius_mm / mm, "pieces": np.where(candidates, 0, -1)}
    else:
        found = yarn_tracks.find_yarns(splat["positions"], splat["scales"], splat["quaternion"], splat["opacity"], candidates, mm)
        yarn = found["yarn"]
        extra = {"source": "directions"}
    np.save(args.run / YARNS_FILE, yarn.astype(np.int32))

    photo_path = args.run / "fabric_photo_colours.npy"
    photo = np.load(photo_path) if photo_path.is_file() else None
    positions = np.asarray(splat["positions"], dtype=np.float64)
    yarns = []
    for k in [k for k in np.unique(yarn) if k >= 0]:
        mine = yarn == k
        middle = positions[mine].mean(axis=0)
        _, _, vt = np.linalg.svd(positions[mine] - middle, full_matrices=False)
        span = float(np.ptp((positions[mine] - middle) @ vt[0]) * mm)
        entry = {"yarn": int(k), "gaussians": int(mine.sum()), "span_mm": round(span, 2),
                 "middle": [round(float(v), 5) for v in middle], "direction": [round(float(v), 4) for v in vt[0]],
                 "colour_group": int(np.bincount(fabric["labels"][mine][fabric["labels"][mine] >= 0]).argmax())}
        if photo is not None:
            known = mine & ~np.isnan(photo[:, 0])
            if known.sum() >= 5:
                entry["photo_colour"] = [round(float(v), 4) for v in fabric_palette.linear_to_srgb(
                    np.median(fabric_palette.srgb_to_linear(photo[known]), axis=0))]
        yarns.append(entry)
    report = {**extra, "radius_mm": round(found["radius"] * mm, 3), "yarns": yarns,
              "candidates": int(candidates.sum()), "linked": int((found["pieces"] >= 0).sum()),
              "pieces": int(found["pieces"].max() + 1) if (found["pieces"] >= 0).any() else 0}
    (args.run / YARNS_JSON).write_text(json.dumps(report, indent=1), encoding="utf-8")
    spans = sorted((y["span_mm"] for y in yarns), reverse=True)
    if source == "fit":
        print(f"from the fit: {extra['curves']} curves, {len(yarns)} yarns (rows); yarn Gaussians to their curve: median "
              f"{extra['median_distance_mm']} mm, {extra['within_1mm']:.0%} within 1 mm, {extra['within_2mm']:.0%} within 2 mm")
    else:
        print(f"yarn radius {report['radius_mm']} mm; {report['linked']} of {report['candidates']} yarn Gaussians linked into "
              f"{report['pieces']} pieces, joined into {len(yarns)} yarns")
    print("longest spans (mm):", spans[:12], "| yarns under 2 mm:", sum(s < 2 for s in spans))


if __name__ == "__main__":
    main()
