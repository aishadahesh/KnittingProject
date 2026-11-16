"""Find the fabric's 40 numbered parts in the carton photo and write config/fabric_parts.json.

    uv run --extra camera python scripts\\map_fabric_parts.py
    uv run --extra camera python scripts\\map_fabric_parts.py --preview robot_scans\\fabric_parts.png

The preview is the photo with every part outlined and numbered, to check against the knitted numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import fabric_parts  # noqa: E402
import paths  # noqa: E402
import workcell  # noqa: E402
from persistence import write_json  # noqa: E402

PREVIEW_SCALE = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preview", type=Path, default=ROOT / "robot_scans" / "fabric_parts.png")
    args = parser.parse_args()

    cell = workcell.cell()
    carton = cell["carton"]
    photo = cv2.imread(str(paths.WORKCELL_CARTON_JPG))
    res = (carton["x"][1] - carton["x"][0]) / photo.shape[1]
    lab = cv2.cvtColor(cv2.GaussianBlur(photo, (0, 0), 3), cv2.COLOR_BGR2LAB).astype(float)
    parts = fabric_parts.find_parts(lab, res, carton["x"][0], carton["y"][1], cell["fabric"])
    for part in parts:
        x, y = float(np.mean(part["x"])), float(np.mean(part["y"]))
        part["centre"] = [x, y, workcell.surface_z(carton["top_z_at_origin"], x, y)]
    write_json(paths.FABRIC_PARTS_JSON, {"frame": cell["frame"], "photo": paths.WORKCELL_CARTON_JPG.name, "parts": parts})

    preview = cv2.resize(photo, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE, interpolation=cv2.INTER_AREA)
    scale = PREVIEW_SCALE / res
    for part in parts:
        (x0, x1), (y0, y1) = part["x"], part["y"]
        corner = lambda x, y: (int((x - carton["x"][0]) * scale), int((carton["y"][1] - y) * scale))  # noqa: E731
        cv2.rectangle(preview, corner(x0, y1), corner(x1, y0), (0, 255, 255), 2)
        cx, cy = corner(*part["centre"][:2])
        cv2.circle(preview, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(preview, f"{part['number']:02d}", (cx - 22, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4)
        cv2.putText(preview, f"{part['number']:02d}", (cx - 22, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.preview), preview)
    print(f"{len(parts)} parts -> {paths.FABRIC_PARTS_JSON}; preview {args.preview}")


if __name__ == "__main__":
    main()
