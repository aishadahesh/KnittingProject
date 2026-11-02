"""Print the ChArUco board the cloth lies on for a splat scan, and record which board it is.

    uv run python scripts\\make_charuco_board.py                      (A4, 10 mm squares)
    uv run python scripts\\make_charuco_board.py --measured-square-mm 9.94   (after printing: what a square really measures)

Writes hardware/charuco_board.pdf (and a .png) and config/charuco_board.json. Print the PDF at 100 %
("actual size", no "fit to page"), glue it flat to card or foam board - a wavy print is a wrong
board - and lay the cloth in the middle with the border showing round it. Then measure ten squares
with a ruler and run again with --measured-square-mm: the board is the ruler of the whole scene.
(A wrong square size only scales the splat; a non-flat print breaks it.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import charuco  # noqa: E402

PX_PER_MM = 12.0     # 304.8 dpi: a 10 mm square is exactly 120 pixels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--squares-x", type=int, default=charuco.DEFAULT_BOARD["squares_x"])
    parser.add_argument("--squares-y", type=int, default=charuco.DEFAULT_BOARD["squares_y"])
    parser.add_argument("--square-mm", type=float, default=charuco.DEFAULT_BOARD["square_mm"])
    parser.add_argument("--marker-mm", type=float, default=charuco.DEFAULT_BOARD["marker_mm"])
    parser.add_argument("--dictionary", default=charuco.DEFAULT_BOARD["dictionary"])
    parser.add_argument("--page-mm", type=float, nargs=2, default=(297.0, 210.0), help="page width and height (A4 landscape)")
    parser.add_argument("--measured-square-mm", type=float, default=None,
                        help="after printing: a square's real size - only config/charuco_board.json is updated")
    parser.add_argument("--output", default=str(ROOT / "hardware" / "charuco_board.pdf"))
    args = parser.parse_args()

    if args.measured_square_mm is not None:
        spec = charuco.load_board_spec()
        if spec is None:
            raise SystemExit("no board printed yet: run without --measured-square-mm first")
        ratio = args.measured_square_mm / float(spec["square_mm"])
        spec.update(square_mm=args.measured_square_mm, marker_mm=float(spec["marker_mm"]) * ratio, printed_scale=ratio)
        charuco.save_board_spec(spec)
        print(f"Board squares set to {args.measured_square_mm:.3f} mm (the print came out at {ratio * 100:.2f} %).")
        return

    spec = {"squares_x": args.squares_x, "squares_y": args.squares_y, "square_mm": args.square_mm,
            "marker_mm": args.marker_mm, "dictionary": args.dictionary}
    if spec["squares_x"] * spec["square_mm"] > args.page_mm[0] - 10 or spec["squares_y"] * spec["square_mm"] > args.page_mm[1] - 10:
        raise SystemExit("the board does not fit the page with 5 mm margins: fewer or smaller squares")
    page = charuco.board_image(spec, PX_PER_MM, args.page_mm)

    from PIL import Image

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    picture = Image.fromarray(page)
    dpi = PX_PER_MM * 25.4
    picture.save(output.with_suffix(".png"), dpi=(dpi, dpi))
    # One bit per pixel: stored losslessly (a JPEG-compressed board would have soft, shifted edges).
    picture.point(lambda v: 255 if v >= 128 else 0).convert("1").save(output.with_suffix(".pdf"), resolution=dpi)
    charuco.save_board_spec(spec)
    print(f"Board: {spec['squares_x']} x {spec['squares_y']} squares of {spec['square_mm']} mm, {spec['dictionary']}")
    print(f"Print {output.with_suffix('.pdf')} at 100 % (actual size), glue it flat, then measure a square and run with --measured-square-mm.")
    print(f"Board recorded in {charuco.BOARD_JSON}")


if __name__ == "__main__":
    main()
