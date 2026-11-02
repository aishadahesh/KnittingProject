# Scanning the fabric over a ChArUco board

The splat needs every picture's camera pose to about one pixel. The arm with the mount calibration
managed ~11 px (40 px worst) and broke whenever the pendant's tool-centre setting changed; COLMAP
cannot tell one stitch from the next. A printed ChArUco board under the cloth fixes both: every
picture that sees part of the board gets its pose from it directly, and the rest are placed through
the arm with a mount fitted to that same scan's board views (`src/charuco.py`).

## One-time setup

1. **Print the board**: `hardware/charuco_board.pdf` (A4, 47 x 33 squares of 6 mm), already
   generated with `uv run python scripts\make_charuco_board.py --squares-x 47 --squares-y 33 --square-mm 6 --marker-mm 4.5`,
   which also wrote `config/charuco_board.json`. 6 mm squares are seen from 7 cm (15+ corners) to
   18 cm (100-230 corners) - the whole range a dome scan flies.
   Print at **100 % / actual size** (no "fit to page"), on matte paper.
2. **Mount it flat**: glue it to stiff card or foam board. A wavy print is a wrong board.
3. **Measure it**: measure 10 squares with a ruler, divide by 10, and record it:
   `uv run python scripts\make_charuco_board.py --measured-square-mm 5.98`

## Before each scan

1. **Fix the focus**: in the UR5 panel, *Find best focus* at the scan height (turns autofocus off).
   Autofocus changes the lens between pictures, and no single lens model fits them all.
2. **Lens**: nothing to do if the scan sees the board in 12 or more pictures (it calibrates the
   lens itself, at exactly that focus). Otherwise calibrate once per focus setting:
   `uv run python scripts\calibrate_lens.py --camera 0 --charuco` (move the board around in front
   of the camera; 20 pictures).
3. **Lay out**: cloth in the middle of the board, with board showing on every side. Don't touch
   the cloth or the board until the scan is done.

## The scan

- **One session per splat.** The export refuses pictures from several scans: the cloth will have
  moved between them.
- **The dome scan** (`scripts\freedrive_splat_scan.py`: freedrive, lift, plan, run) is the one to use:
  free-drive the camera over the cloth near its edge, so the board shows in part of the picture, about
  7-10 cm away. Views that miss the board get their pose through the arm.
- **Variety**: tilted views (15-25 deg) from several directions. They give the splat depth, and the
  per-session mount fit needs the arm in different attitudes.
- **Overlap**: neighbouring pictures should share about 70 % of their view.

## Export and train

Under *4 Results*, a session whose pictures show the board is detected automatically. The steps
read *Calibrate mount: not needed: the ChArUco board places the cameras*, and *Export the dataset
with poses from the ChArUco board*. Then train as before.

Check `splat_dataset/transforms.json` after export:

| Field | Good |
|---|---|
| `board_pose_rms_px` | under 0.5 |
| `mount_fit.rms_px` (only if some pictures miss the board) | under 2 |
| `lens.reprojection_error_px` | under 1 |
| `pose_from` per frame | `board` or `arm` |

The status line lists anything left out, and why (e.g. the board seen too poorly, or no arm pose).

## When it goes wrong

- **"the board is in none of the pictures"**: the cloth covers it, or the scan was too close.
  Scan higher, or move the cloth so more border shows.
- **"the lens is not measured"**: too few pictures saw much of the board. Run
  `calibrate_lens.py --charuco` at the scan's focus.
- **Mount fit over 2 px**: the arm's poses don't explain the board views. Was the board or cloth
  touched during the scan? Did the focus change?
- **Pictures mirrored**: detected and handled (exported flipped). A mirrored camera driver is fine
  as long as it stays that way for the whole scan.
