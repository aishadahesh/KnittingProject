# Rescan of one lace repeat, for fitting the stitch equations

The photo fit (`scripts/fit_stitch_model.py`) needs to see where every post and every opening is. The
Oct 29 scan was taken for a splat: views from far around, the lace 9-15 cm away, many at a slant.
At 0.1 mm per pixel a post is 7-10 pixels wide, and from a slant the yarn in front hides the
openings behind it. This scan is only for the stitches: one or two repeats (about 2 x 3 cm), looked
at square-on and slightly from the side, in soft light.

## Before (you)
1. **Count on the lace**, between two neighbouring openings in the same row: how many posts, and how
   many chains across the opening. Then count the rows from one row of openings to the next. Write
   down the stitch too (double crochet = one diagonal wrap per post).
2. **Light:** two lamps either side, or daylight through a sheet. No spot light; nothing shiny.
3. **Lay the lace flat, gently stretched**, with the printed board (6 mm squares) beside it and
   partly under it, as on Oct 29. Pick the repeat to scan near the middle of the lace, and mark it
   with a pin at its corner if you like.
4. **App:** robot disconnected, camera preview off. **Pendant:** Remote. Close the splat viewer and
   Chrome (memory).

## Robot (me, with you there)
See `docs/ROBOT_CONNECTION.md` for the network: TP-Link_CD91, 192.168.0.101.

1. **Hold the camera above the repeat:**
   `uv run python scripts\freedrive_splat_scan.py freedrive`
   - Move the camera by hand to about 9 cm above the repeat, square-on, with the board in view
     (the preview counts the board's corners).
   - Press **H** to hold.
   - I check `robot_scans\_freedrive\held_view.png`.
2. **If you held it lower than about 5 cm:**
   `uv run python scripts\freedrive_splat_scan.py lift`
3. **Plan a tight dome**, square-on plus two tilted rings, all at the camera's sharp distance:
   `uv run python scripts\freedrive_splat_scan.py plan --layers-at 0.085,0.09 --rings 3 --per-ring 16 --max-tilt 25`
   - That is about 80 views.
   - Every move is solved and checked: at least 30 mm over the cloth, the arm at least 100 mm clear.
   - I show you the plan summary. **The arm moves only after you say go.**
4. **Run:**
   `uv run python scripts\freedrive_splat_scan.py run`
   - About 10-15 minutes. Say "stop" at any time.
   - Afterwards the arm goes home and is disconnected.

## After (me)
1. **Poses:** `uv run python scripts\solve_poses.py <session> --fov 35 --refine-focal` (COLMAP), with
   the board for scale (`scripts\export_board_dataset.py`, giving `model_scale.json`).
2. **Splat:** `scripts\train_splat.ps1 -Detail -Iterations 30000`, then `crop_splat.py` to the lace.
   The splat is only used for the cloth's outline and the start.
3. **Stitch fit:**
   `fit_stitch_model.py <output> --stitches <your count> --rows <your count> --try ...`
   - It reads the openings from the photos, then fits the lattice, the bow and the stitch's
     learned curves to the photos.
   - Judged on held-out photos, beside the free splat.
4. **Edit Mode:** Fabric: Crochet → Fit to a scan → pick the new scan → **Fit to the photos**.
