# Plan: a Gaussian splat that looks like the real fabric

Status 2026-10-26: diagnosis done, ChArUco pose source built and tested on synthetic data.
Next is printing the board and a new scan.

## 1. Why the current fabric splats fail

The fabric splats score 11-12 dB PSNR on held-out pictures. Even the pictures they were trained on
render as fog (median 16 dB, as low as 5 dB). The model cannot reproduce its own input, so the problem
is in the data, not in the training settings.

| # | Problem | Evidence | Effect |
|---|---|---|---|
| 1 | Datasets mix scans from different days | `combined_fabric_oct18` holds Oct 8 and Oct 18 sessions | The cloth moved between them: no single scene fits all pictures |
| 2 | The pendant's tool-centre (TCP) setting changed between days | Reported minus planned TCP: **-108.5 mm** on Oct 18, about 0 mm on Oct 8 and Oct 19 | A mount calibration from one day places the other day's cameras centimetres off |
| 3 | The camera image is mirrored on some days | Picture matching shows Oct 18 mirrored, Oct 8 not | The camera mount or driver changed between days |
| 4 | The camera poses are not accurate enough | Mount calibration error 11 px RMS, 40 px worst | A stitch is 20-30 px, so neighbouring stitches blur into grey. Splatting needs about 1 px |
| 5 | The lens was never measured | No `config/camera_intrinsics.json`; focal length guessed differently per export (929 / 1194 px), COLMAP found 1020 px and real distortion | Every ray is a little wrong, worst at the picture edges |
| 6 | Autofocus is on | `ur5_camera_autofocus: true` in `config/params.json` | Focal length changes between pictures, so no single lens model fits them all |
| 7 | COLMAP can't recover the poses | 11.6 dB with COLMAP poses | Every stitch looks alike, so features match to the wrong stitch |

Problems 2 and 4 come from relying on the arm and a mount calibration. Problem 7 comes from relying
on the pictures. The plan therefore takes the poses from a third source: a printed board under the cloth.

## 2. The approach: a ChArUco board as the reference

A ChArUco board is a checkerboard with a uniquely numbered marker in every white square. Even the
strip of board showing round the cloth tells exactly which corners are visible, and because the board
is flat with known geometry, each picture that sees it gets its camera pose to a fraction of a pixel.

- **Pictures that see the board** get their pose from it directly.
- **Pictures that don't see the board** (close-ups of the cloth's middle) are placed through the arm.
  The mount is fitted per scan from that scan's own board views, so the pendant's TCP setting cancels out.
- **The lens** is calibrated from the scan's own board views when there are 12 or more. It is then
  measured at exactly the focus the scan used.
- **Mirrored pictures** are detected and exported flipped.
- **One scan per splat**: the export refuses pictures from several sessions.

The board becomes the world frame: metres, z up out of the paper.

## 3. Done (in code, not committed)

| Piece | File | Status |
|---|---|---|
| Board detection, per-picture pose, lens calibration, per-scan mount fit, dataset export | `src/charuco.py` | Done; 8 tests pass on synthetic board views |
| Printable board + board record | `scripts/make_charuco_board.py` → `hardware/charuco_board.pdf`, `config/charuco_board.json` | Done, generated (A4, 47 × 33 squares of 6 mm, seen from 7 to 18 cm) |
| Lens calibration from the same board | `scripts/calibrate_lens.py --charuco` | Done; recovered a synthetic lens to 0.03 % |
| App: board scans detected automatically, export with board poses | `src/gui_ur5.py` (*4 Results* flow) | Done |
| Tests | `tests/test_charuco.py` | 8 passing |
| Capture protocol | `docs/CHARUCO_SCAN.md` | Done |

Test results on synthetic scans:
- Board poses: within 0.5 mm and 0.2°.
- Arm-placed cameras: within 1 mm, with a simulated 108.5 mm pendant TCP offset and an unknown mount.
- Lens self-calibration: within 2 %.

## 4. Required steps

### Step 1: prepare the board (you, once)
1. Print `hardware/charuco_board.pdf` at **100 % / actual size**, on matte paper.
2. Glue it flat to stiff card or foam board.
3. Measure 10 squares with a ruler, divide by 10, and record it:
   `uv run python scripts\make_charuco_board.py --measured-square-mm <value>`

### Step 2: prepare the camera (you, before each scan)
1. Lay the cloth in the middle of the board, with board showing on every side.
2. Move the arm to the scan height (12-15 cm) and press **Find best focus**. This turns autofocus off.
3. Optional: run `uv run python scripts\calibrate_lens.py --camera 0 --charuco` at that focus. It is
   only needed if the scan will see the board in fewer than 12 pictures.
4. Check the robot connection (`docs/ROBOT_CONNECTION.md`).

### Step 3: scan (you)
- **One session.** Don't touch the cloth or the board until it ends.
- **Height** 12-15 cm, so the board shows round the cloth. 5 cm close-ups of the middle are fine as well.
- **Tilted views** (15-25°) from several directions, and about 70 % overlap between neighbours.
- **About 100-200 pictures.**

### Step 4: export and check (you or me)
Under *4 Results*, choose **Export the dataset with poses from the ChArUco board**, then check
`splat_dataset/transforms.json`:

| Field | Target |
|---|---|
| `board_pose_rms_px` | < 0.5 |
| `mount_fit.rms_px` (if some pictures miss the board) | < 2 |
| `lens.reprojection_error_px` | < 1 |

If any is off, see the troubleshooting section of `docs/CHARUCO_SCAN.md` before training.

### Step 5: train and judge (me)
1. Train with the current `scripts/train_splat.ps1` (splatfacto, camera optimiser on).
2. Compare to the old runs. Held-out PSNR was 11-12 dB; **the goal is over 28 dB**, the bar the panel
   already uses for "good".
3. Look at the held-out renders next to the real pictures: individual stitches and yarn twist should
   be visible, not just the right colours.

### Step 6: tune training (me, after a good dataset exists)
These only help once the poses are right:
- **Appearance per picture.** Turn on per-picture appearance in splatfacto for the remaining lighting
  differences between views.
- **Splitting and step count.** Retune for a cloth now that the scene is consistent: currently 15,000
  steps and a split threshold of 0.0025.
- **Background.** Paint the board area out (like the existing *object only* mode for the nut) if it
  draws Gaussians away from the cloth.
- **Resolution.** Check whether the camera supports more than 640 × 480. More pixels per stitch is
  the biggest remaining quality lever.

## 5. Risks and what to watch

| Risk | Sign | Response |
|---|---|---|
| Cloth hides too much board | Few `board` frames, export warns | Scan higher, or show more border |
| Board not flat | Lens error > 1 px, board pose RMS > 0.5 px | Re-mount the print on something stiffer |
| Focus drifts mid-scan | Mount fit > 2 px | Check autofocus stayed off; rescan |
| Cloth moves during the scan (it's soft) | Good pose numbers but blurry splat | Pin or tape the cloth edges to the board |
| 640 × 480 is too coarse for fine yarn detail | Stitches sharp, fibres not | Higher camera resolution, or closer close-ups |

## 6. Unrelated issues found on the way

These fail in the current working tree, from code outside this work:
- **Scan planner, most of the 18 failures:** the arm-clearance check rejects every plan ("passes 56 mm above the
  fabric plane, against 100 mm required"). If it does this on the real robot, it blocks the scan in
  Step 3, so check it first.
- **Exposure equalisation:** `test_ur5_mode.py::test_export_writes_both_dataset_formats` doesn't
  expect the new equalisation warning.
- **Video sampler:** one test in `test_video_sampler.py` fails.
- **`tests/test_knitting_core.py`:** it no longer imports, because `_scale_factors_jax` is missing
  from `src/knitting_core.py`.
