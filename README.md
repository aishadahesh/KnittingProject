# Knitting Reconstruction GUI

An interactive graphical user interface for physical-based knitting reconstruction, utilizing JAX for differentiable geometry optimization and Mitsuba 3 for differentiable rendering.

## Installation

This project manages dependencies using `uv`. 

### CPU Execution (Windows, Linux, macOS)
To synchronize the virtual environment for CPU execution:
```bash
uv sync
```

### GPU Execution (Linux & WSL2)
To synchronize the virtual environment with CUDA acceleration enabled:
```bash
uv sync --extra gpu
```
*Note: Python will automatically resolve and preload the local virtual environment's packaged CUDA/cuDNN shared libraries on Linux startup.*

## Usage

To launch the OpenGL-based interactive editor:
```bash
uv run app.py
```

## Project layout

```
app.py            Entry point. Puts src/ on the path, then starts the GUI.
src/              Application modules, importing each other by plain name.
config/           config.json, state_schema.json, initial_params.json,
                  params.json
assets/           refImg.png, model.png
scripts/          Standalone utilities, not part of the running app.
tests/            Pytest suite (src/ is put on the path by conftest.py).
```

File locations are declared once, in `src/paths.py`. Nothing else derives the
project root or joins config filenames itself, so moving a file means editing
one place. Paths stored inside `config.json` (the reference image) are written
relative to the project root and resolved through `paths.resolve()`.

Two things deliberately stay in the project root: `app.py`, and
`imgui_layout.ini` — the latter is window layout the UI writes at runtime, not
configuration anyone edits.

### src/

| Area | Modules |
| --- | --- |
| Layout | `paths.py` — where every file lives; the one place that knows |
| Model and state | `app_state.py`, `knitting_core.py`, `config_schema.py`, `mesh_io.py` |
| Rendering and UI | `rendering.py`, `gui.py`, `gui_ur5.py` |
| Simulation | `yarn_simulation.py` |
| Scanning | `fabric_scanner.py`, `scanner_storage.py`, `rgb_analysis.py` |
| Real UR5 robot | `ur5_robot.py`, `ur5_scan.py`, `robot_camera.py`, `gaussian_splatting.py`, `free_capture.py` |

### scripts/

- `solve_poses.py` — camera poses for a hand-held capture (5 Free capture runs
  it with `.venv-splat`'s Python).
- `profile_app.py` — profiles the geometry pipeline against the project's own
  data. Run as `python scripts/profile_app.py`.
- `main.py` — an older Blender script (`import bpy`). It is kept for reference
  and does not run as part of this application.

## Modes

The app has five modes, the first four simulated and the last driving physical
hardware:

- **Edit** — build and edit the knitted model.
- **Scan** — simulated UR5e scan of a fabric grid, in MuJoCo.
- **Puzzle** — manual seamless-repeat workflow.
- **Database** — browse captures, their metadata and colour analysis.
- **UR5 Robot** — the same scan workflow on a real UR5 over RTDE, capturing
  from a real gripper camera. Requires `opencv-python` for the camera
  (`uv sync --extra camera`); it otherwise falls back to clearly marked
  synthetic frames. The in-app stop button is a software protective stop and is
  not a substitute for the hardware emergency stop.
  Everything is done from the panel, without touching the arm. The stop, pause,
  reset, Home and the latest message sit above five tabs, in the order a session
  uses them:

  UR5 Robot Mode opens as one simple page - 1 Connect, 2 Put the camera over the
  object, 3 Scan, 4 Reconstruct - with defaults for everything else. Tick
  *Expert* at the top for the full five-tab panel described below.
  - **1 Setup**: how the panel reaches the arm, the camera, speed, and calibration
    (tool offset, lens field of view, Straighten Home).
  - **2 Position**: the live preview with a crosshair and a jog pad. The jog pad
    slides the camera left, right, forward or back as seen in the preview, or raises
    and lowers it, in steps from 1 mm to 5 cm. Every jog is solved and collision
    checked like a capture, and it keeps the lens straight down. Under **Fabric**,
    press *Measure fabric height with the camera*. It slides the camera 3 mm each way
    and works the distance out from how far the picture moves, so heights and
    clearances are true; a ruler gap typed in does the same. This is only as accurate
    as the lens field of view under Setup. The height is saved until you set it again.
    Then, optionally, jog each fabric corner under the crosshair and mark it; the
    outline keeps every automatic frame on the cloth. Free drive is still there,
    under *Advanced*.
  - **3 Capture**: the capture height, the pattern and its few settings, image
    quality, then Check, Rehearse and Run. *Use the camera's height now* takes every
    frame at exactly the height you jogged to, so focus holds. A new height re-plans
    the points by itself. Grid stations sit one frame less the *Frame overlap* apart,
    so they spread out as the camera rises and close in as it drops. **taught** flies
    views recorded at poses you jogged to. Taught sets are saved in
    `config/taught_views/`.
    **splat** is the set for Gaussian Splatting: a dome of views aimed at the
    spot under the camera when Check is pressed - straight down, then full rings
    tilted up to *Max tilt* - at one to three distances from the capture height
    out (74 views by default). Put the object under the crosshair, jog to where
    it is sharp, press *Use the camera's height now*, and the dome's first view
    is the camera's own position.
    Its views may frame past the cloth's edge, which anchors a splat in space.
    **sweep** is the grid with its stations only *Overlap between frames* apart in
    both directions, a still at every one. Each still carries the camera pose the
    arm was planned to, so the splat trains from the arm's own poses (4 Results >
    Gaussian Splatting > *the arm's own poses*) with no matching of pictures. That
    matters because a plain knit defeats the matching: every stitch looks like
    every other, so COLMAP finds almost no reliable match even between pictures
    that overlap by 85%, and a solve from the pictures comes out as a few placed
    or several pieces. A solve in pieces, or with few placed, cannot be trained;
    the trainer says so.
    The arm's poses are only right once the **camera mount is calibrated**: the
    planner assumes the camera is the tool centre point looking along the tool
    axis, and on this arm the lens sits about 5 cm beside and behind it, turned.
    Uncalibrated, every exported pose misses the pictures by hundreds of pixels
    and the splat is a cloud of floaters. 4 Results > Gaussian Splatting is a
    five-step list - Scan, Calibrate mount, Export dataset, Train, Judge - with
    one *Next* button that does the next step; COLMAP and the planner's poses
    are under *Advanced*. Scan one small dark object on plain white paper (a
    nut) with the splat pattern, then press Next to calibrate: `hand_eye.py`
    fits where the camera sits on the tool and the lens's field of view from
    where the object landed in every picture, saves it as `config/hand_eye.json`
    (a fit to ~10 px is normal), and *Export splatting dataset with the
    calibrated poses* applies it to every later scan. The trainer
    (`scripts/train_splat.ps1`) lets each camera shift a little during training
    to take up what is left. *Object only* (the default) paints everything but
    the one dark object black in the exported pictures and trains against a
    black background, so the splat is the object alone. `scripts/visual_hull.py`
    builds a coarse model of the same object on the CPU in seconds from the
    exported poses alone (silhouette carving) - a check that the poses agree
    with the pictures - and with `--mesh file.stl` or `.obj` writes it as a mesh
    any 3D tool or slicer opens. `scripts/calibrate_lens.py` measures the lens
    itself from 15-25 photos of a printed checkerboard (focal length, centre,
    distortion) into `config/camera_intrinsics.json`; redo the mount
    calibration afterwards and it, and every export, use the measured lens.
    Any run may also tick *Also keep frames from the video while moving*: frames
    from the live stream are kept between stations when nearly as sharp as the
    last standing one, the whole stream is recorded as `video.mp4` beside them,
    and *Pick the pictures again from the recorded video* (4 Results) makes the
    choice afresh from the file. These frames have no pose and serve the
    picture-matching solve only.
  - **4 Results**: captured images, RGB analysis, Gaussian Splatting, and the
    Scan Mode grid run on the real arm.
  - **5 Free capture**: Gaussian Splatting from pictures taken with the camera in
    hand - no arm. Every picture is matched against the earlier ones as it is
    taken, and a checklist says when there are enough: far, middle and close
    views (distance relative to the first picture, the whole fabric straight
    down), tilted views from at least 6 of 8 sides, and each picture overlapping
    an earlier one. The tab is five steps, all on screen: camera and session,
    shooting, the checklist, the pose solve, and the splat. Pictures are taken
    with the button, Space, or automatically - each time the live view has moved
    on from the last picture while still overlapping it. Every try flashes the
    preview green (kept), red (kept, matches nothing) or amber (skipped: blurred
    against the camera's own recent sharpness, or a repeat), and says why. *Solve camera poses* then runs
    `scripts/solve_poses.py` (COLMAP through pycolmap, in `.venv-splat`), checks
    the same targets on what it measured plus how many pictures it placed, and
    the result trains like a robot export. Sessions are saved as
    `robot_scans/free_*`. The estimates while shooting assume the cloth lies flat.

  Each capture is scored for sharpness, and blurred ones are retaken. Once
  `uv run python scripts/fetch_mobilesam.py` has fetched the MobileSAM model (~45 MB),
  each capture is also checked for how much of it is fabric. Each pose is solved
  from several whole-arm starts (where the arm is, Home, and both turned to face the
  target), so a pose the arm can reach by refolding is not refused just because the
  nearest solution stalled.

## Differentiable GPU Optimization Details

To execute differentiable rendering and JAX geometry optimization concurrently on a single GPU, the following configurations are automated within this codebase:

1. **Dynamic VRAM Allocation**: JAX (via XLA) is configured with `XLA_PYTHON_CLIENT_PREALLOCATE=false` at startup. This prevents JAX from preallocating the majority of device memory, leaving VRAM available for Mitsuba's JIT compiler.
2. **Path Replay Backpropagation (PRB)**: To prevent out-of-memory (OOM) conditions during backpropagation, the Mitsuba scene is configured to use the `prb` integrator. Path Replay Backpropagation recomputes light paths during the backward pass, maintaining a constant memory footprint relative to ray depth.
3. **Dynamic Library Resolution**: The Python environment dynamically resolves and preloads the packaged CUDA/cuDNN shared libraries (`.so` files) on Linux. This avoids runtime initialization errors without requiring manual changes to `LD_LIBRARY_PATH`.

*Note: If Mitsuba JIT compilation fails during OptiX translation (`ptx2llvm-module-001`), verify that a stable NVIDIA driver version (e.g., 550.x or 560.x) is active on the system, as developer/beta driver releases may contain PTX parser regressions.*
