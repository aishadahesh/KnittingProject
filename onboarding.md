# Knitting Reconstruction

A desktop application for editing knitted and crocheted geometry, simulating fabric scans, capturing with a UR5e, and fitting yarn models to reconstructed scans.
The application uses NumPy/SciPy for geometry and fitting, IPC Toolkit for yarn simulation, MuJoCo for robot planning, and Dear ImGui/ModernGL for the interface.
Gaussian splat training runs in a separate environment.

## Setup

Python 3.11 or newer and an OpenGL 3.3 context are required.
Install the application and camera support with:

```sh
uv sync --extra camera
uv run --extra camera python app.py
```

Use `uv sync` without the camera extra when working only with geometry or synthetic captures.
The `camera` extra supplies OpenCV and, on Windows, friendly camera device names.
The default robot transport connects through the separately running CraftLabXRFleet TeleopApp.
For the alternative direct RTDE transport, install `uv sync --extra camera --extra rtde`; its native extension may require a local build toolchain.
Opening the robot panel does not connect to hardware.
Connection and motion require explicit actions in the panel.

Dependencies are declared in `pyproject.toml` and resolved by `uv.lock`.
The editor has no JAX, Mitsuba, VTK, or Blender runtime dependency.

## Checks

```sh
uv run --extra camera ruff check
uv run --extra camera pytest
```

Pytest imports `src/` through the configuration in `pyproject.toml`.
Tests use temporary output directories, synthetic images, dry robot transports, and hidden OpenGL windows.
GUI tests skip when an OpenGL context is unavailable.
These checks do not verify physical robot operation or CUDA training.

## Navigation

| Location | Responsibility |
| --- | --- |
| `app.py` | Window, resource lifecycle, simulation worker, and frame loop |
| `src/paths.py` | Application file locations |
| `src/app_state.py` | Editable state, model mutations, undo, and workspaces |
| `config/config.json` | Knitted geometry parameters and reference-image configuration |
| `config/state_schema.json` | UI configuration, static state defaults, and saved field declarations |
| `config/initial_params.json` | Frozen reset baseline and scanner fabric template |
| `config/params.json` | Working model autosave; preserve local edits |
| `src/knitting_core.py` | Pure knitted control rows, periodic centre lines, Jacobians, and meshes |
| `src/crochet_core.py` | Crochet stitch equations, topology, and meshes |
| `src/yarn_simulation.py` | Stretch, bending, collision, and solver operations |
| `src/rendering.py`, `src/splat_gl.py` | Mesh and Gaussian GPU rendering |
| `src/gui.py` | Mode transitions and Edit, Scan, and Puzzle panels |
| `src/gui_database.py`, `src/gui_ur5.py`, `src/gui_model.py` | Database, robot, and reconstruction panels |
| `src/gui_captures.py` | Capture galleries and splat inspection |
| `src/scanner_core.py`, `src/fabric_scanner.py`, `src/embedded_scanner.py` | Scan orchestration, geometry, and embedded simulation |
| `src/scanner_storage.py`, `src/persistence.py` | SQLite metadata, state decoding, and atomic JSON writes |
| `src/ur5_robot.py`, `src/teleop_transport.py`, `src/ur5_scan.py`, `src/robot_camera.py` | Robot, simulated cell, and camera lifecycles and capture execution |
| `src/scan_geometry.py`, `src/fabric_detector.py` | Reachability, collision checks, capture plans, and fabric/RGB view metrics |
| `src/charuco.py`, `src/hand_eye.py`, `src/session_fit.py` | Board, mount, and lens calibration and camera fitting |
| `src/gaussian_splatting.py`, `src/free_capture.py` | Dataset export, trainer processes, and handheld capture |
| `src/scan_fit.py`, `src/lace_lattice.py`, `src/strand_equations.py` | Fabric and crochet geometry extraction and fitting |
| `src/fabric_palette.py`, `src/yarn_tracks.py` | Colours and yarn assignment |
| `src/drive_store.py` | User-requested scan/model transfers through rclone |
| `scripts/` | CLI tools and training environment setup |
| `hardware/` | Board print and camera mount designs |
| `tests/` | Numerical, persistence, capture, planning, and GUI regressions |

Modules in `src/` import each other by plain module name.
Entry points add `src/` to their import path.
Keep numerical modules independent of file access and GUI state.
Robot, camera, renderer, and worker objects own their resources; they are not model values to serialize or undo.

Model files use `format_version: 2` with GUI settings under `gui_state`.
Loads validate incoming data before committing it and rebuild dependent geometry.
Loading a model in the editor preserves the active mode; startup can restore the saved mode.
JSON files are replaced atomically through `persistence.write_json`.
SQLite captures remain the source of truth; `captures_index.json` is a derived export.

Knitted rows alternate direction.
The mesh and simulator use the same periodic centreline evaluator, and each row's signed period follows its control-point order.
Treat camera and robot coordinate conventions and units as part of each numerical API's contract.

## Workflows

- **Edit:** change the knit bitmap, loop heights, yarn radii, crochet stitches, colours, and display copies.
- **Scan:** generate fabric samples and capture them through the simulated MuJoCo UR5e.
- **Puzzle:** work on an independent model workspace for seamless repeats.
- **Database:** browse capture images, metadata, and colour analysis.
- **UR5 Robot:** connect, position the camera, check/rehearse a capture plan, capture, and reconstruct.
- **Model:** inspect reconstructed splats, extract or fit yarn geometry, and save or share a model.

UR5 Robot offers a simple workflow and an Expert panel with additional controls.
Every automatic motion is planned and checked against the simulated arm.
The software stop is not a substitute for the physical emergency stop.
Only one RTDE control session may connect to the UR5e controller at a time, so use either TeleopApp or direct RTDE.
Place the fabric flat over the printed ChArUco board (`config/charuco_board.json`), capture focused views from ~9–12 cm standoff, and refit `config/hand_eye.json` and `config/camera_intrinsics.json` whenever the mount in `hardware/endoscope_holder/` is disturbed.

`uv run --extra camera python scripts/fetch_mobilesam.py` downloads the optional fabric segmentation model.
Calibration tools include `scripts/calibrate_lens.py` and `scripts/make_charuco_board.py`.
Run each script with `--help` for its accepted inputs.

## Reconstruction environment

On Windows, `scripts/setup_splatting.ps1` creates `.venv-splat` with nerfstudio, gsplat, PyTorch, and pycolmap.
Training requires the CUDA Toolkit and Visual Studio C++ build tools used by that script.
`scripts/splat_env.ps1` configures the trainer and viewer environment.

`scripts/solve_poses.py` solves handheld images with COLMAP.
`scripts/export_board_dataset.py` exports scans using the printed board.
`scripts/fit_session.py` fits a scan's mount and lens to its own pictures.
`scripts/train_splat.ps1` trains the exported dataset; `scripts/view_splat.ps1` opens its viewer.
`scripts/visual_hull.py` provides a CPU silhouette reconstruction for checking object poses.
`scripts/fit_stitch_model.py` fits crochet templates to photos using the separate training environment.

## Local data

`robot_scans/`, `scanner_data/`, `scanner_images/`, `models/`, and `logs/` contain runtime data or downloaded assets and are ignored by Git.
`imgui_layout.ini` is the application's saved window layout.
Do not delete captures, calibrations, trained models, or edited parameters during code cleanup.
Use `scripts/profile_app.py` to profile the geometry and scanning pipeline against local data.


