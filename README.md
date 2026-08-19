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
| Real UR5 robot | `ur5_robot.py`, `ur5_scan.py`, `robot_camera.py`, `gaussian_splatting.py` |

### scripts/

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

## Differentiable GPU Optimization Details

To execute differentiable rendering and JAX geometry optimization concurrently on a single GPU, the following configurations are automated within this codebase:

1. **Dynamic VRAM Allocation**: JAX (via XLA) is configured with `XLA_PYTHON_CLIENT_PREALLOCATE=false` at startup. This prevents JAX from preallocating the majority of device memory, leaving VRAM available for Mitsuba's JIT compiler.
2. **Path Replay Backpropagation (PRB)**: To prevent out-of-memory (OOM) conditions during backpropagation, the Mitsuba scene is configured to use the `prb` integrator. Path Replay Backpropagation recomputes light paths during the backward pass, maintaining a constant memory footprint relative to ray depth.
3. **Dynamic Library Resolution**: The Python environment dynamically resolves and preloads the packaged CUDA/cuDNN shared libraries (`.so` files) on Linux. This avoids runtime initialization errors without requiring manual changes to `LD_LIBRARY_PATH`.

*Note: If Mitsuba JIT compilation fails during OptiX translation (`ptx2llvm-module-001`), verify that a stable NVIDIA driver version (e.g., 550.x or 560.x) is active on the system, as developer/beta driver releases may contain PTX parser regressions.*
