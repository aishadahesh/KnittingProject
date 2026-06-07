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

### Main Interface (Dear ImGui)
To launch the OpenGL-based interactive editor:
```bash
uv run imgui_app.py
```

### Web Interface (Trame)
To launch the alternative web-based VTK editor:
```bash
uv run trame_app.py
```

## Differentiable GPU Optimization Details

To execute differentiable rendering and JAX geometry optimization concurrently on a single GPU, the following configurations are automated within this codebase:

1. **Dynamic VRAM Allocation**: JAX (via XLA) is configured with `XLA_PYTHON_CLIENT_PREALLOCATE=false` at startup. This prevents JAX from preallocating the majority of device memory, leaving VRAM available for Mitsuba's JIT compiler.
2. **Path Replay Backpropagation (PRB)**: To prevent out-of-memory (OOM) conditions during backpropagation, the Mitsuba scene is configured to use the `prb` integrator. Path Replay Backpropagation recomputes light paths during the backward pass, maintaining a constant memory footprint relative to ray depth.
3. **Dynamic Library Resolution**: The Python environment dynamically resolves and preloads the packaged CUDA/cuDNN shared libraries (`.so` files) on Linux. This avoids runtime initialization errors without requiring manual changes to `LD_LIBRARY_PATH`.

*Note: If Mitsuba JIT compilation fails during OptiX translation (`ptx2llvm-module-001`), verify that a stable NVIDIA driver version (e.g., 550.x or 560.x) is active on the system, as developer/beta driver releases may contain PTX parser regressions.*
