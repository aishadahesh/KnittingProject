# Knitting Model Optimizer

This project is a specialized tool for optimizing 3D knitting geometry to match target reference images. It uses differentiable rendering and gradient-based optimization to refine parameters that define the structure and appearance of knitted fabrics.

## Project Overview

The core of the project is a differentiable knitting simulator and optimizer. It takes a reference image of a knitted pattern and attempts to find the geometric parameters (stitch bulge, loop height, spacing, etc.) that, when rendered, produce a similar image.

### Main Technologies
- **Python 3.12**: The primary programming language.
- **Mitsuba 3**: A research-oriented rendering system used here for differentiable rendering (`cuda_ad_rgb` or `llvm_ad_rgb` variants).
- **JAX & DrJit**: Used for high-performance numerical computing, automatic differentiation of geometry logic, and JIT compilation.
- **Optax**: A gradient processing and optimization library for JAX.
- **PyTorch & Torchvision**: Used specifically for calculating **Perceptual Loss** using a pre-trained VGG16 model.
- **Matplotlib & PIL**: Used for image processing, interactive editing interfaces, and generating progress visualizations.
- **Vedo** (Optional): Used for interactive 3D mesh visualization.

## Architecture & Core Functionality

The main logic is contained in `optimize_knitting.py`, which includes:

1.  **Geometry Engine (JAX)**: Functions to evaluate curves and generate knitting mesh vertices based on parameters like `stitch_bulge`, `loop_height`, `radius`, etc.
2.  **Differentiable Pipeline**: Integrates the JAX geometry logic with Mitsuba's differentiable rendering to compute gradients of the image loss with respect to geometric parameters.
3.  **Hybrid Loss Function**: Combines standard Pixel MSE (Mean Squared Error) with VGG-based Perceptual Loss to capture both global structure and semantic details.
4.  **Interactive Editors**:
    *   **Spline Editor**: Allows manual adjustment of control points and radii using a GUI.
    *   **Parameter Editor**: Allows real-time adjustment of global knitting parameters using arrow keys.
5.  **Optimization Loop**: Automates the refinement of parameters using Optax optimizers (e.g., Adam) with features like early stopping and camera calibration.
6.  **Visualization Engine**: Automatically generates before/after comparisons, parameter evolution plots, and high-quality final renders.

## Building and Running

### Dependencies
Ensure you have the required libraries installed in your environment:
```bash
pip install mitsuba drjit jax jaxlib optax torch torchvision matplotlib pillow vedo
```

### Running the Optimizer
The main entry point is `optimize_knitting.py`.
```bash
python optimize_knitting.py
```
Upon execution, the script will:
1.  Load the default reference image (`referenceImage_cropped_new1.jpg`).
2.  Optionally calibrate the camera to match the reference framing.
3.  Enter an **Interactive Model Editor** where you can choose between the Spline Editor and the Parameter Editor.
4.  Allow you to trigger the optimization loop by pressing `O` in the interactive window or finish by pressing `F`.

### Outputs
Results are saved in the following directories:
- `opt_outputs/meshes/`: Contains `.obj` files of the generated knitting structures (e.g., `best_model_combined.obj`).
- `opt_outputs/renders/`: Contains intermediate renders and summary visualizations (e.g., `epoch_XXX_summary.png`).
- `results/`: Pre-generated or historical results.

## Development Conventions

- **Interactive Workflow**: The project heavily relies on an interactive loop between manual editing and automated optimization.
- **Visualization First**: Almost every step of the optimization is visualized to provide insights into how parameters affect the final render.
- **Mitsuba Variants**: The script automatically attempts to set the Mitsuba variant to `cuda_ad_rgb` for GPU acceleration, falling back to `llvm_ad_rgb` if a CUDA-compatible GPU is not available.
- **Center Cropping**: For efficiency and focus, the loss is often calculated on a central crop of the rendered image, controlled by `LOSS_CENTER_CROP`.
