# Tech Stack

## Core Technologies
- **Python 3.12**: Main programming language.
- **Mitsuba 3**: Research-oriented rendering system for differentiable rendering. Uses `cuda_ad_rgb` for GPU or `llvm_ad_rgb` for CPU fallback.
- **JAX & DrJit**: High-performance numerical computing and automatic differentiation. JAX handles geometry logic; DrJit provides the backend for Mitsuba and JAX integration.
- **Optax**: Gradient processing and optimization library for JAX (e.g., Adam optimizer).

## Machine Learning & Computer Vision
- **PyTorch & Torchvision**: Used for calculating Perceptual Loss via pre-trained VGG16 models.

## Visualization & UI
- **Trame & VTK**: Core interactive 3D visualization and reactive UI framework.
- **Matplotlib**: Generating plots, progress visualizations, and simple GUI overlays.
- **PIL (Pillow)**: Image loading, processing, and cropping.

## Development Tools
- **Git**: Version control.
