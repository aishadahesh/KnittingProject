# Initial Concept
This is a research project with the goals of reconstruction a 3d model of colored knit fabric from an image. It is still under active development, focusing on differentiable rendering, parameter optimization, and hybrid loss functions for accurate geometry and appearance matching.

# Product Definition

## Target Audience
- Researchers in Computer Graphics and Vision focusing on material reconstruction and differentiable rendering.
- Technical Artists and Designers specializing in physically-based fabric simulation and procedural modeling.

## Core Goals
- **High-Fidelity Reconstruction:** Automatically derive 3D geometric parameters (stitch bulge, loop height, spacing) that accurately replicate the structure of knitted fabric from a single 2D reference image.
- **Differentiable Pipeline Integration:** Maintain a fully differentiable loop from JAX-based geometry generation to Mitsuba-based rendering.
- **Robust Optimization:** Utilize hybrid loss functions (Pixel MSE and VGG-based Perceptual Loss) to handle both low-level pixel accuracy and high-level structural features.
- **Interactive Refinement:** Provide a bridge between automated optimization and manual control through Spline and Parameter editors.

## Key Features
- **Differentiable Knitting Simulator:** Real-time generation of knitting meshes driven by parametric inputs.
- **Hybrid Optimization Engine:** Multi-objective loss optimization using Optax and JAX.
- **Visual Feedback Loop:** Automated generation of progress visualizations, parameter evolution plots, and before/after comparisons.
- **Interactive Editing Suite:** GUI tools for manual spline adjustment and parameter tuning.
