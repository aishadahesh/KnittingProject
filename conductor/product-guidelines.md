# Product Guidelines

## Development Philosophy
- **Experimental Flexibility:** The codebase must remain modular to allow researchers to easily swap out loss functions, geometry generation logic, or rendering backends (e.g., CUDA vs. LLVM).
- **Research Clarity:** Prioritize code readability and structural transparency over aggressive micro-optimizations that might obscure the underlying mathematical logic.

## Reporting & Visualization
- **Standardized Summaries:** Every optimization run must produce a standardized visual summary (e.g., the `epoch_XXX_summary.png`) that includes the reference image, the current render, and key loss metrics for immediate visual validation.
- **Visual-First Debugging:** Maintain and expand the automated visualization engine to ensure that every step of the parameter evolution is visually verifiable.

## Code Standards
- **Modular Integration:** New features (like new stitch types or loss components) should be implemented as modular extensions rather than monolithic changes to `optimize_knitting.py`.
- **Differentiability Awareness:** All geometry-related code must be compatible with JAX/DrJit to ensure the gradient chain remains unbroken.
