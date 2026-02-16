# Track Specification: Refactor and Unify Knitting Model Optimizer

## Objective
Transform the current `optimize_knitting.py` from a collection of fragmented "apps" and hardcoded constants into a unified, logically organized, and configurable research tool.

## Key Requirements
1. **Unified UI:**
   - Merge the Spline Editor and Parameter Editor into a single Matplotlib/interactive window.
   - Integrate the rendering/optimization view into this same interface.
   - Use a cohesive interaction model (e.g., specific keys to toggle modes or update views).

2. **Configuration Management:**
   - Extract all hardcoded constants (geometry params, rendering settings, optimization hyperparams) into `config.json`.
   - Implement a `load_config()` utility as the single source of truth for these values.

3. **Flat & Logical Reorganization:**
   - Reorder `optimize_knitting.py` so related parts (Imports -> Config -> Geometry Engine -> Rendering Pipeline -> UI/App Loop) are near each other.
   - Definitions must appear in order of execution flow.
   - Consolidate single-use functions to reduce fragmentation.

4. **Code Quality:**
   - Apply DRY, Single Source of Truth, and One True Path principles.
   - Remove deep nesting and unnecessary abstractions.

## Success Criteria
- A single execution of `python optimize_knitting.py` opens one unified window.
- All primary parameters can be tuned via `config.json` without modifying code.
- The script is logically partitioned using comments/folding and follows a linear execution flow.
