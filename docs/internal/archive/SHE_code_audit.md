# SHE Codebase Quick Audit (Core)

**Scope:** /src/core and examples in S.H.E.-main zip

## Critical issues found
1. Packaging missing: no `pyproject.toml`; `she.toml` is a *doc snippet*, not used by build tools.
2. Import surface inconsistent: examples refer to `she_core`, but the code lives under `src/core/*` and `__init__.py` files are empty.
3. Syntax/Name errors:
   - `src/core/diffusion/SHEHodgeDiffusion.py`: uses `lambda_dif` instead of `lambda_diff`.
   - `src/core/config/SheConfig.py`: missing `from dataclasses import dataclass`, missing `import torch`.
   - `src/core/complex/SHESimplicialComplex.py`: truncated method body (`edge_features[...] = ...`).
   - `src/core/engine/SHEEngine.py`: truncated calls in `visualize_diffusion(...)`.
4. Example `examples/SHEDemo.py` is not runnable (commented imports, stray `TOPOX_AVAILABLE` token, heavy external deps).

## High‑priority fixes applied in this patch
- Add **pyproject.toml** with `src` layout and a `she` facade that re‑exports the `core/*` modules.
- Fix **SheConfig** imports and make device/dtype resilient when `torch` is not present.
- Fix **Hodge diffusion** typo and add sparse guards.
- Complete **SHESimplicialComplex.add_simplex** minimal functionality.
- Provide **SHEDemo_min.py** which runs end‑to‑end without torch‑geometric.
- Add **pytest** smoke test and **CI** workflow.

## Next recommended fixes (not in patch)
- Finish `SHEEngine.visualize_diffusion` and ensure all visualizer calls exist.
- Add construction of proper **Hodge L1/L2** from boundary operators and tests (currently toy example uses identity).
- Split requirements into `base` vs `extra` (torch-geometric under an extra).
- Pin versions and document CUDA instructions.
- Replace `examples/SHEDemo.py` with a cleaned, runnable version or convert it to a Jupyter notebook.
