
# SHE Action List

This document summarizes the cleanup and refactoring tasks for the **Simplicial Hyperstructure Engine (SHE)** repository, with specific focus on `src/core/SHE.py`.

---

## A. Packaging & Module Boundaries
1. remove main from core 
2. Add `__init__.py` in `src/`, `src/core/`, `src/morse/`, `src/stanleyreisner/`.

## B. Imports & Optional Dependencies
3. Standardize optional dependencies (`toponetx`, `torch_geometric`, `gtda`, `gudhi`, `seaborn`).
4. Make `seaborn` optional (currently used only in `plot_diffusion_heatmap`). Provide a fallback using matplotlib.

## C. API, Types, and Config
5. Add explicit type hints for all public methods.
6. Add full NumPy-style docstrings for every public class/method.
7. Expand `SHEConfig` with validated fields (`normalized_laplacian`, `eig_k`, `seed`, `heat_times`).

## D. Simplicial Complex Correctness
8. Ensure closure and orientation conventions when adding simplices.
9. Provide consistent incidence/boundary operators with stable indexing.
10. Centralize weight handling (`get_simplex_weights` as single source of truth).

## E. Hodge Diffusion & Numerics
11. Support combinatorial and normalized Laplacians.
12. Improve eigen solver selection (`eigsh` vs `eigh`) with tolerance checks.
13. Document and validate heat kernel / diffusion map pipeline.
14. Extend Hodge decomposition: orthogonal projections (`im d`, `im δ`, `ker L`).
15. Make key diffuser scoring explicit and documented.

## F. Visualization
16. Ensure matplotlib-first plotting; seaborn only optional.
17. Improve axis labeling and add `save_path` option to plots.

## G. Engine & Demos
18. Remove `print` statements; use `logger` only.
19. Move `__main__` demo to `/examples/` notebooks using `synthetic_she_data.csv`.
20. Extend `analyze_diffusion` with options for dimension, `eig_k`, `heat_times`.

## H. Data Loading
21. Harden `SHEDataLoader.from_weighted_networkx`:
    - Parameters: `weight_attr`, `include_cliques`, `max_clique_size`
    - Handle missing weights gracefully
    - Warn on large cliques

## I. Logging, Errors, and Messages
22. Replace all `print(...)` with logging calls.
23. Standardize ImportError messages with exact `pip install ...` hints.

## J. Testing Hooks (to implement later)
24. Unit tests for:
    - Incidence signs and Laplacian PSD property
    - Spectrum shape and harmonic count (toy complexes)
    - Diffusion map invariants
    - Key diffuser ranking monotonicity

## K. Quick Fixes
25. Add missing `__init__.py` files.
26. Rename or remove `SHESimplicialConvolutionalNetwork` file (give `.py` extension).
27. Remove duplicate license files (keep `LICENSE` only).
28. Harden seaborn import fallback in `plot_diffusion_heatmap`.

---

**Assignee:**Mirco 
**Priority:** Start with Section K (quick fixes), then A/B (packaging and imports), and finally work through C–J in order.
