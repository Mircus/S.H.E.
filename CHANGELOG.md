# Changelog

## v0.1.0 — Research Preview (2025)

First public release of SHE as a small, runnable research preview.

### Included

- **Core simplicial-complex construction** via `SHESimplicialComplex` (wraps TopoNetX)
- **Graph / clique lifting** via `SHEDataLoader.from_weighted_networkx()`
- **Hodge-Laplacian diffusion analysis** via `SHEHodgeDiffusion`
- **Minimal visualisation** via `SHEDiffusionVisualizer`
- **Engine convenience wrapper** via `SHEEngine`
- Three runnable examples: `toy_triangle.py`, `social_group_lift.py`, `group_diffusion_demo.py`
- Minimal test suite (imports, complex construction, diffusion smoke)
- GitHub Actions CI

### Experimental (not part of stable API)

- `src/core/SHESimplicialConvolution.py` — simplicial neural network blocks (requires PyTorch)
- `src/morse/` — discrete Morse theory (requires PyTorch, numba, sparse, psutil)
- `src/core/SHE.py` — legacy monolithic module (retained for reference)

### Known limitations

- API may change in future releases
- Hodge analysis returns only the **harmonic component**; exact/coexact decomposition is not yet implemented
- No persistent homology in core; available via `pip install she[tda]`
- Tested with TopoNetX 0.2.x on Python 3.11
- Not tested for production workloads
- Source-available under HNCL v1.0 (non-commercial); not OSI open-source
