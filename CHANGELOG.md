# Changelog

## v0.1.2 — Export, Decay Windows, Real Data (2026-03-26)

### Added

- **Result export** — `ranked_items_to_csv/json`, `bridges_to_csv/json`, `cohesion_to_csv/json`
  for feeding SHE outputs into pandas, dashboards, or downstream tools
- **Decay-weighted temporal accumulation** — `decay_window()` applies exponential
  decay (configurable half-life) so older interactions fade rather than vanish at
  a hard boundary
- **Real public dataset notebook** — `examples/eu_email_analysis.ipynb` analyses the
  SNAP EU Email Dept3 network (20 most-active members, 812 interactions, 27 months)
  with Louvain community detection, temporal bridge/cohesion tracking, and matplotlib
  plots
- **Bundled dataset** — `data/eu_email_dept3.jsonl` (pre-processed from SNAP)

### Tests

- 41 tests passing (10 new: 6 export, 4 decay window)

---

## v0.1.1 — Modeling Layer, Social Analysis, Temporal (2026-03)

### Added

- **`SHEHyperstructure`** — decorated, weighted higher-order relational structure
  with entity attributes, typed relations, and arbitrary per-simplex metadata
- **Social analysis** — `rank_diffusers`, `rank_entity_diffusers`,
  `rank_simplex_diffusers`, `find_bridge_simplices`, `group_cohesion`,
  `rank_influencers` (graph centrality vs simplex diffusion contrast)
- **Bulk ingestion** — `from_records`, `from_csv`, `from_jsonl`
- **Temporal windowing** — `window()` and `rolling_windows()` for time-sliced analysis
- Worked use case: `examples/social_media_diffusers.py`
- Temporal notebook: `examples/temporal_diffusion_analysis.ipynb`

### Fixed

- Diffusion scoring: switched to rank-percentile normalisation with `lambda=1.0`
  (previously scores saturated at 1.0 due to identity-term dominance)
- `analyze_diffusion` no longer truncates to top 10 internally

### Tests

- 31 tests across 7 files (up from 8 in v0.1.0)

---

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
