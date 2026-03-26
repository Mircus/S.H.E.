# S.H.E. — Simplicial Hyperstructure Engine

[![License: HNCL](https://img.shields.io/badge/license-HNCL-blue.svg)](/LICENCE.md)

<p align="center">
  <img src="she_logo.png" alt="SHE logo" width="280">
</p>

SHE is a source-available Python toolkit for building **weighted simplicial
complexes** from relational data, computing **Hodge Laplacians**, and running
**diffusion / spectral analysis** on higher-order structures.

This is a **v0.1 Research Preview** — useful for exploration, not yet hardened
for production.  Released under the non-commercial
[HNCL v1.0](LICENCE.md) license (not OSI open-source).

## What v0.1 includes

- Simplicial-complex construction (wraps [TopoNetX](https://github.com/pyt-team/TopoNetX))
- Graph-to-simplicial lifting via clique detection
- Hodge-Laplacian computation, spectral analysis, and harmonic-form extraction
- Diffusion centrality ranking
- Minimal matplotlib visualisation
- Three runnable examples and a small test suite

## Installation

```bash
# core only
pip install -e .

# with test tooling
pip install -e ".[dev]"

# with optional TDA support (gudhi, giotto-tda)
pip install -e ".[tda]"
```

Requires **Python >= 3.10**.

## Quickstart

```python
import networkx as nx
from she import SHEDataLoader, SHEHodgeDiffusion, SHEConfig

# 1. Start from a NetworkX graph
G = nx.karate_club_graph()

# 2. Lift to a simplicial complex (cliques become higher-order simplices)
sc = SHEDataLoader.from_weighted_networkx(G)

# 3. Run diffusion analysis
config = SHEConfig(max_dimension=2, spectral_k=5)
analyzer = SHEHodgeDiffusion(config)
result = analyzer.analyze_diffusion(sc)

# 4. Inspect top diffusers
for dim, diffusers in result.key_diffusers.items():
    print(f"Dimension {dim}: top diffuser = {diffusers[0]}")
```

## Examples

| Script | Description |
|--------|-------------|
| `examples/toy_triangle.py` | Smallest nontrivial complex — Hodge Laplacian printout |
| `examples/social_group_lift.py` | Lift a small social graph to simplices via cliques |
| `examples/group_diffusion_demo.py` | Weighted Karate Club diffusion analysis with plot |

Run any example with:

```bash
python examples/toy_triangle.py
```

## Experimental modules

The following are **not** part of the stable v0.1 API and require extra
dependencies:

| Module | Requires | Install extra |
|--------|----------|---------------|
| `src/core/SHESimplicialConvolution.py` | PyTorch | `pip install she[ml]` |
| `src/morse/` | PyTorch, numba, sparse, psutil | `pip install she[morse]` |

## Limitations

This is a **Research Preview**.  The API may change between releases.
It has not been optimised or audited for production use.

- Hodge analysis currently computes the **harmonic component** only; exact and
  coexact parts of the decomposition are not yet implemented.
- Tested with TopoNetX 0.2.x on Python 3.11.
- Not OSI open-source — see License section below.

## License

**Holomathics Non-Commercial License (HNCL) v1.0**

Free for personal, academic research, and educational use.
Commercial use requires a separate license from Holomathics.
See [LICENCE.md](LICENCE.md) for full text.

Contact: [info@holomathics.com](mailto:info@holomathics.com)
