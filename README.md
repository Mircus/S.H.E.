# S.H.E. — Simplicial Hyperstructure Engine

[![CI](https://github.com/Mircus/S.H.E./actions/workflows/ci.yml/badge.svg)](https://github.com/Mircus/S.H.E./actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: HNCL](https://img.shields.io/badge/license-HNCL-blue.svg)](/LICENCE.md)
[![Status: Research Preview](https://img.shields.io/badge/status-research%20preview-orange.svg)](#repo-status)

<p align="center">
  <img src="she_logo.png" alt="SHE logo" width="500">
</p>

SHE is a source-available research library for modeling and analyzing
**decorated higher-order relational structures**, with a current computational
focus on weighted simplicial representations and **social / group-level
diffusion analysis**.

This is a **Research Preview** released under the non-commercial
[HNCL v1.0](LICENCE.md) license (not OSI open-source).

## Repo status

| | |
|---|---|
| **Version** | 0.1.2 |
| **Canonical package** | `src/she/` — install with `pip install -e .`, import as `import she` |
| **Maturity** | Research preview — API may change between releases |
| **Stable surface** | `SHEHyperstructure`, social analysis functions, temporal windowing, export |
| **Experimental** | `src/core/` (legacy monolith, convolution), `src/morse/` (discrete Morse) — not part of the public API |
| **Tests** | 41 passing (CI on Python 3.10–3.12) |

## Why SHE?

Standard graph analysis collapses every interaction to a pairwise edge.
When the signal you care about lives in **small-group structure** — triads,
co-engagement cliques, collaborative clusters — graph methods wash it out.

Use SHE when:

- **Triads and higher simplices matter.** A co-amplification group of three
  accounts is a different object from three pairwise edges.
- **Group-level diffusion matters.** The question is not just "who is central?"
  but "which small group carries the most structural weight in diffusion?"
- **Bridge groups matter.** You want to find the cross-community triad, not
  just the cross-community edge.
- **Decorations matter.** Relations carry weight, type, topic, and metadata
  that you want to query and analyze — not just adjacency.

SHE does not replace graph libraries.  It adds a layer for the cases where
graphs are not enough.

## What v0.1.x includes

**Modeling layer**
- `SHEHyperstructure` — decorated, weighted higher-order relational object
  with entity attributes, typed relations, and bulk ingestion (`from_csv`, `from_jsonl`)

**Social analysis**
- `rank_diffusers` / `rank_entity_diffusers` / `rank_simplex_diffusers`
- `find_bridge_simplices` — cross-community higher-order bridges (heuristic)
- `group_cohesion` — structural cohesion scoring for candidate groups (composite)
- `rank_influencers` — graph centrality vs. simplex diffusion comparison

**Temporal**
- `window` / `rolling_windows` — hard time-range slicing
- `decay_window` — exponential decay with configurable half-life

**Export**
- `ranked_items_to_csv/json`, `bridges_to_csv/json`, `cohesion_to_csv/json`

**Core simplicial engine**
- Simplicial-complex construction (wraps [TopoNetX](https://github.com/pyt-team/TopoNetX))
- Graph-to-simplicial lifting via clique detection
- Hodge-Laplacian spectral analysis and harmonic-form extraction
- Diffusion centrality ranking (rank-percentile normalisation)
- Minimal matplotlib visualisation

## Installation

```bash
git clone https://github.com/Mircus/S.H.E.
cd S.H.E
pip install -e .            # core only
pip install -e ".[dev]"     # with pytest / ruff
pip install -e ".[tda]"     # with gudhi / giotto-tda
```

Requires **Python >= 3.10**.

## Start here

**1.** Check your install:

```python
import she
print(she.__version__)  # 0.1.2
```

**2.** Run the smallest example:

```bash
python examples/toy_triangle.py
```

**3.** Run the main worked example (graph centrality vs simplex diffusion):

```bash
python examples/social_media_diffusers.py
```

**4.** Open the real-data notebook:

```
notebooks/eu_email_analysis.ipynb
```

For a full walkthrough, see [Getting Started](docs/tutorials/getting_started.md).

## Quickstart

```python
from she import SHEHyperstructure, rank_diffusers, find_bridge_simplices

# Build a decorated hyperstructure from interaction records
hs = SHEHyperstructure("demo")
hs.add_entity("alice", community="A")
hs.add_entity("bob",   community="A")
hs.add_entity("carol", community="B")

hs.add_relation(["alice", "bob"],          weight=1.0, kind="reply")
hs.add_relation(["alice", "bob", "carol"], weight=2.5, kind="co_amplification")

# Rank simplices by diffusion centrality (Hodge Laplacian, rank-percentile)
for r in rank_diffusers(hs, top_k=3):
    print(f"dim={r.dimension}  {r.target}  score={r.score:.3f}")

# Find simplices that span multiple communities (heuristic bridge score)
for b in find_bridge_simplices(hs):
    print(f"{sorted(b.members)}  communities={b.communities_spanned}")
```

## Examples at a glance

| Example | Type | What you learn |
|---------|------|---------------|
| `examples/toy_triangle.py` | Script | Hodge Laplacians on the smallest complex |
| `examples/social_group_lift.py` | Script | Graph-to-simplicial lifting via cliques |
| `examples/social_media_diffusers.py` | Script | Graph centrality vs simplex diffusion on two communities |
| `examples/group_diffusion_demo.py` | Script | Weighted Karate Club diffusion |
| `notebooks/temporal_diffusion_analysis.ipynb` | Notebook | Bridge formation over three time periods (synthetic) |
| `notebooks/eu_email_analysis.ipynb` | Notebook | Real SNAP email data, temporal bridge/cohesion plots |

## Experimental modules

The following are **not** part of the stable API and require extra
dependencies:

| Module | Requires | Install extra |
|--------|----------|---------------|
| `src/core/SHESimplicialConvolution.py` | PyTorch | `pip install she[ml]` |
| `src/morse/` | PyTorch, numba, sparse, psutil | `pip install she[morse]` |

## Documentation

| Doc | Purpose |
|-----|---------|
| [Getting Started](docs/tutorials/getting_started.md) | Install, first examples, onboarding path |
| [Social Diffusers Tutorial](docs/tutorials/social_diffusers.md) | Diffusers, bridges, cohesion, temporal |
| [API Overview](docs/api/overview.md) | Public API map with stable-vs-heuristic labels |
| [Social Media Use Case](docs/usecases/social_media_diffusers.md) | Conceptual framing |

## Limitations

This is a **Research Preview**.  The API may change between releases.

- Hodge analysis computes the **harmonic component** only; exact/coexact
  decomposition is not yet implemented.
- Bridge detection uses a heuristic (community-span weighted by relation
  weight), not a topological invariant.
- Group cohesion is a simple composite score, not a formal measure.
- Tested with TopoNetX 0.2.x on Python 3.11.
- Not OSI open-source — see License section below.

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

**Holomathics Non-Commercial License (HNCL) v1.0**

Free for personal, academic research, and educational use.
Commercial use requires a separate license from Holomathics.
See [LICENCE.md](LICENCE.md) for full text.

Contact: [info@holomathics.com](mailto:info@holomathics.com)
