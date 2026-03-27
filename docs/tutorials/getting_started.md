# Getting Started with SHE

## Installation

```bash
git clone https://github.com/Mircus/S.H.E.
cd S.H.E
pip install -e .          # core only
pip install -e ".[dev]"   # adds pytest, ruff
```

Requires **Python >= 3.10**. Tested with TopoNetX 0.2.x on Python 3.11.

## First import

```python
import she
print(she.__version__)  # 0.1.2
```

The main objects you will use:

| Object | Purpose |
|---|---|
| `SHEHyperstructure` | Build a decorated higher-order relational structure |
| `rank_diffusers` | Rank simplices by Hodge-Laplacian diffusion centrality |
| `find_bridge_simplices` | Find simplices spanning multiple communities |
| `group_cohesion` | Score structural cohesion of a candidate group |
| `rolling_windows` / `decay_window` | Temporal slicing and decay-weighted views |
| `ranked_items_to_csv` | Export results to CSV or JSON |

## First example: toy triangle

Run the smallest nontrivial complex:

```bash
python examples/toy_triangle.py
```

This builds a triangle (three nodes, three edges, one face), computes Hodge
Laplacians at each dimension, and prints them. If this runs, your install works.

## Second example: social diffusers

```bash
python examples/social_media_diffusers.py
```

This builds a two-community social scenario and compares graph centrality
(eigenvector on the 1-skeleton) with simplex-level analysis:

- **Graph centrality** ranks a high-degree hub first.
- **Bridge detection** surfaces a cross-community triad as the top bridge.
- **Group cohesion** scores the triad as structurally tight.

These are heuristic scores. The point is that graph-only metrics never see
group-level structures, while SHE makes them queryable.

## Third: real-data notebook

Open `examples/eu_email_analysis.ipynb` — it analyses the SNAP EU Email
network (20 researchers, 812 interactions, 27 months) with:

- Louvain community detection
- Temporal bridge/cohesion tracking over rolling windows
- Decay-weighted windowing
- Matplotlib plots of bridge scores and cohesion over time

## What to read next

- [Social Diffusers Tutorial](social_diffusers.md) — deeper look at the
  social-analysis features
- [API Overview](../api/overview.md) — public API map
- [Social Media Use Case](../usecases/social_media_diffusers.md) — the
  conceptual framing behind the diffusers example
