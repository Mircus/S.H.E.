# API Overview

Public API surface of the `she` package (v0.1.2). All imports from `import she`.

## Modeling layer

### `SHEHyperstructure`

The central object. A decorated, weighted higher-order relational structure.

```python
hs = SHEHyperstructure("name", config=SHEConfig(max_dimension=2))
hs.add_entity("alice", community="A", role="researcher")
hs.add_relation(["alice", "bob"], weight=1.0, kind="reply", topic="energy")
```

Key methods:

| Method | Purpose |
|---|---|
| `add_entity(id, **attrs)` | Register an entity with arbitrary attributes |
| `add_relation(members, weight, kind, **meta)` | Add a weighted, typed relation |
| `from_records(records)` | Bulk build from list of dicts |
| `from_csv(path)` | Build from CSV file |
| `from_jsonl(path)` | Build from JSON-Lines file |
| `entities` | List of registered entity ids |
| `relations` | List of relation keys (frozensets) |
| `get_entity_attrs(id)` | Entity attribute dict |
| `get_relation_attrs(members)` | Relation attribute dict |
| `summary()` | Dimension counts and totals |
| `complex` | Access the underlying `SHESimplicialComplex` |

### `SHEConfig`

Configuration dataclass. Key fields: `max_dimension`, `spectral_k`.

## Social analysis

All functions take an `SHEHyperstructure` and return interpretable results.

| Function | Returns | Description |
|---|---|---|
| `rank_diffusers(hs, dimension, top_k)` | `List[RankedItem]` | Rank simplices by Hodge diffusion centrality |
| `rank_entity_diffusers(hs, top_k)` | `List[RankedItem]` | Shortcut for dimension=0 |
| `rank_simplex_diffusers(hs, dimension, top_k)` | `List[RankedItem]` | Shortcut for specific dimension |
| `find_bridge_simplices(hs, community_attr, top_k)` | `List[BridgeSimplex]` | Cross-community bridges (heuristic) |
| `group_cohesion(hs, group)` | `CohesionScore` | Structural cohesion composite |
| `rank_influencers(hs, top_k)` | `dict` | Graph centrality vs simplex diffusion |

### Result types

- **`RankedItem`** — `target`, `dimension`, `score`, `metadata`
- **`BridgeSimplex`** — `members`, `dimension`, `communities_spanned`, `bridge_score`, `metadata`
- **`CohesionScore`** — `members`, `score`, `components`

## Temporal

| Function | Description |
|---|---|
| `window(hs, start, end)` | Hard time-range filter, returns new hyperstructure |
| `rolling_windows(hs, window_size, step)` | Sequence of `(start, end, hs)` snapshots |
| `decay_window(hs, reference_time, half_life)` | Exponential decay weighting |

## Export

| Function | Description |
|---|---|
| `ranked_items_to_csv(items, path)` | Export ranked items to CSV |
| `ranked_items_to_json(items, path)` | Export ranked items to JSON |
| `bridges_to_csv(bridges, path)` | Export bridge simplices to CSV |
| `bridges_to_json(bridges, path)` | Export bridge simplices to JSON |
| `cohesion_to_csv(scores, path)` | Export cohesion scores to CSV |
| `cohesion_to_json(scores, path)` | Export cohesion scores to JSON |

All export functions return the string and optionally write to `path`.

## Core engine (lower-level)

These are available but most users should work through `SHEHyperstructure`.

| Class | Purpose |
|---|---|
| `SHESimplicialComplex` | Wraps TopoNetX `SimplicialComplex` |
| `SHEHodgeDiffusion` | Hodge Laplacian computation and diffusion solve |
| `SHEDataLoader` | Graph-to-simplicial lifting |
| `SHEDiffusionVisualizer` | Matplotlib plots |
| `SHEEngine` | Convenience wrapper |
| `DiffusionResult` | Result container from diffusion analysis |

## What is stable vs heuristic

| Feature | Status |
|---|---|
| Hyperstructure construction, ingestion, temporal | Stable for this release |
| Hodge Laplacian computation | Stable (harmonic component only) |
| Diffusion centrality ranking | Stable method, heuristic interpretation |
| Bridge detection | Heuristic score, not a topological invariant |
| Group cohesion | Simple composite, not a formal measure |
| Export | Stable |
