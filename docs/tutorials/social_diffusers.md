# Social Diffusers, Bridges, and Cohesion

This tutorial explains SHE's social-analysis layer: what the scores mean,
how they differ from graph centrality, and how temporal features fit in.

## What is a diffuser in SHE?

A **diffuser** is a simplex (node, edge, triangle, ...) ranked by its
structural importance for diffusion on the Hodge Laplacian.

SHE solves `(I + L) x = w` where `L` is the Hodge Laplacian at a given
dimension and `w` is the weight vector. The solution `x` is then
rank-percentile normalised to produce scores in [0, 1].

- **Dimension 0:** ranks individual entities (nodes).
- **Dimension 1:** ranks pairwise relations (edges).
- **Dimension 2:** ranks triadic relations (triangles).

This is **not** the same as graph centrality. Eigenvector centrality on the
1-skeleton ranks nodes by pairwise connectivity. Hodge diffusion centrality
ranks simplices at any dimension by their role in the simplicial diffusion
process. They can disagree substantially.

## What is a bridge simplex?

A **bridge simplex** is a relation whose members span multiple communities.

SHE's bridge score is a heuristic:

```
bridge_score = (number of communities spanned / number of members) * weight
```

This rewards simplices that span many communities with few members and high
weight. It is intentionally simple — a legible first-pass signal, not a
topological invariant.

Community membership comes from entity attributes (the `community` field).
You assign these when building the hyperstructure.

## What is group cohesion?

`group_cohesion(hs, group)` scores how tightly a candidate group is bound
in the hyperstructure. It combines three signals:

1. **Relation weight** — total weight of relations fully contained in the
   group, normalised by group size.
2. **Sub-relation density** — fraction of possible pairs within the group
   that actually exist as relations.
3. **Higher-order support** — number of relations of dimension >= 2 fully
   contained in the group, normalised by group size.

The final score is the geometric mean of these three components. This is a
deliberately simple composite, not a formal topological measure.

## Temporal features

### Hard windows

`window(hs, start, end)` returns a new hyperstructure containing only
relations within the time range `[start, end)`. Entity attributes are
preserved for all entities that appear in the window.

`rolling_windows(hs, window_size, step)` produces a sequence of these
snapshots.

### Decay-weighted windows

`decay_window(hs, reference_time, half_life)` builds a snapshot where
relation weights decay exponentially with age:

```
decayed_weight = original_weight * 2^(-age / half_life)
```

Relations decayed below 1% of their original weight are dropped. This is
more realistic than a hard cutoff — recent interactions dominate, but older
ones still contribute.

## Which example demonstrates this?

- **Script:** `examples/social_media_diffusers.py` — synthetic scenario,
  graph vs simplex comparison
- **Notebook:** `notebooks/eu_email_analysis.ipynb` — real email data,
  temporal bridge/cohesion plots
- **Notebook:** `notebooks/temporal_diffusion_analysis.ipynb` — synthetic
  temporal scenario showing bridge formation over three periods

## Caveats

- Bridge detection and cohesion are heuristic scores, not topological
  invariants. They are useful first-pass signals.
- Diffusion centrality depends on the Hodge Laplacian structure and weight
  distribution. On very small complexes, rankings can be sensitive to weight
  tuning.
- These tools are research-grade. The API may change.
