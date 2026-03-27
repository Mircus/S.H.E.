# Use case: simplicial diffusers in social media

## Problem

Standard social-network analysis ranks individuals by graph centrality
(degree, betweenness, eigenvector).  This misses a common real-world
phenomenon: **small groups that co-amplify content are often more important
for information diffusion than any single prominent individual**.

A triad of mid-reach accounts that consistently co-retweet or co-engage on a
topic can act as a diffusion engine that a node-level centrality measure will
never surface, because the signal lives in the *group structure*, not in any
one member's connectivity.

## Graph baseline limitation

In a standard graph:

- Each interaction is collapsed to a pairwise edge.
- A triad {A, B, C} that always acts together is indistinguishable from three
  independent pairwise interactions A-B, B-C, A-C.
- Centrality metrics rank nodes; there is no object corresponding to the group.

## Event-to-hyperstructure construction

SHE lifts interaction records into a **weighted simplicial complex** where:

- Each entity (account, user) is a 0-simplex.
- Each pairwise interaction is a 1-simplex (edge) with weight.
- Each co-engagement group of size k is a (k-1)-simplex with its own weight,
  kind label, and metadata (topic, timestamp window, ...).

This preserves the *higher-order grouping* that graph projection destroys.

## Analysis outputs

Given this structure, SHE computes:

1. **Simplex-level diffusion centrality** via the Hodge Laplacian, ranking
   groups (not just individuals) by their structural role in diffusion.
2. **Bridge simplices** that span community boundaries.
3. **Group cohesion scores** measuring how tightly a candidate group is bound.
4. **Graph vs. simplex ranking comparison** showing where the two disagree.

## What higher-order signal we expect to uncover

In a scenario with two communities and one cross-community triad:

- **Graph centrality** (eigenvector on the 1-skeleton) highlights a high-degree
  hub node — the member with the most and heaviest pairwise edges.
- **Bridge detection** (heuristic: community-span x relation weight) highlights
  the cross-community triad as the top bridge structure, because it spans both
  communities with high group weight.
- **Group cohesion** (geometric mean of internal weight, pair density, and
  higher-order support) scores the triad as structurally tight despite
  containing no individually prominent member.

These are heuristic scores, not topological invariants.  The point is that
graph-only centrality never sees group-level structures; SHE makes them
queryable and trackable over time.
