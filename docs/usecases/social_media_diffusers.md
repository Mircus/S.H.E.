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

- **Graph centrality** highlights a high-degree hub node.
- **Simplex diffusion** highlights the cross-community triad as the actual
  diffusion bottleneck, because information must pass through that group
  structure to bridge the communities.

The triad may contain no individually prominent member, yet it dominates the
diffusion pathway.  SHE makes this visible.
