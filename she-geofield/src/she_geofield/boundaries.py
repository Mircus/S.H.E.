"""Oriented boundary matrices for simplicial complexes."""

import numpy as np
from .toy_complex import WeightedToyComplex


def boundary_1_to_0(complex_: WeightedToyComplex) -> np.ndarray:
    """Oriented boundary operator d1: C_1 -> C_0.

    For edge (u, v) with u < v:  d1[v_idx, e_idx] = +1,  d1[u_idx, e_idx] = -1.
    """
    vtx_index = {v: i for i, v in enumerate(complex_.vertices)}
    n0 = len(complex_.vertices)
    n1 = len(complex_.edges)
    B = np.zeros((n0, n1), dtype=float)
    for j, (u, v) in enumerate(complex_.edges):
        B[vtx_index[u], j] = -1.0
        B[vtx_index[v], j] = +1.0
    return B


def boundary_2_to_1(complex_: WeightedToyComplex) -> np.ndarray:
    """Oriented boundary operator d2: C_2 -> C_1.

    For triangle (a, b, c) with a < b < c:
        d2 = +[b,c] - [a,c] + [a,b]
    Signs are (-1)^i where i is the index of the omitted vertex.
    """
    edge_index = {e: i for i, e in enumerate(complex_.edges)}
    n1 = len(complex_.edges)
    n2 = len(complex_.triangles)
    B = np.zeros((n1, n2), dtype=float)
    for j, (a, b, c) in enumerate(complex_.triangles):
        # face opposite vertex 0 (a): edge (b,c), sign (-1)^0 = +1
        B[edge_index[tuple(sorted((b, c)))], j] = +1.0
        # face opposite vertex 1 (b): edge (a,c), sign (-1)^1 = -1
        B[edge_index[tuple(sorted((a, c)))], j] = -1.0
        # face opposite vertex 2 (c): edge (a,b), sign (-1)^2 = +1
        B[edge_index[tuple(sorted((a, b)))], j] = +1.0
    return B
