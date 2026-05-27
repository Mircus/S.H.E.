"""Weighted Hodge Laplacians on edges and triangles.

L_1 = d1* d1 + d2 d2*

L_2 = d2* d2

where d_k* = W_k^{-1} d_k^T W_{k-1} is the weighted adjoint.

Expanded:
  L_1_down = W_1^{-1} d1^T W_0 d1       (vertex-mediated)
  L_1_up   = d2 W_2^{-1} d2^T W_1        (triangle-mediated)
  L_1      = L_1_down + L_1_up
"""

import numpy as np
from .boundaries import boundary_1_to_0, boundary_2_to_1
from .toy_complex import WeightedToyComplex


def hodge_laplacian_1(complex_: WeightedToyComplex) -> np.ndarray:
    """Weighted Hodge Laplacian on 1-chains (edges)."""
    B10 = boundary_1_to_0(complex_)
    B21 = boundary_2_to_1(complex_)

    w0 = np.array([complex_.vertex_weights[v] for v in complex_.vertices], dtype=float)
    w1 = np.array([complex_.edge_weights[e] for e in complex_.edges], dtype=float)
    w2 = np.array([complex_.triangle_weights[t] for t in complex_.triangles], dtype=float)

    W0 = np.diag(w0)
    W1_inv = np.diag(1.0 / w1)
    W1 = np.diag(w1)
    W2_inv = np.diag(1.0 / w2)

    # L_1_down = W_1^{-1} d1^T W_0 d1
    L1_down = W1_inv @ B10.T @ W0 @ B10

    # L_1_up = d2 W_2^{-1} d2^T W_1
    L1_up = B21 @ W2_inv @ B21.T @ W1

    return L1_down + L1_up


def hodge_laplacian_2(complex_: WeightedToyComplex) -> np.ndarray:
    """Weighted Hodge Laplacian on 2-chains (triangles).

    With no tetrahedra in the current model, this is the down-Laplacian
    L_2 = d2* d2 = W_2^{-1} d2^T W_1 d2.
    """
    B21 = boundary_2_to_1(complex_)

    w1 = np.array([complex_.edge_weights[e] for e in complex_.edges], dtype=float)
    w2 = np.array([complex_.triangle_weights[t] for t in complex_.triangles], dtype=float)

    W1 = np.diag(w1)
    W2_inv = np.diag(1.0 / w2)

    return W2_inv @ B21.T @ W1 @ B21
