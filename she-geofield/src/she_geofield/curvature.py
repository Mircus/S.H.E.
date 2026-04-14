"""Weighted Forman-type discrete curvature for edges and triangles.

Edge curvature (Forman-Ricci on 1-cells, diagnostic):
    F(e) = w(u)/w(e) + w(v)/w(e) - sum_{t > e} w(e)/w(t)

Triangle curvature (drives geometry evolution, field-coupled):
    G(t) = w(t)/mean(w_e) - 1 + lambda*cohesion - mu*E(t)

where E(t) is the normalized field energy on the boundary edges of t.
The field feedback term -mu*E(t) means triangles carrying more signal
have lower effective curvature, so their weight grows: active groups
strengthen.  This is the genuine geometry-field coupling.
"""

import numpy as np
from .toy_complex import WeightedToyComplex, _sorted_edge


def edge_forman_curvature(complex_: WeightedToyComplex, edge):
    """Weighted Forman-Ricci curvature for a 1-cell (edge).

    Diagnostic quantity; does not drive the flow.
    """
    u, v = edge
    we = complex_.edge_weights[edge]
    wu = complex_.vertex_weights[u]
    wv = complex_.vertex_weights[v]
    incident = [t for t in complex_.triangles if u in t and v in t]
    boundary_term = wu / we + wv / we
    coboundary_term = sum(we / complex_.triangle_weights[t] for t in incident)
    return boundary_term - coboundary_term


def triangle_curvature(complex_: WeightedToyComplex, triangle,
                       x1: np.ndarray = None, lam: float = 0.5,
                       mu: float = 0.0):
    """Curvature functional driving triangle-weight evolution.

    G(t) = w(t)/mean(w_e) - 1 + lambda*cohesion - mu*E(t)

    where E(t) = mean(|x_e| for e in boundary(t)) / max(|x|, eps).

    The field feedback -mu*E(t) lowers curvature on active triangles,
    causing their weight to grow.  mu=0 recovers the uncoupled case.
    """
    a, b, c = triangle
    wt = complex_.triangle_weights[triangle]
    edges_of_tri = [_sorted_edge(a, b), _sorted_edge(a, c), _sorted_edge(b, c)]
    mean_we = sum(complex_.edge_weights[e] for e in edges_of_tri) / 3.0
    cohesion = complex_.triangle_cohesion.get(triangle, 0.0)

    F = wt / mean_we - 1.0 + lam * cohesion

    # Field feedback: active triangles are reinforced
    if x1 is not None and mu > 0.0:
        edge_list = list(complex_.edges)
        edge_idx = [edge_list.index(e) for e in edges_of_tri]
        boundary_energy = np.mean(np.abs(x1[edge_idx]))
        total_energy = np.max(np.abs(x1)) + 1e-15
        E_t = boundary_energy / total_energy
        F -= mu * E_t

    return F
