from collections import Counter, defaultdict

import numpy as np

from ..flow import fixed_geometry_step
from .records import DblpSimplicialComplex, PaperRecord


def author_productivity(records: tuple[PaperRecord, ...]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.authors)
    return {author: float(count) for author, count in counts.items()}


def author_weighted_degree(complex_: DblpSimplicialComplex) -> dict[str, float]:
    degree = defaultdict(float)
    for u, v in complex_.edges:
        weight = complex_.edge_weights[(u, v)]
        degree[u] += weight
        degree[v] += weight
    return dict(degree)


def edge_scores_from_node_scores(
    complex_: DblpSimplicialComplex,
    node_scores: dict[str, float],
) -> dict[tuple[str, str], float]:
    return {
        edge: 0.5 * (node_scores.get(edge[0], 0.0) + node_scores.get(edge[1], 0.0))
        for edge in complex_.edges
    }


def triangle_scores_from_node_scores(
    complex_: DblpSimplicialComplex,
    node_scores: dict[str, float],
) -> dict[tuple[str, str, str], float]:
    return {
        triangle: sum(node_scores.get(vertex, 0.0) for vertex in triangle) / 3.0
        for triangle in complex_.triangles
    }


def project_edge_scores_to_triangles(
    complex_: DblpSimplicialComplex,
    edge_scores: dict[tuple[str, str], float],
) -> dict[tuple[str, str, str], float]:
    projected: dict[tuple[str, str, str], float] = {}
    for triangle in complex_.triangles:
        a, b, c = triangle
        edges = [tuple(sorted((a, b))), tuple(sorted((a, c))), tuple(sorted((b, c)))]
        projected[triangle] = sum(edge_scores.get(edge, 0.0) for edge in edges) / 3.0
    return projected


def graph_bridge_scores(complex_: DblpSimplicialComplex) -> dict[tuple[str, str], float]:
    neighbors: dict[str, set[str]] = {vertex: set() for vertex in complex_.vertices}
    for u, v in complex_.edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    scores: dict[tuple[str, str], float] = {}
    for u, v in complex_.edges:
        shared = len(neighbors[u] & neighbors[v])
        scores[(u, v)] = complex_.edge_weights[(u, v)] / (1.0 + shared)
    return scores


def simplex_support_scores(complex_: DblpSimplicialComplex, *, location: str = "edges") -> dict:
    simplices = complex_.edges if location == "edges" else complex_.triangles
    return {
        simplex: float(complex_.simplex_data.get(simplex, {}).get("support_count", 0.0))
        for simplex in simplices
    }


def persistence_scores(complex_: DblpSimplicialComplex, *, location: str = "edges") -> dict:
    simplices = complex_.edges if location == "edges" else complex_.triangles
    return {
        simplex: float(complex_.simplex_data.get(simplex, {}).get("persistence", 0.0))
        for simplex in simplices
    }


def frozen_diffusion_scores(
    complex_: DblpSimplicialComplex,
    x0: np.ndarray,
    *,
    dt: float,
    steps: int,
    location: str = "edges",
) -> dict[tuple[str, str], float]:
    x = x0.copy()
    for _ in range(steps):
        x = fixed_geometry_step(complex_, x, dt=dt, location=location)
    simplices = complex_.edges if location == "edges" else complex_.triangles
    return {simplex: float(x[idx]) for idx, simplex in enumerate(simplices)}
