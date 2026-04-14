from dataclasses import dataclass, field
from typing import Dict, List, Tuple

Edge = Tuple[str, str]
Triangle = Tuple[str, str, str]

def _sorted_edge(a: str, b: str) -> Edge:
    return tuple(sorted((a, b)))

def _sorted_tri(a: str, b: str, c: str) -> Triangle:
    return tuple(sorted((a, b, c)))

@dataclass
class WeightedToyComplex:
    vertices: List[str]
    edges: List[Edge]
    triangles: List[Triangle]
    vertex_weights: Dict[str, float]
    edge_weights: Dict[Edge, float]
    triangle_weights: Dict[Triangle, float]
    triangle_cohesion: Dict[Triangle, float] = field(default_factory=dict)

def build_toy_complex() -> WeightedToyComplex:
    vertices = ["a", "b", "c", "d", "e", "f"]
    triangles = [
        _sorted_tri("a", "b", "c"),
        _sorted_tri("c", "d", "e"),
        _sorted_tri("c", "e", "f"),
    ]
    edge_set = set()
    for t in triangles:
        x, y, z = t
        edge_set.add(_sorted_edge(x, y))
        edge_set.add(_sorted_edge(x, z))
        edge_set.add(_sorted_edge(y, z))
    edges = sorted(edge_set)
    vertex_weights = {v: 1.0 for v in vertices}
    edge_weights = {e: 1.0 for e in edges}
    triangle_weights = {
        _sorted_tri("a", "b", "c"): 1.2,
        _sorted_tri("c", "d", "e"): 0.9,
        _sorted_tri("c", "e", "f"): 1.4,
    }
    triangle_cohesion = {
        _sorted_tri("a", "b", "c"): 0.5,
        _sorted_tri("c", "d", "e"): 0.3,
        _sorted_tri("c", "e", "f"): 0.8,
    }
    return WeightedToyComplex(
        vertices=vertices,
        edges=edges,
        triangles=triangles,
        vertex_weights=vertex_weights,
        edge_weights=edge_weights,
        triangle_weights=triangle_weights,
        triangle_cohesion=triangle_cohesion,
    )
