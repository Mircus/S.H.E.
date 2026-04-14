import numpy as np

from she_geofield.toy_complex import build_toy_complex
from she_geofield.curvature import edge_forman_curvature, triangle_curvature


def test_curvature_returns_finite_values():
    c = build_toy_complex()
    for e in c.edges:
        assert isinstance(edge_forman_curvature(c, e), float)
    for t in c.triangles:
        assert isinstance(triangle_curvature(c, t), float)


def test_field_feedback_lowers_triangle_curvature():
    c = build_toy_complex()
    edge_index = {edge: idx for idx, edge in enumerate(c.edges)}
    tri = c.triangles[0]
    x = np.zeros(len(c.edges))
    for edge in [("a", "b"), ("a", "c"), ("b", "c")]:
        x[edge_index[edge]] = 1.0

    uncoupled = triangle_curvature(c, tri, x1=x, lam=0.5, mu=0.0)
    coupled = triangle_curvature(c, tri, x1=x, lam=0.5, mu=0.5)

    assert coupled < uncoupled
