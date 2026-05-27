from math import isclose

import pytest

pytest.importorskip("she_geofield.dblp.experiments")
from seldon_lab.experiments.dblp_geometry import compute_simplex_observables
records = pytest.importorskip("she_geofield.dblp.records")
DblpSimplicialComplex = records.DblpSimplicialComplex


def test_triangle_reinforcement_and_curvature_are_positive_when_triangle_exceeds_edges():
    complex_ = DblpSimplicialComplex(
        vertices=["a", "b", "c"],
        edges=[("a", "b"), ("a", "c"), ("b", "c")],
        triangles=[("a", "b", "c")],
        vertex_weights={"a": 2.0, "b": 2.0, "c": 2.0},
        edge_weights={("a", "b"): 3.0, ("a", "c"): 3.0, ("b", "c"): 3.0},
        triangle_weights={("a", "b", "c"): 6.0},
    )
    obs = compute_simplex_observables(complex_, ("a", "b", "c"))
    assert isclose(obs["reinforcement"], 2.0, rel_tol=1e-6)
    assert obs["boundary_strain"] == 0.0
    assert obs["curvature_balance"] > 0.0


def test_edge_curvature_turns_negative_when_upward_pull_dominates():
    complex_ = DblpSimplicialComplex(
        vertices=["a", "b", "c"],
        edges=[("a", "b"), ("a", "c"), ("b", "c")],
        triangles=[("a", "b", "c")],
        vertex_weights={"a": 4.0, "b": 4.0, "c": 4.0},
        edge_weights={("a", "b"): 2.0, ("a", "c"): 4.0, ("b", "c"): 4.0},
        triangle_weights={("a", "b", "c"): 8.0},
    )
    obs = compute_simplex_observables(complex_, ("a", "b"))
    assert obs["reinforcement"] < 1.0
    assert obs["boundary_strain"] > 1.0
    assert obs["curvature_balance"] < 0.0
