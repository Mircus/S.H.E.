"""Deterministic tests on a toy triangle complex."""

from she import SHEConfig, SHESimplicialComplex


def _make_triangle():
    config = SHEConfig(max_dimension=2)
    sc = SHESimplicialComplex("tri", config=config)
    for n in [0, 1, 2]:
        sc.add_node(n)
    for e in [(0, 1), (1, 2), (0, 2)]:
        sc.add_edge(e)
    sc.add_simplex([0, 1, 2])
    return sc


def test_dimension():
    sc = _make_triangle()
    assert sc.complex.dim == 2


def test_simplex_counts():
    sc = _make_triangle()
    assert len(sc.get_simplex_list(0)) == 3   # nodes
    assert len(sc.get_simplex_list(1)) == 3   # edges
    assert len(sc.get_simplex_list(2)) == 1   # face


def test_hodge_laplacians_exist():
    sc = _make_triangle()
    laps = sc.get_hodge_laplacians()
    assert 0 in laps
    assert 1 in laps
    assert laps[0].shape[0] == 3
