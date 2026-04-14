import numpy as np
from she_geofield.boundaries import boundary_1_to_0, boundary_2_to_1
from she_geofield.toy_complex import build_toy_complex
from she_geofield.hodge import hodge_laplacian_1


def test_hodge_shape_and_diagonal_nonnegative():
    c = build_toy_complex()
    L = hodge_laplacian_1(c)
    assert L.shape == (len(c.edges), len(c.edges))
    assert np.all(np.diag(L) >= 0)


def test_boundary_chain_complex_identity():
    c = build_toy_complex()
    B10 = boundary_1_to_0(c)
    B21 = boundary_2_to_1(c)

    assert np.allclose(B10 @ B21, 0.0)


def test_hodge_is_symmetric_positive_semidefinite():
    c = build_toy_complex()
    L = hodge_laplacian_1(c)

    assert np.allclose(L, L.T)
    assert np.all(np.linalg.eigvalsh(L) >= -1e-12)
