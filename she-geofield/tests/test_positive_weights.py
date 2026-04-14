from she_geofield.toy_complex import build_toy_complex
from she_geofield.flow import geometry_step


def test_positive_weights_are_preserved_for_large_eta():
    c = build_toy_complex()
    for _ in range(5):
        geometry_step(c, eta=10.0, lam=0.2, mu=0.5)
        assert all(w > 0 for w in c.triangle_weights.values())
