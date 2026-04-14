import numpy as np
from she_geofield.toy_complex import build_toy_complex
from she_geofield.flow import coupled_step


def test_coupled_step_shapes():
    c = build_toy_complex()
    x = np.zeros(len(c.edges))
    x[0] = 1.0
    y = coupled_step(c, x)
    assert y.shape == x.shape


def test_coupled_step_with_field_feedback_changes_triangle_weights():
    c = build_toy_complex()
    x = np.zeros(len(c.edges))
    x[0] = 1.0

    before = dict(c.triangle_weights)
    coupled_step(c, x, mu=0.5)

    assert any(c.triangle_weights[tri] != before[tri] for tri in c.triangles)
