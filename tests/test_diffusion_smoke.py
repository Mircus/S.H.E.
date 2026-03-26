"""Smoke test: diffusion analysis runs and returns expected types."""

import numpy as np

from she import SHEConfig, SHESimplicialComplex, SHEHodgeDiffusion, DiffusionResult


def _make_small_complex():
    config = SHEConfig(max_dimension=2, spectral_k=3)
    sc = SHESimplicialComplex("small", config=config)
    for n in range(4):
        sc.add_node(n)
    for e in [(0, 1), (1, 2), (2, 3), (0, 2), (0, 3)]:
        sc.add_edge(e)
    sc.add_simplex([0, 1, 2])
    sc.add_simplex([0, 2, 3])
    return sc, config


def test_analyze_diffusion_returns_result():
    sc, config = _make_small_complex()
    analyzer = SHEHodgeDiffusion(config)
    result = analyzer.analyze_diffusion(sc)
    assert isinstance(result, DiffusionResult)


def test_eigenvalues_non_empty():
    sc, config = _make_small_complex()
    analyzer = SHEHodgeDiffusion(config)
    result = analyzer.analyze_diffusion(sc)
    assert len(result.eigenvalues) > 0
    for dim, ev in result.eigenvalues.items():
        assert isinstance(ev, np.ndarray)
        assert ev.size > 0


def test_key_diffusers_non_empty():
    sc, config = _make_small_complex()
    analyzer = SHEHodgeDiffusion(config)
    result = analyzer.analyze_diffusion(sc)
    assert len(result.key_diffusers) > 0
