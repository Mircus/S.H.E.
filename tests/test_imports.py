"""Verify that the public API imports cleanly."""


def test_import_she():
    import she
    assert hasattr(she, "__version__")


def test_import_public_symbols():
    from she import (
        SHEConfig,
        DiffusionResult,
        SHESimplicialComplex,
        SHEDataLoader,
        SHEHodgeDiffusion,
        SHEDiffusionVisualizer,
        SHEEngine,
    )
    # basic sanity — classes are callable
    assert callable(SHEConfig)
    assert callable(SHESimplicialComplex)
    assert callable(SHEDataLoader)
    assert callable(SHEHodgeDiffusion)
    assert callable(SHEDiffusionVisualizer)
    assert callable(SHEEngine)
