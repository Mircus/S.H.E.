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
        decay_window,
        ranked_items_to_csv,
        ranked_items_to_json,
        bridges_to_csv,
        bridges_to_json,
        cohesion_to_csv,
        cohesion_to_json,
    )
    # basic sanity — classes are callable
    assert callable(SHEConfig)
    assert callable(SHESimplicialComplex)
    assert callable(SHEDataLoader)
    assert callable(SHEHodgeDiffusion)
    assert callable(SHEDiffusionVisualizer)
    assert callable(SHEEngine)
    assert callable(decay_window)
    assert callable(ranked_items_to_csv)
