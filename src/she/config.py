"""SHE configuration."""

from dataclasses import dataclass


@dataclass
class SHEConfig:
    """Configuration for the SHE engine.

    All fields have sensible defaults so ``SHEConfig()`` is enough for most
    use-cases.  The ``device`` and ``dtype`` fields are kept as plain strings
    so that the core package does **not** depend on PyTorch at import time.
    """

    device: str = "cpu"
    max_dimension: int = 2
    use_cache: bool = True
    batch_size: int = 32
    persistent_homology_backend: str = "gudhi"
    diffusion_steps: int = 100
    diffusion_dt: float = 0.01
    spectral_k: int = 10
