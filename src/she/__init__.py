"""SHE -- Simplicial Hyperstructure Engine (v0.1 Research Preview).

Public API
----------
>>> from she import SHEConfig, SHESimplicialComplex, SHEDataLoader
>>> from she import SHEHodgeDiffusion, DiffusionResult
>>> from she import SHEEngine
"""

from .config import SHEConfig
from .types import DiffusionResult
from .complex import SHESimplicialComplex
from .io import SHEDataLoader
from .diffusion import SHEHodgeDiffusion
from .visualize import SHEDiffusionVisualizer
from .engine import SHEEngine

__version__ = "0.1.0"

__all__ = [
    "SHEConfig",
    "DiffusionResult",
    "SHESimplicialComplex",
    "SHEDataLoader",
    "SHEHodgeDiffusion",
    "SHEDiffusionVisualizer",
    "SHEEngine",
]
