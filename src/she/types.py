"""Shared data containers for SHE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DiffusionResult:
    """Results from diffusion analysis."""

    eigenvalues: Dict[int, np.ndarray]
    eigenvectors: Dict[int, np.ndarray]
    diffusion_maps: Dict[str, np.ndarray]
    key_diffusers: Dict[int, List[Tuple[Any, float]]]
    hodge_decomposition: Dict[str, np.ndarray]
    heat_kernel: Optional[np.ndarray] = None
