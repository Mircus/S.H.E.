
from dataclasses import dataclass
from typing import Optional
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

@dataclass
class SHEConfig:
    """Configuration for SHE engine"""
    device: str = "cuda" if (_TORCH_AVAILABLE and getattr(__import__('torch'), 'cuda').is_available()) else "cpu"
    dtype: "torch.dtype" = getattr(__import__('torch'), 'float32', None) if _TORCH_AVAILABLE else None
    max_dimension: int = 2  # keep up to triangles by default
    use_cache: bool = True
    batch_size: int = 32
    persistent_homology_backend: str = "gudhi"  # "giotto", "gudhi", or "ripser"
    diffusion_steps: int = 100
    diffusion_dt: float = 0.01
    spectral_k: int = 10  # Number of eigenvalues/eigenvectors to compute
