
@dataclass
class SHEConfig:
    """Configuration for SHE engine"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    max_dimension: int = 3
    use_cache: bool = True
    batch_size: int = 32
    persistent_homology_backend: str = "giotto"  # "giotto", "gudhi", or "ripser"
    diffusion_steps: int = 100
    diffusion_dt: float = 0.01
    spectral_k: int = 10  # Number of eigenvalues/eigenvectors to compute
