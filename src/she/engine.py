"""High-level SHE engine that ties analysis components together."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .config import SHEConfig
from .complex import SHESimplicialComplex
from .diffusion import SHEHodgeDiffusion
from .types import DiffusionResult
from .visualize import SHEDiffusionVisualizer

logger = logging.getLogger(__name__)


class SHEEngine:
    """Convenience wrapper: register complexes, run diffusion, visualise."""

    def __init__(self, config: Optional[SHEConfig] = None):
        self.config = config or SHEConfig()
        self.complexes: Dict[str, SHESimplicialComplex] = {}
        self.diffusion_analyzer = SHEHodgeDiffusion(self.config)
        self.visualizer = SHEDiffusionVisualizer()
        logger.info("SHE Engine initialised (device=%s)", self.config.device)

    def register_complex(self, name: str, sc: SHESimplicialComplex) -> None:
        self.complexes[name] = sc
        logger.info("Registered complex: %s", name)

    def analyze_diffusion(self, complex_name: str) -> DiffusionResult:
        if complex_name not in self.complexes:
            raise ValueError(f"Complex '{complex_name}' not found")
        return self.diffusion_analyzer.analyze_diffusion(self.complexes[complex_name])

    def find_key_diffusers(
        self, complex_name: str, dimension: int = 0, top_k: int = 10
    ) -> List[Tuple[Any, float]]:
        result = self.analyze_diffusion(complex_name)
        return result.key_diffusers.get(dimension, [])[:top_k]

    def visualize_diffusion(self, complex_name: str, dimension: int = 0) -> None:
        result = self.analyze_diffusion(complex_name)
        self.visualizer.plot_spectrum(result, dimension, f"{complex_name} - dim {dimension}")
        self.visualizer.plot_key_diffusers(result, dimension)
        self.visualizer.plot_diffusion_heatmap(result, dimension)
