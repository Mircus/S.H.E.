from __future__ import annotations
from typing import Dict, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from ..config.SheConfig import SHEConfig
from ..complex.SHESimplicialComplex import SHESimplicialComplex
from ..diffusion.SHEHodgeDiffusion import SHEHodgeDiffusion
from ..visualize.SHEDiffusionVisualizer import SHEDiffusionVisualizer

logger = logging.getLogger(__name__)




class SHEEngine:
    """
    Enhanced SHE Engine with advanced diffusion analysis
    """
    
    def __init__(self, config: Optional[SHEConfig] = None):
        self.config = config or SHEConfig()
        self.complexes: Dict[str, SHESimplicialComplex] = {}
        self.diffusion_analyzer = SHEHodgeDiffusion(self.config)
        self.visualizer = SHEDiffusionVisualizer()
        
        logger.info(f"Enhanced SHE Engine initialized with device: {self.config.device}")
    
    def register_complex(self, name: str, complex: SHESimplicialComplex):
        """Register a simplicial complex"""
        self.complexes[name] = complex
        logger.info(f"Registered complex: {name}")
    
    def analyze_diffusion(self, complex_name: str) -> DiffusionResult:
        """Perform comprehensive diffusion analysis"""
        if complex_name not in self.complexes:
            raise ValueError(f"Complex {complex_name} not found")
        
        complex = self.complexes[complex_name]
        logger.info(f"Starting diffusion analysis for {complex_name}")
        
        return self.diffusion_analyzer.analyze_diffusion(complex)
    
    def find_key_diffusers(self, complex_name: str, dimension: int = 0, 
                          top_k: int = 10) -> List[Tuple[Any, float]]:
        """Find key diffusers for a specific dimension"""
        diffusion_result = self.analyze_diffusion(complex_name)
        
        if dimension in diffusion_result.key_diffusers:
            return diffusion_result.key_diffusers[dimension][:top_k]
        else:
            return []
    
    def visualize_diffusion(self, complex_name: str, dimension: int = 0):
        """Create comprehensive diffusion visualizations"""
        diffusion_result = self.analyze_diffusion(complex_name)
        
        # Plot spectrum
        self.visualizer.plot_spectrum(diffusion_result, dimension, 
                                    f"{complex_name} - Dimension {dimension} Spectrum")
        
        # Plot key diffusers
        self.visualizer.plot_key_diffusers(diffusion_result, dimension)
        
        # Plot diffusion heatmap
        self.visualizer.plot_diffusion_heatmap(diffusion_result, dimension)
