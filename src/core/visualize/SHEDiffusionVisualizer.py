from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# + any types you annotate



class SHEDiffusionVisualizer:
    """Visualization tools for diffusion analysis"""
    
    @staticmethod
    def plot_spectrum(diffusion_result: DiffusionResult, dimension: int, 
                     title: Optional[str] = None):
        """Plot eigenvalue spectrum"""
        if dimension not in diffusion_result.eigenvalues:
            logger.warning(f"No eigenvalues found for dimension {dimension}")
            return
        
        eigenvals = diffusion_result.eigenvalues[dimension]
        
        plt.figure(figsize=(10, 6))
        plt.plot(eigenvals, 'bo-', markersize=4)
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Eigenvalue')
        plt.title(title or f'Hodge Laplacian Spectrum - Dimension {dimension}')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_key_diffusers(diffusion_result: DiffusionResult, dimension: int,
                          top_k: int = 10):
        """Plot key diffusers ranking"""
        if dimension not in diffusion_result.key_diffusers:
            logger.warning(f"No key diffusers found for dimension {dimension}")
            return
        
        diffusers = diffusion_result.key_diffusers[dimension][:top_k]
        
        if not diffusers:
            return
        
        simplices = [str(d[0]) for d in diffusers]
        scores = [d[1] for d in diffusers]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(simplices)), scores)
        plt.xlabel('Simplex')
        plt.ylabel('Diffusion Centrality')
        plt.title(f'Top {top_k} Key Diffusers - Dimension {dimension}')
        plt.xticks(range(len(simplices)), simplices, rotation=45, ha='right')
        
        # Color bars by score
        for bar, score in zip(bars, scores):
            bar.set_color(plt.cm.viridis(score))
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_diffusion_heatmap(diffusion_result: DiffusionResult, dimension: int):
        """Plot diffusion map as heatmap"""
        map_key = f"dim_{dimension}"
        if map_key not in diffusion_result.diffusion_maps:
            logger.warning(f"No diffusion map found for dimension {dimension}")
            return
        
        diff_map = diffusion_result.diffusion_maps[map_key]
        
        if diff_map.size == 0:
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(diff_map.T, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Diffusion Coordinate'})
        plt.xlabel('Simplex Index')
        plt.ylabel('Diffusion Dimension')
        plt.title(f'Diffusion Map - Dimension {dimension}')
        plt.show()
