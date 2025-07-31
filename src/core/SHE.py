"""
SHE - Simplicial Hyperstructure Engine
Enhanced with Hodge Laplacian diffusion analysis and key diffuser identification
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh, spsolve
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns

# TopoX imports for simplicial complexes
try:
    import toponetx as tnx
    from toponetx import SimplicialComplex as TopoXSimplicialComplex
    from toponetx.classes.simplicial_complex import SimplicialComplex
    TOPOX_AVAILABLE = True
except ImportError:
    print("TopoX not available. Install with: pip install toponetx")
    TOPOX_AVAILABLE = False

# PyTorch Geometric imports
try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import MessagePassing
    PYGTORCH_AVAILABLE = True
except ImportError:
    print("PyTorch Geometric not available. Install with: pip install torch-geometric")
    PYGTORCH_AVAILABLE = False

# Giotto-TDA for persistent homology
try:
    from gtda.homology import VietorisRipsPersistence, CubicalPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude, NumberOfPoints
    from gtda.plotting import plot_diagram
    GIOTTO_AVAILABLE = True
except ImportError:
    print("Giotto-TDA not available. Install with: pip install giotto-tda")
    GIOTTO_AVAILABLE = False

# Gudhi for advanced TDA
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    print("Gudhi not available. Install with: pip install gudhi")
    GUDHI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@dataclass
class DiffusionResult:
    """Results from diffusion analysis"""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    diffusion_maps: Dict[str, np.ndarray]
    key_diffusers: Dict[int, List[Tuple[Any, float]]]  # dimension -> [(simplex, centrality_score)]
    hodge_decomposition: Dict[str, np.ndarray]
    heat_kernel: Optional[np.ndarray] = None

class SHESimplicialComplex:
    """
    Enhanced wrapper around TopoX SimplicialComplex with diffusion capabilities
    """
    
    def __init__(self, name: str = "SHE_Complex", config: Optional[SHEConfig] = None):
        if not TOPOX_AVAILABLE:
            raise ImportError("TopoX is required for SHE. Install with: pip install toponetx")
            
        self.name = name
        self.config = config or SHEConfig()
        self.complex = SimplicialComplex()
        self.node_features = {}
        self.edge_features = {}
        self.face_features = {}
        self.metadata = {}
        self._cached_matrices = {}
        
    def add_node(self, node_id: Any, features: Optional[Dict[str, Any]] = None, **attr):
        """Add a node (0-simplex) to the complex"""
        self.complex.add_node(node_id, **attr)
        if features:
            self.node_features[node_id] = features
    
    def add_edge(self, edge: Tuple, features: Optional[Dict[str, Any]] = None, **attr):
        """Add an edge (1-simplex) to the complex"""
        self.complex.add_simplex(edge, rank=1, **attr)
        if features:
            self.edge_features[edge] = features
    
    def add_simplex(self, simplex: Union[List, Tuple], rank: Optional[int] = None, 
                   features: Optional[Dict[str, Any]] = None, **attr):
        """Add a general k-simplex to the complex"""
        if rank is None:
            rank = len(simplex) - 1
        
        self.complex.add_simplex(simplex, rank=rank, **attr)
        
        if features:
            if rank == 0:
                self.node_features[simplex[0]] = features
            elif rank == 1:
                self.edge_features[tuple(sorted(simplex))] = features
            elif rank == 2:
                self.face_features[tuple(sorted(simplex))] = features
    
    def get_hodge_laplacians(self, use_cache: bool = True) -> Dict[int, csr_matrix]:
        """Get Hodge Laplacian matrices for all dimensions using TopoX"""
        if use_cache and 'hodge_laplacians' in self._cached_matrices:
            return self._cached_matrices['hodge_laplacians']
        
        laplacians = {}
        max_dim = min(self.complex.dim, self.config.max_dimension)
        
        for k in range(max_dim + 1):
            try:
                # Use TopoX's built-in Hodge Laplacian computation
                L_k = self.complex.hodge_laplacian_matrix(rank=k, signed=True)
                if L_k is not None and L_k.shape[0] > 0:
                    laplacians[k] = L_k.tocsr()
                    logger.info(f"Computed Hodge Laplacian L_{k} with shape {L_k.shape}")
                else:
                    logger.warning(f"Empty or None Hodge Laplacian for dimension {k}")
            except Exception as e:
                logger.warning(f"Could not compute Hodge Laplacian L_{k}: {e}")
        
        if use_cache:
            self._cached_matrices['hodge_laplacians'] = laplacians
        
        return laplacians
    
    def get_incidence_matrices(self, use_cache: bool = True) -> Dict[str, csr_matrix]:
        """Get boundary/incidence matrices using TopoX"""
        if use_cache and 'incidence_matrices' in self._cached_matrices:
            return self._cached_matrices['incidence_matrices']
        
        matrices = {}
        max_dim = min(self.complex.dim, self.config.max_dimension)
        
        for k in range(max_dim):
            try:
                # Boundary matrix from k+1 to k simplices
                B_k = self.complex.incidence_matrix(rank=k, to_rank=k+1, signed=True)
                if B_k is not None and B_k.shape[0] > 0:
                    matrices[f"B_{k}"] = B_k.tocsr()
                    logger.info(f"Computed incidence matrix B_{k} with shape {B_k.shape}")
            except Exception as e:
                logger.warning(f"Could not compute incidence matrix B_{k}: {e}")
        
        if use_cache:
            self._cached_matrices['incidence_matrices'] = matrices
        
        return matrices
    
    def get_simplex_weights(self, dimension: int) -> Dict[Any, float]:
        """Extract weights for simplices of given dimension"""
        weights = {}
        
        try:
            for simplex in self.complex.skeleton(dimension):
                attrs = self.complex.get_simplex_attributes(simplex, dimension)
                weight = attrs.get('weight', 1.0)
                weights[simplex] = weight
        except Exception as e:
            logger.warning(f"Could not extract weights for dimension {dimension}: {e}")
        
        return weights
    
    def get_simplex_list(self, dimension: int) -> List[Any]:
        """Get ordered list of simplices for a given dimension"""
        try:
            return list(self.complex.skeleton(dimension))
        except:
            return []

class SHEHodgeDiffusion:
    """
    Advanced diffusion analysis using Hodge Laplacians
    """
    
    def __init__(self, config: Optional[SHEConfig] = None):
        self.config = config or SHEConfig()
    
    def compute_spectral_properties(self, laplacian: csr_matrix, 
                                  k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors of Hodge Laplacian"""
        k = k or min(self.config.spectral_k, laplacian.shape[0] - 1)
        
        if laplacian.shape[0] <= 1:
            return np.array([0.0]), np.array([[1.0]])
        
        try:
            # For symmetric matrices, use eigsh for efficiency
            if k >= laplacian.shape[0] - 1:
                # Use dense solver for small matrices
                eigenvals, eigenvecs = eigh(laplacian.toarray())
            else:
                # Use sparse solver for large matrices
                eigenvals, eigenvecs = eigsh(laplacian, k=k, which='SM', sigma=0.0)
            
            # Sort by eigenvalue
            idx = np.argsort(eigenvals)
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            return eigenvals, eigenvecs
            
        except Exception as e:
            logger.warning(f"Spectral computation failed: {e}")
            return np.array([0.0]), np.array([[1.0]])
    
    def compute_diffusion_centrality(self, laplacian: csr_matrix, 
                                   weights: Dict[Any, float],
                                   simplex_list: List[Any]) -> Dict[Any, float]:
        """Compute diffusion centrality for simplices"""
        
        if laplacian.shape[0] == 0:
            return {}
        
        # Create weight vector
        weight_vector = np.array([weights.get(simplex, 1.0) for simplex in simplex_list])
        
        try:
            # Solve diffusion equation: (I + λL)x = w
            # where λ controls diffusion strength
            lambda_diff = 0.1
            A = diags([1.0], shape=laplacian.shape) + lambda_diff * laplacian
            
            if A.shape[0] > 0:
                diffusion_result = spsolve(A, weight_vector)
                
                # Normalize to get centrality scores
                if np.max(np.abs(diffusion_result)) > 0:
                    centrality_scores = np.abs(diffusion_result) / np.max(np.abs(diffusion_result))
                else:
                    centrality_scores = np.ones_like(diffusion_result)
                
                return {simplex: score for simplex, score in zip(simplex_list, centrality_scores)}
            
        except Exception as e:
            logger.warning(f"Diffusion centrality computation failed: {e}")
        
        # Fallback: uniform centrality
        return {simplex: 1.0 for simplex in simplex_list}
    
    def compute_heat_kernel(self, laplacian: csr_matrix, t: float = 1.0) -> np.ndarray:
        """Compute heat kernel exp(-tL)"""
        if laplacian.shape[0] <= 1:
            return np.eye(laplacian.shape[0])
        
        try:
            eigenvals, eigenvecs = self.compute_spectral_properties(laplacian)
            
            # Heat kernel: exp(-t * eigenval)
            heat_eigenvals = np.exp(-t * eigenvals)
            
            # Reconstruct: U * diag(exp(-t*λ)) * U^T
            heat_kernel = eigenvecs @ np.diag(heat_eigenvals) @ eigenvecs.T
            
            return heat_kernel
            
        except Exception as e:
            logger.warning(f"Heat kernel computation failed: {e}")
            return np.eye(laplacian.shape[0])
    
    def hodge_decomposition(self, complex: SHESimplicialComplex, 
                          dimension: int) -> Dict[str, np.ndarray]:
        """Compute Hodge decomposition for k-forms"""
        
        try:
            # Get relevant matrices
            hodge_laplacians = complex.get_hodge_laplacians()
            incidence_matrices = complex.get_incidence_matrices()
            
            if dimension not in hodge_laplacians:
                return {"harmonic": np.array([]), "exact": np.array([]), "coexact": np.array([])}
            
            L_k = hodge_laplacians[dimension]
            
            # Compute kernel (harmonic forms)
            eigenvals, eigenvecs = self.compute_spectral_properties(L_k)
            
            # Harmonic forms: kernel of Laplacian (eigenvalue ≈ 0)
            tol = 1e-6
            harmonic_idx = np.where(np.abs(eigenvals) < tol)[0]
            harmonic_forms = eigenvecs[:, harmonic_idx] if len(harmonic_idx) > 0 else np.array([]).reshape(L_k.shape[0], 0)
            
            # For exact and coexact forms, we'd need boundary operators
            # This is a simplified version
            exact_forms = np.array([]).reshape(L_k.shape[0], 0)
            coexact_forms = np.array([]).reshape(L_k.shape[0], 0)
            
            return {
                "harmonic": harmonic_forms,
                "exact": exact_forms,
                "coexact": coexact_forms
            }
            
        except Exception as e:
            logger.warning(f"Hodge decomposition failed for dimension {dimension}: {e}")
            return {"harmonic": np.array([]), "exact": np.array([]), "coexact": np.array([])}
    
    def analyze_diffusion(self, complex: SHESimplicialComplex) -> DiffusionResult:
        """Comprehensive diffusion analysis"""
        
        hodge_laplacians = complex.get_hodge_laplacians()
        
        all_eigenvals = {}
        all_eigenvecs = {}
        diffusion_maps = {}
        key_diffusers = {}
        hodge_decompositions = {}
        
        for dim, laplacian in hodge_laplacians.items():
            logger.info(f"Analyzing diffusion for dimension {dim}")
            
            # Spectral analysis
            eigenvals, eigenvecs = self.compute_spectral_properties(laplacian)
            all_eigenvals[dim] = eigenvals
            all_eigenvecs[dim] = eigenvecs
            
            # Diffusion map (using first few non-trivial eigenvectors)
            if len(eigenvals) > 1:
                # Skip first eigenvalue (should be 0 or very small)
                start_idx = 1 if eigenvals[0] < 1e-6 else 0
                end_idx = min(start_idx + 3, len(eigenvals))
                diffusion_maps[f"dim_{dim}"] = eigenvecs[:, start_idx:end_idx]
            
            # Key diffusers analysis
            weights = complex.get_simplex_weights(dim)
            simplex_list = complex.get_simplex_list(dim)
            
            if weights and simplex_list:
                centrality = self.compute_diffusion_centrality(laplacian, weights, simplex_list)
                
                # Sort by centrality score
                sorted_diffusers = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                key_diffusers[dim] = sorted_diffusers[:min(10, len(sorted_diffusers))]
            
            # Hodge decomposition
            hodge_decompositions[f"dim_{dim}"] = self.hodge_decomposition(complex, dim)
        
        return DiffusionResult(
            eigenvalues=all_eigenvals,
            eigenvectors=all_eigenvecs,
            diffusion_maps=diffusion_maps,
            key_diffusers=key_diffusers,
            hodge_decomposition=hodge_decompositions
        )

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

# Enhanced data loader with weighted simplices
class SHEDataLoader:
    """Enhanced data loader with weight support"""
    
    @staticmethod
    def from_weighted_networkx(G, weight_attr: str = 'weight', 
                             include_cliques: bool = True, 
                             max_clique_size: int = 4) -> SHESimplicialComplex:
        """Load from NetworkX graph with weights"""
        complex = SHESimplicialComplex("from_weighted_networkx")
        
        # Add nodes with weights
        for node, data in G.nodes(data=True):
            weight = data.get(weight_attr, 1.0)
            complex.add_node(node, weight=weight, **data)
        
        # Add edges with weights
        for u, v, data in G.edges(data=True):
            weight = data.get(weight_attr, 1.0)
            complex.add_edge((u, v), weight=weight, **data)
        
        # Add higher-order simplices from cliques
        if include_cliques:
            try:
                import networkx as nx
                cliques = list(nx.find_cliques(G))
                for clique in cliques:
                    if 3 <= len(clique) <= max_clique_size:
                        # Weight of clique = average of edge weights
                        edge_weights = []
                        for i in range(len(clique)):
                            for j in range(i + 1, len(clique)):
                                if G.has_edge(clique[i], clique[j]):
                                    edge_weights.append(G[clique[i]][clique[j]].get(weight_attr, 1.0))
                        
                        clique_weight = np.mean(edge_weights) if edge_weights else 1.0
                        complex.add_simplex(list(clique), weight=clique_weight)
            except Exception as e:
                logger.warning(f"Could not compute weighted cliques: {e}")
        
        return complex

# Example usage with diffusion analysis
if __name__ == "__main__":
    # Initialize enhanced SHE
    config = SHEConfig(device="cpu", max_dimension=2, spectral_k=5)
    she = SHEEngine(config)
    
    # Example: Weighted social network analysis
    try:
        import networkx as nx
        
        # Create a weighted graph
        G = nx.karate_club_graph()
        
        # Add random weights to demonstrate diffusion
        np.random.seed(42)
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.exponential(1.0)
        
        for node in G.nodes():
            G.nodes[node]['weight'] = np.random.gamma(2.0, 1.0)
        
        # Load as weighted complex
        karate_complex = SHEDataLoader.from_weighted_networkx(G, include_cliques=True)
        she.register_complex("weighted_karate", karate_complex)
        
        # Comprehensive diffusion analysis
        print("=== Weighted Karate Club Diffusion Analysis ===")
        diffusion_result = she.analyze_diffusion("weighted_karate")
        
        # Print key diffusers for each dimension
        for dim in diffusion_result.key_diffusers:
            print(f"\nTop Key Diffusers (Dimension {dim}):")
            for i, (simplex, score) in enumerate(diffusion_result.key_diffusers[dim][:5]):
                print(f"  {i+1}. {simplex}: {score:.4f}")
        
        # Print spectral properties
        for dim in diffusion_result.eigenvalues:
            eigenvals = diffusion_result.eigenvalues[dim]
            print(f"\nSpectral Properties (Dimension {dim}):")
            print(f"  Number of eigenvalues: {len(eigenvals)}")
            print(f"  Spectral gap: {eigenvals[1] - eigenvals[0] if len(eigenvals) > 1 else 'N/A'}")
            print(f"  Largest eigenvalue: {eigenvals[-1] if len(eigenvals) > 0 else 'N/A'}")
        
        # Find specific key diffusers
        print("\n=== Key Node Diffusers ===")
        key_nodes = she.find_key_diffusers("weighted_karate", dimension=0, top_k=5)
        for i, (node, score) in enumerate(key_nodes):
            print(f"{i+1}. Node {node}: {score:.4f}")
        
        print("\n=== Key Edge Diffusers ===")
        key_edges = she.find_key_diffusers("weighted_karate", dimension=1, top_k=5)
        for i, (edge, score) in enumerate(key_edges):
            print(f"{i+1}. Edge {edge}: {score:.4f}")
        
        # Visualization (comment out if matplotlib not available)
        try:
            she.visualize_diffusion("weighted_karate", dimension=0)
        except Exception as e:
            print(f"Visualization failed: {e}")
        
    except Exception as e:
        print(f"Enhanced example failed: {e}")
    
    logger.info("Enhanced SHE demonstration complete!")
