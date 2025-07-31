# SHE - Simplicial Hyperstructure Engine Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Quick Start](#quick-start)
5. [API Reference](#api-reference)
6. [Advanced Usage](#advanced-usage)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

## Introduction

The Simplicial Hyperstructure Engine (SHE) is an advanced Python framework for analyzing complex systems through the lens of algebraic topology. It provides tools for:

- **Simplicial Complex Construction**: Build and manipulate simplicial complexes from various data sources
- **Hodge Laplacian Analysis**: Compute and analyze Hodge Laplacians for diffusion processes
- **Diffusion Analysis**: Identify key diffusers and analyze information flow patterns
- **Spectral Analysis**: Compute eigenspectra and diffusion maps
- **Topological Data Analysis**: Integration with persistent homology tools

## Installation

### Prerequisites

SHE requires Python 3.7+ and depends on several scientific computing libraries:

```bash
# Core dependencies
pip install torch numpy scipy pandas matplotlib seaborn

# Optional but recommended dependencies
pip install toponetx              # Simplicial complex operations
pip install torch-geometric       # Graph neural networks
pip install giotto-tda           # Persistent homology
pip install gudhi                # Advanced TDA
pip install networkx             # Graph algorithms
```

### Installation from Source

```bash
git clone <repository-url>
cd she
pip install -e .
```

## Core Concepts

### Simplicial Complexes

A simplicial complex is a mathematical structure that generalizes graphs to higher dimensions:

- **0-simplices**: Vertices/nodes
- **1-simplices**: Edges connecting two vertices
- **2-simplices**: Triangular faces connecting three vertices
- **k-simplices**: Higher-dimensional analogs

### Hodge Laplacians

Hodge Laplacians are matrix operators that encode the topology of a simplicial complex and enable diffusion analysis. For dimension k, the Hodge Laplacian L_k captures how information flows through k-dimensional structures.

### Diffusion Analysis

SHE analyzes how information, signals, or processes diffuse through complex topological structures by:

1. Computing spectral properties of Hodge Laplacians
2. Identifying key diffusers (highly influential simplices)
3. Analyzing harmonic, exact, and coexact forms through Hodge decomposition

## Quick Start

### Basic Usage

```python
from she import SHEEngine, SHEConfig, SHESimplicialComplex

# Initialize SHE with configuration
config = SHEConfig(device="cpu", max_dimension=2)
she = SHEEngine(config)

# Create a simplicial complex
complex = SHESimplicialComplex("my_complex")

# Add simplices
complex.add_node("A", features={"weight": 1.0})
complex.add_node("B", features={"weight": 1.5})
complex.add_edge(("A", "B"), features={"weight": 2.0})

# Register and analyze
she.register_complex("my_complex", complex)
diffusion_result = she.analyze_diffusion("my_complex")
```

### Loading from NetworkX

```python
import networkx as nx
from she import SHEDataLoader

# Create or load a NetworkX graph
G = nx.karate_club_graph()

# Convert to weighted simplicial complex
complex = SHEDataLoader.from_weighted_networkx(
    G, 
    weight_attr='weight',
    include_cliques=True,
    max_clique_size=4
)
```

## API Reference

### SHEConfig

Configuration class for the SHE engine.

**Parameters:**
- `device`: str - Computing device ("cpu" or "cuda")
- `dtype`: torch.dtype - Data type for computations
- `max_dimension`: int - Maximum simplex dimension to analyze
- `use_cache`: bool - Enable matrix caching
- `diffusion_steps`: int - Number of diffusion time steps
- `spectral_k`: int - Number of eigenvalues to compute

### SHESimplicialComplex

Main class for building and manipulating simplicial complexes.

#### Methods

**`add_node(node_id, features=None, **attr)`**
- Add a 0-simplex (vertex) to the complex
- `node_id`: Unique identifier for the node
- `features`: Optional dictionary of node features
- `**attr`: Additional attributes

**`add_edge(edge, features=None, **attr)`**
- Add a 1-simplex (edge) to the complex
- `edge`: Tuple of two node identifiers
- `features`: Optional dictionary of edge features

**`add_simplex(simplex, rank=None, features=None, **attr)`**
- Add a k-simplex to the complex
- `simplex`: List/tuple of vertex identifiers
- `rank`: Dimension of the simplex (auto-detected if None)
- `features`: Optional dictionary of simplex features

**`get_hodge_laplacians(use_cache=True)`**
- Returns dictionary of Hodge Laplacian matrices by dimension
- Returns: `Dict[int, csr_matrix]`

**`get_incidence_matrices(use_cache=True)`**
- Returns boundary/incidence matrices
- Returns: `Dict[str, csr_matrix]`

### SHEEngine

Main engine class for diffusion analysis.

#### Methods

**`register_complex(name, complex)`**
- Register a simplicial complex with the engine
- `name`: String identifier for the complex
- `complex`: SHESimplicialComplex instance

**`analyze_diffusion(complex_name)`**
- Perform comprehensive diffusion analysis
- Returns: `DiffusionResult` object

**`find_key_diffusers(complex_name, dimension=0, top_k=10)`**
- Find the most influential diffusers in a given dimension
- Returns: List of (simplex, centrality_score) tuples

**`visualize_diffusion(complex_name, dimension=0)`**
- Generate visualization plots for diffusion analysis

### DiffusionResult

Container for diffusion analysis results.

**Attributes:**
- `eigenvalues`: Dict of eigenvalue arrays by dimension
- `eigenvectors`: Dict of eigenvector matrices by dimension
- `diffusion_maps`: Dict of diffusion coordinate arrays
- `key_diffusers`: Dict of key diffuser rankings by dimension
- `hodge_decomposition`: Dict of Hodge decomposition results

## Advanced Usage

### Custom Weight Functions

```python
# Define custom weights based on simplex properties
def custom_weight_function(simplex, dimension, complex):
    if dimension == 0:  # Nodes
        return complex.node_features.get(simplex, {}).get('importance', 1.0)
    elif dimension == 1:  # Edges
        return len(simplex) * 0.5  # Example: weight by connectivity
    else:
        return 1.0

# Apply weights during complex construction
for node in nodes:
    weight = custom_weight_function(node, 0, complex)
    complex.add_node(node, weight=weight)
```

### Spectral Analysis

```python
# Access detailed spectral properties
diffusion_result = she.analyze_diffusion("my_complex")

for dim in diffusion_result.eigenvalues:
    eigenvals = diffusion_result.eigenvalues[dim]
    eigenvecs = diffusion_result.eigenvectors[dim]
    
    print(f"Dimension {dim}:")
    print(f"  Spectral gap: {eigenvals[1] - eigenvals[0]}")
    print(f"  Effective resistance: {sum(1/ev for ev in eigenvals if ev > 1e-10)}")
```

### Hodge Decomposition

```python
# Analyze harmonic, exact, and coexact forms
hodge_result = diffusion_result.hodge_decomposition

for dim_key, decomp in hodge_result.items():
    print(f"{dim_key}:")
    print(f"  Harmonic forms: {decomp['harmonic'].shape}")
    print(f"  Exact forms: {decomp['exact'].shape}")
    print(f"  Coexact forms: {decomp['coexact'].shape}")
```

### Custom Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create custom diffusion visualizations
def plot_custom_diffusion_analysis(diffusion_result, dimension=0):
    eigenvals = diffusion_result.eigenvalues[dimension]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Eigenvalue distribution
    ax1.hist(eigenvals, bins=20, alpha=0.7)
    ax1.set_xlabel('Eigenvalue')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Eigenvalue Distribution (Dim {dimension})')
    
    # Spectral density
    ax2.plot(eigenvals, 'o-')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Eigenvalue')
    ax2.set_title('Spectral Density')
    
    plt.tight_layout()
    plt.show()
```

## Examples

### Example 1: Social Network Analysis

```python
import networkx as nx
import numpy as np

# Create social network
G = nx.karate_club_graph()

# Add influence weights
np.random.seed(42)
for node in G.nodes():
    G.nodes[node]['influence'] = np.random.gamma(2.0, 1.0)

for u, v in G.edges():
    G[u][v]['strength'] = np.random.exponential(1.0)

# Load and analyze
complex = SHEDataLoader.from_weighted_networkx(G, weight_attr='strength')
she.register_complex("social_network", complex)

# Find key influencers
key_nodes = she.find_key_diffusers("social_network", dimension=0, top_k=5)
print("Key influencers:")
for node, score in key_nodes:
    print(f"  Node {node}: {score:.3f}")

# Find critical connections
key_edges = she.find_key_diffusers("social_network", dimension=1, top_k=5)
print("Critical connections:")
for edge, score in key_edges:
    print(f"  Edge {edge}: {score:.3f}")
```

### Example 2: Protein Interaction Networks

```python
# Load protein interaction data
def create_protein_complex(interactions, proteins):
    complex = SHESimplicialComplex("protein_network")
    
    # Add proteins as nodes
    for protein_id, properties in proteins.items():
        complex.add_node(protein_id, 
                        weight=properties.get('expression_level', 1.0),
                        **properties)
    
    # Add interactions as edges
    for (p1, p2), strength in interactions.items():
        complex.add_edge((p1, p2), weight=strength)
    
    # Add protein complexes as higher-order simplices
    # (This would be based on known protein complex data)
    
    return complex

# Analyze protein diffusion patterns
diffusion_result = she.analyze_diffusion("protein_network")

# Identify critical proteins for signal propagation
critical_proteins = she.find_key_diffusers("protein_network", dimension=0)
```

### Example 3: Brain Connectivity Analysis

```python
# Brain network from connectivity matrix
def create_brain_complex(connectivity_matrix, region_names):
    complex = SHESimplicialComplex("brain_network")
    
    # Add brain regions
    for i, region in enumerate(region_names):
        complex.add_node(region, weight=1.0, index=i)
    
    # Add connections above threshold
    threshold = 0.3
    for i in range(len(region_names)):
        for j in range(i+1, len(region_names)):
            if connectivity_matrix[i, j] > threshold:
                weight = connectivity_matrix[i, j]
                complex.add_edge((region_names[i], region_names[j]), 
                               weight=weight)
    
    return complex

# Analyze information flow in brain
brain_result = she.analyze_diffusion("brain_network")

# Find hub regions
hub_regions = she.find_key_diffusers("brain_network", dimension=0, top_k=10)
print("Brain hub regions:")
for region, centrality in hub_regions:
    print(f"  {region}: {centrality:.3f}")
```

## Troubleshooting

### Common Issues

**1. ImportError for optional dependencies**
- Install missing packages: `pip install toponetx torch-geometric giotto-tda`
- SHE will warn about missing optional dependencies but continue with available features

**2. Memory issues with large complexes**
- Reduce `max_dimension` in SHEConfig
- Decrease `spectral_k` for fewer eigenvalue computations
- Enable caching with `use_cache=True`

**3. Numerical instability in spectral computations**
- Check for disconnected components in your complex
- Verify that simplices are properly oriented
- Try different solvers by adjusting the backend

**4. Empty Hodge Laplacians**
- Ensure your complex has simplices in the requested dimension
- Check that incidence matrices are properly computed
- Verify simplex orientation consistency

### Performance Tips

1. **Use appropriate data types**: Set `dtype=torch.float32` for better performance
2. **Enable caching**: Set `use_cache=True` for repeated computations
3. **Limit dimensions**: Set `max_dimension` to only what you need
4. **Batch processing**: Process multiple complexes together when possible

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# SHE will now provide detailed computation logs
```

### Getting Help

For additional support:
1. Check the examples in the codebase
2. Review the scientific literature on discrete Hodge theory
3. Examine the TopoX documentation for simplicial complex operations

## References

- Schaub, M. T., et al. "Random walks on simplicial complexes and the normalized Hodge 1-Laplacian." SIAM Review 62.2 (2020): 353-391.
- Barbarossa, S., & Sardellitti, S. "Topological signal processing over simplicial complexes." IEEE Transactions on Signal Processing 68 (2020): 2992-3007.

---

*This manual covers the core functionality of SHE. For the latest updates and additional features, please refer to the source code and accompanying documentation.*
