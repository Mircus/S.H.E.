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

## Neural Network Components

SHE includes advanced neural network architectures specifically designed for learning on simplicial complexes and higher-order topological structures.

### Available Architectures

#### Simplicial Convolutional Networks (SCN)

SCNs extend graph convolutional networks to operate on both nodes and edges simultaneously, using the topological structure encoded in incidence matrices.

```python
from she_neural import SHESimplicialConvolutionalNetwork

# Create SCN model
model = SHESimplicialConvolutionalNetwork(
    in_channels_0=16,      # Node feature dimensions
    in_channels_1=8,       # Edge feature dimensions
    hidden_channels=64,    # Hidden layer size
    out_channels=10,       # Output classes
    num_layers=3,          # Number of SCN layers
    dropout=0.5,           # Dropout rate
    task="node_classification"  # Task type
)
```

#### Higher-Order Simplicial Networks (HSN)

HSNs operate on simplicial complexes of arbitrary dimension, enabling learning from triangles, tetrahedra, and higher-dimensional structures.

```python
from she_neural import SHEHigherOrderNetwork

# Define input channels for each dimension
in_channels = {
    0: 16,  # Node features
    1: 8,   # Edge features
    2: 4    # Triangle features
}

model = SHEHigherOrderNetwork(
    in_channels=in_channels,
    hidden_channels=64,
    out_channels=10,
    max_rank=2,           # Maximum simplex dimension
    num_layers=2
)
```

### Data Preparation

#### SimplicialDataset

Specialized dataset class for handling simplicial complex data:

```python
from she_neural import SimplicialDataset

# Prepare your data
complexes = [complex1, complex2, ...]  # List of SHESimplicialComplex
labels = [0, 1, 0, ...]               # Classification labels
node_features = [feat1, feat2, ...]   # Node feature tensors
edge_features = [edge1, edge2, ...]   # Edge feature tensors

# Create dataset
dataset = SimplicialDataset(
    complexes=complexes,
    labels=labels,
    node_features=node_features,
    edge_features=edge_features
)

# Create data loader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Training Framework

#### SHE Neural Engine

High-level interface for managing neural network experiments:

```python
from she_neural import SHENeuralEngine

engine = SHENeuralEngine(config)

# Create and register models
engine.create_scn_model(
    name="social_scn",
    in_channels_0=16,
    in_channels_1=8,
    out_channels=2,
    task="node_classification"
)

# Train model
engine.train_model(
    model_name="social_scn",
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100
)
```

#### Manual Training Loop

For custom training scenarios:

```python
import torch.optim as optim

model = SHESimplicialConvolutionalNetwork(...)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(
            x_0=batch['x_0'],
            x_1=batch['x_1'], 
            incidence_1=batch['incidences']['B_0']
        )
        
        loss = criterion(logits, batch['label'])
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")
```

### Task-Specific Applications

#### Node Classification

Classify nodes based on their topological context:

```python
# Node classification model
node_classifier = SHESimplicialConvolutionalNetwork(
    in_channels_0=node_dim,
    in_channels_1=edge_dim,
    hidden_channels=128,
    out_channels=num_classes,
    task="node_classification"
)

# Training for node classification
for batch in data_loader:
    node_logits = node_classifier(
        batch['x_0'], 
        batch['x_1'], 
        batch['incidences']['B_0']
    )
    loss = F.cross_entropy(node_logits, batch['node_labels'])
```

#### Edge Classification

Classify edges using simplicial structure:

```python
# Edge classification model
edge_classifier = SHESimplicialConvolutionalNetwork(
    task="edge_classification",
    # ... other parameters
)

# Predict edge properties
edge_predictions = edge_classifier(x_0, x_1, incidence_matrix)
```

#### Graph-Level Tasks

Perform classification on entire simplicial complexes:

```python
# Graph classification model
graph_classifier = SHESimplicialConvolutionalNetwork(
    task="graph_classification",
    # ... other parameters
)

# Complex-level prediction
complex_prediction = graph_classifier(x_0, x_1, incidence_matrix)
```

### Advanced Neural Network Features

#### Attention Mechanisms

```python
class SimplicialAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attention = nn.MultiheadAttention(in_channels, num_heads=8)
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x_0, x_1, incidence_1):
        # Apply attention to node features
        attended_nodes, _ = self.attention(x_0, x_0, x_0)
        
        # Combine with edge information through incidence matrix
        edge_aggregated = torch.sparse.mm(incidence_1.t(), attended_nodes)
        
        return self.linear(attended_nodes), self.linear(edge_aggregated)
```

#### Residual Connections

```python
class ResidualSCNLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scn = SCN(channels, channels, channels, channels)
        self.norm_0 = nn.LayerNorm(channels)
        self.norm_1 = nn.LayerNorm(channels)
    
    def forward(self, x_0, x_1, incidence_1):
        # Residual connections
        x_0_out, x_1_out = self.scn(x_0, x_1, incidence_1)
        x_0 = self.norm_0(x_0 + x_0_out)
        x_1 = self.norm_1(x_1 + x_1_out)
        return x_0, x_1
```

#### Multi-Scale Learning

```python
class MultiScaleSCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.local_conv = SCN(in_channels, in_channels, 64, 64)
        self.global_conv = SCN(in_channels, in_channels, 64, 64)
        self.fusion = nn.Linear(128, out_channels)
    
    def forward(self, x_0, x_1, incidence_1):
        # Local processing
        local_0, local_1 = self.local_conv(x_0, x_1, incidence_1)
        
        # Global processing (could use coarsened complex)
        global_0, global_1 = self.global_conv(x_0, x_1, incidence_1)
        
        # Fuse representations
        fused_0 = torch.cat([local_0, global_0], dim=-1)
        return self.fusion(fused_0)
```

### Example: Social Network Analysis with Neural Networks

```python
import torch
import torch.nn.functional as F
from she_neural import *
from she_core import *

# Prepare social network data
def prepare_social_data(G):
    """Convert NetworkX graph to SHE neural network input"""
    complex = SHEDataLoader.from_weighted_networkx(G)
    
    # Node features: degree, centrality, clustering
    node_features = []
    for node in G.nodes():
        degree = G.degree(node)
        centrality = nx.betweenness_centrality(G)[node]
        clustering = nx.clustering(G)[node]
        node_features.append([degree, centrality, clustering])
    
    # Edge features: weight, edge betweenness
    edge_features = []
    edge_centrality = nx.edge_betweenness_centrality(G)
    for edge in G.edges():
        weight = G[edge[0]][edge[1]].get('weight', 1.0)
        betweenness = edge_centrality[edge]
        edge_features.append([weight, betweenness])
    
    return complex, torch.tensor(node_features), torch.tensor(edge_features)

# Load Karate Club dataset
G = nx.karate_club_graph()
complex, node_feat, edge_feat = prepare_social_data(G)

# Create neural network
model = SHESimplicialConvolutionalNetwork(
    in_channels_0=3,  # degree, centrality, clustering
    in_channels_1=2,  # weight, betweenness
    hidden_channels=64,
    out_channels=2,   # two clubs
    task="node_classification"
)

# Get true labels (club membership)
labels = torch.tensor([G.nodes[node]['club'] for node in G.nodes()])

# Get incidence matrix
incidence_matrices = complex.get_incidence_matrices()
incidence_1 = torch.tensor(incidence_matrices['B_0'].toarray(), dtype=torch.float32)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

for epoch in range(200):
    optimizer.zero_grad()
    logits = model(node_feat, edge_feat, incidence_1)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            acc = (pred == labels).float().mean()
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Acc = {acc:.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    logits = model(node_feat, edge_feat, incidence_1)
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == labels).float().mean()
    print(f"Final Accuracy: {accuracy:.4f}")
```

### Example: Protein Interaction Prediction

```python
# Predict protein interactions using higher-order structure
def predict_protein_interactions():
    # Load protein complex data
    proteins = load_protein_data()  # Your protein loading function
    
    # Create higher-order network
    model = SHEHigherOrderNetwork(
        in_channels={0: 128, 1: 64, 2: 32},  # Multi-scale features
        hidden_channels=256,
        out_channels=1,  # Binary interaction prediction
        max_rank=2
    )
    
    # Train on known interactions
    for epoch in range(100):
        for batch in protein_loader:
            # Forward pass through all simplex dimensions
            prediction = model(batch['x_dict'], batch['incidence_dict'])
            loss = F.binary_cross_entropy_with_logits(
                prediction, batch['interaction_labels']
            )
            # ... training loop
    
    return model
```

### Best Practices for SHE Neural Networks

#### 1. Feature Engineering
```python
# Design features that capture topological properties
def extract_topological_features(complex):
    features = {}
    
    # Node features: local topology
    for node in complex.complex.nodes:
        degree = complex.complex.degree(node)
        neighbors = list(complex.complex.neighbors(node))
        local_clustering = compute_local_clustering(complex, node)
        features[node] = [degree, len(neighbors), local_clustering]
    
    return features
```

#### 2. Model Architecture Selection
- Use **SCN** for problems where node-edge interactions are crucial
- Use **HSN** when higher-order structures (triangles, tetrahedra) matter
- Consider **hybrid architectures** for complex scenarios

#### 3. Training Strategies
```python
# Curriculum learning: start with simple structures
def curriculum_training(model, data_loader, epochs_per_stage=50):
    stages = [
        {"max_dimension": 0},  # Only nodes
        {"max_dimension": 1},  # Nodes + edges  
        {"max_dimension": 2}   # Full complex
    ]
    
    for stage in stages:
        print(f"Training stage: max_dim = {stage['max_dimension']}")
        for epoch in range(epochs_per_stage):
            # Filter data by max dimension
            filtered_data = filter_by_dimension(data_loader, stage['max_dimension'])
            train_epoch(model, filtered_data)
```

#### 4. Regularization for Topological Models
```python
class TopologicalRegularizer(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, model_output, incidence_matrices):
        # Penalize solutions that don't respect topological constraints
        boundary_consistency = compute_boundary_consistency(
            model_output, incidence_matrices
        )
        return self.alpha * boundary_consistency
```

## References

- Schaub, M. T., et al. "Random walks on simplicial complexes and the normalized Hodge 1-Laplacian." SIAM Review 62.2 (2020): 353-391.
- Barbarossa, S., & Sardellitti, S. "Topological signal processing over simplicial complexes." IEEE Transactions on Signal Processing 68 (2020): 2992-3007.
- Bunch, E., et al. "Simplicial 2-complex convolutional neural networks." arXiv preprint arXiv:2012.06010 (2020).
- Roddenberry, T. M., et al. "Principled simplicial neural networks for trajectory prediction." International Conference on Machine Learning (2021).

---

*This manual covers both the core topological analysis and neural network capabilities of SHE. For the latest updates and additional features, please refer to the source code and accompanying documentation.*
