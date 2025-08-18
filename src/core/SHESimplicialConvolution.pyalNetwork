"""
SHE - Simplicial Hyperstructure Engine
Neural Network Components using TopoX Simplicial Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

# TopoNetX neural network imports
try:
    from toponetx.nn.simplicial import SCN, SCANConv, SCNN
    from toponetx.nn.simplicial.hsn import HSN
    from toponetx.nn.hypergraph import HGNN
    TOPONETX_NN_AVAILABLE = True
except ImportError:
    print("TopoNetX neural networks not available. Install with: pip install toponetx")
    TOPONETX_NN_AVAILABLE = False

# PyTorch Lightning for training
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    LIGHTNING_AVAILABLE = True
except ImportError:
    print("PyTorch Lightning not available. Install with: pip install pytorch-lightning")
    LIGHTNING_AVAILABLE = False

from she_core import SHESimplicialComplex, SHEConfig

logger = logging.getLogger(__name__)

class SimplicialDataset(Dataset):
    """
    Dataset for simplicial complexes
    """
    
    def __init__(self, complexes: List[SHESimplicialComplex], 
                 labels: Optional[List[Any]] = None,
                 node_features: Optional[List[torch.Tensor]] = None,
                 edge_features: Optional[List[torch.Tensor]] = None):
        self.complexes = complexes
        self.labels = labels
        self.node_features = node_features
        self.edge_features = edge_features
    
    def __len__(self):
        return len(self.complexes)
    
    def __getitem__(self, idx):
        complex = self.complexes[idx]
        
        # Get incidence matrices
        incidences = complex.get_incidence_matrices()
        
        # Get features
        if self.node_features:
            x_0 = self.node_features[idx]
        else:
            # Default node features
            num_nodes = len(complex.complex.nodes)
            x_0 = torch.ones(num_nodes, 1, dtype=torch.float32)
        
        if self.edge_features:
            x_1 = self.edge_features[idx]
        else:
            # Default edge features
            num_edges = len(list(complex.complex.skeleton(1)))
            x_1 = torch.ones(num_edges, 1, dtype=torch.float32)
        
        sample = {
            'x_0': x_0,  # Node features
            'x_1': x_1,  # Edge features
            'incidences': incidences,
            'complex': complex
        }
        
        if self.labels:
            sample['label'] = self.labels[idx]
        
        return sample

class SHESimplicialConvolutionalNetwork(nn.Module):
    """
    Simplicial Convolutional Network using TopoNetX
    """
    
    def __init__(self, 
                 in_channels_0: int,
                 in_channels_1: int, 
                 hidden_channels: int = 64,
                 out_channels: int = 1,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 task: str = "node_classification"):
        super().__init__()
        
        if not TOPONETX_NN_AVAILABLE:
            raise ImportError("TopoNetX neural networks required")
        
        self.task = task
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Simplicial convolutional layers
        self.conv_layers = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(
            SCN(in_channels_0, in_channels_1, hidden_channels, hidden_channels)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(
                SCN(hidden_channels, hidden_channels, hidden_channels, hidden_channels)
            )
        
        # Output layer
        if task == "node_classification":
            self.output = nn.Linear(hidden_channels, out_channels)
        elif task == "edge_classification":
            self.output = nn.Linear(hidden_channels, out_channels)
        elif task == "graph_classification":
            self.output = nn.Linear(hidden_channels * 2, out_channels)  # Node + edge pooling
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x_0, x_1, incidence_1):
        """
        Forward pass
        Args:
            x_0: Node features [num_nodes, in_channels_0]
            x_1: Edge features [num_edges, in_channels_1] 
            incidence_1: Incidence matrix B_1 [num_nodes, num_edges]
        """
        
        # Apply SCN layers
        for i, conv in enumerate(self.conv_layers):
            x_0, x_1 = conv(x_0, x_1, incidence_1)
            
            if i < len(self.conv_layers) - 1:  # Don't apply activation to last layer
                x_0 = F.relu(x_0)
                x_1 = F.relu(x_1)
                x_0 = self.dropout_layer(x_0)
                x_1 = self.dropout_layer(x_1)
        
        # Task-specific output
        if self.task == "node_classification":
            return self.output(x_0)
        elif self.task == "edge_classification":
            return self.output(x_1)
        elif self.task == "graph_classification":
            # Global pooling
            node_pool = torch.mean(x_0, dim=0)
            edge_pool = torch.mean(x_1, dim=0)
            graph_repr = torch.cat([node_pool, edge_pool])
            return self.output(graph_repr.unsqueeze(0))

class SHEHigherOrderNetwork(nn.Module):
    """
    Higher-order simplicial network using HSN from TopoNetX
    """
    
    def __init__(self,
                 in_channels: Dict[int, int],  # {dimension: channels}
                 hidden_channels: int = 64,
                 out_channels: int = 1,
                 max_rank: int = 2,
                 num_layers: int = 2):
        super().__init__()
        
        if not TOPONETX_NN_AVAILABLE:
            raise ImportError("TopoNetX neural networks required")
        
        self.max_rank = max_rank
        self.hsn_layers = nn.ModuleList()
        
        # Build HSN layers
        for _ in range(num_layers):
            self.hsn_layers.append(
                HSN(in_channels, hidden_channels, max_rank=max_rank)
            )
            in_channels = {k: hidden_channels for k in range(max_rank + 1)}
        
        # Output projection
        self.output = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x_dict, incidence_dict):
        """
        Forward pass for higher-order network
        Args:
            x_dict: {rank: features} dictionary
            incidence_dict: {f"B_{k}": incidence_matrix} dictionary
        """
        
        for hsn_layer in self.hsn_layers:
            x_dict = hsn_layer(x_dict, incidence_dict)
        
        # Use 0-dimensional features for final prediction
        return self.output(x_dict[0])

class SHETrainer(pl.LightningModule if LIGHTNING_AVAILABLE else nn.Module):
    """
    Training wrapper for SHE neural networks
    """
    
    def __init__(self, 
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 task_type: str = "classification"):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.task_type = task_type
        
        # Loss function
        if task_type == "classification":
            self.loss_fn = nn.CrossEntropyLoss()
        elif task_type == "regression":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def forward(self, batch):
        return self.model(**batch)
    
    def training_step(self, batch, batch_idx):
        if not LIGHTNING_AVAILABLE:
            raise NotImplementedError("Use manual training loop")
        
        logits = self.forward(batch)
        loss = self.loss_fn(logits, batch['label'])
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if not LIGHTNING_AVAILABLE:
            raise NotImplementedError("Use manual training loop")
        
        logits = self.forward(batch)
        loss = self.loss_fn(logits, batch['label'])
        self.log('val_loss', loss)
        
        # Compute accuracy for classification
        if self.task_type == "classification":
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == batch['label']).float() / len(batch['label'])
            self.log('val_acc', acc)
    
    def configure_optimizers(self):
        if not LIGHTNING_AVAILABLE:
            raise NotImplementedError("Configure optimizer manually")
        
        optimizer = optim.Adam(self.parameters(), 
                              lr=self.learning_rate, 
                              weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

class SHENeuralEngine:
    """
    High-level interface for SHE neural network operations
    """
    
    def __init__(self, config: Optional[SHEConfig] = None):
        self.config = config or SHEConfig()
        self.models: Dict[str, nn.Module] = {}
        self.trainers: Dict[str, SHETrainer] = {}
    
    def create_scn_model(self, 
                        name: str,
                        in_channels_0: int
