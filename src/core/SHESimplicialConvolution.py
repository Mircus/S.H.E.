"""
SHESimplicialConvolution (EXPERIMENTAL)
=======================================

.. warning::
    This module is **experimental** and not part of the stable v0.1 API.
    It requires PyTorch and is not covered by the public import path.
    Optional dependency: ``pip install she[ml]``


A minimal, self-contained implementation of **simplicial convolution blocks**
focused on the 1-skeleton (edge features) with optional coupling to nodes (0-simplices)
and faces (2-simplices). The block supports the canonical Hodge-decomposition
message terms for edges:

    x1' = W_self x1
         + W_grad (B1^T x0)         # gradient inflow from nodes
         + W_curl (B2 x2)           # curl inflow from faces
         - alpha * (L1 x1)          # diffusion (Hodge Laplacian on edges)
         + b

where:
    B1: boundary matrix edges->nodes (shape [n0, n1])
    B2: boundary matrix faces->edges (shape [n1, n2])
    L1 = B1^T B1 + B2 B2^T

This file does **not** depend on torch-geometric or external graph libs; it only
expects SciPy sparse matrices for B1/B2 or precomputed L1. It converts them to
PyTorch sparse tensors internally for efficient multiplications.

The module also provides a simple `create_scn_model(...)` that stacks a few
edge-conv blocks into a small SCN.

Usage (example):
----------------
>>> from scipy.sparse import csr_matrix
>>> import numpy as np, torch
>>> from core.SHESimplicialConvolution import build_hodge_laplacians, SimplicialEdgeConv, create_scn_model
>>>
>>> # toy complex: triangle (3 nodes, 3 edges, 1 face)
>>> # nodes (0,1,2), edges ((0,1),(1,2),(0,2)), face (0,1,2)
>>> # B1: edges->nodes  shape [n0=3, n1=3]
>>> rows = [0,1,1,2,0,2]  # u,v for edges (0,1), (1,2), (0,2)
>>> cols = [0,0,1,1,2,2]
>>> vals = [-1, +1, -1, +1, -1, +1]
>>> B1 = csr_matrix((vals, (rows, cols)), shape=(3,3))
>>> # B2: faces->edges shape [n1=3, n2=1]
>>> # face (0,1,2) touches edges (0,1), (0,2), (1,2) with an orientation
>>> B2 = csr_matrix(([+1, -1, +1], ([0,1,2],[0,0,0])), shape=(3,1))
>>> L0, L1, L2 = build_hodge_laplacians(B1, B2)
>>>
>>> conv = SimplicialEdgeConv(in_channels=8, out_channels=16, alpha=0.1, use_grad=True, use_curl=True)
>>> x0 = torch.randn(3, 4)   # (optional) node features (unused if use_grad=False)
>>> x1 = torch.randn(3, 8)   # edge features
>>> x2 = torch.randn(1, 6)   # (optional) face features (unused if use_curl=False)
>>> out = conv(x1, B1=B1, B2=B2, L1=L1, x0=x0, x2=x2)
>>> print(out.shape)  # (3, 16)

Notes
-----
* If you **only** want diffusion on edges, set `use_grad=False, use_curl=False` and
  pass just `x1` and `L1`.
* If you pass `B1/B2`, `L1` is optional (will be computed on the fly). If you pass
  only `L1`, grad/curl terms are skipped unless you also provide `B1/B2` and `x0/x2`.

Author: SCN minimalization for SHE core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- Sparse helpers ---------

def _to_torch_sparse(mat: csr_matrix) -> torch.Tensor:
    """Convert a SciPy CSR into a PyTorch sparse COO tensor (CPU)."""
    if not isspmatrix_csr(mat):
        raise TypeError("Expected SciPy csr_matrix.")
    mat = mat.tocoo()
    indices = torch.vstack((torch.from_numpy(mat.row.astype(np.int64)),
                            torch.from_numpy(mat.col.astype(np.int64))))
    values = torch.from_numpy(mat.data.astype(np.float32))
    shape = torch.Size(mat.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def build_hodge_laplacians(B1: Optional[csr_matrix], B2: Optional[csr_matrix]) -> Tuple[Optional[csr_matrix], csr_matrix, Optional[csr_matrix]]:
    """Return (L0, L1, L2) from boundary matrices.

    L0 = B1 @ B1^T
    L1 = B1^T @ B1 + B2 @ B2^T
    L2 = B2^T @ B2

    Any of B1 or B2 may be None; missing terms are skipped.
    """
    L0 = None
    L1 = None
    L2 = None
    if B1 is not None:
        L0_part = B1 @ B1.T
        L1_part = B1.T @ B1
        L0 = L0_part.tocsr()
        L1 = L1_part.tocsr() if L1 is None else (L1 + L1_part).tocsr()
    if B2 is not None:
        L1_part2 = B2 @ B2.T
        L2_part = B2.T @ B2
        L1 = L1_part2.tocsr() if L1 is None else (L1 + L1_part2).tocsr()
        L2 = L2_part.tocsr()
    if L1 is None:
        raise ValueError("build_hodge_laplacians: Need at least B1 or B2 to compute L1.")
    return L0, L1, L2


# --------- Core block: Edge-centric simplicial convolution ---------

class SimplicialEdgeConv(nn.Module):
    """Edge-centric simplicial convolution with optional grad/curl sources and L1 diffusion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha: float = 0.1,
        use_grad: bool = True,
        use_curl: bool = True,
        bias: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_grad = use_grad
        self.use_curl = use_curl
        self.alpha = float(alpha)

        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_grad = nn.Linear(in_channels if use_grad else 1, out_channels, bias=False) if use_grad else None
        self.lin_curl = nn.Linear(in_channels if use_curl else 1, out_channels, bias=False) if use_curl else None
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x, inplace=True)
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        if self.activation in (None, "none"):
            return x
        # default
        return F.relu(x, inplace=True)

    @torch.no_grad()
    def _edge_from_nodes(self, B1: csr_matrix, x0: torch.Tensor) -> torch.Tensor:
        """Compute B1^T @ x0 (node-to-edge aggregate)."""
        torch_B1t = _to_torch_sparse(B1.T).to(x0.device)
        return torch.sparse.mm(torch_B1t, x0)

    @torch.no_grad()
    def _edge_from_faces(self, B2: csr_matrix, x2: torch.Tensor) -> torch.Tensor:
        """Compute B2 @ x2 (face-to-edge aggregate)."""
        torch_B2 = _to_torch_sparse(B2).to(x2.device)
        return torch.sparse.mm(torch_B2, x2)

    @torch.no_grad()
    def _laplacian_apply(self, L1: csr_matrix, x1: torch.Tensor) -> torch.Tensor:
        """Compute (L1 @ x1)."""
        torch_L1 = _to_torch_sparse(L1).to(x1.device)
        return torch.sparse.mm(torch_L1, x1)

    def forward(
        self,
        x1: torch.Tensor,
        *,
        B1: Optional[csr_matrix] = None,
        B2: Optional[csr_matrix] = None,
        L1: Optional[csr_matrix] = None,
        x0: Optional[torch.Tensor] = None,
        x2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward.

        Args:
            x1: (n_edges, Cin) edge features.
            B1, B2: SciPy CSR boundary matrices.
            L1: optional SciPy CSR edge-Laplacian; computed if missing but B1/B2 present.
            x0: (n_nodes, C0) node features, required if use_grad=True.
            x2: (n_faces, C2) face features, required if use_curl=True.

        Returns:
            (n_edges, Cout) tensor.
        """
        if self.use_grad and (B1 is None or x0 is None):
            raise ValueError("SimplicialEdgeConv: use_grad=True requires B1 and x0.")
        if self.use_curl and (B2 is None or x2 is None):
            raise ValueError("SimplicialEdgeConv: use_curl=True requires B2 and x2.")
        if L1 is None:
            if B1 is None and B2 is None:
                raise ValueError("SimplicialEdgeConv: need L1 or (B1/B2) to compute diffusion term.")
            _, L1, _ = build_hodge_laplacians(B1, B2)

        # Base self linear
        h = self.lin_self(x1)

        # Gradient inflow: B1^T @ x0
        if self.use_grad:
            g = self._edge_from_nodes(B1, x0)
            if g.shape[1] != x1.shape[1]:
                # project to edge feature dim before linear
                g = F.linear(g, weight=torch.eye(g.shape[1], x1.shape[1], device=g.device)[:x1.shape[1]])
            h = h + self.lin_grad(g)

        # Curl inflow: B2 @ x2
        if self.use_curl:
            c = self._edge_from_faces(B2, x2)
            if c.shape[1] != x1.shape[1]:
                c = F.linear(c, weight=torch.eye(c.shape[1], x1.shape[1], device=c.device)[:x1.shape[1]])
            h = h + self.lin_curl(c)

        # Diffusion: -alpha * L1 x1
        if self.alpha != 0.0:
            diff = self._laplacian_apply(L1, x1)
            h = h - self.alpha * diff

        if self.bias is not None:
            h = h + self.bias

        h = self._apply_activation(h)
        h = self.dropout(h)
        return h


# --------- Simple model builder ---------

class SCN(nn.Module):
    """A tiny SCN that stacks edge-conv blocks."""
    def __init__(self, in_channels: int, hidden: int, out_channels: int, num_layers: int = 2, **block_kwargs) -> None:
        super().__init__()
        layers = []
        last_c = in_channels
        for i in range(num_layers):
            c_out = out_channels if i == num_layers - 1 else hidden
            layers.append(SimplicialEdgeConv(last_c, c_out, **block_kwargs))
            last_c = c_out
        self.layers = nn.ModuleList(layers)

    def forward(self, x1, *, B1=None, B2=None, L1=None, x0=None, x2=None):
        for i, layer in enumerate(self.layers):
            x1 = layer(x1, B1=B1, B2=B2, L1=L1, x0=x0, x2=x2)
        return x1


def create_scn_model(
    name: str,
    in_channels_1: int,
    out_channels: int,
    in_channels_0: Optional[int] = None,
    in_channels_2: Optional[int] = None,
    hidden: int = 64,
    num_layers: int = 2,
    alpha: float = 0.1,
    use_grad: bool = True,
    use_curl: bool = True,
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Module:
    """Factory for a small SCN model (edge-centric).

    Parameters
    ----------
    name : str
        Identifier (unused—kept for compatibility).
    in_channels_1 : int
        Edge feature dimension.
    out_channels : int
        Output dimension for the final layer.
    in_channels_0 : Optional[int]
        Node feature dimension (required at runtime if use_grad=True).
    in_channels_2 : Optional[int]
        Face feature dimension (required at runtime if use_curl=True).
    hidden : int
        Hidden channels for the intermediate layers.
    num_layers : int
        Number of stacked SimplicialEdgeConv layers.
    alpha : float
        Diffusion coefficient on L1.
    use_grad : bool
        Whether to include B1^T x0 inflow.
    use_curl : bool
        Whether to include B2 x2 inflow.
    activation : str
        "relu", "gelu", "tanh", or "none".
    dropout : float
        Dropout probability applied after each layer.

    Returns
    -------
    torch.nn.Module
        An SCN that consumes (x1, B1/B2/L1, x0, x2) in forward().
    """
    block_kwargs = dict(alpha=alpha, use_grad=use_grad, use_curl=use_curl, activation=activation, dropout=dropout)
    return SCN(in_channels=in_channels_1, hidden=hidden, out_channels=out_channels, num_layers=num_layers, **block_kwargs)


__all__ = [
    "build_hodge_laplacians",
    "SimplicialEdgeConv",
    "SCN",
    "create_scn_model",
]
