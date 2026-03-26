"""Simplicial complex wrapper built on TopoNetX."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from toponetx.classes.simplicial_complex import SimplicialComplex

from .config import SHEConfig

logger = logging.getLogger(__name__)


class SHESimplicialComplex:
    """Enhanced wrapper around TopoX SimplicialComplex with diffusion capabilities."""

    def __init__(self, name: str = "SHE_Complex", config: Optional[SHEConfig] = None):
        self.name = name
        self.config = config or SHEConfig()
        self.complex = SimplicialComplex()
        self.node_features: Dict[Any, Dict[str, Any]] = {}
        self.edge_features: Dict[Tuple, Dict[str, Any]] = {}
        self.face_features: Dict[Tuple, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}
        self._cached_matrices: Dict[str, Any] = {}

    # -- mutators ---------------------------------------------------------

    def add_node(self, node_id: Any, features: Optional[Dict[str, Any]] = None, **attr) -> None:
        """Add a node (0-simplex)."""
        self.complex.add_node(node_id, **attr)
        if features:
            self.node_features[node_id] = features

    def add_edge(self, edge: Tuple, features: Optional[Dict[str, Any]] = None, **attr) -> None:
        """Add an edge (1-simplex)."""
        self.complex.add_simplex(edge, rank=1, **attr)
        if features:
            self.edge_features[edge] = features

    def add_simplex(
        self,
        simplex: Union[List, Tuple],
        rank: Optional[int] = None,
        features: Optional[Dict[str, Any]] = None,
        **attr,
    ) -> None:
        """Add a general k-simplex."""
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

    # -- queries ----------------------------------------------------------

    def get_hodge_laplacians(self, use_cache: bool = True) -> Dict[int, csr_matrix]:
        """Hodge Laplacian matrices for all dimensions via TopoX."""
        if use_cache and "hodge_laplacians" in self._cached_matrices:
            return self._cached_matrices["hodge_laplacians"]

        laplacians: Dict[int, csr_matrix] = {}
        max_dim = min(self.complex.dim, self.config.max_dimension)

        for k in range(max_dim + 1):
            try:
                L_k = self.complex.hodge_laplacian_matrix(rank=k, signed=True)
                if L_k is not None and L_k.shape[0] > 0:
                    laplacians[k] = L_k.tocsr()
                    logger.info("Computed Hodge Laplacian L_%d with shape %s", k, L_k.shape)
            except Exception as exc:
                logger.warning("Could not compute Hodge Laplacian L_%d: %s", k, exc)

        if use_cache:
            self._cached_matrices["hodge_laplacians"] = laplacians
        return laplacians

    def get_incidence_matrices(self, use_cache: bool = True) -> Dict[str, csr_matrix]:
        """Boundary / incidence matrices via TopoX."""
        if use_cache and "incidence_matrices" in self._cached_matrices:
            return self._cached_matrices["incidence_matrices"]

        matrices: Dict[str, csr_matrix] = {}
        max_dim = min(self.complex.dim, self.config.max_dimension)

        for k in range(max_dim):
            try:
                B_k = self.complex.incidence_matrix(rank=k, to_rank=k + 1, signed=True)
                if B_k is not None and B_k.shape[0] > 0:
                    matrices[f"B_{k}"] = B_k.tocsr()
                    logger.info("Computed incidence matrix B_%d with shape %s", k, B_k.shape)
            except Exception as exc:
                logger.warning("Could not compute incidence matrix B_%d: %s", k, exc)

        if use_cache:
            self._cached_matrices["incidence_matrices"] = matrices
        return matrices

    def get_simplex_weights(self, dimension: int) -> Dict[Any, float]:
        """Extract weights for simplices of the given dimension."""
        weights: Dict[Any, float] = {}
        try:
            for simplex in self.complex.skeleton(dimension):
                attrs = self.complex.get_simplex_attributes(simplex, dimension)
                weights[simplex] = attrs.get("weight", 1.0)
        except Exception as exc:
            logger.warning("Could not extract weights for dimension %d: %s", dimension, exc)
        return weights

    def get_simplex_list(self, dimension: int) -> List[Any]:
        """Ordered list of simplices for *dimension*."""
        try:
            return list(self.complex.skeleton(dimension))
        except Exception:
            return []
