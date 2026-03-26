"""Hodge-Laplacian diffusion analysis."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh, spsolve

from .config import SHEConfig
from .complex import SHESimplicialComplex
from .types import DiffusionResult

logger = logging.getLogger(__name__)


class SHEHodgeDiffusion:
    """Diffusion analysis on simplicial complexes via Hodge Laplacians."""

    def __init__(self, config: Optional[SHEConfig] = None):
        self.config = config or SHEConfig()

    # -- spectral ---------------------------------------------------------

    def compute_spectral_properties(
        self, laplacian: csr_matrix, k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalues and eigenvectors of a Hodge Laplacian."""
        k = k or min(self.config.spectral_k, laplacian.shape[0] - 1)

        if laplacian.shape[0] <= 1:
            return np.array([0.0]), np.array([[1.0]])

        try:
            if k >= laplacian.shape[0] - 1:
                eigenvals, eigenvecs = eigh(laplacian.toarray())
            else:
                eigenvals, eigenvecs = eigsh(laplacian, k=k, which="SM", sigma=0.0)

            idx = np.argsort(eigenvals)
            return eigenvals[idx], eigenvecs[:, idx]
        except Exception as exc:
            logger.warning("Spectral computation failed: %s", exc)
            return np.array([0.0]), np.array([[1.0]])

    # -- centrality -------------------------------------------------------

    def compute_diffusion_centrality(
        self,
        laplacian: csr_matrix,
        weights: Dict[Any, float],
        simplex_list: List[Any],
    ) -> Dict[Any, float]:
        """Diffusion centrality scores for simplices.

        Uses a stronger diffusion coupling (lambda=1.0) so the Laplacian
        structure actually influences the result, and rank-percentile
        normalisation so scores spread across [0, 1] instead of clustering
        near 1.0.
        """
        if laplacian.shape[0] == 0:
            return {}

        weight_vector = np.array([weights.get(s, 1.0) for s in simplex_list])

        try:
            lambda_diff = 1.0
            A = diags([1.0], shape=laplacian.shape) + lambda_diff * laplacian
            raw = spsolve(A, weight_vector)

            # rank-percentile normalisation: spreads scores across [0, 1]
            n = len(raw)
            if n <= 1:
                scores = np.ones(n)
            else:
                order = np.argsort(np.abs(raw))
                ranks = np.empty_like(order, dtype=float)
                ranks[order] = np.linspace(0.0, 1.0, n)
                scores = ranks

            return dict(zip(simplex_list, scores))
        except Exception as exc:
            logger.warning("Diffusion centrality computation failed: %s", exc)

        return {s: 1.0 for s in simplex_list}

    # -- heat kernel ------------------------------------------------------

    def compute_heat_kernel(self, laplacian: csr_matrix, t: float = 1.0) -> np.ndarray:
        """Heat kernel exp(-tL)."""
        if laplacian.shape[0] <= 1:
            return np.eye(laplacian.shape[0])

        try:
            eigenvals, eigenvecs = self.compute_spectral_properties(laplacian)
            heat_eigenvals = np.exp(-t * eigenvals)
            return eigenvecs @ np.diag(heat_eigenvals) @ eigenvecs.T
        except Exception as exc:
            logger.warning("Heat kernel computation failed: %s", exc)
            return np.eye(laplacian.shape[0])

    # -- Hodge decomposition ----------------------------------------------

    def hodge_decomposition(
        self, sc: SHESimplicialComplex, dimension: int
    ) -> Dict[str, np.ndarray]:
        """Hodge decomposition for k-forms (harmonic component)."""
        try:
            laplacians = sc.get_hodge_laplacians()
            if dimension not in laplacians:
                return {"harmonic": np.array([]), "exact": np.array([]), "coexact": np.array([])}

            L_k = laplacians[dimension]
            eigenvals, eigenvecs = self.compute_spectral_properties(L_k)

            tol = 1e-6
            harmonic_idx = np.where(np.abs(eigenvals) < tol)[0]
            harmonic = (
                eigenvecs[:, harmonic_idx]
                if len(harmonic_idx) > 0
                else np.array([]).reshape(L_k.shape[0], 0)
            )

            empty = np.array([]).reshape(L_k.shape[0], 0)
            return {"harmonic": harmonic, "exact": empty, "coexact": empty}
        except Exception as exc:
            logger.warning("Hodge decomposition failed for dimension %d: %s", dimension, exc)
            return {"harmonic": np.array([]), "exact": np.array([]), "coexact": np.array([])}

    # -- full analysis ----------------------------------------------------

    def analyze_diffusion(self, sc: SHESimplicialComplex) -> DiffusionResult:
        """Comprehensive diffusion analysis across all dimensions."""
        laplacians = sc.get_hodge_laplacians()

        all_eigenvals: Dict[int, np.ndarray] = {}
        all_eigenvecs: Dict[int, np.ndarray] = {}
        diffusion_maps: Dict[str, np.ndarray] = {}
        key_diffusers: Dict[int, List[Tuple[Any, float]]] = {}
        hodge_decomps: Dict[str, np.ndarray] = {}

        for dim, lap in laplacians.items():
            eigenvals, eigenvecs = self.compute_spectral_properties(lap)
            all_eigenvals[dim] = eigenvals
            all_eigenvecs[dim] = eigenvecs

            if len(eigenvals) > 1:
                start = 1 if eigenvals[0] < 1e-6 else 0
                end = min(start + 3, len(eigenvals))
                diffusion_maps[f"dim_{dim}"] = eigenvecs[:, start:end]

            weights = sc.get_simplex_weights(dim)
            simplex_list = sc.get_simplex_list(dim)
            if weights and simplex_list:
                centrality = self.compute_diffusion_centrality(lap, weights, simplex_list)
                sorted_d = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                key_diffusers[dim] = sorted_d

            hodge_decomps[f"dim_{dim}"] = self.hodge_decomposition(sc, dim)

        return DiffusionResult(
            eigenvalues=all_eigenvals,
            eigenvectors=all_eigenvecs,
            diffusion_maps=diffusion_maps,
            key_diffusers=key_diffusers,
            hodge_decomposition=hodge_decomps,
        )
