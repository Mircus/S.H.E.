"""Minimal visualisation helpers for diffusion results."""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .types import DiffusionResult

logger = logging.getLogger(__name__)

# seaborn is optional
try:
    import seaborn as sns  # noqa: F401

    _SNS = True
except ImportError:
    _SNS = False


class SHEDiffusionVisualizer:
    """Static plotting helpers for :class:`DiffusionResult`."""

    @staticmethod
    def plot_spectrum(
        result: DiffusionResult,
        dimension: int,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot eigenvalue spectrum."""
        if dimension not in result.eigenvalues:
            logger.warning("No eigenvalues for dimension %d", dimension)
            return

        eigenvals = result.eigenvalues[dimension]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(eigenvals, "bo-", markersize=4)
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(title or f"Hodge Laplacian Spectrum - Dimension {dimension}")
        ax.grid(True, alpha=0.3)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_key_diffusers(
        result: DiffusionResult,
        dimension: int,
        top_k: int = 10,
        save_path: Optional[str] = None,
    ) -> None:
        """Bar chart of key diffusers."""
        if dimension not in result.key_diffusers:
            logger.warning("No key diffusers for dimension %d", dimension)
            return

        diffusers = result.key_diffusers[dimension][:top_k]
        if not diffusers:
            return

        labels = [str(d[0]) for d in diffusers]
        scores = [d[1] for d in diffusers]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(labels)), scores)
        ax.set_xlabel("Simplex")
        ax.set_ylabel("Diffusion Centrality")
        ax.set_title(f"Top {top_k} Key Diffusers - Dimension {dimension}")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        for bar, score in zip(bars, scores):
            bar.set_color(plt.cm.viridis(score))
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_diffusion_heatmap(
        result: DiffusionResult,
        dimension: int,
        save_path: Optional[str] = None,
    ) -> None:
        """Heatmap of diffusion map coordinates."""
        key = f"dim_{dimension}"
        if key not in result.diffusion_maps:
            logger.warning("No diffusion map for dimension %d", dimension)
            return

        diff_map = result.diffusion_maps[key]
        if diff_map.size == 0:
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        if _SNS:
            sns.heatmap(
                diff_map.T,
                cmap="RdBu_r",
                center=0,
                cbar_kws={"label": "Diffusion Coordinate"},
                ax=ax,
            )
        else:
            im = ax.imshow(diff_map.T, cmap="RdBu_r", aspect="auto")
            fig.colorbar(im, ax=ax, label="Diffusion Coordinate")
        ax.set_xlabel("Simplex Index")
        ax.set_ylabel("Diffusion Dimension")
        ax.set_title(f"Diffusion Map - Dimension {dimension}")
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
