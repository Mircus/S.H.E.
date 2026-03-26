"""Data loading utilities for SHE."""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
import numpy as np

from .complex import SHESimplicialComplex

logger = logging.getLogger(__name__)


class SHEDataLoader:
    """Load graphs / data into :class:`SHESimplicialComplex` instances."""

    @staticmethod
    def from_weighted_networkx(
        G: nx.Graph,
        weight_attr: str = "weight",
        include_cliques: bool = True,
        max_clique_size: int = 4,
    ) -> SHESimplicialComplex:
        """Build a simplicial complex from a NetworkX graph.

        Nodes and edges are added directly.  If *include_cliques* is ``True``,
        maximal cliques up to *max_clique_size* are lifted to higher-order
        simplices whose weight equals the mean edge weight within the clique.
        """
        sc = SHESimplicialComplex("from_weighted_networkx")

        for node, data in G.nodes(data=True):
            node_data = dict(data)
            node_data["weight"] = node_data.get(weight_attr, 1.0)
            sc.add_node(node, **node_data)

        for u, v, data in G.edges(data=True):
            edge_data = dict(data)
            edge_data["weight"] = edge_data.get(weight_attr, 1.0)
            sc.add_edge((u, v), **edge_data)

        if include_cliques:
            try:
                for clique in nx.find_cliques(G):
                    if 3 <= len(clique) <= max_clique_size:
                        edge_weights = []
                        for i in range(len(clique)):
                            for j in range(i + 1, len(clique)):
                                if G.has_edge(clique[i], clique[j]):
                                    edge_weights.append(
                                        G[clique[i]][clique[j]].get(weight_attr, 1.0)
                                    )
                        clique_weight = float(np.mean(edge_weights)) if edge_weights else 1.0
                        sc.add_simplex(list(clique), weight=clique_weight)
            except Exception as exc:
                logger.warning("Could not compute weighted cliques: %s", exc)

        return sc
