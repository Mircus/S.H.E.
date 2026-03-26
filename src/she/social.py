"""Social / group-level analysis on decorated hyperstructures.

This module provides domain-shaped functions for ranking diffusers,
finding bridge simplices, and scoring group cohesion -- the kind of
questions that motivate SHE beyond generic simplicial analysis.

Every function takes an :class:`SHEHyperstructure` and returns
interpretable results keyed to the entities and relations that were
originally registered, not just anonymous simplex indices.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from .config import SHEConfig
from .diffusion import SHEHodgeDiffusion
from .hyperstructure import SHEHyperstructure

logger = logging.getLogger(__name__)


# -- result containers ----------------------------------------------------


@dataclass
class RankedItem:
    """One entry in a diffuser / influencer ranking."""

    target: Any
    dimension: int
    score: float
    metadata: Dict[str, Any]


@dataclass
class BridgeSimplex:
    """A simplex that spans multiple communities."""

    members: frozenset
    dimension: int
    communities_spanned: List[Any]
    bridge_score: float
    metadata: Dict[str, Any]


@dataclass
class CohesionScore:
    """Structural cohesion assessment for a candidate group."""

    members: frozenset
    score: float
    components: Dict[str, float]


# -- ranking --------------------------------------------------------------


def rank_diffusers(
    hs: SHEHyperstructure,
    dimension: Optional[int] = None,
    top_k: int = 20,
) -> List[RankedItem]:
    """Rank simplices by diffusion centrality, decorated with relation metadata.

    If *dimension* is ``None``, ranks across all dimensions and merges.
    Results are sorted descending by score.
    """
    sc = hs.complex
    analyzer = SHEHodgeDiffusion(hs.config)
    result = analyzer.analyze_diffusion(sc)

    items: List[RankedItem] = []
    dims = [dimension] if dimension is not None else sorted(result.key_diffusers)

    for dim in dims:
        for simplex, score in result.key_diffusers.get(dim, []):
            members = _simplex_members(simplex)
            meta = hs.get_relation_attrs(members) if dim >= 1 else hs.get_entity_attrs(members[0] if members else None)
            items.append(RankedItem(target=members, dimension=dim, score=score, metadata=meta))

    items.sort(key=lambda r: r.score, reverse=True)
    return items[:top_k]


def rank_entity_diffusers(hs: SHEHyperstructure, top_k: int = 20) -> List[RankedItem]:
    """Rank entities (dimension 0) by diffusion centrality."""
    return rank_diffusers(hs, dimension=0, top_k=top_k)


def rank_simplex_diffusers(
    hs: SHEHyperstructure, dimension: int = 1, top_k: int = 20
) -> List[RankedItem]:
    """Rank higher-order relations by diffusion centrality."""
    return rank_diffusers(hs, dimension=dimension, top_k=top_k)


# -- bridge detection -----------------------------------------------------


def find_bridge_simplices(
    hs: SHEHyperstructure,
    community_attr: str = "community",
    min_communities: int = 2,
    top_k: int = 10,
) -> List[BridgeSimplex]:
    """Find simplices whose members span multiple communities.

    Community membership is read from entity attributes under
    *community_attr*.  A simplex is a bridge if its members belong to
    at least *min_communities* distinct communities.

    The bridge score is ``n_communities / n_members * weight``, a simple
    heuristic that rewards spanning many communities with few members and
    high weight.  This is intentionally not a deep topological invariant --
    it is a legible first-pass signal.
    """
    bridges: List[BridgeSimplex] = []
    sc = hs.complex

    max_dim = sc.complex.dim
    for dim in range(1, max_dim + 1):
        for simplex in sc.get_simplex_list(dim):
            members = _simplex_members(simplex)
            communities = set()
            for m in members:
                c = hs.get_entity_attrs(m).get(community_attr)
                if c is not None:
                    communities.add(c)

            if len(communities) >= min_communities:
                rel = hs.get_relation_attrs(members)
                weight = rel.get("weight", 1.0)
                bridge_score = len(communities) / max(len(members), 1) * weight
                bridges.append(
                    BridgeSimplex(
                        members=frozenset(members),
                        dimension=dim,
                        communities_spanned=sorted(communities, key=str),
                        bridge_score=bridge_score,
                        metadata=rel,
                    )
                )

    bridges.sort(key=lambda b: b.bridge_score, reverse=True)
    return bridges[:top_k]


# -- cohesion -------------------------------------------------------------


def group_cohesion(
    hs: SHEHyperstructure,
    group: Sequence[Any],
) -> CohesionScore:
    """Score the structural cohesion of a candidate group of entities.

    Combines three signals:

    * **relation_weight** -- total weight of relations whose members are all
      within *group*.
    * **sub_relation_density** -- fraction of possible sub-relations (pairs,
      triples, ...) that actually exist in the hyperstructure.
    * **higher_order_support** -- number of relations of dimension >= 2 that
      are fully contained in *group*, normalised by group size.

    The final score is the geometric mean of the three components (each
    clipped to [0, 1] for the density/support terms).  This is a
    deliberately simple composite; it is not a topological invariant.
    """
    group_set = frozenset(group)
    members_list = sorted(group_set, key=str)

    # collect all relations fully contained in group
    contained: List[Dict[str, Any]] = []
    for rel_key in hs.relations:
        if rel_key <= group_set:
            contained.append(hs.get_relation_attrs(rel_key))

    # component 1: total relation weight
    total_weight = sum(r.get("weight", 1.0) for r in contained)
    # normalise by number of members so larger groups need proportionally more weight
    norm_weight = total_weight / max(len(group_set), 1)

    # component 2: sub-relation density (pairs that exist / pairs possible)
    n = len(group_set)
    possible_pairs = n * (n - 1) / 2 if n >= 2 else 1
    existing_pairs = sum(1 for r in hs.relations if r <= group_set and len(r) == 2)
    density = existing_pairs / possible_pairs

    # component 3: higher-order support
    higher = sum(1 for r in hs.relations if r <= group_set and len(r) >= 3)
    ho_support = min(higher / max(n - 2, 1), 1.0)

    # geometric mean
    vals = [max(v, 1e-12) for v in [norm_weight, density, ho_support]]
    score = float(np.prod(vals) ** (1.0 / len(vals)))

    return CohesionScore(
        members=group_set,
        score=score,
        components={
            "relation_weight": norm_weight,
            "sub_relation_density": density,
            "higher_order_support": ho_support,
        },
    )


# -- graph baseline comparison --------------------------------------------


def rank_influencers(
    hs: SHEHyperstructure,
    top_k: int = 10,
) -> Dict[str, List[RankedItem]]:
    """Compare graph-level node centrality with simplex-level diffusion ranking.

    Returns a dict with two keys:

    * ``"graph_centrality"`` -- entities ranked by eigenvector centrality on
      the 1-skeleton (the ordinary graph that discards higher-order structure).
    * ``"simplex_diffusion"`` -- entities/simplices ranked by Hodge diffusion.

    The contrast makes explicit what higher-order analysis can reveal that
    graph-only methods miss.
    """
    # -- graph centrality on 1-skeleton -----------------------------------
    G = nx.Graph()
    sc = hs.complex
    for simplex in sc.get_simplex_list(1):
        members = _simplex_members(simplex)
        if len(members) == 2:
            rel = hs.get_relation_attrs(members)
            G.add_edge(members[0], members[1], weight=rel.get("weight", 1.0))

    try:
        centrality = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        centrality = nx.degree_centrality(G)

    graph_items: List[RankedItem] = []
    for node, score in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        graph_items.append(
            RankedItem(
                target=[node],
                dimension=0,
                score=score,
                metadata=hs.get_entity_attrs(node),
            )
        )

    # -- simplex diffusion (all dims) -------------------------------------
    simplex_items = rank_diffusers(hs, top_k=top_k)

    return {"graph_centrality": graph_items, "simplex_diffusion": simplex_items}


# -- helpers --------------------------------------------------------------


def _simplex_members(simplex: Any) -> List[Any]:
    """Extract a plain list of member ids from a TopoNetX Simplex or tuple."""
    if hasattr(simplex, "elements"):
        return sorted(simplex.elements, key=str)
    if hasattr(simplex, "__iter__") and not isinstance(simplex, str):
        return sorted(simplex, key=str)
    return [simplex]
