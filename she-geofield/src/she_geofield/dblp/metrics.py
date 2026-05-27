from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from .records import AggregationSnapshot, DblpSimplicialComplex


def group_persistence(complexes: list[DblpSimplicialComplex], *, location: str = "edges") -> dict:
    scores: Counter[tuple[str, ...]] = Counter()
    for complex_ in complexes:
        simplices = complex_.edges if location == "edges" else complex_.triangles
        for simplex in simplices:
            scores[simplex] += 1.0
    return dict(scores)


def future_support_targets(
    complexes: list[DblpSimplicialComplex],
    window_index: int,
    *,
    location: str = "edges",
    horizon: int = 1,
) -> dict:
    targets: Counter[tuple[str, ...]] = Counter()
    for future_complex in complexes[window_index + 1 : window_index + 1 + horizon]:
        simplices = future_complex.edges if location == "edges" else future_complex.triangles
        for simplex in simplices:
            targets[simplex] += float(
                future_complex.simplex_data.get(simplex, {}).get("support_count", 0.0)
            )
    return dict(targets)


def future_branching_targets(
    complexes: list[DblpSimplicialComplex],
    window_index: int,
    *,
    horizon: int = 1,
) -> dict[tuple[str, str], float]:
    """Score edges by future branching into novel triangle contexts.

    For each current edge, count future triangle support that introduces a new
    third collaborator relative to the current window. This targets emerging or
    bridge-like collaboration expansion rather than simple edge recurrence.
    """
    current_complex = complexes[window_index]
    seen_third_authors: dict[tuple[str, str], set[str]] = {
        edge: set() for edge in current_complex.edges
    }
    for triangle in current_complex.triangles:
        a, b, c = triangle
        for edge, third in [((a, b), c), ((a, c), b), ((b, c), a)]:
            seen_third_authors[tuple(sorted(edge))].add(third)

    targets: Counter[tuple[str, str]] = Counter()
    for future_complex in complexes[window_index + 1 : window_index + 1 + horizon]:
        for triangle in future_complex.triangles:
            support = float(future_complex.simplex_data.get(triangle, {}).get("support_count", 0.0))
            a, b, c = triangle
            for edge, third in [((a, b), c), ((a, c), b), ((b, c), a)]:
                normalized_edge = tuple(sorted(edge))
                if third not in seen_third_authors.get(normalized_edge, set()):
                    targets[normalized_edge] += support
    return dict(targets)


def future_triangle_branching_targets(
    complexes: list[DblpSimplicialComplex],
    window_index: int,
    *,
    horizon: int = 1,
) -> dict[tuple[str, str, str], float]:
    """Score triangles by future expansion into new adjacent triangle contexts."""
    current_complex = complexes[window_index]
    current_edge_contexts: dict[tuple[str, str], set[str]] = {}
    for triangle in current_complex.triangles:
        a, b, c = triangle
        for edge, third in [((a, b), c), ((a, c), b), ((b, c), a)]:
            normalized_edge = tuple(sorted(edge))
            current_edge_contexts.setdefault(normalized_edge, set()).add(third)

    targets: Counter[tuple[str, str, str]] = Counter()
    for future_complex in complexes[window_index + 1 : window_index + 1 + horizon]:
        for triangle in future_complex.triangles:
            support = float(future_complex.simplex_data.get(triangle, {}).get("support_count", 0.0))
            future_edges = [tuple(sorted((triangle[0], triangle[1]))),
                            tuple(sorted((triangle[0], triangle[2]))),
                            tuple(sorted((triangle[1], triangle[2])))]
            for current_triangle in current_complex.triangles:
                if current_triangle == triangle:
                    continue
                a, b, c = current_triangle
                current_edges = [tuple(sorted((a, b))), tuple(sorted((a, c))), tuple(sorted((b, c)))]
                shared_edges = set(current_edges) & set(future_edges)
                if not shared_edges:
                    continue
                branched = False
                for shared_edge in shared_edges:
                    future_third = next(vertex for vertex in triangle if vertex not in shared_edge)
                    if future_third not in current_edge_contexts.get(shared_edge, set()):
                        branched = True
                        break
                if branched:
                    targets[current_triangle] += support
    return dict(targets)


def edge_to_triangle_expansion_target(
    complexes: list[DblpSimplicialComplex],
    window_index: int,
    *,
    horizon: int = 1,
    persistence_bonus: float = 0.0,
) -> dict[tuple[str, str], float]:
    """Graded bridge-to-cluster target for edge candidates.

    Count future triangles containing the edge whose third author is new
    relative to the current window. Optionally add a bonus when the new
    triangle context persists across multiple future windows.
    """
    current_complex = complexes[window_index]
    current_contexts: dict[tuple[str, str], set[str]] = {edge: set() for edge in current_complex.edges}
    for triangle in current_complex.triangles:
        a, b, c = triangle
        for edge, third in [((a, b), c), ((a, c), b), ((b, c), a)]:
            current_contexts[tuple(sorted(edge))].add(third)

    future_context_hits: dict[tuple[str, str], Counter[str]] = {
        edge: Counter() for edge in current_complex.edges
    }
    for future_complex in complexes[window_index + 1 : window_index + 1 + horizon]:
        for triangle in future_complex.triangles:
            support = float(future_complex.simplex_data.get(triangle, {}).get("support_count", 0.0))
            a, b, c = triangle
            for edge, third in [((a, b), c), ((a, c), b), ((b, c), a)]:
                normalized_edge = tuple(sorted(edge))
                if third not in current_contexts.get(normalized_edge, set()):
                    future_context_hits.setdefault(normalized_edge, Counter())[third] += support

    targets: dict[tuple[str, str], float] = {}
    for edge, counter in future_context_hits.items():
        value = 0.0
        for third, support in counter.items():
            future_presence = sum(
                1.0
                for future_complex in complexes[window_index + 1 : window_index + 1 + horizon]
                if tuple(sorted((*edge, third))) in future_complex.triangles
            )
            value += support + persistence_bonus * max(future_presence - 1.0, 0.0)
        targets[edge] = value
    return targets


def binary_triangle_emergence_target(
    complexes: list[DblpSimplicialComplex],
    window_index: int,
    *,
    horizon: int = 1,
) -> dict[tuple[str, str], float]:
    graded = edge_to_triangle_expansion_target(complexes, window_index, horizon=horizon)
    return {edge: 1.0 if value > 0.0 else 0.0 for edge, value in graded.items()}


def bridge_emergence_candidates(
    complex_: DblpSimplicialComplex,
    *,
    bridge_quantile: float = 0.8,
    max_persistence: float = 0.5,
    min_support: float = 1.0,
) -> set[tuple[str, str]]:
    """Select edges where recurrence is weak but bridge structure is plausible."""
    from .baselines import graph_bridge_scores

    if not complex_.edges:
        return set()

    bridge_scores = graph_bridge_scores(complex_)
    sorted_scores = sorted(bridge_scores.values())
    cutoff_index = min(
        max(int(bridge_quantile * len(sorted_scores)), 0),
        max(len(sorted_scores) - 1, 0),
    )
    bridge_cutoff = sorted_scores[cutoff_index] if sorted_scores else 0.0

    candidates = set()
    for edge in complex_.edges:
        data = complex_.simplex_data.get(edge, {})
        if float(bridge_scores.get(edge, 0.0)) < bridge_cutoff:
            continue
        if float(data.get("persistence", 0.0)) > max_persistence:
            continue
        if float(data.get("support_count", 0.0)) < min_support:
            continue
        candidates.add(edge)
    return candidates


def bridge_to_cluster_candidates(
    complex_: DblpSimplicialComplex,
    *,
    bridge_quantile: float = 0.8,
    max_persistence: float = 0.5,
    max_triangle_support: float = 1.0,
    min_support: float = 1.0,
) -> set[tuple[str, str]]:
    """Thin bridge-like edges with room to clusterize later."""
    from .baselines import graph_bridge_scores

    bridge_scores = graph_bridge_scores(complex_)
    sorted_scores = sorted(bridge_scores.values())
    if not sorted_scores:
        return set()
    cutoff_index = min(
        max(int(bridge_quantile * len(sorted_scores)), 0),
        max(len(sorted_scores) - 1, 0),
    )
    bridge_cutoff = sorted_scores[cutoff_index]

    candidates = set()
    for edge in complex_.edges:
        data = complex_.simplex_data.get(edge, {})
        if float(bridge_scores.get(edge, 0.0)) < bridge_cutoff:
            continue
        if float(data.get("persistence", 0.0)) > max_persistence:
            continue
        if float(data.get("triangle_support_count", 0.0)) > max_triangle_support:
            continue
        if float(data.get("support_count", 0.0)) < min_support:
            continue
        candidates.add(edge)
    return candidates


def triangle_emergence_candidates(
    complex_: DblpSimplicialComplex,
    *,
    max_persistence: float = 0.5,
    min_support: float = 1.0,
) -> set[tuple[str, str, str]]:
    """Select triangles that are present but not yet strongly recurrent."""
    candidates = set()
    for triangle in complex_.triangles:
        data = complex_.simplex_data.get(triangle, {})
        if float(data.get("persistence", 0.0)) > max_persistence:
            continue
        if float(data.get("support_count", 0.0)) < min_support:
            continue
        candidates.add(triangle)
    return candidates


def local_expansion_features(
    complex_: DblpSimplicialComplex,
    *,
    field_scores: dict[tuple[str, str], float] | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Latent closure and neighborhood reorganization features around edges."""
    neighbors: dict[str, set[str]] = {vertex: set() for vertex in complex_.vertices}
    for u, v in complex_.edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    current_contexts: dict[tuple[str, str], set[str]] = {edge: set() for edge in complex_.edges}
    for triangle in complex_.triangles:
        a, b, c = triangle
        for edge, third in [((a, b), c), ((a, c), b), ((b, c), a)]:
            current_contexts[tuple(sorted(edge))].add(third)

    features: dict[tuple[str, str], dict[str, float]] = {}
    for edge in complex_.edges:
        u, v = edge
        common_neighbors = (neighbors[u] & neighbors[v]) - current_contexts.get(edge, set())
        open_wedge_count = float(len(common_neighbors))

        endpoint_union = (neighbors[u] | neighbors[v]) - {u, v}
        neighborhood_diversity = float(len(endpoint_union))

        off_bridge_activity = 0.0
        for w in common_neighbors:
            uw = tuple(sorted((u, w)))
            vw = tuple(sorted((v, w)))
            off_bridge_activity += float(complex_.simplex_data.get(uw, {}).get("support_count", 0.0))
            off_bridge_activity += float(complex_.simplex_data.get(vw, {}).get("support_count", 0.0))

        triangle_support = float(complex_.simplex_data.get(edge, {}).get("triangle_support_count", 0.0))
        field_value = float(field_scores.get(edge, 0.0)) if field_scores is not None else 0.0

        features[edge] = {
            "open_wedge_count": open_wedge_count,
            "neighborhood_diversity": neighborhood_diversity,
            "off_bridge_activity": off_bridge_activity,
            "triangle_support_count": triangle_support,
            "field_score": field_value,
        }
    return features


def bridge_to_cluster_score(
    complex_: DblpSimplicialComplex,
    *,
    evolving_scores: dict[tuple[str, str], float],
    bridge_scores: dict[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    """Score thin bridges by current bridge-likeness plus closure potential."""
    def _normalize(values: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
        max_value = max(values.values(), default=0.0)
        if max_value <= 0.0:
            return {key: 0.0 for key in values}
        return {key: value / max_value for key, value in values.items()}

    persistence = {
        edge: float(complex_.simplex_data.get(edge, {}).get("persistence", 0.0))
        for edge in complex_.edges
    }
    triangle_support = {
        edge: float(complex_.simplex_data.get(edge, {}).get("triangle_support_count", 0.0))
        for edge in complex_.edges
    }
    expansion_features = local_expansion_features(complex_, field_scores=evolving_scores)

    normalized_bridge = _normalize(bridge_scores)
    normalized_evolving = _normalize(evolving_scores)
    normalized_open_wedges = _normalize(
        {edge: feats["open_wedge_count"] for edge, feats in expansion_features.items()}
    )
    normalized_off_bridge = _normalize(
        {edge: feats["off_bridge_activity"] for edge, feats in expansion_features.items()}
    )
    normalized_diversity = _normalize(
        {edge: feats["neighborhood_diversity"] for edge, feats in expansion_features.items()}
    )
    normalized_triangle_support = _normalize(triangle_support)

    scores: dict[tuple[str, str], float] = {}
    for edge in complex_.edges:
        bridge_now = (
            normalized_bridge[edge]
            * (1.0 - persistence[edge])
            * (1.0 - normalized_triangle_support[edge])
        )
        expansion_potential = (
            0.45 * normalized_open_wedges[edge]
            + 0.35 * normalized_off_bridge[edge]
            + 0.20 * normalized_diversity[edge]
        )
        scores[edge] = (
            0.35 * bridge_now
            + 0.35 * normalized_evolving[edge]
            + 0.30 * expansion_potential
        )
    return scores


def aggregation_boundary_features(
    complex_: DblpSimplicialComplex,
    members: tuple[str, ...],
) -> dict[str, float]:
    member_set = set(members)
    internal_edges = 0.0
    boundary_edges = 0.0
    outward_neighbors: set[str] = set()
    for u, v in complex_.edges:
        in_u = u in member_set
        in_v = v in member_set
        if in_u and in_v:
            internal_edges += 1.0
        elif in_u or in_v:
            boundary_edges += 1.0
            outward_neighbors.add(v if in_u else u)
    total = internal_edges + boundary_edges
    internal_ratio = internal_edges / total if total else 0.0
    boundary_ratio = boundary_edges / total if total else 0.0
    return {
        "internal_edge_ratio": internal_ratio,
        "boundary_edge_ratio": boundary_ratio,
        "outward_adjacency_count": float(len(outward_neighbors)),
    }


def aggregation_growth_features(
    complex_: DblpSimplicialComplex,
    members: tuple[str, ...],
    *,
    field_scores: dict[tuple[str, str], float] | None = None,
) -> dict[str, float]:
    member_set = set(members)
    latent_closure = 0.0
    adjacent_activity = 0.0
    neighboring_regions: set[tuple[str, ...]] = set()
    for triangle in complex_.triangles:
        triangle_set = set(triangle)
        overlap = len(member_set & triangle_set)
        if overlap >= 2 and not triangle_set.issubset(member_set):
            latent_closure += 1.0
            neighboring_regions.add(triangle)
            for edge in [
                tuple(sorted((triangle[0], triangle[1]))),
                tuple(sorted((triangle[0], triangle[2]))),
                tuple(sorted((triangle[1], triangle[2]))),
            ]:
                adjacent_activity += float(
                    field_scores.get(edge, 0.0) if field_scores is not None else complex_.simplex_data.get(edge, {}).get("support_count", 0.0)
                )
    return {
        "latent_closure": latent_closure,
        "adjacent_region_count": float(len(neighboring_regions)),
        "adjacent_activity": adjacent_activity,
    }


def aggregation_state_vector(
    complex_: DblpSimplicialComplex,
    members: tuple[str, ...],
    *,
    anchor_simplex: tuple[str, ...] | None = None,
    field_scores: dict[tuple[str, str], float] | None = None,
) -> dict[str, float]:
    member_set = set(members)
    triangles_inside = [
        triangle for triangle in complex_.triangles if set(triangle).issubset(member_set)
    ]
    possible_triangles = max(len(member_set) * (len(member_set) - 1) * (len(member_set) - 2) / 6.0, 1.0)
    closure = len(triangles_inside) / possible_triangles if len(member_set) >= 3 else 0.0

    anchor = anchor_simplex or members
    anchor_data = complex_.simplex_data.get(anchor, {})
    support = float(anchor_data.get("support_count", 0.0))
    persistence = float(anchor_data.get("persistence", 0.0))

    internal_activity = 0.0
    boundary_activity = 0.0
    for edge in complex_.edges:
        edge_set = set(edge)
        score = float(
            field_scores.get(edge, 0.0)
            if field_scores is not None
            else complex_.simplex_data.get(edge, {}).get("support_count", 0.0)
        )
        if edge_set.issubset(member_set):
            internal_activity += score
        elif edge_set & member_set:
            boundary_activity += score

    boundary = aggregation_boundary_features(complex_, members)
    growth = aggregation_growth_features(complex_, members, field_scores=field_scores)
    state = {
        "closure": closure,
        "persistence": persistence,
        "activity": internal_activity,
        "support": support,
        "internal_activity": internal_activity,
        "boundary_activity": boundary_activity,
        "boundary_role": boundary["boundary_edge_ratio"],
        "growth_potential": growth["latent_closure"],
        "outward_adjacency_count": boundary["outward_adjacency_count"],
        "adjacent_activity": growth["adjacent_activity"],
    }
    return state


def is_bona_fide_aggregation(
    state: dict[str, float],
    *,
    closure_threshold: float = 0.2,
    persistence_threshold: float = 0.5,
    activity_threshold: float = 1.0,
    support_threshold: float = 1.0,
) -> bool:
    return (
        state.get("closure", 0.0) >= closure_threshold
        and state.get("persistence", 0.0) >= persistence_threshold
        and state.get("activity", 0.0) >= activity_threshold
        and state.get("support", 0.0) >= support_threshold
    )


def aggregation_birth_target(
    current: list[AggregationSnapshot],
    matched_future: dict[str, AggregationSnapshot | None],
    *,
    closure_threshold: float = 0.2,
    persistence_threshold: float = 0.5,
    activity_threshold: float = 1.0,
    support_threshold: float = 1.0,
) -> dict[str, float]:
    targets: dict[str, float] = {}
    for snapshot in current:
        now_is_agg = is_bona_fide_aggregation(
            snapshot.state,
            closure_threshold=closure_threshold,
            persistence_threshold=persistence_threshold,
            activity_threshold=activity_threshold,
            support_threshold=support_threshold,
        )
        future = matched_future.get(snapshot.aggregation_id)
        future_is_agg = future is not None and is_bona_fide_aggregation(
            future.state,
            closure_threshold=closure_threshold,
            persistence_threshold=persistence_threshold,
            activity_threshold=activity_threshold,
            support_threshold=support_threshold,
        )
        targets[snapshot.aggregation_id] = 1.0 if (not now_is_agg and future_is_agg) else 0.0
    return targets


def aggregation_reinforcement_target(
    current: list[AggregationSnapshot],
    matched_future: dict[str, AggregationSnapshot | None],
) -> dict[str, float]:
    targets: dict[str, float] = {}
    for snapshot in current:
        future = matched_future.get(snapshot.aggregation_id)
        if future is None:
            targets[snapshot.aggregation_id] = 0.0
            continue
        now = snapshot.state
        later = future.state
        gain = (
            0.35 * max(later["closure"] - now["closure"], 0.0)
            + 0.35 * max(later["persistence"] - now["persistence"], 0.0)
            + 0.30 * max(later["activity"] - now["activity"], 0.0)
        )
        targets[snapshot.aggregation_id] = gain
    return targets


def aggregation_decay_target(
    current: list[AggregationSnapshot],
    matched_future: dict[str, AggregationSnapshot | None],
) -> dict[str, float]:
    targets: dict[str, float] = {}
    for snapshot in current:
        future = matched_future.get(snapshot.aggregation_id)
        if future is None:
            targets[snapshot.aggregation_id] = 1.0
            continue
        now = snapshot.state
        later = future.state
        loss = (
            0.35 * max(now["closure"] - later["closure"], 0.0)
            + 0.35 * max(now["persistence"] - later["persistence"], 0.0)
            + 0.30 * max(now["activity"] - later["activity"], 0.0)
        )
        targets[snapshot.aggregation_id] = loss
    return targets


def top_k_precision(scores: dict, targets: dict, *, k: int) -> float:
    if not scores or k <= 0:
        return 0.0
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
    hits = sum(1 for simplex, _score in ranked if targets.get(simplex, 0.0) > 0.0)
    return hits / max(len(ranked), 1)


def spearman_against_future(scores: dict, targets: dict) -> float:
    simplices = sorted(set(scores) | set(targets))
    if len(simplices) < 2:
        return 0.0
    score_vec = np.asarray([scores.get(simplex, 0.0) for simplex in simplices], dtype=float)
    target_vec = np.asarray([targets.get(simplex, 0.0) for simplex in simplices], dtype=float)
    stat = spearmanr(score_vec, target_vec)
    if np.isnan(stat.statistic):
        return 0.0
    return float(stat.statistic)
