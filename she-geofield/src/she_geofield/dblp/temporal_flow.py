from copy import deepcopy
import math

from ..flow import iterate_coupled
from .baselines import (
    author_productivity,
    author_weighted_degree,
    edge_scores_from_node_scores,
    frozen_diffusion_scores,
    graph_bridge_scores,
    persistence_scores,
    project_edge_scores_to_triangles,
    simplex_support_scores,
    triangle_scores_from_node_scores,
)
from .decorations import apply_window_decorations
from .fields import collaboration_activation_field
from .lift import build_window_complex
from .metrics import (
    aggregation_birth_target,
    aggregation_decay_target,
    aggregation_reinforcement_target,
    aggregation_state_vector,
    bridge_emergence_candidates,
    bridge_to_cluster_candidates,
    bridge_to_cluster_score,
    binary_triangle_emergence_target,
    edge_to_triangle_expansion_target,
    future_branching_targets,
    future_support_targets,
    future_triangle_branching_targets,
    spearman_against_future,
    top_k_precision,
    triangle_emergence_candidates,
)
from .records import AggregationSnapshot, DblpSimplicialComplex, TimeWindow


def _normalized_log_support(
    complex_: DblpSimplicialComplex,
    *,
    location: str = "edges",
) -> dict[tuple[str, ...], float]:
    simplices = complex_.edges if location == "edges" else complex_.triangles
    support_scores = {
        simplex: math.log1p(float(complex_.simplex_data.get(simplex, {}).get("support_count", 0.0)))
        for simplex in simplices
    }
    max_support = max(support_scores.values(), default=1.0)
    if max_support <= 0.0:
        return {simplex: 0.0 for simplex in simplices}
    return {simplex: value / max_support for simplex, value in support_scores.items()}


def _hybrid_evolving_scores(
    complex_: DblpSimplicialComplex,
    raw_scores: dict[tuple[str, ...], float],
    *,
    location: str = "edges",
) -> dict[tuple[str, ...], float]:
    """Blend local field evolution with persistence/support stabilization.

    The raw evolved field alone underperformed strongly on the first real venue
    slices. This hybrid keeps the geometry-field signal but tempers it with
    temporal persistence and bounded support information so the score reflects
    actual recurring collaboration carriers rather than only within-window flow.
    """
    simplices = complex_.edges if location == "edges" else complex_.triangles
    persistence = {
        simplex: float(complex_.simplex_data.get(simplex, {}).get("persistence", 0.0))
        for simplex in simplices
    }
    support = _normalized_log_support(complex_, location=location)

    return {
        simplex: (
            0.45 * raw_scores[simplex]
            + 0.35 * persistence[simplex]
            + 0.20 * support[simplex]
        )
        for simplex in simplices
    }


def build_temporal_complex_sequence(
    windows: list[TimeWindow],
    *,
    weight_mode: str = "contained",
    max_simplex_size: int = 3,
) -> list[DblpSimplicialComplex]:
    complexes = [
        build_window_complex(window, weight_mode=weight_mode, max_simplex_size=max_simplex_size)
        for window in windows
    ]
    apply_window_decorations(complexes)
    return complexes


def build_candidate_aggregations(
    complex_: DblpSimplicialComplex,
    *,
    field_scores: dict[tuple[str, str], float] | None = None,
    unit_types: tuple[str, ...] = ("edges", "triangles", "neighborhoods"),
) -> list[AggregationSnapshot]:
    snapshots: list[AggregationSnapshot] = []

    if "edges" in unit_types:
        for edge in complex_.edges:
            state = aggregation_state_vector(
                complex_,
                edge,
                anchor_simplex=edge,
                field_scores=field_scores,
            )
            snapshots.append(
                AggregationSnapshot(
                    aggregation_id=f"edge:{'-'.join(edge)}",
                    unit_type="edge_seed",
                    members=edge,
                    anchor_simplex=edge,
                    window_label=complex_.label,
                    state=state,
                )
            )

    if "triangles" in unit_types:
        for triangle in complex_.triangles:
            state = aggregation_state_vector(
                complex_,
                triangle,
                anchor_simplex=triangle,
                field_scores=field_scores,
            )
            snapshots.append(
                AggregationSnapshot(
                    aggregation_id=f"triangle:{'-'.join(triangle)}",
                    unit_type="triangle_seed",
                    members=triangle,
                    anchor_simplex=triangle,
                    window_label=complex_.label,
                    state=state,
                )
            )

    if "neighborhoods" in unit_types:
        for triangle in complex_.triangles:
            member_set = set(triangle)
            for other in complex_.triangles:
                if other == triangle:
                    continue
                if len(set(triangle) & set(other)) >= 2:
                    member_set.update(other)
            members = tuple(sorted(member_set))
            state = aggregation_state_vector(
                complex_,
                members,
                anchor_simplex=triangle,
                field_scores=field_scores,
            )
            snapshots.append(
                AggregationSnapshot(
                    aggregation_id=f"neighborhood:{'-'.join(triangle)}",
                    unit_type="local_neighborhood",
                    members=members,
                    anchor_simplex=triangle,
                    window_label=complex_.label,
                    state=state,
                )
            )

    return snapshots


def match_aggregations(
    current: list[AggregationSnapshot],
    future: list[AggregationSnapshot],
    *,
    min_overlap: float = 0.5,
) -> dict[str, AggregationSnapshot | None]:
    matches: dict[str, AggregationSnapshot | None] = {}
    for snapshot in current:
        best: AggregationSnapshot | None = None
        best_score = 0.0
        current_members = set(snapshot.members)
        for candidate in future:
            if snapshot.unit_type != candidate.unit_type:
                continue
            future_members = set(candidate.members)
            overlap = len(current_members & future_members) / max(len(current_members | future_members), 1)
            if overlap < min_overlap:
                continue
            score = overlap + 0.05 * candidate.state.get("support", 0.0)
            if score > best_score:
                best = candidate
                best_score = score
        matches[snapshot.aggregation_id] = best
    return matches


def evaluate_aggregation_events(
    complexes: list[DblpSimplicialComplex],
    *,
    internal_steps: int,
    dt: float,
    eta: float,
    lam: float,
    mu: float,
    field_mode: str,
    horizon: int,
    top_k: int,
    event_type: str,
    unit_types: tuple[str, ...] = ("edges", "triangles", "neighborhoods"),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    comparison_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []

    for idx, complex_ in enumerate(complexes[:-horizon or None]):
        field_scores = run_window_dynamics(
            complex_,
            internal_steps=internal_steps,
            dt=dt,
            eta=eta,
            lam=lam,
            mu=mu,
            field_mode=field_mode,
            field_location="edges",
        )["evolving_geometry"]
        current_aggs = build_candidate_aggregations(
            complex_,
            field_scores=field_scores,
            unit_types=unit_types,
        )
        future_field_scores = run_window_dynamics(
            complexes[idx + horizon],
            internal_steps=internal_steps,
            dt=dt,
            eta=eta,
            lam=lam,
            mu=mu,
            field_mode=field_mode,
            field_location="edges",
        )["evolving_geometry"]
        future_aggs = build_candidate_aggregations(
            complexes[idx + horizon],
            field_scores=future_field_scores,
            unit_types=unit_types,
        )
        matches = match_aggregations(current_aggs, future_aggs)

        if event_type == "birth":
            targets = aggregation_birth_target(current_aggs, matches)
        elif event_type == "reinforcement":
            targets = aggregation_reinforcement_target(current_aggs, matches)
        elif event_type == "decay":
            targets = aggregation_decay_target(current_aggs, matches)
        else:
            raise ValueError(f"unsupported event type: {event_type}")

        # Baseline law-candidate models compared against each event target.
        # author_level_proxy: uses the aggregation's internal_activity score
        # (sum of field/support scores over all edges whose both endpoints are
        # members of the aggregation) as a simple author-productivity proxy.
        # This measures how much co-authorship weight is concentrated inside
        # the unit, without reference to higher-order geometry or boundary
        # structure.  It serves as a node/edge-level baseline to test whether
        # geometric observables (curvature, boundary role, growth potential)
        # provide signal beyond raw activity counts.
        model_scores = {
            "support_only": {
                snapshot.aggregation_id: snapshot.state["support"] for snapshot in current_aggs
            },
            "persistence_only": {
                snapshot.aggregation_id: snapshot.state["persistence"] for snapshot in current_aggs
            },
            "author_level_proxy": {
                snapshot.aggregation_id: snapshot.state["activity"] for snapshot in current_aggs
            },
            "evolving_geometry": {
                snapshot.aggregation_id: 0.4 * snapshot.state["activity"]
                + 0.3 * snapshot.state["boundary_activity"]
                + 0.3 * snapshot.state["growth_potential"]
                for snapshot in current_aggs
            },
            "aggregation_birth_score": {
                snapshot.aggregation_id: (
                    0.30 * snapshot.state["growth_potential"]
                    + 0.25 * snapshot.state["boundary_role"]
                    + 0.20 * snapshot.state["adjacent_activity"]
                    + 0.15 * snapshot.state["activity"]
                    + 0.10 * (1.0 - snapshot.state["persistence"])
                )
                for snapshot in current_aggs
            },
        }

        for model_name, scores in model_scores.items():
            precision = top_k_precision(scores, targets, k=min(top_k, len(scores)))
            rho = spearman_against_future(scores, targets)
            comparison_rows.append(
                {
                    "window": complex_.label,
                    "event_type": event_type,
                    "model": model_name,
                    "candidate_count": len(current_aggs),
                    "positive_targets": sum(1 for value in targets.values() if value > 0.0),
                    "top_k_precision": precision,
                    "spearman_future_support": rho,
                }
            )

        for snapshot in sorted(
            current_aggs,
            key=lambda candidate: model_scores["aggregation_birth_score"][candidate.aggregation_id],
            reverse=True,
        )[:top_k]:
            top_rows.append(
                {
                    "window": complex_.label,
                    "event_type": event_type,
                    "aggregation_id": snapshot.aggregation_id,
                    "unit_type": snapshot.unit_type,
                    "members": "-".join(snapshot.members),
                    "score": model_scores["aggregation_birth_score"][snapshot.aggregation_id],
                    "target": targets.get(snapshot.aggregation_id, 0.0),
                    "closure": snapshot.state["closure"],
                    "persistence": snapshot.state["persistence"],
                    "activity": snapshot.state["activity"],
                }
            )

    return comparison_rows, top_rows


def run_window_dynamics(
    complex_: DblpSimplicialComplex,
    *,
    internal_steps: int,
    dt: float,
    eta: float,
    lam: float,
    mu: float,
    field_mode: str,
    field_location: str,
) -> dict[str, dict]:
    x0 = collaboration_activation_field(
        complex_,
        location=field_location,
        mode=field_mode,
    )
    frozen_scores = frozen_diffusion_scores(
        complex_,
        x0,
        dt=dt,
        steps=internal_steps,
        location=field_location,
    )

    evolving_complex = deepcopy(complex_)
    trajectory, triangle_weights, _edge_weights = iterate_coupled(
        evolving_complex,
        x0,
        steps=internal_steps,
        dt=dt,
        eta=eta,
        lam=lam,
        mu=mu,
        field_location=field_location,
    )
    simplices = evolving_complex.edges if field_location == "edges" else evolving_complex.triangles
    raw_evolving_scores = {
        simplex: float(trajectory[-1][idx])
        for idx, simplex in enumerate(simplices)
    }
    evolving_scores = _hybrid_evolving_scores(complex_, raw_evolving_scores, location=field_location)
    output = {
        "frozen_diffusion": frozen_scores,
        "evolving_geometry": evolving_scores,
        "evolving_field_only": raw_evolving_scores,
        "evolved_triangle_weights": triangle_weights[-1],
    }
    if field_location == "triangles":
        edge_x0 = collaboration_activation_field(
            complex_,
            location="edges",
            mode=field_mode,
        )
        edge_frozen = frozen_diffusion_scores(
            complex_,
            edge_x0,
            dt=dt,
            steps=internal_steps,
            location="edges",
        )
        edge_evolving = run_window_dynamics(
            complex_,
            internal_steps=internal_steps,
            dt=dt,
            eta=eta,
            lam=lam,
            mu=mu,
            field_mode=field_mode,
            field_location="edges",
        )["evolving_geometry"]
        output["dyad_projection"] = project_edge_scores_to_triangles(complex_, edge_evolving)
        output["dyad_frozen_projection"] = project_edge_scores_to_triangles(complex_, edge_frozen)
        output["mixed_geometry"] = {
            triangle: 0.5 * output["evolving_geometry"][triangle] + 0.5 * output["dyad_projection"][triangle]
            for triangle in complex_.triangles
        }
    return output


def evaluate_models(
    complexes: list[DblpSimplicialComplex],
    *,
    internal_steps: int,
    dt: float,
    eta: float,
    lam: float,
    mu: float,
    field_mode: str,
    field_location: str,
    horizon: int,
    top_k: int,
    target_mode: str = "future_support",
    candidate_regime: str = "all",
    bridge_quantile: float = 0.8,
    max_persistence: float = 0.5,
    score_location: str = "edges",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    comparison_rows: list[dict[str, object]] = []
    top_simplex_rows: list[dict[str, object]] = []

    for idx, complex_ in enumerate(complexes[:-horizon or None]):
        if target_mode == "future_support":
            future_targets = future_support_targets(
                complexes,
                idx,
                location="edges" if score_location == "edges" else "triangles",
                horizon=horizon,
            )
        elif target_mode == "future_branching":
            future_targets = (
                future_branching_targets(complexes, idx, horizon=horizon)
                if score_location == "edges"
                else future_triangle_branching_targets(complexes, idx, horizon=horizon)
            )
        elif target_mode == "edge_to_triangle_expansion":
            future_targets = edge_to_triangle_expansion_target(complexes, idx, horizon=horizon)
        elif target_mode == "binary_triangle_emergence":
            future_targets = binary_triangle_emergence_target(complexes, idx, horizon=horizon)
        else:
            raise ValueError(f"unsupported target_mode: {target_mode}")

        if candidate_regime == "all":
            candidates = set(complex_.edges if score_location == "edges" else complex_.triangles)
        elif candidate_regime == "bridge_emergence":
            candidates = (
                bridge_emergence_candidates(
                    complex_,
                    bridge_quantile=bridge_quantile,
                    max_persistence=max_persistence,
                )
                if score_location == "edges"
                else triangle_emergence_candidates(
                    complex_,
                    max_persistence=max_persistence,
                )
            )
        elif candidate_regime == "bridge_to_cluster":
            if score_location != "edges":
                raise ValueError("bridge_to_cluster candidates are currently defined for edges only")
            candidates = bridge_to_cluster_candidates(
                complex_,
                bridge_quantile=bridge_quantile,
                max_persistence=max_persistence,
            )
        else:
            raise ValueError(f"unsupported candidate_regime: {candidate_regime}")

        masked_targets = {simplex: future_targets.get(simplex, 0.0) for simplex in candidates}
        node_degree = author_weighted_degree(complex_)
        node_productivity = author_productivity(complex_.records)
        if score_location == "edges":
            model_scores = {
                "author_degree": edge_scores_from_node_scores(complex_, node_degree),
                "author_productivity": edge_scores_from_node_scores(complex_, node_productivity),
                "graph_bridge": graph_bridge_scores(complex_),
                "simplex_support": simplex_support_scores(complex_, location="edges"),
                "persistence_only": persistence_scores(complex_, location="edges"),
            }
        else:
            model_scores = {
                "dyad_degree_projection": triangle_scores_from_node_scores(complex_, node_degree),
                "dyad_productivity_projection": triangle_scores_from_node_scores(complex_, node_productivity),
                "triangle_support": simplex_support_scores(complex_, location="triangles"),
                "triangle_persistence": persistence_scores(complex_, location="triangles"),
            }
        model_scores.update(
            run_window_dynamics(
                complex_,
                internal_steps=internal_steps,
                dt=dt,
                eta=eta,
                lam=lam,
                mu=mu,
                field_mode=field_mode,
                field_location=score_location,
            )
        )
        if score_location == "edges":
            model_scores["btc_score"] = bridge_to_cluster_score(
                complex_,
                evolving_scores=model_scores["evolving_geometry"],
                bridge_scores=model_scores["graph_bridge"],
            )

        for model_name, scores in model_scores.items():
            if model_name in {"evolved_triangle_weights", "evolving_field_only"}:
                continue
            masked_scores = {simplex: scores.get(simplex, 0.0) for simplex in candidates}
            precision = top_k_precision(masked_scores, masked_targets, k=min(top_k, len(masked_scores)))
            rho = spearman_against_future(masked_scores, masked_targets)
            comparison_rows.append(
                {
                    "window": complex_.label,
                    "model": model_name,
                    "target_mode": target_mode,
                    "candidate_regime": candidate_regime,
                    "score_location": score_location,
                    "candidate_count": len(candidates),
                    "positive_targets": sum(1 for value in masked_targets.values() if value > 0.0),
                    "top_k_precision": precision,
                    "spearman_future_support": rho,
                }
            )

        evolving_scores = model_scores["evolving_geometry"]
        for simplex, score in sorted(
            ((simplex, evolving_scores[simplex]) for simplex in candidates),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]:
            data = complex_.simplex_data.get(simplex, {})
            top_simplex_rows.append(
                {
                    "window": complex_.label,
                    "simplex": "-".join(simplex),
                    "target_mode": target_mode,
                    "candidate_regime": candidate_regime,
                    "score": score,
                    "support_count": data.get("support_count", 0.0),
                    "persistence": data.get("persistence", 0.0),
                    "future_support": masked_targets.get(simplex, 0.0),
                }
            )
    return comparison_rows, top_simplex_rows
