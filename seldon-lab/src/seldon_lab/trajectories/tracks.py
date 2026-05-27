from dataclasses import dataclass

from she_geofield.dblp.temporal_flow import build_candidate_aggregations, match_aggregations


@dataclass(frozen=True)
class MatchDiagnostics:
    unmatched_count: int
    ambiguous_count: int
    split_candidate_count: int
    merge_candidate_count: int


@dataclass(frozen=True)
class AggregationTrack:
    aggregation_id: str
    unit_type: str
    windows: tuple[str, ...]
    members_by_window: tuple[tuple[str, ...], ...]
    states_by_window: tuple[dict[str, float], ...]


def build_tracks(
    complexes,
    *,
    field_scores_by_window: list[dict[tuple[str, ...], float] | None] | None = None,
    unit_types: tuple[str, ...] = ("edges", "triangles", "neighborhoods"),
    min_overlap: float = 0.5,
) -> tuple[list[AggregationTrack], MatchDiagnostics]:
    if not complexes:
        return [], MatchDiagnostics(0, 0, 0, 0)

    scores = field_scores_by_window or [None] * len(complexes)
    snapshots_by_window = [
        build_candidate_aggregations(complex_, field_scores=score_map, unit_types=unit_types)
        for complex_, score_map in zip(complexes, scores, strict=True)
    ]

    tracks: list[AggregationTrack] = []
    unmatched_count = 0
    ambiguous_count = 0
    split_candidate_count = 0
    merge_candidate_count = 0

    for window_idx, current in enumerate(snapshots_by_window[:-1]):
        future = snapshots_by_window[window_idx + 1]
        matches = match_aggregations(current, future, min_overlap=min_overlap)
        future_ids = [match.aggregation_id for match in matches.values() if match is not None]
        future_id_counts = {aggregation_id: future_ids.count(aggregation_id) for aggregation_id in set(future_ids)}
        split_candidate_count += sum(1 for count in future_id_counts.values() if count > 1)
        merge_candidate_count += 0
        for snapshot in current:
            match = matches.get(snapshot.aggregation_id)
            if match is None:
                unmatched_count += 1
                continue
            if future_id_counts.get(match.aggregation_id, 0) > 1:
                ambiguous_count += 1
            tracks.append(
                AggregationTrack(
                    aggregation_id=snapshot.aggregation_id,
                    unit_type=snapshot.unit_type,
                    windows=(snapshot.window_label, match.window_label),
                    members_by_window=(snapshot.members, match.members),
                    states_by_window=(snapshot.state, match.state),
                )
            )

    diagnostics = MatchDiagnostics(
        unmatched_count=unmatched_count,
        ambiguous_count=ambiguous_count,
        split_candidate_count=split_candidate_count,
        merge_candidate_count=merge_candidate_count,
    )
    return tracks, diagnostics
