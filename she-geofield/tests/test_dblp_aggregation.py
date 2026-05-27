from pathlib import Path

from she_geofield.dblp.decorations import apply_window_decorations
from she_geofield.dblp.lift import build_window_complex
from she_geofield.dblp.metrics import aggregation_birth_target, aggregation_reinforcement_target
from she_geofield.dblp.records import PaperRecord, TimeWindow
from she_geofield.dblp.temporal_flow import (
    build_candidate_aggregations,
    match_aggregations,
    run_window_dynamics,
)


def test_match_aggregations_tracks_overlap():
    window_a = TimeWindow(
        2020,
        2021,
        (
            PaperRecord("p1", 2020, ["alice", "bob", "carol"], "V", "article", "ABC"),
        ),
    )
    window_b = TimeWindow(
        2021,
        2022,
        (
            PaperRecord("p2", 2022, ["alice", "bob", "carol"], "V", "article", "ABC again"),
            PaperRecord("p3", 2022, ["alice", "bob", "dave"], "V", "article", "ABD"),
        ),
    )
    complexes = [
        build_window_complex(window_a, weight_mode="contained", max_simplex_size=3),
        build_window_complex(window_b, weight_mode="contained", max_simplex_size=3),
    ]
    apply_window_decorations(complexes)
    current = build_candidate_aggregations(complexes[0], unit_types=("triangles",))
    future = build_candidate_aggregations(complexes[1], unit_types=("triangles",))

    matches = match_aggregations(current, future, min_overlap=0.5)

    assert matches[current[0].aggregation_id] is not None


def test_aggregation_birth_and_reinforcement_targets_are_defined():
    window_a = TimeWindow(
        2020,
        2021,
        (
            PaperRecord("p1", 2020, ["alice", "bob"], "V", "article", "AB"),
            PaperRecord("p2", 2021, ["alice", "bob", "carol"], "V", "article", "ABC"),
        ),
    )
    window_b = TimeWindow(
        2021,
        2022,
        (
            PaperRecord("p3", 2022, ["alice", "bob", "carol"], "V", "article", "ABC2"),
            PaperRecord("p4", 2022, ["alice", "bob", "carol"], "V", "article", "ABC3"),
        ),
    )
    complexes = [
        build_window_complex(window_a, weight_mode="contained", max_simplex_size=3),
        build_window_complex(window_b, weight_mode="contained", max_simplex_size=3),
    ]
    apply_window_decorations(complexes)
    current = build_candidate_aggregations(complexes[0], unit_types=("edges", "triangles"))
    future = build_candidate_aggregations(complexes[1], unit_types=("edges", "triangles"))
    matches = match_aggregations(current, future, min_overlap=0.5)

    birth = aggregation_birth_target(current, matches)
    reinforcement = aggregation_reinforcement_target(current, matches)

    assert set(birth) == {snapshot.aggregation_id for snapshot in current}
    assert set(reinforcement) == {snapshot.aggregation_id for snapshot in current}


def test_candidate_aggregations_include_neighborhoods():
    window = TimeWindow(
        2020,
        2021,
        (
            PaperRecord("p1", 2020, ["alice", "bob", "carol"], "V", "article", "ABC"),
            PaperRecord("p2", 2021, ["alice", "bob", "dave"], "V", "article", "ABD"),
        ),
    )
    complex_ = build_window_complex(window, weight_mode="contained", max_simplex_size=3)
    apply_window_decorations([complex_])
    field_scores = run_window_dynamics(
        complex_,
        internal_steps=1,
        dt=0.5,
        eta=0.1,
        lam=0.3,
        mu=0.2,
        field_mode="support_count",
        field_location="edges",
    )["evolving_geometry"]
    candidates = build_candidate_aggregations(
        complex_,
        field_scores=field_scores,
        unit_types=("neighborhoods",),
    )

    assert candidates
    assert all(candidate.unit_type == "local_neighborhood" for candidate in candidates)
