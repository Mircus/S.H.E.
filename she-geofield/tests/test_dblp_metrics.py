from she_geofield.dblp.decorations import apply_window_decorations
from she_geofield.dblp.lift import build_window_complex
from she_geofield.dblp.metrics import (
    binary_triangle_emergence_target,
    bridge_emergence_candidates,
    bridge_to_cluster_candidates,
    bridge_to_cluster_score,
    edge_to_triangle_expansion_target,
    future_branching_targets,
    future_triangle_branching_targets,
)
from she_geofield.dblp.records import PaperRecord, TimeWindow


def test_future_branching_targets_detect_new_triangle_context():
    window_a = TimeWindow(
        start_year=2020,
        end_year=2021,
        records=(
            PaperRecord("p1", 2020, ["alice", "bob"], "V", "article", "AB"),
            PaperRecord("p2", 2021, ["alice", "bob", "carol"], "V", "article", "ABC"),
        ),
    )
    window_b = TimeWindow(
        start_year=2021,
        end_year=2022,
        records=(
            PaperRecord("p3", 2022, ["alice", "bob", "dave"], "V", "article", "ABD"),
        ),
    )

    complexes = [
        build_window_complex(window_a, weight_mode="contained", max_simplex_size=3),
        build_window_complex(window_b, weight_mode="contained", max_simplex_size=3),
    ]
    apply_window_decorations(complexes)

    targets = future_branching_targets(complexes, 0, horizon=1)

    assert targets[("alice", "bob")] == 1.0
    assert targets.get(("alice", "carol"), 0.0) == 0.0


def test_bridge_emergence_candidates_filter_low_persistence_edges():
    window_a = TimeWindow(
        start_year=2020,
        end_year=2021,
        records=(
            PaperRecord("p1", 2020, ["alice", "bob"], "V", "article", "AB"),
            PaperRecord("p2", 2020, ["alice", "carol"], "V", "article", "AC"),
            PaperRecord("p3", 2021, ["alice", "dave"], "V", "article", "AD"),
        ),
    )
    window_b = TimeWindow(
        start_year=2021,
        end_year=2022,
        records=(
            PaperRecord("p4", 2022, ["alice", "bob"], "V", "article", "AB return"),
        ),
    )

    complexes = [
        build_window_complex(window_a, weight_mode="contained", max_simplex_size=3),
        build_window_complex(window_b, weight_mode="contained", max_simplex_size=3),
    ]
    apply_window_decorations(complexes)

    candidates = bridge_emergence_candidates(
        complexes[0],
        bridge_quantile=0.5,
        max_persistence=0.0,
        min_support=1.0,
    )

    assert ("alice", "carol") in candidates or ("alice", "dave") in candidates
    assert ("alice", "bob") not in candidates


def test_future_triangle_branching_targets_reward_new_adjacent_triangles():
    window_a = TimeWindow(
        start_year=2020,
        end_year=2021,
        records=(
            PaperRecord("p1", 2020, ["alice", "bob", "carol"], "V", "article", "ABC"),
        ),
    )
    window_b = TimeWindow(
        start_year=2021,
        end_year=2022,
        records=(
            PaperRecord("p2", 2022, ["alice", "bob", "dave"], "V", "article", "ABD"),
        ),
    )

    complexes = [
        build_window_complex(window_a, weight_mode="contained", max_simplex_size=3),
        build_window_complex(window_b, weight_mode="contained", max_simplex_size=3),
    ]
    apply_window_decorations(complexes)

    targets = future_triangle_branching_targets(complexes, 0, horizon=1)

    assert targets[("alice", "bob", "carol")] == 1.0


def test_edge_to_triangle_expansion_target_counts_new_triangle_contexts():
    window_a = TimeWindow(
        start_year=2020,
        end_year=2021,
        records=(
            PaperRecord("p1", 2020, ["alice", "bob"], "V", "article", "AB"),
        ),
    )
    window_b = TimeWindow(
        start_year=2021,
        end_year=2022,
        records=(
            PaperRecord("p2", 2022, ["alice", "bob", "carol"], "V", "article", "ABC"),
            PaperRecord("p3", 2022, ["alice", "bob", "dave"], "V", "article", "ABD"),
        ),
    )

    complexes = [
        build_window_complex(window_a, weight_mode="contained", max_simplex_size=3),
        build_window_complex(window_b, weight_mode="contained", max_simplex_size=3),
    ]
    apply_window_decorations(complexes)

    graded = edge_to_triangle_expansion_target(complexes, 0, horizon=1)
    binary = binary_triangle_emergence_target(complexes, 0, horizon=1)

    assert graded[("alice", "bob")] == 2.0
    assert binary[("alice", "bob")] == 1.0


def test_bridge_to_cluster_score_prefers_open_bridge_edge():
    window = TimeWindow(
        start_year=2020,
        end_year=2021,
        records=(
            PaperRecord("p1", 2020, ["alice", "bob"], "V", "article", "AB"),
            PaperRecord("p2", 2020, ["alice", "carol"], "V", "article", "AC"),
            PaperRecord("p3", 2020, ["bob", "carol"], "V", "article", "BC"),
            PaperRecord("p4", 2021, ["alice", "dave"], "V", "article", "AD"),
        ),
    )
    complex_ = build_window_complex(window, weight_mode="contained", max_simplex_size=3)
    apply_window_decorations([complex_])

    candidates = bridge_to_cluster_candidates(
        complex_,
        bridge_quantile=0.0,
        max_persistence=1.0,
        max_triangle_support=0.0,
        min_support=1.0,
    )
    scores = bridge_to_cluster_score(
        complex_,
        evolving_scores={edge: 1.0 for edge in complex_.edges},
        bridge_scores={edge: 1.0 for edge in complex_.edges},
    )

    assert ("alice", "dave") in candidates
    assert scores[("alice", "dave")] > 0.0
