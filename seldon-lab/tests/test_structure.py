from pathlib import Path

from seldon_lab.laws.reports import write_law_candidate_report
from seldon_lab.ontology.state import AggregationState
from seldon_lab.ontology.thresholds import AggregationThresholds
from seldon_lab.trajectories.summaries import summarize_track
from seldon_lab.trajectories.tracks import AggregationTrack


def test_state_thresholds_are_explicit():
    thresholds = AggregationThresholds()
    state = AggregationState(
        closure=thresholds.closure,
        persistence=thresholds.persistence,
        activity=thresholds.activity,
        boundary_role=0.2,
        growth_potential=0.4,
        support=thresholds.support,
    )
    assert state.as_dict()["closure"] == thresholds.closure


def test_track_summary_has_gains():
    track = AggregationTrack(
        aggregation_id="triangle:a-b-c",
        unit_type="triangle_seed",
        windows=("2018-2020", "2019-2021"),
        members_by_window=(("a", "b", "c"), ("a", "b", "c")),
        states_by_window=(
            {"closure": 0.5, "persistence": 0.5, "activity": 1.0, "boundary_role": 0.4},
            {"closure": 1.0, "persistence": 1.0, "activity": 1.6, "boundary_role": 0.2},
        ),
    )
    summary = summarize_track(track)
    assert summary["closure_gain"] == 0.5
    assert summary["boundary_shift"] == -0.2


def test_law_candidate_report_writer(tmp_path: Path):
    path = write_law_candidate_report(
        tmp_path / "law.md",
        title="Birth law candidate",
        candidate_regularity="Closure plus activity precedes stable birth.",
        cross_venue_evidence="Seen on SDM and WSDM.",
        counterexamples="Thin edge seeds without closure usually fail.",
        confidence="moderate",
        unclear="Need stronger neighborhood aggregation tracking.",
    )
    text = path.read_text(encoding="utf-8")
    assert "Birth law candidate" in text
    assert "Seen on SDM and WSDM." in text
