from pathlib import Path

from she_geofield.dblp.experiments import run_experiment


ROOT = Path(__file__).resolve().parents[1]


def test_dblp_experiment_pipeline_runs(tmp_path):
    config_path = tmp_path / "dblp_test.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_file: {ROOT / 'tests' / 'fixtures' / 'dblp_sample.xml'}",
                f"output_dir: {tmp_path / 'out'}",
                "start_year: 2020",
                "end_year: 2023",
                "window_width: 2",
                "window_stride: 1",
                "min_team_size: 2",
                "max_team_size: 4",
                "max_simplex_size: 3",
                "weight_mode: contained",
                "field_location: edges",
                "field_mode: support_count",
                "prediction_horizon: 1",
                "top_k: 3",
                "secondary_target_mode: future_branching",
                "secondary_candidate_regime: bridge_emergence",
                "secondary_output_prefix: bridge_emergence",
                "secondary_top_k: 3",
                "tertiary_target_mode: future_branching",
                "tertiary_candidate_regime: bridge_emergence",
                "tertiary_output_prefix: triangle_branching",
                "tertiary_top_k: 3",
                "tertiary_field_location: triangles",
                "tertiary_score_location: triangles",
                "internal_steps: 2",
                "dt: 0.5",
                "eta: 0.1",
                "lam: 0.3",
                "mu: 0.2",
                "publication_types: article,inproceedings",
            ]
        )
    )

    outdir = run_experiment(config_path)

    assert (outdir / "window_summary.csv").exists()
    assert (outdir / "model_comparison.csv").exists()
    assert (outdir / "top_collaboration_simplices.csv").exists()
    assert (outdir / "predictive_comparison.png").exists()
    assert (outdir / "bridge_emergence_model_comparison.csv").exists()
    assert (outdir / "bridge_emergence_top_collaboration_simplices.csv").exists()
    assert (outdir / "bridge_emergence_predictive_comparison.png").exists()
    assert (outdir / "triangle_branching_model_comparison.csv").exists()
    assert (outdir / "triangle_branching_top_collaboration_simplices.csv").exists()
    assert (outdir / "triangle_branching_predictive_comparison.png").exists()


def test_dblp_aggregation_pipeline_runs(tmp_path):
    config_path = tmp_path / "dblp_aggregation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_family: aggregation",
                "event_type: birth",
                f"input_file: {ROOT / 'tests' / 'fixtures' / 'dblp_sample.xml'}",
                f"output_dir: {tmp_path / 'agg_out'}",
                "start_year: 2020",
                "end_year: 2023",
                "window_width: 2",
                "window_stride: 1",
                "min_team_size: 2",
                "max_team_size: 4",
                "max_simplex_size: 3",
                "weight_mode: contained",
                "field_mode: support_count",
                "prediction_horizon: 1",
                "top_k: 3",
                "internal_steps: 2",
                "dt: 0.5",
                "eta: 0.1",
                "lam: 0.3",
                "mu: 0.2",
                "unit_types: edges,triangles,neighborhoods",
                "publication_types: article,inproceedings",
            ]
        )
    )

    outdir = run_experiment(config_path)

    assert (outdir / "window_summary.csv").exists()
    assert (outdir / "model_comparison.csv").exists()
    assert (outdir / "top_collaboration_simplices.csv").exists()
    assert (outdir / "predictive_comparison.png").exists()
