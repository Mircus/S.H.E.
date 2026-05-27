import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from she_geofield.dblp.experiments import run_experiment
from seldon_lab.datasets.dblp import load_seldon_experiment_config
from seldon_lab.summaries.case_studies import export_event_cases
from seldon_lab.viz.case_studies import plot_case_study_panel, write_case_study_csv
from seldon_lab.viz.trajectories import plot_aggregation_trajectories


def _read_case_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _resolve_mechanism_output_dir(input_config: Path) -> Path:
    from she_geofield.dblp.experiments import load_config

    config = load_config(input_config)
    root = input_config.parent.parent if input_config.parent.name == "configs" else input_config.parent
    return (root / str(config["output_dir"])).resolve()


def _select_birth_cases(case_rows: list[dict[str, str]], *, max_cases: int = 5) -> list[dict[str, str]]:
    positives = [row for row in case_rows if row.get("case_label") == "positive"]
    failed = [row for row in case_rows if row.get("case_label") == "failed"]
    chosen: list[dict[str, str]] = []
    chosen.extend(positives[: max_cases - 1])
    if len(chosen) < max_cases and failed:
        chosen.append(failed[0])
    return chosen[:max_cases]


def _build_case_transition_trajectories(case_rows: list[dict[str, str]]) -> dict[str, list[dict[str, object]]]:
    trajectories: dict[str, list[dict[str, object]]] = {}
    for row in case_rows:
        aggregation_id = str(row["aggregation_id"])
        window = str(row["window"])
        trajectories[aggregation_id] = [
            {
                "window": f"{window} before",
                "closure": float(row.get("before_closure", 0.0)),
                "persistence": float(row.get("before_persistence", 0.0)),
                "activity": float(row.get("before_activity", 0.0)),
                "boundary_role": float(row.get("before_boundary_role", 0.0)),
            },
            {
                "window": f"{window} after",
                "closure": float(row.get("after_closure", 0.0)),
                "persistence": float(row.get("after_persistence", 0.0)),
                "activity": float(row.get("after_activity", 0.0)),
                "boundary_role": float(row.get("after_boundary_role", 0.0)),
            },
        ]
    return trajectories


def run_birth_experiment(config_path: str | Path, *, reports_dir: str | Path | None = None) -> dict[str, Path]:
    experiment = load_seldon_experiment_config(config_path)
    output_dir = _resolve_mechanism_output_dir(experiment.input_config)
    if not (output_dir / "model_comparison.csv").exists():
        output_dir = run_experiment(experiment.input_config)

    reports_root = Path(reports_dir) if reports_dir is not None else experiment.config_path.parents[1] / "reports" / "current"
    case_dir = reports_root / "case_studies"
    figure_dir = reports_root / "figures" / "birth_trajectories"
    artifact_dir = reports_root / "artifacts" / "birth"
    case_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    raw_case_csv = export_event_cases(
        experiment.input_config,
        event_type="birth",
        output_csv=case_dir / f"{experiment.venue.lower()}_birth_cases.csv",
        max_cases=4,
    )
    case_rows = _read_case_rows(raw_case_csv)
    selected = _select_birth_cases(case_rows, max_cases=5)
    curated_csv = write_case_study_csv(
        selected,
        path=artifact_dir / f"{experiment.venue.lower()}_birth_case_selection.csv",
    )
    panel = plot_case_study_panel(
        selected,
        path=artifact_dir / f"{experiment.venue.lower()}_birth_case_panel.png",
    )

    plot_aggregation_trajectories(
        _build_case_transition_trajectories(selected),
        output_dir=figure_dir / experiment.venue.lower(),
    )
    return {
        "experiment_output": output_dir,
        "raw_cases": raw_case_csv,
        "curated_cases": curated_csv,
        "case_panel": panel,
        "trajectory_dir": figure_dir / experiment.venue.lower(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Seldon DBLP birth experiment.")
    parser.add_argument("--config", required=True, help="Path to the Seldon wrapper config.")
    parser.add_argument("--reports-dir", help="Optional reports output directory.")
    args = parser.parse_args()
    outputs = run_birth_experiment(args.config, reports_dir=args.reports_dir)
    print(f"Seldon birth experiment complete. Outputs in {outputs['experiment_output']}")


if __name__ == "__main__":
    main()
