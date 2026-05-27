import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from she_geofield.dblp.experiments import run_experiment
from she_geofield.dblp.experiments import load_config

from seldon_lab.datasets.dblp import load_seldon_experiment_config
from seldon_lab.summaries.case_studies import export_event_cases


def run_reinforcement_experiment(config_path: str | Path, *, reports_dir: str | Path | None = None) -> dict[str, Path]:
    experiment = load_seldon_experiment_config(config_path)
    config = load_config(experiment.input_config)
    root = experiment.input_config.parent.parent if experiment.input_config.parent.name == "configs" else experiment.input_config.parent
    output_dir = (root / str(config["output_dir"])).resolve()
    if not (output_dir / "model_comparison.csv").exists():
        output_dir = run_experiment(experiment.input_config)
    reports_root = Path(reports_dir) if reports_dir is not None else experiment.config_path.parents[1] / "reports" / "current"
    case_dir = reports_root / "case_studies"
    case_dir.mkdir(parents=True, exist_ok=True)
    case_csv = export_event_cases(
        experiment.input_config,
        event_type="reinforcement",
        output_csv=case_dir / f"{experiment.venue.lower()}_reinforcement_cases.csv",
        max_cases=4,
    )
    return {
        "experiment_output": output_dir,
        "raw_cases": case_csv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Seldon DBLP reinforcement experiment.")
    parser.add_argument("--config", required=True, help="Path to the Seldon wrapper config.")
    parser.add_argument("--reports-dir", help="Optional reports output directory.")
    args = parser.parse_args()
    outputs = run_reinforcement_experiment(args.config, reports_dir=args.reports_dir)
    print(f"Seldon reinforcement experiment complete. Outputs in {outputs['experiment_output']}")


if __name__ == "__main__":
    main()
