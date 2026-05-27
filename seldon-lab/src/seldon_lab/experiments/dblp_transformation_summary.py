import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from seldon_lab.experiments.dblp_birth import run_birth_experiment
from seldon_lab.experiments.dblp_reinforcement import run_reinforcement_experiment
from seldon_lab.laws.candidates import birth_law_candidate_from_rows
from seldon_lab.laws.summaries import summarize_candidate
from seldon_lab.summaries.cross_event import build_summary
from seldon_lab.viz.case_studies import plot_case_study_panel, write_case_study_csv
from seldon_lab.viz.summary_plots import copy_summary_plot


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _select_cross_venue_birth_cases(case_paths: list[Path]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for path in case_paths:
        venue = path.stem.split("_", 1)[0].upper()
        rows = _read_csv(path)
        positives = [row for row in rows if row.get("case_label") == "positive"][:2]
        failed = [row for row in rows if row.get("case_label") == "failed"][:1]
        for row in positives + failed:
            selected.append({"venue": venue, **row})
    return selected[:5]


def run_transformation_summary(*, reports_dir: str | Path | None = None) -> dict[str, Path]:
    root = Path(__file__).resolve().parents[3]
    reports_root = Path(reports_dir) if reports_dir is not None else root / "reports" / "current"
    artifact_root = reports_root / "artifacts"
    figure_root = reports_root / "figures"
    artifact_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    birth_sdm = run_birth_experiment(root / "configs" / "dblp_sdm_aggregation_birth.yaml", reports_dir=reports_root)
    birth_wsdm = run_birth_experiment(root / "configs" / "dblp_wsdm_aggregation_birth.yaml", reports_dir=reports_root)
    reinf_sdm = run_reinforcement_experiment(root / "configs" / "dblp_sdm_reinforcement.yaml", reports_dir=reports_root)
    reinf_wsdm = run_reinforcement_experiment(root / "configs" / "dblp_wsdm_reinforcement.yaml", reports_dir=reports_root)

    summary_dir = artifact_root / "cross_event"
    summary_plot = build_summary(
        {
            "SDM": {
                "birth": birth_sdm["experiment_output"],
                "reinforcement": reinf_sdm["experiment_output"],
            },
            "WSDM": {
                "birth": birth_wsdm["experiment_output"],
                "reinforcement": reinf_wsdm["experiment_output"],
            },
        },
        output_dir=summary_dir,
    )

    summary_rows = _read_csv(summary_dir / "cross_event_summary.csv")
    birth_candidate = birth_law_candidate_from_rows(summary_rows)
    birth_report_path = reports_root / "aggregation_birth_cross_venue.md"
    birth_report_path.write_text(summarize_candidate(birth_candidate), encoding="utf-8")

    case_rows = _select_cross_venue_birth_cases(
        [birth_sdm["raw_cases"], birth_wsdm["raw_cases"]]
    )
    curated_cases = write_case_study_csv(
        case_rows,
        path=artifact_root / "birth_case_studies.csv",
    )
    case_panel = plot_case_study_panel(
        case_rows,
        path=figure_root / "birth_case_studies.png",
    )
    copied_summary = copy_summary_plot(
        summary_plot,
        output_path=figure_root / "cross_event_summary.png",
    )

    return {
        "birth_report": birth_report_path,
        "cross_event_summary": summary_dir / "cross_event_summary.csv",
        "best_predictors": summary_dir / "best_predictors.csv",
        "cross_event_plot": copied_summary,
        "birth_cases": curated_cases,
        "birth_case_plot": case_panel,
        "sdm_birth_trajectory_dir": birth_sdm["trajectory_dir"],
        "wsdm_birth_trajectory_dir": birth_wsdm["trajectory_dir"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Seldon DBLP transformation summary.")
    parser.add_argument("--reports-dir", help="Optional reports output directory.")
    args = parser.parse_args()
    outputs = run_transformation_summary(reports_dir=args.reports_dir)
    print(f"Seldon transformation summary complete. Birth report at {outputs['birth_report']}")


if __name__ == "__main__":
    main()
