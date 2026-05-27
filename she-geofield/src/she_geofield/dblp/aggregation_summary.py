import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open()))


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def build_cross_event_summary(
    event_outputs: dict[str, dict[str, str | Path]],
    *,
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    event_rows: list[dict[str, object]] = []
    predictor_rows: list[dict[str, object]] = []

    for venue, event_map in event_outputs.items():
        for event_type, path in event_map.items():
            model_rows = _read_csv(Path(path) / "model_comparison.csv")
            top_rows = _read_csv(Path(path) / "top_collaboration_simplices.csv")

            positive_counts = [int(float(row["positive_targets"])) for row in model_rows]
            top_precision_by_model: dict[str, float] = defaultdict(float)
            mean_rho_by_model: dict[str, float] = defaultdict(float)
            models = sorted({row["model"] for row in model_rows})
            best_model = None
            best_rho = float("-inf")
            for model in models:
                rows = [row for row in model_rows if row["model"] == model]
                mean_precision = _mean([float(row["top_k_precision"]) for row in rows])
                mean_rho = _mean([float(row["spearman_future_support"]) for row in rows])
                top_precision_by_model[model] = mean_precision
                mean_rho_by_model[model] = mean_rho
                if mean_rho > best_rho:
                    best_rho = mean_rho
                    best_model = model

            closure_vals = [float(row.get("closure", 0.0)) for row in top_rows]
            persistence_vals = [float(row.get("persistence", 0.0)) for row in top_rows]
            activity_vals = [float(row.get("activity", 0.0)) for row in top_rows]
            event_rows.append(
                {
                    "venue": venue,
                    "event_type": event_type,
                    "event_count": max(positive_counts) if positive_counts else 0,
                    "mean_closure": _mean(closure_vals),
                    "mean_persistence": _mean(persistence_vals),
                    "mean_activity": _mean(activity_vals),
                    "best_predictor": best_model or "",
                    "best_predictor_rho": best_rho if best_model is not None else 0.0,
                }
            )
            predictor_rows.append(
                {
                    "venue": venue,
                    "event_type": event_type,
                    "best_predictor": best_model or "",
                    "best_predictor_precision": top_precision_by_model.get(best_model or "", 0.0),
                    "best_predictor_rho": best_rho if best_model is not None else 0.0,
                }
            )

    with (output / "cross_event_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "venue",
                "event_type",
                "event_count",
                "mean_closure",
                "mean_persistence",
                "mean_activity",
                "best_predictor",
                "best_predictor_rho",
            ],
        )
        writer.writeheader()
        writer.writerows(event_rows)

    with (output / "best_predictors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "venue",
                "event_type",
                "best_predictor",
                "best_predictor_precision",
                "best_predictor_rho",
            ],
        )
        writer.writeheader()
        writer.writerows(predictor_rows)

    venues = sorted({row["venue"] for row in event_rows})
    event_types = sorted({row["event_type"] for row in event_rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    width = 0.35
    for idx, venue in enumerate(venues):
        birth_rows = [row for row in event_rows if row["venue"] == venue]
        event_counts = [int(next(row["event_count"] for row in birth_rows if row["event_type"] == event)) for event in event_types]
        predictor_rhos = [float(next(row["best_predictor_rho"] for row in birth_rows if row["event_type"] == event)) for event in event_types]
        positions = [x + (idx - 0.5) * width for x in range(len(event_types))]
        axes[0].bar(positions, event_counts, width=width, label=venue)
        axes[1].bar(positions, predictor_rhos, width=width, label=venue)

    for ax, ylabel, title in [
        (axes[0], "event count", "Cross-event event counts"),
        (axes[1], "best predictor mean Spearman", "Cross-event best predictor strength"),
    ]:
        ax.set_xticks(list(range(len(event_types))))
        ax.set_xticklabels(event_types)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[1].legend()
    fig.tight_layout()
    figure_path = output / "cross_event_summary.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path
