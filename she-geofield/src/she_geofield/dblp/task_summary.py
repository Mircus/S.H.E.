import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _aggregate_metrics(csv_path: Path) -> dict[str, tuple[float, float]]:
    rows = list(csv.DictReader(csv_path.open()))
    agg: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for row in rows:
        acc = agg[row["model"]]
        acc[0] += float(row["top_k_precision"])
        acc[1] += float(row["spearman_future_support"])
        acc[2] += 1.0
    return {
        model: (values[0] / values[2], values[1] / values[2])
        for model, values in agg.items()
        if values[2] > 0.0
    }


def build_cross_task_summary(
    task_to_venue_csv: dict[str, dict[str, str | Path]],
    *,
    output_dir: str | Path,
    focus_models: list[str] | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    focus = focus_models or [
        "persistence_only",
        "btc_score",
        "evolving_geometry",
        "frozen_diffusion",
        "author_degree",
    ]

    rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for task, venue_map in task_to_venue_csv.items():
        summary[task] = {}
        for venue, csv_path in venue_map.items():
            summary[task][venue] = _aggregate_metrics(Path(csv_path))
            for model, (precision, spearman) in summary[task][venue].items():
                rows.append(
                    {
                        "task": task,
                        "venue": venue,
                        "model": model,
                        "mean_top_k_precision": precision,
                        "mean_spearman": spearman,
                    }
                )

    with (output / "cross_task_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "venue", "model", "mean_top_k_precision", "mean_spearman"],
        )
        writer.writeheader()
        writer.writerows(rows)

    tasks = list(summary.keys())
    venues = sorted({venue for task_data in summary.values() for venue in task_data})
    fig, axes = plt.subplots(len(venues), 2, figsize=(12, 4 * len(venues)), squeeze=False)
    for row_idx, venue in enumerate(venues):
        for col_idx, (metric_idx, ylabel, title) in enumerate(
            [
                (0, "mean top-k precision", f"{venue}: ranking precision by task"),
                (1, "mean Spearman", f"{venue}: graded correlation by task"),
            ]
        ):
            ax = axes[row_idx][col_idx]
            width = 0.14
            xs = range(len(tasks))
            for offset, model in enumerate(focus):
                values = [
                    summary.get(task, {}).get(venue, {}).get(model, (0.0, 0.0))[metric_idx]
                    for task in tasks
                ]
                positions = [x + (offset - (len(focus) - 1) / 2) * width for x in xs]
                ax.bar(positions, values, width=width, label=model)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(tasks, rotation=20)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
    axes[0][1].legend(fontsize=8)
    fig.tight_layout()
    figure_path = output / "cross_task_summary.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path
