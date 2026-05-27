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
        model: (vals[0] / vals[2], vals[1] / vals[2])
        for model, vals in agg.items()
    }


def build_cross_venue_summary(
    venue_to_csv: dict[str, str | Path],
    *,
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for venue, csv_path in venue_to_csv.items():
        summary[venue] = _aggregate_metrics(Path(csv_path))
        for model, (precision, spearman) in summary[venue].items():
            rows.append(
                {
                    "venue": venue,
                    "model": model,
                    "mean_top_k_precision": precision,
                    "mean_spearman_future_support": spearman,
                }
            )

    with (output / "cross_venue_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "venue",
                "model",
                "mean_top_k_precision",
                "mean_spearman_future_support",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    venues = list(summary.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    focus_models = ["persistence_only", "evolving_geometry", "simplex_support", "author_degree"]
    for ax, metric_idx, ylabel, title in [
        (axes[0], 0, "mean top-k precision", "Cross-venue ranking precision"),
        (axes[1], 1, "mean Spearman", "Cross-venue future-support correlation"),
    ]:
        width = 0.18
        xs = range(len(venues))
        for offset, model in enumerate(focus_models):
            values = [summary[venue].get(model, (0.0, 0.0))[metric_idx] for venue in venues]
            positions = [x + (offset - 1.5) * width for x in xs]
            ax.bar(positions, values, width=width, label=model)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(venues)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    figure_path = output / "cross_venue_summary.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path

