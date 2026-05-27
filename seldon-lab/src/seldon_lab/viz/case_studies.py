import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt


def write_case_study_csv(rows: list[dict[str, object]], *, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return output
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output


def plot_case_study_panel(rows: list[dict[str, object]], *, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return output
    sample = rows[: min(5, len(rows))]
    labels = [str(row["aggregation_id"]).split(":", 1)[-1] for row in sample]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ("closure", axes[0]),
        ("persistence", axes[1]),
        ("activity", axes[2]),
    ]
    positions = list(range(len(sample)))
    width = 0.35
    for metric, ax in metrics:
        before = [float(row.get(f"before_{metric}", 0.0)) for row in sample]
        after = [float(row.get(f"after_{metric}", 0.0)) for row in sample]
        ax.bar([p - width / 2 for p in positions], before, width=width, label="before")
        ax.bar([p + width / 2 for p in positions], after, width=width, label="after")
        ax.set_title(metric)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
