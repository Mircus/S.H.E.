import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt


def plot_aggregation_trajectories(
    trajectories: dict[str, list[dict[str, object]]],
    *,
    output_dir: str | Path,
) -> list[Path]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for aggregation_id, rows in trajectories.items():
        if not rows:
            continue
        windows = [str(row["window"]) for row in rows]
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        metric_axes = [
            ("closure", axes[0][0]),
            ("persistence", axes[0][1]),
            ("activity", axes[1][0]),
            ("boundary_role", axes[1][1]),
        ]
        for metric, ax in metric_axes:
            ax.plot(windows, [float(row.get(metric, 0.0)) for row in rows], marker="o")
            ax.set_title(metric.replace("_", " "))
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle(aggregation_id)
        fig.tight_layout()
        path = outdir / f"{aggregation_id.replace(':', '_').replace('/', '_')}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path)

    return created
