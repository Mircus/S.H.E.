import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt

from .filters import filter_records
from .parse_xml import load_csv_records, load_dblp_records
from .temporal_flow import (
    build_temporal_complex_sequence,
    evaluate_aggregation_events,
    evaluate_models,
)
from .windows import build_rolling_windows, summarize_window


def _parse_scalar(value: str):
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_config(config_path: str | Path) -> dict[str, object]:
    path = Path(config_path)
    config: dict[str, object] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, value = stripped.split(":", 1)
        config[key.strip()] = _parse_scalar(value)
    config["_config_path"] = path
    return config


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_model_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    models = sorted({row["model"] for row in rows})
    averages = []
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        avg = sum(float(row["top_k_precision"]) for row in model_rows) / max(len(model_rows), 1)
        averages.append(avg)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(models, averages, color="C1", alpha=0.85)
    ax.set_ylabel("mean top-k precision")
    ax.set_title("DBLP prediction comparison")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_window_trajectory(path: Path, rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    model_order = []
    for preferred in ["btc_score", "graph_bridge", "persistence_only", "simplex_support", "triangle_support", "frozen_diffusion", "evolving_geometry", "mixed_geometry", "dyad_projection"]:
        if any(row["model"] == preferred for row in rows):
            model_order.append(preferred)
    if not model_order:
        model_order = sorted({row["model"] for row in rows})[:3]
    for model in model_order[:4]:
        model_rows = [row for row in rows if row["model"] == model]
        ax.plot(
            [row["window"] for row in model_rows],
            [float(row["top_k_precision"]) for row in model_rows],
            marker="o",
            label=model,
        )
    ax.set_ylabel("top-k precision")
    ax.set_title("Temporal predictive performance")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_case_study(path: Path, rows: list[dict[str, object]]) -> None:
    sample = rows[: min(8, len(rows))]
    if not sample:
        return
    labels = [
        str(row.get("simplex", row.get("aggregation_id", row.get("members", "item"))))
        for row in sample
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        labels,
        [float(row["score"]) for row in sample],
        color="C2",
        alpha=0.85,
    )
    ax.set_ylabel("evolving score")
    ax.set_title("Top collaboration carriers (case study)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_output_bundle(
    output_dir: Path,
    *,
    prefix: str,
    comparison_rows: list[dict[str, object]],
    top_simplex_rows: list[dict[str, object]],
) -> None:
    stem = f"{prefix}_" if prefix else ""
    _write_csv(output_dir / f"{stem}model_comparison.csv", comparison_rows)
    _write_csv(output_dir / f"{stem}top_collaboration_simplices.csv", top_simplex_rows)
    _plot_model_comparison(output_dir / f"{stem}predictive_comparison.png", comparison_rows)
    _plot_window_trajectory(output_dir / f"{stem}temporal_performance.png", comparison_rows)
    _plot_case_study(output_dir / f"{stem}case_study.png", top_simplex_rows)


def run_experiment(config_path: str | Path) -> Path:
    config = load_config(config_path)
    config_path = Path(config["_config_path"])
    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent

    input_file = _resolve_path(base_dir, str(config["input_file"]))
    output_dir = _resolve_path(base_dir, str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_file.suffix == ".csv":
        records = load_csv_records(input_file)
    else:
        records = load_dblp_records(
            input_file,
            publication_types=set(config.get("publication_types", ["article", "inproceedings"])),
        )
    filtered = filter_records(
        records,
        start_year=int(config.get("start_year")) if "start_year" in config else None,
        end_year=int(config.get("end_year")) if "end_year" in config else None,
        min_team_size=int(config.get("min_team_size")) if "min_team_size" in config else None,
        max_team_size=int(config.get("max_team_size")) if "max_team_size" in config else None,
        venue_list=config.get("venue_list") if isinstance(config.get("venue_list"), list) else None,
        venue_regex=str(config.get("venue_regex")) if "venue_regex" in config else None,
        publication_types=(
            config.get("publication_types") if isinstance(config.get("publication_types"), list) else None
        ),
    )

    windows = build_rolling_windows(
        filtered,
        width=int(config.get("window_width", 3)),
        stride=int(config.get("window_stride", 1)),
        start_year=int(config.get("start_year")) if "start_year" in config else None,
        end_year=int(config.get("end_year")) if "end_year" in config else None,
    )
    complexes = build_temporal_complex_sequence(
        windows,
        weight_mode=str(config.get("weight_mode", "contained")),
        max_simplex_size=int(config.get("max_simplex_size", 3)),
    )

    window_rows = []
    for window, complex_ in zip(windows, complexes):
        summary = summarize_window(window, complex_)
        window_rows.append(
            {
                "window": summary.window_label,
                "paper_count": summary.paper_count,
                "author_count": summary.author_count,
                "vertex_count": summary.simplex_count_by_dim[0],
                "edge_count": summary.simplex_count_by_dim[1],
                "triangle_count": summary.simplex_count_by_dim[2],
                "edge_density": summary.density_stats["edge_density"],
                "triangle_density": summary.density_stats["triangle_density"],
            }
        )

    experiment_family = str(config.get("experiment_family", "score"))
    if experiment_family == "aggregation":
        comparison_rows, top_simplex_rows = evaluate_aggregation_events(
            complexes,
            internal_steps=int(config.get("internal_steps", 2)),
            dt=float(config.get("dt", 0.5)),
            eta=float(config.get("eta", 0.1)),
            lam=float(config.get("lam", 0.3)),
            mu=float(config.get("mu", 0.2)),
            field_mode=str(config.get("field_mode", "support_count")),
            horizon=int(config.get("prediction_horizon", 1)),
            top_k=int(config.get("top_k", 5)),
            event_type=str(config.get("event_type", "birth")),
            unit_types=tuple(config.get("unit_types", ["edges", "triangles", "neighborhoods"])),
        )
    else:
        comparison_rows, top_simplex_rows = evaluate_models(
            complexes,
            internal_steps=int(config.get("internal_steps", 2)),
            dt=float(config.get("dt", 0.5)),
            eta=float(config.get("eta", 0.1)),
            lam=float(config.get("lam", 0.3)),
            mu=float(config.get("mu", 0.2)),
            field_mode=str(config.get("field_mode", "support_count")),
            field_location=str(config.get("field_location", "edges")),
            horizon=int(config.get("prediction_horizon", 1)),
            top_k=int(config.get("top_k", 5)),
            target_mode=str(config.get("target_mode", "future_support")),
            candidate_regime=str(config.get("candidate_regime", "all")),
            bridge_quantile=float(config.get("bridge_quantile", 0.8)),
            max_persistence=float(config.get("max_persistence", 0.5)),
            score_location=str(config.get("score_location", config.get("field_location", "edges"))),
        )

    _write_csv(output_dir / "window_summary.csv", window_rows)
    _write_output_bundle(
        output_dir,
        prefix="",
        comparison_rows=comparison_rows,
        top_simplex_rows=top_simplex_rows,
    )

    if experiment_family != "aggregation" and ("secondary_target_mode" in config or "secondary_candidate_regime" in config):
        secondary_rows, secondary_top_rows = evaluate_models(
            complexes,
            internal_steps=int(config.get("internal_steps", 2)),
            dt=float(config.get("dt", 0.5)),
            eta=float(config.get("eta", 0.1)),
            lam=float(config.get("lam", 0.3)),
            mu=float(config.get("mu", 0.2)),
            field_mode=str(config.get("field_mode", "support_count")),
            field_location=str(config.get("field_location", "edges")),
            horizon=int(config.get("prediction_horizon", 1)),
            top_k=int(config.get("secondary_top_k", config.get("top_k", 5))),
            target_mode=str(config.get("secondary_target_mode", config.get("target_mode", "future_support"))),
            candidate_regime=str(config.get("secondary_candidate_regime", config.get("candidate_regime", "all"))),
            bridge_quantile=float(config.get("secondary_bridge_quantile", config.get("bridge_quantile", 0.8))),
            max_persistence=float(config.get("secondary_max_persistence", config.get("max_persistence", 0.5))),
            score_location=str(config.get("secondary_score_location", config.get("score_location", config.get("field_location", "edges")))),
        )
        _write_output_bundle(
            output_dir,
            prefix=str(config.get("secondary_output_prefix", "secondary")),
            comparison_rows=secondary_rows,
            top_simplex_rows=secondary_top_rows,
        )

    if experiment_family != "aggregation" and ("tertiary_target_mode" in config or "tertiary_candidate_regime" in config):
        tertiary_rows, tertiary_top_rows = evaluate_models(
            complexes,
            internal_steps=int(config.get("internal_steps", 2)),
            dt=float(config.get("dt", 0.5)),
            eta=float(config.get("eta", 0.1)),
            lam=float(config.get("lam", 0.3)),
            mu=float(config.get("mu", 0.2)),
            field_mode=str(config.get("tertiary_field_mode", config.get("field_mode", "support_count"))),
            field_location=str(config.get("tertiary_field_location", config.get("field_location", "edges"))),
            horizon=int(config.get("prediction_horizon", 1)),
            top_k=int(config.get("tertiary_top_k", config.get("top_k", 5))),
            target_mode=str(config.get("tertiary_target_mode", config.get("target_mode", "future_support"))),
            candidate_regime=str(config.get("tertiary_candidate_regime", config.get("candidate_regime", "all"))),
            bridge_quantile=float(config.get("tertiary_bridge_quantile", config.get("bridge_quantile", 0.8))),
            max_persistence=float(config.get("tertiary_max_persistence", config.get("max_persistence", 0.5))),
            score_location=str(config.get("tertiary_score_location", config.get("tertiary_field_location", config.get("score_location", config.get("field_location", "edges"))))),
        )
        _write_output_bundle(
            output_dir,
            prefix=str(config.get("tertiary_output_prefix", "tertiary")),
            comparison_rows=tertiary_rows,
            top_simplex_rows=tertiary_top_rows,
        )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DBLP geometry-field experiment.")
    parser.add_argument("--config", required=True, help="Path to the experiment config file.")
    args = parser.parse_args()
    outdir = run_experiment(args.config)
    print(f"DBLP experiment complete. Outputs in {outdir}")


if __name__ == "__main__":
    main()
