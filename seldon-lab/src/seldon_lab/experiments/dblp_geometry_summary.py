import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt

from .dblp_geometry import _plot_local_case_profiles, run_geometry_experiment


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_cross_venue_profile(path: Path, rows: list[dict[str, str]]) -> None:
    venues = sorted({row["venue"] for row in rows})
    dims = sorted({int(row["dimension"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    width = 0.35
    for idx, venue in enumerate(venues):
        venue_rows = [row for row in rows if row["venue"] == venue]
        positions = [dim + (idx - 0.5) * width for dim in dims]
        reinf = [float(next(row["mean_reinforcement"] for row in venue_rows if int(row["dimension"]) == dim)) for dim in dims]
        curv = [float(next(row["mean_curvature_balance"] for row in venue_rows if int(row["dimension"]) == dim)) for dim in dims]
        axes[0].bar(positions, reinf, width=width, label=venue)
        axes[1].bar(positions, curv, width=width, label=venue)
    for ax, title in [(axes[0], "Mean reinforcement"), (axes[1], "Mean curvature balance")]:
        ax.set_xticks(dims)
        ax.set_xlabel("dimension")
        ax.set_title(title)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_weight_sensitivity(path: Path, rows: list[dict[str, str]]) -> None:
    semantics = sorted({row["weight_semantics"] for row in rows})
    venues = sorted({row["venue"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.22
    for ax, dimension, title in [(axes[0], 1, "Mean edge balance"), (axes[1], 2, "Mean triangle balance")]:
        x_positions = list(range(len(semantics)))
        for idx, venue in enumerate(venues):
            venue_rows = [row for row in rows if row["venue"] == venue and int(row["dimension"]) == dimension]
            values = [float(next(r["mean_curvature_balance"] for r in venue_rows if r["weight_semantics"] == semantic)) for semantic in semantics]
            shifted = [x + (idx - 0.5) * width for x in x_positions]
            ax.bar(shifted, values, width=width, label=venue)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(semantics, rotation=20, ha="right")
        ax.set_title(title)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_validation_summary(path: Path, rows: list[dict[str, str]]) -> None:
    sample_rows = [row for row in rows if row["group"] in {"most_negative", "near_zero", "most_positive"}]
    venues = sorted({row["venue"] for row in sample_rows})
    groups = ["most_negative", "near_zero", "most_positive"]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.22
    x_positions = list(range(len(groups)))
    for idx, venue in enumerate(venues):
        venue_rows = [row for row in sample_rows if row["venue"] == venue and row["weight_semantics"] == "contained-support"]
        values = [float(next(r["mean_edge_betweenness"] for r in venue_rows if r["group"] == group)) for group in groups]
        shifted = [x + (idx - 0.5) * width for x in x_positions]
        ax.bar(shifted, values, width=width, label=venue)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups)
    ax.set_ylabel("mean edge betweenness")
    ax.set_title("Brokerage validation by curvature group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_baseline_vs_ricbal_summary(path: Path, rows: list[dict[str, str]]) -> None:
    sample_rows = [
        row
        for row in rows
        if row["weight_semantics"] == "contained-support"
        and row["dimension"] == "1"
        and row["feature"] == "curvature_balance"
        and row["baseline"] == "edge_betweenness"
    ]
    venues = sorted({row["venue"] for row in sample_rows})
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        venues,
        [float(next(row["spearman"] for row in sample_rows if row["venue"] == venue)) for venue in venues],
        color="C3",
        alpha=0.85,
    )
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Contained-support edge balance vs betweenness")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_geometry_summary(*, configs: list[str | Path], output_dir: str | Path) -> Path:
    outputs = [run_geometry_experiment(config) for config in configs]
    latest_profiles = []
    slice_rows = []
    featured_cases = []
    correlation_rows = []
    validation_rows = []
    region_rows = []
    comparison_rows = []
    for outdir in outputs:
        latest_profiles.extend(_read_csv(Path(outdir) / "latest_dimension_profile.csv"))
        window_rows = _read_csv(Path(outdir) / "window_summary.csv")
        latest_window = window_rows[-1]
        slice_rows.append(latest_window)
        featured_cases.extend(_read_csv(Path(outdir) / "featured_cases.csv"))
        correlation_rows.extend(_read_csv(Path(outdir) / "baseline_correlations.csv"))
        validation_rows.extend(_read_csv(Path(outdir) / "brokerage_validation.csv"))
        region_rows.extend(_read_csv(Path(outdir) / "featured_region_geometry.csv"))
        comparison_rows.extend(_read_csv(Path(outdir) / "group_geometry_comparison.csv"))

    summary_dir = Path(output_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_dir / "cross_venue_latest_profile.csv", latest_profiles)
    _write_csv(summary_dir / "cross_venue_slice_summary.csv", slice_rows)
    _write_csv(summary_dir / "featured_cases.csv", featured_cases)
    _write_csv(summary_dir / "baseline_correlations.csv", correlation_rows)
    _write_csv(summary_dir / "brokerage_validation.csv", validation_rows)
    _write_csv(summary_dir / "weight_sensitivity.csv", latest_profiles)
    _write_csv(summary_dir / "featured_region_geometry.csv", region_rows)
    _write_csv(summary_dir / "group_geometry_comparison.csv", comparison_rows)
    _plot_cross_venue_profile(summary_dir / "cross_venue_profile.png", latest_profiles)
    _plot_local_case_profiles(summary_dir / "featured_cases.png", featured_cases[:4])
    _plot_weight_sensitivity(summary_dir / "weight_sensitivity_plot.png", latest_profiles)
    _plot_validation_summary(summary_dir / "brokerage_validation_plot.png", validation_rows)
    _plot_baseline_vs_ricbal_summary(summary_dir / "baseline_vs_ricbal_plot.png", correlation_rows)
    # Reuse the featured-case records to produce a neighborhood panel from the per-venue runs.
    if outputs:
        # Stitch together one latest complex per venue by reading from the single-venue runs.
        # The featured-case rows already carry venue labels, so we group them below.
        venue_to_cases: dict[str, list[dict[str, str]]] = {}
        for row in featured_cases[:4]:
            venue_to_cases.setdefault(row["venue"], []).append(row)
        all_cases: list[dict[str, str]] = []
        for rows in venue_to_cases.values():
            all_cases.extend(rows)
        # The helper only needs local simplex/weight information already present in the rows.
        # Use the first available single-venue output complex file context implicitly by grouping in the source run.
        # This figure is regenerated directly in the single-venue outputs as well.
        # Here we simply provide the combined artifact for the paper-facing output folder.
        from she_geofield.dblp.experiments import load_config
        from she_geofield.dblp.filters import filter_records
        from she_geofield.dblp.lift import build_window_complex
        from she_geofield.dblp.parse_xml import load_csv_records
        from she_geofield.dblp.windows import build_rolling_windows

        combined_cases: list[dict[str, str]] = []
        complexes_by_venue = {}
        for config_path in configs:
            config = load_config(Path(config_path).resolve())
            base = Path(config_path).resolve().parent.parent if Path(config_path).resolve().parent.name == "configs" else Path(config_path).resolve().parent
            records = load_csv_records((base / str(config["input_file"])).resolve())
            filtered = filter_records(
                records,
                start_year=int(config.get("start_year")) if "start_year" in config else None,
                end_year=int(config.get("end_year")) if "end_year" in config else None,
                min_team_size=int(config.get("min_team_size")) if "min_team_size" in config else None,
                max_team_size=int(config.get("max_team_size")) if "max_team_size" in config else None,
                publication_types=config.get("publication_types") if isinstance(config.get("publication_types"), list) else None,
            )
            windows = build_rolling_windows(
                filtered,
                width=int(config.get("window_width", 3)),
                stride=int(config.get("window_stride", 1)),
                start_year=int(config.get("start_year")) if "start_year" in config else None,
                end_year=int(config.get("end_year")) if "end_year" in config else None,
            )
            complexes_by_venue[str(config.get("venue", "DBLP"))] = build_window_complex(
                windows[-1],
                weight_mode=str(config.get("weight_mode", "contained")),
                max_simplex_size=int(config.get("max_simplex_size", 3)),
            )
        # Build one combined figure by copying venue-specific cases into a temporary list and plotting on a 2x2 grid.
        # We dispatch through the same helper by venue.
        # This is intentionally redundant but keeps the paper artifact self-contained.
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from .dblp_geometry import _node_positions_for_case

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        flat_axes = list(axes.flatten())
        for ax, row in zip(flat_axes, featured_cases[:4]):
            complex_ = complexes_by_venue[row["venue"]]
            simplex = tuple(part for part in str(row.get("simplex_key", row["simplex"])).split(";") if part)
            simplex_set = set(simplex)
            adjacent_triangles = [triangle for triangle in complex_.triangles if simplex_set.issubset(triangle)]
            extras = sorted({node for triangle in adjacent_triangles for node in triangle if node not in simplex_set})
            positions = _node_positions_for_case(simplex, extras)
            for triangle in adjacent_triangles:
                coords = [positions[node] for node in triangle]
                ax.add_patch(Polygon(coords, closed=True, facecolor="C0", alpha=0.16, edgecolor="none"))
            if len(simplex) == 2:
                neighborhood_edges = {
                    tuple(sorted((simplex[0], simplex[1]))),
                    *{tuple(sorted((simplex[0], node))) for node in extras},
                    *{tuple(sorted((simplex[1], node))) for node in extras},
                }
            else:
                neighborhood_edges = {
                    tuple(sorted(edge))
                    for triangle in adjacent_triangles or [simplex]
                    for edge in ((triangle[0], triangle[1]), (triangle[0], triangle[2]), (triangle[1], triangle[2]))
                }
            for edge in sorted(neighborhood_edges):
                x0, y0 = positions[edge[0]]
                x1, y1 = positions[edge[1]]
                weight = complex_.edge_weights.get(edge, 1.0)
                is_core = set(edge) == simplex_set if len(simplex) == 2 else edge[0] in simplex_set and edge[1] in simplex_set
                ax.plot([x0, x1], [y0, y1], color="crimson" if is_core else "0.45", linewidth=2.8 if is_core else 1.0 + 0.35 * min(weight, 4.0), alpha=0.95 if is_core else 0.8, zorder=2)
            if len(simplex) == 3:
                coords = [positions[node] for node in simplex]
                ax.add_patch(Polygon(coords, closed=True, facecolor="crimson", alpha=0.18, edgecolor="crimson", linewidth=2.0))
            for node, (x, y) in positions.items():
                weight = complex_.vertex_weights.get(node, 1.0)
                is_core = node in simplex_set
                ax.scatter([x], [y], s=120 + 18 * min(weight, 8.0), color="crimson" if is_core else "white", edgecolors="black" if is_core else "0.35", linewidths=1.2, zorder=3)
                ax.text(x, y - 0.16, node, ha="center", va="top", fontsize=7)
            ax.set_title(f"{row['venue']} {row['kind']}", fontsize=10)
            ax.text(0.02, 0.98, f"simplex: {row['simplex']}\\nRic={float(row['curvature_balance']):.2f}", transform=ax.transAxes, ha='left', va='top', fontsize=8.5, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"})
            ax.set_xlim(-2.1, 2.1)
            ax.set_ylim(-1.5, 1.9)
            ax.axis("off")
        for ax in flat_axes[len(featured_cases[:4]) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(summary_dir / "featured_neighborhoods.png", dpi=180)
        plt.close(fig)
    return summary_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Seldon DBLP static-geometry cross-venue summary.")
    parser.add_argument("--config", action="append", required=True, help="Geometry config path. Repeat for multiple venues.")
    parser.add_argument("--output-dir", required=True, help="Cross-venue output directory.")
    args = parser.parse_args()
    outdir = run_geometry_summary(configs=args.config, output_dir=args.output_dir)
    print(f"Seldon geometry cross-venue summary complete. Outputs in {outdir}")


if __name__ == "__main__":
    main()
