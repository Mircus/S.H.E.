"""
Disambiguation experiments for the Seldon/SHE paper.

Runs three analyses:
  E3  — R_bal vs betweenness disambiguation (partial correlation, disagreement table, top disagreement edges)
  E1/A3 — Mann-Whitney U test on brokerage groups
  E7/A6 — Generates window_summary.csv and simplex_observables.csv for each venue

Outputs:
  artifacts/paper1/validation/disambiguation_results.csv
  artifacts/paper1/validation/statistical_tests.csv
  artifacts/paper1/validation/ricbal_vs_betweenness_scatter.png
  out/dblp_sdm_geometry/window_summary.csv
  out/dblp_sdm_geometry/simplex_observables.csv
  out/dblp_wsdm_geometry/window_summary.csv
  out/dblp_wsdm_geometry/simplex_observables.csv
"""

import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add the experiment source to the path so we can reuse helpers
sys.path.insert(0, str(REPO_ROOT / "src"))

from seldon_lab.experiments.dblp_geometry import (
    run_geometry_experiment,
    _rankdata,
    _pearson,
    _spearman,
)

ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "paper1" / "validation"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SDM_CONFIG = REPO_ROOT / "configs" / "dblp_sdm_geometry.yaml"
WSDM_CONFIG = REPO_ROOT / "configs" / "dblp_wsdm_geometry.yaml"

EPS = 1e-9


# ---------------------------------------------------------------------------
# Partial Spearman correlation: corr(X, Y | Z)
# Uses the formula: r_{XY.Z} = (r_XY - r_XZ * r_YZ) / sqrt((1-r_XZ^2)(1-r_YZ^2))
# ---------------------------------------------------------------------------
def partial_spearman(xs, ys, zs):
    """Spearman partial correlation of xs and ys controlling for zs."""
    rxy = _spearman(xs, ys)
    rxz = _spearman(xs, zs)
    ryz = _spearman(ys, zs)
    denom = math.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), EPS))
    return (rxy - rxz * ryz) / denom


# ---------------------------------------------------------------------------
# Mann-Whitney U and rank-biserial r
# ---------------------------------------------------------------------------
def mann_whitney_u(group1, group2):
    """
    Compute Mann-Whitney U statistic and two-sided p-value using a normal
    approximation (valid for n1, n2 > ~20).
    Returns (U, p_value, rank_biserial_r).
    """
    n1 = len(group1)
    n2 = len(group2)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")

    combined = [(val, 0) for val in group1] + [(val, 1) for val in group2]
    combined.sort(key=lambda item: item[0])

    # Assign ranks with tie-averaging
    ranks = [0.0] * len(combined)
    idx = 0
    while idx < len(combined):
        end = idx + 1
        while end < len(combined) and combined[end][0] == combined[idx][0]:
            end += 1
        rank = (idx + end - 1) / 2.0 + 1.0
        for pos in range(idx, end):
            ranks[pos] = rank
        idx = end

    rank_sum_1 = sum(ranks[i] for i, (_, g) in enumerate(combined) if g == 0)
    U1 = rank_sum_1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)

    # Normal approximation
    mu_U = n1 * n2 / 2.0
    # Tie correction
    n = n1 + n2
    tie_correction = 0.0
    idx = 0
    all_vals = [v for v, _ in combined]
    while idx < len(all_vals):
        end = idx + 1
        while end < len(all_vals) and all_vals[end] == all_vals[idx]:
            end += 1
        t = end - idx
        tie_correction += t ** 3 - t
        idx = end
    sigma_U = math.sqrt((n1 * n2 / 12.0) * (n + 1 - tie_correction / (n * (n - 1) + EPS)))
    if sigma_U < EPS:
        p_value = 1.0
    else:
        z = (U - mu_U) / sigma_U
        # Two-sided p-value using error function
        p_value = math.erfc(abs(z) / math.sqrt(2))

    # Rank-biserial r = (U1 - U2) / (n1 * n2)
    r_rb = (U1 - U2) / (n1 * n2)

    return U, p_value, r_rb


# ---------------------------------------------------------------------------
# Disagreement table helpers
# ---------------------------------------------------------------------------
def tercile_labels(values):
    """Return list of 'low'/'mid'/'high' tercile labels for each value."""
    n = len(values)
    sorted_vals = sorted(values)
    t1 = sorted_vals[n // 3]
    t2 = sorted_vals[2 * n // 3]
    labels = []
    for v in values:
        if v <= t1:
            labels.append("low")
        elif v <= t2:
            labels.append("mid")
        else:
            labels.append("high")
    return labels


# ---------------------------------------------------------------------------
# E7/A6: Run geometry experiment for both venues
# ---------------------------------------------------------------------------
def run_geometry_for_venues():
    print("=" * 60)
    print("E7/A6: Running geometry experiments for SDM and WSDM")
    print("=" * 60)

    print(f"\nRunning SDM geometry experiment (config: {SDM_CONFIG}) ...")
    sdm_outdir = run_geometry_experiment(SDM_CONFIG)
    print(f"  SDM outputs in: {sdm_outdir}")

    print(f"\nRunning WSDM geometry experiment (config: {WSDM_CONFIG}) ...")
    wsdm_outdir = run_geometry_experiment(WSDM_CONFIG)
    print(f"  WSDM outputs in: {wsdm_outdir}")

    return sdm_outdir, wsdm_outdir


# ---------------------------------------------------------------------------
# Load simplex_observables rows (edge-level) from a geometry output dir
# ---------------------------------------------------------------------------
def load_edge_rows(outdir: Path, venue: str):
    """Load edge-level rows from simplex_extended_geometry.csv (has graph features)."""
    # simplex_extended_geometry.csv contains the latest-window rows WITH extended metrics
    path = outdir / "simplex_extended_geometry.csv"
    if not path.exists():
        # Fall back to simplex_observables.csv filtered to latest window
        path = outdir / "simplex_observables.csv"
        print(f"  Warning: using simplex_observables.csv for {venue}")

    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["dimension"]) == 1:
                rows.append(row)
    print(f"  Loaded {len(rows)} edge rows for {venue} from {path.name}")
    return rows


# ---------------------------------------------------------------------------
# E3: R_bal vs betweenness disambiguation
# ---------------------------------------------------------------------------
def run_e3_disambiguation(venue: str, edge_rows: list):
    print(f"\n  {venue}: {len(edge_rows)} edges")

    curv = [float(r["curvature_balance"]) for r in edge_rows]
    btw = [float(r.get("edge_betweenness", 0.0)) for r in edge_rows]
    strength = [float(r.get("endpoint_mean_strength", 0.0)) for r in edge_rows]

    # 1. Raw Spearman correlation
    rho_raw = _spearman(curv, btw)
    print(f"    Raw Spearman(curv, btw) = {rho_raw:.4f}")

    # 2. Partial Spearman after controlling for endpoint_mean_strength
    rho_partial = partial_spearman(curv, btw, strength)
    print(f"    Partial Spearman(curv, btw | strength) = {rho_partial:.4f}")

    # 3. Tercile disagreement table
    curv_labels = tercile_labels(curv)
    btw_labels = tercile_labels(btw)
    aa = sum(1 for c, b in zip(curv_labels, btw_labels) if c == "low" and b == "low")
    ab = sum(1 for c, b in zip(curv_labels, btw_labels) if c == "low" and b != "low")
    ba = sum(1 for c, b in zip(curv_labels, btw_labels) if c != "low" and b == "low")
    bb = sum(1 for c, b in zip(curv_labels, btw_labels) if c != "low" and b != "low")

    # More precise: bottom-tercile = "low", otherwise "not-low"
    # (a) both low
    cell_aa = sum(1 for c, b in zip(curv_labels, btw_labels) if c == "low" and b == "low")
    # (b) curv low but btw NOT low (R_bal-unique brokerage signal)
    cell_ab = sum(1 for c, b in zip(curv_labels, btw_labels) if c == "low" and b != "low")
    # (c) btw low but curv NOT low
    cell_ba = sum(1 for c, b in zip(curv_labels, btw_labels) if c != "low" and b == "low")
    # (d) both NOT low
    cell_bb = sum(1 for c, b in zip(curv_labels, btw_labels) if c != "low" and b != "low")

    print(f"    Disagreement table (bottom tercile):")
    print(f"      (a) both low      = {cell_aa}")
    print(f"      (b) curv-low only = {cell_ab}  [R_bal-unique brokerage signal]")
    print(f"      (c) btw-low only  = {cell_ba}")
    print(f"      (d) both not-low  = {cell_bb}")

    # 4. Top-10 disagreement edges by |rank(curv) - rank(btw)|
    rank_curv = _rankdata(curv)
    rank_btw = _rankdata(btw)
    rank_diffs = [abs(rc - rb) for rc, rb in zip(rank_curv, rank_btw)]
    top10_indices = sorted(range(len(rank_diffs)), key=lambda i: rank_diffs[i], reverse=True)[:10]
    top10_edges = []
    for i in top10_indices:
        row = edge_rows[i]
        top10_edges.append({
            "venue": venue,
            "simplex": row.get("simplex", ""),
            "curvature_balance": curv[i],
            "edge_betweenness": btw[i],
            "endpoint_mean_strength": strength[i],
            "rank_curv": rank_curv[i],
            "rank_btw": rank_btw[i],
            "rank_diff": rank_diffs[i],
        })

    return {
        "venue": venue,
        "n_edges": len(edge_rows),
        "spearman_raw": rho_raw,
        "partial_spearman_controlling_strength": rho_partial,
        "disagree_both_low": cell_aa,
        "disagree_curv_low_only": cell_ab,
        "disagree_btw_low_only": cell_ba,
        "disagree_both_not_low": cell_bb,
        "top10_edges": top10_edges,
    }


# ---------------------------------------------------------------------------
# E1/A3: Mann-Whitney U on brokerage groups
# ---------------------------------------------------------------------------
def run_e1_mann_whitney(venue: str, edge_rows: list):
    print(f"\n  {venue}: Mann-Whitney U test on brokerage groups")

    curv = [float(r["curvature_balance"]) for r in edge_rows]
    btw = [float(r.get("edge_betweenness", 0.0)) for r in edge_rows]

    sorted_rows_with_vals = sorted(zip(curv, btw, edge_rows), key=lambda x: x[0])
    k = max(5, len(sorted_rows_with_vals) // 10)

    most_neg_btw = [b for _, b, _ in sorted_rows_with_vals[:k]]
    near_zero_btw = sorted(
        [(abs(c), b) for c, b, _ in sorted_rows_with_vals], key=lambda x: x[0]
    )[:k]
    near_zero_btw = [b for _, b in near_zero_btw]

    mean_neg = mean(most_neg_btw) if most_neg_btw else float("nan")
    mean_zero = mean(near_zero_btw) if near_zero_btw else float("nan")
    print(f"    most_negative group: n={k}, mean_betweenness={mean_neg:.2f}")
    print(f"    near_zero group:     n={k}, mean_betweenness={mean_zero:.2f}")

    U, p_value, r_rb = mann_whitney_u(most_neg_btw, near_zero_btw)
    print(f"    Mann-Whitney U={U:.1f}, p={p_value:.4e}, rank-biserial r={r_rb:.4f}")

    return {
        "venue": venue,
        "k": k,
        "mean_betweenness_most_negative": mean_neg,
        "mean_betweenness_near_zero": mean_zero,
        "mann_whitney_U": U,
        "p_value": p_value,
        "rank_biserial_r": r_rb,
    }


# ---------------------------------------------------------------------------
# Scatter plot: R_bal vs betweenness for both venues
# ---------------------------------------------------------------------------
def plot_ricbal_vs_betweenness(sdm_rows, wsdm_rows, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, rows, title in [
        (axes[0], sdm_rows, "SDM: R_bal vs Edge Betweenness"),
        (axes[1], wsdm_rows, "WSDM: R_bal vs Edge Betweenness"),
    ]:
        curv = [float(r["curvature_balance"]) for r in rows]
        btw = [float(r.get("edge_betweenness", 0.0)) for r in rows]
        ax.scatter(btw, curv, alpha=0.45, s=18, color="C3", edgecolors="none")
        ax.axhline(0.0, color="0.7", linewidth=1.0)
        ax.set_xlabel("Edge Betweenness")
        ax.set_ylabel("Curvature Balance (R_bal)")
        ax.set_title(title)
        rho = _spearman(curv, btw)
        ax.text(
            0.97, 0.97,
            f"ρ = {rho:.3f}",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"\nScatter plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# CSV write helpers
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: list):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("Seldon-Lab Disambiguation Experiments")
    print("=" * 60)

    # E7/A6: Run geometry for both venues to get simplex_observables.csv and window_summary.csv
    sdm_outdir, wsdm_outdir = run_geometry_for_venues()

    # Load edge-level data from geometry outputs
    print("\n" + "=" * 60)
    print("Loading edge-level data")
    print("=" * 60)
    sdm_edges = load_edge_rows(sdm_outdir, "SDM")
    wsdm_edges = load_edge_rows(wsdm_outdir, "WSDM")

    # E3: Disambiguation analysis
    print("\n" + "=" * 60)
    print("E3: R_bal vs betweenness disambiguation")
    print("=" * 60)
    sdm_e3 = run_e3_disambiguation("SDM", sdm_edges)
    wsdm_e3 = run_e3_disambiguation("WSDM", wsdm_edges)

    # E1/A3: Mann-Whitney U
    print("\n" + "=" * 60)
    print("E1/A3: Mann-Whitney U test on brokerage groups")
    print("=" * 60)
    sdm_mw = run_e1_mann_whitney("SDM", sdm_edges)
    wsdm_mw = run_e1_mann_whitney("WSDM", wsdm_edges)

    # Scatter plot
    plot_ricbal_vs_betweenness(
        sdm_edges, wsdm_edges,
        ARTIFACTS_DIR / "ricbal_vs_betweenness_scatter.png",
    )

    # Build disambiguation_results.csv rows
    disambiguation_rows = []
    for result in [sdm_e3, wsdm_e3]:
        venue = result["venue"]
        disambiguation_rows.append({
            "venue": venue,
            "analysis": "partial_correlation",
            "description": "Spearman(curvature_balance, edge_betweenness | endpoint_mean_strength)",
            "n_edges": result["n_edges"],
            "spearman_raw": result["spearman_raw"],
            "partial_spearman": result["partial_spearman_controlling_strength"],
            "value": result["partial_spearman_controlling_strength"],
        })
        disambiguation_rows.append({
            "venue": venue,
            "analysis": "disagreement_table",
            "description": "2x2 bottom-tercile disagreement (a=both_low, b=curv_low_only, c=btw_low_only, d=neither)",
            "n_edges": result["n_edges"],
            "a_both_low": result["disagree_both_low"],
            "b_curv_low_only": result["disagree_curv_low_only"],
            "c_btw_low_only": result["disagree_btw_low_only"],
            "d_neither_low": result["disagree_both_not_low"],
        })
        for edge in result["top10_edges"]:
            disambiguation_rows.append({
                "venue": venue,
                "analysis": "top10_disagreement_edges",
                "description": "Edges with largest |rank(curvature_balance) - rank(edge_betweenness)|",
                "n_edges": result["n_edges"],
                **edge,
            })

    # Build statistical_tests.csv rows
    stat_rows = []
    for mw in [sdm_mw, wsdm_mw]:
        stat_rows.append({
            "venue": mw["venue"],
            "test": "Mann-Whitney U",
            "comparison": "most_negative vs near_zero (edge betweenness)",
            "n_per_group": mw["k"],
            "mean_betweenness_most_negative": mw["mean_betweenness_most_negative"],
            "mean_betweenness_near_zero": mw["mean_betweenness_near_zero"],
            "U_statistic": mw["mann_whitney_U"],
            "p_value": mw["p_value"],
            "rank_biserial_r": mw["rank_biserial_r"],
        })

    # Write outputs
    print("\n" + "=" * 60)
    print("Writing outputs")
    print("=" * 60)
    write_csv(ARTIFACTS_DIR / "disambiguation_results.csv", disambiguation_rows)
    write_csv(ARTIFACTS_DIR / "statistical_tests.csv", stat_rows)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    print("\nE3 — R_bal vs betweenness disambiguation:")
    for result in [sdm_e3, wsdm_e3]:
        v = result["venue"]
        print(f"\n  {v}:")
        print(f"    Raw Spearman(R_bal, betweenness)                       = {result['spearman_raw']:.4f}")
        print(f"    Partial Spearman(R_bal, betweenness | node_strength)   = {result['partial_spearman_controlling_strength']:.4f}")
        print(f"    Disagreement table (bottom tercile):")
        print(f"      (a) both low [aligned]         = {result['disagree_both_low']}")
        print(f"      (b) R_bal-low, btw-not-low     = {result['disagree_curv_low_only']}  [R_bal-unique signal]")
        print(f"      (c) btw-low, R_bal-not-low     = {result['disagree_btw_low_only']}")
        print(f"      (d) neither low                = {result['disagree_both_not_low']}")

    print("\nE1/A3 — Mann-Whitney U tests:")
    for mw in [sdm_mw, wsdm_mw]:
        v = mw["venue"]
        print(f"\n  {v}:")
        print(f"    most_negative mean betweenness = {mw['mean_betweenness_most_negative']:.2f}")
        print(f"    near_zero mean betweenness     = {mw['mean_betweenness_near_zero']:.2f}")
        print(f"    Mann-Whitney U = {mw['mann_whitney_U']:.1f}")
        print(f"    p-value        = {mw['p_value']:.4e}")
        print(f"    rank-biserial r = {mw['rank_biserial_r']:.4f}")

    print("\nE7/A6 — Geometry files generated:")
    for outdir, venue in [(sdm_outdir, "SDM"), (wsdm_outdir, "WSDM")]:
        ws = outdir / "window_summary.csv"
        so = outdir / "simplex_observables.csv"
        print(f"  {venue}: window_summary.csv exists={ws.exists()}, simplex_observables.csv exists={so.exists()}")


if __name__ == "__main__":
    main()
