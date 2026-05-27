import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from she_geofield.dblp.experiments import load_config
from she_geofield.dblp.filters import filter_records
from she_geofield.dblp.lift import build_window_complex
from she_geofield.dblp.parse_xml import load_csv_records
from she_geofield.dblp.windows import build_rolling_windows, summarize_window


EPS = 1e-9
SUPPORTED_WEIGHT_SEMANTICS = {"contained-support", "exact-facet", "recency-decayed"}


def _encode_simplex(simplex: tuple[str, ...]) -> str:
    return ";".join(simplex)


def _decode_simplex(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split(";") if part)


def _all_simplices(complex_) -> list[tuple[str, ...]]:
    return [(vertex,) for vertex in complex_.vertices] + list(complex_.edges) + list(complex_.triangles)


def _simplex_dimension(simplex: tuple[str, ...]) -> int:
    return len(simplex) - 1


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


def _simplex_weight(complex_, simplex: tuple[str, ...]) -> float:
    if len(simplex) == 1:
        return float(complex_.vertex_weights.get(simplex[0], 0.0))
    if len(simplex) == 2:
        return float(complex_.edge_weights.get(simplex, 0.0))
    if len(simplex) == 3:
        return float(complex_.triangle_weights.get(simplex, 0.0))
    return float(complex_.simplex_data.get(simplex, {}).get("support_count", 0.0))


def _iter_simplices(authors: tuple[str, ...], max_simplex_size: int):
    upper = min(len(authors), max_simplex_size)
    for size in range(1, upper + 1):
        yield from combinations(authors, size)


def _sorted_simplex(authors: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(authors))


def compute_weight_maps(window, *, weight_semantics: str, max_simplex_size: int, decay_lambda: float) -> dict[int, dict[tuple[str, ...] | str, float]]:
    if weight_semantics not in SUPPORTED_WEIGHT_SEMANTICS:
        raise ValueError(f"unsupported weight semantics: {weight_semantics}")

    simplex_weights: defaultdict[tuple[str, ...], float] = defaultdict(float)
    for record in window.records:
        authors = _sorted_simplex(record.authors)
        if weight_semantics == "contained-support":
            delta = 1.0
            simplices = _iter_simplices(authors, max_simplex_size=max_simplex_size)
        elif weight_semantics == "exact-facet":
            delta = 1.0
            simplices = [authors] if len(authors) <= max_simplex_size else []
        else:
            delta = math.exp(-decay_lambda * (window.end_year - record.year))
            simplices = _iter_simplices(authors, max_simplex_size=max_simplex_size)
        for simplex in simplices:
            simplex_weights[simplex] += delta

    vertex_weights: dict[str, float] = {}
    edge_weights: dict[tuple[str, str], float] = {}
    triangle_weights: dict[tuple[str, str, str], float] = {}
    for simplex, value in simplex_weights.items():
        if len(simplex) == 1:
            vertex_weights[simplex[0]] = value
        elif len(simplex) == 2:
            edge_weights[simplex] = value
        elif len(simplex) == 3:
            triangle_weights[simplex] = value
    return {0: vertex_weights, 1: edge_weights, 2: triangle_weights}


def _weight_from_maps(weight_maps: dict[int, dict], simplex: tuple[str, ...]) -> float:
    if len(simplex) == 1:
        return float(weight_maps[0].get(simplex[0], 0.0))
    return float(weight_maps[len(simplex) - 1].get(simplex, 0.0))


def _cv(values: list[float]) -> float:
    """Return the coefficient of variation (std / mean) over positive values only.

    Zero-weight edges are treated as absent from the collaboration network
    (no co-authorship recurrence), not as weak collaborations.  They are
    excluded from CV computation to avoid inflating dispersion: a pair of
    authors who have never co-authored within the window simply has no edge,
    and including that zero would conflate structural absence with low-but-
    present collaboration weight.  If all values are zero or the list is
    empty, CV is defined as 0.0 (no variation among present edges).
    """
    positives = [value for value in values if value > 0.0]
    if not positives:
        return 0.0
    mu = mean(positives)
    if mu <= EPS:
        return 0.0
    var = sum((value - mu) ** 2 for value in positives) / len(positives)
    return math.sqrt(var) / (mu + EPS)


def _rankdata(values: list[float]) -> list[float]:
    pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][1] == pairs[idx][1]:
            end += 1
        rank = (idx + end - 1) / 2.0 + 1.0
        for pos in range(idx, end):
            ranks[pairs[pos][0]] = rank
        idx = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    xbar = mean(xs)
    ybar = mean(ys)
    num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - xbar) ** 2 for x in xs))
    deny = math.sqrt(sum((y - ybar) ** 2 for y in ys))
    if denx <= EPS or deny <= EPS:
        return 0.0
    return num / (denx * deny)


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_rankdata(xs), _rankdata(ys))


def _codim1_faces(simplex: tuple[str, ...]) -> list[tuple[str, ...]]:
    if len(simplex) <= 1:
        return []
    return [simplex[:idx] + simplex[idx + 1 :] for idx in range(len(simplex))]


def _codim_minus1_cofaces(complex_, simplex: tuple[str, ...]) -> list[tuple[str, ...]]:
    target_size = len(simplex) + 1
    if target_size == 2:
        return [edge for edge in complex_.edges if set(simplex).issubset(edge)]
    if target_size == 3:
        return [triangle for triangle in complex_.triangles if set(simplex).issubset(triangle)]
    return []


def compute_simplex_observables(complex_, simplex: tuple[str, ...], *, weight_maps: dict[int, dict] | None = None) -> dict[str, float]:
    lookup = _simplex_weight if weight_maps is None else (lambda _complex, s: _weight_from_maps(weight_maps, s))
    weight = lookup(complex_, simplex)
    face_weights = [lookup(complex_, face) for face in _codim1_faces(simplex)]
    coface_weights = [lookup(complex_, coface) for coface in _codim_minus1_cofaces(complex_, simplex)]

    mean_face = mean(face_weights) if face_weights else 0.0
    mean_coface = mean(coface_weights) if coface_weights else 0.0
    reinforcement = weight / (mean_face + EPS) if face_weights else 0.0
    boundary_strain = mean_coface / (weight + EPS) if coface_weights else 0.0
    coface_dependence = sum(coface_weights) / (weight + EPS) if coface_weights else 0.0
    terminality = weight / (sum(coface_weights) + EPS)
    tension = mean_coface - mean_face

    if face_weights and coface_weights:
        curvature = math.log((weight + EPS) / (mean_face + EPS)) - math.log((mean_coface + EPS) / (weight + EPS))
    elif face_weights:
        curvature = math.log((weight + EPS) / (mean_face + EPS))
    elif coface_weights:
        curvature = -math.log((mean_coface + EPS) / (weight + EPS))
    else:
        curvature = 0.0

    return {
        "weight": weight,
        "mean_face_weight": mean_face,
        "mean_coface_weight": mean_coface,
        "reinforcement": reinforcement,
        "boundary_strain": boundary_strain,
        "coface_dependence": coface_dependence,
        "terminality": terminality,
        "tension": tension,
        "curvature_balance": curvature,
    }


def _hasse_neighbors(complex_, simplex: tuple[str, ...]) -> set[tuple[str, ...]]:
    neighbors = set(_codim1_faces(simplex))
    neighbors.update(_codim_minus1_cofaces(complex_, simplex))
    return neighbors


def extract_core_simplex_region(complex_, simplex: tuple[str, ...], *, radius: int = 1) -> set[tuple[str, ...]]:
    region = {simplex}
    frontier = {simplex}
    for _ in range(max(radius, 0)):
        new_frontier = set()
        for current in frontier:
            for neighbor in _hasse_neighbors(complex_, current):
                if neighbor not in region:
                    region.add(neighbor)
                    new_frontier.add(neighbor)
        frontier = new_frontier
        if not frontier:
            break
    return region


def extract_author_ego_region(complex_, author: str, *, radius: int = 1) -> set[tuple[str, ...]]:
    seed = {simplex for simplex in _all_simplices(complex_) if author in simplex}
    region = set(seed)
    frontier = set(seed)
    for _ in range(max(radius, 0)):
        new_frontier = set()
        for current in frontier:
            for neighbor in _hasse_neighbors(complex_, current):
                if neighbor not in region:
                    region.add(neighbor)
                    new_frontier.add(neighbor)
        frontier = new_frontier
        if not frontier:
            break
    return region


def compute_region_volume(complex_, region: set[tuple[str, ...]], *, weight_maps: dict[int, dict]) -> float:
    return sum(_weight_from_maps(weight_maps, simplex) for simplex in region)


def compute_boundary_mass(complex_, region: set[tuple[str, ...]], *, weight_maps: dict[int, dict]) -> float:
    total = 0.0
    seen = set()
    for simplex in region:
        for neighbor in _hasse_neighbors(complex_, simplex):
            if neighbor in region:
                continue
            edge = tuple(sorted((simplex, neighbor), key=lambda item: (len(item), item)))
            if edge in seen:
                continue
            seen.add(edge)
            total += min(_weight_from_maps(weight_maps, simplex), _weight_from_maps(weight_maps, neighbor))
    return total


def compute_cohesion_ratio(complex_, region: set[tuple[str, ...]], *, weight_maps: dict[int, dict]) -> float:
    volume = compute_region_volume(complex_, region, weight_maps=weight_maps)
    boundary = compute_boundary_mass(complex_, region, weight_maps=weight_maps)
    return volume / (boundary + EPS)


def compute_conductance_score(complex_, region: set[tuple[str, ...]], *, weight_maps: dict[int, dict]) -> float:
    volume = compute_region_volume(complex_, region, weight_maps=weight_maps)
    full_region = set(_all_simplices(complex_))
    complement = full_region - region
    boundary = compute_boundary_mass(complex_, region, weight_maps=weight_maps)
    complement_volume = compute_region_volume(complex_, complement, weight_maps=weight_maps)
    return boundary / (min(volume, complement_volume) + EPS)


def compute_local_anisotropy(complex_, simplex: tuple[str, ...], *, weight_maps: dict[int, dict]) -> float:
    face_weights = [_weight_from_maps(weight_maps, face) for face in _codim1_faces(simplex)]
    coface_weights = [_weight_from_maps(weight_maps, coface) for coface in _codim_minus1_cofaces(complex_, simplex)]
    return _cv(face_weights) + _cv(coface_weights)


def _ball_region(complex_, simplex: tuple[str, ...], *, radius: int) -> set[tuple[str, ...]]:
    return extract_core_simplex_region(complex_, simplex, radius=radius)


def compute_volume_growth(complex_, simplex: tuple[str, ...], *, weight_maps: dict[int, dict], radius: int = 1) -> float:
    ball_r = _ball_region(complex_, simplex, radius=radius)
    ball_next = _ball_region(complex_, simplex, radius=radius + 1)
    volume_r = compute_region_volume(complex_, ball_r, weight_maps=weight_maps)
    volume_next = compute_region_volume(complex_, ball_next, weight_maps=weight_maps)
    return volume_next / (volume_r + EPS)


def compute_multiscale_consistency(complex_, region: set[tuple[str, ...]], *, weight_maps: dict[int, dict]) -> float:
    by_dimension: dict[int, list[tuple[str, ...]]] = defaultdict(list)
    for simplex in region:
        by_dimension[_simplex_dimension(simplex)].append(simplex)
    q_values = []
    for dimension in sorted(by_dimension):
        simplices = by_dimension[dimension]
        if not simplices:
            continue
        logs = []
        for simplex in simplices:
            if len(simplex) <= 1:
                continue
            weight = _weight_from_maps(weight_maps, simplex)
            m_minus = mean(_weight_from_maps(weight_maps, face) for face in _codim1_faces(simplex))
            logs.append(math.log((m_minus + EPS) / (weight + EPS)))
        if logs:
            q_values.append(mean(logs))
    if len(q_values) <= 1:
        return 1.0
    qbar = mean(q_values)
    variance = sum((value - qbar) ** 2 for value in q_values) / len(q_values)
    return 1.0 / (1.0 + variance)


def _simplex_rows_for_window(venue: str, complex_) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for simplex in complex_.edges + complex_.triangles:
        observables = compute_simplex_observables(complex_, simplex)
        rows.append(
            {
                "venue": venue,
                "window": complex_.label,
                "simplex": "-".join(simplex),
                "simplex_key": _encode_simplex(simplex),
                "dimension": len(simplex) - 1,
                **observables,
            }
        )
    return rows


def _compute_graph_features(complex_) -> dict[str, dict[tuple[str, ...], float]]:
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in complex_.vertices}
    triangle_counts: Counter[tuple[str, str]] = Counter()
    for u, v in complex_.edges:
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    for triangle in complex_.triangles:
        for edge in _codim1_faces(triangle):
            if len(edge) == 2:
                triangle_counts[tuple(sorted(edge))] += 1

    def edge_betweenness() -> dict[tuple[str, str], float]:
        edge_scores = {tuple(sorted(edge)): 0.0 for edge in complex_.edges}
        vertices = list(adjacency)
        for source in vertices:
            stack: list[str] = []
            preds = {node: [] for node in vertices}
            sigma = {node: 0.0 for node in vertices}
            sigma[source] = 1.0
            dist = {node: -1 for node in vertices}
            dist[source] = 0
            queue = [source]
            qidx = 0
            while qidx < len(queue):
                v = queue[qidx]
                qidx += 1
                stack.append(v)
                for w in adjacency.get(v, ()):
                    if dist[w] < 0:
                        queue.append(w)
                        dist[w] = dist[v] + 1
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        preds[w].append(v)
            delta = {node: 0.0 for node in vertices}
            while stack:
                w = stack.pop()
                for v in preds[w]:
                    coeff = (sigma[v] / sigma[w]) * (1.0 + delta[w]) if sigma[w] > 0 else 0.0
                    edge = tuple(sorted((v, w)))
                    edge_scores[edge] += coeff
                    delta[v] += coeff
        for edge in edge_scores:
            edge_scores[edge] /= 2.0
        return edge_scores

    edge_btw = edge_betweenness()
    edge_features: dict[tuple[str, ...], dict[str, float]] = {}
    triangle_features: dict[tuple[str, ...], dict[str, float]] = {}
    vertex_strength = {vertex: float(complex_.vertex_weights.get(vertex, 0.0)) for vertex in complex_.vertices}

    for edge in complex_.edges:
        u, v = edge
        common = adjacency.get(u, set()) & adjacency.get(v, set())
        edge_features[edge] = {
            "endpoint_mean_strength": (vertex_strength.get(u, 0.0) + vertex_strength.get(v, 0.0)) / 2.0,
            "endpoint_mean_degree": (len(adjacency.get(u, ())) + len(adjacency.get(v, ()))) / 2.0,
            "triangle_participation": float(triangle_counts.get(tuple(sorted(edge)), 0.0)),
            "common_neighbor_count": float(len(common)),
            "edge_betweenness": float(edge_btw.get(tuple(sorted(edge)), 0.0)),
        }
    for triangle in complex_.triangles:
        edges = [tuple(sorted(edge)) for edge in _codim1_faces(triangle)]
        triangle_features[triangle] = {
            "mean_boundary_edge_weight": mean(float(complex_.edge_weights.get(edge, 0.0)) for edge in edges),
            "mean_boundary_edge_degree": mean(edge_features[edge]["endpoint_mean_degree"] for edge in edges),
            "triangle_edge_participation": mean(edge_features[edge]["triangle_participation"] for edge in edges),
        }
    return {"edge": edge_features, "triangle": triangle_features}


def _extended_region_metrics(complex_, simplex: tuple[str, ...], *, weight_maps: dict[int, dict]) -> dict[str, float]:
    core_region = extract_core_simplex_region(complex_, simplex, radius=1)
    return {
        "anisotropy": compute_local_anisotropy(complex_, simplex, weight_maps=weight_maps),
        "volume_growth_r1": compute_volume_growth(complex_, simplex, weight_maps=weight_maps, radius=1),
        "volume_growth_r2": compute_volume_growth(complex_, simplex, weight_maps=weight_maps, radius=2),
        "core_region_size": float(len(core_region)),
        "core_region_cohesion": compute_cohesion_ratio(complex_, core_region, weight_maps=weight_maps),
        "core_region_conductance": compute_conductance_score(complex_, core_region, weight_maps=weight_maps),
        "core_region_multiscale_consistency": compute_multiscale_consistency(complex_, core_region, weight_maps=weight_maps),
    }


def _aggregate_dimension_profile(rows: list[dict[str, object]], *, venue: str, scope: str) -> list[dict[str, object]]:
    profile_rows: list[dict[str, object]] = []
    dimensions = sorted({int(row["dimension"]) for row in rows})
    for dimension in dimensions:
        dim_rows = [row for row in rows if int(row["dimension"]) == dimension]
        profile_rows.append(
            {
                "venue": venue,
                "scope": scope,
                "dimension": dimension,
                "simplex_count": len(dim_rows),
                "mean_weight": mean(float(row["weight"]) for row in dim_rows),
                "mean_reinforcement": mean(float(row["reinforcement"]) for row in dim_rows),
                "mean_boundary_strain": mean(float(row["boundary_strain"]) for row in dim_rows),
                "mean_tension": mean(float(row["tension"]) for row in dim_rows),
                "mean_curvature_balance": mean(float(row["curvature_balance"]) for row in dim_rows),
                "negative_curvature_fraction": sum(float(row["curvature_balance"]) < -1e-9 for row in dim_rows) / max(len(dim_rows), 1),
                "zero_curvature_fraction": sum(abs(float(row["curvature_balance"])) <= 1e-9 for row in dim_rows) / max(len(dim_rows), 1),
            }
        )
    return profile_rows


def _baseline_correlation_rows(rows: list[dict[str, object]], *, venue: str, weight_semantics: str) -> list[dict[str, object]]:
    correlation_rows: list[dict[str, object]] = []
    by_dimension: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_dimension[int(row["dimension"])].append(row)

    feature_sets = {
        1: ["weight", "reinforcement", "boundary_strain", "curvature_balance"],
        2: ["weight", "reinforcement", "curvature_balance"],
    }
    baseline_sets = {
        1: ["endpoint_mean_strength", "endpoint_mean_degree", "triangle_participation", "common_neighbor_count", "edge_betweenness"],
        2: ["mean_boundary_edge_weight", "mean_boundary_edge_degree", "triangle_edge_participation"],
    }
    for dimension, dim_rows in by_dimension.items():
        for feature in feature_sets.get(dimension, []):
            xs = [float(row[feature]) for row in dim_rows]
            for baseline in baseline_sets.get(dimension, []):
                ys = [float(row.get(baseline, 0.0)) for row in dim_rows]
                correlation_rows.append(
                    {
                        "venue": venue,
                        "weight_semantics": weight_semantics,
                        "dimension": dimension,
                        "feature": feature,
                        "baseline": baseline,
                        "pearson": _pearson(xs, ys),
                        "spearman": _spearman(xs, ys),
                    }
                )
    return correlation_rows


def _brokerage_validation_rows(rows: list[dict[str, object]], *, venue: str, weight_semantics: str) -> list[dict[str, object]]:
    edge_rows = [row for row in rows if int(row["dimension"]) == 1]
    if len(edge_rows) < 5:
        return []
    sorted_rows = sorted(edge_rows, key=lambda row: float(row["curvature_balance"]))
    k = max(5, len(sorted_rows) // 10)
    negative = sorted_rows[:k]
    neutral = sorted(sorted_rows, key=lambda row: abs(float(row["curvature_balance"])))[:k]
    positive = sorted_rows[-k:]
    output = []
    for label, sample in [("most_negative", negative), ("near_zero", neutral), ("most_positive", positive)]:
        output.append(
            {
                "venue": venue,
                "weight_semantics": weight_semantics,
                "group": label,
                "sample_size": len(sample),
                "mean_curvature_balance": mean(float(row["curvature_balance"]) for row in sample),
                "mean_edge_betweenness": mean(float(row.get("edge_betweenness", 0.0)) for row in sample),
                "mean_triangle_participation": mean(float(row.get("triangle_participation", 0.0)) for row in sample),
                "mean_common_neighbor_count": mean(float(row.get("common_neighbor_count", 0.0)) for row in sample),
            }
        )
    output.append(
        {
            "venue": venue,
            "weight_semantics": weight_semantics,
            "group": "global_correlation",
            "sample_size": len(edge_rows),
            "mean_curvature_balance": mean(float(row["curvature_balance"]) for row in edge_rows),
            "mean_edge_betweenness": _spearman(
                [float(row["curvature_balance"]) for row in edge_rows],
                [float(row.get("edge_betweenness", 0.0)) for row in edge_rows],
            ),
            "mean_triangle_participation": _spearman(
                [float(row["curvature_balance"]) for row in edge_rows],
                [float(row.get("triangle_participation", 0.0)) for row in edge_rows],
            ),
            "mean_common_neighbor_count": _spearman(
                [float(row["curvature_balance"]) for row in edge_rows],
                [float(row.get("common_neighbor_count", 0.0)) for row in edge_rows],
            ),
        }
    )
    for metric in ["anisotropy", "volume_growth_r1", "core_region_cohesion", "core_region_conductance", "core_region_multiscale_consistency"]:
        output.append(
            {
                "venue": venue,
                "weight_semantics": weight_semantics,
                "group": f"diagnostic_{metric}",
                "sample_size": len(negative),
                "mean_curvature_balance": mean(float(row[metric]) for row in negative),
                "mean_edge_betweenness": mean(float(row[metric]) for row in neutral),
                "mean_triangle_participation": mean(float(row[metric]) for row in positive),
                "mean_common_neighbor_count": 0.0,
            }
        )
    return output


def _plot_geometry_scatter(path: Path, rows: list[dict[str, object]], *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for dimension, color, label in [(1, "C1", "edges"), (2, "C0", "triangles")]:
        dim_rows = [row for row in rows if int(row["dimension"]) == dimension]
        ax.scatter(
            [float(row["reinforcement"]) for row in dim_rows],
            [float(row["curvature_balance"]) for row in dim_rows],
            s=22,
            alpha=0.7,
            color=color,
            label=label,
        )
    ax.axhline(0.0, color="0.7", linewidth=1.0)
    ax.set_xlabel("reinforcement")
    ax.set_ylabel("curvature balance")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_baseline_vs_ricbal(path: Path, rows: list[dict[str, object]], *, title: str) -> None:
    edge_rows = [row for row in rows if int(row["dimension"]) == 1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        [float(row.get("edge_betweenness", 0.0)) for row in edge_rows],
        [float(row["curvature_balance"]) for row in edge_rows],
        color="C3",
        alpha=0.65,
        s=20,
    )
    ax.set_xlabel("edge betweenness")
    ax.set_ylabel("curvature balance")
    ax.set_title(title)
    ax.axhline(0.0, color="0.7", linewidth=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_dimension_profile(path: Path, rows: list[dict[str, object]], *, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    dims = [str(int(row["dimension"])) for row in rows]
    axes[0].bar(dims, [float(row["mean_reinforcement"]) for row in rows], color="C0", alpha=0.85)
    axes[0].set_title("Mean reinforcement")
    axes[0].set_xlabel("dimension")
    axes[1].bar(dims, [float(row["mean_curvature_balance"]) for row in rows], color="C2", alpha=0.85)
    axes[1].set_title("Mean curvature balance")
    axes[1].set_xlabel("dimension")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_weight_sensitivity(path: Path, rows: list[dict[str, object]]) -> None:
    semantics = sorted({row["weight_semantics"] for row in rows})
    venues = sorted({row["venue"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.22
    for ax, dimension, title in [(axes[0], 1, "Mean edge balance"), (axes[1], 2, "Mean triangle balance")]:
        x_positions = list(range(len(semantics)))
        for idx, venue in enumerate(venues):
            venue_rows = [row for row in rows if row["venue"] == venue and int(row["dimension"]) == dimension]
            vals = [float(next(r["mean_curvature_balance"] for r in venue_rows if r["weight_semantics"] == semantic)) for semantic in semantics]
            shifted = [x + (idx - 0.5) * width for x in x_positions]
            ax.bar(shifted, vals, width=width, label=venue)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(semantics, rotation=20, ha="right")
        ax.set_title(title)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_local_case_profiles(path: Path, cases: list[dict[str, object]]) -> None:
    if not cases:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    flat_axes = list(axes.flatten())
    for ax, row in zip(flat_axes, cases):
        support_labels = ["simplex", "faces", "cofaces"]
        support_values = [
            float(row["weight"]),
            float(row["mean_face_weight"]),
            float(row["mean_coface_weight"]),
        ]
        ax.bar(support_labels, support_values, color=["C0", "C1", "C2"], alpha=0.85)
        metric_text = (
            f"R={float(row['reinforcement']):.2f}\n"
            f"T={float(row['tension']):.1f}\n"
            f"Ric={float(row['curvature_balance']):.2f}"
        )
        ax.text(
            0.98,
            0.95,
            metric_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )
        ax.set_title(f"{row['venue']} {row['kind']}: {row['simplex']}", fontsize=10)
        ax.set_ylabel("local support")
    for ax in flat_axes[len(cases) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_ricbal_vs_anisotropy(path: Path, rows: list[dict[str, object]], *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for dimension, color, label in [(1, "C1", "edges"), (2, "C0", "triangles")]:
        dim_rows = [row for row in rows if int(row["dimension"]) == dimension]
        ax.scatter(
            [float(row["anisotropy"]) for row in dim_rows],
            [float(row["curvature_balance"]) for row in dim_rows],
            s=22,
            alpha=0.7,
            color=color,
            label=label,
        )
    ax.set_xlabel("anisotropy")
    ax.set_ylabel("curvature balance")
    ax.set_title(title)
    ax.axhline(0.0, color="0.7", linewidth=1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_featured_region_bars(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    labels = [f"{row['venue']} {row['kind']}" for row in rows[:4]]
    cohesion = [float(row["core_region_cohesion"]) for row in rows[:4]]
    conductance = [float(row["core_region_conductance"]) for row in rows[:4]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, cohesion, color="C0", alpha=0.85)
    axes[0].set_title("Core-region cohesion")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, conductance, color="C3", alpha=0.85)
    axes[1].set_title("Core-region conductance")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _node_positions_for_case(simplex: tuple[str, ...], extras: list[str]) -> dict[str, tuple[float, float]]:
    if len(simplex) == 2:
        positions = {
            simplex[0]: (-1.0, 0.0),
            simplex[1]: (1.0, 0.0),
        }
        count = len(extras)
        if count == 1:
            positions[extras[0]] = (0.0, 1.0)
        elif count:
            radius = 1.4
            start = math.pi / 6
            stop = 5 * math.pi / 6
            for idx, node in enumerate(extras):
                angle = start + (stop - start) * idx / max(count - 1, 1)
                positions[node] = (radius * math.cos(angle), radius * math.sin(angle))
        return positions

    if len(simplex) == 3:
        positions = {
            simplex[0]: (-0.9, -0.55),
            simplex[1]: (0.9, -0.55),
            simplex[2]: (0.0, 1.0),
        }
        count = len(extras)
        if count:
            radius = 1.8
            start = -math.pi / 6
            stop = 7 * math.pi / 6
            for idx, node in enumerate(extras):
                angle = start + (stop - start) * idx / max(count - 1, 1)
                positions[node] = (radius * math.cos(angle), radius * math.sin(angle))
        return positions

    return {node: (float(idx), 0.0) for idx, node in enumerate(simplex + tuple(extras))}


def _plot_featured_neighborhoods(path: Path, complex_, cases: list[dict[str, object]]) -> None:
    if not cases:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    flat_axes = list(axes.flatten())
    for ax, row in zip(flat_axes, cases):
        simplex = _decode_simplex(str(row.get("simplex_key", row["simplex"])))
        simplex_set = set(simplex)
        adjacent_triangles = [triangle for triangle in complex_.triangles if simplex_set.issubset(triangle)]
        extras = sorted({node for triangle in adjacent_triangles for node in triangle if node not in simplex_set})
        positions = _node_positions_for_case(simplex, extras)

        for triangle in adjacent_triangles:
            coords = [positions[node] for node in triangle]
            patch = Polygon(coords, closed=True, facecolor="C0", alpha=0.16, edgecolor="none")
            ax.add_patch(patch)

        edge_set = set()
        if len(simplex) == 2:
            neighborhood_edges = {
                tuple(sorted((simplex[0], simplex[1]))),
                *{
                    tuple(sorted((simplex[0], node)))
                    for node in extras
                },
                *{
                    tuple(sorted((simplex[1], node)))
                    for node in extras
                },
            }
        else:
            neighborhood_edges = {
                tuple(sorted(edge))
                for triangle in adjacent_triangles or [simplex]
                for edge in _codim1_faces(triangle)
            }
        for edge in sorted(neighborhood_edges):
            if len(edge) != 2 or edge[0] not in positions or edge[1] not in positions:
                continue
            if edge in edge_set:
                continue
            edge_set.add(edge)
            x0, y0 = positions[edge[0]]
            x1, y1 = positions[edge[1]]
            weight = complex_.edge_weights.get(edge, 1.0)
            is_core = set(edge) == simplex_set if len(simplex) == 2 else edge[0] in simplex_set and edge[1] in simplex_set
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="crimson" if is_core else "0.45",
                linewidth=2.8 if is_core else 1.0 + 0.35 * min(weight, 4.0),
                alpha=0.95 if is_core else 0.8,
                zorder=2,
            )

        if len(simplex) == 3:
            coords = [positions[node] for node in simplex]
            patch = Polygon(coords, closed=True, facecolor="crimson", alpha=0.18, edgecolor="crimson", linewidth=2.0)
            ax.add_patch(patch)

        for node, (x, y) in positions.items():
            weight = complex_.vertex_weights.get(node, 1.0)
            is_core = node in simplex_set
            ax.scatter(
                [x],
                [y],
                s=120 + 18 * min(weight, 8.0),
                color="crimson" if is_core else "white",
                edgecolors="black" if is_core else "0.35",
                linewidths=1.2,
                zorder=3,
            )
            ax.text(x, y - 0.16, node, ha="center", va="top", fontsize=7)

        ax.set_title(f"{row['venue']} {row['kind']}", fontsize=10)
        ax.text(
            0.02,
            0.98,
            f"simplex: {row['simplex']}\nRic={float(row['curvature_balance']):.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )
        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(-1.5, 1.9)
        ax.axis("off")
    for ax in flat_axes[len(cases) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _featured_region_rows(complex_, cases: list[dict[str, object]], *, weight_maps: dict[int, dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in cases:
        simplex = _decode_simplex(str(row.get("simplex_key", row["simplex"])))
        core_region = extract_core_simplex_region(complex_, simplex, radius=1)
        rows.append(
            {
                "venue": row["venue"],
                "window": row["window"],
                "simplex": row["simplex"],
                "simplex_key": row.get("simplex_key", ""),
                "kind": row["kind"],
                "region_type": "core_simplex_radius_1",
                "region_size": len(core_region),
                "region_volume": compute_region_volume(complex_, core_region, weight_maps=weight_maps),
                "boundary_mass": compute_boundary_mass(complex_, core_region, weight_maps=weight_maps),
                "cohesion_ratio": compute_cohesion_ratio(complex_, core_region, weight_maps=weight_maps),
                "conductance_score": compute_conductance_score(complex_, core_region, weight_maps=weight_maps),
                "multiscale_consistency": compute_multiscale_consistency(complex_, core_region, weight_maps=weight_maps),
                "anisotropy": float(row["anisotropy"]),
                "volume_growth_r1": float(row["volume_growth_r1"]),
                "volume_growth_r2": float(row["volume_growth_r2"]),
            }
        )
        author = simplex[0]
        ego_region = extract_author_ego_region(complex_, author, radius=1)
        rows.append(
            {
                "venue": row["venue"],
                "window": row["window"],
                "simplex": row["simplex"],
                "simplex_key": row.get("simplex_key", ""),
                "kind": row["kind"],
                "region_type": f"author_ego_{author}",
                "region_size": len(ego_region),
                "region_volume": compute_region_volume(complex_, ego_region, weight_maps=weight_maps),
                "boundary_mass": compute_boundary_mass(complex_, ego_region, weight_maps=weight_maps),
                "cohesion_ratio": compute_cohesion_ratio(complex_, ego_region, weight_maps=weight_maps),
                "conductance_score": compute_conductance_score(complex_, ego_region, weight_maps=weight_maps),
                "multiscale_consistency": compute_multiscale_consistency(complex_, ego_region, weight_maps=weight_maps),
                "anisotropy": float(row["anisotropy"]),
                "volume_growth_r1": float(row["volume_growth_r1"]),
                "volume_growth_r2": float(row["volume_growth_r2"]),
            }
        )
    return rows


def _comparison_rows(rows: list[dict[str, object]], *, venue: str, weight_semantics: str) -> list[dict[str, object]]:
    edge_rows = sorted([row for row in rows if int(row["dimension"]) == 1], key=lambda row: float(row["curvature_balance"]))[:10]
    triangle_rows = sorted(
        [row for row in rows if int(row["dimension"]) == 2 and abs(float(row["curvature_balance"])) <= 1e-9],
        key=lambda row: float(row["weight"]),
        reverse=True,
    )[:10]
    out = []
    for label, sample in [("brokerage_edges", edge_rows), ("balanced_triangles", triangle_rows)]:
        if not sample:
            continue
        out.append(
            {
                "venue": venue,
                "weight_semantics": weight_semantics,
                "group": label,
                "sample_size": len(sample),
                "mean_anisotropy": mean(float(row["anisotropy"]) for row in sample),
                "mean_volume_growth_r1": mean(float(row["volume_growth_r1"]) for row in sample),
                "mean_core_region_cohesion": mean(float(row["core_region_cohesion"]) for row in sample),
                "mean_core_region_conductance": mean(float(row["core_region_conductance"]) for row in sample),
                "mean_multiscale_consistency": mean(float(row["core_region_multiscale_consistency"]) for row in sample),
            }
        )
    return out


def _resolve_input_config(config_path: Path) -> dict[str, object]:
    return load_config(config_path)


def run_geometry_experiment(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    config = _resolve_input_config(path)
    base = path.parent.parent if path.parent.name == "configs" else path.parent
    output_dir = (base / str(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = (base / str(config["input_file"])).resolve()
    records = load_csv_records(input_file)
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
    weight_semantics = str(config.get("weight_semantics", "contained-support"))
    decay_lambda = float(config.get("decay_lambda", 0.6))
    complexes = [
        build_window_complex(
            window,
            weight_mode=str(config.get("weight_mode", "contained")),
            max_simplex_size=int(config.get("max_simplex_size", 3)),
        )
        for window in windows
    ]

    venue = str(config.get("venue", "DBLP"))
    window_rows = []
    all_simplex_rows: list[dict[str, object]] = []
    all_profile_rows: list[dict[str, object]] = []
    all_correlation_rows: list[dict[str, object]] = []
    all_validation_rows: list[dict[str, object]] = []
    for window, complex_ in zip(windows, complexes, strict=True):
        summary = summarize_window(window, complex_)
        window_rows.append(
            {
                "venue": venue,
                "window": summary.window_label,
                "paper_count": summary.paper_count,
                "author_count": summary.author_count,
                "edge_count": summary.simplex_count_by_dim[1],
                "triangle_count": summary.simplex_count_by_dim[2],
                "edge_density": summary.density_stats["edge_density"],
                "triangle_density": summary.density_stats["triangle_density"],
            }
        )
        weight_maps = compute_weight_maps(
            window,
            weight_semantics=weight_semantics,
            max_simplex_size=int(config.get("max_simplex_size", 3)),
            decay_lambda=decay_lambda,
        )
        simplex_rows = []
        graph_features = _compute_graph_features(complex_)
        for simplex in complex_.edges + complex_.triangles:
            observables = compute_simplex_observables(complex_, simplex, weight_maps=weight_maps)
            row = {
                "venue": venue,
                "window": complex_.label,
                "simplex": "-".join(simplex),
                "simplex_key": _encode_simplex(simplex),
                "dimension": len(simplex) - 1,
                "weight_semantics": weight_semantics,
                **observables,
            }
            if len(simplex) == 2:
                row.update(graph_features["edge"].get(simplex, {}))
            elif len(simplex) == 3:
                row.update(graph_features["triangle"].get(simplex, {}))
            simplex_rows.append(row)
        all_simplex_rows.extend(simplex_rows)
        all_profile_rows.extend(_aggregate_dimension_profile(simplex_rows, venue=venue, scope=complex_.label))

    latest_window = str(config.get("report_window", "latest"))
    latest_label = windows[-1].label if latest_window == "latest" else latest_window
    latest_rows = [row for row in all_simplex_rows if row["window"] == latest_label]
    latest_complex = complexes[-1]
    latest_weight_maps = compute_weight_maps(
        windows[-1],
        weight_semantics=weight_semantics,
        max_simplex_size=int(config.get("max_simplex_size", 3)),
        decay_lambda=decay_lambda,
    )
    for row in latest_rows:
        simplex = _decode_simplex(str(row.get("simplex_key", row["simplex"])))
        row.update(_extended_region_metrics(latest_complex, simplex, weight_maps=latest_weight_maps))
    latest_profile = _aggregate_dimension_profile(latest_rows, venue=venue, scope="latest")
    latest_profile = [{**row, "weight_semantics": weight_semantics} for row in latest_profile]
    all_profile_rows = [{**row, "weight_semantics": weight_semantics} for row in all_profile_rows]
    all_correlation_rows.extend(_baseline_correlation_rows(latest_rows, venue=venue, weight_semantics=weight_semantics))
    all_validation_rows.extend(_brokerage_validation_rows(latest_rows, venue=venue, weight_semantics=weight_semantics))

    top_brokerage = sorted(
        [row for row in latest_rows if int(row["dimension"]) == 1],
        key=lambda row: float(row["curvature_balance"]),
    )[:10]
    top_balanced = sorted(
        [row for row in latest_rows if int(row["dimension"]) == 2],
        key=lambda row: (float(row["weight"]), float(row["curvature_balance"])),
        reverse=True,
    )[:10]

    _write_csv(output_dir / "window_summary.csv", window_rows)
    _write_csv(output_dir / "simplex_observables.csv", all_simplex_rows)
    _write_csv(output_dir / "dimension_profile.csv", all_profile_rows)
    _write_csv(output_dir / "latest_dimension_profile.csv", latest_profile)
    _write_csv(output_dir / "baseline_correlations.csv", all_correlation_rows)
    _write_csv(output_dir / "brokerage_validation.csv", all_validation_rows)
    _write_csv(output_dir / "top_brokerage_edges.csv", top_brokerage)
    _write_csv(output_dir / "top_balanced_triangles.csv", top_balanced)
    _plot_geometry_scatter(
        output_dir / "latest_geometry_scatter.png",
        latest_rows,
        title=f"{venue} {latest_label}: reinforcement vs curvature balance",
    )
    _plot_dimension_profile(
        output_dir / "latest_dimension_profile.png",
        latest_profile,
        title=f"{venue} {latest_label}: dimension profile",
    )
    _plot_baseline_vs_ricbal(
        output_dir / "baseline_vs_ricbal_plot.png",
        latest_rows,
        title=f"{venue} {latest_label}: edge betweenness vs curvature balance",
    )
    local_cases = []
    if top_brokerage:
        local_cases.append({**top_brokerage[0], "kind": "brokerage edge"})
    if top_balanced:
        local_cases.append({**top_balanced[0], "kind": "balanced triangle"})
    featured_region_rows = _featured_region_rows(latest_complex, local_cases, weight_maps=latest_weight_maps)
    comparison_rows = _comparison_rows(latest_rows, venue=venue, weight_semantics=weight_semantics)
    _write_csv(output_dir / "featured_cases.csv", local_cases)
    _write_csv(output_dir / "simplex_extended_geometry.csv", latest_rows)
    _write_csv(output_dir / "featured_region_geometry.csv", featured_region_rows)
    _write_csv(output_dir / "regional_geometry.csv", featured_region_rows)
    _write_csv(output_dir / "group_geometry_comparison.csv", comparison_rows)
    _plot_local_case_profiles(output_dir / "featured_cases.png", local_cases)
    _plot_featured_neighborhoods(output_dir / "featured_neighborhoods.png", latest_complex, local_cases)
    _plot_ricbal_vs_anisotropy(
        output_dir / "ricbal_vs_anisotropy.png",
        latest_rows,
        title=f"{venue} {latest_label}: anisotropy vs balance",
    )
    _plot_featured_region_bars(output_dir / "featured_region_geometry.png", local_cases)
    _write_run_params(output_dir, path, config)
    return output_dir


def _write_run_params(output_dir: Path, config_path: Path, config: dict) -> None:
    """Write a provenance stamp alongside experiment outputs.

    Records the config path and its SHA-256 hash, the Python version, the
    seldon-lab module path, and a UTC timestamp.  This costs almost nothing and
    makes reproducibility auditable (Critic Issue 14 / N4).
    """
    try:
        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]
    except Exception:
        config_hash = "unknown"

    try:
        import seldon_lab
        seldon_lab_version = getattr(seldon_lab, "__version__", "dev")
    except Exception:
        seldon_lab_version = "dev"

    try:
        import she_geofield
        geofield_version = getattr(she_geofield, "__version__", "dev")
    except Exception:
        geofield_version = "dev"

    params = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_sha256_prefix": config_hash,
        "seldon_lab_version": seldon_lab_version,
        "she_geofield_version": geofield_version,
        "config_summary": {k: v for k, v in config.items() if not k.startswith("_")},
    }
    run_params_path = output_dir / "run_params.json"
    run_params_path.write_text(json.dumps(params, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Seldon DBLP static-geometry experiment.")
    parser.add_argument("--config", required=True, help="Path to the geometry config file.")
    args = parser.parse_args()
    outdir = run_geometry_experiment(args.config)
    print(f"Seldon geometry experiment complete. Outputs in {outdir}")


if __name__ == "__main__":
    main()
