from math import comb

from .records import DblpSimplicialComplex, TimeWindow, WindowSummary


def build_rolling_windows(
    records,
    *,
    width: int,
    stride: int,
    start_year: int | None = None,
    end_year: int | None = None,
) -> list[TimeWindow]:
    if not records:
        return []

    min_year = min(record.year for record in records) if start_year is None else start_year
    max_year = max(record.year for record in records) if end_year is None else end_year
    if width <= 0 or stride <= 0:
        raise ValueError("width and stride must be positive")
    if min_year > max_year:
        return []

    windows: list[TimeWindow] = []
    current = min_year
    sorted_records = tuple(sorted(records, key=lambda rec: (rec.year, rec.key)))
    while current + width - 1 <= max_year:
        window_end = current + width - 1
        window_records = tuple(
            record for record in sorted_records if current <= record.year <= window_end
        )
        windows.append(TimeWindow(current, window_end, window_records))
        current += stride
    return windows


def summarize_window(
    window: TimeWindow,
    complex_: DblpSimplicialComplex | None = None,
) -> WindowSummary:
    author_count = len({author for record in window.records for author in record.authors})
    simplex_counts = {0: 0, 1: 0, 2: 0}
    densities = {"edge_density": 0.0, "triangle_density": 0.0}

    if complex_ is not None:
        simplex_counts = {
            0: len(complex_.vertices),
            1: len(complex_.edges),
            2: len(complex_.triangles),
        }
        n_vertices = len(complex_.vertices)
        max_edges = comb(n_vertices, 2) if n_vertices >= 2 else 0
        max_triangles = comb(n_vertices, 3) if n_vertices >= 3 else 0
        densities = {
            "edge_density": len(complex_.edges) / max_edges if max_edges else 0.0,
            "triangle_density": (
                len(complex_.triangles) / max_triangles if max_triangles else 0.0
            ),
        }

    return WindowSummary(
        window_label=window.label,
        paper_count=len(window.records),
        author_count=author_count,
        simplex_count_by_dim=simplex_counts,
        density_stats=densities,
    )
