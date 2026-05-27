from collections import Counter, defaultdict
from itertools import combinations

from .records import DblpSimplicialComplex, PaperRecord, TimeWindow


def _sorted_simplex(authors: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(authors))


def _iter_contained_simplices(authors: tuple[str, ...], max_simplex_size: int):
    upper = min(len(authors), max_simplex_size)
    for size in range(1, upper + 1):
        yield from combinations(authors, size)


def build_window_complex(
    window: TimeWindow,
    *,
    weight_mode: str = "contained",
    max_simplex_size: int = 3,
) -> DblpSimplicialComplex:
    if weight_mode not in {"contained", "exact"}:
        raise ValueError("weight_mode must be 'contained' or 'exact'")

    exact_support: Counter[tuple[str, ...]] = Counter()
    contained_support: Counter[tuple[str, ...]] = Counter()
    supporting_papers: defaultdict[tuple[str, ...], list[str]] = defaultdict(list)
    record_by_key: dict[str, PaperRecord] = {record.key: record for record in window.records}

    for record in window.records:
        authors = _sorted_simplex(record.authors)
        exact_support[authors] += 1.0
        supporting_papers[authors].append(record.key)
        for simplex in _iter_contained_simplices(authors, max_simplex_size=max_simplex_size):
            contained_support[simplex] += 1.0
            supporting_papers[simplex].append(record.key)

    weight_source = exact_support if weight_mode == "exact" else contained_support
    tiny = 1e-9

    vertices = sorted(simplex[0] for simplex in contained_support if len(simplex) == 1)
    edges = sorted(simplex for simplex in contained_support if len(simplex) == 2)
    triangles = sorted(simplex for simplex in contained_support if len(simplex) == 3)

    vertex_weights = {
        vertex: float(contained_support.get((vertex,), tiny))
        for vertex in vertices
    }
    edge_weights = {
        edge: float(weight_source.get(edge, tiny) if weight_mode == "exact" else weight_source[edge])
        for edge in edges
    }
    triangle_weights = {
        triangle: float(
            weight_source.get(triangle, tiny) if weight_mode == "exact" else weight_source[triangle]
        )
        for triangle in triangles
    }

    simplex_data: dict[tuple[str, ...], dict[str, object]] = {}
    for simplex in contained_support:
        support_keys = tuple(sorted(set(supporting_papers.get(simplex, []))))
        support_records = [record_by_key[key] for key in support_keys if key in record_by_key]
        venue_counts = Counter(record.venue for record in support_records if record.venue)
        simplex_data[simplex] = {
            "team_size": len(simplex),
            "exact_support": float(exact_support.get(simplex, 0.0)),
            "contained_support": float(contained_support.get(simplex, 0.0)),
            "support_count": float(
                exact_support.get(simplex, 0.0) if weight_mode == "exact" else contained_support[simplex]
            ),
            "supporting_papers": support_keys,
            "venue_category": venue_counts.most_common(1)[0][0] if venue_counts else None,
        }

    return DblpSimplicialComplex(
        vertices=vertices,
        edges=edges,
        triangles=triangles,
        vertex_weights=vertex_weights,
        edge_weights=edge_weights,
        triangle_weights=triangle_weights,
        exact_support=dict(exact_support),
        contained_support=dict(contained_support),
        supporting_papers={k: tuple(v) for k, v in supporting_papers.items()},
        simplex_data=simplex_data,
        window_start=window.start_year,
        window_end=window.end_year,
        records=window.records,
        weight_mode=weight_mode,
    )
