from .records import DblpSimplicialComplex


def apply_window_decorations(complexes: list[DblpSimplicialComplex]) -> None:
    """Attach minimal DBLP decorations needed by the first experiment."""
    for idx, complex_ in enumerate(complexes):
        prev_support = complexes[idx - 1].contained_support if idx > 0 else {}
        next_support = complexes[idx + 1].contained_support if idx + 1 < len(complexes) else {}
        max_triangle_support = max(
            (complex_.simplex_data[triangle]["contained_support"] for triangle in complex_.triangles),
            default=1.0,
        )

        for simplex, data in complex_.simplex_data.items():
            prev_hit = 1.0 if prev_support.get(simplex, 0.0) > 0 else 0.0
            next_hit = 1.0 if next_support.get(simplex, 0.0) > 0 else 0.0
            persistence = 0.5 * (prev_hit + next_hit)
            support_count = float(data.get("support_count", 0.0))
            novelty_ratio = 1.0 - prev_hit
            reuse_ratio = prev_hit
            data.update(
                {
                    "persistence": persistence,
                    "novelty_ratio": novelty_ratio,
                    "reuse_ratio": reuse_ratio,
                    "window_label": complex_.label,
                }
            )

            if len(simplex) == 3:
                normalized_support = support_count / max_triangle_support if max_triangle_support else 0.0
                # Keep cohesion in [0, 1] so the existing toy flow stays numerically stable.
                complex_.triangle_cohesion[simplex] = min(
                    1.0,
                    0.5 * persistence + 0.5 * normalized_support,
                )

        for edge in complex_.edges:
            triangle_support = 0.0
            adjacent_triangles = 0.0
            u, v = edge
            for triangle in complex_.triangles:
                if u in triangle and v in triangle:
                    adjacent_triangles += 1.0
                    triangle_support += float(complex_.simplex_data.get(triangle, {}).get("support_count", 0.0))
            complex_.simplex_data.setdefault(edge, {}).update(
                {
                    "triangle_support_count": triangle_support,
                    "adjacent_triangle_count": adjacent_triangles,
                }
            )
