import numpy as np

from .records import DblpSimplicialComplex


def collaboration_activation_field(
    complex_: DblpSimplicialComplex,
    *,
    location: str = "edges",
    mode: str = "support_count",
    decay: float = 1.0,
    normalize_by_size: bool = True,
) -> np.ndarray:
    """Initialize a collaboration-activation field on edges or triangles."""
    if location not in {"edges", "triangles"}:
        raise ValueError("location must be 'edges' or 'triangles'")

    simplices = complex_.edges if location == "edges" else complex_.triangles
    values: list[float] = []
    for simplex in simplices:
        data = complex_.simplex_data.get(simplex, {})
        if mode == "support_count":
            value = float(data.get("support_count", 0.0))
        elif mode == "decayed_support":
            value = decay * float(data.get("support_count", 0.0))
        elif mode == "persistence":
            value = float(data.get("persistence", 0.0))
        else:
            raise ValueError(f"unsupported field mode: {mode}")

        if normalize_by_size:
            value /= max(len(simplex), 1)
        values.append(value)
    return np.asarray(values, dtype=float)
