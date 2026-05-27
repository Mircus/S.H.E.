def boundary_role(internal_edges: float, boundary_edges: float) -> float:
    total = internal_edges + boundary_edges
    if total <= 0.0:
        return 0.0
    return boundary_edges / total
