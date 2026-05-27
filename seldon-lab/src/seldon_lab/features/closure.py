def higher_order_closure_ratio(closed_triangles: int, possible_triangles: int) -> float:
    if possible_triangles <= 0:
        return 0.0
    return closed_triangles / possible_triangles
