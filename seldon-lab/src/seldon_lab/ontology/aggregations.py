from dataclasses import dataclass


@dataclass(frozen=True)
class Aggregation:
    aggregation_id: str
    unit_type: str
    members: tuple[str, ...]
    anchor: tuple[str, ...]
    window_label: str


def is_simplex_aggregation(members: tuple[str, ...]) -> bool:
    return len(members) >= 3
