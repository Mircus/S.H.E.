from dataclasses import asdict, dataclass

from .thresholds import AggregationThresholds, DEFAULT_THRESHOLDS


@dataclass(frozen=True)
class AggregationState:
    closure: float
    persistence: float
    activity: float
    boundary_role: float
    growth_potential: float
    internal_activity: float = 0.0
    boundary_activity: float = 0.0
    adjacent_activity: float = 0.0
    support: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def is_bona_fide_aggregation(
    state: AggregationState,
    *,
    thresholds: AggregationThresholds = DEFAULT_THRESHOLDS,
) -> bool:
    return (
        state.closure >= thresholds.closure
        and state.persistence >= thresholds.persistence
        and state.activity >= thresholds.activity
        and state.support >= thresholds.support
    )
