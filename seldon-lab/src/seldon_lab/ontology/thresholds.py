from dataclasses import dataclass


@dataclass(frozen=True)
class AggregationThresholds:
    closure: float = 0.2
    persistence: float = 0.5
    activity: float = 1.0
    support: float = 1.0


DEFAULT_THRESHOLDS = AggregationThresholds()
