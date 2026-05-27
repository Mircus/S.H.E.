from dataclasses import dataclass


@dataclass(frozen=True)
class AggregationEvent:
    event_type: str
    aggregation_id: str
    window_label: str
    target_value: float
