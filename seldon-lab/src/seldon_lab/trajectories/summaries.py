from .tracks import AggregationTrack


def summarize_track(track: AggregationTrack) -> dict[str, object]:
    before = track.states_by_window[0]
    after = track.states_by_window[-1]
    return {
        "aggregation_id": track.aggregation_id,
        "unit_type": track.unit_type,
        "start_window": track.windows[0],
        "end_window": track.windows[-1],
        "closure_gain": after.get("closure", 0.0) - before.get("closure", 0.0),
        "persistence_gain": after.get("persistence", 0.0) - before.get("persistence", 0.0),
        "activity_gain": after.get("activity", 0.0) - before.get("activity", 0.0),
        "boundary_shift": after.get("boundary_role", 0.0) - before.get("boundary_role", 0.0),
    }
