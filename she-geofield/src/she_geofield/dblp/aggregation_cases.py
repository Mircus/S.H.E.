import csv
from pathlib import Path

from .experiments import load_config
from .filters import filter_records
from .parse_xml import load_csv_records, load_dblp_records
from .temporal_flow import (
    build_candidate_aggregations,
    build_temporal_complex_sequence,
    match_aggregations,
    run_window_dynamics,
)
from .windows import build_rolling_windows
from .metrics import aggregation_birth_target, aggregation_reinforcement_target


def _load_complexes_from_config(config_path: str | Path):
    config = load_config(config_path)
    config_path = Path(config["_config_path"])
    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    input_file = Path(config["input_file"])
    if not input_file.is_absolute():
        input_file = (base_dir / input_file).resolve()

    if input_file.suffix == ".csv":
        records = load_csv_records(input_file)
    else:
        records = load_dblp_records(
            input_file,
            publication_types=set(config.get("publication_types", ["article", "inproceedings"])),
        )
    filtered = filter_records(
        records,
        start_year=int(config.get("start_year")) if "start_year" in config else None,
        end_year=int(config.get("end_year")) if "end_year" in config else None,
        min_team_size=int(config.get("min_team_size")) if "min_team_size" in config else None,
        max_team_size=int(config.get("max_team_size")) if "max_team_size" in config else None,
    )
    windows = build_rolling_windows(
        filtered,
        width=int(config.get("window_width", 3)),
        stride=int(config.get("window_stride", 1)),
        start_year=int(config.get("start_year")) if "start_year" in config else None,
        end_year=int(config.get("end_year")) if "end_year" in config else None,
    )
    complexes = build_temporal_complex_sequence(
        windows,
        weight_mode=str(config.get("weight_mode", "contained")),
        max_simplex_size=int(config.get("max_simplex_size", 3)),
    )
    return config, complexes


def export_case_studies(
    config_path: str | Path,
    *,
    event_type: str,
    output_csv: str | Path,
    max_cases: int = 3,
) -> Path:
    config, complexes = _load_complexes_from_config(config_path)
    rows: list[dict[str, object]] = []
    horizon = int(config.get("prediction_horizon", 1))

    for idx, complex_ in enumerate(complexes[:-horizon or None]):
        field_scores = run_window_dynamics(
            complex_,
            internal_steps=int(config.get("internal_steps", 2)),
            dt=float(config.get("dt", 0.5)),
            eta=float(config.get("eta", 0.1)),
            lam=float(config.get("lam", 0.3)),
            mu=float(config.get("mu", 0.2)),
            field_mode=str(config.get("field_mode", "support_count")),
            field_location="edges",
        )["evolving_geometry"]
        current = build_candidate_aggregations(complex_, field_scores=field_scores)

        future_complex = complexes[idx + horizon]
        future_field_scores = run_window_dynamics(
            future_complex,
            internal_steps=int(config.get("internal_steps", 2)),
            dt=float(config.get("dt", 0.5)),
            eta=float(config.get("eta", 0.1)),
            lam=float(config.get("lam", 0.3)),
            mu=float(config.get("mu", 0.2)),
            field_mode=str(config.get("field_mode", "support_count")),
            field_location="edges",
        )["evolving_geometry"]
        future = build_candidate_aggregations(future_complex, field_scores=future_field_scores)
        matches = match_aggregations(current, future)

        if event_type == "birth":
            targets = aggregation_birth_target(current, matches)
        elif event_type == "reinforcement":
            targets = aggregation_reinforcement_target(current, matches)
        else:
            raise ValueError(f"unsupported event type: {event_type}")

        positives = [snapshot for snapshot in current if float(targets.get(snapshot.aggregation_id, 0.0)) > 0.0]
        negatives = [snapshot for snapshot in current if float(targets.get(snapshot.aggregation_id, 0.0)) == 0.0]

        wrote_any = False
        for label, collection in [("positive", positives), ("failed", negatives)]:
            for snapshot in collection[:max_cases]:
                future_snapshot = matches.get(snapshot.aggregation_id)
                row = {
                    "window": snapshot.window_label,
                    "event_type": event_type,
                    "case_label": label,
                    "aggregation_id": snapshot.aggregation_id,
                    "unit_type": snapshot.unit_type,
                    "members": "-".join(snapshot.members),
                    "target": float(targets.get(snapshot.aggregation_id, 0.0)),
                }
                for prefix, state in [("before", snapshot.state), ("after", future_snapshot.state if future_snapshot else {})]:
                    for key in ["closure", "persistence", "activity", "boundary_role", "growth_potential"]:
                        row[f"{prefix}_{key}"] = float(state.get(key, 0.0))
                rows.append(row)
                wrote_any = True
        if wrote_any:
            break

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["event_type"])
        writer.writeheader()
        writer.writerows(rows)
    return output_path
