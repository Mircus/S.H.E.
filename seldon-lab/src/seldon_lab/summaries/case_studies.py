from pathlib import Path

from she_geofield.dblp.aggregation_cases import export_case_studies


def export_event_cases(
    config_path: str | Path,
    *,
    event_type: str,
    output_csv: str | Path,
    max_cases: int = 3,
) -> Path:
    return export_case_studies(
        config_path,
        event_type=event_type,
        output_csv=output_csv,
        max_cases=max_cases,
    )
