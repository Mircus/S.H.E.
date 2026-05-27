from pathlib import Path

from she_geofield.dblp.aggregation_summary import build_cross_event_summary


def build_summary(event_outputs: dict[str, dict[str, str | Path]], *, output_dir: str | Path) -> Path:
    return build_cross_event_summary(event_outputs, output_dir=output_dir)
