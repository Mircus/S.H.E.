from dataclasses import dataclass
from pathlib import Path

from .loaders import load_simple_config, resolve_dataset_path


@dataclass(frozen=True)
class DblpSliceConfig:
    venue: str
    start_year: int
    end_year: int
    min_team_size: int = 2
    max_team_size: int = 6


@dataclass(frozen=True)
class SeldonExperimentConfig:
    venue: str
    event_type: str
    input_config: Path
    config_path: Path


def load_seldon_experiment_config(path: str | Path) -> SeldonExperimentConfig:
    config = load_simple_config(path)
    config_path = Path(config["_config_path"])
    input_config = resolve_dataset_path(config_path.parent, str(config["input_config"]))
    return SeldonExperimentConfig(
        venue=str(config["venue"]),
        event_type=str(config["event_type"]),
        input_config=input_config,
        config_path=config_path,
    )
