from pathlib import Path


def _parse_scalar(value: str):
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_simple_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    config: dict[str, object] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, value = stripped.split(":", 1)
        config[key.strip()] = _parse_scalar(value)
    config["_config_path"] = config_path.resolve()
    return config


def resolve_dataset_path(base: str | Path, value: str | Path) -> Path:
    base_path = Path(base)
    value_path = Path(value)
    if value_path.is_absolute():
        return value_path
    return (base_path / value_path).resolve()
