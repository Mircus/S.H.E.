from pathlib import Path
import shutil


def copy_summary_plot(source: str | Path, *, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination
