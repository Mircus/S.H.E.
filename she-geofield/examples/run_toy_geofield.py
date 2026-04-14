import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from she_geofield.experiment import run_experiment

if __name__ == "__main__":
    outdir = run_experiment(ROOT / "out")
    print(f"Wrote reproducible outputs to {outdir}")
