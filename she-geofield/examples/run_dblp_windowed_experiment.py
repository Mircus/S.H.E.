from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from she_geofield.dblp.experiments import run_experiment


if __name__ == "__main__":
    outdir = run_experiment(ROOT / "configs" / "dblp_medium.yaml")
    print(f"Wrote DBLP windowed outputs to {outdir}")
