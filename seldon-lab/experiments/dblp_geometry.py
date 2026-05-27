"""Thin entry-point wrapper for the seldon-lab DBLP static-geometry experiment.

Delegates immediately to the installed module so this script can be invoked
as::

    python seldon-lab/experiments/dblp_geometry.py --config <config>

from the repo root, mirroring the documented ``python -m`` invocation in the
README without requiring the caller to remember the full module path.

Stderr/warning suppression: Python warnings are suppressed for matplotlib font
manager noise by setting MPLCONFIGDIR to a pre-cached directory (done inside
the main module via os.environ.setdefault).  Any remaining runtime warnings
from scipy/numpy are filtered here so that experiment runs stay clean.
"""
import sys
import warnings

# Suppress noisy but harmless warnings that appear during normal runs.
# These are filtered at source rather than redirected, so they do not appear
# in either stdout or stderr.
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure the seldon_lab src directory is on the path when invoked as a script.
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "seldon-lab" / "src"
_GEOFIELD_SRC = _REPO_ROOT / "she-geofield" / "src"
for _path in [str(_SRC), str(_GEOFIELD_SRC)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from seldon_lab.experiments.dblp_geometry import main  # noqa: E402

if __name__ == "__main__":
    main()
