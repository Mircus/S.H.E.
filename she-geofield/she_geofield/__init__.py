"""In-place import shim so `python -m she_geofield...` works without install."""

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SRC_PACKAGE = _ROOT / "src" / "she_geofield"

__path__ = [str(Path(__file__).resolve().parent), str(_SRC_PACKAGE)]
