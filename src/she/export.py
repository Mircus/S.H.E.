"""Export analysis results to CSV and JSON.

Provides simple serialisation for :class:`RankedItem`, :class:`BridgeSimplex`,
and :class:`CohesionScore` result lists so that downstream tools (pandas,
spreadsheets, dashboards) can consume SHE outputs without writing custom glue.
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .social import BridgeSimplex, CohesionScore, RankedItem


# -- helpers ------------------------------------------------------------------


def _ranked_item_row(r: RankedItem) -> Dict[str, Any]:
    return {
        "target": ";".join(str(t) for t in r.target),
        "dimension": r.dimension,
        "score": r.score,
        **{f"meta_{k}": v for k, v in r.metadata.items()},
    }


def _bridge_row(b: BridgeSimplex) -> Dict[str, Any]:
    return {
        "members": ";".join(str(m) for m in sorted(b.members, key=str)),
        "dimension": b.dimension,
        "communities_spanned": ";".join(str(c) for c in b.communities_spanned),
        "bridge_score": b.bridge_score,
        **{f"meta_{k}": v for k, v in b.metadata.items()},
    }


def _cohesion_row(c: CohesionScore) -> Dict[str, Any]:
    return {
        "members": ";".join(str(m) for m in sorted(c.members, key=str)),
        "score": c.score,
        **{f"comp_{k}": v for k, v in c.components.items()},
    }


# -- CSV export ---------------------------------------------------------------


def ranked_items_to_csv(
    items: Sequence[RankedItem],
    path: Optional[Union[str, Path]] = None,
) -> str:
    """Serialise ranked items to CSV.  Returns the CSV string; also writes to
    *path* if given."""
    rows = [_ranked_item_row(r) for r in items]
    return _write_csv(rows, path)


def bridges_to_csv(
    bridges: Sequence[BridgeSimplex],
    path: Optional[Union[str, Path]] = None,
) -> str:
    rows = [_bridge_row(b) for b in bridges]
    return _write_csv(rows, path)


def cohesion_to_csv(
    scores: Sequence[CohesionScore],
    path: Optional[Union[str, Path]] = None,
) -> str:
    rows = [_cohesion_row(c) for c in scores]
    return _write_csv(rows, path)


# -- JSON export --------------------------------------------------------------


def ranked_items_to_json(
    items: Sequence[RankedItem],
    path: Optional[Union[str, Path]] = None,
    indent: int = 2,
) -> str:
    rows = [_ranked_item_row(r) for r in items]
    return _write_json(rows, path, indent)


def bridges_to_json(
    bridges: Sequence[BridgeSimplex],
    path: Optional[Union[str, Path]] = None,
    indent: int = 2,
) -> str:
    rows = [_bridge_row(b) for b in bridges]
    return _write_json(rows, path, indent)


def cohesion_to_json(
    scores: Sequence[CohesionScore],
    path: Optional[Union[str, Path]] = None,
    indent: int = 2,
) -> str:
    rows = [_cohesion_row(c) for c in scores]
    return _write_json(rows, path, indent)


# -- internal -----------------------------------------------------------------


def _write_csv(rows: List[Dict[str, Any]], path: Optional[Union[str, Path]]) -> str:
    if not rows:
        return ""
    buf = StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    text = buf.getvalue()
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def _write_json(
    rows: List[Dict[str, Any]], path: Optional[Union[str, Path]], indent: int
) -> str:
    text = json.dumps(rows, indent=indent, default=str)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text
