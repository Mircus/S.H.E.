"""Lightweight temporal windowing for hyperstructures.

Adds time-awareness without redesigning the core.  Relations carry a ``time``
metadata field;  :func:`window` and :func:`rolling_windows` filter or slice
the structure by time range so the same entity set can be analyzed across
successive periods.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import SHEConfig
from .hyperstructure import SHEHyperstructure


def window(
    hs: SHEHyperstructure,
    start: float,
    end: float,
    time_key: str = "time",
    name: Optional[str] = None,
) -> SHEHyperstructure:
    """Return a new hyperstructure containing only relations where
    ``start <= relation[time_key] < end``.

    Entity attributes are copied for every entity that participates in at
    least one relation within the window.  Relations without a *time_key*
    field are **excluded**.
    """
    wname = name or f"{hs.name}_t[{start},{end})"
    ws = SHEHyperstructure(wname, config=hs.config)

    for rec in hs._interaction_log:
        t = rec.get(time_key)
        if t is None:
            continue
        try:
            t = float(t)
        except (ValueError, TypeError):
            continue
        if start <= t < end:
            members = rec["members"]
            # bring entity attrs along
            for m in members:
                if m not in ws._entity_attrs:
                    ws.add_entity(m, **hs.get_entity_attrs(m))
            meta = {k: v for k, v in rec.items() if k not in ("members", "weight", "kind")}
            ws.add_relation(
                members,
                weight=rec.get("weight", 1.0),
                kind=rec.get("kind", "interaction"),
                **meta,
            )
    return ws


def rolling_windows(
    hs: SHEHyperstructure,
    window_size: float,
    step: float,
    time_key: str = "time",
    bounds: Optional[Tuple[float, float]] = None,
) -> List[Tuple[float, float, SHEHyperstructure]]:
    """Produce a sequence of windowed snapshots.

    Returns a list of ``(start, end, windowed_hs)`` tuples.  If *bounds* is
    not given, the global min/max of *time_key* across all relations is used.
    """
    if bounds is not None:
        t_min, t_max = bounds
    else:
        times: List[float] = []
        for rec in hs._interaction_log:
            t = rec.get(time_key)
            if t is not None:
                try:
                    times.append(float(t))
                except (ValueError, TypeError):
                    pass
        if not times:
            return []
        t_min, t_max = min(times), max(times)

    snapshots: List[Tuple[float, float, SHEHyperstructure]] = []
    start = t_min
    while start < t_max:
        end = start + window_size
        ws = window(hs, start, end, time_key=time_key)
        if ws.entities:  # skip empty windows
            snapshots.append((start, end, ws))
        start += step
    return snapshots
