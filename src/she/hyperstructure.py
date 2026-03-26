"""Decorated higher-order relational structure.

An :class:`SHEHyperstructure` represents entities (people, accounts, agents)
connected by weighted, typed relations of arbitrary order.  Pairwise edges,
triads, and larger co-engagement groups are all first-class objects.

The simplicial complex is the *computational substrate* -- this class adds the
*modeling vocabulary*: entity attributes, relation kinds, topics, and
arbitrary per-simplex metadata that make the structure interpretable as a
social or relational object rather than a raw topological one.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .complex import SHESimplicialComplex
from .config import SHEConfig

logger = logging.getLogger(__name__)


@dataclass
class RelationRecord:
    """Lightweight record describing one relation to be ingested."""

    members: List[Any]
    weight: float = 1.0
    kind: str = "interaction"
    metadata: Dict[str, Any] = field(default_factory=dict)


class SHEHyperstructure:
    """A decorated, weighted higher-order relational structure.

    Use this when you care about *who* the vertices are and *what kind* of
    relation each simplex encodes -- not just the combinatorial skeleton.

    Parameters
    ----------
    name : str
        Human-readable label for the structure.
    config : SHEConfig, optional
        Passed through to the underlying simplicial complex.

    Examples
    --------
    >>> hs = SHEHyperstructure("demo")
    >>> hs.add_entity("alice", role="journalist", community="A")
    >>> hs.add_entity("bob", role="amplifier", community="A")
    >>> hs.add_relation(["alice", "bob"], weight=0.8, kind="reply")
    >>> hs.add_relation(["alice", "bob", "carol"], weight=1.4,
    ...                 kind="co_amplification", topic="climate")
    >>> print(hs.summary())
    """

    def __init__(self, name: str = "hyperstructure", config: Optional[SHEConfig] = None):
        self.name = name
        self.config = config or SHEConfig()
        self._sc = SHESimplicialComplex(name, self.config)

        # domain-level stores (keyed by entity id / frozenset of members)
        self._entity_attrs: Dict[Any, Dict[str, Any]] = {}
        self._relation_attrs: Dict[frozenset, Dict[str, Any]] = {}
        # every add_relation call is logged here for temporal filtering
        self._interaction_log: List[Dict[str, Any]] = []

    # -- entity management ------------------------------------------------

    def add_entity(self, entity_id: Any, **attrs) -> None:
        """Register an entity (0-simplex) with arbitrary attributes.

        Calling this multiple times for the same *entity_id* merges attributes.
        """
        if entity_id in self._entity_attrs:
            self._entity_attrs[entity_id].update(attrs)
        else:
            self._entity_attrs[entity_id] = dict(attrs)
            self._sc.add_node(entity_id, **attrs)

    def get_entity_attrs(self, entity_id: Any) -> Dict[str, Any]:
        """Return stored attributes for *entity_id*, or empty dict."""
        return dict(self._entity_attrs.get(entity_id, {}))

    @property
    def entities(self) -> List[Any]:
        """All registered entity ids."""
        return list(self._entity_attrs)

    # -- relation management ----------------------------------------------

    def add_relation(
        self,
        members: Sequence[Any],
        weight: float = 1.0,
        kind: str = "interaction",
        **metadata,
    ) -> None:
        """Add a weighted, typed relation among *members*.

        Any member not yet registered is added as an entity with no extra
        attributes.  The relation is stored both as a simplex in the
        underlying complex and as a decorated record accessible via
        :meth:`get_relation_attrs`.
        """
        if len(members) < 2:
            raise ValueError("A relation requires at least two members.")

        # ensure entities exist
        for m in members:
            if m not in self._entity_attrs:
                self.add_entity(m)

        key = frozenset(members)
        attrs: Dict[str, Any] = {"weight": weight, "kind": kind, **metadata}

        if key in self._relation_attrs:
            # accumulate weight, keep latest metadata
            self._relation_attrs[key]["weight"] = (
                self._relation_attrs[key].get("weight", 0.0) + weight
            )
            self._relation_attrs[key].update({k: v for k, v in attrs.items() if k != "weight"})
        else:
            self._relation_attrs[key] = attrs

        # log the individual interaction (for temporal filtering)
        self._interaction_log.append(
            {"members": sorted(members, key=str), "weight": weight, "kind": kind, **metadata}
        )

        # push into simplicial substrate
        self._sc.add_simplex(sorted(members, key=str), weight=self._relation_attrs[key]["weight"])

    def get_relation_attrs(self, members: Iterable[Any]) -> Dict[str, Any]:
        """Return stored decoration for a relation, or empty dict."""
        return dict(self._relation_attrs.get(frozenset(members), {}))

    @property
    def relations(self) -> List[frozenset]:
        """All registered relation keys (frozensets of member ids)."""
        return list(self._relation_attrs)

    # -- bulk ingestion ---------------------------------------------------

    @classmethod
    def from_records(
        cls,
        records: Iterable[Dict[str, Any]],
        name: str = "from_records",
        config: Optional[SHEConfig] = None,
        members_key: str = "users",
        weight_key: str = "weight",
        kind_key: str = "kind",
    ) -> "SHEHyperstructure":
        """Build a hyperstructure from simple interaction records.

        Each record is a dict with at least a *members_key* field containing
        a list of entity ids.  Optional keys: *weight_key*, *kind_key*, and
        any other fields stored as relation metadata.

        Example input::

            [
                {"users": ["alice", "bob"], "weight": 1.0, "kind": "reply"},
                {"users": ["alice", "bob", "carol"], "weight": 2.2,
                 "kind": "co_amplification", "topic": "climate"},
            ]
        """
        hs = cls(name=name, config=config)
        for rec in records:
            members = rec.get(members_key)
            if not members or len(members) < 2:
                continue
            weight = rec.get(weight_key, 1.0)
            kind = rec.get(kind_key, "interaction")
            meta = {k: v for k, v in rec.items() if k not in (members_key, weight_key, kind_key)}
            hs.add_relation(members, weight=weight, kind=kind, **meta)
        return hs

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        name: str = "from_csv",
        config: Optional[SHEConfig] = None,
        members_col: str = "users",
        weight_col: str = "weight",
        kind_col: str = "kind",
        members_sep: str = ";",
    ) -> "SHEHyperstructure":
        """Build a hyperstructure from a CSV file.

        The *members_col* column should contain entity ids separated by
        *members_sep* (default ``";"``) — e.g. ``"alice;bob;carol"``.
        All other columns are stored as relation metadata.
        """
        records: List[Dict[str, Any]] = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw_members = row.get(members_col, "")
                members = [m.strip() for m in raw_members.split(members_sep) if m.strip()]
                if len(members) < 2:
                    continue
                rec: Dict[str, Any] = {"users": members}
                if weight_col in row:
                    try:
                        rec["weight"] = float(row[weight_col])
                    except (ValueError, TypeError):
                        rec["weight"] = 1.0
                if kind_col in row:
                    rec["kind"] = row[kind_col]
                # store remaining columns as metadata
                for k, v in row.items():
                    if k not in (members_col, weight_col, kind_col):
                        rec[k] = v
                records.append(rec)
        return cls.from_records(records, name=name, config=config)

    @classmethod
    def from_jsonl(
        cls,
        path: Union[str, Path],
        name: str = "from_jsonl",
        config: Optional[SHEConfig] = None,
        members_key: str = "users",
        weight_key: str = "weight",
        kind_key: str = "kind",
    ) -> "SHEHyperstructure":
        """Build a hyperstructure from a JSON-Lines file.

        Each line must be a JSON object with at least a *members_key* field
        containing a list of entity ids.
        """
        records: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return cls.from_records(
            records, name=name, config=config,
            members_key=members_key, weight_key=weight_key, kind_key=kind_key,
        )

    # -- access to computational substrate --------------------------------

    @property
    def complex(self) -> SHESimplicialComplex:
        """The underlying simplicial complex used for computation."""
        return self._sc

    # -- inspection -------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Counts by dimension plus entity/relation totals."""
        dim_counts: Dict[int, int] = {}
        max_dim = self._sc.complex.dim if len(self._entity_attrs) > 0 else -1
        for d in range(max_dim + 1):
            dim_counts[d] = len(self._sc.get_simplex_list(d))
        return {
            "name": self.name,
            "entities": len(self._entity_attrs),
            "relations": len(self._relation_attrs),
            "simplices_by_dimension": dim_counts,
            "max_dimension": max_dim,
        }

    def __repr__(self) -> str:
        s = self.summary()
        dim_str = ", ".join(f"dim{d}={c}" for d, c in s["simplices_by_dimension"].items())
        return (
            f"SHEHyperstructure('{self.name}', "
            f"entities={s['entities']}, relations={s['relations']}, {dim_str})"
        )
