"""Tests for CSV / JSON-Lines ingestion."""

import json
import os
import tempfile

from she import SHEHyperstructure


def test_from_csv(tmp_path):
    csv_file = tmp_path / "interactions.csv"
    csv_file.write_text(
        "users,weight,kind,topic\n"
        "alice;bob,1.0,reply,climate\n"
        "alice;bob;carol,2.5,co_amp,climate\n"
        "dave;eve,0.5,mention,\n"
    )
    hs = SHEHyperstructure.from_csv(csv_file)
    assert len(hs.entities) == 5
    assert len(hs.relations) == 3
    rel = hs.get_relation_attrs(["alice", "bob", "carol"])
    assert rel["kind"] == "co_amp"


def test_from_csv_custom_sep(tmp_path):
    csv_file = tmp_path / "pipe.csv"
    csv_file.write_text(
        "members,weight\n"
        "a|b,1.0\n"
        "a|b|c,2.0\n"
    )
    hs = SHEHyperstructure.from_csv(csv_file, members_col="members", members_sep="|")
    assert len(hs.entities) == 3


def test_from_jsonl(tmp_path):
    jsonl_file = tmp_path / "interactions.jsonl"
    records = [
        {"users": ["alice", "bob"], "weight": 1.0, "kind": "reply"},
        {"users": ["alice", "bob", "carol"], "weight": 2.0, "kind": "co_amp", "topic": "X"},
    ]
    jsonl_file.write_text("\n".join(json.dumps(r) for r in records))
    hs = SHEHyperstructure.from_jsonl(jsonl_file)
    assert len(hs.entities) == 3
    assert len(hs.relations) == 2
