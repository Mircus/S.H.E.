"""Tests for export module."""

import json
import tempfile
from pathlib import Path

from she import (
    SHEHyperstructure,
    SHEConfig,
    rank_entity_diffusers,
    find_bridge_simplices,
    group_cohesion,
    ranked_items_to_csv,
    ranked_items_to_json,
    bridges_to_csv,
    bridges_to_json,
    cohesion_to_csv,
    cohesion_to_json,
)


def _make_hs():
    config = SHEConfig(max_dimension=2)
    hs = SHEHyperstructure("export_test", config=config)
    for name in ["a", "b", "c"]:
        hs.add_entity(name, community="X")
    hs.add_entity("d", community="Y")
    hs.add_relation(["a", "b"], weight=1.0, kind="chat")
    hs.add_relation(["b", "c"], weight=1.0, kind="chat")
    hs.add_relation(["c", "d"], weight=1.0, kind="chat")
    hs.add_relation(["a", "b", "c"], weight=2.0, kind="group")
    return hs


def test_ranked_items_csv_roundtrip():
    hs = _make_hs()
    items = rank_entity_diffusers(hs, top_k=4)
    csv_text = ranked_items_to_csv(items)
    assert "target" in csv_text
    assert "score" in csv_text
    lines = csv_text.strip().split("\n")
    assert len(lines) == len(items) + 1  # header + rows


def test_ranked_items_json():
    hs = _make_hs()
    items = rank_entity_diffusers(hs, top_k=4)
    text = ranked_items_to_json(items)
    data = json.loads(text)
    assert len(data) == len(items)
    assert all("score" in row for row in data)


def test_bridges_csv():
    hs = _make_hs()
    bridges = find_bridge_simplices(hs)
    csv_text = bridges_to_csv(bridges)
    assert "bridge_score" in csv_text


def test_bridges_json():
    hs = _make_hs()
    bridges = find_bridge_simplices(hs)
    text = bridges_to_json(bridges)
    data = json.loads(text)
    assert all("bridge_score" in row for row in data)


def test_cohesion_csv():
    hs = _make_hs()
    scores = [group_cohesion(hs, ["a", "b", "c"])]
    csv_text = cohesion_to_csv(scores)
    assert "score" in csv_text
    assert "comp_relation_weight" in csv_text


def test_write_to_file():
    hs = _make_hs()
    items = rank_entity_diffusers(hs, top_k=2)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.csv"
        ranked_items_to_csv(items, path=p)
        assert p.exists()
        assert p.read_text().startswith("target")
