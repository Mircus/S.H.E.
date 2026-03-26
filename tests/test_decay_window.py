"""Tests for decay-weighted temporal accumulation."""

import math

from she import SHEHyperstructure, SHEConfig, decay_window


def _make_temporal_hs():
    config = SHEConfig(max_dimension=2)
    hs = SHEHyperstructure("decay_test", config=config)
    hs.add_entity("a", community="X")
    hs.add_entity("b", community="X")
    hs.add_entity("c", community="Y")

    hs.add_relation(["a", "b"], weight=1.0, kind="chat", time=1.0)
    hs.add_relation(["a", "b"], weight=1.0, kind="chat", time=3.0)
    hs.add_relation(["b", "c"], weight=2.0, kind="chat", time=5.0)
    return hs


def test_decay_recent_dominates():
    hs = _make_temporal_hs()
    ws = decay_window(hs, reference_time=5.0, half_life=1.0)
    # The relation at t=5 (age=0) should have full weight
    # The relation at t=3 (age=2) should be decayed to 0.25
    # The relation at t=1 (age=4) should be decayed to 0.0625
    assert len(ws.entities) >= 2
    bc_attrs = ws.get_relation_attrs(["b", "c"])
    assert bc_attrs["weight"] > 1.5  # ~2.0 with negligible decay


def test_decay_drops_old_interactions():
    hs = _make_temporal_hs()
    # Very short half-life: only most recent should survive
    ws = decay_window(hs, reference_time=5.0, half_life=0.1, cutoff=0.01)
    # t=1 and t=3 are ages 4.0 and 2.0 with half_life=0.1
    # decay at age 2.0: 2^(-20) ≈ 1e-6, well below cutoff
    ab_attrs = ws.get_relation_attrs(["a", "b"])
    assert ab_attrs == {} or ab_attrs.get("weight", 0) < 0.01


def test_decay_preserves_entity_attrs():
    hs = _make_temporal_hs()
    ws = decay_window(hs, reference_time=5.0, half_life=2.0)
    assert ws.get_entity_attrs("a").get("community") == "X"
    assert ws.get_entity_attrs("c").get("community") == "Y"


def test_decay_future_excluded():
    hs = _make_temporal_hs()
    ws = decay_window(hs, reference_time=2.0, half_life=1.0)
    # Only t=1 relation should be included (t=3 and t=5 are future)
    assert "c" not in ws.entities  # c only appears at t=5
