"""Tests for the SHEHyperstructure modeling layer."""

from she import SHEHyperstructure


def _make_hs():
    hs = SHEHyperstructure("test")
    hs.add_entity("alice", role="journalist", community="A")
    hs.add_entity("bob", role="amplifier", community="A")
    hs.add_entity("carol", role="bridge", community="B")
    hs.add_relation(["alice", "bob"], weight=0.8, kind="reply")
    hs.add_relation(["alice", "bob", "carol"], weight=1.4, kind="co_amplification", topic="X")
    return hs


def test_entity_insertion():
    hs = _make_hs()
    assert set(hs.entities) == {"alice", "bob", "carol"}


def test_entity_attrs_retained():
    hs = _make_hs()
    assert hs.get_entity_attrs("alice")["role"] == "journalist"
    assert hs.get_entity_attrs("carol")["community"] == "B"


def test_relation_insertion():
    hs = _make_hs()
    assert len(hs.relations) == 2


def test_relation_metadata_retained():
    hs = _make_hs()
    rel = hs.get_relation_attrs(["alice", "bob", "carol"])
    assert rel["kind"] == "co_amplification"
    assert rel["topic"] == "X"
    assert rel["weight"] == 1.4


def test_dimension_counts():
    hs = _make_hs()
    s = hs.summary()
    assert s["entities"] == 3
    assert s["relations"] == 2
    assert s["simplices_by_dimension"][0] == 3  # nodes
    assert s["simplices_by_dimension"].get(2, 0) == 1  # the triad


def test_complex_accessible():
    hs = _make_hs()
    sc = hs.complex
    assert sc.complex.dim >= 1


def test_from_records():
    records = [
        {"users": ["a", "b"], "weight": 1.0, "kind": "reply"},
        {"users": ["a", "b", "c"], "weight": 2.0, "kind": "co_amp"},
    ]
    hs = SHEHyperstructure.from_records(records)
    assert len(hs.entities) == 3
    assert len(hs.relations) == 2


def test_repr():
    hs = _make_hs()
    r = repr(hs)
    assert "SHEHyperstructure" in r
    assert "test" in r
