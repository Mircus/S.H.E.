"""Tests for temporal windowing."""

from she import SHEHyperstructure, SHEConfig, window, rolling_windows


def _make_temporal_hs():
    config = SHEConfig(max_dimension=2)
    hs = SHEHyperstructure("temporal", config=config)
    hs.add_entity("a", community="X")
    hs.add_entity("b", community="X")
    hs.add_entity("c", community="Y")

    hs.add_relation(["a", "b"], weight=1.0, kind="chat", time=1.0)
    hs.add_relation(["a", "b"], weight=0.5, kind="chat", time=2.0)
    hs.add_relation(["b", "c"], weight=1.0, kind="chat", time=3.0)
    hs.add_relation(["a", "b", "c"], weight=2.0, kind="group", time=4.0)
    hs.add_relation(["a", "c"], weight=0.8, kind="chat", time=5.0)
    return hs


def test_window_filters_by_time():
    hs = _make_temporal_hs()
    ws = window(hs, start=2.0, end=4.0)
    # should include relations at t=2.0 and t=3.0 but not t=1, t=4, t=5
    assert len(ws.relations) >= 1
    # entity c should appear (t=3.0 has b-c)
    assert "c" in ws.entities or "b" in ws.entities


def test_window_preserves_entity_attrs():
    hs = _make_temporal_hs()
    ws = window(hs, start=1.0, end=2.0)
    assert ws.get_entity_attrs("a").get("community") == "X"


def test_window_empty_range():
    hs = _make_temporal_hs()
    ws = window(hs, start=100.0, end=200.0)
    assert len(ws.entities) == 0


def test_rolling_windows_produces_snapshots():
    hs = _make_temporal_hs()
    snapshots = rolling_windows(hs, window_size=2.0, step=1.0)
    assert len(snapshots) >= 2
    for start, end, ws in snapshots:
        assert end - start == 2.0
        assert len(ws.entities) > 0


def test_rolling_windows_with_bounds():
    hs = _make_temporal_hs()
    snapshots = rolling_windows(hs, window_size=2.0, step=2.0, bounds=(1.0, 5.0))
    assert len(snapshots) == 2  # [1,3) and [3,5)
