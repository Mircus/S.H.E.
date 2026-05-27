"""Additional regression and boundary-condition tests (Critic R4).

Three areas:
1. CV with all-zero input — must not crash, must return 0.0.
2. Config parsing with a colon-containing value — documents the known
   limitation that keys must not contain colons.
3. Track-matching boundary — Jaccard < 0.5 must not match at threshold 0.5.
"""

import math
from statistics import mean

import pytest

pytest.importorskip("she_geofield.dblp.experiments")

from seldon_lab.experiments.dblp_geometry import _cv
from she_geofield.dblp.experiments import load_config
from she_geofield.dblp.temporal_flow import match_aggregations
from she_geofield.dblp.records import AggregationSnapshot


# ---------------------------------------------------------------------------
# 1. CV with all-zero / empty input
# ---------------------------------------------------------------------------


def test_cv_all_zeros_returns_zero_without_crashing():
    """_cv([0.0, 0.0, 0.0]) must return 0.0, not raise or produce NaN.

    Zero-weight edges are excluded from CV computation (they represent
    absent collaborations, not weak ones).  When every value is zero the
    positive sub-list is empty and CV is defined as 0.0.
    """
    assert _cv([0.0, 0.0, 0.0]) == 0.0


def test_cv_empty_list_returns_zero():
    """_cv([]) must return 0.0 without raising."""
    assert _cv([]) == 0.0


def test_cv_mixed_zeros_uses_only_positives():
    """Zeros are ignored; result equals the CV of the positive subset."""
    positives = [2.0, 4.0, 6.0]
    mu = mean(positives)
    var = sum((v - mu) ** 2 for v in positives) / len(positives)
    expected = math.sqrt(var) / mu

    result = _cv([0.0, 2.0, 0.0, 4.0, 6.0])
    assert abs(result - expected) < 1e-9


# ---------------------------------------------------------------------------
# 2. Config parsing — colon-in-key known limitation
# ---------------------------------------------------------------------------


def test_config_url_value_is_parsed_correctly(tmp_path):
    """A value that contains a colon (e.g. a URL) is preserved intact.

    load_config() splits on the *first* colon only, so:
        venue: http://example.com
    yields key="venue", value="http://example.com".
    """
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("venue: http://example.com\n")
    config = load_config(cfg_file)
    assert "venue" in config
    assert "http://example.com" in str(config["venue"])


def test_config_colon_in_key_known_limitation(tmp_path):
    """Document the known limitation: a key that contains a colon is mis-parsed.

    load_config() uses ``line.split(':', 1)``, so a line like
        a:b: 42
    produces key="a", value="b: 42" — the intended key "a:b" is never stored.
    Config keys must not contain colons.  This test pins the current behaviour
    so any future fix will be visible.
    """
    cfg_file = tmp_path / "colon_key.yaml"
    cfg_file.write_text("a:b: 42\n")
    config = load_config(cfg_file)
    # Current (broken) behaviour: key "a" is extracted, not "a:b".
    assert "a" in config
    assert "a:b" not in config


# ---------------------------------------------------------------------------
# 3. Track-matching boundary — Jaccard threshold at 0.5
# ---------------------------------------------------------------------------


def _snapshot(agg_id, members, unit_type="edge", window="W1"):
    members_t = tuple(members)
    return AggregationSnapshot(
        aggregation_id=agg_id,
        unit_type=unit_type,
        window_label=window,
        members=members_t,
        anchor_simplex=members_t,
        state={
            "support": 1.0,
            "persistence": 0.5,
            "activity": 1.0,
            "boundary_activity": 0.0,
            "growth_potential": 0.0,
            "boundary_role": 0.0,
            "adjacent_activity": 0.0,
        },
    )


def test_jaccard_below_threshold_does_not_match():
    """Jaccard = 0.40 is strictly below 0.5 and must not produce a match.

    current = {a, b, c}     (3 members)
    future  = {b, c, d, e}  (4 members)
    intersection = {b, c} = 2
    union        = {a, b, c, d, e} = 5
    Jaccard      = 2/5 = 0.40 < 0.5
    """
    current = [_snapshot("snap_c", ("a", "b", "c"))]
    future = [_snapshot("snap_f", ("b", "c", "d", "e"))]

    matches = match_aggregations(current, future, min_overlap=0.5)
    assert matches["snap_c"] is None


def test_jaccard_at_threshold_does_match():
    """Jaccard = 0.5 exactly meets the threshold and must produce a match.

    current = {a, b}        (2 members)
    future  = {a, b, c, d}  (4 members)
    intersection = {a, b} = 2
    union        = {a, b, c, d} = 4
    Jaccard      = 2/4 = 0.5 >= 0.5
    """
    current = [_snapshot("snap_c", ("a", "b"))]
    future = [_snapshot("snap_f", ("a", "b", "c", "d"))]

    matches = match_aggregations(current, future, min_overlap=0.5)
    assert matches["snap_c"] is not None


def test_jaccard_near_miss_just_below_threshold():
    """Jaccard ≈ 0.495 (just below 0.5) must not produce a match.

    Construction: 49 shared members, 50 exclusive to current, 0 exclusive to
    future.
      intersection = 49
      union        = 49 + 50 = 99
      Jaccard      = 49/99 ≈ 0.4949 < 0.5
    """
    shared = [f"s{i}" for i in range(49)]
    only_current = [f"c{i}" for i in range(50)]

    current_members = tuple(shared + only_current)   # 99 members
    future_members = tuple(shared)                    # 49 members

    # Verify the construction.
    overlap = len(set(current_members) & set(future_members)) / len(
        set(current_members) | set(future_members)
    )
    assert overlap < 0.5, f"test construction error: Jaccard={overlap:.4f}"

    current = [_snapshot("snap_c", current_members)]
    future = [_snapshot("snap_f", future_members)]

    matches = match_aggregations(current, future, min_overlap=0.5)
    assert matches["snap_c"] is None, (
        f"Jaccard={overlap:.4f} < 0.5 must not match at threshold=0.5"
    )
