"""Tests for the social analysis layer."""

from she import (
    SHEHyperstructure,
    SHEConfig,
    rank_diffusers,
    rank_entity_diffusers,
    find_bridge_simplices,
    group_cohesion,
    rank_influencers,
)


def _two_community_hs():
    """Two communities linked by one cross-community triad."""
    config = SHEConfig(max_dimension=2, spectral_k=4)
    hs = SHEHyperstructure("two_comm", config=config)

    # Community A
    for name in ["a1", "a2", "a3"]:
        hs.add_entity(name, community="A")
    hs.add_relation(["a1", "a2"], weight=1.0, kind="chat")
    hs.add_relation(["a2", "a3"], weight=1.0, kind="chat")
    hs.add_relation(["a1", "a3"], weight=1.0, kind="chat")
    hs.add_relation(["a1", "a2", "a3"], weight=1.5, kind="group_chat")

    # Community B
    for name in ["b1", "b2", "b3"]:
        hs.add_entity(name, community="B")
    hs.add_relation(["b1", "b2"], weight=1.0, kind="chat")
    hs.add_relation(["b2", "b3"], weight=1.0, kind="chat")
    hs.add_relation(["b1", "b3"], weight=1.0, kind="chat")
    hs.add_relation(["b1", "b2", "b3"], weight=1.5, kind="group_chat")

    # Bridge triad: a3 (A), b1 (B), b2 (B)
    hs.add_relation(["a3", "b1"], weight=2.0, kind="bridge")
    hs.add_relation(["a3", "b2"], weight=1.8, kind="bridge")
    # b1-b2 already exists, just add the triad
    hs.add_relation(["a3", "b1", "b2"], weight=3.0, kind="bridge_triad")

    return hs


def test_rank_diffusers_non_empty():
    hs = _two_community_hs()
    ranked = rank_diffusers(hs, top_k=10)
    assert len(ranked) > 0
    assert all(hasattr(r, "score") for r in ranked)


def test_rank_entity_diffusers_non_empty():
    hs = _two_community_hs()
    ranked = rank_entity_diffusers(hs, top_k=5)
    assert len(ranked) > 0
    assert all(r.dimension == 0 for r in ranked)


def test_find_bridge_simplices_finds_bridge():
    hs = _two_community_hs()
    bridges = find_bridge_simplices(hs, community_attr="community")
    assert len(bridges) > 0
    # the bridge triad {a3, b1, b2} should appear
    bridge_sets = [b.members for b in bridges]
    assert frozenset(["a3", "b1", "b2"]) in bridge_sets


def test_bridge_spans_both_communities():
    hs = _two_community_hs()
    bridges = find_bridge_simplices(hs)
    for b in bridges:
        assert len(b.communities_spanned) >= 2


def test_group_cohesion_stronger_vs_weaker():
    hs = _two_community_hs()
    # tight internal triad
    strong = group_cohesion(hs, ["a1", "a2", "a3"])
    # loose cross-community pair with a dangling member
    weak = group_cohesion(hs, ["a1", "b3", "a2"])
    # the fully-connected internal group should score higher
    assert strong.score > weak.score


def test_group_cohesion_components():
    hs = _two_community_hs()
    cs = group_cohesion(hs, ["a1", "a2", "a3"])
    assert "relation_weight" in cs.components
    assert "sub_relation_density" in cs.components
    assert "higher_order_support" in cs.components


def test_rank_influencers_has_both_keys():
    hs = _two_community_hs()
    result = rank_influencers(hs, top_k=5)
    assert "graph_centrality" in result
    assert "simplex_diffusion" in result
    assert len(result["graph_centrality"]) > 0
    assert len(result["simplex_diffusion"]) > 0
