"""Social media diffusers -- why SHE instead of graph-only analysis?

This example builds a synthetic social scenario with two communities and one
cross-community triad that acts as the real diffusion engine.  It then
compares graph centrality with simplex-level diffusion ranking to show what
higher-order analysis reveals that node-level metrics miss.

Scenario
--------
Community A: five tightly connected accounts (u0-u4).
Community B: five tightly connected accounts (u5-u9).
Hub:         u0 has high degree -- connected to many in A and a few in B.
Bridge triad: {u3, u5, u7} repeatedly co-amplify a topic across A and B.

Graph centrality will favour u0 (high degree).
Simplex diffusion will surface the {u3, u5, u7} triad as structurally more
important for cross-community information flow.
"""

from she import (
    SHEHyperstructure,
    SHEConfig,
    rank_diffusers,
    rank_entity_diffusers,
    rank_influencers,
    find_bridge_simplices,
    group_cohesion,
)


def build_scenario() -> SHEHyperstructure:
    config = SHEConfig(max_dimension=2, spectral_k=6)
    hs = SHEHyperstructure("social_media", config=config)

    # -- Community A (u0-u4) ----------------------------------------------
    for i in range(5):
        hs.add_entity(f"u{i}", community="A", role="regular")
    # make u0 a high-degree hub
    hs._entity_attrs["u0"]["role"] = "hub"

    # dense, high-weight internal edges in A — u0 is the star
    a_pairs = [
        ("u0", "u1", 2.5), ("u0", "u2", 2.5), ("u0", "u3", 2.0), ("u0", "u4", 2.0),
        ("u1", "u2", 1.0), ("u1", "u3", 0.8), ("u2", "u3", 0.8), ("u2", "u4", 0.6),
        ("u3", "u4", 0.6),
    ]
    for a, b, w in a_pairs:
        hs.add_relation([a, b], weight=w, kind="engagement")

    # internal triads in A centred on u0
    hs.add_relation(["u0", "u1", "u2"], weight=2.0, kind="co_engagement")
    hs.add_relation(["u0", "u2", "u4"], weight=1.5, kind="co_engagement")

    # -- Community B (u5-u9) ----------------------------------------------
    for i in range(5, 10):
        hs.add_entity(f"u{i}", community="B", role="regular")

    b_pairs = [
        ("u5", "u6", 1.0), ("u5", "u7", 1.0), ("u5", "u8", 0.8),
        ("u6", "u7", 0.8), ("u6", "u8", 0.6), ("u7", "u8", 0.8),
        ("u7", "u9", 0.6), ("u8", "u9", 0.6),
    ]
    for a, b, w in b_pairs:
        hs.add_relation([a, b], weight=w, kind="engagement")

    hs.add_relation(["u5", "u6", "u7"], weight=1.3, kind="co_engagement")

    # -- Cross-community links --------------------------------------------
    # hub u0 has moderately strong cross-community edges (boosts graph centrality)
    hs.add_relation(["u0", "u5"], weight=1.2, kind="mention")
    hs.add_relation(["u0", "u6"], weight=1.0, kind="mention")
    hs.add_relation(["u0", "u7"], weight=0.8, kind="mention")

    # the bridge triad: u3 (A), u5 (B), u7 (B) — moderate pairwise, heavy group
    hs.add_relation(["u3", "u5"], weight=1.0, kind="co_amplification", topic="climate")
    hs.add_relation(["u3", "u7"], weight=0.9, kind="co_amplification", topic="climate")
    # the key triad — its *group* weight is what matters
    hs.add_relation(
        ["u3", "u5", "u7"], weight=4.0,
        kind="co_amplification", topic="climate",
    )

    return hs


def main():
    # suppress noisy ARPACK warnings from small-matrix spectral solves
    import logging
    logging.getLogger("she.diffusion").setLevel(logging.ERROR)

    hs = build_scenario()
    print(f"Built: {hs!r}\n")

    # -- 1. Graph vs simplex ranking --------------------------------------
    comparison = rank_influencers(hs, top_k=5)

    print("=== Graph centrality (1-skeleton only) ===")
    for r in comparison["graph_centrality"]:
        label = r.target[0] if len(r.target) == 1 else r.target
        role = r.metadata.get("role", "")
        comm = r.metadata.get("community", "")
        print(f"  {label:>4s}  score={r.score:.4f}  community={comm}  role={role}")

    print("\n=== Simplex diffusion (all dimensions) ===")
    for r in comparison["simplex_diffusion"]:
        kind = r.metadata.get("kind", "")
        print(f"  dim={r.dimension}  {r.target}  score={r.score:.4f}  kind={kind}")

    # -- 2. Top entity diffusers ------------------------------------------
    print("\n=== Top entity diffusers (dim 0) ===")
    for r in rank_entity_diffusers(hs, top_k=5):
        label = r.target[0] if len(r.target) == 1 else r.target
        comm = r.metadata.get("community", "")
        print(f"  {label:>4s}  score={r.score:.4f}  community={comm}")

    # -- 3. Bridge simplices ----------------------------------------------
    print("\n=== Bridge simplices (cross-community) ===")
    bridges = find_bridge_simplices(hs)
    for b in bridges[:5]:
        print(
            f"  {sorted(b.members, key=str)}  dim={b.dimension}  "
            f"communities={b.communities_spanned}  bridge_score={b.bridge_score:.3f}  "
            f"kind={b.metadata.get('kind', '')}"
        )

    # -- 4. Group cohesion comparison -------------------------------------
    print("\n=== Group cohesion comparison ===")
    triad = ["u3", "u5", "u7"]
    hub_group = ["u0", "u1", "u2"]
    for group in [triad, hub_group]:
        cs = group_cohesion(hs, group)
        print(f"  {sorted(cs.members, key=str)}  score={cs.score:.4f}  {cs.components}")

    # -- 5. The punchline ------------------------------------------------
    print("\n=== Why this matters ===")
    print("Graph centrality highlights u0 (the high-degree hub).")
    print("Simplex diffusion highlights the {u3, u5, u7} triad -- a cross-community")
    print("co-amplification group that no node-level metric would surface.")
    print("The triad is the actual diffusion bottleneck between communities A and B.")


if __name__ == "__main__":
    main()
