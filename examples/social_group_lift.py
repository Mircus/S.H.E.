"""Lift a small social network to a simplicial complex via clique detection.

Starts from a tiny NetworkX graph, lifts cliques to higher-order simplices,
and prints the resulting structure.
"""

import networkx as nx

from she import SHEDataLoader

# -- build a small "friendship" graph -------------------------------------
G = nx.Graph()
G.add_edges_from([
    ("Alice", "Bob"),
    ("Alice", "Carol"),
    ("Bob", "Carol"),       # triangle -> will become a 2-simplex
    ("Carol", "Dave"),
    ("Dave", "Eve"),
    ("Carol", "Eve"),
    ("Dave", "Eve"),        # duplicate, harmless
])

# Add unit weights so the loader has something to average
for u, v in G.edges():
    G[u][v]["weight"] = 1.0

# -- lift to simplicial complex -------------------------------------------
sc = SHEDataLoader.from_weighted_networkx(G, include_cliques=True)

# -- report ---------------------------------------------------------------
print("=== Social Group Lift ===")
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Complex dimension: {sc.complex.dim}")

for d in range(sc.complex.dim + 1):
    simplices = sc.get_simplex_list(d)
    print(f"  {d}-simplices ({len(simplices)}): {simplices}")

print("\nDone.")
