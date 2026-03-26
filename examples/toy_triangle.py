"""Toy triangle -- the smallest nontrivial simplicial complex.

Builds a single triangle (3 nodes, 3 edges, 1 face), computes the Hodge
Laplacians, and prints a spectral summary.
"""

from she import SHEConfig, SHESimplicialComplex

# -- build complex --------------------------------------------------------
config = SHEConfig(max_dimension=2)
sc = SHESimplicialComplex("triangle", config=config)

for node in [0, 1, 2]:
    sc.add_node(node)
for edge in [(0, 1), (1, 2), (0, 2)]:
    sc.add_edge(edge)
sc.add_simplex([0, 1, 2])  # the face

# -- inspect --------------------------------------------------------------
print("=== Toy Triangle ===")
print(f"Dimension of complex : {sc.complex.dim}")
for d in range(sc.complex.dim + 1):
    simplices = sc.get_simplex_list(d)
    print(f"  {d}-simplices ({len(simplices)}): {simplices}")

# -- Hodge Laplacians -----------------------------------------------------
laplacians = sc.get_hodge_laplacians()
for k, L in laplacians.items():
    print(f"\nHodge Laplacian L_{k}  (shape {L.shape}):")
    print(L.toarray())

print("\nDone.")
