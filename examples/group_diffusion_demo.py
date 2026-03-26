"""Diffusion analysis on a small weighted graph.

Builds a weighted social-style graph, lifts it to a simplicial complex,
runs Hodge diffusion analysis, and prints top diffusers per dimension.
Optionally saves a spectrum plot.
"""

import numpy as np
import networkx as nx

from she import SHEConfig, SHEDataLoader, SHEHodgeDiffusion

# -- build weighted graph -------------------------------------------------
np.random.seed(42)
G = nx.karate_club_graph()

for u, v in G.edges():
    G[u][v]["weight"] = np.random.exponential(1.0)
for node in G.nodes():
    G.nodes[node]["weight"] = np.random.gamma(2.0, 1.0)

# -- lift to complex and analyse ------------------------------------------
config = SHEConfig(max_dimension=2, spectral_k=5)
sc = SHEDataLoader.from_weighted_networkx(G, include_cliques=True, max_clique_size=4)
sc.config = config

analyzer = SHEHodgeDiffusion(config)
result = analyzer.analyze_diffusion(sc)

# -- report ---------------------------------------------------------------
print("=== Group Diffusion Demo (Karate Club) ===")
for dim in sorted(result.key_diffusers):
    top5 = result.key_diffusers[dim][:5]
    print(f"\nTop 5 key diffusers (dimension {dim}):")
    for i, (simplex, score) in enumerate(top5, 1):
        print(f"  {i}. {simplex}  score={score:.4f}")

for dim in sorted(result.eigenvalues):
    ev = result.eigenvalues[dim]
    gap = f"{ev[1] - ev[0]:.4f}" if len(ev) > 1 else "N/A"
    print(f"\nSpectral summary dim {dim}: {len(ev)} eigenvalues, gap={gap}, max={ev[-1]:.4f}")

# -- optional plot --------------------------------------------------------
try:
    from she import SHEDiffusionVisualizer

    SHEDiffusionVisualizer.plot_spectrum(result, 0, save_path="spectrum_dim0.png")
    print("\nSaved spectrum_dim0.png")
except Exception as exc:
    print(f"\nPlot skipped: {exc}")

print("\nDone.")
