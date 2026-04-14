from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .toy_complex import build_toy_complex
from .flow import iterate_coupled, fixed_geometry_step
from .observables import right_region_mass

def generate_demo_plot(outdir: str | Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    complex_fixed = build_toy_complex()
    complex_coupled = build_toy_complex()

    x0 = np.zeros(len(complex_fixed.edges))
    x0[0] = 1.0

    fixed_masses = [right_region_mass(complex_fixed, np.abs(x0))]
    x = x0.copy()
    for _ in range(8):
        x = fixed_geometry_step(complex_fixed, x)
        fixed_masses.append(right_region_mass(complex_fixed, np.abs(x)))

    traj, _tri_w, _edge_w = iterate_coupled(complex_coupled, x0, steps=8, mu=0.5)
    coupled_masses = [right_region_mass(complex_coupled, np.abs(z)) for z in traj]

    xs = list(range(len(fixed_masses)))
    plt.figure(figsize=(6, 4))
    plt.plot(xs, fixed_masses, marker="o", label="fixed geometry")
    plt.plot(xs, coupled_masses, marker="s", label="coupled geometry-field")
    plt.xlabel("step")
    plt.ylabel("right-region edge mass")
    plt.title("Toy bridge complex: fixed vs coupled")
    plt.legend()
    plt.tight_layout()
    path = outdir / "toy_fixed_vs_coupled.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return path
