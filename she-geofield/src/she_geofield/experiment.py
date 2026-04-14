"""Run the toy bridge-complex experiment and produce real figures + tables.

Outputs go to ``out/`` relative to the repo root.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from .toy_complex import build_toy_complex, _sorted_edge
from .hodge import hodge_laplacian_1
from .flow import fixed_geometry_step, iterate_coupled
from .observables import right_region_mass
from .curvature import edge_forman_curvature, triangle_curvature
from .boundaries import boundary_1_to_0, boundary_2_to_1


# ── helpers ────────────────────────────────────────────────────────────

def _hodge_spectrum(complex_):
    L = hodge_laplacian_1(complex_)
    return np.sort(np.linalg.eigvalsh(L))


def _total_mass(x):
    return float(np.sum(np.abs(x)))


def _right_fraction(complex_, x):
    total = _total_mass(x)
    if total < 1e-15:
        return 0.0
    return right_region_mass(complex_, np.abs(x)) / total


def _fmt(x, decimals=4):
    return f"{x:.{decimals}f}"


def _fmte(x):
    return f"{x:.4e}"


def _tri_label(t):
    """CSV-safe triangle label: abc not (a,b,c)."""
    return "".join(t)


# ── seed helpers ──────────────────────────────────────────────────────

def _make_edge_seed(complex_, edge_name):
    """Unit signal on a single edge."""
    x = np.zeros(len(complex_.edges))
    edge_list = list(complex_.edges)
    x[edge_list.index(edge_name)] = 1.0
    return x


def _make_triangle_seed(complex_, triangle):
    """Equal signal on the three boundary edges of a triangle."""
    x = np.zeros(len(complex_.edges))
    edge_list = list(complex_.edges)
    a, b, c = triangle
    for e in [_sorted_edge(a, b), _sorted_edge(a, c), _sorted_edge(b, c)]:
        x[edge_list.index(e)] = 1.0 / 3.0
    return x


# ── main experiment ───────────────────────────────────────────────────

def run_experiment(outdir: str | Path = "out", steps: int = 10,
                   dt: float = 0.5, eta: float = 0.15, lam: float = 0.5,
                   mu: float = 0.5):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── initial state ─────────────────────────────────────────────────
    complex_fixed = build_toy_complex()
    complex_coupled = build_toy_complex()

    # Bridge edge seed
    x0 = _make_edge_seed(complex_fixed, ("a", "c"))

    # ── 1. fixed-geometry run ─────────────────────────────────────────
    fixed_fracs = [_right_fraction(complex_fixed, x0)]
    fixed_totals = [_total_mass(x0)]
    x = x0.copy()
    for _ in range(steps):
        x = fixed_geometry_step(complex_fixed, x, dt=dt)
        fixed_fracs.append(_right_fraction(complex_fixed, x))
        fixed_totals.append(_total_mass(x))

    # ── 2. coupled run (with field feedback mu > 0) ───────────────────
    traj, tri_weights, edge_weights = iterate_coupled(
        complex_coupled, x0, steps=steps, dt=dt, eta=eta, lam=lam, mu=mu
    )
    coupled_fracs = [_right_fraction(complex_coupled, z) for z in traj]
    coupled_totals = [_total_mass(z) for z in traj]

    # ── 3. Hodge spectra (initial vs final) ───────────────────────────
    spec_init = _hodge_spectrum(build_toy_complex())
    spec_final = _hodge_spectrum(complex_coupled)

    # ── 4. seed comparison: bridge edge vs left triangle vs right tri ─
    seeds = {
        "bridge_ac": _make_edge_seed(build_toy_complex(), ("a", "c")),
        "tri_abc": _make_triangle_seed(build_toy_complex(), ("a", "b", "c")),
        "tri_cef": _make_triangle_seed(build_toy_complex(), ("c", "e", "f")),
    }
    seed_results = {}
    for name, s0 in seeds.items():
        c_fix = build_toy_complex()
        c_cpl = build_toy_complex()
        # fixed run
        xf = s0.copy()
        fix_frac = [_right_fraction(c_fix, xf)]
        for _ in range(steps):
            xf = fixed_geometry_step(c_fix, xf, dt=dt)
            fix_frac.append(_right_fraction(c_fix, xf))
        # coupled run
        tr, tw, ew = iterate_coupled(c_cpl, s0, steps=steps, dt=dt,
                                     eta=eta, lam=lam, mu=mu)
        cpl_frac = [_right_fraction(c_cpl, z) for z in tr]
        seed_results[name] = {
            "fixed": fix_frac, "coupled": cpl_frac,
            "tri_weights_final": tw[-1],
        }

    # ── FIGURE 1: right-region mass fraction ─────────────────────────
    xs = list(range(steps + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(xs, fixed_fracs, marker="o", label="fixed geometry")
    ax1.plot(xs, coupled_fracs, marker="s", label="coupled")
    ax1.set_xlabel("step")
    ax1.set_ylabel("right-region mass fraction")
    ax1.set_title("Signal redistribution (bridge edge seed)")
    ax1.legend()

    ax2.semilogy(xs, fixed_totals, marker="o", label="fixed |x|")
    ax2.semilogy(xs, coupled_totals, marker="s", label="coupled |x|")
    ax2.set_xlabel("step")
    ax2.set_ylabel("total mass (log scale)")
    ax2.set_title("Total signal decay")
    ax2.legend()

    fig.suptitle(f"Toy bridge complex ($\\eta$={eta}, $\\lambda$={lam}, $\\mu$={mu})")
    fig.tight_layout()
    fig.savefig(outdir / "fig1_fixed_vs_coupled.png", dpi=200)
    plt.close(fig)

    # ── FIGURE 2: triangle weight trajectories ──────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for tri in complex_coupled.triangles:
        ws = [tw[tri] for tw in tri_weights]
        ax.plot(xs, ws, marker="^", label=_tri_label(tri))
    ax.set_xlabel("step")
    ax.set_ylabel("triangle weight $w_t$")
    ax.set_title("Triangle weight evolution (field-coupled flow)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig2_triangle_weights.png", dpi=200)
    plt.close(fig)

    # ── FIGURE 3: Hodge spectrum shift ───────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 3.5))
    idx = np.arange(len(spec_init))
    w = 0.35
    ax.bar(idx - w/2, spec_init, w, label="initial", color="C0", alpha=0.8)
    ax.bar(idx + w/2, spec_final, w, label=f"after {steps} steps", color="C1", alpha=0.8)
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("$\\lambda_i$")
    ax.set_title("$L_1$ spectrum: initial vs coupled-evolved geometry")
    ax.legend()
    ax.set_xticks(idx)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_hodge_spectrum.png", dpi=200)
    plt.close(fig)

    # ── FIGURE 4: edge curvature comparison ──────────────────────────
    c0 = build_toy_complex()
    edges = complex_coupled.edges
    e_labels = [f"({e[0]},{e[1]})" for e in edges]
    f_init = [edge_forman_curvature(c0, e) for e in edges]
    f_final = [edge_forman_curvature(complex_coupled, e) for e in edges]
    idx = np.arange(len(edges))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(idx - w/2, f_init, w, label="initial", color="C0", alpha=0.8)
    ax.bar(idx + w/2, f_final, w, label=f"after {steps} steps", color="C1", alpha=0.8)
    ax.set_xlabel("edge")
    ax.set_ylabel("Forman curvature $F(e)$")
    ax.set_title("Edge Forman curvature: initial vs final")
    ax.set_xticks(idx)
    ax.set_xticklabels(e_labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig4_edge_curvature.png", dpi=200)
    plt.close(fig)

    # ── FIGURE 5: seed comparison ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (name, res) in zip(axes, seed_results.items()):
        ax.plot(xs, res["fixed"], marker="o", label="fixed")
        ax.plot(xs, res["coupled"], marker="s", label="coupled")
        ax.set_xlabel("step")
        ax.set_title(f"Seed: {name}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("right-region mass fraction")
    fig.suptitle("Seed comparison: bridge edge vs left triangle vs right triangle")
    fig.tight_layout()
    fig.savefig(outdir / "fig5_seed_comparison.png", dpi=200)
    plt.close(fig)

    # ── TABLE 1: full trajectory summary ─────────────────────────────
    tri_cols = [f"w_{_tri_label(t)}" for t in complex_coupled.triangles]
    lines = ["step;fixed_right_frac;coupled_right_frac;fixed_total;coupled_total;"
             + ";".join(tri_cols)]
    for k in range(steps + 1):
        tw_vals = ";".join(_fmt(tri_weights[k][t]) for t in complex_coupled.triangles)
        lines.append(f"{k};{_fmt(fixed_fracs[k])};{_fmt(coupled_fracs[k])};"
                     f"{_fmte(fixed_totals[k])};{_fmte(coupled_totals[k])};{tw_vals}")
    (outdir / "table1_trajectory.csv").write_text("\n".join(lines))

    # ── TABLE 2: edge curvature ──────────────────────────────────────
    lines = ["edge;F_init;F_final;delta_F"]
    for e in edges:
        fi = edge_forman_curvature(c0, e)
        ff = edge_forman_curvature(complex_coupled, e)
        lbl = f"{e[0]}-{e[1]}"
        lines.append(f"{lbl};{_fmt(fi)};{_fmt(ff)};{_fmt(ff-fi)}")
    (outdir / "table2_edge_curvature.csv").write_text("\n".join(lines))

    # ── TABLE 3: Hodge spectrum ──────────────────────────────────────
    lines = ["index;lambda_init;lambda_final;shift"]
    for i in range(len(spec_init)):
        lines.append(f"{i};{_fmt(spec_init[i])};{_fmt(spec_final[i])};{_fmt(spec_final[i]-spec_init[i])}")
    (outdir / "table3_hodge_spectrum.csv").write_text("\n".join(lines))

    # ── TABLE 4: eta sensitivity ─────────────────────────────────────
    lines = ["eta;final_right_frac;max_tri_w;min_tri_w;spectral_gap"]
    for eta_val in [0.01, 0.05, 0.1, 0.15, 0.25]:
        c = build_toy_complex()
        tr, tw, ew = iterate_coupled(c, x0, steps=steps, dt=dt,
                                     eta=eta_val, lam=lam, mu=mu)
        frac = _right_fraction(c, tr[-1])
        tw_f = tw[-1]
        sp = _hodge_spectrum(c)
        lines.append(f"{eta_val};{_fmt(frac)};{_fmt(max(tw_f.values()))};"
                     f"{_fmt(min(tw_f.values()))};{_fmt(sp[0])}")
    (outdir / "table4_eta_sensitivity.csv").write_text("\n".join(lines))

    # ── TABLE 5: lambda sensitivity ──────────────────────────────────
    lines = ["lambda;final_right_frac;max_tri_w;min_tri_w;spectral_gap"]
    for lam_val in [0.0, 0.2, 0.5, 0.8, 1.0]:
        c = build_toy_complex()
        tr, tw, ew = iterate_coupled(c, x0, steps=steps, dt=dt,
                                     eta=eta, lam=lam_val, mu=mu)
        frac = _right_fraction(c, tr[-1])
        tw_f = tw[-1]
        sp = _hodge_spectrum(c)
        lines.append(f"{lam_val};{_fmt(frac)};{_fmt(max(tw_f.values()))};"
                     f"{_fmt(min(tw_f.values()))};{_fmt(sp[0])}")
    (outdir / "table5_lambda_sensitivity.csv").write_text("\n".join(lines))

    # ── TABLE 6: seed comparison summary ─────────────────────────────
    lines = ["seed;final_fixed_frac;final_coupled_frac;delta;w_abc;w_cde;w_cef"]
    for name, res in seed_results.items():
        ff = res["fixed"][-1]
        cf = res["coupled"][-1]
        tw = res["tri_weights_final"]
        tris = list(build_toy_complex().triangles)
        lines.append(f"{name};{_fmt(ff)};{_fmt(cf)};{_fmt(cf-ff)};"
                     f"{_fmt(tw[tris[0]])};{_fmt(tw[tris[1]])};{_fmt(tw[tris[2]])}")
    (outdir / "table6_seed_comparison.csv").write_text("\n".join(lines))

    # ── TABLE 7: mu (field feedback) sensitivity ─────────────────────
    lines = ["mu;final_right_frac;max_tri_w;min_tri_w;spectral_gap"]
    for mu_val in [0.0, 0.25, 0.5, 1.0, 2.0]:
        c = build_toy_complex()
        tr, tw, ew = iterate_coupled(c, x0, steps=steps, dt=dt,
                                     eta=eta, lam=lam, mu=mu_val)
        frac = _right_fraction(c, tr[-1])
        tw_f = tw[-1]
        sp = _hodge_spectrum(c)
        lines.append(f"{mu_val};{_fmt(frac)};{_fmt(max(tw_f.values()))};"
                     f"{_fmt(min(tw_f.values()))};{_fmt(sp[0])}")
    (outdir / "table7_mu_sensitivity.csv").write_text("\n".join(lines))

    # ── verification ─────────────────────────────────────────────────
    c_check = build_toy_complex()
    B10 = boundary_1_to_0(c_check)
    B21 = boundary_2_to_1(c_check)
    chain_zero = np.max(np.abs(B10 @ B21))

    L = hodge_laplacian_1(c_check)
    sym_err = np.max(np.abs(L - L.T))
    psd = np.all(np.linalg.eigvalsh(L) >= -1e-12)

    (outdir / "verification.txt").write_text(
        f"max|B10 @ B21| = {chain_zero:.2e}  (chain complex: should be 0)\n"
        f"max|L1 - L1^T| = {sym_err:.2e}  (symmetry: should be 0)\n"
        f"L1 positive semi-definite: {psd}\n"
        f"L1 eigenvalues: {np.round(_hodge_spectrum(c_check), 4).tolist()}\n"
        f"Parameters: steps={steps}, dt={dt}, eta={eta}, lam={lam}, mu={mu}\n"
    )

    print(f"Experiment complete. Outputs in {outdir.resolve()}")
    print(f"  5 figures (PNG), 7 tables (CSV), 1 verification")
    print(f"  Chain complex check: max|d1 d2| = {chain_zero:.2e}")
    print(f"  L1 symmetric: {sym_err:.2e}, PSD: {psd}")
    return outdir


if __name__ == "__main__":
    run_experiment()
