# she-geofield

A focused extension package for the public **S.H.E. — Simplicial Hyperstructure Engine**.

This directory is the public software companion to the article
`Coupled Geometry--Field Dynamics on Weighted Simplicial Complexes:
A Unified Framework for Higher-Order Propagation`.

This repository adds a **geometry–field layer** to SHE:

- curvature-driven simplex-weight evolution,
- Hodge-Laplacian diffusion on the current geometry,
- coupled discrete-time geometry/field simulation,
- toy experiment and figure generation for the accompanying paper.

## Why this exists

Public SHE already supports decorated higher-order structures, simplicial lifting, diffusion ranking, bridge heuristics, temporal slicing, and export. `she-geofield` is intentionally narrower: it supports the article on **coupled geometry–field dynamics on weighted simplicial complexes**.

## Scope

Version 0.1 intentionally does only a few things:

1. define a minimal weighted simplicial complex object,
2. build boundary matrices and weighted Hodge Laplacians,
3. compute a simple Forman-style curvature score on edges and triangles,
4. update simplex weights under a curvature-driven rule,
5. run fixed-vs-coupled diffusion on a toy bridge complex,
6. generate reproducible figures.

It is not a replacement for SHE. It is a research extension.

## Model implemented here

The code implements the same minimal coupled update used in the root article:

- field step: `x^{n+1} = exp(-dt L_1(w^n)) x^n`,
- geometry step: `w^{n+1}(t) = w^n(t) exp(-eta G(t; w^n, pi^n, x^n))`,
- triangle curvature:
  `G(t) = w(t)/mean(w_e) - 1 + lambda*cohesion(t) - mu*E(t; x)`,
- normalized field energy:
  `E(t; x) = mean(|x_e| : e in boundary(t)) / (max_e |x_e| + eps)`.

Only triangle weights evolve. Edge Forman curvature is included as a diagnostic quantity.

## Reproducibility

Run the tests from the workspace root:

```bash
pytest she-geofield/tests -q
```

Run the toy experiment and regenerate the paper outputs:

```bash
python3 she-geofield/examples/run_toy_geofield.py
```

This writes:

- `out/fig1_fixed_vs_coupled.png` through `out/fig5_seed_comparison.png`,
- `out/table1_trajectory.csv` through `out/table7_mu_sensitivity.csv`,
- `out/verification.txt`.

## Publication role

`she-geofield` is intentionally narrow. It is not the general S.H.E. package.
It lives as a focused subdirectory inside the public `S.H.E.` repository and
provides the exact implementation used to support the article's coupled
geometry-field model and toy experiment.

## Intended connection to SHE

The public SHE repo currently exposes a stable package surface under `src/she/` with modules such as `complex.py`, `diffusion.py`, `hyperstructure.py`, `social.py`, and `temporal.py`. This extension is designed to sit next to that surface and import/export from it rather than rewrite it.
