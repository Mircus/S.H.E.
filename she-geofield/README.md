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

## DBLP pipeline

Version 0.2 adds a narrow DBLP branch for the follow-up paper on temporal
coauthorship carriers. The DBLP support is intentionally scoped to one
reproducible experiment family:

- stream DBLP XML into a canonical `PaperRecord`,
- filter by year, team size, venue, and publication type,
- build deterministic rolling windows,
- lift coauthor sets into weighted simplicial windows,
- compare node baselines, frozen diffusion, and evolving geometry-field scores.

Run the sample DBLP experiment from the `she-geofield` directory:

```bash
python -m she_geofield.dblp.experiments --config configs/dblp_small.yaml
```

This writes:

- `out/dblp_small/window_summary.csv`,
- `out/dblp_small/model_comparison.csv`,
- `out/dblp_small/top_collaboration_simplices.csv`,
- `out/dblp_small/predictive_comparison.png`,
- `out/dblp_small/temporal_performance.png`,
- `out/dblp_small/case_study.png`.

For a first real-data run, extract a filtered slice from the official DBLP dump
and then point the experiment runner at the resulting CSV:

```bash
python -m she_geofield.dblp.extract_subset --input /path/to/dblp.xml.gz --output data/dblp_subset.csv --start-year 2018 --end-year 2024 --min-team-size 2 --max-team-size 6 --venue-regex '^SDM$' --publication-types article,inproceedings
python -m she_geofield.dblp.experiments --config configs/dblp_sdm_real.yaml
```

The real-data configs also emit a secondary bridge/emergence bundle alongside
the broad future-support outputs:

- `bridge_emergence_model_comparison.csv`
- `bridge_emergence_top_collaboration_simplices.csv`
- `bridge_emergence_predictive_comparison.png`
- `bridge_emergence_temporal_performance.png`
- `bridge_emergence_case_study.png`

This secondary evaluation targets low-persistence bridge candidates and future
branching into novel triangle contexts, which is the first regime where the
geometry layer is tested against something recurrence alone is not built to
capture.

There is also a dedicated bridge-to-cluster regime with standalone configs:

- `configs/dblp_sdm_bridge_to_cluster.yaml`
- `configs/dblp_wsdm_bridge_to_cluster.yaml`

These runs target early thin bridge edges that later thicken into new triangle
contexts. The corresponding outputs live in:

- `out/dblp_sdm_bridge_to_cluster/`
- `out/dblp_wsdm_bridge_to_cluster/`
- `out/dblp_task_cross_venue/` for the broad-recurrence vs bridge-emergence vs
  bridge-to-cluster comparison figure.

## Publication role

`she-geofield` is intentionally narrow. It is not the general S.H.E. package.
It lives as a focused subdirectory inside the public `S.H.E.` repository and
provides the exact implementation used to support the article's coupled
geometry-field model and toy experiment.

## Intended connection to SHE

The public SHE repo currently exposes a stable package surface under `src/she/` with modules such as `complex.py`, `diffusion.py`, `hyperstructure.py`, `social.py`, and `temporal.py`. This extension is designed to sit next to that surface and import/export from it rather than rewrite it.
