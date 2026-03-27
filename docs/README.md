# SHE Documentation

**SHE** (Simplicial Hyperstructure Engine) is a source-available research library
for modeling and analyzing decorated higher-order relational structures, with a
focus on social/group-level diffusion analysis.

**Status:** Research Preview (v0.1.2) under [HNCL v1.0](../LICENCE.md).
**Canonical package:** `src/she/` — install with `pip install -e .`, import as `import she`.

## Where to start

| If you want to... | Go to |
|---|---|
| Install and run your first example | [Getting Started](tutorials/getting_started.md) |
| Understand the social-analysis features | [Social Diffusers Tutorial](tutorials/social_diffusers.md) |
| See the public API surface | [API Overview](api/overview.md) |
| Read the social-media use case | [Use Cases](usecases/social_media_diffusers.md) |

## Docs map

```
docs/
  README.md              ← you are here
  tutorials/
    getting_started.md   ← install, first import, first examples
    social_diffusers.md  ← diffusers, bridges, cohesion, temporal
  api/
    overview.md          ← public API map
  usecases/
    social_media_diffusers.md
  archive/               ← historical materials (not current)
```

## What is stable

The `she` package (`src/she/`) is the public surface:
`SHEHyperstructure`, social analysis functions, temporal windowing, export.

Code under `src/core/` and `src/morse/` is legacy/experimental and not part
of the stable API.
