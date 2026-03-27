# Contributing to SHE

SHE is a **Research Preview** under the [HNCL v1.0](LICENCE.md) license
(source-available, non-commercial).

## How to engage

- **Issues** are welcome: bugs, questions, feature ideas.
- **Pull requests** are welcome for the stable package (`src/she/`).
- Please keep PRs focused — one concern per PR.
- New public-surface features should include tests and an example or
  docstring update.

## What is where

| Path | Status |
|---|---|
| `src/she/` | Canonical public package — contributions welcome here |
| `src/core/`, `src/morse/` | Legacy/experimental — not part of stable API |
| `tests/` | pytest suite — run with `pytest` |
| `examples/` | Scripts and notebooks |
| `docs/` | Tutorials, API overview, use cases |

## Running tests

```bash
pip install -e ".[dev]"
pytest -v
```

## Style

- No heavy formatting rules yet. Keep code readable.
- Prefer clear variable names over comments.
- Match the style of surrounding code.

## Reporting security issues

If you find a security issue, please email [info@holomathics.com](mailto:info@holomathics.com)
rather than opening a public issue.
