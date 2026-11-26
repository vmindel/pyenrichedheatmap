# Contributing

## Setup

```bash
pip install -e ".[dev]"
```

`dev` pulls in pyBigWig and pymisha so no test skips. Without them the bigWig and
misha tests skip, and so does the pymisha tutorial page.

## Before pushing

```bash
ruff check src tests docs/run_docs.py
```

```bash
mypy src/enriched_heatmap
```

```bash
pytest -ra
```

`pytest` also runs every tutorial in `docs/tutorials/` (through `tests/test_docs.py`) and
checks that the docs and the bundled skill references only name functions that exist.
CI runs the same three commands on Linux and macOS, Python 3.10 and 3.13, plus one job
on the oldest supported versions (Python 3.10, pandas 2.0, numpy 1.24, matplotlib 3.7).

## Docs

Tutorial code blocks are executed, and the `text` block after each is its expected
output. After changing behaviour that a tutorial shows:

```bash
python docs/run_docs.py --update
```

This rewrites the output blocks and redraws `docs/images/`. The PNGs depend on the
matplotlib version and fonts, so a redraw in a different environment changes every
image by a few pixels. Commit images only when a figure's content changed, and drop the
rest with `git checkout -- docs/images`. `mkdocs build --strict` checks the site.

## Rules the code relies on

- Panels are matched by region identifier, never by position. Region identifiers are
  unique, non-missing and not boolean; `NormalizedMatrix` and `bins` enforce this
  once, so every other lookup can use plain pandas indexing through
  `_matrix._safe_get_indexer` (which only guards the MultiIndex-prefix and
  IntervalIndex-containment traps). Don't add per-element type checks on top.
- `plot` validates every argument before it creates a figure, so a bad call never
  leaves an empty figure behind in pyplot.
- Support pandas 2 and 3. String IDs are object dtype on pandas 2 and `str` on
  pandas 3, which go through different lookup paths.
- Keep the bundled skill references (`src/enriched_heatmap/_skills/data/references/`)
  in step with the API; they are what an assistant reads to use the package.

## Changelog

Every user-visible change gets a line under `## Unreleased` in `CHANGELOG.md`. Mark
anything that changes existing output or makes a previously working call raise as
**(breaking)**, and say what to do instead.

## Release

1. Rename `## Unreleased` to the version.
2. Set the version in `pyproject.toml` and `src/enriched_heatmap/__init__.py`.
3. Merge to `main`. There are no tags or PyPI uploads yet; if we start, tag `vX.Y.Z`
   at this step.
