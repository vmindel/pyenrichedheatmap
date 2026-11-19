# Claude Code skill

`pyenrichedheatmap` ships an
[Agent Skill](https://docs.claude.com/en/docs/claude-code/skills) for
[Claude Code](https://claude.com/claude-code) that teaches the agent how to use
the library accurately. It covers API contracts, coordinate conventions, shared
scales, clustering, and large-scale rendering pitfalls.

## Install

Claude Code does not scan installed Python packages for skills, so copy the
skill once into your personal skills directory:

```bash
pyenrichedheatmap-install-skills
```

This installs it to `~/.claude/skills/pyenrichedheatmap/`, where Claude Code
discovers it automatically in every project from the next session on.

| command | effect |
|---|---|
| `pyenrichedheatmap-install-skills` | install to `~/.claude/skills/pyenrichedheatmap` |
| `pyenrichedheatmap-install-skills --dest .claude/skills/pyenrichedheatmap` | install into the current project only |
| `pyenrichedheatmap-install-skills --force` | overwrite an existing copy after upgrading |
| `pyenrichedheatmap-install-skills --print-path` | print the bundled skill path without copying |

Once installed, ask Claude Code for genomic heatmap tasks in plain language —
for example: *"load my ATAC and ChEC BigWigs around these TSSs, cluster into 3
groups, add our RNA-seq log2FC heatmap on the right, and export a 300 dpi
vector/raster PDF"*.

## What's inside

The skill uses a lightweight router, `SKILL.md`, pointing to topic-specific
reference files under `references/` so it uses minimal context until needed:

| file | covers |
|---|---|
| `references/matrix-and-transforms.md` | `NormalizedMatrix`, modes, alignment, non-destructive transforms, interchange contract |
| `references/loaders-genomics.md` | `from_bigwig`, `from_pymisha`, 0-based `[start, end)` intervals, minus-strand orientation, OOB padding |
| `references/tracks-and-composition.md` | `EnrichedTrack`, `MatrixTrack`, `CategoricalTrack`, `BarTrack`, `PointTrack`, `scale_group`, differential defaults |
| `references/ordering-clustering-splits.md` | `order_by`, `enriched_score`, `split_by`, single- and multi-track k-means/hierarchical clustering |
| `references/rendering-and-large-scale.md` | smart downsampling (`mean`/`q90`/`max`/`absmax`), raster DPI, hybrid PDFs, `return_axes=True` |
| `references/from-r-enrichedheatmap.md` | translation table from R `EnrichedHeatmap` / `ComplexHeatmap` |
| `scripts/verify_figure.py` | standalone verification of vertical alignment, rasterization, and compact PDF output |

The test suite (`tests/test_skills.py`) checks the skill on every commit:

- `SKILL.md` frontmatter is valid and every routed file exists;
- every file in `references/` is routed from `SKILL.md`;
- every `eh.<name>` referenced in the skill is a real export on `enriched_heatmap`;
- the `pyenrichedheatmap-install-skills` CLI works;
- `scripts/verify_figure.py` passes cleanly.
