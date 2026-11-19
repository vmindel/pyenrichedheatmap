# Claude Code skill

`pyenrichedheatmap` ships an
[Agent Skill](https://docs.claude.com/en/docs/claude-code/skills) for
[Claude Code](https://claude.com/claude-code) that teaches the agent how to use
the library accurately: the bins -> extraction -> matrices data path, the pymisha
workflow, row order and splits, and the drawing functions.

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

Once installed, ask for heatmap tasks in plain language, for example: *"extract
my ChEC and ATAC tracks from the misha DB around these TSSs, split into 3 k-means
clusters, add the RNA log2FC as a strip on the right and save a PDF"*.

## What's inside

The skill uses a lightweight router, `SKILL.md`, pointing to topic-specific
reference files under `references/` so it uses minimal context until needed:

| file | covers |
|---|---|
| `references/data-path.md` | `eh.bins` anchors, strand, scale-regions, chromosome edges; `eh.matrices`, `eh.read_bigwig`; matrix transforms |
| `references/pymisha.md` | `gextract` with a bin table, several expressions, virtual tracks, NA fill, why not an integer iterator |
| `references/ordering.md` | `eh.arrange`, `eh.score`, `eh.kmeans`, `eh.hclust` |
| `references/figures.md` | `eh.heatmap`, `eh.profile`, `eh.annotation`, `eh.figure`, `eh.plot`, scales, downsampling |
| `references/from-r.md` | R EnrichedHeatmap / ComplexHeatmap equivalents |
| `scripts/check_regions.py` | checks a region table: columns, strand, duplicate ids, chromosome names, edge windows |

The test suite (`tests/test_skills.py`) checks the skill on every commit:

- `SKILL.md` frontmatter is valid and every routed file exists;
- every file in `references/` is routed from `SKILL.md`;
- every `eh.<name>` referenced in the skill is a real export on `enriched_heatmap`;
- the `pyenrichedheatmap-install-skills` CLI works;
- `scripts/check_regions.py` flags a bad table and passes a good one.
