---
name: pyenrichedheatmap
description: Use for any task involving the pyenrichedheatmap library (import enriched_heatmap as eh) - enriched heatmaps / metaplots / tornado plots in Python, signal around a set of regions with one row per region. Triggers on heatmaps of ChEC-seq, ChIP-seq, ATAC, MNase or CUT&RUN signal around peaks, TSSs, motifs or gene bodies; pymisha gextract into region x bin matrices; bigWig to matrix; k-means or hierarchical row splits; average profiles above heatmaps; R EnrichedHeatmap / ComplexHeatmap or deepTools plotHeatmap equivalents in Python. This is a router skill - read the relevant file under references/ before writing eh code.
---

# pyenrichedheatmap

Builds region x bin matrices from plain tables and draws them on matplotlib
axes. The data path is `eh.bins` (one row per bin) -> any extraction that
returns one value per bin (`pymisha.gextract(..., intervals=b, iterator=b)`,
`eh.read_bigwig`) -> `eh.matrices`. Drawing functions have the shape
`fn(data, ax, rows, **style)`; `rows` from `eh.arrange` is the one object that
aligns every panel. This file only routes; read the file for the task first.

## Task to reference file

| task | read |
|---|---|
| **regions to matrices**: `eh.bins` anchors, strand, scale-regions, chromosome edges, `eh.matrices`, `eh.read_bigwig`, matrix transforms | `references/data-path.md` |
| **pymisha**: gextract with a bin table, several expressions, virtual tracks for sum/max, NA fill, why not `iterator=<int>` | `references/pymisha.md` |
| **row order and splits**: `eh.arrange`, `eh.score`, `eh.kmeans`, `eh.hclust`, groups from a table column | `references/ordering.md` |
| **drawing and layout**: `eh.heatmap`, `eh.profile`, `eh.annotation`, `eh.figure`, `eh.plot`, shared and diverging scales, downsampling, saving | `references/figures.md` |
| **coming from R** EnrichedHeatmap / ComplexHeatmap | `references/from-r.md` |
| "**why are my rows empty / NaN / misaligned?**": check a regions table | `scripts/check_regions.py` |

`scripts/check_regions.py` (path relative to this skill's root) reads a
BED/TSV/CSV/parquet region table and reports missing columns, strand
encoding, duplicate identifiers, chromosome names absent from a genome
table, windows that run off a chromosome, and the size of the bin table.
Run `python scripts/check_regions.py --help`.

## Conventions

- `import enriched_heatmap as eh`. Coordinates are 0-based, half-open
  (BED, misha). Never add or subtract 1 between pymisha and eh.
- Region identifiers are the DataFrame index (or `eh.bins(..., id="col")`)
  and must be unique. Matrices, annotation Series and `rows` are matched by
  identifier, never by position.
- Strand: `"+"`/`"-"` and `1`/`-1` both work. Matrices come out 5' -> 3';
  never flip them afterwards.
- Pass the `eh.bins` table unchanged (or boolean-filtered) to the extraction
  and to `eh.matrices`; its layout lives in `DataFrame.attrs`, which
  `merge`/`concat` drop.
- Every drawing function takes the same `rows`; build it once with
  `eh.arrange(by, groups=...)`.
