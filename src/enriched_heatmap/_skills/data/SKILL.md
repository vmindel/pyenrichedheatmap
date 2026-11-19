---
name: pyenrichedheatmap
description: Use for any task involving pyenrichedheatmap (imported as enriched_heatmap) — publication-grade Python genomic heatmaps and horizontal multi-omics track composition inspired by R's EnrichedHeatmap and ComplexHeatmap. Triggers on plotting ChEC-seq, ChIP-seq, ATAC-seq, MNase-seq, or RNA-seq signal heatmaps with top metaplots; loading signal matrices from BigWig (pyBigWig), pymisha databases/virtual tracks, NumPy arrays, or pandas DataFrames; reference-point (TSS/center) or scale-regions (gene body) normalization; strand-aware bin orientation; horizontal `+` composition of EnrichedTrack, MatrixTrack, CategoricalTrack, BarTrack, and PointTrack; differential log2FC heatmaps and shared-scale colorbars; k-means or hierarchical clustering splits; smart row downsampling for 10k-100k regions into compact hybrid vector/raster PDFs; or porting R EnrichedHeatmap/ComplexHeatmap figures to Python. This is a router skill — read the relevant file under references/ before writing enriched_heatmap code.
---

# pyenrichedheatmap (`enriched_heatmap`)

Builds aligned, publication-grade genomic heatmaps with top aggregate metaplots
and horizontally composed annotation tracks using matplotlib `GridSpec`. This
file only routes: the reference files hold exact signatures, coordinate rules,
and silent pitfalls. Read the relevant reference file before writing code.

## Task to reference file

| task | read |
|---|---|
| **`NormalizedMatrix` & transforms**: `reference_point` vs `scale_regions`, `.reindex()`, `.loc`, `.log1p()`, `.zscore()`, `.smooth()`, `.log2_ratio()`, `.subtract()`, interchange payloads | `references/matrix-and-transforms.md` |
| **loading genomic data**: `from_bigwig`, `from_pymisha`, `from_array`, `from_dataframe`, `from_interchange`, 0-based `[start, end)` intervals, minus-strand flipping, OOB padding, `chr` prefixes | `references/loaders-genomics.md` |
| **tracks & `+` composition**: `EnrichedTrack`, `MatrixTrack`, `CategoricalTrack`, `BarTrack`, `PointTrack`, `HeatmapList`, shared `scale_group` colorbars & synchronized metaplots, differential auto-defaults | `references/tracks-and-composition.md` |
| **ordering, clustering & splits**: `order_by`, `order_metric` (`center_mean`, `enriched_score`), `split_by`, `cluster_kmeans`, `cluster_hierarchical`, joint multi-track clustering, `region_index` zero-desync | `references/ordering-clustering-splits.md` |
| **large-scale rendering & PDF quality**: `downsample_agg` (`mean`, `q90`, `max`, `absmax`), `raster_dpi`, `max_raster_rows`, vector chrome vs raster body, `join` modes, `return_axes=True` | `references/rendering-and-large-scale.md` |
| **porting R `EnrichedHeatmap` / `ComplexHeatmap`** figures to Python | `references/from-r-enrichedheatmap.md` |
| **"verify alignment & compact PDF output"**: independent check script for pixel alignment, rasterization, and PDF size | `scripts/verify_figure.py` |

`scripts/verify_figure.py` (path relative to this skill's root) renders a
multi-omics test figure and verifies exact vertical cluster alignment,
rasterized heatmap bodies vs vector metaplots, and compact PDF size. Run
`python scripts/verify_figure.py --help`.

## Conventions

- Package install name is `pyenrichedheatmap`; import name is
  `import enriched_heatmap as eh`.
- Every track carries a unique `pd.Index` (`region_index`). `eh.HeatmapList.plot()`
  aligns all tracks by `region_index` (default `join="inner"`) before ordering
  or splitting — never manually sort NumPy arrays outside the tracks.
- Genomic intervals are **0-based, half-open `[start, end)`** everywhere.
  When a `strand` column is present and `strand_oriented=True` (default),
  minus-strand rows are automatically reversed 5' -> 3' (including asymmetric
  `upstream != downstream` flanks and `tss` anchoring at `end`).
- All transforms on `eh.NormalizedMatrix` are **non-destructive** and return a
  new copy.
- Save figures with `fig.savefig("out.pdf", dpi=300)` — only heatmap bodies are
  rasterized (`rasterized=True`); metaplots, ticks, borders, labels, legends,
  and colorbars remain 100% vector.
