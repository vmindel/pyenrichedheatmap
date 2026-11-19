# Coming from R

R's EnrichedHeatmap computes matrices with `normalizeToMatrix` and draws them through
ComplexHeatmap's `+` composition. Here the two halves are separate: tables in,
matrices out, then drawing functions on axes you lay out.

| R (EnrichedHeatmap / ComplexHeatmap) | here |
|---|---|
| `normalizeToMatrix(signal, target, extend = 1000, w = 20)` | `b = eh.bins(target, 1000, 1000, 20)`, extract, `eh.matrices(vals, b)` |
| `target_ratio`, scale-regions targets | `eh.bins(..., body_bins=n)` |
| `value_column`, `mean_mode = "w0"` / `"absolute"` | choose in the extraction: a misha virtual track (`func="avg"`, `"sum"`, `"max"`) or `eh.read_bigwig(..., agg=)` |
| `smooth = TRUE` | `mat.smooth(sigma)` |
| `EnrichedHeatmap(mat, col = ...)` | `eh.heatmap(mat, ax, rows, cmap=..., vmax=...)` |
| `top_annotation = HeatmapAnnotation(enriched = anno_enriched())` | `eh.profile(mat, ax, rows)` |
| `row_split = km$cluster` | `eh.arrange(by, groups=clusters)` |
| `row_order = order(enriched_score(mat))` | `eh.arrange(mat, metric="enriched_score")` |
| `ht1 + ht2 + rowAnnotation(...)` | columns from `eh.figure(widths)` or `eh.plot(mats, rows, annotations=...)` |
| shared `col_fun` across heatmaps | `vmin, vmax = eh.limits([m1, m2])`, or `eh.plot(..., shared=[[...]])` |
| `use_raster = TRUE` | always: bodies are rasterised, the rest stays vector |

Two things work differently on purpose:

- **Alignment is by region identifier, not position.** Matrices, annotation Series and
  `rows` are matched on their index, so filtering or reordering one table cannot
  shift the others.
- **The extraction is yours.** Instead of an argument for every averaging mode, the
  per-bin value is whatever your `gextract` expression or bigWig aggregation returns.
