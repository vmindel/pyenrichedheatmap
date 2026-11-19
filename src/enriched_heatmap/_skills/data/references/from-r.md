# Coming from R EnrichedHeatmap / ComplexHeatmap

| R | eh |
|---|---|
| `normalizeToMatrix(sig, target, extend=1000, w=20)` | `b = eh.bins(target, 1000, 1000, 20)`; extract; `eh.matrices(vals, b)` |
| scale-regions target, `target_ratio` | `eh.bins(..., body_bins=n)` |
| `mean_mode`, `value_column` | the extraction: misha vtrack `func=` or `eh.read_bigwig(agg=)` |
| `smooth = TRUE` | `mat.smooth(sigma)` |
| `EnrichedHeatmap(mat, col=)` | `eh.heatmap(mat, ax, rows, cmap=, vmax=)` |
| `anno_enriched()` top annotation | `eh.profile(mat, ax, rows)` |
| `row_split = km$cluster` | `eh.arrange(by, groups=clusters)` |
| `enriched_score()` ordering | `eh.arrange(mat, metric="enriched_score")` |
| `ht1 + ht2 + rowAnnotation()` | `eh.figure(widths)` columns or `eh.plot(mats, rows, annotations=)` |
| shared `col_fun` | `eh.limits([m1, m2])` or `eh.plot(shared=[[...]])` |

No `+` operator and no track classes: lay out axes, then call a function per
panel with the same `rows`.
