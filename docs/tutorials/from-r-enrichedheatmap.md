# Coming from R `EnrichedHeatmap`

If you already know R's `EnrichedHeatmap` and `ComplexHeatmap`, the concepts map
directly onto `pyenrichedheatmap`.

## Translation table

| R (`EnrichedHeatmap` / `ComplexHeatmap`) | Python (`enriched_heatmap`) |
|---|---|
| `normalizeToMatrix(sig, target, extend=c(1000, 1000), w=10)` | `eh.from_bigwig(bw, intervals, upstream=1000, downstream=1000, bin_size=10)` |
| `normalizeToMatrix(..., target_ratio=0.3)` | `mode="scale_regions", target_bins=60` |
| `enriched_score(mat)` | `eh.compute_order_scores(mat, metric="enriched_score")` |
| `EnrichedHeatmap(mat, top_annotation=HeatmapAnnotation(lines=anno_enriched()))` | `eh.EnrichedTrack(mat, top_profile="mean_ci")` |
| `Heatmap(rna_mat, col=colorRamp2(c(-2, 0, 2), ...))` | `eh.MatrixTrack(rna_df, cmap="RdBu_r", center=0.0)` |
| `Heatmap(peak_class)` | `eh.CategoricalTrack(peak_class, palette="Set2")` |
| `rowAnnotation(bar=anno_barplot(score))` | `eh.BarTrack(score)` |
| `rowAnnotation(pt=anno_points(val))` | `eh.PointTrack(val, smooth_line=True)` |
| `ht1 + ht2 + ht3` | `track1 + track2 + track3` |
| `draw(ht_list, row_split=clusters, row_order=ord)` | `hm.plot(split_by=clusters, order_by=ord)` |
| `row_km = 3` | `split_by=3` |
| `use_raster = TRUE` | Built-in `raster_dpi=300` + `downsample_agg` |
| Shared `col_fun` across heatmaps | `scale_group="ATAC"` (also shares one colorbar & top-profile y-axis) |

## Key differences to keep in mind

1. **0-based half-open intervals `[start, end)`:** R `GRanges` are 1-based
   closed `[start, end]`. Subtract `1` from `start` when exporting `GRanges` to
   pandas for `from_bigwig` or `from_pymisha`.
2. **Zero-desync alignment by `region_index`:** Instead of relying on implicit
   row positions across matrices, `pyenrichedheatmap` aligns every track by its
   unique `region_index` (`pd.Index`) before ordering or splitting.
3. **Differential auto-defaults:** `mut.log2_ratio(wt)` and `mut.subtract(wt)`
   automatically trigger `cmap="RdBu_r"` and `center=0.0` (`TwoSlopeNorm`) inside
   `EnrichedTrack`.
