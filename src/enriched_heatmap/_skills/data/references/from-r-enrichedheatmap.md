# Coming from R `EnrichedHeatmap` / `ComplexHeatmap`

Quick translation table from R's `EnrichedHeatmap` + `ComplexHeatmap` to
`enriched_heatmap` (`import enriched_heatmap as eh`).

## Concept mapping

| R (`EnrichedHeatmap` / `ComplexHeatmap`) | Python (`enriched_heatmap`) |
|---|---|
| `normalizeToMatrix(signal, target, extend=c(1000, 1000), w=10, mean_mode="w0")` | `eh.from_bigwig(bw, intervals, upstream=1000, downstream=1000, bin_size=10)` or `eh.from_pymisha(...)` |
| `normalizeToMatrix(..., target_ratio=0.3)` (gene body) | `mode="scale_regions", target_bins=60` |
| `enriched_score(mat)` | `eh.compute_order_scores(mat, metric="enriched_score")` |
| `EnrichedHeatmap(mat, col=..., top_annotation=HeatmapAnnotation(lines=anno_enriched()))` | `eh.EnrichedTrack(mat, cmap="Reds", top_profile="mean_ci")` |
| `Heatmap(rna_mat, col=colorRamp2(c(-2,0,2), ...))` | `eh.MatrixTrack(rna_df, cmap="RdBu_r", center=0.0)` |
| `Heatmap(class_vec, col=...)` | `eh.CategoricalTrack(peak_class, palette="Set2")` |
| `rowAnnotation(bar=anno_barplot(score))` | `eh.BarTrack(score)` |
| `rowAnnotation(pt=anno_points(val))` | `eh.PointTrack(val, smooth_line=True)` |
| `ht1 + ht2 + ht3` | `track1 + track2 + track3` (returns `eh.HeatmapList`) |
| `draw(ht_list, row_split=clusters, row_order=ord)` | `hm.plot(split_by=clusters, order_by=ord)` |
| `row_km = 3` | `split_by=3` (or `eh.cluster_kmeans(...)`) |
| `use_raster=TRUE, raster_quality=2` | Built-in `raster_dpi=300` + smart `downsample_agg` (`"mean"`, `"q90"`, `"max"`, `"absmax"`) |
| Shared color scale (`col = shared_col_fun`) | `scale_group="ATAC signal"` on both tracks (also shares one colorbar & synced top metaplot y-axis) |

## Key behavioral differences

1. **Zero-based half-open intervals `[start, end)`:** R `GRanges` are 1-based
   closed `[start, end]`. When converting a `GRanges` export to a pandas
   DataFrame for `eh.from_bigwig` or `eh.from_pymisha`, subtract 1 from `start`.
2. **Automatic alignment by `region_index`:** R's `ComplexHeatmap` aligns tracks
   by row position (silently misaligning if one matrix was filtered or sorted
   differently). `enriched_heatmap` enforces alignment by unique `region_index`
   across every track.
3. **Differential auto-centering:** `diff = mut.log2_ratio(wt)` flags the matrix
   as differential so `eh.EnrichedTrack(diff)` automatically uses `cmap="RdBu_r"`
   and `center=0.0` (`TwoSlopeNorm`) without manual `colorRamp2`.
