# `NormalizedMatrix` and signal transforms

`eh.NormalizedMatrix` is the core genomic signal container. It pairs a 2D
floating-point matrix `(n_regions, n_bins)` with a **unique** `pd.Index`
(`region_index`) and coordinate geometry.

## Construction & modes

```python
import enriched_heatmap as eh

# Reference-point mode (e.g., ±1 kb around peak center or TSS)
mat = eh.NormalizedMatrix(
    values=arr,                # shape (n_regions, 200), float32/float64
    region_index=peak_ids,     # unique pd.Index (defaults to RangeIndex)
    mode="reference_point",
    upstream=1000,
    downstream=1000,
    bin_size=10,
    name="ATAC-seq",
)

# Scale-regions mode (e.g., -1 kb -> TSS -> scaled gene body -> TES -> +1 kb)
gene_mat = eh.NormalizedMatrix(
    values=gene_arr,           # shape (n_genes, 10 + 30 + 10)
    region_index=gene_ids,
    mode="scale_regions",
    upstream=1000,
    downstream=1000,
    bin_size=100,
    target_bins=30,
    name="Pol II",
)
```

### Coordinate helpers

- `mat.bin_centers()`: exact x-coordinates of each bin center (e.g., `[-995, ..., +995]`).
- `mat.guide_positions()`: `[0.0]` for `reference_point`; `[0.0, mat.body_length]` for `scale_regions`.
- `mat.default_xticks(anchor_labels=...)`: clean formatted ticks (`-1 kb`, `0`, `+1 kb` or `-1 kb`, `TSS`, `TES`, `+1 kb`).

## Subsetting & alignment

- `mat.reindex(new_index, fill_value=np.nan)`: aligns rows to `new_index` without mutating `mat`.
- `mat.loc[ids_or_bool_mask]` / `mat.iloc[positions]`: returns a new `eh.NormalizedMatrix`.
- `mat.dropna_regions(how="any" | "all", thresh=None)`: drops rows containing NaNs.

## Non-destructive signal transforms

Every transform returns a **new** `eh.NormalizedMatrix`:

- `mat.log1p()` — `log(1 + x)`
- `mat.log2p(pseudocount=1.0)` — `log2(x + pseudocount)`
- `mat.zscore(axis=1, ddof=0)` — per-region (`axis=1`) or per-bin (`axis=0`) z-score; constant rows safely become `0.0` (never NaN)
- `mat.clip_quantile(q_low=0.0, q_high=0.99)` — clip finite values to quantiles
- `mat.smooth(sigma=1.0, mode="nearest")` — NaN-aware 1D Gaussian smoothing along bins (uses SciPy when available, pure-NumPy fallback otherwise)
- `mat_a.subtract(mat_b)` — aligns `mat_b` to `mat_a.region_index`, computes `mat_a - mat_b`, sets `metadata["is_differential"] = True`
- `mat_a.log2_ratio(mat_b, pseudocount=1.0)` — aligns `mat_b` to `mat_a.region_index`, computes `log2((mat_a + p) / (mat_b + p))`, sets `metadata["is_differential"] = True`

## Interchange contract (zero cross-package imports)

Any external pipeline (such as a Fragment Geometry package) can hand off data
without importing `enriched_heatmap`:

```python
payload = (values_2d, {"region_index": ids, "upstream": 1000, "downstream": 1000, "bin_size": 10, "name": "Sub-nuc"})
mat = eh.from_interchange(payload)
# Or pass `payload` directly into eh.EnrichedTrack(payload)!
```

## Pitfalls

- `region_index` **must be unique**. Duplicate IDs raise `ValueError` immediately to prevent silent row misalignments across tracks.
- `target_bins` must be `0` in `reference_point` mode and `> 0` in `scale_regions` mode.
- `.subtract()` and `.log2_ratio()` align `other` onto `self.region_index`; missing regions in `other` become `NaN`.
