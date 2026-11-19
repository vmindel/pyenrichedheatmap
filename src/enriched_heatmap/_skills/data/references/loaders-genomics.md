# Genomic loaders (`from_bigwig`, `from_pymisha`, arrays)

All genomic loaders enforce **0-based, half-open `[start, end)`** coordinates
and automatic minus-strand 5' -> 3' orientation when a `strand` column is
present (`strand_oriented=True`).

## Input intervals DataFrame

Required columns: `chrom` (or `chr`), `start`, `end`. Optional: `strand`
(`"+"` / `"-"`).

- By default, `intervals.index` becomes `region_index` (must be unique).
- Pass `region_id_col="peak_id"` to use a column instead, or `use_coord_index=True` for `"chr1:100-200"` IDs.

## `eh.from_bigwig` (requires `pyenrichedheatmap[bigwig]`)

```python
import enriched_heatmap as eh

mat = eh.from_bigwig(
    "signal.bw",
    intervals_df,
    mode="reference_point",      # or "scale_regions"
    reference_point="center",    # "center" | "start" | "end" | "tss" | "tes"
    upstream=1000,
    downstream=1000,
    bin_size=10,
    target_bins=100,             # used only in scale_regions
    agg="mean",                  # "mean" | "max" | "min" | "sum" | "median"
    missing_value=0.0,           # in-bounds NaN replacement (use np.nan for methylation)
    out_of_bounds="nan",         # "nan" | "zero" | float for coords outside [0, chrom_len)
    chr_prefix="auto",           # reconciles "1" <-> "chr1"; "strict" requires exact match
    strand_oriented=True,
    n_jobs=4,                    # per-worker BigWig handles (thread-safe)
    name="ATAC-seq",
)
```

## `eh.from_pymisha` (requires `pyenrichedheatmap[pymisha]`)

```python
mat = eh.from_pymisha(
    "track_wt - track_input",    # any pymisha track expression or virtual track
    intervals_df,
    mode="reference_point",
    reference_point="tss",
    upstream=1000,
    downstream=1000,
    bin_size=10,
    missing_value=0.0,
    out_of_bounds="nan",
    chr_prefix="auto",
    strand_oriented=True,
    name="ChIP - Input",
)
```

- Overlapping anchor windows (e.g., nearby peaks with ±1 kb flanks) are
  automatically partitioned into sorted, strictly disjoint layers before calling
  `pymisha.gextract`, preserving exact input row order and avoiding canonicalization bugs.
- Coordinates extending past chromosome boundaries are clamped using
  `pymisha.gintervals_all()` and padded with `out_of_bounds`.

## Array, DataFrame & interchange loaders

- `eh.from_array(values, region_index=None, upstream=1000, downstream=1000, bin_size=None, ...)`
- `eh.from_dataframe(df, ...)` — preserves `df.index` as `region_index`
- `eh.from_interchange(obj, **overrides)` — accepts `eh.NormalizedMatrix`, `(values, metadata)` tuple, `dict`, `pd.DataFrame`, or `np.ndarray`

## Pitfalls

- **Minus-strand anchoring & asymmetric flanks:** When `strand == "-"` and
  `strand_oriented=True`, `reference_point="tss"` anchors at `end` (not `start`),
  extracts `[anchor - downstream, anchor + upstream)`, and reverses the bins so
  bin 0 is always 5' upstream. Set `strand_oriented=False` only for unoriented features.
- **`upstream` and `downstream` must be divisible by `bin_size`** in `eh.from_bigwig` and `eh.from_pymisha`.
- **`missing_value` vs `out_of_bounds`:** Coverage BigWigs store zero-read bases as NaN — pass `missing_value=0.0` for ChIP/ATAC/ChEC coverage, and leave `missing_value=np.nan` for ratio or CpG methylation tracks.
