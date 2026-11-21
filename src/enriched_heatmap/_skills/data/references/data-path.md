# Regions to matrices

## eh.bins

```python
b = eh.bins(regions, upstream=1000, downstream=1000, bin_size=20,
            anchor="center", body_bins=None, genome=None, id=None,
            cols=("chrom", "start", "end"), strand="strand", snap=None)
# -> DataFrame chrom, start, end, region, bin (one row per bin)
```

- `upstream`/`downstream` must be multiples of `bin_size` (ValueError).
- `anchor`: `"center"` = `(start + end) // 2`; `"start"` = 5' end (TSS for
  genes: `start` on +, `end` on -); `"end"` = 3' end. No `"tss"` alias.
- Bins are anchored per region, **not** on a genome-wide grid.
- Minus-strand rows: `bin` 0 is the genomic right end. The matrix is oriented
  5' -> 3' already. `strand=None` treats all rows as +.
- `body_bins=n` -> scale-regions: `upstream//bin_size` flank bins, `n` body
  bins of equal fractional width, `downstream//bin_size` flank bins. Bodies
  shorter than `n` bp are sampled at 1 bp; zero-length bodies have no body bins.
- `genome=` (dict or a table like `pm.gintervals_all()`): bins past chromosome
  ends are dropped and region chromosome names are mapped with/without `chr`.
  A chromosome absent from `genome` raises KeyError: filter those regions
  first. Without `genome` only bins before 0 are dropped, and pymisha will
  error on bins past a chromosome end.
- Duplicate region identifiers raise ValueError; `reset_index()` or `id=`.
- `snap=n` rounds anchors down to a multiple of `n` (reference-point only), to
  line bins up with a fixed-bin track's stored grid (see `references/pymisha.md`).

## Filling values

Any DataFrame with one row per bin works if it has `region` + `bin` columns,
or `intervalID` (1-based row of `b`), or `start`/`end` identical to `b` row
for row. Anything else raises instead of guessing.

```python
vals = pm.gextract(exprs, intervals=b, iterator=b, colnames=names)  # misha
vals = eh.read_bigwig({"atac": "atac.bw", "chec": "chec.bw"}, b)      # bigWig
vals = eh.example_values(b, ["chec", "atac"])                         # toy data
```

`eh.read_bigwig(paths, bins, agg="mean")`: `agg` in mean/sum/max/min; mean
ignores bp with no data; a bin with none is NaN. Needs pyBigWig.

## eh.matrices / eh.matrix

```python
mats = eh.matrices(vals, b)              # dict column -> NormalizedMatrix
m = eh.matrix(vals, b, value="atac")     # one column; value optional if only one
```

- Value columns = every column except chrom/start/end/intervalID/region/bin.
- Rows follow the region order given to `eh.bins`; missing bins are NaN.
- "lost its layout" ValueError: `b` was rebuilt, merged/concatenated (pandas 2.x
  drops `attrs` there) or written to disk (CSV/parquet). Rebuild it with `eh.bins`.
- `eh.matrices(..., dtype=np.float64)` for raw counts above 16.7 million; the default
  float32 is exact only up to 2**24.

## NormalizedMatrix

Fields: `values` (n_regions, n_bins), `region_index`, `mode`
(`"reference_point"`/`"scale_regions"`), `upstream`, `downstream`,
`bin_size`, `target_bins`, `name`. Build one directly from an array with
`eh.NormalizedMatrix(values, region_index=ids, upstream=, downstream=, bin_size=)`;
the column count must equal `upstream/bin_size + downstream/bin_size` (plus
`target_bins`), and `subtract`/`log2_ratio` need identical geometry. Matrices
compare by identity (`==` is `is`).

Selection: `m.loc[ids]`, `m.iloc[pos]`, `m.reindex(ids)`, `m.dropna_regions()`.
Transforms (all return new matrices): `log1p`, `log2p(pc)`, `zscore(axis=1)`,
`clip_quantile(lo, hi)`, `smooth(sigma)` (NaN-aware, along bins),
`log2_ratio(other, pseudocount=1)`, `subtract(other)` (other aligned by
region). Geometry: `x_extent`, `bin_centers()`, `to_dataframe()`.
