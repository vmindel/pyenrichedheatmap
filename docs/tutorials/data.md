# Regions to matrices

A heatmap row is a region; a column is a bin at a fixed offset from an anchor. This
page covers how `eh.bins` places those bins, how values get into them, and what you
can do with the resulting matrices.

```python
import numpy as np
import pandas as pd
import enriched_heatmap as eh

genome = eh.example_genome()
peaks = eh.example_peaks()
```

## The bin table

`eh.bins` returns one row per bin. Coordinates are 0-based, half-open, as in BED and
misha. `region` is the row identifier (the DataFrame index, or `id=<column>`) and `bin`
the column number in the matrix.

```python
genes = pd.DataFrame(
    {"chrom": ["chr1", "chr1"], "start": [10_000, 20_000], "end": [15_000, 26_000],
     "strand": ["+", "-"]},
    index=["geneA", "geneB"],
)
b = eh.bins(genes, upstream=100, downstream=60, bin_size=20, anchor="start")
print(b.sort_values(["region", "bin"]).groupby("region").head(2))
```

```text
   chrom  start    end region  bin
0   chr1   9900   9920  geneA    0
1   chr1   9920   9940  geneA    1
15  chr1  26080  26100  geneB    0
14  chr1  26060  26080  geneB    1
```

- **Anchors.** `anchor="center"` (default), `"start"` or `"end"`. Start and end are
  strand-aware: `"start"` is the 5' end, so it is the TSS for a table of genes on
  either strand. The centre is `(start + end) // 2`.
- **Strand.** A `strand` column with `"+"`/`"-"` (BED) or `1`/`-1` (misha) is honoured;
  minus-strand bins are numbered from the genomic right, so `bin` 0 is always the most
  upstream bin and the matrix is oriented 5' to 3' with no flipping afterwards.
  `strand=None` ignores it.
- **Bins are relative to each anchor**, not to a genome-wide grid. That is why the
  table lists every bin explicitly instead of asking misha to tile with
  `iterator=20` (which tiles from chromosome start; see [With pymisha](pymisha.md)).
- **Edges.** Bins before position 0 are always dropped. With `genome=` (a dict or a
  table like `pm.gintervals_all()`) bins past a chromosome end are dropped too, and
  chromosome names are matched with or without the `chr` prefix. Dropped bins are NaN
  in the matrix; likewise, dropping rows from `b` masks only those bins to NaN while
  keeping every original region, so filter `regions` before `eh.bins` (or pass
  `index=` to `eh.arrange`) to subset regions.

## Scale-regions mode

`body_bins=` switches to EnrichedHeatmap's scale-regions layout: flanks in fixed
bins, each region body split into `body_bins` bins of equal fractional width.

```python
s = eh.bins(genes, upstream=1000, downstream=1000, bin_size=100, body_bins=20)
body = s[(s.region == "geneA") & s.bin.between(10, 29)]
print(body.head(3))
print("body bin widths:", sorted(set(body.end - body.start)))
```

```text
   chrom  start    end region  bin
10  chr1  10000  10250  geneA   10
11  chr1  10250  10500  geneA   11
12  chr1  10500  10750  geneA   12
body bin widths: [250]
```

Bodies shorter than `body_bins` bp are sampled at 1 bp positions.

## Filling values

Anything that returns one value per bin row works. The contract `eh.matrices` relies on
is the one pymisha's `gextract` already follows: an `intervalID` column giving the
1-based row of `bins` each value belongs to. `region` and `bin` columns work too (that
is what `gextract(..., intervals_join="intervals")` returns). A table with neither is
accepted only if its `start`/`end` match `bins` row for row; otherwise `eh.matrices`
raises rather than guess an order.

| source | call |
|---|---|
| misha database | `pm.gextract(exprs, intervals=b, iterator=b, colnames=[...])` |
| bigWig files | `eh.read_bigwig({"atac": "atac.bw", "chec": "chec.bw"}, b)` |
| your own code | any DataFrame with value columns and `intervalID` |
| example data | `eh.example_values(b, ["chec", "atac"])` |

`eh.read_bigwig` averages base-pair values within each bin (`agg="sum"`, `"max"` or
`"min"` for others) and reads one span per region, so overlapping windows cost nothing
extra. It needs `pip install "pyenrichedheatmap[bigwig]"`.

```python
b = eh.bins(peaks, upstream=1000, downstream=1000, bin_size=20, genome=genome)
mats = eh.matrices(eh.example_values(b, ["chec", "chec_ko", "atac"]), b)
print(list(mats))
```

```text
['chec', 'chec_ko', 'atac']
```

## The matrix object

A `NormalizedMatrix` is a 2D array plus its region index and bin geometry. Rows are
selected by label or position like a pandas object (a boolean Series is aligned to the
region index by label; use `.to_numpy()` for a positional mask), and every transform
returns a new matrix:

```python
m = mats["chec"]
print(m.shape, m.x_extent)
print(m.loc[["peak0003", "peak0000"]].region_index.tolist())
print(m.to_dataframe().iloc[:2, 48:52])
```

```text
(1500, 100) (-1000.0, 1000.0)
['peak0003', 'peak0000']
          -30.0  -10.0   10.0   30.0
name
peak0000  0.695  0.382  0.121  0.054
peak0001  1.624  0.297  1.604  0.269
```

| method | does |
|---|---|
| `.log1p()`, `.log2p(pseudocount)` | log transforms |
| `.zscore(axis=1)` | per-region (or per-bin with `axis=0`) z-score |
| `.clip_quantile(lo, hi)` | clip to global quantiles |
| `.smooth(sigma)` | Gaussian smoothing along bins; NaN bins stay NaN |
| `.log2_ratio(other)`, `.subtract(other)` | differential matrices, aligned by region |
| `.reindex(ids)`, `.dropna_regions()` | change the region set |

A differential matrix is drawn on a diverging scale centred on 0 by `eh.heatmap`
(pass `center=` or `cmap=` to override):

```python
diff = mats["chec_ko"].log2_ratio(mats["chec"], pseudocount=1.0)
print(diff.name)
```

```text
log2(chec_ko/chec)
```

## Matrices from arrays

If a matrix comes from elsewhere (deepTools, a model's predictions), build it
directly:

```python
values = np.random.default_rng(0).gamma(2.0, 1.0, size=(300, 50))
pred = eh.NormalizedMatrix(values, region_index=peaks.index[:300], upstream=500,
                           downstream=500, bin_size=20, name="predicted")
print(pred)
```

```text
NormalizedMatrix(shape=(300, 50), mode='reference_point', upstream=500, downstream=500, bin_size=20 name='predicted')
```

It aligns with the other matrices by region identifier when drawn with the same
`rows`.
