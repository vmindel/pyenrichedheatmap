# Genomic loaders

All genomic loaders enforce **0-based, half-open `[start, end)`** coordinates
and automatic minus-strand **5' → 3' flipping** whenever a `strand` column is
present.

## Interval table format

Pass a pandas `DataFrame` with `chrom` (or `chr`), `start`, `end`, and optional
`strand` (`"+"` / `"-"`):

```python
import pandas as pd

genes = pd.DataFrame(
    {
        "chrom": ["chr1", "chr1", "chr2"],
        "start": [10_000, 50_000, 80_000],
        "end":   [14_000, 56_000, 85_000],
        "strand": ["+", "-", "+"],
    },
    index=["SOX2", "NANOG", "OCT4"],  # becomes region_index!
)
```

!!! note "Strand orientation & asymmetric flanks"
    When `strand == "-"` and `strand_oriented=True` (default):

    - `reference_point="tss"` anchors at `end` (the biological 5' start in 0-based `[start, end)`).
    - Asymmetric flanks (`upstream=2000, downstream=500`) extract `[anchor - 500, anchor + 2000)` and reverse the binned array so bin 0 is always `-2 kb` upstream and the last bin is `+500 bp` downstream.

## Loading from BigWig (`from_bigwig`)

Install the optional extra with `pip install "pyenrichedheatmap[bigwig]"`.

```python
import enriched_heatmap as eh

# Reference-point around TSS
atac_tss = eh.from_bigwig(
    "atac.bw",
    genes,
    mode="reference_point",
    reference_point="tss",
    upstream=1000,
    downstream=1000,
    bin_size=10,
    agg="mean",
    missing_value=0.0,       # replace in-bounds NaNs with 0 for coverage tracks
    out_of_bounds="nan",     # pad coordinates outside [0, chrom_len) with NaN
    chr_prefix="auto",       # auto-reconcile "1" <-> "chr1"
    n_jobs=4,
    name="ATAC-seq",
)

# Scale-regions across full gene bodies (TSS -> TES)
pol2_body = eh.from_bigwig(
    "pol2.bw",
    genes,
    mode="scale_regions",
    upstream=1000,
    downstream=1000,
    bin_size=50,
    target_bins=60,
    missing_value=0.0,
    name="Pol II",
)
```

## Loading from `pymisha` (`from_pymisha`)

Install the optional extra with `pip install "pyenrichedheatmap[pymisha]"`.

`from_pymisha` supports arbitrary `pymisha` track expressions and virtual
tracks, automatically clamps coordinates to `gintervals_all()`, and partitions
overlapping anchor windows into strictly disjoint layers before calling
`gextract`:

```python
import pymisha as pm
import enriched_heatmap as eh

pm.gdb_init("/path/to/misha_db")
pm.gvtrack_create("v_chec_shift", "tf_chec_raw", func="sum")
pm.gvtrack_iterator("v_chec_shift", sshift=-15, eshift=15)

chec = eh.from_pymisha(
    "v_chec_shift",
    peaks_df,
    mode="reference_point",
    reference_point="center",
    upstream=1000,
    downstream=1000,
    bin_size=10,
    missing_value=0.0,
    name="TF ChEC (<120bp)",
)

# Arbitrary track math expressions work directly
chip_norm = eh.from_pymisha(
    "log2((chip_track + 1) / (input_track + 1))",
    peaks_df,
    upstream=1000,
    downstream=1000,
    bin_size=10,
    name="ChIP / Input",
)
```

## Arrays, DataFrames & interchange handoff

For matrices computed by external tools (`moodspy`, ChromBPNet, Fragment
Geometry pipelines):

```python
# From a 2D NumPy array or pandas DataFrame
mat = eh.from_dataframe(precomputed_df, upstream=500, downstream=500, name="Motif hits")

# Zero-coupling interchange tuple: (values_2d, metadata_dict)
mat = eh.from_interchange((values_2d, {"region_index": ids, "upstream": 1000, "downstream": 1000, "bin_size": 10}))
```
