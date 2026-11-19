# pymisha workflows

The eh bin table is a misha intervals frame. Use it as both scope and iterator:

```python
import pymisha as pm
import enriched_heatmap as eh

pm.gdb_init("/path/to/trackdb")
b = eh.bins(peaks, upstream=1000, downstream=1000, bin_size=20, genome=pm.gintervals_all())
vals = pm.gextract(["chec.msn2", "atac.wt"], intervals=b, iterator=b, colnames=["msn2", "atac"])
mats = eh.matrices(vals, b)
```

- `intervals=b, iterator=b` returns one row per bin with `intervalID`
  (1-based row of `b`). Overlapping windows are fine. Row order does not
  matter; `eh.matrices` places rows by `intervalID`, or by `region`/`bin`
  with `intervals_join="intervals"`. `intervals_join="none"` drops both and
  `eh.matrices` refuses it (rows come back genomically sorted).
- **Never** `iterator=<int>` with window intervals: misha tiles from the
  chromosome start, so bins misalign between regions and edge bins are
  partial.
- One `gextract` with all expressions (one genome pass), `colnames=` for
  names. Do not loop gextract per track.
- Per-bin aggregation is chosen by the extraction. A bare track name gives
  the plain mean of the stored bins each heatmap bin touches. For counts use
  a virtual track: `pm.gvtrack_create("chec_sum", "chec.msn2", func="sum")`;
  also `max`, `min`, `avg`, `pwm`, `distance`, with `sshift=`/`eshift=` in bp.
- Stored bins are **not weighted by overlap** (checked on pymisha 0.20): a
  heatmap bin covering 40 bp of one stored bin and 10 bp of the next gets
  their unweighted mean, and `sum` counts both in full. For fixed-bin tracks
  use `bin_size` = a multiple of the stored bin size and
  `eh.bins(..., snap=<stored bin size>)` so heatmap bins cover whole stored
  bins. Stored bin size: `pm.gtrack_info(track)["bin_size"]`.
- pymisha expressions are Python: NA fill is
  `"np.where(np.isnan(t), 0, t)"`, not R's `ifelse(is.na(...))`.
- Thresholds for peak calling come from `pm.gquantiles(expr,
  percentiles=0.99, intervals=pm.gintervals_all())`, not literals.
- `pm.gintervals_load("set")` strands are `1`/`-1`; `eh.bins` reads them.
- Pass `genome=pm.gintervals_all()`, otherwise windows past a chromosome end
  make gextract raise.
- Size: 20k regions x 200 bins = 4M bin rows; gextract takes seconds.
