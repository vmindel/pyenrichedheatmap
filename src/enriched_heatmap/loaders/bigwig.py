"""pyBigWig genomic signal loader supporting reference-point and scale-regions modes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from enriched_heatmap.loaders._intervals import (
    ReferencePoint,
    compute_reference_windows,
    compute_scale_windows,
    normalize_intervals,
)
from enriched_heatmap.matrix import Mode, NormalizedMatrix

AggFunc = Literal["mean", "max", "min", "sum", "median"]


def _aggregate_2d_bins(arr: np.ndarray, agg: AggFunc) -> np.ndarray:
    """Aggregate shape ``(n_bins, bin_size)`` along axis=1 safely with NaNs."""
    valid = np.any(np.isfinite(arr), axis=1)
    out = np.full(arr.shape[0], np.nan, dtype=np.float64)
    if not np.any(valid):
        return out
    sub = arr[valid]
    if agg == "mean":
        out[valid] = np.nanmean(sub, axis=1)
        return out
    if agg == "max":
        out[valid] = np.nanmax(sub, axis=1)
        return out
    if agg == "min":
        out[valid] = np.nanmin(sub, axis=1)
        return out
    if agg == "sum":
        out[valid] = np.nansum(sub, axis=1)
        return out
    if agg == "median":
        out[valid] = np.nanmedian(sub, axis=1)
        return out
    raise ValueError(
        f"Unsupported `agg` {agg!r}. Choose from 'mean', 'max', 'min', 'sum', 'median'."
    )


def _aggregate_1d_slice(arr: np.ndarray, agg: AggFunc) -> float:
    """Aggregate a 1D slice safely with NaNs."""
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    with np.errstate(invalid="ignore", all="ignore"):
        if agg == "mean":
            return float(np.nanmean(arr))
        if agg == "max":
            return float(np.nanmax(arr))
        if agg == "min":
            return float(np.nanmin(arr))
        if agg == "sum":
            return float(np.nansum(arr))
        if agg == "median":
            return float(np.nanmedian(arr))
    raise ValueError(f"Unsupported `agg` {agg!r}.")


def _resolve_chrom(
    chrom: str,
    bw_chroms: dict[str, int],
    chr_prefix: Literal["auto", "strict"],
) -> str:
    """Resolve chromosome name against BigWig header with optional chr-prefix adaptation."""
    if chrom in bw_chroms:
        return chrom
    if chr_prefix == "auto":
        alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
        if alt in bw_chroms:
            return alt
    sample = list(bw_chroms.keys())[:5]
    raise KeyError(
        f"Chromosome {chrom!r} not found in BigWig file (available examples: {sample}). "
        f"Set chr_prefix='auto' to automatically match 'chr' prefixes, or harmonize chromosome names."
    )


def _fetch_padded_bp(
    bw: Any,
    chrom: str,
    chrom_len: int,
    start: int,
    end: int,
    missing_value: float,
    oob_value: float,
) -> np.ndarray:
    """Extract 1bp values over ``[start, end)``, padding out-of-bounds with ``oob_value``."""
    length = int(end - start)
    if length <= 0:
        return np.empty(0, dtype=np.float64)

    buf = np.full(length, oob_value, dtype=np.float64)
    clamped_start = max(0, int(start))
    clamped_end = min(int(chrom_len), int(end))
    if clamped_end <= clamped_start:
        return buf

    vals = np.asarray(
        bw.values(chrom, clamped_start, clamped_end, numpy=True),
        dtype=np.float64,
    )
    if not np.isnan(missing_value):
        vals = np.where(np.isnan(vals), missing_value, vals)

    offset = clamped_start - int(start)
    buf[offset : offset + len(vals)] = vals
    return buf


def _bin_fixed_flank(
    bp_vals: np.ndarray, n_bins: int, bin_size: int, agg: AggFunc
) -> np.ndarray:
    """Bin a 1D bp array of length ``n_bins * bin_size`` into ``n_bins``."""
    if n_bins == 0:
        return np.empty(0, dtype=np.float64)
    return _aggregate_2d_bins(bp_vals.reshape(n_bins, bin_size), agg)


def _bin_variable_body(
    bp_vals: np.ndarray, target_bins: int, agg: AggFunc
) -> np.ndarray:
    """Bin a variable-length 1D bp array into ``target_bins`` equal fractional bins."""
    if target_bins == 0:
        return np.empty(0, dtype=np.float64)
    body_len = bp_vals.size
    if body_len == 0:
        return np.full(target_bins, np.nan, dtype=np.float64)

    if body_len == target_bins:
        return bp_vals.astype(np.float64, copy=False)

    if body_len % target_bins == 0:
        return _aggregate_2d_bins(bp_vals.reshape(target_bins, body_len // target_bins), agg)

    if body_len < target_bins:
        # Sample each target bin center from the nearest bp position
        centers = (np.arange(target_bins, dtype=np.float64) + 0.5) * (body_len / target_bins)
        idx = np.clip(np.floor(centers).astype(int), 0, body_len - 1)
        return bp_vals[idx]

    edges = np.linspace(0, body_len, target_bins + 1, dtype=int)
    out = np.empty(target_bins, dtype=np.float64)
    for b in range(target_bins):
        s, e = edges[b], edges[b + 1]
        if e <= s:
            e = min(s + 1, body_len)
        out[b] = _aggregate_1d_slice(bp_vals[s:e], agg)
    return out


def _open_bigwig(bw_path: str | Path) -> Any:
    try:
        import pyBigWig
    except ImportError as exc:
        raise ImportError(
            "pyBigWig is required for `from_bigwig`. "
            "Install it with: pip install 'pyenrichedheatmap[bigwig]'"
        ) from exc
    return pyBigWig.open(str(bw_path))


def from_bigwig(
    bw_path: str | Path,
    intervals: pd.DataFrame,
    mode: Mode = "reference_point",
    upstream: int = 1000,
    downstream: int = 1000,
    bin_size: int = 10,
    target_bins: int = 100,
    reference_point: ReferencePoint = "center",
    agg: AggFunc = "mean",
    missing_value: float = np.nan,
    out_of_bounds: Literal["nan", "zero"] | float = "nan",
    chr_prefix: Literal["auto", "strict"] = "auto",
    strand_oriented: bool = True,
    region_id_col: str | None = None,
    use_coord_index: bool = False,
    name: str = "",
    n_jobs: int = 1,
    dtype: Any = np.float32,
) -> NormalizedMatrix:
    """Extract a :class:`NormalizedMatrix` from a BigWig file via ``pyBigWig``.

    Parameters
    ----------
    bw_path : str | Path
        Path to the BigWig (``.bw`` / ``.bigWig``) file.
    intervals : pd.DataFrame
        0-based half-open ``[start, end)`` table with ``chrom`` (or ``chr``),
        ``start``, ``end``, and optional ``strand`` columns.
    mode : {"reference_point", "scale_regions"}, default "reference_point"
        Whether to align around a single reference point or scale each interval body.
    upstream, downstream : int, default 1000
        Flanking span in base pairs; must be divisible by ``bin_size``.
    bin_size : int, default 10
        Flank bin width in base pairs.
    target_bins : int, default 100
        Number of body bins when ``mode="scale_regions"``.
    reference_point : {"center", "start", "end", "tss", "tes"}, default "center"
        Anchor position when ``mode="reference_point"``.
    agg : {"mean", "max", "min", "sum", "median"}, default "mean"
        Aggregation function per bin.
    missing_value : float, default np.nan
        Replacement value for missing (NaN) BigWig entries inside chromosome bounds
        (e.g., ``0.0`` for coverage tracks, ``np.nan`` for CpG methylation).
    out_of_bounds : {"nan", "zero"} | float, default "nan"
        Fill value for coordinates extending past chromosome boundaries ``[0, chrom_len)``.
    chr_prefix : {"auto", "strict"}, default "auto"
        Whether to automatically reconcile ``chr1`` vs ``1`` naming mismatches.
    strand_oriented : bool, default True
        If True and a ``strand`` column is present, reverse minus-strand rows 5' -> 3'.
    name : str, default ""
        Track name (defaults to BigWig filename stem if empty).
    n_jobs : int, default 1
        Number of parallel worker threads (each opens its own BigWig file handle).
    """
    if bin_size <= 0:
        raise ValueError("`bin_size` must be positive.")
    if upstream < 0 or downstream < 0:
        raise ValueError("`upstream` and `downstream` must be non-negative.")
    if upstream % bin_size != 0 or downstream % bin_size != 0:
        raise ValueError(
            f"`upstream` ({upstream}) and `downstream` ({downstream}) must be divisible "
            f"by `bin_size` ({bin_size})."
        )
    if mode == "scale_regions" and target_bins <= 0:
        raise ValueError("`target_bins` must be > 0 for mode='scale_regions'.")
    if agg not in ("mean", "max", "min", "sum", "median"):
        raise ValueError(f"Unsupported `agg` {agg!r}.")

    if out_of_bounds == "nan":
        oob_val = float("nan")
    elif out_of_bounds == "zero":
        oob_val = 0.0
    else:
        oob_val = float(out_of_bounds)

    norm = normalize_intervals(
        intervals, region_id_col=region_id_col, use_coord_index=use_coord_index
    )
    n_regions = len(norm)
    n_up_bins = upstream // bin_size
    n_down_bins = downstream // bin_size
    n_body_bins = target_bins if mode == "scale_regions" else 0
    n_bins = n_up_bins + n_body_bins + n_down_bins

    resolved_name = name or Path(bw_path).stem

    if n_regions == 0:
        return NormalizedMatrix(
            values=np.empty((0, n_bins), dtype=dtype),
            region_index=norm.region_index,
            mode=mode,
            upstream=upstream,
            downstream=downstream,
            bin_size=bin_size,
            target_bins=n_body_bins,
            name=resolved_name,
            strand_oriented=strand_oriented,
            metadata={"source": "bigwig", "path": str(bw_path), "agg": agg},
        )

    # Validate chromosome names upfront with a single header read
    bw_probe = _open_bigwig(bw_path)
    try:
        bw_chroms: dict[str, int] = dict(bw_probe.chroms())
    finally:
        bw_probe.close()

    unique_chroms = np.unique(norm.chrom)
    chrom_map = {c: _resolve_chrom(c, bw_chroms, chr_prefix) for c in unique_chroms}

    if mode == "reference_point":
        win_start, win_end, flip = compute_reference_windows(
            norm,
            upstream=upstream,
            downstream=downstream,
            reference_point=reference_point,
            strand_oriented=strand_oriented,
        )

        def _process_ref_chunk(indices: np.ndarray) -> np.ndarray:
            out = np.empty((len(indices), n_bins), dtype=np.float64)
            bw = _open_bigwig(bw_path)
            try:
                for local_i, r_i in enumerate(indices):
                    c = chrom_map[norm.chrom[r_i]]
                    bp = _fetch_padded_bp(
                        bw,
                        c,
                        bw_chroms[c],
                        int(win_start[r_i]),
                        int(win_end[r_i]),
                        missing_value=missing_value,
                        oob_value=oob_val,
                    )
                    row = _bin_fixed_flank(bp, n_bins, bin_size, agg)
                    if flip[r_i]:
                        row = row[::-1]
                    out[local_i] = row
            finally:
                bw.close()
            return out

        worker_fn = _process_ref_chunk
    else:
        left_start, body_start, body_end, right_end, flip = compute_scale_windows(
            norm,
            upstream=upstream,
            downstream=downstream,
            strand_oriented=strand_oriented,
        )

        def _process_scale_chunk(indices: np.ndarray) -> np.ndarray:
            out = np.empty((len(indices), n_bins), dtype=np.float64)
            bw = _open_bigwig(bw_path)
            try:
                for local_i, r_i in enumerate(indices):
                    c = chrom_map[norm.chrom[r_i]]
                    clen = bw_chroms[c]
                    is_flip = bool(flip[r_i])
                    n_left_bins = n_down_bins if is_flip else n_up_bins
                    n_right_bins = n_up_bins if is_flip else n_down_bins

                    left_bp = _fetch_padded_bp(
                        bw, c, clen, int(left_start[r_i]), int(body_start[r_i]),
                        missing_value=missing_value, oob_value=oob_val,
                    )
                    body_bp = _fetch_padded_bp(
                        bw, c, clen, int(body_start[r_i]), int(body_end[r_i]),
                        missing_value=missing_value, oob_value=oob_val,
                    )
                    right_bp = _fetch_padded_bp(
                        bw, c, clen, int(body_end[r_i]), int(right_end[r_i]),
                        missing_value=missing_value, oob_value=oob_val,
                    )

                    left_binned = _bin_fixed_flank(left_bp, n_left_bins, bin_size, agg)
                    body_binned = _bin_variable_body(body_bp, target_bins, agg)
                    right_binned = _bin_fixed_flank(right_bp, n_right_bins, bin_size, agg)
                    row = np.concatenate([left_binned, body_binned, right_binned])
                    if is_flip:
                        row = row[::-1]
                    out[local_i] = row
            finally:
                bw.close()
            return out

        worker_fn = _process_scale_chunk

    all_indices = np.arange(n_regions, dtype=int)
    n_workers = max(1, int(n_jobs))
    if n_workers == 1 or n_regions < 64:
        values = worker_fn(all_indices)
    else:
        chunks = [c for c in np.array_split(all_indices, n_workers) if len(c) > 0]
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            parts = list(pool.map(worker_fn, chunks))
        values = np.vstack(parts)

    return NormalizedMatrix(
        values=values.astype(dtype, copy=False),
        region_index=norm.region_index,
        mode=mode,
        upstream=upstream,
        downstream=downstream,
        bin_size=bin_size,
        target_bins=n_body_bins,
        name=resolved_name,
        strand_oriented=strand_oriented,
        metadata={
            "source": "bigwig",
            "path": str(bw_path),
            "agg": agg,
            "reference_point": reference_point if mode == "reference_point" else None,
        },
    )
