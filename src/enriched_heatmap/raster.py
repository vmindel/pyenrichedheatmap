"""Smart anti-aliased row downsampling for rendering 10k-100k+ genomic regions cleanly."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

DownsampleAgg = (
    Literal["mean", "max", "min", "median", "q90", "q95", "absmax"]
    | str
    | Callable[[np.ndarray], np.ndarray]
)


def _bucket_edges(n_rows: int, target_rows: int) -> np.ndarray:
    """Return strictly increasing integer bucket boundaries of length ``target_rows + 1``."""
    edges = np.linspace(0, n_rows, target_rows + 1, dtype=np.int64)
    return edges


def _parse_quantile_agg(agg: str) -> float | None:
    """Parse ``"q90"`` -> ``0.90``, ``"q95"`` -> ``0.95``, etc."""
    if len(agg) >= 2 and agg[0] in ("q", "Q"):
        try:
            q = float(agg[1:])
        except ValueError:
            return None
        if 0.0 <= q <= 1.0:
            return q
        if 1.0 < q <= 100.0:
            return q / 100.0
    return None


def downsample_rows(
    data: np.ndarray,
    target_rows: int,
    agg: DownsampleAgg = "mean",
) -> np.ndarray:
    """Downsample a 2D array along axis 0 into ``target_rows`` buckets.

    If ``data.shape[0] <= target_rows`` or ``target_rows <= 0``, returns ``data``
    unchanged so small matrices retain exact single-row fidelity.

    Parameters
    ----------
    data : np.ndarray
        2D numeric array of shape ``(n_rows, n_cols)``.
    target_rows : int
        Desired output row count (typically matched to physical pixel height at ``raster_dpi``).
    agg : str | Callable, default "mean"
        Aggregation mode per row bucket:
        - ``"mean"``: NaN-safe bucket average (fast C-level ``reduceat``, ideal for ATAC/MNase).
        - ``"max"``: NaN-safe bucket maximum (preserves sparse narrow TF ChEC peaks).
        - ``"min"``: NaN-safe bucket minimum.
        - ``"absmax"``: Signed value with largest magnitude (ideal for differential log2FC).
        - ``"median"``: NaN-safe bucket median.
        - ``"q90"``, ``"q95"``, ``"qXX"``: NaN-safe bucket quantile (e.g., 90th percentile).
        - Callable ``fn(chunk_2d) -> row_1d`` for custom aggregation.
    """
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"`data` must be 2D, got shape {arr.shape}.")

    n_rows, n_cols = arr.shape
    if target_rows <= 0 or n_rows <= target_rows or n_rows == 0:
        return arr

    edges = _bucket_edges(n_rows, target_rows)
    starts = edges[:-1]
    ends = edges[1:]

    if callable(agg):
        out = np.empty((target_rows, n_cols), dtype=np.float64)
        for i in range(target_rows):
            out[i] = agg(arr[starts[i] : ends[i]])
        return out.astype(arr.dtype, copy=False)

    agg_key = str(agg).lower()
    arr_f = arr.astype(np.float64, copy=False)

    if agg_key == "mean":
        finite = np.isfinite(arr_f)
        filled = np.where(finite, arr_f, 0.0)
        sums = np.add.reduceat(filled, starts, axis=0)
        counts = np.add.reduceat(finite.astype(np.int32), starts, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        return out.astype(arr.dtype, copy=False)

    if agg_key == "max":
        finite = np.isfinite(arr_f)
        filled = np.where(finite, arr_f, -np.inf)
        mx = np.maximum.reduceat(filled, starts, axis=0)
        counts = np.add.reduceat(finite.astype(np.int32), starts, axis=0)
        out = np.where(counts > 0, mx, np.nan)
        return out.astype(arr.dtype, copy=False)

    if agg_key == "min":
        finite = np.isfinite(arr_f)
        filled = np.where(finite, arr_f, np.inf)
        mn = np.minimum.reduceat(filled, starts, axis=0)
        counts = np.add.reduceat(finite.astype(np.int32), starts, axis=0)
        out = np.where(counts > 0, mn, np.nan)
        return out.astype(arr.dtype, copy=False)

    if agg_key == "absmax":
        finite = np.isfinite(arr_f)
        pos_filled = np.where(finite, arr_f, -np.inf)
        neg_filled = np.where(finite, arr_f, np.inf)
        mx = np.maximum.reduceat(pos_filled, starts, axis=0)
        mn = np.minimum.reduceat(neg_filled, starts, axis=0)
        counts = np.add.reduceat(finite.astype(np.int32), starts, axis=0)
        chosen = np.where(np.abs(mx) >= np.abs(mn), mx, mn)
        out = np.where(counts > 0, chosen, np.nan)
        return out.astype(arr.dtype, copy=False)

    if agg_key == "median":
        out = np.full((target_rows, n_cols), np.nan, dtype=np.float64)
        with np.errstate(invalid="ignore", all="ignore"):
            for i in range(target_rows):
                out[i] = np.nanmedian(arr_f[starts[i] : ends[i]], axis=0)
        return out.astype(arr.dtype, copy=False)

    q = _parse_quantile_agg(agg_key)
    if q is not None:
        out = np.full((target_rows, n_cols), np.nan, dtype=np.float64)
        with np.errstate(invalid="ignore", all="ignore"):
            for i in range(target_rows):
                out[i] = np.nanquantile(arr_f[starts[i] : ends[i]], q, axis=0)
        return out.astype(arr.dtype, copy=False)

    raise ValueError(
        f"Unsupported downsample_agg {agg!r}. Choose from "
        "'mean', 'max', 'min', 'absmax', 'median', 'qXX' (e.g. 'q90'), or a callable."
    )


def downsample_1d_rows(
    values: np.ndarray,
    target_rows: int,
    agg: DownsampleAgg = "mean",
) -> np.ndarray:
    """Downsample a 1D row annotation vector into ``target_rows`` buckets."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"`values` must be 1D, got shape {arr.shape}.")
    if target_rows <= 0 or arr.size <= target_rows:
        return arr
    return downsample_rows(arr[:, None], target_rows=target_rows, agg=agg)[:, 0]


def downsample_categorical_rows(
    codes: np.ndarray,
    target_rows: int,
) -> np.ndarray:
    """Downsample 1D integer category codes (``-1`` = missing) by bucket mode."""
    arr = np.asarray(codes, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"`codes` must be 1D, got shape {arr.shape}.")
    n_rows = arr.size
    if target_rows <= 0 or n_rows <= target_rows or n_rows == 0:
        return arr

    edges = _bucket_edges(n_rows, target_rows)
    out = np.full(target_rows, -1, dtype=np.int64)
    for i in range(target_rows):
        chunk = arr[edges[i] : edges[i + 1]]
        valid = chunk[chunk >= 0]
        if valid.size == 0:
            continue
        vals, counts = np.unique(valid, return_counts=True)
        out[i] = vals[int(np.argmax(counts))]
    return out
