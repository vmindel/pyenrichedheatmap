"""Fill a per-bin table from bigWig files, in the same shape pymisha.gextract returns."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_AGGS = ("mean", "sum", "max", "min")


def _coerce_path(p: object, *, context: str = "`paths`") -> str:
    if isinstance(p, (str, os.PathLike)):
        fspath = os.fspath(p)
        if isinstance(fspath, str):
            return fspath
    raise TypeError(
        f"{context} must be a path (str or os.PathLike) or a non-empty mapping of "
        f"{{column_name: path}}, got {type(p).__name__}."
    )


def _open(path: str | os.PathLike[str]) -> Any:
    try:
        import pyBigWig
    except ImportError as exc:
        raise ImportError(
            "read_bigwig needs pyBigWig: pip install 'pyenrichedheatmap[bigwig]'"
        ) from exc
    return pyBigWig.open(_coerce_path(path))


def _resolve(name: str, chroms: dict[str, int]) -> str | None:
    if name in chroms:
        return name
    alt = name[3:] if name.startswith("chr") else f"chr{name}"
    return alt if alt in chroms else None


def _reduce(vals: np.ndarray, s: np.ndarray, e: np.ndarray, agg: str) -> np.ndarray:
    """Aggregate ``vals[s[i]:e[i]]`` for every bin, NaN-aware."""
    finite = np.isfinite(vals)
    if agg in ("mean", "sum"):
        cs = np.concatenate([[0.0], np.cumsum(np.where(finite, vals, 0.0))])
        cn = np.concatenate([[0], np.cumsum(finite)])
        tot, cnt = cs[e] - cs[s], cn[e] - cn[s]
        with np.errstate(invalid="ignore", divide="ignore"):
            res = tot / cnt if agg == "mean" else tot
        return np.where(cnt > 0, res, np.nan)
    fill = -np.inf if agg == "max" else np.inf
    filled = np.where(finite, vals, fill)
    out = np.array(
        [
            float(np.max(filled[a:b]) if agg == "max" else np.min(filled[a:b]))
            if b > a
            else fill
            for a, b in zip(s, e, strict=True)
        ]
    )
    return np.where(np.isfinite(out), out, np.nan)


def read_bigwig(
    paths: str | os.PathLike[str] | Mapping[str, str | os.PathLike[str]],
    bins: pd.DataFrame,
    agg: str = "mean",
) -> pd.DataFrame:
    """Read bigWig signal for every bin of a :func:`bins` table.

    Returns the same layout as ``pymisha.gextract(..., intervals=bins, iterator=bins)``:
    ``chrom, start, end, <one column per file>, intervalID``, so the result goes into
    :func:`matrices` the same way.

    Parameters
    ----------
    paths : str, os.PathLike, or Mapping of str to path
        One bigWig path, or a non-empty mapping keyed by the column name to give each.
    bins : pandas.DataFrame
        Output of :func:`bins`. Bins extending before ``0`` or past a chromosome end are
        left as ``NaN``, matching ``bins(..., genome=...)``.
    agg : {"mean", "sum", "max", "min"}
        How base-pair values are combined within a bin. ``"mean"`` ignores positions
        with no data; a bin with no data at all is NaN.
    """
    if agg not in _AGGS:
        raise ValueError(f"`agg` must be one of {_AGGS}, got {agg!r}.")
    if isinstance(paths, Mapping):
        if not paths:
            raise ValueError("`paths` mapping must not be empty.")
        path_map = {
            str(k): _coerce_path(v, context=f"`paths[{k!r}]`")
            for k, v in paths.items()
        }
    else:
        single = _coerce_path(paths)
        path_map = {Path(single).stem: single}

    out = bins[["chrom", "start", "end"]].copy()
    region_codes = pd.factorize(bins["region"])[0]
    starts = bins["start"].to_numpy(np.int64)
    ends = bins["end"].to_numpy(np.int64)
    chroms = bins["chrom"].to_numpy()
    order = np.argsort(region_codes, kind="stable")
    bounds = np.flatnonzero(np.diff(region_codes[order])) + 1
    groups = np.split(order, bounds) if len(order) else []

    for name, path in path_map.items():
        col = np.full(len(bins), np.nan)
        bw = _open(path)
        try:
            bw_chroms = bw.chroms()
            for idx in groups:
                c = _resolve(str(chroms[idx[0]]), bw_chroms)
                if c is None:
                    continue
                chrom_len = int(bw_chroms[c])
                b_starts = starts[idx]
                b_ends = ends[idx]
                valid = (b_starts >= 0) & (b_ends <= chrom_len) & (b_ends > b_starts)
                if not np.any(valid):
                    continue
                valid_idx = idx[valid]
                lo = int(b_starts[valid].min())
                hi = int(b_ends[valid].max())
                vals = np.asarray(bw.values(c, lo, hi, numpy=True), dtype=np.float64)
                s = b_starts[valid] - lo
                e = b_ends[valid] - lo
                col[valid_idx] = _reduce(vals, s, e, agg)
        finally:
            bw.close()
        out[name] = col
    out["intervalID"] = np.arange(1, len(bins) + 1)
    return out
