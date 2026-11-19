"""Shared genomic interval normalization and strand-aware window calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

ReferencePoint = Literal["center", "start", "end", "tss", "tes"]


@dataclass
class NormalizedIntervals:
    """Standardized 0-based half-open genomic intervals."""

    chrom: np.ndarray
    start: np.ndarray
    end: np.ndarray
    is_minus: np.ndarray
    region_index: pd.Index

    def __len__(self) -> int:
        return len(self.chrom)


def normalize_intervals(
    intervals: pd.DataFrame,
    region_id_col: str | None = None,
    use_coord_index: bool = False,
) -> NormalizedIntervals:
    """Validate and standardize a genomic interval DataFrame.

    Expects 0-based, half-open ``[start, end)`` intervals with ``chrom`` (or ``chr``),
    ``start``, ``end``, and optional ``strand`` columns.
    """
    if not isinstance(intervals, pd.DataFrame):
        raise TypeError(
            f"`intervals` must be a pandas DataFrame, got {type(intervals).__name__}."
        )

    cols = set(intervals.columns)
    if "chrom" in cols:
        chrom_col = "chrom"
    elif "chr" in cols:
        chrom_col = "chr"
    else:
        raise ValueError("`intervals` DataFrame must contain a 'chrom' (or 'chr') column.")

    if "start" not in cols or "end" not in cols:
        raise ValueError("`intervals` DataFrame must contain 'start' and 'end' columns.")

    chrom = intervals[chrom_col].astype(str).to_numpy()
    start = intervals["start"].to_numpy(dtype=np.int64)
    end = intervals["end"].to_numpy(dtype=np.int64)

    if np.any(end < start):
        raise ValueError("Found intervals with end < start; expected 0-based [start, end).")

    if "strand" in cols:
        strand_raw = intervals["strand"].astype(str).str.strip().to_numpy()
        is_minus = strand_raw == "-"
    else:
        is_minus = np.zeros(len(intervals), dtype=bool)

    if region_id_col is not None:
        if region_id_col not in cols:
            raise ValueError(f"`region_id_col` {region_id_col!r} not found in intervals.")
        region_index = pd.Index(intervals[region_id_col])
    elif use_coord_index:
        coords = [f"{c}:{s}-{e}" for c, s, e in zip(chrom, start, end, strict=True)]
        region_index = pd.Index(coords)
    else:
        region_index = intervals.index.copy()

    if not region_index.is_unique:
        raise ValueError(
            "Derived `region_index` is not unique. Pass a DataFrame with a unique index "
            "or specify `region_id_col`."
        )

    return NormalizedIntervals(
        chrom=chrom,
        start=start,
        end=end,
        is_minus=is_minus,
        region_index=region_index,
    )


def compute_reference_windows(
    norm: NormalizedIntervals,
    upstream: int,
    downstream: int,
    reference_point: ReferencePoint = "center",
    strand_oriented: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute genomic ``[win_start, win_end)`` and ``flip_mask`` for reference-point mode.

    On minus-strand regions (when ``strand_oriented=True``):
    - ``tss`` / ``start`` anchors at ``end``; ``tes`` / ``end`` anchors at ``start``.
    - 5' upstream is to the genomic right (``+upstream``) and 3' downstream is to the
      genomic left (``-downstream``), so the genomic window is
      ``[anchor - downstream, anchor + upstream)`` and the extracted bins must be reversed.
    """
    if reference_point not in ("center", "start", "end", "tss", "tes"):
        raise ValueError(
            f"`reference_point` must be one of 'center', 'start', 'end', 'tss', 'tes', "
            f"got {reference_point!r}."
        )

    flip = norm.is_minus if strand_oriented else np.zeros(len(norm), dtype=bool)

    if reference_point == "center":
        anchor = (norm.start + norm.end) // 2
    elif reference_point in ("start", "tss"):
        anchor = np.where(flip, norm.end, norm.start)
    else:  # "end", "tes"
        anchor = np.where(flip, norm.start, norm.end)

    win_start = np.where(flip, anchor - downstream, anchor - upstream)
    win_end = np.where(flip, anchor + upstream, anchor + downstream)
    return win_start, win_end, flip


def compute_scale_windows(
    norm: NormalizedIntervals,
    upstream: int,
    downstream: int,
    strand_oriented: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute genomic left-flank, body, and right-flank boundaries + ``flip_mask``.

    For minus-strand regions (when ``strand_oriented=True``):
    - Genomic left flank has length ``downstream`` (becomes 3' downstream after reversal).
    - Body is ``[start, end)`` (reversed to 5' -> 3').
    - Genomic right flank has length ``upstream`` (becomes 5' upstream after reversal).
    """
    flip = norm.is_minus if strand_oriented else np.zeros(len(norm), dtype=bool)
    left_len = np.where(flip, downstream, upstream)
    right_len = np.where(flip, upstream, downstream)

    left_start = norm.start - left_len
    body_start = norm.start
    body_end = norm.end
    right_end = norm.end + right_len
    return left_start, body_start, body_end, right_end, flip
