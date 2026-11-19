"""pymisha genomic signal loader supporting arbitrary track expressions and virtual tracks."""

from __future__ import annotations

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


def _import_pymisha() -> Any:
    try:
        import pymisha as pm
    except ImportError as exc:
        raise ImportError(
            "pymisha is required for `from_pymisha`. "
            "Install it with: pip install 'pyenrichedheatmap[pymisha]'"
        ) from exc
    return pm


def _get_chrom_sizes(pm: Any) -> dict[str, int]:
    """Query chromosome lengths from the active pymisha database."""
    if not hasattr(pm, "gintervals_all"):
        return {}
    all_iv = pm.gintervals_all()
    if all_iv is None or len(all_iv) == 0:
        return {}
    chrom_col = "chrom" if "chrom" in all_iv.columns else "chr"
    return {
        str(c): int(e)
        for c, e in zip(all_iv[chrom_col], all_iv["end"], strict=True)
    }


def _resolve_chrom(
    chrom: str,
    db_chroms: dict[str, int],
    chr_prefix: Literal["auto", "strict"],
) -> str:
    if not db_chroms or chrom in db_chroms:
        return chrom
    if chr_prefix == "auto":
        alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
        if alt in db_chroms:
            return alt
    sample = list(db_chroms.keys())[:5]
    raise KeyError(
        f"Chromosome {chrom!r} not found in pymisha DB (available examples: {sample})."
    )


def _partition_non_overlapping(
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    valid_indices: np.ndarray,
) -> list[np.ndarray]:
    """Greedy interval coloring into sorted, strictly non-overlapping layers.

    Guarantees that each layer passed to ``pymisha.gextract`` contains canonical
    (sorted, disjoint) intervals even when input anchor windows overlap.
    """
    if len(valid_indices) == 0:
        return []

    order = sorted(
        valid_indices.tolist(),
        key=lambda idx: (str(chroms[idx]), int(starts[idx]), int(ends[idx])),
    )
    layers: list[list[int]] = []
    layer_last_chrom: list[str] = []
    layer_last_end: list[int] = []

    for idx in order:
        c = str(chroms[idx])
        s = int(starts[idx])
        e = int(ends[idx])
        placed = False
        for l_i in range(len(layers)):
            if layer_last_chrom[l_i] != c or layer_last_end[l_i] <= s:
                layers[l_i].append(idx)
                layer_last_chrom[l_i] = c
                layer_last_end[l_i] = e
                placed = True
                break
        if not placed:
            layers.append([idx])
            layer_last_chrom.append(c)
            layer_last_end.append(e)

    return [np.asarray(layer, dtype=int) for layer in layers]


def _extract_value_column(df: pd.DataFrame) -> str:
    reserved = {"chrom", "chr", "start", "end", "intervalID"}
    candidates = [col for col in df.columns if col not in reserved]
    if not candidates:
        raise ValueError(
            f"Could not identify signal value column in pymisha.gextract output: {list(df.columns)}"
        )
    return candidates[0]


def _normalize_interval_ids(ids: np.ndarray, n_intervals: int) -> np.ndarray:
    """Convert pymisha 1-based intervalID (or 0-based in custom mocks) to 0-based indices."""
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size == 0:
        return arr
    if arr.min() >= 1 and arr.max() <= n_intervals:
        return arr - 1
    return arr


def from_pymisha(
    track_expr: str,
    intervals: pd.DataFrame,
    mode: Mode = "reference_point",
    upstream: int = 1000,
    downstream: int = 1000,
    bin_size: int = 10,
    target_bins: int = 100,
    reference_point: ReferencePoint = "center",
    missing_value: float = np.nan,
    out_of_bounds: Literal["nan", "zero"] | float = "nan",
    chr_prefix: Literal["auto", "strict"] = "auto",
    strand_oriented: bool = True,
    region_id_col: str | None = None,
    use_coord_index: bool = False,
    name: str = "",
    dtype: Any = np.float32,
    **gextract_kwargs: Any,
) -> NormalizedMatrix:
    """Extract a :class:`NormalizedMatrix` using ``pymisha.gextract``.

    Supports arbitrary ``pymisha`` track expressions (e.g., ``"track_a - track_b"``,
    virtual tracks created with ``gvtrack_create``, shifted cut-site tracks, PWM tracks).
    Overlapping region windows are automatically partitioned into disjoint layers so
    ``gextract`` preserves exact row order and ``intervalID`` mapping.

    Parameters
    ----------
    track_expr : str
        Any valid ``pymisha`` track expression or virtual track name.
    intervals : pd.DataFrame
        0-based half-open ``[start, end)`` table with ``chrom`` (or ``chr``),
        ``start``, ``end``, and optional ``strand`` columns.
    mode : {"reference_point", "scale_regions"}, default "reference_point"
        Coordinate alignment mode.
    upstream, downstream : int, default 1000
        Flanking span in base pairs; must be divisible by ``bin_size``.
    bin_size : int, default 10
        Flank bin width in base pairs (passed as ``iterator`` in ``reference_point`` mode).
    target_bins : int, default 100
        Number of body bins when ``mode="scale_regions"``.
    reference_point : {"center", "start", "end", "tss", "tes"}, default "center"
        Anchor position when ``mode="reference_point"``.
    missing_value : float, default np.nan
        Replacement value for NaNs returned by ``gextract`` inside chromosome bounds.
    out_of_bounds : {"nan", "zero"} | float, default "nan"
        Fill value for coordinates extending past chromosome boundaries.
    chr_prefix : {"auto", "strict"}, default "auto"
        Reconcile ``chr1`` vs ``1`` naming against ``gintervals_all()``.
    strand_oriented : bool, default True
        Reverse minus-strand rows 5' -> 3' when a ``strand`` column is present.
    name : str, default ""
        Track name (defaults to ``track_expr``).
    **gextract_kwargs : Any
        Additional keyword arguments forwarded to ``pymisha.gextract``.
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

    if out_of_bounds == "nan":
        oob_val = float("nan")
    elif out_of_bounds == "zero":
        oob_val = 0.0
    else:
        oob_val = float(out_of_bounds)

    pm = _import_pymisha()
    norm = normalize_intervals(
        intervals, region_id_col=region_id_col, use_coord_index=use_coord_index
    )
    n_regions = len(norm)
    n_up_bins = upstream // bin_size
    n_down_bins = downstream // bin_size
    n_body_bins = target_bins if mode == "scale_regions" else 0
    n_bins = n_up_bins + n_body_bins + n_down_bins
    resolved_name = name or track_expr

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
            metadata={"source": "pymisha", "track_expr": track_expr},
        )

    db_chroms = _get_chrom_sizes(pm)
    mapped_chroms = np.array(
        [_resolve_chrom(c, db_chroms, chr_prefix) for c in norm.chrom],
        dtype=object,
    )

    out = np.full((n_regions, n_bins), oob_val, dtype=np.float64)

    if mode == "reference_point":
        win_start, win_end, flip = compute_reference_windows(
            norm,
            upstream=upstream,
            downstream=downstream,
            reference_point=reference_point,
            strand_oriented=strand_oriented,
        )

        # Clamp windows to exact bin-aligned in-bounds spans
        q_start = np.empty(n_regions, dtype=np.int64)
        q_end = np.empty(n_regions, dtype=np.int64)
        valid_mask = np.zeros(n_regions, dtype=bool)

        for i in range(n_regions):
            c = str(mapped_chroms[i])
            clen = db_chroms.get(c, None)
            ws = int(win_start[i])
            we = int(win_end[i])

            # Advance start by whole bins if < 0
            if ws < 0:
                skip_bins = (-ws + bin_size - 1) // bin_size
                cs = ws + skip_bins * bin_size
            else:
                cs = ws

            ce = min(we, clen) if clen is not None and we > clen else we

            if ce > cs:
                q_start[i] = cs
                q_end[i] = ce
                valid_mask[i] = True
            else:
                q_start[i] = 0
                q_end[i] = 0

        valid_indices = np.flatnonzero(valid_mask)
        layers = _partition_non_overlapping(mapped_chroms, q_start, q_end, valid_indices)

        for layer in layers:
            layer_df = pd.DataFrame(
                {
                    "chrom": mapped_chroms[layer].astype(str),
                    "start": q_start[layer],
                    "end": q_end[layer],
                }
            )
            extracted = pm.gextract(
                track_expr,
                layer_df,
                iterator=gextract_kwargs.pop("iterator", bin_size)
                if "iterator" in gextract_kwargs
                else bin_size,
                **gextract_kwargs,
            )
            if extracted is None or len(extracted) == 0:
                continue

            val_col = _extract_value_column(extracted)
            local_ids = _normalize_interval_ids(
                extracted["intervalID"].to_numpy(), len(layer)
            )
            ext_starts = extracted["start"].to_numpy(dtype=np.int64)
            ext_vals = extracted[val_col].to_numpy(dtype=np.float64)
            if not np.isnan(missing_value):
                ext_vals = np.where(np.isnan(ext_vals), missing_value, ext_vals)

            region_ids = layer[local_ids]
            bin_idx = (ext_starts - win_start[region_ids]) // bin_size
            in_range = (bin_idx >= 0) & (bin_idx < n_bins)
            out[region_ids[in_range], bin_idx[in_range]] = ext_vals[in_range]

        # Flip minus-strand rows 5' -> 3'
        if np.any(flip):
            out[flip] = out[flip, ::-1]
    else:
        left_start, body_start, body_end, right_end, flip = compute_scale_windows(
            norm,
            upstream=upstream,
            downstream=downstream,
            strand_oriented=strand_oriented,
        )

        # Build explicit per-bin intervals for every region
        bin_chroms: list[str] = []
        bin_starts: list[int] = []
        bin_ends: list[int] = []
        bin_region_i: list[int] = []
        bin_col_j: list[int] = []

        for i in range(n_regions):
            c = str(mapped_chroms[i])
            clen = db_chroms.get(c, None)
            is_flip = bool(flip[i])
            n_left = n_down_bins if is_flip else n_up_bins
            n_right = n_up_bins if is_flip else n_down_bins

            ls = int(left_start[i])
            bs = int(body_start[i])
            be = int(body_end[i])

            # Left flank bins
            col = 0
            for b in range(n_left):
                s = ls + b * bin_size
                e = s + bin_size
                if s >= 0 and (clen is None or e <= clen):
                    bin_chroms.append(c)
                    bin_starts.append(s)
                    bin_ends.append(e)
                    bin_region_i.append(i)
                    bin_col_j.append(col)
                col += 1

            # Body bins (fractional scaling over [bs, be))
            body_len = be - bs
            if body_len > 0:
                if body_len >= target_bins:
                    edges = np.round(np.linspace(bs, be, target_bins + 1)).astype(np.int64)
                    for b in range(target_bins):
                        s = int(edges[b])
                        e = int(max(edges[b + 1], s + 1))
                        if s >= 0 and (clen is None or e <= clen):
                            bin_chroms.append(c)
                            bin_starts.append(s)
                            bin_ends.append(e)
                            bin_region_i.append(i)
                            bin_col_j.append(col)
                        col += 1
                else:
                    centers = bs + (np.arange(target_bins, dtype=np.float64) + 0.5) * (
                        body_len / target_bins
                    )
                    pos = np.clip(np.floor(centers).astype(np.int64), bs, be - 1)
                    for b in range(target_bins):
                        s = int(pos[b])
                        e = s + 1
                        if s >= 0 and (clen is None or e <= clen):
                            bin_chroms.append(c)
                            bin_starts.append(s)
                            bin_ends.append(e)
                            bin_region_i.append(i)
                            bin_col_j.append(col)
                        col += 1
            else:
                col += target_bins

            # Right flank bins
            for b in range(n_right):
                s = be + b * bin_size
                e = s + bin_size
                if s >= 0 and (clen is None or e <= clen):
                    bin_chroms.append(c)
                    bin_starts.append(s)
                    bin_ends.append(e)
                    bin_region_i.append(i)
                    bin_col_j.append(col)
                col += 1

        if bin_chroms:
            b_chrom_arr = np.asarray(bin_chroms, dtype=object)
            b_start_arr = np.asarray(bin_starts, dtype=np.int64)
            b_end_arr = np.asarray(bin_ends, dtype=np.int64)
            b_reg_arr = np.asarray(bin_region_i, dtype=np.int64)
            b_col_arr = np.asarray(bin_col_j, dtype=np.int64)

            layers = _partition_non_overlapping(
                b_chrom_arr,
                b_start_arr,
                b_end_arr,
                np.arange(len(bin_chroms), dtype=int),
            )
            for layer in layers:
                layer_df = pd.DataFrame(
                    {
                        "chrom": b_chrom_arr[layer].astype(str),
                        "start": b_start_arr[layer],
                        "end": b_end_arr[layer],
                    }
                )
                extracted = pm.gextract(
                    track_expr,
                    layer_df,
                    iterator=layer_df,
                    **gextract_kwargs,
                )
                if extracted is None or len(extracted) == 0:
                    continue
                val_col = _extract_value_column(extracted)
                local_ids = _normalize_interval_ids(
                    extracted["intervalID"].to_numpy(), len(layer)
                )
                ext_vals = extracted[val_col].to_numpy(dtype=np.float64)
                if not np.isnan(missing_value):
                    ext_vals = np.where(np.isnan(ext_vals), missing_value, ext_vals)
                global_bin_idx = layer[local_ids]
                out[b_reg_arr[global_bin_idx], b_col_arr[global_bin_idx]] = ext_vals

        if np.any(flip):
            out[flip] = out[flip, ::-1]

    return NormalizedMatrix(
        values=out.astype(dtype, copy=False),
        region_index=norm.region_index,
        mode=mode,
        upstream=upstream,
        downstream=downstream,
        bin_size=bin_size,
        target_bins=n_body_bins,
        name=resolved_name,
        strand_oriented=strand_oriented,
        metadata={
            "source": "pymisha",
            "track_expr": track_expr,
            "reference_point": reference_point if mode == "reference_point" else None,
        },
    )
