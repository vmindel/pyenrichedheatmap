"""Array, DataFrame, and interchange loaders for NormalizedMatrix."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from enriched_heatmap.matrix import Mode, NormalizedMatrix


def from_array(
    values: np.ndarray,
    region_index: pd.Index | list[Any] | None = None,
    mode: Mode = "reference_point",
    upstream: int = 1000,
    downstream: int = 1000,
    bin_size: int | None = None,
    target_bins: int = 0,
    name: str = "",
    strand_oriented: bool = True,
    metadata: dict[str, Any] | None = None,
) -> NormalizedMatrix:
    """Wrap a 2D NumPy array into a :class:`NormalizedMatrix`.

    If ``bin_size`` is omitted in ``reference_point`` mode, it is inferred from
    ``(upstream + downstream) / n_bins`` (falling back to ``1`` if the span is 0).
    """
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"`values` must be a 2D array, got shape {arr.shape}.")

    n_bins = arr.shape[1]
    if bin_size is None:
        if mode == "reference_point":
            total_span = upstream + downstream
            if total_span > 0 and n_bins > 0:
                bin_size = max(1, int(round(total_span / n_bins)))
            else:
                bin_size = 1
        else:
            if target_bins <= 0:
                raise ValueError(
                    "Provide `bin_size` (or both `bin_size` and `target_bins`) "
                    "for mode='scale_regions'."
                )
            flank_bins = n_bins - target_bins
            total_flank = upstream + downstream
            if flank_bins > 0 and total_flank > 0:
                bin_size = max(1, int(round(total_flank / flank_bins)))
            else:
                bin_size = 1

    return NormalizedMatrix(
        values=arr,
        region_index=region_index,
        mode=mode,
        upstream=upstream,
        downstream=downstream,
        bin_size=bin_size,
        target_bins=target_bins,
        name=name,
        strand_oriented=strand_oriented,
        metadata=metadata or {},
    )


def from_dataframe(
    df: pd.DataFrame,
    mode: Mode = "reference_point",
    upstream: int = 1000,
    downstream: int = 1000,
    bin_size: int | None = None,
    target_bins: int = 0,
    name: str = "",
    strand_oriented: bool = True,
    metadata: dict[str, Any] | None = None,
) -> NormalizedMatrix:
    """Wrap a 2D pandas DataFrame into a :class:`NormalizedMatrix`, preserving its index."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")
    return from_array(
        values=df.to_numpy(dtype=np.float64),
        region_index=df.index.copy(),
        mode=mode,
        upstream=upstream,
        downstream=downstream,
        bin_size=bin_size,
        target_bins=target_bins,
        name=name,
        strand_oriented=strand_oriented,
        metadata=metadata,
    )


def from_interchange(
    obj: NormalizedMatrix | tuple[np.ndarray, dict[str, Any]] | dict[str, Any] | pd.DataFrame | np.ndarray,
    **overrides: Any,
) -> NormalizedMatrix:
    """Convert any interchange-compatible object into a :class:`NormalizedMatrix`.

    Supports:
    - :class:`NormalizedMatrix` (returned as-is, or cloned with overrides)
    - ``(values, metadata_dict)`` tuple (e.g., from companion Fragment Geometry tools)
    - ``dict`` with ``"values"`` and optional coordinate metadata keys
    - :class:`pandas.DataFrame`
    - 2D :class:`numpy.ndarray`
    """
    if isinstance(obj, NormalizedMatrix):
        if not overrides:
            return obj
        payload = obj.to_dict()
        payload.update(overrides)
        return NormalizedMatrix.from_dict(payload)

    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], dict):
        values, meta = obj
        payload = dict(meta)
        payload["values"] = values
        payload.update(overrides)
        return NormalizedMatrix.from_dict(payload)

    if isinstance(obj, dict):
        payload = dict(obj)
        payload.update(overrides)
        return NormalizedMatrix.from_dict(payload)

    if isinstance(obj, pd.DataFrame):
        return from_dataframe(obj, **overrides)

    if isinstance(obj, np.ndarray):
        return from_array(obj, **overrides)

    raise TypeError(
        f"Unsupported interchange object type: {type(obj).__name__}. "
        "Expected NormalizedMatrix, (values, metadata) tuple, dict, DataFrame, or ndarray."
    )
