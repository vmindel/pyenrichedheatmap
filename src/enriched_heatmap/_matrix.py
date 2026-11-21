"""Core NormalizedMatrix data model and signal transformations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

Mode = Literal["reference_point", "scale_regions"]


def _format_bp(bp: float) -> str:
    """Format base-pair offsets into concise publication-ready tick labels."""
    if bp == 0:
        return "0"
    sign = "+" if bp > 0 else "-"
    abs_bp = abs(bp)
    if abs_bp >= 1_000_000:
        val = abs_bp / 1_000_000
        return f"{sign}{val:g} Mb"
    if abs_bp >= 1_000:
        val = abs_bp / 1_000
        return f"{sign}{val:g} kb"
    return f"{sign}{abs_bp:g} bp"


_SMOOTH_MODES = {
    # scipy.ndimage boundary mode -> numpy.pad equivalent
    "nearest": "edge",
    "reflect": "symmetric",
    "mirror": "reflect",
    "wrap": "wrap",
    "constant": "constant",
}


def _gaussian_smooth_1d_nan_aware(
    values: np.ndarray, sigma: float, mode: str = "nearest"
) -> np.ndarray:
    """1D Gaussian smoothing along axis=1 that ignores NaN entries.

    NaN bins stay NaN: they are excluded from their neighbours' averages but never
    filled in from them.
    """
    if mode not in _SMOOTH_MODES:
        raise ValueError(f"`mode` must be one of {sorted(_SMOOTH_MODES)}, got {mode!r}.")
    if sigma <= 0:
        return values.copy()

    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0).astype(np.float64, copy=False)
    weights = finite.astype(np.float64)

    try:
        from scipy.ndimage import gaussian_filter1d

        num = gaussian_filter1d(filled, sigma=sigma, axis=1, mode=mode)
        den = gaussian_filter1d(weights, sigma=sigma, axis=1, mode=mode)
    except ImportError:
        radius = max(1, int(4.0 * sigma + 0.5))  # scipy's default truncate=4.0
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()

        pad = ((0, 0), (radius, radius))
        padded_num = np.pad(filled, pad, mode=_SMOOTH_MODES[mode])
        padded_den = np.pad(weights, pad, mode=_SMOOTH_MODES[mode])

        n_bins = values.shape[1]
        num = np.zeros_like(filled)
        den = np.zeros_like(weights)
        for k_idx, w in enumerate(kernel):
            num += w * padded_num[:, k_idx : k_idx + n_bins]
            den += w * padded_den[:, k_idx : k_idx + n_bins]

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(finite & (den > 1e-12), num / den, np.nan)
    return out.astype(values.dtype, copy=False)


class _MatrixLocIndexer:
    """Label-based row indexer for NormalizedMatrix, mirroring pandas .loc."""

    def __init__(self, matrix: NormalizedMatrix) -> None:
        self._matrix = matrix

    def __getitem__(self, key: Any) -> NormalizedMatrix:
        idx = self._matrix.region_index
        if isinstance(key, (np.ndarray, pd.Series, list)) and len(key) > 0:
            arr = np.asarray(key)
            if arr.dtype == bool:
                if len(arr) != len(idx):
                    raise ValueError(
                        f"Boolean mask length ({len(arr)}) does not match "
                        f"n_regions ({len(idx)})."
                    )
                positions = np.flatnonzero(arr)
                return self._matrix._subset_positions(positions)

        if isinstance(key, slice):
            start_pos = 0 if key.start is None else idx.get_loc(key.start)
            stop_pos = len(idx) if key.stop is None else idx.get_loc(key.stop) + 1
            positions = np.arange(start_pos, stop_pos, key.step or 1)
            return self._matrix._subset_positions(positions)

        if np.isscalar(key):
            pos = idx.get_loc(key)
            return self._matrix._subset_positions(np.array([pos], dtype=int))

        key_idx = pd.Index(key)
        positions = idx.get_indexer(key_idx)
        missing = positions < 0
        if np.any(missing):
            missing_labels = list(key_idx[missing][:5])
            raise KeyError(f"Region IDs not found in region_index: {missing_labels}")
        return self._matrix._subset_positions(positions)


class _MatrixILocIndexer:
    """Integer-position row indexer for NormalizedMatrix."""

    def __init__(self, matrix: NormalizedMatrix) -> None:
        self._matrix = matrix

    def __getitem__(self, key: Any) -> NormalizedMatrix:
        if np.isscalar(key):
            positions = np.array([int(key)], dtype=int)
        elif isinstance(key, slice):
            positions = np.arange(*key.indices(self._matrix.n_regions))
        else:
            arr = np.asarray(key)
            positions = np.flatnonzero(arr) if arr.dtype == bool else arr.astype(int)
        return self._matrix._subset_positions(positions)


@dataclass(eq=False)
class NormalizedMatrix:
    """Genomic signal matrix with explicit region alignment and coordinate geometry.

    Parameters
    ----------
    values : np.ndarray
        2D floating-point array of shape ``(n_regions, n_bins)``.
    region_index : pd.Index
        Unique region identifiers of length ``n_regions``.
    mode : {"reference_point", "scale_regions"}
        Coordinate mode for the bin axis.
    upstream : int
        Base pairs upstream of anchor/start.
    downstream : int
        Base pairs downstream of anchor/end.
    bin_size : int
        Base pairs per bin in flanking regions.
    target_bins : int, default 0
        Number of bins across the scaled body (only > 0 for ``"scale_regions"``).
    name : str, default ""
        Human-readable signal/track name.
    strand_oriented : bool, default True
        Whether minus-strand regions have been oriented 5' -> 3'.
    metadata : dict[str, Any]
        Arbitrary metadata (e.g., loader provenance, differential flags, units).

    Notes
    -----
    Instances compare by identity. Compare contents with
    ``np.array_equal(a.values, b.values)`` and ``a.region_index.equals(b.region_index)``.
    """

    values: np.ndarray
    region_index: pd.Index | None = None
    mode: Mode = "reference_point"
    upstream: int = 1000
    downstream: int = 1000
    bin_size: int = 10
    target_bins: int = 0
    name: str = ""
    strand_oriented: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.values)
        if arr.ndim != 2:
            raise ValueError(f"`values` must be a 2D array, got shape {arr.shape}.")
        if arr.dtype not in (np.float32, np.float64):
            arr = arr.astype(np.float64)
        self.values = arr

        if self.region_index is None:
            self.region_index = pd.RangeIndex(arr.shape[0])
        elif not isinstance(self.region_index, pd.Index):
            self.region_index = pd.Index(self.region_index)

        if len(self.region_index) != arr.shape[0]:
            raise ValueError(
                f"`region_index` length ({len(self.region_index)}) does not match "
                f"`values.shape[0]` ({arr.shape[0]})."
            )
        if not self.region_index.is_unique:
            raise ValueError("`region_index` must contain unique region identifiers.")

        if self.mode not in ("reference_point", "scale_regions"):
            raise ValueError(
                f"`mode` must be 'reference_point' or 'scale_regions', got {self.mode!r}."
            )
        if self.upstream < 0 or self.downstream < 0:
            raise ValueError("`upstream` and `downstream` must be non-negative.")
        if self.bin_size <= 0:
            raise ValueError("`bin_size` must be positive.")
        if self.target_bins < 0:
            raise ValueError("`target_bins` must be non-negative.")

        if self.mode == "reference_point":
            if self.target_bins != 0:
                raise ValueError(
                    "`target_bins` must be 0 when mode='reference_point'."
                )
        else:
            if self.target_bins == 0:
                inferred = arr.shape[1] - (self.upstream // self.bin_size) - (
                    self.downstream // self.bin_size
                )
                if inferred <= 0:
                    raise ValueError(
                        "`target_bins` must be > 0 for mode='scale_regions'."
                    )
                self.target_bins = int(inferred)

        self._check_geometry(arr.shape[1])

        if self.metadata is None:
            self.metadata = {}
        else:
            self.metadata = dict(self.metadata)

    def _check_geometry(self, n_bins: int) -> None:
        """The column count must follow from the flanks and bin size, or ticks lie."""
        if self.mode == "reference_point" and self.upstream == 0 and self.downstream == 0:
            return
        if self.upstream % self.bin_size or self.downstream % self.bin_size:
            raise ValueError(
                f"`upstream` ({self.upstream}) and `downstream` ({self.downstream}) must be "
                f"multiples of `bin_size` ({self.bin_size})."
            )
        expected = self.upstream // self.bin_size + self.target_bins + (
            self.downstream // self.bin_size
        )
        if n_bins != expected:
            raise ValueError(
                f"`values` has {n_bins} columns but upstream={self.upstream}, "
                f"downstream={self.downstream}, bin_size={self.bin_size}"
                + (f", target_bins={self.target_bins}" if self.mode == "scale_regions" else "")
                + f" imply {expected}."
            )

    def _geometry(self) -> tuple:
        return (self.mode, self.upstream, self.downstream, self.bin_size, self.target_bins)

    # ------------------------------------------------------------------
    # Properties & indexers
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def n_regions(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.values.shape[1])

    @property
    def n_upstream_bins(self) -> int:
        return int(self.upstream // self.bin_size)

    @property
    def n_downstream_bins(self) -> int:
        return int(self.downstream // self.bin_size)

    @property
    def body_length(self) -> float:
        """Visual x-axis length of the scaled target body (0 for reference_point)."""
        if self.mode == "scale_regions":
            return float(self.target_bins * self.bin_size)
        return 0.0

    @property
    def x_extent(self) -> tuple[float, float]:
        """(x_min, x_max) coordinates spanned by the bins."""
        if self.mode == "reference_point":
            if self.upstream == 0 and self.downstream == 0:
                return (0.0, float(self.n_bins * self.bin_size))
            return (-float(self.upstream), float(self.downstream))
        return (-float(self.upstream), self.body_length + float(self.downstream))

    @property
    def loc(self) -> _MatrixLocIndexer:
        return _MatrixLocIndexer(self)

    @property
    def iloc(self) -> _MatrixILocIndexer:
        return _MatrixILocIndexer(self)

    def _clone(
        self,
        values: np.ndarray,
        region_index: pd.Index | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NormalizedMatrix:
        return NormalizedMatrix(
            values=values,
            region_index=self.region_index.copy() if region_index is None else region_index,
            mode=self.mode,
            upstream=self.upstream,
            downstream=self.downstream,
            bin_size=self.bin_size,
            target_bins=self.target_bins,
            name=self.name if name is None else name,
            strand_oriented=self.strand_oriented,
            metadata=dict(self.metadata) if metadata is None else dict(metadata),
        )

    def _subset_positions(self, positions: np.ndarray) -> NormalizedMatrix:
        return self._clone(
            values=self.values[positions].copy(),
            region_index=self.region_index[positions],
        )

    def copy(self) -> NormalizedMatrix:
        """Return a deep copy of this NormalizedMatrix."""
        return self._clone(values=self.values.copy())

    # ------------------------------------------------------------------
    # Alignment & subsetting
    # ------------------------------------------------------------------
    def reindex(
        self,
        region_index: pd.Index | list[Any],
        fill_value: float = np.nan,
    ) -> NormalizedMatrix:
        """Align rows to ``region_index``, filling absent regions with ``fill_value``."""
        target = region_index if isinstance(region_index, pd.Index) else pd.Index(region_index)
        if not target.is_unique:
            raise ValueError("Target `region_index` must contain unique identifiers.")

        indexer = self.region_index.get_indexer(target)
        out = np.full((len(target), self.n_bins), fill_value, dtype=self.values.dtype)
        present = indexer >= 0
        if np.any(present):
            out[present] = self.values[indexer[present]]
        return self._clone(values=out, region_index=target)

    def dropna_regions(
        self,
        how: Literal["any", "all"] = "any",
        thresh: int | None = None,
    ) -> NormalizedMatrix:
        """Return a copy with rows containing NaNs removed."""
        finite_counts = np.isfinite(self.values).sum(axis=1)
        if thresh is not None:
            keep = finite_counts >= thresh
        elif how == "any":
            keep = finite_counts == self.n_bins
        elif how == "all":
            keep = finite_counts > 0
        else:
            raise ValueError(f"`how` must be 'any' or 'all', got {how!r}.")
        return self._subset_positions(np.flatnonzero(keep))

    # ------------------------------------------------------------------
    # Signal transforms (non-destructive)
    # ------------------------------------------------------------------
    def log1p(self) -> NormalizedMatrix:
        """Return a new matrix transformed by ``log(1 + x)``."""
        with np.errstate(invalid="ignore"):
            vals = np.log1p(self.values)
        return self._clone(values=vals)

    def log2p(self, pseudocount: float = 1.0) -> NormalizedMatrix:
        """Return a new matrix transformed by ``log2(x + pseudocount)``."""
        if pseudocount <= 0:
            raise ValueError("`pseudocount` must be positive.")
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = np.log2(self.values + pseudocount)
        return self._clone(values=vals)

    def zscore(self, axis: Literal[0, 1] = 1, ddof: int = 0) -> NormalizedMatrix:
        """Return a z-scored copy along rows (axis=1) or columns/bins (axis=0)."""
        if axis not in (0, 1):
            raise ValueError(f"`axis` must be 0 or 1, got {axis!r}.")
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nanmean(self.values, axis=axis, keepdims=True)
            std = np.nanstd(self.values, axis=axis, ddof=ddof, keepdims=True)
            z = np.where(std > 0, (self.values - mean) / std, 0.0)
            z = np.where(np.isnan(self.values), np.nan, z)
        return self._clone(values=z.astype(self.values.dtype, copy=False))

    def clip_quantile(
        self, q_low: float = 0.0, q_high: float = 0.99
    ) -> NormalizedMatrix:
        """Clip values to the ``[q_low, q_high]`` quantiles across all finite entries."""
        if not (0.0 <= q_low <= q_high <= 1.0):
            raise ValueError("Require 0.0 <= q_low <= q_high <= 1.0.")
        finite = self.values[np.isfinite(self.values)]
        if finite.size == 0:
            return self.copy()
        lo = float(np.quantile(finite, q_low))
        hi = float(np.quantile(finite, q_high))
        return self._clone(values=np.clip(self.values, lo, hi))

    def smooth(self, sigma: float = 1.0, mode: str = "nearest") -> NormalizedMatrix:
        """Apply 1D NaN-aware Gaussian smoothing along the bin axis."""
        if sigma < 0:
            raise ValueError("`sigma` must be non-negative.")
        smoothed = _gaussian_smooth_1d_nan_aware(self.values, sigma=sigma, mode=mode)
        return self._clone(values=smoothed)

    def _align_other(self, other: NormalizedMatrix) -> NormalizedMatrix:
        if not isinstance(other, NormalizedMatrix):
            raise TypeError(f"Expected NormalizedMatrix, got {type(other).__name__}.")
        if other._geometry() != self._geometry():
            fields = ("mode", "upstream", "downstream", "bin_size", "target_bins")
            diff = [
                f"{f}: {a!r} vs {b!r}"
                for f, a, b in zip(fields, self._geometry(), other._geometry(), strict=True)
                if a != b
            ]
            raise ValueError(
                "Cannot combine matrices with different bin geometry (" + "; ".join(diff) + ")."
            )
        if not other.region_index.equals(self.region_index):
            return other.reindex(self.region_index)
        return other

    def subtract(self, other: NormalizedMatrix) -> NormalizedMatrix:
        """Compute ``self - other`` after aligning ``other`` to ``self.region_index``."""
        aligned = self._align_other(other)
        meta = dict(self.metadata)
        meta["is_differential"] = True
        name = (
            f"{self.name} - {other.name}"
            if (self.name and other.name)
            else (self.name or other.name)
        )
        return self._clone(
            values=self.values - aligned.values,
            name=name,
            metadata=meta,
        )

    def log2_ratio(
        self, other: NormalizedMatrix, pseudocount: float = 1.0
    ) -> NormalizedMatrix:
        """Compute ``log2((self + pseudocount) / (other + pseudocount))``."""
        if pseudocount <= 0:
            raise ValueError("`pseudocount` must be positive.")
        aligned = self._align_other(other)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.log2((self.values + pseudocount) / (aligned.values + pseudocount))
        meta = dict(self.metadata)
        meta["is_differential"] = True
        meta["pseudocount"] = pseudocount
        name = (
            f"log2({self.name}/{other.name})"
            if (self.name and other.name)
            else (self.name or other.name)
        )
        return self._clone(
            values=ratio.astype(self.values.dtype, copy=False),
            name=name,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Coordinate & tick helpers
    # ------------------------------------------------------------------
    def bin_centers(self) -> np.ndarray:
        """Return 1D array of exact x-coordinates for each bin center."""
        x_min, x_max = self.x_extent
        if self.n_bins == 0:
            return np.array([], dtype=np.float64)
        step = (x_max - x_min) / self.n_bins
        return np.linspace(x_min + 0.5 * step, x_max - 0.5 * step, self.n_bins)

    def guide_positions(self) -> list[float]:
        """Return vertical reference guide x-positions."""
        if self.mode == "reference_point":
            return [0.0]
        return [0.0, self.body_length]

    def default_xticks(
        self, anchor_labels: str | tuple[str, str] | None = None
    ) -> tuple[list[float], list[str]]:
        """Return clean default (tick_positions, tick_labels) for the x-axis."""
        x_min, x_max = self.x_extent
        if self.mode == "reference_point":
            center_label = anchor_labels if isinstance(anchor_labels, str) else "0"
            ticks: list[float] = []
            labels: list[str] = []
            if self.upstream > 0:
                ticks.append(x_min)
                labels.append(_format_bp(-self.upstream))
            ticks.append(0.0)
            labels.append(center_label)
            if self.downstream > 0:
                ticks.append(x_max)
                labels.append(_format_bp(self.downstream))
            return ticks, labels

        if isinstance(anchor_labels, tuple) and len(anchor_labels) == 2:
            start_lbl, end_lbl = anchor_labels
        else:
            start_lbl, end_lbl = "Start", "End"
        ticks = []
        labels = []
        if self.upstream > 0:
            ticks.append(x_min)
            labels.append(_format_bp(-self.upstream))
        ticks.extend([0.0, self.body_length])
        labels.extend([start_lbl, end_lbl])
        if self.downstream > 0:
            ticks.append(x_max)
            labels.append(_format_bp(self.downstream))
        return ticks, labels

    # ------------------------------------------------------------------
    # Interchange & DataFrame conversion
    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a DataFrame indexed by ``region_index`` with bin centers as columns."""
        return pd.DataFrame(
            self.values,
            index=self.region_index.copy(),
            columns=self.bin_centers(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Export as a plain interchange dictionary (zero cross-package coupling)."""
        return {
            "values": self.values.copy(),
            "region_index": self.region_index.tolist(),
            "mode": self.mode,
            "upstream": self.upstream,
            "downstream": self.downstream,
            "bin_size": self.bin_size,
            "target_bins": self.target_bins,
            "name": self.name,
            "strand_oriented": self.strand_oriented,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NormalizedMatrix:
        """Construct a NormalizedMatrix from an interchange dictionary."""
        if "values" not in payload:
            raise KeyError("Interchange dict must contain a 'values' key.")
        region_index = payload.get("region_index")
        return cls(
            values=np.asarray(payload["values"]),
            region_index=pd.Index(region_index) if region_index is not None else None,
            mode=payload.get("mode", "reference_point"),
            upstream=int(payload.get("upstream", 1000)),
            downstream=int(payload.get("downstream", 1000)),
            bin_size=int(payload.get("bin_size", 10)),
            target_bins=int(payload.get("target_bins", 0)),
            name=str(payload.get("name", "")),
            strand_oriented=bool(payload.get("strand_oriented", True)),
            metadata=dict(payload.get("metadata", {})),
        )

    def __repr__(self) -> str:
        label = f" name={self.name!r}" if self.name else ""
        extra = f", target_bins={self.target_bins}" if self.mode == "scale_regions" else ""
        return (
            f"NormalizedMatrix(shape={self.shape}, mode={self.mode!r}, "
            f"upstream={self.upstream}, downstream={self.downstream}, "
            f"bin_size={self.bin_size}{extra}{label})"
        )
