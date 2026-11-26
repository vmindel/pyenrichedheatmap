"""Core NormalizedMatrix data model and signal transformations."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

Mode = Literal["reference_point", "scale_regions"]
_PadMode = Literal["edge", "symmetric", "reflect", "wrap", "constant"]


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


_SMOOTH_MODES: dict[str, _PadMode] = {
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


def _validate_region_index(idx: pd.Index, what: str = "`region_index`") -> None:
    """Region identifiers must be unique, non-missing and not booleans.

    Missing identifiers cannot be matched reliably across panels, and booleans collide
    with 0/1 under pandas' hashing (``True == 1``), so both are rejected once here. Every
    lookup further down can then use plain pandas indexing.
    """
    if not idx.is_unique:
        raise ValueError(f"{what} must contain unique region identifiers.")
    if isinstance(idx, pd.MultiIndex):
        missing = any(bool((np.asarray(c) < 0).any()) for c in idx.codes)
    else:
        missing = bool(idx.hasnans)
    if missing:
        raise ValueError(f"{what} must not contain missing (NaN/None) region identifiers.")
    if not isinstance(idx, pd.MultiIndex):
        dt = idx.categories.dtype if isinstance(idx, pd.CategoricalIndex) else idx.dtype
        if pd.api.types.is_bool_dtype(dt) or (
            pd.api.types.is_object_dtype(dt) and pd.api.types.infer_dtype(idx) == "boolean"
        ):
            raise ValueError(f"{what} must not be boolean; use string or integer region IDs.")


def _safe_get_indexer(source: pd.Index, target: pd.Index) -> np.ndarray:
    """Positions of ``target`` labels in ``source`` (-1 where absent), never raising.

    ``Index.get_indexer`` has three traps this guards: a MultiIndex matches a deeper
    MultiIndex by prefix, an IntervalIndex matches numbers by containment, and some
    incompatible index pairs raise instead of returning -1.
    """
    none = np.full(len(target), -1, dtype=np.intp)
    if len(target) == 0 or len(source) == 0:
        return none
    s_mi, t_mi = isinstance(source, pd.MultiIndex), isinstance(target, pd.MultiIndex)
    if s_mi and t_mi and source.nlevels != target.nlevels:
        return none
    if isinstance(source, pd.IntervalIndex) != isinstance(target, pd.IntervalIndex):
        return none
    try:
        return np.asarray(source.get_indexer(target), dtype=np.intp)
    except (TypeError, ValueError, AssertionError, pd.errors.InvalidIndexError):
        return none


def _shares_labels(source: pd.Index, target: pd.Index) -> bool:
    return source is target or bool(np.any(_safe_get_indexer(source, target) >= 0))


def _bool_mask(key: Any, n: int) -> np.ndarray | None:
    """Positions selected by a 1D boolean mask, or None if ``key`` is not one."""
    dt = getattr(key, "dtype", None)
    if isinstance(dt, pd.CategoricalDtype):
        dt = dt.categories.dtype
    if dt is not None and pd.api.types.is_bool_dtype(dt):
        pass
    elif dt is None or pd.api.types.is_object_dtype(dt):
        if pd.api.types.infer_dtype(key, skipna=True) != "boolean":
            return None
    else:
        return None
    if bool(np.any(pd.isna(np.asarray(key, dtype=object)))):
        raise ValueError("Boolean mask cannot contain NA / missing values.")
    arr = np.asarray(key, dtype=bool)
    if len(arr) != n:
        raise ValueError(f"Boolean mask length ({len(arr)}) does not match n_regions ({n}).")
    return np.flatnonzero(arr)


def _as_1d_key(key: Any, what: str) -> Any:
    """Reject containers that are not a 1D row selection; materialise iterators."""
    if isinstance(key, tuple):
        raise IndexError(f"NormalizedMatrix.{what} is 1D (rows only); tuple indexing is not supported.")
    if isinstance(key, np.ndarray) and key.ndim != 1:
        raise IndexError(f"NormalizedMatrix.{what} requires a 1D indexer, got {key.ndim}D array.")
    if key is Ellipsis or isinstance(
        key, (pd.DataFrame, Mapping, set, frozenset, bytes, bytearray, memoryview)
    ):
        raise TypeError(f"Cannot index NormalizedMatrix.{what} with {type(key).__name__}.")
    if not isinstance(key, (np.ndarray, pd.Index, pd.Series, pd.api.extensions.ExtensionArray, list, range)):
        key = list(key)
    return key


class _MatrixLocIndexer:
    """Label-based row indexer for NormalizedMatrix, mirroring pandas .loc."""

    def __init__(self, matrix: NormalizedMatrix) -> None:
        self._matrix = matrix

    def __getitem__(self, key: Any) -> NormalizedMatrix:
        m = self._matrix
        idx = m.region_index
        if isinstance(key, tuple) and isinstance(idx, pd.MultiIndex):
            key = [key]  # a full MultiIndex label selects one row
        elif isinstance(key, slice):
            slc = idx.slice_indexer(key.start, key.stop, key.step)
            return m._subset_positions(np.arange(*slc.indices(len(idx)), dtype=np.intp))
        elif not pd.api.types.is_list_like(key) and not isinstance(key, tuple):
            if key is Ellipsis or isinstance(key, (bytearray, memoryview)):
                raise TypeError(f"Cannot index NormalizedMatrix.loc with {type(key).__name__}.")
            try:
                hash(key)
            except TypeError:
                raise TypeError(
                    f"Cannot index NormalizedMatrix.loc with unhashable {type(key).__name__}."
                ) from None
            if isinstance(idx, pd.IntervalIndex) and not isinstance(key, pd.Interval):
                raise KeyError(key)
            if isinstance(key, (bool, np.bool_)):
                raise KeyError(key)  # region IDs are never boolean; True would match 1
            pos = idx.get_loc(key)
            if not isinstance(pos, (int, np.integer)):
                raise KeyError(key)
            return m._subset_positions(np.array([pos], dtype=np.intp))

        key = _as_1d_key(key, "loc")
        if isinstance(key, pd.Series) and _bool_mask(key.to_numpy(), len(key)) is not None:
            if not key.index.is_unique:
                raise ValueError("Boolean Series index must contain unique identifiers.")
            if len(key) != len(idx):
                raise ValueError(
                    f"Boolean mask length ({len(key)}) does not match n_regions ({len(idx)})."
                )
            pos = _safe_get_indexer(key.index, idx)
            if np.any(pos < 0):
                raise KeyError("Boolean Series index does not match `region_index`.")
            return m._subset_positions(np.flatnonzero(np.asarray(key.to_numpy()[pos], dtype=bool)))
        mask = _bool_mask(key, len(idx))
        if mask is not None:
            return m._subset_positions(mask)

        if isinstance(key, pd.Index):
            key_idx = key
        else:
            key_idx = pd.Index(list(key), tupleize_cols=isinstance(idx, pd.MultiIndex))
        if pd.api.types.is_object_dtype(key_idx.dtype) and not isinstance(key_idx, pd.MultiIndex):
            for x in key_idx:
                try:
                    hash(x)
                except TypeError:
                    raise TypeError(
                        f"Cannot index NormalizedMatrix.loc with unhashable {type(x).__name__}."
                    ) from None
        positions = _safe_get_indexer(idx, key_idx)
        missing = positions < 0
        if np.any(missing):
            raise KeyError(f"Region IDs not found in region_index: {list(key_idx[missing][:5])}")
        return m._subset_positions(positions)


class _MatrixILocIndexer:
    """Integer-position row indexer for NormalizedMatrix."""

    def __init__(self, matrix: NormalizedMatrix) -> None:
        self._matrix = matrix

    def __getitem__(self, key: Any) -> NormalizedMatrix:
        m = self._matrix
        n = m.n_regions
        if isinstance(key, (bool, np.bool_)):
            raise TypeError("Cannot index by location with a boolean scalar.")
        if isinstance(key, (int, np.integer)):
            return m._subset_positions(_check_bounds(np.array([int(key)], dtype=object), n))
        if isinstance(key, slice):
            for bound in (key.start, key.stop, key.step):
                if bound is not None and (
                    isinstance(bound, (bool, np.bool_)) or not isinstance(bound, (int, np.integer))
                ):
                    raise TypeError(
                        f"Slice bounds for iloc must be integers or None, got {type(bound).__name__}."
                    )
            return m._subset_positions(np.arange(*key.indices(n), dtype=np.intp))
        if not pd.api.types.is_list_like(key) or isinstance(key, pd.MultiIndex):
            raise TypeError(f"Cannot index by location with {type(key).__name__}.")
        key = _as_1d_key(key, "iloc")
        if isinstance(key, range):
            if key and not (-n <= key[0] < n and -n <= key[-1] < n):  # len() overflows on huge ranges
                raise IndexError(f"positional index out of bounds for axis 0 with size {n}")
            return m._subset_positions(np.asarray(key, dtype=np.intp))
        if len(key) == 0:
            return m._subset_positions(np.empty(0, dtype=np.intp))
        mask = _bool_mask(key, n)
        if mask is not None:
            return m._subset_positions(mask)
        if pd.api.types.infer_dtype(key, skipna=False) != "integer":
            raise TypeError("Positional indexer must contain integers or a boolean mask.")
        return m._subset_positions(_check_bounds(np.asarray(key, dtype=object), n))


def _check_bounds(pos: np.ndarray, n: int) -> np.ndarray:
    """Integer positions (Python ints in an object array, so huge values cannot wrap)."""
    bad = [int(p) for p in pos if not -n <= int(p) < n]
    if bad:
        raise IndexError(f"positional index {bad[0]} is out of bounds for axis 0 with size {n}")
    return pos.astype(np.intp)


@dataclass(init=False, eq=False)
class NormalizedMatrix:
    """Genomic signal matrix with explicit region alignment and coordinate geometry.

    Parameters
    ----------
    values : np.ndarray
        2D floating-point array of shape ``(n_regions, n_bins)``.
    region_index : pd.Index, optional
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
    metadata : dict[str, Any], optional
        Arbitrary metadata (e.g., loader provenance, differential flags, units).
    centered : bool, default False
        Reference-point only: an extra middle bin is centred on the anchor (from
        ``eh.bins(..., snap=)``). The matrix has ``upstream/bin_size + 1 +
        downstream/bin_size`` columns, x = 0 is the middle of the anchor bin and the
        bins span ``-upstream - bin_size/2`` to ``downstream + bin_size/2``.

    Notes
    -----
    Instances compare by identity. Compare contents with
    ``np.array_equal(a.values, b.values)`` and ``a.region_index.equals(b.region_index)``.
    """

    values: np.ndarray
    region_index: pd.Index
    mode: Mode = "reference_point"
    upstream: int = 1000
    downstream: int = 1000
    bin_size: int = 10
    target_bins: int = 0
    name: str = ""
    strand_oriented: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    centered: bool = False

    def __init__(
        self,
        values: ArrayLike,
        region_index: pd.Index | Sequence[Any] | np.ndarray | None = None,
        mode: Mode = "reference_point",
        upstream: int = 1000,
        downstream: int = 1000,
        bin_size: int = 10,
        target_bins: int = 0,
        name: str | None = "",
        strand_oriented: bool = True,
        metadata: Mapping[str, Any] | None = None,
        centered: bool = False,
    ) -> None:
        arr = np.asarray(values)
        if arr.ndim != 2:
            raise ValueError(f"`values` must be a 2D array, got shape {arr.shape}.")
        if arr.dtype not in (np.float32, np.float64):
            arr = arr.astype(np.float64)
        self.values = arr

        if region_index is None:
            idx: pd.Index = pd.RangeIndex(arr.shape[0])
        elif isinstance(region_index, pd.Index):
            idx = region_index
        else:
            idx = pd.Index(region_index)

        if len(idx) != arr.shape[0]:
            raise ValueError(
                f"`region_index` length ({len(idx)}) does not match "
                f"`values.shape[0]` ({arr.shape[0]})."
            )
        _validate_region_index(idx)
        self.region_index = idx

        if mode not in ("reference_point", "scale_regions"):
            raise ValueError(
                f"`mode` must be 'reference_point' or 'scale_regions', got {mode!r}."
            )
        self.mode = mode
        if upstream < 0 or downstream < 0:
            raise ValueError("`upstream` and `downstream` must be non-negative.")
        if bin_size <= 0:
            raise ValueError("`bin_size` must be positive.")
        if target_bins < 0:
            raise ValueError("`target_bins` must be non-negative.")

        self.upstream = upstream
        self.downstream = downstream
        self.bin_size = bin_size
        self.target_bins = target_bins
        self.name = "" if name is None else str(name)
        self.strand_oriented = bool(strand_oriented)

        self.centered = bool(centered)
        if self.centered and self.mode != "reference_point":
            raise ValueError("`centered` is only valid with mode='reference_point'.")
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

        self.metadata = {} if metadata is None else dict(metadata)

    def _check_geometry(self, n_bins: int) -> None:
        """The column count must follow from the flanks and bin size, or ticks lie."""
        if (self.mode == "reference_point" and self.upstream == 0 and self.downstream == 0
                and not self.centered):
            return
        if self.upstream % self.bin_size or self.downstream % self.bin_size:
            raise ValueError(
                f"`upstream` ({self.upstream}) and `downstream` ({self.downstream}) must be "
                f"multiples of `bin_size` ({self.bin_size})."
            )
        expected = self.upstream // self.bin_size + self.target_bins + int(self.centered) + (
            self.downstream // self.bin_size
        )
        if n_bins != expected:
            raise ValueError(
                f"`values` has {n_bins} columns but upstream={self.upstream}, "
                f"downstream={self.downstream}, bin_size={self.bin_size}"
                + (f", target_bins={self.target_bins}" if self.mode == "scale_regions" else "")
                + (", centered=True" if self.centered else "")
                + f" imply {expected}."
            )

    def _geometry(self) -> tuple[Mode, int, int, int, int, bool, int]:
        return (self.mode, self.upstream, self.downstream, self.bin_size, self.target_bins,
                self.centered, self.n_bins)

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
            if self.centered:
                half = self.bin_size / 2.0
                return (-float(self.upstream) - half, float(self.downstream) + half)
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
        metadata: Mapping[str, Any] | None = None,
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
            centered=self.centered,
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
        region_index: pd.Index | Sequence[Any] | np.ndarray,
        fill_value: float = np.nan,
    ) -> NormalizedMatrix:
        """Align rows to ``region_index``, filling absent regions with ``fill_value``."""
        target = (
            region_index
            if isinstance(region_index, pd.Index)
            else pd.Index(region_index, tupleize_cols=isinstance(self.region_index, pd.MultiIndex))
        )
        if not target.is_unique:
            raise ValueError("Target `region_index` must contain unique identifiers.")

        return self._take_rows(_safe_get_indexer(self.region_index, target), target, fill_value)

    def _take_rows(
        self, indexer: np.ndarray, target: pd.Index, fill_value: float = np.nan
    ) -> NormalizedMatrix:
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
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
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
            fields = (
                "mode", "upstream", "downstream", "bin_size", "target_bins", "centered", "n_bins"
            )
            diff = [
                f"{f}: {a!r} vs {b!r}"
                for f, a, b in zip(fields, self._geometry(), other._geometry(), strict=True)
                if a != b
            ]
            raise ValueError(
                "Cannot combine matrices with different bin geometry (" + "; ".join(diff) + ")."
            )
        if other.region_index is self.region_index or other.region_index.equals(self.region_index):
            return other
        indexer = _safe_get_indexer(other.region_index, self.region_index)
        if len(self.region_index) and len(other.region_index) and not np.any(indexer >= 0):
            raise ValueError(
                "`other` shares no region identifiers with `self` "
                f"(self starts {list(self.region_index[:3])}, "
                f"other starts {list(other.region_index[:3])}). "
                "Pass `region_index=` when constructing matrices from arrays."
            )
        return other._take_rows(indexer, self.region_index)

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
                ticks.append(-float(self.upstream) if self.centered else x_min)
                labels.append(_format_bp(-self.upstream))
            ticks.append(0.0)
            labels.append(center_label)
            if self.downstream > 0:
                ticks.append(float(self.downstream) if self.centered else x_max)
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
        out: dict[str, Any] = {
            "values": self.values.copy(),
            "region_index": self.region_index.tolist(),
            "region_index_name": self.region_index.name,
            "mode": self.mode,
            "upstream": self.upstream,
            "downstream": self.downstream,
            "bin_size": self.bin_size,
            "target_bins": self.target_bins,
            "name": self.name,
            "strand_oriented": self.strand_oriented,
            "metadata": dict(self.metadata),
            "centered": self.centered,
        }
        if isinstance(self.region_index, pd.MultiIndex):
            out["region_index_names"] = list(self.region_index.names)
            if len(self.region_index) == 0:
                out["region_index_level_dtypes"] = [
                    str(lvl.dtype) for lvl in self.region_index.levels
                ]
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NormalizedMatrix:
        """Construct a NormalizedMatrix from an interchange dictionary."""
        if "values" not in payload:
            raise KeyError("Interchange dict must contain a 'values' key.")
        values = np.asarray(payload["values"])
        raw_index = payload.get("region_index")
        has_index_name = "region_index_name" in payload
        index_name = payload.get("region_index_name")
        index_names = payload.get("region_index_names")

        def _empty_multiindex(names_seq: Sequence[Any]) -> pd.MultiIndex:
            names_list = list(names_seq)
            lvl_dtypes = payload.get("region_index_level_dtypes")
            arrays: list[pd.Index] = []
            if lvl_dtypes is not None and len(lvl_dtypes) == len(names_list):
                for dt in lvl_dtypes:
                    try:
                        arrays.append(pd.Index([], dtype=dt))
                    except TypeError:
                        arrays.append(pd.Index([], dtype=object))
            else:
                arrays = [pd.Index([], dtype=object) for _ in names_list]
            return pd.MultiIndex.from_arrays(arrays, names=names_list)

        if raw_index is not None:
            if isinstance(raw_index, pd.MultiIndex):
                idx: pd.Index | None = (
                    raw_index.set_names(list(index_names))
                    if index_names is not None
                    else raw_index.copy()
                )
            elif isinstance(raw_index, pd.Index):
                idx = raw_index.rename(index_name) if has_index_name else raw_index.copy()
            elif index_names is not None:
                raw_list = list(raw_index)
                if len(raw_list) > 0:
                    idx = pd.MultiIndex.from_tuples(raw_list, names=list(index_names))
                else:
                    idx = _empty_multiindex(index_names)
            elif has_index_name:
                idx = pd.Index(raw_index, name=index_name, tupleize_cols=False)
            else:
                idx = pd.Index(raw_index, tupleize_cols=False)
        elif index_names is not None and values.ndim == 2 and values.shape[0] == 0:
            idx = _empty_multiindex(index_names)
        elif has_index_name and index_name is not None and values.ndim == 2:
            idx = pd.RangeIndex(values.shape[0], name=index_name)
        else:
            idx = None
        raw_name = payload.get("name", "")
        return cls(
            values=values,
            region_index=idx,
            mode=payload.get("mode", "reference_point"),
            upstream=int(payload.get("upstream", 1000)),
            downstream=int(payload.get("downstream", 1000)),
            bin_size=int(payload.get("bin_size", 10)),
            target_bins=int(payload.get("target_bins", 0)),
            name="" if raw_name is None else str(raw_name),
            strand_oriented=bool(payload.get("strand_oriented", True)),
            metadata=dict(payload.get("metadata") or {}),
            centered=bool(payload.get("centered", False)),
        )

    def __repr__(self) -> str:
        label = f" name={self.name!r}" if self.name else ""
        extra = f", target_bins={self.target_bins}" if self.mode == "scale_regions" else ""
        if self.centered:
            extra += ", centered=True"
        return (
            f"NormalizedMatrix(shape={self.shape}, mode={self.mode!r}, "
            f"upstream={self.upstream}, downstream={self.downstream}, "
            f"bin_size={self.bin_size}{extra}{label})"
        )
