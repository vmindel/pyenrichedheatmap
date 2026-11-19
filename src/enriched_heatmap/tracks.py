"""Declarative track classes for horizontal heatmap composition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from enriched_heatmap.loaders.array import from_interchange
from enriched_heatmap.matrix import NormalizedMatrix
from enriched_heatmap.ordering import OrderMetric, compute_order_scores
from enriched_heatmap.profiles import ProfileMode
from enriched_heatmap.raster import DownsampleAgg

if TYPE_CHECKING:
    from enriched_heatmap.layout import HeatmapList

_UNSET = object()


def resolve_palette(
    categories: Sequence[Any],
    palette: str | dict[Any, Any] | Sequence[Any] = "Set2",
) -> dict[Any, Any]:
    """Resolve a colormap name, list of colors, or mapping into ``{category: color}``."""
    cats = list(categories)
    if not cats:
        return {}

    if isinstance(palette, dict):
        fallback_cmap = mpl.colormaps.get_cmap("tab10")
        out = {}
        for i, c in enumerate(cats):
            out[c] = palette[c] if c in palette else fallback_cmap(i % fallback_cmap.N)
        return out

    if isinstance(palette, str):
        cmap = mpl.colormaps.get_cmap(palette)
        if hasattr(cmap, "colors") and len(cmap.colors) > 0:
            colors = list(cmap.colors)
            return {c: colors[i % len(colors)] for i, c in enumerate(cats)}
        n = max(len(cats), 1)
        if n == 1:
            return {cats[0]: cmap(0.5)}
        return {c: cmap(i / (n - 1)) for i, c in enumerate(cats)}

    colors = list(palette)
    if not colors:
        raise ValueError("`palette` sequence must not be empty.")
    return {c: colors[i % len(colors)] for i, c in enumerate(cats)}


class BaseTrack:
    """Abstract base class for composable heatmap tracks."""

    name: str = ""
    width: float = 1.0
    scale_group: str | None = None

    @property
    def region_index(self) -> pd.Index:
        raise NotImplementedError

    @property
    def has_top_profile(self) -> bool:
        return False

    @property
    def has_colorbar(self) -> bool:
        return False

    def _clustering_data(self, region_index: pd.Index) -> Any:
        return None

    def _order_scores(
        self, region_index: pd.Index, metric: OrderMetric = "center_mean"
    ) -> np.ndarray | None:
        return None

    def _aligned_numeric_values(self, region_index: pd.Index) -> np.ndarray | None:
        return None

    def __add__(self, other: BaseTrack | HeatmapList) -> HeatmapList:
        from enriched_heatmap.layout import HeatmapList

        if isinstance(other, HeatmapList):
            return HeatmapList([self, *other.tracks])
        if isinstance(other, BaseTrack):
            return HeatmapList([self, other])
        return NotImplemented


class EnrichedTrack(BaseTrack):
    """Genomic signal track displaying a :class:`NormalizedMatrix` + optional top metaplot.

    Parameters
    ----------
    matrix : NormalizedMatrix | pd.DataFrame | np.ndarray | dict | tuple
        Signal matrix (or interchange object).
    name : str | None, default None
        Display title above the track (defaults to ``matrix.name``).
    cmap : str | Colormap, default "Reds" (auto "RdBu_r" for differential matrices)
        Matplotlib colormap.
    vmin, vmax : float | None
        Explicit color scale limits.
    vmin_q, vmax_q : float | None
        Quantile-based color scale limits when ``vmin``/``vmax`` are omitted.
        Default ``vmax_q=0.99``.
    center : float | None
        Center value for diverging ``TwoSlopeNorm`` (auto ``0.0`` for differential matrices).
    top_profile : str | None, default "mean"
        Top metaplot mode (``"mean"``, ``"median"``, ``"mean_sem"``, ``"mean_ci"``,
        ``"mean_boot_ci"``, ``"median_iqr"``, or ``None`` to disable).
    scale_group : str | None, default None
        Optional group name; tracks sharing the same ``scale_group`` share a single
        ``vmin``/``vmax`` range and a single colorbar.
    downsample_agg : str | Callable | None, default None
        Optional per-track override of global ``downsample_agg`` (e.g., ``"q90"`` or ``"max"``
        for narrow TF ChEC peaks alongside ``"mean"`` for ATAC-seq).
    """

    def __init__(
        self,
        matrix: NormalizedMatrix | pd.DataFrame | np.ndarray | dict[str, Any] | tuple[np.ndarray, dict[str, Any]],
        name: str | None = None,
        cmap: str | mcolors.Colormap | Any = _UNSET,
        vmin: float | None = None,
        vmax: float | None = None,
        vmin_q: float | None = None,
        vmax_q: float | None = 0.99,
        center: float | None | Any = _UNSET,
        top_profile: ProfileMode | None = "mean",
        top_profile_ylim: tuple[float, float] | None = None,
        top_profile_ylabel: str | None = None,
        top_profile_ci: float = 0.95,
        guide_lines: bool = True,
        guide_kwargs: dict[str, Any] | None = None,
        xticks: list[float] | None = None,
        xtick_labels: list[str] | None = None,
        anchor_labels: str | tuple[str, str] | None = None,
        width: float = 2.0,
        scale_group: str | None = None,
        colorbar: bool = True,
        colorbar_label: str | None = None,
        downsample_agg: DownsampleAgg | None = None,
        na_color: str = "#eeeeee",
        show_border: bool = True,
    ) -> None:
        mat = from_interchange(matrix)
        self.matrix = mat
        self.name = mat.name if name is None else str(name)

        is_diff = bool(mat.metadata.get("is_differential", False))
        if cmap is _UNSET:
            self.cmap: str | mcolors.Colormap = "RdBu_r" if is_diff else "Reds"
        else:
            self.cmap = cmap

        if center is _UNSET:
            self.center: float | None = 0.0 if is_diff else None
        else:
            self.center = center

        self.vmin = vmin
        self.vmax = vmax
        self.vmin_q = vmin_q
        self.vmax_q = vmax_q
        self.top_profile = top_profile
        self.top_profile_ylim = top_profile_ylim
        self.top_profile_ylabel = top_profile_ylabel
        self.top_profile_ci = top_profile_ci
        self.guide_lines = guide_lines
        self.guide_kwargs = dict(guide_kwargs) if guide_kwargs is not None else {}
        self.xticks = xticks
        self.xtick_labels = xtick_labels
        self.anchor_labels = anchor_labels
        self.width = float(width)
        self.scale_group = scale_group
        self.colorbar = bool(colorbar)
        self.colorbar_label = colorbar_label
        self.downsample_agg = downsample_agg
        self.na_color = na_color
        self.show_border = bool(show_border)

    @property
    def region_index(self) -> pd.Index:
        return self.matrix.region_index

    @property
    def has_top_profile(self) -> bool:
        return self.top_profile is not None

    @property
    def has_colorbar(self) -> bool:
        return self.colorbar

    def _aligned_matrix(self, region_index: pd.Index) -> NormalizedMatrix:
        if self.matrix.region_index.equals(region_index):
            return self.matrix
        return self.matrix.reindex(region_index)

    def _aligned_numeric_values(self, region_index: pd.Index) -> np.ndarray:
        return self._aligned_matrix(region_index).values

    def _clustering_data(self, region_index: pd.Index) -> NormalizedMatrix:
        return self._aligned_matrix(region_index)

    def _order_scores(
        self, region_index: pd.Index, metric: OrderMetric = "center_mean"
    ) -> np.ndarray:
        return compute_order_scores(self._aligned_matrix(region_index), metric=metric)

    def __repr__(self) -> str:
        return f"EnrichedTrack(name={self.name!r}, shape={self.matrix.shape}, cmap={self.cmap!r})"


class MatrixTrack(BaseTrack):
    """Arbitrary numeric 2D DataFrame/array heatmap track (e.g., RNA-seq log2FC, motif scores)."""

    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        name: str = "",
        region_index: pd.Index | Sequence[Any] | None = None,
        columns: Sequence[str] | None = None,
        cmap: str | mcolors.Colormap = "RdBu_r",
        vmin: float | None = None,
        vmax: float | None = None,
        vmin_q: float | None = None,
        vmax_q: float | None = 0.99,
        center: float | None = None,
        cluster_columns: bool = False,
        col_labels: bool = True,
        col_label_rotation: float = 45.0,
        top_profile: ProfileMode | None = None,
        top_profile_ylim: tuple[float, float] | None = None,
        top_profile_ylabel: str | None = None,
        top_profile_ci: float = 0.95,
        width: float = 1.2,
        scale_group: str | None = None,
        colorbar: bool = True,
        colorbar_label: str | None = None,
        downsample_agg: DownsampleAgg | None = None,
        na_color: str = "#eeeeee",
        show_border: bool = True,
    ) -> None:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            if region_index is not None:
                df.index = pd.Index(region_index)
            if columns is not None:
                df.columns = pd.Index(columns)
        else:
            arr = np.asarray(data, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.ndim != 2:
                raise ValueError(f"MatrixTrack `data` must be 2D, got shape {arr.shape}.")
            idx = pd.Index(region_index) if region_index is not None else pd.RangeIndex(arr.shape[0])
            cols = (
                pd.Index(columns)
                if columns is not None
                else pd.Index([f"col{i}" for i in range(arr.shape[1])])
            )
            df = pd.DataFrame(arr, index=idx, columns=cols)

        if not df.index.is_unique:
            raise ValueError("MatrixTrack `region_index` must be unique.")

        self.data = df.astype(np.float64)
        self.name = str(name)
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.vmin_q = vmin_q
        self.vmax_q = vmax_q
        self.center = center
        self.cluster_columns = bool(cluster_columns)
        self.col_labels = bool(col_labels)
        self.col_label_rotation = float(col_label_rotation)
        self.top_profile = top_profile
        self.top_profile_ylim = top_profile_ylim
        self.top_profile_ylabel = top_profile_ylabel
        self.top_profile_ci = top_profile_ci
        self.width = float(width)
        self.scale_group = scale_group
        self.colorbar = bool(colorbar)
        self.colorbar_label = colorbar_label
        self.downsample_agg = downsample_agg
        self.na_color = na_color
        self.show_border = bool(show_border)

    @property
    def region_index(self) -> pd.Index:
        return self.data.index

    @property
    def has_top_profile(self) -> bool:
        return self.top_profile is not None

    @property
    def has_colorbar(self) -> bool:
        return self.colorbar

    def _aligned_df(self, region_index: pd.Index) -> pd.DataFrame:
        df = self.data if self.data.index.equals(region_index) else self.data.reindex(region_index)
        if self.cluster_columns and df.shape[1] > 2:
            try:
                from scipy.cluster.hierarchy import leaves_list, linkage
            except ImportError as exc:
                raise ImportError(
                    "scipy is required for `cluster_columns=True`. "
                    "Install it with: pip install 'pyenrichedheatmap[stats]'"
                ) from exc
            vals = df.to_numpy(dtype=np.float64, copy=True)
            vals = np.where(np.isfinite(vals), vals, 0.0)
            Z = linkage(vals.T, method="ward", metric="euclidean")
            col_order = leaves_list(Z)
            df = df.iloc[:, col_order]
        return df

    def _aligned_numeric_values(self, region_index: pd.Index) -> np.ndarray:
        return self._aligned_df(region_index).to_numpy(dtype=np.float64)

    def _clustering_data(self, region_index: pd.Index) -> pd.DataFrame:
        return self._aligned_df(region_index)

    def _order_scores(
        self, region_index: pd.Index, metric: OrderMetric = "mean"
    ) -> np.ndarray:
        vals = self._aligned_numeric_values(region_index)
        if callable(metric):
            return np.asarray(metric(vals), dtype=np.float64)
        with np.errstate(invalid="ignore", all="ignore"):
            if metric in ("mean", "center_mean", "enriched_score"):
                return np.nanmean(vals, axis=1)
            if metric == "median":
                return np.nanmedian(vals, axis=1)
            if metric in ("max", "center_max"):
                return np.nanmax(vals, axis=1)
            if metric == "sum":
                return np.nansum(vals, axis=1)
        return np.nanmean(vals, axis=1)

    def __repr__(self) -> str:
        return f"MatrixTrack(name={self.name!r}, shape={self.data.shape}, cmap={self.cmap!r})"


class CategoricalTrack(BaseTrack):
    """Discrete row annotation strip (e.g., peak class, promoter/enhancer, cluster)."""

    def __init__(
        self,
        data: pd.Series | pd.Categorical | Sequence[Any],
        name: str = "",
        region_index: pd.Index | Sequence[Any] | None = None,
        palette: str | dict[Any, Any] | Sequence[Any] = "Set2",
        categories: Sequence[Any] | None = None,
        width: float = 0.25,
        show_legend: bool = True,
        na_color: str = "#cccccc",
        show_border: bool = True,
    ) -> None:
        if isinstance(data, pd.Series):
            s = data.copy()
            if region_index is not None:
                s.index = pd.Index(region_index)
        elif isinstance(data, pd.Categorical):
            idx = pd.Index(region_index) if region_index is not None else pd.RangeIndex(len(data))
            s = pd.Series(data, index=idx)
        else:
            arr = list(data)
            idx = pd.Index(region_index) if region_index is not None else pd.RangeIndex(len(arr))
            s = pd.Series(arr, index=idx)

        if not s.index.is_unique:
            raise ValueError("CategoricalTrack `region_index` must be unique.")

        if categories is not None:
            cats = list(categories)
        elif isinstance(s.dtype, pd.CategoricalDtype):
            cats = list(s.cat.categories)
        else:
            cats = list(pd.unique(s.dropna()))

        self.series = s
        self.categories = cats
        self.name = str(name)
        self.palette = palette
        self.color_map = resolve_palette(cats, palette)
        self.width = float(width)
        self.show_legend = bool(show_legend)
        self.na_color = na_color
        self.show_border = bool(show_border)

    @property
    def region_index(self) -> pd.Index:
        return self.series.index

    def _aligned_codes(self, region_index: pd.Index) -> np.ndarray:
        aligned = (
            self.series
            if self.series.index.equals(region_index)
            else self.series.reindex(region_index)
        )
        cat_to_code = {c: i for i, c in enumerate(self.categories)}
        codes = np.full(len(region_index), -1, dtype=np.int64)
        for i, val in enumerate(aligned.to_numpy()):
            if pd.isna(val):
                continue
            code = cat_to_code.get(val, -1)
            codes[i] = code
        return codes

    def __repr__(self) -> str:
        return (
            f"CategoricalTrack(name={self.name!r}, n_regions={len(self.series)}, "
            f"n_categories={len(self.categories)})"
        )


class BarTrack(BaseTrack):
    """1D continuous row annotation rendered as horizontal bars along each region."""

    def __init__(
        self,
        data: pd.Series | Sequence[float] | np.ndarray,
        name: str = "",
        region_index: pd.Index | Sequence[Any] | None = None,
        color: str = "#4c72b0",
        neg_color: str | None = None,
        xlim: tuple[float, float] | None = None,
        zero_line: bool = True,
        width: float = 0.6,
        xlabel: str | None = None,
        downsample_agg: DownsampleAgg | None = None,
        show_border: bool = True,
    ) -> None:
        if isinstance(data, pd.Series):
            s = data.astype(np.float64).copy()
            if region_index is not None:
                s.index = pd.Index(region_index)
        else:
            arr = np.asarray(data, dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(f"BarTrack `data` must be 1D, got shape {arr.shape}.")
            idx = pd.Index(region_index) if region_index is not None else pd.RangeIndex(len(arr))
            s = pd.Series(arr, index=idx)

        if not s.index.is_unique:
            raise ValueError("BarTrack `region_index` must be unique.")

        self.series = s
        self.name = str(name)
        self.color = color
        self.neg_color = neg_color
        self.xlim = xlim
        self.zero_line = bool(zero_line)
        self.width = float(width)
        self.xlabel = xlabel
        self.downsample_agg = downsample_agg
        self.show_border = bool(show_border)

    @property
    def region_index(self) -> pd.Index:
        return self.series.index

    def _aligned_values(self, region_index: pd.Index) -> np.ndarray:
        aligned = (
            self.series
            if self.series.index.equals(region_index)
            else self.series.reindex(region_index)
        )
        return aligned.to_numpy(dtype=np.float64)

    def _order_scores(
        self, region_index: pd.Index, metric: OrderMetric = "mean"
    ) -> np.ndarray:
        return self._aligned_values(region_index)

    def _clustering_data(self, region_index: pd.Index) -> np.ndarray:
        return self._aligned_values(region_index)[:, None]

    def __repr__(self) -> str:
        return f"BarTrack(name={self.name!r}, n_regions={len(self.series)})"


class PointTrack(BaseTrack):
    """1D continuous row annotation rendered as scatter points with optional trendline."""

    def __init__(
        self,
        data: pd.Series | Sequence[float] | np.ndarray,
        name: str = "",
        region_index: pd.Index | Sequence[Any] | None = None,
        color: str = "#333333",
        size: float = 4.0,
        alpha: float = 0.55,
        xlim: tuple[float, float] | None = None,
        zero_line: bool = False,
        smooth_line: bool = False,
        smooth_window: int | None = None,
        smooth_color: str = "#d62728",
        width: float = 0.6,
        xlabel: str | None = None,
        show_border: bool = True,
    ) -> None:
        if isinstance(data, pd.Series):
            s = data.astype(np.float64).copy()
            if region_index is not None:
                s.index = pd.Index(region_index)
        else:
            arr = np.asarray(data, dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(f"PointTrack `data` must be 1D, got shape {arr.shape}.")
            idx = pd.Index(region_index) if region_index is not None else pd.RangeIndex(len(arr))
            s = pd.Series(arr, index=idx)

        if not s.index.is_unique:
            raise ValueError("PointTrack `region_index` must be unique.")

        self.series = s
        self.name = str(name)
        self.color = color
        self.size = float(size)
        self.alpha = float(alpha)
        self.xlim = xlim
        self.zero_line = bool(zero_line)
        self.smooth_line = bool(smooth_line)
        self.smooth_window = smooth_window
        self.smooth_color = smooth_color
        self.width = float(width)
        self.xlabel = xlabel
        self.show_border = bool(show_border)

    @property
    def region_index(self) -> pd.Index:
        return self.series.index

    def _aligned_values(self, region_index: pd.Index) -> np.ndarray:
        aligned = (
            self.series
            if self.series.index.equals(region_index)
            else self.series.reindex(region_index)
        )
        return aligned.to_numpy(dtype=np.float64)

    def _order_scores(
        self, region_index: pd.Index, metric: OrderMetric = "mean"
    ) -> np.ndarray:
        return self._aligned_values(region_index)

    def _clustering_data(self, region_index: pd.Index) -> np.ndarray:
        return self._aligned_values(region_index)[:, None]

    def __repr__(self) -> str:
        return f"PointTrack(name={self.name!r}, n_regions={len(self.series)})"
