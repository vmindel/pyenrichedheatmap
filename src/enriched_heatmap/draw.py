"""Axes-level drawing: every function has the shape ``fn(data, ax, rows, **style)``."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import matplotlib as mpl
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.patches import Patch, Rectangle

from enriched_heatmap._matrix import NormalizedMatrix, _index_isna, _safe_get_indexer
from enriched_heatmap.profiles import ProfileMode, _validate_profile_params, compute_profile
from enriched_heatmap.raster import (
    DownsampleAgg,
    downsample_1d_rows,
    downsample_categorical_rows,
    downsample_rows,
)
from enriched_heatmap.rows import Rows

AnnotationKind = Literal["auto", "color", "categorical", "numeric", "bar"]

_FRAME: dict[str, Any] = {"color": "#333333", "lw": 0.6}
_GUIDE: dict[str, Any] = {"color": "0.3", "lw": 0.6, "ls": "--", "alpha": 0.6}
_PROFILE_MODES: dict[tuple[str, str | None], ProfileMode] = {
    ("mean", None): "mean",
    ("median", None): "median",
    ("mean", "sem"): "mean_sem",
    ("mean", "ci"): "mean_ci",
    ("mean", "boot"): "mean_boot_ci",
    ("median", "iqr"): "median_iqr",
}
_ANNOTATION_KINDS: tuple[str, ...] = ("auto", "color", "categorical", "numeric", "bar")


def _ensure_overlap(source: pd.Index, target: pd.Index, label: str = "`mat`") -> None:
    if len(source) == 0 or len(target) == 0 or source is target:
        return
    indexer = _safe_get_indexer(source, target)
    tgt_na = _index_isna(target)
    both_all_na = bool(tgt_na.all()) and bool(_index_isna(source).all())
    ok = bool(np.any(indexer >= 0)) if both_all_na else bool(np.any((indexer >= 0) & ~tgt_na))
    if not ok:
        raise ValueError(
            f"{label} shares no region identifiers with `rows` "
            f"(data index starts {list(source[:3])}, rows start {list(target[:3])}). "
            "Panels are matched by region identifier, not position."
        )


def _validate_finite_bound(val: float | None, name: str) -> float | None:
    if val is None:
        return None
    if (
        isinstance(val, (bool, np.bool_))
        or not isinstance(val, (int, float, np.number))
        or isinstance(val, (complex, np.complexfloating))
    ):
        raise ValueError(f"`{name}` must be a finite number, got {val!r}.")
    f = float(val)
    if not math.isfinite(f):
        raise ValueError(f"`{name}` must be finite, got {val!r}.")
    return f


def _validate_quantile_q(q: float | None) -> float | None:
    if q is None:
        return None
    if (
        isinstance(q, (bool, np.bool_))
        or not isinstance(q, (int, float, np.number))
        or isinstance(q, (complex, np.complexfloating))
    ):
        raise ValueError(f"`q` must be a quantile in [0, 1], got {q!r}.")
    qf = float(q)
    if not (math.isfinite(qf) and 0.0 <= qf <= 1.0):
        raise ValueError(f"`q` must be in [0, 1], got {q!r}.")
    return qf


def palette(
    categories: Sequence[Any],
    colors: str | mcolors.Colormap | Mapping[Any, Any] | Sequence[Any] = "tab10",
) -> dict[Any, Any]:
    """Map categories to colours.

    ``colors`` is a matplotlib colormap name or object, a list of colours (cycled), or a
    dict; categories missing from a dict fall back to ``tab10``.
    """
    cats = list(categories)
    if isinstance(colors, Mapping):
        fallback = mpl.colormaps["tab10"]
        return {c: colors.get(c, fallback(i % 10)) for i, c in enumerate(cats)}
    if isinstance(colors, (str, mcolors.Colormap)):
        cmap = mpl.colormaps[colors] if isinstance(colors, str) else colors
        if isinstance(cmap, mcolors.ListedColormap) and cmap.N <= 20:
            return {c: cmap(i % cmap.N) for i, c in enumerate(cats)}
        n = max(len(cats) - 1, 1)
        return {c: cmap(i / n) for i, c in enumerate(cats)}
    cols = list(colors)
    if not cols and cats:
        raise ValueError("`colors` must not be empty.")
    return {c: cols[i % len(cols)] for i, c in enumerate(cats)}


def _rows_for(data_index: pd.Index, rows: Rows | None) -> Rows:
    return Rows(index=pd.Index(data_index)) if rows is None else rows


def _root_figure(ax: Axes) -> Figure:
    fig: Any = ax.figure
    while not isinstance(fig, Figure):
        fig = fig.figure
    return fig


def _target_rows(
    ax: Axes,
    n_rows: int,
    max_rows: int | Literal["auto"] | str | None,
    dpi: float = 300.0,
) -> int | None:
    if max_rows is None:
        return None
    if isinstance(max_rows, str):
        if max_rows == "auto":
            fig = _root_figure(ax)
            if fig.get_layout_engine() is not None:
                # constrained/tight layout only positions axes at draw time; run it first
                fig.draw_without_rendering()
            bbox = ax.get_window_extent()
            return max(16, int(round(bbox.height / fig.dpi * dpi)))
        raise ValueError(
            f"`max_rows` must be a positive integer, 'auto', or None, got {max_rows!r}."
        )
    if isinstance(max_rows, (bool, np.bool_)) or not isinstance(max_rows, (int, np.integer)):
        raise ValueError(
            f"`max_rows` must be a positive integer, 'auto', or None, got {max_rows!r}."
        )
    mr = int(max_rows)
    if mr <= 0:
        raise ValueError(
            f"`max_rows` must be a positive integer, 'auto', or None, got {max_rows!r}."
        )
    return mr


def _norm(
    values: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    q: float | None,
    center: float | None,
) -> mcolors.Normalize:
    vmin_f = _validate_finite_bound(vmin, "vmin")
    vmax_f = _validate_finite_bound(vmax, "vmax")
    center_f = _validate_finite_bound(center, "center")
    q_f = _validate_quantile_q(q)

    finite = values[np.isfinite(values)]
    if center_f is not None:
        c = center_f
        if vmin_f is not None and vmin_f >= c:
            raise ValueError(
                f"`vmin` ({vmin_f}) must be strictly less than `center` ({c})."
            )
        if vmax_f is not None and vmax_f <= c:
            raise ValueError(
                f"`vmax` ({vmax_f}) must be strictly greater than `center` ({c})."
            )
        if vmin_f is None and vmax_f is None:
            span = (
                float(np.quantile(np.abs(finite - c), q_f if q_f is not None else 1.0))
                if finite.size
                else 1.0
            )
            span = span if span > 0 else 1.0
            lo, hi = c - span, c + span
        elif vmin_f is None:
            assert vmax_f is not None
            hi = vmax_f
            lo = c - (hi - c)
        elif vmax_f is None:
            lo = vmin_f
            hi = c + (c - lo)
        else:
            lo, hi = vmin_f, vmax_f
        return mcolors.TwoSlopeNorm(vmin=lo, vcenter=c, vmax=hi)

    if vmin_f is not None and vmax_f is not None and vmin_f > vmax_f:
        raise ValueError(f"`vmin` ({vmin_f}) must not exceed `vmax` ({vmax_f}).")

    if finite.size == 0:
        lo = vmin_f if vmin_f is not None else 0.0
        hi = vmax_f if vmax_f is not None else lo + 1.0
    else:
        lo = float(min(0.0, finite.min())) if vmin_f is None else vmin_f
        hi = (
            float(np.quantile(finite, q_f) if q_f is not None else finite.max())
            if vmax_f is None
            else vmax_f
        )
    if hi <= lo:
        hi = lo + 1.0
    return mcolors.Normalize(vmin=lo, vmax=hi)


def _center_for(mats: Sequence[NormalizedMatrix], center: float | None) -> float | None:
    """Differential matrices (``subtract``, ``log2_ratio``) are centred on 0 unless told otherwise."""
    if center is None and mats and all(m.metadata.get("is_differential") for m in mats):
        return 0.0
    return center


def _set_ylim(ax: Axes, rows: Rows) -> None:
    ax.set_ylim(rows.height or 1.0, 0)


def limits(
    matrices: NormalizedMatrix | Sequence[NormalizedMatrix] | Mapping[str, NormalizedMatrix],
    q: float | None = 0.99,
    center: float | None = None,
    rows: Rows | None = None,
) -> tuple[float, float]:
    """``(vmin, vmax)`` over one or several matrices, to put them on one colour scale.

    With ``center`` the range is symmetric around it; otherwise it runs from
    ``min(0, data)`` to the ``q`` quantile of all values. Differential matrices
    (from ``subtract`` or ``log2_ratio``) are centred on 0 when ``center`` is not given.
    When ``rows`` is given, each matrix is aligned to ``rows.index`` first (raising if
    a non-empty matrix shares no identifiers with ``rows``).
    """
    if isinstance(matrices, NormalizedMatrix):
        mats = [matrices]
    elif isinstance(matrices, Mapping):
        mats = list(matrices.values())
    elif isinstance(matrices, Sequence) and not isinstance(matrices, (str, bytes)):
        mats = list(matrices)
    else:
        raise TypeError(
            "`matrices` must be a NormalizedMatrix, a Sequence, or a Mapping of them."
        )
    if not mats:
        raise ValueError("`matrices` must not be empty.")
    _validate_finite_bound(center, "center")
    _validate_quantile_q(q)
    if rows is not None:
        aligned_mats: list[NormalizedMatrix] = []
        for m in mats:
            _ensure_overlap(m.region_index, rows.index, "`matrices`")
            aligned_mats.append(
                m.reindex(rows.index) if not m.region_index.equals(rows.index) else m
            )
        mats = aligned_mats
    eff_center = _center_for(mats, center)
    vals = np.concatenate([m.values.ravel() for m in mats])
    n = _norm(vals, None, None, q, eff_center)
    assert n.vmin is not None and n.vmax is not None
    return float(n.vmin), float(n.vmax)


def _xaxis(
    mat: NormalizedMatrix,
    ax: Axes,
    labels: str | tuple[str, str] | None,
    guides: bool,
    ticks: bool,
) -> None:
    x0, x1 = mat.x_extent
    ax.set_xlim(x0, x1)
    pos, lab = mat.default_xticks(anchor_labels=labels)
    ax.set_xticks(pos, lab)
    # Hide rather than clear: a profile shares its x-axis with the heatmap below.
    ax.tick_params(axis="x", labelsize=7, length=2 if ticks else 0, labelbottom=ticks)
    # Edge labels point inwards so neighbouring columns do not collide.
    tl = ax.get_xticklabels()
    if len(tl) > 1:
        # Centred matrices put the flank ticks on the outer bin centres, half a bin inside.
        slack = mat.bin_size / 2.0 if getattr(mat, "centered", False) else 0.0
        if pos[0] <= x0 + slack:
            tl[0].set_horizontalalignment("left")
        if pos[-1] >= x1 - slack:
            tl[-1].set_horizontalalignment("right")
    if guides:
        for g in mat.guide_positions():
            ax.axvline(g, **_GUIDE)


def _frame_groups(ax: Axes, rows: Rows, x0: float, x1: float, frame: bool) -> None:
    for s in ax.spines.values():
        s.set_visible(False)
    _set_ylim(ax, rows)
    # Hide locally rather than clearing the shared y-axis locator across figure columns.
    ax.tick_params(axis="y", left=False, labelleft=False)
    if frame:
        for _, _, _, y0, y1 in rows.spans():
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    zorder=3,
                    edgecolor=_FRAME["color"],
                    lw=_FRAME["lw"],
                    clip_on=False,
                )
            )


def heatmap(
    mat: NormalizedMatrix,
    ax: Axes,
    rows: Rows | None = None,
    cmap: str | mcolors.Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    q: float | None = 0.99,
    center: float | None = None,
    norm: mcolors.Normalize | None = None,
    na_color: str = "#eeeeee",
    max_rows: int | Literal["auto"] | None = "auto",
    agg: DownsampleAgg = "mean",
    anchor_labels: str | tuple[str, str] | None = None,
    guides: bool = True,
    xticks: bool = True,
    frame: bool = True,
    title: str | None = None,
) -> AxesImage | mcm.ScalarMappable:
    """Draw a region x bin matrix as a heatmap.

    Parameters
    ----------
    mat : NormalizedMatrix
    ax : matplotlib.axes.Axes
    rows : Rows, optional
        Order and groups from :func:`arrange`. Regions in ``rows`` but not in ``mat``
        are drawn as NaN. Default: the matrix order, one group.
    cmap : colormap, optional
        Default ``"Reds"``, or ``"RdBu_r"`` when ``center`` is set.
    vmin, vmax : float, optional
        Colour limits. Pass the same values (see :func:`limits`) to several heatmaps to
        put them on one scale.
    q : float or None
        Quantile used for ``vmax`` when it is not given, so a few hot bins do not wash
        out the rest. ``None`` uses the maximum.
    center : float, optional
        Diverging scale around this value (e.g. 0 for a log2 ratio). Differential
        matrices (from ``subtract`` or ``log2_ratio``) default to ``center=0``.
    norm : matplotlib.colors.Normalize, optional
        Overrides ``vmin``/``vmax``/``q``/``center``.
    max_rows : "auto", int or None
        More rows than this are averaged into this many before drawing. ``"auto"``
        is one row per pixel at 300 dpi, which keeps files small without visible loss.
        It measures the axes when called (running a constrained or tight layout first),
        so after a later resize of the figure pass an explicit number instead. ``None``
        draws every row.
    agg : str
        How rows are combined when downsampling: ``"mean"``, ``"max"``, ``"absmax"``,
        ``"median"``, ``"q90"``. Use ``"max"`` for sparse, narrow peaks.
    anchor_labels : str or (str, str), optional
        Tick label for the anchor (default ``"0"``), or the body start/end labels in
        scale-regions mode (default ``("Start", "End")``).
    guides : bool
        Dashed lines at the anchor (or body edges).
    xticks : bool
        Label the x-axis in bp/kb.
    frame : bool
        Box around each group.
    title : str, optional

    Returns
    -------
    matplotlib.image.AxesImage or matplotlib.cm.ScalarMappable
        For ``fig.colorbar`` or :func:`colorbar`. Every group shares its norm.
    """
    rows = _rows_for(mat.region_index, rows)
    _ensure_overlap(mat.region_index, rows.index, "`mat`")
    target = _target_rows(ax, len(rows), max_rows)
    vals = mat.reindex(rows.index).values if not mat.region_index.equals(rows.index) else mat.values
    _validate_finite_bound(center, "center")
    eff_center = _center_for([mat], center)
    if norm is None:
        norm = _norm(vals, vmin, vmax, q, eff_center)
    if cmap is None:
        cmap = "RdBu_r" if eff_center is not None else "Reds"
    cm = (mpl.colormaps[cmap] if isinstance(cmap, str) else cmap).with_extremes(bad=na_color)

    x0, x1 = mat.x_extent
    image: AxesImage | None = None
    for _, i0, i1, y0, y1 in rows.spans():
        block = vals[i0:i1]
        if block.shape[0] == 0:
            continue
        if target is not None:
            share = max(1, int(round(target * (i1 - i0) / max(len(rows), 1))))
            block = downsample_rows(block, share, agg=agg)
        image = ax.imshow(
            block,
            extent=(x0, x1, y1, y0),
            aspect="auto",
            cmap=cm,
            norm=norm,
            interpolation="nearest",
            rasterized=True,
        )
    _frame_groups(ax, rows, x0, x1, frame)
    _xaxis(mat, ax, anchor_labels, guides, xticks)
    if title is not None:
        ax.set_title(title, fontsize=9)
    if image is None:
        return mcm.ScalarMappable(norm=norm, cmap=cm)
    return image


def profile(
    mat: NormalizedMatrix,
    ax: Axes,
    rows: Rows | None = None,
    stat: str = "mean",
    band: str | None = None,
    by_group: bool = True,
    colors: str | mcolors.Colormap | Mapping[Any, Any] | Sequence[Any] = "tab10",
    color: Any = "0.15",
    ci: float = 0.95,
    n_boot: int = 200,
    lw: float = 1.2,
    alpha: float = 0.25,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
    guides: bool = True,
    xticks: bool = False,
    title: str | None = None,
) -> Axes:
    """Draw the average profile over regions (the metaplot above a heatmap).

    Parameters
    ----------
    mat : NormalizedMatrix
    ax : matplotlib.axes.Axes
    rows : Rows, optional
        Restricts to these regions; with groups and ``by_group`` draws one line each.
    stat : {"mean", "median"}
    band : {None, "sem", "ci", "boot", "iqr"}
        Shaded band: standard error, normal ``ci`` interval, bootstrap ``ci`` interval
        (with mean), or interquartile range (with median).
    ci : float
        Confidence level for the ``"ci"`` and ``"boot"`` bands.
    n_boot : int
        Bootstrap resamples for ``band="boot"``.
    by_group : bool
        One line per group (coloured as in :func:`annotation`, which has the same
        default ``colors``), or a single line over all rows.
    colors : colormap name, list or dict
        Group colours.
    color : colour
        Line colour when there is a single line.
    ylim : (float, float), optional
        Pass the same limits to several profiles to compare them.
    legend : bool
        Add a legend with group names.

    Returns
    -------
    matplotlib.axes.Axes
    """
    key = (stat, band)
    if key not in _PROFILE_MODES:
        raise ValueError(
            f"Unsupported stat/band combination {key}; choose from {list(_PROFILE_MODES)}."
        )
    mode = _PROFILE_MODES[key]
    ci, n_boot = _validate_profile_params(mode, ci, n_boot)
    rows = _rows_for(mat.region_index, rows)
    _ensure_overlap(mat.region_index, rows.index, "`mat`")
    vals = mat.reindex(rows.index).values if not mat.region_index.equals(rows.index) else mat.values
    x = mat.bin_centers()

    if by_group and rows.groups is not None:
        cols = palette(rows.labels, colors)
        parts: list[tuple[Any, np.ndarray, Any]] = [
            (lbl, vals[i0:i1], cols[lbl]) for lbl, i0, i1, _, _ in rows.spans()
        ]
    else:
        parts = [(None, vals, color)]
    for lbl, block, c in parts:
        res = compute_profile(block, mode=mode, ci=ci, n_bootstrap=n_boot)
        ax.plot(x, res.center, color=c, lw=lw, label=None if lbl is None else str(lbl))
        if res.lower is not None and res.upper is not None:
            ax.fill_between(x, res.lower, res.upper, color=c, alpha=alpha, lw=0)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", labelsize=7, length=2)
    _xaxis(mat, ax, None, guides, xticks)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend and len(parts) > 1:
        ax.legend(frameon=False, fontsize=7)
    if title is not None:
        ax.set_title(title, fontsize=9)
    return ax


def _is_categorical(s: pd.Series) -> bool:
    return (
        isinstance(s.dtype, pd.CategoricalDtype)
        or s.dtype == object
        or pd.api.types.is_bool_dtype(s.dtype)
        or pd.api.types.is_string_dtype(s.dtype)
    )


def _categories_for_series(series: pd.Series) -> list[Any]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return list(series.cat.categories)
    uniq = pd.Index(series.dropna().unique()).tolist()
    try:
        return sorted(uniq)
    except TypeError:
        return sorted(uniq, key=str)


def legends(
    ax: Axes,
    items: Sequence[tuple[str | None, Mapping[Any, Any]]],
    x: float = 1.05,
    y: float = 1.0,
) -> None:
    """Stack ``(title, {label: colour})`` legends down the right-hand side of ``ax``.

    ``x`` and ``y`` are axes coordinates of the first legend's upper-left corner.
    """
    fig = _root_figure(ax)
    cur_y = float(y)
    n_items = len(items)
    for idx, (title, colors) in enumerate(items):
        handles = [Patch(color=c, label=str(k)) for k, c in colors.items()]
        leg = Legend(
            ax,
            handles,
            [str(k) for k in colors],
            title=title,
            frameon=False,
            fontsize=7,
            title_fontsize=8,
            loc="upper left",
            bbox_to_anchor=(x, cur_y),
        )
        ax.add_artist(leg)
        leg.set_clip_on(False)  # add_artist clips to the axes, and these sit outside
        if idx + 1 < n_items:
            fig.draw_without_rendering()
            leg_box = leg.get_window_extent()
            pad_px = 6.0 * fig.dpi / 72.0
            _, cur_y = ax.transAxes.inverted().transform((0.0, float(leg_box.y0) - pad_px))


def annotation(
    data: pd.Series | pd.DataFrame,
    ax: Axes,
    rows: Rows | None = None,
    kind: AnnotationKind = "auto",
    colors: str | mcolors.Colormap | Mapping[Any, Any] | Sequence[Any] = "tab10",
    cmap: str | mcolors.Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    q: float | None = 0.99,
    center: float | None = None,
    bar_color: Any = "0.35",
    max_rows: int | Literal["auto"] | None = "auto",
    agg: DownsampleAgg = "mean",
    legend: bool = False,
    frame: bool = True,
    labels: bool = True,
) -> dict[Any, Any] | AxesImage | mcm.ScalarMappable | Axes:
    """Draw per-region values beside a heatmap: a colour strip or bars.

    Parameters
    ----------
    data : Series or DataFrame
        Indexed by region. A DataFrame draws one strip per column. With ``kind="auto"``
        or ``"color"``, each column is categorical or numeric on its own; numeric
        columns share one colour scale.
    ax : matplotlib.axes.Axes
    rows : Rows, optional
        Same object as for the heatmaps. Regions missing from ``data`` are blank.
    kind : {"auto", "color", "categorical", "numeric", "bar"}
        ``"auto"`` and ``"color"`` infer categorical vs numeric per column.
        ``"categorical"`` treats every column (including integer cluster IDs) as
        discrete categories. ``"numeric"`` converts every column with
        ``pandas.to_numeric(..., errors="raise")`` onto a continuous colour scale.
        ``"bar"`` draws one numeric column as horizontal bars.
    colors : colormap name, list or dict
        Category colours. The default matches :func:`profile`, so a group strip and the
        profile lines agree.
    cmap, vmin, vmax, q, center
        Colour scale for numeric strips, as in :func:`heatmap`.
    legend : bool
        Legend of categories, one per categorical column, to the right of the axes.
    labels : bool
        Column names under the strip.

    Returns
    -------
    dict, AxesImage, ScalarMappable or Axes
        For a single column: ``{category: colour}`` for categorical data (use with
        :func:`legend`), the image for numeric strips (use with :func:`colorbar`), the
        axes for bars. For several numeric columns, the image of the last one (all share
        one norm). For several columns of which some are categorical:
        ``{column: <that column's return value>}``.
    """
    if kind not in _ANNOTATION_KINDS:
        raise ValueError(f"`kind` must be one of {_ANNOTATION_KINDS}, got {kind!r}.")
    if isinstance(data, pd.Series):
        df = data.to_frame("" if data.name is None else data.name)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(
            f"`data` must be a pandas Series or DataFrame, got {type(data).__name__}."
        )
    ncol = df.shape[1]
    if ncol == 0:
        raise ValueError("`data` has no columns.")
    if kind == "bar" and ncol != 1:
        raise ValueError("kind='bar' takes a single column.")

    rows = _rows_for(df.index, rows)
    _ensure_overlap(df.index, rows.index, "`data`")
    target = _target_rows(ax, len(rows), max_rows)
    df = df.reindex(rows.index)
    first = df.iloc[:, 0]

    def share(i0: int, i1: int) -> int:
        assert target is not None
        return max(1, int(round(target * (i1 - i0) / max(len(rows), 1))))

    if kind == "bar":
        src_bar = first.astype(object) if isinstance(first.dtype, pd.CategoricalDtype) else first
        v = pd.to_numeric(src_bar, errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
        for _, i0, i1, y0, y1 in rows.spans():
            block = v[i0:i1]
            if target is not None:
                block = downsample_1d_rows(block, share(i0, i1), agg=agg)
            h = (y1 - y0) / max(len(block), 1)
            y = y0 + h * np.arange(len(block))
            ax.barh(
                y,
                np.where(np.isfinite(block), block, 0.0),
                height=h,
                align="edge",
                color=bar_color,
                lw=0,
            )
        _set_ylim(ax, rows)
        ax.tick_params(axis="y", left=False, labelleft=False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelsize=7, length=2)
        ax.axvline(0, color="0.3", lw=0.5)
        if labels and str(first.name):
            ax.set_xlabel(str(first.name), fontsize=8)
        return ax

    if kind == "categorical":
        categorical = [True] * ncol
    elif kind == "numeric":
        categorical = [False] * ncol
    else:
        categorical = [_is_categorical(df.iloc[:, j]) for j in range(ncol)]

    numeric: dict[int, np.ndarray] = {}
    for j in range(ncol):
        if not categorical[j]:
            col_s = df.iloc[:, j]
            if kind == "numeric":
                raw_s = (
                    col_s.astype(object)
                    if isinstance(col_s.dtype, pd.CategoricalDtype)
                    else col_s
                )
                numeric[j] = pd.to_numeric(raw_s, errors="raise").to_numpy(
                    dtype=np.float64, na_value=np.nan
                )
            else:
                numeric[j] = col_s.to_numpy(dtype=np.float64, na_value=np.nan)

    num_norm: mcolors.Normalize | None = None
    num_cm: mcolors.Colormap | None = None
    if numeric:
        num_norm = _norm(np.concatenate(list(numeric.values())), vmin, vmax, q, center)
        num_cm = (mpl.colormaps[cmap] if isinstance(cmap, str) else cmap).with_extremes(
            bad="#eeeeee"
        )

    results: dict[Any, Any] = {}
    legend_items: list[tuple[str | None, Mapping[Any, Any]]] = []
    for j, col in enumerate(df.columns):
        series = df.iloc[:, j]
        if categorical[j]:
            cats = _categories_for_series(series)
            pal = palette(cats, colors)
            codes = pd.Categorical(series, categories=cats).codes.astype(np.int64)
            lut = mcolors.ListedColormap([pal[c] for c in cats] or ["white"]).with_extremes(
                bad="white"
            )
            cat_norm = mcolors.BoundaryNorm(
                np.arange(-0.5, max(len(cats), 1) + 0.5), max(len(cats), 1)
            )
            for _, i0, i1, y0, y1 in rows.spans():
                block = codes[i0:i1]
                if block.shape[0] == 0:
                    continue
                if target is not None:
                    block = downsample_categorical_rows(block, share(i0, i1))
                img = np.where(block < 0, np.nan, block).astype(float)[:, None]
                ax.imshow(
                    img,
                    extent=(j, j + 1, y1, y0),
                    aspect="auto",
                    cmap=lut,
                    norm=cat_norm,
                    interpolation="nearest",
                    rasterized=True,
                )
            results[col] = pal
            legend_items.append((str(col) if ncol > 1 else None, pal))
        else:
            assert num_norm is not None and num_cm is not None
            vals_2d = numeric[j][:, None]
            image: AxesImage | None = None
            for _, i0, i1, y0, y1 in rows.spans():
                block_2d = vals_2d[i0:i1]
                if block_2d.shape[0] == 0:
                    continue
                if target is not None:
                    block_2d = downsample_rows(block_2d, share(i0, i1), agg=agg)
                image = ax.imshow(
                    block_2d,
                    extent=(j, j + 1, y1, y0),
                    aspect="auto",
                    cmap=num_cm,
                    norm=num_norm,
                    interpolation="nearest",
                    rasterized=True,
                )
            results[col] = (
                image if image is not None else mcm.ScalarMappable(norm=num_norm, cmap=num_cm)
            )

    _frame_groups(ax, rows, 0, ncol, frame)
    ax.set_xlim(0, ncol)
    if labels and any(str(c) for c in df.columns):
        ax.set_xticks(np.arange(ncol) + 0.5, [str(c) for c in df.columns], rotation=90, fontsize=8)
        ax.tick_params(axis="x", length=0)
    else:
        ax.set_xticks([])
    if legend and legend_items:
        legends(ax, legend_items)
    if ncol == 1:
        return results[df.columns[0]]
    if not any(categorical):
        return results[df.columns[-1]]  # every strip shares this image's norm
    return results


def group_labels(
    ax: Axes, rows: Rows, counts: bool = True, side: str = "left", fontsize: float = 8
) -> None:
    """Write each group's label (and size) beside its block of rows."""
    if rows.groups is None:
        return
    x, ha = (-0.15, "right") if side == "left" else (1.15, "left")
    for lbl, i0, i1, y0, y1 in rows.spans():
        text = f"{lbl}\nn={i1 - i0:,}" if counts else str(lbl)
        ax.text(
            x,
            (y0 + y1) / 2,
            text,
            transform=ax.get_yaxis_transform(),
            ha=ha,
            va="center",
            fontsize=fontsize,
        )


def legend(
    ax: Axes,
    colors: Mapping[Any, Any],
    title: str | None = None,
    loc: str | tuple[float, float] | int | None = "upper left",
    **kwargs: Any,
) -> None:
    """Legend from a ``{label: colour}`` dict, such as :func:`annotation` returns."""
    handles = [Patch(color=c, label=str(k)) for k, c in colors.items()]
    ax.legend(
        handles=handles,
        title=title,
        frameon=False,
        fontsize=7,
        title_fontsize=8,
        loc=cast(Any, loc),
        **kwargs,
    )


def colorbar(
    mappable: mcm.ScalarMappable,
    cax: Axes,
    label: str | None = None,
    orientation: Literal["vertical", "horizontal"] = "horizontal",
) -> Colorbar:
    """A thin colour bar with small ticks; returns the matplotlib Colorbar."""
    cb = cax.figure.colorbar(mappable, cax=cax, orientation=orientation)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.5)
    if label:
        cb.set_label(label, fontsize=8)
    return cb
