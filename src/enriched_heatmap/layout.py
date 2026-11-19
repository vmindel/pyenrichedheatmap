"""HeatmapList horizontal composition, deterministic GridSpec geometry, and hybrid vector/raster rendering."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from enriched_heatmap.ordering import OrderMetric, resolve_ordering_and_splits
from enriched_heatmap.profiles import compute_profile
from enriched_heatmap.raster import (
    DownsampleAgg,
    compute_target_raster_rows,
    downsample_1d_rows,
    downsample_categorical_rows,
    downsample_rows,
)
from enriched_heatmap.tracks import (
    BarTrack,
    BaseTrack,
    CategoricalTrack,
    EnrichedTrack,
    MatrixTrack,
    PointTrack,
    resolve_palette,
)

JoinMode = Literal["inner", "outer", "left"]


def _build_norm(
    values_list: list[np.ndarray],
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_q: float | None = None,
    vmax_q: float | None = 0.99,
    center: float | None = None,
) -> mcolors.Normalize:
    """Compute Normalize or TwoSlopeNorm from one or more numeric arrays."""
    finite_parts = [v[np.isfinite(v)] for v in values_list if v.size > 0]
    finite = np.concatenate(finite_parts) if finite_parts else np.array([], dtype=np.float64)

    if finite.size == 0:
        lo, hi = (0.0, 1.0) if center is None else (center - 1.0, center + 1.0)
    else:
        if vmin is not None:
            lo = float(vmin)
        elif vmin_q is not None:
            lo = float(np.quantile(finite, vmin_q))
        elif center is not None:
            q = 0.99 if vmax_q is None else vmax_q
            lo = float(np.quantile(finite, 1.0 - q))
        else:
            lo = float(np.min(finite))

        if vmax is not None:
            hi = float(vmax)
        elif vmax_q is not None:
            hi = float(np.quantile(finite, vmax_q))
        else:
            hi = float(np.max(finite))

    if center is not None:
        c = float(center)
        # Symmetric range around center when explicit vmin/vmax weren't both fixed
        if vmin is None and vmax is None:
            span = max(abs(hi - c), abs(c - lo))
            if span <= 0:
                span = 1.0
            lo, hi = c - span, c + span
        else:
            if lo >= c:
                lo = c - max(abs(hi - c), 1e-6)
            if hi <= c:
                hi = c + max(abs(c - lo), 1e-6)
        return mcolors.TwoSlopeNorm(vmin=lo, vcenter=c, vmax=hi)

    if hi <= lo:
        hi = lo + 1.0
    return mcolors.Normalize(vmin=lo, vmax=hi)


def _get_cmap(cmap: str | mcolors.Colormap, na_color: str = "#eeeeee") -> mcolors.Colormap:
    cm = mpl.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
    return cm.with_extremes(bad=na_color)


def _style_box_spines(ax: plt.Axes, visible: bool = True, color: str = "#333333", lw: float = 0.6) -> None:
    for spine in ax.spines.values():
        spine.set_visible(visible)
        if visible:
            spine.set_color(color)
            spine.set_linewidth(lw)


def _running_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Fast NaN-aware centered running mean for PointTrack trendline."""
    n = len(y)
    if n == 0 or window <= 1:
        return y.copy()
    w = min(window, n)
    if w % 2 == 0:
        w = max(1, w - 1)
    pad = w // 2
    finite = np.isfinite(y)
    vals = np.where(finite, y, 0.0)
    wts = finite.astype(np.float64)
    padded_v = np.pad(vals, pad, mode="edge")
    padded_w = np.pad(wts, pad, mode="edge")
    cs_v = np.cumsum(np.insert(padded_v, 0, 0.0))
    cs_w = np.cumsum(np.insert(padded_w, 0, 0.0))
    sum_v = cs_v[w:] - cs_v[:-w]
    sum_w = cs_w[w:] - cs_w[:-w]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sum_w > 0, sum_v / sum_w, np.nan)


class HeatmapList:
    """Horizontally composed collection of aligned genomic and annotation tracks.

    Constructed directly via ``HeatmapList([t1, t2, ...])`` or declaratively with ``+``::

        hm = EnrichedTrack(atac, name="ATAC") + MatrixTrack(rna, name="RNA")
        fig = hm.plot(order_by="ATAC", split_by=3)
    """

    def __init__(self, tracks: Sequence[BaseTrack]) -> None:
        track_list = list(tracks)
        if not track_list:
            raise ValueError("HeatmapList requires at least one track.")
        for t in track_list:
            if not isinstance(t, BaseTrack):
                raise TypeError(f"Expected BaseTrack instance, got {type(t).__name__}.")
        self.tracks: list[BaseTrack] = track_list

    def __add__(self, other: BaseTrack | HeatmapList) -> HeatmapList:
        if isinstance(other, HeatmapList):
            return HeatmapList([*self.tracks, *other.tracks])
        if isinstance(other, BaseTrack):
            return HeatmapList([*self.tracks, other])
        return NotImplemented

    def __len__(self) -> int:
        return len(self.tracks)

    def __iter__(self) -> Iterator[BaseTrack]:
        return iter(self.tracks)

    def __getitem__(self, key: int | str) -> BaseTrack:
        if isinstance(key, str):
            for t in self.tracks:
                if t.name == key:
                    return t
            raise KeyError(f"Track {key!r} not found in HeatmapList.")
        return self.tracks[key]

    def align_index(self, join: JoinMode = "inner") -> pd.Index:
        """Compute the shared ``region_index`` across all tracks with zero desync."""
        idx = self.tracks[0].region_index.copy()
        if join == "left":
            return idx
        for t in self.tracks[1:]:
            other = t.region_index
            if join == "inner":
                idx = idx.intersection(other, sort=False)
            elif join == "outer":
                idx = idx.union(other, sort=False)
            else:
                raise ValueError(f"`join` must be 'inner', 'outer', or 'left', got {join!r}.")

        if len(idx) == 0:
            raise ValueError(
                "Zero overlapping regions across tracks after alignment! "
                "Verify that all tracks share matching `region_index` identifiers."
            )
        return idx

    def _resolve_norms(
        self, region_index: pd.Index
    ) -> tuple[dict[int, mcolors.Normalize], dict[str, list[int]]]:
        """Resolve per-track and shared scale_group color normalizations."""
        groups: dict[str, list[int]] = {}
        norms: dict[int, mcolors.Normalize] = {}

        for i, t in enumerate(self.tracks):
            if isinstance(t, (EnrichedTrack, MatrixTrack)) and t.scale_group is not None:
                groups.setdefault(t.scale_group, []).append(i)

        # Shared scale groups
        for _gname, idxs in groups.items():
            vals_list = [self.tracks[i]._aligned_numeric_values(region_index) for i in idxs]
            ref_tracks = [self.tracks[i] for i in idxs]
            vmins = [t.vmin for t in ref_tracks if t.vmin is not None]
            vmaxs = [t.vmax for t in ref_tracks if t.vmax is not None]
            vmin_qs = [t.vmin_q for t in ref_tracks if t.vmin_q is not None]
            vmax_qs = [t.vmax_q for t in ref_tracks if t.vmax_q is not None]
            centers = [t.center for t in ref_tracks if t.center is not None]

            shared_norm = _build_norm(
                vals_list,
                vmin=min(vmins) if vmins else None,
                vmax=max(vmaxs) if vmaxs else None,
                vmin_q=min(vmin_qs) if vmin_qs else None,
                vmax_q=max(vmax_qs) if vmax_qs else 0.99,
                center=centers[0] if centers else None,
            )
            for i in idxs:
                norms[i] = shared_norm

        # Independent tracks
        for i, t in enumerate(self.tracks):
            if i in norms:
                continue
            if isinstance(t, (EnrichedTrack, MatrixTrack)):
                vals = t._aligned_numeric_values(region_index)
                norms[i] = _build_norm(
                    [vals],
                    vmin=t.vmin,
                    vmax=t.vmax,
                    vmin_q=t.vmin_q,
                    vmax_q=t.vmax_q,
                    center=t.center,
                )

        return norms, groups

    def plot(
        self,
        order_by: str | Callable | pd.Series | pd.Index | np.ndarray | None = None,
        order_metric: OrderMetric = "center_mean",
        ascending: bool = False,
        split_by: pd.Series | pd.Categorical | Sequence[Any] | int | None = None,
        cluster_tracks: list[str] | None = None,
        cluster_palette: str | dict[Any, Any] | Sequence[Any] = "tab10",
        show_cluster_strip: bool | None = None,
        show_row_counts: bool = True,
        row_gap: float = 0.03,
        col_gap: float = 0.22,
        top_profile_height: float = 1.35,
        body_height: float = 6.0,
        colorbar_height: float = 0.22,
        figsize: tuple[float, float] | None = None,
        raster_dpi: int = 300,
        max_raster_rows: int | None = None,
        downsample_agg: DownsampleAgg = "mean",
        join: JoinMode = "inner",
        title: str | None = None,
        random_state: int = 42,
        return_axes: bool = False,
    ) -> plt.Figure | tuple[plt.Figure, dict[str, Any]]:
        """Render the horizontally composed figure with hybrid raster body + vector chrome.

        Parameters
        ----------
        order_by : str | Callable | pd.Series | pd.Index | np.ndarray | None
            Track name, ``"kmeans"``, ``"hierarchical"``, callable, Series, Index, or score array.
        order_metric : str | Callable, default "center_mean"
            Metric used when ordering by an :class:`EnrichedTrack` or :class:`MatrixTrack`.
        ascending : bool, default False
            Sort direction (False = strongest signal at top).
        split_by : pd.Series | pd.Categorical | Sequence | int | None
            Cluster assignments or integer ``k`` for automatic k-means clustering.
        cluster_tracks : list[str] | None
            Track names used when ``split_by`` is an integer or ``order_by="kmeans"``/``"hierarchical"``.
        cluster_palette : str | dict | Sequence, default "tab10"
            Colors for top metaplot cluster curves and the left cluster strip.
        show_cluster_strip : bool | None, default None
            Show colored cluster strip on the left (defaults to True when split into >1 group).
        show_row_counts : bool, default True
            Annotate region counts (``n = ...``) along the right margin.
        row_gap : float, default 0.03
            Vertical gap between split clusters as a fraction of total body height.
        col_gap : float, default 0.14
            Horizontal spacing between track columns (matplotlib ``wspace``).
        figsize : tuple[float, float] | None
            Figure size in inches; auto-computed cleanly from track widths if omitted.
        raster_dpi : int, default 300
            Target DPI for rasterized heatmap bodies and smart row downsampling.
        max_raster_rows : int | None
            Optional hard cap on downsampled raster rows for the total heatmap body.
        downsample_agg : str | Callable, default "mean"
            Default bucket aggregation when ``n_rows`` exceeds pixel height (``"mean"``,
            ``"max"``, ``"q90"``, ``"absmax"``, etc.; per-track overrides take precedence).
        join : {"inner", "outer", "left"}, default "inner"
            How to align differing ``region_index`` across tracks.
        title : str | None
            Optional overall figure super-title.
        return_axes : bool, default False
            If True, return ``(fig, info_dict)`` containing axes, resolved order, splits, and norms.
        """
        base_index = self.align_index(join=join)
        ordered_index, split_series, split_slices = resolve_ordering_and_splits(
            tracks=self.tracks,
            region_index=base_index,
            order_by=order_by,
            order_metric=order_metric,
            ascending=ascending,
            split_by=split_by,
            cluster_tracks=cluster_tracks,
            random_state=random_state,
        )

        n_total = len(ordered_index)
        n_splits = len(split_slices)
        is_split = split_series is not None and n_splits > 0
        use_cluster_strip = is_split if show_cluster_strip is None else (bool(show_cluster_strip) and is_split)

        cluster_labels = [lbl for lbl, _ in split_slices]
        cluster_colors = (
            resolve_palette(cluster_labels, cluster_palette)
            if is_split
            else {"": "#222222"}
        )

        norms, scale_groups = self._resolve_norms(ordered_index)

        # Determine columns: optional left cluster strip + one column per track
        strip_width = 0.18
        col_widths: list[float] = []
        if use_cluster_strip:
            col_widths.append(strip_width)
        track_col_offset = 1 if use_cluster_strip else 0
        for t in self.tracks:
            col_widths.append(t.width)
        n_cols = len(col_widths)

        has_top_row = any(t.has_top_profile for t in self.tracks)
        has_cbar_row = any(t.has_colorbar for t in self.tracks)
        cat_tracks_with_legend = [
            t for t in self.tracks if isinstance(t, CategoricalTrack) and t.show_legend
        ]

        # Auto figure size
        if figsize is None:
            total_w = sum(col_widths)
            fig_w = max(4.2, min(20.0, total_w * 1.45 + (1.2 if show_row_counts else 0.6)))
            fig_h = 5.2 + (1.3 if has_top_row else 0.0) + (0.8 if has_cbar_row else 0.0)
            if cat_tracks_with_legend:
                fig_h += 0.45
            figsize = (fig_w, fig_h)

        # Build outer vertical row ratios with explicit spacer bands for zero collisions
        row_ratios: list[float] = []
        row_roles: list[str] = []
        if has_top_row:
            row_ratios.append(top_profile_height)
            row_roles.append("top")
            row_ratios.append(0.22)
            row_roles.append("top_pad")
        row_ratios.append(body_height)
        row_roles.append("body")
        if has_cbar_row:
            # Leave generous room for rotated MatrixTrack labels / EnrichedTrack x-ticks
            has_matrix_labels = any(
                isinstance(t, MatrixTrack) and t.col_labels for t in self.tracks
            )
            pad_h = 0.85 if has_matrix_labels else 0.55
            row_ratios.append(pad_h)
            row_roles.append("cbar_pad")
            row_ratios.append(colorbar_height)
            row_roles.append("cbar")

        fig = plt.figure(figsize=figsize, dpi=min(raster_dpi, 150))
        fig.set_facecolor("white")

        bottom_margin = 0.14 if cat_tracks_with_legend else 0.08
        top_margin = 0.89 if title else 0.93
        max_cluster_lbl_len = max((len(lbl) for lbl in cluster_labels), default=0) if use_cluster_strip else 0
        if use_cluster_strip:
            left_margin = min(0.18, 0.085 + 0.006 * max_cluster_lbl_len)
        elif has_top_row:
            left_margin = 0.09
        else:
            left_margin = 0.06
        right_margin = 0.89 if show_row_counts else 0.96

        outer_gs = gridspec.GridSpec(
            nrows=len(row_ratios),
            ncols=1,
            figure=fig,
            height_ratios=row_ratios,
            hspace=0.0,
            left=left_margin,
            right=right_margin,
            top=top_margin,
            bottom=bottom_margin,
        )

        role_to_outer_idx = {role: i for i, role in enumerate(row_roles)}

        # Compute body height fraction for smart raster pixel budgeting
        total_ratio = sum(row_ratios)
        body_fig_frac = (top_margin - bottom_margin) * (body_height / total_ratio)
        total_target_rows = compute_target_raster_rows(
            fig_height_inches=figsize[1],
            axes_height_fraction=body_fig_frac,
            raster_dpi=raster_dpi,
            max_rows=max_raster_rows,
        )

        # Build interleaved vertical split ratios inside body_row (exact pixel alignment!)
        split_sizes = [sl.stop - sl.start for _, sl in split_slices]
        gap_units = max(1e-6, row_gap * max(n_total, 1))
        body_row_ratios: list[float] = []
        split_grid_rows: list[int] = []
        for s_i, sz in enumerate(split_sizes):
            split_grid_rows.append(len(body_row_ratios))
            body_row_ratios.append(float(max(sz, 1)))
            if s_i < n_splits - 1:
                body_row_ratios.append(gap_units)

        body_gs = gridspec.GridSpecFromSubplotSpec(
            nrows=len(body_row_ratios),
            ncols=n_cols,
            subplot_spec=outer_gs[role_to_outer_idx["body"]],
            height_ratios=body_row_ratios,
            width_ratios=col_widths,
            hspace=0.0,
            wspace=col_gap,
        )

        top_gs = (
            gridspec.GridSpecFromSubplotSpec(
                nrows=1,
                ncols=n_cols,
                subplot_spec=outer_gs[role_to_outer_idx["top"]],
                width_ratios=col_widths,
                wspace=col_gap,
            )
            if has_top_row
            else None
        )

        cbar_gs = (
            gridspec.GridSpecFromSubplotSpec(
                nrows=1,
                ncols=n_cols,
                subplot_spec=outer_gs[role_to_outer_idx["cbar"]],
                width_ratios=col_widths,
                wspace=col_gap,
            )
            if has_cbar_row
            else None
        )

        axes_info: dict[str, Any] = {
            "top": {},
            "body": {},
            "cluster_strip": [],
            "colorbars": {},
        }

        # --------------------------------------------------------------
        # 1. Render optional left cluster strip
        # --------------------------------------------------------------
        if use_cluster_strip:
            for s_i, (grp_lbl, _sl) in enumerate(split_slices):
                ax_strip = fig.add_subplot(body_gs[split_grid_rows[s_i], 0])
                ax_strip.set_facecolor(cluster_colors[grp_lbl])
                ax_strip.set_xticks([])
                ax_strip.set_yticks([])
                _style_box_spines(ax_strip, visible=True, color="#333333", lw=0.5)
                ax_strip.text(
                    -0.35,
                    0.5,
                    grp_lbl,
                    transform=ax_strip.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="#222222",
                )
                axes_info["cluster_strip"].append(ax_strip)

        # --------------------------------------------------------------
        # 2. Render each track column (top metaplot + split body panels)
        # --------------------------------------------------------------
        for t_i, track in enumerate(self.tracks):
            col_i = track_col_offset + t_i
            track_key = track.name or f"track_{t_i}"
            body_axes_for_track: list[plt.Axes] = []

            if isinstance(track, EnrichedTrack):
                mat = track._aligned_matrix(ordered_index)
                norm = norms[t_i]
                cm = _get_cmap(track.cmap, na_color=track.na_color)
                x_min, x_max = mat.x_extent
                centers = mat.bin_centers()
                guides = mat.guide_positions()
                eff_agg = track.downsample_agg if track.downsample_agg is not None else downsample_agg

                guide_style = {"color": "#222222", "linestyle": "--", "linewidth": 0.7, "alpha": 0.65}
                guide_style.update(track.guide_kwargs)

                # Top metaplot (vector!)
                if top_gs is not None and track.has_top_profile:
                    ax_top = fig.add_subplot(top_gs[0, col_i])
                    for grp_lbl, sl in split_slices:
                        sub_vals = mat.values[sl]
                        prof = compute_profile(
                            sub_vals,
                            mode=track.top_profile,
                            ci=track.top_profile_ci,
                            random_state=random_state,
                        )
                        line_color = cluster_colors[grp_lbl] if is_split else "#222222"
                        ax_top.plot(
                            centers,
                            prof.center,
                            color=line_color,
                            linewidth=1.4,
                            label=grp_lbl if is_split else None,
                            rasterized=False,
                        )
                        if prof.lower is not None and prof.upper is not None:
                            ax_top.fill_between(
                                centers,
                                prof.lower,
                                prof.upper,
                                color=line_color,
                                alpha=0.22,
                                linewidth=0,
                                rasterized=False,
                            )

                    if track.guide_lines:
                        for gx in guides:
                            ax_top.axvline(gx, **{**guide_style, "zorder": 1})

                    ax_top.set_xlim(x_min, x_max)
                    if track.top_profile_ylim is not None:
                        ax_top.set_ylim(*track.top_profile_ylim)
                    ax_top.set_xticks([])
                    ax_top.tick_params(axis="y", labelsize=7.5, length=2.5, pad=2)
                    ax_top.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
                    ax_top.spines["top"].set_visible(False)
                    ax_top.spines["right"].set_visible(False)
                    ax_top.spines["left"].set_linewidth(0.6)
                    ax_top.spines["bottom"].set_linewidth(0.6)
                    if track.top_profile_ylabel:
                        ax_top.set_ylabel(track.top_profile_ylabel, fontsize=7.5)
                    if track.name:
                        ax_top.set_title(track.name, fontsize=9.5, fontweight="bold", pad=5)
                    axes_info["top"][track_key] = ax_top

                # Body panels (rasterized imshow + vector chrome!)
                ticks, tick_labels = (
                    (track.xticks, track.xtick_labels)
                    if track.xticks is not None
                    else mat.default_xticks(anchor_labels=track.anchor_labels)
                )

                for s_i, (_grp_lbl, sl) in enumerate(split_slices):
                    ax_b = fig.add_subplot(body_gs[split_grid_rows[s_i], col_i])
                    sub_vals = mat.values[sl]
                    cluster_n = sub_vals.shape[0]
                    cluster_target = max(
                        1,
                        int(round(total_target_rows * (cluster_n / max(n_total, 1)))),
                    )
                    disp_vals = downsample_rows(sub_vals, target_rows=cluster_target, agg=eff_agg)

                    ax_b.imshow(
                        disp_vals,
                        aspect="auto",
                        interpolation="none",
                        cmap=cm,
                        norm=norm,
                        extent=[x_min, x_max, cluster_n, 0],
                        rasterized=True,
                    )
                    if track.guide_lines:
                        for gx in guides:
                            ax_b.axvline(gx, **guide_style)

                    ax_b.set_xlim(x_min, x_max)
                    ax_b.set_yticks([])
                    _style_box_spines(ax_b, visible=track.show_border)

                    if s_i == 0 and not track.has_top_profile and track.name:
                        ax_b.set_title(track.name, fontsize=9.5, fontweight="bold", pad=5)

                    if s_i == n_splits - 1:
                        ax_b.set_xticks(ticks)
                        if tick_labels is not None:
                            tick_objs = ax_b.set_xticklabels(tick_labels, fontsize=8)
                            tol = max(abs(x_max - x_min) * 1e-6, 1e-9)
                            for pos, lbl_obj in zip(ticks, tick_objs, strict=True):
                                if abs(pos - x_min) <= tol:
                                    lbl_obj.set_horizontalalignment("left")
                                elif abs(pos - x_max) <= tol:
                                    lbl_obj.set_horizontalalignment("right")
                        ax_b.tick_params(axis="x", length=3, pad=2)
                    else:
                        ax_b.set_xticks([])

                    body_axes_for_track.append(ax_b)

            elif isinstance(track, MatrixTrack):
                df = track._aligned_df(ordered_index)
                vals = df.to_numpy(dtype=np.float64)
                n_mcols = vals.shape[1]
                norm = norms[t_i]
                cm = _get_cmap(track.cmap, na_color=track.na_color)
                eff_agg = track.downsample_agg if track.downsample_agg is not None else downsample_agg

                if top_gs is not None and track.has_top_profile:
                    ax_top = fig.add_subplot(top_gs[0, col_i])
                    x_pos = np.arange(n_mcols, dtype=np.float64) + 0.5
                    for grp_lbl, sl in split_slices:
                        prof = compute_profile(
                            vals[sl],
                            mode=track.top_profile,
                            ci=track.top_profile_ci,
                            random_state=random_state,
                        )
                        line_color = cluster_colors[grp_lbl] if is_split else "#222222"
                        ax_top.plot(
                            x_pos,
                            prof.center,
                            color=line_color,
                            linewidth=1.3,
                            marker="o",
                            markersize=2.5,
                            rasterized=False,
                        )
                        if prof.lower is not None and prof.upper is not None:
                            ax_top.fill_between(
                                x_pos,
                                prof.lower,
                                prof.upper,
                                color=line_color,
                                alpha=0.22,
                                linewidth=0,
                                rasterized=False,
                            )
                    ax_top.set_xlim(0, n_mcols)
                    if track.top_profile_ylim is not None:
                        ax_top.set_ylim(*track.top_profile_ylim)
                    ax_top.set_xticks([])
                    ax_top.tick_params(axis="y", labelsize=7.5, length=2.5, pad=2)
                    ax_top.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
                    ax_top.spines["top"].set_visible(False)
                    ax_top.spines["right"].set_visible(False)
                    if track.name:
                        ax_top.set_title(track.name, fontsize=9.5, fontweight="bold", pad=5)
                    axes_info["top"][track_key] = ax_top

                for s_i, (_grp_lbl, sl) in enumerate(split_slices):
                    ax_b = fig.add_subplot(body_gs[split_grid_rows[s_i], col_i])
                    sub_vals = vals[sl]
                    cluster_n = sub_vals.shape[0]
                    cluster_target = max(
                        1,
                        int(round(total_target_rows * (cluster_n / max(n_total, 1)))),
                    )
                    disp_vals = downsample_rows(sub_vals, target_rows=cluster_target, agg=eff_agg)
                    ax_b.imshow(
                        disp_vals,
                        aspect="auto",
                        interpolation="none",
                        cmap=cm,
                        norm=norm,
                        extent=[0, n_mcols, cluster_n, 0],
                        rasterized=True,
                    )
                    ax_b.set_yticks([])
                    _style_box_spines(ax_b, visible=track.show_border)

                    if s_i == 0 and not track.has_top_profile and track.name:
                        ax_b.set_title(track.name, fontsize=9.5, fontweight="bold", pad=5)

                    if s_i == n_splits - 1 and track.col_labels:
                        ax_b.set_xticks(np.arange(n_mcols) + 0.5)
                        ax_b.set_xticklabels(
                            [str(c) for c in df.columns],
                            fontsize=7.5,
                            rotation=track.col_label_rotation,
                            ha="right" if track.col_label_rotation else "center",
                        )
                        ax_b.tick_params(axis="x", length=2.5, pad=2)
                    else:
                        ax_b.set_xticks([])

                    body_axes_for_track.append(ax_b)

            elif isinstance(track, CategoricalTrack):
                codes = track._aligned_codes(ordered_index)
                palette_list = [track.color_map[c] for c in track.categories]
                n_cats = max(len(palette_list), 1)
                cat_cmap = mcolors.ListedColormap(
                    palette_list if palette_list else [track.na_color]
                ).with_extremes(bad=track.na_color)
                cat_norm = mcolors.Normalize(vmin=-0.5, vmax=n_cats - 0.5)

                for s_i, (_grp_lbl, sl) in enumerate(split_slices):
                    ax_b = fig.add_subplot(body_gs[split_grid_rows[s_i], col_i])
                    sub_codes = codes[sl]
                    cluster_n = len(sub_codes)
                    cluster_target = max(
                        1,
                        int(round(total_target_rows * (cluster_n / max(n_total, 1)))),
                    )
                    disp_codes = downsample_categorical_rows(sub_codes, target_rows=cluster_target)
                    img = np.where(disp_codes >= 0, disp_codes.astype(np.float64), np.nan)[:, None]

                    ax_b.imshow(
                        img,
                        aspect="auto",
                        interpolation="none",
                        cmap=cat_cmap,
                        norm=cat_norm,
                        extent=[0, 1, cluster_n, 0],
                        rasterized=True,
                    )
                    ax_b.set_xticks([])
                    ax_b.set_yticks([])
                    _style_box_spines(ax_b, visible=track.show_border)

                    if s_i == 0 and track.name:
                        ax_b.set_title(track.name, fontsize=9.0, fontweight="bold", pad=5)

                    body_axes_for_track.append(ax_b)

            elif isinstance(track, BarTrack):
                vals = track._aligned_values(ordered_index)
                finite = vals[np.isfinite(vals)]
                if track.xlim is not None:
                    x_lo, x_hi = track.xlim
                elif finite.size > 0:
                    lo_v, hi_v = float(np.min(finite)), float(np.max(finite))
                    x_lo = min(0.0, lo_v * 1.05)
                    x_hi = max(0.0, hi_v * 1.05)
                    if x_hi <= x_lo:
                        x_hi = x_lo + 1.0
                else:
                    x_lo, x_hi = 0.0, 1.0

                eff_agg = track.downsample_agg if track.downsample_agg is not None else downsample_agg

                for s_i, (_grp_lbl, sl) in enumerate(split_slices):
                    ax_b = fig.add_subplot(body_gs[split_grid_rows[s_i], col_i])
                    sub_vals = vals[sl]
                    cluster_n = len(sub_vals)
                    cluster_target = max(
                        1,
                        int(round(total_target_rows * (cluster_n / max(n_total, 1)))),
                    )
                    disp = downsample_1d_rows(sub_vals, target_rows=cluster_target, agg=eff_agg)
                    y_pos = np.linspace(0.5, max(cluster_n - 0.5, 0.5), len(disp))
                    safe_disp = np.where(np.isfinite(disp), disp, 0.0)

                    if len(disp) <= 250:
                        bar_h = max(cluster_n / max(len(disp), 1), 1.0) * 0.85
                        colors = (
                            [
                                track.neg_color if (track.neg_color and v < 0) else track.color
                                for v in safe_disp
                            ]
                            if track.neg_color
                            else track.color
                        )
                        ax_b.barh(y_pos, safe_disp, height=bar_h, color=colors, linewidth=0)
                    else:
                        if track.neg_color:
                            ax_b.fill_betweenx(
                                y_pos,
                                0.0,
                                np.maximum(safe_disp, 0.0),
                                color=track.color,
                                linewidth=0,
                                rasterized=True,
                            )
                            ax_b.fill_betweenx(
                                y_pos,
                                np.minimum(safe_disp, 0.0),
                                0.0,
                                color=track.neg_color,
                                linewidth=0,
                                rasterized=True,
                            )
                        else:
                            ax_b.fill_betweenx(
                                y_pos,
                                0.0,
                                safe_disp,
                                color=track.color,
                                linewidth=0,
                                rasterized=True,
                            )

                    if track.zero_line and x_lo < 0.0 < x_hi:
                        ax_b.axvline(0.0, color="#555555", linewidth=0.6, linestyle="-")

                    ax_b.set_xlim(x_lo, x_hi)
                    ax_b.set_ylim(cluster_n, 0)
                    ax_b.set_yticks([])
                    _style_box_spines(ax_b, visible=track.show_border)

                    if s_i == 0 and track.name:
                        ax_b.set_title(track.name, fontsize=9.0, fontweight="bold", pad=5)

                    if s_i == n_splits - 1:
                        ax_b.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2))
                        ax_b.tick_params(axis="x", labelsize=7.5, length=2.5, pad=2)
                        if track.xlabel:
                            ax_b.set_xlabel(track.xlabel, fontsize=7.5)
                    else:
                        ax_b.set_xticks([])

                    body_axes_for_track.append(ax_b)

            elif isinstance(track, PointTrack):
                vals = track._aligned_values(ordered_index)
                finite = vals[np.isfinite(vals)]
                if track.xlim is not None:
                    x_lo, x_hi = track.xlim
                elif finite.size > 0:
                    lo_v, hi_v = float(np.min(finite)), float(np.max(finite))
                    pad = max((hi_v - lo_v) * 0.06, 0.1)
                    x_lo, x_hi = lo_v - pad, hi_v + pad
                else:
                    x_lo, x_hi = 0.0, 1.0

                for s_i, (_grp_lbl, sl) in enumerate(split_slices):
                    ax_b = fig.add_subplot(body_gs[split_grid_rows[s_i], col_i])
                    sub_vals = vals[sl]
                    cluster_n = len(sub_vals)
                    y_pos = np.arange(cluster_n, dtype=np.float64) + 0.5
                    ok = np.isfinite(sub_vals)

                    ax_b.scatter(
                        sub_vals[ok],
                        y_pos[ok],
                        s=track.size,
                        color=track.color,
                        alpha=track.alpha,
                        linewidths=0,
                        rasterized=(cluster_n > 2000),
                    )
                    if track.smooth_line and cluster_n >= 5:
                        win = (
                            track.smooth_window
                            if track.smooth_window is not None
                            else max(5, cluster_n // 20)
                        )
                        sm = _running_mean(sub_vals, window=win)
                        ax_b.plot(
                            sm,
                            y_pos,
                            color=track.smooth_color,
                            linewidth=1.3,
                            rasterized=False,
                        )

                    if track.zero_line and x_lo < 0.0 < x_hi:
                        ax_b.axvline(0.0, color="#555555", linewidth=0.6, linestyle="--")

                    ax_b.set_xlim(x_lo, x_hi)
                    ax_b.set_ylim(cluster_n, 0)
                    ax_b.set_yticks([])
                    _style_box_spines(ax_b, visible=track.show_border)

                    if s_i == 0 and track.name:
                        ax_b.set_title(track.name, fontsize=9.0, fontweight="bold", pad=5)

                    if s_i == n_splits - 1:
                        ax_b.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2))
                        ax_b.tick_params(axis="x", labelsize=7.5, length=2.5, pad=2)
                        if track.xlabel:
                            ax_b.set_xlabel(track.xlabel, fontsize=7.5)
                    else:
                        ax_b.set_xticks([])

                    body_axes_for_track.append(ax_b)

            axes_info["body"][track_key] = body_axes_for_track

            # Right-side row counts on the rightmost track
            if show_row_counts and t_i == len(self.tracks) - 1:
                for s_i, (_grp_lbl, sl) in enumerate(split_slices):
                    ax_right = body_axes_for_track[s_i]
                    ax_right.text(
                        1.06,
                        0.5,
                        f"n = {sl.stop - sl.start:,}",
                        transform=ax_right.transAxes,
                        ha="left",
                        va="center",
                        fontsize=8,
                        color="#333333",
                    )

        # Synchronize top-profile y-limits across tracks sharing a scale_group
        if has_top_row and scale_groups:
            for _gname, member_idxs in scale_groups.items():
                group_top_axes = []
                for idx_i in member_idxs:
                    t_m = self.tracks[idx_i]
                    t_key = t_m.name or f"track_{idx_i}"
                    if (
                        isinstance(t_m, (EnrichedTrack, MatrixTrack))
                        and t_m.top_profile_ylim is None
                        and t_key in axes_info["top"]
                    ):
                        group_top_axes.append(axes_info["top"][t_key])
                if len(group_top_axes) > 1:
                    y_lo = min(ax.get_ylim()[0] for ax in group_top_axes)
                    y_hi = max(ax.get_ylim()[1] for ax in group_top_axes)
                    for ax in group_top_axes:
                        ax.set_ylim(y_lo, y_hi)

        # --------------------------------------------------------------
        # 3. Render per-track & shared scale_group colorbars (100% vector!)
        # --------------------------------------------------------------
        if cbar_gs is not None:
            rendered_groups: set[str] = set()
            for t_i, track in enumerate(self.tracks):
                if not isinstance(track, (EnrichedTrack, MatrixTrack)) or not track.has_colorbar:
                    continue

                if track.scale_group is not None:
                    if track.scale_group in rendered_groups:
                        continue
                    rendered_groups.add(track.scale_group)
                    member_idxs = scale_groups[track.scale_group]
                    # Span across contiguous columns if group members are adjacent
                    is_contiguous = member_idxs == list(range(member_idxs[0], member_idxs[-1] + 1))
                    if is_contiguous and len(member_idxs) > 1:
                        c_start = track_col_offset + member_idxs[0]
                        c_end = track_col_offset + member_idxs[-1] + 1
                        ax_cb = fig.add_subplot(cbar_gs[0, c_start:c_end])
                    else:
                        ax_cb = fig.add_subplot(cbar_gs[0, track_col_offset + t_i])
                    cb_label = track.colorbar_label or track.scale_group
                    cb_key = f"group:{track.scale_group}"
                else:
                    ax_cb = fig.add_subplot(cbar_gs[0, track_col_offset + t_i])
                    cb_label = track.colorbar_label or ""
                    cb_key = track.name or f"track_{t_i}"

                norm = norms[t_i]
                cm = _get_cmap(track.cmap, na_color=track.na_color)
                sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
                sm.set_array([])
                cb = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
                cb.outline.set_linewidth(0.5)
                cb.outline.set_edgecolor("#333333")

                if isinstance(norm, mcolors.TwoSlopeNorm):
                    ticks_cb = [norm.vmin, norm.vcenter, norm.vmax]
                    cb.set_ticks(ticks_cb)
                    cb.set_ticklabels(
                        ["0" if abs(v) < 1e-9 else f"{v:.2g}" for v in ticks_cb]
                    )
                else:
                    cb.locator = mpl.ticker.MaxNLocator(nbins=3)
                    cb.update_ticks()

                cb.ax.tick_params(labelsize=7.5, length=2.5, pad=2)
                if cb_label:
                    cb.set_label(cb_label, fontsize=8, labelpad=2)
                axes_info["colorbars"][cb_key] = ax_cb

        # --------------------------------------------------------------
        # 4. Categorical track legends (clean bottom strip, vector)
        # --------------------------------------------------------------
        if cat_tracks_with_legend:
            handles: list[mpatches.Patch] = []
            for ct in cat_tracks_with_legend:
                prefix = f"{ct.name}: " if (ct.name and len(cat_tracks_with_legend) > 1) else ""
                for cat in ct.categories:
                    handles.append(
                        mpatches.Patch(
                            facecolor=ct.color_map[cat],
                            edgecolor="#333333",
                            linewidth=0.4,
                            label=f"{prefix}{cat}",
                        )
                    )
            if handles:
                ncol = min(len(handles), 6)
                legend_title = (
                    cat_tracks_with_legend[0].name
                    if len(cat_tracks_with_legend) == 1 and cat_tracks_with_legend[0].name
                    else None
                )
                fig.legend(
                    handles=handles,
                    title=legend_title,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.01),
                    ncol=ncol,
                    frameon=False,
                    fontsize=8,
                    title_fontsize=8.5,
                    handlelength=1.1,
                    handleheight=0.9,
                    columnspacing=1.2,
                )

        if title:
            fig.suptitle(title, fontsize=11.5, fontweight="bold", y=0.97)

        if return_axes:
            info = {
                "axes": axes_info,
                "region_order": ordered_index,
                "splits": split_series,
                "split_slices": split_slices,
                "norms": norms,
            }
            return fig, info
        return fig

    def __repr__(self) -> str:
        names = [repr(t) for t in self.tracks]
        return f"HeatmapList([{', '.join(names)}])"
