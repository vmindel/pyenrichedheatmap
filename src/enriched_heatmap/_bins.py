"""Per-bin window tables and the pivot from long extraction output to matrices.

The data path is three steps, each a plain DataFrame you can inspect:

1. :func:`bins` turns regions into one row per bin (``chrom, start, end, region, bin``).
2. Something fills a value per bin: ``pymisha.gextract(expr, intervals=b, iterator=b)``,
   :func:`enriched_heatmap.read_bigwig`, or your own code.
3. :func:`matrix` / :func:`matrices` pivot that long table into region x bin matrices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from enriched_heatmap._matrix import NormalizedMatrix

_ATTR = "enriched_heatmap"
_RESERVED = {"chrom", "start", "end", "intervalID", "region", "bin", "chrom1", "start1", "end1"}
_ANCHORS = ("center", "start", "end")


class _Regions:
    """Region identifiers kept in ``DataFrame.attrs``.

    pandas compares and deep-copies ``attrs`` in operations such as ``merge``; a bare
    ``pandas.Index`` there makes the comparison ambiguous. This wrapper is immutable,
    compares by value to a plain bool and is shared rather than copied.
    """

    __slots__ = ("index",)

    def __init__(self, index: pd.Index) -> None:
        self.index = index

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Regions) and self.index.equals(other.index)

    __hash__ = None  # type: ignore[assignment]

    def __deepcopy__(self, memo: dict) -> _Regions:
        return self

    def __len__(self) -> int:
        return len(self.index)


def _is_minus(strand: pd.Series) -> np.ndarray:
    """``"-"`` (BED) and ``-1`` (misha) both mean minus strand."""
    s = strand.astype(str).str.strip()
    return s.isin(["-", "-1", "-1.0"]).to_numpy()


def _genome_sizes(genome: pd.DataFrame | Mapping[str, int] | None) -> dict[str, int] | None:
    if genome is None:
        return None
    if isinstance(genome, pd.DataFrame):
        chrom_col = "chrom" if "chrom" in genome.columns else genome.columns[0]
        return {str(c): int(e) for c, e in zip(genome[chrom_col], genome["end"], strict=True)}
    return {str(k): int(v) for k, v in genome.items()}


def _match_chroms(chroms: np.ndarray, sizes: dict[str, int]) -> np.ndarray:
    """Map region chromosome names onto genome names, adding or dropping ``chr``."""
    out = chroms.astype(object).copy()
    for name in pd.unique(chroms):
        if name in sizes:
            continue
        alt = name[3:] if name.startswith("chr") else f"chr{name}"
        if alt not in sizes:
            raise KeyError(
                f"Chromosome {name!r} is not in `genome` (examples: {list(sizes)[:5]}). "
                "Drop those regions or pass a matching genome table."
            )
        out[chroms == name] = alt
    return out


def bins(
    regions: pd.DataFrame,
    upstream: int = 1000,
    downstream: int = 1000,
    bin_size: int = 20,
    anchor: str = "center",
    body_bins: int | None = None,
    genome: pd.DataFrame | Mapping[str, int] | None = None,
    id: str | None = None,
    cols: Sequence[str] = ("chrom", "start", "end"),
    strand: str | None = "strand",
    snap: int | None = None,
) -> pd.DataFrame:
    """One row per heatmap bin, ready to pass to ``pymisha.gextract``.

    Bins are laid out 5' to 3': on minus-strand regions ``bin`` 0 is the genomic
    right end, so matrices built from this table are already strand-oriented.

    Parameters
    ----------
    regions : pandas.DataFrame
        Regions with ``chrom``, ``start``, ``end`` (0-based, half-open) and an optional
        strand column holding ``"+"``/``"-"`` or ``1``/``-1``.
    upstream, downstream : int
        Flank lengths in bp. Both must be multiples of ``bin_size``.
    bin_size : int
        Flank bin width in bp.
    anchor : {"center", "start", "end"}
        Reference point when ``body_bins`` is None. ``"start"`` is the 5' end, so it is
        the TSS for a table of genes on either strand.
    body_bins : int, optional
        Switch to scale-regions mode: each region body is split into this many bins of
        equal fractional width, between the two flanks. Bodies shorter than
        ``body_bins`` bp are sampled at 1 bp.
    genome : DataFrame or dict, optional
        Chromosome sizes, e.g. ``pymisha.gintervals_all()`` or ``{"chr1": 248956422}``.
        Bins past a chromosome end are dropped (pymisha refuses them) and come out as
        NaN in the matrix. Without it only bins before position 0 are dropped.
    id : str, optional
        Column holding region identifiers. Defaults to the DataFrame index. Must be unique.
    cols : tuple of str
        Names of the chrom, start and end columns.
    strand : str or None
        Strand column; ``None`` (or a missing column) treats every region as plus.
    snap : int, optional
        Stored bin size of a fixed-bin track (reference-point mode). Heatmap bins are
        laid on that grid so each covers whole stored bins (misha does not weight
        partially covered stored bins by overlap), and the layout becomes *centred*:
        an extra middle bin holds the stored bin that contains the anchor base, on
        both strands. The matrix then has ``upstream/bin_size + 1 +
        downstream/bin_size`` columns and x = 0 is the middle of the anchor bin.
        ``bin_size`` must be an odd multiple of ``snap`` (e.g. ``snap=20`` with
        ``bin_size`` 20 or 60) so the middle bin is centred on the grid.

    Returns
    -------
    pandas.DataFrame
        ``chrom, start, end, region, bin``. The layout (flanks, bin size, region order)
        is kept in ``.attrs`` for :func:`matrix`; filtering rows keeps it.
    """
    c_chrom, c_start, c_end = cols
    for c in cols:
        if c not in regions.columns:
            raise KeyError(f"`regions` has no {c!r} column; pass cols=(chrom, start, end).")
    if bin_size <= 0:
        raise ValueError("`bin_size` must be positive.")
    if upstream < 0 or downstream < 0:
        raise ValueError("`upstream` and `downstream` must be non-negative.")
    if upstream % bin_size or downstream % bin_size:
        raise ValueError(
            f"`upstream` ({upstream}) and `downstream` ({downstream}) must be multiples of "
            f"`bin_size` ({bin_size})."
        )
    if body_bins is None and anchor not in _ANCHORS:
        raise ValueError(f"`anchor` must be one of {_ANCHORS}, got {anchor!r}.")
    if body_bins is not None and body_bins <= 0:
        raise ValueError("`body_bins` must be positive.")
    if snap:
        snap = int(snap)
        if snap < 0:
            raise ValueError("`snap` must be positive.")
        if body_bins is not None:
            raise ValueError("`snap` only applies to reference-point bins (body_bins=None).")
        if bin_size % snap or (bin_size // snap) % 2 == 0:
            raise ValueError(
                f"`bin_size` ({bin_size}) must be an odd multiple of `snap` ({snap}) so the "
                f"anchor bin is centred on the stored grid, e.g. bin_size={snap} or {3 * snap}.")

    region_ids = pd.Index(regions[id] if id is not None else regions.index)
    if not region_ids.is_unique:
        raise ValueError(
            "Region identifiers are not unique. Reset the index or pass id=<unique column>."
        )

    chrom = regions[c_chrom].astype(str).to_numpy()
    for c in (c_start, c_end):
        if regions[c].isna().any():
            raise ValueError(
                f"`regions` has {int(regions[c].isna().sum())} missing values in {c!r}; "
                "drop those rows first."
            )
    start = regions[c_start].to_numpy(dtype=np.int64)
    end = regions[c_end].to_numpy(dtype=np.int64)
    if np.any(end < start):
        raise ValueError("Found regions with end < start.")
    use_strand = strand is not None and strand in regions.columns
    minus = _is_minus(regions[strand]) if use_strand else np.zeros(len(regions), dtype=bool)
    sizes = _genome_sizes(genome)
    if sizes is not None:
        chrom = _match_chroms(chrom, sizes)

    n_up, n_down = upstream // bin_size, downstream // bin_size
    n_body = body_bins or 0
    centered = bool(snap) and body_bins is None
    n_bins = n_up + n_body + n_down + int(centered)
    n = len(regions)

    # Build every bin in genomic left-to-right order, then renumber minus rows 5'->3'.
    n_left = np.where(minus, n_down, n_up)
    k = np.arange(max(n_up, n_down))
    if body_bins is None:
        if centered:
            # The anchor *base*, not a boundary: the 5' base of a minus region is end - 1.
            if anchor == "center":
                p = (start + end) // 2
            elif anchor == "start":
                p = np.where(minus, end - 1, start)
            else:
                p = np.where(minus, start, end - 1)
            cell = p - p % snap  # stored bin holding the anchor base
            a = cell - (bin_size - snap) // 2  # genomic start of the centred middle bin
        elif anchor == "center":
            a = (start + end) // 2
        elif anchor == "start":
            a = np.where(minus, end, start)
        else:
            a = np.where(minus, start, end)
        left_edge = a - n_left * bin_size
        g = np.arange(n_bins)
        b_start = left_edge[:, None] + g[None, :] * bin_size
        b_end = b_start + bin_size
        keep = np.ones_like(b_start, dtype=bool)
    else:
        body_len = (end - start)[:, None]
        kb = np.arange(n_body)[None, :]
        wide = body_len >= n_body
        body_s = np.where(wide, start[:, None] + kb * body_len // n_body,
                          start[:, None] + ((2 * kb + 1) * body_len) // (2 * n_body))
        body_e = np.where(wide, start[:, None] + (kb + 1) * body_len // n_body, body_s + 1)
        left_s = (start - n_left * bin_size)[:, None] + k[None, :] * bin_size
        right_s = end[:, None] + k[None, :] * bin_size
        n_right = n_bins - n_body - n_left
        b_start = np.concatenate([left_s, body_s, right_s], axis=1)
        b_end = np.concatenate([left_s + bin_size, body_e, right_s + bin_size], axis=1)
        # The flank blocks are padded to the longer flank; mask each row's extra columns.
        keep = np.concatenate(
            [k[None, :] < n_left[:, None], np.ones((n, n_body), dtype=bool),
             k[None, :] < n_right[:, None]], axis=1)

    g_idx = np.cumsum(keep, axis=1) - 1
    o_idx = np.where(minus[:, None], n_bins - 1 - g_idx, g_idx)
    if body_bins is not None:
        empty_body = (end - start) == 0
        if np.any(empty_body):
            body_cols = np.zeros_like(keep)
            body_cols[:, k.size : k.size + n_body] = True
            keep &= ~(body_cols & empty_body[:, None])

    keep &= b_start >= 0
    if sizes is not None:
        clen = np.array([sizes[c] for c in chrom], dtype=np.int64)
        keep &= b_end <= clen[:, None]

    rows, cols_ = np.nonzero(keep)
    out = pd.DataFrame(
        {
            "chrom": chrom[rows],
            "start": b_start[rows, cols_],
            "end": b_end[rows, cols_],
            "region": region_ids[rows],
            "bin": o_idx[rows, cols_].astype(np.int64),
        }
    )
    out.attrs[_ATTR] = {
        "regions": _Regions(region_ids),
        "strand_oriented": bool(use_strand),
        "upstream": int(upstream),
        "downstream": int(downstream),
        "bin_size": int(bin_size),
        "body_bins": int(n_body),
        "anchor": anchor if body_bins is None else None,
        "snap": int(snap) if snap else None,
        "centered": centered,
        "n_bins": int(n_bins),
    }
    return out


def _layout(bins_df: pd.DataFrame) -> dict[str, Any]:
    layout = bins_df.attrs.get(_ATTR)
    if layout is None:
        raise ValueError(
            "`bins` has lost its layout (it lives in DataFrame.attrs, which some pandas "
            "operations such as merge or concat drop). Pass the table returned by "
            "eh.bins() unchanged, or filter it with boolean indexing."
        )
    return layout


def _value_columns(values: pd.DataFrame) -> list[str]:
    return [c for c in values.columns if c not in _RESERVED]


def matrices(
    values: pd.DataFrame,
    bins: pd.DataFrame,
    columns: Sequence[str] | None = None,
    dtype: Any = np.float32,
) -> dict[str, NormalizedMatrix]:
    """Pivot a long per-bin value table into one matrix per value column.

    Parameters
    ----------
    values : pandas.DataFrame
        Long table with one row per bin, e.g.
        ``pymisha.gextract(exprs, intervals=bins, iterator=bins, colnames=[...])`` or
        :func:`read_bigwig`. Rows are placed by, in order of preference: ``region`` and
        ``bin`` columns (present with ``gextract(..., intervals_join="intervals")``),
        ``intervalID`` (the 1-based row of ``bins``, gextract's default), or else row
        position, which is accepted only when ``start``/``end`` match ``bins`` row for
        row.
    bins : pandas.DataFrame
        The table from :func:`bins` that was passed to the extraction.
    columns : list of str, optional
        Value columns to pivot. Default: every column that is not a coordinate,
        ``intervalID``, ``region`` or ``bin``.
    dtype : numpy dtype
        Storage type of the matrices. ``float32`` (default) halves memory and is exact
        for integers up to 2**24 (about 16.7 million); pass ``np.float64`` for raw
        counts beyond that.

    Returns
    -------
    dict of str to NormalizedMatrix
        One ``(n_regions, n_bins)`` matrix per column, rows in the order of the regions
        given to :func:`bins`. Bins with no row (off the chromosome) are NaN.
    """
    layout = _layout(bins)
    region_ids: pd.Index = layout["regions"].index
    if {"region", "bin"} <= set(values.columns):
        r = region_ids.get_indexer(values["region"].to_numpy())
        b = values["bin"].to_numpy(dtype=np.int64)
        if np.any(r < 0):
            raise ValueError("`values` has `region` labels that are not in `bins`.")
    else:
        if "intervalID" in values.columns:
            pos = values["intervalID"].to_numpy(dtype=np.int64) - 1
            if pos.size and (pos.min() < 0 or pos.max() >= len(bins)):
                raise ValueError(
                    "`intervalID` values do not fit `bins`. Pass the same bins table as "
                    "both `intervals` and `iterator` to gextract, and the same table here."
                )
        else:
            same = (
                len(values) == len(bins)
                and {"start", "end"} <= set(values.columns)
                and np.array_equal(values["start"].to_numpy(), bins["start"].to_numpy())
                and np.array_equal(values["end"].to_numpy(), bins["end"].to_numpy())
            )
            if not same:
                raise ValueError(
                    "`values` has no `intervalID` (or `region`/`bin`) and its rows do not "
                    "match `bins` row for row. gextract sorts rows genomically, so keep its "
                    "default intervals_join, or use intervals_join='intervals'."
                )
            pos = np.arange(len(bins))
        r = region_ids.get_indexer(bins["region"].to_numpy()[pos])
        b = bins["bin"].to_numpy(dtype=np.int64)[pos]

    cols = list(columns) if columns is not None else _value_columns(values)
    if not cols:
        raise ValueError(f"No value columns found in `values` ({list(values.columns)}).")
    mode = "scale_regions" if layout["body_bins"] else "reference_point"
    out: dict[str, NormalizedMatrix] = {}
    for c in cols:
        arr = np.full((len(region_ids), layout["n_bins"]), np.nan, dtype=dtype)
        arr[r, b] = values[c].to_numpy(dtype=dtype)
        out[c] = NormalizedMatrix(
            values=arr,
            region_index=region_ids,
            mode=mode,
            upstream=layout["upstream"],
            downstream=layout["downstream"],
            bin_size=layout["bin_size"],
            target_bins=layout["body_bins"],
            name=str(c),
            strand_oriented=layout["strand_oriented"],
            metadata={"anchor": layout["anchor"]},
            centered=bool(layout.get("centered", False)),
        )
    return out


def matrix(
    values: pd.DataFrame,
    bins: pd.DataFrame,
    value: str | None = None,
    dtype: Any = np.float32,
) -> NormalizedMatrix:
    """Pivot one value column of a long per-bin table into a matrix.

    Same as :func:`matrices` for a single column. ``value`` may be omitted when the
    table has exactly one value column.
    """
    if value is None:
        cols = _value_columns(values)
        if len(cols) != 1:
            raise ValueError(f"`values` has value columns {cols}; pass value=<column>.")
        value = cols[0]
    if value not in values.columns:
        raise KeyError(f"`values` has no column {value!r} (columns: {list(values.columns)}).")
    return matrices(values, bins, columns=[value], dtype=dtype)[value]
