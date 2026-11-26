"""Core NormalizedMatrix data model and signal transformations."""

from __future__ import annotations

import datetime
import numbers
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
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


_NAProfile = tuple[str, set[str], bool]
_INDEX_LOOKUP_ERRORS = (
    KeyError,
    TypeError,
    ValueError,
    ArithmeticError,
    AssertionError,
    pd.errors.InvalidIndexError,
)


def _scalar_na_kind(val: object) -> str | None:
    if val is pd.NaT:
        return "nat"
    if isinstance(val, np.datetime64) and bool(np.isnat(val)):
        return "dt_nat"
    if isinstance(val, np.timedelta64) and bool(np.isnat(val)):
        return "td_nat"
    if (
        val is None
        or val is pd.NA
        or (isinstance(val, (float, np.floating)) and bool(np.isnan(val)))
    ):
        return "na"
    return None


def _is_scalar_na(val: object) -> bool:
    return _scalar_na_kind(val) is not None


def _is_temporal_na_kind(kind: str | None) -> bool:
    return kind in ("nat", "dt_nat", "td_nat")


def _ensure_hashable(val: object) -> None:
    try:
        hash(val)
    except TypeError:
        raise TypeError(
            f"Cannot index NormalizedMatrix.loc with unhashable type {type(val).__name__}."
        ) from None


def _leaf_dtype(dt: Any) -> Any:
    while isinstance(dt, (pd.CategoricalDtype, pd.SparseDtype)):
        dt = dt.categories.dtype if isinstance(dt, pd.CategoricalDtype) else dt.subtype
    return dt


def _dtype_domain(dt: Any) -> str:
    leaf = _leaf_dtype(dt)
    if pd.api.types.is_object_dtype(leaf):
        return "object"
    if pd.api.types.is_bool_dtype(leaf):
        return "bool"
    if pd.api.types.is_numeric_dtype(leaf):
        return "numeric"
    if pd.api.types.is_string_dtype(leaf):
        return "string"
    if isinstance(leaf, pd.DatetimeTZDtype) or getattr(leaf, "tz", None) is not None:
        return "datetime_tz"
    if pd.api.types.is_datetime64_any_dtype(leaf):
        return "datetime_naive"
    if pd.api.types.is_timedelta64_dtype(leaf):
        return "timedelta"
    if isinstance(leaf, pd.PeriodDtype):
        return f"period:{leaf}"
    if isinstance(leaf, pd.IntervalDtype):
        return f"interval:{leaf.closed}:{_dtype_domain(leaf.subtype)}"
    return f"other:{leaf}"


def _scalar_domain(val: Any) -> str:
    if isinstance(val, (bool, np.bool_)):
        return "bool"
    if isinstance(val, (numbers.Number, Decimal)):
        return "numeric"
    if isinstance(val, str):
        return "string"
    if isinstance(val, (pd.Timestamp, datetime.datetime)):
        return "datetime_tz" if val.tzinfo is not None else "datetime_naive"
    if isinstance(val, (np.datetime64, datetime.date)):
        return "datetime_naive"
    if isinstance(val, (pd.Timedelta, datetime.timedelta, np.timedelta64)):
        return "timedelta"
    if isinstance(val, pd.Period):
        return f"period:{pd.PeriodDtype(val.freqstr)}"
    if isinstance(val, pd.Interval):
        return f"interval:{val.closed}:{_scalar_domain(val.left)}"
    if isinstance(val, bytes):
        return "bytes"
    if isinstance(val, tuple):
        return "tuple"
    return "custom"


def _is_temporal_domain(dom: str) -> bool:
    return dom in ("datetime_naive", "datetime_tz", "timedelta") or dom.startswith("period:")


def _na_kind_matches_typed_domain(na_kind: str, typed_dom: str) -> bool:
    if na_kind == "nat":
        return _is_temporal_domain(typed_dom)
    if na_kind == "dt_nat":
        return typed_dom == "datetime_naive"
    if na_kind == "td_nat":
        return typed_dom == "timedelta"
    return not _is_temporal_domain(typed_dom)


def _na_kinds_compatible(ka: str, kb: str) -> bool:
    if ka == "na" or kb == "na":
        return ka == kb
    if ka == "nat" or kb == "nat":
        return True
    return ka == kb


def _na_profiles_compatible(pa: _NAProfile, pb: _NAProfile) -> bool:
    kind_a, doms_a, typed_a = pa
    kind_b, doms_b, typed_b = pb
    if typed_a and typed_b:
        return doms_a == doms_b
    if typed_b and not typed_a:
        tdom_b = next(iter(doms_b))
        return _na_kind_matches_typed_domain(kind_a, tdom_b) and (
            not doms_a or tdom_b in doms_a
        )
    if typed_a and not typed_b:
        tdom_a = next(iter(doms_a))
        return _na_kind_matches_typed_domain(kind_b, tdom_a) and (
            not doms_b or tdom_a in doms_b
        )
    if not _na_kinds_compatible(kind_a, kind_b):
        return False
    if kind_a == "dt_nat" and doms_b and "datetime_naive" not in doms_b:
        return False
    if kind_b == "dt_nat" and doms_a and "datetime_naive" not in doms_a:
        return False
    if kind_a == "td_nat" and doms_b and "timedelta" not in doms_b:
        return False
    if kind_b == "td_nat" and doms_a and "timedelta" not in doms_a:
        return False
    return not (doms_a and doms_b and not (doms_a & doms_b))


def _underlying_index_dtype(idx: pd.Index) -> Any:
    return idx.categories.dtype if isinstance(idx, pd.CategoricalIndex) else idx.dtype


def _is_nonbool_numeric_dtype(dt: Any) -> bool:
    leaf = _leaf_dtype(dt)
    return not pd.api.types.is_bool_dtype(leaf) and (
        pd.api.types.is_integer_dtype(leaf) or pd.api.types.is_float_dtype(leaf)
    )


def _index_isna(idx: pd.Index) -> np.ndarray:
    if isinstance(idx, pd.MultiIndex):
        if len(idx) == 0 or idx.nlevels == 0:
            return np.zeros(len(idx), dtype=bool)
        level_na: list[np.ndarray] = []
        for lvl in range(idx.nlevels):
            codes = np.asarray(idx.codes[lvl])
            lvl_idx = idx.levels[lvl]
            if len(lvl_idx) > 0 and pd.api.types.is_object_dtype(lvl_idx.dtype):
                lvl_mask = np.array([_is_scalar_na(x) for x in lvl_idx], dtype=bool)
                if np.any(lvl_mask):
                    safe_codes = np.maximum(codes, 0)
                    level_na.append((codes < 0) | lvl_mask[safe_codes])
                    continue
            level_na.append(codes < 0)
        return np.asarray(np.all(level_na, axis=0), dtype=bool)
    return np.asarray(idx.isna(), dtype=bool)


def _object_index_info(idx: pd.Index, na_pos: int) -> tuple[str, set[str]]:
    if isinstance(idx, pd.CategoricalIndex):
        doms = {_scalar_domain(v) for v in idx.categories if not _is_scalar_na(v)}
        na_kind = "nat" if (doms and all(_is_temporal_domain(d) for d in doms)) else "na"
        return na_kind, doms
    raw_na = idx[na_pos]
    na_kind = _scalar_na_kind(raw_na) or "na"
    doms = {_scalar_domain(x) for x in idx if not _is_scalar_na(x)}
    return na_kind, doms


def _index_na_profile(idx: pd.Index, na_pos: int) -> _NAProfile:
    dom = _dtype_domain(idx.dtype)
    if dom != "object":
        return ("nat" if _is_temporal_domain(dom) else "na", {dom}, True)
    na_kind, doms = _object_index_info(idx, na_pos)
    return (na_kind, doms, False)


def _object_na_matches_domain(obj_idx: pd.Index, obj_pos: int, typed_dom: str) -> bool:
    na_kind, doms = _object_index_info(obj_idx, obj_pos)
    return _na_profiles_compatible(
        (na_kind, doms, False),
        ("nat" if _is_temporal_domain(typed_dom) else "na", {typed_dom}, True),
    )


def _are_na_compatible(idx_a: pd.Index, pos_a: int, idx_b: pd.Index, pos_b: int) -> bool:
    return _na_profiles_compatible(
        _index_na_profile(idx_a, pos_a),
        _index_na_profile(idx_b, pos_b),
    )


def _has_coexisting_same_group_na(idx: pd.Index) -> bool:
    na_positions = np.flatnonzero(_index_isna(idx))
    if na_positions.size <= 1:
        return False
    na_vals = [idx[int(p)] for p in na_positions]
    for allowed_kinds in (("na",), ("nat", "dt_nat"), ("nat", "td_nat")):
        grp = [v for v in na_vals if _scalar_na_kind(v) in allowed_kinds]
        if len(grp) > 1 and len(pd.Index(grp, dtype=object).unique()) > 1:
            return True
    return False


def _mi_level_na_profile(lvl_idx: pd.Index, code: int) -> _NAProfile:
    dom = _dtype_domain(lvl_idx.dtype)
    if dom != "object":
        return ("nat" if _is_temporal_domain(dom) else "na", {dom}, True)
    doms = {_scalar_domain(x) for x in lvl_idx if not _is_scalar_na(x)}
    if code >= 0 and _is_scalar_na(lvl_idx[code]):
        na_kind = _scalar_na_kind(lvl_idx[code]) or "na"
    else:
        na_kind = "nat" if (doms and all(_is_temporal_domain(d) for d in doms)) else "na"
    return (na_kind, doms, False)


def _mi_rows_compatible(
    source: pd.MultiIndex, s_row: int, target: pd.MultiIndex, t_row: int
) -> bool:
    for lvl in range(source.nlevels):
        s_lvl = source.levels[lvl]
        t_lvl = target.levels[lvl]
        s_code = int(source.codes[lvl][s_row])
        t_code = int(target.codes[lvl][t_row])
        s_na = s_code < 0 or _is_scalar_na(s_lvl[s_code])
        t_na = t_code < 0 or _is_scalar_na(t_lvl[t_code])
        if s_na != t_na:
            return False
        if not s_na:
            sv = s_lvl[s_code]
            tv = t_lvl[t_code]
            if (
                isinstance(tv, (bool, np.bool_)) != isinstance(sv, (bool, np.bool_))
                or isinstance(tv, pd.Interval) != isinstance(sv, pd.Interval)
            ):
                return False
        elif not _na_profiles_compatible(
            _mi_level_na_profile(s_lvl, s_code),
            _mi_level_na_profile(t_lvl, t_code),
        ):
            return False
    return True


def _tuple_level_domains(tuples: Sequence[tuple[Any, ...]], nlevels: int) -> list[set[str]]:
    return [
        {_scalar_domain(t[lvl]) for t in tuples if not _is_scalar_na(t[lvl])}
        for lvl in range(nlevels)
    ]


def _tuple_elem_can_match_mi_level(
    val: Any, tup_lvl_doms: set[str], mi_lvl: pd.Index
) -> bool:
    lvl_dom = _dtype_domain(mi_lvl.dtype)
    if _is_scalar_na(val):
        tup_prof: _NAProfile = (_scalar_na_kind(val) or "na", tup_lvl_doms, False)
        if lvl_dom != "object":
            mi_prof: _NAProfile = (
                "nat" if _is_temporal_domain(lvl_dom) else "na",
                {lvl_dom},
                True,
            )
            return _na_profiles_compatible(tup_prof, mi_prof)
        mi_doms = {_scalar_domain(x) for x in mi_lvl if not _is_scalar_na(x)}
        mi_kinds = {
            _scalar_na_kind(x) or "na" for x in mi_lvl if _is_scalar_na(x)
        }
        default_kind = (
            "nat" if (mi_doms and all(_is_temporal_domain(d) for d in mi_doms)) else "na"
        )
        mi_kinds.add(default_kind)
        return any(
            _na_profiles_compatible(tup_prof, (mk, mi_doms, False)) for mk in mi_kinds
        )
    if lvl_dom != "object":
        return _scalar_domain(val) == lvl_dom
    return True


def _mi_row_matches_tuple(
    mi: pd.MultiIndex,
    mi_row: int,
    tup: tuple[Any, ...],
    tup_doms_per_lvl: Sequence[set[str]],
) -> bool:
    for lvl in range(mi.nlevels):
        tv = tup[lvl]
        lvl_idx = mi.levels[lvl]
        code = int(mi.codes[lvl][mi_row])
        mi_na = code < 0 or _is_scalar_na(lvl_idx[code])
        tup_na = _is_scalar_na(tv)
        if mi_na != tup_na:
            return False
        if not mi_na:
            mv = lvl_idx[code]
            if (
                isinstance(tv, (bool, np.bool_)) != isinstance(mv, (bool, np.bool_))
                or isinstance(tv, pd.Interval) != isinstance(mv, pd.Interval)
            ):
                return False
        else:
            tup_prof: _NAProfile = (
                _scalar_na_kind(tv) or "na",
                tup_doms_per_lvl[lvl],
                False,
            )
            if not _na_profiles_compatible(_mi_level_na_profile(lvl_idx, code), tup_prof):
                return False
    return True


def _safe_get_indexer(source: pd.Index, target: pd.Index) -> np.ndarray:
    n_target = len(target)
    if n_target == 0 or len(source) == 0:
        return np.full(n_target, -1, dtype=np.intp)

    if isinstance(source, pd.MultiIndex) and isinstance(target, pd.MultiIndex):
        if source.nlevels != target.nlevels:
            return np.full(n_target, -1, dtype=np.intp)
        for lvl in range(source.nlevels):
            sl_dom = _dtype_domain(source.levels[lvl].dtype)
            tl_dom = _dtype_domain(target.levels[lvl].dtype)
            if sl_dom != "object" and tl_dom != "object" and sl_dom != tl_dom:
                return np.full(n_target, -1, dtype=np.intp)
            if (
                sl_dom.startswith("interval:")
                and tl_dom == "object"
                and not any(
                    isinstance(x, pd.Interval) and _scalar_domain(x) == sl_dom
                    for x in target.levels[lvl]
                )
            ):
                return np.full(n_target, -1, dtype=np.intp)
            if (
                tl_dom.startswith("interval:")
                and sl_dom == "object"
                and not any(
                    isinstance(x, pd.Interval) and _scalar_domain(x) == tl_dom
                    for x in source.levels[lvl]
                )
            ):
                return np.full(n_target, -1, dtype=np.intp)
            sl_dt = _underlying_index_dtype(source.levels[lvl])
            tl_dt = _underlying_index_dtype(target.levels[lvl])
            if (pd.api.types.is_bool_dtype(sl_dt) and _is_nonbool_numeric_dtype(tl_dt)) or (
                pd.api.types.is_bool_dtype(tl_dt) and _is_nonbool_numeric_dtype(sl_dt)
            ):
                return np.full(n_target, -1, dtype=np.intp)
        try:
            with np.errstate(invalid="ignore"):
                indexer = np.asarray(source.get_indexer(target), dtype=np.intp)
        except _INDEX_LOOKUP_ERRORS:
            return np.full(n_target, -1, dtype=np.intp)
        if np.any(indexer >= 0) and any(
            _dtype_domain(source.levels[lvl].dtype) == "object"
            or _dtype_domain(target.levels[lvl].dtype) == "object"
            for lvl in range(source.nlevels)
        ):
            for i in np.flatnonzero(indexer >= 0):
                if not _mi_rows_compatible(source, int(indexer[i]), target, int(i)):
                    indexer[i] = -1
        return indexer

    if isinstance(source, pd.MultiIndex):
        t_dt = _underlying_index_dtype(target)
        if not pd.api.types.is_object_dtype(t_dt):
            return np.full(n_target, -1, dtype=np.intp)
        k = source.nlevels
        raw_pos = [i for i, x in enumerate(target) if isinstance(x, tuple) and len(x) == k]
        if not raw_pos:
            return np.full(n_target, -1, dtype=np.intp)
        all_tgt_tuples: list[tuple[Any, ...]] = [target[i] for i in raw_pos]
        tup_doms = _tuple_level_domains(all_tgt_tuples, k)
        valid_pos = [
            i
            for i in raw_pos
            if all(
                _tuple_elem_can_match_mi_level(target[i][lvl], tup_doms[lvl], source.levels[lvl])
                for lvl in range(k)
            )
        ]
        if not valid_pos:
            return np.full(n_target, -1, dtype=np.intp)
        try:
            sub_tgt = pd.MultiIndex.from_arrays(
                [
                    pd.Index([target[i][lvl] for i in valid_pos], dtype=object, tupleize_cols=False)
                    for lvl in range(k)
                ]
            )
        except _INDEX_LOOKUP_ERRORS:
            return np.full(n_target, -1, dtype=np.intp)
        sub_hits = _safe_get_indexer(source, sub_tgt)
        res = np.full(n_target, -1, dtype=np.intp)
        for idx_in_valid, orig_i in enumerate(valid_pos):
            s_hit = int(sub_hits[idx_in_valid])
            if s_hit >= 0 and _mi_row_matches_tuple(
                source, s_hit, target[orig_i], tup_doms
            ):
                res[orig_i] = s_hit
        return res

    if isinstance(target, pd.MultiIndex):
        s_dt = _underlying_index_dtype(source)
        if not pd.api.types.is_object_dtype(s_dt):
            return np.full(n_target, -1, dtype=np.intp)
        k = target.nlevels
        raw_src = [j for j, x in enumerate(source) if isinstance(x, tuple) and len(x) == k]
        if not raw_src:
            return np.full(n_target, -1, dtype=np.intp)
        all_src_tuples: list[tuple[Any, ...]] = [source[j] for j in raw_src]
        src_doms = _tuple_level_domains(all_src_tuples, k)
        valid_src = [
            j
            for j in raw_src
            if all(
                _tuple_elem_can_match_mi_level(source[j][lvl], src_doms[lvl], target.levels[lvl])
                for lvl in range(k)
            )
        ]
        if not valid_src:
            return np.full(n_target, -1, dtype=np.intp)
        try:
            sub_src = pd.MultiIndex.from_arrays(
                [
                    pd.Index([source[j][lvl] for j in valid_src], dtype=object, tupleize_cols=False)
                    for lvl in range(k)
                ]
            )
        except _INDEX_LOOKUP_ERRORS:
            return np.full(n_target, -1, dtype=np.intp)
        if not sub_src.is_unique:
            keep_mask = ~np.asarray(sub_src.duplicated(keep="first"), dtype=bool)
            valid_src = [valid_src[m_i] for m_i in np.flatnonzero(keep_mask)]
            sub_src = sub_src[keep_mask]
        sub_idx = _safe_get_indexer(sub_src, target)
        res = np.full(n_target, -1, dtype=np.intp)
        for t_i in np.flatnonzero(sub_idx >= 0):
            orig_j = valid_src[int(sub_idx[t_i])]
            if _mi_row_matches_tuple(target, int(t_i), source[orig_j], src_doms):
                res[int(t_i)] = orig_j
        return res

    if isinstance(target, pd.CategoricalIndex):
        codes = np.asarray(target.codes, dtype=np.intp)
        out = np.full(n_target, -1, dtype=np.intp)
        valid = codes >= 0
        if np.any(valid) and len(target.categories) > 0:
            cat_indexer = _safe_get_indexer(source, target.categories)
            out[valid] = cat_indexer[codes[valid]]
        missing = codes < 0
        if np.any(missing):
            src_na_positions = np.flatnonzero(_index_isna(source))
            if src_na_positions.size > 0:
                first_miss = int(np.flatnonzero(missing)[0])
                for si in src_na_positions:
                    if _are_na_compatible(source, int(si), target, first_miss):
                        out[missing] = int(si)
                        break
        return out

    if isinstance(source, pd.CategoricalIndex):
        src_codes = np.asarray(source.codes, dtype=np.intp)
        out = np.full(n_target, -1, dtype=np.intp)
        tgt_na = _index_isna(target)
        tgt_valid = ~tgt_na
        if np.any(tgt_valid) and len(source.categories) > 0:
            cat_to_src = np.full(len(source.categories), -1, dtype=np.intp)
            src_valid_pos = np.flatnonzero(src_codes >= 0)
            if src_valid_pos.size > 0:
                cat_to_src[src_codes[src_valid_pos]] = src_valid_pos
            sub_target = target if not np.any(tgt_na) else target[tgt_valid]
            cat_hits = _safe_get_indexer(source.categories, sub_target)
            mapped = np.where(cat_hits >= 0, cat_to_src[np.maximum(cat_hits, 0)], -1)
            out[tgt_valid] = mapped
        if np.any(tgt_na):
            src_na_positions = np.flatnonzero(src_codes < 0)
            if src_na_positions.size > 0:
                for ti in np.flatnonzero(tgt_na):
                    for si in src_na_positions:
                        if _are_na_compatible(source, int(si), target, int(ti)):
                            out[int(ti)] = int(si)
                            break
        return out

    s_dom = _dtype_domain(source.dtype)
    t_dom = _dtype_domain(target.dtype)
    if s_dom != "object" and t_dom != "object" and s_dom != t_dom:
        return np.full(n_target, -1, dtype=np.intp)

    tgt_na = _index_isna(target)
    if np.any(tgt_na):
        out = np.full(n_target, -1, dtype=np.intp)
        tgt_valid = ~tgt_na
        if np.any(tgt_valid):
            out[tgt_valid] = _safe_get_indexer(source, target[tgt_valid])
        src_na_positions = np.flatnonzero(_index_isna(source))
        if src_na_positions.size > 0:
            if (
                s_dom == "object"
                and t_dom == "object"
                and (
                    _has_coexisting_same_group_na(source)
                    or _has_coexisting_same_group_na(target)
                )
            ):
                try:
                    na_hits = np.asarray(source.get_indexer(target[tgt_na]), dtype=np.intp)
                except _INDEX_LOOKUP_ERRORS:
                    na_hits = np.full(int(tgt_na.sum()), -1, dtype=np.intp)
                for k_idx, ti in enumerate(np.flatnonzero(tgt_na)):
                    hit_pos = int(na_hits[k_idx])
                    if hit_pos >= 0 and _are_na_compatible(source, hit_pos, target, int(ti)):
                        out[int(ti)] = hit_pos
            else:
                for ti in np.flatnonzero(tgt_na):
                    for si in src_na_positions:
                        if _are_na_compatible(source, int(si), target, int(ti)):
                            out[int(ti)] = int(si)
                            break
        return out

    if s_dom.startswith("interval:") and t_dom == "object":
        iv_pos = [
            i
            for i, x in enumerate(target)
            if isinstance(x, pd.Interval) and _scalar_domain(x) == s_dom
        ]
        if not iv_pos:
            return np.full(n_target, -1, dtype=np.intp)
        out_iv = np.full(n_target, -1, dtype=np.intp)
        try:
            sub_iv_tgt = pd.IntervalIndex([target[i] for i in iv_pos])
            with np.errstate(invalid="ignore"):
                hits_iv = np.asarray(source.get_indexer(sub_iv_tgt), dtype=np.intp)
        except _INDEX_LOOKUP_ERRORS:
            return out_iv
        out_iv[np.asarray(iv_pos, dtype=np.intp)] = hits_iv
        return out_iv
    if (
        t_dom.startswith("interval:")
        and s_dom == "object"
        and not any(isinstance(x, pd.Interval) and _scalar_domain(x) == t_dom for x in source)
    ):
        return np.full(n_target, -1, dtype=np.intp)

    s_dt = _underlying_index_dtype(source)
    t_dt = _underlying_index_dtype(target)
    if (pd.api.types.is_bool_dtype(s_dt) and _is_nonbool_numeric_dtype(t_dt)) or (
        pd.api.types.is_bool_dtype(t_dt) and _is_nonbool_numeric_dtype(s_dt)
    ):
        return np.full(n_target, -1, dtype=np.intp)
    try:
        with np.errstate(invalid="ignore"):
            indexer = np.asarray(source.get_indexer(target), dtype=np.intp)
    except _INDEX_LOOKUP_ERRORS:
        return np.full(n_target, -1, dtype=np.intp)
    if np.any(indexer >= 0) and (
        pd.api.types.is_object_dtype(s_dt) or pd.api.types.is_object_dtype(t_dt)
    ):
        for i in np.flatnonzero(indexer >= 0):
            tv = target[int(i)]
            sv = source[int(indexer[i])]
            if (
                isinstance(tv, (bool, np.bool_)) != isinstance(sv, (bool, np.bool_))
                or isinstance(tv, pd.Interval) != isinstance(sv, pd.Interval)
            ):
                indexer[i] = -1
    return indexer


class _MatrixLocIndexer:
    """Label-based row indexer for NormalizedMatrix, mirroring pandas .loc."""

    def __init__(self, matrix: NormalizedMatrix) -> None:
        self._matrix = matrix

    def __getitem__(self, key: Any) -> NormalizedMatrix:
        if isinstance(key, tuple):
            raise IndexError(
                "NormalizedMatrix.loc is 1D (rows only); tuple / multi-axis indexing is not supported."
            )
        if key is Ellipsis or isinstance(
            key, (pd.DataFrame, Mapping, set, frozenset, bytearray, memoryview)
        ):
            raise TypeError(f"Cannot index NormalizedMatrix.loc with {type(key).__name__}.")
        if isinstance(key, np.ndarray) and key.ndim != 1:
            raise IndexError(
                f"NormalizedMatrix.loc requires a 1D indexer, got {key.ndim}D array."
            )

        idx = self._matrix.region_index
        if isinstance(key, pd.Series):
            series_elem_dtype = (
                key.dtype.categories.dtype
                if isinstance(key.dtype, pd.CategoricalDtype)
                else key.dtype
            )
            is_series_bool = pd.api.types.is_bool_dtype(key.dtype) or pd.api.types.is_bool_dtype(
                series_elem_dtype
            )
            if not is_series_bool and pd.api.types.is_object_dtype(series_elem_dtype) and len(key) > 0:
                vals = list(key)
                if any(isinstance(x, (bool, np.bool_)) for x in vals) and all(
                    isinstance(x, (bool, np.bool_)) or _is_scalar_na(x) for x in vals
                ):
                    is_series_bool = True
            if is_series_bool:
                if key.isna().any() or any(_is_scalar_na(x) for x in key):
                    raise ValueError("Boolean mask cannot contain NA / missing values.")
                if not key.index.is_unique:
                    raise ValueError("Boolean Series index must contain unique identifiers.")
                if len(key) != len(idx):
                    raise ValueError(
                        f"Boolean mask length ({len(key)}) does not match "
                        f"n_regions ({len(idx)})."
                    )
                pos = _safe_get_indexer(key.index, idx)
                if np.any(pos < 0):
                    raise KeyError("Boolean Series index does not match `region_index`.")
                mask = np.asarray(key.to_numpy()[pos], dtype=bool)
                return self._matrix._subset_positions(np.flatnonzero(mask))

        if isinstance(key, (np.ndarray, pd.Index, pd.api.extensions.ExtensionArray)):
            if getattr(key, "ndim", 1) != 1:
                raise IndexError("NormalizedMatrix.loc requires a 1D indexer.")
            elem_dtype = (
                key.dtype.categories.dtype
                if isinstance(key.dtype, pd.CategoricalDtype)
                else key.dtype
            )
            is_arr_bool = pd.api.types.is_bool_dtype(key.dtype) or pd.api.types.is_bool_dtype(
                elem_dtype
            )
            if (
                not is_arr_bool
                and not isinstance(key, pd.MultiIndex)
                and pd.api.types.is_object_dtype(elem_dtype)
                and len(key) > 0
            ):
                items = list(key)
                if any(isinstance(x, (bool, np.bool_)) for x in items) and all(
                    isinstance(x, (bool, np.bool_)) or _is_scalar_na(x) for x in items
                ):
                    if any(_is_scalar_na(x) for x in items):
                        raise ValueError("Boolean mask cannot contain NA / missing values.")
                    is_arr_bool = True
            if is_arr_bool:
                if pd.isna(key).any():
                    raise ValueError("Boolean mask cannot contain NA / missing values.")
                arr = np.asarray(key, dtype=bool)
                if len(arr) != len(idx):
                    raise ValueError(
                        f"Boolean mask length ({len(arr)}) does not match "
                        f"n_regions ({len(idx)})."
                    )
                return self._matrix._subset_positions(np.flatnonzero(arr))

        if isinstance(key, slice):
            slc = idx.slice_indexer(key.start, key.stop, key.step)
            positions = np.arange(*slc.indices(len(idx)), dtype=np.intp)
            return self._matrix._subset_positions(positions)

        if pd.api.types.is_scalar(key) or not isinstance(key, Iterable):
            _ensure_hashable(key)
            if isinstance(idx, pd.MultiIndex):
                raise KeyError(key)
            key_na_kind = _scalar_na_kind(key)
            if key_na_kind is not None:
                idx_dom = _dtype_domain(idx.dtype)
                if idx_dom != "object" and not _na_kind_matches_typed_domain(key_na_kind, idx_dom):
                    raise KeyError(key)
                pos_scalar: Any = None
                try:
                    pos_scalar = idx.get_loc(key)
                except _INDEX_LOOKUP_ERRORS:
                    raise KeyError(key) from None
                if not isinstance(pos_scalar, (int, np.integer)):
                    raise KeyError(key)
                pos_int = int(pos_scalar)
                if not _index_isna(idx)[pos_int]:
                    raise KeyError(key)
                if idx_dom == "object":
                    idx_na_kind, idx_doms = _object_index_info(idx, pos_int)
                    if not _na_profiles_compatible(
                        (idx_na_kind, idx_doms, False),
                        (key_na_kind, set(), False),
                    ):
                        raise KeyError(key)
                return self._matrix._subset_positions(np.array([pos_int], dtype=np.intp))

            idx_dom = _dtype_domain(idx.dtype)
            if idx_dom.startswith("interval:") and not isinstance(key, pd.Interval):
                raise KeyError(key)
            try:
                with np.errstate(invalid="ignore"):
                    pos_scalar = idx.get_loc(key)
            except _INDEX_LOOKUP_ERRORS:
                raise KeyError(key) from None
            if not isinstance(pos_scalar, (int, np.integer)):
                raise KeyError(key)
            pos_int = int(pos_scalar)
            if _index_isna(idx)[pos_int]:
                raise KeyError(key)
            matched_label = idx[pos_int]
            if (
                isinstance(key, (bool, np.bool_)) != isinstance(matched_label, (bool, np.bool_))
                or isinstance(key, pd.Interval) != isinstance(matched_label, pd.Interval)
            ):
                raise KeyError(key)
            return self._matrix._subset_positions(np.array([pos_int], dtype=np.intp))

        if not isinstance(
            key, (pd.Series, np.ndarray, pd.Index, pd.api.extensions.ExtensionArray)
        ):
            seq_items = list(key)
            if (
                len(seq_items) > 0
                and any(isinstance(x, (bool, np.bool_)) for x in seq_items)
                and all(isinstance(x, (bool, np.bool_)) or _is_scalar_na(x) for x in seq_items)
            ):
                if any(_is_scalar_na(x) for x in seq_items):
                    raise ValueError("Boolean mask cannot contain NA / missing values.")
                arr = np.asarray(seq_items, dtype=bool)
                if len(arr) != len(idx):
                    raise ValueError(
                        f"Boolean mask length ({len(arr)}) does not match "
                        f"n_regions ({len(idx)})."
                    )
                return self._matrix._subset_positions(np.flatnonzero(arr))
            for x in seq_items:
                _ensure_hashable(x)
            if (
                not isinstance(idx, pd.MultiIndex)
                and len(seq_items) > 0
                and any(_is_temporal_na_kind(_scalar_na_kind(x)) for x in seq_items)
            ):
                key_idx: pd.Index = pd.Index(seq_items, dtype=object, tupleize_cols=False)
            else:
                key_idx = pd.Index(seq_items, tupleize_cols=isinstance(idx, pd.MultiIndex))
        else:
            raw_elem_dtype = (
                key.dtype.categories.dtype
                if isinstance(key.dtype, pd.CategoricalDtype)
                else key.dtype
            )
            if isinstance(key, pd.MultiIndex) or pd.api.types.is_object_dtype(raw_elem_dtype):
                for x in key:
                    _ensure_hashable(x)
            key_idx = (
                key
                if isinstance(key, pd.Index)
                else pd.Index(key, tupleize_cols=isinstance(idx, pd.MultiIndex))
            )

        positions = _safe_get_indexer(idx, key_idx)
        missing = positions < 0
        if np.any(missing):
            missing_labels = list(key_idx[missing][:5])
            raise KeyError(f"Region IDs not found in region_index: {missing_labels}")
        return self._matrix._subset_positions(positions)


class _MatrixILocIndexer:
    """Integer-position row indexer for NormalizedMatrix."""

    def __init__(self, matrix: NormalizedMatrix) -> None:
        self._matrix = matrix

    def _validate_sequence(self, items: Sequence[Any]) -> np.ndarray:
        n_regions = self._matrix.n_regions
        if len(items) == 0:
            return np.empty(0, dtype=np.intp)
        if any(isinstance(x, (bool, np.bool_)) for x in items) and all(
            isinstance(x, (bool, np.bool_)) or _is_scalar_na(x) for x in items
        ):
            if any(_is_scalar_na(x) for x in items):
                raise ValueError("Boolean mask cannot contain NA / missing values.")
            arr = np.asarray(items, dtype=bool)
            if len(arr) != n_regions:
                raise ValueError(
                    f"Boolean mask length ({len(arr)}) does not match "
                    f"n_regions ({n_regions})."
                )
            return np.flatnonzero(arr)
        if all(
            isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
            for x in items
        ):
            out: list[int] = []
            for x in items:
                iv = int(x)
                if iv < -n_regions or iv >= n_regions:
                    raise IndexError(
                        f"positional index {iv} is out of bounds for axis 0 with size {n_regions}"
                    )
                out.append(iv)
            return np.array(out, dtype=np.intp)
        raise TypeError("Positional indexer must contain integers or a boolean mask.")

    def __getitem__(self, key: Any) -> NormalizedMatrix:
        if isinstance(key, tuple):
            raise IndexError(
                "NormalizedMatrix.iloc is 1D (rows only); tuple / multi-axis indexing is not supported."
            )
        if isinstance(key, (bool, np.bool_)):
            raise TypeError("Cannot index by location with a boolean scalar.")
        if isinstance(key, (int, np.integer)):
            iv = int(key)
            n_regions = self._matrix.n_regions
            if iv < -n_regions or iv >= n_regions:
                raise IndexError(
                    f"positional index {iv} is out of bounds for axis 0 with size {n_regions}"
                )
            return self._matrix._subset_positions(np.array([iv], dtype=np.intp))
        if (
            key is None
            or key is Ellipsis
            or key is pd.NA
            or key is pd.NaT
            or isinstance(key, (str, bytes, bytearray))
            or np.isscalar(key)
        ):
            raise TypeError(f"Cannot index by location with scalar of type {type(key).__name__}.")
        if isinstance(key, slice):
            for bound in (key.start, key.stop, key.step):
                if bound is not None and (
                    isinstance(bound, (bool, np.bool_))
                    or not isinstance(bound, (int, np.integer))
                ):
                    raise TypeError(
                        f"Slice bounds for iloc must be integers or None, got {type(bound).__name__}."
                    )
            positions = np.arange(*key.indices(self._matrix.n_regions), dtype=np.intp)
            return self._matrix._subset_positions(positions)
        if isinstance(key, (pd.DataFrame, pd.MultiIndex, Mapping, set, frozenset)):
            raise TypeError(f"Cannot index by location with {type(key).__name__}.")
        if isinstance(key, np.ndarray) and key.ndim != 1:
            raise TypeError(
                f"NormalizedMatrix.iloc requires a 1D indexer, got {key.ndim}D array."
            )
        if isinstance(key, (np.ndarray, pd.Series, pd.Index, pd.api.extensions.ExtensionArray)):
            if getattr(key, "ndim", 1) != 1:
                raise TypeError("NormalizedMatrix.iloc requires a 1D indexer.")
            elem_dtype = (
                key.dtype.categories.dtype
                if isinstance(key.dtype, pd.CategoricalDtype)
                else key.dtype
            )
            if pd.api.types.is_bool_dtype(key.dtype) or pd.api.types.is_bool_dtype(elem_dtype):
                if pd.isna(key).any():
                    raise ValueError("Boolean mask cannot contain NA / missing values.")
                arr = np.asarray(key, dtype=bool)
                if len(arr) != self._matrix.n_regions:
                    raise ValueError(
                        f"Boolean mask length ({len(arr)}) does not match "
                        f"n_regions ({self._matrix.n_regions})."
                    )
                return self._matrix._subset_positions(np.flatnonzero(arr))
            if pd.api.types.is_signed_integer_dtype(elem_dtype):
                if pd.isna(key).any():
                    raise TypeError("Cannot index by location with NA / missing values.")
                iarr = np.asarray(key, dtype=np.int64)
                n_regions = self._matrix.n_regions
                if iarr.size and (int(np.min(iarr)) < -n_regions or int(np.max(iarr)) >= n_regions):
                    bad = int(iarr[np.flatnonzero((iarr < -n_regions) | (iarr >= n_regions))[0]])
                    raise IndexError(
                        f"positional index {bad} is out of bounds for axis 0 with size {n_regions}"
                    )
                return self._matrix._subset_positions(iarr.astype(np.intp, copy=False))
            if pd.api.types.is_unsigned_integer_dtype(elem_dtype):
                if pd.isna(key).any():
                    raise TypeError("Cannot index by location with NA / missing values.")
                uarr = np.asarray(key, dtype=np.uint64)
                n_regions = self._matrix.n_regions
                if uarr.size and int(np.max(uarr)) >= n_regions:
                    bad = int(uarr[np.flatnonzero(uarr >= n_regions)[0]])
                    raise IndexError(
                        f"positional index {bad} is out of bounds for axis 0 with size {n_regions}"
                    )
                return self._matrix._subset_positions(uarr.astype(np.intp))
            if pd.api.types.is_object_dtype(elem_dtype):
                return self._matrix._subset_positions(self._validate_sequence(list(key)))
            raise TypeError(f"Cannot index by location with dtype {key.dtype}.")
        if isinstance(key, range):
            if not key:
                return self._matrix._subset_positions(np.empty(0, dtype=np.intp))
            n_regions = self._matrix.n_regions
            for endpoint in (key[0], key[-1]):
                if endpoint < -n_regions or endpoint >= n_regions:
                    raise IndexError(
                        f"positional index {endpoint} is out of bounds for axis 0 with size {n_regions}"
                    )
            return self._matrix._subset_positions(np.asarray(key, dtype=np.intp))
        if isinstance(key, Sequence):
            return self._matrix._subset_positions(self._validate_sequence(key))
        if isinstance(key, Iterable):
            return self._matrix._subset_positions(self._validate_sequence(list(key)))
        raise TypeError(f"Unsupported indexer type for iloc: {type(key).__name__}.")


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
        if not idx.is_unique:
            raise ValueError("`region_index` must contain unique region identifiers.")
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

        indexer = _safe_get_indexer(self.region_index, target)
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
        if other.region_index is self.region_index:
            return other
        if len(self.region_index) == 0 or len(other.region_index) == 0:
            return other.reindex(self.region_index)
        indexer = _safe_get_indexer(other.region_index, self.region_index)
        self_na = _index_isna(self.region_index)
        both_all_na = bool(self_na.all()) and bool(_index_isna(other.region_index).all())
        has_overlap = (
            bool(np.any(indexer >= 0))
            if both_all_na
            else bool(np.any((indexer >= 0) & ~self_na))
        )
        if not has_overlap:
            raise ValueError(
                "`other` shares no region identifiers with `self` "
                f"(self starts {list(self.region_index[:3])}, "
                f"other starts {list(other.region_index[:3])}). "
                "Pass `region_index=` when constructing matrices from arrays."
            )
        if len(other.region_index) == len(self.region_index) and np.array_equal(
            indexer, np.arange(len(self.region_index), dtype=np.intp)
        ):
            return other
        out = np.full((len(self.region_index), other.n_bins), np.nan, dtype=other.values.dtype)
        present = indexer >= 0
        if np.any(present):
            out[present] = other.values[indexer[present]]
        return other._clone(values=out, region_index=self.region_index.copy())

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
