"""Small synthetic data set so the tutorials run without a misha database or bigWigs.

The signals are defined along a toy genome (two chromosomes), so they are consistent
across overlapping windows, and :func:`example_values` returns exactly the table
``pymisha.gextract(tracks, intervals=bins, iterator=bins)`` would.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

_GENOME = {"chr1": 3_000_000, "chr2": 2_000_000}
TRACKS = ("chec", "chec_ko", "atac", "mnase", "k4me3")


def example_genome() -> pd.DataFrame:
    """Chromosome sizes of the toy genome, shaped like ``pymisha.gintervals_all()``."""
    return pd.DataFrame(
        {"chrom": list(_GENOME), "start": 0, "end": list(_GENOME.values())}
    )


def example_peaks(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Synthetic peaks: promoters (stranded), enhancers and CTCF sites.

    ``n`` is limited by the toy genome (about 1,600 peaks 3 kb apart).

    Columns: ``chrom, start, end, strand, class, bound, score``; the index holds peak
    names. ``bound`` marks peaks occupied by the ChEC-profiled factor; about a third of
    the bound promoters lose binding in the ``chec_ko`` track.
    """
    if n < 1:
        raise ValueError("`n` must be positive.")
    rng = np.random.default_rng(seed)
    chroms = rng.choice(list(_GENOME), size=n, p=[0.6, 0.4])
    # Space peaks at least 3 kb apart so windows of +-1 kb do not collide.
    out = []
    for c in _GENOME:
        k = int(np.sum(chroms == c))
        capacity = _GENOME[c] // 3000 - 10
        if k > capacity:
            raise ValueError(
                f"The toy genome has room for about {int(capacity / 0.4)} peaks 3 kb apart; "
                f"n={n} puts {k} on {c}, which holds {capacity}. Use a smaller `n`."
            )
        slots = rng.choice(np.arange(5, _GENOME[c] // 3000 - 5), size=k, replace=False)
        centers = np.sort(slots) * 3000 + rng.integers(-500, 500, size=k)
        out.append(pd.DataFrame({"chrom": c, "center": centers}))
    df = pd.concat(out, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    width = rng.integers(200, 600, size=n)
    df["start"] = df["center"] - width // 2
    df["end"] = df["start"] + width
    df["class"] = pd.Categorical(
        rng.choice(["promoter", "enhancer", "CTCF"], size=n, p=[0.4, 0.45, 0.15]),
        categories=["promoter", "enhancer", "CTCF"],
    )
    df["strand"] = np.where(df["class"] == "promoter", rng.choice(["+", "-"], size=n), "+")
    df["bound"] = rng.random(n) < np.select(
        [df["class"] == "promoter", df["class"] == "enhancer"], [0.55, 0.35], 0.05
    )
    df["score"] = np.round(rng.lognormal(mean=3.0, sigma=0.6, size=n), 1)
    df.index = pd.Index([f"peak{i:04d}" for i in range(n)], name="name")
    return df[["chrom", "start", "end", "strand", "class", "bound", "score"]]


def _noise(chrom_code: np.ndarray, pos: np.ndarray, salt: float) -> np.ndarray:
    """Deterministic pseudo-random uniforms keyed on position, so overlaps agree."""
    x = np.sin(pos * 12.9898e-3 + chrom_code * 78.233 + salt) * 43758.5453
    return np.clip(x - np.floor(x), 1e-9, 1 - 1e-9)


def example_values(
    bins: pd.DataFrame,
    tracks: Sequence[str] = ("chec", "atac"),
    peaks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Synthetic signal for every bin, in ``pymisha.gextract`` output layout.

    Parameters
    ----------
    bins : pandas.DataFrame
        From :func:`enriched_heatmap.bins`, built on :func:`example_peaks` (or any
        regions on the toy genome).
    tracks : list of str
        Any of ``"chec"`` (narrow factor footprint), ``"chec_ko"`` (the same factor
        after a knockout), ``"atac"``, ``"mnase"`` (nucleosome-depleted region with
        phased nucleosomes) and ``"k4me3"`` (promoters only, downstream of the TSS).
    peaks : pandas.DataFrame, optional
        The peaks that generate the signal. Default: :func:`example_peaks`. Pass the
        same table the bins were built from if it is not the default one; a mismatch
        raises.

    Returns
    -------
    pandas.DataFrame
        ``chrom, start, end, <tracks...>, intervalID``.
    """
    bad = [t for t in tracks if t not in TRACKS]
    if bad:
        raise ValueError(f"Unknown example tracks {bad}; choose from {TRACKS}.")
    peaks = example_peaks() if peaks is None else peaks
    span = bins.groupby("region", sort=False).agg(
        chrom=("chrom", "first"), lo=("start", "min"), hi=("end", "max")
    )
    unknown = span.index.difference(peaks.index)
    known = span.drop(index=unknown)
    # Whatever the anchor, the bin window must overlap the peak it was built from.
    moved = (
        (peaks["chrom"].astype(str).reindex(known.index) != known["chrom"].astype(str))
        | (peaks["end"].reindex(known.index) <= known["lo"])
        | (peaks["start"].reindex(known.index) >= known["hi"])
    )
    if len(unknown) or moved.any():
        bad = list(unknown[:3]) + list(known.index[moved.to_numpy()][:3])
        raise ValueError(
            f"{len(unknown) + int(moved.sum())} bin regions (e.g. {bad[:3]}) are not at the "
            "positions given by `peaks`. Pass the table the bins were built from: "
            "example_values(b, tracks, peaks=peaks)."
        )
    rng = np.random.default_rng(1)
    amp = rng.lognormal(0.0, 0.5, size=len(peaks))
    klass = peaks["class"].astype(str).to_numpy()
    bound = peaks["bound"].to_numpy()
    sign = np.where(peaks["strand"].to_numpy() == "-", -1.0, 1.0)
    lost = bound & (klass == "promoter") & (rng.random(len(peaks)) < 0.35)
    centers = ((peaks["start"] + peaks["end"]) // 2).to_numpy()

    mid = ((bins["start"] + bins["end"]) // 2).to_numpy(np.int64)
    chrom = bins["chrom"].astype(str).to_numpy()
    chrom_code = pd.factorize(chrom)[0].astype(float)
    out = bins[["chrom", "start", "end"]].copy()
    cols = {t: np.full(len(bins), 4.0 if t == "mnase" else 0.0) for t in tracks}

    for c in pd.unique(chrom):
        on = np.flatnonzero(peaks["chrom"].to_numpy() == c)
        if on.size == 0:
            continue
        order = on[np.argsort(centers[on])]
        pc = centers[order]
        sel = np.flatnonzero(chrom == c)
        m = mid[sel]
        idx = np.searchsorted(pc, m)
        for o in (-2, -1, 0, 1):
            j = idx + o
            ok = (j >= 0) & (j < pc.size)
            j = np.clip(j, 0, pc.size - 1)
            p = order[j]
            d = (m - pc[j]).astype(float)
            near = ok & (np.abs(d) < 4000)
            a, k = amp[p], klass[p]
            if "chec" in cols or "chec_ko" in cols:
                foot = np.exp(-0.5 * (d / 45.0) ** 2) * 30.0 * a * bound[p]
                if "chec" in cols:
                    cols["chec"][sel] += np.where(near, foot, 0)
                if "chec_ko" in cols:
                    cols["chec_ko"][sel] += np.where(near, foot * np.where(lost[p], 0.15, 1.0), 0)
            if "atac" in cols:
                level = np.select([k == "promoter", k == "enhancer"], [12.0, 6.0], 4.0)
                cols["atac"][sel] += np.where(near, level * a * np.exp(-0.5 * (d / 180.0) ** 2), 0)
            if "mnase" in cols:
                # Background occupancy is added once below; this is each site's deviation.
                phase = 0.5 + 0.5 * np.cos(2 * np.pi * d / 165.0)
                depth = np.select([k == "CTCF", k == "promoter"], [1.0, 0.8], 0.4)
                ndr = np.exp(-0.5 * (d / 90.0) ** 2)
                decay = np.exp(-np.abs(d) / 700.0)
                mn = -depth * 3.5 * ndr + depth * 3.0 * (phase - 0.5) * decay * (1 - ndr)
                cols["mnase"][sel] += np.where(near, mn, 0)
            if "k4me3" in cols:
                dd = d * sign[p]  # transcription direction
                body = np.exp(-0.5 * ((dd - 450.0) / 300.0) ** 2) + 0.4 * np.exp(-0.5 * ((dd + 250.0) / 150.0) ** 2)
                cols["k4me3"][sel] += np.where(near & (k == "promoter"), 15.0 * a * body, 0)

    for i, t in enumerate(tracks):
        u = _noise(chrom_code, mid.astype(float), salt=float(i))
        base = {"chec": 0.8, "chec_ko": 0.8, "atac": 1.0, "mnase": 1.2, "k4me3": 0.6}[t]
        out[t] = np.round(np.maximum(cols[t] + base * -np.log(u), 0.0), 3)
    out["intervalID"] = np.arange(1, len(bins) + 1)
    return out
