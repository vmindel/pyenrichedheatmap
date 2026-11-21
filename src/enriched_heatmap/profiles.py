"""Top metaplot summary profile computation (mean/median with SEM, CI, or IQR ribbons)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np

ProfileMode = Literal["mean", "median", "mean_sem", "mean_ci", "mean_boot_ci", "median_iqr"]


@dataclass(frozen=True)
class ProfileResult:
    """Computed aggregate profile along the bin axis."""

    center: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    n_valid: np.ndarray


def _normal_z(ci: float) -> float:
    """Return two-sided normal quantile z for confidence level ``ci``."""
    alpha = 1.0 - ci
    try:
        from scipy.stats import norm

        return float(norm.ppf(1.0 - alpha / 2.0))
    except ImportError:
        # Common exact z-values + Beasley-Springer-Moro approximation fallback
        common = {0.90: 1.6448536, 0.95: 1.9599640, 0.99: 2.5758293}
        if ci in common:
            return common[ci]
        # Acklam/Abramowitz-Stegun approximation for upper tail p = alpha / 2
        p = alpha / 2.0
        t = np.sqrt(-2.0 * np.log(p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return float(t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t**3))


def compute_profile(
    values: np.ndarray,
    mode: ProfileMode = "mean",
    ci: float = 0.95,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> ProfileResult:
    """Compute column-wise aggregate signal and optional uncertainty band.

    Parameters
    ----------
    values : np.ndarray
        2D array of shape ``(n_regions, n_bins)``.
    mode : {"mean", "median", "mean_sem", "mean_ci", "mean_boot_ci", "median_iqr"}
        Summary statistic and error ribbon type.
    ci : float, default 0.95
        Confidence level for ``"mean_ci"`` and ``"mean_boot_ci"``.
    n_bootstrap : int, default 200
        Number of bootstrap resamples for ``"mean_boot_ci"``.
    random_state : int, default 42
        RNG seed for bootstrap resampling.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"`values` must be 2D, got shape {arr.shape}.")

    n_rows, n_bins = arr.shape
    if n_rows == 0:
        empty = np.full(n_bins, np.nan, dtype=np.float64)
        return ProfileResult(
            center=empty,
            lower=None if mode in ("mean", "median") else empty.copy(),
            upper=None if mode in ("mean", "median") else empty.copy(),
            n_valid=np.zeros(n_bins, dtype=int),
        )

    finite = np.isfinite(arr)
    n_valid = finite.sum(axis=0)
    has_data = n_valid > 0

    with np.errstate(invalid="ignore", divide="ignore", all="ignore"), warnings.catch_warnings():
        # empty bins are expected and come out as NaN; numpy's "empty slice" warnings add nothing
        warnings.simplefilter("ignore", RuntimeWarning)
        if mode == "mean":
            center = np.where(has_data, np.nanmean(arr, axis=0), np.nan)
            return ProfileResult(center=center, lower=None, upper=None, n_valid=n_valid)

        if mode == "median":
            center = np.where(has_data, np.nanmedian(arr, axis=0), np.nan)
            return ProfileResult(center=center, lower=None, upper=None, n_valid=n_valid)

        if mode in ("mean_sem", "mean_ci"):
            center = np.where(has_data, np.nanmean(arr, axis=0), np.nan)
            sd = np.where(n_valid > 1, np.nanstd(arr, axis=0, ddof=1), 0.0)
            sem = np.where(n_valid > 1, sd / np.sqrt(np.maximum(n_valid, 1)), 0.0)
            if mode == "mean_sem":
                half = sem
            else:
                z = _normal_z(ci)
                half = z * sem
            lower = np.where(has_data, center - half, np.nan)
            upper = np.where(has_data, center + half, np.nan)
            return ProfileResult(center=center, lower=lower, upper=upper, n_valid=n_valid)

        if mode == "mean_boot_ci":
            center = np.where(has_data, np.nanmean(arr, axis=0), np.nan)
            if n_rows == 1:
                return ProfileResult(
                    center=center,
                    lower=center.copy(),
                    upper=center.copy(),
                    n_valid=n_valid,
                )
            rng = np.random.default_rng(random_state)
            # Resample by multinomial row counts: each bootstrap mean is then one matrix
            # product over the NaN-aware sums, with no n_rows x n_bins copy per resample.
            filled = np.where(finite, arr, 0.0)
            ok = finite.astype(np.float64)
            pvals = np.full(n_rows, 1.0 / n_rows)
            boot_means = np.empty((n_bootstrap, n_bins), dtype=np.float64)
            step = max(1, int(2e7 // n_rows))
            for lo in range(0, n_bootstrap, step):
                hi = min(n_bootstrap, lo + step)
                counts = rng.multinomial(n_rows, pvals, size=hi - lo).astype(np.float64)
                num, den = counts @ filled, counts @ ok
                boot_means[lo:hi] = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
            alpha = 1.0 - ci
            lower = np.where(has_data, np.nanquantile(boot_means, alpha / 2.0, axis=0), np.nan)
            upper = np.where(
                has_data, np.nanquantile(boot_means, 1.0 - alpha / 2.0, axis=0), np.nan
            )
            return ProfileResult(center=center, lower=lower, upper=upper, n_valid=n_valid)

        if mode == "median_iqr":
            center = np.where(has_data, np.nanmedian(arr, axis=0), np.nan)
            lower = np.where(has_data, np.nanquantile(arr, 0.25, axis=0), np.nan)
            upper = np.where(has_data, np.nanquantile(arr, 0.75, axis=0), np.nan)
            return ProfileResult(center=center, lower=lower, upper=upper, n_valid=n_valid)

    raise ValueError(
        f"Unsupported `mode` {mode!r}. Choose from "
        "'mean', 'median', 'mean_sem', 'mean_ci', 'mean_boot_ci', 'median_iqr'."
    )
