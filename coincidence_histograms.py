# -*- coding: utf-8 -*-
"""
coincidence_histograms.py

Global-time coincidence analysis for multi-channel detector systems.

For each pivot hit t[n]:
    coincidence window = (t[n-1], t[n+1])

All hits from other channels inside this window are collected and
Δt = t_hit - t_ref is histogrammed per channel.

Time units: nanoseconds (ns).
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from tqdm import tqdm
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from dataclasses import dataclass
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

from filter_functions import load_cfd50_lookup, compute_cfd50
import mplhep as hep

# =============================================================================
# Data container
# =============================================================================

@dataclass
class HistogramResult:
    counts:     np.ndarray
    bin_edges:  np.ndarray
    fit_params: dict | None = None

    @property
    def bin_centres(self) -> np.ndarray:
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    @property
    def total_counts(self) -> int:
        return int(np.sum(self.counts))


# =============================================================================
# Models  — defined ONCE, no duplicate
# =============================================================================

def gaussian_background(
    x: np.ndarray,
    amplitude: float,
    mu: float,
    sigma: float,
    background: float,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + background


def multi_gaussian_background(x: np.ndarray, *params) -> np.ndarray:
    """
    Sum of Gaussians + constant background.

    Parameter layout: [A1, mu1, sigma1, A2, mu2, sigma2, ..., background]
    """
    n_peaks = (len(params) - 1) // 3
    bg = params[-1]
    y = np.full_like(x, bg, dtype=np.float64)
    for i in range(n_peaks):
        A, mu, sigma = params[3*i], params[3*i + 1], abs(params[3*i + 2])
        y += A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return y




# =============================================================================
# Coincidence histogram builder
# =============================================================================

def build_coincidence_histograms(
    parquet_path: str,
    pivot_channel: int,
    channel_col: str = "Channel",
    time_col: str = "OrderedCell0Time",
    x_range: float = 100.0,
    channel_filter: List[int] = None,
    bin_width: float = 0.05,
    period: float = 1 / 6.4,
    use_cfd50: bool = True,
    cfd50_path: Optional[str] = None,   # path to precomputed CFD50 parquet
    batch_size: int = 100_000,
    hit_mask: Optional[np.ndarray] = None,
) -> Dict[int, HistogramResult]:
    """
    Build per-channel coincidence timing histograms with optional channel filtering.
    """

# =========================================================================
# PASS 1 — lightweight load + vectorised candidate selection
# =========================================================================

    mask_set = None
    if hit_mask is not None:
        mask_set = set(hit_mask.tolist())

    pqf = pq.ParquetFile(parquet_path)

    hit_ids_list  = []
    times_list    = []
    channels_list = []

    cols = ["HITNumber", channel_col, time_col]

    for batch in pqf.iter_batches(batch_size=batch_size, columns=cols):

        hit_ids = batch["HITNumber"].to_numpy()

        # 1. Apply reusable hit mask (Existing Logic)
        if mask_set is not None:
            keep = np.fromiter(
                (h in mask_set for h in hit_ids),
                dtype=bool,
                count=len(hit_ids),
            )

            if not np.any(keep):
                continue

            batch = batch.take(pa.array(np.where(keep)[0]))
            hit_ids = hit_ids[keep]

        # 2. Apply Channel Filter (NEW LOGIC)
        if channel_filter is not None:
            channels_batch = batch[channel_col].to_numpy()
            
            # CRITICAL: We must keep rows matching the filter OR the pivot channel.
            # Without the pivot channel, we cannot calculate coincidence deltas!
            keep_ch = np.isin(channels_batch, channel_filter) | (channels_batch == pivot_channel)
            
            if not np.any(keep_ch):
                continue
                
            # Slice the PyArrow batch to drop unneeded channels early
            batch = batch.take(pa.array(np.where(keep_ch)[0]))
            hit_ids = hit_ids[keep_ch]

        # Append remaining valid data to global tracking lists
        hit_ids_list.append(hit_ids)

        times_list.append(
            batch[time_col]
            .to_numpy(zero_copy_only=False)
            .astype(np.float64)
        )

        channels_list.append(
            batch[channel_col].to_numpy()
        )

    # Concatenate filtered data
    hit_ids_all  = np.concatenate(hit_ids_list)
    times_all    = np.concatenate(times_list)
    channels_all = np.concatenate(channels_list)

    # Time ordering
    order           = np.argsort(times_all, kind="stable")

    hit_ids_sorted  = hit_ids_all[order]
    times_sorted    = times_all[order]
    channels_sorted = channels_all[order]

    # Pivot selection
    pivot_mask  = channels_sorted == pivot_channel
    pivot_times = times_sorted[pivot_mask]

    if len(pivot_times) < 3:
        raise ValueError(
            f"Pivot channel {pivot_channel} has fewer than 3 hits."
        )

    print(f"Total hits  : {len(times_sorted):,}")
    print(f"Pivot hits  : {len(pivot_times):,}")

    # Candidate selection
    lo_idx = np.searchsorted(
        pivot_times,
        times_sorted - x_range,
        side="left",
    )

    hi_idx = np.searchsorted(
        pivot_times,
        times_sorted + x_range,
        side="right",
    )

    candidate_mask = (lo_idx < hi_idx) | pivot_mask

    # IMPORTANT: candidate HIT IDs, NOT parquet row indices
    candidate_hit_ids = hit_ids_sorted[candidate_mask]
    candidate_set = set(candidate_hit_ids.tolist())

    print(
        f"Candidates  : {len(candidate_hit_ids):,} "
        f"({100 * len(candidate_hit_ids) / len(times_sorted):.1f}%)"
    )

    del (
        hit_ids_sorted,
        times_sorted,
        channels_sorted,
        hit_ids_all,
        times_all,
        channels_all,
        candidate_mask,
    )

    # =========================================================================
    # PASS 2 — load waveform columns only for candidate hits
    # =========================================================================

    waveform_cols = [
        "HITNumber",
        channel_col,
        time_col,
        "Baseline",
        "DataSample",
        "TriggerPosition",
    ]

    pqf = pq.ParquetFile(parquet_path)
    collected_batches = []

    for batch in pqf.iter_batches(
        batch_size=batch_size,
        columns=waveform_cols,
    ):
        hit_ids = batch["HITNumber"].to_numpy()

        # Keep only candidate hits
        keep = np.fromiter(
            (h in candidate_set for h in hit_ids),
            dtype=bool, count=len(hit_ids),
        )

        if not np.any(keep):
            continue

        local_idx = np.where(keep)[0]
        collected_batches.append(
            batch.take(pa.array(local_idx))
        )

    if not collected_batches:
        raise RuntimeError("No candidate hits found in file.")

    candidate_table = pa.Table.from_batches(collected_batches)

    # -------------------------------------------------------------------------
    # Extract arrays EXACTLY like before
    # -------------------------------------------------------------------------
    channels = (
        candidate_table.column(channel_col)
        .combine_chunks()
        .to_numpy()
    )

    hit_ids = (
        candidate_table.column("HITNumber")
        .combine_chunks()
        .to_numpy()
    )

    times_raw = (
        candidate_table.column(time_col)
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype(np.float64)
    )

    baselines = (
        candidate_table.column("Baseline")
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype(np.float64)
    )

    samples_col = candidate_table.column("DataSample")
    triggers_col = candidate_table.column("TriggerPosition")

    del candidate_table

    # =========================================================================
    # CFD50 — only on candidates (already pre-filtered to within x_range)
    # =========================================================================
    if use_cfd50:
        if cfd50_path is not None:
            cfd50_lut = load_cfd50_lookup(cfd50_path)
            corrected_times = times_raw.copy()
            valid_mask      = np.ones(len(times_raw), dtype=bool)
            failed = 0

            for i, hid in enumerate(hit_ids):
                offset = cfd50_lut.get(int(hid))

                if offset is None:
                    valid_mask[i] = False
                    failed += 1
                else:
                    corrected_times[i] += offset
            print(
                f"CFD50 lookup failures: "
                f"{failed:,} / {len(times_raw):,} "
                f"({100 * failed / len(times_raw):.2f}%)"
            )

            # Apply consistently in both paths:
            corrected_times = corrected_times[valid_mask]
            channels        = channels[valid_mask]
            hit_ids         = hit_ids[valid_mask]
            baselines       = baselines[valid_mask]
            print(f"CFD50 lookup: {valid_mask.sum():,} hits used, "
                f"{(~valid_mask).sum():,} dropped (not in cfd50 lookup)")
        else:
            corrected_times = times_raw.copy()
            valid_mask      = np.ones(len(times_raw), dtype=bool)
            failed          = 0

            for i in tqdm(range(len(times_raw)), desc="CFD50"):
                offset = compute_cfd50(
                    np.asarray(samples_col[i].as_py(),  dtype=np.float64),
                    np.asarray(triggers_col[i].as_py(), dtype=np.int32),
                    baselines[i],
                    period,
                )
                if offset is None:
                    valid_mask[i] = False
                    failed += 1
                else:
                    corrected_times[i] += offset

            print(
                f"CFD failures: {failed:,} / {len(times_raw):,} "
                f"({100 * failed / len(times_raw):.2f}%)"
            )
            corrected_times = corrected_times[valid_mask]
            channels        = channels[valid_mask]
            baselines  = baselines[valid_mask]

    # =========================================================================
    # Final sort → coincidence histogram loop
    # =========================================================================
    order           = np.argsort(corrected_times, kind="stable")
    times_sorted    = corrected_times[order]
    channels_sorted = channels[order]

    pivot_times    = times_sorted[channels_sorted == pivot_channel]
    other_channels = sorted(
        [int(ch) for ch in np.unique(channels_sorted) if ch != pivot_channel]
    )

    bin_edges  = np.arange(-x_range, x_range + bin_width, bin_width, dtype=np.float64)
    n_bins     = len(bin_edges) - 1
    histograms = {ch: np.zeros(n_bins, dtype=np.int64) for ch in other_channels}

    for n in tqdm(
        range(1, len(pivot_times) - 1),
        desc=f"Building histograms (pivot ch {pivot_channel})",
    ):
        t_ref  = pivot_times[n]
        t_prev = pivot_times[n - 1]
        t_next = pivot_times[n + 1]
        
        lo_bound = 0.5 * (t_prev + t_ref)
        hi_bound = 0.5 * (t_ref + t_next)

        lo_t = max(t_ref - x_range, lo_bound)
        hi_t = min(t_ref + x_range, hi_bound)

        lo = np.searchsorted(times_sorted, lo_t, side="left")
        hi = np.searchsorted(times_sorted, hi_t, side="right")

        if lo >= hi:
            continue

        wt = times_sorted[lo:hi]
        wc = channels_sorted[lo:hi]

        nonpivot = wc != pivot_channel
        wt, wc   = wt[nonpivot], wc[nonpivot]
        if len(wt) == 0:
            continue

        deltas = wt - t_ref
        valid  = (deltas >= -x_range) & (deltas <= x_range)
        deltas, wc = deltas[valid], wc[valid]
        if len(deltas) == 0:
            continue

        bin_idx  = np.digitize(deltas, bin_edges) - 1
        in_range = (bin_idx >= 0) & (bin_idx < n_bins)
        bin_idx, wc = bin_idx[in_range], wc[in_range]

        for ch in other_channels:
            ch_mask = wc == ch
            if np.any(ch_mask):
                np.add.at(histograms[ch], bin_idx[ch_mask], 1)

    return {
        ch: HistogramResult(counts=histograms[ch], bin_edges=bin_edges)
        for ch in other_channels
    }


# =============================================================================
# Single-peak fit  (used by plot_coincidence_histograms)
# =============================================================================

def fit_histogram(
    result: HistogramResult,
    fit_window: float = 5.0,
) -> dict | None:
    """
    Fit Gaussian + constant background around the tallest peak.

    Parameters
    ----------
    fit_window : half-width in ns of the fit region, centred on peak bin.
                 Should be ~10x the expected peak sigma, NOT the full x_range.
    """
    x = result.bin_centres
    y = result.counts.astype(np.float64)

    if np.sum(y) == 0:
        return None

    peak_idx = int(np.argmax(y))
    amp0     = float(y[peak_idx])
    mu0      = float(x[peak_idx])
    bg0      = float(np.median(y))

    mask  = np.abs(x - mu0) <= fit_window
    x_fit = x[mask]
    y_fit = y[mask]

    half_max   = amp0 / 2.0
    above_half = x_fit[y_fit >= half_max]
    if len(above_half) >= 2:
        sigma0 = float((above_half[-1] - above_half[0]) / 2.355)
    else:
        sigma0 = float(3.0 * np.median(np.diff(x)))

    nz    = y_fit > 0
    x_fit = x_fit[nz]
    y_fit = y_fit[nz]

    if len(x_fit) < 5:
        return None

    n_params = 4
    ndf      = len(x_fit) - n_params
    if ndf <= 0:
        return None

    sigma_y = np.sqrt(np.maximum(y_fit, 1.0))

    try:
        popt, pcov = curve_fit(
            gaussian_background,
            x_fit, y_fit,
            p0=[amp0, mu0, sigma0, bg0],
            sigma=sigma_y,
            absolute_sigma=True,
            bounds=([0, -np.inf, 1e-6, 0], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=20_000,
        )
        perr    = np.sqrt(np.diag(pcov))
        y_model = gaussian_background(x_fit, *popt)
        chi2    = float(np.sum(((y_fit - y_model) / sigma_y) ** 2))

        return {
            "amplitude":     popt[0], "amplitude_err": perr[0],
            "mu":            popt[1], "mu_err":        perr[1],
            "sigma":     abs(popt[2]), "sigma_err":    perr[2],
            "background":    popt[3], "background_err":perr[3],
            "chi2":          chi2,    "ndf":           ndf,
            "chi2_red":      chi2 / ndf,
        }
    except (RuntimeError, ValueError):
        return None


# =============================================================================
# Single-peak plotting
# =============================================================================

def plot_coincidence_histograms(
    histograms: Dict[int, HistogramResult],
    pivot_channel: int,
    time_unit: str = "ns",
    fit_window: float = 5.0,
    plot_range: Optional[float] = None,
    logy: bool = False,
) -> None:
    """
    Plot single-peak Gaussian fits for each channel histogram.

    Parameters
    ----------
    fit_window  : half-width in ns of the fit region around the tallest peak.
                  Should be ~10–20x your expected sigma, e.g. 3 ns for 0.3 ns peaks.
    plot_range  : half-width in ns of the displayed x-axis, centred on fitted μ.
                  If None, uses the full histogram range.
    """
    channels = sorted(histograms.keys())
    n        = len(channels)
    ncols    = min(2, n)
    nrows    = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, ch in tqdm(zip(axes, channels), desc="Building plots", total=n):
        result  = histograms[ch]
        centres = result.bin_centres
        counts  = result.counts
        bw      = result.bin_edges[1] - result.bin_edges[0]

        if np.sum(counts) == 0:
            ax.text(0.5, 0.5, "No coincidences",
                    transform=ax.transAxes, ha="center", va="center", color="gray")
            ax.set_title(f"Ch {pivot_channel} → Ch {ch}")
            continue

        fit_result = fit_histogram(result, fit_window=fit_window)

        if fit_result is not None:
            centre = fit_result["mu"]
        else:
            centre = float(centres[np.argmax(counts)])

        half    = plot_range if plot_range is not None else (centres[-1] - centres[0]) / 2
        lo_plot = centre - half
        hi_plot = centre + half

        vis = (centres >= lo_plot) & (centres <= hi_plot)
        ax.bar(centres[vis], counts[vis], width=bw, align="center", alpha=0.75)

        if fit_result is not None:
            result.fit_params = fit_result
            xd = np.linspace(lo_plot, hi_plot, 5000)
            yd = gaussian_background(
                xd, fit_result["amplitude"], fit_result["mu"],
                fit_result["sigma"], fit_result["background"],
            )
            ax.plot(xd, yd, lw=1, color="red", linestyle="--", label="Gaussian fit")
            ax.text(
                0.97, 0.95,
                f"μ = {fit_result['mu']:.3f} ± {fit_result['mu_err']:.3f} {time_unit}\n"
                f"σ = {fit_result['sigma']:.3f} ± {fit_result['sigma_err']:.3f} {time_unit}\n"
                f"χ²/ndf = {fit_result['chi2_red']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )
            ax.legend(fontsize=8)

        ax.set_xlim(lo_plot, hi_plot)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.set_title(f"Ch {pivot_channel} → Ch {ch}")
        ax.set_xlabel(f"Δt ({time_unit})")
        ax.set_ylabel("Counts")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Coincidence timing — pivot channel {pivot_channel}", fontsize=14)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Peak detection
# =============================================================================

def detect_peaks(
    result: HistogramResult,
    prominence_frac: float = 0.05,
    min_height_frac: float = 0.02,
    min_distance_ns: float = 1.0,
    smooth_sigma_bins: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Detect local maxima in a histogram.

    Parameters
    ----------
    prominence_frac   : minimum peak prominence as fraction of max(y).
                        Lower → picks up weaker peaks; raise to suppress noise.
    min_height_frac   : minimum peak height as fraction of max(y).
    min_distance_ns   : minimum separation between peaks in ns.
    smooth_sigma_bins : Gaussian smoothing sigma applied before detection only
                        (original counts are never modified).
    """
    x = result.bin_centres
    y = result.counts.astype(np.float64)

    if np.sum(y) == 0:
        return np.array([], dtype=int), {}

    y_smooth = gaussian_filter1d(y, sigma=smooth_sigma_bins) if smooth_sigma_bins > 0 else y

    bw                = float(np.median(np.diff(x)))
    min_distance_bins = max(1, int(round(min_distance_ns / bw)))

    peaks, props = find_peaks(
        y_smooth,
        prominence=prominence_frac * np.max(y_smooth),
        height=min_height_frac * np.max(y_smooth),
        distance=min_distance_bins,
    )

    # Sort spatially
    order = np.argsort(x[peaks])
    peaks = peaks[order]
    for k in props:
        if len(props[k]) == len(order):
            props[k] = props[k][order]

    return peaks, props


# =============================================================================
# Multi-peak simultaneous fit
# =============================================================================

def fit_multiple_peaks(
    result: HistogramResult,
    peak_indices: np.ndarray,
    fit_window: float = 3.0,
    sigma_max: float = 5.0,
) -> Optional[List[dict]]:
    """
    Simultaneously fit all detected peaks with sum(Gaussians) + background.

    Parameters
    ----------
    fit_window : half-width in ns of the fit region around the outermost peaks.
                 Should be ~10x your expected sigma, NOT the histogram x_range.
                 For 0.3 ns peaks, use fit_window ~ 2–5 ns.
    sigma_max  : hard upper bound on each Gaussian's sigma in ns.
                 Prevents physically absurd wide fits absorbing background.
                 For 0.3 ns peaks, use sigma_max ~ 1–2 ns.
    """
    x = result.bin_centres
    y = result.counts.astype(np.float64)

    if len(peak_indices) == 0:
        return None

    peak_positions = x[peak_indices]
    lo = float(np.min(peak_positions)) - fit_window
    hi = float(np.max(peak_positions)) + fit_window

    mask  = (x >= lo) & (x <= hi)
    x_fit = x[mask]
    y_fit = y[mask]

    nz    = y_fit > 0
    x_fit = x_fit[nz]
    y_fit = y_fit[nz]

    if len(x_fit) < 5:
        return None

    bw = float(np.median(np.diff(x)))

    p0, lower, upper = [], [], []
    for p in peak_indices:
        amp0   = max(float(y[p]), 1.0)
        mu0    = float(x[p])
        sigma0 = max(2 * bw, 0.05)
        p0    += [amp0, mu0, sigma0]
        lower += [0,    mu0 - fit_window, bw / 10]
        upper += [np.inf, mu0 + fit_window, sigma_max]  # sigma_max, NOT fit_window

    bg0 = float(np.median(y_fit))
    p0    += [bg0]
    lower += [0]
    upper += [np.inf]

    sigma_y = np.sqrt(np.maximum(y_fit, 1.0))

    try:
        popt, pcov = curve_fit(
            multi_gaussian_background,
            x_fit, y_fit,
            p0=p0, sigma=sigma_y,
            absolute_sigma=True,
            bounds=(lower, upper),
            maxfev=100_000,
        )
        perr    = np.sqrt(np.diag(pcov))
        y_model = multi_gaussian_background(x_fit, *popt)
        chi2    = float(np.sum(((y_fit - y_model) / sigma_y) ** 2))
        ndf     = len(y_fit) - len(popt)

        results = []
        for i in range(len(peak_indices)):
            results.append({
                "amplitude":     popt[3*i],     "amplitude_err": perr[3*i],
                "mu":            popt[3*i + 1], "mu_err":        perr[3*i + 1],
                "sigma":     abs(popt[3*i + 2]), "sigma_err":    perr[3*i + 2],
                "background":    popt[-1],       "background_err":perr[-1],
                "chi2": chi2, "ndf": ndf,
            })
        return results

    except Exception as e:
        print(f"  Multi-fit failed for channel: {e}")
        return None


# =============================================================================
# Multi-peak plotting
# — named plot_coincidence_histograms_multipeak to match user calls
# =============================================================================

# =============================================================================
# Duplicate-peak cleanup utilities  (now actually called in the plot function)
# =============================================================================

def merge_close_peaks(
    peaks: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    merge_distance_ns: float,
) -> np.ndarray:
    """
    Merge detected peak indices that are closer than `merge_distance_ns`.

    When two peaks are too close, keep the taller one. Applied *before*
    fitting so the fitter never receives duplicate seed positions.

    Parameters
    ----------
    peaks             : 1-D array of bin indices (will be sorted spatially).
    x                 : bin-centre array.
    y                 : counts array.
    merge_distance_ns : minimum allowed separation in ns between kept peaks.
    """
    if len(peaks) == 0:
        return peaks

    peaks  = list(peaks[np.argsort(x[peaks])])   # ensure spatial order
    merged = [peaks[0]]
    for p in peaks[1:]:
        prev = merged[-1]
        if abs(x[p] - x[prev]) < merge_distance_ns:
            if y[p] > y[prev]:
                merged[-1] = p
        else:
            merged.append(p)
    return np.array(merged, dtype=int)


def remove_duplicate_fits(
    fit_results: List[dict],
    mu_merge_distance_ns: float = 0.5,
) -> List[dict]:
    """
    Remove fitted Gaussians whose mu values are closer than `mu_merge_distance_ns`.

    Keeps the component with the lower reduced chi2/ndf. Applied *after* fitting.

    Parameters
    ----------
    fit_results          : list of fit-result dicts from `fit_multiple_peaks`.
    mu_merge_distance_ns : minimum allowed separation in ns between kept peaks.
    """
    if len(fit_results) <= 1:
        return fit_results

    sorted_fits = sorted(fit_results, key=lambda f: f["mu"])
    cleaned     = [sorted_fits[0]]

    for fr in sorted_fits[1:]:
        prev = cleaned[-1]
        if abs(fr["mu"] - prev["mu"]) < mu_merge_distance_ns:
            def redchi(f):
                ndf = f.get("ndf", 0)
                return f["chi2"] / ndf if ndf > 0 else np.inf
            if redchi(fr) < redchi(prev):
                cleaned[-1] = fr
        else:
            cleaned.append(fr)

    return cleaned

def _is_sane_fit(fr: dict, seed_mu: float, fit_window: float, sigma_max: float) -> bool:
    """Return False if a fit result looks pathological."""
    mu        = fr["mu"]
    sigma     = fr["sigma"]
    amplitude = fr["amplitude"]
    mu_err    = fr["mu_err"]
    sigma_err = fr["sigma_err"]

    if not np.isfinite(mu) or not np.isfinite(sigma):
        return False
    if abs(mu - seed_mu) > fit_window:          # drifted outside the fit window
        return False
    if sigma <= 0 or sigma > sigma_max:
        return False
    if amplitude <= 0:
        return False
    if not np.isfinite(mu_err) or mu_err > fit_window:   # error blew up
        return False
    if not mu_err > 0:
        return False
    if not np.isfinite(sigma_err) or sigma_err > sigma_max:
        return False
    return True

def plot_coincidence_histograms_multipeak(
    histograms: Dict[int, HistogramResult],
    pivot_channel: int,
    time_unit: str = "ns",
    prominence_frac: float = 0.08,
    min_height_frac: float = 0.02,
    min_distance_ns: float = 1.0,
    smooth_sigma_bins: float = 1.0,
    fit_window: float = 3.0,
    sigma_max: float = 2.0,
    merge_distance_ns: float = 0.5,
    plot_range: float = 20.0,
    max_peaks: Optional[int] = 4,
    logy: bool = False,
    llabel: str = "PPS2 Timing Preliminary",
    rlabel: str = "(H8 Test Beam, May 2025)",
    is_data: bool = True,
    min_counts_for_fit: int = 75,          # NEW — skip fit below this
) -> None:

    """
    Detect and fit multiple peaks per channel, then plot.

    Duplicate-peak cleanup is applied at two stages:
    1. Before fitting : `merge_close_peaks` removes seed positions that are
        too close so the fitter does not receive degenerate inputs.
    2. After fitting  : `remove_duplicate_fits` drops fitted Gaussians whose
        mu values ended up too close (keeping the better chi2/ndf).

    Parameters
    ----------
    prominence_frac   : peak prominence threshold as fraction of max(y).
    min_height_frac   : minimum peak height as fraction of max(y).
    min_distance_ns   : minimum separation for find_peaks (detection stage).
    smooth_sigma_bins : Gaussian smoothing for peak detection only.
    fit_window        : half-width in ns of the fit region (~10x expected sigma).
    sigma_max         : hard upper bound on Gaussian sigma in ns.
    merge_distance_ns : minimum separation (ns) enforced at both cleanup stages.
                        Peaks / fitted mu closer than this are merged/removed.
    plot_range        : half-width in ns of the displayed x-axis around peaks.
    max_peaks         : maximum peaks to pass to the fitter (strongest first).
    logy              : use log y-axis.
    
    min_counts_for_fit : channels whose total count is below this threshold are
                        plotted raw (no peak detection, no fit). This avoids
                        spurious fits on noise-dominated histograms.
    """
    channels = sorted(histograms.keys())
    n        = len(channels)
    ncols    = min(2, n)
    nrows    = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows),
                         constrained_layout=True)
    axes = np.array(axes).flatten()

    for ax, ch in zip(axes, channels):
        result  = histograms[ch]
        centres = result.bin_centres
        counts  = result.counts.astype(np.float64)
        bw      = result.bin_edges[1] - result.bin_edges[0]
        total   = int(np.sum(counts))

        ax.set_title(f"Ch {pivot_channel} → Ch {ch}",fontsize = 18)
        ax.set_xlabel(f"Δt ({time_unit})",fontsize = 18)
        ax.set_ylabel("Counts",fontsize = 18)
        ax.grid(alpha=0.3)
        hep.cms.label(llabel, data=is_data, rlabel=rlabel, loc=0, ax=ax, fontsize=(14, 12, 12, 11))


        # ── No data at all ────────────────────────────────────────────────────
        if total == 0:
            ax.text(0.5, 0.5, "No coincidences",
                    transform=ax.transAxes, ha="center", va="center", color="gray")
            continue

        # ── Too few counts: just plot raw histogram, no fit ───────────────────
        if total < min_counts_for_fit:
            peak_bin   = int(np.argmax(counts))
            data_center = float(centres[peak_bin])
            lo_vis = data_center - plot_range
            hi_vis = data_center + plot_range
            vis = (centres >= lo_vis) & (centres <= hi_vis)
            ax.bar(centres[vis], counts[vis], width=bw, alpha=0.75, align="center",
                    color="steelblue")
            ax.set_xlim(lo_vis, hi_vis)
            ax.text(
                0.97, 0.97,
                f"Only {total} counts — fit skipped\n"
                f"(need ≥ {min_counts_for_fit})",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
            )
            if logy:
                ax.set_yscale("log")
            continue

        # ── Normal path: peak detection + fit ────────────────────────────────
        peaks, _ = detect_peaks(
            result,
            prominence_frac=prominence_frac,
            min_height_frac=min_height_frac,
            min_distance_ns=min_distance_ns,
            smooth_sigma_bins=smooth_sigma_bins,
        )

        if len(peaks) > 0:
            peaks = peaks[np.argsort(counts[peaks])[::-1]]
        if max_peaks is not None:
            peaks = peaks[:max_peaks]
        if len(peaks) > 0:
            peaks = peaks[np.argsort(centres[peaks])]

        if len(peaks) > 0:
            peaks = merge_close_peaks(peaks, centres, counts, merge_distance_ns)

        fit_results = fit_multiple_peaks(
            result, peaks, fit_window=fit_window, sigma_max=sigma_max
        )

        if fit_results is not None and len(fit_results) > 1:
            fit_results = remove_duplicate_fits(fit_results, merge_distance_ns)

        # ── Filter fit results against their seed peaks ───────────────────────
# Replace the seed-pairing block in the plotting loop
        if fit_results is not None:
            anchor_bin = int(np.argmax(counts))
            anchor     = float(centres[anchor_bin])
            seed_positions = list(centres[peaks]) if len(peaks) > 0 else [anchor]
            sane = []
            for fr in fit_results:
                nearest_seed = min(seed_positions, key=lambda s: abs(s - fr["mu"]))
                if _is_sane_fit(fr, nearest_seed, fit_window, sigma_max):
                    sane.append(fr)

            if len(sane) < len(fit_results):
                n_dropped = len(fit_results) - len(sane)
                ax.text(
                    0.03, 0.97,
                    f"[!] {n_dropped} bad fit(s) removed",   # was ⚠,
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    color="darkorange",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
                )
            fit_results = sane if sane else None

        # ── Display range: anchor to *seed peaks*, not fit mus ───────────────
        if fit_results is not None:
            mus = [fr["mu"] for fr in fit_results]
        elif len(peaks) > 0:
            mus = list(centres[peaks])
        else:
            mus = [float(centres[np.argmax(counts)])]

        # Hard-clip: never let a single outlier mu stretch the window beyond
        # plot_range from the dominant peak (max count seed or fit)
        anchor = float(centres[np.argmax(counts)])
        mus    = [m for m in mus if abs(m - anchor) <= 2 * plot_range]
        if not mus:
            mus = [anchor]

        lo_vis = min(mus) - plot_range
        hi_vis = max(mus) + plot_range

        vis = (centres >= lo_vis) & (centres <= hi_vis)
        ax.bar(centres[vis], counts[vis], width=bw, alpha=0.75, align="center",
                color="steelblue")

        if fit_results is not None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(fit_results)))
            xd     = np.linspace(lo_vis, hi_vis, 4000)

            total_params = []
            for fr in fit_results:
                total_params += [fr["amplitude"], fr["mu"], fr["sigma"]]
            total_params.append(fit_results[0]["background"])
            ax.plot(
                xd, multi_gaussian_background(xd, *total_params),
                color="black", lw=2, label="Total fit",
            )

            annotation_lines = []
            for i, (fr, color) in enumerate(zip(fit_results, colors), start=1):
                mu_in_view = lo_vis <= fr["mu"] <= hi_vis
                if mu_in_view:
                    ax.axvline(fr["mu"], color=color, alpha=0.35)
                    ax.text(
                        fr["mu"], fr["amplitude"] + fr["background"],
                        str(i), fontsize=8, color=color, ha="center",
                    )
                annotation_lines.append(
                    f"{i}: μ = {fr['mu']:.5f} ± {fr['mu_err']:.5f} {time_unit}\n"
                    f"   σ = {fr['sigma']:.3f} ± {fr['sigma_err']:.3f} {time_unit}"
                    + ("" if mu_in_view else " [off-axis]")
                )

            ndf      = fit_results[0]["ndf"]
            chi2_red = fit_results[0]["chi2"] / ndf if ndf > 0 else float("nan")
            annotation_lines.append(
                f"\nχ²/ndf = {fit_results[0]['chi2']:.1f}/{ndf} = {chi2_red:.2f}"
            )
            ax.text(
                0.97, 0.97, "\n".join(annotation_lines),
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )
            ax.legend(fontsize=10, loc="upper left")

        ax.set_xlim(lo_vis, hi_vis)
        if logy:
            ax.set_yscale("log")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Coincidence timing — pivot channel {pivot_channel}", fontsize=14)
    plt.show()


# Keep the old name as an alias so existing notebooks don't break
plot_coincidence_histograms_multifit = plot_coincidence_histograms_multipeak


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":

    PARQUET_FILE  = "your_sorted_file.parquet"
    PIVOT_CHANNEL = 8

    histograms = build_coincidence_histograms(
        parquet_path  = PARQUET_FILE,
        pivot_channel = PIVOT_CHANNEL,
        x_range       = 1_000,
        bin_width     = 0.05,
        use_cfd50     = True,
        cfd50_path    = None,   # path to precomputed CFD50 parquet

    )

    print("\nCoincidence summary:\n")
    for ch, result in histograms.items():
        print(f"  Ch {PIVOT_CHANNEL} → Ch {ch}: {result.total_counts} counts")

    plot_coincidence_histograms_multipeak(
        histograms,
        pivot_channel     = PIVOT_CHANNEL,
        time_unit         = "ns",
        prominence_frac   = 0.08,
        min_height_frac   = 0.01,
        min_distance_ns   = 0.5,
        smooth_sigma_bins = 1.0,
        fit_window        = 3.0,          # ~10x expected sigma of 0.3 ns
        sigma_max         = 2.0,          # no Gaussian wider than 2 ns
        merge_distance_ns = 0.5,          # merge peaks/fits closer than 0.5 ns
        plot_range        = 20.0,
        max_peaks         = 4,
    )