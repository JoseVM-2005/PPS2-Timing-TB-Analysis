from __future__ import annotations

import os
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pyarrow as pa
from scipy.optimize import curve_fit
from scipy.stats import moyal
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sampiclyser.sampic_tools import reorder_circular_samples_with_trigger, open_hit_reader
import mplhep as hep
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Optional, Tuple
from pathlib import Path
from scipy.signal import find_peaks
from collections import defaultdict

from scipy.stats import exponnorm, moyal

try:
    import mplhep as hep  # user may have a local alias
except Exception:
    try:
        import mplhep as hep
    except Exception:
        hep = None
# =============================================================================
# Histogram container
# =============================================================================
@dataclass
class HistogramResult:
    counts: np.ndarray
    bin_edges: np.ndarray
    @property
    def bin_centres(self):
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

# =============================================================================
# Landau ⊗ Gaussian
# =============================================================================
def langauss(x, mpv, eta, sigma, A, B):
    z = np.linspace(-5*sigma, 5*sigma, 121)
    gauss = np.exp(-0.5 * (z/sigma)**2)
    gauss /= np.trapezoid(gauss, z)
    xx = x[:, None] - z[None, :]
    landau = moyal.pdf(xx, loc=mpv, scale=eta)
    conv = np.trapezoid(landau * gauss, z, axis=1)
    return A * conv + B

def gaussian(x, A, mu, sigma, B):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + B

def snr_composite_model(x, mpv, eta, sigma_lg, A_lg, mu_n, sigma_n, A_n):
    return langauss(x, mpv, eta, sigma_lg, A_lg, 0.0) + gaussian(x, A_n, mu_n, sigma_n, 0.0)

def snr_langaus_model(x, mpv, eta, sigma_lg, A_lg):
    return langauss(x, mpv, eta, sigma_lg, A_lg, 0.0)

# =============================================================================
# Waveform baseline
# =============================================================================

def compute_baseline(
    samp_arr: np.ndarray,
    trig_arr: np.ndarray,
    n_baseline_samples: int = 20,
) -> float:
    """Compute the waveform baseline from the first ordered samples.

    The circular SAMPIC waveform is first put into the same trigger-defined
    order used by the timing calculations. The baseline is then the arithmetic
    mean of the first ``n_baseline_samples`` ordered samples.
    """
    _, samp_ord, _ = reorder_circular_samples_with_trigger(
        trig_arr, samp_arr, reorder_samples=False,
    )
    samp_ord = np.asarray(samp_ord, dtype=np.float64)
    n_use = min(int(n_baseline_samples), len(samp_ord))
    if n_use <= 0:
        return np.nan
    region = samp_ord[:n_use]
    if not np.all(np.isfinite(region)):
        return np.nan
    return float(np.mean(region))


def compute_and_save_baseline_database(
    parquet_path: str,
    output_path: Optional[str] = None,
    n_baseline_samples: int = 20,
    batch_size: int = 100_000,
    root_tree: str = "sampic_hits",
) -> str:
    """Build a HITNumber -> waveform baseline lookup database.

    The baseline is calculated from the first ``n_baseline_samples`` ordered
    waveform samples, rather than using the pre-existing ``Baseline`` column
    in the input parquet file.
    """
    if output_path is None:
        p = Path(parquet_path)
        output_path = str(p.with_stem(p.stem + "_baseline"))

    parquet_path = Path(parquet_path)
    output_path = Path(output_path)

    cols = ["HITNumber", "DataSample", "TriggerPosition"]
    batches = open_hit_reader(
        parquet_path, cols=cols, batch_size=batch_size, root_tree=root_tree
    )

    hit_ids_out = []
    baselines_out = []
    total = 0
    failed = 0

    for batch in tqdm(batches, desc="Building baseline database"):
        hit_ids = batch["HITNumber"].to_numpy().astype(np.int64)
        samples = batch["DataSample"]
        triggers = batch["TriggerPosition"]

        for i in tqdm(range(len(hit_ids)), leave=False, desc="Hits",
                      mininterval=0.1, dynamic_ncols=False):
            total += 1
            samp_arr = np.asarray(samples[i].as_py(), dtype=np.float64)
            trig_arr = np.asarray(triggers[i].as_py(), dtype=np.int32)

            baseline = compute_baseline(
                samp_arr, trig_arr, n_baseline_samples
            )

            if not np.isfinite(baseline):
                failed += 1

            hit_ids_out.append(int(hit_ids[i]))
            baselines_out.append(baseline)

    table = pa.table({
        "HITNumber": pa.array(hit_ids_out, type=pa.int64()),
        "Baseline": pa.array(baselines_out, type=pa.float64()),
    })
    pq.write_table(table, output_path)

    print(
        f"Baseline failures: {failed:,} / {total:,} "
        f"({100 * failed / max(total, 1):.2f}%)"
    )
    print(
        f"Saved baseline lookup to: {output_path} "
        f"(first {n_baseline_samples} ordered samples)"
    )

    return str(output_path)


def load_baseline_lookup(baseline_path: str) -> Dict[int, float]:
    """Load a baseline lookup parquet into HITNumber -> Baseline."""
    table = pq.read_table(baseline_path, columns=["HITNumber", "Baseline"])
    hit_ids = table["HITNumber"].combine_chunks().to_numpy()
    baselines = table["Baseline"].combine_chunks().to_numpy(
        zero_copy_only=False
    )
    return dict(zip(hit_ids.tolist(), baselines.tolist()))


def get_hit_baseline(
    hit_id: int,
    samp_arr: Optional[np.ndarray] = None,
    trig_arr: Optional[np.ndarray] = None,
    baseline_lookup: Optional[Dict[int, float]] = None,
    n_baseline_samples: int = 20,
) -> float:
    """Get a corrected baseline from the lookup, or calculate it on demand."""
    if baseline_lookup is not None:
        try:
            return float(baseline_lookup[int(hit_id)])
        except KeyError:
            pass

    if samp_arr is None or trig_arr is None:
        raise KeyError(
            f"No baseline for HITNumber={hit_id} and no waveform was supplied."
        )

    return compute_baseline(samp_arr, trig_arr, n_baseline_samples)


# =============================================================================
# CFD50
# =============================================================================

def compute_cfd50(
    samp_arr: np.ndarray,
    trig_arr: np.ndarray,
    baseline: float,
    period: float,
    min_amplitude: float = 0.01,      # V — reject noise spikes
    edge_buffer_samples: int = 10,     # samples — peak must not be too close to buffer edge
) -> Optional[float]:
    
    """
    Compute the CFD50 time offset from the start of the ordered waveform.

    Reorders the circular buffer, finds the rising-edge crossing of the
    50% amplitude level, and interpolates linearly between the two
    bracketing samples.

    Parameters
    ----------
    samp_arr : ndarray of float, shape (N,)
        Raw ADC sample values.
    trig_arr : ndarray of {0,1}, shape (N,)
        Trigger markers.
    baseline : float
        Baseline offset (same units as samp_arr).
    period : float
        Time between samples in ns.

    Returns
    -------
    float or None
        Time offset in ns from sample 0 of the ordered buffer to the
        50% crossing, or None if no valid rising-edge crossing is found.
    """

    trig_ord, samp_ord, _ = reorder_circular_samples_with_trigger(
        trig_arr, samp_arr, reorder_samples=False
    )
    samp_ord = np.asarray(samp_ord, dtype=np.float64)
    n = len(samp_ord)

    peak_idx = int(np.argmax(samp_ord))
    peak_val = float(samp_ord[peak_idx])

    # Reject noise: peak must clear baseline by min_amplitude
    if peak_val - baseline <= min_amplitude:
        return None

    # Reject buffer-edge artifacts: peak must be away from both edges
    if peak_idx < edge_buffer_samples or peak_idx > n - edge_buffer_samples:
        return None

    half_level = baseline + 0.5 * (peak_val - baseline)

    for i in range(peak_idx):
        y0 = float(samp_ord[i])
        y1 = float(samp_ord[i + 1])
        if y0 <= half_level <= y1:
            dy = y1 - y0
            if dy <= 0:
                return None
            frac = (half_level - y0) / dy
            return (i + frac) * period

    return None


def compute_and_save_cfd50(
    parquet_path: str,
    output_path: Optional[str] = None,
    period: float = 1 / 6.4,
    batch_size: int = 100_000,
    root_tree: str = "sampic_hits",
    min_amplitude: float = 0.01,      # V — reject noise spikes
    edge_buffer_samples: int = 10,     # samples — peak must not be too close to buffer edge
    baseline_path: Optional[str] = None,
    n_baseline_samples: int = 20,

) -> np.ndarray:
    """
    Compute CFD50 offsets for every hit in a parquet file and save as a
    lightweight lookup parquet with two columns: HITNumber, CFD50Offset.

    HITs where CFD50 fails get NaN — callers can decide whether to drop
    or fall back to OrderedCell0Time.

    Parameters
    ----------
    output_path : path for the output parquet; if None, uses
                  parquet_path with '_cfd50' suffix.

    Returns
    -------
    np.ndarray of float64, shape (N,)
        CFD50 offsets in ns, indexed by file row order.
        NaN for failed hits.
    """
    if output_path is None:
        p = Path(parquet_path)
        output_path = str(p.with_stem(p.stem + "_cfd50"))

    baseline_lookup = load_baseline_lookup(baseline_path) if baseline_path is not None else None

    cols    = ["HITNumber", "DataSample", "TriggerPosition", "Channel"]
    batches = open_hit_reader(parquet_path, cols=cols,
                              batch_size=batch_size, root_tree=root_tree)

    all_hit_ids = []
    all_offsets = []
    failed      = 0
    total       = 0

    fail_ch  = defaultdict(int)
    total_ch = defaultdict(int)


    for batch in tqdm(batches, desc="Computing CFD50"):
        hit_ids   = batch["HITNumber"].to_numpy()
        samples   = batch["DataSample"]
        triggers  = batch["TriggerPosition"]

        for i in tqdm(range(len(hit_ids)), desc="CFD50", leave=False):

            ch = int(batch["Channel"][i])
            total += 1
            total_ch[ch] += 1

            samp_arr = np.asarray(samples[i].as_py(), dtype=np.float64)
            trig_arr = np.asarray(triggers[i].as_py(), dtype=np.int32)
            hit_id = int(hit_ids[i])
            baseline = (
                float(baseline_lookup[hit_id])
                if baseline_lookup is not None and hit_id in baseline_lookup
                else compute_baseline(samp_arr, trig_arr, n_baseline_samples)
            )

            offset = compute_cfd50(
                samp_arr,
                trig_arr,
                baseline,
                period,
                min_amplitude,
                edge_buffer_samples,     # samples — peak must not be too close to buffer edge

            )
            all_hit_ids.append(int(hit_ids[i]))

            if offset is None:
                all_offsets.append(np.nan)
                failed += 1
                fail_ch[ch] += 1
            else:
                all_offsets.append(offset)


    print(f"CFD50 failures: {failed:,} / {total:,} ({100*failed/max(total,1):.2f}%)")
    print("\nPer‑channel CFD50 statistics:")
    for ch in sorted(total_ch):
        tot = total_ch[ch]
        fail = fail_ch[ch]
        print(
            f"  Channel {ch}: {tot} hits | "
            f"CFD50 fails = {fail} ({100*fail/max(tot,1):.2f}%)"
        )

    # Save as a two-column parquet — tiny file, fast to load
    lut = pa.table({
        "HITNumber":   pa.array(all_hit_ids, type=pa.int64()),
        "CFD50Offset": pa.array(all_offsets, type=pa.float64()),
    })
    pq.write_table(lut, output_path)
    print(f"Saved CFD50 lookup to: {output_path}")

    return np.array(all_offsets, dtype=np.float64)


def load_cfd50_lookup(cfd50_path: str) -> Dict[int, float]:
    """
    Load a CFD50 lookup parquet into a HITNumber → offset dict.
    NaN offsets (failed hits) are excluded.
    """
    table    = pq.read_table(cfd50_path, columns=["HITNumber", "CFD50Offset"])
    hit_ids  = table["HITNumber"].combine_chunks().to_numpy()
    offsets  = table["CFD50Offset"].combine_chunks().to_numpy(zero_copy_only=False)
    valid    = np.isfinite(offsets)
    return dict(zip(hit_ids[valid].tolist(), offsets[valid].tolist()))



# =============================================================================
# =============================================================================
# CFD CONFIG
# =============================================================================

CFD_LEVELS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]


def compute_cfd_times_and_slopes(
    samp_arr: np.ndarray,
    trig_arr: np.ndarray,
    baseline: float,
    period: float,
    cfd_levels: List[int] = CFD_LEVELS,
    min_amplitude: float = 0.01,
    edge_buffer_samples: int = 10,
    min_slope: float = 0.0,
    consecutive_points: int = 2,
) -> tuple[dict, dict]:
    """
    Compute CFD crossing times and slopes for multiple CFD levels.

    Returns
    -------
    cfd_offsets : dict  {10: time_ns, 20: time_ns, ...}
    cfd_slopes  : dict  slope at crossing in V/ns
    """

    _, samp_ord, _ = reorder_circular_samples_with_trigger(
        trig_arr,
        samp_arr,
        reorder_samples=False,
    )

    y = np.asarray(samp_ord, dtype=np.float64)
    n = len(y)

    peak_idx = int(np.argmax(y))
    peak_val = float(y[peak_idx])
    amplitude = peak_val - baseline

    cfd_offsets = {k: np.nan for k in cfd_levels}
    cfd_slopes  = {k: np.nan for k in cfd_levels}

    # -------------------------------------------------------------------------
    # Basic quality cuts
    # -------------------------------------------------------------------------
    if amplitude <= min_amplitude:
        return cfd_offsets, cfd_slopes

    if peak_idx < edge_buffer_samples or peak_idx > n - edge_buffer_samples:
        return cfd_offsets, cfd_slopes

    # Rising edge only (inclusive of peak so highest CFD levels can be reached)
    rising = y[: peak_idx + 1]
    n_rising = len(rising)

    # -------------------------------------------------------------------------
    # CFD loop
    # -------------------------------------------------------------------------
    for frac_percent in cfd_levels:
        frac = frac_percent / 100.0
        threshold = baseline + frac * amplitude

        # Search window: need `consecutive_points` samples on each side
        # Upper bound ensures `above` slice is always full length
        search_end = n_rising - consecutive_points
        for i in range(consecutive_points, search_end):
            below = rising[i - consecutive_points : i]
            above = rising[i : i + consecutive_points]

            if np.all(below < threshold) and np.all(above >= threshold):
                # Linear interpolation between the last-below and first-above
                y0 = float(rising[i - 1])
                y1 = float(rising[i])
                dy = y1 - y0

                if dy <= min_slope:
                    continue  # try next candidate (rare: flat region at threshold)

                frac_interp = (threshold - y0) / dy
                t_cross = ((i - 1) + frac_interp) * period   # ns
                slope   = dy / period                          # V/ns

                cfd_offsets[frac_percent] = t_cross
                cfd_slopes[frac_percent]  = slope
                break  # take first valid crossing

    return cfd_offsets, cfd_slopes


# =============================================================================
# CFD DATABASE BUILDER
# =============================================================================

def compute_and_save_cfd_database(
    parquet_path: str,
    output_path: Optional[str] = None,
    period: float = 1 / 6.4,        # ns per sample
    batch_size: int = 100_000,
    root_tree: str = "sampic_hits",
    min_amplitude: float = 0.01,
    edge_buffer_samples: int = 10,
    consecutive_points: int = 2,
    baseline_path: Optional[str] = None,
    n_baseline_samples: int = 20,
) -> str:
    """
    Build a CFD database containing:
        HITNumber
        CFD{10..90}Offset   (time of threshold crossing, ns)
        CFD{10..90}Slope    (slope at crossing, V/ns)
    """

    if output_path is None:
        p = Path(parquet_path)
        output_path = str(p.with_stem(p.stem + "_cfddb"))
    
    parquet_path = Path(parquet_path)
    output_path = Path(output_path)
    baseline_lookup = load_baseline_lookup(baseline_path) if baseline_path is not None else None

    cols = ["HITNumber", "DataSample", "TriggerPosition"]

    batches = open_hit_reader(
        parquet_path,
        cols=cols,
        batch_size=batch_size,
        root_tree=root_tree,
    )

    # -------------------------------------------------------------------------
    # Storage + per-level failure counters
    # -------------------------------------------------------------------------
    out: dict[str, list] = {"HITNumber": []}
    for k in CFD_LEVELS:
        out[f"CFD{k}Offset"] = []
        out[f"CFD{k}Slope"]  = []

    total = 0
    failures: dict[int, int] = {k: 0 for k in CFD_LEVELS}

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    for batch in tqdm(batches, desc="Building CFD database"):
        hit_ids  = batch["HITNumber"].to_numpy().astype(np.int64)
        samples_col  = batch["DataSample"]
        triggers_col = batch["TriggerPosition"]

        for i in tqdm(range(len(hit_ids)), leave=False, desc="Hits", mininterval=0.1, dynamic_ncols=False):
            total += 1
            hit_id   = int(hit_ids[i])
            samp_arr = np.asarray(samples_col[i].as_py(),  dtype=np.float64)
            trig_arr = np.asarray(triggers_col[i].as_py(), dtype=np.int32)

            cfd_offsets, cfd_slopes = compute_cfd_times_and_slopes(
                samp_arr=samp_arr,
                trig_arr=trig_arr,
                baseline=(
                    float(baseline_lookup[hit_id])
                    if baseline_lookup is not None and hit_id in baseline_lookup
                    else compute_baseline(samp_arr, trig_arr, n_baseline_samples)
                ),
                period=period,
                cfd_levels=CFD_LEVELS,
                min_amplitude=min_amplitude,
                edge_buffer_samples=edge_buffer_samples,
                consecutive_points=consecutive_points,
            )

            out["HITNumber"].append(hit_id)
            for k in CFD_LEVELS:
                out[f"CFD{k}Offset"].append(cfd_offsets[k])
                out[f"CFD{k}Slope"].append(cfd_slopes[k])
                if np.isnan(cfd_offsets[k]):
                    failures[k] += 1

    # -------------------------------------------------------------------------
    # Failure summary (once, after all hits)
    # -------------------------------------------------------------------------
    print("\nCFD crossing failures:")
    for k in CFD_LEVELS:
        pct = 100 * failures[k] / max(total, 1)
        print(f"  CFD{k:2d}: {failures[k]:>8,} / {total:,}  ({pct:.2f}%)")

    # -------------------------------------------------------------------------
    # Write Arrow / Parquet
    # -------------------------------------------------------------------------
    table_dict: dict[str, pa.Array] = {
        "HITNumber": pa.array(np.asarray(out["HITNumber"], dtype=np.int64), type=pa.int64())
    }
    for k in CFD_LEVELS:
        table_dict[f"CFD{k}Offset"] = pa.array(
            np.asarray(out[f"CFD{k}Offset"], dtype=np.float32), type=pa.float32()
        )
        table_dict[f"CFD{k}Slope"] = pa.array(
            np.asarray(out[f"CFD{k}Slope"], dtype=np.float32), type=pa.float32()
        )

    table = pa.table(table_dict)
    pq.write_table(table, output_path, compression="zstd")

    print("\n============================================================")
    print(f"CFD database written to: {output_path}")
    print(f"Total hits : {total:,}")
    print("Columns    :", ", ".join(table.column_names))
    print("============================================================\n")

    return output_path


# =============================================================================
# CFD LOOKUP LOADER
# =============================================================================

def load_cfd_lookup(
    cfd_database_path: str,
):
    """
    Returns:

        {
            HITNumber: {
                "CFD10Offset": ...,
                "CFD10Slope": ...,
                ...
            }
        }
    """

    table = pq.read_table(cfd_database_path)

    data = {}

    hit_ids = (
        table["HITNumber"]
        .combine_chunks()
        .to_numpy()
    )

    arrays = {}

    for k in CFD_LEVELS:

        arrays[f"CFD{k}Offset"] = (
            table[f"CFD{k}Offset"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )

        arrays[f"CFD{k}Slope"] = (
            table[f"CFD{k}Slope"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )

    for i, hit_id in enumerate(hit_ids):

        row = {}

        for k in CFD_LEVELS:

            row[f"CFD{k}Offset"] = arrays[f"CFD{k}Offset"][i]
            row[f"CFD{k}Slope"]  = arrays[f"CFD{k}Slope"][i]

        data[int(hit_id)] = row

    return data

def _robust_langauss_seeds(x_fit, y_fit, mpv0):
    """
    Estimate eta and sigma from the LEFT-side HWHM of the peak
    located near mpv0.

    Important:
    Do NOT use np.argmax(y_fit) globally, because the signal peak
    may be smaller than a noise peak.
    """

    x_fit = np.asarray(x_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)

    if len(x_fit) < 3:
        raise ValueError("Not enough points for Landau-Gauss seeds")

    # Index corresponding to the selected signal MPV
    peak_idx = int(np.argmin(np.abs(x_fit - mpv0)))

    peak_val = float(y_fit[peak_idx])

    if peak_val <= 0:
        raise ValueError("Selected peak has non-positive height")

    half_max = 0.5 * peak_val

    # Only inspect the LEFT side of the selected peak
    left_x = x_fit[:peak_idx + 1]
    left_y = y_fit[:peak_idx + 1]

    crossings = np.where(left_y <= half_max)[0]

    if len(crossings) > 0:
        hwhm = mpv0 - left_x[crossings[-1]]
    else:
        # Fallback based on the available left-side range
        if peak_idx > 0:
            hwhm = mpv0 - left_x[0]
        else:
            hwhm = (x_fit[-1] - x_fit[0]) * 0.1

    hwhm = max(hwhm, 1e-4)

    eta0 = max(hwhm * 0.5, 1e-4)
    sigma0 = max(hwhm * 0.3, 1e-5)

    return eta0, sigma0, hwhm


# =============================================================================
# Main plotting function
# =============================================================================

def plot_Amplitude_histograms(
    parquet_path: str,
    Run: str,
    channel_col: str = "Channel",
    baseline_col: str = "Baseline",
    peak_col: str = "RawPeak",
    n_bins: int = 200,
    x_range: Optional[tuple[float, float]] = None,
    channel_filter: Optional[list[int]] = None,
    figsize_per_plot: tuple[float, float] = (8, 5),

    # Kept for backwards compatibility
    fit_window: Optional[float] = None,

    # Manual cut, if desired
    Amplitude_cuts: Optional[dict[int, float]] = None,
    Amplitude_signal_guesses: Optional[dict[int, float]] = None,
    signal_guess_window: float = 0.1,

    # Kept for backwards compatibility, but no longer used.
    # The cut is now always 10% of the fitted Landau-Gauss maximum.
    mpv_fraction: float = 0.10,

    logy: bool = False,
    label: str = "PPS2 Timing Preliminary",
    rlabel: str = "(H8 Test Beam, May 2025)",
    is_data: bool = True,

    # -------------------------------------------------------------------------
    # Peak-selection parameters
    # -------------------------------------------------------------------------

    # Fraction of the highest smoothed peak that a peak must reach
    # to be considered significant.
    min_peak_height_fraction: float = 0.02,

    # Fraction of the highest smoothed peak used for prominence.
    min_peak_prominence_fraction: float = 0.01,

    # Minimum separation between candidate peaks in Amplitude.
    min_peak_distance: float = 0.01,
    baseline_path: Optional[str] = None,
    n_baseline_samples: int = 20,

) -> tuple[dict, dict]:

    """
    Build and plot per-channel Amplitude histograms.

    Signal identification
    ----------------------

    The Amplitude distribution can contain both noise and real detector
    signal.

    The signal is selected according to the following rule:

        1. Smooth the histogram.
        2. Find significant peaks.
        3. If there is one significant peak, use it as the signal peak.
        4. If there are multiple significant peaks, use the RIGHTMOST
           significant peak as the signal peak.

    This deliberately does NOT assume that the highest peak is the signal.

    Therefore, a noise population can contain substantially more events
    than the signal population without determining the signal MPV.

    The selected peak is then fitted with Landau ⊗ Gaussian.

    The Amplitude cut is defined as the LEFT-side crossing of:

        10% × maximum(Landau ⊗ Gaussian)

    The noise population is not included when calculating this cut.

    Parameters
    ----------
    fit_window:
        Optional fixed right-side fit extent in Amplitude units.

        If None, the fit extends to:

            MPV + 3 × HWHM

        where HWHM is estimated from the left side of the selected peak.

    Amplitude_cut:
        Optional manual cut. If supplied, this value is used instead
        of the automatically determined 10% crossing.

    mpv_fraction:
        Retained for backwards compatibility. It is no longer used for
        the automatic cut. The automatic cut is always 10% of the
        fitted Landau-Gaussian maximum.

    min_peak_height_fraction:
        Minimum peak height relative to the highest smoothed peak.

    min_peak_prominence_fraction:
        Minimum peak prominence relative to the highest smoothed peak.

    min_peak_distance:
        Minimum distance between candidate peaks in Amplitude.
    """

    baseline_lookup = load_baseline_lookup(baseline_path) if baseline_path is not None else None

    # =========================================================================
    # Load waveform / peak columns
    # =========================================================================

    table = pq.read_table(
        parquet_path,
        columns=["HITNumber", channel_col, peak_col, "DataSample", "TriggerPosition"]
    )

    channels = (
        table[channel_col]
        .combine_chunks()
        .to_numpy()
    )

    hit_ids = table["HITNumber"].combine_chunks().to_numpy().astype(np.int64)
    samples_col = table["DataSample"]
    triggers_col = table["TriggerPosition"]

    baselines = np.array([
        (float(baseline_lookup[int(hit_ids[i])])
         if baseline_lookup is not None and int(hit_ids[i]) in baseline_lookup
         else compute_baseline(
             np.asarray(samples_col[i].as_py(), dtype=np.float64),
             np.asarray(triggers_col[i].as_py(), dtype=np.int32),
             n_baseline_samples,
         ))
        for i in range(len(hit_ids))
    ], dtype=np.float64)

    peaks_arr = (
        table[peak_col]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype(np.float64)
    )

    del table

    # =========================================================================
    # Calculate Amplitude
    # =========================================================================

    valid = baselines != 0

    Amplitudes = np.where(
        valid,
        (peaks_arr - baselines) ,#/ baselines,
        np.nan
    )

    unique_channels = sorted(
        np.unique(channels).astype(int)
    )

    if channel_filter is not None:
        unique_channels = [
            ch for ch in unique_channels
            if ch in channel_filter
        ]

    # =========================================================================
    # Failure counters
    # =========================================================================

    fail_baseline_zero = defaultdict(int)
    fail_nonfinite = defaultdict(int)
    fail_no_entries = defaultdict(int)
    fail_fit = defaultdict(int)
    total_ch = defaultdict(int)

    # =========================================================================
    # Build histograms
    # =========================================================================

    results = {}

    for ch in unique_channels:

        mask = channels == ch

        total_ch[ch] = int(mask.sum())

        fail_baseline_zero[ch] = int(
            ((channels == ch) & (~valid)).sum()
        )

        values = Amplitudes[mask]

        n_nonfinite = int(
            np.sum(~np.isfinite(values))
        )

        fail_nonfinite[ch] = n_nonfinite

        values = values[np.isfinite(values)]

        if len(values) == 0:
            fail_no_entries[ch] += 1
            continue

        lo, hi = (
            x_range
            if x_range is not None
            else (
                float(values.min()),
                float(values.max())
            )
        )

        counts, bin_edges = np.histogram(
            values,
            bins=n_bins,
            range=(lo, hi)
        )

        results[ch] = HistogramResult(
            counts=counts,
            bin_edges=bin_edges
        )

    # =========================================================================
    # Plot layout
    # =========================================================================

    n = len(results)

    if n == 0:
        print("No valid channels found.")
        return results, {}

    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            figsize_per_plot[0] * ncols,
            figsize_per_plot[1] * nrows
        ),
    )

    axes = np.atleast_1d(axes).flatten()

    thresholds = {}

    # =========================================================================
    # Channel loop
    # =========================================================================

    for ax, ch in tqdm(zip(
        axes,
        sorted(results))
    ):

        result = results[ch]

        centres = result.bin_centres
        counts = result.counts.astype(float)

        bw = (
            result.bin_edges[1]
            - result.bin_edges[0]
        )

        # ---------------------------------------------------------------------
        # Histogram
        # ---------------------------------------------------------------------

        ax.bar(
            centres,
            counts,
            width=bw,
            align="center",
            alpha=0.7,
            label="Data"
        )

        # ---------------------------------------------------------------------
        # Only non-empty bins for fitting
        # ---------------------------------------------------------------------

        fit_mask = counts > 0

        x_fit_full = centres[fit_mask]
        y_fit_full = counts[fit_mask]

        if len(x_fit_full) < 10:

            ax.set_title(
                f"Channel {ch} — too few entries"
            )

            thresholds[ch] = np.nan

            continue

        # =========================================================================
        # STEP 1 — Smooth histogram
        # =========================================================================

        y_smooth = gaussian_filter1d(
            counts.astype(float),
            sigma=2
        )

        max_smooth = float(
            np.max(y_smooth)
        )

        # =========================================================================
        # STEP 2 — Find significant peaks
        # =========================================================================

        min_prominence = max(
            max_smooth * min_peak_prominence_fraction,
            1.0
        )

        min_height = max(
            max_smooth * min_peak_height_fraction,
            1.0
        )

        peak_distance_bins = max(
            1,
            int(
                min_peak_distance / bw
            )
        )

        peaks_idx, peak_properties = find_peaks(
            y_smooth,
            prominence=min_prominence,
            height=min_height,
            distance=peak_distance_bins,
        )

        # =========================================================================
        # STEP 3 — Select signal peak
        # =========================================================================

        signal_guess_used = False

        if (
            Amplitude_signal_guesses is not None
            and ch in Amplitude_signal_guesses
        ):

            # ---------------------------------------------------------------------
            # Manual signal-location hint
            #
            # We don't force the peak to be exactly at the supplied value.
            # Instead, search for the local maximum around that value.
            # ---------------------------------------------------------------------

            guess = float(
                Amplitude_signal_guesses[ch]
            )

            guess_mask = (
                (centres >= guess - signal_guess_window)
                &
                (centres <= guess + signal_guess_window)
            )

            guess_indices = np.where(guess_mask)[0]

            if len(guess_indices) > 0:

                local_idx = guess_indices[
                    np.argmax(
                        y_smooth[guess_indices]
                    )
                ]

                signal_peak_idx = int(local_idx)

                signal_guess_used = True

            else:

                print(
                    f"  Channel {ch}: "
                    f"signal guess {guess:.4f} is outside "
                    f"histogram range"
                )

                signal_peak_idx = None

        else:

            signal_peak_idx = None


        # =========================================================================
        # Automatic peak finding if no manual hint was supplied
        # =========================================================================

        if signal_peak_idx is None:

            if len(peaks_idx) == 0:

                print(
                    f"  Channel {ch}: "
                    f"NO SIGNIFICANT PEAK"
                )

                thresholds[ch] = np.nan

                # ... existing no-signal handling ...

                continue

            # Rightmost significant peak
            signal_peak_idx = int(
                peaks_idx[-1]
            )


        mpv0 = float(
            centres[signal_peak_idx]
        )

        print(
            f"  Channel {ch}: "
            f"signal peak = {mpv0:.4f}"
            + (
                " (manual search)"
                if signal_guess_used
                else " (automatic)"
            )
        )

        

        # ---------------------------------------------------------------------
        # Mark selected signal peak on plot
        # ---------------------------------------------------------------------

        ax.axvline(
            mpv0,
            color="green",
            lw=1.0,
            ls=":",
            alpha=0.7,
        )

        # =========================================================================
        # STEP 4 — Build a local fit region around selected peak
        # =========================================================================

        # We need the selected peak to determine the HWHM.
        # Therefore we first use all populated bins to estimate the local
        # HWHM, but the helper explicitly works relative to mpv0 rather
        # than using the global maximum.

        try:

            eta0, sigma0, hwhm = (
                _robust_langauss_seeds(
                    x_fit_full,
                    y_fit_full,
                    mpv0
                )
            )

        except Exception as e:

            print(
                f"  Ch {ch}: "
                f"failed to estimate Landau-Gauss seeds: {e}"
            )

            fail_fit[ch] += 1
            thresholds[ch] = np.nan

            continue

        # =========================================================================
        # STEP 5 — Fit window
        # =========================================================================

        if fit_window is not None:

            left_lim = mpv0 - fit_window
            right_lim = mpv0 + fit_window

        else:

            # We want enough of the rising edge to constrain the signal,
            # but do not want the entire Landau tail to dominate the fit.
            left_lim = mpv0 - 2.5 * hwhm
            right_lim = mpv0 + 3.0 * hwhm

        fit_mask2 = (
            (x_fit_full >= left_lim)
            & (x_fit_full <= right_lim)
        )

        x_fit = x_fit_full[fit_mask2]
        y_fit = y_fit_full[fit_mask2]

        if len(x_fit) < 10:

            print(
                f"  Ch {ch}: "
                f"fit window too narrow"
            )

            fail_fit[ch] += 1
            thresholds[ch] = np.nan

            continue

        # =========================================================================
        # STEP 6 — Fit Landau ⊗ Gaussian
        # =========================================================================

        sigma_y = np.sqrt(
            np.maximum(
                y_fit,
                1.0
            )
        )

        try:

            popt, pcov = curve_fit(
                langauss,
                x_fit,
                y_fit,

                p0=[
                    mpv0,
                    eta0,
                    sigma0,
                    max(y_fit),
                    0.0,
                ],

                sigma=sigma_y,
                absolute_sigma=True,

                bounds=(
                    [
                        x_fit.min(),
                        1e-5,
                        1e-5,
                        0.0,
                        0.0,
                    ],

                    [
                        x_fit.max(),
                        max(hwhm, 1e-5),
                        max(hwhm, 1e-5),
                        np.inf,
                        np.inf,
                    ],
                ),

                maxfev=50_000,
            )

            perr = np.sqrt(
                np.maximum(
                    np.diag(pcov),
                    0
                )
            )

            y_model = langauss(
                x_fit,
                *popt
            )

            chi2 = float(
                np.sum(
                    (
                        (y_fit - y_model)
                        / sigma_y
                    ) ** 2
                )
            )

            ndf = (
                len(x_fit)
                - len(popt)
            )

            # =========================================================================
            # STEP 7 — Draw fitted signal over full display range
            # =========================================================================

            xd = np.linspace(
                centres.min(),
                centres.max(),
                2000
            )

            yd = langauss(
                xd,
                *popt
            )

            ax.plot(
                xd,
                yd,
                "r--",
                lw=2,
                label="Landau*Gauss"
            )


            # =========================================================================
            # STEP 8 — Determine Amplitude cut
            # =========================================================================

            # The last parameter of langauss() is the fitted background/offset.
            #
            # For the cut we DO NOT want the background.
            # We therefore evaluate the fitted signal with background = 0.

            mpv_fit = popt[0]
            eta_fit = popt[1]
            sigma_fit = popt[2]
            A_fit = popt[3]

            yd_signal = langauss(
                xd,
                mpv_fit,
                eta_fit,
                sigma_fit,
                A_fit,
                0.0,          # remove fitted background
            )

            ax.plot(
                xd,
                yd_signal,
                "r-",
                lw=1.5,
                alpha=0.8,
                label="Signal component"
            )
            # Peak of the SIGNAL component
            signal_peak_idx = int(
                np.argmax(yd_signal)
            )

            signal_peak_x = float(
                xd[signal_peak_idx]
            )

            signal_peak_y = float(
                yd_signal[signal_peak_idx]
            )

            # -------------------------------------------------------------------------
            # Automatic cut:
            #
            # 10% of the SIGNAL peak height, NOT 10% of the total fit.
            # -------------------------------------------------------------------------

            target = 0.10 * signal_peak_y

            left_side = np.where(
                yd_signal[:signal_peak_idx] <= target
            )[0]

            if len(left_side) > 0:

                auto_thr = float(
                    xd[left_side[-1]]
                )

            else:

                auto_thr = float(
                    xd[0]
                )

            # -------------------------------------------------------------------------
            # Manual channel-specific override
            # -------------------------------------------------------------------------

            if (
                Amplitude_cuts is not None
                and ch in Amplitude_cuts
            ):

                thr = float(
                    Amplitude_cuts[ch]
                )

                cut_source = "manual"

            else:

                thr = auto_thr

                cut_source = "automatic"

            thresholds[ch] = thr

            # =========================================================================
            # STEP 9 — Draw cut
            # =========================================================================

            ax.axvline(
                thr,
                color="orange",
                lw=1.5,
                ls="--",
                label=f"Cut = {thr:.3f}"
            )

            # =========================================================================
            # STEP 10 — Diagnostics
            # =========================================================================

            ax.text(
                0.97,
                0.97,

                f"MPV = {popt[0]:.4f} ± {perr[0]:.4f}\n"
                f"η   = {popt[1]:.4f} ± {perr[1]:.4f}\n"
                f"σ   = {popt[2]:.4f} ± {perr[2]:.4f}\n"
                f"χ²/ndf = {chi2:.1f}/{ndf} "
                f"= {chi2 / max(ndf, 1):.2f}",

                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,

                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.85,
                ),
            )
            
            print(
                f"  Channel {ch}: "
                f"MPV={popt[0]:.4f} | "
                f"auto cut Amplitude >{auto_thr:.4f}V| "
                f"final cut Amplitude > {thr:.4f}V"
            )

        except Exception as e:

            print(
                f"  Ch {ch} fit failed: {e}"
            )

            fail_fit[ch] += 1
            thresholds[ch] = np.nan

        # =========================================================================
        # Axes
        # =========================================================================

        ax.set_title(
            f"Channel {ch}",
            fontsize=18
        )

        ax.set_xlabel(
            "Amplitude (V)",
            fontsize=18
        )

        ax.set_ylabel(
            "Counts",
            fontsize=18
        )

        hep.cms.label(
            label,
            data=is_data,
            rlabel=rlabel,
            loc=0,
            ax=ax,
            fontsize=(14, 12, 12, 11),
        )

        ax.legend(fontsize=10)

        if logy:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1.0, top=None)

        ax.grid(alpha=0.3)

    # =========================================================================
    # Hide unused axes
    # =========================================================================

    for ax in axes[n:]:
        ax.set_visible(False)

    # =========================================================================
    # Figure
    # =========================================================================

    fig.suptitle(
        f"Amplitude distribution per channel - {Run}",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

    # =========================================================================
    # Print thresholds
    # =========================================================================

    print("\nSuggested Amplitude cut thresholds:")

    for ch, thr in sorted(thresholds.items()):

        if np.isfinite(thr):

            print(
                f"  Channel {ch}: "
                f"Amplitude > {thr:.4f} V({cut_source})"
            )

        else:

            print(
                f"  Channel {ch}: "
                f"NO RELIABLE CUT"
            )

    return results, thresholds


def compute_rise_time(
    samp_arr: np.ndarray,
    trig_arr: np.ndarray,
    baseline: float,
    period: float,
    low_frac: float = 0.1,
    high_frac: float = 0.9,
    smooth_window: int = 2,
) -> Optional[float]:
    """
    Compute rise time between low_frac and high_frac of the amplitude.

    More robust than the naive version:
    - Optionally smooths the waveform before threshold crossing search,
      to avoid noise-induced fake crossings (original samples never modified).
    - Searches for the LAST crossing of lo_level before the peak, and the
      LAST crossing of hi_level before the peak — this correctly handles
      noisy rising edges that cross the threshold multiple times.
    - Does not reject on dy <= 0: uses the crossing point even if the
      local slope is flat, as long as the level is bracketed.
    - Falls back gracefully if smoothing doesn't help.

    Parameters
    ----------
    smooth_window : number of samples for moving-average pre-smoothing
                    used ONLY for finding crossings (1 = no smoothing).
    """
    _, samp_ord, _ = reorder_circular_samples_with_trigger(
        trig_arr, samp_arr, reorder_samples=False
    )
    samp_ord = np.asarray(samp_ord, dtype=np.float64)
    n        = len(samp_ord)


    # --- Robust peak finding: use smoothed signal to find peak region,
    #     then take argmax of raw signal in that neighbourhood ----------
    if smooth_window > 1:
        kernel    = np.ones(smooth_window) / smooth_window
        samp_smooth = np.convolve(samp_ord, kernel, mode="same")
    else:
        samp_smooth = samp_ord

    peak_idx  = int(np.argmax(samp_smooth))
    peak_val  = float(samp_ord[peak_idx])   # amplitude from raw sample at peak
    amplitude = peak_val - baseline

    if amplitude <= 0:
        return None

    lo_level = baseline + low_frac  * amplitude
    hi_level = baseline + high_frac * amplitude

    # Sanity check: levels must be within the actual sample range
    if lo_level >= peak_val or hi_level >= peak_val:
        return None

    # --- Find crossings on the smoothed signal (more stable),
    #     but interpolate crossing time using raw samples --------------

    # We want the LAST crossing before the peak for each level,
    # so the search uses the final upward crossing — correct even if
    # the signal crosses the threshold multiple times on the way up.
    t_lo = t_hi = None

    for i in range(peak_idx):
        y0_raw = float(samp_ord[i])
        y1_raw = float(samp_ord[i + 1])
        y0_smo = float(samp_smooth[i])
        y1_smo = float(samp_smooth[i + 1])

        # lo_level crossing — keep updating so we get the LAST one
        if y0_smo <= lo_level <= y1_smo:
            dy = y1_raw - y0_raw
            if abs(dy) > 0:
                t_lo = (i + (lo_level - y0_raw) / dy) * period
            else:
                t_lo = i * period   # flat crossing: use left edge

        # hi_level crossing — keep updating so we get the LAST one
        if y0_smo <= hi_level <= y1_smo:
            dy = y1_raw - y0_raw
            if abs(dy) > 0:
                t_hi = (i + (hi_level - y0_raw) / dy) * period
            else:
                t_hi = i * period

    if t_lo is None or t_hi is None:
        return None

    rt = t_hi - t_lo
    # Sanity: rise time must be positive and physically reasonable
    # (cannot be longer than half the waveform)
    if rt <= 0 or rt > (n // 2) * period:
        return None

    return rt


from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
import mplhep as hep

def plot_rise_time_histograms(
    parquet_path: str,
    Amplitude_cuts: Dict[int, float],
    apply_Amp_filter: bool = False,
    channel_filter: Optional[List[int]] = None,
    period: float = 1 / 6.4,
    channel_col: str = "Channel",
    baseline_col: str = "Baseline",
    peak_col: str = "RawPeak",
    batch_size: int = 100_000,
    root_tree: str = "sampic_hits",
    low_frac: float = 0.1,
    high_frac: float = 0.9,
    n_bins: int = 100,
    sigma_cut: Tuple[float, float] = (2.0, 2.0),
    x_range: Optional[Tuple[float, float]] = None,
    figsize_per_plot: Tuple[float, float] = (8, 5),
    logy: bool = False,
    label: str = "PPS2 Timing Preliminary",
    rlabel: str = "(H8 Test Beam, May 2025)",
    is_data: bool = True,
    fit_guesses: Optional[Dict[int, Union[float, Tuple[float, float]]]] = None,
    baseline_path: Optional[str] = None,
    n_baseline_samples: int = 20,
) -> Tuple[Dict[int, "HistogramResult"], Dict[int, Tuple[float, float]]]:
    """
    Stream waveforms, apply per-channel Amplitude cuts, compute rise times,
    and plot per-channel histograms with sigma-cut threshold markers.
    """
    parquet_path = Path(parquet_path)
    baseline_lookup = load_baseline_lookup(baseline_path) if baseline_path is not None else None

    cols = [channel_col, peak_col, "DataSample", "TriggerPosition"]
    if baseline_lookup is not None:
        cols.insert(0, "HITNumber")
    batches = open_hit_reader(parquet_path, cols=cols, batch_size=batch_size, root_tree=root_tree)

    thresholds = {}
    rise_times: Dict[int, list] = {}
    skipped_Amplitude = 0
    skipped_rt   = 0
    total        = 0
    skipped_Amplitude_ch = defaultdict(int)
    skipped_rt_ch       = defaultdict(int)
    total_ch            = defaultdict(int)

    for batch in tqdm(batches, desc="Computing rise times"):
        
        # Early Channel Filter
        if channel_filter is not None:
            channels_batch = batch[channel_col].to_numpy()
            keep_ch = np.isin(channels_batch, channel_filter)
            if not np.any(keep_ch):
                continue
            batch = batch.take(pa.array(np.where(keep_ch)[0]))

        channels  = batch[channel_col].to_numpy()
        peaks     = batch[peak_col].to_numpy(zero_copy_only=False).astype(np.float64)
        samples   = batch["DataSample"]
        triggers  = batch["TriggerPosition"]

        for i in range(len(channels)):
            total += 1
            ch = int(channels[i])
            total_ch[ch] += 1
            samp_arr = np.asarray(samples[i].as_py(), dtype=np.float64)
            trig_arr = np.asarray(triggers[i].as_py(), dtype=np.int32)
            bl = (
                float(baseline_lookup[int(batch["HITNumber"][i])])
                if baseline_lookup is not None and "HITNumber" in batch
                and int(batch["HITNumber"][i]) in baseline_lookup
                else compute_baseline(samp_arr, trig_arr, n_baseline_samples)
            )
            pk = float(peaks[i])

            if apply_Amp_filter:
                if bl != 0:
                    Amplitude = (pk - bl)
                else:
                    skipped_Amplitude += 1
                    skipped_Amplitude_ch[ch] += 1
                    continue

                cut = Amplitude_cuts.get(ch, 0.0)
                if Amplitude < cut:
                    skipped_Amplitude += 1
                    skipped_Amplitude_ch[ch] += 1
                    continue

            rt = compute_rise_time(
                samp_arr, trig_arr,
                bl, period, low_frac, high_frac,
            )
            if rt is None:
                skipped_rt += 1
                skipped_rt_ch[ch] += 1
                continue

            rise_times.setdefault(ch, []).append(rt)

    print(f"Total hits     : {total:,}")
    if apply_Amp_filter:
        print(f"Skipped (Amplitude)  : {skipped_Amplitude:,} ({100*skipped_Amplitude/max(total,1):.1f}%)")
    print(f"Skipped (no RT): {skipped_rt:,} ({100*skipped_rt/max(total,1):.1f}%)")

    if not rise_times:
        raise RuntimeError("No rise times computed — check Amplitude cuts, filter, and data.")

    # Build histograms
    results = {}
    for ch, values in rise_times.items():
        arr = np.array(values, dtype=np.float64)
        lo, hi = x_range if x_range is not None else (arr.min(), arr.max())
        counts, bin_edges = np.histogram(arr, bins=n_bins, range=(lo, hi))
        results[ch] = HistogramResult(counts=counts, bin_edges=bin_edges)

    # Plotting setup
    channels_sorted = sorted(results)
    n     = len(channels_sorted)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )
    axes = np.array(axes).flatten()

    for ax, ch in tqdm(zip(axes, channels_sorted), desc = "Creating plots"):
        result  = results[ch]
        centres = result.bin_centres
        bw      = result.bin_edges[1] - result.bin_edges[0]

        counts = result.counts.astype(float)
        ax.bar(centres, counts, width=bw, align="center", alpha=0.75, label="Data")

        # Gaussian fit
        fit_mask = counts > 0
        x_fit = centres[fit_mask]
        y_fit = counts[fit_mask]

        if len(x_fit) > 5:
            # Default automated initial guesses
            peak_bin = np.argmax(y_fit)
            mu0      = x_fit[peak_bin]
            sigma0   = 0.2 * (x_fit.max() - x_fit.min())
            A0       = y_fit.max()
            B0       = 0.0

            # =====================================================================
            # NEW LOGIC: Override initial guess for specified channels
            # =====================================================================
            if fit_guesses and ch in fit_guesses:
                guess = fit_guesses[ch]
                if isinstance(guess, (tuple, list)):
                    mu0 = float(guess[0])
                    if len(guess) > 1:
                        sigma0 = float(guess[1])
                else:
                    mu0 = float(guess)
                    sigma0 = 0.05  # Tighter default width (in ns) for signal peak

                # Calculate local count height around mu0 for initial amplitude A0
                near_idx = np.argmin(np.abs(x_fit - mu0))
                A0 = float(y_fit[near_idx])

            sigma_y = np.sqrt(np.maximum(y_fit, 1.0))

            try:
                popt, pcov = curve_fit(
                    gaussian, x_fit, y_fit,
                    p0=[A0, mu0, sigma0, B0],
                    sigma=sigma_y, absolute_sigma=True,
                    bounds=([0, x_fit.min(), 1e-6, -np.inf],
                            [np.inf, x_fit.max(), np.inf,  np.inf]),
                    maxfev=20000,
                )
                perr = np.sqrt(np.diag(pcov))

                y_model = gaussian(x_fit, *popt)
                chi2    = float(np.sum(((y_fit - y_model) / sigma_y)**2))
                ndf     = len(x_fit) - len(popt)

                xd = np.linspace(centres.min(), centres.max(), 2000)
                if not logy:
                    ax.plot(xd, gaussian(xd, *popt), "r--", lw=2, label="Gaussian fit")

                ax.text(
                    0.97, 0.97,
                    f"μ = {popt[1]:.3f} ± {perr[1]:.4f}\n"
                    f"σ = {popt[2]:.3f} ± {perr[2]:.4f}\n"
                    f"χ²/ndf = {chi2:.1f}/{ndf} = {chi2/max(ndf,1):.2f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
                )

                if (popt[1] is not None) and (popt[2] is not None):
                    thr_low = popt[1] - sigma_cut[0] * popt[2]
                    thr_high = popt[1] + sigma_cut[1] * popt[2]
                    thresholds[ch] = (thr_low, thr_high)

                    ax.axvline(thr_low, color="darkorange", linestyle="-.", lw=1.5, 
                               label=f"+{sigma_cut[1]}σ -{sigma_cut[0]}σ cut thresholds")
                    ax.axvline(thr_high, color="darkorange", linestyle="-.", lw=1.5)

            except Exception as e:
                print(f"Gaussian fit failed for channel {ch}: {e}")

        ax.set_title(f"Channel {ch}", fontsize = 18)
        ax.set_xlabel(f"Rise time {int(low_frac*100)}%–{int(high_frac*100)}% (ns)", fontsize = 18)
        ax.set_ylabel("Counts", fontsize = 18)
        hep.cms.label(label, data=is_data, rlabel=rlabel, loc=0, ax=ax, fontsize=(14, 12, 12, 11))
        ax.legend(fontsize=10)

        if logy:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1.0, top=None)
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Rise time {int(low_frac*100)}%–{int(high_frac*100)}%  "
        f"(Amplitude-filtered)", fontsize=14
    )
    plt.tight_layout()
    plt.show()

    print("\nSuggested Rise-Time cut thresholds:")
    for ch, thr in sorted(thresholds.items()):
        print(f"  Channel {ch}: {thr[0]:.4f} < RT < {thr[1]:.4f} ns")
        
    return results, thresholds



hep.style.use("CMS")

def snr_composite_model(x, mpv, eta, sigma_lg, A_lg, A_n, mu_n, sigma_n):
    return langauss(x, mpv, eta, sigma_lg, A_lg, 0.0) + gaussian(x, A_n, mu_n, sigma_n, 0.0)


# -----------------------------------------------------------------------------
# PDF models
# -----------------------------------------------------------------------------

def noise_emg(x: np.ndarray, A: float, K: float, loc: float, scale: float) -> np.ndarray:
    """Exponentially modified Gaussian (Gaussian ⊗ Exponential)."""
    return A * exponnorm.pdf(x, K, loc=loc, scale=scale)


def signal_moyal(x: np.ndarray, A: float, loc: float, scale: float) -> np.ndarray:
    """Stable Landau-like signal model."""
    return A * moyal.pdf(x, loc=loc, scale=scale)


def composite_model(
    x: np.ndarray,
    A_n: float,
    K: float,
    loc_n: float,
    scale_n: float,
    A_s: float,
    loc_s: float,
    scale_s: float,
) -> np.ndarray:
    return noise_emg(x, A_n, K, loc_n, scale_n) + signal_moyal(x, A_s, loc_s, scale_s)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def compute_noise_rms(
    samp_arr: np.ndarray,
    trig_arr: np.ndarray,
    baseline: Optional[float] = None,
    n_noise_samples: int = 20,
) -> float:
    """Compute baseline noise RMS from the first ordered samples.

    If ``baseline`` is omitted, it is calculated from the same first
    ``n_noise_samples`` samples. This keeps the baseline and noise definitions
    internally consistent.
    """
    _, samp_ord, _ = reorder_circular_samples_with_trigger(
        trig_arr, samp_arr, reorder_samples=False,
    )
    samp_ord = np.asarray(samp_ord, dtype=np.float64)
    n_use = min(n_noise_samples, len(samp_ord))
    if n_use <= 1:
        return np.nan

    if baseline is None:
        baseline = float(np.mean(samp_ord[:n_use]))

    noise_region = samp_ord[:n_use] - float(baseline)
    return float(np.sqrt(np.mean(noise_region**2)))


def _safe_percentile(arr: np.ndarray, q: float, default: float) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.percentile(arr, q))


def _detect_peaks(
    centres: np.ndarray,
    counts: np.ndarray,
    prominence_frac: float = 0.03,
    smooth_sigma: float = 2.0,
    min_distance_bins: int = 5,
) -> np.ndarray:
    smoothed = gaussian_filter1d(counts.astype(float), sigma=smooth_sigma)
    prom = max(prominence_frac * float(np.max(smoothed)), 1.0)
    peaks, _ = find_peaks(smoothed, prominence=prom, distance=min_distance_bins)
    return peaks


def _choose_noise_signal_peaks(
    centres: np.ndarray,
    counts: np.ndarray,
    peaks: np.ndarray,
    noise_x_max: float = 15.0,
) -> Tuple[Optional[int], Optional[int]]:
    """Pick a left noise peak and a right signal peak robustly.

    Preference order:
      1) Any peak at x <= noise_x_max is noise, choose the tallest among them.
      2) Signal is the tallest peak at x > noise peak (or the global tallest right-side peak).
      3) Fallbacks when peak finding is sparse.
    """
    if len(peaks) == 0:
        return None, None

    # Left noise peak: prefer a real peak at low x.
    left = peaks[centres[peaks] <= noise_x_max]
    if len(left) > 0:
        noise_idx = int(left[np.argmax(counts[left])])
    else:
        noise_idx = int(peaks[np.argmin(centres[peaks])])

    # Right signal peak: prefer the tallest peak to the right of the noise peak.
    right = peaks[centres[peaks] > centres[noise_idx] + 2.0]
    if len(right) > 0:
        signal_idx = int(right[np.argmax(counts[right])])
    else:
        # If no distinct right peak exists, take the global max after the noise region.
        mask = centres > max(noise_x_max, centres[noise_idx] + 2.0)
        if np.any(mask):
            signal_idx = int(np.argmax(counts * mask))
            if counts[signal_idx] == 0:
                signal_idx = int(peaks[np.argmax(counts[peaks])])
        else:
            signal_idx = int(peaks[np.argmax(counts[peaks])])

    if signal_idx == noise_idx:
        # Final fallback: use the strongest peak and a nearby neighbour if possible.
        order = peaks[np.argsort(centres[peaks])]
        noise_idx = int(order[0])
        signal_idx = int(order[-1])

    return noise_idx, signal_idx


def _valley_between_peaks(
    centres: np.ndarray,
    counts: np.ndarray,
    noise_idx: int,
    signal_idx: int,
) -> Tuple[float, int]:
    """Valley from the smoothed histogram between two peak indices."""
    lo = min(noise_idx, signal_idx)
    hi = max(noise_idx, signal_idx)
    if hi - lo < 2:
        return float(centres[lo]), lo

    smoothed = gaussian_filter1d(counts.astype(float), sigma=1.5)
    region = smoothed[lo : hi + 1]
    valley_rel = int(np.argmin(region))
    valley_idx = lo + valley_rel
    return float(centres[valley_idx]), valley_idx


def _fit_noise_only(
    centres: np.ndarray,
    counts: np.ndarray,
    valley_x: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the noise component on the left side of the valley."""
    mask = (centres >= centres.min()) & (centres <= valley_x)
    x = centres[mask]
    y = counts[mask].astype(float)

    if x.size < 6 or np.sum(y) <= 0:
        raise RuntimeError("Insufficient data for noise fit")

    peak_idx = int(np.argmax(y))
    loc0 = float(x[peak_idx])
    scale0 = max(float(np.std(np.repeat(x, np.maximum(y.astype(int), 1)))), 1.0)
    K0 = 1.5
    A0 = float(np.max(y)) / max(float(exponnorm.pdf(loc0, K0, loc=loc0, scale=max(scale0, 1.0))), 1e-8)

    p0 = [A0, K0, loc0, scale0]
    bounds = ([0.0, 0.05, 0.0, 0.2], [np.inf, 50.0, max(60.0, float(np.max(x)) + 5.0), 80.0])

    popt, pcov = curve_fit(
        noise_emg,
        x,
        y,
        p0=p0,
        bounds=bounds,
        sigma=np.sqrt(np.maximum(y, 1.0)),
        absolute_sigma=True,
        maxfev=20_000,
    )
    return popt, pcov


def _fit_signal_only(
    centres: np.ndarray,
    counts: np.ndarray,
    valley_x: float,
    noise_params: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the signal component on the right side of the valley.

    If noise_params are provided, subtract the noise model before fitting.
    """
    mask = centres >= valley_x
    x = centres[mask]
    y = counts[mask].astype(float)

    if x.size < 6 or np.sum(y) <= 0:
        raise RuntimeError("Insufficient data for signal fit")

    if noise_params is not None:
        y = np.clip(y - noise_emg(x, *noise_params), 0.0, None)

    if np.sum(y) <= 0:
        raise RuntimeError("Signal residual is empty after noise subtraction")

    peak_idx = int(np.argmax(y))
    loc0 = float(x[peak_idx])
    scale0 = max(float(np.std(np.repeat(x, np.maximum(y.astype(int), 1)))), 1.0)
    A0 = float(np.max(y)) / max(float(moyal.pdf(loc0, loc=loc0, scale=max(scale0, 1.0))), 1e-8)

    p0 = [A0, loc0, scale0]
    bounds = ([0.0, 0.0, 0.2], [np.inf, 200.0, 100.0])

    popt, pcov = curve_fit(
        signal_moyal,
        x,
        y,
        p0=p0,
        bounds=bounds,
        sigma=np.sqrt(np.maximum(y, 1.0)),
        absolute_sigma=True,
        maxfev=20_000,
    )
    return popt, pcov


def _fit_composite(
    centres: np.ndarray,
    counts: np.ndarray,
    noise_popt: np.ndarray,
    signal_popt: np.ndarray,
    fit_max_x: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the full composite model using the per-component fits as seeds."""
    mask = (centres >= centres.min()) & (centres <= fit_max_x)
    x = centres[mask]
    y = counts[mask].astype(float)

    if x.size < 8 or np.sum(y) <= 0:
        raise RuntimeError("Insufficient data for composite fit")

    A_n, K, loc_n, scale_n = noise_popt
    A_s, loc_s, scale_s = signal_popt

    p0 = [A_n, K, loc_n, scale_n, A_s, loc_s, scale_s]
    bounds = (
        [0.0, 0.05, 0.0, 0.2, 0.0, 0.0, 0.2],
        [np.inf, 80.0, max(80.0, fit_max_x), 120.0, np.inf, 250.0, 120.0],
    )

    popt, pcov = curve_fit(
        composite_model,
        x,
        y,
        p0=p0,
        bounds=bounds,
        sigma=np.sqrt(np.maximum(y, 1.0)),
        absolute_sigma=True,
        maxfev=40_000,
    )
    return popt, pcov


def _intersection_threshold(
    x: np.ndarray,
    noise_y: np.ndarray,
    signal_y: np.ndarray,
    fallback: float,
) -> float:
    diff = noise_y - signal_y
    sgn = np.sign(diff)
    idx = np.where(np.diff(sgn) != 0)[0]
    if len(idx) == 0:
        return float(fallback)

    # Take the first crossing to the right of the noise peak, if possible.
    i = int(idx[0])
    x0, x1 = x[i], x[i + 1]
    y0, y1 = diff[i], diff[i + 1]
    if y1 == y0:
        return float(x0)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


# -----------------------------------------------------------------------------
# Main plotting function
# -----------------------------------------------------------------------------

 
# -----------------------------------------------------------------------------
# SNR histograms — streamed straight from the raw parquet, with NO Amplitude
# or rise-time filtering applied, so the SNR threshold is determined from
# the channel's natural, unfiltered SNR distribution — independently of the
# Amplitude and rise-time thresholds. All three are combined only afterwards,
# in build_hit_mask.
# -----------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

# Assuming these exist in your module context:
# load_baseline_lookup, open_hit_reader, compute_baseline, compute_noise_rms,
# _detect_peaks, _choose_noise_signal_peaks, _valley_between_peaks, _safe_percentile,
# _fit_noise_only, _fit_signal_only, _fit_composite, composite_model, noise_emg, signal_moyal, hep


def plot_SNR_histograms(
    parquet_path: str,
    channel_col: str = "Channel",
    baseline_col: str = "Baseline",
    peak_col: str = "RawPeak",
    channel_filter: Optional[list[int]] = None,
    n_noise_samples: int = 20,
    batch_size: int = 100_000,
    root_tree: str = "sampic_hits",
    n_bins: int = 120,
    x_range: Optional[Tuple[float, float]] = (0.0, 200.0),
    sigma_cut: float = 2.5,
    figsize_per_plot: Tuple[float, float] = (8, 5),
    logy: bool = False,
    label: str = "PPS2 Timing Preliminary",
    rlabel: str = "(H8 Test Beam, May 2025)",
    is_data: bool = True,
    baseline_path: Optional[str] = None,
    manual_thresholds: Optional[Dict[int, Union[Tuple[Optional[float], Optional[float]], list]]] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Tuple[float, float]]]:
    """Plot per-channel SNR histograms with robust peak finding, fits, and optional manual threshold overrides.

    Parameters
    ----------
    manual_thresholds : Optional[Dict[int, Union[Tuple, list]]]
        Dictionary mapping channel IDs to [low, high] or (low, high) thresholds.
        If a channel key is present, its value will override the automated logic.
        Use `None` or `np.nan` for a single bound to keep the automatic calculation.
        Example: {0: [12.5, 150.0], 1: [8.0, None]}
    """
    parquet_path = Path(parquet_path)
    baseline_lookup = load_baseline_lookup(baseline_path) if baseline_path is not None else None

    cols = [channel_col, peak_col, "DataSample", "TriggerPosition"]
    if baseline_lookup is not None:
        cols.insert(0, "HITNumber")
    batches = open_hit_reader(parquet_path, cols=cols, batch_size=batch_size, root_tree=root_tree)

    snr_by_channel: Dict[int, list] = {}
    total = 0
    skipped = 0        # non-finite / negative SNR values
    skipped_none = 0   # noise_rms could not be computed (<=0 or NaN)

    for batch in tqdm(batches, desc="Computing SNR"):
        channels = batch[channel_col].to_numpy()
        peaks = batch[peak_col].to_numpy(zero_copy_only=False).astype(np.float64)
        samples = batch["DataSample"]
        triggers = batch["TriggerPosition"]

        for i in range(len(channels)):
            total += 1
            ch = int(channels[i])

            if channel_filter is not None and ch not in channel_filter:
                continue

            pk = float(peaks[i])

            samp_arr = np.asarray(samples[i].as_py(), dtype=np.float64)
            trig_arr = np.asarray(triggers[i].as_py(), dtype=np.int32)

            hit_id = int(batch["HITNumber"][i]) if "HITNumber" in batch else None
            bl = (
                float(baseline_lookup[hit_id])
                if baseline_lookup is not None and hit_id in baseline_lookup
                else compute_baseline(samp_arr, trig_arr, n_noise_samples)
            )
            noise_rms = compute_noise_rms(samp_arr, trig_arr, bl, n_noise_samples)
            if not (np.isfinite(noise_rms) and noise_rms > 0):
                skipped_none += 1
                continue

            snr_val = (pk - bl) / noise_rms
            if not np.isfinite(snr_val) or snr_val < 0:
                skipped += 1
                continue

            snr_by_channel.setdefault(ch, []).append(snr_val)

    print(f"Total hits           : {total:,}")
    print(f"No noise_rms (<=0)   : {skipped_none:,} ({100*skipped_none/max(total,1):.1f}%)")
    print(f"Skipped (bad SNR)    : {skipped:,} ({100*skipped/max(total,1):.1f}%)")

    if not snr_by_channel:
        raise RuntimeError("No SNR data found — check column names and channel_filter.")

    snr_by_channel = {ch: np.array(v, dtype=np.float64) for ch, v in snr_by_channel.items()}

    channels_sorted = sorted(snr_by_channel)
    results: Dict[int, np.ndarray] = {}
    bin_edges_by_ch: Dict[int, np.ndarray] = {}
    hist_smooth_by_ch: Dict[int, np.ndarray] = {}

    # Build histograms.
    for ch in channels_sorted:
        arr = snr_by_channel[ch]
        lo, hi = x_range if x_range is not None else (float(np.min(arr)), float(np.max(arr)))
        counts, edges = np.histogram(arr, bins=n_bins, range=(lo, hi))
        results[ch] = counts
        bin_edges_by_ch[ch] = edges
        hist_smooth_by_ch[ch] = gaussian_filter1d(counts.astype(float), sigma=1.5)

    n = len(channels_sorted)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )
    axes = np.array(axes).flatten()

    thresholds: Dict[int, Tuple[float, float]] = {}

    for ax, ch in tqdm(zip(axes, channels_sorted), desc = "Plotting"):
        counts = results[ch].astype(float)
        edges = bin_edges_by_ch[ch]
        bw = edges[1] - edges[0]
        centres = 0.5 * (edges[:-1] + edges[1:])
        arr = snr_by_channel[ch]
        smoothed = hist_smooth_by_ch[ch]

        ax.bar(centres, counts, width=bw, align="center", alpha=0.75, label="Data")

        # Peak finding.
        peaks = _detect_peaks(centres, counts)
        noise_idx, signal_idx = _choose_noise_signal_peaks(centres, counts, peaks)

        if noise_idx is None or signal_idx is None:
            noise_idx = int(np.argmax(smoothed[: max(2, min(len(smoothed), int(0.15 * len(smoothed))))]))
            right_mask = centres > centres[max(noise_idx + 2, 1)]
            if np.any(right_mask):
                signal_idx = int(np.argmax(np.where(right_mask, smoothed, -np.inf)))
            else:
                signal_idx = int(np.argmax(smoothed))

        valley_x, valley_idx = _valley_between_peaks(centres, counts, noise_idx, signal_idx)

        # High cut: robust upper percentile.
        thr_high = _safe_percentile(arr, 95, default=float(np.max(centres)))

        if not np.isfinite(valley_x):
            valley_x = float(centres[min(noise_idx, signal_idx)])
        valley_x = float(np.clip(valley_x, centres.min(), thr_high))

        # Fit components separately.
        comp_ok = False
        try:
            noise_popt, noise_pcov = _fit_noise_only(centres, counts, valley_x)
        except Exception as e:
            print(f"Channel {ch}: noise fit failed: {e}")
            noise_popt = None

        try:
            signal_popt, signal_pcov = _fit_signal_only(
                centres,
                counts,
                valley_x,
                noise_params=noise_popt if noise_popt is not None else None,
            )
        except Exception as e:
            print(f"Channel {ch}: signal fit failed: {e}")
            signal_popt = None

        # Composite fit if both component fits worked.
        popt = None
        pcov = None
        chi2 = np.nan
        ndf = np.nan

        fit_max_x = float(x_range[1]) if x_range is not None else float(centres.max())
        fit_mask = (centres >= centres.min()) & (centres <= fit_max_x) & (counts > 0)
        x_fit = centres[fit_mask]
        y_fit = counts[fit_mask]

        if noise_popt is not None and signal_popt is not None and x_fit.size >= 8:
            try:
                popt, pcov = _fit_composite(centres, counts, noise_popt, signal_popt, fit_max_x)
                y_model = composite_model(x_fit, *popt)
                chi2 = float(np.sum(((y_fit - y_model) / np.sqrt(np.maximum(y_fit, 1.0))) ** 2))
                ndf = len(y_fit) - len(popt)
                comp_ok = True
            except Exception as e:
                print(f"Channel {ch}: composite fit failed: {e}")

        # Plot model(s).
        xd = np.linspace(float(centres.min()), float(centres.max()), 2000)

        if comp_ok and popt is not None:
            ax.plot(xd, composite_model(xd, *popt), "r-", lw=2.0, label="Composite fit")

        # ------------------------------------------------------------------
        # ROBUST LOW-CUT DETERMINATION
        # ------------------------------------------------------------------

        thr_low = None

        fit_signal_mpv = None
        if comp_ok and popt is not None:
            fit_signal_mpv = popt[5]
        elif signal_popt is not None:
            fit_signal_mpv = signal_popt[1]

        # 1) Try histogram valley between two peaks
        smooth = gaussian_filter1d(counts.astype(float), sigma=2.0)
        peaks, _ = find_peaks(smooth, prominence=0.05 * smooth.max(), distance=10)
        peaks = peaks[smooth[peaks] > 0.10 * smooth.max()]

        noise_peak_x = None
        signal_peak_x = None

        if len(peaks) >= 2:
            noise_idx = np.argmin(centres[peaks])
            noise_peak = peaks[noise_idx]

            remaining_peaks_indices = [i for i in range(len(peaks)) if i != noise_idx]
            remaining_peaks = peaks[remaining_peaks_indices]
            signal_peak = remaining_peaks[np.argmax(smooth[remaining_peaks])]

            noise_peak_x = centres[noise_peak]
            signal_peak_x = centres[signal_peak]

            if signal_peak > noise_peak + 2:
                if fit_signal_mpv is not None:
                    mpv_idx = int(np.argmin(np.abs(centres - fit_signal_mpv)))
                else:
                    mpv_idx = signal_peak

                search_right = min(signal_peak, mpv_idx)

                if search_right > noise_peak + 1:
                    valley_idx = noise_peak + np.argmin(smooth[noise_peak:search_right])
                    thr_low = float(centres[valley_idx])

        # 2) Try signal fit if valley wasn't found
        if thr_low is None and signal_popt is not None:
            xd_cut = np.linspace(centres.min(), centres.max(), 5000)
            signal_y = signal_moyal(xd_cut, *signal_popt)
            peak_idx = np.argmax(signal_y)
            peak_height = signal_y[peak_idx]
            target = 0.05 * peak_height
            left_side = signal_y[:peak_idx]
            below = np.where(left_side <= target)[0]

            if len(below):
                thr_low = float(xd_cut[below[-1]])

        # 3) Pure histogram fallback
        if thr_low is None:
            peak_idx = np.argmax(smooth)
            peak_height = smooth[peak_idx]
            target = 0.05 * peak_height
            left_side = smooth[:peak_idx]
            below = np.where(left_side <= target)[0]

            if len(below):
                thr_low = float(centres[below[-1]])

        # 4) Absolute emergency fallback
        if thr_low is None:
            peak_idx = np.argmax(smooth)
            thr_low = float(max(centres[0], centres[peak_idx] * 0.25))

        # 5) Enforce physical sanity
        if noise_peak_x is not None:
            if thr_low < noise_peak_x:
                thr_low = max(thr_low, noise_peak_x)

        if signal_peak_x is not None:
            thr_low = min(thr_low, 0.90 * signal_peak_x)

        if fit_signal_mpv is not None:
            thr_low = min(thr_low, 0.95 * fit_signal_mpv)

        thr_low = min(thr_low, thr_high - bw)
        thr_low = max(thr_low, centres.min())

        # ------------------------------------------------------------------
        # APPLY MANUAL OVERRIDES (IF PROVIDED)
        # ------------------------------------------------------------------
        low_is_manual = False
        high_is_manual = False

        if manual_thresholds is not None and ch in manual_thresholds:
            man_low, man_high = manual_thresholds[ch]

            if man_low is not None and np.isfinite(man_low):
                thr_low = float(man_low)
                low_is_manual = True

            if man_high is not None and np.isfinite(man_high):
                thr_high = float(man_high)
                high_is_manual = True

        thresholds[ch] = (float(thr_low), float(thr_high))

        # Draw thresholds.
        low_lbl = f"Low cut ({'manual' if low_is_manual else 'valley'} = {thr_low:.1f})"
        high_lbl = f"High cut ({'manual' if high_is_manual else '95th'} = {thr_high:.1f})"

        ax.axvline(thr_low, color="firebrick" if low_is_manual else "darkorange", ls="-.", lw=1.5, label=low_lbl)
        ax.axvline(thr_high, color="firebrick" if high_is_manual else "darkorange", ls=":", lw=1.5, label=high_lbl)

        # Text box.
        if comp_ok and popt is not None:
            A_n, K, loc_n, scale_n, A_s, loc_s, scale_s = popt
            text = (
                f"Noise: loc={loc_n:.2f}, σ={scale_n:.2f}, K={K:.2f}\n"
                f"Signal: loc={loc_s:.2f}, σ={scale_s:.2f}\n"
                f"χ²/ndf = {chi2:.1f}/{ndf:.0f} = {chi2 / max(ndf, 1):.2f}"
            )
        else:
            text = f"Valley ≈ {thr_low:.2f}\nHigh cut ≈ {thr_high:.2f}"

        ax.text(
            0.97,
            0.97,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        ax.set_title(f"Channel {ch}", fontsize=18)
        ax.set_xlabel("SNR", fontsize=18)
        ax.set_ylabel("Counts", fontsize=18)
        if hep is not None:
            try:
                hep.cms.label(label, data=is_data, rlabel=rlabel, loc=0, ax=ax, fontsize=(14, 12, 12, 11))
            except Exception:
                pass
        ax.legend(fontsize=9)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.set_xlim(x_range)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("SNR per Channel (unfiltered, raw distribution)", fontsize=14)
    plt.tight_layout()
    plt.show()

    print("\nFinal SNR cut thresholds:")
    for ch, (lo, hi) in sorted(thresholds.items()):
        print(f"  Channel {ch}: {lo:.3f} < SNR < {hi:.3f}")

    return results, thresholds





def build_hit_mask(
    parquet_path: str,
    Amplitude_cuts:   Optional[Dict[int, float]] = None,
    rise_time_cuts:  Optional[Dict[int, tuple]] = None,
    SNR_cuts:        Optional[Dict[int, tuple]] = None,
    channel_filter:  Optional[list[int]] = None,
    first_hit:       int = 0,
    num_hits:        Optional[int] = None,
    period:          float = 1 / 6.4,
    batch_size:      int = 100_000,
    root_tree:       str = "sampic_hits",
    low_frac:        float = 0.1,
    high_frac:       float = 0.9,
    smooth_window:   int = 2,
    n_noise_samples: int = 20,
    save_path:       Optional[str] = None,
    baseline_path:   Optional[str] = None,
) -> np.ndarray:
    """
    Stream a parquet file and return the HITNumbers of waveforms that pass
    all requested filters.
 
    Filters applied in order (each is optional):
      1. channel_filter  — keep only listed channels
      2. Amplitude_cuts   — per-channel minimum Amplitude = (RawPeak-Baseline)/Baseline
      3. rise_time_cuts  — per-channel (min_rt, max_rt) rise time window in ns
      4. SNR_cuts        — per-channel (min_snr, max_snr) SNR window,
                            where SNR = (RawPeak-Baseline) / noise_rms
      5. first_hit       — skip the first N passing hits (global, across all channels)
      6. num_hits        — stop after N passing hits
 
    Parameters
    ----------
    Amplitude_cuts : dict mapping channel → minimum Amplitude.
               e.g. {4: 0.10, 8: 0.05}
               Channels absent from the dict pass with no Amplitude cut.
 
    rise_time_cuts : dict mapping channel → (min_ns, max_ns).
               e.g. {4: (0.5, 3.0), 8: (0.3, 2.0)}
               Channels absent from the dict pass with no rise time cut.
               Rise time is computed from the raw waveform via
               compute_rise_time only when this filter is active.
 
    SNR_cuts : dict mapping channel → (min_snr, max_snr).
               e.g. {4: (8.0, 200.0), 8: (6.0, 200.0)}
               Channels absent from the dict pass with no SNR cut.
               SNR = (RawPeak - Baseline) / noise_rms, where noise_rms comes
               from compute_noise_rms on the raw waveform (same definition
               used when building the full database) — computed only when
               this filter is active.
 
    channel_filter : list of channels to keep; None = all channels.
 
    first_hit  : number of passing hits to skip before collecting.
    num_hits   : maximum number of passing hits to collect; None = all.
 
    n_noise_samples : number of leading ordered samples used by
               compute_noise_rms to estimate the baseline noise RMS.
 
    save_path  : if provided, saves the HITNumber array as a .npy file
                 so the mask can be reloaded without re-running the filter.
 
    Returns
    -------
    np.ndarray of int
        1-D array of HITNumbers that passed all filters, in file order.
 
    Examples
    --------
    >>> good_hits = build_hit_mask(
    ...     parquet_file,
    ...     Amplitude_cuts  = {4: 0.10, 8: 0.05},
    ...     rise_time_cuts = {4: (0.5, 3.0), 8: (0.3, 2.0)},
    ...     SNR_cuts       = {4: (8.0, 200.0), 8: (6.0, 200.0)},
    ...     channel_filter = [4, 8],
    ...     save_path      = "good_hits_run42.npy",
    ... )
    >>> # reload later without reprocessing:
    >>> good_hits = np.load("good_hits_run42.npy")
    """
    baseline_lookup = load_baseline_lookup(baseline_path) if baseline_path is not None else None
    need_waveform = (rise_time_cuts is not None) or (SNR_cuts is not None) or (baseline_lookup is None and Amplitude_cuts is not None)
 
    # Columns to load — only add waveform columns if actually needed
    cols = ["HITNumber", "Channel", "RawPeak"]
    if need_waveform:
        cols += ["DataSample", "TriggerPosition"]
 
    batches = open_hit_reader(
        parquet_path, cols=cols,
        batch_size=batch_size, root_tree=root_tree,
    )
 
    good_hit_numbers = []
    total_ch         = defaultdict(int)
    fail_Amplitude_ch = defaultdict(int)
    fail_rt_ch       = defaultdict(int)
    fail_rt_none_ch  = defaultdict(int)
    fail_snr_ch      = defaultdict(int)
    fail_snr_none_ch = defaultdict(int)
    passed_ch        = defaultdict(int)
 
    # Counters for diagnostics
    total          = 0
    fail_Amplitude  = 0
    fail_rt        = 0
    fail_rt_none   = 0
    fail_snr       = 0
    fail_snr_none  = 0
    skipped        = 0   # first_hit skip
    collected      = 0   # passing hits collected
 
    for batch in tqdm(batches, desc="Filtering waveforms"):
        hit_ids   = batch["HITNumber"].to_numpy()
        channels  = batch["Channel"].to_numpy()
        peaks     = batch["RawPeak"].to_numpy(zero_copy_only=False).astype(np.float64)
 
        if need_waveform:
            samples  = batch["DataSample"]
            triggers = batch["TriggerPosition"]
 
        for i in range(len(hit_ids)):
            total += 1
            ch = int(channels[i])
            pk = float(peaks[i])
            total_ch[ch] += 1
 
            # Apply cuts only to selected channels
            apply_filters = (
                channel_filter is None or
                ch in channel_filter
            )
 
            # Compute the corrected waveform baseline once when needed.
            needs_rt  = apply_filters and rise_time_cuts is not None and ch in rise_time_cuts
            needs_snr = apply_filters and SNR_cuts is not None and ch in SNR_cuts
            needs_baseline = apply_filters and (ch in (Amplitude_cuts or {}) or needs_rt or needs_snr)

            if needs_baseline:
                if baseline_lookup is not None and int(hit_ids[i]) in baseline_lookup:
                    bl = float(baseline_lookup[int(hit_ids[i])])
                else:
                    samp_arr = np.asarray(samples[i].as_py(), dtype=np.float64)
                    trig_arr = np.asarray(triggers[i].as_py(), dtype=np.int32)
                    bl = compute_baseline(samp_arr, trig_arr, n_noise_samples)

            # 2. Amplitude cut
            if apply_filters and Amplitude_cuts is not None and ch in Amplitude_cuts:
                if not np.isfinite(bl):
                    fail_Amplitude += 1
                    fail_Amplitude_ch[ch] += 1
                    continue
                Amplitude = (pk - bl)
                if Amplitude < Amplitude_cuts[ch]:
                    fail_Amplitude += 1
                    fail_Amplitude_ch[ch] += 1
                    continue
 
            # Decode the waveform once if either rise time or SNR is needed.
            if needs_rt or needs_snr:
                samp_arr = np.asarray(samples[i].as_py(), dtype=np.float64)
                trig_arr = np.asarray(triggers[i].as_py(), dtype=np.int32)
 
            # 3. Rise time cut
            if needs_rt:
                rt = compute_rise_time(
                    samp_arr, trig_arr,
                    bl, period, low_frac, high_frac, smooth_window,
                )
                if rt is None:
                    fail_rt_none += 1
                    fail_rt_ch[ch] += 1
                    fail_rt_none_ch[ch] += 1
                    continue
                rt_min, rt_max = rise_time_cuts[ch]
                if not (rt_min <= rt <= rt_max):
                    fail_rt += 1
                    fail_rt_ch[ch] += 1
                    continue
 
            # 4. SNR cut — SNR = (RawPeak - Baseline) / noise_rms
            if needs_snr:
                noise_rms = compute_noise_rms(samp_arr, trig_arr, bl, n_noise_samples)
                amplitude = pk - bl
                if not (np.isfinite(noise_rms) and noise_rms > 0):
                    fail_snr_none += 1
                    fail_snr_ch[ch] += 1
                    fail_snr_none_ch[ch] += 1
                    continue
                snr_val = amplitude / noise_rms
                snr_min, snr_max = SNR_cuts[ch]
                if not (snr_min <= snr_val <= snr_max):
                    fail_snr += 1
                    fail_snr_ch[ch] += 1
                    continue
 
            # 5. first_hit skip
            if skipped < first_hit:
                skipped += 1
                continue
 
            # 6. num_hits cap
            if num_hits is not None and collected >= num_hits:
                break   # inner loop — will also need to break outer below
 
            good_hit_numbers.append(int(hit_ids[i]))
            collected += 1
            passed_ch[ch] += 1
 
        # Break outer loop too if cap reached
        if num_hits is not None and collected >= num_hits:
            break
 
    good = np.array(good_hit_numbers, dtype=np.int64)
 
    # --- Diagnostics --------------------------------------------------------
    print("\nPer‑channel filter summary:")
    for ch in sorted(total_ch):
        tot   = total_ch[ch]
        amp_f = fail_Amplitude_ch[ch]
        rt_f  = fail_rt_ch[ch]
        rt_n  = fail_rt_none_ch[ch]
        snr_f = fail_snr_ch[ch]
        snr_n = fail_snr_none_ch[ch]
        ok    = passed_ch[ch]
 
        print(
            f"  Channel {ch}: {tot} hits | "
            f"Amplitude fails = {amp_f} ({100*amp_f/max(tot,1):.1f}%) | "
            f"RT fails = {rt_f} ({100*rt_f/max(tot,1):.1f}%) | "
            f"RT None = {rt_n} ({100*rt_n/max(tot,1):.1f}%) | "
            f"SNR fails = {snr_f} ({100*snr_f/max(tot,1):.1f}%) | "
            f"SNR None = {snr_n} ({100*snr_n/max(tot,1):.1f}%) | "
            f"Passed = {ok} ({100*ok/max(tot,1):.1f}%)"
        )
 
    print(f"\nFilter summary ({parquet_path})")
    print(f"  Total hits seen      : {total:,}")
    print(f"  Failed Amplitude cut  : {fail_Amplitude:,} ({100*fail_Amplitude/max(total,1):.1f}%)")
    print(f"  Failed rise time cut : {fail_rt:,} ({100*fail_rt/max(total,1):.1f}%)")
    print(f"  No rise time found   : {fail_rt_none:,} ({100*fail_rt_none/max(total,1):.1f}%)")
    print(f"  Failed SNR cut       : {fail_snr:,} ({100*fail_snr/max(total,1):.1f}%)")
    print(f"  No SNR (noise_rms<=0): {fail_snr_none:,} ({100*fail_snr_none/max(total,1):.1f}%)")
    print(f"  Skipped (first_hit)  : {skipped:,}")
    print(f"  Passing hits         : {len(good):,} ({100*len(good)/max(total,1):.1f}%)")
 
    if save_path is not None:
        np.save(save_path, good)
        print(f"  Saved mask to       : {save_path}")
 
    return good




def load_hit_mask(save_path: str) -> np.ndarray:
    """
    Loads a previously saved hit mask from a .npy file 
    and returns a 'good'-like NumPy array.
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Hit mask file not found at: {save_path}")
        
    print(f"Successfully loaded mask from {save_path}")
    return np.load(save_path)


def apply_hit_mask(
    parquet_path: str,
    hit_mask:     np.ndarray,
    columns:      Optional[list[str]] = None,
    batch_size:   int = 100_000,
    root_tree:    str = "sampic_hits",
) -> pa.Table:
    """
    Load only the rows matching `hit_mask` from a parquet file.

    Uses the same batch-by-batch strategy as build_coincidence_histograms
    — never reads non-matching rows into memory.

    Parameters
    ----------
    hit_mask : array of HITNumbers returned by build_hit_mask,
               or loaded from a .npy file.
    columns  : columns to load; None = all columns.

    Returns
    -------
    pyarrow.Table with only the matching rows.
    """
    # Build a set for O(1) lookup
    mask_set = set(hit_mask.tolist())

    pqf        = pq.ParquetFile(parquet_path)
    load_cols  = columns or pqf.schema_arrow.names
    if "HITNumber" not in load_cols:
        load_cols = ["HITNumber"] + list(load_cols)

    collected  = []
    global_row = 0

    for batch in pqf.iter_batches(batch_size=batch_size, columns=load_cols):
        hit_ids   = batch["HITNumber"].to_numpy()
        local_idx = np.where(np.isin(hit_ids, hit_mask))[0]
        if len(local_idx) > 0:
            collected.append(batch.take(pa.array(local_idx)))
        global_row += batch.num_rows

    if not collected:
        raise RuntimeError("No matching hits found — check hit_mask.")

    return pa.Table.from_batches(collected)




#---------------------------------------------------------------------
#   Older version before correcting baseline calculation on 12/08/26:
#---------------------------------------------------------------------


# from __future__ import annotations

# import os
# import matplotlib.pyplot as plt
# import pyarrow.parquet as pq
# import pyarrow as pa
# from scipy.optimize import curve_fit
# from scipy.stats import moyal
# from typing import Optional, Dict, List, Tuple
# from dataclasses import dataclass
# import numpy as np
# from tqdm import tqdm
# from pathlib import Path
# from sampiclyser.sampic_tools import reorder_circular_samples_with_trigger, open_hit_reader
# import mplhep as hep
# from scipy.ndimage import gaussian_filter1d
# from typing import Dict, Optional, Tuple
# from pathlib import Path
# from scipy.signal import find_peaks
# from collections import defaultdict

# from scipy.stats import exponnorm, moyal

# try:
#     import mplhep as hep  # user may have a local alias
# except Exception:
#     try:
#         import mplhep as hep
#     except Exception:
#         hep = None
# # =============================================================================
# # Histogram container
# # =============================================================================
# @dataclass
# class HistogramResult:
#     counts: np.ndarray
#     bin_edges: np.ndarray
#     @property
#     def bin_centres(self):
#         return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

# # =============================================================================
# # Landau ⊗ Gaussian
# # =============================================================================
# def langauss(x, mpv, eta, sigma, A, B):
#     z = np.linspace(-5*sigma, 5*sigma, 121)
#     gauss = np.exp(-0.5 * (z/sigma)**2)
#     gauss /= np.trapezoid(gauss, z)
#     xx = x[:, None] - z[None, :]
#     landau = moyal.pdf(xx, loc=mpv, scale=eta)
#     conv = np.trapezoid(landau * gauss, z, axis=1)
#     return A * conv + B

# def gaussian(x, A, mu, sigma, B):
#     return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + B

# def snr_composite_model(x, mpv, eta, sigma_lg, A_lg, mu_n, sigma_n, A_n):
#     return langauss(x, mpv, eta, sigma_lg, A_lg, 0.0) + gaussian(x, A_n, mu_n, sigma_n, 0.0)

# def snr_langaus_model(x, mpv, eta, sigma_lg, A_lg):
#     return langauss(x, mpv, eta, sigma_lg, A_lg, 0.0)

# # =============================================================================
# # CFD50
# # =============================================================================

# def compute_cfd50(
#     samp_arr: np.ndarray,
#     trig_arr: np.ndarray,
#     baseline: float,
#     period: float,
#     min_amplitude: float = 0.01,      # V — reject noise spikes
#     edge_buffer_samples: int = 10,     # samples — peak must not be too close to buffer edge
# ) -> Optional[float]:
    
#     """
#     Compute the CFD50 time offset from the start of the ordered waveform.

#     Reorders the circular buffer, finds the rising-edge crossing of the
#     50% amplitude level, and interpolates linearly between the two
#     bracketing samples.

#     Parameters
#     ----------
#     samp_arr : ndarray of float, shape (N,)
#         Raw ADC sample values.
#     trig_arr : ndarray of {0,1}, shape (N,)
#         Trigger markers.
#     baseline : float
#         Baseline offset (same units as samp_arr).
#     period : float
#         Time between samples in ns.

#     Returns
#     -------
#     float or None
#         Time offset in ns from sample 0 of the ordered buffer to the
#         50% crossing, or None if no valid rising-edge crossing is found.
#     """

#     trig_ord, samp_ord, _ = reorder_circular_samples_with_trigger(
#         trig_arr, samp_arr, reorder_samples=False
#     )
#     samp_ord = np.asarray(samp_ord, dtype=np.float64)
#     n = len(samp_ord)

#     peak_idx = int(np.argmax(samp_ord))
#     peak_val = float(samp_ord[peak_idx])

#     # Reject noise: peak must clear baseline by min_amplitude
#     if peak_val - baseline <= min_amplitude:
#         return None

#     # Reject buffer-edge artifacts: peak must be away from both edges
#     if peak_idx < edge_buffer_samples or peak_idx > n - edge_buffer_samples:
#         return None

#     half_level = baseline + 0.5 * (peak_val - baseline)

#     for i in range(peak_idx):
#         y0 = float(samp_ord[i])
#         y1 = float(samp_ord[i + 1])
#         if y0 <= half_level <= y1:
#             dy = y1 - y0
#             if dy <= 0:
#                 return None
#             frac = (half_level - y0) / dy
#             return (i + frac) * period

#     return None


# def compute_and_save_cfd50(
#     parquet_path: str,
#     output_path: Optional[str] = None,
#     period: float = 1 / 6.4,
#     batch_size: int = 100_000,
#     root_tree: str = "sampic_hits",
#     min_amplitude: float = 0.01,      # V — reject noise spikes
#     edge_buffer_samples: int = 10,     # samples — peak must not be too close to buffer edge

# ) -> np.ndarray:
#     """
#     Compute CFD50 offsets for every hit in a parquet file and save as a
#     lightweight lookup parquet with two columns: HITNumber, CFD50Offset.

#     HITs where CFD50 fails get NaN — callers can decide whether to drop
#     or fall back to OrderedCell0Time.

#     Parameters
#     ----------
#     output_path : path for the output parquet; if None, uses
#                   parquet_path with '_cfd50' suffix.

#     Returns
#     -------
#     np.ndarray of float64, shape (N,)
#         CFD50 offsets in ns, indexed by file row order.
#         NaN for failed hits.
#     """
#     if output_path is None:
#         p = Path(parquet_path)
#         output_path = str(p.with_stem(p.stem + "_cfd50"))

#     cols    = ["HITNumber", "Baseline", "DataSample", "TriggerPosition", "Channel"]
#     batches = open_hit_reader(parquet_path, cols=cols,
#                               batch_size=batch_size, root_tree=root_tree)

#     all_hit_ids = []
#     all_offsets = []
#     failed      = 0
#     total       = 0

#     fail_ch  = defaultdict(int)
#     total_ch = defaultdict(int)


#     for batch in tqdm(batches, desc="Computing CFD50"):
#         hit_ids   = batch["HITNumber"].to_numpy()
#         baselines = batch["Baseline"].to_numpy(zero_copy_only=False).astype(np.float64)
#         samples   = batch["DataSample"]
#         triggers  = batch["TriggerPosition"]

#         for i in tqdm(range(len(hit_ids)), desc="CFD50", leave=False):

#             ch = int(batch["Channel"][i])
#             total += 1
#             total_ch[ch] += 1

#             offset = compute_cfd50(
#                 np.asarray(samples[i].as_py(),  dtype=np.float64),
#                 np.asarray(triggers[i].as_py(), dtype=np.int32),
#                 float(baselines[i]),
#                 period,
#                 min_amplitude,
#                 edge_buffer_samples,     # samples — peak must not be too close to buffer edge

#             )
#             all_hit_ids.append(int(hit_ids[i]))

#             if offset is None:
#                 all_offsets.append(np.nan)
#                 failed += 1
#                 fail_ch[ch] += 1
#             else:
#                 all_offsets.append(offset)


#     print(f"CFD50 failures: {failed:,} / {total:,} ({100*failed/max(total,1):.2f}%)")
#     print("\nPer‑channel CFD50 statistics:")
#     for ch in sorted(total_ch):
#         tot = total_ch[ch]
#         fail = fail_ch[ch]
#         print(
#             f"  Channel {ch}: {tot} hits | "
#             f"CFD50 fails = {fail} ({100*fail/max(tot,1):.2f}%)"
#         )

#     # Save as a two-column parquet — tiny file, fast to load
#     lut = pa.table({
#         "HITNumber":   pa.array(all_hit_ids, type=pa.int64()),
#         "CFD50Offset": pa.array(all_offsets, type=pa.float64()),
#     })
#     pq.write_table(lut, output_path)
#     print(f"Saved CFD50 lookup to: {output_path}")

#     return np.array(all_offsets, dtype=np.float64)


# def load_cfd50_lookup(cfd50_path: str) -> Dict[int, float]:
#     """
#     Load a CFD50 lookup parquet into a HITNumber → offset dict.
#     NaN offsets (failed hits) are excluded.
#     """
#     table    = pq.read_table(cfd50_path, columns=["HITNumber", "CFD50Offset"])
#     hit_ids  = table["HITNumber"].combine_chunks().to_numpy()
#     offsets  = table["CFD50Offset"].combine_chunks().to_numpy(zero_copy_only=False)
#     valid    = np.isfinite(offsets)
#     return dict(zip(hit_ids[valid].tolist(), offsets[valid].tolist()))



# # =============================================================================
# # =============================================================================
# # CFD CONFIG
# # =============================================================================

# CFD_LEVELS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]


# def compute_cfd_times_and_slopes(
#     samp_arr: np.ndarray,
#     trig_arr: np.ndarray,
#     baseline: float,
#     period: float,
#     cfd_levels: List[int] = CFD_LEVELS,
#     min_amplitude: float = 0.01,
#     edge_buffer_samples: int = 10,
#     min_slope: float = 0.0,
#     consecutive_points: int = 2,
# ) -> tuple[dict, dict]:
#     """
#     Compute CFD crossing times and slopes for multiple CFD levels.

#     Returns
#     -------
#     cfd_offsets : dict  {10: time_ns, 20: time_ns, ...}
#     cfd_slopes  : dict  slope at crossing in V/ns
#     """

#     _, samp_ord, _ = reorder_circular_samples_with_trigger(
#         trig_arr,
#         samp_arr,
#         reorder_samples=False,
#     )

#     y = np.asarray(samp_ord, dtype=np.float64)
#     n = len(y)

#     peak_idx = int(np.argmax(y))
#     peak_val = float(y[peak_idx])
#     amplitude = peak_val - baseline

#     cfd_offsets = {k: np.nan for k in cfd_levels}
#     cfd_slopes  = {k: np.nan for k in cfd_levels}

#     # -------------------------------------------------------------------------
#     # Basic quality cuts
#     # -------------------------------------------------------------------------
#     if amplitude <= min_amplitude:
#         return cfd_offsets, cfd_slopes

#     if peak_idx < edge_buffer_samples or peak_idx > n - edge_buffer_samples:
#         return cfd_offsets, cfd_slopes

#     # Rising edge only (inclusive of peak so highest CFD levels can be reached)
#     rising = y[: peak_idx + 1]
#     n_rising = len(rising)

#     # -------------------------------------------------------------------------
#     # CFD loop
#     # -------------------------------------------------------------------------
#     for frac_percent in cfd_levels:
#         frac = frac_percent / 100.0
#         threshold = baseline + frac * amplitude

#         # Search window: need `consecutive_points` samples on each side
#         # Upper bound ensures `above` slice is always full length
#         search_end = n_rising - consecutive_points
#         for i in range(consecutive_points, search_end):
#             below = rising[i - consecutive_points : i]
#             above = rising[i : i + consecutive_points]

#             if np.all(below < threshold) and np.all(above >= threshold):
#                 # Linear interpolation between the last-below and first-above
#                 y0 = float(rising[i - 1])
#                 y1 = float(rising[i])
#                 dy = y1 - y0

#                 if dy <= min_slope:
#                     continue  # try next candidate (rare: flat region at threshold)

#                 frac_interp = (threshold - y0) / dy
#                 t_cross = ((i - 1) + frac_interp) * period   # ns
#                 slope   = dy / period                          # V/ns

#                 cfd_offsets[frac_percent] = t_cross
#                 cfd_slopes[frac_percent]  = slope
#                 break  # take first valid crossing

#     return cfd_offsets, cfd_slopes


# # =============================================================================
# # CFD DATABASE BUILDER
# # =============================================================================

# def compute_and_save_cfd_database(
#     parquet_path: str,
#     output_path: Optional[str] = None,
#     period: float = 1 / 6.4,        # ns per sample
#     batch_size: int = 100_000,
#     root_tree: str = "sampic_hits",
#     min_amplitude: float = 0.01,
#     edge_buffer_samples: int = 10,
#     consecutive_points: int = 2,
# ) -> str:
#     """
#     Build a CFD database containing:
#         HITNumber
#         CFD{10..90}Offset   (time of threshold crossing, ns)
#         CFD{10..90}Slope    (slope at crossing, V/ns)
#     """

#     if output_path is None:
#         p = Path(parquet_path)
#         output_path = str(p.with_stem(p.stem + "_cfddb"))
    
#     parquet_path = Path(parquet_path)
#     output_path = Path(output_path)

#     cols = ["HITNumber", "Baseline", "DataSample", "TriggerPosition"]

#     batches = open_hit_reader(
#         parquet_path,
#         cols=cols,
#         batch_size=batch_size,
#         root_tree=root_tree,
#     )

#     # -------------------------------------------------------------------------
#     # Storage + per-level failure counters
#     # -------------------------------------------------------------------------
#     out: dict[str, list] = {"HITNumber": []}
#     for k in CFD_LEVELS:
#         out[f"CFD{k}Offset"] = []
#         out[f"CFD{k}Slope"]  = []

#     total = 0
#     failures: dict[int, int] = {k: 0 for k in CFD_LEVELS}

#     # -------------------------------------------------------------------------
#     # Main loop
#     # -------------------------------------------------------------------------
#     for batch in tqdm(batches, desc="Building CFD database"):
#         hit_ids  = batch["HITNumber"].to_numpy().astype(np.int64)
#         baselines = batch["Baseline"].to_numpy(zero_copy_only=False).astype(np.float64)
#         samples_col  = batch["DataSample"]
#         triggers_col = batch["TriggerPosition"]

#         for i in tqdm(range(len(hit_ids)), leave=False, desc="Hits", mininterval=0.1, dynamic_ncols=False):
#             total += 1
#             hit_id   = int(hit_ids[i])
#             baseline = float(baselines[i])

#             samp_arr = np.asarray(samples_col[i].as_py(),  dtype=np.float64)
#             trig_arr = np.asarray(triggers_col[i].as_py(), dtype=np.int32)

#             cfd_offsets, cfd_slopes = compute_cfd_times_and_slopes(
#                 samp_arr=samp_arr,
#                 trig_arr=trig_arr,
#                 baseline=baseline,
#                 period=period,
#                 cfd_levels=CFD_LEVELS,
#                 min_amplitude=min_amplitude,
#                 edge_buffer_samples=edge_buffer_samples,
#                 consecutive_points=consecutive_points,
#             )

#             out["HITNumber"].append(hit_id)
#             for k in CFD_LEVELS:
#                 out[f"CFD{k}Offset"].append(cfd_offsets[k])
#                 out[f"CFD{k}Slope"].append(cfd_slopes[k])
#                 if np.isnan(cfd_offsets[k]):
#                     failures[k] += 1

#     # -------------------------------------------------------------------------
#     # Failure summary (once, after all hits)
#     # -------------------------------------------------------------------------
#     print("\nCFD crossing failures:")
#     for k in CFD_LEVELS:
#         pct = 100 * failures[k] / max(total, 1)
#         print(f"  CFD{k:2d}: {failures[k]:>8,} / {total:,}  ({pct:.2f}%)")

#     # -------------------------------------------------------------------------
#     # Write Arrow / Parquet
#     # -------------------------------------------------------------------------
#     table_dict: dict[str, pa.Array] = {
#         "HITNumber": pa.array(np.asarray(out["HITNumber"], dtype=np.int64), type=pa.int64())
#     }
#     for k in CFD_LEVELS:
#         table_dict[f"CFD{k}Offset"] = pa.array(
#             np.asarray(out[f"CFD{k}Offset"], dtype=np.float32), type=pa.float32()
#         )
#         table_dict[f"CFD{k}Slope"] = pa.array(
#             np.asarray(out[f"CFD{k}Slope"], dtype=np.float32), type=pa.float32()
#         )

#     table = pa.table(table_dict)
#     pq.write_table(table, output_path, compression="zstd")

#     print("\n============================================================")
#     print(f"CFD database written to: {output_path}")
#     print(f"Total hits : {total:,}")
#     print("Columns    :", ", ".join(table.column_names))
#     print("============================================================\n")

#     return output_path


# # =============================================================================
# # CFD LOOKUP LOADER
# # =============================================================================

# def load_cfd_lookup(
#     cfd_database_path: str,
# ):
#     """
#     Returns:

#         {
#             HITNumber: {
#                 "CFD10Offset": ...,
#                 "CFD10Slope": ...,
#                 ...
#             }
#         }
#     """

#     table = pq.read_table(cfd_database_path)

#     data = {}

#     hit_ids = (
#         table["HITNumber"]
#         .combine_chunks()
#         .to_numpy()
#     )

#     arrays = {}

#     for k in CFD_LEVELS:

#         arrays[f"CFD{k}Offset"] = (
#             table[f"CFD{k}Offset"]
#             .combine_chunks()
#             .to_numpy(zero_copy_only=False)
#         )

#         arrays[f"CFD{k}Slope"] = (
#             table[f"CFD{k}Slope"]
#             .combine_chunks()
#             .to_numpy(zero_copy_only=False)
#         )

#     for i, hit_id in enumerate(hit_ids):

#         row = {}

#         for k in CFD_LEVELS:

#             row[f"CFD{k}Offset"] = arrays[f"CFD{k}Offset"][i]
#             row[f"CFD{k}Slope"]  = arrays[f"CFD{k}Slope"][i]

#         data[int(hit_id)] = row

#     return data

# def _robust_langauss_seeds(x_fit, y_fit, mpv0):
#     """
#     Estimate eta and sigma from the LEFT-side HWHM of the peak
#     located near mpv0.

#     Important:
#     Do NOT use np.argmax(y_fit) globally, because the signal peak
#     may be smaller than a noise peak.
#     """

#     x_fit = np.asarray(x_fit, dtype=float)
#     y_fit = np.asarray(y_fit, dtype=float)

#     if len(x_fit) < 3:
#         raise ValueError("Not enough points for Landau-Gauss seeds")

#     # Index corresponding to the selected signal MPV
#     peak_idx = int(np.argmin(np.abs(x_fit - mpv0)))

#     peak_val = float(y_fit[peak_idx])

#     if peak_val <= 0:
#         raise ValueError("Selected peak has non-positive height")

#     half_max = 0.5 * peak_val

#     # Only inspect the LEFT side of the selected peak
#     left_x = x_fit[:peak_idx + 1]
#     left_y = y_fit[:peak_idx + 1]

#     crossings = np.where(left_y <= half_max)[0]

#     if len(crossings) > 0:
#         hwhm = mpv0 - left_x[crossings[-1]]
#     else:
#         # Fallback based on the available left-side range
#         if peak_idx > 0:
#             hwhm = mpv0 - left_x[0]
#         else:
#             hwhm = (x_fit[-1] - x_fit[0]) * 0.1

#     hwhm = max(hwhm, 1e-4)

#     eta0 = max(hwhm * 0.5, 1e-4)
#     sigma0 = max(hwhm * 0.3, 1e-5)

#     return eta0, sigma0, hwhm


# # =============================================================================
# # Main plotting function
# # =============================================================================

# def plot_Amplitude_histograms(
#     parquet_path: str,
#     Run: str,
#     channel_col: str = "Channel",
#     baseline_col: str = "Baseline",
#     peak_col: str = "RawPeak",
#     n_bins: int = 200,
#     x_range: Optional[tuple[float, float]] = None,
#     channel_filter: Optional[list[int]] = None,
#     figsize_per_plot: tuple[float, float] = (8, 5),

#     # Kept for backwards compatibility
#     fit_window: Optional[float] = None,

#     # Manual cut, if desired
#     Amplitude_cuts: Optional[dict[int, float]] = None,
#     Amplitude_signal_guesses: Optional[dict[int, float]] = None,
#     signal_guess_window: float = 0.1,

#     # Kept for backwards compatibility, but no longer used.
#     # The cut is now always 10% of the fitted Landau-Gauss maximum.
#     mpv_fraction: float = 0.10,

#     logy: bool = False,
#     label: str = "PPS2 Timing Preliminary",
#     rlabel: str = "(H8 Test Beam, May 2025)",
#     is_data: bool = True,

#     # -------------------------------------------------------------------------
#     # Peak-selection parameters
#     # -------------------------------------------------------------------------

#     # Fraction of the highest smoothed peak that a peak must reach
#     # to be considered significant.
#     min_peak_height_fraction: float = 0.02,

#     # Fraction of the highest smoothed peak used for prominence.
#     min_peak_prominence_fraction: float = 0.01,

#     # Minimum separation between candidate peaks in Amplitude.
#     min_peak_distance: float = 0.01,

# ) -> tuple[dict, dict]:

#     """
#     Build and plot per-channel Amplitude histograms.

#     Signal identification
#     ----------------------

#     The Amplitude distribution can contain both noise and real detector
#     signal.

#     The signal is selected according to the following rule:

#         1. Smooth the histogram.
#         2. Find significant peaks.
#         3. If there is one significant peak, use it as the signal peak.
#         4. If there are multiple significant peaks, use the RIGHTMOST
#            significant peak as the signal peak.

#     This deliberately does NOT assume that the highest peak is the signal.

#     Therefore, a noise population can contain substantially more events
#     than the signal population without determining the signal MPV.

#     The selected peak is then fitted with Landau ⊗ Gaussian.

#     The Amplitude cut is defined as the LEFT-side crossing of:

#         10% × maximum(Landau ⊗ Gaussian)

#     The noise population is not included when calculating this cut.

#     Parameters
#     ----------
#     fit_window:
#         Optional fixed right-side fit extent in Amplitude units.

#         If None, the fit extends to:

#             MPV + 3 × HWHM

#         where HWHM is estimated from the left side of the selected peak.

#     Amplitude_cut:
#         Optional manual cut. If supplied, this value is used instead
#         of the automatically determined 10% crossing.

#     mpv_fraction:
#         Retained for backwards compatibility. It is no longer used for
#         the automatic cut. The automatic cut is always 10% of the
#         fitted Landau-Gaussian maximum.

#     min_peak_height_fraction:
#         Minimum peak height relative to the highest smoothed peak.

#     min_peak_prominence_fraction:
#         Minimum peak prominence relative to the highest smoothed peak.

#     min_peak_distance:
#         Minimum distance between candidate peaks in Amplitude.
#     """

#     # =========================================================================
#     # Load scalar columns only
#     # =========================================================================

#     table = pq.read_table(
#         parquet_path,
#         columns=[channel_col, baseline_col, peak_col]
#     )

#     channels = (
#         table[channel_col]
#         .combine_chunks()
#         .to_numpy()
#     )

#     baselines = (
#         table[baseline_col]
#         .combine_chunks()
#         .to_numpy(zero_copy_only=False)
#         .astype(np.float64)
#     )

#     peaks_arr = (
#         table[peak_col]
#         .combine_chunks()
#         .to_numpy(zero_copy_only=False)
#         .astype(np.float64)
#     )

#     del table

#     # =========================================================================
#     # Calculate Amplitude
#     # =========================================================================

#     valid = baselines != 0

#     Amplitudes = np.where(
#         valid,
#         (peaks_arr - baselines) ,#/ baselines,
#         np.nan
#     )

#     unique_channels = sorted(
#         np.unique(channels).astype(int)
#     )

#     if channel_filter is not None:
#         unique_channels = [
#             ch for ch in unique_channels
#             if ch in channel_filter
#         ]

#     # =========================================================================
#     # Failure counters
#     # =========================================================================

#     fail_baseline_zero = defaultdict(int)
#     fail_nonfinite = defaultdict(int)
#     fail_no_entries = defaultdict(int)
#     fail_fit = defaultdict(int)
#     total_ch = defaultdict(int)

#     # =========================================================================
#     # Build histograms
#     # =========================================================================

#     results = {}

#     for ch in unique_channels:

#         mask = channels == ch

#         total_ch[ch] = int(mask.sum())

#         fail_baseline_zero[ch] = int(
#             ((channels == ch) & (~valid)).sum()
#         )

#         values = Amplitudes[mask]

#         n_nonfinite = int(
#             np.sum(~np.isfinite(values))
#         )

#         fail_nonfinite[ch] = n_nonfinite

#         values = values[np.isfinite(values)]

#         if len(values) == 0:
#             fail_no_entries[ch] += 1
#             continue

#         lo, hi = (
#             x_range
#             if x_range is not None
#             else (
#                 float(values.min()),
#                 float(values.max())
#             )
#         )

#         counts, bin_edges = np.histogram(
#             values,
#             bins=n_bins,
#             range=(lo, hi)
#         )

#         results[ch] = HistogramResult(
#             counts=counts,
#             bin_edges=bin_edges
#         )

#     # =========================================================================
#     # Plot layout
#     # =========================================================================

#     n = len(results)

#     if n == 0:
#         print("No valid channels found.")
#         return results, {}

#     ncols = min(2, n)
#     nrows = (n + ncols - 1) // ncols

#     fig, axes = plt.subplots(
#         nrows,
#         ncols,
#         figsize=(
#             figsize_per_plot[0] * ncols,
#             figsize_per_plot[1] * nrows
#         ),
#     )

#     axes = np.atleast_1d(axes).flatten()

#     thresholds = {}

#     # =========================================================================
#     # Channel loop
#     # =========================================================================

#     for ax, ch in tqdm(zip(
#         axes,
#         sorted(results))
#     ):

#         result = results[ch]

#         centres = result.bin_centres
#         counts = result.counts.astype(float)

#         bw = (
#             result.bin_edges[1]
#             - result.bin_edges[0]
#         )

#         # ---------------------------------------------------------------------
#         # Histogram
#         # ---------------------------------------------------------------------

#         ax.bar(
#             centres,
#             counts,
#             width=bw,
#             align="center",
#             alpha=0.7,
#             label="Data"
#         )

#         # ---------------------------------------------------------------------
#         # Only non-empty bins for fitting
#         # ---------------------------------------------------------------------

#         fit_mask = counts > 0

#         x_fit_full = centres[fit_mask]
#         y_fit_full = counts[fit_mask]

#         if len(x_fit_full) < 10:

#             ax.set_title(
#                 f"Channel {ch} — too few entries"
#             )

#             thresholds[ch] = np.nan

#             continue

#         # =========================================================================
#         # STEP 1 — Smooth histogram
#         # =========================================================================

#         y_smooth = gaussian_filter1d(
#             counts.astype(float),
#             sigma=2
#         )

#         max_smooth = float(
#             np.max(y_smooth)
#         )

#         # =========================================================================
#         # STEP 2 — Find significant peaks
#         # =========================================================================

#         min_prominence = max(
#             max_smooth * min_peak_prominence_fraction,
#             1.0
#         )

#         min_height = max(
#             max_smooth * min_peak_height_fraction,
#             1.0
#         )

#         peak_distance_bins = max(
#             1,
#             int(
#                 min_peak_distance / bw
#             )
#         )

#         peaks_idx, peak_properties = find_peaks(
#             y_smooth,
#             prominence=min_prominence,
#             height=min_height,
#             distance=peak_distance_bins,
#         )

#         # =========================================================================
#         # STEP 3 — Select signal peak
#         # =========================================================================

#         signal_guess_used = False

#         if (
#             Amplitude_signal_guesses is not None
#             and ch in Amplitude_signal_guesses
#         ):

#             # ---------------------------------------------------------------------
#             # Manual signal-location hint
#             #
#             # We don't force the peak to be exactly at the supplied value.
#             # Instead, search for the local maximum around that value.
#             # ---------------------------------------------------------------------

#             guess = float(
#                 Amplitude_signal_guesses[ch]
#             )

#             guess_mask = (
#                 (centres >= guess - signal_guess_window)
#                 &
#                 (centres <= guess + signal_guess_window)
#             )

#             guess_indices = np.where(guess_mask)[0]

#             if len(guess_indices) > 0:

#                 local_idx = guess_indices[
#                     np.argmax(
#                         y_smooth[guess_indices]
#                     )
#                 ]

#                 signal_peak_idx = int(local_idx)

#                 signal_guess_used = True

#             else:

#                 print(
#                     f"  Channel {ch}: "
#                     f"signal guess {guess:.4f} is outside "
#                     f"histogram range"
#                 )

#                 signal_peak_idx = None

#         else:

#             signal_peak_idx = None


#         # =========================================================================
#         # Automatic peak finding if no manual hint was supplied
#         # =========================================================================

#         if signal_peak_idx is None:

#             if len(peaks_idx) == 0:

#                 print(
#                     f"  Channel {ch}: "
#                     f"NO SIGNIFICANT PEAK"
#                 )

#                 thresholds[ch] = np.nan

#                 # ... existing no-signal handling ...

#                 continue

#             # Rightmost significant peak
#             signal_peak_idx = int(
#                 peaks_idx[-1]
#             )


#         mpv0 = float(
#             centres[signal_peak_idx]
#         )

#         print(
#             f"  Channel {ch}: "
#             f"signal peak = {mpv0:.4f}"
#             + (
#                 " (manual search)"
#                 if signal_guess_used
#                 else " (automatic)"
#             )
#         )

        

#         # ---------------------------------------------------------------------
#         # Mark selected signal peak on plot
#         # ---------------------------------------------------------------------

#         ax.axvline(
#             mpv0,
#             color="green",
#             lw=1.0,
#             ls=":",
#             alpha=0.7,
#         )

#         # =========================================================================
#         # STEP 4 — Build a local fit region around selected peak
#         # =========================================================================

#         # We need the selected peak to determine the HWHM.
#         # Therefore we first use all populated bins to estimate the local
#         # HWHM, but the helper explicitly works relative to mpv0 rather
#         # than using the global maximum.

#         try:

#             eta0, sigma0, hwhm = (
#                 _robust_langauss_seeds(
#                     x_fit_full,
#                     y_fit_full,
#                     mpv0
#                 )
#             )

#         except Exception as e:

#             print(
#                 f"  Ch {ch}: "
#                 f"failed to estimate Landau-Gauss seeds: {e}"
#             )

#             fail_fit[ch] += 1
#             thresholds[ch] = np.nan

#             continue

#         # =========================================================================
#         # STEP 5 — Fit window
#         # =========================================================================

#         if fit_window is not None:

#             left_lim = mpv0 - fit_window
#             right_lim = mpv0 + fit_window

#         else:

#             # We want enough of the rising edge to constrain the signal,
#             # but do not want the entire Landau tail to dominate the fit.
#             left_lim = mpv0 - 2.5 * hwhm
#             right_lim = mpv0 + 3.0 * hwhm

#         fit_mask2 = (
#             (x_fit_full >= left_lim)
#             & (x_fit_full <= right_lim)
#         )

#         x_fit = x_fit_full[fit_mask2]
#         y_fit = y_fit_full[fit_mask2]

#         if len(x_fit) < 10:

#             print(
#                 f"  Ch {ch}: "
#                 f"fit window too narrow"
#             )

#             fail_fit[ch] += 1
#             thresholds[ch] = np.nan

#             continue

#         # =========================================================================
#         # STEP 6 — Fit Landau ⊗ Gaussian
#         # =========================================================================

#         sigma_y = np.sqrt(
#             np.maximum(
#                 y_fit,
#                 1.0
#             )
#         )

#         try:

#             popt, pcov = curve_fit(
#                 langauss,
#                 x_fit,
#                 y_fit,

#                 p0=[
#                     mpv0,
#                     eta0,
#                     sigma0,
#                     max(y_fit),
#                     0.0,
#                 ],

#                 sigma=sigma_y,
#                 absolute_sigma=True,

#                 bounds=(
#                     [
#                         x_fit.min(),
#                         1e-5,
#                         1e-5,
#                         0.0,
#                         0.0,
#                     ],

#                     [
#                         x_fit.max(),
#                         max(hwhm, 1e-5),
#                         max(hwhm, 1e-5),
#                         np.inf,
#                         np.inf,
#                     ],
#                 ),

#                 maxfev=50_000,
#             )

#             perr = np.sqrt(
#                 np.maximum(
#                     np.diag(pcov),
#                     0
#                 )
#             )

#             y_model = langauss(
#                 x_fit,
#                 *popt
#             )

#             chi2 = float(
#                 np.sum(
#                     (
#                         (y_fit - y_model)
#                         / sigma_y
#                     ) ** 2
#                 )
#             )

#             ndf = (
#                 len(x_fit)
#                 - len(popt)
#             )

#             # =========================================================================
#             # STEP 7 — Draw fitted signal over full display range
#             # =========================================================================

#             xd = np.linspace(
#                 centres.min(),
#                 centres.max(),
#                 2000
#             )

#             yd = langauss(
#                 xd,
#                 *popt
#             )

#             ax.plot(
#                 xd,
#                 yd,
#                 "r--",
#                 lw=2,
#                 label="Landau*Gauss"
#             )


#             # =========================================================================
#             # STEP 8 — Determine Amplitude cut
#             # =========================================================================

#             # The last parameter of langauss() is the fitted background/offset.
#             #
#             # For the cut we DO NOT want the background.
#             # We therefore evaluate the fitted signal with background = 0.

#             mpv_fit = popt[0]
#             eta_fit = popt[1]
#             sigma_fit = popt[2]
#             A_fit = popt[3]

#             yd_signal = langauss(
#                 xd,
#                 mpv_fit,
#                 eta_fit,
#                 sigma_fit,
#                 A_fit,
#                 0.0,          # remove fitted background
#             )

#             ax.plot(
#                 xd,
#                 yd_signal,
#                 "r-",
#                 lw=1.5,
#                 alpha=0.8,
#                 label="Signal component"
#             )
#             # Peak of the SIGNAL component
#             signal_peak_idx = int(
#                 np.argmax(yd_signal)
#             )

#             signal_peak_x = float(
#                 xd[signal_peak_idx]
#             )

#             signal_peak_y = float(
#                 yd_signal[signal_peak_idx]
#             )

#             # -------------------------------------------------------------------------
#             # Automatic cut:
#             #
#             # 10% of the SIGNAL peak height, NOT 10% of the total fit.
#             # -------------------------------------------------------------------------

#             target = 0.10 * signal_peak_y

#             left_side = np.where(
#                 yd_signal[:signal_peak_idx] <= target
#             )[0]

#             if len(left_side) > 0:

#                 auto_thr = float(
#                     xd[left_side[-1]]
#                 )

#             else:

#                 auto_thr = float(
#                     xd[0]
#                 )

#             # -------------------------------------------------------------------------
#             # Manual channel-specific override
#             # -------------------------------------------------------------------------

#             if (
#                 Amplitude_cuts is not None
#                 and ch in Amplitude_cuts
#             ):

#                 thr = float(
#                     Amplitude_cuts[ch]
#                 )

#                 cut_source = "manual"

#             else:

#                 thr = auto_thr

#                 cut_source = "automatic"

#             thresholds[ch] = thr

#             # =========================================================================
#             # STEP 9 — Draw cut
#             # =========================================================================

#             ax.axvline(
#                 thr,
#                 color="orange",
#                 lw=1.5,
#                 ls="--",
#                 label=f"Cut = {thr:.3f}"
#             )

#             # =========================================================================
#             # STEP 10 — Diagnostics
#             # =========================================================================

#             ax.text(
#                 0.97,
#                 0.97,

#                 f"MPV = {popt[0]:.4f} ± {perr[0]:.4f}\n"
#                 f"η   = {popt[1]:.4f} ± {perr[1]:.4f}\n"
#                 f"σ   = {popt[2]:.4f} ± {perr[2]:.4f}\n"
#                 f"χ²/ndf = {chi2:.1f}/{ndf} "
#                 f"= {chi2 / max(ndf, 1):.2f}",

#                 transform=ax.transAxes,
#                 ha="right",
#                 va="top",
#                 fontsize=9,

#                 bbox=dict(
#                     boxstyle="round",
#                     facecolor="white",
#                     alpha=0.85,
#                 ),
#             )
            
#             print(
#                 f"  Channel {ch}: "
#                 f"MPV={popt[0]:.4f} | "
#                 f"auto cut Amplitude >{auto_thr:.4f}V| "
#                 f"final cut Amplitude > {thr:.4f}V"
#             )

#         except Exception as e:

#             print(
#                 f"  Ch {ch} fit failed: {e}"
#             )

#             fail_fit[ch] += 1
#             thresholds[ch] = np.nan

#         # =========================================================================
#         # Axes
#         # =========================================================================

#         ax.set_title(
#             f"Channel {ch}",
#             fontsize=18
#         )

#         ax.set_xlabel(
#             "Amplitude (V)",
#             fontsize=18
#         )

#         ax.set_ylabel(
#             "Counts",
#             fontsize=18
#         )

#         hep.cms.label(
#             label,
#             data=is_data,
#             rlabel=rlabel,
#             loc=0,
#             ax=ax,
#             fontsize=(14, 12, 12, 11),
#         )

#         ax.legend(fontsize=10)

#         if logy:
#             ax.set_yscale("log")

#         ax.grid(alpha=0.3)

#     # =========================================================================
#     # Hide unused axes
#     # =========================================================================

#     for ax in axes[n:]:
#         ax.set_visible(False)

#     # =========================================================================
#     # Figure
#     # =========================================================================

#     fig.suptitle(
#         f"Amplitude distribution per channel - {Run}",
#         fontsize=14
#     )

#     plt.tight_layout()
#     plt.show()

#     # =========================================================================
#     # Print thresholds
#     # =========================================================================

#     print("\nSuggested Amplitude cut thresholds:")

#     for ch, thr in sorted(thresholds.items()):

#         if np.isfinite(thr):

#             print(
#                 f"  Channel {ch}: "
#                 f"Amplitude > {thr:.4f} V({cut_source})"
#             )

#         else:

#             print(
#                 f"  Channel {ch}: "
#                 f"NO RELIABLE CUT"
#             )

#     return results, thresholds


# def compute_rise_time(
#     samp_arr: np.ndarray,
#     trig_arr: np.ndarray,
#     baseline: float,
#     period: float,
#     low_frac: float = 0.1,
#     high_frac: float = 0.9,
#     smooth_window: int = 2,
# ) -> Optional[float]:
#     """
#     Compute rise time between low_frac and high_frac of the amplitude.

#     More robust than the naive version:
#     - Optionally smooths the waveform before threshold crossing search,
#       to avoid noise-induced fake crossings (original samples never modified).
#     - Searches for the LAST crossing of lo_level before the peak, and the
#       LAST crossing of hi_level before the peak — this correctly handles
#       noisy rising edges that cross the threshold multiple times.
#     - Does not reject on dy <= 0: uses the crossing point even if the
#       local slope is flat, as long as the level is bracketed.
#     - Falls back gracefully if smoothing doesn't help.

#     Parameters
#     ----------
#     smooth_window : number of samples for moving-average pre-smoothing
#                     used ONLY for finding crossings (1 = no smoothing).
#     """
#     _, samp_ord, _ = reorder_circular_samples_with_trigger(
#         trig_arr, samp_arr, reorder_samples=False
#     )
#     samp_ord = np.asarray(samp_ord, dtype=np.float64)
#     n        = len(samp_ord)


#     # --- Robust peak finding: use smoothed signal to find peak region,
#     #     then take argmax of raw signal in that neighbourhood ----------
#     if smooth_window > 1:
#         kernel    = np.ones(smooth_window) / smooth_window
#         samp_smooth = np.convolve(samp_ord, kernel, mode="same")
#     else:
#         samp_smooth = samp_ord

#     peak_idx  = int(np.argmax(samp_smooth))
#     peak_val  = float(samp_ord[peak_idx])   # amplitude from raw sample at peak
#     amplitude = peak_val - baseline

#     if amplitude <= 0:
#         return None

#     lo_level = baseline + low_frac  * amplitude
#     hi_level = baseline + high_frac * amplitude

#     # Sanity check: levels must be within the actual sample range
#     if lo_level >= peak_val or hi_level >= peak_val:
#         return None

#     # --- Find crossings on the smoothed signal (more stable),
#     #     but interpolate crossing time using raw samples --------------

#     # We want the LAST crossing before the peak for each level,
#     # so the search uses the final upward crossing — correct even if
#     # the signal crosses the threshold multiple times on the way up.
#     t_lo = t_hi = None

#     for i in range(peak_idx):
#         y0_raw = float(samp_ord[i])
#         y1_raw = float(samp_ord[i + 1])
#         y0_smo = float(samp_smooth[i])
#         y1_smo = float(samp_smooth[i + 1])

#         # lo_level crossing — keep updating so we get the LAST one
#         if y0_smo <= lo_level <= y1_smo:
#             dy = y1_raw - y0_raw
#             if abs(dy) > 0:
#                 t_lo = (i + (lo_level - y0_raw) / dy) * period
#             else:
#                 t_lo = i * period   # flat crossing: use left edge

#         # hi_level crossing — keep updating so we get the LAST one
#         if y0_smo <= hi_level <= y1_smo:
#             dy = y1_raw - y0_raw
#             if abs(dy) > 0:
#                 t_hi = (i + (hi_level - y0_raw) / dy) * period
#             else:
#                 t_hi = i * period

#     if t_lo is None or t_hi is None:
#         return None

#     rt = t_hi - t_lo
#     # Sanity: rise time must be positive and physically reasonable
#     # (cannot be longer than half the waveform)
#     if rt <= 0 or rt > (n // 2) * period:
#         return None

#     return rt


# from typing import Dict, List, Optional, Tuple, Union
# from pathlib import Path
# from collections import defaultdict
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from tqdm import tqdm
# import mplhep as hep

# def plot_rise_time_histograms(
#     parquet_path: str,
#     Amplitude_cuts: Dict[int, float],
#     apply_Amp_filter: bool = False,
#     channel_filter: Optional[List[int]] = None,
#     period: float = 1 / 6.4,
#     channel_col: str = "Channel",
#     baseline_col: str = "Baseline",
#     peak_col: str = "RawPeak",
#     batch_size: int = 100_000,
#     root_tree: str = "sampic_hits",
#     low_frac: float = 0.1,
#     high_frac: float = 0.9,
#     n_bins: int = 100,
#     sigma_cut: Tuple[float, float] = (2.0, 2.0),
#     x_range: Optional[Tuple[float, float]] = None,
#     figsize_per_plot: Tuple[float, float] = (8, 5),
#     logy: bool = False,
#     label: str = "PPS2 Timing Preliminary",
#     rlabel: str = "(H8 Test Beam, May 2025)",
#     is_data: bool = True,
#     fit_guesses: Optional[Dict[int, Union[float, Tuple[float, float]]]] = None,  # NEW PARAMETER
# ) -> Tuple[Dict[int, "HistogramResult"], Dict[int, Tuple[float, float]]]:
#     """
#     Stream waveforms, apply per-channel Amplitude cuts, compute rise times,
#     and plot per-channel histograms with sigma-cut threshold markers.
#     """
#     parquet_path = Path(parquet_path)

#     cols = [channel_col, baseline_col, peak_col, "DataSample", "TriggerPosition"]
#     batches = open_hit_reader(parquet_path, cols=cols, batch_size=batch_size, root_tree=root_tree)

#     thresholds = {}
#     rise_times: Dict[int, list] = {}
#     skipped_Amplitude = 0
#     skipped_rt   = 0
#     total        = 0
#     skipped_Amplitude_ch = defaultdict(int)
#     skipped_rt_ch       = defaultdict(int)
#     total_ch            = defaultdict(int)

#     for batch in tqdm(batches, desc="Computing rise times"):
        
#         # Early Channel Filter
#         if channel_filter is not None:
#             channels_batch = batch[channel_col].to_numpy()
#             keep_ch = np.isin(channels_batch, channel_filter)
#             if not np.any(keep_ch):
#                 continue
#             batch = batch.take(pa.array(np.where(keep_ch)[0]))

#         channels  = batch[channel_col].to_numpy()
#         baselines = batch[baseline_col].to_numpy(zero_copy_only=False).astype(np.float64)
#         peaks     = batch[peak_col].to_numpy(zero_copy_only=False).astype(np.float64)
#         samples   = batch["DataSample"]
#         triggers  = batch["TriggerPosition"]

#         for i in range(len(channels)):
#             total += 1
#             ch = int(channels[i])
#             total_ch[ch] += 1
#             bl = float(baselines[i])
#             pk = float(peaks[i])

#             if apply_Amp_filter:
#                 if bl != 0:
#                     Amplitude = (pk - bl)
#                 else:
#                     skipped_Amplitude += 1
#                     skipped_Amplitude_ch[ch] += 1
#                     continue

#                 cut = Amplitude_cuts.get(ch, 0.0)
#                 if Amplitude < cut:
#                     skipped_Amplitude += 1
#                     skipped_Amplitude_ch[ch] += 1
#                     continue

#             rt = compute_rise_time(
#                 np.asarray(samples[i].as_py(), dtype=np.float64),
#                 np.asarray(triggers[i].as_py(), dtype=np.int32),
#                 bl, period, low_frac, high_frac,
#             )
#             if rt is None:
#                 skipped_rt += 1
#                 skipped_rt_ch[ch] += 1
#                 continue

#             rise_times.setdefault(ch, []).append(rt)

#     print(f"Total hits     : {total:,}")
#     if apply_Amp_filter:
#         print(f"Skipped (Amplitude)  : {skipped_Amplitude:,} ({100*skipped_Amplitude/max(total,1):.1f}%)")
#     print(f"Skipped (no RT): {skipped_rt:,} ({100*skipped_rt/max(total,1):.1f}%)")

#     if not rise_times:
#         raise RuntimeError("No rise times computed — check Amplitude cuts, filter, and data.")

#     # Build histograms
#     results = {}
#     for ch, values in rise_times.items():
#         arr = np.array(values, dtype=np.float64)
#         lo, hi = x_range if x_range is not None else (arr.min(), arr.max())
#         counts, bin_edges = np.histogram(arr, bins=n_bins, range=(lo, hi))
#         results[ch] = HistogramResult(counts=counts, bin_edges=bin_edges)

#     # Plotting setup
#     channels_sorted = sorted(results)
#     n     = len(channels_sorted)
#     ncols = min(2, n)
#     nrows = (n + ncols - 1) // ncols

#     fig, axes = plt.subplots(
#         nrows, ncols,
#         figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
#     )
#     axes = np.array(axes).flatten()

#     for ax, ch in zip(axes, channels_sorted):
#         result  = results[ch]
#         centres = result.bin_centres
#         bw      = result.bin_edges[1] - result.bin_edges[0]

#         counts = result.counts.astype(float)
#         ax.bar(centres, counts, width=bw, align="center", alpha=0.75, label="Data")

#         # Gaussian fit
#         fit_mask = counts > 0
#         x_fit = centres[fit_mask]
#         y_fit = counts[fit_mask]

#         if len(x_fit) > 5:
#             # Default automated initial guesses
#             peak_bin = np.argmax(y_fit)
#             mu0      = x_fit[peak_bin]
#             sigma0   = 0.2 * (x_fit.max() - x_fit.min())
#             A0       = y_fit.max()
#             B0       = 0.0

#             # =====================================================================
#             # NEW LOGIC: Override initial guess for specified channels
#             # =====================================================================
#             if fit_guesses and ch in fit_guesses:
#                 guess = fit_guesses[ch]
#                 if isinstance(guess, (tuple, list)):
#                     mu0 = float(guess[0])
#                     if len(guess) > 1:
#                         sigma0 = float(guess[1])
#                 else:
#                     mu0 = float(guess)
#                     sigma0 = 0.05  # Tighter default width (in ns) for signal peak

#                 # Calculate local count height around mu0 for initial amplitude A0
#                 near_idx = np.argmin(np.abs(x_fit - mu0))
#                 A0 = float(y_fit[near_idx])

#             sigma_y = np.sqrt(np.maximum(y_fit, 1.0))

#             try:
#                 popt, pcov = curve_fit(
#                     gaussian, x_fit, y_fit,
#                     p0=[A0, mu0, sigma0, B0],
#                     sigma=sigma_y, absolute_sigma=True,
#                     bounds=([0, x_fit.min(), 1e-6, -np.inf],
#                             [np.inf, x_fit.max(), np.inf,  np.inf]),
#                     maxfev=20000,
#                 )
#                 perr = np.sqrt(np.diag(pcov))

#                 y_model = gaussian(x_fit, *popt)
#                 chi2    = float(np.sum(((y_fit - y_model) / sigma_y)**2))
#                 ndf     = len(x_fit) - len(popt)

#                 xd = np.linspace(centres.min(), centres.max(), 2000)
#                 ax.plot(xd, gaussian(xd, *popt), "r--", lw=2, label="Gaussian fit")

#                 ax.text(
#                     0.97, 0.97,
#                     f"μ = {popt[1]:.3f} ± {perr[1]:.4f}\n"
#                     f"σ = {popt[2]:.3f} ± {perr[2]:.4f}\n"
#                     f"χ²/ndf = {chi2:.1f}/{ndf} = {chi2/max(ndf,1):.2f}",
#                     transform=ax.transAxes, ha="right", va="top",
#                     fontsize=9,
#                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
#                 )

#                 if (popt[1] is not None) and (popt[2] is not None):
#                     thr_low = popt[1] - sigma_cut[0] * popt[2]
#                     thr_high = popt[1] + sigma_cut[1] * popt[2]
#                     thresholds[ch] = (thr_low, thr_high)

#                     ax.axvline(thr_low, color="darkorange", linestyle="-.", lw=1.5, 
#                                label=f"+{sigma_cut[1]}σ -{sigma_cut[0]}σ cut thresholds")
#                     ax.axvline(thr_high, color="darkorange", linestyle="-.", lw=1.5)

#             except Exception as e:
#                 print(f"Gaussian fit failed for channel {ch}: {e}")

#         ax.set_title(f"Channel {ch}", fontsize = 18)
#         ax.set_xlabel(f"Rise time {int(low_frac*100)}%–{int(high_frac*100)}% (ns)", fontsize = 18)
#         ax.set_ylabel("Counts", fontsize = 18)
#         hep.cms.label(label, data=is_data, rlabel=rlabel, loc=0, ax=ax, fontsize=(14, 12, 12, 11))
#         ax.legend(fontsize=10)

#         if logy:
#             ax.set_yscale("log")
#         ax.grid(alpha=0.3)

#     for ax in axes[n:]:
#         ax.set_visible(False)

#     fig.suptitle(
#         f"Rise time {int(low_frac*100)}%–{int(high_frac*100)}%  "
#         f"(Amplitude-filtered)", fontsize=14
#     )
#     plt.tight_layout()
#     plt.show()

#     print("\nSuggested Rise-Time cut thresholds:")
#     for ch, thr in sorted(thresholds.items()):
#         print(f"  Channel {ch}: {thr[0]:.4f} < RT < {thr[1]:.4f} ns")
        
#     return results, thresholds



# hep.style.use("CMS")

# def snr_composite_model(x, mpv, eta, sigma_lg, A_lg, A_n, mu_n, sigma_n):
#     return langauss(x, mpv, eta, sigma_lg, A_lg, 0.0) + gaussian(x, A_n, mu_n, sigma_n, 0.0)


# # -----------------------------------------------------------------------------
# # PDF models
# # -----------------------------------------------------------------------------

# def noise_emg(x: np.ndarray, A: float, K: float, loc: float, scale: float) -> np.ndarray:
#     """Exponentially modified Gaussian (Gaussian ⊗ Exponential)."""
#     return A * exponnorm.pdf(x, K, loc=loc, scale=scale)


# def signal_moyal(x: np.ndarray, A: float, loc: float, scale: float) -> np.ndarray:
#     """Stable Landau-like signal model."""
#     return A * moyal.pdf(x, loc=loc, scale=scale)


# def composite_model(
#     x: np.ndarray,
#     A_n: float,
#     K: float,
#     loc_n: float,
#     scale_n: float,
#     A_s: float,
#     loc_s: float,
#     scale_s: float,
# ) -> np.ndarray:
#     return noise_emg(x, A_n, K, loc_n, scale_n) + signal_moyal(x, A_s, loc_s, scale_s)


# # -----------------------------------------------------------------------------
# # Helpers
# # -----------------------------------------------------------------------------

# def compute_noise_rms(
#     samp_arr: np.ndarray,
#     trig_arr: np.ndarray,
#     baseline: float,
#     n_noise_samples: int = 20,
# ) -> float:
#     """Compute baseline noise RMS from the first ordered samples."""
#     _, samp_ord, _ = reorder_circular_samples_with_trigger(
#         trig_arr, samp_arr, reorder_samples=False,
#     )
#     samp_ord = np.asarray(samp_ord, dtype=np.float64)
#     n_use = min(n_noise_samples, len(samp_ord))
#     if n_use <= 1:
#         return np.nan

#     noise_region = samp_ord[:n_use] - baseline
#     return float(np.sqrt(np.mean(noise_region**2)))


# def _safe_percentile(arr: np.ndarray, q: float, default: float) -> float:
#     arr = np.asarray(arr, dtype=float)
#     arr = arr[np.isfinite(arr)]
#     if arr.size == 0:
#         return default
#     return float(np.percentile(arr, q))


# def _detect_peaks(
#     centres: np.ndarray,
#     counts: np.ndarray,
#     prominence_frac: float = 0.03,
#     smooth_sigma: float = 2.0,
#     min_distance_bins: int = 5,
# ) -> np.ndarray:
#     smoothed = gaussian_filter1d(counts.astype(float), sigma=smooth_sigma)
#     prom = max(prominence_frac * float(np.max(smoothed)), 1.0)
#     peaks, _ = find_peaks(smoothed, prominence=prom, distance=min_distance_bins)
#     return peaks


# def _choose_noise_signal_peaks(
#     centres: np.ndarray,
#     counts: np.ndarray,
#     peaks: np.ndarray,
#     noise_x_max: float = 15.0,
# ) -> Tuple[Optional[int], Optional[int]]:
#     """Pick a left noise peak and a right signal peak robustly.

#     Preference order:
#       1) Any peak at x <= noise_x_max is noise, choose the tallest among them.
#       2) Signal is the tallest peak at x > noise peak (or the global tallest right-side peak).
#       3) Fallbacks when peak finding is sparse.
#     """
#     if len(peaks) == 0:
#         return None, None

#     # Left noise peak: prefer a real peak at low x.
#     left = peaks[centres[peaks] <= noise_x_max]
#     if len(left) > 0:
#         noise_idx = int(left[np.argmax(counts[left])])
#     else:
#         noise_idx = int(peaks[np.argmin(centres[peaks])])

#     # Right signal peak: prefer the tallest peak to the right of the noise peak.
#     right = peaks[centres[peaks] > centres[noise_idx] + 2.0]
#     if len(right) > 0:
#         signal_idx = int(right[np.argmax(counts[right])])
#     else:
#         # If no distinct right peak exists, take the global max after the noise region.
#         mask = centres > max(noise_x_max, centres[noise_idx] + 2.0)
#         if np.any(mask):
#             signal_idx = int(np.argmax(counts * mask))
#             if counts[signal_idx] == 0:
#                 signal_idx = int(peaks[np.argmax(counts[peaks])])
#         else:
#             signal_idx = int(peaks[np.argmax(counts[peaks])])

#     if signal_idx == noise_idx:
#         # Final fallback: use the strongest peak and a nearby neighbour if possible.
#         order = peaks[np.argsort(centres[peaks])]
#         noise_idx = int(order[0])
#         signal_idx = int(order[-1])

#     return noise_idx, signal_idx


# def _valley_between_peaks(
#     centres: np.ndarray,
#     counts: np.ndarray,
#     noise_idx: int,
#     signal_idx: int,
# ) -> Tuple[float, int]:
#     """Valley from the smoothed histogram between two peak indices."""
#     lo = min(noise_idx, signal_idx)
#     hi = max(noise_idx, signal_idx)
#     if hi - lo < 2:
#         return float(centres[lo]), lo

#     smoothed = gaussian_filter1d(counts.astype(float), sigma=1.5)
#     region = smoothed[lo : hi + 1]
#     valley_rel = int(np.argmin(region))
#     valley_idx = lo + valley_rel
#     return float(centres[valley_idx]), valley_idx


# def _fit_noise_only(
#     centres: np.ndarray,
#     counts: np.ndarray,
#     valley_x: float,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Fit the noise component on the left side of the valley."""
#     mask = (centres >= centres.min()) & (centres <= valley_x)
#     x = centres[mask]
#     y = counts[mask].astype(float)

#     if x.size < 6 or np.sum(y) <= 0:
#         raise RuntimeError("Insufficient data for noise fit")

#     peak_idx = int(np.argmax(y))
#     loc0 = float(x[peak_idx])
#     scale0 = max(float(np.std(np.repeat(x, np.maximum(y.astype(int), 1)))), 1.0)
#     K0 = 1.5
#     A0 = float(np.max(y)) / max(float(exponnorm.pdf(loc0, K0, loc=loc0, scale=max(scale0, 1.0))), 1e-8)

#     p0 = [A0, K0, loc0, scale0]
#     bounds = ([0.0, 0.05, 0.0, 0.2], [np.inf, 50.0, max(60.0, float(np.max(x)) + 5.0), 80.0])

#     popt, pcov = curve_fit(
#         noise_emg,
#         x,
#         y,
#         p0=p0,
#         bounds=bounds,
#         sigma=np.sqrt(np.maximum(y, 1.0)),
#         absolute_sigma=True,
#         maxfev=20_000,
#     )
#     return popt, pcov


# def _fit_signal_only(
#     centres: np.ndarray,
#     counts: np.ndarray,
#     valley_x: float,
#     noise_params: Optional[np.ndarray] = None,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Fit the signal component on the right side of the valley.

#     If noise_params are provided, subtract the noise model before fitting.
#     """
#     mask = centres >= valley_x
#     x = centres[mask]
#     y = counts[mask].astype(float)

#     if x.size < 6 or np.sum(y) <= 0:
#         raise RuntimeError("Insufficient data for signal fit")

#     if noise_params is not None:
#         y = np.clip(y - noise_emg(x, *noise_params), 0.0, None)

#     if np.sum(y) <= 0:
#         raise RuntimeError("Signal residual is empty after noise subtraction")

#     peak_idx = int(np.argmax(y))
#     loc0 = float(x[peak_idx])
#     scale0 = max(float(np.std(np.repeat(x, np.maximum(y.astype(int), 1)))), 1.0)
#     A0 = float(np.max(y)) / max(float(moyal.pdf(loc0, loc=loc0, scale=max(scale0, 1.0))), 1e-8)

#     p0 = [A0, loc0, scale0]
#     bounds = ([0.0, 0.0, 0.2], [np.inf, 200.0, 100.0])

#     popt, pcov = curve_fit(
#         signal_moyal,
#         x,
#         y,
#         p0=p0,
#         bounds=bounds,
#         sigma=np.sqrt(np.maximum(y, 1.0)),
#         absolute_sigma=True,
#         maxfev=20_000,
#     )
#     return popt, pcov


# def _fit_composite(
#     centres: np.ndarray,
#     counts: np.ndarray,
#     noise_popt: np.ndarray,
#     signal_popt: np.ndarray,
#     fit_max_x: float,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Fit the full composite model using the per-component fits as seeds."""
#     mask = (centres >= centres.min()) & (centres <= fit_max_x)
#     x = centres[mask]
#     y = counts[mask].astype(float)

#     if x.size < 8 or np.sum(y) <= 0:
#         raise RuntimeError("Insufficient data for composite fit")

#     A_n, K, loc_n, scale_n = noise_popt
#     A_s, loc_s, scale_s = signal_popt

#     p0 = [A_n, K, loc_n, scale_n, A_s, loc_s, scale_s]
#     bounds = (
#         [0.0, 0.05, 0.0, 0.2, 0.0, 0.0, 0.2],
#         [np.inf, 80.0, max(80.0, fit_max_x), 120.0, np.inf, 250.0, 120.0],
#     )

#     popt, pcov = curve_fit(
#         composite_model,
#         x,
#         y,
#         p0=p0,
#         bounds=bounds,
#         sigma=np.sqrt(np.maximum(y, 1.0)),
#         absolute_sigma=True,
#         maxfev=40_000,
#     )
#     return popt, pcov


# def _intersection_threshold(
#     x: np.ndarray,
#     noise_y: np.ndarray,
#     signal_y: np.ndarray,
#     fallback: float,
# ) -> float:
#     diff = noise_y - signal_y
#     sgn = np.sign(diff)
#     idx = np.where(np.diff(sgn) != 0)[0]
#     if len(idx) == 0:
#         return float(fallback)

#     # Take the first crossing to the right of the noise peak, if possible.
#     i = int(idx[0])
#     x0, x1 = x[i], x[i + 1]
#     y0, y1 = diff[i], diff[i + 1]
#     if y1 == y0:
#         return float(x0)
#     return float(x0 - y0 * (x1 - x0) / (y1 - y0))


# # -----------------------------------------------------------------------------
# # Main plotting function
# # -----------------------------------------------------------------------------

 
# # -----------------------------------------------------------------------------
# # SNR histograms — streamed straight from the raw parquet, with NO Amplitude
# # or rise-time filtering applied, so the SNR threshold is determined from
# # the channel's natural, unfiltered SNR distribution — independently of the
# # Amplitude and rise-time thresholds. All three are combined only afterwards,
# # in build_hit_mask.
# # -----------------------------------------------------------------------------
 
# def plot_SNR_histograms(
#     parquet_path: str,
#     channel_col: str = "Channel",
#     baseline_col: str = "Baseline",
#     peak_col: str = "RawPeak",
#     channel_filter: Optional[list[int]] = None,
#     n_noise_samples: int = 20,
#     batch_size: int = 100_000,
#     root_tree: str = "sampic_hits",
#     n_bins: int = 120,
#     x_range: Optional[Tuple[float, float]] = (0.0, 200.0),
#     sigma_cut: float = 2.5,
#     figsize_per_plot: Tuple[float, float] = (8, 5),
#     logy: bool = False,
#     label: str = "PPS2 Timing Preliminary",
#     rlabel: str = "(H8 Test Beam, May 2025)",
#     is_data: bool = True,
# ) -> Tuple[Dict[int, np.ndarray], Dict[int, Tuple[float, float]]]:
#     """Plot per-channel SNR histograms with more robust peak finding and fits.
 
#     SNR = (RawPeak - Baseline) / noise_rms, where noise_rms comes from
#     compute_noise_rms on the raw waveform — the same definition used when
#     building the full database. There's no precomputed SNR column at this
#     stage, so it's derived here directly from DataSample/TriggerPosition,
#     the same way compute_rise_time derives rise time from the waveform.
 
#     SNR is streamed directly from parquet_path, per channel, with NO
#     Amplitude or rise-time filtering applied. Amplitude_cuts, rise_time_cuts,
#     and SNR_cuts are each determined independently from the raw per-channel
#     distribution, so that none of the three thresholds is conditioned on
#     the others having already been applied. They are only combined together
#     (via AND) afterwards, in build_hit_mask, when building the final mask.
 
#     Recommended order:
#         Amplitude_cuts  = plot_Amplitude_histograms(parquet)
#         rise_time_cuts = plot_rise_time_histograms(parquet)
#         SNR_cuts       = plot_SNR_histograms(parquet)
#         mask           = build_hit_mask(parquet, Amplitude_cuts, rise_time_cuts, SNR_cuts)
 
#     channel_filter restricts which channels are read/histogrammed; it is a
#     selection, not a quality cut, so it doesn't reintroduce the
#     cut-on-a-cut issue.
 
#     Everything below the histogram-building step (peak finding, EMG + Moyal
#     component fits, composite fit, valley/percentile cut logic, plotting)
#     is unchanged from the original.
#     """
#     parquet_path = Path(parquet_path)
 
#     cols = [channel_col, baseline_col, peak_col, "DataSample", "TriggerPosition"]
#     batches = open_hit_reader(parquet_path, cols=cols, batch_size=batch_size, root_tree=root_tree)
 
#     snr_by_channel: Dict[int, list] = {}
#     total        = 0
#     skipped      = 0   # non-finite / negative SNR values
#     skipped_none = 0   # noise_rms could not be computed (<=0 or NaN)
 
#     for batch in tqdm(batches, desc="Computing SNR"):
#         channels  = batch[channel_col].to_numpy()
#         baselines = batch[baseline_col].to_numpy(zero_copy_only=False).astype(np.float64)
#         peaks     = batch[peak_col].to_numpy(zero_copy_only=False).astype(np.float64)
#         samples   = batch["DataSample"]
#         triggers  = batch["TriggerPosition"]
 
#         for i in range(len(channels)):
#             total += 1
#             ch = int(channels[i])
 
#             if channel_filter is not None and ch not in channel_filter:
#                 continue
 
#             bl = float(baselines[i])
#             pk = float(peaks[i])
 
#             samp_arr = np.asarray(samples[i].as_py(),  dtype=np.float64)
#             trig_arr = np.asarray(triggers[i].as_py(), dtype=np.int32)
 
#             noise_rms = compute_noise_rms(samp_arr, trig_arr, bl, n_noise_samples)
#             if not (np.isfinite(noise_rms) and noise_rms > 0):
#                 skipped_none += 1
#                 continue
 
#             snr_val = (pk - bl) / noise_rms
#             if not np.isfinite(snr_val) or snr_val < 0:
#                 skipped += 1
#                 continue
 
#             snr_by_channel.setdefault(ch, []).append(snr_val)
 
#     print(f"Total hits           : {total:,}")
#     print(f"No noise_rms (<=0)   : {skipped_none:,} ({100*skipped_none/max(total,1):.1f}%)")
#     print(f"Skipped (bad SNR)    : {skipped:,} ({100*skipped/max(total,1):.1f}%)")
 
#     if not snr_by_channel:
#         raise RuntimeError("No SNR data found — check column names and channel_filter.")
 
#     snr_by_channel = {ch: np.array(v, dtype=np.float64) for ch, v in snr_by_channel.items()}
 
#     channels_sorted = sorted(snr_by_channel)
#     results: Dict[int, np.ndarray] = {}
#     bin_edges_by_ch: Dict[int, np.ndarray] = {}
#     hist_smooth_by_ch: Dict[int, np.ndarray] = {}
 
#     # Build histograms.
#     for ch in channels_sorted:
#         arr = snr_by_channel[ch]
#         lo, hi = x_range if x_range is not None else (float(np.min(arr)), float(np.max(arr)))
#         counts, edges = np.histogram(arr, bins=n_bins, range=(lo, hi))
#         results[ch] = counts
#         bin_edges_by_ch[ch] = edges
#         hist_smooth_by_ch[ch] = gaussian_filter1d(counts.astype(float), sigma=1.5)
 
#     n = len(channels_sorted)
#     ncols = min(2, n)
#     nrows = (n + ncols - 1) // ncols
 
#     fig, axes = plt.subplots(
#         nrows,
#         ncols,
#         figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
#     )
#     axes = np.array(axes).flatten()
 
#     thresholds: Dict[int, Tuple[float, float]] = {}
 
#     for ax, ch in zip(axes, channels_sorted):
#         counts = results[ch].astype(float)
#         edges = bin_edges_by_ch[ch]
#         bw = edges[1] - edges[0]
#         centres = 0.5 * (edges[:-1] + edges[1:])
#         arr = snr_by_channel[ch]
#         smoothed = hist_smooth_by_ch[ch]
 
#         ax.bar(centres, counts, width=bw, align="center", alpha=0.75, label="Data")
 
#         # Peak finding.
#         peaks = _detect_peaks(centres, counts)
#         noise_idx, signal_idx = _choose_noise_signal_peaks(centres, counts, peaks)
 
#         # Some channels are very overlapping; if peak finding is weak, force
#         # a noise peak near the left and a signal peak near the global maximum.
#         if noise_idx is None or signal_idx is None:
#             noise_idx = int(np.argmax(smoothed[: max(2, min(len(smoothed), int(0.15 * len(smoothed))))]))
#             right_mask = centres > centres[max(noise_idx + 2, 1)]
#             if np.any(right_mask):
#                 signal_idx = int(np.argmax(np.where(right_mask, smoothed, -np.inf)))
#             else:
#                 signal_idx = int(np.argmax(smoothed))
 
#         valley_x, valley_idx = _valley_between_peaks(centres, counts, noise_idx, signal_idx)
 
#         # High cut: robust upper percentile.
#         thr_high = _safe_percentile(arr, 95, default=float(np.max(centres)))
 
#         # If the valley lands too far to the right because of a very broad
#         # shoulder, clamp it to a reasonable region between the two main peaks.
#         if not np.isfinite(valley_x):
#             valley_x = float(centres[min(noise_idx, signal_idx)])
#         valley_x = float(np.clip(valley_x, centres.min(), thr_high))
 
#         # Fit components separately.
#         comp_ok = False
#         try:
#             noise_popt, noise_pcov = _fit_noise_only(centres, counts, valley_x)
#         except Exception as e:
#             print(f"Channel {ch}: noise fit failed: {e}")
#             noise_popt = None
 
#         try:
#             signal_popt, signal_pcov = _fit_signal_only(
#                 centres,
#                 counts,
#                 valley_x,
#                 noise_params=noise_popt if noise_popt is not None else None,
#             )
#         except Exception as e:
#             print(f"Channel {ch}: signal fit failed: {e}")
#             signal_popt = None
 
#         # Composite fit if both component fits worked.
#         popt = None
#         pcov = None
#         chi2 = np.nan
#         ndf = np.nan
 
#         fit_max_x = float(x_range[1]) if x_range is not None else float(centres.max())
#         fit_mask = (centres >= centres.min()) & (centres <= fit_max_x) & (counts > 0)
#         x_fit = centres[fit_mask]
#         y_fit = counts[fit_mask]
 
#         if noise_popt is not None and signal_popt is not None and x_fit.size >= 8:
#             try:
#                 popt, pcov = _fit_composite(centres, counts, noise_popt, signal_popt, fit_max_x)
#                 y_model = composite_model(x_fit, *popt)
#                 chi2 = float(np.sum(((y_fit - y_model) / np.sqrt(np.maximum(y_fit, 1.0))) ** 2))
#                 ndf = len(y_fit) - len(popt)
#                 comp_ok = True
#             except Exception as e:
#                 print(f"Channel {ch}: composite fit failed: {e}")
 
#         # Plot model(s).
#         xd = np.linspace(float(centres.min()), float(centres.max()), 2000)
 
#         noise_line = None
#         signal_line = None
#         if noise_popt is not None:
#             noise_line = noise_emg(xd, *noise_popt)
#             #ax.plot(xd, noise_line, color="tab:green", ls=":", lw=1.6, alpha=0.9, label="Noise EMG")
#         if signal_popt is not None:
#             signal_line = signal_moyal(xd, *signal_popt)
#             #ax.plot(xd, signal_line, color="tab:blue", ls="--", lw=1.6, alpha=0.9, label="Signal Moyal")
#         if comp_ok and popt is not None:
#             ax.plot(xd, composite_model(xd, *popt), "r-", lw=2.0, label="Composite fit")
 
#         # ------------------------------------------------------------------
#         # ROBUST LOW-CUT DETERMINATION
#         # ------------------------------------------------------------------
 
#         thr_low = None
 
#         # Extract the true fitted signal MPV if a fit exists to use as a hard guard
#         fit_signal_mpv = None
#         if comp_ok and popt is not None:
#             # Structure: A_n, K, loc_n, scale_n, A_s, loc_s, scale_s
#             fit_signal_mpv = popt[5]  # loc_s is the fitted MPV
#         elif signal_popt is not None:
#             fit_signal_mpv = signal_popt[1]  # Assuming standard (Amp, loc, scale)
 
#         # ==============================================================
#         # 1) Try histogram valley between two peaks
#         # ==============================================================
 
#         smooth = gaussian_filter1d(counts.astype(float), sigma=2.0)
#         peaks, _ = find_peaks(smooth, prominence=0.05 * smooth.max(), distance=10)
#         peaks = peaks[smooth[peaks] > 0.10 * smooth.max()]
 
#         noise_peak_x = None
#         signal_peak_x = None
 
#         if len(peaks) >= 2:
#             noise_idx = np.argmin(centres[peaks])
#             noise_peak = peaks[noise_idx]
 
#             remaining_peaks_indices = [i for i in range(len(peaks)) if i != noise_idx]
#             remaining_peaks = peaks[remaining_peaks_indices]
#             signal_peak = remaining_peaks[np.argmax(smooth[remaining_peaks])]
 
#             noise_peak_x = centres[noise_peak]
#             signal_peak_x = centres[signal_peak]
 
#             if signal_peak > noise_peak + 2:
#                 # ── KEY FIX: clamp the right boundary of the valley search ──
#                 # Use the fitted MPV if available, otherwise the histogram signal peak.
#                 # This guarantees the valley (and therefore the low cut) is always
#                 # strictly to the LEFT of the true signal peak.
#                 if fit_signal_mpv is not None:
#                     mpv_idx = int(np.argmin(np.abs(centres - fit_signal_mpv)))
#                 else:
#                     mpv_idx = signal_peak  # fallback: histogram peak
 
#                 # Search only between noise peak and MPV, not beyond it
#                 search_right = min(signal_peak, mpv_idx)
 
#                 if search_right > noise_peak + 1:
#                     valley_idx = noise_peak + np.argmin(smooth[noise_peak:search_right])
#                     thr_low = float(centres[valley_idx])
 
#         # ==============================================================
#         # 2) Try signal fit if valley wasn't found
#         # ==============================================================
 
#         if thr_low is None and signal_popt is not None:
 
#             xd_cut = np.linspace(
#                 centres.min(),
#                 centres.max(),
#                 5000,
#             )
 
#             signal_y = signal_moyal(
#                 xd_cut,
#                 *signal_popt
#             )
 
#             peak_idx = np.argmax(signal_y)
 
#             peak_height = signal_y[peak_idx]
 
#             target = 0.05 * peak_height
 
#             left_side = signal_y[:peak_idx]
 
#             below = np.where(
#                 left_side <= target
#             )[0]
 
#             if len(below):
 
#                 # last point before reaching peak
#                 thr_low = float(
#                     xd_cut[below[-1]]
#                 )
 
#         # ==============================================================
#         # 3) Pure histogram fallback
#         # ==============================================================
 
#         if thr_low is None:
 
#             peak_idx = np.argmax(smooth)
 
#             peak_height = smooth[peak_idx]
 
#             target = 0.05 * peak_height
 
#             left_side = smooth[:peak_idx]
 
#             below = np.where(
#                 left_side <= target
#             )[0]
 
#             if len(below):
 
#                 thr_low = float(
#                     centres[below[-1]]
#                 )
 
#         # ==============================================================
#         # 4) Absolute emergency fallback
#         # ==============================================================
 
#         if thr_low is None:
 
#             peak_idx = np.argmax(smooth)
 
#             thr_low = float(
#                 max(
#                     centres[0],
#                     centres[peak_idx] * 0.25
#                 )
#             )
 
#         # ==============================================================
#         # 5) Enforce physical sanity
#         # ==============================================================
 
#         # Only push the cut right of the noise peak if the valley landed
#         # *inside* or *left of* the noise peak — not when it's already
#         # correctly placed in the inter-peak valley.
#         if noise_peak_x is not None:
#             if thr_low < noise_peak_x:          # cut is buried in noise — push it right
#                 thr_low = max(thr_low, noise_peak_x)
#             # else: valley is already to the right of the noise peak → leave it alone
 
#         if signal_peak_x is not None:
#             thr_low = min(thr_low, 0.90 * signal_peak_x)
 
#         # Hard enforcement to stay strictly to the left of the fitted MPV
#         if fit_signal_mpv is not None:
#             thr_low = min(thr_low, 0.95 * fit_signal_mpv)
 
#         # Never exceed high cut
 
#         thr_low = min(
#             thr_low,
#             thr_high - bw
#         )
 
#         thr_low = max(
#             thr_low,
#             centres.min()
#         )
 
#         thresholds[ch] = (
#             float(thr_low),
#             float(thr_high),
#         )
#         # Draw thresholds.
#         ax.axvline(thr_low, color="darkorange", ls="-.", lw=1.5, label=f"Low cut (valley ≈ {thr_low:.1f})")
#         ax.axvline(thr_high, color="darkorange", ls=":", lw=1.5, label=f"High cut (95th ≈ {thr_high:.1f})")
 
#         # Text box.
#         if comp_ok and popt is not None:
#             A_n, K, loc_n, scale_n, A_s, loc_s, scale_s = popt
#             text = (
#                 f"Noise: loc={loc_n:.2f}, σ={scale_n:.2f}, K={K:.2f}\n"
#                 f"Signal: loc={loc_s:.2f}, σ={scale_s:.2f}\n"
#                 f"χ²/ndf = {chi2:.1f}/{ndf:.0f} = {chi2 / max(ndf, 1):.2f}"
#             )
#         else:
#             text = f"Valley ≈ {thr_low:.2f}\nHigh cut ≈ {thr_high:.2f}"
 
#         ax.text(
#             0.97,
#             0.97,
#             text,
#             transform=ax.transAxes,
#             ha="right",
#             va="top",
#             fontsize=9,
#             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
#         )
 
#         ax.set_title(f"Channel {ch}", fontsize=18)
#         ax.set_xlabel("SNR", fontsize=18)
#         ax.set_ylabel("Counts", fontsize=18)
#         if hep is not None:
#             try:
#                 hep.cms.label(label, data=is_data, rlabel=rlabel, loc=0, ax=ax, fontsize=(14, 12, 12, 11))
#             except Exception:
#                 pass
#         ax.legend(fontsize=9)
#         if logy:
#             ax.set_yscale("log")
#         ax.grid(alpha=0.3)
#         ax.set_xlim(x_range)
 
#     for ax in axes[n:]:
#         ax.set_visible(False)
 
#     fig.suptitle("SNR per Channel (unfiltered, raw distribution)", fontsize=14)
#     plt.tight_layout()
#     plt.show()
 
#     print("\nSuggested SNR cut thresholds:")
#     for ch, (lo, hi) in sorted(thresholds.items()):
#         print(f"  Channel {ch}: {lo:.3f} < SNR < {hi:.3f}")
 
#     return results, thresholds





# def build_hit_mask(
#     parquet_path: str,
#     Amplitude_cuts:   Optional[Dict[int, float]] = None,
#     rise_time_cuts:  Optional[Dict[int, tuple]] = None,
#     SNR_cuts:        Optional[Dict[int, tuple]] = None,
#     channel_filter:  Optional[list[int]] = None,
#     first_hit:       int = 0,
#     num_hits:        Optional[int] = None,
#     period:          float = 1 / 6.4,
#     batch_size:      int = 100_000,
#     root_tree:       str = "sampic_hits",
#     low_frac:        float = 0.1,
#     high_frac:       float = 0.9,
#     smooth_window:   int = 2,
#     n_noise_samples: int = 20,
#     save_path:       Optional[str] = None,
# ) -> np.ndarray:
#     """
#     Stream a parquet file and return the HITNumbers of waveforms that pass
#     all requested filters.
 
#     Filters applied in order (each is optional):
#       1. channel_filter  — keep only listed channels
#       2. Amplitude_cuts   — per-channel minimum Amplitude = (RawPeak-Baseline)/Baseline
#       3. rise_time_cuts  — per-channel (min_rt, max_rt) rise time window in ns
#       4. SNR_cuts        — per-channel (min_snr, max_snr) SNR window,
#                             where SNR = (RawPeak-Baseline) / noise_rms
#       5. first_hit       — skip the first N passing hits (global, across all channels)
#       6. num_hits        — stop after N passing hits
 
#     Parameters
#     ----------
#     Amplitude_cuts : dict mapping channel → minimum Amplitude.
#                e.g. {4: 0.10, 8: 0.05}
#                Channels absent from the dict pass with no Amplitude cut.
 
#     rise_time_cuts : dict mapping channel → (min_ns, max_ns).
#                e.g. {4: (0.5, 3.0), 8: (0.3, 2.0)}
#                Channels absent from the dict pass with no rise time cut.
#                Rise time is computed from the raw waveform via
#                compute_rise_time only when this filter is active.
 
#     SNR_cuts : dict mapping channel → (min_snr, max_snr).
#                e.g. {4: (8.0, 200.0), 8: (6.0, 200.0)}
#                Channels absent from the dict pass with no SNR cut.
#                SNR = (RawPeak - Baseline) / noise_rms, where noise_rms comes
#                from compute_noise_rms on the raw waveform (same definition
#                used when building the full database) — computed only when
#                this filter is active.
 
#     channel_filter : list of channels to keep; None = all channels.
 
#     first_hit  : number of passing hits to skip before collecting.
#     num_hits   : maximum number of passing hits to collect; None = all.
 
#     n_noise_samples : number of leading ordered samples used by
#                compute_noise_rms to estimate the baseline noise RMS.
 
#     save_path  : if provided, saves the HITNumber array as a .npy file
#                  so the mask can be reloaded without re-running the filter.
 
#     Returns
#     -------
#     np.ndarray of int
#         1-D array of HITNumbers that passed all filters, in file order.
 
#     Examples
#     --------
#     >>> good_hits = build_hit_mask(
#     ...     parquet_file,
#     ...     Amplitude_cuts  = {4: 0.10, 8: 0.05},
#     ...     rise_time_cuts = {4: (0.5, 3.0), 8: (0.3, 2.0)},
#     ...     SNR_cuts       = {4: (8.0, 200.0), 8: (6.0, 200.0)},
#     ...     channel_filter = [4, 8],
#     ...     save_path      = "good_hits_run42.npy",
#     ... )
#     >>> # reload later without reprocessing:
#     >>> good_hits = np.load("good_hits_run42.npy")
#     """
#     need_waveform = (rise_time_cuts is not None) or (SNR_cuts is not None)
 
#     # Columns to load — only add waveform columns if actually needed
#     cols = ["HITNumber", "Channel", "Baseline", "RawPeak"]
#     if need_waveform:
#         cols += ["DataSample", "TriggerPosition"]
 
#     batches = open_hit_reader(
#         parquet_path, cols=cols,
#         batch_size=batch_size, root_tree=root_tree,
#     )
 
#     good_hit_numbers = []
#     total_ch         = defaultdict(int)
#     fail_Amplitude_ch = defaultdict(int)
#     fail_rt_ch       = defaultdict(int)
#     fail_rt_none_ch  = defaultdict(int)
#     fail_snr_ch      = defaultdict(int)
#     fail_snr_none_ch = defaultdict(int)
#     passed_ch        = defaultdict(int)
 
#     # Counters for diagnostics
#     total          = 0
#     fail_Amplitude  = 0
#     fail_rt        = 0
#     fail_rt_none   = 0
#     fail_snr       = 0
#     fail_snr_none  = 0
#     skipped        = 0   # first_hit skip
#     collected      = 0   # passing hits collected
 
#     for batch in tqdm(batches, desc="Filtering waveforms"):
#         hit_ids   = batch["HITNumber"].to_numpy()
#         channels  = batch["Channel"].to_numpy()
#         baselines = batch["Baseline"].to_numpy(zero_copy_only=False).astype(np.float64)
#         peaks     = batch["RawPeak"].to_numpy(zero_copy_only=False).astype(np.float64)
 
#         if need_waveform:
#             samples  = batch["DataSample"]
#             triggers = batch["TriggerPosition"]
 
#         for i in range(len(hit_ids)):
#             total += 1
#             ch = int(channels[i])
#             bl = float(baselines[i])
#             pk = float(peaks[i])
#             total_ch[ch] += 1
 
#             # Apply cuts only to selected channels
#             apply_filters = (
#                 channel_filter is None or
#                 ch in channel_filter
#             )
 
#             # 2. Amplitude cut
#             if apply_filters and Amplitude_cuts is not None and ch in Amplitude_cuts:
#                 if bl == 0:
#                     fail_Amplitude += 1
#                     fail_Amplitude_ch[ch] += 1
#                     continue
#                 Amplitude = (pk - bl)
#                 if Amplitude < Amplitude_cuts[ch]:
#                     fail_Amplitude += 1
#                     fail_Amplitude_ch[ch] += 1
#                     continue
 
#             # Decode the waveform once if either rise time or SNR is needed
#             # for this hit's channel.
#             needs_rt  = apply_filters and rise_time_cuts is not None and ch in rise_time_cuts
#             needs_snr = apply_filters and SNR_cuts is not None and ch in SNR_cuts
 
#             if needs_rt or needs_snr:
#                 samp_arr = np.asarray(samples[i].as_py(),  dtype=np.float64)
#                 trig_arr = np.asarray(triggers[i].as_py(), dtype=np.int32)
 
#             # 3. Rise time cut
#             if needs_rt:
#                 rt = compute_rise_time(
#                     samp_arr, trig_arr,
#                     bl, period, low_frac, high_frac, smooth_window,
#                 )
#                 if rt is None:
#                     fail_rt_none += 1
#                     fail_rt_ch[ch] += 1
#                     fail_rt_none_ch[ch] += 1
#                     continue
#                 rt_min, rt_max = rise_time_cuts[ch]
#                 if not (rt_min <= rt <= rt_max):
#                     fail_rt += 1
#                     fail_rt_ch[ch] += 1
#                     continue
 
#             # 4. SNR cut — SNR = (RawPeak - Baseline) / noise_rms
#             if needs_snr:
#                 noise_rms = compute_noise_rms(samp_arr, trig_arr, bl, n_noise_samples)
#                 amplitude = pk - bl
#                 if not (np.isfinite(noise_rms) and noise_rms > 0):
#                     fail_snr_none += 1
#                     fail_snr_ch[ch] += 1
#                     fail_snr_none_ch[ch] += 1
#                     continue
#                 snr_val = amplitude / noise_rms
#                 snr_min, snr_max = SNR_cuts[ch]
#                 if not (snr_min <= snr_val <= snr_max):
#                     fail_snr += 1
#                     fail_snr_ch[ch] += 1
#                     continue
 
#             # 5. first_hit skip
#             if skipped < first_hit:
#                 skipped += 1
#                 continue
 
#             # 6. num_hits cap
#             if num_hits is not None and collected >= num_hits:
#                 break   # inner loop — will also need to break outer below
 
#             good_hit_numbers.append(int(hit_ids[i]))
#             collected += 1
#             passed_ch[ch] += 1
 
#         # Break outer loop too if cap reached
#         if num_hits is not None and collected >= num_hits:
#             break
 
#     good = np.array(good_hit_numbers, dtype=np.int64)
 
#     # --- Diagnostics --------------------------------------------------------
#     print("\nPer‑channel filter summary:")
#     for ch in sorted(total_ch):
#         tot   = total_ch[ch]
#         amp_f = fail_Amplitude_ch[ch]
#         rt_f  = fail_rt_ch[ch]
#         rt_n  = fail_rt_none_ch[ch]
#         snr_f = fail_snr_ch[ch]
#         snr_n = fail_snr_none_ch[ch]
#         ok    = passed_ch[ch]
 
#         print(
#             f"  Channel {ch}: {tot} hits | "
#             f"Amplitude fails = {amp_f} ({100*amp_f/max(tot,1):.1f}%) | "
#             f"RT fails = {rt_f} ({100*rt_f/max(tot,1):.1f}%) | "
#             f"RT None = {rt_n} ({100*rt_n/max(tot,1):.1f}%) | "
#             f"SNR fails = {snr_f} ({100*snr_f/max(tot,1):.1f}%) | "
#             f"SNR None = {snr_n} ({100*snr_n/max(tot,1):.1f}%) | "
#             f"Passed = {ok} ({100*ok/max(tot,1):.1f}%)"
#         )
 
#     print(f"\nFilter summary ({parquet_path})")
#     print(f"  Total hits seen      : {total:,}")
#     print(f"  Failed Amplitude cut  : {fail_Amplitude:,} ({100*fail_Amplitude/max(total,1):.1f}%)")
#     print(f"  Failed rise time cut : {fail_rt:,} ({100*fail_rt/max(total,1):.1f}%)")
#     print(f"  No rise time found   : {fail_rt_none:,} ({100*fail_rt_none/max(total,1):.1f}%)")
#     print(f"  Failed SNR cut       : {fail_snr:,} ({100*fail_snr/max(total,1):.1f}%)")
#     print(f"  No SNR (noise_rms<=0): {fail_snr_none:,} ({100*fail_snr_none/max(total,1):.1f}%)")
#     print(f"  Skipped (first_hit)  : {skipped:,}")
#     print(f"  Passing hits         : {len(good):,} ({100*len(good)/max(total,1):.1f}%)")
 
#     if save_path is not None:
#         np.save(save_path, good)
#         print(f"  Saved mask to       : {save_path}")
 
#     return good




# def load_hit_mask(save_path: str) -> np.ndarray:
#     """
#     Loads a previously saved hit mask from a .npy file 
#     and returns a 'good'-like NumPy array.
#     """
#     if not os.path.exists(save_path):
#         raise FileNotFoundError(f"Hit mask file not found at: {save_path}")
        
#     print(f"Successfully loaded mask from {save_path}")
#     return np.load(save_path)


# def apply_hit_mask(
#     parquet_path: str,
#     hit_mask:     np.ndarray,
#     columns:      Optional[list[str]] = None,
#     batch_size:   int = 100_000,
#     root_tree:    str = "sampic_hits",
# ) -> pa.Table:
#     """
#     Load only the rows matching `hit_mask` from a parquet file.

#     Uses the same batch-by-batch strategy as build_coincidence_histograms
#     — never reads non-matching rows into memory.

#     Parameters
#     ----------
#     hit_mask : array of HITNumbers returned by build_hit_mask,
#                or loaded from a .npy file.
#     columns  : columns to load; None = all columns.

#     Returns
#     -------
#     pyarrow.Table with only the matching rows.
#     """
#     # Build a set for O(1) lookup
#     mask_set = set(hit_mask.tolist())

#     pqf        = pq.ParquetFile(parquet_path)
#     load_cols  = columns or pqf.schema_arrow.names
#     if "HITNumber" not in load_cols:
#         load_cols = ["HITNumber"] + list(load_cols)

#     collected  = []
#     global_row = 0

#     for batch in pqf.iter_batches(batch_size=batch_size, columns=load_cols):
#         hit_ids   = batch["HITNumber"].to_numpy()
#         local_idx = np.where(np.isin(hit_ids, hit_mask))[0]
#         if len(local_idx) > 0:
#             collected.append(batch.take(pa.array(local_idx)))
#         global_row += batch.num_rows

#     if not collected:
#         raise RuntimeError("No matching hits found — check hit_mask.")

#     return pa.Table.from_batches(collected)