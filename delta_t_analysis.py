from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Set
from dataclasses import dataclass, field
import delays_db
import pandas as pd
import parameters_database
import importlib
from scipy.ndimage import zoom
importlib.reload(parameters_database)
importlib.reload(delays_db)
import mplhep as hep
from itertools import combinations


DELAY_REGISTRY = delays_db.DELAY_REGISTRY
CFD_LEVELS = parameters_database.CFD_LEVELS


# =============================================================================
# Adjacency helpers
# =============================================================================
def get_pixel_map_from_config(config):
    """
    Creates a dict mapping SAMPIC channel -> list of (row, col) tuples.
    """
    pixel_map = {}
    # Invert sampic_to_board to get a flat map of all channels
    for board_name, ch_dict in config.sampic_to_board.items():
        for sampic_ch, lgad_board_ch in ch_dict.items():
            if lgad_board_ch in config.sensor_channels:
                pixel_map[sampic_ch] = config.sensor_channels[lgad_board_ch]
    return pixel_map


def _chebyshev_adjacent(p1: tuple, p2: tuple) -> bool:
    """True if two (row, col) pixels are Chebyshev-adjacent (share edge or corner)."""
    return abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1

def _hits_are_adjacent(pixels1: list[tuple], pixels2: list[tuple]) -> bool:
    """Returns True if ANY pixel in hit 1 is adjacent to ANY pixel in hit 2."""
    for p1 in pixels1:
        for p2 in pixels2:
            # Chebyshev distance <= 1
            if abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1:
                return True
    return False

def _is_hit_cluster_connected(hits_pixels_list: list[list[tuple]]) -> bool:
    """
    True if all hits form a single connected component.
    hits_pixels_list: list where each element is a list of (row, col) tuples for a single hit.
    """
    n = len(hits_pixels_list)
    if n <= 1:
        return True

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if _hits_are_adjacent(hits_pixels_list[i], hits_pixels_list[j]):
                union(i, j)

    return len({find(i) for i in range(n)}) == 1


def _is_connected(pixels: list[tuple]) -> bool:
    """
    True if all pixels form a single connected component under Chebyshev adjacency.
    Uses union-find.
    """
    n = len(pixels)
    if n <= 1:
        return True

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if _chebyshev_adjacent(pixels[i], pixels[j]):
                union(i, j)

    return len({find(i) for i in range(n)}) == 1

def _cross_sensor_adjacent(pixels1: list[tuple], pixels2: list[tuple]) -> bool:
    """
    Returns True if ANY pixel in channel 1 is adjacent to ANY pixel in channel 2.
    """
    for p1 in pixels1:
        for p2 in pixels2:
            # Chebyshev distance <= 1
            if abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1:
                return True
    return False

#=============================
# Diagnostics
#=============================
"""
mean_spread_diagnostic.py
--------------------------
Drop-in addon for delta_t_analysis.py / channel_pair_analysis.py.

Adds:
  1. Two small edits to plot_channel_pair_histograms (marked PATCH 1, PATCH 2)
     that record mu_pair / mu_pair_err per channel pair and print them in
     the summary table.
  2. A new function diagnose_mean_spread() that takes the per-pair (n, mu,
     mu_err, sigma) values and the pooled (global) resolution, and reports
     whether the pooled resolution is inflated by inter-channel-pair
     miscalibration of the mean delay.

Nothing here changes the existing return value or signature of
plot_channel_pair_histograms beyond adding two columns to the printed
table — the function still returns the same dict of figures.
"""



# =============================================================================
# PATCH 1 — inside plot_channel_pair_histograms, in the per-pair loop,
# right after computing mu50/mu50_err (CFD_plot/CFD_plot resolution block),
# nothing needs to change there — mu50 and mu50_err already exist.
# The only change is in what gets appended to summary_rows:
#
# BEFORE:
#   summary_rows.append((c1, c2, n_ev,
#                         res50, res50_err, res50_std, res50_std_err,
#                         best_k1, best_k2, best_res, best_err,
#                         best_k1_std, best_k2_std, best_res_std, best_err_std))
#
# AFTER:
#   summary_rows.append((c1, c2, n_ev,
#                         mu50, mu50_err,                     # <-- ADD
#                         res50, res50_err, res50_std, res50_std_err,
#                         best_k1, best_k2, best_res, best_err,
#                         best_k1_std, best_k2_std, best_res_std, best_err_std))
# =============================================================================


# =============================================================================
# PATCH 2 — summary table print loop, unpack + print the two new columns
#
# BEFORE:
#   for row in summary_rows:
#       (c1, c2, n_ev,
#        r50, e50, r50s, e50s,
#        bk1, bk2, br, be,
#        bk1s, bk2s, brs, bes) = row
#
# AFTER:
#   for row in summary_rows:
#       (c1, c2, n_ev,
#        mu50, mu50_err,                                       # <-- ADD
#        r50, e50, r50s, e50s,
#        bk1, bk2, br, be,
#        bk1s, bk2s, brs, bes) = row
#
# And add a column to the header / print f-string, e.g.:
#   f"{'mu (ps)':>14}"   ...   f"{mu50*1e3:.1f} ± {mu50_err*1e3:.1f}":>14
#
# A full ready-to-paste version of plot_channel_pair_histograms section 6
# is given in `summary_table_block.py` alongside this file if you'd rather
# copy-paste than hand-edit.
# =============================================================================


def diagnose_mean_spread(
    pair_results: List[Tuple[int, int, int, float, float, float, float]],
    pooled_resolution_ps: float,
    pooled_resolution_err_ps: Optional[float] = None,
    method_label: str = "Gaussian fit",
) -> dict:
    """
    Quantify whether the pooled (global) resolution is inflated by
    residual inter-channel-pair delay miscalibration.

    Parameters
    ----------
    pair_results : list of tuples
        (c1, c2, n_events, mu_ns, mu_err_ns, sigma_res_ps, sigma_res_err_ps)
        — one entry per channel pair. mu_ns is the fitted Δt centroid for
        that pair (ns); sigma_res_ps is that pair's own timing resolution
        in ps (i.e. sigma/sqrt(2)*1e3, NOT the raw Gaussian sigma).
    pooled_resolution_ps : float
        The resolution measured from the heatmap / plot_delta_t_histogram
        on the full pooled (all-pairs-combined) Δt distribution, in ps.
    pooled_resolution_err_ps : float, optional
        Uncertainty on the pooled resolution, for reporting only.
    method_label : str
        Just a label for the printout (e.g. "Gaussian fit" or "Std dev").

    Returns
    -------
    dict with keys:
        weighted_mean_mu_ns   : event-count-weighted average of mu_pair
        spread_of_means_ps    : weighted std of mu_pair values, in ps
        weighted_mean_sigma_ps: event-count-weighted average of per-pair
                                 resolutions, in ps
        quadrature_prediction_ps : sqrt(weighted_mean_sigma² + spread_of_means²)
        pooled_resolution_ps  : as given
        inflation_ps          : pooled_resolution_ps - weighted_mean_sigma_ps
        inflation_fraction    : inflation_ps / weighted_mean_sigma_ps
    """
    # Filter out pairs with non-finite values — can't use them
    clean = [
        (c1, c2, n, mu, mu_err, sig, sig_err)
        for (c1, c2, n, mu, mu_err, sig, sig_err) in pair_results
        if np.isfinite(mu) and np.isfinite(sig) and n > 0
    ]

    if len(clean) < 2:
        print("Not enough valid pairs (<2) to compute a mean-spread diagnostic.")
        return {}

    n_arr   = np.array([c[2] for c in clean], dtype=np.float64)
    mu_arr  = np.array([c[3] for c in clean], dtype=np.float64)   # ns
    sig_arr = np.array([c[5] for c in clean], dtype=np.float64)   # ps (resolution, not raw sigma)

    weights = n_arr  # event-count weighting

    # --- Weighted mean of the per-pair centroids ---
    weighted_mean_mu_ns = np.average(mu_arr, weights=weights)

    # --- Weighted spread of the centroids around that mean ---
    # This directly measures how mis-centered the pairs are relative to
    # each other — the quantity that inflates the pooled distribution.
    weighted_var_mu = np.average((mu_arr - weighted_mean_mu_ns) ** 2, weights=weights)
    spread_of_means_ps = np.sqrt(weighted_var_mu) * 1e3   # ns -> ps

    # --- Weighted mean of the per-pair *true* resolutions ---
    weighted_mean_sigma_ps = np.average(sig_arr, weights=weights)

    # --- Quadrature prediction for the pooled resolution ---
    # If pooling several Gaussians of similar width sigma, each offset
    # from a common mean by delta_i, the resulting pooled variance is
    # (to leading order, equal weights) sigma_true^2 + spread_of_means^2.
    quadrature_prediction_ps = np.sqrt(weighted_mean_sigma_ps ** 2 + spread_of_means_ps ** 2)

    inflation_ps = pooled_resolution_ps - weighted_mean_sigma_ps
    inflation_fraction = (
        inflation_ps / weighted_mean_sigma_ps if weighted_mean_sigma_ps > 0 else np.nan
    )

    # --- Report ---
    print()
    print("=" * 78)
    print(f"  MEAN-SPREAD DIAGNOSTIC  ({method_label})")
    print("=" * 78)
    print(f"  Pairs used                         : {len(clean)}")
    print(f"  Weighted mean of pair resolutions   : {weighted_mean_sigma_ps:6.1f} ps   "
          f"<- 'true' single-pair resolution estimate")
    print(f"  Weighted spread of pair means (μ)   : {spread_of_means_ps:6.1f} ps   "
          f"<- inter-channel-pair miscalibration")
    print(f"  Quadrature-predicted pooled res.    : {quadrature_prediction_ps:6.1f} ps   "
          f"<- sqrt(mean_sigma^2 + spread_of_means^2)")
    print("  " + "-" * 76)
    pooled_str = f"{pooled_resolution_ps:.1f}"
    if pooled_resolution_err_ps is not None and np.isfinite(pooled_resolution_err_ps):
        pooled_str += f" ± {pooled_resolution_err_ps:.1f}"
    print(f"  Actual pooled (heatmap) resolution  : {pooled_str:>8} ps")
    print(f"  Inflation (pooled - mean_sigma)     : {inflation_ps:6.1f} ps "
          f"({inflation_fraction*100:+.1f}%)" if np.isfinite(inflation_fraction) else "  Inflation: N/A")
    print("=" * 78)

    if spread_of_means_ps > 0.3 * weighted_mean_sigma_ps:
        print("  -> Mean spread is >30% of the single-pair resolution: ")
        print("     residual inter-channel delay miscalibration is likely")
        print("     contributing meaningfully to the pooled resolution.")
    else:
        print("  -> Mean spread is small relative to the single-pair resolution:")
        print("     pooled inflation from delay miscalibration appears minor.")
    print()

    return {
        "weighted_mean_mu_ns":       weighted_mean_mu_ns,
        "spread_of_means_ps":        spread_of_means_ps,
        "weighted_mean_sigma_ps":    weighted_mean_sigma_ps,
        "quadrature_prediction_ps":  quadrature_prediction_ps,
        "pooled_resolution_ps":      pooled_resolution_ps,
        "inflation_ps":              inflation_ps,
        "inflation_fraction":        inflation_fraction,
    }


# =============================================================================
# Event dataclass — one per coincidence window
# =============================================================================

@dataclass
class _LGADHit:
    """Representative time + metadata for one LGAD within an event."""
    lgad:         str
    n_pixels:     int
    is_cluster:   bool              # True if >1 pixel was averaged
    snr:          float
    times:        Dict[int, float]  # {cfd_level: corrected_time_ns}
    channel:      int = -1          # raw SAMPIC channel; -1 for clusters


MAX_LGADS_DEFAULT = 4  # highest number of LGADs supported in one coincidence event


# =============================================================================
# Core event builder
# =============================================================================

def build_events(
    corrdb_path: str,
    run_name: str,
    config,
    output_path: Optional[str] = None,
    coincidence_window_ns: float = 50.0,
    event_building_cfd: int = 50,
    cfd_levels: List[int] = CFD_LEVELS,
    max_lgads: int = MAX_LGADS_DEFAULT,
    lgad_to_slot: Optional[Dict[str, int]] = None,
) -> str:
    """
    Group hits from the corrected-time DB into coincidence events, apply
    the single/cluster/discard logic per LGAD, and compute Δt for every
    (k1, k2) CFD pair.

    Output parquet — one row per valid event
    ----------------------------------------
    EventID                     : int32
    LGAD{1..max_lgads}          : string (board name; null if that slot's LGAD wasn't hit)
    LGAD{1..max_lgads}_NPixels  : int8   (1 = single hit, >1 = cluster; null if unhit)
    LGAD{1..max_lgads}_IsCluster: bool   (null if unhit)
    LGAD{1..max_lgads}_SNR      : float32 (NaN if unhit)
    LGAD{1..max_lgads}_Channel  : int32  (null if unhit)
    Dt_k1_k2                    : float32  LGAD-pair (1,2), all 9×9 = 81 CFD pairs (ns);
                                           kept unprefixed for backward compatibility
    Dt_{i}{j}_k1_k2              : float32  same 81-pair grid for every other
                                           LGAD-pair slot (i,j) — NaN where that pair
                                           isn't present in the event or is not adjacent

    Parameters
    ----------
    corrdb_path : str
        Path to the output of build_full_hit_database (contains
        CFD{k}Time_corr, SampicChannel, LGAD, PixelRow, PixelCol, SNR).
        SNR (along with Amplitude and rise-time) is already filtered at this
        stage via the hit mask used to build corrdb, so no SNR cut is
        applied here — `snrs` is only used downstream for cluster
        SNR-weighting and the reported LGAD1_SNR/LGAD2_SNR columns.
    coincidence_window_ns : float
        Maximum time span from first to last hit in an event window.
    event_building_cfd : int
        CFD level used to sort and group hits into events.
    cfd_levels : list of int
        CFD levels stored in corrdb (must match what was built).
    max_lgads : int
        Highest number of LGADs allowed in one coincidence event (2-4).
        Events with more hit LGADs than this are discarded. Output gets
        LGAD1..LGAD{max_lgads} metadata columns and one Δt matrix per
        LGAD-pair slot — combinations(max_lgads, 2) pairs.
    """

    if output_path is None:
        p = Path(corrdb_path)
        run_dir = Path(run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(run_dir / p.with_stem(p.stem + "_events").name)

    # ------------------------------------------------------------------
    # Load corrdb fully into memory — it's the compact DB, should be fine
    # ------------------------------------------------------------------
    print(f"Loading corrected DB: {corrdb_path}")
    table = pq.read_table(corrdb_path)

    hit_ids   = table["HITNumber"].to_pylist()
    lgads     = table["LGAD"].to_pylist()          # already decoded
    lgad_board_chs = table["SensorPad"].to_pylist()
    snrs      = table["SNR"].to_pylist()

    ordered_times = np.array(table["OrderedCell0Time"].to_pylist(), dtype=np.float64)

    cfd_offsets = {
        k: np.array(table[f"CFD{k}Offset"].to_pylist(), dtype=np.float64)
        for k in cfd_levels
    }
    channels = table["Channel"].to_pylist()

    delays = DELAY_REGISTRY[run_name]

    # Precompute per-hit delay as a float64 array — avoids dict lookup in inner loop
    delay_per_hit = np.array(
        [delays.channel_delays.get(int(ch), np.nan) for ch in channels],
        dtype=np.float64,
    )

    def get_corrected_time(i: int, k: int) -> float:
        offset = cfd_offsets[k][i]
        delay  = delay_per_hit[i]
        if np.isfinite(offset) and np.isfinite(delay):
            return ordered_times[i] + offset - delay
        return np.nan

    n_hits = len(hit_ids)
    print(f"  {n_hits:,} hits loaded")

    # --- Fixed slot assignment: one board = one slot, for the WHOLE run,
    #     not re-derived per event. Prevents slot identity drifting when
    #     fewer than max_lgads boards fire in a given event.
    all_lgad_names = sorted(set(lg for lg in lgads if lg is not None))
    if len(all_lgad_names) > max_lgads:
        raise ValueError(
            f"Found {len(all_lgad_names)} distinct LGAD names in corrdb "
            f"({all_lgad_names}) but max_lgads={max_lgads}. Increase max_lgads "
            f"or check for a naming inconsistency upstream."
        )
    if lgad_to_slot is None:
        print("  [!] No canonical lgad_to_slot given -- deriving one from the "
              "board names present in THIS run only. Slot numbers are only "
              "guaranteed stable within this run; if the active board set "
              "differs between runs, pass an explicit lgad_to_slot= (the same "
              "dict for every run) before pooling slot-indexed Dt_{i}{j}_* "
              "data across runs.")
        lgad_to_slot = {name: i + 1 for i, name in enumerate(all_lgad_names)}
    else:
        missing = [n for n in all_lgad_names if n not in lgad_to_slot]
        if missing:
            raise ValueError(
                f"lgad_to_slot was provided but has no entry for {missing} "
                f"(present in corrdb) -- add them explicitly."
            )
    print(f"Fixed slot assignment: {lgad_to_slot}")

    channels_arr = np.array(channels, dtype=np.int32)
    unique_channels = np.unique(channels_arr)

    # ------------------------------------------------------------------
    # Drop hits with invalid sorting time
    # (SNR/Amplitude/rise-time were already filtered upstream via the hit
    # mask used to build corrdb — no quality cuts are reapplied here.)
    # ------------------------------------------------------------------
    cut_counts = {int(ch): {"invalid_time": 0} for ch in unique_channels}

    valid_mask = []
    for i in range(n_hits):
        t   = get_corrected_time(i, event_building_cfd)
        ch  = int(channels[i])

        passes_time = np.isfinite(t)
        if not passes_time:
            cut_counts[ch]["invalid_time"] += 1

        valid_mask.append(passes_time)

    idx_valid = [i for i, v in enumerate(valid_mask) if v]
    n_valid   = len(idx_valid)

    # Cutflow summary
    print(f"\n  {'Channel':<10} {'Total':>8} {'Bad time':>10} {'Passing':>10}")
    print(f"  {'-'*42}")
    ch_counts = {}
    for i, ch in enumerate(channels):
        ch_counts[int(ch)] = ch_counts.get(int(ch), 0) + 1

    for ch in sorted(unique_channels):
        total    = ch_counts.get(ch, 0)
        bad_time = cut_counts[ch]["invalid_time"]
        passing  = total - bad_time
        print(f"  {ch:<10} {total:>8,} {bad_time:>10,} {passing:>10,}")
    print(f"  {'-'*42}")
    print(f"  {'TOTAL':<10} {n_hits:>8,} "
          f"{sum(c['invalid_time'] for c in cut_counts.values()):>10,} "
          f"{n_valid:>10,}")

    # Sort hits by corrected CFD50 time for event building
    idx_valid.sort(key=lambda i: get_corrected_time(i, event_building_cfd))


    # ------------------------------------------------------------------
    # Coincidence grouping — fixed window from first hit in group
    # ------------------------------------------------------------------
    groups: list[list[int]] = []         # list of [hit_index, ...]
    if n_valid == 0:
        print("No valid hits — nothing to group.")
        return output_path

    current_group  = [idx_valid[0]]
    window_start_t = get_corrected_time(idx_valid[0], event_building_cfd)

    for i in idx_valid[1:]:
        t = get_corrected_time(i, event_building_cfd)
        if t is None or not np.isfinite(t):
            continue
        if t - window_start_t <= coincidence_window_ns:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group  = [i]
            window_start_t = t

    groups.append(current_group)
    print(f"  {len(groups):,} coincidence groups formed")

    # ------------------------------------------------------------------
    # Per-event processing
    # ------------------------------------------------------------------

    pair_slots: List[Tuple[int, int]] = list(combinations(range(1, max_lgads + 1), 2))
    n_cfd      = len(cfd_levels)
    nan_matrix = np.full((n_cfd, n_cfd), np.nan)

    # Storage for output
    out_event_id  = []
    out_lgad      = {slot: [] for slot in range(1, max_lgads + 1)}
    out_npix      = {slot: [] for slot in range(1, max_lgads + 1)}
    out_cluster   = {slot: [] for slot in range(1, max_lgads + 1)}
    out_snr       = {slot: [] for slot in range(1, max_lgads + 1)}
    out_ch        = {slot: [] for slot in range(1, max_lgads + 1)}

    # One (n_cfd x n_cfd) Δt matrix per event per LGAD-pair slot; NaN-filled
    # when that pair isn't present in a given event (fewer than max_lgads hit)
    out_dt_matrices: Dict[Tuple[int, int], list] = {pair: [] for pair in pair_slots}

    stats = {
            "total_groups":          len(groups),
            "discarded_lt2lgad":     0,
            "discarded_toomanylgad": 0,          
            "discarded_case3":       0,
            "discarded_nonadjacent_pairs": 0,    # Updated to track discarded pairs instead of events
            "kept":                  0,
            "cluster_events":        0,
    }
    useful_lgads: Dict[str, int] = {}

    event_id = 0
    
    
    for group in groups:

        # --- Partition hits by LGAD ---
        lgad_hits: Dict[str, list[int]] = {}
        for i in group:
            lg = lgads[i]
            lgad_hits.setdefault(lg, []).append(i)

        n_lgads = len(lgad_hits)

        # Require between 2 and max_lgads LGADs in coincidence
        if n_lgads < 2:
            stats["discarded_lt2lgad"] += 1
            continue
        if n_lgads > max_lgads:
            stats["discarded_toomanylgad"] += 1
            continue

        lgad_names = sorted(list(lgad_hits.keys()))

        # --- Per-LGAD clustering ---
        lgad_representatives: Dict[str, Optional[_LGADHit]] = {}
        lgad_all_pixels: Dict[str, List[Tuple[int, int]]] = {} # Flat list for cross-sensor check
        discard = False

        for lg, hit_indices in lgad_hits.items():
            
            # 1. Gather pixel coordinates PER HIT
            hits_pixels_list = []
            all_pixels_flat = []
            
            for i in hit_indices:
                pad_id = lgad_board_chs[i]
                pad_coords = config.sensor_channels[pad_id]
                hits_pixels_list.append(pad_coords)
                all_pixels_flat.extend(pad_coords)
                
            # Store the flat list of all possible pixels for the cross-sensor adjacency check later
            lgad_all_pixels[lg] = all_pixels_flat
            
            n_hits = len(hit_indices)

            if n_hits == 1:
                # Case 1 — Single hit (even if it physically maps to multiple pixels, it's NOT a cluster)
                i = hit_indices[0]
                snr = float(snrs[i]) if snrs[i] is not None else np.nan
                times = {k: get_corrected_time(i, k) for k in cfd_levels}
                
                lgad_representatives[lg] = _LGADHit(
                    lgad=lg, 
                    n_pixels=1,  # Represents cluster size (1 hit)
                    is_cluster=False,
                    snr=snr, 
                    times=times,
                    channel=int(channels[i]),
                )

            else:
                # Case 2 & 3 — Multiple hits: Check if they are physically connected
                if _is_hit_cluster_connected(hits_pixels_list):
                    # Case 2 — Connected cluster: SNR-weighted average
                    snr_vals = np.array([
                        float(snrs[i]) if snrs[i] is not None else 0.0
                        for i in hit_indices
                    ])
                    snr_total = snr_vals.sum()
                    weights = snr_vals / snr_total if snr_total > 0 else np.ones(n_hits) / n_hits
                    rep_snr = float(snr_total / n_hits)

                    times = {}
                    for k in cfd_levels:
                        raw = np.array([get_corrected_time(i, k) for i in hit_indices])
                        finite = np.isfinite(raw)
                        if finite.any():
                            w = weights.copy()
                            w[~finite] = 0.0
                            w /= w.sum()
                            times[k] = float(np.dot(w, raw))
                        else:
                            times[k] = np.nan

                    lgad_representatives[lg] = _LGADHit(
                        lgad=lg,
                        n_pixels=n_hits, # Cluster size (number of hits combined)
                        is_cluster=True,
                        snr=rep_snr,
                        times=times,
                    )
                else:
                    # Case 3 — Disconnected hits within one LGAD: discard
                    discard = True
                    break

        if discard:
            stats["discarded_case3"] += 1
            continue

        # Fixed-slot placement: physical board -> slot is constant across
        # the whole run, regardless of how many other boards fired this event
        event_slots = {lgad_to_slot[name]: lgad_representatives[name] for name in lgad_names}

        for slot in range(1, max_lgads + 1):
            r = event_slots.get(slot)
            if r is not None:
                out_lgad[slot].append(r.lgad)
                out_npix[slot].append(r.n_pixels)
                out_cluster[slot].append(r.is_cluster)
                out_snr[slot].append(r.snr)
                out_ch[slot].append(r.channel)
            else:
                out_lgad[slot].append(None)
                out_npix[slot].append(None)
                out_cluster[slot].append(None)
                out_snr[slot].append(np.nan)
                out_ch[slot].append(None)

        # --- Δt matrices, one per LGAD-pair slot (NaN-filled if the pair isn't present OR if not adjacent) ---
        event_useful_lgads: Set[str] = set()

        for pair in pair_slots:
            slot_i, slot_j = pair
            r_i = event_slots.get(slot_i)
            r_j = event_slots.get(slot_j)
            if r_i is not None and r_j is not None:
                lg_a, lg_b = r_i.lgad, r_j.lgad
                if _cross_sensor_adjacent(lgad_all_pixels[lg_a], lgad_all_pixels[lg_b]):
                    times_a = np.array([r_i.times[k] for k in cfd_levels])
                    times_b = np.array([r_j.times[k] for k in cfd_levels])
                    out_dt_matrices[pair].append(times_a[:, None] - times_b[None, :])
                    event_useful_lgads.add(lg_a)
                    event_useful_lgads.add(lg_b)
                else:
                    out_dt_matrices[pair].append(nan_matrix)
                    stats["discarded_nonadjacent_pairs"] += 1
            else:
                out_dt_matrices[pair].append(nan_matrix)

        out_event_id.append(event_id)
        
        if any(r.is_cluster for r in event_slots.values()):
            stats["cluster_events"] += 1

        event_id += 1
        stats["kept"] += 1
        for lg in event_useful_lgads:
            useful_lgads[lg] = useful_lgads.get(lg, 0) + 1

    # ------------------------------------------------------------------
    # Write output parquet
    # ------------------------------------------------------------------

    n_events = len(out_event_id)

    table_dict = {
        "EventID": pa.array(out_event_id, type=pa.int32()),
    }

    for slot in range(1, max_lgads + 1):
        table_dict[f"LGAD{slot}"]           = pa.array(out_lgad[slot],    type=pa.string()).dictionary_encode()
        table_dict[f"LGAD{slot}_NPixels"]   = pa.array(out_npix[slot],    type=pa.int8())
        table_dict[f"LGAD{slot}_IsCluster"] = pa.array(out_cluster[slot], type=pa.bool_())
        table_dict[f"LGAD{slot}_SNR"]       = pa.array(out_snr[slot],     type=pa.float32())
        table_dict[f"LGAD{slot}_Channel"]   = pa.array(out_ch[slot],      type=pa.int32())

    for pair in pair_slots:
        stacked = (np.stack(out_dt_matrices[pair], axis=0) if out_dt_matrices[pair]
                   else np.empty((0, n_cfd, n_cfd)))
        for a, k1 in enumerate(cfd_levels):
            for b, k2 in enumerate(cfd_levels):
                # (1, 2) keeps the original unprefixed name for backward compatibility
                colname = f"Dt_{k1}_{k2}" if pair == (1, 2) else f"Dt_{pair[0]}{pair[1]}_{k1}_{k2}"
                table_dict[colname] = pa.array(stacked[:, a, b].astype(np.float32), type=pa.float32())

    table = pa.table(table_dict)
    pq.write_table(table, output_path, compression="zstd")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Event database built")
    print("=" * 60)
    print(f"  Total groups formed             : {stats['total_groups']:>10,}")
    print(f"  Discarded (<2 LGADs)            : {stats['discarded_lt2lgad']:>10,}")
    print(f"  Discarded (disconnected cluster): {stats['discarded_case3']:>10,}")
    print(f"  Discarded non-adjacent *pairs*  : {stats['discarded_nonadjacent_pairs']:>10,}")
    print(f"  Valid events kept               : {stats['kept']:>10,}")
    print(f"  Of which cluster hits           : {stats['cluster_events']:>10,}")
    print(f"LGADS used for event building (valid events per board):")
    for lg, count in sorted(useful_lgads.items()):
        print(f"    {lg:<30}: {count:,}")
    print("=" * 60 + "\n")

    return output_path



#-----------------------------------------
# Determine resolutions from least squares regression, fall back to assuming 2 equal resolutions for 2 planes only
#-----------------------------------------

def solve_detector_resolutions(
    pair_sigmas: Dict[Tuple[int, int], float],
    pair_sigma_errs: Dict[Tuple[int, int], float],
    known_resolutions: Optional[Dict[int, float]] = None
) -> Dict[int, Tuple[float, float]]:
    """
    Solves for individual detector time resolutions using Weighted Least Squares.
    
    Parameters
    ----------
    pair_sigmas : Dict[Tuple[int, int], float]
        Mapping of detector pairs to their measured Gaussian delta-t sigma.
    pair_sigma_errs : Dict[Tuple[int, int], float]
        Mapping of detector pairs to the uncertainty on their delta-t sigma.
    known_resolutions : Dict[int, float], optional
        Mapping of detector indices to a fixed known resolution (e.g., an MCP).
        
    Returns
    -------
    Dict[int, Tuple[float, float]]
        Mapping of detector index to (resolution, error) in the same units provided.
    """
    known = known_resolutions or {}
    
    # Identify all unique detectors across all pairs
    detectors = set()
    for d1, d2 in pair_sigmas.keys():
        detectors.update([d1, d2])
    detectors = sorted(list(detectors))
    
    unknown_dets = [d for d in detectors if d not in known]
    
    # Base case: 2 identical detectors, no knowns provided
    if len(detectors) == 2 and len(unknown_dets) == 2:
        d1, d2 = detectors
        sig = pair_sigmas[(d1, d2)]
        err = pair_sigma_errs[(d1, d2)]
        res = sig / np.sqrt(2)
        res_err = err / np.sqrt(2)
        return {d1: (res, res_err), d2: (res, res_err)}
        
    # WLS regression formulation
    n_eq = len(pair_sigmas)
    n_unk = len(unknown_dets)
    
    A = np.zeros((n_eq, n_unk))
    Y = np.zeros(n_eq)
    W = np.zeros((n_eq, n_eq))
    
    unk_idx = {d: i for i, d in enumerate(unknown_dets)}
    
    for eq_idx, (pair, sig) in enumerate(pair_sigmas.items()):
        d1, d2 = pair
        sig_err = pair_sigma_errs[pair]
        
        # y = sigma_ij^2
        y_val = sig**2
        # Error propagation: delta(sigma^2) = 2 * sigma * delta(sigma)
        sy = 2 * sig * sig_err
        
        if d1 in known:
            y_val -= known[d1]**2
        else:
            A[eq_idx, unk_idx[d1]] = 1
            
        if d2 in known:
            y_val -= known[d2]**2
        else:
            A[eq_idx, unk_idx[d2]] = 1
            
        Y[eq_idx] = y_val
        # W = diag(1 / sy^2)
        W[eq_idx, eq_idx] = 1.0 / (sy**2) if sy > 0 else 1.0

    # r_i^2 + r_j^2 = sigma_ij^2 only pins down every detector in this
    # component if it contains an odd cycle or has >=1 known_resolutions
    # anchor. A chain/tree component leaves a free gauge direction that
    # pinv fills in via its minimum-norm solution -- numeric-looking,
    # deceptively small error bar, not actually determined by the data.
    rank_A = np.linalg.matrix_rank(A) if n_unk > 0 else 0
    if rank_A < n_unk:
        deficiency = n_unk - rank_A
        print(f"  [!] UNDERDETERMINED: {deficiency} unconstrained degree(s) of "
              f"freedom among detectors {unknown_dets}. Returning NaN. Each "
              f"direction below splits these detectors into two groups related "
              f"by a free additive shift -- a known_resolutions anchor for ONE "
              f"detector on either side removes that direction:")
        _, _, Vt = np.linalg.svd(A)
        for k, vec in enumerate(Vt[rank_A:]):
            vec = vec / np.max(np.abs(vec))
            side_a = [unknown_dets[i] for i in range(n_unk) if vec[i] > 0.3]
            side_b = [unknown_dets[i] for i in range(n_unk) if vec[i] < -0.3]
            print(f"      direction {k}: {side_a}  vs  {side_b}")
        X = np.full(n_unk, np.nan)
        Cx = np.full((n_unk, n_unk), np.nan)
    else:
        # Solve WLS: x = (A^T W A)^-1 A^T W Y
        AtW = A.T @ W
        try:
            Cx = np.linalg.pinv(AtW @ A)
            X = Cx @ AtW @ Y
        except np.linalg.LinAlgError:
            X = np.full(n_unk, np.nan)
            Cx = np.full((n_unk, n_unk), np.nan)
        
    results = {}
    for d, val in known.items():
        results[d] = (val, 0.0)
        
    for i, d in enumerate(unknown_dets):
        var = X[i]
        var_err = np.sqrt(Cx[i, i]) if (Cx[i, i] > 0) else np.nan
        
        if var > 0 and np.isfinite(var):
            res = np.sqrt(var)
            # Propagate back to linear: delta(sigma) = delta(sigma^2) / (2 * sigma)
            res_err = var_err / (2 * res)
        else:
            res = np.nan
            res_err = np.nan
            
        results[d] = (res, res_err)
        
    return results


# =============================================================================
# Gaussian helper
# =============================================================================


# =============================================================================
# Gaussian helpers (Kept identical)
# =============================================================================

def _gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _fit_gaussian(
    values: np.ndarray,
    bin_edges: Optional[np.ndarray] = None,
):
    v = values[np.isfinite(values)]
    if len(v) < 10:
        return (np.array([np.nan, np.nan, np.nan]),
                np.full((3, 3), np.nan))

    bins = bin_edges if bin_edges is not None else "auto"
    counts, edges = np.histogram(v, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    filled   = counts > 0
    x_fit    = centres[filled]
    y_fit    = counts[filled].astype(float)
    sigma_y  = np.sqrt(y_fit)          

    if len(x_fit) < 4:
        return (np.array([np.nan, np.nan, np.nan]),
                np.full((3, 3), np.nan))

    p0 = [np.median(v), np.std(v), float(counts[filled].max())]

    try:
        popt, pcov = curve_fit(
            _gaussian,
            x_fit, y_fit,
            p0=p0,
            sigma=sigma_y,
            absolute_sigma=True,
            maxfev=5000,
        )
        return popt, pcov

    except RuntimeError:
        return (np.array([np.nan, np.nan, np.nan]),
                np.full((3, 3), np.nan))

# =============================================================================
# Upgraded Multi-Pair Histogrammer
# =============================================================================
def plot_delta_t_histogram(
    events_path: str,
    pairs: Optional[List[Tuple[int, int]]] = None,
    known_resolutions: Optional[Dict[int, float]] = None,
    k1: int = 50,
    k2: int = 50,
    cluster_filter: Optional[str] = None,
    n_sigma_window: float = 6.0,
    n_sigma_window_std: float = 3.0,
    n_bins: int = 100,       
    axes: Optional[Union[plt.Axes, List[plt.Axes]]] = None,
    llabel: str = "PPS2 Timing Preliminary",
    rlabel: str = "(H8 Test Beam, May 2025)",
    is_data: bool = True,
) -> plt.Figure:
    """
    Fit Gaussian distributions to Delta-t time difference histograms for specified LGAD pairs,
    solve for individual detector resolutions via Weighted Least Squares, and produce 
    CMS-styled plots.
    """
    
    # ------------------------------------------------------------------
    # Step 0: Auto-discover LGAD slots & build pair combinations if None
    # ------------------------------------------------------------------
    if pairs is None:
        try:
            schema = pq.read_schema(events_path)
            discovered_slots = set()
            for name in schema.names:
                if name.startswith("LGAD") and name[4:].isdigit():
                    discovered_slots.add(int(name[4:]))
            
            sorted_slots = sorted(list(discovered_slots))
            if len(sorted_slots) >= 2:
                pairs = list(combinations(sorted_slots, 2))
            else:
                pairs = [(1, 2)]
        except Exception as e:
            print(f"Warning: Failed to inspect schema for LGAD pairs ({e}). Defaulting to [(1, 2)].")
            pairs = [(1, 2)]

    n_pairs = len(pairs)
    
    # ------------------------------------------------------------------
    # Step 1: Extract and Fit ALL pairs first (to feed the WLS solver)
    # ------------------------------------------------------------------
    fit_results = {}
    pair_sigmas_ps = {}
    pair_sigmas_errs_ps = {}
    pair_sigmas_ps_std = {}
    pair_sigmas_errs_ps_std = {}
    slot_to_name = {}  # Map slot number (e.g. 1) to string name (e.g. "Board_A")
    
    for pair in pairs:
        col = f"Dt_{k1}_{k2}" if pair == (1, 2) else f"Dt_{pair[0]}{pair[1]}_{k1}_{k2}"
        s1, s2 = f"LGAD{pair[0]}", f"LGAD{pair[1]}"
        c1, c2 = f"{s1}_IsCluster", f"{s2}_IsCluster"
        
        try:
            table = pq.read_table(events_path, columns=[col, s1, s2, c1, c2])
        except Exception as e:
            print(f"Skipping pair {pair} - columns not found: {e}")
            continue
            
        dt_np = np.array(table[col].to_pylist(), dtype=np.float64)
        is_cluster_1 = np.array(table[c1].to_pylist(), dtype=bool)
        is_cluster_2 = np.array(table[c2].to_pylist(), dtype=bool)
        
        # Capture actual sensor names from the first non-null entry
        if pair[0] not in slot_to_name:
            names1 = [x for x in table[s1].to_pylist() if x is not None]
            slot_to_name[pair[0]] = names1[0] if len(names1) > 0 else f"LGAD {pair[0]}"
        if pair[1] not in slot_to_name:
            names2 = [x for x in table[s2].to_pylist() if x is not None]
            slot_to_name[pair[1]] = names2[0] if len(names2) > 0 else f"LGAD {pair[1]}"

        all_valid = np.isfinite(dt_np)
        cluster_arr = is_cluster_1 | is_cluster_2
        
        if cluster_filter == "only":
            keep = all_valid & cluster_arr
        elif cluster_filter == "exclude":
            keep = all_valid & ~cluster_arr
        else:
            keep = all_valid
            
        vals = dt_np[keep]
        
        if len(vals) < 10:
            continue
            
        centre      = np.median(vals[np.isfinite(vals)])
        kMAD        = 1.4826 * np.median(np.abs(vals[np.isfinite(vals)] - centre))
        window_half = n_sigma_window * kMAD
        
        in_window   = vals[np.abs(vals - centre) < window_half]
        n_out       = len(vals) - len(in_window)
        bin_edges   = np.linspace(centre - window_half, centre + window_half, n_bins + 1)
        
        popt, pcov = _fit_gaussian(in_window, bin_edges=bin_edges)
        mu, sigma, amp = popt
        sigma = abs(float(sigma))
        perr = np.sqrt(np.diag(pcov))
        mu_err, sigma_err, amp_err = perr
        
        fit_results[pair] = {
            "vals": in_window,
            "edges": bin_edges,
            "mu": mu, "mu_err": mu_err,
            "sigma": sigma, "sigma_err": sigma_err,
            "amp": amp, "n_out": n_out
        }
        
        if np.isfinite(sigma) and np.isfinite(sigma_err):
            pair_sigmas_ps[pair] = sigma * 1e3
            pair_sigmas_errs_ps[pair] = sigma_err * 1e3

        sigma_std, sigma_std_err = _raw_std_from_values(vals, n_sigma_window=n_sigma_window_std)
        if np.isfinite(sigma_std) and np.isfinite(sigma_std_err):
            pair_sigmas_ps_std[pair] = sigma_std * 1e3
            pair_sigmas_errs_ps_std[pair] = sigma_std_err * 1e3

    # ------------------------------------------------------------------
    # Step 2: Solve global WLS resolutions -- both methods
    # ------------------------------------------------------------------
    indiv_res = solve_detector_resolutions(pair_sigmas_ps, pair_sigmas_errs_ps, known_resolutions)
    indiv_res_std = solve_detector_resolutions(pair_sigmas_ps_std, pair_sigmas_errs_ps_std, known_resolutions)

    # ------------------------------------------------------------------
    # Step 3: Setup Plot Layout
    # ------------------------------------------------------------------
    if axes is None:
        cols = min(n_pairs, 2)
        rows = (n_pairs - 1) // cols + 1
        fig, axs_arr = plt.subplots(rows, cols, figsize=(7*cols, 5*rows), tight_layout=True)
        axs_list = [axs_arr] if n_pairs == 1 else axs_arr.flatten()
    else:
        axs_list = [axes] if isinstance(axes, plt.Axes) else axes
        fig = axs_list[0].get_figure()

    # ------------------------------------------------------------------
    # Step 4: Draw Histograms
    # ------------------------------------------------------------------
    for idx, pair in enumerate(pairs):
        ax = axs_list[idx]
        name1 = slot_to_name.get(pair[0], f"LGAD {pair[0]}")
        name2 = slot_to_name.get(pair[1], f"LGAD {pair[1]}")
        
        if pair not in fit_results:
            ax.text(0.5, 0.5, f"Insufficient data for {name1} - {name2}", ha='center')
            continue
            
        res = fit_results[pair]
        in_window, bin_edges = res["vals"], res["edges"]
        mu, mu_err = res["mu"], res["mu_err"]
        sigma, sigma_err = res["sigma"], res["sigma_err"]
        amp, n_out = res["amp"], res["n_out"]
        
        bin_width_ps = (bin_edges[1] - bin_edges[0]) * 1e3
        counts, edges = np.histogram(in_window, bins=bin_edges)
        centres = 0.5 * (edges[:-1] + edges[1:])
        poisson_err = np.sqrt(counts.astype(float))
        
        # Plotting elements
        ax.stairs(counts, bin_edges, baseline=0, fill=True, color="#87DFAA", alpha=0.12)
        ax.stairs(counts, bin_edges, color="#2a8a50", lw=1.4, label="Data")
        
        filled = counts > 0
        ax.errorbar(
            centres[filled], counts[filled], yerr=poisson_err[filled],
            fmt="none", ecolor="#2a8a50", elinewidth=1.0, capsize=2, capthick=0.8, zorder=3
        )

        chi2 = -1.0
        ndf  = -1.0
        if np.isfinite(sigma):
            x_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)
            y_model = _gaussian(centres[filled], mu, sigma, amp)
            chi2 = float(np.sum(((counts[filled] - y_model) / np.sqrt(counts[filled])) ** 2))
            ndf = filled.sum() - 3
            
            ax.plot(
                x_fit, _gaussian(x_fit, mu, sigma, amp),
                color="#3C3489", lw=2,
                label=rf"Fit: $\sigma(\Delta t)={sigma*1e3:.1f}\pm{sigma_err*1e3:.1f}$ ps",
                zorder=4,
            )
            ax.axvline(mu, color="#3C3489", lw=1, linestyle="--", alpha=0.6, zorder=3)
        
        ax.set_xlabel(r"$\Delta t$ (ns)", fontsize=12)
        ax.set_ylabel(f"Events | Bin width: {bin_width_ps:.0f} ps", fontsize=12)
        ax.tick_params(axis="both", labelsize=12)
        
        # Resolution text construction using dynamic names
        res1, err1 = indiv_res.get(pair[0], (np.nan, np.nan))
        res2, err2 = indiv_res.get(pair[1], (np.nan, np.nan))
        res1s, err1s = indiv_res_std.get(pair[0], (np.nan, np.nan))
        res2s, err2s = indiv_res_std.get(pair[1], (np.nan, np.nan))
        
        text_str = (
            f"μ = {mu:.4f} ± {mu_err:.4f} ns\n"
            f"σ = {sigma:.4f} ± {sigma_err:.4f} ns\n"
            f"χ²/ndf = {chi2:.1f}/{ndf} = {chi2/max(ndf,1):.2f}\n"
            f"Res({name1}) fit={res1:.1f}±{err1:.1f}  std={res1s:.1f}±{err1s:.1f} ps\n"
            f"Res({name2}) fit={res2:.1f}±{err2:.1f}  std={res2s:.1f}±{err2s:.1f} ps"
        )
        if n_out > 0:
            text_str += f"\n{n_out:,} events outside window"
            
        ax.text(
            0.97, 0.97, text_str,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )
        
        ax.set_title(rf"$\Delta t$ distribution: {name1} - {name2}", fontsize=11, pad=16)
        hep.cms.label(llabel, data=is_data, rlabel=rlabel, loc=0, ax=ax, fontsize=(14, 12, 12, 11))
        ax.legend(fontsize=10)

    # Clean up empty subplots if grid is larger than n_pairs
    if axes is None:
        for j in range(n_pairs, len(axs_list)):
            fig.delaxes(axs_list[j])
            
    fig.subplots_adjust(top=0.82 if n_pairs == 1 else 0.9)

    # ------------------------------------------------------------------
    # Step 5: Console summary print
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"{'TIME RESOLUTION RESULTS (CFD' + str(k1) + ')':^65}")
    print("=" * 65)
    print("Pairs Fitted σ(Δt):")
    print("-" * 65)
    for pair, s_ps in pair_sigmas_ps.items():
        err_ps = pair_sigmas_errs_ps[pair]
        n1 = slot_to_name.get(pair[0], f"Slot {pair[0]}")
        n2 = slot_to_name.get(pair[1], f"Slot {pair[1]}")
        print(f"  • {n1} - {n2:<25} : {s_ps:>6.1f} ± {err_ps:<5.1f} ps")
        
    print("\nIndividual Detector Resolutions (via WLS, fit method):")
    print("-" * 65)
    for slot, (r_ps, r_err_ps) in sorted(indiv_res.items()):
        name = slot_to_name.get(slot, f"Slot {slot}")
        print(f"  • {name:<35} : {r_ps:>6.1f} ± {r_err_ps:<5.1f} ps")
    print("\nIndividual Detector Resolutions (via WLS, std method):")
    print("-" * 65)
    for slot, (r_ps, r_err_ps) in sorted(indiv_res_std.items()):
        name = slot_to_name.get(slot, f"Slot {slot}")
        print(f"  • {name:<35} : {r_ps:>6.1f} ± {r_err_ps:<5.1f} ps")
    print("=" * 65 + "\n")

    return fig

# =============================================================================
# 2D resolution heatmap
# =============================================================================

def plot_resolution_heatmap(
    events_path: str,
    cfd_levels: List[int] = CFD_LEVELS,
    cluster_filter: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    n_bins: int = 200,
    n_contours: int = 8,
    lgad1_label: str = "LGAD 1",
    lgad2_label: str = "LGAD 2",
    llabel: str = "PPS2 Timing Preliminary",
    rlabel: str = "(H8 Test Beam, May 2025)",
    is_data: bool = True,
) -> Tuple[plt.Figure, plt.Figure, float, float, float, float]:

    cols = [f"Dt_{k1}_{k2}" for k1 in cfd_levels for k2 in cfd_levels]
    cols += ["LGAD1_IsCluster", "LGAD2_IsCluster"]
    table = pq.read_table(events_path, columns=cols)

    cluster_arr = (
        np.array(table["LGAD1_IsCluster"].to_pylist(), dtype=bool)
        | np.array(table["LGAD2_IsCluster"].to_pylist(), dtype=bool)
    )

    if cluster_filter == "only":
        row_mask = cluster_arr
    elif cluster_filter == "exclude":
        row_mask = ~cluster_arr
    else:
        row_mask = np.ones(len(table), dtype=bool)

    n = len(cfd_levels)
    
    # 1. Matrices for Gaussian Fit
    res_matrix_fit = np.full((n, n), np.nan)
    err_matrix_fit = np.full((n, n), np.nan)
    
    # 2. Matrices for NumPy Std Dev
    res_matrix_std = np.full((n, n), np.nan)
    err_matrix_std = np.full((n, n), np.nan)

    for i, k1 in enumerate(cfd_levels):
        for j, k2 in enumerate(cfd_levels):
            raw  = np.array(table[f"Dt_{k1}_{k2}"].to_pylist(), dtype=np.float64)
            vals = raw[row_mask & np.isfinite(raw)]
            if len(vals) < 20:
                continue

            centre = np.median(vals)
            kMAD   = 1.4826 * np.median(np.abs(vals - centre))

            # Keep this for the Gaussian fit
            trimmed_fit = vals[np.abs(vals - centre) < 6 * kMAD]
            bin_edges = np.linspace(centre - 6 * kMAD, centre + 6 * kMAD, n_bins + 1)

            # Create a tighter trim strictly for the NumPy STD
            trimmed_std = vals[np.abs(vals - centre) < 3 * kMAD]  # <-- 3 sigma core

            if len(trimmed_fit) < 10 or len(trimmed_std) < 10:
                continue

            # --- Method 1: Gaussian Fit ---
            popt, pcov = _fit_gaussian(trimmed_fit, bin_edges=bin_edges)
            sigma_fit  = abs(float(popt[1]))
            sigma_fit_err = float(np.sqrt(pcov[1, 1])) if np.isfinite(pcov[1, 1]) else np.nan

            if np.isfinite(sigma_fit):
                res_matrix_fit[i, j] = sigma_fit / np.sqrt(2) * 1e3
                err_matrix_fit[i, j] = sigma_fit_err / np.sqrt(2) * 1e3

            # --- Method 2: NumPy Standard Deviation ---
            sigma_std = np.std(trimmed_std)
            sigma_std_err = sigma_std / np.sqrt(2 * len(trimmed_std))
            
            res_matrix_std[i, j] = sigma_std / np.sqrt(2) * 1e3
            err_matrix_std[i, j] = sigma_std_err / np.sqrt(2) * 1e3

    # Print outs for benchmark pair
    idx_50 = cfd_levels.index(50)

    print(f"Fit cfd(50,50): {res_matrix_fit[idx_50, idx_50]:.1f} ± {err_matrix_fit[idx_50, idx_50]:.1f} ps")
    print(f"Std cfd(50,50): {res_matrix_std[idx_50, idx_50]:.1f} ± {err_matrix_std[idx_50, idx_50]:.1f} ps")

    # ------------------------------------------------------------------
    # Reusable Plotting Helper to avoid copy-pasting the layout
    # ------------------------------------------------------------------
    def generate_heatmap(res_matrix, err_matrix, method_title, cbar_label_text):
        if np.all(np.isnan(res_matrix)):
            print(f"WARNING: Not enough data points ( < 20 ) for ANY CFD pair ({method_title}).")
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.text(0.5, 0.5, f"Insufficient Data\n({method_title} Matrix is all NaNs)",
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            return fig, np.nan, np.nan

        res_filled = res_matrix.copy()
        err_filled = err_matrix.copy()
        nan_mask = np.isnan(res_filled)
        if np.any(nan_mask):
            worst_res = np.nanmax(res_filled)
            worst_err = np.nanmax(err_filled)
            res_filled[nan_mask] = worst_res
            err_filled[nan_mask] = worst_err

        fig, ax = plt.subplots(figsize=(7, 6), tight_layout=True)

        flat_min_raw = np.nanargmin(res_matrix)
        i_min_idx, j_min_idx = np.unravel_index(flat_min_raw, res_matrix.shape)

        res_min = res_matrix[i_min_idx, j_min_idx]
        err_min = err_matrix[i_min_idx, j_min_idx]

        zoom_factor = 2
        res_smooth  = zoom(res_filled, zoom_factor, order=1)
        n_smooth = res_smooth.shape[0]

        X, Y = np.meshgrid(
            np.linspace(0, n - 1, n_smooth),
            np.linspace(0, n - 1, n_smooth),
        )

        plot_vmax = vmax if vmax is not None else np.nanpercentile(res_matrix, 90)
        plot_vmin = vmin if vmin is not None else np.nanmin(res_matrix)

        im = ax.imshow(
            res_smooth,
            origin="lower",
            cmap="viridis_r",
            vmin=plot_vmin,
            vmax=plot_vmax,
            aspect="auto",
            extent=[0, n-1, 0, n-1],
        )
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label_text, fontsize=11)
        cbar.ax.tick_params(labelsize=10)

        focus_levels = np.linspace(plot_vmin, plot_vmax, n_contours)
        cs = ax.contour(
            X, Y, res_smooth,
            levels=focus_levels,
            colors="black",
            linewidths=0.8,
            alpha=0.6,
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt=lambda v: f"{v:.1f}p")

        label_text = (f"Min: {res_min:.1f} ± {err_min:.1f} ps\n"
                      f"CFD{cfd_levels[j_min_idx]} / CFD{cfd_levels[i_min_idx]}")

        ax.errorbar(
            j_min_idx, i_min_idx,
            xerr=0.4, yerr=0.4,
            fmt="o",
            color="red",
            markersize=8,
            capsize=0,
            linewidth=1.5,
            label=label_text,
            zorder=5,
        )

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"{k}" for k in cfd_levels], rotation=45, ha="right")
        ax.set_yticklabels([f"{k}" for k in cfd_levels])
        ax.set_xlabel(f"k_{lgad2_label} (%)", fontsize=11)
        ax.set_ylabel(f"k_{lgad1_label} (%)", fontsize=11)
        ax.set_title(f"Detector Timing Resolution Matrix ({method_title})", fontsize=12, pad=16)
        ax.tick_params(axis="both", labelsize=11)
        
        
        hep.cms.label(llabel, data=is_data, rlabel=rlabel, loc=0, ax=ax, fontsize=(14, 12, 12, 11))
        ax.legend(fontsize=9, loc="lower left")
        legend = ax.legend(fontsize=9, loc="lower left")
        for text in legend.get_texts():
            text.set_color("white")
        return fig, res_min, err_min

    # Generate both figures seamlessly
    fig_fit, res_min_fit, err_min_fit = generate_heatmap(
        res_matrix_fit, err_matrix_fit, 
        method_title="Gaussian Fit", 
        cbar_label_text=r"Time Resolution $\sigma(\Delta t) / √2$ (ps)"
    )
    
    fig_std, res_min_std, err_min_std = generate_heatmap(
        res_matrix_std, err_matrix_std, 
        method_title="NumPy Std Dev", 
        cbar_label_text=r"Time Resolution $\text{STD}(\Delta t) / √2$ (ps)"
    )

    return fig_fit, fig_std, res_min_fit, err_min_fit, res_min_std, err_min_std

 
 
# =============================================================================
# Core analysis helper (shared with existing functions)
# =============================================================================
 
def _resolution_from_values(
    vals: np.ndarray,
    n_bins: int,
    n_sigma_window: float,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, float, int]:
    """
    Gaussian fit on kMAD-windowed data.
    Returns: mu, mu_err, sigma, sigma_err, bin_edges, counts, chi2, ndf
    All in ns.  Resolution = sigma / sqrt(2).
    """
    vals = vals[np.isfinite(vals)]
    if len(vals) < 20:
        return np.nan, np.nan, np.nan, np.nan, np.array([]), np.array([]), np.nan, 0

    centre    = np.median(vals)
    kMAD      = 1.4826 * np.median(np.abs(vals - centre))
    hw        = n_sigma_window * kMAD
    trimmed   = vals[np.abs(vals - centre) < hw]
    bin_edges = np.linspace(centre - hw, centre + hw, n_bins + 1)

    popt, pcov = _fit_gaussian(trimmed, bin_edges=bin_edges)
    mu, sigma, amp = popt
    sigma = abs(float(sigma))
    mu_err, sigma_err, _ = np.sqrt(np.diag(pcov))

    counts, _ = np.histogram(trimmed, bins=bin_edges)
    centres_b = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    filled    = counts > 0
    if filled.sum() >= 4 and np.isfinite(sigma):
        y_model = _gaussian(centres_b[filled], mu, sigma, amp)
        chi2    = float(np.sum(((counts[filled] - y_model) / np.sqrt(counts[filled])) ** 2))
        ndf     = int(filled.sum()) - 3
    else:
        chi2, ndf = np.nan, 0

    return mu, mu_err, sigma, sigma_err, bin_edges, counts, chi2, ndf
 
 
def _raw_std_from_values(
    vals: np.ndarray,
    n_sigma_window: float = 3.0,
) -> Tuple[float, float]:
    """
    Unbinned std-dev sigma of the raw Delta-t distribution (kMAD core
    trim), NOT divided by sqrt(2) -- the std-based counterpart to
    _fit_gaussian's raw `sigma`. Use this (not _std_resolution_from_values)
    when feeding solve_detector_resolutions, which expects the raw pair
    sigma (sigma_ij^2 = r_i^2 + r_j^2), not a pre-divided resolution.
    Returns (sigma_ns, sigma_err_ns).
    """
    vals = vals[np.isfinite(vals)]
    if len(vals) < 20:
        return np.nan, np.nan

    centre  = np.median(vals)
    kMAD    = 1.4826 * np.median(np.abs(vals - centre))
    trimmed = vals[np.abs(vals - centre) < n_sigma_window * kMAD]
    if len(trimmed) < 20:
        return np.nan, np.nan

    sigma     = np.std(trimmed)
    sigma_err = sigma / np.sqrt(2 * len(trimmed))
    return sigma, sigma_err


def _std_resolution_from_values(
    vals: np.ndarray,
    n_sigma_window: float = 3.0,
) -> Tuple[float, float]:
    """
    Unbinned std-dev resolution (3σ_kMAD core trim, matching heatmap).
    Returns (resolution_ps, error_ps) where resolution = std / sqrt(2).
    """
    sigma, sigma_err = _raw_std_from_values(vals, n_sigma_window)
    if not np.isfinite(sigma):
        return np.nan, np.nan
    return sigma / np.sqrt(2) * 1e3, sigma_err / np.sqrt(2) * 1e3


##----Helper to plot histograms----##
def _draw_histogram_with_errors(
    ax: plt.Axes,
    counts: np.ndarray,
    bin_edges: np.ndarray,
    mu: float,
    sigma: float,
    sigma_err: float,
    amp: float,
) -> None:
    """
    Draw a Δt histogram with:
      • Professional unfilled step outline with a faint background tint
      • Poisson √N error bars on every filled bin (no empty bin markers)
      • Gaussian fit curve (without uncertainty band)

    Parameters
    ----------
    ax         : matplotlib Axes object
    counts     : event counts per bin from np.histogram
    bin_edges  : edges used for histogramming and fit
    mu, sigma, amp : Gaussian fit parameters (ns)
    sigma_err  : uncertainty on sigma from covariance (ns)
    """
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    poisson_err = np.sqrt(counts.astype(float))   # √N per bin

    # --- 1. Subtle, clean background fill underneath the data ---
    ax.stairs(
        counts, bin_edges,
        baseline=0, fill=True,
        color="#87DFAA", alpha=0.12,
    )

    # --- 2. Professional step outline (Unfilled edge) ---
    ax.stairs(
        counts, bin_edges,
        color="#2a8a50", lw=1.4,
        label="Data",
    )

    # --- 3. Centralized Poisson error bars (Only where data exists) ---
    filled = counts > 0
    ax.errorbar(
        centres[filled], counts[filled],
        yerr=poisson_err[filled],
        fmt="none",
        ecolor="#2a8a50",
        elinewidth=1.0,
        capsize=2,
        capthick=0.8,
        zorder=3,
    )

    # Note: The 'if np.any(~filled):' empty bin block has been completely removed.

    if not np.isfinite(sigma):
        return

    # --- 4. Gaussian fit curve ---
    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)
    y_fit = _gaussian(x_fit, mu, sigma, amp)
    ax.plot(
        x_fit, y_fit,
        color="#3C3489", lw=2,
        label=rf"Gaussian fit  $\sigma={sigma*1e3:.1f}\pm{sigma_err*1e3:.1f}$ ps",
        zorder=4,
    )

    # --- 5. μ line ---
    ax.axvline(mu, color="#3C3489", lw=1, linestyle="--", alpha=0.55, zorder=3)


def plot_channel_pair_histograms(
    events_path: str,
    corrdb_path: str,
    config,
    cfd_levels: List[int] = CFD_LEVELS,
    cfd_plot: int = 50,
    n_sigma_window: float = 6.0,
    n_sigma_window_std: float = 3.0,
    n_bins: int = 100,
    output_dir: Optional[str] = None,
    plot_pairs: Optional[List[Tuple[int, int]]] = None,
    pooled_res_fit_ps: Optional[float] = None,
    pooled_res_fit_err_ps: Optional[float] = None,
    pooled_res_std_ps: Optional[float] = None,
    pooled_res_std_err_ps: Optional[float] = None,
    llabel: str = "PPS2 Timing Preliminary",
    rlabel: str = "(H8 Test Beam, May 2025)",
    is_data: bool = True,
) -> Dict[Tuple[int, int], plt.Figure]:
    """For every adjacent cross-board channel pair, produce a Δt histogram

    (CFD `cfd_plot`/`cfd_plot`) with Poisson error bars and a Gaussian fit.
    
    Parameters
    ----------
    plot_pairs : Optional[List[Tuple[int, int]]]
        If provided, limits the generation of plots to only these specific 
        channel pairs (e.g., `[(2, 3), (15, 16)]`) to save memory. Summary 
        statistics are still calculated for all pairs.
    """
    # 1. Channel → pixel coordinates map from corrdb
    print("Building channel→pixels map from config and corrdb …")
    meta = pq.read_table(corrdb_path, columns=["Channel", "SensorPad"])
    ch_arr = np.array(meta["Channel"].to_pylist(), dtype=np.int32)
    pad_arr = np.array(meta["SensorPad"].to_pylist())

    unique_indices = np.unique(ch_arr, return_index=True)[1]
    pixel_map: Dict[int, List[Tuple[int, int]]] = {}

    for idx in unique_indices:
        ch = int(ch_arr[idx])
        pad = pad_arr[idx]
        pixel_map[ch] = config.sensor_channels[pad]

    del meta
    print(f"  {len(pixel_map)} unique channels mapped.")

    # 2. Load events — fixed physical slots
    max_lgads = getattr(config, "max_lgads", MAX_LGADS_DEFAULT)
    pair_slots = list(combinations(range(1, max_lgads + 1), 2))

    needed_cols = []
    for slot in range(1, max_lgads + 1):
        needed_cols.extend([f"LGAD{slot}_Channel", f"LGAD{slot}_NPixels"])

    for slot_i, slot_j in pair_slots:
        for k1 in cfd_levels:
            for k2 in cfd_levels:
                colname = f"Dt_{k1}_{k2}" if (slot_i, slot_j) == (1, 2) else f"Dt_{slot_i}{slot_j}_{k1}_{k2}"
                needed_cols.append(colname)

    print("Loading events parquet …")
    tbl = pq.read_table(events_path, columns=needed_cols)
    n_events = tbl.num_rows
    print(f"  {n_events:,} events loaded.")

    slot_channels = {
        slot: np.array(tbl[f"LGAD{slot}_Channel"].to_pylist(), dtype=object)
        for slot in range(1, max_lgads + 1)
    }
    slot_npix = {
        slot: np.array(tbl[f"LGAD{slot}_NPixels"].to_pylist(), dtype=object)
        for slot in range(1, max_lgads + 1)
    }

    dt_arrays: Dict[Tuple[int, int, int, int], np.ndarray] = {}
    for slot_i, slot_j in pair_slots:
        for k1 in cfd_levels:
            for k2 in cfd_levels:
                colname = f"Dt_{k1}_{k2}" if (slot_i, slot_j) == (1, 2) else f"Dt_{slot_i}{slot_j}_{k1}_{k2}"
                dt_arrays[(slot_i, slot_j, k1, k2)] = np.array(tbl[colname].to_pylist(), dtype=np.float32)

    del tbl

    # 3. Enumerate unique adjacent cross-board channel pairs
    adjacent_pairs, skipped_pairs = [], []

    for slot_i, slot_j in pair_slots:
        ch_i, ch_j = slot_channels[slot_i], slot_channels[slot_j]
        np_i, np_j = slot_npix[slot_i], slot_npix[slot_j]

        valid = np.array([
            ci is not None and cj is not None and ni is not None and nj is not None and int(ni) == 1 and int(nj) == 1
            for ci, cj, ni, nj in zip(ch_i, ch_j, np_i, np_j)
        ])

        if not np.any(valid):
            continue

        pair_keys = set(zip(ch_i[valid].astype(int).tolist(), ch_j[valid].astype(int).tolist()))

        for c1, c2 in sorted(pair_keys):
            ps1, ps2 = pixel_map.get(c1), pixel_map.get(c2)

            if ps1 is None or ps2 is None:
                skipped_pairs.append((slot_i, slot_j, c1, c2, "coord missing"))
            elif _cross_sensor_adjacent(ps1, ps2):
                adjacent_pairs.append((slot_i, slot_j, c1, c2))
            else:
                skipped_pairs.append((slot_i, slot_j, c1, c2, f"not adjacent {ps1}↔{ps2}"))

    print(f"\n  Adjacent pairs : {len(adjacent_pairs)}")
    print(f"  Skipped pairs  : {len(skipped_pairs)}")
    for slot_i, slot_j, c1, c2, reason in skipped_pairs:
        print(f"    slots {slot_i}–{slot_j}, ch{c1}–ch{c2} : {reason}")

    if not adjacent_pairs:
        print("No adjacent pairs found — nothing to plot.")
        return {}

    # 4. Per-channel-pair boolean masks
    pair_mask: Dict[Tuple[int, int, int, int], np.ndarray] = {}

    for slot_i, slot_j, c1, c2 in adjacent_pairs:
        ch_i, ch_j = slot_channels[slot_i], slot_channels[slot_j]
        np_i, np_j = slot_npix[slot_i], slot_npix[slot_j]

        pair_mask[(slot_i, slot_j, c1, c2)] = np.array([
            ci is not None and cj is not None and ni is not None and nj is not None 
            and int(ci) == c1 and int(cj) == c2 and int(ni) == 1 and int(nj) == 1
            for ci, cj, ni, nj in zip(ch_i, ch_j, np_i, np_j)
        ])

    # 5. Output directory
    out_path = Path(output_dir) if output_dir is not None else None
    if out_path is not None:
        for parent in reversed(out_path.parents):
            if parent.exists() and not parent.is_dir():
                raise FileExistsError(f"'{parent}' exists as a file, not a directory.")
        out_path.mkdir(parents=True, exist_ok=True)

    # 6. Per-pair analysis + figure
    figures: Dict[Tuple[int, int], plt.Figure] = {}
    summary_rows = []

    print(f"\nProcessing {len(adjacent_pairs)} adjacent pairs …\n")
    total_n_ev = 0

    for slot_i, slot_j, c1, c2 in adjacent_pairs:
        mask = pair_mask[(slot_i, slot_j, c1, c2)]
        n_ev = int(mask.sum())
        total_n_ev += n_ev

        p1, p2 = pixel_map[c1], pixel_map[c2]
        print(f"  slots {slot_i}–{slot_j}: ch{c1}{p1} — ch{c2}{p2}  :  {n_ev:,} events", end="")

        if n_ev < 20:
            print("  [SKIP — too few events]")
            continue

        # CFD scans
        best_res, best_err = np.inf, np.nan
        best_k1, best_k2 = cfd_plot, cfd_plot

        best_res_std, best_err_std = np.inf, np.nan
        best_k1_std, best_k2_std = cfd_plot, cfd_plot

        for k1 in cfd_levels:
            for k2 in cfd_levels:
                v = dt_arrays[(slot_i, slot_j, k1, k2)][mask].astype(np.float64)
                _, _, sigma, sigma_err, *_ = _resolution_from_values(v, n_bins=n_bins, n_sigma_window=n_sigma_window)

                if np.isfinite(sigma):
                    r = sigma / np.sqrt(2) * 1e3
                    if r < best_res:
                        best_res = r
                        best_err = sigma_err / np.sqrt(2) * 1e3 if np.isfinite(sigma_err) else np.nan
                        best_k1, best_k2 = k1, k2

                r_std, e_std = _std_resolution_from_values(v, n_sigma_window_std)
                if np.isfinite(r_std) and r_std < best_res_std:
                    best_res_std, best_err_std = r_std, e_std
                    best_k1_std, best_k2_std = k1, k2

        # CFD_plot resolutions
        vals_50 = dt_arrays[(slot_i, slot_j, cfd_plot, cfd_plot)][mask].astype(np.float64)
        mu50, mu50_err, sig50, sig50_err, bin_edges, counts, chi2, ndf = _resolution_from_values(
            vals_50, n_bins=n_bins, n_sigma_window=n_sigma_window
        )

        res50 = sig50 / np.sqrt(2) * 1e3 if np.isfinite(sig50) else np.nan
        res50_err = sig50_err / np.sqrt(2) * 1e3 if np.isfinite(sig50_err) else np.nan
        res50_std, res50_std_err = _std_resolution_from_values(vals_50, n_sigma_window_std)

        print(
            f"  CFD{cfd_plot}/{cfd_plot}: fit={res50:.1f} ps  std={res50_std:.1f} ps  |  "
            f"Best fit CFD{best_k1}/{best_k2}: {best_res:.1f} ps  "
            f"Best std CFD{best_k1_std}/{best_k2_std}: {best_res_std:.1f} ps"
        )

        summary_rows.append((
            slot_i, slot_j, c1, c2, n_ev, mu50, mu50_err, res50, res50_err,
            res50_std, res50_std_err, best_k1, best_k2, best_res, best_err,
            best_k1_std, best_k2_std, best_res_std, best_err_std
        ))

        # ==============================================================================
        # Plotting (Filtered by plot_pairs to save memory)
        # ==============================================================================
        should_plot = plot_pairs is None or [c1, c2] in plot_pairs or [c2, c1] in plot_pairs
        
        if should_plot:
            fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
            bin_width_ps = (bin_edges[1] - bin_edges[0]) * 1e3 if len(bin_edges) > 1 else np.nan
            amp50 = float(counts.max()) if len(counts) > 0 else 1.0

            _draw_histogram_with_errors(
                ax=ax, counts=counts, bin_edges=bin_edges, mu=mu50, 
                sigma=sig50, sigma_err=sig50_err, amp=amp50
            )

            # Outliers calculation
            n_out = 0
            if np.isfinite(mu50):
                v_fin = vals_50[np.isfinite(vals_50)]
                if len(v_fin) > 0:
                    kMAD_50 = 1.4826 * np.median(np.abs(v_fin - np.median(v_fin)))
                    n_out = int(np.sum(np.abs(v_fin - mu50) >= n_sigma_window * kMAD_50))

            # Decorate Axes
            ax.set_xlabel(r"$\Delta t$ (ns)", fontsize=12)
            ax.set_ylabel(f"Events / {bin_width_ps:.0f} ps" if np.isfinite(bin_width_ps) else "Events", fontsize=12)
            ax.tick_params(axis="both", labelsize=12)

            res_str = f"\nTime resolution = {res50:.1f} ± {res50_err:.1f} ps" if np.isfinite(res50) else ""
            ax.set_title(
                rf"$\Delta t$ distribution  (CFD{cfd_plot}/CFD{cfd_plot})\n"
                f"slots {slot_i}–{slot_j}: ch{c1} {p1} — ch{c2} {p2}{res_str}",
                fontsize=11, pad=16
            )
            fig.subplots_adjust(top=0.82)

            # Safeguard in case hep is not imported globally
            try:
                hep.cms.label(llabel, data=is_data, rlabel=rlabel, loc=0, ax=ax, fontsize=(14, 12, 12, 11))
            except Exception:
                pass
                
            ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.01, 0.99), framealpha=0.85)

            # Stats Overlay
            out_str = f"\n{n_out:,} events outside window" if n_out > 0 else ""
            textstr = (
                f"μ = {mu50:.4f} ± {mu50_err:.4f} ns\n"
                f"σ = {sig50:.4f} ± {sig50_err:.4f} ns\n"
                f"χ²/ndf = {chi2:.1f}/{ndf} = {chi2/max(ndf, 1):.2f}\n"
                f"Res (fit) = {res50:.1f} ± {res50_err:.1f} ps\n"
                f"Res (std) = {res50_std:.1f} ± {res50_std_err:.1f} ps\n"
                f"Best fit: CFD{best_k1}/CFD{best_k2} → {best_res:.1f} ± {best_err:.1f} ps\n"
                f"Best std: CFD{best_k1_std}/CFD{best_k2_std} → {best_res_std:.1f} ± {best_err_std:.1f} ps"
                f"{out_str}"
            )

            ax.text(
                0.97, 0.97, textstr, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
            )

            figures[(c1, c2)] = fig

            if output_dir is not None:
                fig.savefig(Path(output_dir) / f"dt_pair_ch{c1}_ch{c2}.png", dpi=300, bbox_inches="tight")

    if plot_pairs is not None and not figures:
        available = sorted(set((c1, c2) for _, _, c1, c2 in adjacent_pairs))
        print(f"  [!] plot_pairs={plot_pairs} matched none of the {len(available)} "
              f"available adjacent channel pairs -- available: {available}. "
              f"(plot_pairs takes SAMPIC channel numbers, not slot numbers.)")

    # 7. Summary table

    # 7. Summary table
    W = 155
    print("\n" + "=" * W)
    print(
        f"  {'Slots':<8} {'Pair':<12} {'N ev':>7}  {'mu_fit (ps)':>14}  "
        f"{'CFD50 fit (ps)':>16}  {'CFD50 std (ps)':>16}  {'Best fit CFD':>13}  "
        f"{'Best fit res (ps)':>18}  {'Best std CFD':>13}  {'Best std res (ps)':>18}"
    )
    print("  " + "-" * (W - 2))

    for row in summary_rows:
        (slot_i, slot_j, c1, c2, n_ev, mu50, mu50_err, r50, e50, r50s, e50s,
         bk1, bk2, br, be, bk1s, bk2s, brs, bes) = row

        _fmt = lambda r, e: f"{r:.1f} ± {e:.1f}" if np.isfinite(r) else "N/A"
        mu_str = f"{mu50 * 1e3:.1f} ± {mu50_err * 1e3:.1f}" if np.isfinite(mu50) else "N/A"
        percentage = (n_ev / total_n_ev * 100) if total_n_ev > 0 else 0.0

        print(
            f"  {f'{slot_i}–{slot_j}':<8} {f'ch{c1}–ch{c2}':<12} {n_ev:>7,}({percentage:.2f}%)  "
            f"{mu_str:>14}  {_fmt(r50, e50):>16}  {_fmt(r50s, e50s):>16}  "
            f"{f'CFD{bk1}/{bk2}':>13}  {_fmt(br, be):>18}  "
            f"{f'CFD{bk1s}/{bk2s}':>13}  {_fmt(brs, bes):>18}"
        )
    print("=" * W)

    # 8. Summary DataFrame
    summary_df = pd.DataFrame([
        {
            "Slots": f"{row[0]}-{row[1]}",
            "Pair": f"ch{row[2]}-ch{row[3]}",
            "Events": row[4],
            "Fraction (%)": (100 * row[4] / total_n_ev) if total_n_ev > 0 else 0.0,
            "μ (ps)": row[5] * 1e3,
            "μ err (ps)": row[6] * 1e3,
            "CFD50 fit (ps)": row[7],
            "CFD50 fit err": row[8],
            "CFD50 std (ps)": row[9],
            "CFD50 std err": row[10],
            "Best fit CFD": f"{row[11]}/{row[12]}",
            "Best fit (ps)": row[13],
            "Best fit err": row[14],
            "Best std CFD": f"{row[15]}/{row[16]}",
            "Best std (ps)": row[17],
            "Best std err": row[18],
        }
        for row in summary_rows
    ])

    print(summary_df.round(2))
    if output_dir is not None:
        summary_df.to_csv(Path(output_dir) / "pair_summary.csv", index=False)

    # 9. Mean-spread diagnostic
    if pooled_res_fit_ps is not None:
        pair_results_fit = [(r[2], r[3], r[4], r[5], r[6], r[7], r[8]) for r in summary_rows]
        diagnose_mean_spread(
            pair_results=pair_results_fit,
            pooled_resolution_ps=pooled_res_fit_ps,
            pooled_resolution_err_ps=pooled_res_fit_err_ps,
            method_label="Gaussian fit",
        )

    if pooled_res_std_ps is not None:
        pair_results_std = [(r[2], r[3], r[4], r[5], r[6], r[9], r[10]) for r in summary_rows]
        diagnose_mean_spread(
            pair_results=pair_results_std,
            pooled_resolution_ps=pooled_res_std_ps,
            pooled_resolution_err_ps=pooled_res_std_err_ps,
            method_label="Std dev",
        )

    return figures


def solve_pixel_resolutions(
    events_path: str,
    corrdb_path: str,
    config,
    cfd_eval: int = 50,
    n_sigma_window: float = 6.0,
    n_sigma_window_std: float = 3.0,
    n_bins: int = 100,
    min_events: int = 20,
    known_resolutions: Optional[Dict[int, float]] = None
) -> pd.DataFrame:
    """Builds a system of paired delta-t equations from adjacent cross-sensor pixels,

    isolates mathematically independent subgraphs, and solves for the individual
    resolution of each pixel using Weighted Least Squares per subgraph.
    """
    # ------------------------------------------------------------------
    # 1. Build channel → list[(row, col)] map
    # ------------------------------------------------------------------
    print("Building channel→pixels map from config and corrdb …")
    meta = pq.read_table(corrdb_path, columns=["Channel", "SensorPad"])
    
    ch_arr = np.array(meta["Channel"].to_pylist(), dtype=np.int32)
    pad_arr = np.array(meta["SensorPad"].to_pylist())
    unique_indices = np.unique(ch_arr, return_index=True)[1]
    
    pixel_map: Dict[int, List[Tuple[int, int]]] = {
        int(ch_arr[idx]): config.sensor_channels[pad_arr[idx]]
        for idx in unique_indices
    }
    del meta
    
    # ------------------------------------------------------------------
    # 2. Load events dynamically across physical LGAD slots
    # ------------------------------------------------------------------
    print(f"Loading events for CFD {cfd_eval}/{cfd_eval} …")
    max_lgads = getattr(config, "max_lgads", MAX_LGADS_DEFAULT)
    pair_slots = list(combinations(range(1, max_lgads + 1), 2))

    needed_cols = []
    for slot in range(1, max_lgads + 1):
        needed_cols.extend([f"LGAD{slot}_Channel", f"LGAD{slot}_NPixels"])

    for slot_i, slot_j in pair_slots:
        colname = f"Dt_{cfd_eval}_{cfd_eval}" if (slot_i, slot_j) == (1, 2) else f"Dt_{slot_i}{slot_j}_{cfd_eval}_{cfd_eval}"
        needed_cols.append(colname)

    tbl = pq.read_table(events_path, columns=needed_cols)

    # Use dtype=object to safely handle None values from unhit slots
    slot_channels = {
        slot: np.array(tbl[f"LGAD{slot}_Channel"].to_pylist(), dtype=object)
        for slot in range(1, max_lgads + 1)
    }
    slot_npix = {
        slot: np.array(tbl[f"LGAD{slot}_NPixels"].to_pylist(), dtype=object)
        for slot in range(1, max_lgads + 1)
    }

    dt_arrays: Dict[Tuple[int, int], np.ndarray] = {}
    for slot_i, slot_j in pair_slots:
        colname = f"Dt_{cfd_eval}_{cfd_eval}" if (slot_i, slot_j) == (1, 2) else f"Dt_{slot_i}{slot_j}_{cfd_eval}_{cfd_eval}"
        dt_arrays[(slot_i, slot_j)] = np.array(tbl[colname].to_pylist(), dtype=np.float32)

    del tbl

    # ------------------------------------------------------------------
    # 3. Enumerate valid adjacent pairs and extract sigma -- both fit and
    #    std, from the same events and the same min_events cut, so the
    #    two are directly comparable.
    # ------------------------------------------------------------------
    pair_sigmas_ps: Dict[str, Dict[Tuple[int, int], float]] = {"fit": {}, "std": {}}
    pair_sigmas_errs_ps: Dict[str, Dict[Tuple[int, int], float]] = {"fit": {}, "std": {}}

    print("Fitting valid adjacent pixel pairs to populate WLS matrix (fit + std)...")

    for slot_i, slot_j in pair_slots:
        ch_i, ch_j = slot_channels[slot_i], slot_channels[slot_j]
        np_i, np_j = slot_npix[slot_i], slot_npix[slot_j]

        valid = np.array([
            ci is not None and cj is not None and ni is not None and nj is not None and int(ni) == 1 and int(nj) == 1
            for ci, cj, ni, nj in zip(ch_i, ch_j, np_i, np_j)
        ])

        if not np.any(valid):
            continue

        pair_keys = set(zip(ch_i[valid].astype(int).tolist(), ch_j[valid].astype(int).tolist()))
        dt_arr = dt_arrays[(slot_i, slot_j)]

        for c1, c2 in sorted(pair_keys):
            ps1, ps2 = pixel_map.get(c1), pixel_map.get(c2)
            if ps1 is None or ps2 is None or not _cross_sensor_adjacent(ps1, ps2):
                continue

            pair_mask = valid & np.array([
                ci is not None and cj is not None and int(ci) == c1 and int(cj) == c2
                for ci, cj in zip(ch_i, ch_j)
            ])

            v = dt_arr[pair_mask].astype(np.float64)

            if len(v) < min_events:
                continue

            _, _, sigma_fit, sigma_fit_err, *_ = _resolution_from_values(
                v, n_bins=n_bins, n_sigma_window=n_sigma_window
            )
            if np.isfinite(sigma_fit) and np.isfinite(sigma_fit_err):
                pair_sigmas_ps["fit"][(c1, c2)] = sigma_fit * 1e3
                pair_sigmas_errs_ps["fit"][(c1, c2)] = sigma_fit_err * 1e3

            sigma_std, sigma_std_err = _raw_std_from_values(v, n_sigma_window=n_sigma_window_std)
            if np.isfinite(sigma_std) and np.isfinite(sigma_std_err):
                pair_sigmas_ps["std"][(c1, c2)] = sigma_std * 1e3
                pair_sigmas_errs_ps["std"][(c1, c2)] = sigma_std_err * 1e3

    print(f"  fit: {len(pair_sigmas_ps['fit'])} pairs.  std: {len(pair_sigmas_ps['std'])} pairs.")

    # ------------------------------------------------------------------
    # 4-5. Graph connectivity + WLS solve, once per method -- the set of
    #    pairs passing each method's own finite-value cut can differ.
    # ------------------------------------------------------------------
    def _solve_method(sigmas_ps, errs_ps, label):
        print(f"Analyzing pixel connectivity graph ({label})...")
        adj: Dict[int, Set[int]] = {}
        for u, v in sigmas_ps.keys():
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        visited: Set[int] = set()
        components: List[Set[int]] = []
        for node in adj.keys():
            if node not in visited:
                comp = set()
                queue = [node]
                while queue:
                    curr = queue.pop(0)
                    if curr not in visited:
                        visited.add(curr)
                        comp.add(curr)
                        queue.extend(adj[curr] - visited)
                components.append(comp)
        print(f"  ({label}) Discovered {len(components)} independent connected component(s).")

        master: Dict[int, Tuple[float, float]] = {}
        for idx, comp in enumerate(components, 1):
            if len(comp) < 3:
                has_ref = known_resolutions and any(n in known_resolutions for n in comp)
                if not has_ref:
                    print(f"  [{label} cluster {idx}] Isolated {len(comp)}-pixel cluster, no "
                          f"reference -- unsolvable. Listing as NaN: {sorted(comp)}")
                    for ch in comp:
                        master[ch] = (np.nan, np.nan)
                    continue

            sub_sigmas = {k: v for k, v in sigmas_ps.items() if k[0] in comp and k[1] in comp}
            sub_errs = {k: v for k, v in errs_ps.items() if k[0] in comp and k[1] in comp}
            print(f"  [{label} cluster {idx}] Solving {len(comp)} pixels with {len(sub_sigmas)} pairwise equations...")
            try:
                comp_res = solve_detector_resolutions(sub_sigmas, sub_errs, known_resolutions)
                master.update(comp_res)
            except Exception as e:
                print(f"  [{label} cluster {idx}] Error solving system: {e}")
                for ch in comp:
                    master.setdefault(ch, (np.nan, np.nan))
        return master

    master_indiv_res_fit = _solve_method(pair_sigmas_ps["fit"], pair_sigmas_errs_ps["fit"], "fit")
    master_indiv_res_std = _solve_method(pair_sigmas_ps["std"], pair_sigmas_errs_ps["std"], "std")

    # ------------------------------------------------------------------
    # 6. Format Output -- both methods side by side
    # ------------------------------------------------------------------
    all_channels = sorted(set(master_indiv_res_fit) | set(master_indiv_res_std))
    records = []
    for ch in all_channels:
        res_fit, err_fit = master_indiv_res_fit.get(ch, (np.nan, np.nan))
        res_std, err_std = master_indiv_res_std.get(ch, (np.nan, np.nan))
        records.append({
            "Channel": ch,
            "Mapped_Pixels": str(pixel_map.get(ch, "Unknown")),
            "Resolution_ps_fit": res_fit,
            "Error_ps_fit": err_fit,
            "Resolution_ps_std": res_std,
            "Error_ps_std": err_std,
        })

    res_df = pd.DataFrame(records)

    print("\n" + "=" * 90)
    print(f"{'SOLVED PIXEL RESOLUTIONS (CFD ' + str(cfd_eval) + ')':^90}")
    print("=" * 90)
    print(f"{'Channel':<10} | {'Mapped Pixels':<20} | {'Fit (ps)':>16} | {'Std (ps)':>16}")
    print("-" * 90)
    for _, row in res_df.iterrows():
        fit_str = f"{row['Resolution_ps_fit']:.1f} ± {row['Error_ps_fit']:.1f}" if np.isfinite(row['Resolution_ps_fit']) else "N/A"
        std_str = f"{row['Resolution_ps_std']:.1f} ± {row['Error_ps_std']:.1f}" if np.isfinite(row['Resolution_ps_std']) else "N/A"
        print(f" {row['Channel']:<9} | {str(row['Mapped_Pixels']):<20} | {fit_str:>16} | {std_str:>16}")
    print("=" * 90)

    return res_df



# =============================================================================
# Execution Pipeline Example
# =============================================================================
if __name__ == "__main__":
    import numpy as np

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    RAW_PARQUET = "Run_0xx.parquet"

    # [!] Assume `my_delays` and `my_config` are instantiated earlier in your code
    # my_delays = Channel_Delays(...)
    # my_config = ConfigInformation(...)

    # Optional: A mask of verified good hit IDs from an earlier filtering step
    # valid_hit_mask = np.array([10, 11, 15, 102, ...])

    # -------------------------------------------------------------------------
    # 2. Build the hit-level database (1 row = 1 hit)
    # -------------------------------------------------------------------------
    # print(">>> STEP 1: Building unified hit database...")
    # corrdb_file = build_full_hit_database(
    #     raw_parquet_path=RAW_PARQUET,
    #     delays=my_delays,
    #     config=my_config,
    #     # hit_mask=valid_hit_mask,   # Uncomment to skip unverified hits instantly
    #     cfd_levels=CFD_LEVELS,
    #     batch_size=100_000,          # Adjust based on your RAM limits
    #     min_amplitude=0.01
    # )

    # -------------------------------------------------------------------------
    # 3. Build the event-level database (1 row = 1 coincidence event)
    # -------------------------------------------------------------------------
    print("\n>>> STEP 2: Grouping hits into coincidence events...")
    events_file = build_events(
        corrdb_path= "Run_0xx_full_hit_db.parquet",
        coincidence_window_ns= 1000.0, # 1 microsecond window
        event_building_cfd=50,        # Sort chronological groups using CFD 50%
        min_snr= 5                   # Aggressive cut: ignore noisy hits
    )

    # -------------------------------------------------------------------------
    # 4. Generate Visual Diagnostics
    # -------------------------------------------------------------------------
    print("\n>>> STEP 3: Generating diagnostic plots...")

    # Plot A: 1D Histogram for the 50% - 50% CFD pair, EXCLUDING clusters
    fig_hist = plot_delta_t_histogram(
        events_path=events_file,
        k1=50,
        k2=50,
        cluster_filter="exclude",
        bins=200
    )
    fig_hist.savefig("dt_histogram_CFD50.png", dpi=300)
    print("Saved -> dt_histogram_CFD50.png")

    # Plot B: 2D Heatmap of time resolution across ALL CFD pairs
    fig_heatmap = plot_resolution_heatmap(
        events_path=events_file,
        cluster_filter=None           # Include single AND cluster events
    )
    fig_heatmap.savefig("resolution_heatmap.png", dpi=300)
    print("Saved -> resolution_heatmap.png")

    # Render plots to screen if running interactively
    plt.show()

    print("\n>>> Pipeline complete.")