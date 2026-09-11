# -*- coding: utf-8 -*-
"""
channel_delay_calibration.py

Weighted graph-Laplacian channel delay calibration for PPS2.

Replaces the hand-built DELAY_REGISTRY entries with a least-squares solve
over channel-pair coincidence edges, reusing the SAME peak-detection/fit
pipeline and thresholds already used in coincidence_histograms.py's
plot_coincidence_histograms_multipeak (detect_peaks, fit_multiple_peaks,
remove_duplicate_fits, _is_sane_fit) -- nothing new to tune, whatever you've
already verified visually in that plot carries over directly.

Pipeline
--------
1. collect_delay_edges() -> for each pivot channel (upper-triangular dedup,
   so each physical pair is only computed once, not both directions),
   build the coincidence histogram, run the existing detect/fit/sanity
   pipeline. Among fits that pass _is_sane_fit, take the highest-amplitude
   one -- there is exactly one real Δt per channel pair; a secondary peak
   is a reflection artifact (cable/trace echo), not an independent
   measurement, and is discarded rather than used as its own edge. If NONE
   pass, the pair is simply skipped -- never forced, never filled in from
   a prior. Missing edges are fine as long as the overall graph stays
   connected (checked via n_components); a pair with no clean peak against
   one pivot is often clean against a different one.
2. solve_channel_delays() -> weighted graph-Laplacian LS solve (weight =
   1/mu_err^2) for tau_i (per-channel delay), plus chi2/ndof closure
   diagnostics and per-edge residuals/pulls.
3. build_pivot_filter_map() / build_upper_triangular_filter_map() -> keep
   the edge set to physically-adjacent pairs, each measured once.
4. run_calibration() -> runs the above end to end.
5. build_channel_delays_object() -> packages a solve into a
   delays_db.Channel_Delays object, ready for DELAY_REGISTRY.

No hand-calibrated prior or previous run's output is used or needed
anywhere in this file. add_systematic_floor() is available as an optional
secondary refinement (widen sigma_mu by a quadrature floor before solving)
but is NOT applied by default and the algorithm does not depend on it.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from coincidence_histograms import (
    build_coincidence_histograms,
    detect_peaks,
    fit_multiple_peaks,
    merge_close_peaks,
    remove_duplicate_fits,
    _is_sane_fit,
)
from delays_db import Channel_Delays
from delta_t_analysis import get_pixel_map_from_config, _cross_sensor_adjacent
# =============================================================================
# Optional secondary refinement -- NOT applied by default
# =============================================================================

def add_systematic_floor(edges, floor_ps):
    # floor_ps: minimum uncertainty (in ps) added in quadrature to every
    # edge's sigma_mu before the LS solve, so no single edge (especially a
    # high-n, artificially-tight fit) can dominate the fit purely from
    # underestimated statistical error. Tune by scanning until chi2_red ~ 1;
    # check tau barely moves vs. unfloored -> confirms it's fixing the
    # diagnostic, not the answer. Optional: the core algorithm does not
    # require this, since amplitude-based selection + skipping pairs with
    # no sane peak already prevents any single bad edge from existing at all.
    floor_ns = floor_ps * 1e-3
    return [
        (ci, cj, mu, np.sqrt(sigma**2 + floor_ns**2), n)
        for (ci, cj, mu, sigma, n) in edges
    ]


# =============================================================================
# Physically-restricted pivot -> partner-channel map
# =============================================================================

def build_pivot_filter_map(config, pivot_channels: List[int]) -> Dict[int, List[int]]:
    """
    For each pivot channel, restrict comparison channels to those with
    Chebyshev-adjacent pixels (same board or cross-sensor), reusing the
    adjacency logic already in delta_t_analysis.py (get_pixel_map_from_config,
    _cross_sensor_adjacent).
    """
    

    pixel_map = get_pixel_map_from_config(config)
    filter_map: Dict[int, List[int]] = {}
    for ch in pivot_channels:
        if ch not in pixel_map.keys():
            print(ch, "MISSING FROM PIXEL MAP")

    for p in pivot_channels:
        if p not in pixel_map:
            continue
        p_pixels = pixel_map[p]
        partners = [
            ch for ch, pixels in pixel_map.items()
            if ch != p and _cross_sensor_adjacent(p_pixels, pixels)
        ]
        filter_map[p] = partners

    return filter_map


# =============================================================================
# Upper-triangular dedup -- full pairwise coverage, no duplicate direction
# =============================================================================

def build_upper_triangular_filter_map(
    pivot_channels: List[int],
    adjacency_filter_map: Optional[Dict[int, List[int]]] = None,
) -> Dict[int, List[int]]:
    """
    For each pivot, restrict partners to channels with a HIGHER index than
    the pivot. Looping pivot over every channel this way still covers every
    unordered pair exactly once (each pair's pivot is its lower-index
    channel) -- full coverage, without ever computing both (i pivot -> j)
    and (j pivot -> i) for the same pair.

    If adjacency_filter_map is given (from build_pivot_filter_map), the
    physical-adjacency restriction is applied on top.
    """
    sorted_channels = sorted(pivot_channels)
    filter_map: Dict[int, List[int]] = {}

    for i, p in enumerate(sorted_channels):
        higher = sorted_channels[i + 1:]
        if adjacency_filter_map is not None:
            allowed = set(adjacency_filter_map.get(p, []))
            higher = [ch for ch in higher if ch in allowed]
        if higher:
            filter_map[p] = higher

    return filter_map


# =============================================================================
# Edge collection -- reuses the existing detect/fit/sanity pipeline as-is
# =============================================================================

def collect_delay_edges(
    parquet_path: str,
    pivot_channels: List[int],
    channel_filter_map: Optional[Dict[int, List[int]]] = None,
    hit_mask=None,
    cfd50_path: Optional[str] = None,
    x_range: float = 30.0,
    bin_width: float = 0.05,
    prominence_frac: float = 0.1,
    min_height_frac: float = 0.01,
    min_distance_ns: float = 0.5,
    smooth_sigma_bins: float = 0.0,
    fit_window: float = 10.0,
    sigma_max: float = 2.0,
    merge_distance_ns: float = 0.6,
    min_counts_for_fit: int = 20,
    max_peaks: Optional[int] = 4,
    verbose: bool = True, #Just means it will print out some status messages along the way
) -> List[Tuple[int, int, float, float, int]]:
    """
    Build the calibration edge list: one (ch_i, ch_j, mu_ij, sigma_ij, n)
    tuple per pair with at least one sane fitted peak, meaning
    tau_i - tau_j = mu_ij (mu_ij = mean of t_i - t_j).

    Parameters mirror plot_coincidence_histograms_multipeak's tunables
    exactly, so whatever you've already visually verified there (clean
    single peak, borderline-but-usable, or garbage) transfers directly.

    Selection rule per pair: among fits passing _is_sane_fit, take the
    highest-amplitude one -- one real Δt per channel pair; a secondary
    peak is a reflection artifact, not an independent measurement. If none
    pass, the pair is skipped -- never forced, never inferred from a
    prior. Missing edges are fine as long as the graph stays connected;
    check via solve_channel_delays()'s n_components.
    """
    edges: List[Tuple[int, int, float, float, int]] = []
    

    for pivot in pivot_channels:
        channel_edges: List[Tuple[int, int, float, float, int]] = []
        ch_filter = channel_filter_map.get(pivot) if channel_filter_map else None
        try:
            histograms = build_coincidence_histograms(
                parquet_path=parquet_path,
                pivot_channel=pivot,
                channel_filter=ch_filter,
                x_range=x_range,
                bin_width=bin_width,
                use_cfd50=True,
                hit_mask=hit_mask,
                cfd50_path=cfd50_path,
            )
        except Exception as e:
            print(f"Error in building histogram (skipping): {e}")
            continue


        for ch, result in histograms.items():
            if result.total_counts < min_counts_for_fit:
                if verbose:
                    print(f"  [skip] ch{ch}-ch{pivot}: {result.total_counts} "
                          f"counts < min_counts_for_fit={min_counts_for_fit}")
                continue

            peaks, _ = detect_peaks(
                result,
                prominence_frac=prominence_frac,
                min_height_frac=min_height_frac,
                min_distance_ns=min_distance_ns,
                smooth_sigma_bins=smooth_sigma_bins,
            )
            if len(peaks) == 0:
                if verbose:
                    print(f"  [skip] ch{ch}-ch{pivot}: no peaks detected")
                continue

            # tallest first, then cap, then re-sort spatially -- matches
            # plot_coincidence_histograms_multipeak's own ordering exactly
            counts = result.counts.astype(np.float64)
            peaks = peaks[np.argsort(counts[peaks])[::-1]]
            if max_peaks is not None:
                peaks = peaks[:max_peaks]
            peaks = peaks[np.argsort(result.bin_centres[peaks])]

            peaks = merge_close_peaks(
                peaks, result.bin_centres, result.counts, merge_distance_ns
            )
            fit_results = fit_multiple_peaks(
                result, peaks, fit_window=fit_window, sigma_max=sigma_max
            )
            if fit_results is None:
                if verbose:
                    print(f"  [skip] ch{ch}-ch{pivot}: fit failed")
                continue
            if len(fit_results) > 1:
                fit_results = remove_duplicate_fits(fit_results, merge_distance_ns)

            seed_positions = list(result.bin_centres[peaks])
            sane = [
                fr for fr in fit_results
                if _is_sane_fit(
                    fr,
                    min(seed_positions, key=lambda s: abs(s - fr["mu"])),
                    fit_window,
                    sigma_max,
                )
            ]
            if not sane:
                if verbose:
                    print(f"  [skip] ch{ch}-ch{pivot}: no sane fit among "
                          f"{len(fit_results)} peak(s)")
                continue

            if verbose and len(sane) > 1:
                print(f"  [note] ch{ch}-ch{pivot}: {len(sane)} sane peaks "
                      f"detected, using tallest (mu = "
                      f"{[round(fr['mu'], 3) for fr in sane]})")

            chosen = max(sane, key=lambda fr: fr["amplitude"])
            edges.append((ch, pivot, chosen["mu"], chosen["mu_err"], result.total_counts))
            channel_edges.append((ch, pivot, chosen["mu"], chosen["mu_err"], result.total_counts))

        if verbose and len(channel_edges)>0:
            print(f"Edges selected for ch {pivot}:")
            print(channel_edges)
            print("-"*10)

    return edges


# =============================================================================
# Weighted graph-Laplacian LS solve
# =============================================================================

def solve_channel_delays(
    edges: List[Tuple[int, int, float, float, int]],
    channels: List[int],
    reference_channel: Optional[int] = None,
    min_edge_counts: int = 0,
) -> dict:
    """
    Weighted least-squares solve for per-channel delays tau_i from pairwise
    edges (ch_i, ch_j, mu_ij, sigmamu_ij, n_counts) meaning tau_i - tau_j = mu_ij.

    Uses all edges simultaneously (inverse-variance weighted: 1/sigma_mu^2),
    gauge-fixed to the minimum-norm solution via lstsq, optionally
    re-anchored to a chosen reference_channel afterward (pure shift,
    doesn't change fit quality). Returns per-edge residuals/pulls and
    chi2/ndof as the closure diagnostic.
    """
    edges = [
        e for e in edges
        if e[4] >= min_edge_counts and np.isfinite(e[2]) and e[3] > 0
    ]
    if not edges:
        raise ValueError("No valid edges to solve.")

    idx = {ch: k for k, ch in enumerate(sorted(channels))}
    n_ch = len(idx)
    n_edges = len(edges)

    A = np.zeros((n_edges, n_ch))
    b = np.zeros(n_edges)
    sigma = np.zeros(n_edges)

    for k, (ci, cj, mu, mu_err, _n) in enumerate(edges):
        A[k, idx[ci]] = 1.0
        A[k, idx[cj]] = -1.0
        b[k] = mu
        sigma[k] = mu_err

    w = 1.0 / sigma
    A_w = A * w[:, None]
    b_w = b * w

    tau, _, rank, _ = np.linalg.lstsq(A_w, b_w, rcond=None)

    if reference_channel is not None:
        tau = tau - tau[idx[reference_channel]]

    resid = A @ tau - b
    pull = resid / sigma
    chi2 = float(np.sum(pull ** 2))

    rows = [idx[e[0]] for e in edges] + [idx[e[1]] for e in edges]
    cols = [idx[e[1]] for e in edges] + [idx[e[0]] for e in edges]
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_ch, n_ch))
    n_components, labels = connected_components(adj, directed=False)

    if n_components > 1:
        print(
            f"[!] Calibration graph has {n_components} disconnected components — "
            f"channels in different components have no relative delay constraint "
            f"between them. Check coverage for: "
            f"{[ch for ch in idx if labels[idx[ch]] != labels[idx[channels[0]]]]}"
        )

    ndof = n_edges - (n_ch - n_components)

    return {
        "tau": {ch: float(tau[idx[ch]]) for ch in idx},
        "chi2": chi2,
        "ndof": ndof,
        "chi2_red": chi2 / ndof if ndof > 0 else float("nan"),
        "edge_residuals": [
            {"ch_i": e[0], "ch_j": e[1], "mu": e[2], "sigma": e[3],
             "n": e[4], "resid": float(r), "pull": float(p)}
            for e, r, p in zip(edges, resid, pull)
        ],
        "n_components": n_components,
        "component_labels": {ch: int(labels[idx[ch]]) for ch in idx},
    }


# =============================================================================
# Full pipeline driver
# =============================================================================

def run_calibration(
    parquet_path: str,
    channels: List[int],
    config=None,
    hit_mask=None,
    cfd50_path: Optional[str] = None,
    reference_channel: Optional[int] = None,
    floor_ps: float = 0.0,
    **edge_kwargs,
) -> dict:
    """
    Physical adjacency filter (if config given) + upper-triangular dedup
    -> edge collection -> (optional floor) -> LS solve.

    floor_ps=0.0 by default -- the algorithm does not depend on it; pass a
    nonzero value only if solve_channel_delays()'s chi2_red diagnostic
    shows a real need for it after the fact.
    """
    adjacency_map = build_pivot_filter_map(config, channels) if config is not None else None
    filter_map = build_upper_triangular_filter_map(channels, adjacency_map)
    pivots = sorted(filter_map.keys())

    edges = collect_delay_edges(
        parquet_path=parquet_path,
        pivot_channels=pivots,
        channel_filter_map=filter_map,
        hit_mask=hit_mask,
        cfd50_path=cfd50_path,
        **edge_kwargs,
    )
    print(f"\n{len(edges)} edges collected.")

    if floor_ps > 0.0:
        edges = add_systematic_floor(edges, floor_ps)

    result = solve_channel_delays(edges, channels, reference_channel=reference_channel)
    result["edges"] = edges
    return result


# =============================================================================
# Writer -- package a solve into a delays_db.Channel_Delays object
# =============================================================================

def build_channel_delays_object(
    solve_result: dict,
    name: str,
    reference_channel: int,
):
    """
    Package a solve_channel_delays() result into a delays_db.Channel_Delays
    object, ready to drop into DELAY_REGISTRY[run_name].

    Sign convention (verified against Run018's hand entries): channel_delays[ch]
    is defined so that corrected_time = raw_time + cfd_offset - channel_delays[ch].
    The edges here are built as mu_ij = mean(t_i - t_j) with tau_i - tau_j = mu_ij,
    so solve_result["tau"], once shifted to zero at reference_channel, drops in
    with no sign flip needed.
    """
    

    tau = dict(solve_result["tau"])
    shift = tau.get(reference_channel, 0.0)
    if shift != 0.0:
        tau = {ch: v - shift for ch, v in tau.items()}

    connections_info = {}
    for e in solve_result["edge_residuals"]:
        connections_info[(e["ch_i"], e["ch_j"])] = {
            "delta_t":  e["mu"],
            "Amp[Cts]": e["n"],        # coincidence counts, not peak amplitude
            "resid_ps": e["resid"] * 1e3,
            "pull":     e["pull"],
            "degree":   None,          # no tree structure — every edge is direct
        }

    return Channel_Delays(
        name=name,
        channel_origin=reference_channel,
        channel_delays=tau,
        connections_info=connections_info,
    )


def print_channel_delays_literal(cd, varname):
    """Pretty-print a Channel_Delays object as paste-able source for delays_db.py."""
    print(f"{varname} = Channel_Delays(")
    print(f'    name="{cd.name}",')
    print(f"    channel_origin={cd.channel_origin},\n")

    print("    channel_delays={")
    for ch, v in sorted(cd.channel_delays.items()):
        print(f"        {ch:>3}: {v:>9.5f},")
    print("    },\n")

    print("    connections_info={")
    for (i, j), info in sorted(cd.connections_info.items()):
        pair = f"({i}, {j})"
        print(
            f"        {pair:>9}: "
            f"{{'delta_t': {info['delta_t']:>9.5f}, "
            f"'Amp[Cts]': {info['Amp[Cts]']:>6}, "
            f"'resid_ps': {info['resid_ps']:>+7.2f}, "
            f"'pull': {info['pull']:>+5.2f}, "
            f"'degree': {info['degree']}}},"
        )
    print("    }")
    print(")")