"""
plot_coincidence_diagnostic.py

For each matched pivot/target pair it shows TWO panels side by side:
  LEFT  — buffer-start aligned  (pivot buffer start = t=0)
  RIGHT — CFD50 aligned         (pivot 50% crossing = t=0)

Visual style matches plot_channel_waveforms:
  - sinc-interpolated lines
  - '>' buffer-start marker
  - '.' hit-sample markers
  - 'x' trigger markers
  - CMS label + set_waveform_titles_and_labels axes labelling

Key pipeline additions vs the original plot_coincidence_waveforms:
  1. times_raw is kept in lockstep through valid_mask and sort
  2. cfd50_offset per hit = times[idx] - times_raw[idx]
  3. _shift_new_artists() lets us call plot_waveform() then slide its
     output along the x-axis, so we get the full interpolation +
     marker machinery for free.

Usage:
    plot_coincidence_waveforms_diagnostic(
        parquet_path   = "...",
        pivot_channel  = 8,
        target_channel = 9,
        peaks          = peaks,
        hit_mask       = hit_mask,
        n_total        = 6,
        cfd50_path     = "...",
    )
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
#Define functions for lables
from pathlib import Path
from matplotlib.axes import Axes
from typing import Optional, List
import mplhep as hep

from sampiclyser.sampic_tools import (
    plot_waveform,
    finalize_waveform_legend,
    apply_interpolation_method,
    reorder_circular_samples_with_trigger,
)
from filter_functions import compute_cfd50, load_cfd50_lookup

# Taken from: https://stackoverflow.com/a/20007730
def ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix

def set_waveform_titles_and_labels(
    ax: Axes,
    file_path: Path,
    file_name_id: Optional[str] = None,
    title: Optional[str] = None,
    channel_filter: Optional[List[int]] = None,
    first_hit: int = 0,
    hits_plotted: int = 1,
    time_scale: float = 1.0,
) -> None:
    
    auto = (title is None)

    if not file_name_id:
        file_name_id = file_path.name

    if auto:
        qualifier = ""
        if channel_filter is not None:
            if len(channel_filter) == 1:
                qualifier = f" Channel {channel_filter[0]}"
            else:
                qualifier = " Selected Channels"

        if first_hit == 0:
            prefix = f"First {hits_plotted}"
            suffix = ""
        else:
            prefix = f"{hits_plotted} sequential"
            suffix = f" after {ordinal(first_hit)} hit"

        ax.set_title(
            f"{prefix}{qualifier} Waveforms from {file_name_id}{suffix}",
            pad=12,
            weight="bold"
        )
    else:
        ax.set_title(title, pad=12, weight="bold")

    # Determine time units
    units_map = {
        1.0:    "s",
        1e3:   "ms",
        1e6:   "µs",
        1e9:   "ns",
        1e12:  "ps",
    }
    try:
        units = units_map[time_scale]
    except KeyError:
        raise RuntimeError(f"Unknown time scale: {time_scale}")

    ax.set_xlabel(f"Time [{units}]")
    ax.set_ylabel("Voltage [V]")



# ======================================================================
# Internal helper: shift artists added by plot_waveform
# ======================================================================

def _count_artists(ax: plt.Axes) -> tuple[int, int]:
    """Return (n_lines, n_collections) currently on ax."""
    return len(ax.lines), len(ax.collections)


def _shift_new_artists(
    ax: plt.Axes,
    n_lines_before: int,
    n_colls_before: int,
    x_offset: float,
) -> None:
    """
    Shift all artists added since the last _count_artists call by x_offset.
    Works on Line2D objects and PathCollection (scatter) objects.
    """
    for line in ax.lines[n_lines_before:]:
        line.set_xdata(line.get_xdata() + x_offset)

    for coll in ax.collections[n_colls_before:]:
        offsets = coll.get_offsets().copy()
        offsets[:, 0] += x_offset
        coll.set_offsets(offsets)



# ======================================================================
# Main diagnostic function
# ======================================================================



def plot_coincidence_waveforms_diagnostic(
    parquet_path: str,
    pivot_channel: int,
    target_channel: int,
    peaks: list[dict],
    hit_mask: Optional[np.ndarray] = None,
    n_total: int = 6,
    n_sigma: float = 2.0,
    period: float = 1 / 6.4,          # ns per sample
    use_cfd50: bool = True,
    cfd50_path: Optional[str] = None,
    batch_size: int = 100_000,
    time_col: str = "OrderedCell0Time",
    interpolation_method: str = "sinc",
    interpolation_factor: int = 4,
    interpolation_parameter: int = 8,
    cms_label: str = "PPS2 Timing Preliminary",
    rlabel: str = "(H8 Test Beam, May 2025)",
    is_data: bool = True,
) -> None:
    """
    Two-panel diagnostic coincidence plotter with full visual parity to
    plot_channel_waveforms (sinc interpolation, buffer/trigger markers).

    LEFT panel  — buffer-start aligned
        Pivot buffer start at t=0.  Target shifted by its OrderedCell0Time
        offset.  Vertical dashed lines show where CFD50 fired in each buffer —
        if Peak 1 (Δt≈−10 ns) is a wrap artifact the target CFD marker will
        sit at or near a buffer edge while the visible pulse is elsewhere.

    RIGHT panel — CFD50 aligned
        Pivot 50% crossing forced to t=0.  Target crossing appears at Δt.
        Use this to judge pulse-shape consistency once CFD is confirmed valid.

    Parameters
    ----------
    peaks : list of dict with keys 'mu', 'sigma', 'label'
    n_total : int
        Max pairs per peak (keep small — each pair produces two panels).
    period : float
        Sample period in **ns**.  Default 1/6.4 ns (6.4 GSa/s SAMPIC rate).
    
    parquet_path: str,
    pivot_channel: int,
    target_channel: int,
    peaks: list[dict],
    hit_mask: Optional[np.ndarray] = None,
    n_total: int = 6,
    n_sigma: float = 2.0,
    period: float = 1 / 6.4,          # ns per sample
    use_cfd50: bool = True,
    cfd50_path: Optional[str] = None,
    batch_size: int = 100_000,
    time_col: str = "OrderedCell0Time",
    interpolation_method: str = "sinc",
    interpolation_factor: int = 4,
    interpolation_parameter: int = 8,
    cms_label: str = "PPS2 Timing Preliminary",
    rlabel: str = "(H8 Test Beam, May 2025)",
    is_data: bool = True,
    """

    plt.style.use(hep.style.CMS)

    interp_kwargs = dict(
        interpolation_method=interpolation_method,
        interpolation_factor=interpolation_factor,
        interpolation_parameter=interpolation_parameter,
    )

    # time_scale=1.0 because period is already in ns — no unit conversion needed
    time_scale = 1.0

    # ------------------------------------------------------------------
    # Load parquet
    # ------------------------------------------------------------------

    cols = [
        "HITNumber", "Channel", time_col,
        "Baseline", "DataSample", "TriggerPosition",
    ]

    pqf = pq.ParquetFile(parquet_path)
    collected = []
    mask_set = set(hit_mask.tolist()) if hit_mask is not None else None

    for batch in pqf.iter_batches(batch_size=batch_size, columns=cols):
        hit_ids = batch["HITNumber"].to_numpy()
        if mask_set is not None:
            keep = np.fromiter(
                (hid in mask_set for hid in hit_ids),
                dtype=bool, count=len(hit_ids),
            )
            if not np.any(keep):
                continue
            batch = batch.take(pa.array(np.where(keep)[0]))
        collected.append(batch)

    if not collected:
        raise RuntimeError("No hits survived masking.")

    table = pa.Table.from_batches(collected)

    # ------------------------------------------------------------------
    # Extract arrays
    # ------------------------------------------------------------------

    hit_ids  = table["HITNumber"].combine_chunks().to_numpy().astype(np.int64)
    channels = table["Channel"].combine_chunks().to_numpy().astype(np.int32)

    times_raw = (
        table[time_col].combine_chunks()
        .to_numpy(zero_copy_only=False).astype(np.float64)
    )
    baselines = (
        table["Baseline"].combine_chunks()
        .to_numpy(zero_copy_only=False).astype(np.float64)
    )

    samples_col  = table["DataSample"].combine_chunks()
    triggers_col = table["TriggerPosition"].combine_chunks()

    times = times_raw.copy()

    # ------------------------------------------------------------------
    # CFD50 correction
    # ------------------------------------------------------------------

    if use_cfd50:

        valid_mask = np.ones(len(times), dtype=bool)

        if cfd50_path is not None:
            print("Loading CFD50 lookup...")
            cfd50_lut = load_cfd50_lookup(cfd50_path)
            failed = 0
            for i, hid in enumerate(hit_ids):
                offset = cfd50_lut.get(int(hid))
                if offset is None:
                    valid_mask[i] = False
                    failed += 1
                else:
                    times[i] += offset
            print(f"CFD50 lookup failures: {failed:,} / {len(times):,}")
        else:
            failed = 0
            for i in tqdm(range(len(times)), desc="Computing CFD50"):
                offset = compute_cfd50(
                    np.asarray(samples_col[i].as_py(), dtype=np.float64),
                    np.asarray(triggers_col[i].as_py(), dtype=np.int32),
                    baselines[i],
                    period,
                )
                if offset is None:
                    valid_mask[i] = False
                    failed += 1
                else:
                    times[i] += offset
            print(f"CFD50 failures: {failed:,} / {len(times):,}")

        # Mask all arrays consistently — times_raw included
        times     = times[valid_mask]
        times_raw = times_raw[valid_mask]      # ← addition vs original
        hit_ids   = hit_ids[valid_mask]
        channels  = channels[valid_mask]
        baselines = baselines[valid_mask]
        vi = np.where(valid_mask)[0]
        samples_col  = samples_col.take(pa.array(vi))
        triggers_col = triggers_col.take(pa.array(vi))

    # ------------------------------------------------------------------
    # Sort by corrected time — keep times_raw in lockstep
    # ------------------------------------------------------------------

    order     = np.argsort(times, kind="stable")
    times     = times[order]
    times_raw = times_raw[order]               # ← addition vs original
    hit_ids   = hit_ids[order]
    channels  = channels[order]
    baselines = baselines[order]
    samples_col  = samples_col.take(pa.array(order))
    triggers_col = triggers_col.take(pa.array(order))

    # ------------------------------------------------------------------
    # Split pivot / target
    # ------------------------------------------------------------------

    pivot_mask  = channels == pivot_channel
    target_mask = channels == target_channel

    pivot_times   = times[pivot_mask]
    target_times  = times[target_mask]
    pivot_indices  = np.where(pivot_mask)[0]
    target_indices = np.where(target_mask)[0]

    print(f"Pivot hits : {len(pivot_times):,}")
    print(f"Target hits: {len(target_times):,}")

    # ------------------------------------------------------------------
    # Waveform accessor (raw arrays only — plot_waveform handles the rest)
    # ------------------------------------------------------------------

    def get_raw(idx: int) -> dict:
        return dict(
            hid      = int(hit_ids[idx]),
            channel  = int(channels[idx]),
            baseline = float(baselines[idx]),
            samp     = np.asarray(samples_col[idx].as_py(),  dtype=np.float64),
            trig     = np.asarray(triggers_col[idx].as_py(), dtype=np.int32),
            # CFD50 crossing position within the ordered buffer (ns from buf start)
            cfd50_offset = float(times[idx] - times_raw[idx]),
            buf_len_ns   = len(samples_col[idx].as_py()) * period,
        )

    # ------------------------------------------------------------------
    # Per-peak loop
    # ------------------------------------------------------------------

    for peak_i, peak in enumerate(peaks):

        mu    = peak["mu"]
        sigma = peak["sigma"]
        label = peak.get("label", f"Peak {peak_i}")

        print(f"\n[{label}]  μ={mu:.3f} ns  σ={sigma:.3f} ns")

        matches = []
        for p_idx, pt in zip(pivot_indices, pivot_times):
            lo = np.searchsorted(target_times, pt + mu - n_sigma * sigma, "left")
            hi = np.searchsorted(target_times, pt + mu + n_sigma * sigma, "right")
            if lo >= hi:
                continue
            best  = lo + np.argmin(np.abs(target_times[lo:hi] - (pt + mu)))
            t_idx = target_indices[best]
            dt    = times[t_idx] - times[p_idx]
            matches.append((p_idx, t_idx, dt))
            if len(matches) >= n_total:
                break

        print(f"Matches found: {len(matches):,}")
        if not matches:
            continue

        # --------------------------------------------------------------
        # One figure per matched pair
        # --------------------------------------------------------------
        #Reorder Samples????
        reorder_samp_arr = False
        reorder_circular_buffer= True

        def _plot_wf(ax, wf, x_offset, is_pivot):
            nl, nc = _count_artists(ax)
            plot_waveform(
                ax=ax,
                hid=wf["hid"],
                channel=wf["channel"],
                baseline=wf["baseline"],
                samp_arr=wf["samp"],
                trig_arr=wf["trig"],
                period=period,
                color="C0" if is_pivot else "C1",
                interp_kwargs=interp_kwargs,
                label_mode="channel",
                reorder_circular_buffer=reorder_circular_buffer,
                reorder_samp_arr = reorder_samp_arr,
                plot_sample_types=True,
                plot_buffer_start=True,
                explicit_labels=is_pivot,   # marker type labels only from pivot
                time_scale=1.0,
            )
            _shift_new_artists(ax, nl, nc, x_offset)
            return x_offset + wf["cfd50_offset"]

        for pair_i, (p_idx, t_idx, dt) in enumerate(matches):


            wf_p = get_raw(p_idx)
            wf_t = get_raw(t_idx)

            # Time of target buffer start relative to pivot buffer start (ns)
            buf_offset = times_raw[t_idx] - times_raw[p_idx]

            cfd_p = wf_p["cfd50_offset"]   # ns into pivot ordered buffer
            cfd_t = wf_t["cfd50_offset"]   # ns into target ordered buffer

            buf_len = wf_p["buf_len_ns"]   # same for both channels

# ---------------------------------------------------------------
# Replace everything from  fig, (ax_buf, ax_cfd) = ...
# down to  plt.show()
# inside the pair loop with the block below.
# ---------------------------------------------------------------

            fig, (ax_buf, ax_cfd) = plt.subplots(
                1, 2,
                figsize=(16, 6),
                sharey=False,
            )

            # Figure-level title carries all the metadata
            fig.suptitle(
                f"Ch{pivot_channel} → Ch{target_channel}  |  "
                f"{label}  |  pair {pair_i + 1}/{len(matches)}  |  "
                f"Δt = {dt:.3f} ns  |  "
                f"reorder_samp_arr = {reorder_samp_arr} | "
                f"reorder_circular_buffer = {reorder_circular_buffer} | "
                f"HIT {wf_p['hid']} & {wf_t['hid']}",
                fontsize=11,
                y=1.01,
            )

            # ===========================================================
            # LEFT — buffer-start aligned
            # ===========================================================

            cfd_x_p = _plot_wf(ax_buf, wf_p, x_offset=0,         is_pivot=True)
            cfd_x_t = _plot_wf(ax_buf, wf_t, x_offset=buf_offset, is_pivot=False)

            ax_buf.axvline(cfd_x_p, color="C0", ls=":", lw=1.3,
                           label=f"Pivot  {cfd_p:.2f} ns")
            ax_buf.axvline(cfd_x_t, color="C1", ls=":", lw=1.3,
                           label=f"Target {buf_offset + cfd_t:.2f} ns")

            for xv, ls in [(0, "--"), (buf_len, "--"),
                           (buf_offset, "-."), (buf_offset + buf_len, "-.")]:
                ax_buf.axvline(xv, color="grey", ls=ls, lw=0.5, alpha=0.35)

            x_lo = min(0, buf_offset) - 1
            x_hi = max(buf_len, buf_offset + buf_len) + 1
            ax_buf.set_xlim(x_lo, x_hi)

            ax_buf.set_xlabel("Time from pivot buffer start [ns]")
            ax_buf.set_ylabel("Voltage [V]")
            ax_buf.set_title("Buffer-start aligned", fontsize=10, pad=4)

            # Two-part legend: channel lines (upper right) +
            #                  CFD lines + marker types (upper left, smaller)
            ch_handles, ch_labels = [], []
            mk_handles, mk_labels = [], []

            for h, l in zip(*ax_buf.get_legend_handles_labels()):
                if l.startswith("Channel") or l.startswith("Pivot") or l.startswith("Target"):
                    ch_handles.append(h); ch_labels.append(l)
                elif l not in ("", None):
                    mk_handles.append(h); mk_labels.append(l)

            leg1 = ax_buf.legend(ch_handles, ch_labels,
                                 loc="upper right", fontsize=8, framealpha=0.7)
            ax_buf.add_artist(leg1)
            if mk_handles:
                ax_buf.legend(mk_handles, mk_labels,
                              loc="upper left", fontsize=7,
                              framealpha=0.6, markerscale=0.85)

            hep.cms.label(cms_label, data=is_data, rlabel=rlabel, ax=ax_buf,
                          fontsize=11)

            # ===========================================================
            # RIGHT — CFD50 aligned
            # ===========================================================

            _plot_wf(ax_cfd, wf_p, x_offset=-cfd_p,             is_pivot=True)
            _plot_wf(ax_cfd, wf_t, x_offset=buf_offset - cfd_p, is_pivot=False)

            ax_cfd.axvline(0,  color="C0", ls=":", lw=1.3, label="Pivot = 0")
            ax_cfd.axvline(dt, color="C1", ls=":", lw=1.3,
                           label=f"Target = {dt:.3f} ns")

            half = max(cfd_p, buf_len - cfd_p, abs(dt) + cfd_t) + 3
            ax_cfd.set_xlim(-half, half)

            ax_cfd.set_xlabel("Time from pivot CFD50 crossing [ns]")
            ax_cfd.set_ylabel("Voltage [V]")
            ax_cfd.set_title("CFD50 aligned", fontsize=10, pad=4)

            ch_handles, ch_labels = [], []
            mk_handles, mk_labels = [], []

            for h, l in zip(*ax_cfd.get_legend_handles_labels()):
                if l.startswith("Channel") or l.startswith("Pivot") or l.startswith("Target"):
                    ch_handles.append(h); ch_labels.append(l)
                elif l not in ("", None):
                    mk_handles.append(h); mk_labels.append(l)

            leg1 = ax_cfd.legend(ch_handles, ch_labels,
                                 loc="upper right", fontsize=8, framealpha=0.7)
            ax_cfd.add_artist(leg1)
            if mk_handles:
                ax_cfd.legend(mk_handles, mk_labels,
                              loc="upper left", fontsize=7,
                              framealpha=0.6, markerscale=0.85)

            plt.tight_layout()
            plt.show()




def build_average_waveform(
    parquet_path: str,
    channel: int,
    hit_mask: Optional[np.ndarray] = None,
    cfd50_path: Optional[str] = None,     # None -> compute_cfd50 on the fly per hit
    period: float = 1 / 6.4,              # ns/sample (6.4 GSa/s SAMPIC rate)
    batch_size: int = 100_000,
    time_col: str = "OrderedCell0Time",
    n_hits: Optional[int] = None,         # cap for speed; None = use all
    window_ns: float = 5.0,              # half-width of output window around CFD50
    interpolation_method: str = "sinc",
    interpolation_factor: int = 4,
    interpolation_parameter: int = 8,
    reorder_circular_buffer: bool = True,
    reorder_samp_arr: bool = False,       # SAMPIC samples are already time-ordered
    seed: int = 0,
) -> dict:
    """
    Average CFD50-aligned, baseline-subtracted waveforms on one channel to
    get a representative pulse shape, using the same reorder/interpolation
    settings as plot_coincidence_waveforms_diagnostic / plot_waveform.

    Returns
    -------
    dict with keys:
        't'         : ndarray, common time grid [ns], t=0 at CFD50 crossing
        'mean'      : ndarray, mean voltage [V] (baseline-subtracted)
        'std'       : ndarray, per-sample std across hits [V]
        'n'         : int, number of waveforms actually averaged
        'waveforms' : ndarray [n, len(t)], individual aligned+interpolated waveforms
        'hit_ids'   : ndarray [n], HITNumber for each row of 'waveforms'
    """
    cols = ["HITNumber", "Channel", time_col, "Baseline", "DataSample", "TriggerPosition"]

    pqf = pq.ParquetFile(parquet_path)
    collected = []
    mask_set = set(hit_mask.tolist()) if hit_mask is not None else None

    for batch in pqf.iter_batches(batch_size=batch_size, columns=cols):
        ch = batch["Channel"].to_numpy()
        keep = ch == channel
        if mask_set is not None:
            hit_ids_b = batch["HITNumber"].to_numpy()
            keep &= np.fromiter(
                (hid in mask_set for hid in hit_ids_b),
                dtype=bool, count=len(hit_ids_b),
            )
        if not np.any(keep):
            continue
        collected.append(batch.take(pa.array(np.where(keep)[0])))

    if not collected:
        raise RuntimeError(f"No hits survived masking on channel {channel}.")

    table = pa.Table.from_batches(collected)

    hit_ids      = table["HITNumber"].combine_chunks().to_numpy().astype(np.int64)
    baselines    = table["Baseline"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    samples_col  = table["DataSample"].combine_chunks()
    triggers_col = table["TriggerPosition"].combine_chunks()

    if n_hits is not None and len(hit_ids) > n_hits:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(hit_ids), size=n_hits, replace=False))
        hit_ids      = hit_ids[sel]
        baselines    = baselines[sel]
        samples_col  = samples_col.take(pa.array(sel))
        triggers_col = triggers_col.take(pa.array(sel))

    cfd50_lut = load_cfd50_lookup(cfd50_path) if cfd50_path is not None else None

    interp_kwargs = dict(
        interpolation_method=interpolation_method,
        interpolation_factor=interpolation_factor,
        interpolation_parameter=interpolation_parameter,
    )

    

    common_t = np.arange(-window_ns, window_ns, period / interpolation_factor)

    aligned = []
    used_ids = []
    n_cfd_fail = 0
    n_oob = 0

    for i in tqdm(range(len(hit_ids)), desc=f"Averaging ch{channel}"):
        samp = np.asarray(samples_col[i].as_py(),  dtype=np.float64)
        trig = np.asarray(triggers_col[i].as_py(), dtype=np.int32)
        baseline = baselines[i]

        if cfd50_lut is not None:
            offset = cfd50_lut.get(int(hit_ids[i]))
        else:
            offset = compute_cfd50(samp, trig, baseline, period)

        if offset is None:
            n_cfd_fail += 1
            continue

        if reorder_circular_buffer:
            _, samp_shifted, _ = reorder_circular_samples_with_trigger(
                trig, samp, reorder_samp_arr,
            )
        else:
            samp_shifted = samp

        t_orig = np.arange(samp_shifted.size, dtype=float) * period

        t_intp, y_intp = apply_interpolation_method(
            x_orig=t_orig,
            y_orig=samp_shifted,
            period=period,
            offset=baseline,
            **interp_kwargs,
        )

        t_aligned = t_intp - offset
        y_aligned = y_intp - baseline  # center pulse around 0 for averaging

        # if t_aligned[0] > common_t[0] or t_aligned[-1] < common_t[-1]:
        #     n_oob += 1
        #     continue


        # 1. Force out-of-bounds values to become NaNs
        interp_wf = np.interp(
            common_t, 
            t_aligned, 
            y_aligned, 
            left=np.nan,  # Pad left with NaN instead of flatlining
            right=np.nan  # Pad right with NaN instead of flatlining
        )


        aligned.append(interp_wf)
        used_ids.append(int(hit_ids[i]))

    print(f"CFD50 failures: {n_cfd_fail:,} | out-of-window: {n_oob:,} | used: {len(aligned):,}")

    if not aligned:
        raise RuntimeError("No valid CFD50-aligned waveforms found.")

    aligned = np.array(aligned)

    valid_fraction = np.mean(~np.isnan(aligned), axis=0)

    mask = valid_fraction > 0.95   # require 95% of waveforms present

    common_t = common_t[mask]
    aligned = aligned[:, mask]

    return dict(
        t=common_t,
        # 2. Use nanmean and nanstd to ignore the NaNs during math
        mean=np.nanmean(aligned, axis=0),
        std=np.nanstd(aligned, axis=0),
        n=len(aligned),
        waveforms=aligned,
        hit_ids=np.array(used_ids, dtype=np.int64),
    )



def plot_average_waveform(
    result: dict,
    channel: int,
    ax: Optional[Axes] = None,
    show_band: bool = True,
    show_individual: bool = False,
    n_individual: int = 30,
    cms_label: str = "PPS2 Timing Preliminary",
    rlabel: str = "(H8 Test Beam, May 2025)",
    is_data: bool = True,
) -> Axes:
    """Quick-look plot for the output of build_average_waveform()."""

    created_fig = False
    if ax is None:
        plt.style.use(hep.style.CMS)
        fig, ax = plt.subplots(figsize=(8, 5))
        created_fig = True

    t, mean, std, n = result["t"], result["mean"], result["std"], result["n"]

    # --------------------------------------------------------------
    # Individual waveforms
    # --------------------------------------------------------------
    if show_individual:
        idx = np.linspace(
            0, len(result["waveforms"]) - 1,
            min(n_individual, len(result["waveforms"]))
        ).astype(int)

        for k in idx:
            ax.plot(t, result["waveforms"][k],
                    color="grey", alpha=0.15, lw=0.6)

    # --------------------------------------------------------------
    # ±1σ band
    # --------------------------------------------------------------
    if show_band:
        ax.fill_between(
            t, mean - std, mean + std,
            color="C0", alpha=0.25,
            label=f"±1σ (n={n})"
        )

    # --------------------------------------------------------------
    # Mean waveform
    # --------------------------------------------------------------
    ax.plot(t, mean, color="C0", lw=1.8,
            label=f"Mean waveform (Ch{channel})")

    ax.axvline(0, color="grey", ls=":", lw=1.0)
    
    ax.set_title(f"Average ch{channel} waveforms", fontsize=12)
    ax.set_xlabel("Time from CFD50 crossing [ns]", fontsize=12)
    ax.set_ylabel("Voltage [V] (baseline-subtracted)", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)

    ax.legend(loc="best", fontsize=9)

    hep.cms.label(cms_label, data=is_data, rlabel=rlabel,
                  ax=ax, fontsize=11)

    # --------------------------------------------------------------
    # Finalization only if we created the figure
    # --------------------------------------------------------------
    if created_fig:
        plt.tight_layout()
        plt.show()

    return ax
