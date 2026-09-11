from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from sampiclyser.sampic_tools import (
    open_hit_reader,
    reorder_circular_samples_with_trigger,
)

# Assuming these are available in your filter_functions
from filter_functions import (
    compute_cfd50,
    compute_rise_time,
    compute_cfd_times_and_slopes,
    compute_noise_rms,
    load_baseline_lookup,
)

CFD_LEVELS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

def _build_sampic_lookup(config) -> dict[int, tuple[str, int]]:
    """Build lookup: sampic_channel -> (lgad_name, lgad_board_ch_index)"""
    lookup = {}
    for lgad_name, ch_map in config.sampic_to_board.items():
        for sampic_ch, lgad_board_ch in ch_map.items():
            lookup[sampic_ch] = (lgad_name, lgad_board_ch)
    return lookup



def build_full_hit_database(
    raw_parquet_path: str,
    config,
    output_path: Optional[str] = None,
    baseline_path: Optional[str] = None,
    hit_mask: Optional[np.ndarray] = None,
    channel_filter: Optional[List[int]] = None,  # NEW PARAMETER
    cfd_levels: List[int] = CFD_LEVELS,
    period: float = 1 / 6.4,
    batch_size: int = 100_000,
    root_tree: str = "sampic_hits",
    low_frac: float = 0.1,
    high_frac: float = 0.9,
    smooth_window: int = 2,
    n_noise_samples: int = 20,
    min_amplitude: float = 0.01,
    edge_buffer_samples: int = 10,
    consecutive_points: int = 2,
) -> str:
    """
    Builds a complete, corrected hit database in a single pass from raw waveforms.

    The baseline is taken from the per-hit baseline database generated from
    the first n_noise_samples waveform samples. The Baseline column stored
    in the raw parquet file is deliberately not used.
    """
    
    if output_path is None:
        p = Path(raw_parquet_path)
        output_path = str(p.with_stem(p.stem + "_full_hit_db"))
        
    raw_parquet_path = Path(raw_parquet_path)
    output_path = Path(output_path)

    if baseline_path is None:
        raise ValueError(
            "baseline_path must be provided. Generate the baseline database "
            "with compute_and_save_baseline_database() first."
        )
    baseline_lookup = load_baseline_lookup(baseline_path)

    # -------------------------------------------------------------------------
    # Lookups & Masks
    # -------------------------------------------------------------------------
    mask_set = set(hit_mask.astype(np.int64).tolist()) if hit_mask is not None else None
    if mask_set:
        print(f"Using hit mask with {len(mask_set):,} valid hits. (Bad hits will be skipped instantly)")

    sampic_lookup = _build_sampic_lookup(config)

    # -------------------------------------------------------------------------
    # Setup Reader
    # -------------------------------------------------------------------------
    cols = [
        "HITNumber", "Channel", "RawPeak", 
        "OrderedCell0Time", "DataSample", "TriggerPosition"
    ]
    
    batches = open_hit_reader(
        raw_parquet_path,
        cols=cols,
        batch_size=batch_size,
        root_tree=root_tree,
    )

    # -------------------------------------------------------------------------
    # Output Storage
    # -------------------------------------------------------------------------
    out = {
        "HITNumber": [], "Channel": [], "LGAD": [], "SensorPad": [],
        "Baseline": [], "RawPeak": [], "Amplitude": [], "RiseTime": [], "SNR": [], "OrderedCell0Time": [],
    }
    
    for k in cfd_levels:
        out[f"CFD{k}Offset"] = []
        out[f"CFD{k}Slope"] = []
        

    total_seen = 0
    total_kept = 0
    skipped_mask = 0
    skipped_ch = 0
    rt_fail = 0

    # -------------------------------------------------------------------------
    # Single-Pass Main Loop
    # -------------------------------------------------------------------------
    for batch in tqdm(batches, desc="Building Full Database"):
        
        # =====================================================================
        # NEW LOGIC: Early Batch-Level Channel Filter
        # =====================================================================
        if channel_filter is not None:
            channels_batch = batch["Channel"].to_numpy()
            keep_ch = np.isin(channels_batch, channel_filter)
            
            # Update metric counters for dropped channels before slicing
            skipped_ch += len(channels_batch) - np.sum(keep_ch)
            total_seen += len(channels_batch)
            
            if not np.any(keep_ch):
                continue
                
            # Slice down the entire PyArrow batch chunk instantly
            batch = batch.take(pa.array(np.where(keep_ch)[0]))
        else:
            # If no channel filter, increment total_seen normally by the whole batch size
            total_seen += len(batch)

        # =====================================================================
        # Continue extraction on the filtered batch subset
        # =====================================================================
        hit_ids = batch["HITNumber"].to_numpy().astype(np.int64)
        channels = batch["Channel"].to_numpy().astype(np.int16)
        raw_peaks = batch["RawPeak"].to_numpy(zero_copy_only=False).astype(np.float64)
        ordered_times = batch["OrderedCell0Time"].to_numpy(zero_copy_only=False).astype(np.float64)
        
        samples_col = batch["DataSample"]
        triggers_col = batch["TriggerPosition"]

        for i in tqdm(range(len(hit_ids)), leave=False, desc="Hits", mininterval=0.5):
            # Note: total_seen increment removed from inside this loop to avoid double-counting 
            # with the batch-level step above.

            hit_id = int(hit_ids[i])

            # EARLY EXIT: Do not parse waveforms if hit isn't in the mask
            if mask_set is not None and hit_id not in mask_set:
                skipped_mask += 1
                continue

            ch = int(channels[i])
            if ch not in sampic_lookup:
                skipped_ch += 1
                continue
            
            # Extract Geometry
            lgad_name, lgad_board_ch = sampic_lookup[ch]
       

            baseline = baseline_lookup.get(hit_id)
            if baseline is None or not np.isfinite(baseline):
                continue
            baseline = float(baseline)
            raw_peak = float(raw_peaks[i])
            ordered_time = float(ordered_times[i])

            # Process Waveforms
            samp_arr = np.asarray(samples_col[i].as_py(), dtype=np.float64)
            trig_arr = np.asarray(triggers_col[i].as_py(), dtype=np.int32)

            # --- Signal Metrics ---
            noise_rms = compute_noise_rms(samp_arr, trig_arr, baseline, n_noise_samples)
            amplitude = raw_peak - baseline
            snr = (amplitude / noise_rms) if (np.isfinite(noise_rms) and noise_rms > 0) else np.nan
            amp_ratio = (amplitude / baseline) if baseline != 0 else np.nan

            rise_time = compute_rise_time(
                samp_arr=samp_arr, trig_arr=trig_arr, baseline=baseline, period=period,
                low_frac=low_frac, high_frac=high_frac, smooth_window=smooth_window,
            )
            if rise_time is None:
                rise_time = np.nan
                rt_fail += 1

            # --- CFD Calculations ---
            cfd_offsets, cfd_slopes = compute_cfd_times_and_slopes(
                samp_arr=samp_arr, trig_arr=trig_arr, baseline=baseline, period=period,
                cfd_levels=cfd_levels, min_amplitude=min_amplitude, 
                edge_buffer_samples=edge_buffer_samples, consecutive_points=consecutive_points,
            )

            # Append Core Data
            out["HITNumber"].append(hit_id)
            out["Channel"].append(ch)
            out["LGAD"].append(lgad_name)
            out["SensorPad"].append(lgad_board_ch)
            out["Baseline"].append(baseline)
            out["RawPeak"].append(raw_peak)
            out["Amplitude"].append(amp_ratio)
            out["RiseTime"].append(rise_time)
            out["SNR"].append(snr)

            # Append CFD Data
            for k in cfd_levels:
                offset = cfd_offsets[k]
                slope = cfd_slopes[k]
                out[f"CFD{k}Offset"].append(offset)
                out[f"CFD{k}Slope"].append(slope)
                
            # Store OrderedCell0Time once per hit as float64
            out["OrderedCell0Time"].append(ordered_time)
                    
            total_kept += 1
                

    # -------------------------------------------------------------------------
    # Build Table and Write Output
    # -------------------------------------------------------------------------
    table_dict = {
        "HITNumber": pa.array(out["HITNumber"], type=pa.int64()),
        "Channel": pa.array(out["Channel"], type=pa.int16()),
        "LGAD": pa.array(out["LGAD"], type=pa.string()).dictionary_encode(),
        "SensorPad": pa.array(out["SensorPad"], type=pa.int8()),
    }
    
    # Batch convert floats — float32 is fine for small-range quantities
    float_cols = ["Baseline", "RawPeak", "Amplitude", "RiseTime", "SNR"]
    for k in cfd_levels:
        float_cols.extend([f"CFD{k}Offset", f"CFD{k}Slope"])

    for col in float_cols:
        table_dict[col] = pa.array(np.asarray(out[col], dtype=np.float32), type=pa.float32())

    table_dict["OrderedCell0Time"] = pa.array(
        np.asarray(out["OrderedCell0Time"], dtype=np.float64), type=pa.float64()
    )

    table = pa.table(table_dict)
    pq.write_table(table, output_path, compression="zstd", row_group_size=batch_size)
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Full Corrected Database Built Successfully")
    print("="*60)
    print(f"Raw Input         : {raw_parquet_path}")
    print(f"Output Database   : {output_path}")
    print(f"Baseline Database : {baseline_path}")
    print()
    print(f"Total Hits Seen   : {total_seen:,}")
    print(f"Total Hits Kept   : {total_kept:,}")
    print(f"Rejected by Mask  : {skipped_mask:,}")
    print(f"Unknown/Filtered Ch: {skipped_ch:,}")
    print(f"RiseTime Failures : {rt_fail:,}")
    print("="*60 + "\n")

    return str(output_path)