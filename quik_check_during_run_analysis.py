from pathlib import Path

import matplotlib.pyplot as plt
from natsort import natsorted

from PPS_TB_202505_RunDB import ( #Don't forget run info here
    runs_to_configs,
    run_config_info,
    run_info as run_info_dict,
)
from delays_db import Run018  # CHANGE HERE for delays

from sampiclyser import SAMPIC_Run_Decoder
from sampiclyser.sampic_tools import (
    get_channel_hits,
    plot_channel_hits,
    plot_hit_rate,
)

from filter_functions import (
    compute_and_save_cfd50,
    plot_AmpRatio_histograms,
    plot_rise_time_histograms,
    plot_SNR_histograms,
    build_hit_mask,
    apply_hit_mask,
    load_hit_mask,
)

from parameters_database import build_full_hit_database

from delta_t_analysis import (
    build_events,
    plot_delta_t_histogram,
    plot_resolution_heatmap,
    plot_channel_pair_histograms,
)

#-----------------------------#
## CONFIG ##     Don't forget config info in PPS_TB_202505_RunDB or equivalent
#-----------------------------#
base_path = Path("./data")

run_path = base_path/"Run018_SAMPIC_5_17_2025_23h_13min_Binary" ##Change file name here
run_files = natsorted(list(run_path.glob("*.bin*")))
run_to_process = "Run018" # adapt accoringly

from pathlib import Path
import os

print("run path exists:", run_path.exists())


print(run_files)
print("\n")


tmp = SAMPIC_Run_Decoder(run_path)
tmp.decode_data(parquet_path=Path(f"./{run_to_process}.parquet"))

parquet_file = Path(f"./{run_to_process}.parquet")
print(run_info_dict)
print("\n")

run_info = run_info_dict[run_to_process]
config_key = runs_to_configs[run_to_process]
config_info = run_config_info[config_key]
print(config_info)
print("\n")
print(run_info)
print("\n")
#------------------------#
## General run details
# #----------------------# 

print("\n General run details \n")

hit_summary = get_channel_hits(parquet_file)
print(hit_summary)
print("\n")
title = run_info.name
fig = plot_channel_hits(hit_summary, 0, 39, label="PPS2 Timing Preliminary", rlabel="(H8 Test Beam, May 2025)", log_y=True, figsize = (15, 10), title=title)



plot_title = run_info.name + " - Hit Rate"

fig = plot_hit_rate(
    parquet_file,
    plot_hits=False,
    bin_size=1,
    #start_time=datetime.datetime(2025, 5, 15, 23, 27),
    #end_time=datetime.datetime(2025, 5, 15, 23, 57),
    #start_time=datetime.datetime(2025, 5, 17, 1, 10),
    #end_time=datetime.datetime(2025, 5, 17, 1, 30),
    label="PPS2 Timing Preliminary",
    rlabel="(H8 Test Beam, May 2025)",
    #log_y=True,
    figsize = (17, 10),
    title=plot_title,
    )



#-------------------------#
## Analysis
#-------------------------#
print("\n Analysis \n")

compute_and_save_cfd50(
    parquet_file,
    output_path = f"{run_to_process}_cfd50.parquet",
    min_amplitude = 0.00,      # V — reject noise spikes
    edge_buffer_samples = 10,     # samples — peak must not be too close to buffer edge
    
)

#AmpRatio hist cuts 


print(run_to_process, "AmpRatio \n")
# amplitude_histos = plot_AmpRatio_histograms(
#     parquet_file,
#     n_bins         = 500,
#     x_range        = (0, 1.5),   # set based on your expected signal range
#     channel_filter = [4, 8],
#     logy           = False,
# )

results, thr_AmpRatio = plot_AmpRatio_histograms(
    parquet_file,
    Run = run_to_process,
    channel_filter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 23, 16, 17, 18, 19],
    n_bins         = 500,
    x_range        = (0.00, 0.7),
    fit_window     = None,      # auto: fits rising edge + modest right tail
    mpv_fraction   = 0.10,       # cut at 5% of MPV — adjust based on plots
    logy           = False,
)

#rise time hist cuts

print(run_to_process, "Rise-time \n")
rise_time_histos, thr_rt = plot_rise_time_histograms(
    parquet_file,
    AmpRatio_cuts   = thr_AmpRatio,    # {4: 0.10, 8: 0.05} etc.
    apply_Amp_filter= False,
    channel_filter  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 23, 16, 17, 18, 19],
    n_bins    = 500,
    x_range   = (0.0, 2),   # expected rise time range in ns; set after first look
    low_frac  = 0.1,
    high_frac = 0.9,
    sigma_cut = (2.5,4),
    logy      = False,
)

#SNR hist cuts


print(run_to_process, "SNR \n")
SNR_hists, thr_snr = plot_SNR_histograms(
    parquet_file,
    channel_filter= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 23, 16, 17, 18, 19],
    n_bins      = 500,
    x_range     = (0.0, 80.0),
    sigma_cut   = 2.5,
)

#hit mask
print("||")
print("|| Build Hit mask")
print("|| \n")


good_hits = build_hit_mask(
    parquet_file,
    AmpRatio_cuts  = thr_AmpRatio,               # from plot_AmpRatio_histograms
    rise_time_cuts = thr_rt,          # from plot_rise_time_histograms
    SNR_cuts       = thr_snr,
    channel_filter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 23, 16, 17, 18, 19],
    save_path      = f"good_hits_{run_to_process}.npy",
)


# Use the mask to load only good waveforms for any downstream step
clean_parquet = apply_hit_mask(parquet_file, good_hits)

#build full database
print("||")
print("|| Build full database")
print("||", "\n")

print(f"Loading good_hits from: 'good_hits_{run_to_process}.npy'")
good_hits = load_hit_mask(f"good_hits_{run_to_process}.npy")

config_key = runs_to_configs[run_to_process]
config_info = run_config_info[config_key]
print("\n", config_info, "\n")

print(">>> STEP 1: Building unified hit database...", "\n")
# Just run the unified function directly:
final_db_path = build_full_hit_database(
    raw_parquet_path = f"{run_to_process}.parquet",
    config           = config_info,
    hit_mask         = good_hits,
    channel_filter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 23, 16, 17, 18, 19],

)

#resolution results
print("||")
print("|| Events and Resoltion")
print("||", "\n")



print("\n>>> STEP 2: Grouping hits into coincidence events...", "\n")
events_file = build_events(
    corrdb_path           = f"{run_to_process}_full_hit_db.parquet",
    run_name              = run_to_process,
    config = config_info,
    coincidence_window_ns = 50.0,
    event_building_cfd    = 50,
)



print("\n>>> STEP 3: Generating diagnostic plots...", "\n")

# Plot A: 1D Histogram for the 50% - 50% CFD pair, EXCLUDING clusters
fig_hist = plot_delta_t_histogram(
    events_path    = events_file,
    k1             = 80, #yaxis on heatmap
    k2             = 80, #xaxis on heatmap
    cluster_filter = "exclude",
    n_bins= 75
)
fig_hist.savefig("dt_histogram_CFD50.png", dpi=300)
print("Saved -> dt_histogram_CFD50.png", "\n")

# Plot B: 2D Heatmaps of time resolution across ALL CFD pairs
fig_fit, fig_std, res_min_fit, err_min_fit, res_min_std, err_min_std = plot_resolution_heatmap(
    events_path  = events_file,
    lgad1_label  = "PPS_LGAD_03",
    lgad2_label  = "PPS_TILGAD_AIDA_01",
    n_contours   = 10,
    n_bins       = 200,
    cfd_levels   = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
)
fig_fit.savefig("heatmap_gaussian_fit.png", dpi=300)
fig_std.savefig("heatmap_numpy_std.png",    dpi=300)
print("Saved -> heatmap_gaussian_fit.png, heatmap_numpy_std.png", "\n")

print(f"Resolution from fit = {res_min_fit} ± {err_min_fit}ps")
print(f"Resolution from unbined std = {res_min_std} ± {err_min_std}ps")

# Plot C: Per adjacent channel-pair Δt histograms + summary table
print("\n>>> STEP 4: Per-channel-pair analysis...", "\n")
pair_figures = plot_channel_pair_histograms(
    events_path = events_file,
    config = config_info,
    corrdb_path = f"{run_to_process}_full_hit_db.parquet",
    cfd_levels  = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
    cfd_plot    = 50,
    n_bins      = 50,
    pooled_res_fit_ps     = res_min_fit,
    pooled_res_fit_err_ps = err_min_fit,
    pooled_res_std_ps     = res_min_std,
    pooled_res_std_err_ps = err_min_std, 
    output_dir            = "channel_pair_plots",   # saves dt_pair_chA_chB.png for each pair
)

plt.show()
print("\n>>>Event Pipeline complete.")