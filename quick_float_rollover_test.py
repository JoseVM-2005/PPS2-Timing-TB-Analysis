import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import datetime

cols = ["HITNumber", "OrderedCell0Time", "UnixTime"]

run_to_process = "Run018_S"
print(run_to_process, "\n")
parquet_file = Path(rf"C:\Users\josev\OneDrive\coisas externas\LIP\PPS2-Timing-TB-Analysis\{run_to_process}\{run_to_process}.parquet")
print(parquet_file)

# 1. Read only the columns you need directly into an Arrow Table
table = pq.read_table(parquet_file, columns=cols)

# 2. Extract columns directly to NumPy arrays (zero-copy when possible)
hit_ids  = table["HITNumber"].to_numpy()
oc0times = table["OrderedCell0Time"].to_numpy()
uxtimes  = table["UnixTime"].to_numpy()

time_str = str(datetime.timedelta(seconds= oc0times.max() / 1e9))
uxtime_str = str(datetime.timedelta(seconds= (uxtimes[-1] - uxtimes[0])))
print(f"Run length (OrderedCell0time): {oc0times[-1]}ns | {time_str}")
print(f"Run length (UnixTime): {uxtimes[-1] - uxtimes[0]}s | {uxtime_str}")

# 3. Plotting
fig_oc0, ax_oc0 = plt.subplots(figsize=(8, 4))
fig_unix, ax_unix = plt.subplots(figsize=(8, 4))

ax_oc0.plot(hit_ids, oc0times, color="red")
ax_oc0.set_title(f"OrderedCell0Time for every hit | {run_to_process}", fontsize = 12)
ax_oc0.grid(True, linestyle="--", alpha=0.3)
ax_oc0.set_ylabel("OrderedCell0Time",fontsize = 12)
ax_oc0.set_xlabel("Hit Number",fontsize = 12)

ax_unix.plot(hit_ids, uxtimes, color="blue")
ax_unix.set_title(f"UnixTime for every hit | {run_to_process}",fontsize = 12)
ax_unix.grid(True, linestyle="--", alpha=0.3)
ax_unix.set_ylabel("UnixTime",fontsize = 12)
ax_unix.set_xlabel("Hit Number",fontsize = 12)

# Add hourly marker lines
max_oc0 = oc0times.max()
hour_in_ns = 3600e9  # assuming OrderedCell0Time is in nanoseconds

for h in range(1, int(max_oc0 // hour_in_ns) + 1):
    ax_oc0.axhline(h * hour_in_ns, color='black', linestyle='--', linewidth=1.5, label=f"Hour {h}")
plt.tight_layout()
plt.show()