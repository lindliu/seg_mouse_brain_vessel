import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

paths = glob.glob('./data/histgram/*.csv')
print(paths)

# # --- Load and merge ---
# dfs = []
# for path in paths:
#     df = pd.read_csv(path, index_col='thickness')
#     dfs.append(df)

# df_all = pd.concat(dfs, axis=1)        # merge by index
# df_all = df_all.sort_index()           # sort by thickness

# # --- Save merged table ---
# df_all.to_csv('./data/histgram/thickness_statistics.csv')
# print("Saved merged CSV")


df_all = pd.read_csv('./data/histgram/thickness_statictics.csv', index_col='thickness')

# Ensure everything is numeric
df_all.index = pd.to_numeric(df_all.index, errors='coerce')
df_all = df_all.apply(pd.to_numeric, errors='coerce')

# Drop rows where thickness (index) is NaN
df_all = df_all[~pd.isna(df_all.index)]

# Build explicit bins across the full thickness range
n_bins = 50
xmin, xmax = float(np.nanmin(df_all.index.values)), float(np.nanmax(df_all.index.values))
if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
    raise ValueError("Thickness range is invalid (min/max are NaN/inf or identical).")

bin_edges = np.linspace(xmin, xmax, n_bins + 1)

for col in df_all.columns:
    x = df_all.index.values
    w = df_all[col].values

    # Keep only finite rows
    mask = np.isfinite(x) & np.isfinite(w)
    x, w = x[mask], w[mask]

    # If nothing to plot, skip
    if x.size == 0 or np.nansum(w) == 0:
        print(f"Skipping {col}: empty or all-zero weights.")
        continue

    plt.figure()
    plt.hist(x, bins=bin_edges, weights=w, edgecolor='black')
    plt.xlabel("Thickness")
    plt.ylabel("Count")
    plt.title(col)
    plt.tight_layout()
    plt.savefig(f'./data/histgram/{col}.png', dpi=150)
    plt.close()

    print("Saved", col)
