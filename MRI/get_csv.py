import os
import numpy as np
import pandas as pd
import glob


which = 'frangi' #  'threshold' #

if which == 'frangi':
    npz_path = glob.glob('./data/Mouse brain MRI/*/Frangi_archived/*.npz')
elif which == 'threshold':
    npz_path = glob.glob('./data/Mouse brain MRI/*/Threshold_archived/*.npz')

col_lists, val_lists = [], []
name_data_list = []

for pp in npz_path:
    name_data = os.path.split(os.path.split(os.path.split(pp)[0])[0])[1]
    name_data_list.append(name_data)

    dd = np.load(pp)
    col_lists.append(dd['val_vessel'])
    val_lists.append(dd['num_vessel'])

rows = []
for name, cols, vals in zip(name_data_list, col_lists, val_lists):
    if len(cols) != len(vals):
        raise ValueError("One row has different lengths for cols and vals")

    pairs = sorted(zip(cols, vals), key=lambda x: x[0])
    row_dict = dict(pairs)

    # add dataset name as first column
    row_dict["dataset"] = name

    rows.append(row_dict)

df = pd.DataFrame(rows)

# sort numeric columns, but keep "dataset" first
numeric_cols = sorted([c for c in df.columns if c != "dataset"])
df = df[["dataset"] + numeric_cols]

df.to_csv(f"./data/Mouse brain MRI/statistics_{which}_vessel.csv", index=False)






col_lists, val_lists = [], []
name_data_list = []

for pp in npz_path:
    name_data = os.path.split(os.path.split(os.path.split(pp)[0])[0])[1]
    name_data_list.append(name_data)

    dd = np.load(pp)
    col_lists.append(dd['val_skel'])
    val_lists.append(dd['num_skel'])

rows = []
for name, cols, vals in zip(name_data_list, col_lists, val_lists):
    if len(cols) != len(vals):
        raise ValueError("One row has different lengths for cols and vals")

    pairs = sorted(zip(cols, vals), key=lambda x: x[0])
    row_dict = dict(pairs)

    # add dataset name as first column
    row_dict["dataset"] = name

    rows.append(row_dict)

df = pd.DataFrame(rows)

# sort numeric columns, but keep "dataset" first
numeric_cols = sorted([c for c in df.columns if c != "dataset"])
df = df[["dataset"] + numeric_cols]

df.to_csv(f"./data/Mouse brain MRI/statistics_{which}_vessel_skeleton.csv", index=False)
