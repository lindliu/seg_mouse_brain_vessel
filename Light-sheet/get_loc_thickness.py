import numpy as np
import glob
import os 
import re 
from PIL import Image
import tifffile as tiff

num_re = re.compile(r'(\d+)(?!.*\d)')

# filter_98_paths = glob.glob('./data/241030_659__13-02-02/3_filter_mask_98/*')
# filter_98_paths = sorted(filter_98_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

# current_loc_paths = glob.glob('./data/241030_659__13-02-02/5_thickness/*')
# current_loc_paths = sorted(current_loc_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

# save_root_path = './data/241030_659__13-02-02/temp'
# save_area_root_path = './data/241030_659__13-02-02/temp_area'

# filter_98_paths = filter_98_paths[370:570]
# current_loc_paths = current_loc_paths[370:570]

# value = 19
# for loc_path, filter_path in zip(current_loc_paths, filter_98_paths):
#     name = os.path.split(loc_path)[1]
#     name = name.replace('loc', 'mask')
#     print(name)
#     loc_arr = np.array(Image.open(loc_path))

#     loc_arr_mask = loc_arr>value
#     filter_arr = np.array(Image.open(filter_path)) * loc_arr_mask

#     curr_arr = ~loc_arr_mask*(loc_arr>0) + filter_arr>0

#     curr_arr = curr_arr.astype(np.uint8) * 255
#     save_path = os.path.join(save_root_path, name)
#     tiff.imwrite(save_path, curr_arr)
    

#     area = loc_arr>value
#     area = area.astype(np.uint8) * 255
#     save_area_path = os.path.join(save_area_root_path, name)
#     tiff.imwrite(save_area_path, area)




import qim3d
from scipy.ndimage import zoom

import plotly.graph_objects as go
def plot_3d_save(mask, spacing_zyx=[1,1,1], value_map=None, save_html=None, seed=0):
    pts = np.argwhere(mask)
    print("points:", pts.shape[0])

    max_pts = 200000
    if pts.shape[0] > max_pts:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], max_pts, replace=False)
        pts = pts[idx]
    else:
        idx = None

    if value_map is not None:
        value_map = np.asarray(value_map)
        val = value_map[pts[:, 0], pts[:, 1], pts[:, 2]].astype(np.float32)
        # val = value_map[mask]
    else:
        val = np.ones(pts.shape[0], dtype=np.float32)

    pz, py, px = spacing_zyx
    z = pts[:,0]*pz
    y = pts[:,1]*py
    x = pts[:,2]*px

    fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode="markers",
                                marker=dict(size=1, opacity=0.5, color=val,
                                            colorscale="Turbo")))
    fig.update_layout(scene=dict(aspectmode="data"))


        # ---- saving ----
    if save_html is not None:
        fig.write_html(save_html)
        print("Saved HTML:", save_html)

    # fig.show()

# --- Remove empty boundary slices (all False) ---
def trim_empty_slices(vol):
    z_min, z_max = 0, vol.shape[0] - 1
    y_min, y_max = 0, vol.shape[1] - 1
    x_min, x_max = 0, vol.shape[2] - 1

    # Trim Z
    while z_min <= z_max and not np.any(vol[z_min]):
        z_min += 1
    while z_max >= z_min and not np.any(vol[z_max]):
        z_max -= 1

    # Trim Y
    while y_min <= y_max and not np.any(vol[:, y_min, :]):
        y_min += 1
    while y_max >= y_min and not np.any(vol[:, y_max, :]):
        y_max -= 1

    # Trim X
    while x_min <= x_max and not np.any(vol[:, :, x_min]):
        x_min += 1
    while x_max >= x_min and not np.any(vol[:, :, x_max]):
        x_max -= 1

    return z_min, z_max+1, y_min, y_max+1, x_min, x_max+1


def find_boundary(paths):
    # --- Load images into a 3D boolean volume ---
    volume = []
    for path in paths:
        array = np.array(Image.open(path)) > 0   # convert to boolean
        volume.append(array)
    volume = np.array(volume)   # shape: (Z, Y, X)

    cleaned = trim_empty_slices(volume)
    return cleaned


def get_loc_thichness(paths, step=1, spacing_zyx=[1,1,1], target_spacing=None):
    dz, dy, dx = spacing_zyx
    dz_eff = dz * step
    dy_eff = dy * step
    dx_eff = dx * step
    spacing_zyx = dz_eff, dy_eff,dx_eff

    if target_spacing is None:
        target_spacing = dz_eff # min(spacing_zyx)
    factors = (dy_eff / target_spacing, dx_eff / target_spacing)

    paths = paths[::step]

    volume = []
    for path in paths:
        array = np.array(Image.open(path)) > 0   # convert to boolean
        array = array[::step,::step]
        volume.append(array)
    volume = np.array(volume)   # shape: (Z, Y, X)
    z_min, z_max, y_min, y_max, x_min, x_max = trim_empty_slices(volume)
    print(z_min, z_max, y_min, y_max, x_min, x_max)

    volume = []
    for path in paths[z_min:z_max]:
        array = np.array(Image.open(path))>0
        array = array[::step,::step]
        array = array[y_min:y_max, x_min:x_max]
        array_iso = zoom(array, zoom=factors, order=0)
        
        volume.append(array_iso)
    volume = np.array(volume)
    print('original: ', volume.shape)

    loc_thichness = qim3d.processing.local_thickness(volume, visualize=False, axis=0)
    loc_thichness_phy = loc_thichness * target_spacing
    print(np.unique(loc_thichness_phy))
    return loc_thichness_phy

root_paths = glob.glob(os.path.join('./data/*'))
for root_path in root_paths:
    print(root_path)

    final_mask_paths = glob.glob(os.path.join(root_path, '4_mask', f'*.tif'))
    final_mask_paths = sorted(final_mask_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))
    # print(final_mask_paths)

    ############ loc thichness of final combined mask #############
    spacing_zyx = [5,5.91,5.91] #5,5.91,5.91
    final_loc = get_loc_thichness(final_mask_paths, step=1, spacing_zyx=spacing_zyx)
    save_html = os.path.join(root_path, 'loc_final.html')
    step = 4
    plot_3d_save(final_loc[::step,::step,::step]>0, value_map=final_loc[::step,::step,::step], save_html=save_html)

    for i in range(final_loc.shape[0]):
        save_path = os.path.join(root_path, '5_thickness', f'loc_{i:04d}.tif')
        tiff.imwrite(save_path, final_loc[i])





# paths = glob.glob('./data/*/5_thickness')
# num_re = re.compile(r'(\d+)(?!.*\d)')

# want_path = []
# for root in paths:
#     end_paths = glob.glob(os.path.join(root, '*.tif'))
#     end_paths = sorted(end_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

#     # print(end_paths[-1])
#     want_path.append(end_paths[-1])

# for p in want_path:
#     arr = np.array(Image.open(p))
#     # print(arr.shape)

#     arr[-1,-1] = 110
#     img = Image.fromarray(arr)
#     img.save(p)