


import numpy as np
import glob
import matplotlib.pyplot as plt
import skimage
import os
from PIL import Image
from skimage.filters import meijering, sato, frangi, hessian
import copy
import qim3d
import re
import tifffile as tiff

from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, measure, util
from skimage.filters import sato, frangi, apply_hysteresis_threshold
from skimage.segmentation import clear_border
from scipy.ndimage import zoom



import open3d as o3d
import matplotlib.cm as cm
def plot_3d_show(masks, value_map=None, cmap_name="turbo"):
    points = np.argwhere(masks > 0)  ## masks (Z,X,Y), points (Z,X,Y)
    
    # max_pts = 200000
    # if points.shape[0] > max_pts:
    #     points = points[np.random.choice(points.shape[0], max_pts, replace=False)]

    if value_map is not None:
        vals = value_map[masks > 0].astype(np.float64)
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        
        norm = (vals - vmin) / (vmax - vmin)
        cmap = cm.get_cmap(cmap_name)
        colors = cmap(norm)[:, :3].astype(np.float64)

        # colors = np.repeat(norm[:, None], 3, axis=1)
        # colors = np.clip(colors, 0.0, 1.0)
    else:        
        # 自定义颜色，每个点对应一个 RGB 值（范围 [0, 1]）
        colors = np.ones_like(points, dtype=float)*.5   

    # 构建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 可视化
    o3d.visualization.draw_geometries([pcd])


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


##### get thresholds #####
def get_thresholds(paths, quatiles=[99], bits=8):
    maxv = 2**bits
    if bits==8:
        counts = np.zeros(maxv, dtype=np.uint8)
        counts = np.zeros(maxv, dtype=np.uint8)
    total = 0

    for path in paths:
        imarray = np.array(Image.open(path))
        flat = imarray.ravel()
        counts = counts+np.bincount(flat, minlength=maxv)
        total += flat.size
    cdf = np.cumsum(counts)

    thresholds = {}
    for q in quatiles:
        target = int(np.ceil((q / 100.0) * total))
        idx = int(np.searchsorted(cdf, target, side="left"))
        thresholds[q] = idx  # 这个 idx 就是 q% 的像素强度阈值
    print(thresholds)

    return thresholds


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


num_re = re.compile(r'(\d+)(?!.*\d)')

# root_path = glob.glob(os.path.join(os.getcwd(),'data/241030_652__11-18-58'))[0]
root_paths = glob.glob(os.path.join(os.getcwd(),'data/*'))
print(root_paths)


for root_path in root_paths[:]:
    
    print(root_path)
    filter_mask_99_dir = '3_filter_mask_99.0'
    filter_mask_99_paths = glob.glob(os.path.join(root_path, f'{filter_mask_99_dir}/*.png'))
    filter_mask_99_paths = sorted(filter_mask_99_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

    filter_mask_dir = '3_filter_mask_99.5'
    filter_mask_paths = glob.glob(os.path.join(root_path, f'{filter_mask_dir}/*.png'))
    filter_mask_paths = sorted(filter_mask_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

    thr_mask_dir = '3_thr_mask_99.0'
    thr_mask_paths = glob.glob(os.path.join(root_path, f'{thr_mask_dir}/*.png'))
    thr_mask_paths = sorted(thr_mask_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

    thr_mask_996_dir  = '3_thr_mask_99.6'
    thr_mask_996_paths = glob.glob(os.path.join(root_path, f'{thr_mask_996_dir}/*'))
    thr_mask_996_paths = sorted(thr_mask_996_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))


    save_mask_dir = os.path.join(root_path, '4_mask')
    save_thickness_dir = os.path.join(root_path, '5_thickness')
    os.makedirs(save_mask_dir, exist_ok=True)
    os.makedirs(save_thickness_dir, exist_ok=True)

    save_tif = True  #  False # 
    i = 0
    for filter_mask_99_path, filter_mask_path, thr_mask_path, thr_mask_996_path in zip(filter_mask_99_paths, filter_mask_paths, thr_mask_paths, thr_mask_996_paths):
        mask_filter_99 = np.array(Image.open(filter_mask_99_path))>0
        mask_filter = np.array(Image.open(filter_mask_path))>0
        mask_thr = np.array(Image.open(thr_mask_path))>0
        mask_thr_996 = np.array(Image.open(thr_mask_996_path))>0

        ############ final vessel mask by mask_filter+mask_thr ##############
        mask1 = (mask_thr*mask_filter_99)>0  # remove low confidence area from filter mask
        mask2 = (mask1 + mask_filter)>0      # add high confidence area from filter mask
        mask = (mask2 + mask_thr_996)>0      # add high confidence area from threshold mask
        # mask = (mask_thr*mask_filter_97 + mask_filter + mask_thr_996)>0
        if i%100==0:
            print(mask_thr.sum(), mask_filter_99.sum(), mask1.sum(), mask2.sum(), mask.sum())


        if save_tif:
            mask = mask.astype(np.uint8) * 255
            save_path = os.path.join(save_mask_dir, f'mask_{i:04d}.tif')
            tiff.imwrite(save_path, mask)
        else:
            save_path = os.path.join(save_mask_dir, f'mask_{i:04d}.png')
            img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
            img.save(save_path)
        i += 1
        #####################################################################
    
    if save_tif:
        final_mask_paths = glob.glob(os.path.join(save_mask_dir, f'*.tif'))
    else:
        final_mask_paths = glob.glob(os.path.join(save_mask_dir, f'*.png'))
    final_mask_paths = sorted(final_mask_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

    # ############ loc thichness of final combined mask #############
    # spacing_zyx = [5,5.91,5.91] #5,5.91,5.91
    # final_loc = get_loc_thichness(final_mask_paths, step=1, spacing_zyx=spacing_zyx)
    # save_html = os.path.join(root_path, 'loc_final.html')
    # step = 4
    # plot_3d_save(final_loc[::step,::step,::step]>0, value_map=final_loc[::step,::step,::step], save_html=save_html)

    # for i in range(final_loc.shape[0]):
    #     save_path = os.path.join(save_thickness_dir, f'loc_{i:04d}.tif')
    #     tiff.imwrite(save_path, final_loc[i])
