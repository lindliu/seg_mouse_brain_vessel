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
    

def get_loc_thichness(paths, step=1):
    volumn = []
    for path in paths[::step]:
        array = np.array(Image.open(path))>0
        array = array[::step,::step]
        volumn.append(array)
    volumn = np.array(volumn)
    loc_thichness = qim3d.processing.local_thickness(volumn, visualize=False, axis=0)
    return loc_thichness


num_re = re.compile(r'(\d+)(?!.*\d)')

# root_path = glob.glob(os.path.join(os.getcwd(),'data/241030_652__11-18-58'))[0]
root_paths = glob.glob(os.path.join(os.getcwd(),'data/*'))
print(root_paths)

for root_path in root_paths[:]:
    print(root_path)
    tif_paths = glob.glob(os.path.join(root_path, '2_8bit/*'))
    tif_paths = sorted(tif_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))


    save_fil_dir = os.path.join(root_path, '3_filter')
    os.makedirs(save_fil_dir, exist_ok=True)
    idx = list(np.arange(len(tif_paths)))

    ###################### process image by filter #########################

    for i, path in zip(idx, tif_paths):
        print(i)
        arr_ = np.array(Image.open(path))

        arr = filters.gaussian(arr_, sigma=1, preserve_range=True)
        #arr = frangi(arr, sigmas=np.arange(.1,20,1), black_ridges=False)
        arr = sato(arr, sigmas=np.arange(.01,6,.3), black_ridges=False)
        #arr = meijering(arr, sigmas=np.arange(.01,4,.2), black_ridges=False)
        #arr = hessian(ararrr_, sigmas=[1,2,3,3,5])

        # Normalize per-slice (or globally if preferred)
        arr_norm = arr / arr.max() if arr.max() != 0 else arr
        arr_filter = (arr_norm * 255).astype(np.uint8)

        img = Image.fromarray(arr_filter)
        save_path = os.path.join(save_fil_dir, f'sato_{i:04d}.tif')
        img.save(save_path)
    #########################################################################


    # # org_dir  = '2_8bit'
    # # org_paths = glob.glob(os.path.join(root_path, f'{org_dir}/*.tif'))
    # # org_paths = sorted(org_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

    # # filter_dir = '3_filter'
    # # filter_paths = glob.glob(os.path.join(root_path, f'{filter_dir}/*.tif'))
    # # filter_paths = sorted(filter_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))
    
    # # save_weig_dir = os.path.join(root_path, '3_weighted_tif')
    # # os.makedirs(save_weig_dir, exist_ok=True)
    
    # # i=0
    # # for org_path, filter_path in zip(org_paths,filter_paths):
    # #     org_img = np.array(Image.open(org_path))
    # #     filter_img = np.array(Image.open(filter_path))
    # #     print(np.mean(org_img), np.mean(filter_img))

    # #     img = Image.fromarray(org_img*filter_img)
    # #     save_path = os.path.join(save_weig_dir, f'weighted_{i:04d}.tif')
    # #     img.save(save_path)
    # #     i += 1


    filter_paths = glob.glob(os.path.join(save_fil_dir, f'sato_*.tif'))
    filter_paths = sorted(filter_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

    # [99,99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9]
    thresholds_filter = get_thresholds(filter_paths, quatiles=[95,96,97,98,99,\
                                                               99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9], bits=8)

    thresholds_thr = get_thresholds(tif_paths, quatiles=np.linspace(99,100,10,endpoint=False), bits=8)
    
    # thr_filter = thresholds_filter[99.4]
    # thr_thr = thresholds_thr[99.2]

    # counts1, bins1 = np.histogram(volumn1, bins=100)
    # counts2, bins2 = np.histogram(volumn2, bins=100)

    # fig,axe = plt.subplots(1,3,figsize=[10,5])
    # axe[0].bar(bins1[:-1], counts1, width=np.diff(bins1), edgecolor='black')
    # axe[1].bar(bins2[:-1], counts2, width=np.diff(bins2), edgecolor='black')

    save_tif = False
    idx = list(np.arange(len(tif_paths)))
    for thr in [95,96,97,98,99,99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9]:
        thr_filter = thresholds_filter[thr]

        save_fil_mask_dir = os.path.join(root_path, f'3_filter_mask_{thr}')
        os.makedirs(save_fil_mask_dir, exist_ok=True)

        for i, path in zip(idx, tif_paths):
            print(i)

            ################### filter mask by threhold ######################
            arr_filter = np.array(Image.open(os.path.join(save_fil_dir, f'sato_{i:04d}.tif')))
            # thr_filter = 50
            mask_filter = arr_filter>thr_filter
            # #mask_filter = apply_hysteresis_threshold(arr_filter, low=15, high=100)
            # #mask_filter = filters.threshold_local(arr_filter, block_size=11,method='gaussian',offset=-.01)
            # mask_filter = morphology.binary_closing(mask_filter, morphology.disk(1))
            # mask_filter = morphology.remove_small_objects(mask_filter, min_size=15)
            # arr_filter[~mask_filter] = 0

            if save_tif:
                mask_filter = mask_filter.astype(np.uint8) * 255
                save_path = os.path.join(save_fil_mask_dir, f'fil_mask_{i:04d}.tif')
                tiff.imwrite(save_path, mask_filter)
            else:
                save_path = os.path.join(save_fil_mask_dir, f'fil_mask_{i:04d}.png')
                img = Image.fromarray((mask_filter.astype(np.uint8) * 255), mode="L")
                img.save(save_path)
            ###################################################################

        filter_mask_paths = glob.glob(os.path.join(save_fil_mask_dir, f'*.png'))
        filter_mask_paths = sorted(filter_mask_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

        step = 4
        ############## loc thichness of filter mask ##############
        filter_loc = get_loc_thichness(filter_mask_paths, step=step)
        save_html = os.path.join(root_path, f'loc_filter_{thr}.html')
        plot_3d_save(filter_loc>0, value_map=filter_loc, save_html=save_html)
        # plot_3d_show(mask_filter, mask_diamfilter_loceter_filter)


    for thr in np.linspace(99,100,10,endpoint=False):
        thr_thr = thresholds_thr[thr]

        save_thr_mask_dir = os.path.join(root_path, f'3_thr_mask_{thr}')
        os.makedirs(save_thr_mask_dir, exist_ok=True)

        for i, path in zip(idx, tif_paths):
            print(i)

            ##################### mask by threshold ####################
            arr_orig = np.array(Image.open(path))
            # thr_thr = 26
            mask_thr = arr_orig>thr_thr
            # #mask = apply_hysteresis_threshold(arr_uint8, low=80, high=200)
            # mask_thr = morphology.binary_closing(mask_thr, morphology.disk(1))
            # mask_thr = morphology.remove_small_objects(mask_thr, min_size=15)
            # arr_orig[~mask_thr] = 0
            
            if save_tif:
                mask_thr = mask_thr.astype(np.uint8) * 255
                save_path = os.path.join(save_thr_mask_dir, f'thr_{i:04d}.tif')
                tiff.imwrite(save_path, mask_thr)
            else:
                save_path = os.path.join(save_thr_mask_dir, f'thr_{i:04d}.png')
                img = Image.fromarray((mask_thr.astype(np.uint8) * 255), mode="L")
                img.save(save_path)
            ############################################################


        thr_mask_paths = glob.glob(os.path.join(save_thr_mask_dir, f'*.png'))
        thr_mask_paths = sorted(thr_mask_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

        step = 4
        ############## loc thichness of threshold mask ##############
        thr_loc = get_loc_thichness(thr_mask_paths, step=step)
        save_html = os.path.join(root_path, f'loc_thr_{thr}.html')
        plot_3d_save(thr_loc>0, value_map=thr_loc, save_html=save_html)
        # plot_3d_show(mask_thr, thr_loc)
