#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 12:17:55 2026

@author: dliu
"""

from glob import glob
import pickle
import re
import os 
import numpy as np
from cellpose import models, core, io, plot
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy import ndimage as ndi

io.logger_setup() # run this to get printing of progress

#Check if colab notebook instance has GPU access
if core.use_gpu()==False:
  raise ImportError("No GPU access, change your runtime")




### load model
new_model_path = './models/model_2'
model = models.CellposeModel(gpu=True, pretrained_model=new_model_path)





### load dataset
num_re = re.compile(r'(\d+)(?!.*\d)')


root_path_list = ['./data/Mouse brain MRI/220210_SA_MR_mouse_338-control-male__E7_P1',
                './data/Mouse brain MRI/220210_SA_MR_mouse_321-knockout-mal__E6_P1',
                './data/Mouse brain MRI/220210_SA_MR_mouse_329-control-male__E4_P1',
                './data/Mouse brain MRI/220210_SA_MR_mouse_378-knockout-mal__E4_P1', ## green
                './data/Mouse brain MRI/220210_SA_MR_mouse_330-knockout-mal__E11_P1',
                './data/Mouse brain MRI/220210_SA_MR_mouse_531-control-fema__E7_P1',
                './data/Mouse brain MRI/220210_SA_MR_mouse_570-control-fema__E2_P1',
                './data/Mouse brain MRI/220210_SA_MR_mouse_572_control_fema__E2_P1',
                './data/Mouse brain MRI/220210_SA_MR_mouse_573-knockout-fem__E3_P1',
                './data/Mouse brain MRI/220210_SA_MR_mouse_582-knockout-fem__E2_P1']

for path_root in root_path_list:
    path_list = glob(os.path.join(path_root, '2_tiff/*.tif'))

    path_list = sorted(path_list, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

    Z = len(path_list)
    X, Y = plt.imread(path_list[0]).shape

    volume = np.zeros([Z, X, Y]).astype('float32')
    for i in range(Z):
        volume[i] = plt.imread(path_list[i])[:,:]

    img_3D = volume




    # 1. computes flows from 2D slices and combines into 3D flows to create masks
    masks, flows, _ = model.eval(img_3D, z_axis=0, channel_axis=None,
                                    batch_size=32,
                                    do_3D=True, flow3D_smooth=1)

    masks = masks!=0

    def has_holes_3d(vol, connectivity=1):
        vol = vol.astype(bool)
        bg = ~vol

        # 3D 连通性：1=6邻域, 2=26邻域
        st = ndi.generate_binary_structure(3, connectivity)

        # 标记背景连通域
        lab, n = ndi.label(bg, structure=st)
        if n == 0:
            return False, 0, None

        # 边界上的背景标签 = 外部背景
        border = np.zeros_like(vol, dtype=bool)
        border[0,:,:] = border[-1,:,:] = True
        border[:,0,:] = border[:,-1,:] = True
        border[:,:,0] = border[:,:,-1] = True

        external_ids = np.unique(lab[border & bg])
        external = np.isin(lab, external_ids)

        holes = bg & (~external)
        holes_count = int(ndi.label(holes, structure=st)[1])
        return holes_count > 0, holes_count, holes

    # 用法
    has, num, holes_mask = has_holes_3d(masks, connectivity=1)
    print(f'the number of holes: {num}')
    masks[holes_mask] = True

    masks = (masks.astype(np.uint8) * 255)
    for i in range(masks.shape[0]):
        tif_path = os.path.join(path_root, f'2_tiff_mask/mask_{i}.tif')
        tiff.imwrite(tif_path, masks[i])

        plt.figure()
        plt.imshow(img_3D[i])
        plt.imshow(masks[i], alpha=.25)
        plt.savefig(os.path.join(path_root, f'2_tiff_mask_overlap/mask_overlap_{i}.png'))
        plt.close()

    import matplotlib.pyplot as plt
    # plt.hist(df.iloc[1,1:])
    np.save(os.path.join(path_root, 'masks.npy'), masks)
    # with open(os.path.join(path_root, 'flows.pkl'), 'wb') as f:
    #     pickle.dump(flows, f)
        

    # # 2. computes masks in 2D slices and stitches masks in 3D based on mask overlap
    # print('running cellpose 2D + stitching masks')
    # masks_stitched, flows_stitched, _ = model.eval(img_3D, z_axis=0, channel_axis=None,
    #                                                   batch_size=32,
    #                                                   do_3D=False, stitch_threshold=0.5)

    # for i in range(masks_stitched.shape[0]):
    #     tif_path = os.path.join(path_root, f'2_tiff_mask_stitched/mask_{i}.tif')
    #     tiff.imwrite(tif_path, masks_stitched[i])

    # # np.save(os.path.join(path_root, 'masks_stitched.npy'), masks_stitched)
    # # with open(os.path.join(path_root, 'flows_stitched.pkl'), 'wb') as f:
    # #     pickle.dump(flows_stitched, f)



    # fig, ax = plt.subplots(2,3,figsize=[10,10])

    # ax[0,0].imshow(img_3D[5])
    # ax[0,1].imshow(img_3D[10])
    # ax[0,2].imshow(img_3D[200])
    # ax[1,0].imshow(masks[5])
    # ax[1,1].imshow(masks[10])
    # ax[1,2].imshow(masks[200])
    # plt.tight_layout()



labels, counts = np.unique(masks[masks != 0], return_counts=True)

# Sort by count descending
order = np.argsort(counts)[::-1]
labels, counts = labels[order], counts[order]


total = counts.sum()
frac = counts / total
cum = np.cumsum(frac) 

masks_new_ = np.isin(masks, labels[cum<.9]).astype(bool)



from scipy import ndimage as ndi

vol = (masks_new_ != 0)  # or (masks == 1) if you already have a binary mask

# Connectivity:
# 6-connected (faces): generate_binary_structure(3, 1)
# 26-connected (faces+edges+corners): generate_binary_structure(3, 2)
structure = ndi.generate_binary_structure(rank=3, connectivity=1)

cc, n = ndi.label(vol, structure=structure)   # cc has labels 0..n
sizes = np.bincount(cc.ravel())[1:]           # drop background count at index 0

print("num components:", n)
print("sizes (voxels) per component:", sizes)  # sizes[i] is size of component (i+1)













def keep_top_k_components(mask2d, k=3, connectivity=1):
    # connectivity=1 -> 4-connected, connectivity=2 -> 8-connected
    bw = mask2d != 0
    structure = ndi.generate_binary_structure(2, connectivity)

    cc, n = ndi.label(bw, structure=structure)
    if n == 0:
        return np.zeros_like(mask2d)

    sizes = np.bincount(cc.ravel())  # sizes[0] is background
    sizes[0] = 0                     # ignore background

    # component ids to keep (largest k)
    keep_ids = np.argsort(sizes)[::-1][:min(k, n)]

    out = np.zeros_like(mask2d)
    out[np.isin(cc, keep_ids)] = mask2d[np.isin(cc, keep_ids)]  # keep original values
    return out

masks_new = np.zeros_like(masks_new_)
for i in range(masks_new_.shape[0]):
    masks_new[i] = keep_top_k_components(masks_new_[i], k=5)











import open3d as o3d

points = np.argwhere(masks > 0)  ## masks (Z,X,Y), points (Z,X,Y)
color = masks[points[:,0], points[:,1], points[:,2]]/np.max(masks)


# 自定义颜色，每个点对应一个 RGB 值（范围 [0, 1]）
colors = np.zeros_like(points, dtype=float)      # 所有点初始为黑色
colors[:, 0] = color          # 红色通道 = x 坐标
colors[:, 1] = color/1.5          # 绿色通道 = y 坐标
colors[:, 2] = color/2          # 蓝色通道 = z 坐标

# 构建点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 可视化
o3d.visualization.draw_geometries([pcd])

