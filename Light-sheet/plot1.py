#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:45:29 2026

@author: dliu
"""

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob, os, re

num_re = re.compile(r'(\d+)(?!.*\d)')

root_path = './data/241030_659__13-02-02/'
print(root_path)

org_dir = '2_8bit'
org_dir_paths = glob.glob(os.path.join(root_path, f'{org_dir}/*.tif'))
org_dir_paths = sorted(org_dir_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

filter_dir = '3_filter'
filter_dir_paths = glob.glob(os.path.join(root_path, f'{filter_dir}/*.tif'))
filter_dir_paths = sorted(filter_dir_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))



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


idx = 400
filter_mask_99_path = filter_mask_99_paths[idx]
filter_mask_path = filter_mask_paths[idx]
thr_mask_path = thr_mask_paths[idx]
thr_mask_996_path = thr_mask_996_paths[idx]

mask_filter_99 = np.array(Image.open(filter_mask_99_path))>0
mask_filter = np.array(Image.open(filter_mask_path))>0
mask_thr = np.array(Image.open(thr_mask_path))>0
mask_thr_996 = np.array(Image.open(thr_mask_996_path))>0


mask1 = (mask_thr*mask_filter_99)>0  # remove low confidence area from filter mask
mask2 = (mask1 + mask_filter)>0      # add high confidence area from filter mask
mask = (mask2 + mask_thr_996)>0      # add high confidence area from threshold mask



import tifffile as tf

org_dir_path = org_dir_paths[idx]
filter_dir_path = filter_dir_paths[idx]

org_img = tf.imread(org_dir_path)
filter_img = tf.imread(filter_dir_path)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(
    org_img,
    cmap=plt.cm.gray,
    interpolation='nearest',
    resample=False
)
ax.set_aspect('equal')
ax.axis('off')  # 去掉横纵坐标和坐标轴边框
plt.savefig('./figure/org_img.png',
            bbox_inches='tight',
            pad_inches=0)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(
    filter_img,
    cmap=plt.cm.gray,
    interpolation='nearest',
    resample=False
)
ax.set_aspect('equal')
ax.axis('off')  # 去掉横纵坐标和坐标轴边框
plt.savefig('./figure/filter_img.png',
            bbox_inches='tight',
            pad_inches=0)









fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(
    mask_thr,
    cmap=colors.ListedColormap(['black', 'white']),
    interpolation='nearest',
    resample=False
)
ax.set_aspect('equal')
ax.axis('off')
plt.savefig('./figure/mask_thr_99.png',
            bbox_inches='tight',
            pad_inches=0)



fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(
    mask_filter_99,
    cmap=colors.ListedColormap(['black', 'white']),
    interpolation='nearest',
    resample=False
)
ax.set_aspect('equal')
ax.axis('off')
plt.savefig('./figure/mask_filter_99.png',
            bbox_inches='tight',
            pad_inches=0)



fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(
    mask_thr*mask_filter_99,
    cmap=colors.ListedColormap(['black', 'white']),
    interpolation='nearest',
    resample=False
)
ax.set_aspect('equal')
ax.axis('off')
plt.savefig('./figure/mask1.png',
            bbox_inches='tight',
            pad_inches=0)




fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(
    mask_filter,
    cmap=colors.ListedColormap(['black', 'red']),
    interpolation='nearest',
    resample=False
)
ax.set_aspect('equal')
ax.axis('off')
plt.savefig('./figure/mask_filter_995.png',
            bbox_inches='tight',
            pad_inches=0)


fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(
    mask_thr_996,
    cmap=colors.ListedColormap(['black', 'red']),
    interpolation='nearest',
    resample=False
)
ax.set_aspect('equal')
ax.axis('off')
plt.savefig('./figure/mask_thr_996.png',
            bbox_inches='tight',
            pad_inches=0)


mask2 = mask_filter+mask_thr_996
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(
    mask2,
    cmap=colors.ListedColormap(['black', 'red']),
    interpolation='nearest',
    resample=False
)
ax.set_aspect('equal')
ax.axis('off')
plt.savefig('./figure/mask2.png',
            bbox_inches='tight',
            pad_inches=0)




mask1 = mask_thr*mask_filter_99
mask2 = mask_filter+mask_thr_996
h, w = mask2.shape
img = np.zeros((h, w, 3), dtype=np.uint8)  # 默认黑色背景

img[mask2 > 0] = [255, 0, 0]        # 后赋值，红色覆盖白色
img[mask1 > 0] = [255, 255, 255]  # 白色


fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(
    img,
    interpolation='nearest',
    resample=False
)
ax.set_aspect('equal')
ax.axis('off')
plt.savefig('./figure/mask1_mask2.png',
            bbox_inches='tight',
            pad_inches=0)

