import numpy as np
# import torch
import pydicom
import matplotlib.pylab as plt
import glob
import os 
import re
import tifffile as tiff
from PIL import Image

# print(torch.cuda.is_available())
def dcm_to_tif_8bit(dcm_path, tif_path, step_y=1, step_x=1, apply_rescale=False):
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)

    # optional rescale
    if apply_rescale:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

    # arr = arr[650:1650,400:1400] # [325:825,200:700] #
    arr = arr[::step_y, ::step_x]

    # percentile window for display
    lo, hi = np.percentile(arr, (.1, 99.9))
    # lo, hi = arr.min(), arr.max()
    arr = np.clip(arr, lo, hi)
    arr8 = ((arr - lo) / (hi - lo + 1e-10) * 255.0).astype(np.uint8)

    tiff.imwrite(tif_path, arr8)
    

num_re = re.compile(r'(\d+)(?!.*\d)')

# specify your image path
# root_path = glob.glob('./data/241030_652__11-18-58')[0]
root_path = glob.glob('./data/241030_653__10-01-07')[0]

root_paths = glob.glob('./data/*')
print(root_paths)

for root_path in root_paths:
    tif_paths = glob.glob(os.path.join(root_path,'1_original/*.tif'))
    tif_paths = sorted(tif_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))


    ##### get lower and higher boundary #####
    bits = 16
    maxv = 2**bits
    counts = np.zeros(maxv, dtype=np.uint64)
    total = 0

    for path in tif_paths:
        im = Image.open(path)
        imarray = np.array(im)
        flat = imarray.ravel()
        counts = counts+np.bincount(flat, minlength=maxv)
        total += flat.size
    cdf = np.cumsum(counts)

    thresholds = {}
    for q in [.3,99.7]:
        target = int(np.ceil((q / 100.0) * total))
        idx = int(np.searchsorted(cdf, target, side="left"))
        thresholds[q] = idx  # 这个 idx 就是 q% 的像素强度阈值
    print(thresholds)

    lo, hi = list(thresholds.values())
    np.save(os.path.join(root_path,'lo_hi_boundary.npy'), np.array([lo,hi]))


    tif_path = os.path.join(root_path, '2_8bit')
    os.makedirs(tif_path, exist_ok=True)

    ##### convert 16bit to 8bit #####
    # volumn = []
    for path in tif_paths:
        im = Image.open(path)
        imarray = np.array(im)
        
        arr = np.clip(imarray, lo, hi)
        arr8 = ((arr - lo) / (hi - lo + 1e-10) * 255.0).astype(np.uint8)


        tif_path_ = os.path.join(tif_path, os.path.split(path)[1])
        tiff.imwrite(tif_path_, arr8)

        # volumn.append(imarray)
    # volumn = np.array(volumn)






    # # 快速计算
    # counts, bins = np.histogram(imarray, bins=100)
    # counts_8, bins_8 = np.histogram(arr8, bins=100)
    # # 快速绘制
    # fig,axe = plt.subplots(1,2,figsize=[10,5])
    # axe[0].bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black')
    # axe[1].bar(bins_8[:-1], counts_8, width=np.diff(bins_8), edgecolor='black')
    # plt.savefig(os.path.join(root_path, 'hist_comparison.png'))


