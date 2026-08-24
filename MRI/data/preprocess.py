import numpy as np
# import torch
import pydicom as dicom
import matplotlib.pylab as plt
import glob
import os 
import re
import tifffile as tiff


# print(torch.cuda.is_available())


def dcm_to_tif_8bit(dcm_path, tif_path, apply_rescale=False):
    ds = dicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)

    # optional rescale
    if apply_rescale:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

    # percentile window for display
    lo, hi = np.percentile(arr, (.1, 99.9))
    # lo, hi = arr.min(), arr.max()
    arr = np.clip(arr, lo, hi)
    arr8 = ((arr - lo) / (hi - lo + 1e-10) * 255.0).astype(np.uint8)

    tiff.imwrite(tif_path, arr8)
    
    
def dcm_to_tif_float32(dcm_path, tif_path, apply_rescale=False):
    ds = dicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)

    # Keep physical/intended values (if present)
    if apply_rescale:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

    # Save exactly (no normalization, no 65535 scaling)
    tiff.imwrite(tif_path, arr.astype(np.float32))


# specify your image path
root_path = glob.glob('./Mouse brain MRI/*/')

for i in range(len(root_path)):
    dcm_path = glob.glob(os.path.join(root_path[i], '1_original/*.dcm'))
    
    num_re = re.compile(r'(\d+)(?!.*\d)')
    dcm_path = sorted(dcm_path, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))
    
    tif_path = os.path.join(root_path[i], '2_tiff')
    os.makedirs(tif_path, exist_ok=True)
    
    for j in range(len(dcm_path)):
        dcm_path_ = dcm_path[j]
        
        tif_path_ = os.path.join(tif_path, os.path.split(dcm_path_)[1][:-4]+'.tif')
        # dcm_to_tif_float32(dcm_path_, tif_path_)
        dcm_to_tif_8bit(dcm_path_, tif_path_)


# plt.imshow(ds.pixel_array)


