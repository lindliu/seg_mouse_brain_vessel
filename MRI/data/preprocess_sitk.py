#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 10:27:18 2026

@author: dliu
"""

import SimpleITK as sitk

def dicom_series_to_nifti(dicom_dir: str, out_nii_path: str) -> sitk.Image:
    # Find all series in the folder
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise ValueError(f"No DICOM series found in: {dicom_dir}")

    # If there's more than one series, pick the first (or choose by your logic)
    series_id = series_ids[0]

    # Get sorted filenames for that series (SimpleITK sorts by DICOM metadata)
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir, series_id)

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)

    # Optional but helpful for some datasets
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    image3d = reader.Execute()
    # image3d = sitk.Cast(image3d, sitk.sitkInt16)
    # Save as NIfTI (keeps origin/spacing/direction in the header)
    sitk.WriteImage(image3d, out_nii_path)
    return image3d


def load_image_as_array_with_meta(image_path: str, get_meta_info: bool = True):
    sitk_image = sitk.ReadImage(image_path)
    arr = sitk.GetArrayFromImage(sitk_image)  # Z, Y, X

    if not get_meta_info:
        return arr

    meta_info = {
        "sitk_image_object": sitk_image,
        "sitk_origin": sitk_image.GetOrigin(),
        "sitk_direction": sitk_image.GetDirection(),
        "sitk_spacing": sitk_image.GetSpacing(),
        "original_numpy_shape": arr.shape,  # Z, Y, X
    }
    return arr, meta_info




import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# specify your image path
root_path = glob.glob('./Mouse brain MRI/*')

i = -2
dcm_dir = glob.glob(os.path.join(root_path[i], '1_original'))[0]
out_nii_path = os.path.join(root_path[i], 'imagesVa', os.path.split(root_path[i])[1]+'.nii.gz')

dicom_series_to_nifti(dcm_dir, out_nii_path)

# img16 = sitk.Cast(arr, sitk.sitkInt16)
# sitk.WriteImage(img16, "out.nii.gz")  # saving as NIfTI is easiest



arr, meta_info = load_image_as_array_with_meta(out_nii_path, get_meta_info = True)
plt.imshow(arr[20])



label_paths = glob.glob(os.path.join(root_path[i], '2_tiff/*.npy'))
label_paths = sorted(label_paths, key=lambda x: int(os.path.split(x)[1][4:6]))

arr_mask = []
for j in range(len(label_paths)):
    label = np.load(label_paths[j], allow_pickle=True).item()
    arr_mask.append(label['masks'])
    
arr_mask = np.array(arr_mask,dtype=np.float32)

out_nii_path = os.path.join(root_path[i], 'labelsVa', os.path.split(root_path[i])[1]+'.nii.gz')
arr_mask = sitk.GetImageFromArray(arr_mask) 
sitk.WriteImage(arr_mask, out_nii_path)