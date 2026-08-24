# Mouse Brain Vascular Segmentation

This repository contains image-processing workflows for **mouse brain MRI** and **light-sheet microscopy** data, including preprocessing, brain/vessel segmentation, vessel diameter analysis, and 3D visualization.

## Workflow

### MRI

**DICOM MRI -> preprocessing -> Cellpose brain segmentation -> mask expansion -> Frangi vessel segmentation -> local thickness -> skeleton -> 3D visualization**

- 16 mouse head MRI volumes
- 256 slices per volume
- Matrix size: 512 x 256 pixels
- Voxel size: 58.59375 um
- 140 manually annotated slices from 4 volumes were used to fine-tune the Cellpose model
- The trained model was applied to all 16 volumes using Cellpose 2.5D segmentation
- Frangi filtering and percentile thresholding were used for vessel segmentation

### Light-sheet

**Light-sheet images -> Gaussian smoothing -> Sato filtering -> threshold combination -> vessel mask -> local thickness -> 3D visualization**

- 5 explanted mouse brain light-sheet datasets
- Sato tubeness filtering was used to enhance vascular structures
- Original-image and filtered-image thresholds were combined to generate the final vessel masks

## Vessel diameter analysis

For both MRI and light-sheet datasets:

1. Final 3D vessel masks were generated.
2. Local vessel thickness was calculated using **qim3d** with the physical voxel spacing of each dataset.
3. Vessel masks were skeletonized using **scikit-image**.
4. Local thickness values were sampled along the skeleton to obtain centerline-based vessel diameters.
5. Diameter distributions and summary statistics were generated for downstream analysis.

## Main software

- Python
- Cellpose
- scikit-image
- qim3d
- NumPy
- SciPy
- tifffile
- Plotly


