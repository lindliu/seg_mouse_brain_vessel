import numpy as np
import os, glob, re
import matplotlib.pyplot as plt
import pydicom
import tifffile as tiff
from PIL import Image
import pandas as pd
import qim3d
import kimimaro

from skimage.filters import frangi, threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, ball, binary_opening, skeletonize, binary_dilation
from scipy.ndimage import distance_transform_edt


def read_dicom_volume(dcm_paths):
    """Return volume (z,y,x), spacing (z,y,x in mm), and sorted paths."""
    # dcm_paths = sort_dicom_paths(dcm_paths)
    slices = [pydicom.dcmread(p, force=True) for p in dcm_paths]

    vol = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)  # (z,y,x)

    # spacing
    py, px = map(float, slices[0].PixelSpacing) if hasattr(slices[0], "PixelSpacing") else (1.0, 1.0)

    if hasattr(slices[0], "SpacingBetweenSlices"):
        pz = float(slices[0].SpacingBetweenSlices)
    elif hasattr(slices[0], "SliceThickness"):
        pz = float(slices[0].SliceThickness)
    elif all(hasattr(s, "ImagePositionPatient") for s in slices) and len(slices) > 1:
        zs = [float(s.ImagePositionPatient[2]) for s in slices]
        pz = float(np.median(np.diff(sorted(zs))))
    else:
        pz = 1.0

    spacing_zyx = (pz, py, px)
    return vol, spacing_zyx


def read_mask_volume(mask_paths):
    # """Read per-slice tif masks into (z,y,x) boolean volume, sorted by trailing number."""
    # num_re = re.compile(r'(\d+)(?!.*\d)')
    # mask_paths = sorted(mask_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))

    masks = []
    for p in mask_paths:
        m = np.array(Image.open(p))
        masks.append(m != 0)
    mask_vol = np.stack(masks, axis=0)
    return mask_vol


def robust_normalize_in_mask(vol, mask, p_low=.1, p_high=99.9):
    vals = vol[mask]
    if vals.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    lo, hi = np.percentile(vals, (p_low, p_high))
    vol_c = np.clip(vol, lo, hi)
    return ((vol_c - lo) / (hi - lo + 1e-6)).astype(np.float32)


def dilate_one_sided(mask: np.ndarray, iterations: int = 1, axis: int = 0, direction: int = 1) -> np.ndarray:
    """
    One-sided (directional) dilation for a binary 3D array.
    Grows ONLY along `axis` in `direction`:
      - direction=+1 grows toward increasing index on that axis
      - direction=-1 grows toward decreasing index on that axis

    No growth happens sideways or "upward" (opposite direction).
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {mask.shape}")
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")

    out = mask.astype(bool, copy=True)

    for _ in range(int(iterations)):
        shifted = np.zeros_like(out, dtype=bool)

        src = [slice(None)] * out.ndim
        dst = [slice(None)] * out.ndim

        if direction == 1:
            # move voxels one step toward +axis (adds new voxels "below" if bottom is +axis)
            src[axis] = slice(0, -1)
            dst[axis] = slice(1, None)
        else:
            # move voxels one step toward -axis (adds new voxels "below" if bottom is -axis)
            src[axis] = slice(1, None)
            dst[axis] = slice(0, -1)

        shifted[tuple(dst)] = out[tuple(src)]
        out |= shifted  # union -> directional dilation

    return out


root_path_list = root_path = [
                            './data/Mouse brain MRI/220210_SA_MR_mouse_338-control-male__E7_P1',
                            './data/Mouse brain MRI/220210_SA_MR_mouse_321-knockout-mal__E6_P1',
                            './data/Mouse brain MRI/220210_SA_MR_mouse_329-control-male__E4_P1',
                            './data/Mouse brain MRI/220210_SA_MR_mouse_378-knockout-mal__E4_P1',
                            './data/Mouse brain MRI/220210_SA_MR_mouse_330-knockout-mal__E11_P1',
                            './data/Mouse brain MRI/220210_SA_MR_mouse_531-control-fema__E7_P1',
                            './data/Mouse brain MRI/220210_SA_MR_mouse_570-control-fema__E2_P1',
                            './data/Mouse brain MRI/220210_SA_MR_mouse_572_control_fema__E2_P1',
                            './data/Mouse brain MRI/220210_SA_MR_mouse_573-knockout-fem__E3_P1',
                            './data/Mouse brain MRI/220210_SA_MR_mouse_582-knockout-fem__E2_P1']


for root_path in root_path_list:
    print(root_path)

    dcm_paths = glob.glob(os.path.join(root_path, '1_original/*.dcm'))
    mask_paths = glob.glob(os.path.join(root_path, '2_tiff_mask/*.tif'))

    num_re = re.compile(r'(\d+)(?!.*\d)')
    dcm_paths = sorted(dcm_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))
    mask_paths = sorted(mask_paths, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))


    # ---------- load 3D ----------
    vol, spacing_zyx = read_dicom_volume(dcm_paths)
    mask_vol = read_mask_volume(mask_paths)

    se = ball(2)                      # radius 2 (in voxels)
    mask_vol = binary_dilation(mask_vol, footprint=se)
    ## dilation along y axis to bottom for 6 pixels
    mask_vol = dilate_one_sided(mask_vol, iterations=20, axis=1, direction=1)
    assert vol.shape == mask_vol.shape, f"Shape mismatch: vol{vol.shape} vs mask{mask_vol.shape}"
    
    
    # np.save(os.path.join(root_path, 'mask_extend.npy'), mask_vol)
    # masks = (mask_vol.astype(np.uint8) * 255)
    # for i in range(masks.shape[0]):
    #     os.makedirs(os.path.join(root_path, '2_tiff_mask_extend'), exist_ok=True)
    #     tiff.imwrite(os.path.join(root_path, f'2_tiff_mask_extend/mask_{i}.tif'), masks[i])
        
    #     os.makedirs(os.path.join(root_path, '2_tiff_mask_overlap_extend'), exist_ok=True)
    #     plt.figure()
    #     plt.imshow(vol[i])
    #     plt.imshow(masks[i], alpha=.25)
    #     plt.savefig(os.path.join(root_path, f'2_tiff_mask_overlap_extend/mask_overlap_{i}.png'))
    #     plt.close()


    # ---------- brain-only + normalize ----------
    brain = vol * mask_vol
    brain_n = robust_normalize_in_mask(brain, mask_vol)
    Z = vol.shape[0]

    # # ---------- save slice-by-slice ----------
    # # output dirs (same as your 2D case)
    # seg_dir = os.path.join(root_path, "3_tiff_segmented")
    # os.makedirs(seg_dir, exist_ok=True)
    # for i in range(Z):
    #     # brain slice
    #     lo, hi = np.percentile(brain_n, (.1, 99.9))
    #     # lo, hi = arr.min(), arr.max()
    #     brain_n = np.clip(brain_n, lo, hi)
    #     brain_int8 = ((brain_n - lo) / (hi - lo + 1e-10) * 255.0).astype(np.uint8)
    #     tiff.imwrite(os.path.join(seg_dir, f"brain_{i}.tif"), brain_int8[i])


    using_fragi = True # False # 
    # ---------- 3D vesselness ----------
    if using_fragi:
        vesselness = frangi(
            brain_n,
            sigmas=np.linspace(0.5, 10.0, 15),  # tune  sigma_vox ~ radius_mm / spacing_xy_mm
            black_ridges=False                 # True if vessels are dark
        ).astype(np.float32)
        vesselness *= mask_vol

        save_path = os.path.join(root_path, "Frangi_archived")
        vesness_dir = os.path.join(save_path, "3_tiff_vessels_frangi")
        ves_dir = os.path.join(save_path, "4_tiff_vessels")
        thk_dir = os.path.join(save_path, "5_tiff_vessels_thickness")

        os.makedirs(vesness_dir, exist_ok=True)
        os.makedirs(ves_dir, exist_ok=True)
        os.makedirs(thk_dir, exist_ok=True)

        for i in range(Z):
            # vesselness slice
            tiff.imwrite(os.path.join(vesness_dir, f"vesselness_{i}.tif"), vesselness[i].astype(np.float32))

    else:
        vesselness = brain_n*mask_vol

        save_path = os.path.join(root_path, "Threshold_archived")
        ves_dir = os.path.join(save_path, "4_tiff_vessels")
        thk_dir = os.path.join(save_path, "5_tiff_vessels_thickness")
        os.makedirs(ves_dir, exist_ok=True)
        os.makedirs(thk_dir, exist_ok=True)
    

    # ---------- 3D vessel mask ----------
    vn_vals = vesselness[mask_vol]
    # thr = threshold_otsu(vn_vals) if vn_vals.size else 0.0
    thr = np.percentile(vn_vals, 96)  # tune 95–99.5
    vessel_bin = vesselness > thr

    ## cleanup
    # vessel_bin = binary_opening(vessel_bin, ball(1)) ## Opening = erosion then dilation.
    vessel_bin = remove_small_objects(vessel_bin, min_size=3)       # tune
    vessel_bin = remove_small_holes(vessel_bin, area_threshold=500)   # tune

    for i in range(Z):
        # vessel binary slice
        tiff.imwrite(os.path.join(ves_dir, f"vessel_{i}.tif"), (vessel_bin[i].astype(np.uint8) * 255))



    # save_path = os.path.join(root_path, "Frangi_archived")
    # ves_dir = os.path.join(save_path, "4_tiff_vessels")
    # thk_dir = os.path.join(save_path, "5_tiff_vessels_thickness")
    # from PIL import Image
    # ppp = glob.glob(os.path.join(save_path,'4_tiff_vessels/*.tif'))
    # ppp = sorted(ppp, key=lambda x: int(num_re.search(os.path.split(x)[1]).group(1)))
    # vessel_bin = []
    # for p in ppp:
    #     im = np.array(Image.open(p))
    #     vessel_bin.append(im)
    # vessel_bin = np.array(vessel_bin)
    # vessel_bin = vessel_bin>0

    # ---------- 3D thickness (mm) ----------
    assert len(np.unique(spacing_zyx))==1, 'not isotropic voxels'
    diameter_mm_uniform = qim3d.processing.local_thickness(vessel_bin, visualize=False, axis=0)
    diameter_mm_uniform = diameter_mm_uniform * spacing_zyx[0]
    diameter_um_uniform = diameter_mm_uniform*10**3

    skel = skeletonize(vessel_bin)
    diameter_mm_skel = np.zeros_like(diameter_mm_uniform, dtype=np.float32)
    diameter_mm_skel[skel] = diameter_mm_uniform[skel]
    diameter_um_skel = diameter_mm_skel*10**3

    #################### using kimimaro to get skeleton ####################
    # labels = vessel_bin.astype(np.uint32)  # 0 background, 1 foreground
    # skeletons = kimimaro.skeletonize(
    #     labels,
    #     teasar_params={
    #         "scale": 1,
    #         "const": 500,
    #         "pdrf_exponent": 4,
    #         "pdrf_scale": 100000,
    #         "soma_detection_threshold": 1100,
    #         "soma_acceptance_threshold": 3500,
    #         "soma_invalidation_scale": 1.0,
    #         "soma_invalidation_const": 300,
    #         "max_paths": 300,
    #     },
    #     dust_threshold=0,
    #     progress=True,
    # )

    # skel1 = skeletons[1]  # skeleton for label 1
    # skel = np.zeros(labels.shape, dtype=bool)

    # # skel1.vertices is an (N, 3) array of coordinates
    # # If you used anisotropy in kimimaro, you may need to divide back (see note below).
    # verts = np.asarray(skel1.vertices)

    # # Convert to integer voxel indices
    # verts = np.rint(verts).astype(int)

    # # Clip to be safe
    # zmax, ymax, xmax = np.array(labels.shape) - 1
    # verts[:, 0] = np.clip(verts[:, 0], 0, zmax)
    # verts[:, 1] = np.clip(verts[:, 1], 0, ymax)
    # verts[:, 2] = np.clip(verts[:, 2], 0, xmax)

    # skel[verts[:, 0], verts[:, 1], verts[:, 2]] = True
    # diameter_mm_skel = np.zeros_like(diameter_mm_uniform, dtype=np.float32)
    # diameter_mm_skel[skel] = diameter_mm_uniform[skel]
    # diameter_um_skel = diameter_mm_skel*10**3
    #################################################################


    ############ skimage to get thickness and skeleton ##############
    # dist_mm, inds = distance_transform_edt(vessel_bin, 
    #                                     return_indices=True, 
    #                                     sampling=spacing_zyx)

    # # optional: skeleton thickness (mm) per slice too
    # skel = skeletonize(vessel_bin)

    # def thickness_from_skeleton(vessel_bin, dist_mm, skel, spacing_zyx):
    #     """
    #     vessel_bin: bool (z,y,x)
    #     dist_mm: float (z,y,x) EDT of vessel_bin in mm  (radius field)
    #     skel: bool (z,y,x) centerline/ridge points
    #     spacing_zyx: (pz,py,px) in mm
    #     returns: diameter_mm_uniform (float32) where vessel cross-sections are ~constant
    #     """
    #     assert skel.any(), "skel is empty"

    #     # diameter only on skeleton (mm)
    #     diameter_mm_skel = np.zeros_like(dist_mm, dtype=np.float32)
    #     diameter_mm_skel[skel] = (2.0 * dist_mm[skel]).astype(np.float32)

    #     # nearest skeleton voxel for every voxel (in mm-space)
    #     _, inds = distance_transform_edt(
    #         ~skel,                  # skeleton voxels are targets (zeros)
    #         return_indices=True,
    #         sampling=spacing_zyx
    #     )
    #     nz, ny, nx = inds
    #     diameter_mm_uniform = diameter_mm_skel[nz, ny, nx].astype(np.float32)
    #     diameter_mm_uniform[~vessel_bin] = 0.0
    #     return diameter_mm_uniform

    # diameter_mm_uniform = thickness_from_skeleton(vessel_bin, dist_mm, skel, spacing_zyx)
    # diameter_mm_skel = np.zeros_like(diameter_mm_uniform, dtype=np.float32)
    # diameter_mm_skel[skel] = diameter_mm_uniform[skel]

    # diameter_um_uniform = diameter_mm_uniform*10**3
    # diameter_um_skel = diameter_mm_skel*10**3
    ####################################################################

    # ---------- save slice-by-slice ----------
    for i in range(Z):
        # diameter map slice (mm)
        tiff.imwrite(os.path.join(thk_dir, f"diameter_um_{i}.tif"), diameter_um_uniform[i])

    diameter_um_uniform[-1,-1,-1]=215
    tiff.imwrite(os.path.join(thk_dir, f"diameter_um_{i}.tif"), diameter_um_uniform[i])

    # for root_path in root_path_list:
    #     print(root_path)
    #     p = os.path.join(root_path,'Frangi_archived/5_tiff_vessels_thickness/diameter_um_255.tif')
    #     im = np.array(Image.open(p))
    #     im[-1,-1]=215
    #     tiff.imwrite(p, im)


    val_vessel, num_vessel = np.unique(diameter_um_uniform[diameter_um_uniform!=0], return_counts=True)
    val_skel, num_skel = np.unique(diameter_um_skel[diameter_um_skel!=0], return_counts=True)
    np.savez(save_path+'/hist.npz',val_vessel=val_vessel,num_vessel=num_vessel,val_skel=val_skel,num_skel=num_skel)





    import numpy as np
    import plotly.graph_objects as go

    def plot_surface(vessel_bin, spacing_zyx):
        # vessel_bin: (z,y,x) boolean
        # spacing_zyx: (pz,py,px) in mm (use (1,1,1) if unknown)
        pz, py, px = spacing_zyx

        # ---- quick sanity check ----
        nvox = int((vessel_bin!=0).sum())
        print("vessel voxels:", nvox)
        assert nvox > 0, "vessel_bin is empty (no True voxels). Check threshold/cleanup."

        # ---- downsample for speed (recommended) ----
        step = 2  # try 1 for full-res, 2/3 for faster
        vb = vessel_bin[::step, ::step, ::step].astype(np.uint8)
        pz_s, py_s, px_s = pz * step, py * step, px * step

        Z, Y, X = vb.shape

        # build coordinates grid in mm
        zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
        x = (xx * px_s).ravel()
        y = (yy * py_s).ravel()
        z = (zz * pz_s).ravel()
        val = vb.ravel()

        fig = go.Figure(
            data=go.Isosurface(
                x=x, y=y, z=z,
                value=val,
                isomin=0.5,
                isomax=1.0,
                surface_count=1,
                caps=dict(x_show=False, y_show=False, z_show=False),
                opacity=0.6
            )
        )

        fig.update_layout(
            title="Vessel mask (Isosurface)",
            scene=dict(
                xaxis_title="x (mm)",
                yaxis_title="y (mm)",
                zaxis_title="z (mm)",
                aspectmode="data"
            )
        )
        fig.show()





    def plot_points(mask, spacing_zyx, value_map=None, save_html=None):

        pts = np.argwhere(mask)
        print("points:", pts.shape[0])

        max_pts = 200000
        if pts.shape[0] > max_pts:
            pts = pts[np.random.choice(pts.shape[0], max_pts, replace=False)]

        if value_map is not None:
            val = value_map[mask]
        else:
            val = 1

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


    plot_points(vessel_bin, spacing_zyx, value_map=diameter_mm_uniform, save_html=save_path+'/vessels.html')
    plot_points(skel, spacing_zyx, value_map=diameter_um_skel, save_html=save_path+'/skel.html')