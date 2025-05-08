"""
Required Libraries
"""

import os
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

"""Normalize a slice to 0-255 uint8."""

def normalize_slice(slice):
    """ Normalize a slice to 0-255 uint8. """
    slice = slice - np.min(slice)
    if np.max(slice) != 0:
        slice = slice / np.max(slice)
    slice = (slice * 255).astype(np.uint8)
    return slice

"""Extract 2D Axial Slices from T1c .nii.gz"""

def extract_slices_from_nifti(input_dir, output_dir, slice_axis=2):
    """
    Traverse input_dir for .nii.gz files and save 2D slices as PNGs to output_dir.

    Args:
        input_dir: Root folder with T1c .nii.gz files
        output_dir: Where to store PNGs in patient-wise folders
        slice_axis: Axis to slice along (default: axial → axis 2)
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith("-t1c.nii.gz"):
                nii_path = os.path.join(root, file)

                # Preserve relative patient folder path
                rel_path = os.path.relpath(root, input_dir)
                out_folder = os.path.join(output_dir, rel_path)
                os.makedirs(out_folder, exist_ok=True)

                # Load the NIfTI file
                img = nib.load(nii_path)
                volume = img.get_fdata()

                num_slices = volume.shape[slice_axis]
                for i in range(num_slices):
                    if slice_axis == 0:
                        slice_2d = volume[i, :, :]
                    elif slice_axis == 1:
                        slice_2d = volume[:, i, :]
                    else:
                        slice_2d = volume[:, :, i]

                    normalized = normalize_slice(slice_2d)
                    im = Image.fromarray(normalized)

                    slice_name = f"slice_{i:03}.png"
                    save_path = os.path.join(out_folder, slice_name)
                    im.save(save_path)

                print(f"Extracted {num_slices} slices from {file} to {rel_path}")

# === Example Usage ===
input_t1c_folder = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-t1c_only"
output_png_folder = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-t1c_onlynifti_to_png2D"

extract_slices_from_nifti(input_t1c_folder, output_png_folder)

"""Extraction of segmentation only files"""

def extract_slices_from_nifti(input_dir, output_dir, slice_axis=2):
    """
    Traverse input_dir for .nii.gz files and save 2D slices as PNGs to output_dir.

    Args:
        input_dir: Root folder with T1c .nii.gz files
        output_dir: Where to store PNGs in patient-wise folders
        slice_axis: Axis to slice along (default: axial → axis 2)
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith("-seg.nii.gz"):
                nii_path = os.path.join(root, file)

                # Preserve relative patient folder path
                rel_path = os.path.relpath(root, input_dir)
                out_folder = os.path.join(output_dir, rel_path)
                os.makedirs(out_folder, exist_ok=True)

                # Load the NIfTI file
                img = nib.load(nii_path)
                volume = img.get_fdata()

                num_slices = volume.shape[slice_axis]
                for i in range(num_slices):
                    if slice_axis == 0:
                        slice_2d = volume[i, :, :]
                    elif slice_axis == 1:
                        slice_2d = volume[:, i, :]
                    else:
                        slice_2d = volume[:, :, i]

                    # Convert label image to uint8 directly (no normalization)
                    slice_2d = slice_2d.astype(np.uint8)
                    im = Image.fromarray(slice_2d, mode='L')  # 'L' = grayscale

                    slice_name = f"slice_{i:03}.png"
                    save_path = os.path.join(out_folder, slice_name)
                    im.save(save_path)

                print(f"Extracted {num_slices} slices from {file} to {rel_path}")

# === Example Usage ===
input_seg_folder = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-seg_only"
output_png_folder = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-seg_onlynifti_to_png2D"

extract_slices_from_nifti(input_t1c_folder, output_png_folder)

"""Converting segmentation NIfTI files to 2D PNGs with visualizations"""

def extract_segmentation_slices_dual(
    input_dir,
    output_dir_raw,
    output_dir_visual,
    slice_axis=2,
    suffix='seg.nii.gz'
):
    """
    Extract 2D slices from segmentation volumes and save:
    - raw grayscale PNGs for training
    - enhanced PNGs for visualization

    Args:
        input_dir: Input folder with segmentation NIfTI files
        output_dir_raw: Output path for raw label PNGs (0/1/2/4 preserved)
        output_dir_visual: Output path for visual PNGs (label * 60)
        slice_axis: Axis to slice (default=2 for axial)
        suffix: Filename suffix to identify segmentation files
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(suffix):
                nii_path = os.path.join(root, file)

                # Preserve patient-relative path
                rel_path = os.path.relpath(root, input_dir)
                out_folder_raw = os.path.join(output_dir_raw, rel_path)
                out_folder_vis = os.path.join(output_dir_visual, rel_path)
                os.makedirs(out_folder_raw, exist_ok=True)
                os.makedirs(out_folder_vis, exist_ok=True)

                # Load and process
                img = nib.load(nii_path)
                volume = img.get_fdata()
                num_slices = volume.shape[slice_axis]

                for i in range(num_slices):
                    if slice_axis == 0:
                        slice_2d = volume[i, :, :]
                    elif slice_axis == 1:
                        slice_2d = volume[:, i, :]
                    else:
                        slice_2d = volume[:, :, i]

                    # Raw labels: save for training
                    raw_slice = slice_2d.astype(np.uint8)
                    raw_im = Image.fromarray(raw_slice, mode='L')

                    # Visual version: amplify label values
                    vis_slice = (slice_2d * 60).clip(0, 255).astype(np.uint8)
                    vis_im = Image.fromarray(vis_slice, mode='L')

                    # Save files
                    slice_name = f"slice_{i:03}.png"
                    raw_im.save(os.path.join(out_folder_raw, slice_name))
                    vis_im.save(os.path.join(out_folder_vis, slice_name))

                print(f"Saved {num_slices} slices for {file} → {rel_path}")

# === Example Usage ===
input_seg_folder = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-seg_only"
output_raw_folder = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-seg_onlynifti_to_png2D"
output_visual_folder = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-seg_onlynifti_to_png2D_color"

extract_segmentation_slices_dual(
    input_dir=input_seg_folder,
    output_dir_raw=output_raw_folder,
    output_dir_visual=output_visual_folder,
    suffix="seg.nii.gz"  # or adjust suffix if your files differ
)

"""Generate 2D MIP(Maximum Intensity Projection) Images from 3D T1c NIfti format.

Converting T1-c Nifti format to Axial (Transverse) plane 2D png image.

Axis=2 or 1 (Coronal/frontal plane) or 0 (Saggital/lateral or right and left cut into half)
"""

def normalize_to_uint8(img_2d):
    """Normalize 2D image to 0-255 (uint8)."""
    img = img_2d - np.min(img_2d)
    if np.max(img) != 0:
        img = img / np.max(img)
    return (img * 255).astype(np.uint8)

def convert_t1c_to_mip(input_dir, output_dir, suffix="-t1c.nii.gz", axis=2):
    """
    Convert each T1c 3D volume to a 2D MIP image and save as PNG.

    Args:
        input_dir: Folder containing T1c .nii.gz files in patient folders
        output_dir: Folder to save MIP PNGs
        suffix: File suffix to identify T1c files
        axis: Axis for MIP (default: 2 = axial)
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(suffix):
                nii_path = os.path.join(root, file)

                # Load volume
                volume = nib.load(nii_path).get_fdata()

                # Max intensity projection
                mip = np.max(volume, axis=axis)
                mip_img = normalize_to_uint8(mip)

                # Save as PNG
                patient_id = os.path.basename(root)  # keep folder name as ID
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f"{patient_id}_mip.png")

                Image.fromarray(mip_img).save(out_path)
                print(f"MIP saved for {patient_id} → {out_path}")

# === Example Usage ===
input_t1c_dir = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-t1c_only"
output_mip_dir = "/Users/parthsalke/D Drive/MRI Data- Gliomas Tumors/PKG - BraTS-Africa-t1c_MPI_2D"

convert_t1c_to_mip(input_t1c_dir, output_mip_dir)

