import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Dict


def load_nifti_header(path: Path) -> Dict[str, any]:
    """
    Load NIfTI header information without loading the full array data.

    Args:
        path: Path to the NIfTI file

    Returns:
        Dictionary containing shape, spacing, and affine matrix
    """
    img_nii = nib.load(str(path))
    shape = img_nii.shape
    spacing = np.abs(np.diag(img_nii.affine)[:3])  # Extract voxel spacing
    affine = img_nii.affine

    return {"shape": shape, "spacing": spacing, "affine": affine}


def load_nifti_array(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load NIfTI file as numpy array with affine matrix.

    Args:
        path: Path to the NIfTI file

    Returns:
        Tuple of (data array, affine matrix)
    """
    img_nii = nib.load(str(path))
    data = img_nii.get_fdata()
    affine = img_nii.affine

    return data, affine


def get_axial_slice_with_most_lesion(mask: np.ndarray) -> int:
    """
    Find the axial slice (z-axis) with the most lesion voxels.

    Args:
        mask: 3D binary mask array (assumed to be in RAS orientation)

    Returns:
        Index of the axial slice with the most foreground voxels
    """
    # Sum over x and y dimensions to get lesion count per z-slice
    lesion_counts_per_slice = np.sum(mask, axis=(0, 1))
    # Return the index of the slice with maximum lesion count
    return int(np.argmax(lesion_counts_per_slice))
