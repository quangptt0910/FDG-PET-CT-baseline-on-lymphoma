import numpy as np
from scipy import ndimage
from typing import Tuple


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Dice coefficient between prediction and ground truth.

    Args:
        pred: Binary prediction array
        gt: Binary ground truth array

    Returns:
        Dice coefficient as float
        - Both empty → 1.0 (true negative)
        - GT empty, pred non-empty → 0.0 (false positive study)
        - Otherwise: 2 * |pred ∩ gt| / (|pred| + |gt|)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt)
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)

    # Handle edge cases
    if pred_sum == 0 and gt_sum == 0:
        return 1.0  # Both empty → true negative
    elif gt_sum == 0 and pred_sum > 0:
        return 0.0  # GT empty but pred non-empty → false positive
    else:
        return 2.0 * np.sum(intersection) / (pred_sum + gt_sum)


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate sensitivity (recall) between prediction and ground truth.

    Args:
        pred: Binary prediction array
        gt: Binary ground truth array

    Returns:
        Sensitivity as float
        - GT empty, pred empty → 1.0 (true negative)
        - Otherwise: |pred ∩ gt| / |gt|
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt)
    gt_sum = np.sum(gt)

    # Handle edge case
    if gt_sum == 0:
        return 1.0  # GT empty → true negative regardless of pred
    else:
        return np.sum(intersection) / gt_sum


def count_false_positive_components(
    pred: np.ndarray, gt: np.ndarray, min_voxels: int
) -> int:
    """
    Count false positive components in prediction that don't overlap with ground truth.

    Args:
        pred: Binary prediction array
        gt: Binary ground truth array
        min_voxels: Minimum component size to consider (ignore smaller components)

    Returns:
        Number of false positive components
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # Label connected components in prediction
    labeled_pred, num_components = ndimage.label(pred)

    fp_count = 0
    for i in range(1, num_components + 1):
        # Get component mask
        component_mask = labeled_pred == i
        component_size = np.sum(component_mask)

        # Ignore small components
        if component_size < min_voxels:
            continue

        # Check if component overlaps with any ground truth
        overlap = np.logical_and(component_mask, gt)
        if not np.any(overlap):
            fp_count += 1

    return fp_count
