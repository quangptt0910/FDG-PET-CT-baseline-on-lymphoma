#!/usr/bin/env python3
"""
Step 07: Evaluation.
Computes Dice, Sensitivity, and FP components for validation cases.
Generates per-case and summary metric files.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np

# Add the project root to sys.path to allow imports from src/utils
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.io_utils import load_yaml, save_json, setup_logger
from src.utils.nifti_utils import load_nifti_array
from src.utils.metrics_utils import dice_coefficient, sensitivity, count_false_positive_components


def main() -> None:
    """Main function for step 07 evaluation."""
    # Define project root
    project_root = Path(__file__).resolve().parent.parent

    # Load configuration files
    paths = load_yaml(project_root / "configs" / "paths.yaml")
    eval_cfg = load_yaml(project_root / "configs" / "evaluation.yaml")

    # Set up output directories (make absolute relative to project root)
    logs_dir = project_root / paths["logs_dir"]
    metrics_dir = project_root / paths["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logger = setup_logger("step07_evaluate", logs_dir / "step07_evaluate.log")
    logger.info("Starting Step 07: Evaluation")

    # Load validation subset (make absolute relative to project root)
    subset_val_path = project_root / paths["manifests_dir"] / "subset_val.csv"
    subset_val = pd.read_csv(subset_val_path)
    logger.info(f"Loaded {len(subset_val)} cases for evaluation.")

    # Define directories (make absolute relative to project root)
    predictions_dir = project_root / "output" / "predictions" / "output_val"
    labels_dir = project_root / paths["labels_dir"]

    results = []
    min_voxels = eval_cfg.get("min_lesion_voxels", 10)

    for _, row in subset_val.iterrows():
        case_id = row["case_id"]
        pred_path = predictions_dir / f"{case_id}.nii.gz"
        gt_path = labels_dir / f"{case_id}.nii.gz"

        if not pred_path.exists():
            logger.error(f"Prediction missing for {case_id}")
            continue

        try:
            pred_arr, _ = load_nifti_array(pred_path)
            gt_arr, _ = load_nifti_array(gt_path)

            # Binarize
            pred_bin = (pred_arr > 0.5).astype(np.uint8)
            gt_bin = (gt_arr > 0.5).astype(np.uint8)

            # Compute metrics
            dice = dice_coefficient(pred_bin, gt_bin)
            sens = sensitivity(pred_bin, gt_bin)
            fp_comp = count_false_positive_components(pred_bin, gt_bin, min_voxels)
            
            # Case level detection flag
            has_gt = np.any(gt_bin)
            has_pred = np.any(pred_bin)
            detected = has_gt and has_pred

            res_row = row.to_dict()
            res_row.update({
                "dice": round(dice, 4),
                "sensitivity": round(sens, 4),
                "fp_components": fp_comp,
                "gt_positive": has_gt,
                "pred_positive": has_pred,
                "detected": detected
            })
            results.append(res_row)
            logger.info(f"Case {case_id}: Dice={dice:.4f}, Sens={sens:.4f}, FP={fp_comp}")

        except Exception as e:
            logger.error(f"Error evaluating {case_id}: {e}")

    results_df = pd.DataFrame(results)
    per_case_csv = metrics_dir / "per_case_metrics.csv"
    results_df.to_csv(per_case_csv, index=False)
    logger.info(f"Per-case metrics saved to {per_case_csv}")

    # Aggregations
    # 1. Overall
    summary_overall = results_df[["dice", "sensitivity", "fp_components"]].agg(["mean", "std", "median"]).round(4)
    summary_overall.to_csv(metrics_dir / "summary_overall.csv")

    # 2. By Modality
    summary_modality = results_df.groupby("modality")[["dice", "sensitivity", "fp_components"]].agg(["mean", "std"]).round(4)
    summary_modality.to_csv(metrics_dir / "summary_by_modality.csv")

    # 3. By Scanner (Manufacturer)
    summary_scanner = results_df.groupby("manufacturer")[["dice", "sensitivity", "fp_components"]].agg(["mean", "std"]).round(4)
    summary_scanner.to_csv(metrics_dir / "summary_by_scanner.csv")

    # JSON Summary
    summary = {
        "total_evaluated": len(results_df),
        "mean_dice": float(results_df["dice"].mean()),
        "mean_sensitivity": float(results_df["sensitivity"].mean()),
        "mean_fp_components": float(results_df["fp_components"].mean())
    }
    save_json(summary, logs_dir / "step07_summary.json")
    logger.info(f"Step 07 summary: Mean Dice={summary['mean_dice']:.4f}")


if __name__ == "__main__":
    main()
