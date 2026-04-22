#!/usr/bin/env python3
"""
Step 08: Visualisation.
Generates dataset statistics plots and prediction overlays.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Add the project root to sys.path to allow imports from src/utils
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.io_utils import load_yaml, save_json, setup_logger
from src.utils.plot_utils import save_figure, make_overlay_panel
from src.utils.nifti_utils import load_nifti_array, get_axial_slice_with_most_lesion


def plot_dataset_stats(master_df: pd.DataFrame, subset_combined: pd.DataFrame, figures_dir: Path, logger: Any) -> None:
    """Generate demographic and composition plots."""
    logger.info("Generating dataset statistics plots...")
    
    # 1. Age distribution by modality
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for modality in ["PSMA", "FDG"]:
        # age might have 'Y' suffix, need to clean
        ages = master_df[master_df["modality"] == modality]["age"]
        if ages.dtype == object:
            ages = ages.str.replace('Y', '').astype(float)
        ax1.hist(ages, bins=20, alpha=0.5, label=modality)
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Count")
    ax1.set_title("Age Distribution by Modality (Master Manifest)")
    ax1.legend()
    save_figure(fig1, figures_dir / "fig01_age_distribution.png")

    # 2. Scanner distribution
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    master_df["manufacturer"].value_counts().plot(kind="barh", ax=ax2)
    ax2.set_xlabel("Count")
    ax2.set_ylabel("Scanner Model")
    ax2.set_title("Scanner Distribution (Master Manifest)")
    save_figure(fig2, figures_dir / "fig02_scanner_distribution.png")

    # 3. Tracer distribution
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    master_df["pet_tracer"].value_counts().plot(kind="bar", ax=ax3)
    ax3.set_ylabel("Count")
    ax3.set_title("PET Tracer Distribution")
    save_figure(fig3, figures_dir / "fig03_tracer_distribution.png")

    # 4. Split Composition
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    composition = subset_combined.groupby(["role", "modality"]).size().unstack()
    composition.plot(kind="bar", stacked=True, ax=ax5)
    ax5.set_ylabel("Count")
    ax5.set_title("Subset Composition (Train vs Val)")
    save_figure(fig5, figures_dir / "fig05_split_composition.png")


def plot_metrics(per_case_metrics: pd.DataFrame, figures_dir: Path, logger: Any) -> None:
    """Generate metric distribution and analysis plots."""
    logger.info("Generating metric plots...")
    
    # 6. Dice distribution
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    data = [
        per_case_metrics["dice"],
        per_case_metrics[per_case_metrics["modality"] == "PSMA"]["dice"],
        per_case_metrics[per_case_metrics["modality"] == "FDG"]["dice"]
    ]
    ax6.boxplot(data, labels=["Overall", "PSMA", "FDG"])
    ax6.set_ylabel("Dice Coefficient")
    ax6.set_title("Dice Distribution")
    save_figure(fig6, figures_dir / "fig06_dice_distribution.png")

    # 7. Dice vs Volume (approximate volume by pet_size_mb for demo)
    fig7, ax7 = plt.subplots(figsize=(8, 6))
    ax7.scatter(per_case_metrics["pet_size_mb"], per_case_metrics["dice"], alpha=0.6)
    ax7.set_xlabel("PET File Size (MB) - Proxy for Resolution/Volume")
    ax7.set_ylabel("Dice")
    ax7.set_title("Dice vs File Size")
    save_figure(fig7, figures_dir / "fig07_dice_vs_volume.png")


def generate_overlays(per_case_metrics: pd.DataFrame, paths: Dict[str, Any], project_root: Path, figures_dir: Path, logger: Any) -> List[str]:
    """Generate slice overlays for best/median/worst cases."""
    logger.info("Generating overlay figures...")
    generated_files = []
    
    # Define directories (make absolute relative to project root)
    predictions_dir = project_root / "output" / "predictions" / "output_val"
    labels_dir = project_root / paths["labels_dir"]
    images_dir = project_root / paths["images_dir"]

    for modality in ["PSMA", "FDG"]:
        m_df = per_case_metrics[per_case_metrics["modality"] == modality].sort_values("dice")
        if m_df.empty: continue
        
        # Select best, median, worst
        indices = [0, len(m_df)//2, len(m_df)-1]
        labels = ["worst", "median", "best"]
        
        for idx, label in zip(indices, labels):
            row = m_df.iloc[idx]
            case_id = row["case_id"]
            dice = row["dice"]
            
            try:
                # Load PET and masks (ensure absolute paths)
                pet_arr, _ = load_nifti_array(images_dir / f"{case_id}_0001.nii.gz")
                gt_arr, _ = load_nifti_array(labels_dir / f"{case_id}.nii.gz")
                pred_arr, _ = load_nifti_array(predictions_dir / f"{case_id}.nii.gz")
                
                # Find best slice
                z_idx = get_axial_slice_with_most_lesion(gt_arr)
                
                pet_slice = pet_arr[:, :, z_idx]
                gt_slice = (gt_arr[:, :, z_idx] > 0.5).astype(np.uint8)
                pred_slice = (pred_arr[:, :, z_idx] > 0.5).astype(np.uint8)
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Panel 1: PET only
                axes[0].imshow(pet_slice.T, cmap="hot", origin="lower")
                axes[0].set_title(f"PET ({case_id})")
                axes[0].axis("off")
                
                # Panel 2: PET + GT (Green)
                make_overlay_panel(axes[1], pet_slice, gt_slice, color="green")
                axes[1].set_title(f"GT Overlay (Dice={dice:.2f})")
                
                # Panel 3: PET + Pred (Red)
                make_overlay_panel(axes[2], pet_slice, pred_slice, color="red")
                axes[2].set_title("Pred Overlay")
                
                fname = f"overlay_{modality}_{label}_{case_id}.png"
                save_figure(fig, figures_dir / fname)
                generated_files.append(fname)
                logger.info(f"Generated overlay: {fname}")
                
            except Exception as e:
                logger.error(f"Error generating overlay for {case_id}: {e}")
                
    return generated_files


def main() -> None:
    """Main function for step 08."""
    # Define project root
    project_root = Path(__file__).resolve().parent.parent

    # Load configuration
    paths = load_yaml(project_root / "configs" / "paths.yaml")

    # Set up output directories (make absolute relative to project root)
    logs_dir = project_root / paths["logs_dir"]
    figures_dir = project_root / paths["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logger = setup_logger("step08_visualise", logs_dir / "step08_visualise.log")
    logger.info("Starting Step 08: Visualisation")

    # Load data (make absolute relative to project root)
    master_df = pd.read_csv(project_root / paths["manifests_dir"] / "master_manifest.csv")
    subset_combined = pd.read_csv(project_root / paths["manifests_dir"] / "subset_combined.csv")
    per_case_metrics = pd.read_csv(project_root / paths["metrics_dir"] / "per_case_metrics.csv")

    # 1. Dataset stats
    plot_dataset_stats(master_df, subset_combined, figures_dir, logger)

    # 2. Metric plots
    plot_metrics(per_case_metrics, figures_dir, logger)

    # 3. Overlays
    overlay_files = generate_overlays(per_case_metrics, paths, project_root, figures_dir, logger)

    # Summary
    summary = {
        "figures_generated": [f.name for f in figures_dir.glob("*.png")],
        "overlay_count": len(overlay_files)
    }
    save_json(summary, logs_dir / "step08_summary.json")
    logger.info(f"Step 08 completed. {len(summary['figures_generated'])} figures generated.")


if __name__ == "__main__":
    main()
