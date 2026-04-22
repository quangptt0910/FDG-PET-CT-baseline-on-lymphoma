#!/usr/bin/env python3
"""
Step 03: Subset selection.
Reproducibly sample 80 train and 20 val cases (50% PSMA, 50% FDG).
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

# Add the project root to sys.path to allow imports from src/utils
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.io_utils import load_yaml, save_json, setup_logger


def main() -> None:
    """Main function for step 03 subset."""
    # Define project root
    project_root = Path(__file__).resolve().parent.parent

    # Load paths configuration
    paths = load_yaml(project_root / "configs" / "paths.yaml")
    subset_cfg = load_yaml(project_root / "configs" / "subset.yaml")

    # Set up output directories (make absolute relative to project root)
    logs_dir = project_root / paths["logs_dir"]
    manifests_dir = project_root / paths["manifests_dir"]

    logs_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logger = setup_logger("step03_subset", logs_dir / "step03_subset.log")
    logger.info("Starting Step 03: Subset Selection")

    # Load master manifest
    master_path = manifests_dir / "master_manifest.csv"
    if not master_path.exists():
        logger.error(f"Master manifest not found at {master_path}. Run step 02 first.")
        sys.exit(1)
    
    master_df = pd.read_csv(master_path)
    logger.info(f"Loaded master manifest with {len(master_df)} cases.")

    # Parameters
    train_size = subset_cfg["train_size"]
    val_size = subset_cfg["val_size"]
    diagnosis_filter = subset_cfg.get("diagnosis_filter")
    seed = subset_cfg["random_seed"]

    # Apply diagnosis filter if present
    if diagnosis_filter:
        master_df = master_df[master_df["diagnosis"] == diagnosis_filter]
        logger.info(f"Filtered manifest to {diagnosis_filter} only: {len(master_df)} cases.")

    # 1. Sample TRAIN subset
    train_pool = master_df[master_df["split"] == "train"]
    logger.info(f"Train pool size: {len(train_pool)}")
    
    # If filtering by diagnosis, we don't force PSMA/FDG fraction (Lymphoma is usually all FDG)
    if diagnosis_filter:
        subset_train = train_pool.sample(n=min(train_size, len(train_pool)), random_state=seed)
    else:
        psma_frac = subset_cfg["psma_fraction"]
        fdg_frac = subset_cfg["fdg_fraction"]
        psma_train_size = int(train_size * psma_frac)
        fdg_train_size = int(train_size * fdg_frac)
        psma_train = train_pool[train_pool["modality"] == "PSMA"].sample(n=psma_train_size, random_state=seed)
        fdg_train = train_pool[train_pool["modality"] == "FDG"].sample(n=fdg_train_size, random_state=seed)
        subset_train = pd.concat([psma_train, fdg_train])

    subset_train = subset_train.sort_values("case_id")
    logger.info(f"Selected train subset: {len(subset_train)} cases.")

    # 2. Sample VAL subset
    val_pool = master_df[master_df["split"] == "val"]
    logger.info(f"Val pool size: {len(val_pool)}")
    
    if diagnosis_filter:
        subset_val = val_pool.sample(n=min(val_size, len(val_pool)), random_state=seed)
    else:
        psma_val_size = int(val_size * psma_frac)
        fdg_val_size = int(val_size * fdg_frac)
        psma_val = val_pool[val_pool["modality"] == "PSMA"].sample(n=psma_val_size, random_state=seed)
        fdg_val = val_pool[val_pool["modality"] == "FDG"].sample(n=fdg_val_size, random_state=seed)
        subset_val = pd.concat([psma_val, fdg_val])

    subset_val = subset_val.sort_values("case_id")
    logger.info(f"Selected val subset: {len(subset_val)} cases.")

    # 3. Verification: No patient-level overlap
    train_patients = set(subset_train["subject_id"].unique())
    val_patients = set(subset_val["subject_id"].unique())
    overlap = train_patients.intersection(val_patients)
    
    if overlap:
        logger.warning(f"Patient-level overlap detected between train and val subsets: {overlap}")
        # Note: In a real scenario, we'd reshuffle or ensure the split was patient-level.
        # But we're using the official split which should already be patient-level.
    else:
        logger.info("No patient-level overlap detected between train and val subsets.")

    # 4. Save CSVs
    train_csv_path = manifests_dir / "subset_train.csv"
    val_csv_path = manifests_dir / "subset_val.csv"
    combined_csv_path = manifests_dir / "subset_combined.csv"

    subset_train.to_csv(train_csv_path, index=False)
    subset_val.to_csv(val_csv_path, index=False)

    # Combined with role
    subset_train_role = subset_train.copy()
    subset_train_role["role"] = "train"
    subset_val_role = subset_val.copy()
    subset_val_role["role"] = "val"
    subset_combined = pd.concat([subset_train_role, subset_val_role]).sort_values("case_id")
    subset_combined.to_csv(combined_csv_path, index=False)

    logger.info(f"CSVs saved to {manifests_dir}")

    # 5. Summary
    summary = {
        "train": {
            "total": len(subset_train),
            "modalities": subset_train["modality"].value_counts().to_dict(),
            "manufacturers": subset_train["manufacturer"].value_counts().to_dict(),
            "tracers": subset_train["pet_tracer"].value_counts().to_dict()
        },
        "val": {
            "total": len(subset_val),
            "modalities": subset_val["modality"].value_counts().to_dict(),
            "manufacturers": subset_val["manufacturer"].value_counts().to_dict(),
            "tracers": subset_val["pet_tracer"].value_counts().to_dict()
        },
        "patient_overlap": list(overlap)
    }
    summary_path = logs_dir / "step03_summary.json"
    save_json(summary, summary_path)
    logger.info(f"Step 03 summary saved to {summary_path}")


if __name__ == "__main__":
    main()
