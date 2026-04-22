#!/usr/bin/env python3
"""
Step 02: Master manifest creation.
Joins inventory with metadata (FDG/PSMA) and splits to create master_manifest.csv.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np

# Add the project root to sys.path to allow imports from src/utils
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.io_utils import load_yaml, save_json, setup_logger


def main() -> None:
    """Main function for step 02 manifest."""
    # Define project root
    project_root = Path(__file__).resolve().parent.parent

    # Load paths configuration
    config_path = project_root / "configs" / "paths.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    paths = load_yaml(config_path)

    # Set up output directories (make absolute relative to project root)
    logs_dir = project_root / paths["logs_dir"]
    manifests_dir = project_root / paths["manifests_dir"]

    logs_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logger = setup_logger("step02_manifest", logs_dir / "step02_manifest.log")
    logger.info("Starting Step 02: Master Manifest Creation")

    # 1. Load inventory
    inventory_path = manifests_dir / "inventory.csv"
    if not inventory_path.exists():
        logger.error(f"Inventory not found at {inventory_path}. Run step 01 first.")
        sys.exit(1)
    
    inventory_df = pd.read_csv(inventory_path)
    logger.info(f"Loaded inventory with {len(inventory_df)} cases.")

    # 2. Load Metadata (make absolute relative to project root)
    fdg_meta_path = project_root / paths["fdg_meta"]
    psma_meta_path = project_root / paths["psma_meta"]

    # Load FDG metadata
    fdg_meta = pd.read_csv(fdg_meta_path)
    logger.info(f"Loaded FDG metadata with {len(fdg_meta)} rows. Columns: {fdg_meta.columns.tolist()}")
    
    # Load PSMA metadata
    psma_meta = pd.read_csv(psma_meta_path)
    logger.info(f"Loaded PSMA metadata with {len(psma_meta)} rows. Columns: {psma_meta.columns.tolist()}")

    # 3. Process FDG Metadata
    # FDG metadata has multiple rows per study (CT, PET, SEG). 
    # We want one row per Study UID.
    # Group by Study UID and keep first row for common metadata (Subject ID, age, sex, diagnosis)
    fdg_meta_grouped = fdg_meta.groupby("Study UID").first().reset_index()
    fdg_meta_grouped["modality"] = "FDG"
    
    # Extract study suffix (last 5 digits) for matching
    fdg_meta_grouped["study_suffix"] = fdg_meta_grouped["Study UID"].astype(str).str[-5:]
    # Extract subject short (remove PETCT_)
    fdg_meta_grouped["subject_short"] = fdg_meta_grouped["Subject ID"].str.replace("PETCT_", "")
    
    # 4. Process PSMA Metadata
    psma_meta["modality"] = "PSMA"
    # Subject ID in PSMA meta is PSMA_...
    psma_meta["subject_short"] = psma_meta["Subject ID"].str.replace("PSMA_", "")
    # Standardize column names to match FDG if possible, or just keep them
    # PSMA columns: Subject ID, Study Date, age, manufacturer_model_name, pet_radionuclide, ct_contrast_agent
    # FDG columns: Subject ID, Study Date, age, sex, diagnosis, Manufacturer, ...
    
    # We'll use a mapping to join them
    # For PSMA, case_id is psma_<subject_short>_<study_date>
    psma_meta["join_key"] = "psma_" + psma_meta["subject_short"] + "_" + psma_meta["Study Date"]

    # 5. Load splits (make absolute relative to project root)
    splits_path = project_root / paths["splits_json"]
    with open(splits_path, "r") as f:
        splits_data = json.load(f)
    
    # Fold 0 is usually the first element in the list
    fold0 = splits_data[0]
    train_ids = set(fold0["train"])
    val_ids = set(fold0["val"])
    
    logger.info(f"Fold 0 splits: {len(train_ids)} train, {len(val_ids)} val cases.")

    # 6. Build Master Manifest
    master_rows = []
    missing_metadata_count = 0

    for _, inv_row in inventory_df.iterrows():
        case_id = inv_row["case_id"]
        modality = "PSMA" if case_id.startswith("psma") else "FDG"
        
        # Determine split
        split = "test"
        if case_id in train_ids:
            split = "train"
        elif case_id in val_ids:
            split = "val"
        
        meta_row = {}
        if modality == "PSMA":
            # Direct match using join_key
            match = psma_meta[psma_meta["join_key"] == case_id]
            if not match.empty:
                row = match.iloc[0]
                meta_row = {
                    "subject_id": row["Subject ID"],
                    "study_date": row["Study Date"],
                    "age": row["age"],
                    "sex": "M",  # PSMA dataset is all male (prostate cancer)
                    "manufacturer": row["manufacturer_model_name"],
                    "modality": "PSMA",
                    "diagnosis": "Prostate Cancer",
                    "pet_tracer": row["pet_radionuclide"],
                    "ct_contrast": row["ct_contrast_agent"]
                }
            else:
                missing_metadata_count += 1
                logger.warning(f"Metadata missing for PSMA case: {case_id}")
        else:
            # FDG case
            # case_id format: fdg_<subject_short>_<study_date>-NA-<desc>-<suffix>
            parts = case_id.split("-")
            suffix = parts[-1]
            
            match = fdg_meta_grouped[fdg_meta_grouped["study_suffix"] == suffix]
            if not match.empty:
                # If multiple matches by suffix (unlikely but possible), match by subject_short
                if len(match) > 1:
                    # Extract subject_short from case_id
                    # fdg_0011f3deaf_...
                    subj_short = case_id.split("_")[1]
                    match = match[match["subject_short"] == subj_short]
                
                if not match.empty:
                    row = match.iloc[0]
                    meta_row = {
                        "subject_id": row["Subject ID"],
                        "study_date": row["Study Date"],
                        "age": row["age"],
                        "sex": row["sex"],
                        "manufacturer": row["Manufacturer"],
                        "modality": "FDG",
                        "diagnosis": row["diagnosis"],
                        "pet_tracer": "18F-FDG",
                        "ct_contrast": "unknown" # Need to check Study Description maybe
                    }
                    # Attempt to refine ct_contrast from Study Description
                    study_desc = str(row["Study Description"]).lower()
                    if "mit km" in study_desc or "iv contrast" in study_desc:
                        meta_row["ct_contrast"] = "yes"
                    elif "ohne km" in study_desc or "nativ" in study_desc:
                        meta_row["ct_contrast"] = "no"
                else:
                    missing_metadata_count += 1
                    logger.warning(f"Metadata missing for FDG case (subj mismatch): {case_id}")
            else:
                missing_metadata_count += 1
                logger.warning(f"Metadata missing for FDG case (suffix mismatch): {case_id}")

        # Combine inventory info with metadata and split
        full_row = inv_row.to_dict()
        full_row.update(meta_row)
        full_row["split"] = split
        master_rows.append(full_row)

    master_df = pd.DataFrame(master_rows)

    # 7. Filtering and Validation
    logger.info(f"Filtering to complete cases only.")
    initial_count = len(master_df)
    master_df = master_df[master_df["complete"] == True]
    filtered_count = len(master_df)
    logger.info(f"Kept {filtered_count} / {initial_count} complete cases.")

    # Validation
    if master_df["case_id"].duplicated().any():
        logger.warning("Duplicate case_ids found in master manifest!")
    
    null_modality = master_df["modality"].isnull().sum()
    if null_modality > 0:
        logger.warning(f"Found {null_modality} cases with null modality.")
    
    null_split = master_df["split"].isnull().sum()
    if null_split > 0:
        logger.warning(f"Found {null_split} cases with null split.")

    # 8. Save results
    master_csv_path = manifests_dir / "master_manifest.csv"
    master_df.to_csv(master_csv_path, index=False)
    logger.info(f"Master manifest saved to {master_csv_path}")

    # Summary
    summary = {
        "total_cases": len(master_df),
        "modality_counts": master_df["modality"].value_counts().to_dict(),
        "split_counts": master_df["split"].value_counts().to_dict(),
        "manufacturer_counts": master_df["manufacturer"].value_counts().to_dict(),
        "cases_missing_metadata": missing_metadata_count
    }
    summary_path = logs_dir / "step02_summary.json"
    save_json(summary, summary_path)
    logger.info(f"Step 02 summary saved to {summary_path}")
    logger.info(f"Final manifest: {summary['total_cases']} cases. Missing meta: {missing_metadata_count}")


if __name__ == "__main__":
    main()
