#!/usr/bin/env python3
"""
Step 01: Inventory scanning for PET/CT lesion segmentation pipeline.
Scans imagesTr and labelsTr; verifies CT + PET + label triplets exist;
records shape and spacing per case; writes inventory.csv and summary.json.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Any

import pandas as pd

# Add the project root to sys.path to allow imports from src/utils
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.io_utils import load_yaml, save_json, setup_logger
from src.utils.nifti_utils import load_nifti_header


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def main() -> None:
    """Main function for step 01 inventory."""
    # Add the project root to sys.path to allow imports from src/utils
    # (Already done at module level, but we use project_root for paths here)
    project_root = Path(__file__).resolve().parent.parent

    # Load paths configuration
    config_path = project_root / "configs" / "paths.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    paths = load_yaml(config_path)

    # Set up output directories (make absolute relative to project root)
    output_root = project_root / paths["output_root"]
    logs_dir = project_root / paths["logs_dir"]
    manifests_dir = project_root / paths["manifests_dir"]

    logs_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logger = setup_logger("step01_inventory", logs_dir / "step01_inventory.log")
    logger.info("Starting Step 01: Data Inventory")

    # Define directories (make absolute relative to project root)
    images_dir = project_root / paths["images_dir"]
    labels_dir = project_root / paths["labels_dir"]

    # Verify input directories
    if not images_dir.is_dir():
        logger.error(f"Images directory not found: {images_dir}")
        sys.exit(1)
    if not labels_dir.is_dir():
        logger.error(f"Labels directory not found: {labels_dir}")
        sys.exit(1)

    # Scan for unique case_ids
    # imagesTr contains <case_id>_0000.nii.gz (CT) and <case_id>_0001.nii.gz (PET)
    case_ids: Set[str] = set()
    for file_path in images_dir.glob("*.nii.gz"):
        name = file_path.name
        if "_0000.nii.gz" in name:
            case_ids.add(name.replace("_0000.nii.gz", ""))
        elif "_0001.nii.gz" in name:
            case_ids.add(name.replace("_0001.nii.gz", ""))

    logger.info(f"Found {len(case_ids)} unique candidate case IDs in {images_dir}")

    inventory_rows: List[Dict[str, Any]] = []
    incomplete_cases: List[str] = []

    for case_id in sorted(list(case_ids)):
        ct_file = images_dir / f"{case_id}_0000.nii.gz"
        pet_file = images_dir / f"{case_id}_0001.nii.gz"
        label_file = labels_dir / f"{case_id}.nii.gz"

        has_ct = ct_file.exists()
        has_pet = pet_file.exists()
        has_label = label_file.exists()

        ct_size = get_file_size_mb(ct_file)
        pet_size = get_file_size_mb(pet_file)
        label_size = get_file_size_mb(label_file)

        complete = has_ct and has_pet and has_label

        row = {
            "case_id": case_id,
            "has_ct": has_ct,
            "has_pet": has_pet,
            "has_label": has_label,
            "ct_size_mb": round(ct_size, 2),
            "pet_size_mb": round(pet_size, 2),
            "label_size_mb": round(label_size, 2),
            "complete": complete,
        }

        # Header info (using CT as primary reference for shape/spacing if available)
        # Requirement: load header only, no data
        ref_file = ct_file if has_ct else (pet_file if has_pet else None)
        shape_x, shape_y, shape_z = None, None, None
        spacing_x, spacing_y, spacing_z = None, None, None

        if ref_file:
            try:
                header_info = load_nifti_header(ref_file)
                shape = header_info["shape"]
                spacing = header_info["spacing"]

                if len(shape) >= 3:
                    shape_x, shape_y, shape_z = shape[:3]
                if len(spacing) >= 3:
                    spacing_x, spacing_y, spacing_z = spacing[:3]
            except Exception as e:
                logger.error(f"Error reading header for {case_id}: {e}")
                complete = False
                row["complete"] = False

        row.update({
            "shape_x": shape_x,
            "shape_y": shape_y,
            "shape_z": shape_z,
            "spacing_x": spacing_x,
            "spacing_y": spacing_y,
            "spacing_z": spacing_z,
        })

        inventory_rows.append(row)

        if not complete:
            incomplete_cases.append(case_id)
            logger.warning(f"Case {case_id} is incomplete: CT={has_ct}, PET={has_pet}, Label={has_label}")

    # Create DataFrame and save CSV
    inventory_df = pd.DataFrame(inventory_rows)
    # Reorder columns to match guidelines
    cols = [
        "case_id", "has_ct", "has_pet", "has_label",
        "ct_size_mb", "pet_size_mb", "label_size_mb",
        "shape_x", "shape_y", "shape_z",
        "spacing_x", "spacing_y", "spacing_z",
        "complete"
    ]
    inventory_df = inventory_df[cols]
    inventory_csv_path = manifests_dir / "inventory.csv"
    inventory_df.to_csv(inventory_csv_path, index=False)
    logger.info(f"Inventory saved to {inventory_csv_path}")

    # Create summary JSON
    summary = {
        "total_cases": len(case_ids),
        "complete_count": len(case_ids) - len(incomplete_cases),
        "incomplete_case_list": incomplete_cases
    }
    summary_path = logs_dir / "step01_summary.json"
    save_json(summary, summary_path)
    logger.info(f"Step 01 summary saved to {summary_path}")
    logger.info(f"Summary: {summary['complete_count']} complete, {len(incomplete_cases)} incomplete.")


if __name__ == "__main__":
    main()
