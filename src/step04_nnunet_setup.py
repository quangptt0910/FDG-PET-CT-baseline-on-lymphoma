#!/usr/bin/env python3
"""
Step 04: nnU-Net setup and preprocessing.
Builds the nnU-Net raw dataset directory with symlinks, sets environment variables,
and runs planning and preprocessing.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

# Add the project root to sys.path to allow imports from src/utils
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.io_utils import load_yaml, save_json, setup_logger


def set_nnunet_env(paths: Dict[str, Any], project_root: Path, logger: Any) -> None:
    """Set nnU-Net environment variables in the current process."""
    os.environ["nnUNet_raw"] = str((project_root / paths["nnunet_raw"]).resolve())
    os.environ["nnUNet_preprocessed"] = str((project_root / paths["nnunet_preprocessed"]).resolve())
    os.environ["nnUNet_results"] = str((project_root / paths["nnunet_results"]).resolve())
    os.environ["nnUNet_preprocessed_no_blosc"] = "True"
    
    logger.info("nnU-Net environment variables set:")
    logger.info(f"  nnUNet_raw: {os.environ['nnUNet_raw']}")
    logger.info(f"  nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
    logger.info(f"  nnUNet_results: {os.environ['nnUNet_results']}")
    logger.info(f"  nnUNet_preprocessed_no_blosc: {os.environ['nnUNet_preprocessed_no_blosc']}")


def setup_nnunet_raw(paths: Dict[str, Any], project_root: Path, training_cfg: Dict[str, Any], logger: Any) -> None:
    """Build the nnU-Net raw dataset directory structure and symlinks."""
    dataset_id = training_cfg["dataset_id"]
    dataset_name = f"Dataset{dataset_id:03d}_AutoPET_Subset"
    
    raw_root = project_root / paths["nnunet_raw"]
    dataset_dir = raw_root / dataset_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    
    # Create directories
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    
    # Load subset_train
    manifests_dir = project_root / paths["manifests_dir"]
    subset_train_path = manifests_dir / "subset_train.csv"
    if not subset_train_path.exists():
        logger.error(f"Subset train manifest not found at {subset_train_path}. Run step 03 first.")
        sys.exit(1)
        
    subset_train = pd.read_csv(subset_train_path)
    logger.info(f"Loaded training subset with {len(subset_train)} cases.")
    
    # Symlink files
    data_images_dir = project_root / paths["images_dir"]
    data_labels_dir = project_root / paths["labels_dir"]
    
    for _, row in subset_train.iterrows():
        case_id = row["case_id"]
        
        # Images: _0000 (CT) and _0001 (PET)
        src_ct = (data_images_dir / f"{case_id}_0000.nii.gz").resolve()
        src_pet = (data_images_dir / f"{case_id}_0001.nii.gz").resolve()
        src_label = (data_labels_dir / f"{case_id}.nii.gz").resolve()
        
        # Destinations
        dst_ct = images_tr / f"{case_id}_0000.nii.gz"
        dst_pet = images_tr / f"{case_id}_0001.nii.gz"
        dst_label = labels_tr / f"{case_id}.nii.gz"
        
        # Create hard links
        for src, dst in [(src_ct, dst_ct), (src_pet, dst_pet), (src_label, dst_label)]:
            # If dst exists but is 0 bytes or wrong, remove it
            if dst.exists() and dst.stat().st_size == 0:
                dst.unlink()
            
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError as e:
                    logger.warning(f"Failed to create hard link {dst} -> {src}: {e}")
                    # Fallback to copy if link fails
                    import shutil
                    try:
                        shutil.copy2(src, dst)
                        logger.info(f"Copied {src} to {dst} as fallback.")
                    except Exception as e2:
                        logger.error(f"Failed to copy {src} to {dst}: {e2}")
    
    # Copy dataset.json and update numTraining
    src_dataset_json = project_root / paths["dataset_json"]
    dst_dataset_json = dataset_dir / "dataset.json"
    
    if src_dataset_json.exists():
        with open(src_dataset_json, "r") as f:
            ds_info = json.load(f)
        
        ds_info["numTraining"] = len(subset_train)
        
        with open(dst_dataset_json, "w") as f:
            json.dump(ds_info, f, indent=2)
        logger.info(f"Updated dataset.json written to {dst_dataset_json}")
    else:
        logger.error(f"Source dataset.json not found at {src_dataset_json}")


def run_preprocessing(training_cfg: Dict[str, Any], logger: Any, log_file: Path) -> bool:
    """Run nnUNetv2_plan_and_preprocess via subprocess."""
    dataset_id = training_cfg["dataset_id"]
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "--verify_dataset_integrity",
        "-np", "2"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Open log file for streaming output
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ
        )
        
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
            
        process.wait()
        return process.returncode == 0


def main() -> None:
    """Main function for step 04."""
    # Define project root
    project_root = Path(__file__).resolve().parent.parent

    # Load paths configuration
    paths = load_yaml(project_root / "configs" / "paths.yaml")
    training_cfg = load_yaml(project_root / "configs" / "training.yaml")

    # Set up output directories (make absolute relative to project root)
    logs_dir = project_root / paths["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logger = setup_logger("step04_nnunet_setup", logs_dir / "step04_nnunet_setup.log")
    logger.info("Starting Step 04: nnU-Net Setup and Preprocessing")

    # 1. Set environment variables
    set_nnunet_env(paths, project_root, logger)

    # 2. Setup raw data
    setup_nnunet_raw(paths, project_root, training_cfg, logger)

    # 3. Run preprocessing
    planning_log = logs_dir / "step04_planning.log"
    success = run_preprocessing(training_cfg, logger, planning_log)

    # 4. Final summary
    dataset_id = training_cfg["dataset_id"]
    dataset_name = f"Dataset{dataset_id:03d}_AutoPET_Subset"
    preprocessed_dir = project_root / paths["nnunet_preprocessed"] / dataset_name
    
    summary = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "preprocessing_success": success,
        "preprocessed_dir_exists": preprocessed_dir.exists()
    }
    
    summary_path = logs_dir / "step04_summary.json"
    save_json(summary, summary_path)
    
    if success and preprocessed_dir.exists():
        logger.info("Step 04 completed successfully.")
    else:
        logger.error("Step 04 failed or preprocessed directory missing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
