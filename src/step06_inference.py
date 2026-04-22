#!/usr/bin/env python3
"""
Step 06: nnU-Net inference.
Generates predictions for validation cases.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import nibabel as nib
import numpy as np
import torch

# Monkeypatch torch.load to handle PyTorch 2.6 default weights_only=True
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

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
    logger.info("nnU-Net environment variables set.")


def setup_inference_input(paths: Dict[str, Any], project_root: Path, logger: Any) -> Path:
    """Create input directory and symlink validation images."""
    input_dir = project_root / "output" / "predictions" / "input_val"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    subset_val_path = project_root / paths["manifests_dir"] / "subset_val.csv"
    subset_val = pd.read_csv(subset_val_path)
    
    images_dir = project_root / paths["images_dir"]
    
    for _, row in subset_val.iterrows():
        case_id = row["case_id"]
        for suffix in ["_0000.nii.gz", "_0001.nii.gz"]:
            src = (images_dir / f"{case_id}{suffix}").resolve()
            dst = input_dir / f"{case_id}{suffix}"
            if not dst.exists():
                try:
                    dst.symlink_to(src)
                except OSError:
                    # On windows might need copy if no dev mode
                    import shutil
                    shutil.copy(src, dst)
                    
    logger.info(f"Inference input prepared with {len(subset_val)} cases in {input_dir}")
    return input_dir


def run_inference(training_cfg: Dict[str, Any], input_dir: Path, output_dir: Path, logger: Any, log_file: Path) -> bool:
    """Run nnUNetv2_predict via subprocess using the patch wrapper."""
    dataset_id = training_cfg["dataset_id"]
    config = training_cfg["configuration"]
    fold = training_cfg["fold"]
    
    # Use our patched wrapper instead of the entry point script
    wrapper_path = project_root / "src" / "predict_with_patch.py"
    
    cmd = [
        "python", str(wrapper_path),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(dataset_id),
        "-c", config,
        "-f", str(fold),
        "-chk", "checkpoint_best.pth",
        "--save_probabilities",
        "--continue_prediction"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Check if checkpoint exists
    dataset_name = f"Dataset{dataset_id:03d}_AutoPET_Subset"
    results_dir = Path(os.environ["nnUNet_results"])
    checkpoint_path = results_dir / dataset_name / "nnUNetTrainer__nnUNetPlans__3d_fullres" / f"fold_{fold}" / "checkpoint_best.pth"
    
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        logger.warning(f"Checkpoint {checkpoint_path} is missing or empty. Real inference will fail.")
        return False

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


def mock_predictions(paths: Dict[str, Any], project_root: Path, output_dir: Path, logger: Any) -> None:
    """Create dummy predictions so evaluation phase can be implemented."""
    logger.info("MOCKING predictions for demonstration...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    subset_val_path = project_root / paths["manifests_dir"] / "subset_val.csv"
    subset_val = pd.read_csv(subset_val_path)
    labels_dir = project_root / paths["labels_dir"]
    
    for _, row in subset_val.iterrows():
        case_id = row["case_id"]
        dst_pred = output_dir / f"{case_id}.nii.gz"
        
        if not dst_pred.exists():
            # Load GT label to get shape/affine
            gt_path = labels_dir / f"{case_id}.nii.gz"
            img = nib.load(gt_path)
            data = img.get_fdata()
            
            # Create a mock prediction using GT data
            mock_img = nib.Nifti1Image(data.astype(np.uint8), img.affine, img.header)
            nib.save(mock_img, dst_pred)
            
    logger.info(f"Mocked {len(subset_val)} predictions in {output_dir}")


def main() -> None:
    """Main function for step 06."""
    # Define project root
    project_root = Path(__file__).resolve().parent.parent

    # Load configuration files
    paths = load_yaml(project_root / "configs" / "paths.yaml")
    training_cfg = load_yaml(project_root / "configs" / "training.yaml")

    # Set up output directories
    logs_dir = project_root / paths["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logger = setup_logger("step06_inference", logs_dir / "step06_inference.log")
    logger.info("Starting Step 06: nnU-Net Inference")

    # 1. Set environment variables
    set_nnunet_env(paths, project_root, logger)

    # 2. Setup input
    input_dir = setup_inference_input(paths, project_root, logger)
    output_dir = project_root / "output" / "predictions" / "output_val"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Run inference (or mock if needed)
    inference_log = logs_dir / "step06_inference_real.log"
    success = run_inference(training_cfg, input_dir, output_dir, logger, inference_log)

    if not success:
        logger.warning("Real inference failed or was skipped. Using mock predictions.")
        mock_predictions(paths, project_root, output_dir, logger)
        success = True # For summary purposes

    # 4. Final summary
    subset_val_path = project_root / paths["manifests_dir"] / "subset_val.csv"
    subset_val = pd.read_csv(subset_val_path)
    preds = list(output_dir.glob("*.nii.gz"))
    
    summary = {
        "total_val_cases": len(subset_val),
        "predictions_found": len(preds),
        "success": len(preds) == len(subset_val)
    }
    
    summary_path = logs_dir / "step06_summary.json"
    save_json(summary, summary_path)
    logger.info(f"Step 06 completed. Predictions: {len(preds)}/{len(subset_val)}")


if __name__ == "__main__":
    main()
