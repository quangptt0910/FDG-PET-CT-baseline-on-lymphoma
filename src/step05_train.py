
#!/usr/bin/env python3
"""
Step 05: nnU-Net training.
Runs the training command for the specified number of epochs.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to sys.path to allow imports from src/utils
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.io_utils import load_yaml, save_json, setup_logger


def set_nnunet_env(paths: Dict[str, Any], project_root: Path, training_cfg: Dict[str, Any], logger: Any) -> None:
    """Set nnU-Net environment variables in the current process."""
    os.environ["nnUNet_raw"] = str((project_root / paths["nnunet_raw"]).resolve())
    os.environ["nnUNet_preprocessed"] = str((project_root / paths["nnunet_preprocessed"]).resolve())
    os.environ["nnUNet_results"] = str((project_root / paths["nnunet_results"]).resolve())
    os.environ["nnUNet_preprocessed_no_blosc"] = "True"
    os.environ["nnUNet_n_proc_DA"] = "0"
    os.environ["nnUNet_def_n_proc"] = "1"
    
    # Set number of epochs
    if "num_epochs" in training_cfg:
        os.environ["nnUNet_n_epochs"] = str(training_cfg["num_epochs"])
    
    logger.info("nnU-Net environment variables set.")


def run_training(training_cfg: Dict[str, Any], logger: Any, log_file: Path) -> bool:
    """Run nnUNetv2_train via subprocess."""
    dataset_id = training_cfg["dataset_id"]
    config = training_cfg["configuration"]
    fold = training_cfg["fold"]
    device = training_cfg.get("device", "cuda")
    
    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        config,
        str(fold),
        "--npz",
        "-device", device
    ]
    
    if training_cfg.get("continue_training", False):
        cmd.append("--c")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    start_time = time.time()
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
        duration = time.time() - start_time
        logger.info(f"Training process finished in {duration/3600:.2f} hours.")
        return process.returncode == 0


def main() -> None:
    """Main function for step 05."""
    # Define project root
    project_root = Path(__file__).resolve().parent.parent

    # Load paths configuration
    paths = load_yaml(project_root / "configs" / "paths.yaml")
    training_cfg = load_yaml(project_root / "configs" / "training.yaml")

    # Set up output directories (make absolute relative to project root)
    logs_dir = project_root / paths["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logger = setup_logger("step05_train", logs_dir / "step05_train.log")
    logger.info("Starting Step 05: nnU-Net Training")

    # 1. Set environment variables
    set_nnunet_env(paths, project_root, training_cfg, logger)

    # 2. Run training
    training_log = logs_dir / "step05_training.log"
    success = run_training(training_cfg, logger, training_log)

    # 3. Verify checkpoint
    dataset_id = training_cfg["dataset_id"]
    dataset_name = f"Dataset{dataset_id:03d}_AutoPET_Subset"
    results_dir = project_root / paths["nnunet_results"]
    # Path depends on trainer class and plans, defaults to nnUNetTrainer and nnUNetPlans
    checkpoint_path = results_dir / dataset_name / "nnUNetTrainer__nnUNetPlans__3d_fullres" / f"fold_{training_cfg['fold']}" / "checkpoint_best.pth"
    
    summary = {
        "training_success": success,
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else None
    }
    
    summary_path = logs_dir / "step05_summary.json"
    save_json(summary, summary_path)
    
    if success and checkpoint_path.exists():
        logger.info("Step 05 completed successfully.")
    else:
        logger.error(f"Step 05 failed or checkpoint missing at {checkpoint_path}")
        # Note: We don't exit(1) here if it's just a time-out or similar, 
        # but we follow the summary rule.
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
