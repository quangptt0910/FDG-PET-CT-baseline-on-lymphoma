import os
from pathlib import Path
import pytest
from src.step04_nnunet_setup import set_nnunet_env
from src.utils.io_utils import load_yaml

def test_set_nnunet_env():
    # Define project root for testing
    project_root = Path(__file__).resolve().parent.parent
    
    # Mock paths
    paths = {
        "nnunet_raw": "output/nnunet_raw",
        "nnunet_preprocessed": "output/nnunet_preprocessed",
        "nnunet_results": "output/nnunet_results"
    }
    
    class MockLogger:
        def info(self, msg):
            pass
            
    logger = MockLogger()
    set_nnunet_env(paths, project_root, logger)
    
    assert os.environ["nnUNet_raw"] == str((project_root / "output/nnunet_raw").resolve())
    assert os.environ["nnUNet_preprocessed"] == str((project_root / "output/nnunet_preprocessed").resolve())
    assert os.environ["nnUNet_results"] == str((project_root / "output/nnunet_results").resolve())

def test_load_configs():
    project_root = Path(__file__).resolve().parent.parent
    paths = load_yaml(project_root / "configs" / "paths.yaml")
    training_cfg = load_yaml(project_root / "configs" / "training.yaml")
    
    assert "nnunet_raw" in paths
    assert "dataset_id" in training_cfg
    assert training_cfg["dataset_id"] == 101
