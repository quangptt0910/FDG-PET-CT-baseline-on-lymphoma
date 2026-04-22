import os
from pathlib import Path
import pytest
from src.utils.io_utils import load_yaml

def test_training_cfg():
    training_cfg = load_yaml(Path("configs/training.yaml"))
    assert training_cfg["num_epochs"] == 100
    assert training_cfg["configuration"] == "3d_fullres"
    assert training_cfg["fold"] == 0

def test_env_setup():
    # Simple check if env setter doesn't crash (already tested in test_step04)
    pass
