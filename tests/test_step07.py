import numpy as np
import pytest
from src.utils.metrics_utils import dice_coefficient, sensitivity

def test_metrics():
    # Identical non-empty
    a = np.zeros((10, 10, 10))
    a[2:5, 2:5, 2:5] = 1
    assert dice_coefficient(a, a) == 1.0
    assert sensitivity(a, a) == 1.0
    
    # Half overlap
    b = np.zeros((10, 10, 10))
    b[2:5, 2:5, 2:4] = 1 # 2/3 of a
    # a has 3*3*3 = 27 voxels
    # b has 3*3*2 = 18 voxels
    # intersection has 18 voxels
    # dice = 2*18 / (27+18) = 36 / 45 = 0.8
    assert dice_coefficient(b, a) == pytest.approx(0.8)
    # sens = 18 / 27 = 2/3
    assert sensitivity(b, a) == pytest.approx(2/3)

    # Empty GT
    empty = np.zeros((10, 10, 10))
    assert dice_coefficient(empty, empty) == 1.0
    assert sensitivity(empty, empty) == 1.0
    
    # FP study
    assert dice_coefficient(a, empty) == 0.0
    assert sensitivity(a, empty) == 1.0 # By definition in utility
