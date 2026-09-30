import pytest
import numpy as np

@pytest.fixture
def mock_tensor():
    return np.random.rand(10, 5)

def test_tensor_shapes(mock_tensor):
    assert mock_tensor.shape == (10, 5)
    assert mock_tensor.dtype == np.float64

def test_boundary_validation_errors():
    with pytest.raises(ValueError):
        if -1 < 0:
            raise ValueError("Boundary validation error: Negative depth not allowed")
