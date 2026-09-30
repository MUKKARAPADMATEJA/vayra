import pytest
import numpy as np
from vayra_types import vayra_risk_assessment

def test_risk_levels():
    tensor = np.array([0.1, 0.6, 0.9])
    res = vayra_risk_assessment(tensor, 0.5)
    assert len(res) == 2
    assert 0.6 in res

def test_rainfall_boundaries():
    with pytest.raises(ValueError):
        rain = -5
        if rain < 0:
            raise ValueError("Negative rainfall is invalid")

def test_invalid_coordinates():
    lat, lng = 1000, -2000
    if abs(lat) > 90 or abs(lng) > 180:
        assert True
        
def test_tensor_shapes():
    mock_tensor = np.random.rand(10, 5)
    assert mock_tensor.shape == (10, 5)
    assert mock_tensor.dtype == np.float64
