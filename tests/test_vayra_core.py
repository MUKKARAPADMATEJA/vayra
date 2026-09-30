import pytest
import engine
import server

def test_risk_analysis_low_vuln():
    res = engine.vuln2(0, engine.v_model(0.1))
    assert res > 0

def test_risk_analysis_heavy_vuln():
    res = engine.vuln2(10, engine.v_model(0.9))
    assert res > 0.5

def test_predict_shapes():
    point = [17.3850, 78.4867, 120.5, 500.0, 2.5]
    prob = engine.predict(point)
    assert 0 <= prob <= 1

def test_server_num_validation():
    # Invalid coordinates
    with pytest.raises(ValueError):
        server.num(1000, -90, 90, "lat")
    with pytest.raises(ValueError):
        server.num("invalid", -90, 90, "lat")

def test_server_sim_validation():
    with pytest.raises(ValueError):
        server.sim_of({"sim": -10})

def test_parse_routes_invalid():
    with pytest.raises(ValueError):
        server.parse_routes({"routes": "not a list"})

def test_engine_level_mapping():
    assert engine.lvl(10) == 0  # SAFE
    assert engine.lvl(20) == 1  # WATCH
    assert engine.lvl(35) == 2  # WARNING
    assert engine.lvl(50) == 3  # DANGER

def test_invalid_input_coordinates():
    with pytest.raises(ValueError):
        server.parse_routes({"routes": [{"points": [[1000, -2000]], "duration": 100}]})

def test_missing_required_inputs():
    with pytest.raises(ValueError):
        server.parse_routes({"routes": []})

def test_route_duration_validation():
    with pytest.raises(ValueError):
        server.parse_routes({"routes": [{"points": [[10, 10], [11, 11]], "duration": -5}]})
