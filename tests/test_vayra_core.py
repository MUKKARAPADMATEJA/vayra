import pytest
import engine
import server

def test_rainfall_input_validation():
    with pytest.raises(ValueError):
        server.sim_of({"sim": -10})
        
def test_flood_risk_calculation():
    res = engine.calculate_flood_risk(17.3850, 78.4867, sim=50)
    assert "score" in res
    assert "classification" in res
    assert res["score"] > 0
    
def test_risk_classification():
    assert engine.classify_risk(10) == "LOW"
    assert engine.classify_risk(20) == "MODERATE"
    assert engine.classify_risk(40) == "HIGH"
    assert engine.classify_risk(60) == "CRITICAL"
    
def test_road_segment_analysis():
    route_points = [(17.3, 78.4), (17.31, 78.41)]
    elevs = [500, 500]
    ar = [1000, 1000]
    wh = [[0, 0, 0, 50]] * 2
    segments = engine.identify_risky_segments(route_points, elevs, ar, wh, 3600, 0.8, sim=50)
    assert len(segments) == 2
    
def test_risky_segment_detection():
    route_points = [(17.3, 78.4), (17.31, 78.41)]
    elevs = [500, 500]
    ar = [1000, 1000]
    wh = [[0, 0, 0, 50]] * 2
    analysis = engine.analyze_route(route_points, elevs, ar, wh, 3600, 0.8, sim=50)
    assert not analysis["is_safe"]
    assert len(analysis["risky_segments"]) > 0

def test_early_warning_generation():
    safe_analysis = {"is_safe": True, "classification": "LOW", "risky_segments": []}
    warn1 = engine.generate_early_warning(safe_analysis)
    assert "safe" in warn1.lower()
    
    danger_analysis = {"is_safe": False, "classification": "CRITICAL", "risky_segments": [1, 2, 3]}
    warn2 = engine.generate_early_warning(danger_analysis)
    assert "CRITICAL" in warn2
    assert "3" in warn2

def test_alternative_route_evaluation():
    routes = [
        {"points": [[17.3, 78.4], [17.31, 78.41]], "duration": 3600},
        {"points": [[17.4, 78.5], [17.41, 78.51]], "duration": 1800}
    ]
    eval_res = engine.evaluate_alternative_routes(routes, sim=10)
    assert "recommended_route_index" in eval_res
    assert len(eval_res["routes"]) == 2

def test_invalid_input_coordinates():
    with pytest.raises(ValueError):
        server.parse_routes({"routes": [{"points": [[1000, -2000]], "duration": 100}]})

def test_missing_required_inputs():
    with pytest.raises(ValueError):
        server.parse_routes({"routes": []})
