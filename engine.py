"""VAYRA risk engine - standard library only."""
import json, math, os, time, urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = Path(__file__).resolve().parent
MODEL = json.loads((ROOT / "flood_model.json").read_text())
FORECAST = os.getenv("OPENMETEO_FORECAST", "https://api.open-meteo.com/v1/forecast")
ELEVATION = os.getenv("OPENMETEO_ELEVATION", "https://api.open-meteo.com/v1/elevation")
ARCHIVE = os.getenv("OPENMETEO_ARCHIVE", "https://archive-api.open-meteo.com/v1/archive")
RISK_CLASSIFICATIONS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
PAST_HOURS, FORECAST_HOURS = 12, 7

clamp = lambda x, a, b: min(b, max(a, x))
rnd = lambda x: math.floor(x + 0.5)

def calculate_travel_distance(point_a: list, point_b: list) -> float:
    r = math.pi / 180
    dl, dg = (point_b[0] - point_a[0]) * r, (point_b[1] - point_a[1]) * r
    x = math.sin(dl / 2) ** 2 + math.cos(point_a[0] * r) * math.cos(point_b[0] * r) * math.sin(dg / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(x))

# ---------- ML model ----------
def predict_flood_prone_area(features: list) -> float:
    """Predict flood probability using tree-based ML model. Complexity: O(Trees * Depth)"""
    s = MODEL["init"]
    for t in MODEL["trees"]:
        n = 0
        while t["l"][n] != -1:
            n = t["l"][n] if features[t["f"][n]] <= t["t"][n] else t["r"][n]
        s += MODEL["lr"] * t["v"][n]
    return 1 / (1 + math.exp(-s))

def calculate_terrain_vulnerability(elevation_diff: float) -> float: return clamp(.45 - elevation_diff / 10, .1, 1)
def calculate_ml_vulnerability(probability: float) -> float: return clamp(.5 + (probability - .75) * 2.5, 0, 1)
def combine_vulnerability(terrain_vuln: float, ml_vuln: float) -> float: return .7 * terrain_vuln + .3 * ml_vuln

def classify_risk_level(risk_score: float) -> str:
    """Risk Classification mapping to standard severity."""
    if risk_score < 12: return "LOW"
    elif risk_score < 28: return "MODERATE"
    elif risk_score < 45: return "HIGH"
    return "CRITICAL"

def fetch_json(url: str, tries: int = 2, timeout: int = 8) -> dict:
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:
            last = e
    raise last

_elev, _rain, _wx = {}, {}, {}
ek = lambda p: f"{p[0]:.3f},{p[1]:.3f}"
rk = lambda p: f"{round(p[0] * 4) / 4:.2f},{round(p[1] * 4) / 4:.2f}"
csv = lambda ps, i: ",".join(f"{p[i]:.4f}" for p in ps)

def elevations(points):
    need = list({ek(p): p for p in points if ek(p) not in _elev}.values())
    for i in range(0, len(need), 100):
        c = need[i:i + 100]
        d = fetch_json(f"{ELEVATION}?latitude={csv(c, 0)}&longitude={csv(c, 1)}")
        for p, v in zip(c, d["elevation"]): _elev[ek(p)] = v
    return [_elev[ek(p)] for p in points]

def annual_rain(points):
    need = list({rk(p): p for p in points if rk(p) not in _rain}.values())
    for i in range(0, len(need), 10):
        c = need[i:i + 10]
        try:
            d = fetch_json(f"{ARCHIVE}?latitude={csv(c, 0)}&longitude={csv(c, 1)}&start_date=2022-01-01&end_date=2023-12-31&daily=precipitation_sum&timezone=UTC")
            for p, x in zip(c, d if isinstance(d, list) else [d]):
                _rain[rk(p)] = round(sum(v or 0 for v in x["daily"]["precipitation_sum"]) / 2)
        except Exception:
            pass
    return [_rain.get(rk(p), 1000) for p in points]

def weather(points):
    keys = [f"{p[0]:.2f},{p[1]:.2f}" for p in points]
    fresh = lambda k: k in _wx and time.time() - _wx[k][0] < 540
    need = list({k: 1 for k in keys if not fresh(k)})
    for i in range(0, len(need), 50):
        c = need[i:i + 50]
        la = ",".join(k.split(",")[0] for k in c); lo = ",".join(k.split(",")[1] for k in c)
        d = fetch_json(f"{FORECAST}?latitude={la}&longitude={lo}&current=precipitation&hourly=precipitation&past_hours={PAST_HOURS}&forecast_hours={FORECAST_HOURS}&timezone=auto")
        for k, x in zip(c, d if isinstance(d, list) else [d]):
            _wx[k] = (time.time(), {"h": [v or 0 for v in x["hourly"]["precipitation"]], "now": x["current"]["precipitation"] or 0})
    return [_wx[k][1] for k in keys]

# ---------- VAYRA EXPLICIT DOMAIN CAPABILITIES ----------

def detect_road_flood_risk(lat: float, lng: float, simulated_rainfall=None) -> Dict[str, Any]:
    """Road Flood Risk Detection combining terrain, rainfall, and ML."""
    weather_data = weather([(lat, lng)])[0]
    hourly_rainfall = weather_data["h"]
    past_rainfall, next_rainfall = sum(hourly_rainfall[:PAST_HOURS]), max(hourly_rainfall[PAST_HOURS:PAST_HOURS + 4] or [0])
    
    dl = .0036; dg = dl / math.cos(lat * math.pi / 180)
    ring = [(lat, lng)] + [(lat + dl * math.sin(k * math.pi / 4), lng + dg * math.cos(k * math.pi / 4)) for k in range(8)]
    terrain_elevations = elevations(ring)
    elevation_diff = terrain_elevations[0] - sum(terrain_elevations[1:]) / 8
    road_slope = sum(math.atan(abs(v - terrain_elevations[0]) / 400) for v in terrain_elevations[1:]) / 8 * 180 / math.pi
    
    annual_rainfall = annual_rain([(lat, lng)])[0]
    flood_probability = predict_flood_prone_area([lat, lng, annual_rainfall, terrain_elevations[0], road_slope])
    
    current_rainfall = simulated_rainfall if simulated_rainfall is not None else max(weather_data["now"], next_rainfall * .8)
    ground_saturation = .8 if simulated_rainfall is not None else clamp(.3 + past_rainfall / 40, .3, .95)
    
    flood_risk_score = current_rainfall * (.6 + ground_saturation * .4) * combine_vulnerability(elevation_diff, calculate_ml_vulnerability(flood_probability))
    risk_classification = classify_risk_level(flood_risk_score)
    
    return {
        "classification": risk_classification,
        "flood_risk_score": round(flood_risk_score, 2),
        "rainfall_now": round(current_rainfall, 2),
        "elevation": round(terrain_elevations[0]),
        "road_slope": round(road_slope, 1),
        "flood_prone_probability": round(flood_probability, 3)
    }

def analyze_road_segments(route_points: list, elevations_data: list, annual_rainfall_data: list, rainfall_forecast: list, travel_time_seconds: float, ground_saturation: float, simulated_rainfall=None) -> list:
    """Road Segment Analysis: Checks each segment for flood risk."""
    n = len(route_points)
    risky_road_segments = []
    
    for k, point in enumerate(route_points):
        neighbors = [elevations_data[j] for j in range(max(0, k - 3), min(n - 1, k + 3) + 1) if j != k]
        progress_fraction = k / (n - 1) if n > 1 else 0
        weather_segment = rainfall_forecast[min(len(rainfall_forecast) - 1, rnd(progress_fraction * (len(rainfall_forecast) - 1)))]
        forecast_hour = min(len(weather_segment) - 1, rnd(progress_fraction * travel_time_seconds / 3600))
        segment_rainfall = simulated_rainfall if simulated_rainfall is not None else weather_segment[forecast_hour]
        
        gradients = []
        if k > 0: gradients.append(abs(elevations_data[k] - elevations_data[k - 1]) / max(50, calculate_travel_distance(point, route_points[k - 1])))
        if k < n - 1: gradients.append(abs(elevations_data[k + 1] - elevations_data[k]) / max(50, calculate_travel_distance(point, route_points[k + 1])))
        segment_slope = math.atan(sum(gradients) / (len(gradients) or 1)) * 180 / math.pi
        
        ml_vuln = calculate_ml_vulnerability(predict_flood_prone_area([point[0], point[1], annual_rainfall_data[k], elevations_data[k], segment_slope]))
        elevation_diff = elevations_data[k] - sum(neighbors) / len(neighbors) if neighbors else 0
        
        segment_risk_score = segment_rainfall * (.6 + ground_saturation * .4) * combine_vulnerability(elevation_diff, ml_vuln)
        
        if segment_risk_score >= 12: # MODERATE, HIGH, or CRITICAL
            risky_road_segments.append({
                "road_segment": list(point), 
                "segment_risk_score": segment_risk_score, 
                "risk_classification": classify_risk_level(segment_risk_score)
            })
            
    return risky_road_segments

def evaluate_route_risk(route_points: list, elevations_data: list, annual_rainfall_data: list, rainfall_forecast: list, travel_time_seconds: float, ground_saturation: float, simulated_rainfall=None) -> dict:
    """Route Evaluation: Scores the entire route and identifies worst segments."""
    risky_road_segments = analyze_road_segments(route_points, elevations_data, annual_rainfall_data, rainfall_forecast, travel_time_seconds, ground_saturation, simulated_rainfall)
    
    route_risk_score = max([seg["segment_risk_score"] for seg in risky_road_segments]) if risky_road_segments else 0
    route_risk_classification = classify_risk_level(route_risk_score)
    
    return {
        "route_risk_classification": route_risk_classification,
        "route_risk_score": round(route_risk_score, 2),
        "risky_road_segments": risky_road_segments,
        "is_safe": route_risk_score < 12
    }

def generate_early_warning(route_analysis: dict) -> str:
    """Early Warning: Generates an alert message based on route evaluation."""
    if route_analysis["is_safe"]:
        return "Route is safe. No flood risk detected."
    
    worst_class = route_analysis["route_risk_classification"]
    segment_count = len(route_analysis["risky_road_segments"])
    return f"EARLY WARNING: {worst_class} flood risk detected on {segment_count} road segments ahead! Consider alternative routes."

def evaluate_alternative_routes(routes: list, simulated_rainfall=None) -> dict:
    """Alternative Route Evaluation: Evaluates multiple route decisions and selects the safest."""
    pts = [[tuple(p) for p in r["points"]] for r in routes]
    flat_points = [p for r in pts for p in r]
    all_elevations = elevations(flat_points)
    all_annual_rainfall = annual_rain(flat_points)
    
    waypoints = [[r[rnd(f * (len(r) - 1))] for f in (0, .25, .5, .75, 1)] for r in pts]
    weather_data = weather([p for r in waypoints for p in r])
    ground_saturation = .8 if simulated_rainfall is not None else clamp(.3 + sum(weather_data[0]["h"][:PAST_HOURS]) / 40, .3, .95)
    
    evaluated_route_decisions = []
    offset = 0
    for i, route in enumerate(routes):
        num_points = len(pts[i])
        rainfall_forecast = [x["h"][PAST_HOURS:] or [0] for x in weather_data[i * 5:i * 5 + 5]]
        
        route_analysis = evaluate_route_risk(
            pts[i], 
            all_elevations[offset:offset + num_points], 
            all_annual_rainfall[offset:offset + num_points], 
            rainfall_forecast, 
            route["duration"], 
            ground_saturation, 
            simulated_rainfall
        )
        
        route_analysis["route_index"] = i
        route_analysis["travel_time_seconds"] = route["duration"]
        route_analysis["early_warning"] = generate_early_warning(route_analysis)
        evaluated_route_decisions.append(route_analysis)
        offset += num_points
        
    safest_alternative_route = min(evaluated_route_decisions, key=lambda x: (x["route_risk_score"], x["travel_time_seconds"]))
    
    return {
        "evaluated_route_decisions": evaluated_route_decisions,
        "recommended_alternative_route_index": safest_alternative_route["route_index"],
        "safest_route_classification": safest_alternative_route["route_risk_classification"]
    }
