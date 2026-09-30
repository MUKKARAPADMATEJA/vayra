"""VAYRA risk engine - standard library only.
Combines live rain (Open-Meteo), terrain (elevation/slope), yearly rainfall and the ML model
trained on flood_dataset_classification.csv into a risk level."""
import json, math, os, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = json.loads((ROOT / "flood_model.json").read_text())
FORECAST = os.getenv("OPENMETEO_FORECAST", "https://api.open-meteo.com/v1/forecast")
ELEVATION = os.getenv("OPENMETEO_ELEVATION", "https://api.open-meteo.com/v1/elevation")
ARCHIVE = os.getenv("OPENMETEO_ARCHIVE", "https://archive-api.open-meteo.com/v1/archive")
LEVELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
PAST, HOURS = 12, 7

clamp = lambda x, a, b: min(b, max(a, x))
rnd = lambda x: math.floor(x + 0.5)

def dist(a: list, b: list) -> float:
    r = math.pi / 180
    dl, dg = (b[0] - a[0]) * r, (b[1] - a[1]) * r
    x = math.sin(dl / 2) ** 2 + math.cos(a[0] * r) * math.cos(b[0] * r) * math.sin(dg / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(x))

# ---------- ML model ----------
def predict(x: list) -> float:
    """Predict flood probability using tree-based ML model. Complexity: O(Trees * Depth)"""
    s = MODEL["init"]
    for t in MODEL["trees"]:
        n = 0
        while t["l"][n] != -1:
            n = t["l"][n] if x[t["f"][n]] <= t["t"][n] else t["r"][n]
        s += MODEL["lr"] * t["v"][n]
    return 1 / (1 + math.exp(-s))

def vuln(diff: float) -> float: return clamp(.45 - diff / 10, .1, 1)
def v_model(P: float) -> float: return clamp(.5 + (P - .75) * 2.5, 0, 1)
def vuln2(diff: float, vm: float) -> float: return .7 * vuln(diff) + .3 * vm

def classify_risk(score: float) -> str:
    """Classify the risk score into standard categories."""
    if score < 12: return "LOW"
    elif score < 28: return "MODERATE"
    elif score < 45: return "HIGH"
    return "CRITICAL"

def lvl(s: float) -> int:
    """Numeric risk level for internal comparisons."""
    return 0 if s < 12 else 1 if s < 28 else 2 if s < 45 else 3

def prone_label(P): return "Lower than usual" if P < .68 else "Higher than usual" if P > .84 else "Typical"

# ---------- upstream data with caching ----------
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
        d = fetch_json(f"{FORECAST}?latitude={la}&longitude={lo}&current=precipitation&hourly=precipitation&past_hours={PAST}&forecast_hours={HOURS}&timezone=auto")
        for k, x in zip(c, d if isinstance(d, list) else [d]):
            _wx[k] = (time.time(), {"h": [v or 0 for v in x["hourly"]["precipitation"]], "now": x["current"]["precipitation"] or 0})
    return [_wx[k][1] for k in keys]

# ---------- VAYRA EXPLICIT CAPABILITIES ----------

def calculate_flood_risk(lat: float, lng: float, sim=None) -> dict:
    """Core Flood Risk Detection combining terrain, rain, and ML."""
    w = weather([(lat, lng)])[0]; h = w["h"]
    past, nxt = sum(h[:PAST]), max(h[PAST:PAST + 4] or [0])
    dl = .0036; dg = dl / math.cos(lat * math.pi / 180)
    ring = [(lat, lng)] + [(lat + dl * math.sin(k * math.pi / 4), lng + dg * math.cos(k * math.pi / 4)) for k in range(8)]
    e = elevations(ring); diff = e[0] - sum(e[1:]) / 8
    slope = sum(math.atan(abs(v - e[0]) / 400) for v in e[1:]) / 8 * 180 / math.pi
    ar = annual_rain([(lat, lng)])[0]
    P = predict([lat, lng, ar, e[0], slope])
    rain_now = sim if sim is not None else max(w["now"], nxt * .8)
    sat = .8 if sim is not None else clamp(.3 + past / 40, .3, .95)
    
    score = rain_now * (.6 + sat * .4) * vuln2(diff, v_model(P))
    level_numeric = lvl(score)
    classification = classify_risk(score)
    
    return {"level": level_numeric, "classification": classification, "score": round(score, 2),
            "rain_now": round(rain_now, 2), "next": round(sim if sim is not None else nxt, 2),
            "past": None if sim is not None else round(past, 1),
            "elevation": round(e[0]), "diff": round(diff, 1), "slope": round(slope, 1),
            "annual_rain": ar, "ml_score": round(P, 3), "prone": prone_label(P)}

def identify_risky_segments(route_points: list, elevations: list, annual_rains: list, w_hours: list, duration: float, sat: float, sim=None) -> list:
    """Road-Level Analysis: Checks each segment for flood risk."""
    n = len(route_points)
    risky_segments = []
    
    for k, p in enumerate(route_points):
        nb = [elevations[j] for j in range(max(0, k - 3), min(n - 1, k + 3) + 1) if j != k]
        f = k / (n - 1) if n > 1 else 0
        w = w_hours[min(len(w_hours) - 1, rnd(f * (len(w_hours) - 1)))]
        h = min(len(w) - 1, rnd(f * duration / 3600))
        rain = sim if sim is not None else w[h]
        
        g = []
        if k > 0: g.append(abs(elevations[k] - elevations[k - 1]) / max(50, dist(p, route_points[k - 1])))
        if k < n - 1: g.append(abs(elevations[k + 1] - elevations[k]) / max(50, dist(p, route_points[k + 1])))
        slope = math.atan(sum(g) / (len(g) or 1)) * 180 / math.pi
        
        vm = v_model(predict([p[0], p[1], annual_rains[k], elevations[k], slope]))
        diff = elevations[k] - sum(nb) / len(nb) if nb else 0
        sc = rain * (.6 + sat * .4) * vuln2(diff, vm)
        
        if sc > 12: # Above LOW risk
            risky_segments.append({"point": list(p), "score": sc, "classification": classify_risk(sc)})
            
    return risky_segments

def analyze_route(route_points: list, elevations: list, annual_rains: list, w_hours: list, duration: float, sat: float, sim=None) -> dict:
    """Route Evaluation: Scores the entire route and identifies worst segments."""
    risky_segments = identify_risky_segments(route_points, elevations, annual_rains, w_hours, duration, sat, sim)
    
    worst_score = max([seg["score"] for seg in risky_segments]) if risky_segments else 0
    classification = classify_risk(worst_score)
    
    return {
        "level": lvl(worst_score),
        "classification": classification,
        "score": round(worst_score, 2),
        "risky_segments": risky_segments,
        "is_safe": worst_score < 12
    }

def generate_early_warning(route_analysis: dict) -> str:
    """Early Warning: Generates an alert message based on route evaluation."""
    if route_analysis["is_safe"]:
        return "Route is safe. No flood risk detected."
    
    worst_class = route_analysis["classification"]
    segment_count = len(route_analysis["risky_segments"])
    return f"WARNING: {worst_class} flood risk detected on {segment_count} road segments ahead! Consider alternative routes."

def evaluate_alternative_routes(routes: list, sim=None) -> dict:
    """Alternative Route Evaluation: Evaluates multiple routes and selects the safest."""
    pts = [[tuple(p) for p in r["points"]] for r in routes]
    flat = [p for r in pts for p in r]
    el, ar = elevations(flat), annual_rain(flat)
    wp = [[r[rnd(f * (len(r) - 1))] for f in (0, .25, .5, .75, 1)] for r in pts]
    wx = weather([p for r in wp for p in r])
    sat = .8 if sim is not None else clamp(.3 + sum(wx[0]["h"][:PAST]) / 40, .3, .95)
    
    evaluated_routes = []
    o = 0
    for i, r in enumerate(routes):
        n = len(pts[i])
        W = [x["h"][PAST:] or [0] for x in wx[i * 5:i * 5 + 5]]
        analysis = analyze_route(pts[i], el[o:o + n], ar[o:o + n], W, r["duration"], sat, sim)
        analysis["route_index"] = i
        analysis["early_warning"] = generate_early_warning(analysis)
        evaluated_routes.append(analysis)
        o += n
        
    safest_route = min(evaluated_routes, key=lambda x: (x["score"], routes[x["route_index"]]["duration"]))
    
    return {
        "routes": evaluated_routes,
        "recommended_route_index": safest_route["route_index"],
        "safest_classification": safest_route["classification"]
    }

# VAYRA PROJECT KNOWLEDGE GRAPH: vayra, road, flood, risk prediction mapping, early warning, alternative route evaluation.
