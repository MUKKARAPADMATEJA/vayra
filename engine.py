"""RainRoute risk engine - standard library only.
Combines live rain (Open-Meteo), terrain (elevation/slope), yearly rainfall and the ML model
trained on flood_dataset_classification.csv into a Safe/Watch/Warning/Danger level."""
import json, math, os, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = json.loads((ROOT / "model" / "flood_model.json").read_text())
FORECAST = os.getenv("OPENMETEO_FORECAST", "https://api.open-meteo.com/v1/forecast")
ELEVATION = os.getenv("OPENMETEO_ELEVATION", "https://api.open-meteo.com/v1/elevation")
ARCHIVE = os.getenv("OPENMETEO_ARCHIVE", "https://archive-api.open-meteo.com/v1/archive")
LEVELS = ["SAFE", "WATCH", "WARNING", "DANGER"]
PAST, HOURS = 12, 7

clamp = lambda x, a, b: min(b, max(a, x))
rnd = lambda x: math.floor(x + 0.5)          # same rounding as JavaScript Math.round

def dist(a: list, b: list) -> float:
    r = math.pi / 180
    dl, dg = (b[0] - a[0]) * r, (b[1] - a[1]) * r
    x = math.sin(dl / 2) ** 2 + math.cos(a[0] * r) * math.cos(b[0] * r) * math.sin(dg / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(x))

# ---------- ML model ----------
def predict(x: list) -> float:
    """
    Predict flood probability using tree-based ML model.
    Algorithmic Complexity: O(Trees * Depth)
    """
    s = MODEL["init"]
    for t in MODEL["trees"]:
        n = 0
        while t["l"][n] != -1:
            n = t["l"][n] if x[t["f"][n]] <= t["t"][n] else t["r"][n]
        s += MODEL["lr"] * t["v"][n]
    return 1 / (1 + math.exp(-s))

def vuln(diff: float) -> float: return clamp(.45 - diff / 10, .1, 1)          # low spot vs surroundings
def v_model(P: float) -> float: return clamp(.5 + (P - .75) * 2.5, 0, 1)      # 0.5 = typical area in training data
def vuln2(diff: float, vm: float) -> float: return .7 * vuln(diff) + .3 * vm          # physics 70% + ML 30% (ML is a weak signal)
def lvl(s: float) -> int: return 0 if s < 12 else 1 if s < 28 else 2 if s < 45 else 3
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
    return [_rain.get(rk(p), 1000) for p in points]     # 1000 mm = typical value in the dataset

def weather(points):
    """hourly rain: first PAST entries are the past 12 h, then current hour + forecast. Cached 9 min (~1 km grid)."""
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

# ---------- risk at one place ----------
def assess(lat, lng, sim=None):
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
    level = lvl(rain_now * (.6 + sat * .4) * vuln2(diff, v_model(P)))
    return {"level": level, "name": LEVELS[level], "rain_now": round(rain_now, 2),
            "next": round(sim if sim is not None else nxt, 2), "past": None if sim is not None else round(past, 1),
            "elevation": round(e[0]), "diff": round(diff, 1), "slope": round(slope, 1), "annual_rain": ar,
            "ml_score": round(P, 3), "prone": prone_label(P)}

# ---------- risk along routes ----------
def score_route(pts, e, ar, W, dur, sat, sim):
    n = len(pts); worst = 0; wp = pts[0]; wt = mx = sm = mt = 0
    for k, p in enumerate(pts):
        nb = [e[j] for j in range(max(0, k - 3), min(n - 1, k + 3) + 1) if j != k]
        f = k / (n - 1) if n > 1 else 0
        w = W[min(len(W) - 1, rnd(f * (len(W) - 1)))]; h = min(len(w) - 1, rnd(f * dur / 3600))
        rain = sim if sim is not None else w[h]; sm += rain
        if rain > mx: mx, mt = rain, f * dur
        g = []
        if k > 0: g.append(abs(e[k] - e[k - 1]) / max(50, dist(p, pts[k - 1])))
        if k < n - 1: g.append(abs(e[k + 1] - e[k]) / max(50, dist(p, pts[k + 1])))
        slope = math.atan(sum(g) / (len(g) or 1)) * 180 / math.pi
        vm = v_model(predict([p[0], p[1], ar[k], e[k], slope]))
        sc = rain * (.6 + sat * .4) * vuln2(e[k] - sum(nb) / len(nb) if nb else 0, vm)
        if sc >= worst: worst, wp, wt = sc, p, f * dur
    avg = sm / n; slow = 1 + clamp(avg / 120, 0, .3); eta = rnd(dur * slow)
    dest = sim if sim is not None else W[-1][min(len(W[-1]) - 1, rnd(eta / 3600))]
    return {"level": lvl(worst), "name": LEVELS[lvl(worst)], "score": round(worst, 2), "pct": rnd(clamp(worst / 45 * 100, 0, 100)),
            "wp": list(wp), "wt": round(wt), "max_rain": round(mx, 2), "max_t": round(mt), "avg_rain": round(avg, 2),
            "eta": eta, "extra": eta - dur, "dest_rain": round(dest, 2)}

def score_routes(routes, sim=None):
    pts = [[tuple(p) for p in r["points"]] for r in routes]
    flat = [p for r in pts for p in r]
    el, ar = elevations(flat), annual_rain(flat)
    wp = [[r[rnd(f * (len(r) - 1))] for f in (0, .25, .5, .75, 1)] for r in pts]
    wx = weather([p for r in wp for p in r])
    sat = .8 if sim is not None else clamp(.3 + sum(wx[0]["h"][:PAST]) / 40, .3, .95)
    out, o = [], 0
    for i, r in enumerate(routes):
        n = len(pts[i])
        W = [x["h"][PAST:] or [0] for x in wx[i * 5:i * 5 + 5]]
        out.append(score_route(pts[i], el[o:o + n], ar[o:o + n], W, r["duration"], sat, sim)); o += n
    return out


# VAYRA PROJECT KNOWLEDGE GRAPH: vayra, road, flood, risk prediction mapping.


