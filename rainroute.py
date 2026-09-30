#!/usr/bin/env python3
"""RainRoute - RoadFlood Early Warning System (OptiForge 2026, algorithm design).

Pipeline
    1. Hydrology   : grid flood model (rain -> runoff -> drainage -> lateral flow).
    2. Forecaster  : data assimilation (sensor nudging + radar rain correction)
                     issues rolling forecasts every REPLAN_EVERY steps.
    3. Fuzzy risk  : Sugeno fuzzy inference on (depth, rise-rate) -> risk in [0, 1].
    4. Router      : time-dependent A* on predicted depth (block + risk penalty).
    5. Tuner       : real-coded Genetic Algorithm over 9 normalised genes.
    6. Round 2     : hidden shifts (cloudburst / drain block / sensor loss /
                     closures) handled by replanning + warm-started robust GA.

Usage
    python rainroute.py --round 1          # design + tune on clean storms
    python rainroute.py --round 2          # shift robustness + re-tune
    python rainroute.py --round all        # full run (default)
    python rainroute.py --quick            # small budget smoke test
Only numpy is required.
"""
from __future__ import annotations

import argparse
import heapq
import json
import time
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Configuration (every tunable constant lives here -> easy live patching)
# --------------------------------------------------------------------------
CFG = {
    "N": 16,                 # grid is N x N road junctions
    "T": 48,                 # simulation steps
    "DT_MIN": 5,             # minutes per step
    "CELL_KM": 1.0,
    "SPEED_KMH": 24.0,
    "D_REF": 0.30,           # reference stranding depth (m) for a car
    "N_SENSORS": 28,
    "SENSOR_NOISE": 0.01,    # metres
    "N_TRIPS": 10,
    "REPLAN_EVERY": 4,       # forecast cycle length (steps)
    "MAX_WAIT": 8,           # steps a vehicle may hold when no safe route
    "ISSUE_STEPS": (8, 14),  # when early warnings are evaluated
    # LIVE PATCH POINT: vehicle classes -> maximum wading depth (m)
    "VEHICLES": {"car": 0.30, "bus": 0.45, "bike": 0.18, "ambulance": 0.50},
    "VEHICLE_MIX": {"car": 1.0},
}
BASE_STEPS = CFG["CELL_KM"] / CFG["SPEED_KMH"] * 60.0 / CFG["DT_MIN"]

# Genes: name, low, high  (GA works on [0, 1]^9 and decodes with these)
GENES = [
    ("alpha", 0.5, 12.0),        # risk penalty weight in routing cost
    ("d_block", 0.08, 0.32),     # predicted depth that blocks a road
    ("d_med", 0.04, 0.18),       # fuzzy "medium" depth centre
    ("d_high", 0.20, 0.45),      # fuzzy "high" depth start
    ("rise_w", 0.0, 1.0),        # boost for fast-rising water
    ("theta", 0.15, 0.85),       # alert threshold on risk
    ("assim_gain", 0.0, 0.95),   # sensor nudging strength
    ("rain_gain", 0.0, 1.0),     # radar/forecast rain correction strength
    ("margin", 0.0, 6.0),        # look-ahead margin (steps)
]
DEFAULT = {"alpha": 4.0, "d_block": 0.20, "d_med": 0.10, "d_high": 0.30,
           "rise_w": 0.5, "theta": 0.5, "assim_gain": 0.5, "rain_gain": 0.5,
           "margin": 2}

# Pre-tuned by Round-1 GA (pop 16, 10 gens, seed 0): ships with the script so it runs instantly.
TUNED = {"alpha": 3.301, "d_block": 0.295, "d_med": 0.18, "d_high": 0.395,
         "rise_w": 0.414, "theta": 0.842, "assim_gain": 0.733, "rain_gain": 0.553, "margin": 4}


def decode(x: np.ndarray) -> dict:
    """Map a normalised gene vector to a parameter dict (with validity repair)."""
    p = {n: lo + float(v) * (hi - lo) for (n, lo, hi), v in zip(GENES, x)}
    p["margin"] = int(round(p["margin"]))
    p["d_high"] = max(p["d_high"], p["d_med"] + 0.05)
    return p


def encode(p: dict) -> np.ndarray:
    """Inverse of decode (used for warm starts)."""
    return np.array([(p[n] - lo) / (hi - lo) for n, lo, hi in GENES]).clip(0, 1)


# --------------------------------------------------------------------------
# 1. World + hydrology
# --------------------------------------------------------------------------
@dataclass
class World:
    N: int
    elev: np.ndarray
    runoff: np.ndarray
    drain: np.ndarray
    sensors: np.ndarray
    nbrs: list = field(default_factory=list)


def make_world(seed: int = 7) -> World:
    """Synthetic city: smooth terrain with basins, patchy imperviousness."""
    N = CFG["N"]
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:N, 0:N]
    elev = 0.03 * xx
    for _ in range(6):
        cx, cy = rng.uniform(0, N, 2)
        s = rng.uniform(0.15, 0.3) * N
        elev = elev + rng.uniform(-1.5, 1.5) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    elev -= elev.min()
    sensors = rng.choice(N * N, CFG["N_SENSORS"], replace=False)
    nbrs = [[(y + dy) * N + x + dx for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
             if 0 <= y + dy < N and 0 <= x + dx < N]
            for y in range(N) for x in range(N)]
    return World(N, elev, rng.uniform(0.5, 0.95, (N, N)),
                 rng.uniform(0.0012, 0.0030, (N, N)), sensors, nbrs)


def hydro_step(d, rain_mm, elev, runoff, drain, k=0.2):
    """One 5-min step: rain->runoff, drainage sink, then downhill lateral flow."""
    d = d + rain_mm * 1e-3 * runoff
    d = d - np.minimum(d, drain)
    h = np.pad(elev + d, 1, mode="edge")
    c = h[1:-1, 1:-1]
    fl = np.stack([np.maximum(c - h[2:, 1:-1], 0), np.maximum(c - h[:-2, 1:-1], 0),
                   np.maximum(c - h[1:-1, 2:], 0), np.maximum(c - h[1:-1, :-2], 0)]) * k
    fl *= np.minimum(1.0, d / (fl.sum(0) + 1e-9))       # cannot export more than stored
    inflow = np.zeros_like(d)
    inflow[1:] += fl[0][:-1]
    inflow[:-1] += fl[1][1:]
    inflow[:, 1:] += fl[2][:, :-1]
    inflow[:, :-1] += fl[3][:, 1:]
    return d - fl.sum(0) + inflow


def simulate(w: World, rain, drain, d0=None):
    """Run the model; returns depth[0..T] (depth[t] = state at start of step t)."""
    d = np.zeros((w.N, w.N)) if d0 is None else d0
    out = np.empty((len(rain) + 1, w.N, w.N))
    out[0] = d
    for t, r in enumerate(rain):
        d = hydro_step(d, r, w.elev, w.runoff, drain)
        out[t + 1] = d
    return out


# --------------------------------------------------------------------------
# 2. Scenarios (ground truth, forecast, radar, sensors, hidden shifts)
# --------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    rain_true: np.ndarray
    rain_fc: np.ndarray
    rain_obs: np.ndarray
    depth: np.ndarray
    sens_obs: np.ndarray
    closures: dict
    trips: list


def _blur(a):
    """3x3 box blur over the last two axes (zero padded)."""
    p = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)])
    n, m = a.shape[-2:]
    return sum(p[..., i:i + n, j:j + m] for i in range(3) for j in range(3)) / 9.0


def _storm(rng, N, T):
    """Moving Gaussian storm cell with triangular hyetograph (mm per step)."""
    yy, xx = np.mgrid[0:N, 0:N]
    s0, dur, peak = rng.integers(2, 8), rng.integers(18, 26), rng.uniform(260, 380)
    (cx, cy), (vx, vy) = rng.uniform(0, N, 2), rng.uniform(-0.15, 0.15, 2) * N / dur
    sig = rng.uniform(0.25, 0.4) * N
    rain = np.zeros((T, N, N))
    for t in range(T):
        u = (t - s0) / dur
        if 0 <= u <= 1:
            h = u / 0.4 if u < 0.4 else (1 - u) / 0.6
            g = np.exp(-((xx - cx - vx * (t - s0)) ** 2 + (yy - cy - vy * (t - s0)) ** 2) / (2 * sig ** 2))
            rain[t] = peak * h * (0.25 + 0.75 * g) * CFG["DT_MIN"] / 60.0
    return rain


def _disk(N, rng, r=4.0):
    yy, xx = np.mgrid[0:N, 0:N]
    cy, cx = rng.uniform(3, N - 3, 2)
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def make_trips(w: World, rng, n: int, mix: dict) -> list:
    """Random long trips: (src, dst, departure_step, vehicle_class)."""
    names, probs = list(mix), np.array(list(mix.values()), float)
    trips = []
    while len(trips) < n:
        s, d = (int(v) for v in rng.integers(0, w.N * w.N, 2))
        if abs(s // w.N - d // w.N) + abs(s % w.N - d % w.N) >= w.N:
            trips.append((s, d, int(rng.integers(0, int(CFG["T"] * 0.55))),
                          str(rng.choice(names, p=probs / probs.sum()))))
    return trips


def make_scenario(w: World, seed: int, shift: str = "none") -> Scenario:
    """Build one event. `shift` is a hidden perturbation the predictor is never told about."""
    N, T = w.N, CFG["T"]
    rng = np.random.default_rng(seed)
    rain = _storm(rng, N, T)
    fc = np.roll(_blur(_blur(rain)), 1, axis=2) * rng.uniform(0.75, 1.25)     # biased, displaced
    drain = w.drain * rng.uniform(0.9, 1.1)
    closures, sens_drop = {}, None
    t_s = int(rng.integers(10, 20))
    if shift == "cloudburst":
        rain = rain.copy()
        rain[t_s:] += rain[t_s:] * 1.6 * _disk(N, rng)
    elif shift == "drain_block":
        drain = np.where(_disk(N, rng), drain * 0.15, drain)
    elif shift == "closure":
        closures = {int(n): int(rng.integers(8, 24)) for n in rng.choice(N * N, 5, replace=False)}
    elif shift == "sensor_drop":
        sens_drop = (t_s, rng.random(len(w.sensors)) < 0.7)
    depth = simulate(w, rain, drain)
    obs = depth.reshape(T + 1, -1)[:, w.sensors] + rng.normal(0, CFG["SENSOR_NOISE"], (T + 1, len(w.sensors)))
    if sens_drop:
        obs[sens_drop[0]:, sens_drop[1]] = np.nan
    radar = rain * rng.lognormal(0, 0.15, rain.shape)
    return Scenario(shift, rain, fc, radar, depth, obs, closures,
                    make_trips(w, rng, CFG["N_TRIPS"], CFG["VEHICLE_MIX"]))


# --------------------------------------------------------------------------
# 3. Forecaster (assimilation) + fuzzy risk
# --------------------------------------------------------------------------
def fuzzy_risk(depth, rise, p):
    """Sugeno fuzzy inference: memberships LOW/MED/HIGH depth; rising water boosts risk."""
    a, b = p["d_med"], p["d_high"]
    low = np.clip((a - depth) / a, 0, 1)
    high = np.clip((depth - a) / (b - a), 0, 1)
    base = 0.05 * low + 0.45 * (1 - low - high) + 0.95 * high
    rising = np.clip(rise / 0.01, 0, 1)
    return np.clip(base * (1 + p["rise_w"] * rising), 0, 1)


def _nudge(d, obs, sensors, gain):
    """Spread sensor innovations over a ~2-cell radius (cheap optimal-interpolation)."""
    ok = ~np.isnan(obs)
    if gain <= 0 or not ok.any():
        return d
    idx = sensors[ok]
    inn, wt = np.zeros(d.size), np.zeros(d.size)
    inn[idx] = obs[ok] - d.ravel()[idx]
    wt[idx] = 1.0
    inn, wt = _blur(_blur(inn.reshape(d.shape))), _blur(_blur(wt.reshape(d.shape)))
    corr = np.where(wt > 1e-4, inn / np.maximum(wt, 1e-4), 0) * np.clip(wt * 9, 0, 1)
    return np.maximum(d + gain * corr, 0)


class Forecaster:
    """mode: 'full' (assimilate + forecast) | 'persist' (freeze current state) | 'static' (no water info)."""

    def __init__(self, w: World, sc: Scenario, p: dict, mode: str = "full"):
        self.w, self.sc, self.p, self.mode = w, sc, p, mode
        self.T = CFG["T"]
        self._pred, self._grid = {}, {}
        self.states = self._assimilate() if mode != "static" else None

    def _assimilate(self):
        """Causal analysis pass: radar rain drives the model, sensors correct it."""
        w, sc = self.w, self.sc
        d, states = np.zeros((w.N, w.N)), [np.zeros((w.N, w.N))]
        for t in range(self.T):
            d = hydro_step(d, sc.rain_obs[t], w.elev, w.runoff, w.drain)
            d = _nudge(d, sc.sens_obs[t + 1], w.sensors, self.p["assim_gain"])
            states.append(d)
        return states

    def _rain_factor(self, t0):
        """Per-region ratio of radar to forecast rain over the last 6 steps (shrunk by rain_gain)."""
        N, g = self.w.N, self.p["rain_gain"]
        f, lo, q = np.ones((N, N)), max(0, t0 - 6)
        if t0 == lo or g <= 0:
            return f
        for y in range(0, N, q):
            for x in range(0, N, q):
                o = self.sc.rain_obs[lo:t0, y:y + q, x:x + q].sum() + 1e-3
                c = self.sc.rain_fc[lo:t0, y:y + q, x:x + q].sum() + 1e-3
                f[y:y + q, x:x + q] = 1 + g * (np.clip(o / c, 0.6, 3.0) - 1)
        return f

    def predict(self, t0: int) -> np.ndarray:
        """Predicted depth for steps t0..T, shape (T-t0+1, N, N)."""
        if t0 in self._pred:
            return self._pred[t0]
        w, K = self.w, self.T - t0 + 1
        if self.mode == "static":
            out = np.zeros((K, w.N, w.N))
        elif self.mode == "persist":
            out = np.repeat(self.states[t0][None], K, axis=0)
        else:
            f = self._rain_factor(t0)
            out = simulate(w, self.sc.rain_fc[t0:] * f, w.drain, self.states[t0])
        self._pred[t0] = out
        return out

    def fields(self, t0: int, scale: float):
        """Routing lookup tables (blocked, time-multiplier, cost-multiplier) for one vehicle class."""
        key = (t0, scale)
        if key in self._grid:
            return self._grid[key]
        pred, p = self.predict(t0), self.p
        pmax = pred.copy()
        for s in range(1, p["margin"] + 1):                  # look-ahead margin
            pmax[:-s] = np.maximum(pmax[:-s], pred[s:])
        rise = np.maximum(np.diff(pred, axis=0, prepend=pred[:1]), 0)
        risk = fuzzy_risk(pmax, rise, p)
        ttm = 1.0 / np.clip(1 - pred / (CFG["D_REF"] * scale + 0.05), 0.2, 1)
        K = len(pred)
        res = ((pmax > p["d_block"] * scale).reshape(K, -1).tolist(),
               ttm.reshape(K, -1).tolist(),
               (ttm * (1 + p["alpha"] * risk)).reshape(K, -1).tolist())
        self._grid[key] = res
        return res

    def alerts(self, t_issue: int):
        """Per-node alert flag = peak fuzzy risk over horizon >= theta."""
        pred = self.predict(t_issue)
        rise = np.maximum(np.diff(pred, axis=0, prepend=pred[:1]), 0)
        return fuzzy_risk(pred, rise, self.p).max(axis=0) >= self.p["theta"]


# --------------------------------------------------------------------------
# 4. Router: time-dependent A*
# --------------------------------------------------------------------------
def plan(fc: Forecaster, t0: int, t_now: float, src: int, dst: int, scale: float, closed) -> list | None:
    """Least-risk path from src to dst using forecast issued at t0. None if no safe path."""
    N = fc.w.N
    blocked, ttm, cm = fc.fields(t0, scale)
    K = len(blocked)
    dy, dx = dst // N, dst % N

    def heur(v: int) -> float:
        return (abs(v // N - dy) + abs(v % N - dx)) * BASE_STEPS

    best, parent = {src: 0.0}, {}
    heap = [(heur(src), 0.0, t_now, src)]
    while heap:
        _, g, t, u = heapq.heappop(heap)
        if g > best[u]:
            continue
        if u == dst:
            path = [u]
            while path[-1] != src:
                path.append(parent[path[-1]])
            return path[::-1]
        k = min(int(t + BASE_STEPS) - t0, K - 1)
        for v in fc.w.nbrs[u]:
            if blocked[k][v] or v in closed:
                continue
            ng = g + BASE_STEPS * cm[k][v]
            if ng < best.get(v, 1e18):
                best[v], parent[v] = ng, u
                heapq.heappush(heap, (ng + heur(v), ng, t + BASE_STEPS * ttm[k][v], v))
    return None


# --------------------------------------------------------------------------
# 5. Trip executor (ground truth) and metrics
# --------------------------------------------------------------------------
def run_trip(trip, w: World, sc: Scenario, fc: Forecaster) -> float:
    """Drive one trip through the TRUE flood, replanning each forecast cycle.

    Score: 1/(1+0.25*(delay_ratio-1)) if safe, 0.3 if abandoned, 0 if stranded.
    """
    src, dst, t_dep, vclass = trip
    d_fail = CFG["VEHICLES"][vclass]
    scale, R, T = d_fail / CFG["D_REF"], CFG["REPLAN_EVERY"], CFG["T"]
    t, u, path, cycle, known, waited = float(t_dep), src, [], -1, frozenset(), 0
    free = (abs(src // w.N - dst // w.N) + abs(src % w.N - dst % w.N)) * BASE_STEPS
    while u != dst:
        if t >= T:
            return 0.3
        now = frozenset(n for n, s in sc.closures.items() if s <= t)
        if int(t) // R != cycle or now != known or (path and path[1] in now):
            cycle, known = int(t) // R, now
            path = plan(fc, cycle * R, t, u, dst, scale, now)
            if path is None:                             # hold position, re-plan next step
                if sc.depth[min(int(t), T)].ravel()[u] > d_fail:
                    return 0.0
                t, waited, cycle = t + 1, waited + 1, -1
                if waited > CFG["MAX_WAIT"]:
                    return 0.3
                continue
        v = path[1]
        d_now = sc.depth[min(int(t + BASE_STEPS), T)].ravel()[v]
        ta = t + BASE_STEPS / float(np.clip(1 - d_now / (d_fail + 0.05), 0.2, 1))
        d_arr = sc.depth[min(int(ta), T)].ravel()[v]
        if d_arr > d_fail or sc.closures.get(v, 1e9) <= ta:
            return 0.0
        t, u, path = ta, v, path[1:]
    return 1.0 / (1.0 + 0.25 * max(0.0, (t - t_dep) / free - 1.0))


def warning_score(sc: Scenario, fc: Forecaster, t_issue: int):
    """F2 (recall-weighted) of node alerts vs true flooding, plus mean lead time (steps)."""
    truth_depth = sc.depth[t_issue:].reshape(len(sc.depth) - t_issue, -1)
    already = truth_depth[0] > CFG["D_REF"]
    flood = (truth_depth.max(0) > CFG["D_REF"]) & ~already
    alert = fc.alerts(t_issue).ravel() & ~already
    tp, fp, fn = (flood & alert).sum(), (~flood & alert).sum(), (flood & ~alert).sum()
    f2 = 1.0 if tp + fp + fn == 0 else 5 * tp / (5 * tp + 4 * fn + fp)
    lead = float(np.mean(np.argmax(truth_depth[:, flood & alert] > CFG["D_REF"], axis=0))) if tp else 0.0
    return f2, lead


def evaluate(p: dict, scens: list, w: World, mode: str = "full") -> dict:
    """Aggregate metrics over scenarios. fitness = 0.65*route + 0.35*F2 (worst-case blended in)."""
    per = []
    for sc in scens:
        fc = Forecaster(w, sc, p, mode)
        scores = np.array([run_trip(t, w, sc, fc) for t in sc.trips])
        route, strand = float(scores.mean()), float((scores == 0.0).mean())
        ws = [warning_score(sc, fc, ti) for ti in CFG["ISSUE_STEPS"]]
        per.append((route, float(np.mean([a for a, _ in ws])), float(np.mean([b for _, b in ws])), strand))
    a = np.array(per)
    fit = 0.65 * a[:, 0] + 0.35 * a[:, 1]
    return {"route": a[:, 0].mean(), "f2": a[:, 1].mean(), "lead": a[:, 2].mean(),
            "strand": a[:, 3].mean(), "fitness": 0.7 * fit.mean() + 0.3 * fit.min()}


# --------------------------------------------------------------------------
# 6. Genetic algorithm (real coded, BLX-alpha, tournament, elitism, decaying mutation)
# --------------------------------------------------------------------------
def ga(fitness, pop: int, gens: int, rng, init: np.ndarray | None = None, log=print):
    """Maximise fitness(gene_vector). Returns (best_vector, best_fitness, history)."""
    dim = len(GENES)
    P = rng.random((pop, dim))
    if init is not None:
        P[0] = init
        P[1:pop // 4] = np.clip(init + rng.normal(0, 0.1, (pop // 4 - 1, dim)), 0, 1)
    F = np.array([fitness(x) for x in P])
    hist = [float(F.max())]
    for g in range(gens):
        sigma = 0.02 + 0.13 * (1 - g / gens)
        elite = np.argsort(-F)[:2]
        kids = []
        while len(kids) < pop - 2:
            a, b = (P[max(rng.integers(0, pop, 3), key=lambda i: F[i])] for _ in range(2))
            span = 0.3 * np.abs(a - b)
            kid = rng.uniform(np.minimum(a, b) - span, np.maximum(a, b) + span)
            kid += (rng.random(dim) < 0.2) * rng.normal(0, sigma, dim)
            kids.append(np.clip(kid, 0, 1))
        Fk = np.array([fitness(k) for k in kids])
        P, F = np.vstack([P[elite], kids]), np.concatenate([F[elite], Fk])
        hist.append(float(F.max()))
        log(f"    gen {g + 1:2d}/{gens}  best fitness {F.max():.4f}")
    i = int(np.argmax(F))
    return P[i], float(F[i]), hist


# --------------------------------------------------------------------------
# 7. Rounds
# --------------------------------------------------------------------------
SHIFTS = ("cloudburst", "drain_block", "sensor_drop", "closure")


def table(rows: list) -> None:
    print(f"    {'method':34s} {'route':>6s} {'F2':>6s} {'lead':>5s} {'strand%':>8s} {'fitness':>8s}")
    for name, m in rows:
        print(f"    {name:34s} {m['route']:6.3f} {m['f2']:6.3f} {m['lead']:5.1f} "
              f"{100 * m['strand']:8.1f} {m['fitness']:8.4f}")


def compare(p: dict, scens: list, w: World) -> list:
    """Baselines vs tuned model on the same scenarios."""
    naive = dict(DEFAULT, alpha=0.0, d_block=9.0)
    return [("shortest path (no water info)", evaluate(naive, scens, w, "static")),
            ("reactive (current depth only)", evaluate(DEFAULT, scens, w, "persist")),
            ("forecast, untuned defaults", evaluate(DEFAULT, scens, w)),
            ("forecast + GA tuned (ours)", evaluate(p, scens, w))]


def round1(w: World, pop: int, gens: int, seed: int) -> dict:
    print("\n=== ROUND 1: design + tune on clean storms ===")
    train = [make_scenario(w, 100 + i) for i in range(4)]
    hold = [make_scenario(w, 900 + i) for i in range(6)]
    t0 = time.time()
    x, fit, _ = ga(lambda v: evaluate(decode(v), train, w)["fitness"], pop, gens, np.random.default_rng(seed))
    p = decode(x)
    print(f"  GA finished in {time.time() - t0:.1f}s (train fitness {fit:.4f})")
    print("  tuned parameters:", json.dumps({k: round(v, 3) for k, v in p.items()}))
    print("  HELD-OUT storms (unseen seeds):")
    table(compare(p, hold, w))
    json.dump(p, open("round1_params.json", "w"), indent=2)
    return p


def round2(w: World, p1: dict, pop: int, gens: int, seed: int) -> dict:
    print("\n=== ROUND 2: hidden scenario shifts + live patch ===")
    CFG["VEHICLE_MIX"] = {"car": 0.5, "bus": 0.2, "bike": 0.2, "ambulance": 0.1}   # <- live patch
    train = [make_scenario(w, 300 + 10 * i + j, s) for j, s in enumerate(SHIFTS) for i in range(2)]
    hold = {s: [make_scenario(w, 950 + 10 * i + j, s) for i in range(2)] for j, s in enumerate(SHIFTS)}
    print("  Round-1 params under each shift (degradation check):")
    for s, sc in hold.items():
        m = evaluate(p1, sc, w)
        print(f"    {s:12s} route {m['route']:.3f}  F2 {m['f2']:.3f}  strand {100 * m['strand']:.1f}%")
    t0 = time.time()
    x, fit, _ = ga(lambda v: evaluate(decode(v), train, w)["fitness"], pop, gens,
                   np.random.default_rng(seed + 1), init=encode(p1))
    print(f"  robust warm-start GA finished in {time.time() - t0:.1f}s (fitness {fit:.4f})")
    val = [make_scenario(w, 700 + 10 * i + j, s) for j, s in enumerate(SHIFTS) for i in range(2)]
    cands = {"re-tuned": decode(x), "round-1": p1}
    pick = max(cands, key=lambda k: evaluate(cands[k], val, w)["fitness"])     # guards against overfit
    p2 = cands[pick]
    print(f"  validation-based selection kept: {pick} parameters")
    all_hold = [sc for v in hold.values() for sc in v]
    print("  HELD-OUT shifted storms (all four shift types pooled):")
    table(compare(p2, all_hold, w) + [("ablation: no adaptation (open loop)",
                                       evaluate(dict(p2, assim_gain=0.0, rain_gain=0.0), all_hold, w))])
    json.dump(p2, open("round2_params.json", "w"), indent=2)
    return p2


def demo(w: World, p: dict) -> None:
    """Operator view for the panel: alert levels, lead time, and a rerouted trip."""
    sc = make_scenario(w, 4242, "cloudburst")
    fc, t_issue = Forecaster(w, sc, p), CFG["ISSUE_STEPS"][1]
    pred = fc.predict(t_issue).reshape(len(fc.predict(t_issue)), -1)
    risk = fuzzy_risk(pred, np.maximum(np.diff(pred, axis=0, prepend=pred[:1]), 0), p).max(0)
    lvl = np.digitize(risk, [p["theta"] * 0.5, p["theta"], 0.85])
    print(f"\n=== DEMO (cloudburst storm, warning issued at t={t_issue * CFG['DT_MIN']} min) ===")
    print("  junctions by level:", {n: int((lvl == i).sum()) for i, n in enumerate(["GREEN", "YELLOW", "ORANGE", "RED"])})
    for n in np.argsort(-risk)[:5]:
        eta = int(np.argmax(pred[:, n] > CFG["D_REF"])) if (pred[:, n] > CFG["D_REF"]).any() else None
        real = int(np.argmax(sc.depth[t_issue:].reshape(-1, w.N * w.N)[:, n] > CFG["D_REF"])) \
            if (sc.depth[t_issue:].reshape(-1, w.N * w.N)[:, n] > CFG["D_REF"]).any() else None
        print(f"  node {n:3d}  risk {risk[n]:.2f}  predicted flood in {None if eta is None else eta * CFG['DT_MIN']} min,"
              f" actual {None if real is None else real * CFG['DT_MIN']} min")
    src, dst, t_dep, vc = sc.trips[0]
    safe = plan(fc, (t_dep // CFG["REPLAN_EVERY"]) * CFG["REPLAN_EVERY"], float(t_dep), src, dst, 1.0, frozenset())
    short = plan(Forecaster(w, sc, p, "static"), 0, float(t_dep), src, dst, 1.0, frozenset())
    print(f"  trip {src}->{dst}: shortest path {len(short) - 1} hops, RainRoute path "
          f"{'NONE (hold)' if safe is None else str(len(safe) - 1) + ' hops'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="RainRoute RoadFlood Early Warning System")
    ap.add_argument("--round", default="all", choices=["1", "2", "all"])
    ap.add_argument("--quick", action="store_true", help="tiny GA budget for smoke tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", action="store_true", help="print operator alert view after the rounds")
    a = ap.parse_args()
    pop, gens = (8, 3) if a.quick else (16, 10)
    w = make_world()
    p1 = round1(w, pop, gens, a.seed) if a.round in ("1", "all") else TUNED
    p_final = round2(w, p1, pop, gens // 2 + 1, a.seed) if a.round in ("2", "all") else p1
    if a.demo:
        demo(w, p_final)


if __name__ == "__main__":
    main()
