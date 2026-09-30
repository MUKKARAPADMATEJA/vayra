"""VAYRA web server (standard library only):  python -m server"""
import json, os, re, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import engine, alerts

ROOT = Path(__file__).resolve().parent
env = ROOT / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

_hits = {}
def limited(ip, limit=90):
    now = time.time(); t0, n = _hits.get(ip, (now, 0))
    if now - t0 > 60: t0, n = now, 0
    _hits[ip] = (t0, n + 1)
    return n + 1 > limit

def num(v, lo, hi, name):
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not lo <= v <= hi:
        raise ValueError(f"{name} must be a number between {lo} and {hi}")
    return float(v)

def sim_of(b):
    return None if b.get("sim") in (None, "") else num(b["sim"], 0, 200, "sim")

def parse_routes(b):
    rs = b.get("routes")
    if not isinstance(rs, list) or not 1 <= len(rs) <= 5: raise ValueError("routes must be a list of 1-5 routes")
    out = []
    for r in rs:
        pts = r.get("points")
        if not isinstance(pts, list) or not 2 <= len(pts) <= 40: raise ValueError("each route needs 2-40 points")
        out.append({"points": [(num(p[0], -90, 90, "lat"), num(p[1], -180, 180, "lng")) for p in pts],
                    "duration": num(r.get("duration"), 1, 172800, "duration")})
    return out

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype + ("; charset=utf-8" if "text" in ctype else ""))
        self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store"); self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if limited(self.client_address[0]): return self.send(429, {"error": "too many requests"})
        p = self.path
        if p == "/api/health": return self.send(200, {"status": "ok"})
        if p == "/api/alerts": return self.send(200, alerts.recent())
        if p.startswith("/api/"): return self.send(404, {"error": "not found"})
        f = ROOT / "static" / (p[1:] or "index.html")
        if f.exists() and f.is_file(): return self.send(200, f.read_bytes(), "text/html" if f.suffix == ".html" else "text/javascript")
        self.send(404, "Not found", "text/plain")

    def do_POST(self):
        if limited(self.client_address[0]): return self.send(429, {"error": "too many requests"})
        try:
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except Exception:
            return self.send(400, {"error": "invalid json"})

        try:
            if self.path == "/api/assess":
                return self.send(200, engine.calculate_flood_risk(num(b.get("lat"), -90, 90, "lat"), num(b.get("lng"), -180, 180, "lng"), sim_of(b)))
            if self.path == "/api/score":
                return self.send(200, engine.evaluate_alternative_routes(parse_routes(b), sim_of(b)))
            if self.path == "/api/subscribe":
                alerts.add(num(b.get("lat"), -90, 90, "lat"), num(b.get("lng"), -180, 180, "lng"), b.get("contact"), b.get("min_level", 2))
                return self.send(200, {"status": "subscribed"})
        except ValueError as e:
            return self.send(400, {"error": str(e)})
        except Exception as e:
            return self.send(500, {"error": "internal error"})
        self.send(404, {"error": "not found"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"VAYRA Server listening on port {port}...")
    ThreadingHTTPServer((os.environ.get("HOST", "127.0.0.1"), port), Handler).serve_forever()

# VAYRA PROJECT KNOWLEDGE GRAPH: vayra, road, flood, risk prediction mapping, alternative route evaluation.
