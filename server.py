"""RainRoute web server (standard library only):  python -m app.server"""
import json, os, re, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from . import engine, alerts

ROOT = Path(__file__).resolve().parent.parent
env = ROOT / ".env"
if env.exists():                                   # tiny .env loader, no extra packages
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

_hits = {}
def limited(ip, limit=90):                         # 90 requests / minute / IP
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
        if self.path in ("/", "/index.html"):
            html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
            return self.send(200, html.replace("YOUR_GOOGLE_MAPS_API_KEY", os.getenv("GOOGLE_MAPS_API_KEY", "")).encode(), "text/html")
        if self.path == "/api/alerts": return self.send(200, {"alerts": alerts.recent()})
        if self.path == "/api/health": return self.send(200, {"ok": True, "model_rows": engine.MODEL["trained_on"]})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        if limited(self.client_address[0]): return self.send(429, {"error": "too many requests"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 200_000: return self.send(413, {"error": "request too large"})
            b = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/assess":
                return self.send(200, engine.assess(num(b.get("lat"), -90, 90, "lat"), num(b.get("lng"), -180, 180, "lng"), sim_of(b)))
            if self.path == "/api/score":
                return self.send(200, {"routes": engine.score_routes(parse_routes(b), sim_of(b))})
            if self.path == "/api/subscribe":
                name = re.sub(r"[^\w .,'-]", "", str(b.get("name") or "My location"))[:60]
                contact = re.sub(r"[^\w@.+ -]", "", str(b.get("contact") or ""))[:80]
                mn = int(num(b.get("min_level", 2), 1, 3, "min_level"))
                sid = alerts.subscribe(name, num(b.get("lat"), -90, 90, "lat"), num(b.get("lng"), -180, 180, "lng"), contact, mn)
                return self.send(200, {"id": sid, "min_level": engine.LEVELS[mn]})
            self.send(404, {"error": "not found"})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, IndexError) as e:
            self.send(400, {"error": str(e)})
        except Exception as e:                      # upstream weather/terrain service failed
            self.send(502, {"error": "weather service unavailable: " + str(e)[:120]})

def main():
    host, port = os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8000"))
    alerts.start(engine.assess, int(os.getenv("SCAN_SECONDS", "600")))
    print(f"RainRoute running on http://{host}:{port}  (Google key {'set' if os.getenv('GOOGLE_MAPS_API_KEY') else 'MISSING'})")
    ThreadingHTTPServer((host, port), Handler).serve_forever()

if __name__ == "__main__":
    main()


# VAYRA PROJECT KNOWLEDGE GRAPH: vayra, road, flood, risk prediction mapping.

