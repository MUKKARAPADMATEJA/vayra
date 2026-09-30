"""Offline tests: the weather/terrain services are faked, so no internet or API keys are needed.
Run:  python -m unittest discover -s tests -v"""
import json, os, sys, tempfile, threading, unittest, urllib.request, urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["RAINROUTE_DB"] = os.path.join(tempfile.mkdtemp(), "t.db")
os.environ["GOOGLE_MAPS_API_KEY"] = "TESTKEY123"
from app import engine, alerts, server

calls = {"n": 0}
def fake_fetch(url, tries=2, timeout=8):
    calls["n"] += 1
    n = len(url.split("latitude=")[1].split("&")[0].split(","))
    if "elevation" in url and "archive" not in url and "forecast" not in url:
        return {"elevation": [50 - (i % 7 == 3) * 12 for i in range(n)]}
    if "archive" in url:
        one = {"daily": {"precipitation_sum": [3] * 730}}
        return [one] * n if n > 1 else one
    one = {"current": {"precipitation": 0}, "hourly": {"precipitation": [1] * 12 + [0, 2, 25, 25, 3, 0, 0]}}
    return [one] * n if n > 1 else one
engine.fetch_json = fake_fetch

PATH = [(17.38 + i * .004, 78.48 + i * .002) for i in range(21)]

class Engine(unittest.TestCase):
    def setUp(self): engine._elev.clear(); engine._rain.clear(); engine._wx.clear()

    def test_model_matches_sklearn(self):
        with open(ROOT / "model" / "reference_predictions.json") as f: ref = json.load(f)
        for x, p in zip(ref["X"], ref["p"]): self.assertAlmostEqual(engine.predict(x), p, places=4)

    def test_assess_fields_and_sim(self):
        dry, storm = engine.assess(17.4, 78.5, None), engine.assess(17.4, 78.5, 100)
        for k in ("level", "name", "rain_now", "elevation", "ml_score", "prone", "annual_rain"): self.assertIn(k, dry)
        self.assertGreater(storm["level"], dry["level"]); self.assertGreaterEqual(storm["level"], 2)   # cloudburst => at least WARNING

    def test_route_timing_matters(self):
        short = engine.score_routes([{"points": PATH, "duration": 600}])[0]
        long_ = engine.score_routes([{"points": PATH, "duration": 3 * 3600}])[0]
        self.assertLess(short["pct"], long_["pct"])           # the long trip runs into the storm hours
        self.assertGreater(long_["eta"], 3 * 3600 - 1)

    def test_caching_saves_requests(self):
        engine.score_routes([{"points": PATH, "duration": 900}]); before = calls["n"]
        engine.score_routes([{"points": PATH, "duration": 900}])
        self.assertEqual(calls["n"], before)

class Alerts(unittest.TestCase):
    def test_notify_once_then_reset(self):
        alerts.subscribe("Home", 17.4, 78.5, "me@x.com", 2)
        lvl = {"v": 3}
        f = lambda la, lo: {"level": lvl["v"], "rain_now": 40, "elevation": 12}
        alerts.scan(f); alerts.scan(f)
        self.assertEqual(len(alerts.recent()), 1)              # no repeat spam
        lvl["v"] = 0; alerts.scan(f); lvl["v"] = 2; alerts.scan(f)
        self.assertEqual(len(alerts.recent()), 2)              # calmed down, then rose again

class Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
    @classmethod
    def tearDownClass(cls): cls.srv.shutdown()

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode() if body is not None else None, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r: return r.status, r.read()
        except urllib.error.HTTPError as e: return e.code, e.read()

    def test_home_injects_key(self):
        s, b = self.call("/"); self.assertEqual(s, 200); self.assertIn(b"TESTKEY123", b); self.assertNotIn(b"YOUR_GOOGLE_MAPS_API_KEY", b)
    def test_assess_and_score(self):
        s, b = self.call("/api/assess", {"lat": 17.4, "lng": 78.5, "sim": 60}); self.assertEqual(s, 200); self.assertIn("level", json.loads(b))
        s, b = self.call("/api/score", {"routes": [{"points": PATH, "duration": 1200}]}); self.assertEqual(s, 200)
        self.assertEqual(len(json.loads(b)["routes"]), 1)
    def test_validation(self):
        self.assertEqual(self.call("/api/assess", {"lat": 999, "lng": 0})[0], 400)
        self.assertEqual(self.call("/api/score", {"routes": []})[0], 400)
        self.assertEqual(self.call("/api/nope", {})[0], 404)
    def test_subscribe_and_feed(self):
        s, b = self.call("/api/subscribe", {"lat": 17.4, "lng": 78.5, "contact": "a@b.com<script>"}); self.assertEqual(s, 200)
        self.assertEqual(self.call("/api/alerts")[0], 200)

if __name__ == "__main__": unittest.main()
