"""Early-warning alerts: people subscribe a place; a background job re-checks it and notifies when risk rises."""
import json, os, sqlite3, threading, time, urllib.request
from pathlib import Path

DB = os.getenv("RAINROUTE_DB", str(Path(__file__).resolve().parent.parent / "rainroute.db"))
WEBHOOK = os.getenv("ALERT_WEBHOOK_URL", "")      # Slack / Discord incoming-webhook URL (optional)
NAMES = ["SAFE", "WATCH", "WARNING", "DANGER"]

def db():
    c = sqlite3.connect(DB, timeout=10)
    c.executescript("""CREATE TABLE IF NOT EXISTS subs(id INTEGER PRIMARY KEY, name TEXT, lat REAL, lng REAL, contact TEXT, min_level INT, last_level INT DEFAULT 0);
    CREATE TABLE IF NOT EXISTS log(id INTEGER PRIMARY KEY, ts REAL, name TEXT, level INT, text TEXT);""")
    return c

def subscribe(name, lat, lng, contact, min_level):
    c = db()
    with c:
        return c.execute("INSERT INTO subs(name,lat,lng,contact,min_level) VALUES(?,?,?,?,?)", (name, lat, lng, contact, min_level)).lastrowid

def recent(limit=8):
    c = db()
    return [{"ts": t, "place": n, "level": l, "text": x} for t, n, l, x in
            c.execute("SELECT ts,name,level,text FROM log ORDER BY id DESC LIMIT ?", (limit,))]

def notify(sub, res):
    text = f"{NAMES[res['level']]} flood risk at {sub['name']}: rain {res['rain_now']} mm/h, elevation {res['elevation']} m. Avoid low-lying roads."
    c = db()
    with c:
        c.execute("INSERT INTO log(ts,name,level,text) VALUES(?,?,?,?)", (time.time(), sub["name"], res["level"], text))
    print("[ALERT]", text, "| contact:", sub["contact"] or "-", flush=True)
    if WEBHOOK:
        try:
            msg = text + (f" (contact: {sub['contact']})" if sub["contact"] else "")
            req = urllib.request.Request(WEBHOOK, json.dumps({"text": msg, "content": msg}).encode(), {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=8)
        except Exception as e:
            print("[ALERT] webhook failed:", e, flush=True)

def scan(assess_fn):
    """Notify only when a place's level rises to/above its threshold (no repeat spam); reset when it calms down."""
    rows = db().execute("SELECT id,name,lat,lng,contact,min_level,last_level FROM subs").fetchall()
    for id_, name, lat, lng, contact, mn, last in rows:
        try:
            res = assess_fn(lat, lng)
        except Exception as e:
            print("[scan] skipped", name, e, flush=True); continue
        if res["level"] >= mn and res["level"] > last:
            notify({"name": name, "contact": contact}, res)
        c = db()
        with c:
            c.execute("UPDATE subs SET last_level=? WHERE id=?", (res["level"] if res["level"] >= mn else 0, id_))

def start(assess_fn, every=600):
    def loop():
        while True:
            time.sleep(every)
            try: scan(assess_fn)
            except Exception as e: print("[scan] error", e, flush=True)
    threading.Thread(target=loop, daemon=True).start()
