# RainRoute – RoadFlood Early Warning System

Live GPS + live rain forecast + terrain + a machine-learning model → flood risk where you are,
and the safest route to any place you search, with risk % and arrival time for every route.

```
Browser (Google Maps, GPS)  ──►  Python server  ──►  Open-Meteo (rain forecast, elevation, yearly rain)
       index.html          server.py       ML model (flood_model.json)
                                  engine.py       SQLite alerts (alerts.py) ──► console / Slack / Discord
```

## Run it (5 minutes, no pip installs)
1. Google Cloud Console → enable **Maps JavaScript API, Places API, Directions API** → create an API key
   (restrict it to your website/`localhost` under "HTTP referrers").
2. `cp .env.example .env` and paste the key into `GOOGLE_MAPS_API_KEY`.
3. `python -m server` and open **http://localhost:8000** (allow location access).
4. Tests (offline, no key needed): `python -m unittest discover -s tests -v`

Phone test on the same Wi-Fi: browsers only allow GPS on HTTPS or localhost, so use a tunnel
(e.g. `ngrok http 8000`) or deploy (below).

## How the risk is calculated
`risk = rain intensity × ground wetness × vulnerability`, mapped to Safe / Watch / Warning / Danger.
- **Rain** – Open-Meteo hourly forecast at ~5 points along each route, *at the hour you will be there*.
  A short trip that ends before the storm scores safe; a longer one into the storm scores higher.
- **Ground wetness** – rain in the last 12 hours.
- **Vulnerability (70%)** – how much lower the road sits than the ground ~400 m around it (underpasses, dips).
- **Vulnerability (30%)** – ML flood-proneness of the area from latitude, longitude, yearly rainfall,
  elevation and slope.
- **Arrival time** – Google live-traffic time, plus up to +30% for heavy rain.

## The ML model (``)
Trained on `flood_dataset_classification.csv` with `python train_model.py flood_dataset_classification.csv`.
Findings about the dataset (all handled in the script):
- `Disaster Type` equals the label exactly → removed (would give a fake 100%).
- Two rainfall values are fill-in constants, one per class (521 flood rows, 694 non-flood rows) → rows dropped, 5,019 remain.
- Deaths / affected / duration / distance / year exist only after a flood → not used.
- The "non-flood" rows are *other disasters*, not safe places, so the model measures flood-proneness relative to other disasters.

Spatial cross-validation (tested on unseen regions): gradient boosting **AUC 0.63, accuracy 74%**.
That is a weak signal, which is why it only carries 30% of the score. For a stronger model, add real
"never flooded" locations or flood records for your own city and retrain.

## API
| Endpoint | Purpose |
|---|---|
| `POST /api/assess` `{lat,lng,sim?}` | risk level, rain, elevation, ML flood-proneness at a place |
| `POST /api/score` `{routes:[{points:[[lat,lng]..],duration}],sim?}` | risk %, timing, ETA for each route |
| `POST /api/subscribe` `{lat,lng,contact?,min_level?}` | early-warning alerts for a place |
| `GET /api/alerts` / `GET /api/health` | recent alerts / status |

`sim` (mm/h) overrides live rain, for demos and tests. The server validates all input, limits requests
to 90/min/IP, caches upstream data (rain 9 min, terrain and yearly rain permanently) and never exposes
subscriber contacts.

## Alerts
Every 10 minutes (`SCAN_SECONDS`) the server re-checks each subscribed place and alerts **once** when risk
rises to the chosen level (no spam; resets after it calms down). Alerts print to the console and, if
`ALERT_WEBHOOK_URL` is a Slack/Discord webhook, are posted there. SMS/WhatsApp needs a provider such as
Twilio: add its call inside `notify()` in `alerts.py`.

## Deploy
`docker build -t rainroute . && docker run -p 8000:8000 --env-file .env rainroute`
(or any host that runs Python, e.g. Render/Railway). Put `HOST=0.0.0.0` and your key in the environment,
and add your site URL to the Google key's referrer list.

## Limits (be upfront about these in your report)
- Risk is an estimate from rain, terrain and a weak ML signal – not an official warning, and it has not
  been validated against real flooded roads.
- Elevation data is ~90 m resolution, so small underpasses can be missed.
- Google Maps/Directions need billing enabled; Google is phasing out the older Directions API, so
  new projects may need the Routes API instead.
- Weather calls need internet from the server; if Open-Meteo is down the API returns a clear 502 error.

## Socio-Technical Impact
This project directly contributes to **SDG 9: Industry, Innovation & Infrastructure** by building resilient road infrastructure, enabling real-time emergency routing, and providing critical decision support to prevent logistics bottlenecks during severe weather events.


