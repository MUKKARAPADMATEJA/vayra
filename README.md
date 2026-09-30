# VAYRA: Road Flood Risk Detection and Alternative Route Assistance

## 1. Problem
Urban areas frequently experience flash floods that paralyze traffic and endanger lives. Current GPS systems route drivers based on traffic, but ignore imminent road-level flood risks.

## 2. Solution
VAYRA is a smart routing assistant that evaluates real-time flood risk on planned routes and warns drivers before they reach impassable zones, offering safer alternatives.

## 3. Architecture
```text
User
 ↓
VAYRA Server
 ↓
Route Acquisition
 ↓
Environmental/Flood Data (Open-Meteo)
 ↓
Risk Engine
 ↓
Road Segment Analysis
 ↓
Warning + Alternative Route (Future Implementation)
 ↓
User
```

## 4. Technology Stack
- **Backend:** Python (Standard Library)
- **ML:** Tree-based predictive model (gradient boosting)
- **Data:** Open-Meteo API
- **Frontend:** HTML/JS/Leaflet (Google Maps integration)

## 5. Repository Structure
```text
server.py
engine.py
alerts.py
flood_model.json
benchmark.py
tests/
.env.example
static/
```

## 6. Installation
Clone the repository:
`git clone https://github.com/MUKKARAPADMATEJA/vayra.git`

## 7. Configuration
Copy the environment template:
`cp .env.example .env`
Edit `.env` to add your `GOOGLE_MAPS_API_KEY`.

## 8. Running the Application
Start the application from the root directory:
`python server.py`
Access the dashboard at `http://localhost:8000`.

## 9. API Usage
- `POST /api/assess` `{lat,lng,sim?}` : Get risk level and flood-proneness.
- `POST /api/score` `{routes:[{points:[[lat,lng]..],duration}],sim?}` : Evaluate route safety.

## 10. Risk Analysis Method
VAYRA combines terrain vulnerability (elevation differences) with a machine learning model trained on historical rainfall to predict flood probability, scaling the risk based on live rain intensity.

## 11. Route Analysis
The server breaks down GPS routes into segments and applies the Risk Analysis Method to each segment, tracking the maximum risk level encountered.

## 12. Alternative Route Logic
*Note: Full alternative route suggestion is a future/optional integration.* Currently, VAYRA evaluates provided routes and flags the safest available option from the inputs.

## 13. Testing
Run the test suite using pytest:
`pytest -v`

## 14. Benchmarking
Run the automated benchmark to measure inference latency:
`python benchmark.py`

## 15. Limitations
The current ML model relies on a weak signal (spatial CV AUC 0.63, accuracy 74%). The system acts as a heuristic early warning and does not consume live municipal waterlogging reports or user crowdsourced data (future integrations).

## 16. SDG 9
**UN SDG 9 — Industry, Innovation and Infrastructure**
VAYRA contributes to SDG 9 by building resilient transportation infrastructure, employing data-driven infrastructure decisions to reduce disruption, and enabling scalable technology-assisted road safety.

## 17. Future Improvements
- Consume live traffic data.
- Integrate real-time municipal flood reports and user crowdsourced data.
- Improve ML accuracy with high-resolution historical flood datasets.
