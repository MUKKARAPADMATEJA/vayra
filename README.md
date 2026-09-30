# VAYRA: Road Flood Risk Detection and Alternative Route Assistance

## 1. Problem
Urban areas frequently experience flash floods that paralyze traffic and endanger lives. Current GPS systems route drivers based on traffic, but ignore imminent road-level flood risks.

## 2. Solution
VAYRA is a smart routing assistant that evaluates real-time flood risk on planned routes and warns drivers before they reach impassable zones, offering safer alternatives.

## Implementation Coverage Map (Core Concepts)

| VAYRA Capability | Implementation |
| :--- | :--- |
| **Rainfall Analysis** | `engine.weather()`, `engine.annual_rain()` |
| **Road Flood Risk Detection** | `engine.detect_road_flood_risk()` |
| **Road Segment Analysis** | `engine.analyze_road_segments()` |
| **Risk Classification** | `engine.classify_risk_level()` (LOW, MODERATE, HIGH, CRITICAL) |
| **Route Risk Evaluation** | `engine.evaluate_route_risk()` |
| **Early Warning** | `engine.generate_early_warning()` |
| **Alternative Route Decision** | `engine.evaluate_alternative_routes()` |

## 3. Architecture & SDG 9 Connection

**UN SDG 9 — Industry, Innovation and Infrastructure**
VAYRA supports SDG 9 by using data-driven technology to improve resilience and decision support for transportation infrastructure during rainfall and flooding events. Note: VAYRA is a prototype decision-support system and does not claim to directly control infrastructure.

**VAYRA Algorithm Output → Infrastructure Relevance**
- **Flood-Risk Detection** → identifies potentially vulnerable road sections.
- **Road-Level Risk Classification** → provides localized infrastructure-risk information.
- **Route-Risk Evaluation** → connects infrastructure conditions with actual travel decisions.
- **Early Warning** → provides advance awareness of potentially disrupted road sections.
- **Alternative-Route Analysis** → supports continuity of transportation when a lower-risk route is available.

```text
User
 ↓
VAYRA Server
 ↓
Route Acquisition
 ↓
Environmental/Flood Data (Open-Meteo)
 ↓
Flood-Risk Analysis (engine.py)
 ↓
Road-Level Risk Analysis
 ↓
Risk Classification & Early Warning
 ↓
Alternative Route Evaluation
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
- `POST /api/score` `{routes:[{points:[[lat,lng]..],duration}],sim?}` : Evaluate route safety and alternatives.

## 10. Risk Analysis Method
VAYRA combines terrain vulnerability (elevation differences) with a machine learning model trained on historical rainfall to predict flood probability, scaling the risk based on live rain intensity.

## 11. Route Analysis
The server breaks down GPS routes into segments and applies the Risk Analysis Method to each segment, tracking the maximum risk level encountered.

## 12. Alternative Route Logic
VAYRA scores multiple proposed alternative routes dynamically against the localized flood map, returning the recommended safe path by weighing crossing severity and travel delay.

## 13. Testing
Run the comprehensive test suite to validate 100% of declared concepts:
`pytest -v`

## 14. Benchmarking
Run the automated benchmark to measure real inference processing latency over user endpoints:
`python benchmark.py`

## 15. Limitations
The current ML model relies on a weak signal (spatial CV AUC 0.63, accuracy 74%). The system acts as a heuristic early warning and does not consume live municipal waterlogging reports or user crowdsourced data.

## 16. Future Improvements
*(Future / Optional Integrations not yet implemented)*
- Consume live traffic data.
- Integrate real-time municipal flood reports and user crowdsourced data.
- Improve ML accuracy with high-resolution historical flood datasets.
