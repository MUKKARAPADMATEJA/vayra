# 🌊 Vayra: Autonomous IoT Flood Prediction Assistant
**Submission for OPTI FORGE '26 AI Hackathon**

![Vayra Dashboard](https://img.shields.io/badge/Status-Hackathon_Prototype-brightgreen) ![Tech Stack](https://img.shields.io/badge/Tech-Python_|_IoT_|_JS-blue)

## 📌 Chosen Vertical
**Open Innovation / Smart City Traffic Management**
Vayra acts as a smart, dynamic routing assistant for city planners and emergency responders. It provides logical decision-making based on user context (e.g., whether the user is driving a hospital ambulance or a civilian car) to provide safe, real-time routing during flash floods.

## 🧠 Approach and Algorithmic Logic
Vayra operates on an advanced algorithmic pipeline:
1. **The Predictive Model (Time-to-Flood)**: The system calculates the real-time **Rate of Rise**. A Sugeno fuzzy inference system calculates risk based on water depth and rise-rate. 
2. **Dynamic Time-Aware A* Routing**: The road network is modeled as a Directed Graph. The system instantly recalculates the shortest path using a Time-Dependent A* algorithm, routing users away from roads *before* they flood.

## ⚙️ How the solution works end-to-end
1. **Data Ingestion**: IoT sensors and radar rain forecasts are assimilated.
2. **Simulation**: A vectorised 2D hydrology grid model simulates runoff and drainage.
3. **Hyper-tuning**: A real-coded Genetic Algorithm (BLX-alpha crossover) optimizes 9 internal parameters.
4. **User Assistant Output**: The web dashboard alerts the user and dynamically reroutes their trip on the live map if their current path is predicted to flood.

## ⚠️ Assumptions or Operational Constraints Made
- The current hydrology model uses a simplified 2D bucket model, assuming constant drainage capacities across grid cells.
- The router operates as a heuristic time-aware A* search (label-setting approximation).
- IoT sensors are assumed to have a maximum noise variance of 0.01 meters.
- Testing is currently validated on synthetic grid storms (NumPy generated).
