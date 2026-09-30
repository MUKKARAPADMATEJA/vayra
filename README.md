# 🌊 Vayra: Predictive Flood Routing Algorithm
**Submission for OPTI FORGE '26 AI Hackathon - Algorithm Design Challenge**

![Vayra Dashboard](https://img.shields.io/badge/Status-Hackathon_Prototype-brightgreen) ![Tech Stack](https://img.shields.io/badge/Tech-Python_|_IoT_|_JS-blue)

## 📌 The Problem
Current navigation systems (like Google Maps) only route traffic away from flooded roads **after** users get stuck and report the blockage. This reactive approach causes massive traffic bottlenecks and delays emergency vehicles. 

## 💡 Our Solution
**Vayra** is an IoT-enabled algorithmic early-warning system. We place low-cost water sensors in known flood-prone underpasses. Instead of just reading water levels, our system runs a **predictive algorithm** to determine *when* the road will become impassable, and dynamically reroutes traffic *before* the bottleneck occurs.

---

## 🧠 Core Algorithm Design (How it works)

To solve the Algorithm Design Challenge, Vayra operates on a two-step algorithmic pipeline:

### 1. The Predictive Model (Time-to-Flood)
Instead of a static warning, the algorithm calculates the real-time **Rate of Rise**.
Rate of Rise = Δ Water Level / Δ Time
Estimated Time to Flood = (Danger Threshold - Current Level) / Rate of Rise
*This allows the system to warn users "Road will be flooded in 15 minutes" rather than waiting for it to happen.*

### 2. Dynamic Weight Dijkstra Routing
The road network is modeled as a Directed Graph.
* Under normal conditions, edge weights represent standard travel time.
* When the Predictive Model flags a node (e.g., an underpass) as nearing the danger threshold, the **Algorithm dynamically updates the graph weight of that specific edge to Infinity (∞)**.
* The system instantly recalculates the shortest path using Dijkstra's Algorithm, forcing the UI to suggest an alternative route.

---

## 📁 Repository Structure
* /frontend - Contains the interactive UI Dashboard mimicking the live map and sensor predictions.
* /algorithm - Contains the Python implementation of the dynamic Dijkstra routing and predictive logic.
* /hardware - Contains the C++/Arduino code for the ESP32 IoT sensor nodes.

## 🚀 How to Run the Demo (For Judges)
1. Navigate to the /frontend folder.
2. Download and open Vayra_Dashboard.html in any web browser.
3. Click **"Start Rain Simulation"** in the top right to watch the algorithmic rerouting happen in real-time as the simulated water level rises.

## 🔮 Future Scope (AI Integration)
In future iterations, Vayra will integrate historical rainfall data and live weather API forecasts into a **Random Forest regression model**. This will allow the system to predict underpass flooding hours in advance based purely on weather patterns, without waiting for the water to physically start rising.
