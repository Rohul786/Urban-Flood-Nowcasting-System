# 🌊 FloodGuard AI – Urban Flood Nowcasting & Emergency Response System
### Smart India Hackathon (SIH 2026) | Problem Statement SIH26085

---

## 🎯 Executive Summary
**FloodGuard AI** is an AI-powered, GIS-enabled disaster response command center designed to solve urban flash flooding in dense Indian metropolitan regions (demonstrated on the Kolkata metropolitan testbed).

Traditional municipal flood response is **reactive**—citizens report waterlogging only after roads submerge and traffic collapses. **FloodGuard AI** shifts the paradigm to **proactive 0–3 hour nowcasting**:
1. **Predicts** micro-spatial flood probability for individual municipal wards 0–3 hours ahead.
2. **Detects** flooded and blocked road corridors before vehicles enter hazard zones.
3. **Recommends** AI-optimized flood-safe evacuation routes bypassing inundated basins.
4. **Broadcasts** automated multi-tier early warnings (Critical, Warning, Watch, Info) with citizen action advisories.
5. **Monitors** critical infrastructure (hospitals, fire stations, emergency shelters, power grids).
6. **Explains** hydrological risk drivers using Explainable AI (XAI).

---

## 🏗️ System Architecture

```
WEATHER / RAINFALL DATA (Live OpenWeather API or Calibrated Simulator)
                   ↓
DATA PREPROCESSING & FEATURE ENGINEERING (11 Hydrological & Spatial Parameters)
                   ↓
AI/ML MODEL CLASSIFIER (Random Forest & Gradient Boosting Ensembles)
                   ↓
0–3 HOUR SPATIAL NOWCASTING ENGINE (T+0h, T+1h, T+2h, T+3h Horizons)
                   ↓
GIS FLOOD RISK & ROAD INUNDATION ANALYTICS (Folium Layered Mapping)
                   ↓
SAFE ROUTE RECOMMENDATION ENGINE (Risk-Weighted NetworkX Dijkstra/A*)
                   ↓
EARLY WARNING BROADCAST & DISASTER MANAGEMENT DASHBOARD
```

---

## 📁 Complete Project Structure

```
floodguard_ai/
│
├── app.py                          # Master Streamlit Disaster Command Center
├── requirements.txt                # Production dependency specifications
├── README.md                       # Complete documentation & SIH demo script
├── .env.example                    # Sample environment variables config
│
├── data/
│   ├── sample_flood_data.csv       # Synthetic historical dataset (1,800+ records)
│   ├── kolkata_wards.geojson       # 16 Kolkata municipal ward polygons & properties
│   ├── roads.geojson               # 31 arterial road network corridors & drainage tags
│   ├── infrastructure.geojson      # 20 emergency hospitals, fire stations, shelters
│   └── floodguard.db               # SQLite database for persistent alerts & logs
│
├── models/
│   ├── train_model.py              # Standalone training & benchmarking CLI script
│   └── flood_model.pkl             # Serialized Random Forest + Gradient Boosting bundle
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Hydrological data generator & GeoJSON loader
│   ├── preprocessing.py           # Feature scaling, column transformation, splitting
│   ├── model.py                    # ML model training, metrics, inference, XAI
│   ├── prediction.py               # 0–3 hour nowcasting calculation engine
│   ├── flood_mapping.py            # Folium GIS layered mapping & interactive popups
│   ├── routing.py                  # NetworkX flood-risk-aware safe routing engine
│   ├── alerts.py                   # Multi-tier early warning generator & DB logger
│   └── simulation.py               # Dynamic rainfall scenario orchestrator
│
├── utils/
│   ├── __init__.py
│   ├── db.py                       # SQLite database manager
│   ├── weather_api.py              # OpenWeatherMap connector with demo fallback
│   └── helpers.py                  # Disaster UI styling, KPI cards & badges
│
└── assets/
    └── custom.css                  # Modern dark navy/cyan disaster UI theme
```

---

## 🤖 How the Machine Learning Model Works

### 1. Feature Engineering (11 Input Parameters)
- `rainfall_1h`, `rainfall_3h`, `rainfall_6h`: Accumulated precipitation rates (mm).
- `elevation`: Terrain elevation in meters above Mean Sea Level (MSL).
- `slope`: Topographic slope gradient (degrees).
- `drainage_capacity`: Storm sewer discharge capacity ($m^3/s$).
- `impervious_surface`: Urban concrete and asphalt coverage percentage (%).
- `historical_flood_frequency`: Past flood events recorded over a 5-year period.
- `road_density`: Road network density ($km/km^2$).
- `tide_level_m`: Tidal head level in the adjoining Hooghly river ($m$).
- `soil_saturation`: Antecedent moisture and infiltration capacity index ($0.0 - 1.0$).

### 2. Model Performance Benchmarks
Trained on 1,800 calibrated hydrologic records with stratified 80/20 train-test split:
- **Random Forest Classifier**:
  - Accuracy: **93.3%**
  - Precision: **94.7%**
  - Recall: **89.9%**
  - F1-Score: **0.923**
  - ROC-AUC: **0.976**
- **Gradient Boosting Classifier**:
  - Accuracy: **93.6%**
  - Precision: **94.7%**
  - Recall: **90.6%**
  - F1-Score: **0.926**
  - ROC-AUC: **0.986**

### 3. Explainable AI (XAI) Factor Breakdown
For every predicted flood zone, FloodGuard AI calculates localized feature attributions answering *"Why is this location at high risk?"*:
- 🌧️ Heavy Rainfall Intensity: **+36.0%**
- 💧 Inadequate / Choked Drainage: **+24.0%**
- 📍 Low Terrain Elevation: **+18.0%**
- 🏙️ High Urban Concrete Coverage: **+12.0%**
- 📊 Historical Flood Vulnerability: **+6.0%**
- 🌊 High River Tidal Head: **+4.0%**

---

## 🧭 How the Flood-Aware Safe Route Engine Works
Standard navigation systems (e.g., standard Dijkstra) find the shortest physical path, which often directs emergency ambulances and citizens directly through flooded low-elevation basins.

FloodGuard AI builds an urban road graph in **NetworkX** and computes:
$$\text{Weight}_{\text{Safe}} = \text{Length} \times \left(1 + 25 \times \text{Risk}^3\right) + \text{Penalty}_{\text{Blocked}}$$
- **Shortest Route (Red dashed)**: Passes through low-elevation corridors (e.g. Central Avenue, Amherst St), suffering **85%+ flood exposure**.
- **AI Flood-Safe Route (Cyan solid)**: Automatically diverts traffic via elevated arterial expressways (e.g. EM Bypass, AJC Bose flyover), cutting flood exposure to **under 15%** with minimal detour distance.

---

## ⚡ Quick Start & Installation

### Prerequisites
- Python 3.10+ (Tested up to Python 3.14)
- Pip package manager

### 1. Clone or Open Project Directory
```bash
cd C:\Users\mathe\.gemini\antigravity\scratch\floodguard_ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure Live Weather API
Copy `.env.example` to `.env` and insert your OpenWeatherMap API key:
```bash
cp .env.example .env
```
*(If no API key is provided, the application automatically runs in high-fidelity simulation mode).*

### 4. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ⏱️ 3-Minute SIH Presentation & Live Demo Script

Follow this step-by-step walkthrough during jury evaluation:

| Step | Action | What to Demonstrate to Judges |
|---|---|---|
| **1** | Open **1. Dashboard** | Show top KPI cards (Rainfall 68mm, City Risk, 12 Affected Roads, 5 Active Alerts). |
| **2** | Move **Rainfall Slider** in Sidebar | Move from 25 mm (Normal) → 115 mm (Extreme). Show all KPI cards, map colors, and alert counters updating instantly. |
| **3** | Click **2. Live Flood Map** | Show interactive GIS layer with ward risk polygons, road statuses (Green, Yellow, Orange, Red), and hospital markers. Click a red ward to show real-time popup diagnostics. |
| **4** | Open **3. 0–3 Hour Forecast** | Switch between `Current`, `+1h`, `+2h`, `+3h` tabs to show the spatial expansion of flood water across municipal boundaries. |
| **5** | Open **4. Risk Analysis & XAI** | Select *Ward 42 (Burrabazar)*. Show the Explainable AI bar chart proving **why** it floods (Rainfall + Choked Drainage + Low Elevation). Show ML benchmark comparison table. |
| **6** | Open **5. Safe Routes** | Set Origin: `Shyambazar`, Destination: `Gariahat Junction`. Show the Red dashed shortest path vs Cyan solid AI Safe Route. Point out: **"Exposure reduced by 25.4% with only +3.4 km detour."** |
| **7** | Open **6. Alerts & Warnings** | Show multi-tier early warnings and click **Broadcast Siren & Emergency SMS Alert** button. |
| **8** | Open **7. Critical Infrastructure** | Filter by `HOSPITALS` to show SSKM Hospital and Calcutta Medical College flood risk status. |
| **9** | Open **8. Historical Analysis** | Show the **Traditional Response vs FloodGuard AI** quantified comparison graph (Lead time increased from 15m to 150m, response time reduced from 120m to 25m). |

---

## 🔌 Replacing Demo/Simulation Data with Real Production Data

| Component | Prototype / Demo Mode | Production / Real Implementation |
|---|---|---|
| **Weather Ingestion** | OpenWeatherMap fallback + slider simulator | IMD Doppler Radar (DWR) API, Automated Weather Stations (AWS) |
| **Terrain Elevation** | High-resolution synthetic DEM for Kolkata | Shuttle Radar Topography Mission (SRTM 30m) or Cartosat-1 DEM |
| **Drainage Telemetry** | Municipal GIS drainage shapefile with flow capacity | SCADA ultrasonic IoT water level sensors installed at major storm sluice gates |
| **Database** | SQLite (`floodguard.db`) | PostgreSQL + PostGIS spatial database cluster |
| **Alert Dispatch** | Streamlit UI + CAP gateway simulation | NDMA Common Alerting Protocol (CAP) SMS & Cell Broadcast gateway |

---

## ⚖️ Disclaimer
*This system is a student prototype developed for the Smart India Hackathon (SIH 2026). Calibrated synthetic hydrological data and realistic urban spatial features have been utilized to demonstrate the end-to-end operational workflow.*
