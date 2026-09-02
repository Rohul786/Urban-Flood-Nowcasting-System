"""
FloodGuard AI - Data Loader & GIS Asset Generator
Generates realistic hydrologic simulation datasets and Kolkata GeoJSON layers
for wards, roads, drainage corridors, and critical infrastructure.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CSV_PATH = os.path.join(DATA_DIR, "sample_flood_data.csv")
WARDS_GEOJSON = os.path.join(DATA_DIR, "kolkata_wards.geojson")
ROADS_GEOJSON = os.path.join(DATA_DIR, "roads.geojson")
INFRA_GEOJSON = os.path.join(DATA_DIR, "infrastructure.geojson")


def generate_sample_flood_dataset(n_samples: int = 1800, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic hydrologic dataset modeling urban flood physics."""
    np.random.seed(random_state)

    lat_min, lat_max = 22.46, 22.65
    lon_min, lon_max = 88.28, 88.48

    lats = np.random.uniform(lat_min, lat_max, n_samples)
    lons = np.random.uniform(lon_min, lon_max, n_samples)

    rainfall_1h = np.random.exponential(scale=25.0, size=n_samples)
    rainfall_1h = np.clip(rainfall_1h, 0, 160.0)
    rainfall_3h = rainfall_1h * np.random.uniform(1.6, 2.8, n_samples) + np.random.normal(5, 2, n_samples)
    rainfall_3h = np.clip(rainfall_3h, rainfall_1h, 300.0)
    rainfall_6h = rainfall_3h * np.random.uniform(1.2, 1.9, n_samples) + np.random.normal(8, 3, n_samples)
    rainfall_6h = np.clip(rainfall_6h, rainfall_3h, 450.0)

    elevation = np.random.normal(loc=6.5, scale=2.2, size=n_samples)
    elevation = np.clip(elevation, 2.0, 15.0)

    slope = np.random.gamma(shape=1.5, scale=0.8, size=n_samples)
    slope = np.clip(slope, 0.1, 5.5)

    drainage_capacity = np.random.uniform(8.0, 42.0, n_samples)
    impervious_surface = np.random.uniform(45.0, 95.0, n_samples)
    road_density = np.random.uniform(5.0, 26.0, n_samples)
    historical_flood_frequency = np.random.poisson(lam=3.5, size=n_samples)
    tide_level_m = np.random.uniform(1.0, 4.2, n_samples)
    soil_saturation = np.random.uniform(0.3, 0.98, n_samples)

    z_score = (
        0.38 * (rainfall_1h / 60.0) +
        0.28 * (rainfall_3h / 120.0) +
        0.18 * (rainfall_6h / 200.0) -
        0.30 * ((elevation - 2.0) / 13.0) -
        0.12 * (slope / 5.0) -
        0.25 * ((drainage_capacity - 8.0) / 34.0) +
        0.22 * ((impervious_surface - 45.0) / 50.0) +
        0.15 * (historical_flood_frequency / 10.0) +
        0.18 * ((tide_level_m - 1.0) / 3.2) +
        0.15 * (soil_saturation - 0.5)
    )

    flood_prob = 1.0 / (1.0 + np.exp(-3.5 * (z_score - 0.35)))
    flood_prob = np.clip(flood_prob, 0.01, 0.99)
    flood_occurred = (flood_prob >= 0.50).astype(int)

    df = pd.DataFrame({
        "latitude": np.round(lats, 5),
        "longitude": np.round(lons, 5),
        "rainfall_1h": np.round(rainfall_1h, 2),
        "rainfall_3h": np.round(rainfall_3h, 2),
        "rainfall_6h": np.round(rainfall_6h, 2),
        "elevation": np.round(elevation, 2),
        "slope": np.round(slope, 2),
        "drainage_capacity": np.round(drainage_capacity, 2),
        "impervious_surface": np.round(impervious_surface, 1),
        "historical_flood_frequency": historical_flood_frequency,
        "road_density": np.round(road_density, 2),
        "tide_level_m": np.round(tide_level_m, 2),
        "soil_saturation": np.round(soil_saturation, 2),
        "flood_probability": np.round(flood_prob, 4),
        "flood_occurred": flood_occurred
    })

    return df


def generate_kolkata_wards_geojson() -> Dict[str, Any]:
    """Generate ward polygons and hydrological vulnerability profiles for Kolkata."""
    wards = [
        {"id": "W-42", "name": "Ward 42 (Burrabazar / Central Basin)", "lat": 22.5840, "lon": 88.3530, "elev": 3.8, "slope": 0.4, "drain": 12.0, "imp": 94, "hist_freq": 11, "rad": 0.012},
        {"id": "W-45", "name": "Ward 45 (BBD Bagh / Secretariat)", "lat": 22.5710, "lon": 88.3490, "elev": 5.2, "slope": 0.6, "drain": 24.0, "imp": 88, "hist_freq": 5, "rad": 0.013},
        {"id": "W-63", "name": "Ward 63 (Park Street / Camac St)", "lat": 22.5510, "lon": 88.3540, "elev": 4.5, "slope": 0.5, "drain": 18.0, "imp": 90, "hist_freq": 8, "rad": 0.012},
        {"id": "W-69", "name": "Ward 69 (Ballygunge / Gariahat)", "lat": 22.5280, "lon": 88.3680, "elev": 5.8, "slope": 0.8, "drain": 28.0, "imp": 82, "hist_freq": 4, "rad": 0.014},
        {"id": "W-83", "name": "Ward 83 (Kalighat / Rashbehari)", "lat": 22.5200, "lon": 88.3450, "elev": 4.1, "slope": 0.5, "drain": 15.0, "imp": 89, "hist_freq": 9, "rad": 0.013},
        {"id": "W-88", "name": "Ward 88 (Alipore / Chetla)", "lat": 22.5320, "lon": 88.3320, "elev": 6.8, "slope": 1.2, "drain": 32.0, "imp": 70, "hist_freq": 3, "rad": 0.015},
        {"id": "W-89", "name": "Ward 89 (Tollygunge / Circular Canal)", "lat": 22.5020, "lon": 88.3440, "elev": 3.9, "slope": 0.4, "drain": 14.0, "imp": 85, "hist_freq": 10, "rad": 0.014},
        {"id": "W-93", "name": "Ward 93 (Jadavpur / Prince Anwar)", "lat": 22.4980, "lon": 88.3690, "elev": 5.5, "slope": 0.7, "drain": 22.0, "imp": 80, "hist_freq": 6, "rad": 0.015},
        {"id": "W-101", "name": "Ward 101 (EM Bypass / Ruby Hub)", "lat": 22.5130, "lon": 88.3990, "elev": 6.2, "slope": 1.0, "drain": 34.0, "imp": 70, "hist_freq": 3, "rad": 0.016},
        {"id": "W-108", "name": "Ward 108 (Kasba / Anandapur)", "lat": 22.5260, "lon": 88.4040, "elev": 5.8, "slope": 0.8, "drain": 30.0, "imp": 74, "hist_freq": 4, "rad": 0.015},
        {"id": "W-115", "name": "Ward 115 (Behala Chowrasta)", "lat": 22.4890, "lon": 88.3180, "elev": 3.4, "slope": 0.3, "drain": 10.0, "imp": 92, "hist_freq": 14, "rad": 0.016},
        {"id": "W-122", "name": "Ward 122 (Taratala / Hyde Road)", "lat": 22.5080, "lon": 88.3090, "elev": 3.7, "slope": 0.4, "drain": 13.0, "imp": 91, "hist_freq": 12, "rad": 0.015},
        {"id": "W-130", "name": "Ward 130 (Salt Lake Sector V)", "lat": 22.5760, "lon": 88.4320, "elev": 6.9, "slope": 1.2, "drain": 38.0, "imp": 68, "hist_freq": 2, "rad": 0.016},
        {"id": "W-135", "name": "Ward 135 (New Town Action Area I)", "lat": 22.5920, "lon": 88.4680, "elev": 7.4, "slope": 1.4, "drain": 42.0, "imp": 62, "hist_freq": 1, "rad": 0.018},
        {"id": "W-142", "name": "Ward 142 (Howrah Bank / Station)", "lat": 22.5850, "lon": 88.3390, "elev": 4.3, "slope": 0.5, "drain": 19.0, "imp": 89, "hist_freq": 8, "rad": 0.014},
        {"id": "W-144", "name": "Ward 144 (Amherst St / Sealdah)", "lat": 22.5690, "lon": 88.3710, "elev": 3.6, "slope": 0.3, "drain": 11.0, "imp": 93, "hist_freq": 13, "rad": 0.013}
    ]

    features = []
    for w in wards:
        lat, lon, r = w["lat"], w["lon"], w["rad"]
        angles = np.linspace(0, 2 * np.pi, 7)
        coords = [[round(lon + r * np.cos(a) * 1.1, 5), round(lat + r * np.sin(a) * 0.9, 5)] for a in angles]
        
        feature = {
            "type": "Feature",
            "properties": {
                "ward_id": w["id"],
                "ward_name": w["name"],
                "elevation": w["elev"],
                "slope": w["slope"],
                "drainage_capacity": w["drain"],
                "impervious_surface": w["imp"],
                "historical_flood_frequency": w["hist_freq"],
                "center_lat": w["lat"],
                "center_lon": w["lon"],
                "road_density": round(np.random.uniform(14.0, 22.0), 1),
                "soil_saturation": 0.75,
                "tide_level_m": 2.8
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            }
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def generate_kolkata_roads_geojson() -> Dict[str, Any]:
    """Generate connected urban road network corridors in Kolkata for routing and risk scoring."""
    roads = [
        # North - Central Spine (Low lying, historical waterlogging)
        {"id": "R01", "name": "Central Avenue (CR Avenue)", "u": "Shyambazar", "v": "Esplanade", "coords": [[88.370, 22.598], [88.365, 22.580], [88.358, 22.565]], "elev": 3.8, "drain": "Poor", "len_km": 4.2},
        {"id": "R02", "name": "Park Street Corridor", "u": "Park Street Metro", "v": "Park Circus", "coords": [[88.350, 22.553], [88.357, 22.551], [88.368, 22.544]], "elev": 4.5, "drain": "Fair", "len_km": 2.4},
        
        # Eastern Elevated Expressway (High drainage, elevated flyover/corridor)
        {"id": "R03A", "name": "Shyambazar-Ultadanga Connector", "u": "Shyambazar", "v": "Ultadanga", "coords": [[88.370, 22.598], [88.384, 22.593], [88.398, 22.588]], "elev": 6.2, "drain": "Good", "len_km": 2.2},
        {"id": "R03B", "name": "EM Bypass (North-South Express)", "u": "Ultadanga", "v": "Ruby Crossing", "coords": [[88.398, 22.588], [88.401, 22.555], [88.399, 22.513]], "elev": 6.8, "drain": "Good", "len_km": 8.5},
        {"id": "R03C", "name": "EM Bypass South Extension", "u": "Ruby Crossing", "v": "Jadavpur PS", "coords": [[88.399, 22.513], [88.385, 22.505], [88.369, 22.498]], "elev": 6.4, "drain": "Good", "len_km": 3.4},

        {"id": "R04", "name": "AJC Bose Road Arterial", "u": "Exide Crossing", "v": "Moulali", "coords": [[88.347, 22.540], [88.364, 22.546], [88.369, 22.560]], "elev": 5.1, "drain": "Fair", "len_km": 3.8},
        {"id": "R05", "name": "Strand Road (Riverfront Basin)", "u": "Howrah Approach", "v": "Prinsep Ghat", "coords": [[88.346, 22.582], [88.342, 22.568], [88.337, 22.552]], "elev": 3.5, "drain": "Choked", "len_km": 3.9},
        {"id": "R06", "name": "Diamond Harbour Road (Behala)", "u": "Taratala Crossing", "v": "Behala Chowrasta", "coords": [[88.318, 22.512], [88.318, 22.489]], "elev": 3.3, "drain": "Choked", "len_km": 3.1},
        {"id": "R07", "name": "Gariahat Road", "u": "Ballygunge Phari", "v": "Gariahat Junction", "coords": [[88.366, 22.534], [88.368, 22.524]], "elev": 5.6, "drain": "Good", "len_km": 1.5},
        {"id": "R08", "name": "VIP Road (Airport Expressway)", "u": "Ultadanga", "v": "Airport Gate 1", "coords": [[88.398, 22.588], [88.420, 22.610], [88.440, 22.645]], "elev": 6.8, "drain": "Good", "len_km": 7.4},
        {"id": "R09", "name": "Red Road (Maidan Bypass)", "u": "Esplanade", "v": "PTS Race Course", "coords": [[88.348, 22.562], [88.342, 22.545]], "elev": 6.2, "drain": "Good", "len_km": 2.1},
        {"id": "R10", "name": "Rashbehari Avenue", "u": "Chetla Bridge", "v": "Gariahat Junction", "coords": [[88.342, 22.518], [88.355, 22.520], [88.368, 22.524]], "elev": 4.8, "drain": "Fair", "len_km": 2.9},
        {"id": "R11", "name": "Amherst Street Corridor", "u": "Shyambazar", "v": "Sealdah Station", "coords": [[88.370, 22.598], [88.368, 22.578], [88.371, 22.565]], "elev": 3.2, "drain": "Choked", "len_km": 3.6},
        {"id": "R12", "name": "Sector V Main Ring (Salt Lake)", "u": "Ultadanga", "v": "Technopolis", "coords": [[88.398, 22.588], [88.432, 22.576], [88.441, 22.578]], "elev": 6.0, "drain": "Good", "len_km": 4.5},
        {"id": "R13", "name": "Prince Anwar Shah Road", "u": "Tollygunge Phari", "v": "Jadavpur PS", "coords": [[88.345, 22.502], [88.360, 22.500], [88.369, 22.498]], "elev": 5.2, "drain": "Fair", "len_km": 2.7},
        {"id": "R14", "name": "New Town Major Arterial", "u": "Technopolis", "v": "City Centre 2", "coords": [[88.441, 22.578], [88.468, 22.592], [88.475, 22.615]], "elev": 6.7, "drain": "Good", "len_km": 5.5},
        {"id": "R15", "name": "Hyde Road Industrial Connector", "u": "Taratala Crossing", "v": "Prinsep Ghat", "coords": [[88.318, 22.512], [88.337, 22.552]], "elev": 3.4, "drain": "Choked", "len_km": 4.8},
        {"id": "R16", "name": "Shakespeare Sarani", "u": "Exide Crossing", "v": "Park Circus", "coords": [[88.349, 22.545], [88.368, 22.544]], "elev": 5.0, "drain": "Fair", "len_km": 2.0},
        {"id": "R17", "name": "Southern Avenue (Lake Side)", "u": "Golpark", "v": "Kalighat Metro", "coords": [[88.365, 22.512], [88.350, 22.514]], "elev": 5.4, "drain": "Good", "len_km": 1.6},
        {"id": "R18", "name": "Beliaghata Main Road", "u": "Sealdah Station", "v": "Ultadanga", "coords": [[88.371, 22.565], [88.398, 22.588]], "elev": 4.1, "drain": "Poor", "len_km": 3.2},
        {"id": "R19", "name": "Chowringhee Road", "u": "Esplanade", "v": "Park Street Metro", "coords": [[88.358, 22.565], [88.350, 22.553]], "elev": 5.5, "drain": "Good", "len_km": 1.4},
        {"id": "R20", "name": "Jawaharlal Nehru Road", "u": "Park Street Metro", "v": "Exide Crossing", "coords": [[88.350, 22.553], [88.347, 22.540]], "elev": 5.3, "drain": "Good", "len_km": 1.6},
        {"id": "R21", "name": "Syed Amir Ali Avenue", "u": "Park Circus", "v": "Ballygunge Phari", "coords": [[88.368, 22.544], [88.366, 22.534]], "elev": 5.4, "drain": "Good", "len_km": 1.3},
        {"id": "R22", "name": "Ruby-Gariahat Connector", "u": "Ruby Crossing", "v": "Gariahat Junction", "coords": [[88.399, 22.513], [88.368, 22.524]], "elev": 5.8, "drain": "Good", "len_km": 3.5},
        {"id": "R23", "name": "Jadavpur-Golpark Road", "u": "Jadavpur PS", "v": "Golpark", "coords": [[88.369, 22.498], [88.365, 22.512]], "elev": 5.6, "drain": "Good", "len_km": 1.7},
        {"id": "R24", "name": "Golpark-Gariahat Link", "u": "Golpark", "v": "Gariahat Junction", "coords": [[88.365, 22.512], [88.368, 22.524]], "elev": 5.5, "drain": "Good", "len_km": 1.2},
        {"id": "R25", "name": "Alipore-Chetla Corridor", "u": "Exide Crossing", "v": "Chetla Bridge", "coords": [[88.347, 22.540], [88.342, 22.518]], "elev": 6.4, "drain": "Good", "len_km": 2.6},
        {"id": "R26", "name": "Chetla-Taratala Link", "u": "Chetla Bridge", "v": "Taratala Crossing", "coords": [[88.342, 22.518], [88.318, 22.512]], "elev": 4.5, "drain": "Fair", "len_km": 2.7},
        {"id": "R27", "name": "Kalighat-Chetla Bridge Link", "u": "Kalighat Metro", "v": "Chetla Bridge", "coords": [[88.350, 22.514], [88.342, 22.518]], "elev": 4.6, "drain": "Fair", "len_km": 1.1},
        {"id": "R28", "name": "Sealdah-Moulali Link", "u": "Sealdah Station", "v": "Moulali", "coords": [[88.371, 22.565], [88.369, 22.560]], "elev": 4.2, "drain": "Poor", "len_km": 0.8},
        {"id": "R29", "name": "Esplanade-Howrah Bridge", "u": "Esplanade", "v": "Howrah Approach", "coords": [[88.358, 22.565], [88.346, 22.582]], "elev": 4.5, "drain": "Fair", "len_km": 2.3}
    ]

    features = []
    for r in roads:
        feature = {
            "type": "Feature",
            "properties": {
                "road_id": r["id"],
                "road_name": r["name"],
                "from_node": r["u"],
                "to_node": r["v"],
                "elevation": r["elev"],
                "drainage_condition": r["drain"],
                "length_km": r["len_km"],
                "speed_limit_kmh": 40
            },
            "geometry": {
                "type": "LineString",
                "coordinates": r["coords"]
            }
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def generate_infrastructure_geojson() -> Dict[str, Any]:
    """Generate critical emergency infrastructure points (Hospitals, Fire, Shelters, Police, Power)."""
    facilities = [
        {"id": "H01", "name": "SSKM Multi-Speciality Hospital", "type": "HOSPITAL", "lat": 22.5395, "lon": 88.3435, "elev": 6.5, "drain": 28.0, "beds": 1800},
        {"id": "H02", "name": "Calcutta Medical College & Hospital", "lat": 22.5732, "lon": 88.3620, "elev": 4.0, "drain": 14.0, "type": "HOSPITAL", "beds": 1200},
        {"id": "H03", "name": "AMRI Hospital Dhakuria", "lat": 22.5120, "lon": 88.3645, "elev": 5.4, "drain": 24.0, "type": "HOSPITAL", "beds": 350},
        {"id": "H04", "name": "Ruby General Hospital (EM Bypass)", "lat": 22.5135, "lon": 88.3995, "elev": 4.8, "drain": 22.0, "type": "HOSPITAL", "beds": 400},
        {"id": "H05", "name": "Apollo Gleneagles Hospital", "lat": 22.5715, "lon": 88.4040, "elev": 5.9, "drain": 30.0, "type": "HOSPITAL", "beds": 550},
        {"id": "H06", "name": "Fortis Hospital Anandapur", "lat": 22.5240, "lon": 88.4020, "elev": 4.5, "drain": 18.0, "type": "HOSPITAL", "beds": 300},
        
        {"id": "F01", "name": "Central Fire HQ (Free School St)", "lat": 22.5560, "lon": 88.3525, "elev": 5.0, "drain": 20.0, "type": "FIRE_STATION", "vehicles": 14},
        {"id": "F02", "name": "Alipore Fire Station", "lat": 22.5310, "lon": 88.3340, "elev": 6.6, "drain": 32.0, "type": "FIRE_STATION", "vehicles": 8},
        {"id": "F03", "name": "Behala Fire Station", "lat": 22.4920, "lon": 88.3190, "elev": 3.5, "drain": 11.0, "type": "FIRE_STATION", "vehicles": 6},
        {"id": "F04", "name": "Salt Lake Sector V Fire Station", "lat": 22.5780, "lon": 88.4350, "elev": 6.1, "drain": 30.0, "type": "FIRE_STATION", "vehicles": 10},

        {"id": "P01", "name": "Kolkata Police HQ (Lalbazar)", "lat": 22.5710, "lon": 88.3520, "elev": 5.4, "drain": 25.0, "type": "POLICE_HQ", "personnel": 450},
        {"id": "P02", "name": "Park Street Police Station", "lat": 22.5515, "lon": 88.3560, "elev": 4.6, "drain": 19.0, "type": "POLICE_STATION", "personnel": 120},
        {"id": "P03", "name": "Jadavpur Police Station", "lat": 22.4985, "lon": 88.3685, "elev": 5.6, "drain": 22.0, "type": "POLICE_STATION", "personnel": 110},

        {"id": "S01", "name": "Netaji Indoor Stadium Relief Center", "lat": 22.5670, "lon": 88.3410, "elev": 6.2, "drain": 28.0, "type": "SHELTER", "capacity": 3500},
        {"id": "S02", "name": "Salt Lake Yuva Bharati Relief Camp", "lat": 22.5695, "lon": 88.4060, "elev": 6.5, "drain": 35.0, "type": "SHELTER", "capacity": 5000},
        {"id": "S03", "name": "Rabindra Sarobar Community Shelter", "lat": 22.5140, "lon": 88.3560, "elev": 5.5, "drain": 25.0, "type": "SHELTER", "capacity": 1500},
        {"id": "S04", "name": "Behala Indoor Relief Center", "lat": 22.4850, "lon": 88.3210, "elev": 4.2, "drain": 15.0, "type": "SHELTER", "capacity": 1200},

        {"id": "PW01", "name": "CESC Central Power Substation", "lat": 22.5620, "lon": 88.3580, "elev": 4.8, "drain": 20.0, "type": "POWER_STATION", "mw": 250},
        {"id": "PW02", "name": "Salt Lake Sector V Grid Substation", "lat": 22.5810, "lon": 88.4310, "elev": 6.3, "drain": 34.0, "type": "POWER_STATION", "mw": 180},
        {"id": "PW03", "name": "Taratala Industrial Grid Node", "lat": 22.5050, "lon": 88.3120, "elev": 3.6, "drain": 12.0, "type": "POWER_STATION", "mw": 150}
    ]

    features = []
    for f in facilities:
        feature = {
            "type": "Feature",
            "properties": {
                "id": f["id"],
                "name": f["name"],
                "type": f["type"],
                "elevation": f["elev"],
                "drainage_capacity": f["drain"],
                "capacity": f.get("capacity", f.get("beds", f.get("vehicles", f.get("personnel", f.get("mw", 0)))))
            },
            "geometry": {
                "type": "Point",
                "coordinates": [f["lon"], f["lat"]]
            }
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def ensure_all_datasets_exist(force_regenerate: bool = True):
    """Ensure all synthetic CSV and GeoJSON files are present, writing them if absent."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if force_regenerate or not os.path.exists(CSV_PATH):
        df = generate_sample_flood_dataset()
        df.to_csv(CSV_PATH, index=False)

    if force_regenerate or not os.path.exists(WARDS_GEOJSON):
        wards_geo = generate_kolkata_wards_geojson()
        with open(WARDS_GEOJSON, "w") as f:
            json.dump(wards_geo, f, indent=2)

    if force_regenerate or not os.path.exists(ROADS_GEOJSON):
        roads_geo = generate_kolkata_roads_geojson()
        with open(ROADS_GEOJSON, "w") as f:
            json.dump(roads_geo, f, indent=2)

    if force_regenerate or not os.path.exists(INFRA_GEOJSON):
        infra_geo = generate_infrastructure_geojson()
        with open(INFRA_GEOJSON, "w") as f:
            json.dump(infra_geo, f, indent=2)


def load_dataset() -> pd.DataFrame:
    """Load sample flood data into a pandas DataFrame."""
    ensure_all_datasets_exist(force_regenerate=False)
    return pd.read_csv(CSV_PATH)


def load_wards_geojson() -> Dict[str, Any]:
    """Load Kolkata wards GeoJSON."""
    ensure_all_datasets_exist(force_regenerate=False)
    with open(WARDS_GEOJSON, "r") as f:
        return json.load(f)


def load_roads_geojson() -> Dict[str, Any]:
    """Load Kolkata roads GeoJSON."""
    ensure_all_datasets_exist(force_regenerate=False)
    with open(ROADS_GEOJSON, "r") as f:
        return json.load(f)


def load_infrastructure_geojson() -> Dict[str, Any]:
    """Load Kolkata critical infrastructure GeoJSON."""
    ensure_all_datasets_exist(force_regenerate=False)
    with open(INFRA_GEOJSON, "r") as f:
        return json.load(f)


# Automatically initialize upon import
ensure_all_datasets_exist(force_regenerate=True)
