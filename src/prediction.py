"""
FloodGuard AI - 0 to 3 Hour Nowcasting Engine
Computes spatial flood risk progression for urban wards and road networks
across T+0h, T+1h, T+2h, and T+3h forecast horizons.
"""

from typing import Dict, Any, List
import numpy as np
from src.model import load_model, predict_risk_probability, explain_prediction
from utils.helpers import get_risk_category, get_road_status_category


def compute_ward_nowcast(ward_properties: Dict[str, Any],
                         base_rainfall_1h: float,
                         model_bundle: Dict[str, Any],
                         rainfall_multiplier: float = 1.0) -> Dict[str, Any]:
    """
    Compute 0-3 hour nowcasting progression for a single ward.
    Simulates physical rainfall accumulation, drainage discharge lag, and terrain waterlogging.
    """
    elev = ward_properties.get("elevation", 5.0)
    slope = ward_properties.get("slope", 0.5)
    drain_cap = ward_properties.get("drainage_capacity", 20.0)
    imp_surf = ward_properties.get("impervious_surface", 80.0)
    hist_freq = ward_properties.get("historical_flood_frequency", 5)
    road_dens = ward_properties.get("road_density", 16.0)
    tide = ward_properties.get("tide_level_m", 2.5)
    soil_sat = ward_properties.get("soil_saturation", 0.7)

    time_steps = ["Current (T+0h)", "+1 Hour", "+2 Hours", "+3 Hours"]
    horizons = [0, 1, 2, 3]
    rain_factors = [1.0, 1.35, 1.65, 1.85]
    
    predictions = []
    
    for h, factor in zip(horizons, rain_factors):
        cur_r1h = base_rainfall_1h * factor * rainfall_multiplier
        cur_r3h = cur_r1h * (1.8 + h * 0.35)
        cur_r6h = cur_r3h * 1.5

        feat = {
            "rainfall_1h": cur_r1h,
            "rainfall_3h": cur_r3h,
            "rainfall_6h": cur_r6h,
            "elevation": elev,
            "slope": slope,
            "drainage_capacity": drain_cap,
            "impervious_surface": imp_surf,
            "historical_flood_frequency": hist_freq,
            "road_density": road_dens,
            "tide_level_m": tide + (0.15 * h),
            "soil_saturation": min(1.0, soil_sat + (0.06 * h))
        }

        prob = predict_risk_probability(model_bundle, feat)
        cat, color, badge_class, icon = get_risk_category(prob)
        
        water_depth_cm = round(max(0, (prob - 0.20) * 85 * (10.0 / max(2.0, elev))), 1) if prob > 0.25 else 0.0

        predictions.append({
            "horizon_hours": h,
            "time_label": time_steps[h],
            "rainfall_1h_mm": round(cur_r1h, 1),
            "rainfall_3h_mm": round(cur_r3h, 1),
            "flood_probability": round(prob, 3),
            "risk_category": cat,
            "color": color,
            "badge_class": badge_class,
            "icon": icon,
            "water_depth_cm": water_depth_cm,
            "feature_dict": feat
        })

    xai_info = explain_prediction(model_bundle, predictions[1]["feature_dict"])

    return {
        "ward_id": ward_properties.get("ward_id"),
        "ward_name": ward_properties.get("ward_name"),
        "elevation": elev,
        "drainage_capacity": drain_cap,
        "impervious_surface": imp_surf,
        "center_lat": ward_properties.get("center_lat"),
        "center_lon": ward_properties.get("center_lon"),
        "nowcast": predictions,
        "current_risk": predictions[0],
        "plus_1h_risk": predictions[1],
        "plus_2h_risk": predictions[2],
        "plus_3h_risk": predictions[3],
        "xai_explanation": xai_info
    }


def compute_all_wards_nowcast(wards_geojson: Dict[str, Any],
                              base_rainfall_1h: float,
                              model_bundle: Dict[str, Any],
                              rainfall_multiplier: float = 1.0) -> List[Dict[str, Any]]:
    """Compute 0-3h nowcasting for all wards in the urban region."""
    results = []
    for feat in wards_geojson["features"]:
        props = feat["properties"]
        res = compute_ward_nowcast(props, base_rainfall_1h, model_bundle, rainfall_multiplier)
        res["geometry"] = feat["geometry"]
        results.append(res)
    return results


def compute_road_risk_assessment(roads_geojson: Dict[str, Any],
                                 ward_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assess flood risk for each road segment based on road elevation, drainage condition,
    and the nearest/intersecting ward's predicted flood probability.
    """
    road_assessments = []
    for feat in roads_geojson["features"]:
        props = feat["properties"]
        road_id = props["road_id"]
        road_name = props["road_name"]
        road_elev = props["elevation"]
        drain_cond = props["drainage_condition"]
        length_km = props["length_km"]
        coords = feat["geometry"]["coordinates"]

        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        mid_lat = np.mean(lats)
        mid_lon = np.mean(lons)

        # Find nearest ward
        min_dist = float("inf")
        nearest_ward = ward_results[0]
        for w in ward_results:
            d = (w["center_lat"] - mid_lat)**2 + (w["center_lon"] - mid_lon)**2
            if d < min_dist:
                min_dist = d
                nearest_ward = w

        ward_prob_1h = nearest_ward["plus_1h_risk"]["flood_probability"]

        # Calibrated attenuation for drainage and road elevation
        drainage_penalties = {"Choked": 0.28, "Poor": 0.16, "Fair": 0.02, "Good": -0.25}
        elevation_penalty = (5.0 - road_elev) * 0.09

        road_risk_score = ward_prob_1h + drainage_penalties.get(drain_cond, 0.0) + elevation_penalty
        road_risk_score = float(np.clip(road_risk_score, 0.02, 0.98))

        status, color = get_road_status_category(road_risk_score)
        water_depth_cm = round(max(0, (road_risk_score - 0.35) * 65), 1) if road_risk_score >= 0.40 else 0.0

        road_assessments.append({
            "road_id": road_id,
            "road_name": road_name,
            "from_node": props["from_node"],
            "to_node": props["to_node"],
            "length_km": length_km,
            "elevation": road_elev,
            "drainage_condition": drain_cond,
            "coordinates": coords,
            "nearest_ward": nearest_ward["ward_name"],
            "risk_score": round(road_risk_score, 3),
            "status": status,
            "color": color,
            "water_depth_cm": water_depth_cm,
            "geometry": feat["geometry"]
        })

    return road_assessments
