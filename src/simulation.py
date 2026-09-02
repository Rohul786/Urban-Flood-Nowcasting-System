"""
FloodGuard AI - Simulation Engine & Scenario Orchestrator
Manages rainfall simulation presets, dynamic slider inputs, and coordinates
instant multi-layer recalculation across ML models, GIS layers, road routing, and alerts.
"""

from typing import Dict, Any, List
import numpy as np
from src.prediction import compute_all_wards_nowcast, compute_road_risk_assessment
from src.alerts import generate_early_warnings
from utils.db import log_simulation_run
from utils.helpers import get_risk_category

SIMULATION_PRESETS = {
    "NORMAL_RAIN": {
        "name": "🌧️ Normal Monsoon Rain",
        "rainfall_mm": 24.0,
        "description": "Standard monsoon showers with normal municipal drainage capacity."
    },
    "HEAVY_RAIN": {
        "name": "⛈️ Heavy Monsoon Downpour",
        "rainfall_mm": 68.0,
        "description": "Intense continuous downpour causing localized waterlogging in depressions."
    },
    "EXTREME_RAIN": {
        "name": "🌊 Extreme Rainfall Event",
        "rainfall_mm": 115.0,
        "description": "Severe weather event triggering widespread road inundation and emergency alerts."
    },
    "CLOUD_BURST": {
        "name": "🚨 Cloudburst / Torrential Surge",
        "rainfall_mm": 160.0,
        "description": "Catastrophic precipitation overwhelming drainage, requiring immediate evacuation."
    }
}


def run_flood_simulation(rainfall_mm: float,
                         model_bundle: Dict[str, Any],
                         wards_geojson: Dict[str, Any],
                         roads_geojson: Dict[str, Any],
                         infra_geojson: Dict[str, Any],
                         scenario_name: str = "Simulation") -> Dict[str, Any]:
    """
    Execute full pipeline simulation for a given rainfall intensity.
    Returns synchronized results for all 8 application views.
    """
    # 1. Compute 0-3h nowcasting for all wards
    ward_results = compute_all_wards_nowcast(
        wards_geojson=wards_geojson,
        base_rainfall_1h=rainfall_mm,
        model_bundle=model_bundle
    )

    # 2. Compute road network risk scores
    road_assessments = compute_road_risk_assessment(
        roads_geojson=roads_geojson,
        ward_results=ward_results
    )

    # 3. Generate early warning alerts
    alerts = generate_early_warnings(
        ward_results=ward_results,
        road_assessments=road_assessments,
        save_to_db=True
    )

    # 4. Compute Top KPI Summaries
    probs_current = [w["current_risk"]["flood_probability"] for w in ward_results]
    probs_1h = [w["plus_1h_risk"]["flood_probability"] for w in ward_results]
    
    avg_risk_prob = float(np.mean(probs_1h))
    max_risk_prob = float(np.max(probs_1h))
    risk_label, risk_color, risk_badge, risk_icon = get_risk_category(avg_risk_prob)

    affected_roads = [r for r in road_assessments if r["status"] in ["CAUTION", "HIGH RISK", "BLOCKED"]]
    blocked_roads = [r for r in road_assessments if r["status"] == "BLOCKED"]
    high_risk_wards = [w for w in ward_results if w["plus_1h_risk"]["flood_probability"] >= 0.50]
    critical_alerts = [a for a in alerts if a["alert_level"] in ["CRITICAL", "WARNING"]]

    # 5. Assess Infrastructure Impact
    infra_at_risk = 0
    for feat in infra_geojson["features"]:
        lon, lat = feat["geometry"]["coordinates"]
        # Nearest ward
        nearest_w = min(ward_results, key=lambda w: (w["center_lat"] - lat)**2 + (w["center_lon"] - lon)**2)
        if nearest_w["plus_1h_risk"]["flood_probability"] >= 0.50:
            infra_at_risk += 1

    # 6. Log run to database
    log_simulation_run(
        rainfall_mm=rainfall_mm,
        scenario_name=scenario_name,
        avg_flood_prob=avg_risk_prob,
        affected_roads=len(affected_roads),
        active_alerts=len(alerts),
        high_risk_wards=len(high_risk_wards)
    )

    return {
        "rainfall_mm": rainfall_mm,
        "scenario_name": scenario_name,
        "ward_results": ward_results,
        "road_assessments": road_assessments,
        "alerts": alerts,
        "kpi": {
            "rainfall_display": f"{rainfall_mm:.1f} mm",
            "avg_risk_prob": avg_risk_prob,
            "max_risk_prob": max_risk_prob,
            "risk_label": risk_label,
            "risk_color": risk_color,
            "risk_badge": risk_badge,
            "risk_icon": risk_icon,
            "total_roads": len(road_assessments),
            "affected_roads_count": len(affected_roads),
            "blocked_roads_count": len(blocked_roads),
            "safe_roads_count": len(road_assessments) - len(affected_roads),
            "active_alerts_count": len(alerts),
            "critical_alerts_count": len(critical_alerts),
            "high_risk_wards_count": len(high_risk_wards),
            "total_wards_count": len(ward_results),
            "infra_at_risk_count": infra_at_risk,
            "total_infra_count": len(infra_geojson["features"])
        }
    }
