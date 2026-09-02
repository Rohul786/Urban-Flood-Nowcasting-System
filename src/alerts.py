"""
FloodGuard AI - Early Warning System & Alerts Engine
Evaluates dynamic hydrologic thresholds, generates multi-tier early warning advisories,
and logs actionable alerts to the database for citizen and administrative response.
"""

from typing import List, Dict, Any
from datetime import datetime
from utils.db import insert_alert, get_active_alerts, clear_alerts


def generate_early_warnings(ward_results: List[Dict[str, Any]],
                            road_assessments: List[Dict[str, Any]],
                            save_to_db: bool = True) -> List[Dict[str, Any]]:
    """
    Generate early warnings for all wards and roads exceeding risk thresholds.
    Alert Levels: CRITICAL, WARNING, WATCH, INFO
    """
    alerts = []
    
    # Check wards (focusing on +1h and +2h nowcasting forecasts)
    for w in ward_results:
        p1_prob = w["plus_1h_risk"]["flood_probability"]
        p2_prob = w["plus_2h_risk"]["flood_probability"]
        max_prob = max(p1_prob, p2_prob)
        ward_name = w["ward_name"]
        ward_id = w["ward_id"]
        elev = w["elevation"]
        drain_cap = w["drainage_capacity"]
        rain_1h = w["plus_1h_risk"]["rainfall_1h_mm"]

        top_drivers = w["xai_explanation"]["factors"][:2]
        driver_str = " + ".join([f"{d[0].split(' ')[1]} ({d[1]}%)" for d in top_drivers])

        if max_prob >= 0.75:
            level = "CRITICAL"
            expected = "Within 30–60 minutes (Peak water depth ~35-50 cm)"
            reason = f"Severe precipitation ({rain_1h} mm/h) combined with {driver_str} and low elevation ({elev} m)."
            action = "IMMEDIATE EVACUATION of ground-floor assets and basements. Avoid all non-emergency travel. Deploy municipal de-watering pumps immediately."
        elif max_prob >= 0.50:
            level = "WARNING"
            expected = "Within 60–90 minutes (Potential water depth ~15-30 cm)"
            reason = f"Heavy rainfall accumulation ({rain_1h} mm/h) exceeding drainage capacity ({drain_cap} m³/s)."
            action = "AVOID low-lying arterial corridors. Move vehicles to elevated parking. NDRF/SDRF emergency quick-response teams placed on standby."
        elif max_prob >= 0.30:
            level = "WATCH"
            expected = "Within 2–3 hours"
            reason = f"Moderate to heavy continuous rainfall with rising soil saturation."
            action = "Monitor municipal drainage gates. Keep emergency power backups active and avoid parking near storm canals."
        else:
            continue

        alert_item = {
            "location": ward_name,
            "ward_id": ward_id,
            "alert_level": level,
            "flood_probability": round(max_prob, 3),
            "expected_timeframe": expected,
            "reason": reason,
            "recommended_action": action,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        alerts.append(alert_item)

    # Check blocked roads
    blocked_roads = [r for r in road_assessments if r["status"] in ["BLOCKED", "HIGH RISK"]]
    if len(blocked_roads) >= 3:
        road_names = ", ".join([r["road_name"] for r in blocked_roads[:3]])
        alerts.append({
            "location": f"Critical Arterial Road Corridors ({len(blocked_roads)} Segments)",
            "ward_id": "ROAD-NETWORK",
            "alert_level": "CRITICAL" if len(blocked_roads) >= 5 else "WARNING",
            "flood_probability": 0.88,
            "expected_timeframe": "Immediate / Next 90 minutes",
            "reason": f"Severe inundation detected across key transit corridors including {road_names}.",
            "recommended_action": "DIVERT ALL VEHICULAR TRAFFIC to elevated bypass routes. Municipal police to enforce roadblocks on affected underpasses.",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # Sort alerts by severity
    severity_order = {"CRITICAL": 0, "WARNING": 1, "WATCH": 2, "INFO": 3}
    alerts.sort(key=lambda x: (severity_order.get(x["alert_level"], 4), -x["flood_probability"]))

    if save_to_db:
        clear_alerts()
        for a in alerts:
            insert_alert(
                location=a["location"],
                ward_id=a["ward_id"],
                alert_level=a["alert_level"],
                flood_probability=a["flood_probability"],
                expected_timeframe=a["expected_timeframe"],
                reason=a["reason"],
                recommended_action=a["recommended_action"]
            )

    return alerts
