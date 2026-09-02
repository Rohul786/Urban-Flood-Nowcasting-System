"""
FloodGuard AI - GIS Flood Mapping Engine
Generates interactive Folium maps with ward vulnerability polygons, road risk vectors,
critical infrastructure status markers, and routing geometry overlays.
"""

from typing import Dict, Any, List, Optional
import folium
from branca.element import Element

DEFAULT_CENTER = [22.5600, 88.3650]  # Kolkata Metropolitan Center


def get_infrastructure_icon(facility_type: str, risk_status: str) -> Dict[str, str]:
    """Return Folium icon configuration for facility types."""
    icon_map = {
        "HOSPITAL": ("plus", "red", "fa"),
        "FIRE_STATION": ("fire", "orange", "fa"),
        "POLICE_HQ": ("shield", "blue", "fa"),
        "POLICE_STATION": ("shield", "blue", "fa"),
        "SHELTER": ("home", "green", "fa"),
        "POWER_STATION": ("bolt", "darkpurple", "fa")
    }
    return icon_map.get(facility_type, ("info-circle", "gray", "fa"))


def create_base_map(center: List[float] = DEFAULT_CENTER, zoom_start: int = 12) -> folium.Map:
    """Create a Folium map instance with dark CartoDB tiles suitable for disaster command centers."""
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="CartoDB dark_matter",
        control_scale=True,
        prefer_canvas=True
    )
    return m


def add_floating_legend(m: folium.Map):
    """Add a responsive HTML legend to the top-right corner of the Folium map."""
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 25px; left: 25px; width: 220px; height: auto;
        background-color: rgba(15, 23, 42, 0.92);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 14px;
        font-family: 'Segoe UI', Roboto, sans-serif;
        font-size: 12px;
        color: #f8fafc;
        z-index: 9999;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    ">
        <b style="color: #00f0ff; text-transform: uppercase; letter-spacing: 0.5px; font-size: 11px;">Flood Risk Zones</b><br>
        <span style="display:inline-block; width:12px; height:12px; background:#06d6a0; border-radius:2px; margin-right:6px;"></span>Low (0–25%)<br>
        <span style="display:inline-block; width:12px; height:12px; background:#ffbe0b; border-radius:2px; margin-right:6px;"></span>Moderate (25–50%)<br>
        <span style="display:inline-block; width:12px; height:12px; background:#fb5607; border-radius:2px; margin-right:6px;"></span>High (50–75%)<br>
        <span style="display:inline-block; width:12px; height:12px; background:#ef233c; border-radius:2px; margin-right:6px;"></span>Severe (75–100%)<br>
        
        <hr style="border: 0; border-top: 1px solid #334155; margin: 6px 0;">
        <b style="color: #3a86ff; text-transform: uppercase; letter-spacing: 0.5px; font-size: 11px;">Road Conditions</b><br>
        <span style="color:#06d6a0; font-weight:bold;">━━</span> Safe Corridor<br>
        <span style="color:#ffbe0b; font-weight:bold;">━━</span> Caution / Slow<br>
        <span style="color:#fb5607; font-weight:bold;">━━</span> High Risk<br>
        <span style="color:#ef233c; font-weight:bold;">━━</span> Inundated / Blocked<br>
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))


def add_ward_polygons_to_map(m: folium.Map, ward_results: List[Dict[str, Any]], horizon_idx: int = 0):
    """Add colored polygon overlays for each ward corresponding to a specific nowcasting horizon."""
    fg = folium.FeatureGroup(name=f"Ward Risk Layer ({['Current', '+1h', '+2h', '+3h'][horizon_idx]})")

    for w in ward_results:
        risk_info = w["nowcast"][horizon_idx]
        cat = risk_info["risk_category"]
        prob = risk_info["flood_probability"]
        color = risk_info["color"]
        coords = w["geometry"]["coordinates"][0]  # [ [lon, lat], ... ]
        
        # Folium polygon expects [ [lat, lon], ... ]
        poly_lat_lons = [[c[1], c[0]] for c in coords]

        # Top XAI factors
        xai_factors = w["xai_explanation"]["factors"][:2]
        xai_snippet = "<br>".join([f"• {f[0]}: <b>{f[1]}%</b>" for f in xai_factors])

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px; min-width: 200px; color: #1e293b;">
            <div style="font-size: 14px; font-weight: bold; border-bottom: 2px solid {color}; padding-bottom: 4px; margin-bottom: 6px;">
                {w["ward_name"]}
            </div>
            <b>Forecast Horizon:</b> {risk_info["time_label"]}<br>
            <b>Rainfall (1h):</b> {risk_info["rainfall_1h_mm"]} mm<br>
            <b>Elevation:</b> {w["elevation"]} m MSL<br>
            <b>Drainage Capacity:</b> {w["drainage_capacity"]} m³/s<br>
            <b>Est. Inundation Depth:</b> {risk_info["water_depth_cm"]} cm<br>
            <hr style="margin: 5px 0;">
            <b>Flood Probability:</b> <span style="font-size: 13px; font-weight: bold; color: {color};">{prob * 100:.1f}% ({cat})</span><br>
            <div style="margin-top: 6px; background: #f1f5f9; padding: 6px; border-radius: 4px;">
                <b style="color: #475569;">Key Contributing Drivers:</b><br>
                {xai_snippet}
            </div>
        </div>
        """

        folium.Polygon(
            locations=poly_lat_lons,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.45,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{w['ward_name']}: {cat} ({prob*100:.1f}%)"
        ).add_to(fg)

    fg.add_to(m)


def add_roads_to_map(m: folium.Map, road_assessments: List[Dict[str, Any]]):
    """Add road segments with risk-colored styling to the map."""
    fg = folium.FeatureGroup(name="Road Risk Status")

    for r in road_assessments:
        coords = r["coordinates"]
        line_lat_lons = [[c[1], c[0]] for c in coords]
        color = r["color"]
        status = r["status"]
        risk_score = r["risk_score"]

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px; min-width: 180px; color: #1e293b;">
            <b style="font-size: 13px; color: #0f172a;">{r["road_name"]}</b><br>
            <b>Status:</b> <span style="font-weight:bold; color:{color};">{status}</span><br>
            <b>Risk Score:</b> {risk_score * 100:.1f}%<br>
            <b>Length:</b> {r["length_km"]} km<br>
            <b>Elevation:</b> {r["elevation"]} m<br>
            <b>Drainage:</b> {r["drainage_condition"]}<br>
            <b>Est. Water Depth:</b> {r["water_depth_cm"]} cm<br>
        </div>
        """

        weight = 5 if status in ["BLOCKED", "HIGH RISK"] else 3

        folium.PolyLine(
            locations=line_lat_lons,
            color=color,
            weight=weight,
            opacity=0.85,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{r['road_name']} — {status}"
        ).add_to(fg)

    fg.add_to(m)


def add_infrastructure_to_map(m: folium.Map,
                              infra_geojson: Dict[str, Any],
                              ward_results: List[Dict[str, Any]]):
    """Add critical facilities (Hospitals, Fire, Shelters, Police, Power) with vulnerability assessment."""
    fg = folium.FeatureGroup(name="Critical Infrastructure")

    for feat in infra_geojson["features"]:
        props = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"]
        f_type = props["type"]
        f_name = props["name"]

        # Determine nearest ward risk
        min_dist = float("inf")
        nearest_ward = ward_results[0]
        for w in ward_results:
            d = (w["center_lat"] - lat)**2 + (w["center_lon"] - lon)**2
            if d < min_dist:
                min_dist = d
                nearest_ward = w

        prob = nearest_ward["plus_1h_risk"]["flood_probability"]
        status = "HIGH RISK" if prob >= 0.60 else "AT RISK" if prob >= 0.35 else "SAFE"
        status_color = "#ef233c" if status == "HIGH RISK" else "#ffbe0b" if status == "AT RISK" else "#06d6a0"

        icon_name, icon_color, prefix = get_infrastructure_icon(f_type, status)

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px; min-width: 190px; color: #1e293b;">
            <b style="font-size: 13px; color: #0f172a;">{f_name}</b><br>
            <b>Facility Type:</b> {f_type.replace('_', ' ')}<br>
            <b>Nearest Zone:</b> {nearest_ward['ward_name']}<br>
            <b>Zone Flood Probability:</b> {prob * 100:.1f}%<br>
            <b>Operational Status:</b> <span style="font-weight:bold; color:{status_color};">{status}</span><br>
            <b>Capacity/Resources:</b> {props.get('capacity', 'N/A')}
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{f_name} ({status})",
            icon=folium.Icon(color=icon_color, icon=icon_name, prefix=prefix)
        ).add_to(fg)

    fg.add_to(m)


def build_complete_gis_map(ward_results: List[Dict[str, Any]],
                           road_assessments: List[Dict[str, Any]],
                           infra_geojson: Dict[str, Any],
                           horizon_idx: int = 0) -> folium.Map:
    """Construct the complete interactive disaster GIS map."""
    m = create_base_map()
    add_ward_polygons_to_map(m, ward_results, horizon_idx)
    add_roads_to_map(m, road_assessments)
    add_infrastructure_to_map(m, infra_geojson, ward_results)
    add_floating_legend(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m
