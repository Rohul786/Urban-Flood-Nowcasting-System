"""
FloodGuard AI - Safe Route Recommendation Engine
Builds an urban road network graph and computes shortest vs. flood-safe evacuation routes
using risk-weighted graph traversal algorithms.
"""

from typing import Dict, Any, List, Tuple, Optional
import networkx as nx
import folium
import numpy as np


def build_road_network_graph(road_assessments: List[Dict[str, Any]]) -> nx.Graph:
    """Construct a NetworkX graph with road geometries and dynamic flood risk weights."""
    G = nx.Graph()

    for r in road_assessments:
        u = r["from_node"]
        v = r["to_node"]
        length = r["length_km"]
        risk = r["risk_score"]
        status = r["status"]
        coords = r["coordinates"]

        base_time = (length / 35.0) * 60.0
        shortest_weight = length

        # Heavy exponential penalty for risky/blocked roads in safe pathfinding
        status_penalty = 80.0 if status == "BLOCKED" else 30.0 if status == "HIGH RISK" else 5.0 if status == "CAUTION" else 0.0
        safe_weight = length * (1.0 + 25.0 * (risk ** 3.0)) + status_penalty

        G.add_edge(
            u, v,
            road_id=r["road_id"],
            road_name=r["road_name"],
            length_km=length,
            base_time_mins=base_time,
            risk_score=risk,
            status=status,
            water_depth_cm=r.get("water_depth_cm", 0.0),
            coordinates=coords,
            shortest_weight=shortest_weight,
            safe_weight=safe_weight
        )

    return G


def compute_route_details(G: nx.Graph, path: List[str]) -> Dict[str, Any]:
    """Extract metrics, coordinate geometry, and flood exposure for a path."""
    total_km = 0.0
    total_time = 0.0
    weighted_risk_sum = 0.0
    all_coords = []
    edges_info = []

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        edge_data = G[u][v]
        length = edge_data.get("length_km", 1.0)
        risk = edge_data.get("risk_score", 0.1)
        coords = edge_data.get("coordinates", [])

        total_km += length
        slowdown = 3.0 if edge_data.get("status") in ["BLOCKED", "HIGH RISK"] else 1.5 if edge_data.get("status") == "CAUTION" else 1.0
        total_time += edge_data.get("base_time_mins", 2.0) * slowdown
        weighted_risk_sum += length * risk

        if coords:
            all_coords.extend([[c[1], c[0]] for c in coords])
        edges_info.append(edge_data)

    # Average risk exposure percentage across the full route
    exposure_pct = round((weighted_risk_sum / max(0.1, total_km)) * 100, 1)

    return {
        "nodes": path,
        "total_distance_km": round(total_km, 2),
        "estimated_time_mins": round(total_time, 1),
        "flood_exposure_pct": exposure_pct,
        "is_severely_inundated": exposure_pct > 50.0,
        "coordinates": all_coords,
        "edges": edges_info
    }


def find_shortest_and_safe_routes(G: nx.Graph, start_node: str, dest_node: str) -> Dict[str, Any]:
    """
    Find and compare standard shortest route vs. flood-risk-avoiding safe route.
    """
    if not G.has_node(start_node) or not G.has_node(dest_node):
        return {
            "shortest_route": None,
            "safe_route": None,
            "comparison": None,
            "error": f"One or both locations ('{start_node}', '{dest_node}') not in road graph."
        }

    try:
        shortest_path = nx.shortest_path(G, source=start_node, target=dest_node, weight="shortest_weight")
        shortest_res = compute_route_details(G, shortest_path)
    except nx.NetworkXNoPath:
        shortest_res = None

    try:
        safe_path = nx.shortest_path(G, source=start_node, target=dest_node, weight="safe_weight")
        safe_res = compute_route_details(G, safe_path)
    except nx.NetworkXNoPath:
        safe_res = None

    # Calculate comparison delta
    if shortest_res and safe_res:
        dist_diff = round(safe_res["total_distance_km"] - shortest_res["total_distance_km"], 2)
        time_diff = round(safe_res["estimated_time_mins"] - shortest_res["estimated_time_mins"], 1)
        exposure_reduction = round(shortest_res["flood_exposure_pct"] - safe_res["flood_exposure_pct"], 1)
        
        comparison = {
            "distance_delta_km": dist_diff,
            "time_delta_mins": time_diff,
            "flood_exposure_reduction_pct": max(0.0, exposure_reduction),
            "recommendation": "Use AI Recommended Safe Route to avoid inundated road corridors." if shortest_res["flood_exposure_pct"] > 35 else "Both routes are currently safe to traverse."
        }
    else:
        comparison = None

    return {
        "shortest_route": shortest_res,
        "safe_route": safe_res,
        "comparison": comparison
    }


def render_routes_folium_map(G: nx.Graph,
                             shortest_res: Optional[Dict[str, Any]],
                             safe_res: Optional[Dict[str, Any]],
                             start_node: str,
                             dest_node: str) -> folium.Map:
    """Render an interactive Folium map comparing both routes."""
    m = folium.Map(location=[22.5450, 88.3650], zoom_start=12, tiles="CartoDB dark_matter")

    # Draw Shortest Route (Red / Orange dashed)
    if shortest_res and shortest_res["coordinates"]:
        folium.PolyLine(
            locations=shortest_res["coordinates"],
            color="#ef233c",
            weight=6,
            dash_array="8, 8",
            opacity=0.9,
            tooltip=f"Shortest Route: {shortest_res['total_distance_km']} km | {shortest_res['flood_exposure_pct']}% Flood Exposure",
            popup=f"Shortest Route<br>Distance: {shortest_res['total_distance_km']} km<br>Time: {shortest_res['estimated_time_mins']} mins<br>Risk Exposure: {shortest_res['flood_exposure_pct']}%"
        ).add_to(m)

    # Draw Safe Route (Bright Cyan Solid)
    if safe_res and safe_res["coordinates"]:
        folium.PolyLine(
            locations=safe_res["coordinates"],
            color="#00f0ff",
            weight=5,
            opacity=0.95,
            tooltip=f"AI Safe Route: {safe_res['total_distance_km']} km | {safe_res['flood_exposure_pct']}% Flood Exposure",
            popup=f"AI Flood-Safe Route<br>Distance: {safe_res['total_distance_km']} km<br>Time: {safe_res['estimated_time_mins']} mins<br>Risk Exposure: {safe_res['flood_exposure_pct']}%"
        ).add_to(m)

    # Markers for Start and Destination
    start_coords = None
    dest_coords = None
    if safe_res and safe_res["coordinates"]:
        start_coords = safe_res["coordinates"][0]
        dest_coords = safe_res["coordinates"][-1]
    elif shortest_res and shortest_res["coordinates"]:
        start_coords = shortest_res["coordinates"][0]
        dest_coords = shortest_res["coordinates"][-1]

    if start_coords:
        folium.Marker(
            location=start_coords,
            popup=f"Origin: {start_node}",
            tooltip=f"Start: {start_node}",
            icon=folium.Icon(color="green", icon="play", prefix="fa")
        ).add_to(m)

    if dest_coords:
        folium.Marker(
            location=dest_coords,
            popup=f"Destination: {dest_node}",
            tooltip=f"Destination: {dest_node}",
            icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")
        ).add_to(m)

    # Route Legend
    from branca.element import Element
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 25px; right: 25px; width: 230px;
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
        <b style="color: #00f0ff; text-transform: uppercase; font-size: 11px;">Routing Comparison</b><br>
        <span style="color:#ef233c; font-weight:bold;">- - -</span> Shortest Route (Risky / Inundated)<br>
        <span style="color:#00f0ff; font-weight:bold;">━━━</span> AI Flood-Safe Route (Recommended)<br>
        <hr style="border:0; border-top:1px solid #334155; margin:6px 0;">
        <span style="color:#06d6a0;">🟢 Start</span> &nbsp;|&nbsp; <span style="color:#ef233c;">🔴 Destination</span>
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))

    return m
