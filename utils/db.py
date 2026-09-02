"""
FloodGuard AI - Database Layer (SQLite)
Provides persistent storage for early warnings, alerts, simulation logs, and dispatch actions.
Easily upgradable to PostgreSQL / PostGIS.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "floodguard.db")


def get_db_connection():
    """Create and return a database connection with dictionary row access."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Alerts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            location TEXT NOT NULL,
            ward_id TEXT,
            alert_level TEXT NOT NULL,
            flood_probability REAL NOT NULL,
            expected_timeframe TEXT NOT NULL,
            reason TEXT NOT NULL,
            recommended_action TEXT NOT NULL,
            status TEXT DEFAULT 'ACTIVE',
            dispatched INTEGER DEFAULT 0
        )
    """)

    # Simulation logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS simulation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            rainfall_mm REAL NOT NULL,
            scenario_name TEXT,
            avg_flood_prob REAL,
            affected_roads_count INTEGER,
            active_alerts_count INTEGER,
            high_risk_wards_count INTEGER
        )
    """)

    # Critical infrastructure status cache
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS infrastructure_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            facility_name TEXT NOT NULL,
            facility_type TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            current_risk TEXT NOT NULL,
            flood_probability REAL NOT NULL,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def insert_alert(location: str, ward_id: str, alert_level: str,
                 flood_probability: float, expected_timeframe: str,
                 reason: str, recommended_action: str) -> int:
    """Insert a new flood alert record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO alerts (location, ward_id, alert_level, flood_probability, expected_timeframe, reason, recommended_action)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (location, ward_id, alert_level, flood_probability, expected_timeframe, reason, recommended_action))
    alert_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return alert_id


def get_active_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve active alerts ordered by severity and time."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM alerts 
        WHERE status = 'ACTIVE' 
        ORDER BY 
            CASE alert_level 
                WHEN 'CRITICAL' THEN 1 
                WHEN 'WARNING' THEN 2 
                WHEN 'WATCH' THEN 3 
                ELSE 4 
            END, timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def log_simulation_run(rainfall_mm: float, scenario_name: str, avg_flood_prob: float,
                       affected_roads: int, active_alerts: int, high_risk_wards: int):
    """Log a simulation run for historical metrics."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO simulation_logs (rainfall_mm, scenario_name, avg_flood_prob, affected_roads_count, active_alerts_count, high_risk_wards_count)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (rainfall_mm, scenario_name, avg_flood_prob, affected_roads, active_alerts, high_risk_wards))
    conn.commit()
    conn.close()


def clear_alerts():
    """Clear all active alerts (for simulation reset)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()


# Ensure DB is initialized upon module load
init_db()
