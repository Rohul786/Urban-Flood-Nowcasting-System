"""
FloodGuard AI - Weather API Connector
Handles live weather fetching from OpenWeatherMap with automatic, graceful fallback
to high-fidelity prototype simulation mode if API key is not supplied.
"""

import os
import requests
from typing import Dict, Any, Optional

DEFAULT_LAT = 22.5726
DEFAULT_LON = 88.3639
DEFAULT_CITY = "Kolkata, IN"


def get_weather_data(api_key: Optional[str] = None,
                     city: str = DEFAULT_CITY,
                     lat: float = DEFAULT_LAT,
                     lon: float = DEFAULT_LON) -> Dict[str, Any]:
    """
    Fetch current weather metrics.
    If API key is valid, returns live data from OpenWeatherMap.
    Otherwise, returns calibrated simulation data clearly labeled.
    """
    api_key = api_key or os.environ.get("OPENWEATHER_API_KEY", "").strip()

    if api_key and api_key != "your_openweather_api_key_here":
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                rain_1h = data.get("rain", {}).get("1h", 0.0)
                # Estimate 3h/6h from 1h if not directly present in current endpoint
                rain_3h = data.get("rain", {}).get("3h", rain_1h * 2.6)
                rain_6h = rain_3h * 1.8
                return {
                    "is_live": True,
                    "source_label": "🟢 LIVE WEATHER ACTIVE (OpenWeatherMap API)",
                    "city": data.get("name", city),
                    "temperature_c": round(data["main"]["temp"], 1),
                    "humidity_percent": data["main"]["humidity"],
                    "pressure_hpa": data["main"]["pressure"],
                    "wind_speed_kmh": round(data["wind"]["speed"] * 3.6, 1),
                    "weather_description": data["weather"][0]["description"].title(),
                    "rainfall_1h_mm": round(rain_1h, 1),
                    "rainfall_3h_mm": round(rain_3h, 1),
                    "rainfall_6h_mm": round(rain_6h, 1),
                    "cloud_cover_percent": data.get("clouds", {}).get("all", 80),
                }
        except Exception as e:
            # Fall back gracefully to simulation mode if network fails
            pass

    # High-fidelity prototype simulation mode (Monsoon in Kolkata)
    return {
        "is_live": False,
        "source_label": "🔴 PROTOTYPE SIMULATION MODE (Demo Calibration)",
        "city": city,
        "temperature_c": 28.4,
        "humidity_percent": 88,
        "pressure_hpa": 1002,
        "wind_speed_kmh": 22.5,
        "weather_description": "Heavy Monsoon Thunderstorm",
        "rainfall_1h_mm": 34.5,
        "rainfall_3h_mm": 78.2,
        "rainfall_6h_mm": 125.0,
        "cloud_cover_percent": 95,
    }
