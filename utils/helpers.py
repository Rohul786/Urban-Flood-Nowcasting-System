"""
FloodGuard AI - UI & Format Helpers
Standardized visual styling, risk classification, and KPI card renderers.
"""

from typing import Tuple, Dict, Any


def get_risk_category(prob: float) -> Tuple[str, str, str, str]:
    """
    Map a continuous flood risk probability [0, 1] to category, hex color, badge class, and status.
    Returns: (category_name, hex_color, badge_class, icon)
    """
    if prob < 0.25:
        return "LOW", "#06d6a0", "badge-low", "🟢"
    elif prob < 0.50:
        return "MODERATE", "#ffbe0b", "badge-moderate", "🟡"
    elif prob < 0.75:
        return "HIGH", "#fb5607", "badge-high", "🟠"
    else:
        return "SEVERE", "#ef233c", "badge-severe", "🔴"


def get_road_status_category(risk_score: float) -> Tuple[str, str]:
    """
    Map road risk score to operational status and color.
    Returns: (status, hex_color)
    """
    if risk_score < 0.30:
        return "SAFE", "#06d6a0"
    elif risk_score < 0.55:
        return "CAUTION", "#ffbe0b"
    elif risk_score < 0.75:
        return "HIGH RISK", "#fb5607"
    else:
        return "BLOCKED", "#ef233c"


def render_kpi_card(title: str, value: str, sub: str = "", card_type: str = "blue") -> str:
    """Render a styled disaster management KPI card in HTML."""
    return f"""
    <div class="kpi-card kpi-{card_type}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def render_emergency_banner(title: str, message: str, level: str = "CRITICAL") -> str:
    """Render an urgent emergency alert banner."""
    color = "#ef233c" if level in ["CRITICAL", "SEVERE"] else "#fb5607" if level == "WARNING" else "#ffbe0b"
    return f"""
    <div class="emergency-banner" style="border-left-color: {color};">
        <div class="emergency-banner-title" style="color: {color};">
            🚨 {level} ALERT: {title}
        </div>
        <div style="font-size: 0.95rem; line-height: 1.4; color: #f1f5f9;">
            {message}
        </div>
    </div>
    """


def get_feature_importance_breakdown(feature_dict: Dict[str, float]) -> Dict[str, float]:
    """Normalize and format feature contributions for Explainable AI display."""
    total = sum(abs(v) for v in feature_dict.values()) or 1.0
    return {k: round((v / total) * 100, 1) for k, v in feature_dict.items()}
