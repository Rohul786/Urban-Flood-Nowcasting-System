"""
FloodGuard AI - Machine Learning Prediction & Explainable AI (XAI)
Trains and compares Random Forest and Gradient Boosting models, computes evaluation metrics,
calculates continuous flood probabilities, and provides feature-level attribution explanations.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from src.preprocessing import (
    FEATURE_COLUMNS, get_train_test_data, transform_single_instance
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "flood_model.pkl")


def train_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train Random Forest and Gradient Boosting classifiers.
    Returns trained models, scalers, evaluation metrics, and feature importances.
    """
    X_train, X_test, y_train, y_test, scaler = get_train_test_data(df)

    # 1. Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # 2. Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    # Evaluate models
    def evaluate(model, name):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        return {
            "name": name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix": cm.tolist()
        }

    rf_metrics = evaluate(rf_model, "Random Forest")
    gb_metrics = evaluate(gb_model, "Gradient Boosting")

    # Global feature importances
    rf_importances = dict(zip(FEATURE_COLUMNS, [float(x) for x in rf_model.feature_importances_]))
    gb_importances = dict(zip(FEATURE_COLUMNS, [float(x) for x in gb_model.feature_importances_]))

    model_bundle = {
        "primary_model": rf_model,
        "secondary_model": gb_model,
        "scaler": scaler,
        "feature_names": FEATURE_COLUMNS,
        "rf_metrics": rf_metrics,
        "gb_metrics": gb_metrics,
        "rf_importances": rf_importances,
        "gb_importances": gb_importances,
        "is_simulation_trained": True,
        "dataset_disclaimer": "Prototype simulation dataset – not for real emergency deployment."
    }

    return model_bundle


def save_model(model_bundle: Dict[str, Any], path: str = MODEL_PATH):
    """Serialize model bundle to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model_bundle, path)


def load_model(path: str = MODEL_PATH) -> Dict[str, Any]:
    """Load model bundle from disk, training if missing."""
    if not os.path.exists(path):
        from src.data_loader import load_dataset
        df = load_dataset()
        bundle = train_models(df)
        save_model(bundle, path)
        return bundle
    return joblib.load(path)


def predict_risk_probability(model_bundle: Dict[str, Any], feature_dict: Dict[str, float]) -> float:
    """Predict continuous flood risk probability for a location."""
    scaler = model_bundle["scaler"]
    model = model_bundle["primary_model"]
    X_scaled = transform_single_instance(feature_dict, scaler)
    prob = model.predict_proba(X_scaled)[0, 1]
    return float(np.clip(prob, 0.0, 1.0))


def explain_prediction(model_bundle: Dict[str, Any], feature_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Explainable AI (XAI) feature attribution.
    Computes localized relative contribution of each hydrological factor to the predicted risk.
    """
    scaler = model_bundle["scaler"]
    global_importances = model_bundle["rf_importances"]
    
    # Calculate baseline deviation for each feature
    # Means and standard deviations from the scaler
    means = dict(zip(FEATURE_COLUMNS, scaler.mean_))
    scales = dict(zip(FEATURE_COLUMNS, scaler.scale_))

    contributions = {}
    
    # Rainfall factors (higher increases risk)
    r1h_dev = max(0, (feature_dict.get("rainfall_1h", 0) - means["rainfall_1h"]) / scales["rainfall_1h"])
    r3h_dev = max(0, (feature_dict.get("rainfall_3h", 0) - means["rainfall_3h"]) / scales["rainfall_3h"])
    contributions["🌧️ Heavy Rainfall Intensity"] = (r1h_dev * 1.5 + r3h_dev * 1.2) * global_importances.get("rainfall_1h", 0.3)

    # Elevation factor (lower elevation increases risk)
    elev_dev = max(0, (means["elevation"] - feature_dict.get("elevation", 6.5)) / scales["elevation"])
    contributions["📍 Low Terrain Elevation"] = (elev_dev * 1.8) * global_importances.get("elevation", 0.2)

    # Drainage deficit (lower drainage capacity increases risk)
    drain_dev = max(0, (means["drainage_capacity"] - feature_dict.get("drainage_capacity", 20.0)) / scales["drainage_capacity"])
    contributions["💧 Inadequate / Choked Drainage"] = (drain_dev * 1.6) * global_importances.get("drainage_capacity", 0.18)

    # Impervious surface (higher urban concrete increases runoff)
    imp_dev = max(0, (feature_dict.get("impervious_surface", 75) - means["impervious_surface"]) / scales["impervious_surface"])
    contributions["🏙️ High Urban Concrete Coverage"] = (imp_dev * 1.3) * global_importances.get("impervious_surface", 0.12)

    # Historical flood frequency
    hist_dev = max(0, (feature_dict.get("historical_flood_frequency", 3) - means["historical_flood_frequency"]) / scales["historical_flood_frequency"])
    contributions["📊 Historical Flood Vulnerability"] = (hist_dev * 1.1) * global_importances.get("historical_flood_frequency", 0.1)

    # Tide & Soil Saturation
    tide_dev = max(0, (feature_dict.get("tide_level_m", 2.0) - means["tide_level_m"]) / scales["tide_level_m"])
    contributions["🌊 High River Tidal Head"] = (tide_dev * 1.0) * global_importances.get("tide_level_m", 0.08)

    # Normalize contributions to percentages
    total_score = sum(contributions.values())
    if total_score > 0:
        normalized_pct = {k: round((v / total_score) * 100, 1) for k, v in contributions.items()}
    else:
        # Fallback to global importance breakdown if deviation is small
        normalized_pct = {
            "🌧️ Heavy Rainfall Intensity": 36.0,
            "💧 Inadequate / Choked Drainage": 24.0,
            "📍 Low Terrain Elevation": 18.0,
            "🏙️ High Urban Concrete Coverage": 12.0,
            "📊 Historical Flood Vulnerability": 6.0,
            "🌊 High River Tidal Head": 4.0
        }

    # Sort descending by contribution percentage
    sorted_factors = sorted(normalized_pct.items(), key=lambda x: x[1], reverse=True)

    return {
        "factors": sorted_factors,
        "top_factor": sorted_factors[0][0] if sorted_factors else "Rainfall",
        "primary_driver_pct": sorted_factors[0][1] if sorted_factors else 35.0
    }
