"""
FloodGuard AI - Preprocessing & Feature Engineering
Extracts, scales, and transforms hydrologic and geospatial features for ML models.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "rainfall_1h",
    "rainfall_3h",
    "rainfall_6h",
    "elevation",
    "slope",
    "drainage_capacity",
    "impervious_surface",
    "historical_flood_frequency",
    "road_density",
    "tide_level_m",
    "soil_saturation"
]

TARGET_COLUMN = "flood_occurred"


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Extract feature matrix X, target vector y, and fit a StandardScaler."""
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def get_train_test_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split dataframe into train and test sets."""
    X_scaled, y, scaler = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler


def transform_single_instance(features_dict: Dict[str, float], scaler: StandardScaler) -> np.ndarray:
    """Format and scale a single dictionary of inputs into a 2D numpy array for inference."""
    row = [features_dict.get(col, 0.0) for col in FEATURE_COLUMNS]
    row_arr = np.array(row).reshape(1, -1)
    return scaler.transform(row_arr)
