"""
FloodGuard AI - Model Training Script
Trains Random Forest and Gradient Boosting models and serializes the model bundle.
"""

import sys
import os

# Configure stdout encoding for Windows console compatibility
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_dataset
from src.model import train_models, save_model, MODEL_PATH


def main():
    print("==================================================")
    print("FloodGuard AI - Model Training & Benchmarking")
    print("==================================================")

    print("Loading synthetic hydrologic dataset...")
    df = load_dataset()
    print(f"Dataset loaded: {len(df)} records with {len(df.columns)} features.")

    print("\nTraining Random Forest and Gradient Boosting classifiers...")
    bundle = train_models(df)

    rf_m = bundle["rf_metrics"]
    gb_m = bundle["gb_metrics"]

    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    print(f"Random Forest     -> Accuracy: {rf_m['accuracy']:.4f} | Precision: {rf_m['precision']:.4f} | Recall: {rf_m['recall']:.4f} | F1: {rf_m['f1']:.4f} | ROC-AUC: {rf_m['roc_auc']:.4f}")
    print(f"Gradient Boosting -> Accuracy: {gb_m['accuracy']:.4f} | Precision: {gb_m['precision']:.4f} | Recall: {gb_m['recall']:.4f} | F1: {gb_m['f1']:.4f} | ROC-AUC: {gb_m['roc_auc']:.4f}")

    print("\n--- TOP GLOBAL FEATURE IMPORTANCES (Random Forest) ---")
    for feat, imp in sorted(bundle["rf_importances"].items(), key=lambda x: x[1], reverse=True)[:6]:
        print(f" - {feat:30s}: {imp * 100:.2f}%")

    print(f"\nSaving serialized model bundle to: {MODEL_PATH}")
    save_model(bundle, MODEL_PATH)
    print("Model training complete and successfully serialized!")


if __name__ == "__main__":
    main()
