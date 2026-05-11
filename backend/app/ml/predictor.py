"""
Prediction engine.

WHY load from MLflow Model Registry instead of disk:
The registry is the single source of truth for what model is in production.
If we retrain and register a new version, predictions automatically
use the latest without any code changes.
"""

import json
import numpy as np
import pandas as pd
import mlflow.sklearn
import shap
import joblib
from pathlib import Path
from loguru import logger

from app.core.config import settings


def load_latest_model():
    """
    Load the latest version of the registered best model from MLflow.
    WHY 'models:/' URI: This always points to the latest registered version,
    so retraining auto-updates predictions without touching this code.
    """
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    model_uri = f"models:/{settings.REGISTERED_MODEL_NAME}/latest"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Loaded model from registry: {}", model_uri)
        return model
    except Exception as e:
        logger.error("Failed to load model from registry: {}", e)
        raise RuntimeError(
            f"No registered model found. Please train first. Error: {e}"
        )


def load_session_artifacts(file_id: str) -> dict:
    """
    Load preprocessor, label encoder, and column profile for a given file_id.
    These were saved during training and are needed to transform new data
    the exact same way the training data was transformed.
    """
    upload_folder = settings.UPLOAD_DIR / file_id

    preprocessor = joblib.load(upload_folder / "preprocessor.pkl")
    label_encoder = joblib.load(upload_folder / "label_encoder.pkl")

    with open(upload_folder / "profile.json", "r") as f:
        profile = json.load(f)

    return {
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "profile": profile,
    }


def compute_shap_values(
    model,
    X_processed: np.ndarray,
    feature_names: list,
    file_id: str
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    try:
        # Load training data as SHAP background
        X_train_path = settings.UPLOAD_DIR / file_id / "X_train.pkl"
        if X_train_path.exists():
            X_background = joblib.load(X_train_path)
            # Use a sample of training data as background (max 100 rows)
            if len(X_background) > 100:
                idx = np.random.choice(len(X_background), 100, replace=False)
                X_background = X_background[idx]
        else:
            X_background = X_processed

        # Tree-based models
        if hasattr(model, "estimators_") or \
           "XGB" in type(model).__name__ or \
           "LGBM" in type(model).__name__ or \
           isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model, X_background)
            shap_values = explainer.shap_values(X_processed)

        # Linear models
        elif isinstance(model, LogisticRegression):
            masker = shap.maskers.Independent(X_background)
            explainer = shap.LinearExplainer(model, masker)
            shap_values = explainer.shap_values(X_processed)

        # Fallback
        else:
            explainer = shap.KernelExplainer(
                model.predict_proba, X_background[:20]
            )
            shap_values = explainer.shap_values(X_processed)

        shap_array = np.array(shap_values)
        logger.info("SHAP array shape: {}", shap_array.shape)

        # Handle all possible shapes:
        # (1, features) - binary
        # (1, features, classes) - multiclass 3D
        # list of (1, features) - multiclass list
        if isinstance(shap_values, list):
            # list of arrays, one per class
            combined = np.stack([np.abs(sv) for sv in shap_values], axis=-1)
            sample_shap = combined[0].mean(axis=-1)
        elif shap_array.ndim == 3:
            # (samples, features, classes)
            sample_shap = np.abs(shap_array[0]).mean(axis=-1)
        elif shap_array.ndim == 2:
            # (samples, features)
            sample_shap = np.abs(shap_array[0])
        else:
            sample_shap = np.abs(shap_array)

        logger.info("Sample SHAP: {}", sample_shap)

        shap_dict = {
            feature_names[i]: round(float(sample_shap[i]), 6)
            for i in range(min(len(feature_names), len(sample_shap)))
        }
        shap_sorted = dict(
            sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return {"shap_values": shap_sorted, "error": None}

    except Exception as e:
        logger.warning("SHAP computation failed: {}", e)
        return {"shap_values": {}, "error": str(e)}
    

def predict_single(file_id: str, input_data: dict) -> dict:
    """
    Full prediction pipeline for a single input row.

    Steps:
    1. Load model from MLflow registry
    2. Load preprocessor artifacts from session
    3. Convert input dict to DataFrame
    4. Transform with fitted preprocessor
    5. Get prediction + probabilities
    6. Compute SHAP explanation
    7. Return everything
    """

    # 1. Load artifacts
    artifacts = load_session_artifacts(file_id)
    preprocessor = artifacts["preprocessor"]
    label_encoder = artifacts["label_encoder"]
    profile = artifacts["profile"]

    # 2. Load model
    model = load_latest_model()

    # 3. Build feature list from profile
    feature_cols = (
        profile["numerical_features"] +
        profile["categorical_features"]
    )

    # 4. Validate input has required features
    missing = [f for f in feature_cols if f not in input_data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # 5. Convert to DataFrame (preprocessor expects DataFrame)
    input_df = pd.DataFrame([input_data])[feature_cols]

    # 6. Transform using fitted preprocessor
    X_processed = preprocessor.transform(input_df)

    # 7. Get feature names after preprocessing
    # (OHE expands categorical columns so we need the new names)
    try:
        feature_names_out = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names_out = [f"feature_{i}" for i in range(X_processed.shape[1])]

    # 8. Predict
    prediction_encoded = model.predict(X_processed)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    # 9. Prediction probabilities
    try:
        probabilities_raw = model.predict_proba(X_processed)[0]
        probabilities = {
            str(label_encoder.classes_[i]): round(float(probabilities_raw[i]), 4)
            for i in range(len(label_encoder.classes_))
        }
        confidence = round(float(probabilities_raw.max()), 4)
    except Exception:
        probabilities = {}
        confidence = None

    # 10. SHAP explanation
    shap_result = compute_shap_values(model, X_processed, feature_names_out, file_id)

    return {
        "prediction": str(prediction_label),
        "confidence": confidence,
        "probabilities": probabilities,
        "shap_values": shap_result["shap_values"],
        "shap_error": shap_result["error"],
        "features_used": feature_cols,
        "model_name": type(model).__name__,
    }