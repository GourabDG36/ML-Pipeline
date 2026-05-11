"""
Scikit-learn preprocessing pipeline.

WHY a Pipeline object instead of manual preprocessing:
1. No data leakage - fit only on train set, transform applied to test/predict
2. Single object to save/load - preprocessor + model stay in sync
3. Identical transformations at training time and prediction time guaranteed
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
from loguru import logger

from app.ml.preprocessing.detector import ColumnProfile


def build_preprocessor(profile: ColumnProfile) -> ColumnTransformer:
    """
    Builds a ColumnTransformer that handles numerical and categorical
    columns differently, all in one sklearn-compatible object.

    WHY ColumnTransformer: Lets us apply different transformations to
    different column types simultaneously without writing loops.
    """

    transformers = []

    # --- Numerical pipeline ---
    # Step 1: Fill missing values with column median (robust to outliers)
    # Step 2: Scale to zero mean, unit variance (helps LR and distance-based models)
    if profile.numerical_features:
        numerical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append((
            "numerical",
            numerical_pipeline,
            profile.numerical_features
        ))

    # --- Categorical pipeline ---
    # Step 1: Fill missing values with most frequent value
    # Step 2: One-hot encode (handle_unknown="ignore" so new categories
    #         at prediction time don't crash the model)
    if profile.categorical_features:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False  # return dense array, easier to work with
            )),
        ])
        transformers.append((
            "categorical",
            categorical_pipeline,
            profile.categorical_features
        ))

    # remainder="drop" means dropped_features are automatically excluded
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    profile: ColumnProfile,
    test_size: float = 0.2,
    random_seed: int = 42,
):
    """
    Full data preparation flow:
    1. Drop useless columns
    2. Split into X and y
    3. Encode target labels
    4. Train/test split
    5. Fit preprocessor on train, transform both splits

    Returns everything needed for training.
    """
    from sklearn.model_selection import train_test_split

    # 1. Select only useful columns
    useful_cols = profile.numerical_features + profile.categorical_features
    X = df[useful_cols].copy()
    y = df[target_column].copy()

    logger.info("Preparing data: {} samples, {} features", len(X), len(useful_cols))

    # 2. Encode target labels to integers
    # WHY: XGBoost and LightGBM require integer targets, not strings
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logger.info("Target classes: {}", list(label_encoder.classes_))

    # 3. Train/test split BEFORE fitting preprocessor
    # WHY: If we fit on the full dataset first, test data leaks into training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_seed,
        stratify=y_encoded  # preserve class distribution in both splits
    )

    logger.info("Train size: {}, Test size: {}", len(X_train), len(X_test))

    # 4. Build and fit preprocessor on TRAIN only
    preprocessor = build_preprocessor(profile)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)  # only transform, never fit

    logger.info(
        "Preprocessing complete. Feature shape: {} -> {}",
        X_train.shape,
        X_train_processed.shape
    )

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "feature_names": useful_cols,
        "class_names": list(label_encoder.classes_),
        "n_classes": len(label_encoder.classes_),
    }


def save_preprocessor(preprocessor, label_encoder, upload_folder: Path):
    """Save fitted preprocessor and label encoder to disk."""
    joblib.dump(preprocessor, upload_folder / "preprocessor.pkl")
    joblib.dump(label_encoder, upload_folder / "label_encoder.pkl")
    logger.info("Preprocessor saved to {}", upload_folder)


def load_preprocessor(upload_folder: Path):
    """Load fitted preprocessor and label encoder from disk."""
    preprocessor = joblib.load(upload_folder / "preprocessor.pkl")
    label_encoder = joblib.load(upload_folder / "label_encoder.pkl")
    return preprocessor, label_encoder

def save_training_data(X_train: np.ndarray, upload_folder: Path):
    """Save processed training data for SHAP background."""
    joblib.dump(X_train, upload_folder / "X_train.pkl")
    logger.info("Training data saved for SHAP background")