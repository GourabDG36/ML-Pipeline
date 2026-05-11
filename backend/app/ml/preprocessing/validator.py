"""
Data validation before training.

WHY validate before training: Training on bad data silently produces
bad models. Catching issues early gives the user actionable feedback
instead of a cryptic training crash 2 minutes later.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def validate_dataset(df: pd.DataFrame, target_column: str) -> ValidationResult:
    """
    Runs all validation checks on the uploaded dataframe.
    Returns a ValidationResult with errors (blockers) and warnings (non-blockers).
    """
    errors = []
    warnings = []
    info = {}

    # --- CHECK 1: Target column exists ---
    if target_column not in df.columns:
        errors.append(
            f"Target column '{target_column}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )
        return ValidationResult(is_valid=False, errors=errors)

    # --- CHECK 2: Minimum row count ---
    if len(df) < 50:
        errors.append(
            f"Dataset too small: {len(df)} rows. Minimum required is 50."
        )

    # --- CHECK 3: Maximum row count ---
    if len(df) > settings_max_rows():
        warnings.append(
            f"Dataset has {len(df)} rows. Will be capped at {settings_max_rows()} for training."
        )

    # --- CHECK 4: Target column has at least 2 classes ---
    unique_classes = df[target_column].nunique()
    if unique_classes < 2:
        errors.append(
            f"Target column '{target_column}' has only {unique_classes} unique value. "
            f"Need at least 2 classes for classification."
        )

    if unique_classes > 20:
        warnings.append(
            f"Target column has {unique_classes} classes. "
            f"Consider whether this is really a classification problem."
        )

    # --- CHECK 5: Class imbalance ---
    class_counts = df[target_column].value_counts(normalize=True)
    min_class_pct = class_counts.min() * 100
    if min_class_pct < 5:
        warnings.append(
            f"Severe class imbalance detected. "
            f"Smallest class has only {min_class_pct:.1f}% of samples. "
            f"Consider oversampling or adjusting class weights."
        )

    # --- CHECK 6: Missing values ---
    null_counts = df.isnull().sum()
    null_columns = null_counts[null_counts > 0]
    if len(null_columns) > 0:
        null_info = {col: int(count) for col, count in null_columns.items()}
        warnings.append(
            f"Missing values found in {len(null_columns)} columns: {null_info}. "
            f"These will be imputed automatically."
        )
        info["null_columns"] = null_info

    # --- CHECK 7: All-null columns ---
    all_null = [col for col in df.columns if df[col].isnull().all()]
    if all_null:
        errors.append(
            f"These columns are completely empty and must be removed: {all_null}"
        )

    # --- CHECK 8: Duplicate rows ---
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        warnings.append(
            f"{n_duplicates} duplicate rows found. They will be kept but may affect model quality."
        )

    # --- CHECK 9: Target column must not be float (regression target) ---
    if df[target_column].dtype == float:
        unique_vals = df[target_column].nunique()
        if unique_vals > 20:
            errors.append(
                f"Target column '{target_column}' looks like a continuous variable "
                f"({unique_vals} unique float values). This pipeline only supports classification."
            )

    # --- CHECK 10: Feature columns (everything except target) ---
    feature_cols = [c for c in df.columns if c != target_column]
    if len(feature_cols) == 0:
        errors.append("No feature columns found. Dataset must have at least one feature besides the target.")

    # Collect summary info
    info["total_rows"] = len(df)
    info["total_columns"] = len(df.columns)
    info["feature_count"] = len(feature_cols)
    info["target_classes"] = int(unique_classes)
    info["class_distribution"] = class_counts.to_dict()

    is_valid = len(errors) == 0

    if is_valid:
        logger.info("Dataset validation passed. {} rows, {} features, {} classes",
                    len(df), len(feature_cols), unique_classes)
    else:
        logger.warning("Dataset validation failed: {}", errors)

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        info=info
    )


def settings_max_rows():
    from app.core.config import settings
    return settings.MAX_TRAINING_ROWS