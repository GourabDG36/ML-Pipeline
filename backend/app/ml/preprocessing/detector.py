"""
Auto-detect column types.

WHY auto-detect: Users shouldn't need to tell us which columns are
categorical vs numerical. We infer it from the data, which is what
any good AutoML tool does.
"""

import pandas as pd
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ColumnProfile:
    numerical_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    datetime_features: list[str] = field(default_factory=list)
    dropped_features: list[str] = field(default_factory=list)
    drop_reasons: dict[str, str] = field(default_factory=dict)


def detect_column_types(df: pd.DataFrame, target_column: str) -> ColumnProfile:
    """
    Automatically detect and categorize column types.
    Drops columns that would hurt model quality (IDs, free text, constants).
    """
    profile = ColumnProfile()
    feature_cols = [c for c in df.columns if c != target_column]

    for col in feature_cols:
        series = df[col]
        n_unique = series.nunique()
        n_rows = len(series)

        # --- DROP: Constant columns (zero variance) ---
        if n_unique <= 1:
            profile.dropped_features.append(col)
            profile.drop_reasons[col] = "Constant column (only one unique value)"
            continue

        # --- DROP: ID-like columns (almost all unique) ---
        uniqueness_ratio = n_unique / n_rows
        if uniqueness_ratio > 0.95 and series.dtype == object:
            profile.dropped_features.append(col)
            profile.drop_reasons[col] = f"Likely an ID column ({n_unique} unique values out of {n_rows} rows)"
            continue

        # --- DATETIME: Try to parse as date ---
        if series.dtype == object:
            try:
                pd.to_datetime(series.dropna().head(50))
                profile.datetime_features.append(col)
                continue
            except Exception:
                pass

        # --- CATEGORICAL: String columns or low-cardinality integers ---
        if series.dtype == object:
            profile.categorical_features.append(col)
            continue

        # --- CATEGORICAL: Integers with very few unique values ---
        if pd.api.types.is_integer_dtype(series) and n_unique <= 15:
            profile.categorical_features.append(col)
            continue

        # --- NUMERICAL: Everything else ---
        if pd.api.types.is_numeric_dtype(series):
            profile.numerical_features.append(col)
            continue

        # --- DEFAULT: Treat as categorical ---
        profile.categorical_features.append(col)

    logger.info(
        "Column detection complete. Numerical: {}, Categorical: {}, Dropped: {}",
        len(profile.numerical_features),
        len(profile.categorical_features),
        len(profile.dropped_features)
    )

    if profile.dropped_features:
        logger.warning("Dropped columns: {}", profile.drop_reasons)

    return profile