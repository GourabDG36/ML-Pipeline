"""
Data drift monitoring using Evidently AI.

WHY drift monitoring: Models are trained on historical data. When the
real world changes, incoming data distribution shifts away from training
data. The model keeps predicting confidently but incorrectly.
Drift monitoring catches this BEFORE it becomes a business problem.
"""

import json
import pandas as pd
import numpy as np
from loguru import logger

from app.core.config import settings


def load_reference_data(file_id: str):
    upload_folder = settings.UPLOAD_DIR / file_id

    with open(upload_folder / "profile.json", "r") as f:
        profile = json.load(f)

    df = pd.read_csv(upload_folder / "data.csv")

    feature_cols = (
        profile["numerical_features"] +
        profile["categorical_features"]
    )
    return df[feature_cols], profile


def compute_drift_report(file_id: str, current_data: list[dict]) -> dict:
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.metrics import DatasetDriftMetric

    # 1. Load reference data
    reference_df, profile = load_reference_data(file_id)
    feature_cols = (
        profile["numerical_features"] +
        profile["categorical_features"]
    )

    # 2. Convert current data to DataFrame
    current_df = pd.DataFrame(current_data)[feature_cols]

    logger.info(
        "Computing drift: {} reference rows vs {} current rows",
        len(reference_df),
        len(current_df)
    )

    # 3. Minimum data check
    if len(current_df) < 10:
        return {
            "status": "insufficient_data",
            "message": f"Need at least 10 rows to compute drift. Got {len(current_df)}.",
            "dataset_drift_detected": False,
            "drift_score": 0.0,
            "feature_drift": {},
            "n_features_drifted": 0,
            "reference_rows": len(reference_df),
            "current_rows": len(current_df),
        }

    # 4. Build and run Evidently report
    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftPreset(),
    ])

    report.run(
        reference_data=reference_df,
        current_data=current_df
    )

    # 5. Extract results
    report_dict = report.as_dict()

    return _parse_evidently_report(
        report_dict,
        feature_cols,
        len(reference_df),
        len(current_df)
    )


def _parse_evidently_report(
    report_dict: dict,
    feature_cols: list,
    n_reference: int,
    n_current: int
) -> dict:
    feature_drift = {}
    dataset_drift_detected = False
    dataset_drift_score = 0.0
    n_drifted = 0
    dataset_metric_parsed = False

    try:
        metrics = report_dict.get("metrics", [])

        for metric in metrics:
            metric_id = metric.get("metric", "")
            result = metric.get("result", {})

            # Dataset level — parse only first occurrence
            if metric_id == "DatasetDriftMetric" and not dataset_metric_parsed:
                dataset_drift_detected = result.get("dataset_drift", False)
                dataset_drift_score = round(float(result.get("drift_share", 0.0)), 4)
                n_drifted = int(result.get("number_of_drifted_columns", 0))
                dataset_metric_parsed = True

            # Per column drift from DataDriftTable
            if metric_id == "DataDriftTable":
                drift_by_col = result.get("drift_by_columns", {})
                for col_name, col_data in drift_by_col.items():
                    if col_name in feature_cols:
                        feature_drift[col_name] = {
                            "drift_detected": bool(col_data.get("drift_detected", False)),
                            "drift_score": round(float(col_data.get("drift_score", 0.0)), 4),
                            "stat_test": col_data.get("stattest_name", "unknown"),
                            "threshold": round(float(col_data.get("stattest_threshold", 0.05)), 4),
                        }

    except Exception as e:
        logger.warning("Error parsing Evidently report: {}", e)

    return {
        "status": "drift_detected" if dataset_drift_detected else "no_drift",
        "message": (
            f"Drift detected in {n_drifted} out of {len(feature_cols)} features."
            if dataset_drift_detected
            else "No significant drift detected. Data distribution is stable."
        ),
        "dataset_drift_detected": dataset_drift_detected,
        "drift_score": dataset_drift_score,
        "n_features_drifted": n_drifted,
        "feature_drift": feature_drift,
        "reference_rows": n_reference,
        "current_rows": n_current,
    }