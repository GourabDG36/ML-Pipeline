"""
Model evaluation metrics.

WHY F1 as the selection metric: Accuracy is misleading on imbalanced datasets.
F1 balances precision and recall. We use macro-F1 which treats all classes equally.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from loguru import logger


def evaluate_model(model, X_test, y_test, n_classes: int) -> dict:
    """
    Evaluate a trained model and return all metrics as a flat dict.
    Flat dict is important because MLflow logs metrics as key-value pairs.
    """
    y_pred = model.predict(X_test)

    # For ROC-AUC we need probability scores
    try:
        y_prob = model.predict_proba(X_test)
    except Exception:
        y_prob = None

    average = "macro"  # macro = unweighted mean across classes

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=average, zero_division=0)),
    }

    # ROC-AUC needs special handling for multiclass
    if y_prob is not None:
        try:
            if n_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_test, y_prob, multi_class="ovr", average=average)
                )
        except Exception as e:
            logger.warning("Could not compute ROC-AUC: {}", e)
            metrics["roc_auc"] = 0.0

    logger.info("Metrics: {}", metrics)
    return metrics