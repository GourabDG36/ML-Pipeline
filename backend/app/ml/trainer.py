"""
Core training engine.

Orchestrates: preprocessing -> baseline training -> Optuna tuning
-> MLflow tracking -> best model selection -> model registration
"""

import time
import mlflow
import mlflow.sklearn
import optuna
import numpy as np
from loguru import logger

from app.core.config import settings
from app.ml.models.classifiers import get_baseline_models
from app.ml.evaluation.metrics import evaluate_model

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_all_models(data: dict, file_id: str) -> dict:
    """
    Train all 4 models, track with MLflow, tune with Optuna,
    return results for all models and the best model name.
    """

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    n_classes = data["n_classes"]

    results = {}
    best_model = None
    best_f1 = -1
    best_model_name = None

    baseline_models = get_baseline_models(n_classes)

    for model_name, model in baseline_models.items():
        logger.info("Training: {}", model_name)

        with mlflow.start_run(run_name=f"{file_id[:8]}-{model_name}"):

            # Log metadata
            mlflow.set_tag("file_id", file_id)
            mlflow.set_tag("model_name", model_name)

            # Train and time it
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            # Evaluate
            metrics = evaluate_model(model, X_test, y_test, n_classes)
            metrics["training_time_seconds"] = round(train_time, 3)

            # Log metrics to MLflow
            mlflow.log_metrics(metrics)
            mlflow.log_params(model.get_params())

            # Log model artifact
            mlflow.sklearn.log_model(model, artifact_path="model")

            results[model_name] = {
                "metrics": metrics,
                "run_id": mlflow.active_run().info.run_id,
            }

            logger.info("{} -> F1: {:.4f}", model_name, metrics["f1"])

            # Track best
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model = model
                best_model_name = model_name

    logger.info("Best model: {} with F1: {:.4f}", best_model_name, best_f1)

    # Run Optuna tuning on the best model only
    logger.info("Starting Optuna tuning for: {}", best_model_name)
    tuned_model, tuned_metrics, best_params = tune_best_model(
        best_model_name, X_train, y_train, X_test, y_test, n_classes
    )

    # Register the tuned best model in MLflow Model Registry
    with mlflow.start_run(run_name=f"{file_id[:8]}-{best_model_name}-tuned"):
        mlflow.set_tag("file_id", file_id)
        mlflow.set_tag("model_name", best_model_name)
        mlflow.set_tag("tuned", "true")
        mlflow.log_metrics(tuned_metrics)
        mlflow.log_params(best_params)

        model_info = mlflow.sklearn.log_model(
            tuned_model,
            artifact_path="model",
            registered_model_name=settings.REGISTERED_MODEL_NAME,
        )

        results["tuned_best"] = {
            "model_name": best_model_name,
            "metrics": tuned_metrics,
            "run_id": mlflow.active_run().info.run_id,
            "model_uri": model_info.model_uri,
        }

    return {
        "all_results": results,
        "best_model_name": best_model_name,
        "best_model": tuned_model,
        "best_metrics": tuned_metrics,
    }


def tune_best_model(model_name, X_train, y_train, X_test, y_test, n_classes):
    """
    Run Optuna hyperparameter search on the best model.
    WHY only tune the best: Tuning all 4 would take too long and
    the best baseline model is usually still best after tuning.
    """

    def objective(trial):
        if model_name == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            params = {
                "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
                "max_iter": 1000,
                "random_state": settings.RANDOM_SEED,
            }
            model = LogisticRegression(**params)

        elif model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "random_state": settings.RANDOM_SEED,
            }
            model = RandomForestClassifier(**params)

        elif model_name == "xgboost":
            from xgboost import XGBClassifier
            is_multiclass = n_classes > 2
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "objective": "multi:softprob" if is_multiclass else "binary:logistic",
                "num_class": n_classes if is_multiclass else None,
                "random_state": settings.RANDOM_SEED,
                "verbosity": 0,
            }
            model = XGBClassifier(**params)

        elif model_name == "lightgbm":
            from lightgbm import LGBMClassifier
            is_multiclass = n_classes > 2
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "objective": "multiclass" if is_multiclass else "binary",
                "num_class": n_classes if is_multiclass else None,
                "random_state": settings.RANDOM_SEED,
                "verbosity": -1,
            }
            model = LGBMClassifier(**params)

        model.fit(X_train, y_train)
        from sklearn.metrics import f1_score
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred, average="macro", zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=settings.OPTUNA_TRIALS)

    # Retrain with best params on full training data
    best_params = study.best_params
    logger.info("Best Optuna params: {}", best_params)

    # Rebuild model with best params
    objective(study.best_trial)  # just to get the model - rebuild properly below
    best_trial_model = _build_model_from_params(model_name, best_params, n_classes)
    best_trial_model.fit(X_train, y_train)

    metrics = evaluate_model(best_trial_model, X_test, y_test, n_classes)
    return best_trial_model, metrics, best_params


def _build_model_from_params(model_name, params, n_classes):
    """Rebuild a model from a flat params dict."""
    is_multiclass = n_classes > 2

    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            **{k: v for k, v in params.items()},
            max_iter=1000,
            random_state=settings.RANDOM_SEED
        )
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            **params,
            random_state=settings.RANDOM_SEED
        )
    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            **params,
            objective="multi:softprob" if is_multiclass else "binary:logistic",
            num_class=n_classes if is_multiclass else None,
            random_state=settings.RANDOM_SEED,
            verbosity=0,
        )
    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            **params,
            objective="multiclass" if is_multiclass else "binary",
            num_class=n_classes if is_multiclass else None,
            random_state=settings.RANDOM_SEED,
            verbosity=-1,
        )