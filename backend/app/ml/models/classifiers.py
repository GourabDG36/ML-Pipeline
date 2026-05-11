"""
Model definitions for all 4 classifiers.

WHY these 4 models:
- Logistic Regression: Fast baseline, interpretable, works well on linear problems
- Random Forest: Robust ensemble, handles non-linearity, less hyperparameter sensitive
- XGBoost: Industry standard gradient boosting, usually best on tabular data
- LightGBM: Faster than XGBoost on large datasets, handles categoricals natively

Training all 4 and comparing gives the user confidence the best model was selected.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from app.core.config import settings


def get_baseline_models(n_classes: int) -> dict:
    """
    Returns all 4 models with sensible default hyperparameters.
    These defaults are good enough to establish a baseline before Optuna tuning.

    WHY sensible defaults matter: If defaults are terrible, Optuna wastes
    trials just getting back to reasonable territory instead of fine-tuning.
    """

    # For multiclass we need to set objective explicitly for XGBoost/LightGBM
    is_multiclass = n_classes > 2
    xgb_objective = "multi:softprob" if is_multiclass else "binary:logistic"
    lgbm_objective = "multiclass" if is_multiclass else "binary"

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,          # enough iterations to converge
            random_state=settings.RANDOM_SEED,
            n_jobs=-1,              # use all CPU cores
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=settings.RANDOM_SEED,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            objective=xgb_objective,
            num_class=n_classes if is_multiclass else None,
            random_state=settings.RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,            # suppress XGBoost output
            eval_metric="logloss",
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=100,
            objective=lgbm_objective,
            num_class=n_classes if is_multiclass else None,
            random_state=settings.RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,           # suppress LightGBM output
        ),
    }

    return models