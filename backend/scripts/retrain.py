"""
Standalone retraining script for CI/CD.
Runs when new data is pushed to data/sample/.
Finds the latest CSV in data/sample/, trains all models,
registers the best model in MLflow.
"""

import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.ml.preprocessing.validator import validate_dataset
from app.ml.preprocessing.detector import detect_column_types
from app.ml.preprocessing.pipeline import prepare_data, save_preprocessor, save_training_data
from app.ml.trainer import train_all_models
from loguru import logger
import pandas as pd


def find_latest_csv() -> Path:
    """Find the most recently modified CSV in data/sample/."""
    sample_dir = Path(__file__).parent.parent.parent / "data" / "sample"
    csvs = list(sample_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {sample_dir}")
    latest = max(csvs, key=lambda p: p.stat().st_mtime)
    logger.info("Found dataset: {}", latest)
    return latest


def main():
    logger.info("Starting retraining pipeline")

    # 1. Find dataset
    csv_path = find_latest_csv()
    df = pd.read_csv(csv_path)
    logger.info("Loaded {} rows, {} columns", len(df), len(df.columns))

    # 2. Get target column from env or use last column as default
    import os
    target_column = os.getenv("TARGET_COLUMN", df.columns[-1])
    logger.info("Target column: {}", target_column)

    # 3. Validate
    validation = validate_dataset(df, target_column)
    if not validation.is_valid:
        logger.error("Validation failed: {}", validation.errors)
        sys.exit(1)

    # 4. Detect columns
    profile = detect_column_types(df, target_column)

    # 5. Prepare data
    data = prepare_data(
        df=df,
        target_column=target_column,
        profile=profile,
        test_size=settings.TEST_SIZE,
        random_seed=settings.RANDOM_SEED,
    )

    # 6. Save artifacts to a fixed location for CI
    output_dir = Path(__file__).parent.parent / "ci_artifacts"
    output_dir.mkdir(exist_ok=True)
    save_preprocessor(data["preprocessor"], data["label_encoder"], output_dir)
    save_training_data(data["X_train"], output_dir)

    # Save profile
    profile_data = {
        "numerical_features": profile.numerical_features,
        "categorical_features": profile.categorical_features,
        "target_column": target_column,
        "class_names": data["class_names"],
    }
    with open(output_dir / "profile.json", "w") as f:
        json.dump(profile_data, f)

    # 7. Train
    file_id = "ci-retrain"
    results = train_all_models(data, file_id)

    # 8. Save report
    report = {
        "dataset": str(csv_path.name),
        "rows": len(df),
        "best_model": results["best_model_name"],
        "best_metrics": results["best_metrics"],
        "all_results": {
            k: {"metrics": v["metrics"]}
            for k, v in results["all_results"].items()
            if "metrics" in v
        }
    }

    report_path = Path(__file__).parent.parent / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Training complete. Best model: {}", results["best_model_name"])
    logger.info("Report saved to: {}", report_path)
    logger.info("Best metrics: {}", results["best_metrics"])


if __name__ == "__main__":
    main()
