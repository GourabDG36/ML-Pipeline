"""
Unit tests for the data layer.
These run in GitHub Actions on every push.
"""

import pytest
import pandas as pd
import numpy as np
from app.ml.preprocessing.validator import validate_dataset
from app.ml.preprocessing.detector import detect_column_types


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def valid_df():
    """A clean, valid classification dataset."""
    np.random.seed(42)
    return pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.choice(["a", "b", "c"], 100),
        "target": np.random.choice(["class1", "class2", "class3"], 100),
    })


@pytest.fixture
def iris_df():
    """Minimal iris-like dataset."""
    return pd.DataFrame({
        "sepal_length": [5.1, 4.9, 6.7, 6.3, 5.8] * 20,
        "sepal_width":  [3.5, 3.0, 3.1, 3.3, 2.7] * 20,
        "petal_length": [1.4, 1.4, 4.7, 6.0, 5.1] * 20,
        "petal_width":  [0.2, 0.2, 1.5, 2.5, 1.9] * 20,
        "species": ["setosa", "setosa", "versicolor", "virginica", "virginica"] * 20,
    })


# ── Validator Tests ────────────────────────────────────────────────────────

class TestValidator:

    def test_valid_dataset_passes(self, valid_df):
        result = validate_dataset(valid_df, "target")
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_target_column(self, valid_df):
        result = validate_dataset(valid_df, "nonexistent_column")
        assert result.is_valid is False
        assert any("not found" in e for e in result.errors)

    def test_too_few_rows(self):
        small_df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "target": ["a", "b", "c"]
        })
        result = validate_dataset(small_df, "target")
        assert result.is_valid is False
        assert any("too small" in e for e in result.errors)

    def test_single_class_target(self):
        df = pd.DataFrame({
            "feature1": range(100),
            "target": ["only_class"] * 100
        })
        result = validate_dataset(df, "target")
        assert result.is_valid is False
        assert any("classes" in e for e in result.errors)

    def test_class_imbalance_warning(self):
        df = pd.DataFrame({
            "feature1": range(100),
            "target": ["majority"] * 97 + ["minority"] * 3
        })
        result = validate_dataset(df, "target")
        assert any("imbalance" in w.lower() for w in result.warnings)

    def test_missing_values_warning(self, valid_df):
        valid_df.loc[0:5, "feature1"] = np.nan
        result = validate_dataset(valid_df, "target")
        assert any("Missing" in w for w in result.warnings)

    def test_all_null_column_blocked(self, valid_df):
        valid_df["empty_col"] = np.nan
        result = validate_dataset(valid_df, "target")
        assert result.is_valid is False
        assert any("completely empty" in e for e in result.errors)

    def test_info_contains_class_distribution(self, iris_df):
        result = validate_dataset(iris_df, "species")
        assert "class_distribution" in result.info
        assert "target_classes" in result.info
        assert result.info["target_classes"] == 3


# ── Detector Tests ─────────────────────────────────────────────────────────

class TestDetector:

    def test_numerical_columns_detected(self, iris_df):
        profile = detect_column_types(iris_df, "species")
        assert "sepal_length" in profile.numerical_features
        assert "petal_width" in profile.numerical_features

    def test_categorical_columns_detected(self, valid_df):
        profile = detect_column_types(valid_df, "target")
        assert "feature3" in profile.categorical_features

    def test_target_not_in_features(self, iris_df):
        profile = detect_column_types(iris_df, "species")
        assert "species" not in profile.numerical_features
        assert "species" not in profile.categorical_features

    def test_constant_column_dropped(self, valid_df):
        valid_df["constant"] = 42
        profile = detect_column_types(valid_df, "target")
        assert "constant" in profile.dropped_features

    def test_id_column_dropped(self):
        df = pd.DataFrame({
            "id": [f"user_{i}" for i in range(100)],
            "feature1": range(100),
            "target": ["a", "b"] * 50,
        })
        profile = detect_column_types(df, "target")
        assert "id" in profile.dropped_features

    def test_low_cardinality_integer_is_categorical(self):
        df = pd.DataFrame({
            "category_int": [1, 2, 3, 1, 2, 3] * 20,
            "feature1": range(120),
            "target": ["a", "b"] * 60,
        })
        profile = detect_column_types(df, "target")
        assert "category_int" in profile.categorical_features


# ── Preprocessing Pipeline Tests ───────────────────────────────────────────

class TestPreprocessingPipeline:

    def test_prepare_data_returns_correct_keys(self, iris_df):
        from app.ml.preprocessing.detector import detect_column_types
        from app.ml.preprocessing.pipeline import prepare_data

        profile = detect_column_types(iris_df, "species")
        data = prepare_data(iris_df, "species", profile)

        assert "X_train" in data
        assert "X_test" in data
        assert "y_train" in data
        assert "y_test" in data
        assert "preprocessor" in data
        assert "label_encoder" in data
        assert "class_names" in data

    def test_train_test_split_sizes(self, iris_df):
        from app.ml.preprocessing.detector import detect_column_types
        from app.ml.preprocessing.pipeline import prepare_data

        profile = detect_column_types(iris_df, "species")
        data = prepare_data(iris_df, "species", profile, test_size=0.2)

        total = len(data["X_train"]) + len(data["X_test"])
        assert total == len(iris_df)
        assert len(data["X_test"]) == pytest.approx(len(iris_df) * 0.2, abs=2)

    def test_preprocessor_transforms_correctly(self, iris_df):
        from app.ml.preprocessing.detector import detect_column_types
        from app.ml.preprocessing.pipeline import prepare_data

        profile = detect_column_types(iris_df, "species")
        data = prepare_data(iris_df, "species", profile)

        # Output should have same number of rows as input split
        assert data["X_train"].shape[0] == len(data["y_train"])
        assert data["X_test"].shape[0] == len(data["y_test"])

    def test_label_encoder_covers_all_classes(self, iris_df):
        from app.ml.preprocessing.detector import detect_column_types
        from app.ml.preprocessing.pipeline import prepare_data

        profile = detect_column_types(iris_df, "species")
        data = prepare_data(iris_df, "species", profile)

        assert set(data["class_names"]) == {"setosa", "versicolor", "virginica"}