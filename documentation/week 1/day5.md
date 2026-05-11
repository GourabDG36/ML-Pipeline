# Day 5 - Evidently AI Drift Monitoring

## What We Built
- POST /drift-report endpoint
- Drift computation using Evidently AI legacy API
- Per-feature K-S test drift scores
- Dataset-level drift detection
- Clean parser for Evidently's nested output format

---

## Files Created

### `backend/app/monitoring/drift.py`
- `load_reference_data()` - loads original training CSV as reference distribution
- `compute_drift_report()` - runs Evidently report comparing reference vs current
- `_parse_evidently_report()` - flattens Evidently's nested dict into clean format
- "file_id": "6bbfbc14-be21-462b-b1bd-9d7f73ac3ab6"

### `backend/app/api/routes/monitor.py`
- POST /drift-report endpoint
- DriftRequest, DriftResponse, FeatureDriftDetail Pydantic models
- Minimum 10 rows check before running drift computation

---

## Endpoints Added

| Method | URL | Description |
|--------|-----|-------------|
| POST | /api/v1/drift-report | Compare current data vs training data for drift |

---

## How Drift Detection Works

```
POST /drift-report (file_id + current_data list)
        |
        v
Load reference data from data/uploads/{file_id}/data.csv
Load profile.json to get feature column names
        |
        v
Check minimum 10 rows in current_data
        |
        v
Run Evidently Report with:
  - DatasetDriftMetric (overall drift)
  - DataDriftPreset (per-column drift)
        |
        v
Parse report_dict:
  - DatasetDriftMetric -> dataset_drift, drift_share, n_drifted_columns
  - DataDriftTable -> per-column drift_score, drift_detected, stat_test
        |
        v
Return clean drift report
```

---

## Evidently Version Notes

### Installed version: 0.7.21
- In 0.7.x, classic drift detection moved to `evidently.legacy`
- Correct imports:
  ```python
  from evidently.legacy.report import Report
  from evidently.legacy.metric_preset import DataDriftPreset
  from evidently.legacy.metrics import DatasetDriftMetric
  ```

### Evidently Output Structure (0.7.x)
```json
{
  "metrics": [
    {
      "metric": "DatasetDriftMetric",
      "result": {
        "drift_share": 0.5,
        "number_of_drifted_columns": 0,
        "dataset_drift": false
      }
    },
    {
      "metric": "DataDriftTable",
      "result": {
        "drift_by_columns": {
          "column_name": {
            "drift_score": 0.337,
            "drift_detected": false,
            "stattest_name": "K-S p_value",
            "stattest_threshold": 0.05
          }
        }
      }
    }
  ]
}
```

Key differences from older Evidently versions:
- Metric name is `DataDriftTable` not `DataDriftPreset`
- Threshold field is `stattest_threshold` not `threshold`
- `DatasetDriftMetric` appears twice in output - parse only first occurrence

---

## Statistical Test Used

### K-S Test (Kolmogorov-Smirnov)
- Used for numerical features
- Compares the full distribution shape, not just mean/variance
- Returns p-value: low p-value = distributions are different = drift
- Default threshold: 0.05 (5% significance level)
- WHY K-S: It's non-parametric (no assumption about distribution shape)
  and sensitive to shifts, scale changes, and shape changes

### Minimum Sample Size
- Need at least 10 rows for meaningful statistical testing
- More rows = more statistical power = more sensitive drift detection
- In production: run drift report on last 100-500 predictions per day

---

## Test Results

### With 10 rows including extreme 99.0 values:
```json
{
  "status": "no_drift",
  "drift_score": 0.5,
  "n_features_drifted": 0,
  "feature_drift": {
    "petal_length": {"drift_score": 0.3373, "drift_detected": false},
    "petal_width":  {"drift_score": 0.4218, "drift_detected": false},
    "sepal_length": {"drift_score": 0.4845, "drift_detected": false},
    "sepal_width":  {"drift_score": 0.3923, "drift_detected": false}
  }
}
```

WHY no drift despite 99.0 values: K-S test with only 10 samples lacks
statistical power to confidently flag drift. In production with 100+
samples the extreme values would clearly trigger drift detection.

---

## Complete API Summary (End of Week 1)

| Method | URL | Description |
|--------|-----|-------------|
| GET  | /api/v1/health | App + MLflow health check |
| POST | /api/v1/upload | Upload CSV, get file_id |
| POST | /api/v1/validate | Validate dataset + detect column types |
| POST | /api/v1/train | Train 4 models + Optuna + MLflow |
| POST | /api/v1/predict | Predict + SHAP explanation |
| POST | /api/v1/drift-report | Drift monitoring report |

---

## Key Concepts Learned

- **Reference vs Current data**: Drift is always relative - need a baseline
- **K-S test**: Tests if two samples come from the same distribution
- **Statistical power**: More samples = more reliable drift detection
- **Evidently legacy API**: v0.7.x moved classic monitoring to legacy module
- **DataDriftTable vs DataDriftPreset**: Internal metric name changed in newer Evidently
- **Production drift workflow**: Collect predictions, run drift report daily/weekly
