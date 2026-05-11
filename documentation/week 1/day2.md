# Day 2 - CSV Upload & Data Validation

## What We Built
- CSV file upload endpoint with size and extension validation
- Unique file ID system so multiple users don't overwrite each other
- Automatic data validation (nulls, imbalance, duplicates, wrong types)
- Automatic column type detection (numerical vs categorical vs ID vs datetime)
- Validate endpoint that returns full dataset profile

---

## Files Created

### `backend/app/api/routes/upload.py`
Two endpoints: /upload and /validate

### `backend/app/ml/preprocessing/validator.py`
Runs 10 validation checks on any uploaded dataset

### `backend/app/ml/preprocessing/detector.py`
Auto-detects column types from data characteristics

---

## Endpoints Added

| Method | URL | Description |
|--------|-----|-------------|
| POST | /api/v1/upload | Upload CSV, get back file_id |
| POST | /api/v1/validate | Validate dataset, detect column types |

---

## How the Upload Flow Works

```
User uploads CSV
      |
      v
Check extension (.csv only)
      |
      v
Check file size (max 50MB)
      |
      v
Generate UUID as file_id
      |
      v
Save to data/uploads/{file_id}/data.csv
      |
      v
Quick parse to get rows/columns
      |
      v
Return file_id to user
```

---

## Validation Checks (validator.py)

| Check | Type | What it does |
|-------|------|-------------|
| Target column exists | ERROR | Blocks training if target not found |
| Minimum 50 rows | ERROR | Blocks training on tiny datasets |
| Maximum row count | WARNING | Warns if dataset will be capped |
| At least 2 classes | ERROR | Blocks if only one class in target |
| More than 20 classes | WARNING | Warns about high cardinality target |
| Class imbalance < 5% | WARNING | Warns about severe imbalance |
| Missing values | WARNING | Reports which columns have nulls |
| All-null columns | ERROR | Blocks if any column is completely empty |
| Duplicate rows | WARNING | Reports count of duplicates |
| Float target column | ERROR | Blocks regression targets |
| No feature columns | ERROR | Blocks if only target column exists |

- ERRORS = blockers, training will not proceed
- WARNINGS = non-blockers, training proceeds but user is informed

---

## Column Type Detection Rules (detector.py)

```
For each feature column:
      |
      |-- n_unique <= 1 -----------> DROP (constant column)
      |
      |-- uniqueness > 95%
      |   AND dtype == object -----> DROP (likely ID column)
      |
      |-- parseable as datetime ---> DATETIME (logged, not used in training)
      |
      |-- dtype == object ---------> CATEGORICAL
      |
      |-- integer AND
      |   n_unique <= 15 ----------> CATEGORICAL
      |
      |-- numeric -----------------> NUMERICAL
      |
      |-- default -----------------> CATEGORICAL
```

---

## Test Results with Iris Dataset

Upload response:
```json
{
  "file_id": "ab7271d4-a8fd-4b41-b2c9-4d1a0c1c4d0c",
  "filename": "iris.csv",
  "rows": 150,
  "columns": 5,
  "column_names": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
  "message": "File uploaded successfully. Use file_id for training."
}
```

Validate response:
```json
{
  "file_id": "ab7271d4-a8fd-4b41-b2c9-4d1a0c1c4d0c",
  "target_column": "species",
  "is_valid": true,
  "errors": [],
  "warnings": ["1 duplicate rows found."],
  "numerical_features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "categorical_features": [],
  "dropped_features": [],
  "drop_reasons": {},
  "info": {
    "total_rows": 150,
    "total_columns": 5,
    "feature_count": 4,
    "target_classes": 3,
    "class_distribution": {
      "setosa": 0.333,
      "versicolor": 0.333,
      "virginica": 0.333
    }
  }
}
```

---

## Key Concepts Learned

- **UUID for file isolation**: Each upload gets its own folder, preventing overwrites
- **Errors vs Warnings**: Errors block training, warnings inform but allow training to proceed
- **Column profiling**: Auto-detecting types is what makes the pipeline work on ANY dataset
- **Uniqueness ratio**: Columns where 95%+ of values are unique are almost certainly IDs
- **Data validation before training**: Catching bad data early saves the user from waiting
  2 minutes for training to crash with a cryptic error

---

## File Storage Structure
```
data/
└── uploads/
    └── {file_id}/
        └── data.csv      <- raw uploaded file
```
Future days will add more files to this folder:
- `preprocessor.pkl` - fitted sklearn pipeline
- `model.pkl` - trained best model
- `training_report.json` - metrics from all 4 models
