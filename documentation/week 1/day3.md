# Day 3 - Preprocessing Pipeline + ML Training + MLflow Tracking

## What We Built
- Scikit-learn ColumnTransformer preprocessing pipeline
- Full training engine for 4 models (LR, RF, XGBoost, LightGBM)
- Optuna hyperparameter tuning on best model (20 trials)
- MLflow experiment tracking for all runs
- MLflow Model Registry auto-registration of best model
- POST /train endpoint that orchestrates the entire pipeline

---

## Files Created

### `backend/app/ml/preprocessing/pipeline.py`
- `build_preprocessor()` - builds ColumnTransformer for numerical + categorical columns
- `prepare_data()` - full data prep: split, encode target, fit preprocessor, transform
- `save_preprocessor()` - saves preprocessor.pkl and label_encoder.pkl to disk
- `load_preprocessor()` - loads them back for prediction time

### `backend/app/ml/models/classifiers.py`
- `get_baseline_models()` - returns all 4 models with sensible defaults
- Handles binary vs multiclass automatically for XGBoost and LightGBM

### `backend/app/ml/evaluation/metrics.py`
- `evaluate_model()` - computes accuracy, F1, precision, recall, ROC-AUC
- Uses macro averaging for multiclass fairness
- Handles ROC-AUC for both binary and multiclass

### `backend/app/ml/trainer.py`
- `train_all_models()` - orchestrates full training + MLflow tracking
- `tune_best_model()` - runs Optuna study on winning baseline model
- `_build_model_from_params()` - rebuilds model from Optuna best params

### `backend/app/api/routes/upload.py` (updated)
- Added POST /train endpoint
- Added TrainRequest and TrainResponse Pydantic models

---

## Training Flow

```
POST /train (file_id + target_column)
        |
        v
Load CSV from disk
        |
        v
Validate dataset (reuse Day 2 validator)
        |
        v
Detect column types (reuse Day 2 detector)
        |
        v
prepare_data()
  - Select useful columns
  - LabelEncode target
  - Train/test split (stratified)
  - Fit ColumnTransformer on TRAIN only
  - Transform both splits
        |
        v
Save preprocessor.pkl + label_encoder.pkl + profile.json
        |
        v
train_all_models()
  For each of 4 models:
    - Start MLflow run
    - Train model
    - Evaluate metrics
    - Log metrics + params + artifact to MLflow
    - Track best F1
        |
        v
tune_best_model() with Optuna
  - 20 trials
  - Maximize macro F1
  - Rebuild model with best params
  - Evaluate tuned model
        |
        v
Register tuned model in MLflow Model Registry
as "best-classifier" version 1
        |
        v
Return all results to user
```

---

## Preprocessing Pipeline Details

### Numerical columns
```
SimpleImputer(strategy="median")
        |
        v
StandardScaler()
```
WHY median imputation: Robust to outliers unlike mean
WHY StandardScaler: Helps Logistic Regression converge faster

### Categorical columns
```
SimpleImputer(strategy="most_frequent")
        |
        v
OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```
WHY handle_unknown="ignore": New categories at prediction time
won't crash the model, they just get all-zero encoding

### ColumnTransformer
- remainder="drop" automatically excludes ID columns and datetime columns
- Single object that handles everything in one .transform() call

---

## MLflow Tracking Details

### What gets logged per run
- **Metrics**: accuracy, f1, precision, recall, roc_auc, training_time_seconds
- **Params**: all model hyperparameters
- **Tags**: file_id, model_name, tuned (true/false)
- **Artifacts**: serialized model as MLflow sklearn flavor

### Model Registry
- Best tuned model registered as `best-classifier`
- Version 1 created automatically
- Visible in MLflow UI under Model training tab

---

## Optuna Tuning Search Spaces

### Logistic Regression
- C: log-uniform [0.01, 10.0]
- solver: categorical [lbfgs, saga]

### Random Forest
- n_estimators: int [50, 300]
- max_depth: int [3, 20]
- min_samples_split: int [2, 10]

### XGBoost
- n_estimators: int [50, 300]
- max_depth: int [3, 10]
- learning_rate: log-uniform [0.01, 0.3]
- subsample: float [0.6, 1.0]

### LightGBM
- n_estimators: int [50, 300]
- max_depth: int [3, 10]
- learning_rate: log-uniform [0.01, 0.3]
- num_leaves: int [20, 100]

---

## Files Saved Per Upload Session

```
data/uploads/{file_id}/
    data.csv              <- raw uploaded file (Day 2)
    preprocessor.pkl      <- fitted ColumnTransformer (Day 3)
    label_encoder.pkl     <- fitted LabelEncoder for target (Day 3)
    profile.json          <- column names + class names (Day 3)
```

---

## Test Results with Iris Dataset

### Baseline models
| Model | F1 | Accuracy | ROC-AUC | Train Time |
|-------|----|----------|---------|------------|
| Logistic Regression | 0.933 | 0.933 | 0.997 | 0.035s |
| XGBoost | 0.933 | 0.933 | 0.965 | 1.429s |
| Random Forest | 0.899 | 0.900 | 0.993 | 0.215s |
| LightGBM | 0.899 | 0.900 | 0.963 | 0.051s |

### After Optuna tuning (Logistic Regression)
| F1 | Accuracy | ROC-AUC |
|----|----------|---------|
| 1.0 | 1.0 | 1.0 |

Perfect score expected on iris - well-separated dataset.

---

## MLflow UI Verification
- URL: http://localhost:5000
- Command: python -m mlflow ui --backend-store-uri file:./mlruns --port 5000
- 5 runs visible under Training runs tab
- best-classifier v1 registered in Model Registry
- All runs show green checkmarks (completed)

---

## Key Concepts Learned

- **ColumnTransformer**: Apply different sklearn pipelines to different column subsets
- **Data leakage**: Fitting preprocessor on full dataset before split leaks test info into training
- **Stratified split**: Preserves class distribution in train and test sets
- **LabelEncoder for target**: XGBoost/LightGBM need integer targets, not strings
- **Optuna direction="maximize"**: We maximize F1, not minimize loss
- **MLflow Model Registry**: Separate from experiment runs, stores production-ready models
- **handle_unknown="ignore"**: Critical for OneHotEncoder at prediction time
