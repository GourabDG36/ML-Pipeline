# Day 4 - Prediction Endpoint + SHAP Explanations

## What We Built
- POST /predict endpoint that loads best model from MLflow registry
- Full prediction pipeline: input validation, preprocessing, inference
- SHAP explanations attached to every prediction response
- Training data saved as SHAP background for accurate explanations

---

## Files Created

### `backend/app/ml/predictor.py`
- `load_latest_model()` - loads best-classifier from MLflow Model Registry
- `load_session_artifacts()` - loads preprocessor, label_encoder, profile
- `compute_shap_values()` - computes SHAP using correct explainer per model type
- `predict_single()` - full prediction pipeline for one input row

### `backend/app/api/routes/predict.py`
- POST /predict endpoint
- PredictRequest and PredictResponse Pydantic models
- Proper error handling for missing session, missing features, model not found

---

## Endpoints Added

| Method | URL | Description |
|--------|-----|-------------|
| POST | /api/v1/predict | Predict class + SHAP explanation for one row |

---

## Prediction Flow

```
POST /predict (file_id + features dict)
        |
        v
Load session artifacts from disk
  - preprocessor.pkl
  - label_encoder.pkl
  - profile.json
  - X_train.pkl (for SHAP background)
        |
        v
Load best-classifier from MLflow Model Registry
        |
        v
Validate input has all required features
        |
        v
Convert features dict to DataFrame
        |
        v
Transform with fitted preprocessor
(same transformation as training time)
        |
        v
model.predict() -> encoded integer
label_encoder.inverse_transform() -> class string
        |
        v
model.predict_proba() -> probability per class
        |
        v
compute_shap_values() -> feature importance dict
        |
        v
Return prediction + confidence + probabilities + SHAP
```

---

## SHAP Explainer Selection

Different model types need different SHAP explainers:

| Model | Explainer | Why |
|-------|-----------|-----|
| Random Forest | TreeExplainer | Fast, exact for tree models |
| XGBoost | TreeExplainer | Native support |
| LightGBM | TreeExplainer | Native support |
| Logistic Regression | LinearExplainer | Exact for linear models |
| Any other | KernelExplainer | Universal fallback, slow |

## SHAP Background Data
- WHY background data matters: SHAP computes feature importance by comparing
  the prediction to a baseline (background). Without real background data,
  SHAP compares to itself and returns near-zero values.
- Solution: Save X_train.pkl during training, load it at prediction time
- Use max 100 random rows from training data as background (performance)

## SHAP Array Shape Handling
LinearExplainer for multiclass returns shape (samples, features, classes).
We handle all cases:
- (samples, features) - binary: take abs of row 0
- (samples, features, classes) - multiclass 3D: mean abs across classes
- list of (samples, features) - multiclass list: stack and mean abs

---

## Files Saved Per Upload Session (Complete)

```
data/uploads/{file_id}/
    data.csv              <- raw uploaded file (Day 2)
    preprocessor.pkl      <- fitted ColumnTransformer (Day 3)
    label_encoder.pkl     <- fitted LabelEncoder for target (Day 3)
    profile.json          <- column names + class names (Day 3)
    X_train.pkl           <- processed training data for SHAP (Day 4)
```

---

## Key Problems Solved

### Problem 1: SHAP values all zero
- Cause: LinearExplainer was using single input row as background
- Fix: Load X_train.pkl as background, use Independent masker

### Problem 2: SHAP 3D array for multiclass
- Cause: LogisticRegression with 3 classes returns (1, 4, 3) shaped array
- Fix: Mean absolute SHAP values across class dimension

### Problem 3: sample_shap not defined
- Cause: logger line was added before the assignment line during editing
- Fix: Always define sample_shap = shap_array[0] before using it

---

## Test Results with Iris Dataset

Request:
```json
{
  "file_id": "your-file-id",
  "features": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
}
```

Response:
```json
{
  "prediction": "setosa",
  "confidence": 0.9881,
  "probabilities": {
    "setosa": 0.9881,
    "versicolor": 0.0119,
    "virginica": 0.0
  },
  "shap_values": {
    "numerical__petal_width": 2.785718,
    "numerical__petal_length": 2.404791,
    "numerical__sepal_width": 0.842761,
    "numerical__sepal_length": 0.799809
  },
  "shap_error": null,
  "features_used": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "model_name": "LogisticRegression"
}
```

SHAP interpretation: petal_width and petal_length are the strongest
predictors for setosa classification. This matches botanical reality -
setosa has distinctly smaller petals than other species.

---

## Key Concepts Learned

- **MLflow Model Registry URI**: `models:/best-classifier/latest` always
  points to latest version, no code changes needed after retraining
- **SHAP background data**: Must represent the training distribution,
  not just the single input being explained
- **LinearExplainer masker**: shap.maskers.Independent wraps background
  data correctly for linear models
- **Multiclass SHAP**: Average absolute values across classes to get
  a single per-feature importance score
- **predict_proba vs predict**: Always get probabilities for confidence
  scores and SHAP, use predict only for final class label
