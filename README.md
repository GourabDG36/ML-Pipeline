# ML Pipeline — Automated Classification with Full MLOps Stack

An end-to-end automated machine learning pipeline for classification tasks. Upload any CSV dataset, select a target column, and the system automatically trains 4 models, tracks experiments with MLflow, tunes hyperparameters with Optuna, serves predictions with SHAP explanations, and monitors data drift with Evidently AI.

Everything runs with a single `docker-compose up` command.

---

## Demo

### Upload & Validate
- Upload any classification CSV
- Auto-detects column types (numerical vs categorical)
- Validates data quality (missing values, class imbalance, duplicates)

### Train
- Trains 4 models: Logistic Regression, Random Forest, XGBoost, LightGBM
- Tracks all experiments in MLflow
- Optuna hyperparameter tuning on best model
- Auto-registers best model in MLflow Model Registry

### Predict
- REST API endpoint for predictions
- Every prediction includes SHAP feature importance explanation
- Returns confidence scores and class probabilities

### Drift Monitor
- Detects data distribution shift between training and production data
- Per-feature K-S test drift scores
- Alerts when model retraining may be needed

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| ML | Scikit-learn, XGBoost, LightGBM |
| Tracking | MLflow 2.13.0 |
| Tuning | Optuna |
| Explainability | SHAP |
| Drift Monitoring | Evidently AI |
| Frontend | Next.js 14 + Tailwind CSS + Recharts |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |

---

## Project Structure

```
ml-pipeline/
├── .github/
│   └── workflows/
│       ├── ci.yml              # runs tests on every push
│       └── retrain.yml         # retrains when new data pushed
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   │   ├── health.py       # GET /health
│   │   │   ├── upload.py       # POST /upload, /validate, /train
│   │   │   ├── predict.py      # POST /predict
│   │   │   └── monitor.py      # POST /drift-report
│   │   ├── core/
│   │   │   ├── config.py       # centralized settings
│   │   │   └── logging.py      # loguru setup
│   │   ├── ml/
│   │   │   ├── preprocessing/  # validator, detector, pipeline
│   │   │   ├── models/         # 4 classifier definitions
│   │   │   ├── evaluation/     # metrics computation
│   │   │   ├── trainer.py      # training + MLflow + Optuna
│   │   │   └── predictor.py    # inference + SHAP
│   │   ├── monitoring/
│   │   │   └── drift.py        # Evidently drift detection
│   │   └── tests/
│   │       └── test_data_layer.py  # 18 unit tests
│   ├── scripts/
│   │   └── retrain.py          # standalone CI retraining script
│   ├── Dockerfile
│   ├── requirements.txt        # local dev (Python 3.14)
│   └── requirements.docker.txt # container (Python 3.11)
├── frontend/
│   ├── app/
│   │   ├── page.tsx            # Upload page
│   │   ├── train/page.tsx      # Train page
│   │   ├── predict/page.tsx    # Predict page
│   │   └── drift/page.tsx      # Drift Monitor page
│   └── lib/
│       └── api.ts              # Axios API client + TypeScript types
├── data/
│   └── sample/
│       └── iris.csv            # sample dataset for testing
└── docker-compose.yml          # runs everything
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check + MLflow connectivity |
| POST | `/api/v1/upload` | Upload CSV, returns file_id |
| POST | `/api/v1/validate` | Validate dataset + detect column types |
| POST | `/api/v1/train` | Train 4 models + Optuna + MLflow tracking |
| POST | `/api/v1/predict` | Predict + SHAP explanation |
| POST | `/api/v1/drift-report` | Data drift monitoring report |

---

## Quick Start

### Prerequisites
- Docker Desktop
- Node.js 18+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ml-pipeline.git
cd ml-pipeline
```

### 2. Start the backend
```bash
docker-compose up --build
```

Wait for:
```
fastapi-backend | INFO: Application startup complete.
```

### 3. Start the frontend
```bash
cd frontend
npm install
npm run dev
```

### 4. Open in browser
- Frontend: http://localhost:3000
- Backend API docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5000

---

## Usage

### Using the Frontend
1. Go to `http://localhost:3000`
2. Upload any classification CSV file
3. Select the target column and click Validate
4. Go to Train page and click Start Training (1-2 minutes)
5. Go to Predict page, enter feature values, click Predict
6. Go to Drift Monitor, paste recent data as JSON array, run report

### Using the API directly

**Upload a dataset:**
```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@data/sample/iris.csv"
```

**Train models:**
```bash
curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{"file_id": "your-file-id", "target_column": "species"}'
```

**Get a prediction:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "your-file-id",
    "features": {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }
  }'
```

**Sample prediction response:**
```json
{
  "prediction": "setosa",
  "confidence": 0.9909,
  "probabilities": {
    "setosa": 0.9909,
    "versicolor": 0.0091,
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

---

## ML Pipeline Architecture

```
CSV Upload
    |
    v
Data Validation (10 checks)
    |
    v
Column Type Detection
(numerical / categorical / ID / datetime)
    |
    v
Scikit-learn Preprocessing Pipeline
  - Numerical: median imputation + StandardScaler
  - Categorical: mode imputation + OneHotEncoder
    |
    v
Train 4 Models in parallel
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
    |
    v
MLflow tracks metrics, params, artifacts
    |
    v
Optuna tunes best model (20 trials)
    |
    v
Best model registered in MLflow Model Registry
    |
    v
FastAPI serves predictions
    |
    v
SHAP explains every prediction
    |
    v
Evidently monitors incoming data for drift
```

---

## Data Validation Checks

The pipeline automatically validates any uploaded dataset:

| Check | Type | Description |
|-------|------|-------------|
| Target column exists | Error | Blocks if target not found |
| Minimum 50 rows | Error | Blocks tiny datasets |
| At least 2 classes | Error | Blocks single-class targets |
| All-null columns | Error | Blocks empty columns |
| No feature columns | Error | Blocks target-only datasets |
| Class imbalance < 5% | Warning | Warns about severe imbalance |
| Missing values | Warning | Reports null counts per column |
| Duplicate rows | Warning | Reports duplicate count |
| High cardinality target | Warning | Warns if > 20 classes |
| Dataset too large | Warning | Warns if > 100k rows |

---

## CI/CD

### On every push to main:
- Installs dependencies on Ubuntu
- Runs 18 unit tests covering validation, detection, preprocessing
- Fails fast if any test breaks

### On new data pushed to data/sample/:
- Automatically retrains all 4 models
- Registers new best model in MLflow
- Uploads training report as GitHub artifact

---

## Running Tests Locally

```bash
cd backend
pip install -r requirements.txt
python -m pytest app/tests/ -v
```

Expected output:
```
18 passed, 3 warnings in 25.97s
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MLFLOW_TRACKING_URI | file:./mlruns | MLflow server URI |
| MLFLOW_EXPERIMENT_NAME | ml-pipeline | Experiment name |
| REGISTERED_MODEL_NAME | best-classifier | Model registry name |
| OPTUNA_TRIALS | 20 | Hyperparameter tuning trials |
| RANDOM_SEED | 42 | Reproducibility seed |
| MAX_FILE_SIZE_MB | 50 | Upload size limit |
| DEBUG | false | Enable debug logging |

---

## Key Design Decisions

**WHY Scikit-learn Pipeline object:**
Guarantees identical transformations at training and prediction time.
No data leakage. Single object to serialize and load.

**WHY MLflow Model Registry:**
Single source of truth for production model. Retraining automatically
updates predictions without code changes.

**WHY SHAP on every prediction:**
Black-box models are unacceptable in production. Every prediction
must be explainable. SHAP provides mathematically rigorous feature attribution.

**WHY Evidently for drift:**
Models degrade silently when data distribution shifts. Drift monitoring
catches this before it becomes a business problem.

**WHY Docker Compose:**
Reproducible environment. One command runs the entire stack.
No "works on my machine" problems.

---

## Built With

- [FastAPI](https://fastapi.tiangolo.com/)
- [MLflow](https://mlflow.org/)
- [Optuna](https://optuna.org/)
- [SHAP](https://shap.readthedocs.io/)
- [Evidently AI](https://www.evidentlyai.com/)
- [Next.js](https://nextjs.org/)
- [Recharts](https://recharts.org/)
