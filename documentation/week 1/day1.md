# Day 1 - Project Foundation & Repository Setup

## What We Built
- Complete project folder structure
- Python virtual environment (Python 3.14.4)
- FastAPI application with factory pattern
- Centralized configuration system
- Structured logging with Loguru
- Health check endpoint with MLflow connectivity check

---

## Folder Structure Created

```
ml-pipeline/
├── .github/
│   └── workflows/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes/
│   │   │       ├── __init__.py
│   │   │       └── health.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   └── logging.py
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── models/
│   │   │   ├── preprocessing/
│   │   │   └── evaluation/
│   │   ├── monitoring/
│   │   │   └── __init__.py
│   │   └── tests/
│   │       └── __init__.py
│   ├── mlruns/
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── app/
│       ├── components/
│       └── lib/
├── data/
│   └── sample/
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## Files Created

### `backend/requirements.txt`
All packages pinned for Python 3.14.4 compatibility:
- fastapi, uvicorn, python-multipart
- scikit-learn, xgboost, lightgbm, pandas, numpy, joblib
- mlflow, optuna, shap, evidently
- pytest, pytest-asyncio, httpx
- pydantic, pydantic-settings, python-dotenv, loguru

### `backend/app/core/config.py`
- Uses Pydantic Settings to centralize all configuration
- Reads from .env file automatically
- Key settings: MAX_FILE_SIZE_MB, OPTUNA_TRIALS, RANDOM_SEED, MLFLOW_TRACKING_URI
- Creates upload and sample data directories on startup
- WHY: Changing config for dev/prod requires only env var changes, zero code changes

### `backend/app/core/logging.py`
- Uses Loguru instead of stdlib logging
- Console handler with colors in dev
- File handler with rotation at 10MB, 7 day retention
- WHY: Structured logs are essential for debugging CI/CD failures

### `backend/app/main.py`
- Creates FastAPI app directly (not factory pattern - simplified for reliability)
- Lifespan context manager handles startup/shutdown
- CORS middleware configured for Next.js frontend
- Registers all routers

### `backend/app/api/routes/health.py`
- GET /api/v1/health
- Checks MLflow connectivity on every call
- Returns: status, timestamp, version, mlflow_uri, mlflow_reachable

---

## Key Problems Solved

### Problem 1: Python 3.14 wheel compatibility
- Most ML packages did not have pre-built Windows wheels for Python 3.14
- Solution: Install numpy==2.4.4 first (confirmed cp314 wheel), then let pip resolve others
- Packages installed without version pins where needed

### Problem 2: PowerShell execution policy
- venv activation script blocked by default policy
- Solution: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Problem 3: MLflow 3.x URI format change
- MLflow 3.x requires `file:./mlruns` prefix instead of plain path `./mlruns`
- Solution: Updated MLFLOW_TRACKING_URI default to `file:./mlruns` in config.py

### Problem 4: Em dash character in code
- Copying code from chat introduced Unicode em dash (--) causing SyntaxError
- Solution: Always paste code into VS Code and check for invalid character errors

---

## How to Run

```powershell
cd D:\my projects\mini_project\ml-pipeline\backend
venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --port 8000
```

## Endpoints
| Method | URL | Description |
|--------|-----|-------------|
| GET | /api/v1/health | Check app and MLflow status |

## Verification
- http://localhost:8000/docs - Swagger UI loads
- http://localhost:8000/api/v1/health - Returns `"status": "healthy"` and `"mlflow_reachable": true`

---

## Key Concepts Learned
- **Pydantic Settings**: Type-safe configuration that reads from environment variables
- **Lifespan context manager**: Modern FastAPI startup/shutdown (replaces deprecated @app.on_event)
- **Health endpoints**: Every production API needs one for load balancers and monitoring
- **MLflow local tracking**: Uses file-based SQLite store, requires `file:` URI prefix in v3.x
