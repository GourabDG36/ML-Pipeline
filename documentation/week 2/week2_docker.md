# Week 2 - Docker + Docker Compose

## What We Built
- Dockerfile for FastAPI backend using Python 3.11
- Separate requirements.docker.txt for container dependencies
- docker-compose.yml running FastAPI + MLflow together
- Single docker-compose up command runs entire stack

---

## Files Created

### `backend/Dockerfile`
- Base image: python:3.11-slim
- Installs gcc/g++ for compiled packages
- Copies requirements.docker.txt and installs dependencies
- Copies application code
- Runs uvicorn on 0.0.0.0:8000

### `backend/requirements.docker.txt`
- Unpinned versions for Python 3.11 compatibility
- Separate from local requirements.txt (Python 3.14)
- Let pip resolve best compatible versions

### `backend/.dockerignore`
- Excludes venv/, __pycache__/, .env, logs/, mlruns/
- Keeps Docker build context small and fast

### `docker-compose.yml`
- Two services: mlflow and backend
- Named volumes for persistent data
- Environment variables for service configuration

---

## Docker Compose Services

### mlflow service
- Image: python:3.11-slim
- Installs mlflow==2.13.0 + gunicorn at startup
- Runs MLflow tracking server on port 5000
- Uses SQLite as backend store
- Stores artifacts in /mlflow/artifacts volume
- Data persisted in mlflow-data named volume

### backend service
- Built from ./backend/Dockerfile
- Runs FastAPI on port 8000
- Depends on mlflow service
- upload-data volume persists uploaded CSV files
- Environment variables override config.py defaults

---

## Key Docker Networking Concept

### WHY http://mlflow:5000 not http://localhost:5000

Inside Docker, each container has its own network namespace.
When FastAPI tries to connect to "localhost", it reaches itself,
not the MLflow container.

Docker Compose creates a shared network between all services.
Services can reach each other using their SERVICE NAME as hostname.

```
Your Browser
     |
     | http://localhost:8000
     v
fastapi-backend container (port 8000)
     |
     | http://mlflow:5000  <- service name as hostname
     v
mlflow-server container (port 5000)
```

This is set via environment variable in docker-compose.yml:
- MLFLOW_TRACKING_URI=http://mlflow:5000

---

## Key Problems Solved

### Problem 1: MLflow binding to 127.0.0.1 instead of 0.0.0.0
- Cause: Newer gunicorn ignores --host flag passed to mlflow server
- Fix: Set GUNICORN_CMD_ARGS=--bind=0.0.0.0:5000 environment variable
- This forces gunicorn (MLflow's WSGI server) to bind to all interfaces

### Problem 2: Docker Desktop not starting
- Cause: Hardware virtualization (VT-x/AMD-V) disabled in BIOS
- Fix: Enable virtualization in BIOS settings
- Required for WSL 2 which Docker Desktop uses on Windows

### Problem 3: version attribute obsolete warning
- Cause: version: "3.9" is deprecated in newer Docker Compose
- Fix: Remove version line from docker-compose.yml entirely

### Problem 4: curl not available for healthcheck
- Cause: python:3.11-slim doesn't include curl
- Fix: Removed healthcheck, used simple depends_on instead

---

## How to Run

### Start everything
```powershell
cd D:\my projects\mini_project\ml-pipeline
docker-compose up --build
```

### Start in background
```powershell
docker-compose up --build -d
```

### Stop everything
```powershell
docker-compose down
```

### View logs
```powershell
docker logs fastapi-backend
docker logs mlflow-server
```

### Check running containers
```powershell
docker ps
```

---

## Verification URLs
- http://localhost:8000/docs - FastAPI Swagger UI
- http://localhost:5000 - MLflow UI
- http://localhost:8000/api/v1/health - {"status":"healthy","mlflow_reachable":true}

---

## Named Volumes
- mlflow-data: stores MLflow SQLite DB and model artifacts (persists between restarts)
- upload-data: stores user uploaded CSV files (persists between restarts)

WHY named volumes: If we used bind mounts for everything, stopping
docker-compose down would lose all trained models and experiments.
Named volumes persist independently of container lifecycle.

---

## Key Concepts Learned
- **Docker networking**: Containers use service names as hostnames, not localhost
- **0.0.0.0 vs 127.0.0.1**: Must bind to 0.0.0.0 inside containers for external access
- **GUNICORN_CMD_ARGS**: Environment variable to override gunicorn binding
- **Named volumes**: Persist data between container restarts
- **depends_on**: Ensures MLflow starts before FastAPI
- **.dockerignore**: Excludes unnecessary files from build context
- **Python 3.11 in container**: Stable for ML packages regardless of local Python version
