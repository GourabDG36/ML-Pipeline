from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import settings
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(debug=settings.DEBUG)
    logger.info("Starting {}", settings.APP_NAME)
    
    # Wait for MLflow to be ready
    import time
    import requests
    mlflow_uri = settings.MLFLOW_TRACKING_URI
    if mlflow_uri.startswith("http"):
        logger.info("Waiting for MLflow at {}", mlflow_uri)
        for i in range(30):  # try for 5 minutes
            try:
                requests.get(f"{mlflow_uri}/health", timeout=3)
                logger.info("MLflow is ready")
                break
            except Exception:
                logger.info("MLflow not ready yet, retrying in 10s... ({}/30)", i+1)
                time.sleep(10)
    
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Automated ML Classification Pipeline with full MLOps stack",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.routes import health, upload, predict, monitor
app.include_router(health.router, prefix=settings.API_V1_PREFIX, tags=["health"])
app.include_router(upload.router, prefix=settings.API_V1_PREFIX, tags=["upload"])
app.include_router(predict.router, prefix=settings.API_V1_PREFIX, tags=["predict"])
app.include_router(monitor.router, prefix=settings.API_V1_PREFIX, tags=["monitor"])