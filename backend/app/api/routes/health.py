from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime, timezone
import mlflow

from app.core.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    mlflow_uri: str
    mlflow_reachable: bool


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    mlflow_ok = False
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.search_experiments()
        mlflow_ok = True
    except Exception as e:
        mlflow_ok = False

    return HealthResponse(
        status="healthy" if mlflow_ok else "degraded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=settings.APP_VERSION,
        mlflow_uri=settings.MLFLOW_TRACKING_URI,
        mlflow_reachable=mlflow_ok,
    )