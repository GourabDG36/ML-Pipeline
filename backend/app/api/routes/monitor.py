"""
Drift monitoring endpoint.

WHY a separate router: Monitoring is a distinct concern from
prediction. Keeping it separate makes it easier to scale
independently and add more monitoring types later.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from loguru import logger

from app.monitoring.drift import compute_drift_report

router = APIRouter()


class DriftRequest(BaseModel):
    file_id: str
    current_data: list[dict]

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "your-file-id-here",
                "current_data": [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2
                    },
                    {
                        "sepal_length": 7.0,
                        "sepal_width": 3.2,
                        "petal_length": 4.7,
                        "petal_width": 1.4
                    }
                ]
            }
        }


class FeatureDriftDetail(BaseModel):
    drift_detected: bool
    drift_score: float
    stat_test: str
    threshold: float


class DriftResponse(BaseModel):
    status: str
    message: str
    dataset_drift_detected: bool
    drift_score: float
    n_features_drifted: int
    feature_drift: dict[str, FeatureDriftDetail]
    reference_rows: int
    current_rows: int


@router.post("/drift-report", response_model=DriftResponse)
async def drift_report(request: DriftRequest) -> DriftResponse:
    """
    Compare incoming data against training data to detect distribution shift.
    Send at least 10 rows of recent prediction inputs to get meaningful results.
    """
    try:
        result = compute_drift_report(
            file_id=request.file_id,
            current_data=request.current_data,
        )
        return DriftResponse(**result)

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No training session found for file_id '{request.file_id}'."
        )
    except Exception as e:
        logger.error("Drift report failed: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift computation failed: {str(e)}"
        )