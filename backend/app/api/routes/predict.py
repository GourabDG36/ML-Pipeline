"""
Prediction endpoint.

Accepts a file_id (links to the trained session) and
a dict of feature values. Returns prediction + SHAP explanation.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from loguru import logger

from app.ml.predictor import predict_single

router = APIRouter()


class PredictRequest(BaseModel):
    file_id: str
    features: dict

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "your-file-id-here",
                "features": {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }
            }
        }


class PredictResponse(BaseModel):
    prediction: str
    confidence: float | None
    probabilities: dict
    shap_values: dict
    shap_error: str | None
    features_used: list[str]
    model_name: str


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Run prediction on a single row of features.
    Returns the predicted class, confidence, probabilities,
    and SHAP values explaining the prediction.
    """
    try:
        result = predict_single(
            file_id=request.file_id,
            input_data=request.features,
        )
        return PredictResponse(**result)

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No training session found for file_id '{request.file_id}'. "
                   f"Please upload and train first."
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Prediction failed: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )