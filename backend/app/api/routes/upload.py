import uuid
import shutil
import pandas as pd

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel

from app.core.config import settings
from app.ml.preprocessing.validator import validate_dataset
from app.ml.preprocessing.detector import detect_column_types
from loguru import logger

router = APIRouter()


# ── Response Models ────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    rows: int
    columns: int
    column_names: list[str]
    message: str


class ValidationResponse(BaseModel):
    file_id: str
    target_column: str
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    numerical_features: list[str]
    categorical_features: list[str]
    dropped_features: list[str]
    drop_reasons: dict[str, str]
    info: dict


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_csv(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a CSV file for training.
    Returns a file_id to use in subsequent /validate and /train requests.
    """

    # 1. Validate file extension
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are allowed."
        )

    # 2. Read file bytes into memory
    content = await file.read()

    # 3. Validate file size
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB."
        )

    # 4. Generate unique ID for this upload session
    file_id = str(uuid.uuid4())

    # 5. Create a dedicated folder for this upload
    upload_folder = settings.UPLOAD_DIR / file_id
    upload_folder.mkdir(parents=True, exist_ok=True)

    # 6. Save the file to disk
    file_path = upload_folder / "data.csv"
    with open(file_path, "wb") as f:
        f.write(content)

    logger.info("File uploaded: {} -> {}", file.filename, file_path)

    # 7. Quick peek to return basic info to the user
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        shutil.rmtree(upload_folder)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse CSV file: {str(e)}"
        )

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        rows=len(df),
        columns=len(df.columns),
        column_names=df.columns.tolist(),
        message="File uploaded successfully. Use file_id for training."
    )


@router.post("/validate", response_model=ValidationResponse)
async def validate_csv(file_id: str, target_column: str) -> ValidationResponse:
    """
    Validate an uploaded CSV and detect column types.
    Call this after /upload with the returned file_id.
    """

    # 1. Find the uploaded file
    file_path = settings.UPLOAD_DIR / file_id / "data.csv"
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No file found for file_id '{file_id}'. Please upload first."
        )

    # 2. Load into dataframe
    df = pd.read_csv(file_path)

    # 3. Run validation checks
    validation = validate_dataset(df, target_column)

    # 4. Run column detection only if validation passed
    profile = detect_column_types(df, target_column) if validation.is_valid else None

    return ValidationResponse(
        file_id=file_id,
        target_column=target_column,
        is_valid=validation.is_valid,
        errors=validation.errors,
        warnings=validation.warnings,
        numerical_features=profile.numerical_features if profile else [],
        categorical_features=profile.categorical_features if profile else [],
        dropped_features=profile.dropped_features if profile else [],
        drop_reasons=profile.drop_reasons if profile else {},
        info=validation.info
    )

from app.ml.preprocessing.pipeline import prepare_data, save_preprocessor
from app.ml.trainer import train_all_models
import json


class TrainRequest(BaseModel):
    file_id: str
    target_column: str


class ModelResult(BaseModel):
    metrics: dict
    run_id: str


class TrainResponse(BaseModel):
    file_id: str
    best_model_name: str
    best_metrics: dict
    all_results: dict
    message: str


@router.post("/train", response_model=TrainResponse)
async def train_models(request: TrainRequest) -> TrainResponse:
    """
    Trigger full training pipeline on an uploaded and validated dataset.
    Trains 4 models, tunes the best with Optuna, registers in MLflow.
    """

    # 1. Find uploaded file
    file_path = settings.UPLOAD_DIR / request.file_id / "data.csv"
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No file found for file_id '{request.file_id}'. Please upload first."
        )

    df = pd.read_csv(file_path)

    # 2. Validate
    validation = validate_dataset(df, request.target_column)
    if not validation.is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"errors": validation.errors}
        )

    # 3. Detect columns
    profile = detect_column_types(df, request.target_column)

    # 4. Preprocess
    data = prepare_data(
        df=df,
        target_column=request.target_column,
        profile=profile,
        test_size=settings.TEST_SIZE,
        random_seed=settings.RANDOM_SEED,
    )

    # 5. Save preprocessor for prediction time
    upload_folder = settings.UPLOAD_DIR / request.file_id
    from app.ml.preprocessing.pipeline import save_preprocessor, save_training_data
    save_preprocessor(data["preprocessor"], data["label_encoder"], upload_folder)
    save_training_data(data["X_train"], upload_folder)

    # Save profile for prediction time
    profile_data = {
        "numerical_features": profile.numerical_features,
        "categorical_features": profile.categorical_features,
        "target_column": request.target_column,
        "class_names": data["class_names"],
    }
    with open(upload_folder / "profile.json", "w") as f:
        json.dump(profile_data, f)

    # 6. Train all models
    results = train_all_models(data, request.file_id)

    return TrainResponse(
        file_id=request.file_id,
        best_model_name=results["best_model_name"],
        best_metrics=results["best_metrics"],
        all_results={
            k: v for k, v in results["all_results"].items()
            if k != "tuned_best"
        },
        message=f"Training complete. Best model: {results['best_model_name']}"
    )