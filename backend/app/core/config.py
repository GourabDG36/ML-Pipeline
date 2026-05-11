"""
Central configuration for the ML Pipeline application.

WHY Pydantic Settings: It validates config values at startup, gives you
type safety, and makes it trivial to swap values between dev/prod
by just changing environment variables — no code changes needed.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

# Resolve the backend/ directory regardless of where the script is run from
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "ML Pipeline API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False)
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    
    # File upload limits
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list[str] = [".csv"]
    
    # ML training
    MAX_TRAINING_ROWS: int = 100_000   # cap dataset size for free-tier deployment
    OPTUNA_TRIALS: int = 20            # keep tuning fast but meaningful
    RANDOM_SEED: int = 42              # reproducibility
    TEST_SIZE: float = 0.2
    
    # MLflow
    MLFLOW_TRACKING_URI: str = Field(
    default="file:./mlruns",
    description="Local path or remote MLflow server URI"
    )
    MLFLOW_EXPERIMENT_NAME: str = "ml-pipeline"
    REGISTERED_MODEL_NAME: str = "best-classifier"
    
    # Paths
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    SAMPLE_DATA_DIR: Path = BASE_DIR / "data" / "sample"
    
    class Config:
        env_file = ".env"          # read from .env file if it exists
        env_file_encoding = "utf-8"
        case_sensitive = True


# Singleton — import this object everywhere instead of instantiating Settings()
settings = Settings()

# Ensure required directories exist on startup
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)