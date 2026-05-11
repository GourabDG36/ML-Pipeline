"""
Centralized logging configuration.

WHY Loguru over stdlib logging: Loguru requires zero configuration,
gives structured context with .bind(), and formats tracebacks beautifully.
In production you'd add a JSON sink for log aggregation services.
"""

import sys
from loguru import logger


def setup_logging(debug: bool = False) -> None:
    """Configure application-wide logging."""
    
    # Remove the default handler
    logger.remove()
    
    log_level = "DEBUG" if debug else "INFO"
    
    # Console handler with color
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    
    # File handler for persistent logs (useful for debugging CI failures)
    logger.add(
        "logs/app.log",
        level="INFO",
        rotation="10 MB",    # create new file when current hits 10MB
        retention="7 days",  # delete logs older than 7 days
        compression="zip",
        format="{time} | {level} | {name}:{function}:{line} | {message}",
    )
    
    logger.info("Logging initialized at level: {}", log_level)