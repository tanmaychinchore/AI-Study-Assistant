"""
Uvicorn entry point for the RAG service.

Usage:
    python run.py

This reads host/port from the settings (which loads .env),
so configuration changes only require editing the .env file.
"""

import uvicorn

from app.core.config import settings


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.RAG_SERVICE_HOST,
        port=settings.RAG_SERVICE_PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
    )
