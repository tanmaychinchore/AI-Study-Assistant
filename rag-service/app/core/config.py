"""
Application configuration using Pydantic Settings.

All environment variables are loaded from the .env file and validated here.
Provides a single source of truth for all configurable values across the service.
"""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    RAG service configuration.

    Values are loaded from environment variables and .env file.
    Defaults are provided for development; override in production via env vars.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    APP_NAME: str = "AI Study Assistant — RAG Service"
    APP_VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # --- Server ---
    RAG_SERVICE_HOST: str = "0.0.0.0"
    RAG_SERVICE_PORT: int = 8000

    # --- Groq LLM ---
    GROQ_API_KEY: str = ""

    # --- Astra DB ---
    ASTRA_DB_API_ENDPOINT: str = ""
    ASTRA_DB_APPLICATION_TOKEN: str = ""
    ASTRA_DB_KEYSPACE: str = "default_keyspace"
    ASTRA_DB_COLLECTION_NAME: str = "study_chunks"

    # --- Embedding ---
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "auto"  # auto | cpu | cuda

    # --- Chunking ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # --- Retrieval ---
    TOP_K: int = 5
    MIN_TOP_K: int = 1
    MAX_TOP_K: int = 50
    DEFAULT_SIMILARITY_THRESHOLD: Optional[float] = None


# Singleton settings instance — import this everywhere
settings = Settings()
