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

    # --- Groq LLM (Task 8) ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE: float = 0.2
    GROQ_MAX_COMPLETION_TOKENS: int = 1024
    GROQ_TIMEOUT: float = 30.0
    GROQ_MAX_RETRIES: int = 2

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

    # --- RAG Generation (Task 9) ---
    RAG_MAX_CONTEXT_CHUNKS: int = 5
    RAG_MAX_CONTEXT_CHARACTERS: int = 12000
    RAG_TEMPERATURE: float = 0.2
    RAG_MAX_COMPLETION_TOKENS: int = 1024

    # --- MongoDB Conversation Storage (Task 10) ---
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "ai_study_assistant"
    MONGODB_CONVERSATIONS_COLLECTION: str = "conversations"
    MONGODB_MESSAGES_COLLECTION: str = "messages"
    CHAT_MAX_HISTORY_MESSAGES: int = 10

    # --- RAG Evaluation (Task 11) ---
    EVAL_MIN_HIT_AT_5: float = 0.70
    EVAL_MIN_KEYWORD_COVERAGE: float = 0.70


# Singleton settings instance — import this everywhere
settings = Settings()
