"""
FastAPI application entry point.

Creates the FastAPI app, configures CORS middleware, registers
the API router, and sets up startup/shutdown lifecycle events.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.router import api_router
from app.core.config import settings
from app.core.logging import get_logger, setup_logging

# Initialize logging as early as possible
setup_logging()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup & shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("=" * 60)
    logger.info("  %s v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info("  Environment : %s", settings.ENVIRONMENT)
    logger.info("  Host        : %s:%s", settings.RAG_SERVICE_HOST, settings.RAG_SERVICE_PORT)
    logger.info("=" * 60)

    # --- Load embedding model ---
    from app.services.embedding_service import EmbeddingService

    try:
        embedding_service = EmbeddingService()
        embedding_service.load_model()
        app.state.embedding_service = embedding_service
        logger.info("Embedding service ready: %s", embedding_service.get_model_info())
    except Exception as exc:
        logger.warning(
            "Embedding model failed to load: %s — embedding endpoints will be unavailable",
            exc,
        )
        app.state.embedding_service = None

    # --- Initialize Astra DB ---
    from app.services.astra_db_service import AstraDBService

    try:
        astra_service = AstraDBService()
        if astra_service.is_configured:
            astra_service.connect()
            astra_service.initialize_collection()
            app.state.astra_db_service = astra_service
            logger.info("Astra DB service ready: keyspace=%s collection=%s", astra_service.keyspace, astra_service.collection_name)
        else:
            app.state.astra_db_service = astra_service
            logger.info("Astra DB service not configured (missing credentials in .env)")
    except Exception as exc:
        logger.warning(
            "Astra DB initialization failed: %s — vector storage endpoints will be unavailable",
            exc,
        )
        app.state.astra_db_service = None

    # --- Initialize Groq LLM (Task 8) ---
    from app.services.groq_service import GroqService

    try:
        groq_service = GroqService()
        app.state.groq_service = groq_service
        if groq_service.is_configured:
            logger.info("Groq LLM service ready: model=%s", groq_service.model)
        else:
            logger.info("Groq LLM service not configured (missing GROQ_API_KEY in .env)")
    except Exception as exc:
        logger.warning(
            "Groq LLM service initialization failed: %s — LLM generation endpoints will be unavailable",
            exc,
        )
        app.state.groq_service = None

    # --- Initialize Retrieval & RAG Orchestration (Tasks 7 & 9) ---
    from app.services.retrieval_service import RetrievalService
    from app.services.rag_service import RAGService

    try:
        retrieval_service = RetrievalService(
            embedding_service=app.state.embedding_service,
            astra_service=app.state.astra_db_service,
        )
        app.state.retrieval_service = retrieval_service

        rag_service = RAGService(
            retrieval_service=retrieval_service,
            groq_service=app.state.groq_service,
        )
        app.state.rag_service = rag_service
        logger.info("RAG orchestration service ready.")
    except Exception as exc:
        logger.warning("RAG service initialization failed: %s", exc)
        app.state.retrieval_service = None
        app.state.rag_service = None

    # --- Initialize MongoDB Conversation Store (Task 10) ---
    from app.services.conversation_service import ConversationService

    try:
        conversation_service = ConversationService()
        try:
            conversation_service.connect()
            logger.info("Conversation service ready: db=%s", conversation_service.database_name)
        except Exception as exc:
            logger.warning(
                "MongoDB conversation store connection deferred/failed: %s — conversation endpoints will connect on demand",
                exc,
            )
        app.state.conversation_service = conversation_service
    except Exception as exc:
        logger.warning("Conversation service initialization failed: %s", exc)
        app.state.conversation_service = None

    yield  # App is running

    logger.info("Shutting down %s …", settings.APP_NAME)
    # Release model, DB, LLM, RAG, and MongoDB resources
    if getattr(app.state, "conversation_service", None) is not None:
        try:
            app.state.conversation_service.close()
        except Exception:
            pass

    app.state.embedding_service = None
    app.state.astra_db_service = None
    app.state.groq_service = None
    app.state.retrieval_service = None
    app.state.rag_service = None
    app.state.conversation_service = None


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI Study Assistant — RAG microservice. "
        "Provides document ingestion, semantic retrieval, and "
        "grounded AI generation for the study platform."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# CORS — allow the Node.js backend (and dev tools) to reach the service.
# In production, restrict origins to the actual backend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# Mount all versioned API routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect bare root to API docs for developer convenience."""
    return RedirectResponse(url="/docs")
