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

    # Future: pre-load embedding model, verify Astra DB connection, etc.

    yield  # App is running

    logger.info("Shutting down %s …", settings.APP_NAME)
    # Future: release model resources, close DB connections, etc.


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
