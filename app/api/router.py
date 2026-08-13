"""
Central API router.

Aggregates all versioned route modules and mounts them
under the /api/v1 prefix defined in settings.
"""

from fastapi import APIRouter

from app.api.routes import health, documents

api_router = APIRouter()

# --- v1 routes ---
api_router.include_router(health.router)
api_router.include_router(documents.router)

# Future task routes will be registered here:
# api_router.include_router(rag.router)
