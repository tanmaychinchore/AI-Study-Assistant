"""
Central API router.

Aggregates all versioned route modules and mounts them
under the /api/v1 prefix defined in settings.
"""

from fastapi import APIRouter

from app.api.routes import health, documents, embeddings, vector_db, retrieval, llm, rag

api_router = APIRouter()

# --- v1 routes ---
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(embeddings.router)
api_router.include_router(vector_db.router)
api_router.include_router(retrieval.router)
api_router.include_router(llm.router)
api_router.include_router(rag.router)

