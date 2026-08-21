"""
Smart Tutor Service.
Responsible for validating tutor requests, checking document scope/ownership,
and invoking user-isolated semantic retrieval.
"""

from typing import Optional

from app.core.logging import get_logger
from app.schemas.tutor import TutorRequest, TutorContextData
from app.schemas.retrieval import RetrievedChunk
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


class TutorService:
    """Orchestrates multi-tenant learning context retrieval and validation for Smart Tutor features."""

    def __init__(self, retrieval_service: RetrievalService) -> None:
        self.retrieval_service = retrieval_service

    def get_study_context(self, request: TutorRequest, user_id: str) -> TutorContextData:
        """
        Verify user ownership, retrieve relevant study chunks scoped strictly to user_id,
        and format the learning context.
        """
        if not user_id or not user_id.strip():
            raise ValueError("Authenticated user_id is required.")

        if not request.query or not request.query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info(
            "Tutor retrieving study context: query='%s' user='%s' doc='%s' subject='%s' topic='%s'",
            request.query[:50] + ("..." if len(request.query) > 50 else ""),
            user_id,
            request.document_id,
            request.subject,
            request.topic,
        )

        # Retrieve relevant chunks scoped strictly to user_id
        # RetrievalService already wraps AstraDB query which enforces user_id matching
        from app.schemas.retrieval import RetrievalRequest
        retrieval_req = RetrievalRequest(
            query=request.query,
            user_id=user_id,
            document_id=request.document_id,
            subject=request.subject,
            topic=request.topic,
        )
        retrieval_res = self.retrieval_service.retrieve(retrieval_req)
        chunks = retrieval_res.results

        # Map directly to RetrievedChunk schema
        chunk_infos = chunks

        # Calculate count of unique document citations
        unique_docs = {c.document_id for c in chunk_infos if c.document_id}
        citations_count = len(unique_docs)

        logger.info(
            "Tutor context retrieved: chunks=%d citations=%d",
            len(chunk_infos),
            citations_count,
        )

        return TutorContextData(
            query=request.query,
            chunks=chunk_infos,
            citations_count=citations_count,
        )
