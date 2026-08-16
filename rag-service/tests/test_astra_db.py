"""
Comprehensive tests for the Astra DB Vector Storage Service (Task 5).

Covers:
  - Configuration verification (dimension=1024, metric=cosine)
  - Document conversion and metadata mapping (_id, $vector, page, slide, etc.)
  - Vector dimension validation (strictly 1024)
  - Error handling (empty lists, invalid credentials, uninitialized collection)
  - API endpoints (health, test-insert, test-document retrieval, deletion)
  - Live integration test (real BGE-M3 embedding -> Astra DB -> retrieve -> delete)
"""

import os
from unittest.mock import MagicMock, patch
import uuid

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.chunk import DocumentChunk
from app.schemas.document import FileType
from app.schemas.embedding import EmbeddedDocumentChunk
from app.services.astra_db_service import AstraDBService
from app.services.embedding_service import EmbeddingService


def _make_embedded_chunk(
    text: str = "Deadlock avoidance with Banker's Algorithm.",
    chunk_id: str = "test_doc_chunk_001",
    document_id: str = "doc_001",
    dimension: int = 1024,
    page_number: int = 4,
    slide_number: int = None,
    slide_title: str = None,
) -> EmbeddedDocumentChunk:
    """Helper to create a test EmbeddedDocumentChunk with dummy vector."""
    doc_chunk = DocumentChunk(
        chunk_id=chunk_id,
        chunk_index=0,
        text=text,
        char_count=len(text),
        document_id=document_id,
        document_name="Operating_Systems.pdf",
        file_type=FileType.PDF,
        user_id="user_test_123",
        subject="Computer Science",
        topic="Deadlocks",
        page_number=page_number,
        slide_number=slide_number,
        slide_title=slide_title,
        heading="Deadlock Avoidance",
        source_type="document",
    )
    vector = [0.01 * (i % 10) for i in range(dimension)]
    return EmbeddedDocumentChunk.from_chunk_and_vector(doc_chunk, vector)


# ===========================================================================
# 1. CONFIGURATION TESTS
# ===========================================================================

class TestAstraDBConfiguration:
    """Tests for Astra DB settings and defaults."""

    def test_default_collection_name(self):
        """Default collection name should be 'study_chunks'."""
        assert settings.ASTRA_DB_COLLECTION_NAME == "study_chunks"

    def test_expected_vector_dimension(self):
        """Astra DB expected vector dimension must match BGE-M3 (1024)."""
        assert settings.EMBEDDING_DIMENSION == 1024

    def test_service_initialization_defaults(self):
        """AstraDBService should initialize with configured defaults."""
        service = AstraDBService()
        assert service.collection_name == settings.ASTRA_DB_COLLECTION_NAME
        assert service.expected_dimension == 1024
        assert service.metric == "cosine"
        assert service.keyspace == settings.ASTRA_DB_KEYSPACE


# ===========================================================================
# 2. DOCUMENT MAPPING & VALIDATION TESTS
# ===========================================================================

class TestAstraDBDocumentMapping:
    """Tests for chunk-to-AstraDB document conversion."""

    def test_chunk_to_document_fields(self):
        """All chunk fields and metadata must map correctly to Astra document format."""
        service = AstraDBService()
        chunk = _make_embedded_chunk(
            text="Process synchronization primitives.",
            chunk_id="sync_chunk_001",
            document_id="doc_sync_100",
            slide_number=12,
            slide_title="Semaphores and Mutexes",
        )

        doc = service._chunk_to_document(chunk)

        assert doc["_id"] == "sync_chunk_001"
        assert doc["chunk_id"] == "sync_chunk_001"
        assert doc["document_id"] == "doc_sync_100"
        assert doc["document_name"] == "Operating_Systems.pdf"
        assert doc["user_id"] == "user_test_123"
        assert doc["text"] == "Process synchronization primitives."
        assert doc["char_count"] == len("Process synchronization primitives.")
        assert doc["file_type"] == "pdf"
        assert doc["slide_number"] == 12
        assert doc["slide_title"] == "Semaphores and Mutexes"
        assert doc["subject"] == "Computer Science"
        assert doc["topic"] == "Deadlocks"
        assert doc["source_type"] == "document"
        assert "$vector" in doc
        assert len(doc["$vector"]) == 1024
        assert "created_at" in doc

    def test_vector_dimension_mismatch_raises(self):
        """Document mapping must reject chunks with invalid vector dimensions."""
        service = AstraDBService()
        invalid_chunk = _make_embedded_chunk(dimension=512)  # Wrong dimension

        with pytest.raises(ValueError, match="vector dimension 512, expected 1024"):
            service._chunk_to_document(invalid_chunk)

    def test_mask_endpoint(self):
        """Endpoint string should be properly masked for logging."""
        masked = AstraDBService._mask_endpoint("https://abc12345-6789-0123-4567-89abcdef0123-us-east-2.apps.astra.datastax.com")
        assert "..." in masked
        assert masked.startswith("https://abc1234")


# ===========================================================================
# 3. SERVICE ERROR HANDLING & EDGE CASES
# ===========================================================================

class TestAstraDBServiceErrorHandling:
    """Tests for edge cases and error states."""

    def test_unconfigured_service(self):
        """Service without token/endpoint should report unconfigured."""
        service = AstraDBService(api_endpoint="", token="")
        assert service.is_configured is False
        assert service.is_connected is False
        assert service.is_ready is False

        health = service.get_health()
        assert health["status"] == "not_configured"
        assert health["is_connected"] is False

    def test_insert_empty_chunks_raises(self):
        """Inserting empty list should raise ValueError."""
        service = AstraDBService()
        with pytest.raises(ValueError, match="Cannot insert empty list"):
            service.insert_embedded_chunks([])

    def test_insert_when_not_ready_raises(self):
        """Inserting when collection is not initialized should raise RuntimeError."""
        service = AstraDBService(api_endpoint="", token="")
        chunk = _make_embedded_chunk()
        with pytest.raises(RuntimeError, match="not initialized"):
            service.insert_embedded_chunks([chunk])

    def test_get_chunk_when_not_ready_raises(self):
        """get_chunk when collection not ready should raise RuntimeError."""
        service = AstraDBService(api_endpoint="", token="")
        with pytest.raises(RuntimeError, match="not initialized"):
            service.get_chunk("some_id")

    def test_delete_chunk_when_not_ready_raises(self):
        """delete_chunk when collection not ready should raise RuntimeError."""
        service = AstraDBService(api_endpoint="", token="")
        with pytest.raises(RuntimeError, match="not initialized"):
            service.delete_chunk("some_id")

    def test_connection_error_on_invalid_endpoint(self):
        """Invalid endpoint or token should raise ConnectionError cleanly."""
        service = AstraDBService(
            api_endpoint="https://invalid-endpoint.apps.astra.datastax.com",
            token="AstraCS:invalid_token_123",
        )
        with pytest.raises(ConnectionError, match="Failed to connect to Astra DB"):
            service.connect()


# ===========================================================================
# 4. API ENDPOINT TESTS
# ===========================================================================

class TestAstraDBAPIEndpoints:
    """Tests for /api/v1/vector-db and /api/v1/health routes."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_check_includes_components(self, client):
        """GET /api/v1/health should include astra_db and embedding_model components."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "components" in data["data"]
        assert "astra_db" in data["data"]["components"]
        assert "embedding_model" in data["data"]["components"]

    def test_vector_db_health_endpoint(self, client):
        """GET /api/v1/vector-db/health should return collection health structure."""
        response = client.get("/api/v1/vector-db/health")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "collection" in data["data"]
        assert "vector_dimension" in data["data"]
        assert data["data"]["vector_dimension"] == 1024
        assert data["data"]["metric"] == "cosine"

    def test_test_insert_empty_text_returns_400(self, client):
        """POST /api/v1/vector-db/test-insert with empty text should return 400 Bad Request."""
        # Mock ready service if not configured
        with patch("app.api.routes.vector_db._get_astra_db_service") as mock_db:
            mock_db.return_value = MagicMock(is_ready=True)
            response = client.post(
                "/api/v1/vector-db/test-insert",
                json={"text": "   ", "document_name": "test.txt"},
            )
            assert response.status_code == 400


# ===========================================================================
# 5. LIVE INTEGRATION TEST (Runs when live Astra credentials are in .env)
# ===========================================================================

class TestAstraDBLiveIntegration:
    """
    Live end-to-end integration test with Astra DB Serverless.
    Tests real BGE-M3 embedding -> Astra DB insertion -> Retrieval by ID -> Deletion cleanup.
    """

    @pytest.fixture(scope="class")
    def live_service(self):
        """Create and connect live AstraDBService instance if configured."""
        service = AstraDBService()
        if not service.is_configured:
            pytest.skip("Astra DB credentials not configured in .env. Skipping live integration test.")
        try:
            service.connect()
            service.initialize_collection()
        except Exception as exc:
            pytest.skip(f"Could not connect to live Astra DB: {exc}. Skipping integration test.")
        return service

    @pytest.fixture(scope="class")
    def embedding_service(self):
        """Load BGE-M3 embedding service."""
        emb = EmbeddingService()
        emb.load_model()
        return emb

    def test_live_insert_retrieve_delete_flow(self, live_service, embedding_service):
        """
        Complete lifecycle test:
          1. Embed real educational text via BGE-M3
          2. Batch insert into Astra DB
          3. Retrieve by chunk_id and verify 1024-dim vector + metadata
          4. Delete test chunk and verify cleanup
        """
        unique_test_id = f"test_integration_{uuid.uuid4().hex[:8]}"
        test_text = (
            "In operating systems, a semaphore is a variable or abstract data type used to control "
            "access to a common resource by multiple processes in a concurrent system."
        )

        # 1. Generate real 1024-dim embedding
        vector = embedding_service.embed_query(test_text)
        assert len(vector) == 1024

        chunk = DocumentChunk(
            chunk_id=unique_test_id,
            chunk_index=0,
            text=test_text,
            char_count=len(test_text),
            document_id="integration_test_doc",
            document_name="OS_Synchronization.pdf",
            file_type=FileType.PDF,
            user_id="integration_test_user",
            subject="Operating Systems",
            topic="Semaphores",
            page_number=15,
            source_type="document",
        )
        embedded_chunk = EmbeddedDocumentChunk.from_chunk_and_vector(chunk, vector)

        # 2. Insert into Astra DB
        inserted_count, inserted_ids, duration_ms = live_service.insert_embedded_chunks([embedded_chunk])
        assert inserted_count == 1
        assert unique_test_id in inserted_ids
        assert duration_ms > 0

        # 3. Retrieve and verify
        retrieved = live_service.get_chunk(unique_test_id)
        assert retrieved is not None
        assert retrieved["chunk_id"] == unique_test_id
        assert retrieved["text"] == test_text
        assert retrieved["page_number"] == 15
        assert retrieved["subject"] == "Operating Systems"
        assert retrieved["user_id"] == "integration_test_user"
        assert retrieved["has_vector"] is True
        assert retrieved["vector_dimension"] == 1024

        # 4. Clean up
        deleted = live_service.delete_chunk(unique_test_id)
        assert deleted is True

        # Verify it no longer exists
        assert live_service.get_chunk(unique_test_id) is None
