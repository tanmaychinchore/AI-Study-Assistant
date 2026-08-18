"""
Comprehensive test suite for Task 7: Retrieval Engine.

Covers:
  - Query embedding generation & validation
  - Vector similarity search & Top-K ranking
  - Score ranges & descending ordering
  - Strict user isolation (cross-user privacy)
  - Metadata filters (document_id, subject, topic)
  - Similarity threshold cutoff filtering
  - Empty / unmatched search results
  - Input validation (empty query, user_id, invalid top_k, threshold)
  - Service failure / unready handling
  - Semantic discrimination (OS vs. DBMS topic differentiation)
  - HTTP API endpoint POST /api/v1/retrieval/search
"""

import uuid
from typing import Generator
import pytest
from starlette.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.chunk import DocumentChunk
from app.schemas.document import FileType
from app.schemas.embedding import EmbeddedDocumentChunk
from app.schemas.retrieval import RetrievalRequest, RetrievalResult, RetrievedChunk
from app.services.astra_db_service import AstraDBService
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService, retrieve_chunks


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture(scope="module")
def embedding_service() -> Generator[EmbeddingService, None, None]:
    """Module-scoped embedding service with preloaded BGE-M3 model."""
    svc = EmbeddingService(device="cpu")
    svc.load_model()
    yield svc


@pytest.fixture(scope="module")
def live_astra_service() -> Generator[AstraDBService, None, None]:
    """Module-scoped connected AstraDBService."""
    svc = AstraDBService()
    if svc.is_configured:
        svc.connect()
        svc.initialize_collection()
    yield svc


@pytest.fixture
def retrieval_service(
    embedding_service: EmbeddingService,
    live_astra_service: AstraDBService,
) -> RetrievalService:
    """RetrievalService instance wired to live embedding & Astra DB services."""
    return RetrievalService(
        embedding_service=embedding_service,
        astra_service=live_astra_service,
    )


# Helper function to seed test chunks into Astra DB
def _seed_test_chunk(
    astra_service: AstraDBService,
    embedding_service: EmbeddingService,
    user_id: str,
    doc_id: str,
    doc_name: str,
    text: str,
    subject: str = "Computer Science",
    topic: str = "General",
    chunk_idx: int = 0,
) -> EmbeddedDocumentChunk:
    chunk = DocumentChunk(
        chunk_id=f"{doc_id}_chunk_{chunk_idx:04d}",
        chunk_index=chunk_idx,
        text=text,
        char_count=len(text),
        document_id=doc_id,
        document_name=doc_name,
        file_type=FileType.TXT,
        user_id=user_id,
        subject=subject,
        topic=topic,
    )
    vec = embedding_service.embed_texts([text])[0]
    embedded_chunk = EmbeddedDocumentChunk.from_chunk_and_vector(chunk, vec)
    astra_service.insert_embedded_chunks([embedded_chunk])
    return embedded_chunk


# ===========================================================================
# 1. QUERY EMBEDDING TESTS
# ===========================================================================

class TestQueryEmbedding:
    """Test embedding generation for search queries."""

    def test_embed_query_produces_1024_dim_vector(self, embedding_service):
        vector = embedding_service.embed_query("What is virtual memory paging?")
        assert isinstance(vector, list)
        assert len(vector) == 1024
        assert all(isinstance(x, float) for x in vector)

    def test_embed_query_is_normalized(self, embedding_service):
        vector = embedding_service.embed_query("Explain deadlock avoidance algorithms.")
        norm = sum(x * x for x in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-4

    def test_empty_query_raises_value_error(self, embedding_service):
        with pytest.raises(ValueError, match="cannot be empty"):
            embedding_service.embed_query("   ")


# ===========================================================================
# 2. RETRIEVAL SERVICE & VECTOR SEARCH TESTS
# ===========================================================================

class TestRetrievalService:
    """Core retrieval functionality tests."""

    def test_basic_retrieval_success(
        self,
        retrieval_service: RetrievalService,
        live_astra_service: AstraDBService,
        embedding_service: EmbeddingService,
    ):
        """Seed a chunk and retrieve it via semantic query."""
        user_id = f"test_user_{uuid.uuid4().hex[:6]}"
        doc_id = f"doc_basic_{uuid.uuid4().hex[:6]}"
        seed_text = "Semaphores and mutex locks provide synchronization primitives for concurrent threads."

        _seed_test_chunk(
            astra_service=live_astra_service,
            embedding_service=embedding_service,
            user_id=user_id,
            doc_id=doc_id,
            doc_name="concurrency.txt",
            text=seed_text,
            subject="Operating Systems",
            topic="Synchronization",
        )

        try:
            request = RetrievalRequest(
                query="How do synchronization primitives like mutex locks work?",
                user_id=user_id,
                top_k=3,
            )
            result = retrieval_service.retrieve(request)

            assert isinstance(result, RetrievalResult)
            assert len(result.results) >= 1
            top_hit = result.results[0]
            assert top_hit.document_id == doc_id
            assert top_hit.user_id == user_id
            assert "Semaphores" in top_hit.text
            assert 0.0 <= top_hit.similarity_score <= 1.0
            assert result.statistics.embedding_time_ms > 0
            assert result.statistics.search_time_ms > 0
            assert result.statistics.total_time_ms > 0
        finally:
            live_astra_service.delete_document_chunks(doc_id)

    def test_top_k_ordering_descending(
        self,
        retrieval_service: RetrievalService,
        live_astra_service: AstraDBService,
        embedding_service: EmbeddingService,
    ):
        """Returned results must be sorted strictly from highest similarity to lowest."""
        user_id = f"test_user_{uuid.uuid4().hex[:6]}"
        doc_id = f"doc_order_{uuid.uuid4().hex[:6]}"

        # Seed 3 chunks with varying relevance
        texts = [
            "Cache memory uses LRU replacement policy to manage spatial and temporal locality.",
            "Photosynthesis in green plants converts solar energy into chemical energy.",
            "Relational database management systems use B-trees for fast index searching.",
        ]
        for i, text in enumerate(texts):
            _seed_test_chunk(
                astra_service=live_astra_service,
                embedding_service=embedding_service,
                user_id=user_id,
                doc_id=doc_id,
                doc_name="multi_topic.txt",
                text=text,
                chunk_idx=i,
            )

        try:
            request = RetrievalRequest(
                query="How does cache replacement policy handle locality in computer architecture?",
                user_id=user_id,
                top_k=3,
            )
            result = retrieval_service.retrieve(request)

            assert len(result.results) == 3
            scores = [r.similarity_score for r in result.results]
            assert scores == sorted(scores, reverse=True), "Scores must be in strictly descending order"
            assert "Cache memory" in result.results[0].text
        finally:
            live_astra_service.delete_document_chunks(doc_id)

    def test_similarity_threshold_filtering(
        self,
        retrieval_service: RetrievalService,
        live_astra_service: AstraDBService,
        embedding_service: EmbeddingService,
    ):
        """Chunks below the similarity threshold must be pruned from results."""
        user_id = f"test_user_{uuid.uuid4().hex[:6]}"
        doc_id = f"doc_thresh_{uuid.uuid4().hex[:6]}"

        _seed_test_chunk(
            astra_service=live_astra_service,
            embedding_service=embedding_service,
            user_id=user_id,
            doc_id=doc_id,
            doc_name="test.txt",
            text="The French Revolution began in 1789 with the storming of the Bastille.",
        )

        try:
            # Query on an unrelated topic with a high threshold
            request = RetrievalRequest(
                query="What is quantum entanglement in theoretical physics?",
                user_id=user_id,
                top_k=5,
                similarity_threshold=0.85,
            )
            result = retrieval_service.retrieve(request)

            assert len(result.results) == 0
            assert result.statistics.chunks_returned == 0
        finally:
            live_astra_service.delete_document_chunks(doc_id)


# ===========================================================================
# 3. USER ISOLATION & FILTERING TESTS
# ===========================================================================

class TestUserIsolationAndFiltering:
    """Verify strict tenant isolation and metadata filter enforcement."""

    def test_user_isolation_prevent_cross_tenant_leakage(
        self,
        retrieval_service: RetrievalService,
        live_astra_service: AstraDBService,
        embedding_service: EmbeddingService,
    ):
        """User A must never retrieve User B's documents, even for identical queries."""
        user_a = f"user_alice_{uuid.uuid4().hex[:6]}"
        user_b = f"user_bob_{uuid.uuid4().hex[:6]}"
        doc_a = f"doc_alice_{uuid.uuid4().hex[:6]}"
        doc_b = f"doc_bob_{uuid.uuid4().hex[:6]}"

        _seed_test_chunk(
            astra_service=live_astra_service,
            embedding_service=embedding_service,
            user_id=user_a,
            doc_id=doc_a,
            doc_name="alice_notes.txt",
            text="Alice's confidential research on quantum computing cryptography algorithms.",
        )
        _seed_test_chunk(
            astra_service=live_astra_service,
            embedding_service=embedding_service,
            user_id=user_b,
            doc_id=doc_b,
            doc_name="bob_notes.txt",
            text="Bob's public guide to organic gardening and composting techniques.",
        )

        try:
            # Bob searches for quantum cryptography
            bob_request = RetrievalRequest(
                query="quantum computing cryptography algorithms",
                user_id=user_b,
                top_k=5,
            )
            bob_result = retrieval_service.retrieve(bob_request)

            # Bob must NOT get Alice's document
            for hit in bob_result.results:
                assert hit.user_id != user_a
                assert hit.document_id != doc_a

            # Alice searches for quantum cryptography and gets her doc
            alice_request = RetrievalRequest(
                query="quantum computing cryptography algorithms",
                user_id=user_a,
                top_k=5,
            )
            alice_result = retrieval_service.retrieve(alice_request)
            assert len(alice_result.results) >= 1
            assert alice_result.results[0].document_id == doc_a
            assert alice_result.results[0].user_id == user_a
        finally:
            live_astra_service.delete_document_chunks(doc_a)
            live_astra_service.delete_document_chunks(doc_b)

    def test_document_id_filter(
        self,
        retrieval_service: RetrievalService,
        live_astra_service: AstraDBService,
        embedding_service: EmbeddingService,
    ):
        """Filter by document_id should only return chunks from that document."""
        user_id = f"test_user_{uuid.uuid4().hex[:6]}"
        doc_1 = f"doc_os_1_{uuid.uuid4().hex[:6]}"
        doc_2 = f"doc_os_2_{uuid.uuid4().hex[:6]}"

        _seed_test_chunk(
            astra_service=live_astra_service,
            embedding_service=embedding_service,
            user_id=user_id,
            doc_id=doc_1,
            doc_name="os_intro.txt",
            text="Operating systems manage hardware resources and provide abstraction layers.",
        )
        _seed_test_chunk(
            astra_service=live_astra_service,
            embedding_service=embedding_service,
            user_id=user_id,
            doc_id=doc_2,
            doc_name="os_memory.txt",
            text="Virtual memory management uses page tables and TLB for address translation.",
        )

        try:
            request = RetrievalRequest(
                query="How does memory management and address translation work?",
                user_id=user_id,
                document_id=doc_2,
                top_k=5,
            )
            result = retrieval_service.retrieve(request)

            assert len(result.results) >= 1
            for hit in result.results:
                assert hit.document_id == doc_2
        finally:
            live_astra_service.delete_document_chunks(doc_1)
            live_astra_service.delete_document_chunks(doc_2)

    def test_subject_and_topic_filter(
        self,
        retrieval_service: RetrievalService,
        live_astra_service: AstraDBService,
        embedding_service: EmbeddingService,
    ):
        """Filter by subject and topic tags."""
        user_id = f"test_user_{uuid.uuid4().hex[:6]}"
        doc_id = f"doc_filtered_{uuid.uuid4().hex[:6]}"

        _seed_test_chunk(
            astra_service=live_astra_service,
            embedding_service=embedding_service,
            user_id=user_id,
            doc_id=doc_id,
            doc_name="topics.txt",
            text="TCP three-way handshake establishes a reliable stream connection.",
            subject="Networking",
            topic="Transport Layer",
            chunk_idx=0,
        )
        _seed_test_chunk(
            astra_service=live_astra_service,
            embedding_service=embedding_service,
            user_id=user_id,
            doc_id=doc_id,
            doc_name="topics.txt",
            text="Binary search trees maintain sorted keys for logarithmic search.",
            subject="Algorithms",
            topic="Trees",
            chunk_idx=1,
        )

        try:
            request = RetrievalRequest(
                query="How does a handshake work?",
                user_id=user_id,
                subject="Networking",
                topic="Transport Layer",
                top_k=5,
            )
            result = retrieval_service.retrieve(request)

            assert len(result.results) == 1
            assert result.results[0].subject == "Networking"
            assert result.results[0].topic == "Transport Layer"
            assert "handshake" in result.results[0].text
        finally:
            live_astra_service.delete_document_chunks(doc_id)

    def test_no_matching_filter_returns_empty_results(
        self,
        retrieval_service: RetrievalService,
    ):
        """Non-existent user or filter returns empty list with 200 OK structure."""
        request = RetrievalRequest(
            query="What is an operating system kernel?",
            user_id=f"ghost_user_{uuid.uuid4().hex[:8]}",
            top_k=5,
        )
        result = retrieval_service.retrieve(request)

        assert isinstance(result, RetrievalResult)
        assert len(result.results) == 0
        assert result.statistics.chunks_returned == 0


# ===========================================================================
# 4. SEMANTIC DISCRIMINATION (REAL BGE-M3 INTEGRATION)
# ===========================================================================

class TestSemanticDiscriminationIntegration:
    """
    Test real semantic topic discrimination between Operating Systems and DBMS.
    Verify that an OS question ranks OS chunks higher than DBMS chunks.
    """

    def test_semantic_discrimination_os_vs_dbms(
        self,
        retrieval_service: RetrievalService,
        live_astra_service: AstraDBService,
        embedding_service: EmbeddingService,
    ):
        user_id = f"student_cs_{uuid.uuid4().hex[:6]}"
        os_doc_id = f"os_doc_{uuid.uuid4().hex[:6]}"
        db_doc_id = f"db_doc_{uuid.uuid4().hex[:6]}"

        os_text = (
            "CPU Scheduling and Process Control Block: The OS kernel uses a round-robin "
            "scheduler with priority queues to allocate CPU time slices to active processes. "
            "Context switches save CPU registers and program counters into the PCB."
        )
        db_text = (
            "Relational DBMS ACID Properties: Database transactions guarantee Atomicity, "
            "Consistency, Isolation, and Durability. Two-Phase Locking prevents concurrency "
            "anomalies in SQL databases."
        )

        _seed_test_chunk(
            astra_service=live_astra_service,
            embedding_service=embedding_service,
            user_id=user_id,
            doc_id=os_doc_id,
            doc_name="os_lecture.txt",
            text=os_text,
            subject="Operating Systems",
            topic="CPU Scheduling",
        )
        _seed_test_chunk(
            astra_service=live_astra_service,
            embedding_service=embedding_service,
            user_id=user_id,
            doc_id=db_doc_id,
            doc_name="dbms_lecture.txt",
            text=db_text,
            subject="DBMS",
            topic="Transactions",
        )

        try:
            # Query 1: Operating Systems question
            os_query = RetrievalRequest(
                query="How does the CPU scheduler allocate time slices using PCB and context switching?",
                user_id=user_id,
                top_k=2,
            )
            os_res = retrieval_service.retrieve(os_query)
            assert len(os_res.results) == 2
            assert os_res.results[0].document_id == os_doc_id
            assert os_res.results[0].subject == "Operating Systems"
            assert os_res.results[0].similarity_score > os_res.results[1].similarity_score

            # Query 2: DBMS question
            db_query = RetrievalRequest(
                query="What are ACID transaction properties and two-phase locking in databases?",
                user_id=user_id,
                top_k=2,
            )
            db_res = retrieval_service.retrieve(db_query)
            assert len(db_res.results) == 2
            assert db_res.results[0].document_id == db_doc_id
            assert db_res.results[0].subject == "DBMS"
            assert db_res.results[0].similarity_score > db_res.results[1].similarity_score

        finally:
            live_astra_service.delete_document_chunks(os_doc_id)
            live_astra_service.delete_document_chunks(db_doc_id)

    def test_cross_user_semantic_match_security(
        self,
        retrieval_service: RetrievalService,
        live_astra_service: AstraDBService,
        embedding_service: EmbeddingService,
    ):
        """
        Security Test: User A has high-match confidential data.
        User B submits an EXACT semantic query matching User A's document.
        Astra DB must return 0 results from User A for User B.
        """
        user_alice = f"alice_sec_{uuid.uuid4().hex[:6]}"
        user_attacker = f"attacker_{uuid.uuid4().hex[:6]}"
        doc_alice = f"doc_alice_sec_{uuid.uuid4().hex[:6]}"

        secret_text = "Master secret key token and confidential encryption passwords for production server."
        _seed_test_chunk(
            astra_service=live_astra_service,
            embedding_service=embedding_service,
            user_id=user_alice,
            doc_id=doc_alice,
            doc_name="secrets.txt",
            text=secret_text,
        )

        try:
            # Attacker searches for the exact phrase
            attacker_req = RetrievalRequest(
                query="Master secret key token and confidential encryption passwords",
                user_id=user_attacker,
                top_k=5,
            )
            result = retrieval_service.retrieve(attacker_req)

            # Strict isolation check: 0 results returned
            assert len(result.results) == 0
            for r in result.results:
                assert r.user_id != user_alice
                assert r.document_id != doc_alice
        finally:
            live_astra_service.delete_document_chunks(doc_alice)

    def test_unrelated_query_evaluation_and_observation(
        self,
        retrieval_service: RetrievalService,
        live_astra_service: AstraDBService,
        embedding_service: EmbeddingService,
    ):
        """
        Evaluation & Observation: Test how BGE-M3 + Astra DB score an unrelated query.
        An unrelated query (e.g. Italian pasta baking) against CS notes should produce
        a low similarity score and can be filtered by threshold.
        """
        user_id = f"eval_user_{uuid.uuid4().hex[:6]}"
        doc_id = f"eval_doc_{uuid.uuid4().hex[:6]}"

        cs_text = "Binary search algorithms require a sorted array and operate in logarithmic O(log n) time complexity."
        _seed_test_chunk(
            astra_service=live_astra_service,
            embedding_service=embedding_service,
            user_id=user_id,
            doc_id=doc_id,
            doc_name="algorithms.txt",
            text=cs_text,
            subject="Computer Science",
            topic="Algorithms",
        )

        try:
            # Query 1: Relevant query
            rel_req = RetrievalRequest(
                query="What is the time complexity of binary search on sorted arrays?",
                user_id=user_id,
                top_k=1,
            )
            rel_res = retrieval_service.retrieve(rel_req)
            assert len(rel_res.results) == 1
            rel_score = rel_res.results[0].similarity_score

            # Query 2: Completely unrelated query
            unrel_req = RetrievalRequest(
                query="How do I bake traditional Italian sourdough bread with rosemary and olive oil?",
                user_id=user_id,
                top_k=1,
            )
            unrel_res = retrieval_service.retrieve(unrel_req)
            assert len(unrel_res.results) == 1
            unrel_score = unrel_res.results[0].similarity_score

            # Relevant query must score significantly higher than unrelated query
            assert rel_score > unrel_score
            assert rel_score >= 0.80
            assert rel_score - unrel_score >= 0.15
            assert unrel_score < 0.72

            # If threshold is set to 0.75, unrelated query returns 0 chunks
            thresh_req = RetrievalRequest(
                query="How do I bake traditional Italian sourdough bread with rosemary and olive oil?",
                user_id=user_id,
                top_k=1,
                similarity_threshold=0.75,
            )
            thresh_res = retrieval_service.retrieve(thresh_req)
            assert len(thresh_res.results) == 0

        finally:
            live_astra_service.delete_document_chunks(doc_id)


# ===========================================================================
# 5. ERROR HANDLING & VALIDATION TESTS
# ===========================================================================

class TestRetrievalErrorHandling:
    """Test failure states and validation bounds."""

    def test_missing_embedding_service_raises(self, live_astra_service):
        svc = RetrievalService(embedding_service=None, astra_service=live_astra_service)
        req = RetrievalRequest(query="test query", user_id="user_1")
        with pytest.raises(RuntimeError, match="EmbeddingService is not available"):
            svc.retrieve(req)

    def test_missing_astra_service_raises(self, embedding_service):
        svc = RetrievalService(embedding_service=embedding_service, astra_service=None)
        req = RetrievalRequest(query="test query", user_id="user_1")
        with pytest.raises(RuntimeError, match="AstraDBService is not connected"):
            svc.retrieve(req)

    def test_empty_query_validation(self):
        with pytest.raises(ValueError):
            RetrievalRequest(query="   ", user_id="user_1")

    def test_empty_user_id_validation(self):
        with pytest.raises(ValueError):
            RetrievalRequest(query="Valid query", user_id="  ")

    def test_invalid_top_k_validation(self):
        with pytest.raises(ValueError):
            RetrievalRequest(query="Valid query", user_id="user_1", top_k=0)

        with pytest.raises(ValueError):
            RetrievalRequest(query="Valid query", user_id="user_1", top_k=51)

    def test_invalid_similarity_threshold_validation(self):
        with pytest.raises(ValueError):
            RetrievalRequest(query="Valid query", user_id="user_1", similarity_threshold=-0.1)

        with pytest.raises(ValueError):
            RetrievalRequest(query="Valid query", user_id="user_1", similarity_threshold=1.5)


# ===========================================================================
# 6. HTTP API ENDPOINT TESTS (POST /api/v1/retrieval/search)
# ===========================================================================

class TestRetrievalAPIEndpoint:
    """Tests for POST /api/v1/retrieval/search HTTP endpoint."""

    @pytest.fixture
    def client(self, embedding_service, live_astra_service):
        app.state.embedding_service = embedding_service
        app.state.astra_db_service = live_astra_service
        with TestClient(app) as test_client:
            yield test_client

    def test_api_retrieval_endpoint_success(self, client, live_astra_service, embedding_service):
        user_id = f"api_user_{uuid.uuid4().hex[:6]}"
        doc_id = f"api_doc_{uuid.uuid4().hex[:6]}"

        _seed_test_chunk(
            astra_service=live_astra_service,
            embedding_service=embedding_service,
            user_id=user_id,
            doc_id=doc_id,
            doc_name="api_test.txt",
            text="API testing for semantic search and document retrieval in study assistant.",
        )

        try:
            response = client.post(
                "/api/v1/retrieval/search",
                json={
                    "query": "How does semantic search work in the API?",
                    "user_id": user_id,
                    "top_k": 3,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "data" in data
            result = data["data"]
            assert result["user_id"] == user_id
            assert len(result["results"]) >= 1
            assert result["results"][0]["document_id"] == doc_id
            assert "statistics" in result
            assert result["statistics"]["embedding_time_ms"] > 0
        finally:
            live_astra_service.delete_document_chunks(doc_id)

    def test_api_retrieval_missing_query_returns_422(self, client):
        response = client.post(
            "/api/v1/retrieval/search",
            json={"user_id": "user_1"},
        )
        assert response.status_code == 422

    def test_api_retrieval_empty_query_returns_422(self, client):
        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "   ", "user_id": "user_1"},
        )
        assert response.status_code == 422
