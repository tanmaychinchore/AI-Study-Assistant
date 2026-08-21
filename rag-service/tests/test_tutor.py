"""
Smart Tutor context retrieval and multi-tenant security verification tests.
Runs offline using mocked services.
"""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.auth_service import AuthService
from app.services.tutor_service import TutorService
from app.schemas.retrieval import RetrievedChunk


@pytest.fixture
def tutor_client():
    import mongomock
    mongo_client = mongomock.MongoClient()
    auth_svc = AuthService(client=mongo_client)
    auth_svc.connect()

    # Setup mocked RetrievalService
    mock_retrieval = MagicMock()
    tutor_svc = TutorService(retrieval_service=mock_retrieval)

    with TestClient(app) as c:
        original_auth = getattr(app.state, "auth_service", None)
        original_tutor = getattr(app.state, "tutor_service", None)
        
        app.state.auth_service = auth_svc
        app.state.tutor_service = tutor_svc
        try:
            yield c, auth_svc, mock_retrieval
        finally:
            app.state.auth_service = original_auth
            app.state.tutor_service = original_tutor
            auth_svc.close()


# ===========================================================================
# 1. Route Authorization and Validation
# ===========================================================================

def test_unauthenticated_tutor_request_rejected(tutor_client):
    client, _, _ = tutor_client
    response = client.post("/api/v1/tutor/context", json={"query": "Operating Systems"})
    assert response.status_code == 401


def test_authenticated_tutor_request_succeeds(tutor_client):
    client, auth_svc, mock_retrieval = tutor_client
    # Register & login user
    user = auth_svc.register_user("tutoruser@gmail.com", "password123", "Tutor User")
    token, _ = auth_svc.create_access_token(user["user_id"])

    # Mock retrieval return
    from app.schemas.retrieval import RetrievalResult, RetrievalStatistics
    mock_stats = RetrievalStatistics(
        embedding_time_ms=10.0,
        search_time_ms=5.0,
        total_time_ms=15.0,
        chunks_retrieved=1,
        chunks_returned=1
    )
    mock_result = RetrievalResult(
        query="Operating Systems",
        user_id=user["user_id"],
        top_k=5,
        filters_applied={"user_id": user["user_id"]},
        results=[
            RetrievedChunk(
                chunk_id="chunk_1",
                document_id="doc_1",
                document_name="OS_notes.pdf",
                user_id=user["user_id"],
                text="Processes and threads in operating systems.",
                similarity_score=0.92,
                char_count=40,
                file_type="pdf",
                chunk_index=0
            )
        ],
        statistics=mock_stats
    )
    mock_retrieval.retrieve.return_value = mock_result

    # Call tutor context endpoint
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post(
        "/api/v1/tutor/context",
        json={"query": "Operating Systems"},
        headers=headers
    )
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["success"] is True
    assert res_data["data"]["citations_count"] == 1
    assert res_data["data"]["chunks"][0]["document_name"] == "OS_notes.pdf"


def test_invalid_tutor_request_rejected(tutor_client):
    client, auth_svc, _ = tutor_client
    user = auth_svc.register_user("tutoruser@gmail.com", "password123", "Tutor User")
    token, _ = auth_svc.create_access_token(user["user_id"])

    headers = {"Authorization": f"Bearer {token}"}
    # Empty query validation check
    response = client.post(
        "/api/v1/tutor/context",
        json={"query": "   "},
        headers=headers
    )
    assert response.status_code == 400


# ===========================================================================
# 2. Multi-Tenant Context Isolation
# ===========================================================================

def test_tutor_retrieves_only_correct_user_context(tutor_client):
    client, auth_svc, mock_retrieval = tutor_client
    
    # Register user_a and user_b
    user_a = auth_svc.register_user("usera@gmail.com", "password123", "User A")
    user_b = auth_svc.register_user("userb@gmail.com", "password123", "User B")
    
    token_a, _ = auth_svc.create_access_token(user_a["user_id"])

    # Mock retrieval return
    from app.schemas.retrieval import RetrievalResult, RetrievalStatistics
    mock_stats = RetrievalStatistics(
        embedding_time_ms=10.0,
        search_time_ms=5.0,
        total_time_ms=15.0,
        chunks_retrieved=1,
        chunks_returned=1
    )
    # The Mocked RetrievalService strictly returns user_a data when queried
    mock_result_a = RetrievalResult(
        query="Operating Systems",
        user_id=user_a["user_id"],
        top_k=5,
        filters_applied={"user_id": user_a["user_id"]},
        results=[
            RetrievedChunk(
                chunk_id="chunk_a",
                document_id="doc_a",
                document_name="UserA_notes.pdf",
                user_id=user_a["user_id"],
                text="This is User A's private OS document.",
                similarity_score=0.95,
                char_count=36,
                file_type="pdf",
                chunk_index=0
            )
        ],
        statistics=mock_stats
    )
    mock_retrieval.retrieve.return_value = mock_result_a

    # Request context as User A
    headers_a = {"Authorization": f"Bearer {token_a}"}
    response = client.post(
        "/api/v1/tutor/context",
        json={"query": "Operating Systems"},
        headers=headers_a
    )
    assert response.status_code == 200
    res_data = response.json()
    # Confirm retrieval call was issued with user_a's user_id context
    mock_retrieval.retrieve.assert_called_once()
    called_request = mock_retrieval.retrieve.call_args[0][0]
    assert called_request.user_id == user_a["user_id"]
    assert res_data["data"]["chunks"][0]["document_name"] == "UserA_notes.pdf"
