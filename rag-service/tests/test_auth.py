"""
Auth and security isolation tests for Task 13.
Runs fully offline using mongomock for user database persistence.
"""

from datetime import datetime, timedelta
import jwt
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.auth_service import AuthService


@pytest.fixture
def auth_client():
    # Setup clean mongomock AuthService for test run
    import mongomock
    mongo_client = mongomock.MongoClient()
    auth_svc = AuthService(client=mongo_client)
    auth_svc.connect()
    
    with TestClient(app) as c:
        original_auth = getattr(app.state, "auth_service", None)
        app.state.auth_service = auth_svc
        try:
            yield c, auth_svc
        finally:
            app.state.auth_service = original_auth
            auth_svc.close()


# ===========================================================================
# 1. Registration, Login, and Password Hashing Tests
# ===========================================================================

class TestAuthRegistrationAndLogin:
    def test_successful_registration(self, auth_client):
        client, auth_svc = auth_client
        payload = {
            "email": "alice@gmail.com",
            "password": "securepassword123",
            "name": "Alice Cooper"
        }
        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 201
        data = response.json()["data"]
        assert data["email"] == "alice@gmail.com"
        assert data["name"] == "Alice Cooper"
        assert "_id" in data
        assert "password" not in data
        assert "password_hash" not in data

    def test_duplicate_email_registration_fails(self, auth_client):
        client, auth_svc = auth_client
        payload = {
            "email": "alice@gmail.com",
            "password": "securepassword123",
            "name": "Alice"
        }
        # First register
        client.post("/api/v1/auth/register", json=payload)
        
        # Second register with same email
        response2 = client.post("/api/v1/auth/register", json=payload)
        assert response2.status_code == 400
        assert "exists" in response2.json()["detail"].lower()

    def test_password_hashing(self, auth_client):
        client, auth_svc = auth_client
        password = "alicepassword123"
        hashed = auth_svc.hash_password(password)
        assert hashed != password
        assert auth_svc.verify_password(password, hashed)
        assert not auth_svc.verify_password("wrongpassword", hashed)

    def test_successful_login(self, auth_client):
        client, auth_svc = auth_client
        # Register first
        reg_payload = {"email": "bob@gmail.com", "password": "bobsecretpassword", "name": "Bob"}
        client.post("/api/v1/auth/register", json=reg_payload)

        # LoginBob
        login_payload = {"email": "bob@gmail.com", "password": "bobsecretpassword"}
        response = client.post("/api/v1/auth/login", json=login_payload)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == "bob@gmail.com"

    def test_incorrect_password_fails(self, auth_client):
        client, auth_svc = auth_client
        reg_payload = {"email": "bob@gmail.com", "password": "bobsecretpassword", "name": "Bob"}
        client.post("/api/v1/auth/register", json=reg_payload)

        login_payload = {"email": "bob@gmail.com", "password": "incorrectpassword"}
        response = client.post("/api/v1/auth/login", json=login_payload)
        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()

    def test_nonexistent_user_login_fails(self, auth_client):
        client, auth_svc = auth_client
        login_payload = {"email": "nonexistent@gmail.com", "password": "password123"}
        response = client.post("/api/v1/auth/login", json=login_payload)
        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()


# ===========================================================================
# 2. JWT Verification and Expiry Tests
# ===========================================================================

class TestJWTVerification:
    def test_valid_jwt_token(self, auth_client):
        client, auth_svc = auth_client
        # Register & Login
        reg_payload = {"email": "jack@gmail.com", "password": "jackpassword", "name": "Jack"}
        client.post("/api/v1/auth/register", json=reg_payload)
        login_res = client.post("/api/v1/auth/login", json={"email": "jack@gmail.com", "password": "jackpassword"}).json()
        token = login_res["access_token"]

        # Call health endpoint preflight (or any endpoint, but health is open).
        # Let's call /api/v1/tutor/context which is protected under Task 13
        headers = {"Authorization": f"Bearer {token}"}
        req_body = {"query": "Operating Systems"}
        # Mock tutor service on app state to prevent 503
        from unittest.mock import MagicMock
        original_tutor = getattr(app.state, "tutor_service", None)
        mock_tutor = MagicMock()
        mock_tutor.get_study_context.return_value = {"query": "Operating Systems", "chunks": [], "citations_count": 0}
        app.state.tutor_service = mock_tutor
        
        try:
            response = client.post("/api/v1/tutor/context", json=req_body, headers=headers)
            assert response.status_code == 200
        finally:
            app.state.tutor_service = original_tutor

    def test_expired_jwt_fails(self, auth_client):
        client, auth_svc = auth_client
        # Create an expired token manually
        expire_time = datetime.utcnow() - timedelta(minutes=5)
        payload = {"sub": "usr_expired_id", "exp": expire_time, "iat": datetime.utcnow() - timedelta(minutes=10)}
        token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

        headers = {"Authorization": f"Bearer {token}"}
        response = client.post("/api/v1/tutor/context", json={"query": "test"}, headers=headers)
        assert response.status_code == 401
        assert "expired" in response.json()["detail"].lower()

    def test_malformed_jwt_fails(self, auth_client):
        client, auth_svc = auth_client
        headers = {"Authorization": "Bearer malformed_token_string_here"}
        response = client.post("/api/v1/tutor/context", json={"query": "test"}, headers=headers)
        assert response.status_code == 401
        assert "invalid" in response.json()["detail"].lower()

    def test_missing_and_malformed_authorization_headers(self, auth_client):
        client, auth_svc = auth_client
        # No header
        response1 = client.post("/api/v1/tutor/context", json={"query": "test"})
        assert response1.status_code == 401
        assert "credentials" in response1.json()["detail"].lower()

        # Malformed header (no Bearer)
        headers2 = {"Authorization": "invalidprefix token123"}
        response2 = client.post("/api/v1/tutor/context", json={"query": "test"}, headers=headers2)
        assert response2.status_code == 401


# ===========================================================================
# 3. Multi-Tenant Resource Isolation Tests
# ===========================================================================

class TestMultiTenantIsolation:
    def test_cross_user_isolation_attack(self, auth_client):
        client, auth_svc = auth_client
        
        # Register User A and User B
        user_a = auth_svc.register_user("usera@gmail.com", "password123", "User A")
        user_b = auth_svc.register_user("userb@gmail.com", "password123", "User B")
        
        token_a, _ = auth_svc.create_access_token(user_a["user_id"])
        token_b, _ = auth_svc.create_access_token(user_b["user_id"])

        # Setup conversation service mock or mongomock instance
        import mongomock
        from app.services.conversation_service import ConversationService
        conv_svc = ConversationService(client=mongomock.MongoClient())
        conv_svc.connect()
        
        # Attach conv_svc to app
        original_conv = getattr(app.state, "conversation_service", None)
        app.state.conversation_service = conv_svc

        try:
            # User B creates conversation B
            conv_b = conv_svc.create_conversation(user_id=user_b["user_id"], title="User B Chat")
            
            # User A attempts to access conversation B
            headers_a = {"Authorization": f"Bearer {token_a}"}
            response = client.get(
                f"/api/v1/conversations/{conv_b.conversation_id}?user_id={user_a['user_id']}",
                headers=headers_a
            )
            # Should return 404 to avoid leaking existence of conversation B to User A
            assert response.status_code == 404
            
            # User A attempts to delete conversation B
            response_delete = client.delete(
                f"/api/v1/conversations/{conv_b.conversation_id}?user_id={user_a['user_id']}",
                headers=headers_a
            )
            assert response_delete.status_code == 404

        finally:
            app.state.conversation_service = original_conv
            conv_svc.close()

    def test_client_userid_override_attack(self, auth_client):
        client, auth_svc = auth_client
        
        # Register User A and User B
        user_a = auth_svc.register_user("usera@gmail.com", "password123", "User A")
        user_b = auth_svc.register_user("userb@gmail.com", "password123", "User B")
        
        token_a, _ = auth_svc.create_access_token(user_a["user_id"])

        # Mock conversation service
        import mongomock
        from app.services.conversation_service import ConversationService
        conv_svc = ConversationService(client=mongomock.MongoClient())
        conv_svc.connect()
        original_conv = getattr(app.state, "conversation_service", None)
        app.state.conversation_service = conv_svc

        try:
            # User A calls create conversation but passes user_id = User B in body
            headers_a = {"Authorization": f"Bearer {token_a}"}
            payload = {
                "user_id": user_b["user_id"],
                "title": "Malicious Chat"
            }
            response = client.post("/api/v1/conversations", json=payload, headers=headers_a)
            # Must reject with 403 Forbidden since the client user_id doesn't match the JWT sub identity
            assert response.status_code == 403
            assert "match" in response.json()["detail"].lower()
            
        finally:
            app.state.conversation_service = original_conv
            conv_svc.close()
