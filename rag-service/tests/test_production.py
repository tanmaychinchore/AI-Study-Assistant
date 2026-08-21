"""
Unit and integration tests for Task 12 — Production Hardening & Security.
Runs fully offline with no external service dependencies.
"""

import os
import shutil
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.response import SuccessResponse, ErrorResponse


@pytest.fixture
def client():
    # Use TestClient to invoke endpoints
    with TestClient(app) as c:
        yield c


# ===========================================================================
# 1. Configuration Validation & Security Tests
# ===========================================================================

class TestProductionConfiguration:
    def test_settings_validation_errors(self):
        # Verify settings loads and configures defaults
        assert settings.APP_NAME is not None
        assert settings.MAX_UPLOAD_SIZE_BYTES == 50 * 1024 * 1024

    def test_cors_origins_security(self, client):
        # 1. Authorized origin preflight
        headers = {
            "Origin": "http://localhost:8000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
        response = client.options("/api/v1/health", headers=headers)
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:8000"
        assert response.headers.get("access-control-allow-credentials") == "true"

        # 2. Unauthorized origin preflight
        headers_evil = {
            "Origin": "http://evil.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
        response_evil = client.options("/api/v1/health", headers=headers_evil)
        # CORS preflight gets rejected with 400 Bad Request by CORSMiddleware
        assert response_evil.status_code in (200, 400)
        # Should NOT allow evil.com
        assert response_evil.headers.get("access-control-allow-origin") != "http://evil.com"


# ===========================================================================
# 2. File Ingestion & Magic Byte Validation
# ===========================================================================

class TestFileIngestionSecurity:
    def test_empty_file_rejected(self, client):
        # Empty text file upload
        files = {"file": ("empty.txt", b"", "text/plain")}
        data = {"user_id": "student_alice"}
        response = client.post("/api/v1/documents/process", files=files, data=data)
        assert response.status_code == 400
        assert "empty" in response.json()["message"].lower()

    def test_file_size_exceeded_rejected(self, monkeypatch, client):
        # Patch the max size limit in the documents route module to 10 bytes
        from app.api.routes import documents
        monkeypatch.setattr(documents, "MAX_FILE_SIZE_BYTES", 10)

        # Upload a 20-byte PDF (over the 10-byte patched limit)
        files = {"file": ("large.pdf", b"%PDF-1.4 dummy large content", "application/pdf")}
        data = {"user_id": "student_alice"}
        response = client.post("/api/v1/documents/process", files=files, data=data)
        assert response.status_code == 413
        assert "exceeds" in response.json()["message"].lower()

    def test_unsupported_file_rejected(self, client):
        # Uploading executable file format
        files = {"file": ("dangerous.exe", b"MZ\x90\x00...", "application/octet-stream")}
        data = {"user_id": "student_alice"}
        response = client.post("/api/v1/documents/process", files=files, data=data)
        assert response.status_code == 400
        assert "unsupported" in response.json()["message"].lower()

    def test_invalid_pdf_magic_bytes_rejected(self, client):
        # Uploading PDF that lacks %PDF magic header
        files = {"file": ("fake_report.pdf", b"fake PDF header info text here", "application/pdf")}
        data = {"user_id": "student_alice"}
        response = client.post("/api/v1/documents/process", files=files, data=data)
        assert response.status_code == 400
        assert "structure" in response.json()["message"].lower()

    def test_invalid_ooxml_magic_bytes_rejected(self, client):
        # Uploading PPTX that lacks PK\x03\x04 zip archive header
        files = {"file": ("fake_presentation.pptx", b"fake presentation header info", "application/vnd.openxmlformats-officedocument.presentationml.presentation")}
        data = {"user_id": "student_alice"}
        response = client.post("/api/v1/documents/process", files=files, data=data)
        assert response.status_code == 400
        assert "structure" in response.json()["message"].lower()

    def test_path_traversal_sanitized(self, client):
        # Malicious path injection filename
        files = {"file": ("../../../../etc/passwd.txt", b"simple document text content", "text/plain")}
        data = {"user_id": "student_alice"}
        response = client.post("/api/v1/documents/process", files=files, data=data)
        assert response.status_code == 200
        # Check that the response contains the sanitized basename
        data_res = response.json()["data"]
        assert "passwd.txt" in data_res["document_name"]
        assert "etc" not in data_res["document_name"]


# ===========================================================================
# 3. Error Handling Sanitization
# ===========================================================================

class TestErrorSanitization:
    def test_exception_handler_standardized(self, client):
        # Call with invalid payload to trigger validation error
        response = client.post("/api/v1/conversations", json={"fake_field": "invalid"})
        assert response.status_code == 422
        resp_json = response.json()
        assert resp_json["success"] is False
        assert "validation" in resp_json["message"].lower()
        assert resp_json["data"] is not None
        assert "errors" in resp_json["data"]


# ===========================================================================
# 4. Health, Liveness, and Readiness Probe Tests
# ===========================================================================

class TestHealthProbes:
    def test_liveness_endpoint(self, client):
        response = client.get("/api/v1/health/liveness")
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["success"] is True
        assert resp_json["data"]["status"] == "alive"

    def test_health_check_endpoint(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["success"] is True
        assert "components" in resp_json["data"]
        # Mongo/Astra DB status must be reported
        components = resp_json["data"]["components"]
        assert "mongodb" in components
        assert "astra_db" in components
        assert "llm" in components

    def test_readiness_endpoint_healthy(self, client):
        response = client.get("/api/v1/health/readiness")
        # In offline testing, some components are mocked or starting up.
        # Check that it returns either 200 or 503 depending on configuration.
        assert response.status_code in (200, 503)


# ===========================================================================
# 5. Development Endpoint Protection
# ===========================================================================

class TestDevelopmentEndpointProtection:
    def test_dev_endpoints_blocked_in_production(self, monkeypatch, client):
        # Temporarily set environment to production
        monkeypatch.setattr(settings, "ENVIRONMENT", "production")
        
        # Call embedding test endpoint
        response = client.post("/api/v1/embeddings/test", json={"texts": ["hello"]})
        assert response.status_code == 403
        assert "disabled" in response.json()["message"].lower()

        # Call vector-db insert test endpoint
        response = client.post("/api/v1/vector-db/test-insert", json={"text": "hello"})
        assert response.status_code == 403
        assert "disabled" in response.json()["message"].lower()
