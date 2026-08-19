"""
Tests for Groq LLM Service (Task 8).

Covers:
1. Configuration validation and singleton client initialization
2. Message validation (roles, content, formatting)
3. Successful text generation and GenerationResult parsing
4. Token usage accounting and latency measurement
5. Error mapping (auth, rate limits, timeouts, model errors)
6. Bounded transient retry logic
7. Health check reporting
8. FastAPI route testing (POST /api/v1/llm/test)
9. Live integration test with real Groq API (marked with skipif when no key)
"""

import os
import time
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
load_dotenv()

import pytest
from fastapi.testclient import TestClient
from groq import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)
import httpx

from app.core.config import settings
from app.main import app
from app.schemas.llm import ChatMessage, GenerationResult, LLMHealthInfo, LLMTestRequest
from app.services.groq_service import (
    GroqAuthError,
    GroqModelError,
    GroqRateLimitError,
    GroqService,
    GroqServiceError,
    GroqTimeoutError,
)


# ===========================================================================
# Fixtures & Helpers
# ===========================================================================

@pytest.fixture
def mock_groq_response():
    """Build a mock ChatCompletion response matching the Groq SDK structure."""
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Deadlock is a condition where two or more processes are unable to proceed because each is waiting for the other to release a resource."
    mock_choice.finish_reason = "stop"
    mock_resp.choices = [mock_choice]

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 24
    mock_usage.completion_tokens = 28
    mock_usage.total_tokens = 52
    mock_resp.usage = mock_usage
    mock_resp.id = "chatcmpl-test-request-id-12345"
    return mock_resp


@pytest.fixture
def configured_groq_service():
    """Return a GroqService instance with a dummy API key."""
    return GroqService(
        api_key="gsk_test_mock_dummy_api_key_for_testing",
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_completion_tokens=1024,
        timeout=10.0,
        max_retries=2,
    )


@pytest.fixture
def test_client():
    """Return a TestClient instance."""
    return TestClient(app)


# ===========================================================================
# 1. Configuration & Client Initialization Tests
# ===========================================================================

class TestGroqConfigurationAndInit:
    """Test configuration loading, client initialization, and readiness."""

    def test_groq_service_init_defaults(self):
        service = GroqService(api_key="test_key")
        assert service.model == settings.GROQ_MODEL
        assert service.temperature == settings.GROQ_TEMPERATURE
        assert service.max_completion_tokens == settings.GROQ_MAX_COMPLETION_TOKENS
        assert service.timeout == settings.GROQ_TIMEOUT
        assert service.max_retries == settings.GROQ_MAX_RETRIES
        assert service.is_configured is True
        assert service.is_ready is True

    def test_groq_service_custom_config(self):
        service = GroqService(
            api_key="custom_key",
            model="custom-model-id",
            temperature=0.7,
            max_completion_tokens=512,
            timeout=15.0,
            max_retries=1,
        )
        assert service.model == "custom-model-id"
        assert service.temperature == 0.7
        assert service.max_completion_tokens == 512
        assert service.timeout == 15.0
        assert service.max_retries == 1

    def test_groq_service_missing_api_key(self):
        service = GroqService(api_key="")
        assert service.is_configured is False
        assert service.is_ready is False
        with pytest.raises(GroqAuthError, match="Groq API key is not configured"):
            _ = service.client

    def test_groq_client_singleton_reuse(self, configured_groq_service):
        client1 = configured_groq_service.client
        client2 = configured_groq_service.client
        assert client1 is client2, "Client instance must be reused across invocations"


# ===========================================================================
# 2. Message Validation Tests
# ===========================================================================

class TestMessageValidation:
    """Test validation of conversation messages and roles."""

    def test_empty_messages_raises_model_error(self, configured_groq_service):
        with pytest.raises(GroqModelError, match="Messages list cannot be empty"):
            configured_groq_service.generate([])

    def test_invalid_role_raises_model_error(self, configured_groq_service):
        with pytest.raises(GroqModelError, match="invalid role 'admin'"):
            configured_groq_service.generate([{"role": "admin", "content": "Hello"}])

    def test_empty_content_raises_model_error(self, configured_groq_service):
        with pytest.raises(GroqModelError, match="empty content"):
            configured_groq_service.generate([{"role": "user", "content": "   "}])

    def test_system_user_assistant_messages_formatted(self, configured_groq_service):
        msgs = [
            ChatMessage(role="system", content="You are a tutor."),
            ChatMessage(role="user", content="What is an OS?"),
            ChatMessage(role="assistant", content="An Operating System manages hardware."),
            {"role": "user", "content": "Tell me more."},
        ]
        formatted = configured_groq_service._format_messages(msgs)
        assert len(formatted) == 4
        assert formatted[0] == {"role": "system", "content": "You are a tutor."}
        assert formatted[1] == {"role": "user", "content": "What is an OS?"}
        assert formatted[2] == {"role": "assistant", "content": "An Operating System manages hardware."}
        assert formatted[3] == {"role": "user", "content": "Tell me more."}


# ===========================================================================
# 3. Successful Generation & Result Parsing (Mocked)
# ===========================================================================

class TestSuccessfulGenerationMocked:
    """Test text generation with mocked Groq SDK completions."""

    def test_successful_generation_mocked(self, configured_groq_service, mock_groq_response):
        with patch.object(configured_groq_service.client.chat.completions, "create", return_value=mock_groq_response) as mock_create:
            messages = [
                ChatMessage(role="system", content="You are an educational assistant."),
                ChatMessage(role="user", content="Explain deadlock."),
            ]
            result = configured_groq_service.generate(messages)

            mock_create.assert_called_once_with(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an educational assistant."},
                    {"role": "user", "content": "Explain deadlock."},
                ],
                temperature=0.2,
                max_completion_tokens=1024,
            )

            assert isinstance(result, GenerationResult)
            assert "Deadlock is a condition" in result.content
            assert result.model == "llama-3.3-70b-versatile"
            assert result.finish_reason == "stop"
            assert result.input_tokens == 24
            assert result.output_tokens == 28
            assert result.total_tokens == 52
            assert result.latency_ms >= 0.0
            assert result.request_id == "chatcmpl-test-request-id-12345"

    def test_custom_parameters_override(self, configured_groq_service, mock_groq_response):
        with patch.object(configured_groq_service.client.chat.completions, "create", return_value=mock_groq_response) as mock_create:
            result = configured_groq_service.generate(
                messages=[{"role": "user", "content": "Custom test"}],
                temperature=0.8,
                max_completion_tokens=256,
                model="llama-3.1-8b-instant",
            )
            mock_create.assert_called_once_with(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "Custom test"}],
                temperature=0.8,
                max_completion_tokens=256,
            )
            assert result.model == "llama-3.1-8b-instant"


# ===========================================================================
# 4. Error Handling & Retry Behavior (Mocked)
# ===========================================================================

class TestErrorHandlingAndRetries:
    """Test mapping of provider exceptions and bounded retry logic."""

    def test_auth_error_handling(self, configured_groq_service):
        mock_req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        mock_res = httpx.Response(401, request=mock_req, json={"error": {"message": "Invalid API Key"}})

        with patch.object(
            configured_groq_service.client.chat.completions,
            "create",
            side_effect=AuthenticationError(message="Invalid API Key", response=mock_res, body=None),
        ):
            with pytest.raises(GroqAuthError, match="Groq API authentication failed"):
                configured_groq_service.generate([{"role": "user", "content": "Hi"}])

    def test_bad_request_error_handling(self, configured_groq_service):
        mock_req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        mock_res = httpx.Response(400, request=mock_req, json={"error": {"message": "Model not found"}})

        with patch.object(
            configured_groq_service.client.chat.completions,
            "create",
            side_effect=BadRequestError(message="Model not found", response=mock_res, body=None),
        ):
            with pytest.raises(GroqModelError, match="Invalid Groq request or model"):
                configured_groq_service.generate([{"role": "user", "content": "Hi"}])

    def test_rate_limit_retry_and_eventual_fail(self, configured_groq_service):
        mock_req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        mock_res = httpx.Response(429, request=mock_req, json={"error": {"message": "Rate limit reached"}})

        with patch.object(
            configured_groq_service.client.chat.completions,
            "create",
            side_effect=RateLimitError(message="Rate limit reached", response=mock_res, body=None),
        ) as mock_create:
            with patch("time.sleep") as mock_sleep:
                with pytest.raises(GroqRateLimitError, match="Groq rate limit exceeded"):
                    configured_groq_service.generate([{"role": "user", "content": "Hi"}])

                # Max retries = 2 -> Total 3 attempts -> 2 sleep calls
                assert mock_create.call_count == 3
                assert mock_sleep.call_count == 2

    def test_timeout_retry_and_eventual_fail(self, configured_groq_service):
        mock_req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")

        with patch.object(
            configured_groq_service.client.chat.completions,
            "create",
            side_effect=APITimeoutError(request=mock_req),
        ) as mock_create:
            with patch("time.sleep"):
                with pytest.raises(GroqTimeoutError, match="Groq API request timed out"):
                    configured_groq_service.generate([{"role": "user", "content": "Hi"}])

                assert mock_create.call_count == 3

    def test_transient_error_retry_success(self, configured_groq_service, mock_groq_response):
        mock_req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")

        # First call fails with APIConnectionError, second succeeds
        with patch.object(
            configured_groq_service.client.chat.completions,
            "create",
            side_effect=[APIConnectionError(request=mock_req), mock_groq_response],
        ) as mock_create:
            with patch("time.sleep"):
                result = configured_groq_service.generate([{"role": "user", "content": "Hi"}])

                assert mock_create.call_count == 2
                assert "Deadlock is a condition" in result.content

    def test_empty_choices_raises_groq_service_error(self, configured_groq_service):
        mock_resp = MagicMock()
        mock_resp.choices = []

        with patch.object(
            configured_groq_service.client.chat.completions,
            "create",
            return_value=mock_resp,
        ):
            with pytest.raises(GroqServiceError, match="Groq returned an empty response"):
                configured_groq_service.generate([{"role": "user", "content": "Hi"}])


# ===========================================================================
# 5. Health & Readiness Tests
# ===========================================================================

class TestHealthAndReadiness:
    """Test health check reporting without API token consumption."""

    def test_health_info_when_configured(self, configured_groq_service):
        info = configured_groq_service.get_health_info()
        assert isinstance(info, LLMHealthInfo)
        assert info.status == "ready"
        assert info.provider == "groq"
        assert info.model == "llama-3.3-70b-versatile"
        assert info.configured is True

    def test_health_info_when_not_configured(self):
        service = GroqService(api_key="")
        info = service.get_health_info()
        assert info.status == "not_configured"
        assert info.configured is False

    def test_health_endpoint_reports_groq_status(self, test_client):
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        components = data["data"]["components"]
        assert "llm" in components


# ===========================================================================
# 6. API Route Endpoint Tests (POST /api/v1/llm/test)
# ===========================================================================

class TestLLMAPIRoute:
    """Test FastAPI route endpoint POST /api/v1/llm/test."""

    def test_api_llm_test_endpoint_success(self, test_client, mock_groq_response):
        app.state.groq_service = GroqService(api_key="gsk_mock_test_key")

        with patch.object(
            app.state.groq_service.client.chat.completions,
            "create",
            return_value=mock_groq_response,
        ):
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a helpful educational assistant."},
                    {"role": "user", "content": "Explain deadlock in simple terms."},
                ],
                "temperature": 0.2,
                "max_completion_tokens": 512,
            }
            response = test_client.post("/api/v1/llm/test", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "Deadlock is a condition" in data["data"]["content"]
            assert data["data"]["model"] == app.state.groq_service.model
            assert data["data"]["input_tokens"] == 24
            assert data["data"]["output_tokens"] == 28
            assert data["data"]["total_tokens"] == 52

    def test_api_llm_test_endpoint_missing_messages_returns_422(self, test_client):
        response = test_client.post("/api/v1/llm/test", json={})
        assert response.status_code == 422

    def test_api_llm_test_endpoint_empty_messages_returns_422(self, test_client):
        response = test_client.post("/api/v1/llm/test", json={"messages": []})
        assert response.status_code == 422

    def test_api_llm_test_endpoint_invalid_role_returns_422(self, test_client):
        payload = {
            "messages": [{"role": "superuser", "content": "Hello"}]
        }
        response = test_client.post("/api/v1/llm/test", json=payload)
        assert response.status_code == 422

    def test_api_llm_test_endpoint_empty_content_returns_422(self, test_client):
        payload = {
            "messages": [{"role": "user", "content": "   "}]
        }
        response = test_client.post("/api/v1/llm/test", json=payload)
        assert response.status_code == 422


# ===========================================================================
# 7. Live Integration Test (Real Groq API Call)
# ===========================================================================

@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY") or not os.getenv("GROQ_API_KEY").strip(),
    reason="GROQ_API_KEY not configured in environment.",
)
class TestLiveGroqIntegration:
    """
    Live integration test executing a real, lightweight API call to Groq.
    Verifies that Groq inference returns valid completions with token usage and latency.
    """

    def test_live_groq_generation(self):
        service = GroqService()
        assert service.is_configured, "GROQ_API_KEY must be configured for live test"

        try:
            available_models = [m.id for m in service.client.models.list().data]
        except Exception:
            available_models = []

        test_model = service.model
        if test_model not in available_models and available_models:
            chat_candidates = ["openai/gpt-oss-120b", "groq/compound-mini", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
            for candidate in chat_candidates:
                if candidate in available_models:
                    test_model = candidate
                    break

        messages = [
            ChatMessage(role="system", content="You are an expert computer science tutor. Be extremely concise."),
            ChatMessage(role="user", content="Explain deadlock in one sentence."),
        ]

        result = service.generate(
            messages=messages,
            temperature=0.1,
            max_completion_tokens=256,
            model=test_model,
        )

        assert isinstance(result, GenerationResult)
        assert len(result.content.strip()) > 0
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.total_tokens == result.input_tokens + result.output_tokens
        assert result.latency_ms > 0.0
        assert result.finish_reason in {"stop", "length"}
