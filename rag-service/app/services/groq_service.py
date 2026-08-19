"""
Groq LLM Service.

Provides a dedicated, production-grade integration with Groq's low-latency
LLM inference API using the official Python SDK (groq).

Features:
- Configurable model (default: llama-3.3-70b-versatile)
- Singleton client reuse across application lifecycle
- Zero-token startup configuration validation
- Standardized GenerationResult response schema with token counts and latency
- Conservative exponential backoff retry for transient network/rate-limit errors
- Strict exception mapping avoiding leakage of credentials or internals
"""

import time
from typing import Any, Optional, Union

import groq
from groq import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.llm import ChatMessage, GenerationResult, LLMHealthInfo

logger = get_logger(__name__)


# ===========================================================================
# Domain Exceptions
# ===========================================================================

class GroqServiceError(Exception):
    """Base exception for all Groq service failures."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class GroqAuthError(GroqServiceError):
    """Raised when the Groq API key is missing or unauthorized."""

    def __init__(self, message: str = "Groq authentication failed. Please verify GROQ_API_KEY in .env."):
        super().__init__(message, status_code=503)


class GroqRateLimitError(GroqServiceError):
    """Raised when Groq API rate limits are exceeded."""

    def __init__(self, message: str = "Groq rate limit exceeded. Please retry after a brief delay."):
        super().__init__(message, status_code=429)


class GroqTimeoutError(GroqServiceError):
    """Raised when a request to Groq times out."""

    def __init__(self, message: str = "Groq request timed out."):
        super().__init__(message, status_code=504)


class GroqModelError(GroqServiceError):
    """Raised when an invalid model or malformed payload is requested."""

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


# ===========================================================================
# GroqService Implementation
# ===========================================================================

class GroqService:
    """
    Service wrapper for Groq LLM API.

    Manages client lifecycle, executes text completions, applies transient
    retry policies, records token consumption and latency, and returns
    strongly-typed GenerationResult objects.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self.temperature = temperature if temperature is not None else settings.GROQ_TEMPERATURE
        self.max_completion_tokens = (
            max_completion_tokens if max_completion_tokens is not None else settings.GROQ_MAX_COMPLETION_TOKENS
        )
        self.timeout = timeout if timeout is not None else settings.GROQ_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else settings.GROQ_MAX_RETRIES

        self._client: Optional[groq.Groq] = None

        logger.info(
            "GroqService initialized: model='%s'  temp=%.2f  max_tokens=%d  timeout=%.1fs  configured=%s",
            self.model,
            self.temperature,
            self.max_completion_tokens,
            self.timeout,
            self.is_configured,
        )

    @property
    def is_configured(self) -> bool:
        """Return True if an API key is provided and non-empty."""
        return bool(self.api_key and self.api_key.strip())

    @property
    def is_ready(self) -> bool:
        """Return True if the service has valid configuration to make requests."""
        return self.is_configured

    @property
    def client(self) -> groq.Groq:
        """
        Get or initialize the reusable Groq SDK client instance.

        Raises:
            GroqAuthError: If GROQ_API_KEY is not configured.
        """
        if not self.is_configured:
            raise GroqAuthError("Groq API key is not configured. Set GROQ_API_KEY in .env.")

        if self._client is None:
            self._client = groq.Groq(
                api_key=self.api_key.strip(),
                timeout=self.timeout,
            )
        return self._client

    def _format_messages(
        self, messages: list[Union[ChatMessage, dict[str, Any]]]
    ) -> list[dict[str, str]]:
        """
        Validate and format input messages into Groq SDK dict format.

        Raises:
            GroqModelError: If messages list is empty or contains invalid roles/content.
        """
        if not messages:
            raise GroqModelError("Messages list cannot be empty.")

        formatted: list[dict[str, str]] = []
        valid_roles = {"system", "user", "assistant"}

        for idx, msg in enumerate(messages):
            if isinstance(msg, ChatMessage):
                role = msg.role
                content = msg.content
            elif isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
            else:
                raise GroqModelError(f"Message at index {idx} must be a ChatMessage or dict.")

            if not role or role not in valid_roles:
                raise GroqModelError(
                    f"Message at index {idx} has invalid role '{role}'. Allowed roles: {valid_roles}."
                )

            if not content or not str(content).strip():
                raise GroqModelError(f"Message at index {idx} has empty content.")

            formatted.append({"role": role, "content": str(content).strip()})

        return formatted

    def generate(
        self,
        messages: list[Union[ChatMessage, dict[str, Any]]],
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> GenerationResult:
        """
        Execute a chat completion request with the Groq API.

        Args:
            messages: List of ChatMessage objects or dicts with 'role' and 'content'.
            temperature: Optional temperature override (0.0 to 2.0).
            max_completion_tokens: Optional token limit override.
            model: Optional model identifier override.

        Returns:
            GenerationResult: Strongly-typed completion output with tokens and latency.

        Raises:
            GroqAuthError: If credentials are missing or invalid.
            GroqRateLimitError: If provider rate limits are exceeded.
            GroqTimeoutError: If request exceeds configured timeout.
            GroqModelError: If request payload or model is invalid.
            GroqServiceError: For unrecoverable provider failures.
        """
        client = self.client
        formatted_messages = self._format_messages(messages)

        target_model = model or self.model
        target_temp = temperature if temperature is not None else self.temperature
        target_max_tokens = (
            max_completion_tokens if max_completion_tokens is not None else self.max_completion_tokens
        )

        logger.info(
            "Calling Groq API: model='%s'  messages=%d  temp=%.2f  max_tokens=%d",
            target_model,
            len(formatted_messages),
            target_temp,
            target_max_tokens,
        )

        start_time = time.perf_counter()
        last_exception: Optional[Exception] = None

        # Bounded transient retry loop
        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=target_model,
                    messages=formatted_messages,  # type: ignore[arg-type]
                    temperature=target_temp,
                    max_completion_tokens=target_max_tokens,
                )
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

                if not response.choices:
                    raise GroqServiceError("Groq returned an empty response with no choices.")

                choice = response.choices[0]
                content = choice.message.content or ""
                finish_reason = choice.finish_reason or "stop"

                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                total_tokens = usage.total_tokens if usage else (input_tokens + output_tokens)

                request_id = getattr(response, "id", None)

                logger.info(
                    "Groq response received: model='%s'  finish_reason='%s'  tokens=%d (in=%d, out=%d)  latency=%.1fms",
                    target_model,
                    finish_reason,
                    total_tokens,
                    input_tokens,
                    output_tokens,
                    latency_ms,
                )

                return GenerationResult(
                    content=content,
                    model=target_model,
                    finish_reason=finish_reason,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    request_id=request_id,
                )

            except AuthenticationError as exc:
                logger.error("Groq authentication error: %s", exc)
                raise GroqAuthError(
                    "Groq API authentication failed. Please verify your GROQ_API_KEY."
                ) from exc

            except (BadRequestError, NotFoundError) as exc:
                logger.error("Groq bad request error: %s", exc)
                hint = ""
                if isinstance(exc, NotFoundError) or "model_not_found" in str(exc):
                    try:
                        avail = [
                            m.id
                            for m in client.models.list().data
                            if "whisper" not in m.id and "guard" not in m.id
                        ]
                        if avail:
                            hint = f" Available models on your Groq key: {', '.join(avail)}."
                    except Exception:
                        pass
                raise GroqModelError(f"Invalid Groq request or model: {exc}.{hint}") from exc

            except RateLimitError as exc:
                last_exception = exc
                logger.warning(
                    "Groq rate limit encountered (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt < self.max_retries:
                    backoff = 0.5 * (2 ** attempt)
                    time.sleep(backoff)
                    continue
                raise GroqRateLimitError(
                    "Groq rate limit exceeded. Please retry after a brief delay."
                ) from exc

            except APITimeoutError as exc:
                last_exception = exc
                logger.warning(
                    "Groq request timeout (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt < self.max_retries:
                    backoff = 0.5 * (2 ** attempt)
                    time.sleep(backoff)
                    continue
                raise GroqTimeoutError(
                    f"Groq API request timed out after {self.timeout}s."
                ) from exc

            except (APIConnectionError, InternalServerError) as exc:
                last_exception = exc
                logger.warning(
                    "Groq transient connection/server error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt < self.max_retries:
                    backoff = 0.5 * (2 ** attempt)
                    time.sleep(backoff)
                    continue
                raise GroqServiceError(
                    "Groq service is temporarily unreachable. Please try again later."
                ) from exc

            except APIError as exc:
                logger.error("Groq API error: %s", exc)
                raise GroqServiceError(f"Groq API error: {exc}") from exc

            except Exception as exc:
                if isinstance(exc, GroqServiceError):
                    raise
                logger.error("Unexpected error during Groq generation: %s", exc)
                raise GroqServiceError(f"Unexpected generation failure: {exc}") from exc

        # Fallback if loop finishes unexpectedly
        raise GroqServiceError(
            f"Groq generation failed after {self.max_retries + 1} attempts: {last_exception}"
        )

    def get_health_info(self) -> LLMHealthInfo:
        """
        Return readiness and configuration status without executing an API call.
        """
        if self.is_ready:
            status = "ready"
        elif not self.is_configured:
            status = "not_configured"
        else:
            status = "unavailable"

        return LLMHealthInfo(
            status=status,
            provider="groq",
            model=self.model,
            configured=self.is_configured,
        )
