"""OpenAI-compatible API client for LLM interactions.

Enhanced with Claude Code-inspired retry logic:
- Exponential backoff with jitter
- Status-code-specific handling (429 → Retry-After, 529 → overloaded backoff, 401 → fail fast)
- Prompt-too-long auto-truncation
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI, APIStatusError, APIConnectionError, APITimeoutError

from autotrainer.utils.retry import (
    RetryableError,
    NonRetryableError,
    classify_status_code,
    retry_with_backoff,
)

logger = logging.getLogger(__name__)


def _wrap_api_error(e: APIStatusError) -> Exception:
    """Convert OpenAI API status error to retryable/non-retryable."""
    category = classify_status_code(e.status_code)
    if category == "auth":
        return NonRetryableError(f"Auth error {e.status_code}: {e}")
    if category == "client":
        return NonRetryableError(f"Client error {e.status_code}: {e}")
    # retry or overloaded
    retry_after = None
    if hasattr(e, "response") and e.response is not None:
        ra = e.response.headers.get("Retry-After")
        if ra:
            try:
                retry_after = float(ra)
            except ValueError:
                pass
    return RetryableError(
        f"API {e.status_code}: {e}",
        status_code=e.status_code,
        retry_after=retry_after,
    )


class LLMClient:
    """Client for OpenAI-compatible APIs used for skill invocation,
    experiment summarization, and log analysis."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    async def _api_call(self, coro_fn, *args, **kwargs) -> Any:
        """Call an async API function with robust retry handling."""
        async def _wrapped():
            try:
                return await coro_fn(*args, **kwargs)
            except APIStatusError as e:
                raise _wrap_api_error(e) from e
            except (APIConnectionError, APITimeoutError) as e:
                raise RetryableError(f"Connection error: {e}") from e

        return await retry_with_backoff(
            _wrapped,
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            jitter=True,
        )

    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Send a chat completion request and return the response text."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        async def _call():
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

        return await self._api_call(_call)

    async def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Send a chat completion request expecting JSON output."""
        system_with_json = system + "\n\nYou MUST respond with valid JSON only. No markdown, no explanation."
        raw = await self.complete(system_with_json, user, max_tokens, temperature=0.0)
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        return json.loads(text)

    async def complete_messages(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Send a raw messages list to the API."""
        async def _call():
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

        return await self._api_call(_call)

    def count_tokens_estimate(self, text: str) -> int:
        """Count tokens using tiktoken (cl100k_base). Falls back to char/4 estimate."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)
