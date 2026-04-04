"""OpenAI-compatible API client for LLM interactions."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI, APIStatusError, APIConnectionError, APITimeoutError

logger = logging.getLogger(__name__)

# Retry config
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0   # seconds, doubles each attempt


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

    async def _call_with_retry(self, coro_fn, *args, **kwargs) -> Any:
        """Call an async function with exponential-backoff retry on transient errors."""
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await coro_fn(*args, **kwargs)
            except APIStatusError as e:
                # 429 rate-limit and 5xx server errors are retryable
                if e.status_code in (429, 500, 502, 503, 504):
                    last_exc = e
                    if attempt < _MAX_RETRIES:
                        delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        logger.warning(
                            "LLM API %s (attempt %d/%d), retrying in %.1fs",
                            e.status_code, attempt, _MAX_RETRIES, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                raise
            except (APIConnectionError, APITimeoutError) as e:
                last_exc = e
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM connection error (attempt %d/%d): %s, retrying in %.1fs",
                        attempt, _MAX_RETRIES, e, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
        raise last_exc  # should not reach here

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

        return await self._call_with_retry(_call)

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

        return await self._call_with_retry(_call)

    def count_tokens_estimate(self, text: str) -> int:
        """Count tokens using tiktoken (cl100k_base). Falls back to char/4 estimate."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)
