"""OpenAI-compatible API client for LLM interactions."""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI


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
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

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
            # Remove first line (```json) and last line (```)
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        return json.loads(text)

    async def complete_messages(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Send a raw messages list to the API."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def count_tokens_estimate(self, text: str) -> int:
        """Rough token count estimate (1 token ≈ 4 chars for English)."""
        return max(1, len(text) // 4)
