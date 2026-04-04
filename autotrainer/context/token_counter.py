"""Token counting utilities with tiktoken support.

Inspired by Claude Code's truncate.ts:
- Width-aware text truncation
- tiktoken integration for accurate counting
- Head-based truncation (discard oldest, keep newest)
"""

from __future__ import annotations

import re


def estimate_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base). Falls back to char/3 heuristic."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 3)


def estimate_tokens_from_lines(lines: list[str]) -> int:
    """Estimate token count from a list of lines."""
    return sum(estimate_tokens(line) for line in lines)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text from the end to fit within a token budget."""
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text
    # Proportionally trim from the end
    ratio = max_tokens / estimated
    keep_chars = int(len(text) * ratio)
    return text[:keep_chars] + "\n...[truncated]"


def truncate_head(text: str, max_tokens: int) -> str:
    """Truncate text from the HEAD (oldest content), keeping the tail.

    Claude Code's approach: when context is too long, drop the oldest messages
    first, not the newest ones. This preserves recent context.
    """
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text
    ratio = max_tokens / estimated
    drop_chars = int(len(text) * (1 - ratio))
    return "...[earlier content truncated]...\n" + text[drop_chars:]


def truncate_head_messages(messages: list[dict], max_tokens: int) -> list[dict]:
    """Truncate a message list from the head, keeping recent messages.

    Returns a subset of messages that fit within max_tokens.
    Always keeps at least the last message.
    """
    total = sum(estimate_tokens(str(m)) for m in messages)
    if total <= max_tokens:
        return messages

    # Drop from the front until we fit
    result = list(messages)
    while len(result) > 1:
        total -= estimate_tokens(str(result[0]))
        result.pop(0)
        if total <= max_tokens:
            break

    return result


def truncate_lines(lines: list[str], max_tokens: int) -> list[str]:
    """Truncate lines to fit within a token budget, keeping the end."""
    total = 0
    result = []
    for line in reversed(lines):
        line_tokens = estimate_tokens(line)
        if total + line_tokens > max_tokens:
            break
        result.insert(0, line)
        total += line_tokens
    return result
