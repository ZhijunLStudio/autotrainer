"""Token counting utilities."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Rough token count estimate.

    Uses the 1 token ≈ 4 chars heuristic (for English/Chinese mixed text).
    This is deliberately conservative — slightly overcounts to prevent overflow.
    """
    return max(1, len(text) // 3)


def estimate_tokens_from_lines(lines: list[str]) -> int:
    """Estimate token count from a list of lines."""
    return sum(estimate_tokens(line) for line in lines)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately fit within a token budget."""
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text
    # Proportionally trim
    ratio = max_tokens / estimated
    keep_chars = int(len(text) * ratio)
    return text[:keep_chars] + "\n...[truncated]"


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
