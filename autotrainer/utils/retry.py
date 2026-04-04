"""Retry utilities with exponential backoff, jitter, and status-code-aware handling.

Inspired by Claude Code's withRetry.ts:
- Exponential backoff with jitter
- Status-code-specific handling (429, 529, 401, network errors)
- Configurable retry strategies per error type
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryableError(Exception):
    """Errors that should trigger a retry."""

    def __init__(self, message: str, *, status_code: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class NonRetryableError(Exception):
    """Errors that should NOT be retried (401, 403, bad request, etc.)."""


def get_retry_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, jitter: bool = True) -> float:
    """Compute retry delay with exponential backoff + optional jitter.

    Args:
        attempt: 1-indexed attempt number
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        jitter: Add random jitter (±25%) to prevent thundering herd
    """
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    if jitter:
        delay = delay * (0.75 + random.random() * 0.5)  # ±25%
    return delay


def classify_status_code(status_code: int) -> str:
    """Classify HTTP status code into retry category.

    Returns:
        "retry" — transient, safe to retry (429, 5xx)
        "overloaded" — server overloaded (529), retry with longer backoff
        "auth" — authentication error, never retry (401, 403)
        "client" — client error, never retry (4xx except 429)
        "success" — 2xx
    """
    if 200 <= status_code < 300:
        return "success"
    if status_code == 429:
        return "retry"
    if status_code == 529:
        return "overloaded"
    if status_code in (401, 403):
        return "auth"
    if 400 <= status_code < 500:
        return "client"
    if status_code >= 500:
        return "retry"
    return "client"


async def retry_with_backoff(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retry_on: tuple[type[Exception], ...] | None = None,
    **kwargs: Any,
) -> Any:
    """Execute an async function with exponential backoff retry.

    Args:
        fn: Async callable to execute
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries (seconds)
        max_delay: Maximum delay cap (seconds)
        jitter: Add random jitter to delays
        retry_on: Tuple of exception types to retry on. If None, retries on
                  RetryableError, ConnectionError, TimeoutError, OSError.
    """
    if retry_on is None:
        retry_on = (RetryableError, ConnectionError, TimeoutError, OSError)

    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except retry_on as e:
            last_exc = e

            # Check for Retry-After header
            retry_after = getattr(e, "retry_after", None)

            if attempt >= max_retries:
                logger.error("All %d retries exhausted: %s", max_retries, e)
                raise

            if retry_after and retry_after > 0:
                delay = min(retry_after, max_delay)
                logger.warning(
                    "Retryable error (attempt %d/%d), server requested %.1fs delay: %s",
                    attempt, max_retries, delay, e,
                )
            else:
                delay = get_retry_delay(attempt, base_delay, max_delay, jitter)
                logger.warning(
                    "Retryable error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt, max_retries, delay, e,
                )

            await asyncio.sleep(delay)

    raise last_exc  # should not reach here


def retry_sync(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
):
    """Decorator for sync functions with retry logic."""
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last_exc: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except (ConnectionError, TimeoutError, OSError) as e:
                    last_exc = e
                    if attempt >= max_retries:
                        raise
                    delay = get_retry_delay(attempt, base_delay, max_delay, jitter)
                    logger.warning(
                        "Sync retry (attempt %d/%d), retrying in %.1fs: %s",
                        attempt, max_retries, delay, e,
                    )
                    import time
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator
