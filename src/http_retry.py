"""Small, deterministic HTTP retry helper for official data downloads."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import requests

DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_MAX_DELAY_SECONDS = 60.0
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _retry_after_seconds(value: object, *, now: datetime | None = None) -> float | None:
    """Parse an HTTP Retry-After delay in seconds or HTTP-date form."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    return max(0.0, (retry_at - current).total_seconds())


def get_with_retry(
    session: requests.Session,
    url: str,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    max_delay_seconds: float = DEFAULT_MAX_DELAY_SECONDS,
    sleep: Callable[[float], None] | None = None,
    **request_kwargs: Any,
) -> requests.Response:
    """GET a URL with bounded retries for throttling and transient failures.

    ``Retry-After`` takes precedence over exponential backoff when present.
    Delays are capped so a bad server hint cannot stall a scheduled workflow
    indefinitely. Non-retryable HTTP responses are returned to the caller,
    which remains responsible for calling ``raise_for_status``.
    """
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if backoff_seconds < 0:
        raise ValueError("backoff_seconds cannot be negative")
    if max_delay_seconds < 0:
        raise ValueError("max_delay_seconds cannot be negative")

    sleep_fn = sleep or time.sleep
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, **request_kwargs)
        except requests.RequestException:
            if attempt == max_attempts:
                raise
            delay = min(backoff_seconds * (2 ** (attempt - 1)), max_delay_seconds)
            sleep_fn(delay)
            continue

        if response.status_code not in RETRYABLE_STATUS_CODES:
            return response
        if attempt == max_attempts:
            return response

        retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
        delay = (
            retry_after
            if retry_after is not None
            else backoff_seconds * (2 ** (attempt - 1))
        )
        response.close()
        sleep_fn(min(delay, max_delay_seconds))

    raise AssertionError("unreachable")
