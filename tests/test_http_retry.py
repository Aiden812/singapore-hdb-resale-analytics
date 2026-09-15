"""Deterministic tests for bounded official-data HTTP retries."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import Mock, call

import requests

from src.http_retry import _retry_after_seconds, get_with_retry


class HttpRetryTests(unittest.TestCase):
    def test_honours_retry_after_before_retrying_rate_limit(self) -> None:
        throttled = Mock(status_code=429, headers={"Retry-After": "7"})
        succeeded = Mock(status_code=200, headers={})
        session = Mock()
        session.get.side_effect = [throttled, succeeded]
        sleep = Mock()

        result = get_with_retry(session, "https://example.test", sleep=sleep)

        self.assertIs(result, succeeded)
        self.assertEqual(session.get.call_count, 2)
        sleep.assert_called_once_with(7.0)
        throttled.close.assert_called_once_with()

    def test_retries_network_failures_with_bounded_backoff(self) -> None:
        succeeded = Mock(status_code=200, headers={})
        session = Mock()
        session.get.side_effect = [
            requests.ConnectionError("connection reset"),
            requests.Timeout("upstream timeout"),
            succeeded,
        ]
        sleep = Mock()

        result = get_with_retry(
            session,
            "https://example.test",
            max_attempts=3,
            backoff_seconds=0.5,
            sleep=sleep,
        )

        self.assertIs(result, succeeded)
        self.assertEqual(sleep.call_args_list, [call(0.5), call(1.0)])

    def test_stops_after_configured_number_of_server_errors(self) -> None:
        responses = [Mock(status_code=503, headers={}) for _ in range(3)]
        session = Mock()
        session.get.side_effect = responses
        sleep = Mock()

        result = get_with_retry(
            session,
            "https://example.test",
            max_attempts=3,
            backoff_seconds=1.0,
            sleep=sleep,
        )

        self.assertIs(result, responses[-1])
        self.assertEqual(session.get.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(1.0), call(2.0)])
        responses[-1].close.assert_not_called()

    def test_parses_retry_after_http_date(self) -> None:
        now = datetime(2026, 9, 15, 3, 0, tzinfo=UTC)

        delay = _retry_after_seconds(
            "Tue, 15 Sep 2026 03:00:12 GMT",
            now=now,
        )

        self.assertEqual(delay, 12.0)


if __name__ == "__main__":
    unittest.main()
