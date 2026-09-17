"""Retry behaviour for ingest HTTP.

A dropped connection killed a 34-minute Docker build. These tests pin down
what is retried and, just as importantly, what is not: a 404 retried four
times is four times the delay before someone reads the error.
"""
from __future__ import annotations

import httpx
import pytest

from atkv.ingest.http import get_with_retry


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_recovers_from_a_dropped_connection():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return httpx.Response(200, content=b"%PDF-1.7 ok")

    r = get_with_retry(_client(handler), "https://example.test/x", base_delay=0.01)
    assert r.status_code == 200
    assert calls["n"] == 3, "should have retried twice then succeeded"


def test_retries_5xx_and_429():
    for status in (429, 503):
        calls = {"n": 0}

        def handler(request, _s=status):
            calls["n"] += 1
            return httpx.Response(200 if calls["n"] > 1 else _s, content=b"ok")

        r = get_with_retry(_client(handler), "https://example.test/x", base_delay=0.01)
        assert r.status_code == 200 and calls["n"] == 2, status


def test_does_not_retry_a_404():
    """It will still be a 404 in two seconds."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        get_with_retry(_client(handler), "https://example.test/missing", base_delay=0.01)
    assert calls["n"] == 1, "a 404 must fail immediately, not after four attempts"


def test_gives_up_and_says_why():
    def handler(request):
        raise httpx.ConnectError("no route to host")

    with pytest.raises(RuntimeError, match="failed after 4 attempts"):
        get_with_retry(_client(handler), "https://example.test/x",
                       base_delay=0.01, label="it-kv-2026-de")
