"""Retrying HTTP for ingest.

WHY THIS EXISTS
---------------
The Docker build fetches four PDFs from wko.at and ~100 documents from RIS. A
34-minute build died on:

    httpx.RemoteProtocolError: Server disconnected without sending a response.

One dropped connection, thirty-four minutes lost, and nothing wrong with the
code. Public web servers drop connections; an ingest that treats every blip as
fatal is not usable in a build pipeline, and it fails at the least convenient
moment -- after the expensive layers have already been rebuilt.

WHAT IS AND IS NOT RETRIED
--------------------------
Retried: transport errors (connection dropped, reset, timed out) and 5xx or
429 responses. These are conditions that plausibly differ on a second attempt.

NOT retried: 4xx other than 429. A 404 will still be a 404 in two seconds, and
retrying it only delays a failure that needs a human to look at the manifest.

The backoff is exponential with a cap, because hammering a free public service
that is already struggling is both rude and counterproductive.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import httpx

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}
TRANSIENT = (
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.ReadError,
    httpx.WriteError,
    httpx.PoolTimeout,
)


def get_with_retry(client: httpx.Client, url: str, *, attempts: int = 4,
                   base_delay: float = 1.5, label: str = "",
                   on_retry: Callable[[int, str], None] | None = None,
                   **kwargs) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            r = client.get(url, **kwargs)
            if r.status_code in RETRY_STATUS and attempt < attempts:
                reason = f"HTTP {r.status_code}"
            else:
                r.raise_for_status()
                return r
        except TRANSIENT as e:
            last = e
            reason = f"{type(e).__name__}: {e}"
        except httpx.HTTPStatusError as e:
            # A 404 will still be a 404 in two seconds.
            raise

        if attempt == attempts:
            break
        delay = min(base_delay * (2 ** (attempt - 1)), 20.0)
        msg = f"{label or url}: {reason}; retry {attempt}/{attempts - 1} in {delay:.1f}s"
        log.warning(msg)
        if on_retry:
            on_retry(attempt, reason)
        time.sleep(delay)

    raise RuntimeError(
        f"{label or url}: failed after {attempts} attempts. Last error: {last}"
    ) from last
