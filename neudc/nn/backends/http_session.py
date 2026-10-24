"""Shared HTTP session builder for the OpenAI-compatible ASR/LLM/TTS backends.

A remote inference server's keep-alive connection can go stale between calls — voice
turns are gappy by nature (a call only fires when the user actually speaks or a reply
needs synthesizing), easily longer than a typical server-side keep-alive timeout.
Reusing a pooled connection the server already closed fails with
`requests.exceptions.ConnectionError` (`RemoteDisconnected: Remote end closed
connection without response`) before the request is even sent — and since ASR/LLM/TTS
nodes run under the default `skip` error policy, that exception just drops the whole
turn silently rather than retrying.
"""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

__all__ = ("build_retrying_session",)


def build_retrying_session(total_retries: int = 2, backoff_factor: float = 0.3) -> requests.Session:
    """Build a `requests.Session` that retries a request whose connection dropped before any response arrived.

    `POST` is explicitly included in `allowed_methods`: `urllib3`'s `Retry` excludes
    it by default, since retrying a non-idempotent request could double-process it if
    the server had already started handling it before the connection dropped. That
    risk doesn't apply here — a connection that closed *before any response at all*
    means the request essentially never reached the server's handler (a dead pooled
    connection rejected on reuse), so retrying is safe, not a duplicate-inference risk.

    Args:
    ----
        total_retries: Maximum retry attempts before giving up and raising.
        backoff_factor: Delay multiplier between retries (`urllib3`'s own backoff formula).

    Returns:
    -------
        A `requests.Session` with the retrying adapter mounted for both `http://`/`https://`.

    """
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
