"""Tests for build_retrying_session: recovers from a dropped-before-any-response connection.

Reproduces the exact failure class this exists for — `RemoteDisconnected('Remote end
closed connection without response')` — with a real local TCP server (no mocking of
`requests`/`urllib3` internals, since the whole point is the *transport-layer* retry
behavior those internals implement) that drops the connection on its first accept and
responds normally on the second, so the retry path is exercised for real.
"""

from __future__ import annotations

import socket
import threading

import pytest

from neudc.nn.backends.http_session import build_retrying_session


def _serve_drop_then_ok(server_socket: socket.socket, attempts: list[int], drop_first_n: int) -> None:
    while True:
        conn, _ = server_socket.accept()
        attempts.append(len(attempts) + 1)
        if len(attempts) <= drop_first_n:
            conn.close()  # closed with no response at all -- the RemoteDisconnected case
            continue
        conn.recv(4096)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
        conn.close()
        return


def _start_flaky_server(drop_first_n: int) -> tuple[int, list[int]]:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("127.0.0.1", 0))
    server_socket.listen(5)
    port = server_socket.getsockname()[1]

    attempts: list[int] = []
    thread = threading.Thread(
        target=_serve_drop_then_ok,
        args=(server_socket, attempts, drop_first_n),
        daemon=True,
    )
    thread.start()
    return port, attempts


def test_session_retries_a_connection_dropped_before_any_response() -> None:
    port, attempts = _start_flaky_server(drop_first_n=1)
    session = build_retrying_session(total_retries=2, backoff_factor=0.01)

    response = session.post(f"http://127.0.0.1:{port}/", timeout=5)

    assert response.status_code == 200
    assert response.text == "ok"
    assert len(attempts) == 2  # first attempt dropped, second (the retry) succeeded


def test_session_gives_up_after_exhausting_retries() -> None:
    # Every attempt drops -- more than total_retries can cover -- so the caller must
    # still see a real error rather than hang or retry forever.
    port, attempts = _start_flaky_server(drop_first_n=10)
    session = build_retrying_session(total_retries=1, backoff_factor=0.01)

    with pytest.raises(Exception):  # noqa: B017,PT011 - asserting *some* connection-level error, not a specific class
        session.post(f"http://127.0.0.1:{port}/", timeout=5)

    assert len(attempts) == 2  # the original attempt + 1 retry, then it gave up


def test_session_mounts_adapter_for_http_and_https() -> None:
    session = build_retrying_session()

    assert "http://" in session.adapters
    assert "https://" in session.adapters


def test_session_allows_post_retries() -> None:
    session = build_retrying_session(total_retries=3, backoff_factor=0.2)

    adapter = session.get_adapter("http://example.test")
    retry = adapter.max_retries

    assert retry.total == 3
    assert "POST" in retry.allowed_methods
    assert "GET" in retry.allowed_methods
