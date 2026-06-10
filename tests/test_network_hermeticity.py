"""Canary for the suite-wide socket guard (audit 2026-06-10).

pyproject's addopts pin the whole suite to
``--disable-socket --allow-unix-socket --allow-hosts=127.0.0.1,::1``.
This is the structural fix for the 862s-stall class: a credential leaked
from .env can no longer turn a test into a live API call — the connect
dies instantly instead of retrying against a real endpoint. These tests
prove the guard is armed (a silently-disabled guard would pass the suite
green while protecting nothing). pytest-socket rejects the address BEFORE
dialing, so no packet ever leaves the machine.
"""

from __future__ import annotations

import socket

import pytest
from pytest_socket import SocketConnectBlockedError


def test_external_connections_are_blocked():
    # Literal IP — avoids DNS, which pytest-socket does not intercept.
    with pytest.raises(SocketConnectBlockedError):
        socket.create_connection(("93.184.216.34", 443), timeout=1)


def test_loopback_stays_allowed():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        client = socket.create_connection(server.getsockname(), timeout=2)
        client.close()
    finally:
        server.close()
