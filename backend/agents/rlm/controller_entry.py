"""In-Pod entrypoint for the durable controller (WS3).

Runs inside the controller Job Pod: acquires the run's drive-lease, renews it on
a heartbeat cadence (so an expired lease reliably means "controller gone" —
which is what makes :func:`controller_cluster.sweep_orphaned_controllers` safe),
exports the stable fence into the env the campaign reads to fence its cell Jobs,
and drives the campaign to a terminal state. On losing the lease (superseded by
a takeover) it stops, so two controller generations never write concurrently.

The heavy cluster/campaign wiring here executes only inside a real Pod and is
exercised at drill time; :func:`heartbeat_loop` is the pure, unit-tested core.

Design: ``docs/superpowers/specs/2026-07-12-cloud-native-durable-and-cpu-lane-design.md`` §3.6, §7.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def heartbeat_loop(
    *,
    lease: Any,
    token: Any,
    interval_s: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    is_running: Callable[[], bool],
    on_lost: Callable[[], None],
) -> None:
    """Renew ``token`` every ``interval_s`` while ``is_running()`` holds.

    Fail-closed on supersede: if ``lease.renew`` returns ``None`` the lease was
    taken over, so ``on_lost`` is called (the caller stops the campaign) and the
    loop exits. Clock and sleep are injected so the loop is deterministic under
    test — nothing here calls ``time`` directly.
    """
    while is_running():
        sleep(interval_s)
        if not is_running():
            return
        token = lease.renew(token, clock())
        if token is None:
            on_lost()
            return
