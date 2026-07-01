"""Unit tests for the GCP<->Azure provision-time failover selector (Phase 1c, Unit A).

Hermetic: ``availability``/``backend_factory`` are always injected fakes, so no
real cloud SDK or credential is ever touched. Confirms (against the real
``SandboxRuntimeError``/``RuntimeCauseKind`` from
``backend.services.runtime.interface``): failover to the next cloud on a
"backend_unavailable" probe, first-cloud-wins when everyone is up, a raise when
every cloud is down, a real bug (any non-"cloud down" exception) propagating
instead of being treated as failover, and OPENRESEARCH_CLOUD_FAILOVER parsing.
"""

import pytest
from backend.services.runtime.cloud_failover import (
    select_backend_with_failover,
    failover_preference,
    BackendSelection,
)
from backend.services.runtime.interface import SandboxRuntimeError, RuntimeCauseKind


def _down(kind: RuntimeCauseKind = RuntimeCauseKind.backend_unavailable):
    def _raise():
        raise SandboxRuntimeError(kind, "simulated cloud down")

    return _raise


def test_failover_picks_azure_when_gcp_down():
    made = []
    sel = select_backend_with_failover(
        ["gcp", "azure"],
        availability={"gcp": _down(), "azure": lambda: None},
        backend_factory=lambda cloud, **_: made.append(cloud) or f"backend:{cloud}",
    )
    assert isinstance(sel, BackendSelection)
    assert sel.cloud == "azure" and sel.backend == "backend:azure"
    assert made == ["azure"]  # gcp never built (unavailable)
    assert sel.attempts[0][0] == "gcp" and "unavailable" in sel.attempts[0][1].lower()


def test_first_cloud_wins_when_available():
    sel = select_backend_with_failover(
        ["gcp", "azure"],
        availability={"gcp": lambda: None, "azure": lambda: None},
        backend_factory=lambda cloud, **_: f"backend:{cloud}",
    )
    assert sel.cloud == "gcp"


def test_all_clouds_down_raises():
    with pytest.raises(SandboxRuntimeError):
        select_backend_with_failover(
            ["gcp", "azure"],
            availability={"gcp": _down(), "azure": _down()},
            backend_factory=lambda cloud, **_: f"backend:{cloud}",
        )


def test_non_capacity_error_propagates_not_failed_over():
    made = []
    with pytest.raises(RuntimeError):
        select_backend_with_failover(
            ["gcp", "azure"],
            availability={
                "gcp": lambda: (_ for _ in ()).throw(RuntimeError("bug")),
                "azure": lambda: None,
            },
            backend_factory=lambda cloud, **_: made.append(cloud) or f"backend:{cloud}",
        )
    assert made == []  # a real bug is not masked as "cloud down"


def test_preference_parsing(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_CLOUD_FAILOVER", raising=False)
    assert failover_preference() == []
    monkeypatch.setenv("OPENRESEARCH_CLOUD_FAILOVER", "gcp, azure")
    assert failover_preference() == ["gcp", "azure"]
