"""Task 14: hermetic backend integration coverage for the autonomous
upload-to-launch flow — composes T1 (``StartRunRequest.autonomous``/
``run_spec`` fields), T2 (request threading), and T3
(``apply_autonomous_profile_override``) the exact way
``FileLiveRunService._start_python_run`` does, proving the full override
chain forces the autonomous profile end-to-end (not just T3 in isolation,
which ``tests/routes/test_autonomous_profile_override.py`` already covers).

``_start_python_run`` itself spawns a real ``subprocess.Popen`` (see
``live_runs.py:892`` and ``tests/routes/test_rerun.py``, which fully
monkeypatches the method out rather than exercising the real subprocess
spawn) — so, per this task's own escape hatch, this file composes the three
override functions directly, in ``_start_python_run``'s own order
(``live_runs.py:840-845``)::

    request = apply_sandbox_override(request, force_sandbox)
    request = apply_provider_override(request, force_provider)
    request = apply_autonomous_profile_override(request)   # applied LAST — wins

CRITICAL CORRECTION vs. the original task-14 brief text: ``autonomous=True``
forces ``sandbox="gcp"``, NOT the literal ``"gke"``. ``StartRunRequest.sandbox``
is typed ``Literal["auto", "docker", "local", "azure", "aws", "gcp"]``
(``live_runs.py:43``) — "gke" is not a member of that Literal at all (the
gke->gcp alias lives only in the separate ``backend.agents.execution.SandboxMode``
enum's ``_missing_`` hook). See ``test_autonomous_profile_override.py``'s
module docstring for the same note. Every assertion below checks "gcp".
"""

from __future__ import annotations

from backend.services.events.live_runs import (
    StartRunRequest,
    apply_autonomous_profile_override,
    apply_provider_override,
    apply_sandbox_override,
)

_AUTONOMOUS_RUN_SPEC = "configs/autonomous_reproduction_run_spec.json"


def _compose(
    request: StartRunRequest,
    *,
    force_sandbox: str = "",
    force_provider: str = "",
) -> StartRunRequest:
    """Mirror ``FileLiveRunService._start_python_run``'s exact override
    composition order: sandbox override, then provider override, then the
    autonomous profile override LAST — so autonomous wins over any
    deployment-level force (see the comment at ``live_runs.py:842-844``)."""
    request = apply_sandbox_override(request, force_sandbox)
    request = apply_provider_override(request, force_provider)
    request = apply_autonomous_profile_override(request)
    return request


# --------------------------------------------------------------------------- #
# autonomous=True forces the profile end-to-end through the full chain
# --------------------------------------------------------------------------- #


def test_autonomous_true_forces_gcp_opus_foundry_and_run_spec():
    """The common case: autonomous=True, no deployment-level force set."""
    request = _compose(StartRunRequest(autonomous=True))
    assert request.sandbox == "gcp"
    assert request.sandbox != "gke"  # "gke" is not a StartRunRequest.sandbox Literal member
    assert request.model == "opus-foundry"
    assert request.run_spec == _AUTONOMOUS_RUN_SPEC


def test_autonomous_true_preserves_explicit_azure_target():
    request = _compose(StartRunRequest(autonomous=True, sandbox="azure"))
    assert request.sandbox == "azure"
    assert request.model == "opus-foundry"
    assert request.run_spec == _AUTONOMOUS_RUN_SPEC


def test_autonomous_wins_over_deployment_forced_sandbox_and_provider():
    """apply_autonomous_profile_override applies LAST in _start_python_run,
    so it must win even when a deployment has forced a different
    sandbox/provider via OPENRESEARCH_FORCE_SANDBOX/_FORCE_LLM_PROVIDER."""
    request = _compose(
        StartRunRequest(autonomous=True, sandbox="azure", provider="anthropic", model="sonnet"),
        force_sandbox="docker",
        force_provider="openai",
    )
    assert request.sandbox == "gcp"
    assert request.model == "opus-foundry"
    assert request.run_spec == _AUTONOMOUS_RUN_SPEC


def test_caller_supplied_run_spec_is_preserved_through_the_full_chain():
    """The profile only fills the run_spec gap when unset — a caller (or an
    earlier stage) that already picked one keeps it, even composed through
    the full sandbox+provider+autonomous chain."""
    request = _compose(
        StartRunRequest(autonomous=True, run_spec="configs/sdar_execute_run_spec.json")
    )
    assert request.run_spec == "configs/sdar_execute_run_spec.json"
    # sandbox/model are still forced — autonomous wins on those two fields
    # regardless of what it does to run_spec.
    assert request.sandbox == "gcp"
    assert request.model == "opus-foundry"


# --------------------------------------------------------------------------- #
# autonomous=False is byte-identical (the override chain's identity path)
# --------------------------------------------------------------------------- #


def test_autonomous_false_is_the_identity_function_through_the_full_chain():
    """With autonomous=False, apply_autonomous_profile_override must not
    touch a request already shaped by the (deployment-level) sandbox/
    provider overrides — the sandbox/model/run_spec are whatever the caller
    (or an earlier override) set; autonomous contributes nothing."""
    request = StartRunRequest(autonomous=False, sandbox="azure", provider="anthropic")
    after_deployment_overrides = apply_provider_override(
        apply_sandbox_override(request, "docker"), "openai"
    )
    result = apply_autonomous_profile_override(after_deployment_overrides)

    assert result is after_deployment_overrides  # identity: autonomous=False changes nothing
    assert result.sandbox == "docker"
    assert result.provider == "openai"
    assert result.model == "sonnet"
    assert result.run_spec is None


def test_autonomous_false_with_no_deployment_overrides_is_byte_identical():
    """The plainest byte-identical case: no deployment force + autonomous
    off leaves the request completely untouched through the full chain
    (same object identity end to end, mirroring the no-op behavior of
    apply_sandbox_override/apply_provider_override on empty force values)."""
    request = StartRunRequest()
    result = _compose(request)
    assert result is request
    assert result.sandbox == "local"  # StartRunRequest's own default (single coherent local), untouched
    assert result.model == "sonnet"
    assert result.run_spec is None
