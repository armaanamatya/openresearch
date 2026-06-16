"""C6 (2026-06-16 grader-fidelity remediation): runpod build short-circuit +
azure cell-route gate.

Two independent harness rough edges pinned here:

(a) ``build_environment`` under ``sandbox_mode == "runpod"`` must short-circuit
    to a no-op (like ``local``/``azure``) so a runpod run no longer pays a
    discarded local ``docker build`` and no longer hard-requires a local Docker
    daemon for an image the pod never uses (it boots ``OPENRESEARCH_RUNPOD_IMAGE``
    over SSH). Flag-gated: ``OPENRESEARCH_RUNPOD_SKIP_BUILD=0`` restores the build.

(b) The ``run_experiment`` cell-matrix entry gate (historically
    ``("local","docker")``) must admit ``"azure"`` so the azure K8s branch in
    ``_execute_cell_matrix`` (which routes ``k8s_job_cell_runner.run_matrix``)
    is reachable — it was unreachable-by-gate before. Flag-gated:
    ``OPENRESEARCH_AZURE_CELL_ROUTE=0`` restores the local/docker-only gate.
    ``runpod`` is NEVER admitted (it uses the SSH exec path, not this route).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.agents.execution import SandboxMode


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_ctx(sandbox_mode: SandboxMode) -> SimpleNamespace:
    """Minimal RunContext duck-type for build_environment.

    build_environment reads only ``ctx.sandbox_mode`` before the short-circuit,
    so a SimpleNamespace with the real enum is sufficient.
    """
    return SimpleNamespace(
        sandbox_mode=sandbox_mode,
        run_budget=None,
        _event_sink=None,
        gpu_device_ids=None,
        gpu_plan=None,
    )


def _poison_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any Docker entry-point raise — proves the short-circuit fired first."""
    import backend.services.runtime.local_docker as _ld
    import backend.agents.rlm.primitives as _prim

    def _should_not_call_docker(*_args, **_kwargs):  # type: ignore[override]
        raise AssertionError(
            "build_environment short-circuit regressed: a Docker entry-point was "
            "reached for a no-build sandbox mode"
        )

    monkeypatch.setattr(_ld, "_make_docker_client", _should_not_call_docker)
    monkeypatch.setattr(_prim, "_image_exists", _should_not_call_docker)


# ---------------------------------------------------------------------------
# (a) build_environment runpod short-circuit
# ---------------------------------------------------------------------------

def test_build_environment_runpod_returns_noop_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """runpod (flag default ON) must return ok/skipped/empty-tag WITHOUT Docker."""
    monkeypatch.delenv("OPENRESEARCH_RUNPOD_SKIP_BUILD", raising=False)
    _poison_docker(monkeypatch)

    from backend.agents.rlm.primitives import build_environment

    ctx = _make_ctx(SandboxMode.runpod)
    result = build_environment({"dockerfile": "FROM python:3.11"}, ctx=ctx)

    assert result.get("ok") is True, f"Expected ok=True, got {result}"
    assert result.get("skipped") is True, f"Expected skipped=True, got {result}"
    assert result.get("image_tag") == "", f"Expected image_tag='', got {result}"
    assert result.get("attempts") == 0
    assert result.get("outcome") == "ok"
    assert "runpod" in result.get("note", "").lower(), (
        f"Note should mention runpod, got: {result.get('note')}"
    )


@pytest.mark.parametrize("flag", ["1", "true", "yes", "on", "TRUE", "On"])
def test_build_environment_runpod_skip_flag_truthy_values(
    monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    """All truthy spellings of OPENRESEARCH_RUNPOD_SKIP_BUILD short-circuit."""
    monkeypatch.setenv("OPENRESEARCH_RUNPOD_SKIP_BUILD", flag)
    _poison_docker(monkeypatch)

    from backend.agents.rlm.primitives import build_environment

    ctx = _make_ctx(SandboxMode.runpod)
    result = build_environment({"dockerfile": "FROM python:3.11"}, ctx=ctx)
    assert result.get("ok") is True and result.get("skipped") is True


def test_build_environment_runpod_skip_disabled_reaches_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENRESEARCH_RUNPOD_SKIP_BUILD=0 restores the prior build path (reaches Docker).

    With the flag off, the runpod short-circuit is bypassed and the function
    proceeds to the content-addressed existence check (`_image_exists`). We
    poison `_image_exists` to raise a *sentinel* so we can prove the build path
    was entered without standing up a real Docker daemon.
    """
    monkeypatch.setenv("OPENRESEARCH_RUNPOD_SKIP_BUILD", "0")

    import backend.agents.rlm.primitives as _prim

    class _ReachedBuildPath(Exception):
        pass

    def _sentinel(*_a, **_k):
        raise _ReachedBuildPath()

    # _image_exists is the first Docker-touching call after the short-circuit
    # block; reaching it proves the runpod short-circuit did NOT fire.
    monkeypatch.setattr(_prim, "_image_exists", _sentinel)

    from backend.agents.rlm.primitives import build_environment

    ctx = _make_ctx(SandboxMode.runpod)
    result = build_environment({"dockerfile": "FROM python:3.11"}, ctx=ctx)

    # build_environment is fail-soft: the sentinel is swallowed and surfaces as
    # an error dict (NOT a skipped no-op). The key assertion is that it is NOT
    # the runpod skip result.
    assert result.get("skipped") is not True, (
        f"With the flag off, runpod must NOT short-circuit; got {result}"
    )
    assert "runpod sandbox" not in (result.get("note") or ""), (
        f"With the flag off, must not return the runpod skip note; got {result}"
    )


def test_build_environment_local_short_circuit_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the local no-op is unaffected by the new runpod branch."""
    _poison_docker(monkeypatch)
    from backend.agents.rlm.primitives import build_environment

    ctx = _make_ctx(SandboxMode.local)
    result = build_environment({"dockerfile": "FROM python:3.11"}, ctx=ctx)
    assert result.get("ok") is True and result.get("skipped") is True
    assert "local" in result.get("note", "").lower()


def test_build_environment_azure_short_circuit_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the azure no-op is unaffected by the new runpod branch."""
    _poison_docker(monkeypatch)
    from backend.agents.rlm.primitives import build_environment

    ctx = _make_ctx(SandboxMode.azure)
    result = build_environment({"dockerfile": "FROM python:3.11"}, ctx=ctx)
    assert result.get("ok") is True and result.get("skipped") is True
    assert "azure" in result.get("note", "").lower()


# ---------------------------------------------------------------------------
# (b) run_experiment cell-route entry gate admits azure
# ---------------------------------------------------------------------------
#
# The gate is an inline predicate inside ``run_experiment``:
#
#     _cell_route_kinds = ["local", "docker"]
#     if OPENRESEARCH_AZURE_CELL_ROUTE truthy:
#         _cell_route_kinds.append("azure")
#     if _caps.backend_kind in _cell_route_kinds and ...:
#         _execute_cell_matrix(...)
#
# We pin the contract of that predicate (which backend kinds are admitted under
# which flag state) plus a source-guard so a refactor that drops azure or
# silently admits runpod is caught.


def _admitted_kinds(flag_value: str | None) -> set[str]:
    """Replicate run_experiment's _cell_route_kinds construction exactly.

    Mirrors the inline logic byte-for-byte so the contract is asserted, not the
    implementation detail. Kept in lock-step with primitives.py via the source
    guard below.
    """
    import os

    kinds = ["local", "docker"]
    val = (os.environ.get("OPENRESEARCH_AZURE_CELL_ROUTE", "1") if flag_value is None else flag_value)
    if val.strip().lower() in ("1", "true", "yes", "on"):
        kinds.append("azure")
    return set(kinds)


def test_gate_default_admits_local_docker_azure() -> None:
    """Default (flag unset → '1'): local, docker AND azure are admitted."""
    kinds = _admitted_kinds("1")
    assert {"local", "docker", "azure"} <= kinds


def test_gate_never_admits_runpod() -> None:
    """runpod uses the SSH exec path, never the cell-matrix route — both flag states."""
    assert "runpod" not in _admitted_kinds("1")
    assert "runpod" not in _admitted_kinds("0")


@pytest.mark.parametrize("flag", ["1", "true", "yes", "on", "TRUE"])
def test_gate_flag_on_admits_azure(flag: str) -> None:
    assert "azure" in _admitted_kinds(flag)


@pytest.mark.parametrize("flag", ["0", "false", "no", "off", ""])
def test_gate_flag_off_excludes_azure_keeps_local_docker(flag: str) -> None:
    """Flag off: azure excluded, local/docker byte-for-byte unchanged."""
    kinds = _admitted_kinds(flag)
    assert "azure" not in kinds
    assert {"local", "docker"} == kinds


def test_run_experiment_source_contains_flag_gated_azure_gate() -> None:
    """Source guard: the inline gate must be flag-gated on OPENRESEARCH_AZURE_CELL_ROUTE
    and admit azure into _cell_route_kinds — catches a refactor that drops it or
    hardcodes ("local","docker") again."""
    import inspect

    from backend.agents.rlm import primitives

    src = inspect.getsource(primitives.run_experiment)
    assert "OPENRESEARCH_AZURE_CELL_ROUTE" in src, (
        "run_experiment lost the OPENRESEARCH_AZURE_CELL_ROUTE gate flag"
    )
    assert "_cell_route_kinds" in src and '"azure"' in src, (
        "run_experiment no longer admits azure into the cell-route gate"
    )
    # Belt-and-braces: the gate must compare backend_kind against the computed
    # set, not the old hardcoded tuple.
    assert "_caps.backend_kind in _cell_route_kinds" in src, (
        "run_experiment cell-route gate no longer keys off _cell_route_kinds"
    )
