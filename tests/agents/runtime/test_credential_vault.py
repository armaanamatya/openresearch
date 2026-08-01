"""Credential vault — the REPL-boundary blast-radius guard.

Threat model (``docs/design/rlm-pivot-brief.md`` §7): ``RLM(environment="local")``
is mandatory, so root-model-written Python is ``exec``'d in the orchestrator's own
process, and ``rlm``'s ``_SAFE_BUILTINS`` retains ``__import__`` — root code can
read ``os.environ``. The paper steering the root is an arbitrary uploaded PDF, i.e.
attacker-influenceable. Until the vault landed, one line of prompt-injected Python
harvested every API key the process held.

These tests pin the mitigation:
  * no credential env var is readable at the REPL boundary;
  * the guard fails LOUDLY (and value-free) if a key leaks back in;
  * every auth surface still works — root (oauth / api-key / foundry / openai),
    the claude-agent-sdk executor subprocess, and the primitives;
  * the RunPod pod env still carries NO ANTHROPIC_API_KEY (invariant I4).

NOTE: every credential value in this file is a fake sentinel. Never a real key.
"""

from __future__ import annotations

import json
import os

import pytest

from backend.agents.runtime import credential_vault as vault

# Fake sentinels — deliberately not shaped like real keys.
_FAKE_ANTHROPIC = "sentinel-anthropic-not-a-real-key"
_FAKE_OPENAI = "sentinel-openai-not-a-real-key"
_FAKE_FOUNDRY = "sentinel-foundry-not-a-real-key"
_FAKE_OPENROUTER = "sentinel-openrouter-not-a-real-key"


@pytest.fixture(autouse=True)
def _always_disarm():
    """The vault is module-global state; never let it leak across tests."""
    yield
    vault.disarm()


@pytest.fixture
def full_credentials(monkeypatch):
    """A process env holding one of every credential family."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_ANTHROPIC)
    monkeypatch.setenv("OPENAI_API_KEY", _FAKE_OPENAI)
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", _FAKE_FOUNDRY)
    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_OPENROUTER)


# ---------------------------------------------------------------------------
# 1. The REPL boundary is credential-free.
# ---------------------------------------------------------------------------


def test_no_credential_env_var_is_present_at_the_repl_boundary(full_credentials):
    """THE core invariant: root-written REPL code sees no API key in os.environ."""
    assert vault.leaked_names(), "precondition: credentials must start in os.environ"

    with vault.armed_vault():
        # This is exactly what injected REPL code would evaluate.
        harvested = {
            name: os.environ.get(name)
            for name in vault.CREDENTIAL_ENV_VARS
            if os.environ.get(name, "").strip()
        }
        assert harvested == {}, f"credentials readable from the REPL: {sorted(harvested)}"
        vault.assert_repl_boundary_clean()  # must not raise


def test_every_managed_name_is_scrubbed_not_just_the_bare_ones(monkeypatch):
    """The OPENRESEARCH_*/REPROLAB_* aliases hold live keys too (config.py bridges them)."""
    monkeypatch.setenv("OPENRESEARCH_ANTHROPIC_API_KEY", _FAKE_ANTHROPIC)
    monkeypatch.setenv("REPROLAB_OPENAI_API_KEY", _FAKE_OPENAI)

    with vault.armed_vault():
        assert "OPENRESEARCH_ANTHROPIC_API_KEY" not in os.environ
        assert "REPROLAB_OPENAI_API_KEY" not in os.environ


def test_disarm_restores_the_environment_exactly(full_credentials):
    before = {n: os.environ.get(n) for n in vault.CREDENTIAL_ENV_VARS}

    with vault.armed_vault():
        pass

    after = {n: os.environ.get(n) for n in vault.CREDENTIAL_ENV_VARS}
    assert after == before, "post-completion env must be byte-identical (finalize/report run here)"


# ---------------------------------------------------------------------------
# 2. The regression guard fails loudly.
# ---------------------------------------------------------------------------


def test_guard_fails_loudly_when_a_key_leaks_back_in(full_credentials):
    """The guard that keeps this fixed: a future os.environ credential bridge must fire it."""
    vault.arm()
    # Simulate a new upstream bridge doing `os.environ["ANTHROPIC_API_KEY"] = key`.
    os.environ["ANTHROPIC_API_KEY"] = _FAKE_ANTHROPIC

    with pytest.raises(vault.CredentialLeak) as exc:
        vault.assert_repl_boundary_clean()

    assert "ANTHROPIC_API_KEY" in str(exc.value), "the guard must name the leaking var"


def test_guard_never_discloses_a_credential_value(full_credentials):
    vault.arm()
    os.environ["OPENAI_API_KEY"] = _FAKE_OPENAI

    with pytest.raises(vault.CredentialLeak) as exc:
        vault.assert_repl_boundary_clean()

    assert _FAKE_OPENAI not in str(exc.value), "the guard must report NAMES ONLY, never a value"


def test_an_empty_credential_var_is_not_a_leak(monkeypatch):
    """`ANTHROPIC_API_KEY=` is the documented local-dev pattern for forcing OAuth."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    vault.arm()
    os.environ["ANTHROPIC_API_KEY"] = ""
    vault.assert_repl_boundary_clean()  # empty carries no secret -> not a leak


def test_guard_is_a_noop_when_the_operator_disables_the_vault(monkeypatch, full_credentials):
    """Escape hatch: OPENRESEARCH_CREDENTIAL_VAULT=0 (emergency rollback only)."""
    monkeypatch.setenv("OPENRESEARCH_CREDENTIAL_VAULT", "0")
    assert vault.is_enabled() is False

    vault.arm()
    assert os.environ.get("ANTHROPIC_API_KEY") == _FAKE_ANTHROPIC  # not scrubbed
    vault.assert_repl_boundary_clean()  # and the guard stands down


# ---------------------------------------------------------------------------
# 3. Primitives still get their credentials (set -> use -> immediately restore).
# ---------------------------------------------------------------------------


def test_primitive_call_re_exposes_then_rescrubs(full_credentials):
    """Lazy consumers (grader/azure/openrouter) read os.environ from INSIDE a primitive."""
    seen: dict[str, str | None] = {}

    with vault.armed_vault():
        assert os.environ.get("ANTHROPIC_API_KEY") is None
        with vault.exposed():
            seen["anthropic"] = os.environ.get("ANTHROPIC_API_KEY")
            seen["openrouter"] = os.environ.get("OPENROUTER_API_KEY")
        assert os.environ.get("ANTHROPIC_API_KEY") is None, "must re-scrub on primitive exit"

    assert seen["anthropic"] == _FAKE_ANTHROPIC
    assert seen["openrouter"] == _FAKE_OPENROUTER


def test_consumer_repollution_during_a_primitive_cannot_escape(full_credentials):
    """factory.configure_openai_agents_sdk_credentials writes OPENAI_API_KEY back into
    os.environ. That write must not survive the primitive it happened in."""
    with vault.armed_vault():
        with vault.exposed():
            os.environ["OPENAI_API_KEY"] = _FAKE_OPENAI  # the bridge re-polluting
        assert "OPENAI_API_KEY" not in os.environ, "re-pollution leaked into the REPL window"
        vault.assert_repl_boundary_clean()


def test_nested_primitive_calls_are_reentrant(full_credentials):
    with vault.armed_vault():
        with vault.exposed():
            with vault.exposed():
                assert os.environ.get("ANTHROPIC_API_KEY") == _FAKE_ANTHROPIC
            # inner exit must NOT scrub while the outer primitive is still running
            assert os.environ.get("ANTHROPIC_API_KEY") == _FAKE_ANTHROPIC
        assert "ANTHROPIC_API_KEY" not in os.environ


def test_exposed_is_a_noop_when_not_armed(full_credentials):
    with vault.exposed():
        assert os.environ.get("ANTHROPIC_API_KEY") == _FAKE_ANTHROPIC


# ---------------------------------------------------------------------------
# 4. run.py's tool wrapper — the REPL-facing shape.
# ---------------------------------------------------------------------------


def test_credential_scoped_tools_exposes_only_inside_the_tool(full_credentials):
    from backend.agents.rlm.run import _credential_scoped_tools

    observed: dict[str, str | None] = {}

    def run_experiment(code_path: str, env_id: str = "x") -> dict:
        observed["inside"] = os.environ.get("OPENROUTER_API_KEY")
        return {"ok": True, "code_path": code_path, "env_id": env_id}

    tools = _credential_scoped_tools(
        {
            "run_experiment": {"tool": run_experiment, "description": "d"},
            "rubric_spec": {"leaves": []},  # non-callable data -> passed through
        }
    )

    with vault.armed_vault():
        observed["outside"] = os.environ.get("OPENROUTER_API_KEY")
        result = tools["run_experiment"]["tool"]("code/", env_id="e1")

    assert observed["outside"] is None, "the REPL must not see the key between primitives"
    assert observed["inside"] == _FAKE_OPENROUTER, "the primitive must still get the key"
    assert result == {"ok": True, "code_path": "code/", "env_id": "e1"}, "args must pass through"
    assert tools["rubric_spec"] == {"leaves": []}, "non-callable tools must be untouched"


def test_credential_scoped_tools_preserves_the_tool_name():
    """rlm binds tools into the REPL globals by name; the wrapper must stay transparent."""
    from backend.agents.rlm.run import _credential_scoped_tools

    def implement_baseline(plan: dict) -> dict:
        return plan

    tools = _credential_scoped_tools({"implement_baseline": {"tool": implement_baseline}})
    assert tools["implement_baseline"]["tool"].__name__ == "implement_baseline"


# ---------------------------------------------------------------------------
# 5. The root model still authenticates — every backend.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "env_var", "fake"),
    [
        ("gpt-5", "OPENAI_API_KEY", _FAKE_OPENAI),
        ("claude", "ANTHROPIC_API_KEY", _FAKE_ANTHROPIC),
        ("opus-foundry", "AZURE_FOUNDRY_API_KEY", _FAKE_FOUNDRY),
        ("sonnet-foundry", "AZURE_FOUNDRY_API_KEY", _FAKE_FOUNDRY),
    ],
)
def test_root_model_key_is_bound_before_the_boundary_and_survives_the_scrub(
    monkeypatch, model, env_var, fake
):
    """The root's key is injected into backend_kwargs at RESOLVE time (models._inject_api_key),
    which happens before rlm.completion(). So the root authenticates from the client it already
    holds — the scrub cannot starve it."""
    from backend.agents.rlm.models import resolve_root_model

    monkeypatch.setenv(env_var, fake)
    root = resolve_root_model(model)
    assert root.backend_kwargs.get("api_key") == fake, "key must be bound at resolve time"

    with vault.armed_vault():
        # The REPL cannot see it...
        assert env_var not in os.environ
        # ...but the already-resolved root client still carries it.
        assert root.backend_kwargs.get("api_key") == fake


def test_oauth_root_needs_no_api_key_at_all(monkeypatch):
    """claude-oauth authenticates via the `claude` CLI subscription (~/.claude), not env."""
    from backend.agents.rlm import models

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.has_provider_credentials", lambda _p: True
    )

    root = models.resolve_root_model("claude-oauth")
    assert root.rlm_backend == "anthropic-oauth"
    assert not root.backend_kwargs.get("api_key"), "OAuth root must not need an env key"

    with vault.armed_vault():
        vault.assert_repl_boundary_clean()  # nothing to scrub, nothing to break


# ---------------------------------------------------------------------------
# 6. The claude-agent-sdk executor subprocess still receives its credentials.
# ---------------------------------------------------------------------------


def test_executor_subprocess_gets_the_key_explicitly_while_armed(full_credentials):
    """Requirement (b): pass the key into ClaudeAgentOptions(env=...) rather than relying on
    the parent's os.environ, which the vault has (correctly) emptied."""
    from backend.agents.runtime.claude_runtime import _subprocess_env

    with vault.armed_vault():
        assert "ANTHROPIC_API_KEY" not in os.environ  # parent env is clean...
        env = _subprocess_env(None)
        assert env["ANTHROPIC_API_KEY"] == _FAKE_ANTHROPIC  # ...subprocess still authenticates


def test_executor_oauth_path_is_never_given_an_api_key(monkeypatch):
    """A no-credit ANTHROPIC_API_KEY does NOT fall back to OAuth — it 400s. So when there is
    no key, none must be injected: the SDK then uses the `claude` CLI subscription login."""
    from backend.agents.runtime.claude_runtime import _subprocess_env

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with vault.armed_vault():
        assert _subprocess_env(None) == {}, "OAuth path must stay key-free"


def test_executor_foundry_subprocess_env_is_not_overwritten(full_credentials):
    """Anthropic-on-Foundry attaches its OWN per-subprocess key + base_url. The vault must
    not clobber it with the plain Anthropic key (that would hijack the Foundry route)."""
    from backend.agents.runtime.claude_runtime import _subprocess_env

    foundry_env = {
        "ANTHROPIC_BASE_URL": "https://example.services.ai.azure.com/anthropic",
        "ANTHROPIC_API_KEY": _FAKE_FOUNDRY,
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    }

    with vault.armed_vault():
        env = _subprocess_env(foundry_env)

    assert env["ANTHROPIC_API_KEY"] == _FAKE_FOUNDRY, "Foundry key must win"
    assert env["ANTHROPIC_BASE_URL"].endswith("/anthropic")
    assert env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] == "1"


def test_executor_env_is_unchanged_when_the_vault_is_not_armed(full_credentials):
    """Outside the REPL window the accessor is exactly os.environ — byte-identical behaviour."""
    from backend.agents.runtime.claude_runtime import _subprocess_env

    assert _subprocess_env(None)["ANTHROPIC_API_KEY"] == _FAKE_ANTHROPIC


# ---------------------------------------------------------------------------
# 8. The wall-clock watchdog / SIGTERM hard-stop fires DURING the REPL window.
# ---------------------------------------------------------------------------


def test_hard_stop_regrade_runs_inside_an_exposure_window():
    """The watchdog thread and the SIGTERM handler call regrade_for_hard_stop at an ARBITRARY
    moment — including mid-REPL, when the keys are scrubbed. It lazily builds a grader
    transport (AnthropicMessagesClient with api_key=None -> the SDK's own env lookup) whenever
    OPENRESEARCH_GRADER_BACKEND is set, so it MUST hold an exposure window or it ships an
    un-regraded report."""
    import inspect

    from backend.agents.rlm import run as run_mod

    src = inspect.getsource(run_mod._hard_stop_with_report)
    assert "regrade_for_hard_stop" in src, "test is stale — the regrade call moved"
    assert "exposed()" in src, (
        "regrade_for_hard_stop must run inside credential_vault.exposed(): it executes on the "
        "watchdog/SIGTERM thread while the REPL window has os.environ scrubbed"
    )


def test_exposure_window_is_thread_safe(full_credentials):
    """The watchdog opens its window from a DIFFERENT thread than the REPL worker."""
    import threading

    seen: dict[str, object] = {}
    started = threading.Event()
    release = threading.Event()

    def _watchdog_like() -> None:
        with vault.exposed():
            seen["inside_other_thread"] = os.environ.get("ANTHROPIC_API_KEY")
            started.set()
            release.wait(timeout=5)

    with vault.armed_vault():
        t = threading.Thread(target=_watchdog_like, daemon=True)
        t.start()
        assert started.wait(timeout=5), "watchdog thread never opened its window"
        release.set()
        t.join(timeout=5)
        assert not t.is_alive()
        # Once the other thread's window closes, the REPL must be clean again.
        assert "ANTHROPIC_API_KEY" not in os.environ

    assert seen["inside_other_thread"] == _FAKE_ANTHROPIC


# ---------------------------------------------------------------------------
# 9. The spawn handoff — keep credentials out of the child's execve snapshot.
# ---------------------------------------------------------------------------
# WHY: execve freezes envp into a kernel snapshot that /proc/self/environ reads
# back verbatim; no in-process os.environ scrub (arm) can touch it. So the run
# subprocess must be SPAWNED credential-free, taking its keys over an inherited
# pipe. These pin the parent-side split + child-side intake + the fail-closed
# /proc guard. The end-to-end exploit proof lives in
# tests/services/events/test_live_runs_proc_environ_handoff.py.


def test_split_spawn_env_lifts_every_managed_name_out(full_credentials):
    """Parent side: the child env goes over credential-free; the blob carries the keys."""
    env = {
        **{n: os.environ[n] for n in vault.CREDENTIAL_ENV_VARS if n in os.environ},
        "PATH": "/usr/bin",
        "OPENRESEARCH_GPU_MODE": "auto",  # non-secret run knob — must stay in the env
    }
    clean, blob = vault.split_spawn_env(env)

    assert blob["ANTHROPIC_API_KEY"] == _FAKE_ANTHROPIC
    assert blob["OPENAI_API_KEY"] == _FAKE_OPENAI
    for name in vault.CREDENTIAL_ENV_VARS:
        assert name not in clean, f"{name} must NOT ride the spawn env"
    assert clean["PATH"] == "/usr/bin"
    assert clean["OPENRESEARCH_GPU_MODE"] == "auto", "non-secret knobs stay in the env"


def test_split_spawn_env_uses_the_same_list_as_arm():
    """No second credential list to drift: split keys off CREDENTIAL_ENV_VARS itself."""
    env = {name: f"val-{name}" for name in vault.CREDENTIAL_ENV_VARS}
    clean, blob = vault.split_spawn_env(env)
    assert clean == {}
    assert set(blob) == set(vault.CREDENTIAL_ENV_VARS)


def test_split_spawn_env_round_trips_an_empty_forcing_value():
    """`ANTHROPIC_API_KEY=` (set, empty) forces OAuth — it must survive the handoff, not vanish."""
    clean, blob = vault.split_spawn_env({"ANTHROPIC_API_KEY": "", "PATH": "/usr/bin"})
    assert "ANTHROPIC_API_KEY" in blob and blob["ANTHROPIC_API_KEY"] == ""
    assert "ANTHROPIC_API_KEY" not in clean


def test_handoff_round_trip_restores_os_environ(monkeypatch):
    """Child side: receive_handoff reads the pipe and restores the keys to os.environ."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    blob = {"ANTHROPIC_API_KEY": _FAKE_ANTHROPIC, "OPENAI_API_KEY": _FAKE_OPENAI}

    read_fd, write_fd = os.pipe()
    vault.write_handoff(os.fdopen(write_fd, "wb"), blob)
    monkeypatch.setenv(vault.HANDOFF_FD_ENV, str(read_fd))

    count = vault.receive_handoff()

    assert count == 2
    assert os.environ["ANTHROPIC_API_KEY"] == _FAKE_ANTHROPIC
    assert os.environ["OPENAI_API_KEY"] == _FAKE_OPENAI
    # The fd-number env var must be consumed so the child's own children can't re-read it.
    assert vault.HANDOFF_FD_ENV not in os.environ


def test_handoff_is_a_legal_noop_when_there_is_nothing_to_hand(monkeypatch):
    """A zero-credential deployment (pure OAuth subscription) hands nothing and stays fine."""
    monkeypatch.delenv(vault.HANDOFF_FD_ENV, raising=False)
    assert vault.receive_handoff() == 0


def test_handoff_rejects_an_unmanaged_name(monkeypatch):
    """Fail-closed: a blob carrying a name the vault does not manage is a wiring bug, not silent."""
    read_fd, write_fd = os.pipe()
    os.write(write_fd, json.dumps({"NOT_A_MANAGED_KEY": "x"}).encode())
    os.close(write_fd)
    monkeypatch.setenv(vault.HANDOFF_FD_ENV, str(read_fd))

    with pytest.raises(vault.CredentialHandoffError) as exc:
        vault.receive_handoff()
    assert "NOT_A_MANAGED_KEY" in str(exc.value)


def test_handoff_rejects_malformed_json(monkeypatch):
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"this is not json")
    os.close(write_fd)
    monkeypatch.setenv(vault.HANDOFF_FD_ENV, str(read_fd))

    with pytest.raises(vault.CredentialHandoffError):
        vault.receive_handoff()


@pytest.mark.skipif(
    not os.path.exists("/proc/self/environ"),
    reason="/proc/self/environ is Linux-only; the execve-snapshot disclosure vector "
    "(and this guard test) does not exist on macOS/Windows",
)
def test_proc_environ_guard_detects_a_credential_in_the_exec_snapshot():
    """The headline guard: a credential in THIS process's execve env must be caught.

    We run a real child spawned WITH a sentinel in env= (the vulnerable spawn) and let it
    self-check via assert_proc_environ_clean — which reads its own /proc/self/environ.
    """
    import subprocess
    import sys

    child = (
        "import sys; sys.path.insert(0, %r)\n" % os.getcwd()
        + "from backend.agents.runtime import credential_vault as v\n"
        + "leaked = v.proc_environ_leaked_names()\n"
        + "print('LEAKED', 'ANTHROPIC_API_KEY' in leaked)\n"
        + "raised = False\n"
        + "try:\n"
        + "    v.assert_proc_environ_clean()\n"
        + "except v.CredentialLeak as e:\n"
        + "    raised = True\n"
        + "    assert 'sentinel' not in str(e)\n"  # names only, never the value
        + "print('RAISED', raised)\n"
    )
    env = {**os.environ, "ANTHROPIC_API_KEY": "sentinel-proc-guard-not-real"}
    out = subprocess.run(
        [sys.executable, "-c", child], env=env, capture_output=True, text=True, timeout=60
    ).stdout
    assert "LEAKED True" in out, f"the guard must see the key in /proc/self/environ; got {out!r}"
    assert "RAISED True" in out, f"the guard must fail closed on a leaked exec env; got {out!r}"


def test_proc_environ_guard_is_a_noop_when_the_vault_is_disabled(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_CREDENTIAL_VAULT", "0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_ANTHROPIC)
    vault.assert_proc_environ_clean()  # disabled ⇒ must not raise even with a key in env
