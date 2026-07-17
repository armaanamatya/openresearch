"""THE headline exploit test: /proc/self/environ of a spawned run subprocess is credential-free.

Threat model (``docs/design/rlm-pivot-brief.md`` §7): the run subprocess is the RLM
orchestrator — root-MODEL-written Python is ``exec``'d inside it, ``rlm``'s
``_SAFE_BUILTINS`` keeps ``__import__``/``open``, and the paper steering the root is an
arbitrary uploaded PDF (attacker-influenceable). ``execve`` freezes the child's ``envp``
into a kernel snapshot that ``open("/proc/self/environ").read()`` hands straight back —
a region **no in-process ``os.environ`` scrub can touch**. So the run subprocess must be
SPAWNED credential-free, taking its keys over an inherited pipe instead.

This test spawns a REAL subprocess through the exact production spawn functions
(``credential_vault.split_spawn_env`` + ``HANDOFF_FD_ENV`` + ``write_handoff`` +
``_credential_handoff_preamble`` + ``receive_handoff``) and, from inside it, runs the
attack — the same ``open("/proc/self/environ")`` a prompt-injected root would write.

  BEFORE (credentials in ``env=``): every sentinel is harvested from ``/proc``.
  AFTER  (the hardened spawn):      ZERO sentinels in ``/proc`` — and the child still
                                    holds them in ``os.environ`` (auth unbroken).

Every credential value here is a fabricated sentinel. Never a real key.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from backend.agents.runtime import credential_vault as vault
from backend.services.events.live_runs import _credential_handoff_preamble

_SENTINELS = {
    "ANTHROPIC_API_KEY": "sentinel-anthropic-e2e-not-real",
    "OPENAI_API_KEY": "sentinel-openai-e2e-not-real",
    "OPENRESEARCH_RUNPOD_API_KEY": "sentinel-runpod-e2e-not-real",
}
_VALUES = sorted(_SENTINELS.values())

# The attacker's payload: harvest sentinel VALUES straight from the kernel's execve
# snapshot, exactly as injected REPL code would, and separately report os.environ so
# we can prove auth still works after the handoff.
_HOSTILE = f"""
import json, os
try:
    with open("/proc/self/environ", "rb") as fh:
        raw = fh.read()
except OSError:
    raw = b""
wanted = {_VALUES!r}
proc_hits = sorted(v for v in wanted if v.encode() in raw)
env_hits = sorted(v for v in wanted if v in set(os.environ.values()))
print("RESULT " + json.dumps({{"proc": proc_hits, "env": env_hits}}))
"""


@pytest.fixture(autouse=True)
def _always_disarm():
    yield
    vault.disarm()


def _run_child(script: str, env: dict, *, blob: dict | None) -> dict:
    """Spawn a child EXACTLY as live_runs._start_python_run does and return its RESULT dict."""
    stdin_target = subprocess.PIPE if blob is not None else subprocess.DEVNULL
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        env=env,
        stdin=stdin_target,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if blob is not None and proc.stdin is not None:
        vault.write_handoff(proc.stdin, blob)  # writes the blob + closes the pipe
        proc.stdin = None  # detach so communicate() (test-only) won't reflush a closed fd
    out_b, err_b = proc.communicate(timeout=60)
    out, err = out_b.decode(), err_b.decode()
    line = next((ln for ln in out.splitlines() if ln.startswith("RESULT ")), None)
    assert line, f"child produced no RESULT\nstdout={out!r}\nstderr={err!r}"
    return json.loads(line[len("RESULT "):])


def test_before_the_fix_proc_environ_leaks_every_credential(monkeypatch):
    """Precondition / the vector we are closing: keys in env= are readable via /proc."""
    import os

    env = {**os.environ, **_SENTINELS}
    result = _run_child(_HOSTILE, env, blob=None)
    assert sorted(result["proc"]) == _VALUES, (
        "BEFORE the fix, /proc/self/environ must disclose the spawned-in credentials"
    )


def test_hardened_spawn_leaves_zero_credentials_in_proc_environ(monkeypatch):
    """THE headline invariant: the hardened spawn puts NOTHING in the execve snapshot.

    Drives the real production spawn functions. After: 0 credential values in
    /proc/self/environ, yet all 3 present in os.environ (the orchestrator still
    authenticates on every backend).
    """
    import os

    full_env = {**os.environ, **_SENTINELS}
    clean_env, blob = vault.split_spawn_env(full_env)
    clean_env[vault.HANDOFF_FD_ENV] = str(vault.HANDOFF_FD)
    child_script = _credential_handoff_preamble() + "\n" + _HOSTILE

    result = _run_child(child_script, clean_env, blob=blob)

    assert result["proc"] == [], (
        f"credentials STILL readable via /proc/self/environ: {result['proc']} — the "
        "execve snapshot vector is NOT closed"
    )
    assert sorted(result["env"]) == _VALUES, (
        f"the orchestrator lost its credentials (auth broken): only {result['env']} in "
        "os.environ; the handoff must restore all three"
    )


def test_hardened_spawn_env_dict_carries_no_credential_names(monkeypatch):
    """The env dict handed to Popen must not contain any managed credential NAME at all.

    (Not just empty values — the name itself must be absent, so /proc has nothing to show.)
    """
    import os

    full_env = {**os.environ, **_SENTINELS, "OPENRESEARCH_GPU_MODE": "auto"}
    clean_env, _blob = vault.split_spawn_env(full_env)
    clean_env[vault.HANDOFF_FD_ENV] = str(vault.HANDOFF_FD)

    for name in vault.CREDENTIAL_ENV_VARS:
        assert name not in clean_env, f"{name} must never enter the spawn env"
    # The fd number is not a secret and legitimately rides the env.
    assert clean_env[vault.HANDOFF_FD_ENV] == str(vault.HANDOFF_FD)
    assert clean_env["OPENRESEARCH_GPU_MODE"] == "auto", "non-secret run knobs must survive"


def test_disabled_vault_falls_back_to_the_plain_env_spawn(monkeypatch):
    """Escape hatch OPENRESEARCH_CREDENTIAL_VAULT=0: byte-identical to the pre-vault spawn.

    is_enabled() is False, so _start_python_run keeps stdin=DEVNULL and the full env — the
    preamble is inert (guarded on the fd var, which is never set). We assert the preamble
    no-ops here; the live_runs branch that keeps the plain env is covered by is_enabled().
    """
    import os

    monkeypatch.setenv("OPENRESEARCH_CREDENTIAL_VAULT", "0")
    assert vault.is_enabled() is False
    # With the fd var unset, the preamble body must not run receive_handoff at all.
    env = {**os.environ, **_SENTINELS}  # no HANDOFF_FD_ENV
    result = _run_child(_credential_handoff_preamble() + "\n" + _HOSTILE, env, blob=None)
    # Preamble is inert → this is exactly the BEFORE behaviour (keys in env, visible in /proc).
    assert sorted(result["proc"]) == _VALUES
