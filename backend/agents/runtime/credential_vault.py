"""Process-local credential vault — keep API keys out of the REPL's ``os.environ``.

WHY THIS EXISTS — the threat model (``docs/design/rlm-pivot-brief.md`` §7)
-------------------------------------------------------------------------
The RLM root model's Python runs via ``exec`` **in this process**: ``run.py``
constructs ``RLM(..., environment="local")`` and that is *mandatory* (``rlm``'s
``DockerREPL`` drops ``custom_tools``). ``rlm``'s ``LocalREPL._SAFE_BUILTINS``
blocks ``eval``/``exec``/``compile``/``input`` but **retains ``__import__`` and
``open``** — so root-written REPL code can evaluate ``__import__("os").environ``.

The root model is trusted; the *paper that steers it* is **not**. A paper is an
arXiv id or an arbitrary uploaded PDF (deepinvent.ai accepts user uploads), i.e.
attacker-influenceable. A prompt-injected paper can steer the root into writing
hostile REPL code — and until this module existed, one line of injected Python
harvested every LLM and cloud credential the orchestrator process held, because
the credential bridge (``factory.configure_*_credentials``) copies keys from
``.env``/Settings straight into ``os.environ``.

This module removes those credentials from ``os.environ`` for the duration of the
REPL window (``rlm.completion``) and hands them back **only inside a primitive
call** — i.e. only while harness code, not root-written code, is on the stack.

THE SPAWN HANDOFF — why ``os.environ`` alone was not enough
-----------------------------------------------------------
``execve`` freezes ``envp`` into a kernel-held snapshot that ``/proc/self/environ``
reads back verbatim. That snapshot is **immune to every in-process mutation**:
:func:`arm` can empty ``os.environ`` completely and the credentials the process was
*spawned* with are still one ``open("/proc/self/environ").read()`` away. On the UI
upload path — the product path, where the PDF is a stranger's — that made the whole
scrub cosmetic against an attacker who knew where to look.

The fix is to never put them in ``envp`` at all. The parent
(``live_runs._start_python_run``) now:

1. builds the child env exactly as before, then :func:`split_spawn_env` lifts every
   managed name OUT of it into a separate blob;
2. spawns the child with the credential-free env + an inherited **pipe on fd 0**
   (the fd number rides ``OPENRESEARCH_CREDENTIAL_HANDOFF_FD``; a fd number is not
   a secret);
3. writes the blob to that pipe once and closes it (:func:`write_handoff`).

The child calls :func:`receive_handoff` as its very first statement — before any
``backend`` import — which reads the blob into memory, closes the fd, and restores
the credentials to ``os.environ``. Because that write happens **after** ``execve``,
libc puts the strings on the heap and the kernel's ``[env_start, env_end)`` snapshot
never sees them: ``/proc/self/environ`` stays credential-free for the life of the
process. :func:`assert_proc_environ_clean` is the fail-closed guard that *proves* it
at runtime rather than trusting this paragraph.

Nothing touches disk, and it behaves identically in dev and in a k8s pod (where
there is no ``.env`` to fall back on and env-injection is the only channel).

WHY THE CHILD STILL PUTS THEM IN ``os.environ``
-----------------------------------------------
Every credential consumer in the process — ``Settings`` (``config.py``), the
``factory.configure_*_credentials`` bridge, ``foundry_endpoint``, ``grader_transport``,
``runpod_backend`` — resolves its key lazily from ``os.environ``, and most of them run
**outside** the REPL window (rubric-gen, root-client construction, finalize, the
validator panel), where the vault is by design disarmed. Keeping the blob out of
``os.environ`` entirely would therefore mean rewriting all of them; until that lands,
it would silently break auth in production, which is a strictly worse outcome than the
threat this closes. So the handoff restores the *pre-``execve`` status quo ante* in
``os.environ`` and hands ownership straight to the existing :func:`arm` /
:func:`exposed` / :func:`disarm` machinery — which empties ``os.environ`` again for
the whole REPL window, the only window in which attacker-steered code runs.

Net effect for root-written REPL code: ``os.environ`` is empty (arm) **and**
``/proc/self/environ`` is empty (the handoff). Both doors, not one.

WHAT THIS DOES *NOT* FIX — read before trusting it
--------------------------------------------------
This is a **blast-radius mitigation, not a sandbox.** Root-written REPL code still
holds ``__import__`` and ``open``, therefore:

* ``open(".env").read()`` still discloses every credential *in dev* — the file is on
  disk. (A k8s/docker deployment has no ``.env``; there, env-injection is the only
  channel, which is exactly the one the handoff closes.)
* ``/proc/<ppid>/environ`` — the **parent** (uvicorn) still holds the credentials in
  its own exec snapshot. Whether the child can read it is up to the kernel
  (``kernel.yama.ptrace_scope``: ``0`` ⇒ readable by a same-uid process, ``1`` ⇒
  denied to a non-descendant). Do not rely on that sysctl for confidentiality.
* the CLI path (``python -m backend.cli``) inherits the operator's real shell env, so
  its own ``/proc/self/environ`` *does* carry keys — an accepted operator-trusted
  path (that shell, and the ``.env`` beside it, already hold the keys).
* ``__import__("os").system(...)`` still owns the host.

Only real process isolation closes those. Do not let this module delay the
sandboxing work.

MECHANICS
---------
* :func:`split_spawn_env` / :func:`write_handoff` / :func:`receive_handoff` — the
  out-of-band spawn handoff described above (parent side / parent side / child side).
* :func:`assert_proc_environ_clean` — fail-closed guard on the ``execve`` snapshot.
* :func:`arm` — snapshot every managed name that is present in ``os.environ``
  into a module-private dict, then delete it from ``os.environ``. Ends by
  asserting the boundary is clean (fail-closed self-check).
* :func:`exposed` — re-inject the vaulted values for the duration of a primitive
  call, then remove them again. Re-entrant (nested primitives) and thread-safe.
  On exit it deletes **every** managed name — so a consumer that re-pollutes
  ``os.environ`` mid-primitive (e.g. ``factory.configure_openai_agents_sdk_credentials``)
  cannot leak a key past the primitive's return.
* :func:`get` — the read accessor for harness code that must pass a credential
  **explicitly** (e.g. into a subprocess env) rather than relying on ``os.environ``
  inheritance. Falls back to ``os.environ`` when the vault is not armed, so every
  call site stays byte-identical outside the REPL window.
* :func:`assert_repl_boundary_clean` — the regression guard. Raises
  :class:`CredentialLeak` if any managed name holds a non-empty value.

Never log, print, or serialise a credential VALUE from this module — names only.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import IO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The managed set — NAMES ONLY, never values.
# ---------------------------------------------------------------------------
# Every env var that carries an LLM or cloud credential the orchestrator process
# can hold. Keep this in sync with `backend/config.py`'s credential Field aliases
# and `cli.py::_warn_on_shell_env_override`'s suspect-key list.
CREDENTIAL_ENV_VARS: tuple[str, ...] = (
    # --- LLM providers (root model + sub-agents) ---
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "OPENAI_ADMIN_KEY",
    "OPENROUTER_API_KEY",
    "FEATHERLESS_API_KEY",
    "GEMINI_API_KEY",
    "AI_GATEWAY_API_KEY",
    # --- Azure OpenAI + Azure AI Foundry (KEY1/KEY2 are the portal's labels) ---
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_KEY1",
    "AZURE_OPENAI_KEY2",
    "AZURE_FOUNDRY_API_KEY",
    # --- Navigation accelerator endpoint credential ---
    "OPENRESEARCH_ACCELERATOR_API_KEY",
    # --- Model hub ---
    "HUGGING_FACE_HUB_TOKEN",
    "HF_TOKEN",
    # --- Run-start gate secret ---
    "OPENRESEARCH_DEMO_SECRET",
    # --- Prefixed aliases: config.py accepts these AND bidirectionally bridges
    #     OPENRESEARCH_* <-> REPROLAB_* at import, so both spellings can hold a
    #     live key. Scrubbing only the bare name would leave the alias behind.
    "OPENRESEARCH_ANTHROPIC_API_KEY",
    "OPENRESEARCH_OPENAI_API_KEY",
    "OPENRESEARCH_OPENAI_ADMIN_KEY",
    "OPENRESEARCH_AZURE_OPENAI_API_KEY",
    "REPROLAB_ANTHROPIC_API_KEY",
    "REPROLAB_OPENAI_API_KEY",
    "REPROLAB_OPENAI_ADMIN_KEY",
)

# GOOGLE_APPLICATION_CREDENTIALS is DELIBERATELY NOT managed here. It is a
# *path* to a key file, not a secret; the file itself stays readable through the
# REPL's `open()` either way, so scrubbing buys no confidentiality — while
# google-auth refreshes credentials on a background thread that would resolve it
# at an arbitrary moment (outside any primitive's exposure window) and silently
# lose GCS/GKE auth mid-run. Net: real breakage risk, zero security gain.

_FLAG = "OPENRESEARCH_CREDENTIAL_VAULT"

_LOCK = threading.RLock()
_vault: dict[str, str] = {}
_armed: bool = False
_exposure_depth: int = 0


class CredentialLeak(RuntimeError):
    """A managed credential was present in ``os.environ`` at the REPL boundary.

    Carries NAMES ONLY — never a value.
    """


def is_enabled() -> bool:
    """Whether the vault is active. Default **ON** — this is a security control.

    Escape hatch for an operator emergency only:
    ``OPENRESEARCH_CREDENTIAL_VAULT=0`` (also ``false``/``no``/``off``).
    Deliberately inverted from the repo's default-OFF feature-flag convention:
    a credential scrub that ships default-OFF protects nobody.
    """
    return os.environ.get(_FLAG, "").strip().lower() not in ("0", "false", "no", "off")


def armed() -> bool:
    """True while the credentials are held in the vault and out of ``os.environ``."""
    return _armed


def managed_names() -> tuple[str, ...]:
    """The managed env-var names (no values)."""
    return CREDENTIAL_ENV_VARS


def _scrub_locked() -> None:
    """Delete every managed name from ``os.environ``. Caller holds ``_LOCK``."""
    for name in CREDENTIAL_ENV_VARS:
        os.environ.pop(name, None)


def _expose_locked() -> None:
    """Re-inject the vaulted values into ``os.environ``. Caller holds ``_LOCK``."""
    for name, value in _vault.items():
        os.environ[name] = value


def leaked_names() -> list[str]:
    """Managed names currently holding a NON-EMPTY value in ``os.environ``.

    An empty value carries no secret (``ANTHROPIC_API_KEY=`` is the documented
    local-dev pattern for forcing the OAuth path), so it is not a leak.
    """
    return [n for n in CREDENTIAL_ENV_VARS if os.environ.get(n, "").strip()]


def assert_repl_boundary_clean() -> None:
    """**The regression guard.** Raise if any credential is reachable from the REPL.

    Called at the REPL boundary in ``run.py`` immediately after :func:`arm`, so by
    construction it passes today. Its job is to fail loudly the day someone adds a
    new ``os.environ[...] = <secret>`` bridge upstream of ``rlm.completion()`` —
    which is exactly how this vulnerability was introduced in the first place.

    Fail-CLOSED, matching the repo's other trust gates: a leak raises rather than
    warning, because root-written REPL code is about to run with whatever is there.
    """
    if not is_enabled():
        return
    leaked = leaked_names()
    if leaked:
        raise CredentialLeak(
            "credential(s) reachable from the RLM REPL at the completion boundary: "
            + ", ".join(sorted(leaked))
            + " — root-written code can read os.environ (rlm's _SAFE_BUILTINS keeps "
            "__import__), and the paper steering it is attacker-influenceable. Route "
            "the credential through backend.agents.runtime.credential_vault instead "
            "of writing it into os.environ."
        )


# ---------------------------------------------------------------------------
# The spawn handoff — keep credentials out of the child's execve snapshot.
# ---------------------------------------------------------------------------
# Names the CHILD reads to find its inherited pipe. Carries an fd NUMBER, not a
# secret, so it is safe in the child's env (and therefore in /proc/self/environ).
HANDOFF_FD_ENV = "OPENRESEARCH_CREDENTIAL_HANDOFF_FD"

# fd 0. The run subprocess is spawned with stdin=DEVNULL and never reads stdin,
# so it is a free, already-inherited channel — and unlike subprocess's `pass_fds`
# (POSIX-only) it works on Windows too, which this repo still supports.
HANDOFF_FD = 0

_PROC_ENVIRON = "/proc/self/environ"


class CredentialHandoffError(RuntimeError):
    """The out-of-band credential handoff was malformed. Carries NAMES ONLY."""


def split_spawn_env(env: Mapping[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Split a child-process env into ``(env_without_credentials, credential_blob)``.

    The single source of truth is :data:`CREDENTIAL_ENV_VARS` — the same list
    :func:`arm` scrubs, deliberately not a second one.

    EMPTY values ride the blob too, rather than being dropped: ``ANTHROPIC_API_KEY=``
    (set, empty) is the documented local-dev pattern for *forcing* the OAuth path, and
    is distinguishable from an unset key by an ``in os.environ`` test. Round-tripping
    it keeps the child's ``os.environ`` byte-identical to what it inherited before
    this handoff existed.
    """
    clean: dict[str, str] = {}
    blob: dict[str, str] = {}
    managed = frozenset(CREDENTIAL_ENV_VARS)
    for name, value in env.items():
        if name in managed:
            blob[name] = value
        else:
            clean[name] = value
    return clean, blob


def write_handoff(stream: IO[bytes], blob: Mapping[str, str]) -> None:
    """Parent side: write the credential blob to the child's inherited pipe, once.

    ALWAYS closes ``stream`` — the child blocks on ``read()`` until EOF, so a missed
    close would wedge every run. A dead child (``BrokenPipeError``) is not this
    function's problem to report: the run fails loudly on its own, and raising here
    would turn a crashed child into a 500 on the API that spawned it.
    """
    try:
        stream.write(json.dumps(dict(blob), separators=(",", ":")).encode("utf-8"))
        stream.flush()
    except (BrokenPipeError, OSError, ValueError):
        logger.warning(
            "credential_vault: could not hand %d credential(s) to the run subprocess "
            "— it exited before reading the pipe",
            len(blob),
        )
    finally:
        try:
            stream.close()
        except (BrokenPipeError, OSError):
            pass


def _close_handoff_fd(fd: int) -> None:
    """Close the handoff fd, restoring ``/dev/null`` when it was stdin.

    The run subprocess was spawned with ``stdin=DEVNULL`` before the handoff existed,
    and its own children (the ``claude`` CLI, docker, training code) inherit fd 0. Hand
    them the same empty-but-valid fd they always had, rather than a closed one.
    """
    try:
        if fd == HANDOFF_FD:
            devnull = os.open(os.devnull, os.O_RDONLY)
            try:
                os.dup2(devnull, fd)
            finally:
                os.close(devnull)
        else:
            os.close(fd)
    except OSError:  # already closed / not a real fd — nothing to protect
        pass


def receive_handoff(fd: int | None = None) -> int:
    """Child side: read the credential blob from the inherited pipe. Returns the count.

    **Call this before importing anything from ``backend``.** ``config.py`` bridges the
    legacy ``REPROLAB_*`` ⇄ ``OPENRESEARCH_*`` spellings *at import time*, reading
    whatever is in ``os.environ`` at that moment — so a handoff that lands after that
    import would leave a ``.env``-sourced legacy-spelled key un-bridged and silently
    break, e.g., RunPod auth.

    Returns 0 (legally) when there is no handoff: a deployment can hold zero API keys
    and authenticate purely via the ``claude`` CLI OAuth subscription.

    :data:`HANDOFF_FD_ENV` is popped from ``os.environ`` FIRST, so the run subprocess's
    own children can never inherit a stale fd number and try to re-read it.
    """
    raw = os.environ.pop(HANDOFF_FD_ENV, "").strip()
    if fd is None:
        if not raw:
            return 0
        try:
            fd = int(raw)
        except ValueError as exc:
            raise CredentialHandoffError(
                f"{HANDOFF_FD_ENV} is not a file descriptor number"
            ) from exc

    chunks: list[bytes] = []
    try:
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    except OSError as exc:
        _close_handoff_fd(fd)
        raise CredentialHandoffError(
            f"could not read the credential handoff from fd {fd}"
        ) from exc
    else:
        _close_handoff_fd(fd)

    payload = b"".join(chunks)
    if not payload:
        return 0

    try:
        blob = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CredentialHandoffError("credential handoff payload is not valid JSON") from exc
    if not isinstance(blob, dict):
        raise CredentialHandoffError("credential handoff payload is not a JSON object")

    managed = frozenset(CREDENTIAL_ENV_VARS)
    for name, value in blob.items():
        # Names only in every error path — never a value.
        if name not in managed:
            raise CredentialHandoffError(
                f"credential handoff carries an unmanaged name: {name!r} — add it to "
                "CREDENTIAL_ENV_VARS or keep it in the spawn env"
            )
        if not isinstance(value, str):
            raise CredentialHandoffError(f"credential handoff value for {name} is not a string")
        # Post-execve, so libc allocates this on the heap: the kernel's
        # [env_start, env_end) snapshot that /proc/self/environ exposes never
        # sees it. assert_proc_environ_clean() is the guard that proves it.
        os.environ[name] = value

    logger.info(
        "credential_vault: received %d credential(s) out-of-band (names: %s) — "
        "they were never in this process's execve environment",
        len(blob),
        ", ".join(sorted(blob)) or "none",
    )
    return len(blob)


def proc_environ_leaked_names() -> list[str]:
    """Managed names holding a NON-EMPTY value in the kernel's ``execve`` snapshot.

    This is what root-written REPL code reads when it does
    ``open("/proc/self/environ").read()`` — a region frozen at ``execve`` that no
    ``os.environ`` mutation can alter. Returns ``[]`` where ``/proc`` does not exist
    (Windows/macOS): the disclosure surface is Linux-specific, and so is the check.
    """
    try:
        with open(_PROC_ENVIRON, "rb") as handle:
            raw = handle.read()
    except OSError:
        return []
    managed = frozenset(CREDENTIAL_ENV_VARS)
    leaked: set[str] = set()
    for entry in raw.split(b"\0"):
        name, sep, value = entry.partition(b"=")
        if not sep or not value.strip():
            continue
        try:
            decoded = name.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if decoded in managed:
            leaked.add(decoded)
    return sorted(leaked)


def assert_proc_environ_clean() -> None:
    """**The spawn-boundary guard.** Raise if a credential is in the ``execve`` snapshot.

    Called by the run subprocess immediately after :func:`receive_handoff`, so by
    construction it passes today. Its job is to fail loudly the day someone re-adds a
    credential to the child's spawn ``env=`` dict — which no ``os.environ`` scrub, and
    therefore no other guard in this module, can defend against.

    Fail-CLOSED, like :func:`assert_repl_boundary_clean`: the process is about to run
    attacker-steerable code that can read this file.
    """
    if not is_enabled():
        return
    leaked = proc_environ_leaked_names()
    if leaked:
        raise CredentialLeak(
            "credential(s) baked into this process's execve environment, readable via "
            + _PROC_ENVIRON
            + " by root-written REPL code (an os.environ scrub CANNOT remove them — the "
            "kernel snapshot is frozen at exec): "
            + ", ".join(leaked)
            + " — spawn the run subprocess with credential_vault.split_spawn_env() + "
            "write_handoff() instead of passing the credential in env=."
        )


def arm() -> None:
    """Snapshot managed credentials out of ``os.environ`` into the vault.

    Idempotent. Ends with :func:`assert_repl_boundary_clean` — a fail-closed
    self-check that the scrub actually worked.
    """
    global _armed
    if not is_enabled():
        logger.warning(
            "credential_vault: DISABLED via %s — API keys stay readable from the "
            "RLM REPL (os.environ). This is an operator escape hatch, not a default.",
            _FLAG,
        )
        return
    with _LOCK:
        if _armed:
            return
        _vault.clear()
        for name in CREDENTIAL_ENV_VARS:
            if name in os.environ:
                # Store presence AND value (possibly "") so disarm restores exactly.
                _vault[name] = os.environ[name]
        _scrub_locked()
        _armed = True
    logger.info(
        "credential_vault: armed — %d credential env var(s) held out of os.environ "
        "for the REPL window (names: %s)",
        len(_vault),
        ", ".join(sorted(_vault)) or "none",
    )
    assert_repl_boundary_clean()


def disarm() -> None:
    """Restore the vaulted credentials to ``os.environ`` exactly as they were.

    Called once the REPL window closes. Everything after ``rlm.completion()``
    (finalize, the external-validator panel, report writing, the campaign loop)
    then sees a byte-identical environment.
    """
    global _armed, _exposure_depth
    with _LOCK:
        if not _armed:
            return
        _scrub_locked()
        _expose_locked()
        _vault.clear()
        _armed = False
        _exposure_depth = 0
    logger.info("credential_vault: disarmed — os.environ restored")


def get(name: str) -> str:
    """Read a credential by name — from the vault when armed, else ``os.environ``.

    The accessor for harness code that must hand a credential to a **subprocess**
    (``claude_runtime``'s ``ClaudeAgentOptions(env=...)``) instead of relying on
    parent-``os.environ`` inheritance. Outside the REPL window this is exactly
    ``os.environ.get(name, "")``, so every call site stays byte-identical.
    """
    with _LOCK:
        if _armed and name in _vault:
            return _vault[name]
    return os.environ.get(name, "")


@contextmanager
def exposed() -> Iterator[None]:
    """Re-expose the vaulted credentials for the duration of a primitive call.

    This is the ``set -> use -> immediately restore`` window. It exists because
    several credential consumers resolve their key **lazily from ``os.environ``**
    at call time, and all of them run inside a primitive:

    * ``grader_transport.build_transport_client`` → ``AnthropicMessagesClient()``
      with ``api_key=None`` (the Anthropic SDK reads ``ANTHROPIC_API_KEY``) —
      reached from ``verify_against_rubric`` via ``leaf_scorer``;
    * ``runpod_backend`` → ``OPENRESEARCH_RUNPOD_API_KEY`` — reached from
      ``run_experiment``;
    * ``azure_openai_runtime`` / the Foundry resolvers → ``AZURE_*`` — reached
      from ``implement_baseline``;
    * ``factory.configure_*_credentials`` → re-bridges keys into ``os.environ``.

    Those modules are owned elsewhere; wrapping the primitive boundary keeps them
    byte-identical instead of rewriting each one to read the vault.

    Re-entrant (a primitive may call another) and thread-safe. On exit it deletes
    **every** managed name — including any a consumer wrote back mid-primitive —
    so nothing survives into the next REPL turn.

    No-op when the vault is not armed.

    RESIDUAL RISK (honest): the exposure window is process-global, so root-written
    REPL code that starts a background thread *before* calling a primitive can poll
    ``os.environ`` and catch the credentials during the window. That race is real
    and only process isolation closes it. It requires a far more sophisticated
    injection than ``print(os.environ)``, which is what this stops.
    """
    global _exposure_depth
    if not _armed:
        yield
        return
    with _LOCK:
        _exposure_depth += 1
        if _exposure_depth == 1:
            _expose_locked()
    try:
        yield
    finally:
        with _LOCK:
            _exposure_depth -= 1
            if _exposure_depth <= 0:
                _exposure_depth = 0
                # Blanket scrub, not a targeted restore: a consumer may have
                # written a managed name back into os.environ during the call.
                _scrub_locked()


@contextmanager
def armed_vault() -> Iterator[None]:
    """Arm the vault for a block (the REPL window), restoring on the way out."""
    arm()
    try:
        yield
    finally:
        disarm()


__all__ = [
    "CREDENTIAL_ENV_VARS",
    "HANDOFF_FD",
    "HANDOFF_FD_ENV",
    "CredentialHandoffError",
    "CredentialLeak",
    "armed",
    "armed_vault",
    "arm",
    "assert_proc_environ_clean",
    "assert_repl_boundary_clean",
    "disarm",
    "exposed",
    "get",
    "is_enabled",
    "leaked_names",
    "managed_names",
    "proc_environ_leaked_names",
    "receive_handoff",
    "split_spawn_env",
    "write_handoff",
]
