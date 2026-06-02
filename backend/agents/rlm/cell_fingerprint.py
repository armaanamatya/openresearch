"""Cell fingerprinting — the resume re-run predicate for the training matrix.

Cell-level checkpoint/resume (Track B) needs a single, deterministic answer to
one question: *"has anything that would change this cell's output changed since
it last ran ok?"*.  This module computes that answer as a content hash — the
**fingerprint** — over exactly the inputs that affect a cell's result:

* the bytes of the env-specific helper module the cell exercises
  (``alfworld_env.py`` for an ALFWorld cell, ``search_qa_env.py`` for a
  Search-QA cell, …) — a bug-fix to ALFWorld's reward shaping MUST re-run the
  ALFWorld cells,
* the bytes of the SHARED helper modules every cell uses
  (``sdar_env_base.py`` + ``agentic_rollout.py``) — a change to the rollout
  loop affects every cell,
* the cell's own training parameters (model id, baseline, seed, steps, …),
* the values of a small allow-list of behaviour-affecting environment vars,
  env-scoped so an ALFWorld flag perturbs only ALFWorld cells (see
  ``FLAG_ENV_PREFIXES``).

The **split** between env-specific and shared helpers is the load-bearing
guarantee: editing ``alfworld_env.py`` flips the fingerprint of an ALFWorld
cell but leaves a Search-QA cell's fingerprint untouched, so resume re-runs
only the cells the edit actually touched.

We deliberately do **NOT** hash ``train_cell.py`` itself.  That file is the
single shared single-cell trainer, regenerated wholesale on every codegen pass,
so hashing it would invalidate *every* cell on *every* warm-retry — defeating
resume.  The ``REPROLAB_TRAINER_VERSION`` env var stands in: bump it to force a
matrix-wide re-run when the trainer's behaviour genuinely changed.

Pure stdlib (``hashlib`` / ``json`` / ``os`` / ``pathlib``).  Fail-soft: a
missing helper file contributes a sentinel hash and never raises, so a
half-populated ``code/`` directory degrades to "fingerprint differs → re-run"
rather than crashing the resume path.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "ENV_HELPER_FILES",
    "SHARED_HELPER_FILES",
    "FLAG_ALLOWLIST",
    "FLAG_ENV_PREFIXES",
    "FINGERPRINT_CELL_PARAMS",
    "MISSING_SENTINEL",
    "compute_fingerprint",
    "dep_files_for_env",
]


# Env name → the single env-specific helper module that env's cells exercise.
# A cell whose ``env`` is not in this map contributes no env-specific helper
# (only the shared files + params + flags decide its fingerprint).
ENV_HELPER_FILES: dict[str, str] = {
    "alfworld": "alfworld_env.py",
    "search_qa": "search_qa_env.py",
    "webshop": "webshop_env.py",
}

# Helper modules EVERY cell depends on regardless of env.  Sorted at hash time.
SHARED_HELPER_FILES: tuple[str, ...] = (
    "sdar_env_base.py",
    "agentic_rollout.py",
)

# Behaviour-affecting env vars folded into the fingerprint.  REPROLAB_TRAINER_VERSION
# is the deliberate stand-in for the un-hashed (codegen-churning) train_cell.py:
# bump it to force a matrix-wide re-run.  Missing vars contribute "".
FLAG_ALLOWLIST: tuple[str, ...] = (
    "REPROLAB_ALFWORLD_SHAPED_REWARD",
    "REPROLAB_ALFWORLD_MAX_TURNS",
    "REPROLAB_SEARCH_QA_DENSE",
    "REPROLAB_TRAINER_VERSION",
)

# Env-scoping for allow-listed flags (2026-06-02 integration fix). A flag whose
# prefix names an env is folded into ONLY that env's cells' fingerprints — so
# flipping REPROLAB_ALFWORLD_SHAPED_REWARD re-runs the ALFWorld cells but leaves
# the Search-QA cells untouched (the same split guarantee the helper files give).
# A flag matching no prefix (e.g. REPROLAB_TRAINER_VERSION) is GLOBAL: folded into
# every cell, so bumping it forces a matrix-wide re-run.
FLAG_ENV_PREFIXES: dict[str, str] = {
    "REPROLAB_ALFWORLD_": "alfworld",
    "REPROLAB_SEARCH_QA_": "search_qa",
    "REPROLAB_WEBSHOP_": "webshop",
}

# The subset of a cell dict that affects its training output.  Sorted before
# hashing so key-order in cells.json never perturbs the fingerprint.  Volatile /
# placement-only keys (id, model_key alias, est_vram_gb) are intentionally
# excluded — they don't change what the cell computes.
FINGERPRINT_CELL_PARAMS: tuple[str, ...] = (
    "model_id",
    "model_key",
    "baseline",
    "env",
    "seed",
    "steps",
    "group_size",
    "tasks_per_batch",
    "max_new_tokens",
    "max_turns",
)

# Contributed in place of a real file hash when a helper file is absent.  A
# stable sentinel (not a random value) so a consistently-missing file yields a
# consistent fingerprint — only its appearance/disappearance flips the hash.
MISSING_SENTINEL = "MISSING"


def _hash_file(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``'s bytes, or the missing sentinel.

    Fail-soft by contract: a missing file, a directory, or an unreadable path
    yields :data:`MISSING_SENTINEL` rather than raising — the resume predicate
    treats an absent helper as "changed" and re-runs, which is the safe default.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return MISSING_SENTINEL
    return hashlib.sha256(data).hexdigest()


def dep_files_for_env(env_name: str, code_dir: str | Path) -> list[Path]:
    """Return the helper-file paths whose bytes a cell of ``env_name`` depends on.

    The list is the env-specific helper (if any) followed by the shared helpers
    in sorted order, each resolved under ``code_dir``.  Files need not exist —
    :func:`_hash_file` resolves a missing path to the sentinel.
    """
    code = Path(code_dir)
    files: list[Path] = []
    env_helper = ENV_HELPER_FILES.get(str(env_name or "").strip().lower())
    if env_helper:
        files.append(code / env_helper)
    files.extend(code / name for name in sorted(SHARED_HELPER_FILES))
    return files


def _flag_scope(flag: str) -> str | None:
    """Return the env a flag is scoped to (by prefix), or ``None`` if it is global."""
    for prefix, env_name in FLAG_ENV_PREFIXES.items():
        if flag.startswith(prefix):
            return env_name
    return None


def _flag_values(env: Mapping[str, str] | None, env_name: str) -> dict[str, str]:
    """Resolve the allow-listed env vars relevant to a cell of ``env_name``.

    Env-scoped flags (those whose prefix names an env in :data:`FLAG_ENV_PREFIXES`)
    are included ONLY when they belong to this cell's env; global flags (no env
    prefix, e.g. ``REPROLAB_TRAINER_VERSION``) are always included.  This is what
    makes flipping ``REPROLAB_ALFWORLD_SHAPED_REWARD`` re-run only the ALFWorld
    cells and leave the Search-QA cells' fingerprints unchanged.  Values are read
    from ``env`` (or ``os.environ`` when ``None``); a missing var contributes ``""``.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    out: dict[str, str] = {}
    for flag in FLAG_ALLOWLIST:
        scope = _flag_scope(flag)
        if scope is None or scope == env_name:
            out[flag] = str(source.get(flag, ""))
    return out


def compute_fingerprint(
    cell: dict,
    code_dir: str | Path,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return the SHA-256 hex fingerprint of one cell's reproducibility inputs.

    The fingerprint is a hash over a CANONICAL JSON blob (sorted keys, no
    whitespace jitter) of four parts:

    * ``env_helper`` — SHA-256 of the env-specific helper module's bytes (or
      :data:`MISSING_SENTINEL` when the cell's ``env`` has no mapped helper or
      the file is absent),
    * ``shared`` — SHA-256 of each shared helper module's bytes, keyed by file
      name and sorted,
    * ``params`` — the :data:`FINGERPRINT_CELL_PARAMS` subset of ``cell``,
      sorted, JSON-normalised,
    * ``flags`` — the :data:`FLAG_ALLOWLIST` env-var values (missing → ``""``).

    Guarantees (all unit-tested):

    * **Determinism** — identical inputs always yield the same hash.
    * **Split** — editing the env-specific helper's bytes changes a cell of that
      env, but NOT a cell of a different env (only shared helpers are common).
    * **Sensitivity** — changing any fingerprinted cell param, or any
      allow-listed flag value, flips the hash.
    * **Fail-soft** — a missing helper contributes the sentinel, never raises.

    Args:
        cell:     One cell dict from ``cells.json`` (uses ``env`` to pick the
                  helper + :data:`FINGERPRINT_CELL_PARAMS` for params).
        code_dir: Directory holding the vendored helper modules (``code/``).
        env:      Mapping to read allow-listed flags from.  ``None`` →
                  ``os.environ``.

    Returns:
        64-char lowercase SHA-256 hex string.
    """
    cell = cell if isinstance(cell, dict) else {}
    env_name = str(cell.get("env", "") or "").strip().lower()
    code = Path(code_dir)

    env_helper_name = ENV_HELPER_FILES.get(env_name)
    env_helper_hash = (
        _hash_file(code / env_helper_name) if env_helper_name else MISSING_SENTINEL
    )

    shared_hashes = {
        name: _hash_file(code / name) for name in sorted(SHARED_HELPER_FILES)
    }

    params = {key: cell.get(key) for key in sorted(FINGERPRINT_CELL_PARAMS)}

    blob = {
        "env_helper": env_helper_hash,
        "shared": shared_hashes,
        "params": params,
        "flags": _flag_values(env, env_name),
    }

    canonical = json.dumps(blob, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
