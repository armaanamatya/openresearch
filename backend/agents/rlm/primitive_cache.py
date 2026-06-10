"""Content-addressed cache for deterministic RLM primitives.

The earlier-shipped analysis flagged sub-RLM amplification as the biggest
runtime tax: the root model calls ``understand_section`` and
``extract_hyperparameters`` dozens of times per run, each spawning a
sub-RLM that is an LLM call.  When the agent retries a paper (or the
operator re-runs after a fix), the SAME inputs trigger the SAME LLM work
again — wasted subscription rate-limit budget and wasted minutes.

This module is a tiny content-addressed cache: same input → same output,
returned from disk in microseconds instead of a fresh LLM round-trip.

Storage shape::

    runs/<project_id>/rlm_state/primitive_cache.jsonl  (append-only JSONL)
    {"key": "<sha256-prefix>", "primitive": "understand_section",
     "result": {...}, "ts": "2026-05-24T19:30:00+00:00"}

Survives across attempts of the same paper because ``project_id`` is keyed
by ``arxiv_id``.  Retry of paper X hits the prior attempt's cache for the
paper-analysis primitives.

Design contract:

  * Pure-function primitives only (see ``CACHEABLE_PRIMITIVES``).
  * Fail-soft on every path — observability and persistence must never
    block a run.  A corrupt JSONL line is skipped, not raised.
  * Versioned key prefix (``v1:``) so a future contract change can
    invalidate the cache without a manual purge.
  * Opt-out per run via ``OPENRESEARCH_PRIMITIVE_CACHE=disabled``.

Cached with extra care (Lane A — warm retry):

  * ``implement_baseline`` — cached on ``{plan, repair_context, arxiv_id,
    sandbox_mode, gpu_mode}`` (NOT ``remaining_s`` — that changes every
    call).  The primitive ALSO verifies the on-disk ``code/commands.json``
    exists on hit and recomputes if attempt_isolation archived the code
    between cache write and re-read.

Not cacheable (intentional exclusions):

  * ``run_experiment`` — depends on real-world state (datasets fetched,
    GPU availability, sandbox).
  * ``build_environment`` — Docker layer-cached already; double-caching
    adds no value.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

# Bump this when any cacheable primitive's contract / output shape changes.
_CACHE_VERSION: Final[str] = "v1"

# Allow-list of primitive names whose outputs are PURE functions of their
# inputs and therefore safe to cache.  Anything not in this set is never
# cached even if a caller invokes ``maybe_get`` / ``put``.
CACHEABLE_PRIMITIVES: Final[frozenset[str]] = frozenset({
    "understand_section",
    "extract_hyperparameters",
    "detect_environment",
    "plan_reproduction",
    "verify_against_rubric",
    # Lane A — warm-retry cache. ``implement_baseline`` is expensive (~5 min,
    # ~$0.50 Sonnet) and on a kill-and-relaunch the prior code/ usually already
    # holds the answer. The cache key intentionally excludes ``remaining_s``
    # (changes every call); the primitive ALSO verifies code/commands.json
    # exists on hit and treats the miss as recompute-from-scratch.
    "implement_baseline",
})

_CACHE_FILENAME: Final[str] = "primitive_cache.jsonl"
_DISABLE_ENV_VAR: Final[str] = "OPENRESEARCH_PRIMITIVE_CACHE"


# ---------------------------------------------------------------------------
# Hit-time schema validators
#
# Every cache HIT runs the matching validator before returning the cached
# result.  A bad first call can poison the cache (e.g. an LLM-call primitive
# returns a malformed dict during a transient outage); validating on every hit
# means the poison gets evicted on the very next request instead of returning
# the stale wrong answer forever.  Validators are *structural* only — they
# check shape, not correctness; correctness checks would defeat the cache.
#
# On failed validation: log a warning, treat the entry as a miss, continue
# scanning subsequent JSONL entries.  ``put`` will append a fresh good entry
# after the recompute, so the bad entry is effectively retired.
# ---------------------------------------------------------------------------


def _v_understand_section(r: dict) -> bool:
    """understand_section returns 5 list-valued keys (Lane I §2)."""
    if not isinstance(r, dict):
        return False
    required = {"datasets", "metrics", "training_recipe", "hardware_clues", "ambiguities"}
    return required.issubset(set(r))


def _v_extract_hyperparameters(r: dict) -> bool:
    """extract_hyperparameters returns hparam slots; accept partial returns."""
    if not isinstance(r, dict):
        return False
    expected = {
        "optimizer", "learning_rate", "batch_size", "epochs_or_steps",
        "scheduler", "other_hparams", "_meta",
    }
    return bool(expected & set(r))


def _v_detect_environment(r: dict) -> bool:
    """detect_environment returns an EnvironmentSpec dict (must have dockerfile)."""
    if not isinstance(r, dict):
        return False
    return "dockerfile" in r and "framework" in r and "python_version" in r


def _v_plan_reproduction(r: dict) -> bool:
    """plan_reproduction returns a ReproductionContract; reject error dicts."""
    if not isinstance(r, dict):
        return False
    if r.get("success") is False:
        return False  # don't keep a cached failure that can self-heal on retry
    return any(
        k in r for k in (
            "smoke_test_plan", "eval_plan", "verification_checklist",
            "datasets", "primary_metric",
        )
    )


def _v_verify_against_rubric(r: dict) -> bool:
    """verify_against_rubric returns {overall_score, target_score, areas, ...}."""
    if not isinstance(r, dict):
        return False
    return "overall_score" in r and "target_score" in r and "areas" in r


def _v_implement_baseline(r: dict) -> bool:
    """implement_baseline cache stores an ok/error envelope or legacy path wrapper."""
    if not isinstance(r, dict):
        return False
    if r.get("ok") is True:
        return (
            isinstance(r.get("code_path"), str)
            and bool(r["code_path"].strip())
            and isinstance(r.get("files"), list)
            and "commands.json" in r["files"]
        )
    if r.get("ok") is False:
        return bool(r.get("error")) and bool(r.get("error_code"))
    if r.get("_kind") == "path":
        return isinstance(r.get("value"), str) and len(r["value"]) > 0
    # Cached error dict (e.g. timeout). Reject empty/malformed errors so the
    # next attempt actually re-tries the agent.
    if "success" in r:
        return bool(r.get("error")) or bool(r.get("logs"))
    return False


_CACHE_VALIDATORS: Final[dict[str, Any]] = {
    "understand_section":      _v_understand_section,
    "extract_hyperparameters": _v_extract_hyperparameters,
    "detect_environment":      _v_detect_environment,
    "plan_reproduction":       _v_plan_reproduction,
    "verify_against_rubric":   _v_verify_against_rubric,
    "implement_baseline":      _v_implement_baseline,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """Return False when ``OPENRESEARCH_PRIMITIVE_CACHE=disabled`` is set."""
    return os.environ.get(_DISABLE_ENV_VAR, "enabled").lower() != "disabled"


def make_key(primitive: str, *, payload: Any) -> str:
    """Compose a versioned content-addressed cache key.

    ``payload`` is the canonical input shape — anything JSON-serialisable.
    The hash is byte-identical for byte-identical inputs after canonical
    JSON encoding (sorted keys, default=str fallback for non-JSON types).
    """
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()[:32]
    return f"{_CACHE_VERSION}:{primitive}:{digest}"


def maybe_get(project_dir: Path, primitive: str, *, payload: Any) -> dict | None:
    """Look up a cached result for (primitive, payload).

    Returns ``None`` on miss, on disable, on non-cacheable primitive, or on
    any I/O / JSON error (fail-soft).  Returns the cached result dict on hit.
    """
    if not is_enabled():
        return None
    if primitive not in CACHEABLE_PRIMITIVES:
        return None
    if not isinstance(project_dir, Path) or not project_dir.exists():
        return None
    cache_path = project_dir / "rlm_state" / _CACHE_FILENAME
    if not cache_path.exists():
        return None

    key = make_key(primitive, payload=payload)
    try:
        with cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                if entry.get("key") == key:
                    result = entry.get("result")
                    if not isinstance(result, dict):
                        continue
                    validator = _CACHE_VALIDATORS.get(primitive)
                    if validator is not None and not validator(result):
                        logger.warning(
                            "primitive_cache: hit-validation failed for %s key=%s "
                            "(stale or poisoned entry) — treating as miss",
                            primitive, key[-8:],
                        )
                        continue
                    logger.debug("primitive_cache HIT %s key=%s", primitive, key[-8:])
                    return result
    except OSError:
        return None
    return None


def put(project_dir: Path, primitive: str, *, payload: Any, result: dict) -> None:
    """Append a primitive result to the cache.  Fail-soft.

    Skips when the primitive is not in ``CACHEABLE_PRIMITIVES``, when the
    cache is disabled, when ``result`` is not a dict, or on any I/O error.
    """
    if not is_enabled():
        return
    if primitive not in CACHEABLE_PRIMITIVES:
        return
    if not isinstance(result, dict):
        return
    try:
        cache_dir = project_dir / "rlm_state"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / _CACHE_FILENAME
        entry = {
            "key": make_key(primitive, payload=payload),
            "primitive": primitive,
            "ts": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except Exception:  # noqa: BLE001 — observability MUST NOT block the run
        logger.exception("primitive_cache: put failed for %s", primitive)


def stats(project_dir: Path) -> dict[str, int]:
    """Return per-primitive entry counts for the cache file (best-effort)."""
    counts: dict[str, int] = {}
    cache_path = project_dir / "rlm_state" / _CACHE_FILENAME
    if not cache_path.exists():
        return counts
    try:
        with cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                name = entry.get("primitive") or "?"
                counts[name] = counts.get(name, 0) + 1
    except OSError:
        return counts
    return counts


__all__ = [
    "CACHEABLE_PRIMITIVES",
    "is_enabled",
    "make_key",
    "maybe_get",
    "put",
    "stats",
]
