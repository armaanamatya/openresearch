"""Cell-matrix aggregation — turn one-GPU-per-cell results into harness metrics.

The risk-H core of the 2026-05-31 OOM/GPU remediation.  ``gpu_cell_runner``
solves *placement* (one physical GPU per cell, OOM retries) but each cell only
writes a tiny FLAT leaf dict for its single ``(model, env, baseline)`` triple.
The harness scorer (``backend/evals/paperbench/leaf_scorer.py``) and the
postflight scope guard (``rubric_guard._validate_scope_metrics``) consume a
single nested ``metrics.json`` shape::

    {
      "status": "complete" | "partial" | "failed",
      "per_model": {
        "<model_key>": {
          "<env>": {
            "<baseline>": {"status": "ok"|"failed", "metric": <float|null>, ...}
          }
        }
      },
      "scope": {
        "models_run": [...], "models_skipped": [...],
        "environments_skipped": [...], "gaps": [...]
      }
    }

This module owns the *translation*.  It is intentionally **pure**: no
``RunContext``, no ``nvidia-smi``, no network unless a probe is injected.  Every
function takes data in and returns JSON-serialisable data out, so the entire
contract is testable without a GPU, a torch install, or a live endpoint.

Three pure stages, composed by the caller in order:

1. :func:`capacity_gate` — drop cells whose estimated VRAM (× headroom) exceeds
   a single GPU's budget, BEFORE any subprocess is launched.  Never blocks on
   unknown capacity.  Emits ``capacity`` gap entries + ``models_skipped``.
2. :func:`dataset_url_preflight` — probe each distinct ``dataset_url`` once and
   drop cells whose endpoint is CONFIRMED dead (HTTP 404/410).  **Fail-soft**: a
   transient/unknown probe result NEVER drops a live env (the 2026-05-31
   requirement — a single network blip must not silently de-scope WebShop).
   Emits ``dataset_unavailable`` gap entries + ``environments_skipped``.
3. :func:`aggregate_cell_metrics` — fold the ``run_matrix`` result + the
   surviving cells into the canonical nested shape above, merging in the
   capacity/dataset gaps and skip lists from stages 1–2.

The env is keyed DIRECTLY under the model (no ``per_dataset`` wrapper) — this
matches the real on-disk SDAR sample
(``runs/.../code/outputs/.../metrics.json``) and the postflight's single-model
fast path.  Do NOT add a ``per_dataset`` layer.

Zero non-stdlib dependencies (``json`` / ``urllib`` / ``typing`` only), so this
file is copy-pasteable into an agent sandbox exactly like ``gpu_cell_runner``.
Auth-agnostic by construction (no provider branching, no LLM calls).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from typing import Any, Callable

__all__ = [
    "DEFAULT_HEADROOM",
    "capacity_gate",
    "dataset_url_preflight",
    "aggregate_cell_metrics",
    "default_dataset_probe",
    "normalize_cell_axes",
]

# Default VRAM headroom multiplier — must match the dynamic-GPU resolver
# (``OPENRESEARCH_DYNAMIC_GPU_HEADROOM=1.25``) so the gate and the scheduler agree on
# what "fits" a card.  Kept here as a named constant for the same reason
# ``gpu_cell_runner`` names ``_OOM_BATCH_SCALES``.
DEFAULT_HEADROOM: float = 1.25

# Truncation bound for the error string copied into a failed leaf.  A raw CUDA
# OOM traceback is multi-KB; the scorer only needs the leading signature, and an
# unbounded copy bloats every metrics.json.
_MAX_LEAF_ERROR_CHARS: int = 500

# HTTP status codes that prove a dataset endpoint is permanently dead.  Anything
# else (200/3xx success, 401/403 auth-gated, 5xx server error, timeouts) is
# treated as NOT-confirmed-dead so a transient or auth blip never de-scopes a
# real env.
_DEAD_HTTP_STATUSES: frozenset[int] = frozenset({404, 410})


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _dedup_sorted(values: "list[str] | None") -> list[str]:
    """Return the distinct non-empty strings in ``values``, sorted.

    Used to normalise the ``models_skipped`` / ``environments_skipped`` lists so
    the emitted scope is deterministic and free of accidental duplicates (a
    model can contribute a skip entry from several of its cells).
    """
    if not values:
        return []
    seen: set[str] = set()
    for v in values:
        if isinstance(v, str) and v:
            seen.add(v)
    return sorted(seen)


def _coerce_metric(value: Any) -> float | None:
    """Coerce a leaf ``metric`` to ``float`` | ``None`` (never raise).

    A cell may emit its headline metric as an int, a numeric string, or omit it
    entirely.  The scorer expects a ``float`` or JSON ``null``; anything
    non-numeric collapses to ``None`` rather than propagating a type error into
    the nested tree.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass — exclude explicitly
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Stage 0 — axis normalization
# ---------------------------------------------------------------------------

# Synonym fallbacks for the three per_model tree axes.  The cells.json contract
# asks for explicit ``model_key`` / ``env`` / ``baseline`` per cell, but agents
# reproducing non-SDAR papers routinely emit their own axis vocabulary
# (``variant`` / ``dataset`` / ``optimizer`` ...).  2026-06-09: an All-CNN run
# trained 14 cells to paper-grade accuracy and ``aggregate_cell_metrics``
# silently dropped every one (``per_model={}``) because the manifest carried no
# SDAR-shaped axes — the scorer saw "no measured metrics" and the run burned
# its remaining budget re-running work it had already done.  A derived,
# coarse-but-present leaf is strictly better than a discarded measured result.
_AXIS_SYNONYMS: dict[str, tuple[str, ...]] = {
    "model_key": ("model_key", "model", "model_id", "model_name", "arch", "architecture"),
    "env": ("env", "environment", "dataset", "dataset_id", "task", "benchmark"),
    "baseline": ("baseline", "method", "variant", "optimizer", "algorithm"),
}


def _cell_axes(cell: dict[str, Any], index: int) -> tuple[str, str, str, bool]:
    """Resolve ``(model_key, env, baseline, was_derived)`` for one cell.

    Explicit axis keys win.  A missing axis falls back through
    ``_AXIS_SYNONYMS`` (each source key feeds at most ONE axis, so ``variant``
    can't become both the model and the baseline), then to the cell ``id`` for
    ``model_key`` (token-rich and unique — exactly what the scorer's token
    matching wants) and the literal ``"default"`` for ``env``/``baseline``.
    Deterministic and pure; never raises.
    """
    used: set[str] = set()
    resolved: dict[str, str] = {}
    derived = False
    for axis, synonyms in _AXIS_SYNONYMS.items():
        value = ""
        for key in synonyms:
            if key in used:
                continue
            raw = cell.get(key)
            if isinstance(raw, (str, int, float)) and not isinstance(raw, bool) and str(raw).strip():
                value = str(raw).strip()
                used.add(key)
                if key != axis:
                    derived = True
                break
        resolved[axis] = value
    if not resolved["model_key"]:
        cid = cell.get("id")
        resolved["model_key"] = (
            cid.strip() if isinstance(cid, str) and cid.strip() else f"cell_{index}"
        )
        derived = True
    if not resolved["env"]:
        resolved["env"] = "default"
        derived = True
    if not resolved["baseline"]:
        resolved["baseline"] = "default"
        derived = True
    return resolved["model_key"], resolved["env"], resolved["baseline"], derived


def normalize_cell_axes(
    cells: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Ensure every cell carries non-empty ``model_key``/``env``/``baseline``.

    Returns ``(normalized_cells, notes)``.  ``notes`` is empty when every cell
    already had explicit axes (the common SDAR path — byte-for-byte unchanged);
    otherwise it holds one human/agent-readable sentence describing the
    derivation so the warning can ride the result back to the agent.

    Guarantees:

    * **Never drops a cell** (non-dict entries excepted — they can't run).
    * Input is not mutated; patched cells are shallow copies.
    * When two cells DERIVE to the same ``(model_key, env, baseline)`` triple,
      the later one's ``baseline`` is suffixed with its cell id so no leaf can
      silently overwrite another.  Explicit duplicate triples are preserved
      verbatim (existing behavior for agent-authored manifests).
    """
    if not isinstance(cells, list):
        return [], []
    out: list[dict[str, Any]] = []
    seen_triples: set[tuple[str, str, str]] = set()
    n_derived = 0
    for i, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue
        model_key, env, baseline, derived = _cell_axes(cell, i)
        if derived:
            triple = (model_key, env, baseline)
            if triple in seen_triples:
                cid = str(cell.get("id") or f"cell_{i}")
                baseline = f"{baseline}__{cid}"
            n_derived += 1
        seen_triples.add((model_key, env, baseline))
        if (
            cell.get("model_key") == model_key
            and cell.get("env") == env
            and cell.get("baseline") == baseline
        ):
            out.append(cell)
        else:
            patched = dict(cell)
            patched["model_key"] = model_key
            patched["env"] = env
            patched["baseline"] = baseline
            out.append(patched)
    notes: list[str] = []
    if n_derived:
        notes.append(
            f"cells.json contract: {n_derived}/{len(out)} cell(s) were missing explicit "
            "model_key/env/baseline axes; the harness derived them from "
            "model/dataset/variant-style fields (falling back to the cell id). "
            "Metrics are keyed under the derived axes. For precise rubric matching, "
            "emit explicit model_key, env, and baseline for every cell in cells.json."
        )
    return out, notes


# ---------------------------------------------------------------------------
# Stage 1 — capacity gate
# ---------------------------------------------------------------------------

def capacity_gate(
    cells: list[dict[str, Any]],
    per_gpu_vram_gb: float,
    *,
    headroom: float = DEFAULT_HEADROOM,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Drop cells that cannot fit on a single GPU, before any launch.

    A cell is dropped when ``per_gpu_vram_gb > 0`` AND
    ``cell["est_vram_gb"] * headroom > per_gpu_vram_gb`` — i.e. its estimated
    footprint, padded by the same headroom the scheduler uses, exceeds one
    card's budget.  The gate is **per-cell**: a model with one env that fits and
    one that does not keeps the fitting env.  A model only lands in
    ``models_skipped`` when EVERY one of its cells exceeds budget (no surviving
    cell carries that ``model_key``).

    Never blocks on unknown capacity:

    * ``per_gpu_vram_gb <= 0`` (capacity unknown) → keep ALL cells, emit nothing.
    * A cell missing ``est_vram_gb`` (or non-numeric) → kept (unknown footprint
      must not be dropped on a guess).

    Args:
        cells:            Cell-description dicts (see module docstring for the
                          per-cell schema).  ``model_key`` and ``est_vram_gb``
                          are read; all other keys pass through untouched.
        per_gpu_vram_gb:  VRAM of a single target GPU, in GiB.  ``<= 0`` means
                          unknown — keep everything.
        headroom:         Multiplier applied to ``est_vram_gb`` before the
                          comparison.  Defaults to :data:`DEFAULT_HEADROOM`
                          (1.25), matching ``OPENRESEARCH_DYNAMIC_GPU_HEADROOM``.

    Returns:
        ``(kept_cells, gap_entries, models_skipped)`` where:

        * ``kept_cells`` — the cells that fit (order preserved).
        * ``gap_entries`` — one dict per fully-dropped ``model_key``::

              {"item": <model_key>,
               "reason": "needs ~{est:.0f}GB > per-GPU budget {budget:.0f}GB "
                         "(headroom {headroom})",
               "kind": "capacity"}

          The scorer reads ``gap["item"]`` (see ``leaf_scorer._collect_gaps``).
        * ``models_skipped`` — deduped, sorted ``model_key`` list for fully-
          dropped models, ready to splice into ``scope.models_skipped``.
    """
    if not isinstance(cells, list):
        return [], [], []

    # Unknown capacity → never block.  Keep everything, emit nothing.
    if not isinstance(per_gpu_vram_gb, (int, float)) or per_gpu_vram_gb <= 0:
        return list(cells), [], []

    budget = float(per_gpu_vram_gb)
    hr = float(headroom) if isinstance(headroom, (int, float)) and headroom > 0 else DEFAULT_HEADROOM

    kept: list[dict[str, Any]] = []
    # Track, per model_key, whether ANY of its cells survived and a representative
    # estimate for the gap message of fully-dropped models.
    kept_models: set[str] = set()
    dropped_est: dict[str, float] = {}

    for cell in cells:
        if not isinstance(cell, dict):
            # Malformed entry — keep it (don't drop on a parse failure); it will
            # surface as a failed leaf in aggregation if it has no usable axes.
            kept.append(cell)
            continue

        model_key = str(cell.get("model_key", "") or "")
        est_raw = cell.get("est_vram_gb")
        est = _coerce_metric(est_raw)

        # Missing/non-numeric estimate → unknown footprint → keep.
        if est is None:
            kept.append(cell)
            if model_key:
                kept_models.add(model_key)
            continue

        if est * hr > budget:
            # This cell is too big for one card.  Record its estimate against the
            # model_key; it is only a *model* skip if no sibling cell survives.
            if model_key:
                dropped_est[model_key] = max(dropped_est.get(model_key, 0.0), est)
            continue

        kept.append(cell)
        if model_key:
            kept_models.add(model_key)

    # A model is skipped only when EVERY one of its cells was dropped.
    fully_dropped = sorted(mk for mk in dropped_est if mk and mk not in kept_models)

    gap_entries: list[dict[str, Any]] = []
    for mk in fully_dropped:
        est = dropped_est[mk]
        gap_entries.append({
            "item": mk,
            "reason": (
                f"needs ~{est:.0f}GB > per-GPU budget {budget:.0f}GB "
                f"(headroom {hr})"
            ),
            "kind": "capacity",
        })

    return kept, gap_entries, fully_dropped


# ---------------------------------------------------------------------------
# Stage 2 — dataset-url preflight
# ---------------------------------------------------------------------------

def default_dataset_probe(url: str, *, timeout_s: float = 5.0) -> bool | None:
    """Bounded HEAD probe of a dataset endpoint.  ``True`` / ``False`` / ``None``.

    The default :func:`dataset_url_preflight` probe — separated out so callers
    (and tests) can wrap or replace it.  Semantics are FAIL-SOFT-biased:

    * ``200`` / any ``3xx`` redirect           → ``True``  (endpoint live)
    * ``404`` / ``410``                          → ``False`` (CONFIRMED dead)
    * any other status (``401``/``403``/``5xx``) → ``None``  (unknown — keep)
    * timeout / DNS / connection / any exception → ``None``  (transient — keep)

    Only a positively-confirmed dead status (404/410) returns ``False``; nothing
    else does.  This is deliberate: an auth gate, a flaky 503, or a DNS blip must
    NOT be read as "dataset gone" and silently de-scope a real environment.

    Args:
        url:        The endpoint to probe.
        timeout_s:  Socket timeout in seconds.

    Returns:
        ``True`` available, ``False`` confirmed-dead, ``None`` unknown/transient.
    """
    if not url or not isinstance(url, str):
        return None
    # Dataset-hub URLs (HuggingFace, Kaggle, …) are resolved by a CLIENT LIBRARY
    # (datasets.load_dataset), NOT a direct HTTP GET — a HEAD 404 on the
    # human-facing page is NOT authoritative (the 2026-05-31 nq_open false-drop
    # that de-scoped a loadable Search-QA env -> capacity_exhausted). Never
    # confirm-dead these; the cell's own load surfaces a real data_load_failure.
    if any(h in url.lower() for h in ("huggingface.co", "hf.co", "kaggle.com")):
        return None
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = int(getattr(resp, "status", 0) or getattr(resp, "code", 0) or 0)
            if 200 <= status < 400:
                return True
            if status in _DEAD_HTTP_STATUSES:
                return False
            return None
    except urllib.error.HTTPError as exc:
        # An HTTPError carries the real status even though urlopen raised.
        status = int(getattr(exc, "code", 0) or 0)
        if status in _DEAD_HTTP_STATUSES:
            return False
        return None
    except Exception:
        # URLError (DNS/connection), socket timeout, bad URL, anything else →
        # unknown.  NEVER False — a transient blip must not drop a live env.
        return None


def dataset_url_preflight(
    cells: list[dict[str, Any]],
    *,
    probe: "Callable[[str], bool | None] | None" = None,
    timeout_s: float = 5.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Drop cells whose ``dataset_url`` is CONFIRMED dead; keep on any doubt.

    Each DISTINCT ``dataset_url`` is probed exactly once (results cached), so a
    matrix with N cells sharing one url makes a single probe call.  A cell is
    dropped only when its url's probe returned ``False`` (confirmed dead, e.g.
    HTTP 404).  Fail-soft on everything else:

    * probe returned ``None`` (unknown/transient) → keep the cell.
    * probe raised                                → keep the cell.
    * cell has no ``dataset_url``                 → keep the cell.

    Dropping an env adds its ``env`` axis to ``environments_skipped`` and emits
    one gap entry; a dropped env is reported once even if many of its cells share
    the dead url.

    Args:
        cells:      Cell-description dicts.  ``dataset_url`` and ``env`` are read.
        probe:      ``probe(url) -> True|False|None`` — injected for tests so no
                    real network is touched.  Defaults to
                    :func:`default_dataset_probe` bound to ``timeout_s``.
        timeout_s:  Socket timeout passed to the default probe (ignored when a
                    custom ``probe`` is supplied).

    Returns:
        ``(kept_cells, gap_entries, environments_skipped)`` where each gap is::

            {"item": <env>,
             "reason": "dataset_url returned 404 (dead endpoint): {url}",
             "kind": "dataset_unavailable"}

        and ``environments_skipped`` is the deduped, sorted list of dropped envs.
    """
    if not isinstance(cells, list):
        return [], [], []

    if probe is None:
        def probe(u: str) -> bool | None:  # type: ignore[misc]
            return default_dataset_probe(u, timeout_s=timeout_s)

    # Probe each distinct url once; cache True/False/None keyed by url.
    probe_cache: dict[str, bool | None] = {}

    def _probe_cached(url: str) -> bool | None:
        if url in probe_cache:
            return probe_cache[url]
        try:
            result = probe(url)
        except Exception:
            # A throwing probe is treated as unknown — fail-soft, keep the cell.
            result = None
        # Normalise to strictly True / False / None.
        if result is True:
            verdict: bool | None = True
        elif result is False:
            verdict = False
        else:
            verdict = None
        probe_cache[url] = verdict
        return verdict

    kept: list[dict[str, Any]] = []
    skipped_envs: list[str] = []
    # env -> the dead url that de-scoped it (first one wins for the message).
    dead_env_url: dict[str, str] = {}

    for cell in cells:
        if not isinstance(cell, dict):
            kept.append(cell)
            continue

        url = cell.get("dataset_url")
        if not url or not isinstance(url, str):
            kept.append(cell)
            continue

        verdict = _probe_cached(url)
        if verdict is False:
            # CONFIRMED dead — drop this cell and remember its env.
            env = str(cell.get("env", "") or "")
            if env:
                skipped_envs.append(env)
                dead_env_url.setdefault(env, url)
            continue

        # True or None → keep (fail-soft).
        kept.append(cell)

    environments_skipped = _dedup_sorted(skipped_envs)
    gap_entries: list[dict[str, Any]] = [
        {
            "item": env,
            "reason": f"dataset_url returned 404 (dead endpoint): {dead_env_url[env]}",
            "kind": "dataset_unavailable",
        }
        for env in environments_skipped
    ]

    return kept, gap_entries, environments_skipped


# ---------------------------------------------------------------------------
# Stage 3 — aggregate into the canonical nested metrics shape
# ---------------------------------------------------------------------------

def _build_ok_leaf(cell_metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Build the leaf for an ``ok`` cell from its flat per-cell metrics dict.

    Pass through every scalar key the cell emitted (``reward_mean``,
    ``steps_run``, …) and guarantee the two keys the scorer always reads:
    ``status`` (forced to ``"ok"``) and ``metric`` (coerced to float|null).
    """
    leaf: dict[str, Any] = {}
    if isinstance(cell_metrics, dict):
        leaf.update(cell_metrics)
    leaf["status"] = "ok"
    leaf["metric"] = _coerce_metric(leaf.get("metric"))
    return leaf


def _build_failed_leaf(
    cell_metrics: dict[str, Any] | None,
    error: Any,
) -> dict[str, Any]:
    """Build the leaf for a failed/oom/error/missing cell.

    Any partial metrics the cell managed to write are preserved (merged under
    the leaf), but ``status`` is forced to ``"failed"`` and ``metric`` defaults
    to ``null`` unless the partial dict carried a numeric one.  The error string
    is truncated to :data:`_MAX_LEAF_ERROR_CHARS`.
    """
    leaf: dict[str, Any] = {}
    if isinstance(cell_metrics, dict):
        leaf.update(cell_metrics)
    leaf["status"] = "failed"
    leaf["metric"] = _coerce_metric(leaf.get("metric"))
    err_text = str(error) if error else "no result"
    leaf["error"] = err_text[:_MAX_LEAF_ERROR_CHARS]
    return leaf


def aggregate_cell_metrics(
    matrix_result: dict[str, dict[str, Any]],
    cells: list[dict[str, Any]],
    *,
    capacity_gaps: list[dict[str, Any]] | None = None,
    dataset_gaps: list[dict[str, Any]] | None = None,
    models_skipped: list[str] | None = None,
    environments_skipped: list[str] | None = None,
) -> dict[str, Any]:
    """Fold per-cell results into the canonical harness ``metrics.json`` shape.

    For every cell in ``cells`` the matching ``matrix_result[cell["id"]]`` record
    decides the leaf:

    * record ``status in {"ok", "skipped"}`` → leaf = the cell's own ``metrics``
      dict (or ``{}``), forced to ``status="ok"`` with a ``metric`` key (null if
      absent).  ``"skipped"`` is the resume case — a prior ok cell reused
      verbatim (``gpu_cell_runner`` Track B) — and aggregates identically to a
      freshly-run ok cell.
    * record ``status in {"oom_failed", "error", "timeout"}`` OR the record is missing →
      leaf = ``{"status": "failed", "metric": null, "error": <record error,
      truncated, or "no result">}``, with any partial ``metrics`` the failed
      cell wrote merged underneath.

    Each leaf is nested at ``per_model[model_key][env][baseline]`` (env keyed
    DIRECTLY under model — no ``per_dataset`` wrapper, matching the real SDAR
    sample and the postflight single-model path).

    Top-level ``status``:

    * ``"complete"`` — every kept cell is ok.
    * ``"partial"``  — at least one ok and at least one failed.
    * ``"failed"``   — no cell is ok (or there are no cells).

    ``scope`` is assembled as:

    * ``models_run`` — sorted distinct ``model_key`` values with ≥1 ok cell.
    * ``models_skipped`` — the passed list, deduped + sorted.
    * ``environments_skipped`` — the passed list, deduped + sorted.
    * ``gaps`` — ``capacity_gaps + dataset_gaps`` concatenated (gap dicts the
      scorer reads via ``item`` / ``name`` / ``id``).

    Defensive throughout: a malformed cell or result is skipped, never raised on.
    The returned dict is plain and JSON-serialisable.

    Args:
        matrix_result:        ``gpu_cell_runner.run_matrix`` output —
                              ``{cell_id: {"status", "metrics", "gpu",
                              "retries", "error"}}``.
        cells:                The cells that were actually run (post-gating).
                              Each provides the ``id`` to look up plus the
                              ``model_key`` / ``env`` / ``baseline`` axes.
        capacity_gaps:        Gap dicts from :func:`capacity_gate` (stage 1).
        dataset_gaps:         Gap dicts from :func:`dataset_url_preflight`
                              (stage 2).
        models_skipped:       ``model_key`` list from stage 1 (capacity).
        environments_skipped: ``env`` list from stage 2 (dataset preflight).

    Returns:
        The canonical nested metrics dict (see module docstring).
    """
    if not isinstance(matrix_result, dict):
        matrix_result = {}

    per_model: dict[str, dict[str, dict[str, Any]]] = {}
    models_with_ok: set[str] = set()
    any_ok = False
    any_failed = False

    for idx, cell in enumerate(cells if isinstance(cells, list) else []):
        if not isinstance(cell, dict):
            continue
        cell_id = cell.get("id")
        model_key = str(cell.get("model_key", "") or "")
        env = str(cell.get("env", "") or "")
        baseline = str(cell.get("baseline", "") or "")
        # A ran cell must NEVER vanish from the aggregate (2026-06-09 All-CNN:
        # 14 paper-grade cells aggregated to per_model={} because the agent's
        # manifest used its own axis names, and this loop silently skipped all
        # of them).  Callers normally pre-normalize via normalize_cell_axes();
        # this is the belt-and-braces layer for direct callers — derive the
        # missing axes instead of dropping measured results.
        if not model_key or not env or not baseline:
            model_key, env, baseline, _ = _cell_axes(cell, idx)

        record = matrix_result.get(cell_id) if isinstance(cell_id, str) else None
        if not isinstance(record, dict):
            record = None

        status = str(record.get("status", "")) if record else ""
        cell_metrics = record.get("metrics") if record else None
        if not isinstance(cell_metrics, dict):
            cell_metrics = None

        if status in ("ok", "skipped"):
            # A "skipped" cell (resume: prior ok cell reused without re-launching,
            # gpu_cell_runner Track B) carries valid prior metrics from its
            # output_dir — it is an ok leaf for every downstream consumer.
            leaf = _build_ok_leaf(cell_metrics)
            models_with_ok.add(model_key)
            any_ok = True
        else:
            # oom_failed / error / timeout / unknown / missing record → failed leaf.
            err = record.get("error") if record else None
            leaf = _build_failed_leaf(cell_metrics, err)
            any_failed = True

        per_model.setdefault(model_key, {}).setdefault(env, {})[baseline] = leaf

    # Top-level status from the ok/failed tallies.
    if any_ok and not any_failed:
        top_status = "complete"
    elif any_ok and any_failed:
        top_status = "partial"
    else:
        top_status = "failed"

    gaps: list[dict[str, Any]] = []
    if capacity_gaps:
        gaps.extend(g for g in capacity_gaps if isinstance(g, dict))
    if dataset_gaps:
        gaps.extend(g for g in dataset_gaps if isinstance(g, dict))

    scope: dict[str, Any] = {
        "models_run": sorted(models_with_ok),
        "models_skipped": _dedup_sorted(models_skipped),
        "environments_skipped": _dedup_sorted(environments_skipped),
        "gaps": gaps,
    }

    return {
        "status": top_status,
        "per_model": per_model,
        "scope": scope,
    }
