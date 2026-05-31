"""Post-run PaperBench rubric leaf scorer.

Grades a reproduction run against a PaperBench rubric.json tree by:
1. Flattening the tree to leaves.
2. LLM-grading leaves in batches against gathered run evidence.
3. Rolling up leaf scores through the weighted tree.
4. Amending final_report.json with the rubric block.

Deterministic invariant gate (paper-hint invariants, 2026-05-29):
When ``score_reproduction`` is called with a non-empty ``invariants`` list
(a list of :class:`backend.agents.schemas.InvariantSpec`), the gate runs
*before* the LLM-graded ``overall_score`` is returned:

  * ``must_not_match`` violation  → ``overall_score`` is capped to 0.0 (hard gate)
  * Any ``must_match`` miss        → ``overall_score`` is capped at 0.5 (soft cap)
  * All invariants pass            → ``overall_score`` is unchanged

Both cases surface a structured ``invariant_results`` list and an
``invariant_gate_applied`` bool in the returned dict so the caller / final
report can show exactly why the score was capped.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Cap applied when a must_not_match invariant fires (hard gate — surrogate model, etc.)
INVARIANT_HARD_CAP: float = 0.0
# Cap applied when a must_match invariant is entirely missing (soft gate — missing
# algorithm token).  Must be > INVARIANT_HARD_CAP so must_not_match always wins.
INVARIANT_SOFT_CAP: float = 0.5

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class LlmClient(Protocol):
    def complete(self, *, system: str, user: str) -> str:
        ...


# ---------------------------------------------------------------------------
# 1. flatten_leaves
# ---------------------------------------------------------------------------


def flatten_leaves(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Recursively collect all leaf nodes (nodes with empty/missing sub_tasks)."""
    children: list[dict[str, Any]] = [
        c for c in (node.get("sub_tasks") or []) if isinstance(c, dict)
    ]
    if not children:
        return [node]
    leaves: list[dict[str, Any]] = []
    for child in children:
        leaves.extend(flatten_leaves(child))
    return leaves


# ---------------------------------------------------------------------------
# 2. roll_up
# ---------------------------------------------------------------------------


def roll_up(
    node: dict[str, Any],
    leaf_scores: dict[str, float],
    skip_set: frozenset[str] = frozenset(),
) -> float:
    """Recursive weighted roll-up.

    Leaf: return leaf_scores.get(node["id"], 0.0), or skip entirely when the
    leaf id is in skip_set (data-unavailable leaves — excluded from BOTH
    numerator AND denominator so they don't drag the parent score down).
    Non-leaf: weighted average of children scores, excluding skipped leaves.

    ``skip_set`` defaults to an empty frozenset so existing callers that pass
    only ``node`` and ``leaf_scores`` are unaffected (backward compat).
    """
    children: list[dict[str, Any]] = [
        c for c in (node.get("sub_tasks") or []) if isinstance(c, dict)
    ]
    if not children:
        lid = str(node.get("id", ""))
        if lid in skip_set:
            # Signal to the parent that this leaf is ineligible.
            # Callers must filter children by skip_set before computing the
            # weighted average — see the non-leaf branch below.
            return None  # type: ignore[return-value]
        return leaf_scores.get(lid, 0.0)

    # Non-leaf: exclude skipped children from both weight sum and weighted sum.
    eligible_children = [
        c for c in children
        if str(c.get("id", "")) not in skip_set
        # A child is skipped when it IS a leaf and its id is in skip_set.
        # For non-leaf children we recurse and check below.
    ]

    # Build (child, subtree_score) pairs; drop children whose entire subtree
    # is fully skipped (roll_up returns None for a skipped leaf, but for a
    # non-leaf intermediate we need a sentinel — use a recursive helper).
    scored_children: list[tuple[dict[str, Any], float]] = []
    for child in children:
        child_score = roll_up(child, leaf_scores, skip_set)
        if child_score is None:
            # Entire subtree is unavailable — exclude from this level too.
            continue
        scored_children.append((child, child_score))

    if not scored_children:
        return None  # type: ignore[return-value]  # entire subtree skipped

    total_weight = sum(float(c.get("weight", 0.0) or 0.0) for c, _ in scored_children)
    if total_weight == 0.0:
        return 0.0

    weighted_sum = sum(
        score * float(c.get("weight", 0.0) or 0.0)
        for c, score in scored_children
    )
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Honesty backstop (C2b)
#
# A run that reached _finalize() without producing measured numeric metrics
# (baseline_metrics={}) is "degraded": the experiment either never ran or ran
# without writing metrics.json. A lenient LLM grader on metric-less evidence
# can still hand out high leaf scores by reading the code; that score does not
# describe a reproduction. Cap each leaf at DEGRADED_LEAF_CEILING so the
# rolled-up overall_score is bounded by the same ceiling.
#
# The 0.35 number is inherited from the verify_against_rubric backstop that
# lived in primitives.py before 2e1ce37 consolidated the in-loop and post-run
# scoring paths through score_reproduction.
# ---------------------------------------------------------------------------

DEGRADED_LEAF_CEILING: float = 0.35

# Minimal field set that distinguishes an RLM-mode final_report from an SDK-mode
# one. Used by _rerender_report_markdown to detect RLM reports without requiring
# ALL RLMFinalReport fields — that prior approach re-broke every time the schema
# gained a new field (regression of T21: primitive_provider + degraded added).
_RLM_SIGNATURE_FIELDS: frozenset[str] = frozenset({"verdict", "baseline_metrics", "paper", "rubric"})


def _is_degraded_run(run_dir: Path) -> bool:
    """Decide whether the run produced no measured metrics.

    A run is degraded when final_report.json exists with baseline_metrics
    empty/missing — the RLMFinalReport contract for "no metrics were measured."
    Missing or unreadable final_report.json is treated as NOT degraded (do not
    cap on uncertainty) so this is safe to call in-loop, before the report has
    been written.

    Callers with a results dict in hand (verify_against_rubric) should NOT
    rely on this auto-detection alone — pass `degraded` explicitly via
    score_reproduction's kwarg so the in-loop signal is correct too.
    """
    report_path = run_dir / "final_report.json"
    if not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — unreadable → don't cap on uncertainty
        return False
    if not isinstance(report, dict):
        return False
    metrics = report.get("baseline_metrics") or {}
    verdict = report.get("verdict", "")
    return (not metrics) or verdict == "failed"


# ---------------------------------------------------------------------------
# Evidence gathering
# ---------------------------------------------------------------------------

_MAX_FILE_BYTES = 6 * 1024          # 6 KB per file
_MAX_TOTAL_EVIDENCE_BYTES = 40 * 1024  # 40 KB total


def _latest_metrics_path(run_dir: Path) -> Path | None:
    """Return the NEWEST-by-mtime metrics.json — the canonical latest experiment.

    A run accumulates one ``code/outputs/<run-id>/metrics.json`` per
    ``run_experiment`` call (including failed/OOM/superseded attempts), plus an
    optional top-level ``code/metrics.json``. Selecting the lexicographically
    first (the old behaviour) reads an ARBITRARY STALE result — e.g. a
    SDAR-loses-to-GRPO attempt that was later improved — so result-match and
    experiment leaves are graded against a superseded outcome. Selecting by
    mtime reads the actual most-recent result. Returns ``None`` when no
    metrics.json exists.
    """
    cands: list[Path] = []
    outputs = run_dir / "code" / "outputs"
    if outputs.exists():
        cands.extend(outputs.rglob("metrics.json"))
    top = run_dir / "code" / "metrics.json"
    if top.exists():
        cands.append(top)
    if not cands:
        return None

    # Prefer the NEWEST metrics that actually carries RESULTS — a real
    # experiment populates per_model and/or comparison. An in-progress or
    # just-created experiment dir (the run keeps iterating) can be newest by
    # mtime yet empty; selecting it would lose both the result AND the scope
    # declaration. Rank (has_results, mtime) so an empty newest loses to the
    # most recent results-bearing metrics; fall back to newest-overall.
    def _rank(p: Path) -> tuple[int, float]:
        has_results = False
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            has_results = bool(d.get("per_model")) or bool(d.get("comparison"))
        except Exception:
            has_results = False
        try:
            mt = p.stat().st_mtime
        except OSError:
            mt = 0.0
        return (1 if has_results else 0, mt)

    return max(cands, key=_rank)


def _gather_evidence(run_dir: Path) -> str:
    """Gather bounded reproduction evidence from a run directory."""
    parts: list[str] = []
    total = 0

    # final_report.json — reproduction_summary + measured metrics + paper id
    # C2a fix: read the RLMFinalReport schema's real keys.  The previous list
    # ("metrics", "paper_title") was a guess at SDK-mode field names; RLM-mode
    # reports carry "baseline_metrics" (dict) and "paper" (dict).  Reading the
    # wrong keys meant every RLM run was graded against evidence with no
    # metrics and no paper identity — the grader had nothing to ground on.
    report_path = run_dir / "final_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            snippet = {
                k: report[k]
                for k in ("reproduction_summary", "baseline_metrics", "verdict", "paper")
                if k in report
            }
            text = f"=== final_report.json (key fields) ===\n{json.dumps(snippet, indent=2)}\n"
            parts.append(text)
            total += len(text)
        except Exception as exc:
            logger.warning("Could not read final_report.json: %s", exc)

    # Latest experiment metrics.json — the actual run RESULTS (per-model scores,
    # the SDAR-vs-GRPO comparison, reward/gate curves). Without this the grader
    # sees only the CODE, never the OUTCOME, so result-match / experiment-execution
    # / data-fidelity leaves score ~0 even when the run succeeded and (e.g.) SDAR
    # beat GRPO. metrics.json is not a priority code extension, so it would
    # otherwise never reach the grader. Read the NEWEST experiment's metrics.
    metrics_path = _latest_metrics_path(run_dir)
    if metrics_path is not None:
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            body = json.dumps(metrics, indent=2)[:_MAX_FILE_BYTES]
            text = f"=== latest experiment metrics.json (measured run results) ===\n{body}\n"
            parts.append(text)
            total += len(text)
        except Exception as exc:
            logger.warning("Could not read latest metrics.json: %s", exc)

    # code/ directory listing
    code_dir = run_dir / "code"
    if code_dir.exists():
        listing_lines: list[str] = []
        for path in sorted(code_dir.rglob("*"))[:200]:
            if path.is_file():
                listing_lines.append(str(path.relative_to(code_dir)))
        listing = "=== code/ listing (first 200 files) ===\n" + "\n".join(listing_lines) + "\n"
        parts.append(listing)
        total += len(listing)

    # Key code files
    if code_dir.exists() and total < _MAX_TOTAL_EVIDENCE_BYTES:
        priority_extensions = {".py", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".txt"}
        for path in sorted(code_dir.rglob("*")):
            if total >= _MAX_TOTAL_EVIDENCE_BYTES:
                break
            if not path.is_file():
                continue
            if path.suffix not in priority_extensions:
                continue
            try:
                raw = path.read_bytes()[:_MAX_FILE_BYTES]
                content = raw.decode("utf-8", errors="replace")
                header = f"\n=== code/{path.relative_to(code_dir)} ===\n"
                chunk = header + content + "\n"
                parts.append(chunk)
                total += len(chunk)
            except Exception:
                pass

    return "".join(parts) if parts else "(no reproduction evidence found)"


# ---------------------------------------------------------------------------
# Pre-filter: data-unavailable leaf detection (PR-κ)
# ---------------------------------------------------------------------------


def _normalise_dataset_name(name: str) -> frozenset[str]:
    """Tokenise a dataset name into a set of lowercase, punctuation-stripped tokens.

    "frey_face" → {"frey", "face"}; "Frey Face" → {"frey", "face"}.
    Used for fuzzy matching between dataset names and leaf descriptions.
    """
    return frozenset(t.lower() for t in re.split(r"[^a-z0-9]+", name.lower()) if t)


def _leaf_mentions_dataset(leaf: dict[str, Any], dataset_tokens: frozenset[str]) -> bool:
    """Return True iff the leaf's id or requirements text contains every token.

    A leaf is linked to a dataset when every token in the normalised dataset name
    appears in the leaf's text (requirements + id).  The in-order subsequence
    requirement from rubric_guard is NOT enforced here — we only need set
    membership because token order is less constrained in leaf descriptions than
    in metric key names.
    """
    text = " ".join([
        str(leaf.get("id", "")),
        str(leaf.get("requirements", "")),
    ]).lower()
    text_tokens = frozenset(t for t in re.split(r"[^a-z0-9]+", text) if t)
    return dataset_tokens.issubset(text_tokens)


def _detect_data_unavailable_leaves(
    leaves: list[dict[str, Any]],
    run_dir: Path,
    metrics_shape: list[dict] | None = None,
) -> set[str]:
    """Return the set of leaf ids that depend on a dataset declared unavailable.

    A leaf is marked unavailable when EITHER of the following runtime signals
    reports a dataset the leaf depends on as unloadable:

    1. ``metrics.json::data_load_failures[]`` — the agent's runtime record of
       "I tried to load this dataset and failed (HTTP 403, licence gate, etc.)".
    2. ``final_report.json::scope.gaps[]`` — the agent's structured declaration
       of datasets that are out of scope for this run.

    Matching strategy:

    * When ``metrics_shape`` (from PR-θ) is present: each MetricPath entry that
      declares a ``rubric_leaf_ids`` list is checked by json_path lookup in
      metrics.json.  If the json_path is absent AND the metric_id or json_path
      contains a failed-dataset token, all declared ``rubric_leaf_ids`` are
      marked unavailable.  This is the authoritative path — no fuzzy guessing.

    * When ``metrics_shape`` is absent or a leaf is not covered by it: fuzzy
      token match between the leaf's id/requirements text and each
      failed-dataset name.

    Anti-gaming: declaring a dataset in scope.gaps without a corresponding
    runtime failure record in data_load_failures does NOT automatically skip a
    leaf — the agent must actually try AND fail (data_load_failures is written
    only by the agent's own exception handler, not prompted by the LLM).
    However, scope.gaps alone IS honoured when the leaf has no matching metric
    in metrics.json — this covers the case where the agent never attempted the
    dataset at all and honestly declared it out of scope.  The conservative
    mode (both signals required) would be over-restrictive: an agent that
    declared a dataset out of scope before even trying is being transparent.

    Returns an empty set when neither signal file exists or is parseable —
    backward-compatible with pre-κ behaviour.
    """
    if not leaves:
        return set()

    # --- Load signals ---
    # Signal 1: data_load_failures from the most recent metrics.json
    failed_datasets: list[str] = []
    metrics_data: dict[str, Any] = {}
    # Read the NEWEST experiment's metrics.json (not the lexicographically-first,
    # which is an arbitrary stale/superseded per-experiment dir — see
    # _latest_metrics_path).
    _mpath = _latest_metrics_path(run_dir)
    if _mpath is not None:
        try:
            metrics_data = json.loads(_mpath.read_text(encoding="utf-8"))
            for entry in metrics_data.get("data_load_failures") or []:
                if isinstance(entry, dict) and entry.get("dataset"):
                    failed_datasets.append(str(entry["dataset"]))
                elif isinstance(entry, str) and entry:
                    failed_datasets.append(entry)
        except Exception:
            pass

    # Signal 1b: failed / skipped MODELS. Graceful degradation (2026-05-30 user
    # mandate) excludes a model's leaves from the rubric — numerator AND
    # denominator — exactly like an unobtainable dataset, so a model that could
    # not be loaded or run does not drag the score to zero. Honest runtime
    # signals: per_model[m].status in {model_load_failed, failed, ...},
    # scope.models_skipped (intentional or forced scope reduction, e.g. the SDAR
    # 7B), and an optional model_load_failures list mirroring data_load_failures.
    failed_models: list[str] = []
    for _m, _mv in (metrics_data.get("per_model") or {}).items():
        if isinstance(_mv, dict) and str(_mv.get("status", "")).lower() in {
            "model_load_failed", "failed", "skipped", "data_unavailable", "unavailable",
        }:
            failed_models.append(str(_m))
    for _m in ((metrics_data.get("scope") or {}).get("models_skipped") or []):
        if isinstance(_m, str) and _m:
            failed_models.append(_m)
    for _entry in metrics_data.get("model_load_failures") or []:
        if isinstance(_entry, dict) and (_entry.get("model") or _entry.get("name")):
            failed_models.append(str(_entry.get("model") or _entry.get("name")))
        elif isinstance(_entry, str) and _entry:
            failed_models.append(_entry)

    # Signal 1c: environments_skipped — the agent's structured declaration that an
    # environment (e.g. ALFWorld / WebShop for a Search-QA-only run) is out of
    # scope.  Mirrors models_skipped: its leaves are excluded from numerator AND
    # denominator exactly like a skipped model (2026-05-31 fix — previously only
    # models_skipped was honoured, so honestly de-scoped environments scored 0.0
    # and dragged the overall score down).
    failed_envs: list[str] = []
    for _e in ((metrics_data.get("scope") or {}).get("environments_skipped") or []):
        if isinstance(_e, str) and _e:
            failed_envs.append(_e)

    # Signal 2: scope.gaps — read from BOTH metrics.json::scope (where the agent
    # writes structured scope, mirroring where models_skipped is already read) AND
    # final_report.json::scope.  Entries may be plain prose strings ("ALFWorld —
    # out of scope") OR structured dicts ({"item": "alfworld", "reason": "..."}).
    # The agent emits the dict form, so both are honoured (2026-05-31 fix —
    # previously only str entries from final_report.json were read, silently
    # dropping every dict-form gap and ignoring metrics.json entirely).
    gap_texts: list[str] = []      # prose form → leading-name extraction below
    gap_items: list[str] = []      # structured form → clean short identifier

    def _collect_gaps(scope_obj: dict | None) -> None:
        for gap in (scope_obj or {}).get("gaps") or []:
            if isinstance(gap, str) and gap:
                gap_texts.append(gap)
            elif isinstance(gap, dict):
                item = gap.get("item") or gap.get("name") or gap.get("id")
                if isinstance(item, str) and item:
                    gap_items.append(item)
                elif isinstance(gap.get("reason"), str) and gap["reason"]:
                    gap_texts.append(gap["reason"])

    _collect_gaps(metrics_data.get("scope"))
    report_path = run_dir / "final_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            _collect_gaps(report.get("scope") or {})
        except Exception:
            pass

    if (not failed_datasets and not gap_texts and not gap_items
            and not failed_models and not failed_envs):
        return set()

    # Normalised token sets for each compact dataset name from data_load_failures.
    # These are short identifiers like "frey_face", "timit", "rcv1" — each
    # tokenises to a small meaningful set.  The check is: do all tokens of the
    # dataset name appear in the leaf description?  (leaf_tokens ⊇ ds_tokens)
    failed_token_sets: list[frozenset[str]] = [
        _normalise_dataset_name(d) for d in failed_datasets if d
    ]

    # For scope.gaps, the text is long prose ("Frey Face: HTTP 403 — licence gated"
    # or "Frey Face dataset not downloaded — licence gated").
    # Tokenising the whole phrase gives a large set; the leaf description is shorter
    # and won't contain all prose tokens.  Instead: extract the leading "dataset name"
    # portion from each gap text — the content words before the first delimiter
    # (":", "—", "-", "http") that are not common English explanation words.
    # "Frey Face: HTTP 403 ..." → "Frey Face" → {"frey", "face"}.
    # "Frey Face dataset not downloaded — ..." → strip stop words → {"frey", "face"}.
    _GAP_STOP = frozenset({
        "dataset", "not", "downloaded", "unavailable", "missing", "data",
        "file", "required", "needed", "access", "gated", "restricted",
        "licence", "license", "institutional", "out", "of", "scope",
        "http", "403", "error", "failed", "load", "loading", "available",
    })
    gap_name_token_sets: list[frozenset[str]] = []
    for g in gap_texts:
        if not g:
            continue
        # Strip everything after the first colon, em-dash, " - ", or " http"
        leading = re.split(r"[:—]| - | http", g, maxsplit=1)[0].strip()
        # Tokenise and drop stop words to isolate the dataset name tokens
        raw_tokens = _normalise_dataset_name(leading)
        name_tokens = raw_tokens - _GAP_STOP
        if name_tokens:
            gap_name_token_sets.append(name_tokens)

    # Failed/skipped model names tokenise like compact dataset ids
    # ("qwen2_5_7b" → {qwen2, 5, 7b}); a leaf is excluded only when it contains
    # ALL of a failed component's tokens (leaf_tokens ⊇ component_tokens), so a
    # 7B-specific leaf matches the failed 7B but a generic "Qwen2.5" leaf does not.
    failed_model_token_sets: list[frozenset[str]] = [
        _normalise_dataset_name(m) for m in failed_models if m
    ]
    # Skipped environments and structured gap items tokenise the same compact way
    # ("alfworld" → {alfworld}, "grpo_baseline_run" → {grpo, baseline, run}); the
    # leaf-token-superset rule means each excludes only leaves specifically about
    # that component, never an in-scope SDAR leaf.
    failed_env_token_sets: list[frozenset[str]] = [
        _normalise_dataset_name(e) for e in failed_envs if e
    ]
    gap_item_token_sets: list[frozenset[str]] = [
        _normalise_dataset_name(g) for g in gap_items if g
    ]

    # Combined signal token sets — datasets, scope.gaps prose + structured items,
    # models, and environments all use the same leaf-token-superset matching logic.
    all_unavailable_token_sets = (
        failed_token_sets
        + gap_name_token_sets
        + gap_item_token_sets
        + failed_model_token_sets
        + failed_env_token_sets
    )

    if not all_unavailable_token_sets:
        return set()

    # --- metrics_shape path (PR-θ authoritative) ---
    unavailable_ids: set[str] = set()

    if metrics_shape:
        from backend.agents.rlm.rubric_guard import _path_resolves  # lazy import

        for mp in metrics_shape:
            if not isinstance(mp, dict):
                continue
            json_path = mp.get("json_path") or ""
            metric_id = mp.get("metric_id") or json_path
            leaf_ids = mp.get("rubric_leaf_ids") or []
            if not json_path or not leaf_ids:
                continue
            # Check if this metric's path is absent from metrics
            if _path_resolves(metrics_data, json_path):
                continue  # metric is present — no skip
            # Check if metric_id / json_path contains any unavailable-dataset token set
            metric_tokens = frozenset(
                t for t in re.split(r"[^a-z0-9]+", (metric_id + " " + json_path).lower()) if t
            )
            for ds_tokens in all_unavailable_token_sets:
                if ds_tokens and ds_tokens.issubset(metric_tokens):
                    for lid in leaf_ids:
                        unavailable_ids.add(str(lid))
                    break

    # --- Fuzzy path: leaves not covered by metrics_shape ---
    # Build the set of leaf ids already handled by metrics_shape
    metrics_shape_covered: set[str] = set()
    if metrics_shape:
        for mp in metrics_shape:
            if isinstance(mp, dict):
                for lid in mp.get("rubric_leaf_ids") or []:
                    metrics_shape_covered.add(str(lid))

    for leaf in leaves:
        lid = str(leaf.get("id", ""))
        if lid in metrics_shape_covered or lid in unavailable_ids:
            continue
        for ds_tokens in all_unavailable_token_sets:
            if ds_tokens and _leaf_mentions_dataset(leaf, ds_tokens):
                unavailable_ids.add(lid)
                break

    return unavailable_ids


# ---------------------------------------------------------------------------
# 3. score_reproduction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a strict research reproducibility judge evaluating whether a paper reproduction \
satisfies specific rubric requirements.

You will be given:
1. Evidence from the reproduction run (code, reports, logs).
2. A batch of rubric leaf tasks, each with an id and requirements text.

For EACH leaf task, output a JSON object with:
- "leaf_id": the task id (string, copy exactly)
- "score": float 0.0 to 1.0 (0.0 = not satisfied at all, 1.0 = fully satisfied)
- "justification": one sentence explaining the score

Output ONLY a JSON array of these objects, no other text. Example:
[{"leaf_id": "abc-123", "score": 0.8, "justification": "The model is implemented but missing dropout."}]

Be conservative: score 0.0 when there is no evidence either way.
"""

_USER_TEMPLATE = """\
## Reproduction evidence

{evidence}

## Rubric leaf tasks to grade (batch {batch_num})

{tasks_json}

Grade EACH task based solely on what the evidence shows. Return a JSON array.
"""


# ---------------------------------------------------------------------------
# Deterministic invariant gate (paper-hint InvariantSpec checks, 2026-05-29)
# ---------------------------------------------------------------------------


def run_invariant_checks(
    invariants: list[Any],
    code_dir: Path,
) -> list[dict[str, Any]]:
    """Run each :class:`~backend.agents.schemas.InvariantSpec` against ``code_dir``.

    Returns a list of per-invariant result dicts, each with the shape::

        {
            "name": str,
            "passed": bool,
            "hard_gate_tripped": bool,      # True iff a must_not_match fired
            "soft_gate_tripped": bool,       # True iff a must_match was empty
            "must_match_evidence": {pat: [<file:line: excerpt>, ...]},
            "must_not_match_violations": {pat: [<file:line: excerpt>, ...]},
            "files_scanned": int,
            "rationale": str,
        }

    The gate logic (applied by the caller, not here):
      * Any ``hard_gate_tripped=True`` → cap ``overall_score`` to 0.0.
      * Any ``soft_gate_tripped=True``  → cap ``overall_score`` to 0.5 (unless
        a hard gate already fires — hard gate wins).

    Fail-soft: if ``code_dir`` does not exist or ``invariants`` is empty, returns [].
    Pattern compilation errors are skipped per-invariant (the schema validator
    already rejects malformed patterns, so this is a defensive last resort).
    """
    if not invariants or not code_dir.exists():
        return []

    # Collect Python source files matching each invariant's file_glob.
    # We do this once per invariant because globs may differ, though in practice
    # SDAR uses the default "**/*.py" for all invariants.
    results: list[dict[str, Any]] = []

    for inv in invariants:
        name: str = getattr(inv, "name", str(inv))
        rationale: str = getattr(inv, "rationale", "")
        file_glob: str = getattr(inv, "file_glob", "**/*.py") or "**/*.py"
        must_match_pats: list[str] = list(getattr(inv, "must_match", []) or [])
        must_not_pats: list[str] = list(getattr(inv, "must_not_match", []) or [])

        # Gather matching files.
        try:
            matched_files = list(code_dir.rglob(file_glob.lstrip("/")))
        except Exception:
            matched_files = []

        # Only look at regular files; skip __pycache__ and compiled bytecode.
        source_files = [
            f for f in matched_files
            if f.is_file()
            and "__pycache__" not in f.parts
            and not f.name.endswith(".pyc")
        ]
        files_scanned = len(source_files)

        # Read all source texts upfront (bounded: skip files > 1 MB each).
        _MAX_FILE_BYTES = 1024 * 1024
        file_contents: list[tuple[Path, str]] = []
        for fpath in source_files:
            try:
                raw = fpath.read_bytes()
                if len(raw) > _MAX_FILE_BYTES:
                    raw = raw[:_MAX_FILE_BYTES]
                file_contents.append((fpath, raw.decode("utf-8", errors="replace")))
            except OSError:
                pass

        # ---- must_match: at least ONE pattern must appear in at least one file
        # (OR semantics across patterns — mirrors InvariantSpec docstring).
        # Soft gate fires only when the ENTIRE must_match list has zero matches
        # across ALL patterns.  A list with two alternatives (e.g. ".detach()"
        # OR "torch.no_grad") is satisfied by finding either alternative.
        must_match_evidence: dict[str, list[str]] = {}
        _any_must_match_hit = False

        for pat in must_match_pats:
            evidence_lines: list[str] = []
            try:
                compiled = re.compile(pat, re.MULTILINE)
            except re.error:
                # Defensive: schema validator prevents this, but skip gracefully.
                continue
            for fpath, text in file_contents:
                for lineno, line in enumerate(text.splitlines(), 1):
                    if compiled.search(line):
                        rel = fpath.relative_to(code_dir)
                        excerpt = line.strip()[:120]
                        evidence_lines.append(f"{rel}:{lineno}: {excerpt}")
            must_match_evidence[pat] = evidence_lines
            if evidence_lines:
                _any_must_match_hit = True

        # Soft gate: fires when must_match is non-empty AND zero patterns matched.
        soft_gate_tripped = bool(must_match_pats) and not _any_must_match_hit

        # ---- must_not_match: no file may match any pattern. ----
        must_not_violations: dict[str, list[str]] = {}
        hard_gate_tripped = False

        for pat in must_not_pats:
            violation_lines: list[str] = []
            try:
                compiled = re.compile(pat, re.MULTILINE)
            except re.error:
                continue
            for fpath, text in file_contents:
                for lineno, line in enumerate(text.splitlines(), 1):
                    if compiled.search(line):
                        rel = fpath.relative_to(code_dir)
                        excerpt = line.strip()[:120]
                        violation_lines.append(f"{rel}:{lineno}: {excerpt}")
            if violation_lines:
                must_not_violations[pat] = violation_lines
                hard_gate_tripped = True

        passed = (not hard_gate_tripped) and (not soft_gate_tripped)
        results.append({
            "name": name,
            "passed": passed,
            "hard_gate_tripped": hard_gate_tripped,
            "soft_gate_tripped": soft_gate_tripped,
            "must_match_evidence": must_match_evidence,
            "must_not_match_violations": must_not_violations,
            "files_scanned": files_scanned,
            "rationale": rationale,
        })

    return results


def _apply_invariant_gate(
    overall_score: float,
    invariant_results: list[dict[str, Any]],
) -> tuple[float, bool]:
    """Apply the deterministic invariant gate to ``overall_score``.

    Returns ``(gated_score, gate_applied)`` where ``gate_applied`` is True iff
    any invariant tripped (hard or soft).  The caller stores this in the score
    dict as ``invariant_gate_applied``.

    Gate precedence (hard wins over soft):
      * Any ``hard_gate_tripped``  → 0.0
      * Any ``soft_gate_tripped``  → min(overall_score, INVARIANT_SOFT_CAP)
      * All pass                   → overall_score unchanged
    """
    has_hard = any(r.get("hard_gate_tripped") for r in invariant_results)
    has_soft = any(r.get("soft_gate_tripped") for r in invariant_results)

    if has_hard:
        return INVARIANT_HARD_CAP, True
    if has_soft:
        return min(overall_score, INVARIANT_SOFT_CAP), True
    return overall_score, False


def score_reproduction(
    rubric_tree: dict[str, Any],
    run_dir: Path,
    llm_client: LlmClient,
    *,
    batch_size: int = 15,
    rubric_source: str = "paperbench_bundle",
    degraded: bool | None = None,
    metrics_shape: list[dict] | None = None,
    invariants: list[Any] | None = None,
) -> dict[str, Any]:
    """Grade a reproduction run against a PaperBench rubric tree.

    Returns a dict with overall_score, leaf_count, graded, rubric_source,
    leaf_scores, degraded, coverage_pct, eligible_count, unavailable_count,
    target_score, invariant_results, and invariant_gate_applied.

    ``rubric_source`` is passed through to the result dict unchanged — callers set
    it to "generated" when the rubric was derived at run-time rather than from a
    vendored bundle.

    ``degraded`` (C2b): when True, every leaf score is capped at
    DEGRADED_LEAF_CEILING (0.35) before roll-up — the honesty backstop for runs
    that produced no measured metrics. ``None`` (default) auto-detects via
    :func:`_is_degraded_run` (reads ``final_report.json`` for an empty
    ``baseline_metrics``). Callers with a results dict in hand should pass
    ``degraded`` explicitly so the in-loop case (no final_report.json on disk
    yet) is also capped.

    ``coverage_pct`` (β2 / κ): fraction of *eligible* leaves that received a
    real LLM grade (0.0–1.0). "Eligible" means not marked data-unavailable by
    :func:`_detect_data_unavailable_leaves`. On degraded runs this is 0.0.
    When 3 of 4 leaves are graded and 1 is skipped as unavailable, eligible=3,
    coverage_pct = 3/3 = 1.0 — not 3/4 = 0.75. This reflects real grading
    fidelity rather than penalising the run for datasets it couldn't reach.

    ``metrics_shape`` (PR-θ): agent-declared metric paths from
    ``ReproductionContract.metrics_shape``. When provided, leaf unavailability
    detection uses exact json_path lookup rather than fuzzy text matching —
    more precise and less likely to false-positive.

    ``invariants`` (paper-hint gate, 2026-05-29): a list of
    :class:`~backend.agents.schemas.InvariantSpec` objects from
    ``PaperHint.invariants``.  When provided and non-empty, the invariant gate
    runs *after* LLM grading and *before* the score dict is returned:

      * A ``must_not_match`` violation (e.g. surrogate model detected) is a
        hard gate — ``overall_score`` is forced to ``INVARIANT_HARD_CAP`` (0.0).
      * A ``must_match`` miss (e.g. sigmoid gate absent) is a soft gate —
        ``overall_score`` is capped at ``INVARIANT_SOFT_CAP`` (0.5).
      * Hard gate always wins over soft.
      * All pass → ``overall_score`` unchanged.

    The structured ``invariant_results`` list and ``invariant_gate_applied``
    bool are always present in the returned dict (empty / False when no
    invariants are provided) so downstream consumers can surface the gate reason.
    """
    leaves = flatten_leaves(rubric_tree)
    evidence = _gather_evidence(run_dir)
    if degraded is None:
        degraded = _is_degraded_run(run_dir)

    # Run invariant checks upfront so they apply on both the degraded and
    # normal paths.  The code dir is run_dir/code/ (the implement_baseline
    # output contract).  Fail-soft: if the dir doesn't exist, returns [].
    code_dir = run_dir / "code"
    invariant_results: list[dict[str, Any]] = run_invariant_checks(
        invariants or [], code_dir
    )

    leaf_scores: dict[str, float] = {}
    leaf_score_records: list[dict[str, Any]] = []
    graded_count = 0

    if degraded:
        for leaf in leaves:
            lid = str(leaf.get("id", ""))
            leaf_scores[lid] = 0.0
            leaf_score_records.append(
                {
                    "id": lid,
                    "score": 0.0,
                    "justification": "degraded_no_metrics",
                }
            )

        raw_target = rubric_tree.get("target_score")
        try:
            target_score: float | None = (
                None if raw_target is None else max(0.0, min(1.0, float(raw_target)))
            )
        except (TypeError, ValueError):
            target_score = None

        degraded_overall = roll_up(rubric_tree, leaf_scores)
        # Apply invariant gate even on degraded runs so the hard gate (surrogate
        # model) is visible in the result dict; score is already 0.0 on degraded
        # so the gate is a no-op numerically, but invariant_results still carries
        # the violation records.
        _inv_score, _inv_gate = _apply_invariant_gate(
            degraded_overall if degraded_overall is not None else 0.0,
            invariant_results,
        )
        return {
            "overall_score": _inv_score,
            "leaf_count": len(leaves),
            "graded": graded_count,
            "rubric_source": rubric_source,
            "leaf_scores": leaf_score_records,
            "degraded": True,
            "coverage_pct": 0.0,
            "eligible_count": len(leaves),
            "unavailable_count": 0,
            "target_score": target_score,
            "invariant_results": invariant_results,
            "invariant_gate_applied": _inv_gate,
        }

    # PR-κ: pre-filter leaves that depend on unavailable datasets.
    # These are excluded from LLM grading and from both numerator AND
    # denominator of the roll-up — they don't drag the score down.
    unavailable_ids: set[str] = _detect_data_unavailable_leaves(
        leaves, run_dir, metrics_shape
    )
    skip_set: frozenset[str] = frozenset(unavailable_ids)

    eligible_count = len(leaves) - len(unavailable_ids)

    # Grade only the eligible leaves.
    eligible_leaves = [l for l in leaves if str(l.get("id", "")) not in unavailable_ids]

    # Build the list of batches first (no LLM calls yet).
    batches: list[tuple[int, list[dict[str, Any]]]] = []
    for batch_num, start in enumerate(range(0, len(eligible_leaves), batch_size), 1):
        batch = eligible_leaves[start : start + batch_size]
        if not batch:
            continue
        batches.append((batch_num, batch))

    def _grade_batch(batch_num: int, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build prompt, call LLM, parse response for one batch. Thread-safe."""
        tasks_payload = [
            {"leaf_id": str(leaf.get("id", "")), "requirements": str(leaf.get("requirements", ""))}
            for leaf in batch
        ]
        user_msg = _USER_TEMPLATE.format(
            evidence=evidence,
            tasks_json=json.dumps(tasks_payload, indent=2),
            batch_num=batch_num,
        )
        try:
            raw = llm_client.complete(system=_SYSTEM_PROMPT, user=user_msg)
            return _parse_batch_response(raw, batch)
        except Exception as exc:
            logger.warning(
                "Batch %d LLM call failed (%s); defaulting all %d leaves to 0.0",
                batch_num,
                exc,
                len(batch),
            )
            return [
                {
                    "id": str(leaf.get("id", "")),
                    "score": 0.0,
                    "justification": "batch_error",
                    "_graded": False,
                }
                for leaf in batch
            ]

    # Submit all batches concurrently; width ≤8 avoids rate-limit bursts.
    # I12: explicit shutdown(wait=False) so a wedged batch cannot block cleanup.
    max_workers = min(len(batches), 8) if batches else 1
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_batch: dict[concurrent.futures.Future[list[dict[str, Any]]], tuple[int, list[dict[str, Any]]]] = {
            executor.submit(_grade_batch, batch_num, batch): (batch_num, batch)
            for batch_num, batch in batches
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            results = future.result()  # exceptions already handled inside _grade_batch
            for rec in results:
                lid = rec["id"]
                score = rec["score"]
                # C2b: clamp degraded leaves to the honesty ceiling before storing
                # so the rolled-up overall_score, the returned leaf_score_records,
                # and any "weak leaves" surface all reflect the cap consistently.
                if degraded and score > DEGRADED_LEAF_CEILING:
                    score = DEGRADED_LEAF_CEILING
                leaf_scores[lid] = score
                leaf_score_records.append(
                    {"id": lid, "score": score, "justification": rec["justification"]}
                )
                if rec.get("_graded", True):
                    graded_count += 1
    finally:
        executor.shutdown(wait=False)

    # PR-κ: append skipped-data-unavailable records.
    # score=None signals "unscored" (not 0) so downstream consumers can
    # distinguish missing data from failing data.  These records are NOT
    # added to leaf_scores — that dict gates the roll_up computation.
    for lid in sorted(unavailable_ids):  # sorted for deterministic output
        leaf_score_records.append({
            "id": lid,
            "score": None,  # explicitly unscored — NOT 0
            "justification": "data_unavailable: dataset declared in data_load_failures or scope.gaps",
            "state": "skipped_data_unavailable",
        })

    # PR-κ: pass skip_set to roll_up so skipped leaves are excluded from BOTH
    # numerator AND denominator at every level of the rubric tree.
    overall_score_raw = roll_up(rubric_tree, leaf_scores, skip_set)
    overall_score = overall_score_raw if overall_score_raw is not None else 0.0

    # C2c: surface target_score so amend_final_report can compute meets_target
    # honestly. None when the rubric tree has no target — never fabricate.
    raw_target = rubric_tree.get("target_score")
    try:
        target_score: float | None = (
            None if raw_target is None else max(0.0, min(1.0, float(raw_target)))
        )
    except (TypeError, ValueError):
        target_score = None

    # β2/κ: coverage_pct = fraction of *eligible* leaves that got a real LLM
    # grade.  Eligible = total - unavailable.  Ungraded (batch_error) leaves
    # count against coverage; skipped (data_unavailable) leaves do not.
    coverage_pct: float = (graded_count / eligible_count) if eligible_count > 0 else 1.0

    # Paper-hint invariant gate (2026-05-29): apply deterministic regex gate
    # AFTER LLM grading, BEFORE returning.  This is the primary scoring gate —
    # the LLM score stands only if all invariants pass.
    gated_overall, inv_gate_applied = _apply_invariant_gate(overall_score, invariant_results)
    if inv_gate_applied:
        logger.info(
            "score_reproduction: invariant gate applied — overall_score %.3f → %.3f "
            "(%d hard, %d soft trips)",
            overall_score,
            gated_overall,
            sum(1 for r in invariant_results if r.get("hard_gate_tripped")),
            sum(1 for r in invariant_results if r.get("soft_gate_tripped")),
        )

    return {
        "overall_score": gated_overall,
        "leaf_count": len(leaves),
        "graded": graded_count,
        "rubric_source": rubric_source,
        "leaf_scores": leaf_score_records,
        "degraded": degraded,
        "coverage_pct": coverage_pct,
        "eligible_count": eligible_count,
        "unavailable_count": len(unavailable_ids),
        "target_score": target_score,
        "invariant_results": invariant_results,
        "invariant_gate_applied": inv_gate_applied,
    }


def _parse_batch_response(
    raw: str, batch: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Parse LLM batch response robustly. Ungraded/malformed leaves -> 0.0."""
    batch_ids = {str(leaf.get("id", "")): leaf for leaf in batch}
    results: dict[str, dict[str, Any]] = {}

    # Try to extract JSON array from response
    raw = raw.strip()
    try:
        from backend.agents.rlm.primitives import _extract_json_array
        parsed = _extract_json_array(raw)
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                lid = str(item.get("leaf_id", ""))
                if not lid or lid not in batch_ids:
                    continue
                try:
                    score = max(0.0, min(1.0, float(item.get("score", 0.0))))
                except (TypeError, ValueError):
                    score = 0.0
                justification = str(item.get("justification", ""))
                results[lid] = {"id": lid, "score": score, "justification": justification, "_graded": True}
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Could not parse batch response as JSON: %s", exc)

    # Fill in any missing leaves with 0.0
    out: list[dict[str, Any]] = []
    for lid in batch_ids:
        if lid in results:
            out.append(results[lid])
        else:
            out.append({"id": lid, "score": 0.0, "justification": "ungraded", "_graded": False})
    return out


# ---------------------------------------------------------------------------
# 4. amend_final_report
# ---------------------------------------------------------------------------


def amend_final_report(run_dir: Path, score: dict[str, Any]) -> None:
    """Load final_report.json, set its rubric field, write back atomically.

    Also re-renders final_report.md so ``GET /runs/{id}/final-report`` (which
    serves the markdown) reflects this authoritative leaf score — not the stale
    in-loop ``verify_against_rubric`` score the run wrote at finish time.
    """
    report_path = run_dir / "final_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {}

    # C2c: compute meets_target from the real target_score score_reproduction
    # now threads through. When the rubric tree has no target_score (e.g. a
    # self-generated arXiv rubric without a configured target), both
    # target_score and meets_target are written as null — never a fabricated
    # False, which used to flip a legitimate high score to "✘ below target".
    target_score = score.get("target_score")
    if target_score is None:
        meets_target: bool | None = None
    else:
        meets_target = bool(score["overall_score"] >= target_score)

    # T5: preserve the in-loop tree-rubric areas list so the markdown areas
    # table is not silently dropped when we replace report["rubric"].
    previous_rubric = report.get("rubric", {}) or {}
    report["rubric"] = {
        "overall_score": score["overall_score"],
        "rubric_source": score.get("rubric_source", "paperbench_bundle"),
        "leaf_count": score["leaf_count"],
        "graded": score["graded"],
        "target_score": target_score,
        "meets_target": meets_target,
        # C2b: surface the degraded flag so the UI / human reviewer can see
        # *why* a low score was reached. False/missing → run was honest.
        "degraded": bool(score.get("degraded", False)),
        # β2: coverage_pct — fraction of leaves that received a real grade.
        "coverage_pct": float(score.get("coverage_pct", 1.0) or 1.0),
        # β3: preserve compute_adjusted_score + compute_scope from the in-loop
        # verify_against_rubric call (which applied floor-anchored scoring).
        # Falls back to overall_score on max-mode or legacy runs (always-emit).
        "compute_adjusted_score": previous_rubric.get(
            "compute_adjusted_score",
            score["overall_score"],
        ),
        "compute_scope": previous_rubric.get("compute_scope"),
        "areas": previous_rubric.get("areas", []),
        # Paper-hint invariant gate (2026-05-29): surface per-invariant
        # pass/fail so the human reviewer can see exactly which algorithmic
        # invariants tripped and why the score was capped.
        "invariant_results": score.get("invariant_results", []),
        "invariant_gate_applied": bool(score.get("invariant_gate_applied", False)),
    }

    # Reconcile the self-reported verdict against the authoritative leaf score.
    # Symptom: the `ftrl` run wrote verdict="reproduced" at overall_score=0.0.
    # This must happen BEFORE the atomic write and before _rerender_report_markdown
    # so the markdown re-render picks up the corrected verdict automatically.
    if "verdict" in report:
        try:
            from backend.agents.rlm.report import reconcile_verdict_with_score  # lazy import
            report["verdict"] = reconcile_verdict_with_score(
                report["verdict"], score["overall_score"]
            )
        except Exception as exc:  # noqa: BLE001 — reconciliation is best-effort
            logger.warning(
                "amend_final_report: verdict reconciliation failed (%s) — "
                "verdict may be inconsistent with rubric score",
                exc,
            )

    tmp_fd, tmp_path = tempfile.mkstemp(dir=run_dir, prefix=".final_report_", suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        os.replace(tmp_path, report_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    _rerender_report_markdown(run_dir, report)


def _rerender_report_markdown(run_dir: Path, report: dict[str, Any]) -> None:
    """Re-render final_report.md from an amended RLM report dict.

    The post-run leaf scorer updates final_report.json's rubric block; the
    markdown the HTTP layer serves must stay consistent with it. Only RLM-mode
    reports are re-rendered — the markdown renderer is RLM-specific; for any
    other report shape (or a missing markdown file) this is a no-op.
    """
    md_path = run_dir / "final_report.md"
    if not md_path.exists():
        return
    try:
        # Lazy import — keeps backend.evals import-light and breaks no cycle.
        from backend.agents.rlm.report import RLMFinalReport, _render_markdown

        # Detect RLM-mode reports by signature fields, not by full-set equality —
        # the schema can grow without breaking this re-render path (regression of T21).
        if not _RLM_SIGNATURE_FIELDS.issubset(report.keys()):
            return  # not an RLM-mode report — leave its markdown untouched
        all_fields = set(RLMFinalReport.model_fields)
        obj = RLMFinalReport(**{k: v for k, v in report.items() if k in all_fields})
        md = _render_markdown(obj)
    except Exception as exc:  # noqa: BLE001 — markdown refresh is best-effort
        logger.warning(
            "amend_final_report: could not re-render final_report.md (%s) — "
            "it may show a stale rubric score",
            exc,
        )
        return
    tmp_fd, tmp_path = tempfile.mkstemp(dir=run_dir, prefix=".final_report_", suffix=".md")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(md)
        os.replace(tmp_path, md_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
