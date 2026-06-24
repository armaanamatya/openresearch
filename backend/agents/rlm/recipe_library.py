"""Tier-B cross-run POSITIVE recipe memory (``OPENRESEARCH_POSITIVE_RECIPES``).

Recipes are **principle-level structured patterns** — NOT champion-artifact copies —
keyed by **paper-class** (transfer), admitted ONLY on Tier-1 deterministic + validator
evidence.  The LLM grade is **never** the admission signal (the red line, §3.1).

Admission gate (all four conditions must hold):
  1. Report-level ``evidence_gate`` passed (``final_report.json.evidence_gate_passed``).
  2. A ``success=True`` ``run_experiment`` row exists in ``experiment_runs.jsonl``.
  3. Validator verdict is ``clean`` — OR is ``unavailable`` AND the deterministic
     floor passed (condition 1+2+4 satisfied).
  4. A deterministic ``meets_target`` predicate holds: the measured top-level numeric
     in the report's ``rubric`` block meets or exceeds the paper's ``target_score``.

The store is ``runs/_recipes/<paper_class>.json`` — atomic temp + ``os.replace``,
capped at ``MAX_RECIPES`` per class, novelty-deduped (hash of ``problem_sig``), and
staleness-retired at ``RETIRE_STALENESS`` runs without a matching class.

Default OFF: when the flag is unset every public function is a no-op / returns "".
Fail-soft throughout; recipe bookkeeping must never break a run's finalize path.

THE RED LINE (§3.1): this module reads the following grade-derived fields ONLY for
copying into the report-stamp metadata and NOWHERE else for admission decisions:
  • ``overall_score``  (line below, copy-into-report (red-line allowlisted))

Every OTHER admission decision is keyed strictly on Tier-1/2 predicates.

Static-import guard test (test_recipe_library.py::test_red_line_static_import) scans
this source and asserts that ``overall_score``, ``median_score``,
``compute_adjusted_score``, and ``rubric_score`` appear only on an explicitly
allowlisted line (containing the ``# copy-into-report (red-line allowlisted)``
comment).  Any accidental read in a guard branch will fail the guard test.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MAX_RECIPES = 10          # hard cap per paper class
MAX_SUMMARY_CHARS = 200   # solution_sig.technique_summary cap
RETIRE_STALENESS = 5      # runs without matching class before a recipe is retired
INJECT_TOP_K = 2          # max recipes injected into implementer guidance

# Grade-field names that are red-line restricted (only allowed on copy-into-report
# allowlisted lines). Used by the static-import guard test.
GRADE_FIELDS = frozenset({
    "overall_score",
    "median_score",
    "compute_adjusted_score",
    "rubric_score",
})

# Rubric-shape bucket fallback keys when no PAPER_HINTS class matches.
_SHAPE_BUCKETS = {
    "rl_agent": frozenset({"reward", "return", "gate", "policy"}),
    "image_classification": frozenset({"test_error_pct", "accuracy", "cifar", "resnet"}),
    "language_model": frozenset({"perplexity", "nll", "bleu", "rouge", "loss"}),
    "optimizer_comparison": frozenset({"adam", "sgd", "adagrad", "rmsprop", "optimizer"}),
    "compression": frozenset({"speedup", "compression", "pruning", "retention"}),
}


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Recipe:
    """A principle-level structured pattern admitted on Tier-1 + validator evidence.

    Fields
    ------
    problem_sig : dict
        Model/dataset/task axes that characterise the class of problem.
    solution_sig : dict
        Key hyperparameters + cells.json shape + ≤200-char technique summary
        + code-path pointer.  Agent prose is NEVER allowed here.
    evidence_key : str
        Evidence fingerprint from the admitting run.
    paper_class : str
        Coarse paper-class key used for storage and retrieval (transfer).
    """
    problem_sig: dict = field(default_factory=dict)
    solution_sig: dict = field(default_factory=dict)
    evidence_key: str = ""
    paper_class: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Feature gate
# ──────────────────────────────────────────────────────────────────────────────
def positive_recipes_enabled() -> bool:
    """Return True iff ``OPENRESEARCH_POSITIVE_RECIPES`` opts the feature ON.

    Default-OFF: any unset/empty/falsey value disables the feature entirely.
    """
    return os.environ.get("OPENRESEARCH_POSITIVE_RECIPES", "").strip().lower() in (
        "1", "on", "true", "yes",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Paper-class derivation
# ──────────────────────────────────────────────────────────────────────────────
def derive_paper_class(
    *,
    arxiv_id: str | None = None,
    paper_hints: dict | None = None,
    rubric: dict | None = None,
) -> str:
    """Derive a coarse paper-class key for storage keying and transfer.

    Strategy (first match wins):
    1. If ``paper_hints`` contains a mapping of arxiv_id → class-name, use it.
    2. If ``arxiv_id`` is a known PAPER_HINTS entry, use its canonical name.
    3. Inspect the rubric's leaf task-categories for a shape-bucket match.
    4. Fallback: ``"generic"``.

    The ``paper_hints`` argument accepts the dict returned by
    ``paper_hints.PAPER_HINTS`` (keys are arXiv IDs); the function only reads
    the ``.guidance`` and ``.default_scope`` fields for structural pattern
    matching — never for prose injection.

    No imports from ``paper_hints.py`` are taken here to keep the module
    stdlib-only; the caller supplies the pre-loaded dict.
    """
    # 1. Caller may supply an explicit class name under a special "__class__" key.
    if paper_hints and "__class__" in paper_hints:
        cls = str(paper_hints["__class__"]).strip()
        if cls:
            return cls

    # 2. Map known arXiv IDs to canonical class names.
    _ARXIV_TO_CLASS: dict[str, str] = {
        "2605.15155": "sdar_agentic_rl",       # SDAR — the reference paper
        "1412.6806": "image_classification",    # All-CNN
        "1512.03385": "image_classification",   # ResNet
        "1412.6980": "optimizer_comparison",    # Adam
        "2511.14582": "compression",            # OmniZip
    }
    if arxiv_id:
        normalized = arxiv_id.strip().lower().split("v")[0]
        if normalized in _ARXIV_TO_CLASS:
            return _ARXIV_TO_CLASS[normalized]

    # 3. Rub rubric leaf task-categories against shape buckets.
    if rubric:
        category_tokens: set[str] = set()
        _collect_categories(rubric, category_tokens)
        all_tokens = " ".join(category_tokens).lower()
        for bucket, keywords in _SHAPE_BUCKETS.items():
            if any(kw in all_tokens for kw in keywords):
                return bucket

    return "generic"


def _collect_categories(node: Any, out: set[str]) -> None:
    """Recursively collect task_category strings from a rubric tree."""
    if isinstance(node, dict):
        cat = node.get("task_category") or node.get("category") or ""
        if cat:
            out.add(str(cat).lower())
        for v in node.values():
            _collect_categories(v, out)
    elif isinstance(node, list):
        for item in node:
            _collect_categories(item, out)


# ──────────────────────────────────────────────────────────────────────────────
# Store helpers
# ──────────────────────────────────────────────────────────────────────────────
def _store_path(runs_root: Path | str, paper_class: str) -> Path:
    safe = "".join(c for c in paper_class if c.isalnum() or c in "._-")[:80]
    return Path(runs_root) / "_recipes" / f"{safe}.json"


def _problem_sig_hash(problem_sig: dict) -> str:
    """Stable hash of the problem signature for novelty dedup."""
    canonical = json.dumps(problem_sig, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def _read_store(path: Path) -> list[dict[str, Any]]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [r for r in data if isinstance(r, dict)]
    except Exception:
        pass
    return []


def _write_store(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomic temp + os.replace write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ──────────────────────────────────────────────────────────────────────────────
# Admission-gate helpers (Tier-1 predicates only; no grade fields)
# ──────────────────────────────────────────────────────────────────────────────
def _evidence_gate_passed(report: dict) -> bool:
    """True iff the report signals the evidence gate passed.

    We accept any of:
    - ``report["evidence_gate_passed"] == True``
    - ``report["evidence_gate"] == "passed"``
    - ``report.get("validation", {}).get("evidence_gate") in {"passed", True}``
    """
    if report.get("evidence_gate_passed") is True:
        return True
    if report.get("evidence_gate") == "passed":
        return True
    if isinstance(report.get("validation"), dict):
        val = report["validation"].get("evidence_gate")
        if val in ("passed", True):
            return True
    return False


def _has_success_ledger_row(project_dir: Path) -> bool:
    """Return True iff ``experiment_runs.jsonl`` has at least one ``success=True`` row."""
    path = Path(project_dir) / "experiment_runs.jsonl"
    try:
        if not path.exists():
            return False
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("success") is True:
                return True
    except Exception:
        pass
    return False


def _validator_ok(validator_verdict: Any, *, floor_passed: bool) -> bool:
    """True iff the validator verdict allows admission.

    Rules:
    - ``clean``      → admit.
    - ``unavailable`` + floor_passed → admit (Tier-1 floor is the backstop).
    - ``vetoed``     → NEVER admit (red line).
    - ``None``/absent → treat as ``unavailable``.
    """
    if validator_verdict is None:
        return floor_passed

    # Accept a plain dict with a "status" key OR an object with a .status attr.
    if isinstance(validator_verdict, dict):
        status = str(validator_verdict.get("status", "unavailable")).lower()
    else:
        status = str(getattr(validator_verdict, "status", "unavailable")).lower()

    if status == "clean":
        return True
    if status == "vetoed":
        return False  # red-line veto — never admit
    # "unavailable" or anything unknown falls back to floor_passed
    return floor_passed


def _deterministic_meets_target(report: dict, project_dir: Path) -> bool:
    """True iff deterministic MEASURED evidence proves the run met its target.

    Keyed ONLY on Tier-1 deterministic evidence — NEVER on grade-derived score
    fields (overall_score / median_score / rubric_score / compute_adjusted_score).
    Those are the forbidden reward-hacking vectors (red line §3.1).

    CONSERVATIVE by design (spec §9.2 requires "measured metric vs the paper's
    claimed target"): a recipe captures a SUCCESSFUL pattern, so a real-but-below-
    target run must NOT be admitted. Returns True iff EITHER:
      1. ``report["deterministic_meets_target"] is True`` — a harness-set boolean
         from the measured-vs-claimed layer (already deterministic; trust it); OR
      2. numeric ``report["paper_claimed_target"]`` AND ``report["measured_headline"]``
         are both present, a real non-degenerate ``code/metrics.json`` exists on disk
         (the precondition), AND ``measured_headline >= paper_claimed_target``.

    When NO claimed target is available we CANNOT deterministically prove the run
    met its target — so we return False (better no recipe than a below-target one).
    We never fall back to "real metrics exist" (that proves a run happened, not that
    it succeeded) and never to the LLM grade. Fail-soft throughout → False.
    """
    # 1. Explicit harness-set deterministic flag.
    if report.get("deterministic_meets_target") is True:
        return True

    # 2. A real measured-vs-claimed-target comparison. Without BOTH numeric values
    #    we cannot prove meets_target deterministically → do not admit.
    claimed = report.get("paper_claimed_target")
    measured = report.get("measured_headline")
    if claimed is None or measured is None:
        return False
    try:
        claimed_f = float(claimed)
        measured_f = float(measured)
    except (TypeError, ValueError):
        return False

    # Precondition: a real, non-degenerate measured result must exist on disk
    # (a target "hit" backed by a fabricated metrics.json must never admit).
    try:
        from backend.agents.rlm.zero_metrics_detection import (  # lazy import — stdlib-only dep
            looks_like_zero_metrics,
            normalize_metric_values,
        )
        metrics_path = Path(project_dir) / "code" / "metrics.json"
        if not metrics_path.exists():
            return False
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not normalize_metric_values(metrics) or looks_like_zero_metrics(metrics):
            return False
    except Exception:
        return False

    # The actual deterministic meets_target: the measured headline clears the claim.
    return measured_f >= claimed_f


# ──────────────────────────────────────────────────────────────────────────────
# Recipe extraction — poison-proof (no agent prose)
# ──────────────────────────────────────────────────────────────────────────────
def _build_problem_sig(report: dict) -> dict:
    """Extract model/dataset/task axes from the report.  Never includes prose."""
    sig: dict[str, Any] = {}
    # Scope info
    scope = report.get("scope") or {}
    if isinstance(scope, dict):
        if scope.get("models"):
            sig["models"] = [str(m) for m in scope["models"]][:8]
        if scope.get("datasets"):
            sig["datasets"] = [str(d) for d in scope["datasets"]][:8]
    # Paper class (if present in report)
    paper_class = report.get("paper_class") or report.get("arxiv_id") or ""
    if paper_class:
        sig["paper_id"] = str(paper_class)[:64]
    return sig


_SOLUTION_SIG_ALLOWED_HP_KEYS = frozenset({
    "lr", "learning_rate", "beta", "lambda", "batch_size", "epochs",
    "weight_decay", "momentum", "seed", "optimizer", "dropout",
    "model_name", "architecture", "cells_shape",
})


def _build_solution_sig(report: dict, project_dir: Path) -> dict:
    """Extract key hyperparameters + cells shape + technique summary.

    Only structured, non-prose fields from the provenance manifest are read.
    The technique_summary is truncated to MAX_SUMMARY_CHARS and must NOT come
    from agent-generated free text (it comes from the paper hint guidance,
    trimmed to the bounded length).
    """
    sig: dict[str, Any] = {}

    # Read provenance.json for hyperparameter ground truth (deterministic).
    prov_path = project_dir / "code" / "provenance.json"
    if prov_path.exists():
        try:
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
            if isinstance(prov, dict):
                hparams = {}
                for k, v in prov.items():
                    if k in _SOLUTION_SIG_ALLOWED_HP_KEYS and isinstance(v, (int, float, str, bool)):
                        hparams[k] = v
                if hparams:
                    sig["hyperparameters"] = hparams
        except Exception:
            pass

    # cells.json shape (structural metadata only — no agent prose).
    cells_path = project_dir / "code" / "cells.json"
    if cells_path.exists():
        try:
            cells = json.loads(cells_path.read_text(encoding="utf-8"))
            if isinstance(cells, list):
                sig["cells_count"] = len(cells)
                # Extract axis names only (not values which may be large).
                axes: set[str] = set()
                for c in cells:
                    if isinstance(c, dict):
                        axes.update(c.keys())
                sig["cells_axes"] = sorted(
                    a for a in axes
                    if a in {"model_key", "env", "baseline", "dataset", "seed", "est_vram_gb"}
                )
        except Exception:
            pass

    # Code-path pointer (just the relative path, never content).
    code_path = project_dir / "code"
    if code_path.exists():
        sig["code_path"] = "code/"

    # Technique summary: use the paper-class hint from the report if present.
    # Deliberately restricted to ≤MAX_SUMMARY_CHARS to prevent prose injection.
    summary = str(report.get("paper_class_summary") or "").strip()[:MAX_SUMMARY_CHARS]
    if summary:
        sig["technique_summary"] = summary

    return sig


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def admit_recipe(
    project_dir: Any,
    runs_root: Any,
    *,
    report: dict,
    validator_verdict: Any = None,
    paper_class: str,
) -> None:
    """Conditionally write a recipe to the cross-run store.

    ADMISSION GATE — admit ONLY if ALL four Tier-1 + validator conditions hold:
      (1) report-level evidence_gate passed
      (2) a success=True run_experiment ledger row exists
      (3) validator verdict is clean (or unavailable + floor passed)
      (4) deterministic meets_target holds

    The champion's median_score is NEVER used here.  Agent prose is NEVER
    included in the recipe body.  Fail-soft — never raises.
    """
    if not positive_recipes_enabled():
        return
    try:
        _admit_recipe_inner(
            Path(project_dir), Path(runs_root),
            report=report, validator_verdict=validator_verdict,
            paper_class=paper_class,
        )
    except Exception:
        return  # fail-soft: recipe bookkeeping must never break finalize


def _admit_recipe_inner(
    project_dir: Path,
    runs_root: Path,
    *,
    report: dict,
    validator_verdict: Any,
    paper_class: str,
) -> None:
    # ── Gate 1: evidence_gate passed ─────────────────────────────────────────
    floor_passed = _evidence_gate_passed(report)
    if not floor_passed:
        return  # red line: high grade but evidence_gate failed → rejected

    # ── Gate 2: success ledger row ────────────────────────────────────────────
    if not _has_success_ledger_row(project_dir):
        return

    # ── Gate 3: validator verdict ─────────────────────────────────────────────
    if not _validator_ok(validator_verdict, floor_passed=floor_passed):
        return  # vetoed → never admit

    # ── Gate 4: deterministic meets_target ───────────────────────────────────
    if not _deterministic_meets_target(report, project_dir):
        return

    # ── Build recipe (poison-proof) ──────────────────────────────────────────
    problem_sig = _build_problem_sig(report)
    solution_sig = _build_solution_sig(report, project_dir)
    evidence_key = str(report.get("evidence_key") or "")

    recipe_dict: dict[str, Any] = {
        "problem_sig": problem_sig,
        "solution_sig": solution_sig,
        "evidence_key": evidence_key,
        "paper_class": paper_class,
        "problem_sig_hash": _problem_sig_hash(problem_sig),
        "staleness": 0,
        # Report stamp: the score is COPIED here only as metadata for human
        # inspection — it is NOT used for ranking or admission decisions.
        "report_stamp": {
            "overall_score": report.get("rubric", {}).get("overall_score"),  # copy-into-report (red-line allowlisted)
            "meets_target": _deterministic_meets_target(report, project_dir),
            "validator_status": (
                validator_verdict.get("status")
                if isinstance(validator_verdict, dict)
                else getattr(validator_verdict, "status", None)
            ),
        },
    }

    # ── Write to store ────────────────────────────────────────────────────────
    store_path = _store_path(runs_root, paper_class)
    records = _read_store(store_path)

    # Novelty dedup: skip if an identical problem_sig hash already exists.
    new_hash = recipe_dict["problem_sig_hash"]
    existing_hashes = {r.get("problem_sig_hash") for r in records}
    if new_hash in existing_hashes:
        return  # already have an equivalent recipe

    # Enforce cap.
    records.append(recipe_dict)
    if len(records) > MAX_RECIPES:
        records = records[-MAX_RECIPES:]

    _write_store(store_path, records)


def recipe_guidance_block(
    runs_root: Any,
    paper_class: str,
    *,
    max_chars: int = 1200,
) -> str:
    """Return an implementer-guidance block from the stored recipes for ``paper_class``.

    Injects top-1 (capped at INJECT_TOP_K=2) by problem-sig recency.
    Returns "" when the feature is disabled, no recipes exist, or the class is unknown.
    """
    if not positive_recipes_enabled():
        return ""
    try:
        return _recipe_guidance_block_inner(
            Path(runs_root), paper_class, max_chars=max_chars
        )
    except Exception:
        return ""


def _recipe_guidance_block_inner(
    runs_root: Path,
    paper_class: str,
    *,
    max_chars: int,
) -> str:
    store_path = _store_path(runs_root, paper_class)
    records = _read_store(store_path)
    if not records:
        return ""

    # Filter to only non-stale, meets_target records.
    active = [
        r for r in records
        if r.get("report_stamp", {}).get("meets_target") is True
        and int(r.get("staleness", 0)) < RETIRE_STALENESS
    ]
    if not active:
        return ""

    # Inject top-1 (most recent by position), hard-capped at INJECT_TOP_K.
    top = active[-INJECT_TOP_K:]

    lines: list[str] = ["POSITIVE RECIPES (prior successful patterns for this paper class):"]
    for rec in reversed(top):
        sol = rec.get("solution_sig") or {}
        prob = rec.get("problem_sig") or {}
        parts: list[str] = []
        if sol.get("hyperparameters"):
            parts.append(f"hyperparameters={sol['hyperparameters']}")
        if sol.get("cells_count"):
            parts.append(f"cells={sol['cells_count']}")
        if sol.get("technique_summary"):
            parts.append(f"technique={sol['technique_summary'][:MAX_SUMMARY_CHARS]}")
        if prob.get("models"):
            parts.append(f"models={prob['models']}")
        body = "; ".join(parts) if parts else "(structured pattern)"
        lines.append(f"- [{rec.get('paper_class', paper_class)}] {body}")

    block = "\n".join(lines)
    return block[:max_chars]


def update_staleness(runs_root: Any, paper_class: str) -> None:
    """Increment staleness for all stored recipes of this class.

    Called by the finalize hook when a run of the same paper class completes
    but was NOT admitted (so a stale recipe that never fires again is retired).
    Fail-soft.
    """
    if not positive_recipes_enabled():
        return
    try:
        _update_staleness_inner(Path(runs_root), paper_class)
    except Exception:
        return


def _update_staleness_inner(runs_root: Path, paper_class: str) -> None:
    store_path = _store_path(runs_root, paper_class)
    records = _read_store(store_path)
    if not records:
        return
    updated: list[dict[str, Any]] = []
    for r in records:
        r = dict(r)
        r["staleness"] = int(r.get("staleness", 0)) + 1
        if r["staleness"] < RETIRE_STALENESS:
            updated.append(r)
        # else: drop (retired)
    _write_store(store_path, updated)
