"""Tier-2 external adversarial validator (P2.2 — 2026-06-20 grounded self-improvement).

An LLM panel is asked to *point at suspicions* by naming typed predicates; every
veto rests on a HARNESS-side deterministic machine-check of the named predicate
(not the LLM's opinion).  This ensures the veto is grounded even when the panel
is weak (same provider, different deployment) or degraded (same model, seed-only).

Min-aggregation: a metric_ref is vetoed iff ANY panelist's predicate machine-verifies
as violated.  No majority vote (consensus collapse).

Verdict store: persisted to ``rlm_state/validation_verdict.json`` (atomic), keyed by
evidence fingerprint.  A stale verdict (fingerprint mismatch) is ignored.

Default-OFF: ``OPENRESEARCH_EXTERNAL_VALIDATOR`` must be set to enable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredicateVerdict:
    """Result of a single HARNESS-side machine-check on one panelist suspicion."""

    predicate: str       # provenance_present | not_all_constant | gpu_claim_plausible | rerun_agrees
    metric_ref: str      # the cited metric key / leaf
    violated: bool       # True = machine-verified violated (NOT the LLM's opinion)
    detail: str          # human-readable explanation


@dataclass(frozen=True)
class ValidatorVerdict:
    """Aggregated verdict from the full panel."""

    status: str                    # clean | vetoed | unavailable
    veto_set: list[str]            # metric_refs vetoed (min-aggregation)
    predicates: list[PredicateVerdict]
    panel_models: list[str]
    separation: str                # independent | weak | degraded | unavailable (caller-supplied)
    evidence_fingerprint: str


# ---------------------------------------------------------------------------
# Feature flag + config
# ---------------------------------------------------------------------------

_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})


def external_validator_enabled() -> bool:
    """True iff ``OPENRESEARCH_EXTERNAL_VALIDATOR`` opts the panel ON.

    Default-OFF: with the var unset or falsey the module is inert and the
    caller gets ``status="unavailable"`` (Tier-1 floor is the backstop).
    """
    return os.environ.get("OPENRESEARCH_EXTERNAL_VALIDATOR", "").strip().lower() in _ENABLED_VALUES


def validator_panel_n() -> int:
    """Number of independent panel completions to request.

    Reads ``OPENRESEARCH_VALIDATOR_PANEL_N``; defaults to 2 (small, as specified).
    """
    raw = os.environ.get("OPENRESEARCH_VALIDATOR_PANEL_N", "").strip()
    try:
        n = int(raw)
        return max(1, n)
    except (ValueError, TypeError):
        return 2


# ---------------------------------------------------------------------------
# Deterministic machine-checks (the four typed predicates)
# ---------------------------------------------------------------------------


def check_provenance_present(metrics: Any, project_dir: Path) -> bool:
    """True iff a ``provenance.json`` file exists under the project's code dir.

    A provenance.json links a metric to a real output.  Its absence (when the
    run claims GPU training with all-zero metrics) is the primary hallucination
    signal.  Fail-soft: any I/O error returns False (conservative — no veto on
    uncertainty).

    Note: returns True == HEALTHY (provenance IS present, nothing suspicious).
    The violated field is set to ``not check_provenance_present(...)``.
    """
    try:
        code_dir = project_dir / "code"
        if code_dir.exists():
            # Search for provenance.json anywhere under code/
            for _ in code_dir.rglob("provenance.json"):
                return True
        return False
    except Exception:  # noqa: BLE001
        return False


def check_not_all_constant(metrics: Any) -> bool:
    """True iff the metrics are NOT all-zero or all-constant (healthy).

    Lazy-imports ``normalize_metric_values`` from ``zero_metrics_detection``
    (built in parallel — this deferred import keeps tests hermetic when that
    module is absent and the caller passes simple flat dicts).

    Fail-soft: import failure or any error returns True (healthy, no veto).
    """
    try:
        from backend.agents.rlm.zero_metrics_detection import normalize_metric_values  # noqa: PLC0415
        values = normalize_metric_values(metrics)
        if not values:
            return True  # nothing to check — conservatively healthy
        if all(v == 0.0 for v in values):
            return False  # all-zero (even a single 0.0 is suspect)
        if len(values) >= 2 and len(set(values)) == 1:
            return False  # all bit-identical (constant across cells; needs >= 2 values)
        return True  # at least some variation — healthy
    except Exception:  # noqa: BLE001
        return True  # fail-soft: cannot determine → treat as healthy


def check_gpu_claim_plausible(metrics: Any, evidence: dict[str, Any]) -> bool:
    """True iff the GPU training claim is plausible given the available evidence.

    Uses ``metrics_claim_gpu_training`` from ``gpu_cell_runner`` to detect the
    GPU claim.  If no GPU claim is present the result is immediately plausible
    (no veto candidate).  When a GPU claim IS present, plausibility requires
    that either:
      - there is wall-time evidence (``wall_time_s`` > 0, or ``elapsed_s`` > 0), OR
      - a vram_used or step-count field is present and positive.

    The ``evidence`` dict is the caller-supplied context (may contain runtime
    info from the experiment result).  Fail-soft: any error returns True.
    """
    try:
        from backend.agents.rlm.gpu_cell_runner import metrics_claim_gpu_training  # noqa: PLC0415
        if not metrics_claim_gpu_training(metrics):
            return True  # no GPU claim — plausible by default
        # GPU was claimed.  Look for corroborating evidence.
        def _find(d: Any, keys: set[str]) -> float:
            if isinstance(d, dict):
                for k, v in d.items():
                    if k.lower() in keys:
                        try:
                            f = float(v)
                            if f > 0:
                                return f
                        except (TypeError, ValueError):
                            pass
                    sub = _find(v, keys)
                    if sub > 0:
                        return sub
            elif isinstance(d, (list, tuple)):
                for item in d:
                    sub = _find(item, keys)
                    if sub > 0:
                        return sub
            return 0.0
        time_keys = {"wall_time_s", "elapsed_s", "elapsed_time_s", "runtime_s", "train_time_s"}
        compute_keys = {"steps", "steps_run", "num_steps", "total_steps", "vram_used_gb", "vram_gb"}
        has_time = _find(metrics, time_keys) > 0 or _find(evidence, time_keys) > 0
        has_compute = _find(metrics, compute_keys) > 0 or _find(evidence, compute_keys) > 0
        return has_time or has_compute
    except Exception:  # noqa: BLE001
        return True  # fail-soft


def check_rerun_agrees(  # noqa: ANN002
    *args: Any,
    **kwargs: Any,
) -> bool | None:
    """P2 stub — real re-run-on-suspicion is implemented in P3.

    Returns None to signal "skipped" (not vetoed).  The predicate record
    will carry ``violated=False`` and a ``detail`` explaining this is pending.
    """
    return None  # P2: always skipped


def check_report_claims_grounded(
    report_claims: list[Any] | None,
    project_dir: Path,
) -> bool:
    """Machine-check: True (grounded/clean) iff all report claims are grounded.

    The machine-check IS ``claim_grounding.check_claims_grounded`` — the
    deterministic engine, not an LLM judgment.  Returns True (not violated) when
    report_claims is None/empty or when no ungrounded claims are found; returns
    False (violated) when ≥1 ungrounded result claim exists and measured evidence
    is present (evidence_gate owns the no-evidence case).
    Fail-soft → True (clean).
    """
    try:
        if not report_claims:
            return True
        from backend.agents.rlm.claim_grounding import (  # noqa: PLC0415
            check_claims_grounded,
            flatten_measured_values,
        )
        measured = flatten_measured_values(project_dir)
        if not measured:
            return True  # unverifiable — evidence_gate owns the no-evidence case
        grounding = check_claims_grounded(report_claims, measured)
        return len(grounding.get("ungrounded", [])) == 0
    except Exception:  # noqa: BLE001 — fail-soft
        return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VALID_PREDICATES = frozenset({
    "provenance_present",
    "not_all_constant",
    "gpu_claim_plausible",
    "rerun_agrees",
    "report_claims_grounded",
})

_ADVERSARIAL_SYSTEM = """You are an adversarial ML research verifier.
Your job is to FIND potential hallucinations or fabrications in experimental results.
Be skeptical. Assume the metrics might be fake until proven otherwise.

For each suspicion you have, output a JSON object with:
  {"predicate": "<name>", "metric_ref": "<key or label>"}

Valid predicates:
- provenance_present: suspect the metric lacks a provenance.json linking it to a real output
- not_all_constant: suspect all result values are zero or suspiciously constant
- gpu_claim_plausible: suspect the GPU/training claim is implausible given the evidence
- rerun_agrees: suspect a re-run would not reproduce the claimed result
- report_claims_grounded: suspect the report narrative claims a result value absent from code/metrics.json

Output a JSON array of suspicions. If you find no issues output [].
Do not explain — only output the JSON array."""

_ADVERSARIAL_USER_TEMPLATE = """Examine these experimental metrics and flag suspicious predicates.

METRICS:
{metrics_json}

LEAF RECORDS (rubric evaluation context):
{leaf_records_json}

REPORT CLAIMS (root's achieved-result narrative, optional):
{report_claims_json}

Output ONLY a JSON array of {{predicate, metric_ref}} suspicions. Example:
[{{"predicate": "not_all_constant", "metric_ref": "mean_reward"}}, {{"predicate": "provenance_present", "metric_ref": "accuracy_avg"}}]"""


def _parse_suspicions(text: str) -> list[dict[str, str]]:
    """Parse a panelist's response into a list of {predicate, metric_ref} dicts.

    Tolerates markdown fences, extra prose before/after the JSON array.
    Returns [] on any parse failure (fail-soft).
    """
    text = text.strip()
    # Strip markdown fences if present
    for fence in ("```json", "```"):
        if fence in text:
            start = text.find(fence) + len(fence)
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()
                break
    # Try to find a JSON array anywhere in the text
    for start_ch in ("[",):
        idx = text.find(start_ch)
        if idx != -1:
            # Find the matching closing bracket
            depth = 0
            for i, ch in enumerate(text[idx:]):
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = text[idx: idx + i + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, list):
                                result = []
                                for item in parsed:
                                    if isinstance(item, dict):
                                        pred = str(item.get("predicate", "")).strip()
                                        ref = str(item.get("metric_ref", "")).strip()
                                        if pred and ref and pred in _VALID_PREDICATES:
                                            result.append({"predicate": pred, "metric_ref": ref})
                                return result
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break
    return []


def _machine_check(
    predicate: str,
    metric_ref: str,
    metrics: Any,
    project_dir: Path,
    evidence: dict[str, Any],
    *,
    report_claims: list[Any] | None = None,
) -> PredicateVerdict:
    """Run the deterministic machine-check for ``predicate`` and return a PredicateVerdict.

    ``violated=True`` means the machine-check confirmed the predicate is violated
    (i.e. the suspicion is substantiated).
    ``report_claims`` is only consulted for the ``report_claims_grounded`` predicate.
    """
    if predicate == "provenance_present":
        present = check_provenance_present(metrics, project_dir)
        if present:
            return PredicateVerdict(
                predicate=predicate,
                metric_ref=metric_ref,
                violated=False,
                detail="provenance.json found — metric links to a real output",
            )
        else:
            return PredicateVerdict(
                predicate=predicate,
                metric_ref=metric_ref,
                violated=True,
                detail="provenance.json absent — no link from metric to a real output",
            )

    elif predicate == "not_all_constant":
        healthy = check_not_all_constant(metrics)
        if healthy:
            return PredicateVerdict(
                predicate=predicate,
                metric_ref=metric_ref,
                violated=False,
                detail="metric values show variation — not constant",
            )
        else:
            return PredicateVerdict(
                predicate=predicate,
                metric_ref=metric_ref,
                violated=True,
                detail="all result metric values are zero or identical (constant) — likely fabricated",
            )

    elif predicate == "gpu_claim_plausible":
        plausible = check_gpu_claim_plausible(metrics, evidence)
        if plausible:
            return PredicateVerdict(
                predicate=predicate,
                metric_ref=metric_ref,
                violated=False,
                detail="GPU training claim is consistent with available evidence",
            )
        else:
            return PredicateVerdict(
                predicate=predicate,
                metric_ref=metric_ref,
                violated=True,
                detail="GPU training claimed but corroborating evidence (wall_time, steps, vram) absent",
            )

    elif predicate == "rerun_agrees":
        check_rerun_agrees(metrics=metrics, metric_ref=metric_ref)
        # P2: always returns None (skipped)
        return PredicateVerdict(
            predicate=predicate,
            metric_ref=metric_ref,
            violated=False,
            detail="rerun_agrees check is pending (P3) — not vetoed at P2",
        )

    elif predicate == "report_claims_grounded":
        grounded = check_report_claims_grounded(report_claims, project_dir)
        if grounded:
            return PredicateVerdict(
                predicate=predicate,
                metric_ref=metric_ref,
                violated=False,
                detail="report claims are grounded by code/metrics.json",
            )
        else:
            return PredicateVerdict(
                predicate=predicate,
                metric_ref=metric_ref,
                violated=True,
                detail=(
                    "report narrative claims result values absent from code/metrics.json "
                    "(within 5% tolerance)"
                ),
            )

    else:
        # Unknown predicate — conservatively not violated
        return PredicateVerdict(
            predicate=predicate,
            metric_ref=metric_ref,
            violated=False,
            detail=f"unknown predicate {predicate!r} — ignored",
        )


def _compute_evidence_fingerprint(metrics: Any) -> str:
    """Stable sha256 fingerprint of the metrics dict for the verdict store.

    Uses a simple canonical JSON serialisation (sorted keys, default=str).
    Fails soft to a hash of "empty" if metrics is not a dict.
    """
    try:
        canonical = json.dumps(
            metrics if isinstance(metrics, dict) else {},
            sort_keys=True,
            default=str,
        )
    except Exception:  # noqa: BLE001
        canonical = "{}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evidence_fingerprint(metrics: object) -> str:
    """Public alias for :func:`_compute_evidence_fingerprint`.

    Exposes the canonical fingerprint function so ``report.py`` (and tests)
    can compute the SAME fingerprint the panel used without importing a
    private symbol.  The panel and the report-stamp chokepoint both call this
    alias, ensuring they are always in sync.
    """
    return _compute_evidence_fingerprint(metrics)


# ---------------------------------------------------------------------------
# Core panel runner
# ---------------------------------------------------------------------------


def run_validation_panel(
    *,
    validator_client: Any,
    panel_models: list[str],
    metrics: Any,
    project_dir: Path,
    leaf_records: list[dict[str, Any]],
    separation: str,
    report_claims: list[Any] | None = None,
) -> ValidatorVerdict:
    """Run the adversarial validation panel and return a min-aggregated verdict.

    Algorithm:
      1. If ``validator_client`` is None → return ``status="unavailable"`` immediately.
      2. Call ``sample_completions`` with the adversarial prompt, n=validator_panel_n().
      3. Parse each completion into a list of ``{predicate, metric_ref}`` suspicions.
      4. For each suspicion, run the HARNESS deterministic machine-check.
      5. Min-aggregation: a metric_ref is in the veto_set iff ANY panelist's predicate
         machine-verifies as violated.
      6. status = "vetoed" if veto_set else "clean".
    """
    evidence_fp = _compute_evidence_fingerprint(metrics)

    if validator_client is None:
        return ValidatorVerdict(
            status="unavailable",
            veto_set=[],
            predicates=[],
            panel_models=list(panel_models),
            separation=separation,
            evidence_fingerprint=evidence_fp,
        )

    # Build the prompt
    try:
        metrics_json = json.dumps(
            metrics if isinstance(metrics, dict) else {},
            sort_keys=True,
            indent=2,
            default=str,
        )
    except Exception:  # noqa: BLE001
        metrics_json = "{}"

    try:
        leaf_records_json = json.dumps(leaf_records[:20], indent=2, default=str)
    except Exception:  # noqa: BLE001
        leaf_records_json = "[]"

    try:
        _rc_list = [{"term": c.term, "value": c.value, "context": c.context} for c in (report_claims or [])]
        report_claims_json = json.dumps(_rc_list[:20], indent=2, default=str) if _rc_list else "null"
    except Exception:  # noqa: BLE001
        report_claims_json = "null"

    user_prompt = _ADVERSARIAL_USER_TEMPLATE.format(
        metrics_json=metrics_json,
        leaf_records_json=leaf_records_json,
        report_claims_json=report_claims_json,
    )

    n = validator_panel_n()

    # Call the panel
    try:
        from backend.agents.rlm.grader_transport import sample_completions  # noqa: PLC0415
        completions = sample_completions(
            validator_client,
            system=_ADVERSARIAL_SYSTEM,
            user=user_prompt,
            n=n,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("external_validator: panel call failed (%s) — unavailable", exc)
        return ValidatorVerdict(
            status="unavailable",
            veto_set=[],
            predicates=[],
            panel_models=list(panel_models),
            separation=separation,
            evidence_fingerprint=evidence_fp,
        )

    # Parse suspicions from each panelist and machine-check them
    all_predicate_verdicts: list[PredicateVerdict] = []
    # Track which metric_refs are violated by ANY panelist (min-aggregation)
    vetoed_refs: set[str] = set()

    evidence_ctx: dict[str, Any] = {}

    for completion in completions:
        suspicions = _parse_suspicions(completion)
        for sus in suspicions:
            pred = sus["predicate"]
            ref = sus["metric_ref"]
            verdict = _machine_check(pred, ref, metrics, project_dir, evidence_ctx, report_claims=report_claims)
            all_predicate_verdicts.append(verdict)
            if verdict.violated:
                vetoed_refs.add(ref)

    veto_set = sorted(vetoed_refs)
    status = "vetoed" if veto_set else "clean"

    return ValidatorVerdict(
        status=status,
        veto_set=veto_set,
        predicates=all_predicate_verdicts,
        panel_models=list(panel_models),
        separation=separation,
        evidence_fingerprint=evidence_fp,
    )


# ---------------------------------------------------------------------------
# Verdict store — atomic persist + load
# ---------------------------------------------------------------------------

_VERDICT_FILENAME = "validation_verdict.json"


def _verdict_path(project_dir: Path) -> Path:
    """Return the canonical path for the validation verdict file."""
    return project_dir / "rlm_state" / _VERDICT_FILENAME


def _verdict_to_dict(verdict: ValidatorVerdict) -> dict[str, Any]:
    """Serialise a ValidatorVerdict to a JSON-compatible dict."""
    d = asdict(verdict)
    # predicates are list[PredicateVerdict] — asdict handles the nesting
    return d


def _dict_to_verdict(d: dict[str, Any]) -> ValidatorVerdict:
    """Reconstruct a ValidatorVerdict from its dict form."""
    predicates = [
        PredicateVerdict(**p) if isinstance(p, dict) else p
        for p in d.get("predicates", [])
    ]
    return ValidatorVerdict(
        status=d.get("status", "unavailable"),
        veto_set=list(d.get("veto_set", [])),
        predicates=predicates,
        panel_models=list(d.get("panel_models", [])),
        separation=d.get("separation", "unavailable"),
        evidence_fingerprint=d.get("evidence_fingerprint", ""),
    )


def persist_verdict(project_dir: Path, verdict: ValidatorVerdict) -> None:
    """Atomically persist the verdict to ``rlm_state/validation_verdict.json``.

    Uses a temp file + os.replace so a concurrent reader never sees a partial
    write.  Fail-soft: logs and returns on any error.
    """
    target = _verdict_path(project_dir)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = _verdict_to_dict(verdict)
        blob = json.dumps(payload, indent=2, sort_keys=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=target.parent,
            prefix=".tmp_validation_verdict_",
            suffix=".json",
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(blob)
            os.replace(tmp_path, target)
        except Exception:  # noqa: BLE001
            # Clean up the temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("external_validator: failed to persist verdict (%s)", exc)


def load_verdict(
    project_dir: Path,
    *,
    expect_fingerprint: str | None = None,
) -> ValidatorVerdict | None:
    """Load the persisted verdict, or return None if absent or stale.

    If ``expect_fingerprint`` is given and the stored verdict's
    ``evidence_fingerprint`` does not match, returns None (stale verdict
    ignored — a verdict for different evidence must never influence the
    current run).
    """
    target = _verdict_path(project_dir)
    try:
        if not target.exists():
            return None
        with open(target) as f:
            d = json.load(f)
        verdict = _dict_to_verdict(d)
        if expect_fingerprint is not None and verdict.evidence_fingerprint != expect_fingerprint:
            logger.debug(
                "external_validator: stale verdict (stored fp=%r, expected fp=%r) — ignored",
                verdict.evidence_fingerprint,
                expect_fingerprint,
            )
            return None
        return verdict
    except Exception as exc:  # noqa: BLE001
        logger.warning("external_validator: failed to load verdict (%s)", exc)
        return None
