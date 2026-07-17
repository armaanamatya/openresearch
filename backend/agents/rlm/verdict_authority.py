"""verdict_authority.py — the single, last, grade-free reproduction-verdict
decision (Track A, closes findings #1/#4/#5/#10/#11; design
``docs/superpowers/specs/2026-07-09-eval-integrity-track-a-design.md`` §4.3).

The north-star invariant this module exists to protect: the reproduction
verdict is keyed on the **deterministic evidence layer**, never the LLM
grade (root ``CLAUDE.md``, "Evidence, not grade"). Historically the grade
reached the headline verdict two ways — a score-threshold upgrade in
``report.py`` and ``two_axis_report`` overwriting ``report["verdict"]`` with
a fidelity-derived (i.e. grade-derived) legacy projection. Both are being
retired in favour of this one function.

``decide`` takes **no grade / ``impl_fidelity`` parameter**. This is not a
convention to remember to honour — it is structural: there is no signature
slot for a grade to flow through, so this function cannot fold one into the
verdict even by accident. It decides purely from:

  * ``result_fidelity`` — the deterministic per-claim checker's output
    (``result_fidelity.evaluate``, Task 2): a dict carrying ``per_claim``,
    a list of ``{claim_id, status, is_primary, ambiguous, ...}`` entries
    where ``status`` is one of ``"pass"``/``"fail"``/``"unmeasured"``.
  * ``evidence_gate`` — whether the run has >=1 success-compatible
    in-process ``run_experiment`` ledger row plus real on-disk metrics.
  * ``fidelity_certificate`` — accepted for interface completeness. Per the
    design (§4.4) the certificate is an input to the EVIDENCE GATE upstream
    ("registers the paper's declared constants ... strengthens the evidence
    gate's anti-forgery signal"), never an independent verdict cap here — a
    non-registrable certificate is an evidence gap the caller folds into
    ``evidence_gate`` before calling ``decide``, not a silent global pin to
    ``partial``. This function does not read it.
  * ``claim_gate_cap`` — an optional str verdict ceiling (e.g. from
    ``report_claim_gate``). Clamps the result DOWNWARD only, never up.
  * ``ruler_quality`` — Spec-B seam (the gold-set ruler-quality verdict).
    Accepted (default ``None``, read as "no ruler-quality signal yet, i.e.
    trusted") but not otherwise acted on in Track A — Spec B is what will
    actually gate on it. Present purely so Spec B does not need a signature
    change later.

Taxonomy — deterministic, explicit precedence over the claim's PRIMARY
claims (locked, design §4.3), checked in this order:

  1. **No measurable primary** — no claim is genuinely ``is_primary``, or
     the only primaries are ``ambiguous``/unresolved-bind → ``inconclusive``
     (``reason="no_measurable_target"``). The extractor's "ensure exactly
     one primary claim" fallback (``repro_spec_extractor.py``, auto-promotes
     ``claims_out[0]`` when nothing was genuinely marked primary) is a
     spec-shape convenience, not a paper-endorsed headline claim, and is
     IGNORED here — an arbitrarily-promoted secondary never yields a
     headline verdict on its own. (No dedicated "was this auto-promoted"
     marker exists on a claim today; ``ambiguous`` is the load-bearing
     signal per the locked design. ``auto_promoted`` /
     ``primary_source == "auto_promoted"`` are read defensively too, so a
     future extractor change that DOES stamp such a marker is honoured
     without touching this module again.)
  2. **Any primary ``fail``** → ``contradicted`` (a measured miss dominates;
     never hidden inside ``inconclusive``, and never masked by an
     ``unmeasured`` sibling).
  3. **Any primary ``unmeasured``** (none ``fail``) → ``partial``
     (faithfully attempted, unmeasured).
  4. **All primaries ``pass``** AND ``evidence_gate`` is satisfied →
     ``reproduced``. All-pass with an unsatisfied evidence gate stays
     ``partial`` — the measured claim is real, but the run cannot be
     certified as the legitimate source of it.

Multiple genuine primaries are rolled up by "the weakest decisive outcome
governs" — reusing ``reproducibility_verdict._ROLLUP_ORDER`` for the
severity numbers (translated through this module's own status vocabulary)
rather than inventing a second, parallel ordering.

Also implements ``freeze_contract`` — the Spec-C seam (§4.6): Track A calls
it immediately after the ruler (``rlm_state/repro_spec.json``) is written
(auto-freeze, no human in the loop yet); Spec C will interpose the async
ApprovalService offer + a default-auto timeout on this SAME call.

Pure(-ish) and stdlib-only: no LLM call, no network. ``freeze_contract``
does local file I/O (by definition — "freeze this file on disk"), but reads
no run state beyond the one JSON file and never fabricates content.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

from backend.agents.rlm.reproducibility_verdict import _ROLLUP_ORDER

__all__ = [
    "VerdictLabel",
    "Verdict",
    "decide",
    "freeze_contract",
    "is_enabled",
    "VERDICT_SURFACE_KEYS",
    "PostAuthorityVerdictMutation",
    "assert_verdict_surface_unchanged",
]


VerdictLabel = Literal["reproduced", "contradicted", "partial", "inconclusive"]


class Verdict(TypedDict):
    verdict: VerdictLabel
    reason: str


# ---------------------------------------------------------------------------
# The sever's activation gate (§4.3) — the ONE place every historical
# grade-derived verdict writer checks before deciding whether to keep minting
# a verdict (legacy, flag off) or defer to this module (severed, flag on).
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """True iff the VerdictAuthority sever is fully active.

    Requires BOTH the existing two-axis master flag
    (``OPENRESEARCH_TWO_AXIS_VERDICT``) AND this module's own sub-flag
    (``OPENRESEARCH_VERDICT_AUTHORITY``) to be truthy. Either off => every
    grade-derived verdict writer this module supersedes
    (``two_axis_report.compute_and_attach``, ``finalize_regrade.regrade_and_emit``,
    ``leaf_scorer.amend_final_report``, ``rdr/controller.py``,
    ``run.py``'s hard-stop salvage) keeps its pre-Track-A behaviour, byte-for-
    byte identical.

    Single canonical gate: every sever-point module imports and calls THIS
    function rather than re-deriving the AND of two env vars locally, so the
    activation condition can never drift between call sites.
    """
    from backend.agents.rlm.feature_flags import env_truthy  # noqa: PLC0415

    return env_truthy("OPENRESEARCH_TWO_AXIS_VERDICT") and env_truthy(
        "OPENRESEARCH_VERDICT_AUTHORITY"
    )


# ---------------------------------------------------------------------------
# Single-writer runtime guard (§4.3 acceptance: "a static + runtime assertion
# that NO module writes report.verdict / demo_status.verdict for a
# reproduction run AFTER VerdictAuthority.decide").
# ---------------------------------------------------------------------------

# The reproduction-verdict surface this module governs (design §4.3 "Scope
# (explicit)"): the headline verdict plus the two top-level diagnostic mirrors
# and the demo_status.json mirror. Once ``decide()`` has stamped these for a
# given finalize call, nothing may write to any of them again.
VERDICT_SURFACE_KEYS: tuple[str, ...] = (
    "verdict",
    "implementation_verdict",
    "replication_verdict",
)


class PostAuthorityVerdictMutation(RuntimeError):
    """Raised when a reproduction-verdict key changed after ``decide()`` had
    already stamped it — a violation of the single-last-writer invariant this
    module exists to enforce (Track A §4.3). Should never fire on healthy
    code; it is a regression tripwire, not a normal-path branch.
    """


def assert_verdict_surface_unchanged(
    stamped: dict[str, Any], current: dict[str, Any], *, context: str = ""
) -> None:
    """Raise :class:`PostAuthorityVerdictMutation` iff any key in
    :data:`VERDICT_SURFACE_KEYS` present in ``stamped`` differs in ``current``.

    ``stamped`` is the snapshot of the verdict surface captured immediately
    after :func:`decide` stamped it; ``current`` is the surface re-read at (or
    just before) the moment the report is actually shipped. A mismatch means
    something wrote to the surface in between — exactly the class of
    regression the design's runtime guard exists to catch. A key absent from
    ``stamped`` is not checked (nothing was stamped for it, so there is
    nothing to protect).
    """
    for key in VERDICT_SURFACE_KEYS:
        if key not in stamped:
            continue
        if current.get(key) != stamped.get(key):
            prefix = f"{context}: " if context else ""
            raise PostAuthorityVerdictMutation(
                f"{prefix}{key!r} changed from {stamped.get(key)!r} (stamped by "
                f"VerdictAuthority.decide()) to {current.get(key)!r} afterward — "
                "no writer may mutate the reproduction-verdict surface once the "
                "single authority has spoken (Track A §4.3 sever)."
            )


# ---------------------------------------------------------------------------
# Severity ordering — reused, not reinvented
# ---------------------------------------------------------------------------

# The four possible reproduction verdicts, ranked from least to most
# "successful" — exactly the locked precedence order in the module docstring
# (inconclusive -> contradicted -> partial -> reproduced). `claim_gate_cap`
# clamps DOWNWARD only: it can move a verdict to a strictly lower rank, never
# raise it to a higher one.
_VERDICT_RANK: dict[str, int] = {
    "inconclusive": 0,
    "contradicted": 1,
    "partial": 2,
    "reproduced": 3,
}

# Translates THIS module's per-claim status vocabulary ("pass"/"fail"/
# "unmeasured", produced by result_fidelity.evaluate) onto
# reproducibility_verdict's existing ReplicationVerdict severity vocabulary,
# so "the weakest primary governs" reuses the ONE severity ordering already
# in the codebase (`_ROLLUP_ORDER`) instead of a second, parallel one. An
# unrecognized/missing status is treated exactly like "unmeasured" — never
# silently promoted to a passing result.
_STATUS_TO_ROLLUP_KEY: dict[str, str] = {
    "fail": "contradicted",
    "unmeasured": "inconclusive",
    "pass": "replicated",
}
_UNKNOWN_STATUS_ROLLUP_KEY = "inconclusive"


def _result(verdict: VerdictLabel, reason: str) -> Verdict:
    return {"verdict": verdict, "reason": reason}


def _apply_cap(result: Verdict, claim_gate_cap: str | None) -> Verdict:
    """Clamp ``result["verdict"]`` DOWNWARD only against ``claim_gate_cap``.

    ``None`` (no cap supplied) and any string that is not one of the four
    verdict words are both documented no-ops — capping is a strict, known
    ceiling, never a guess at what an unrecognized value might mean.
    """
    if claim_gate_cap is None:
        return result
    cap_rank = _VERDICT_RANK.get(claim_gate_cap)
    if cap_rank is None:
        return result
    if cap_rank >= _VERDICT_RANK[result["verdict"]]:
        return result
    capped_verdict: VerdictLabel = claim_gate_cap  # type: ignore[assignment]
    return _result(capped_verdict, f"claim_gate_cap_applied:{result['reason']}")


# ---------------------------------------------------------------------------
# Primary-claim eligibility (Rule 1) + rollup (Rules 2-4)
# ---------------------------------------------------------------------------


def _is_genuine_primary(claim: dict[str, Any]) -> bool:
    """True iff ``claim`` is a genuinely primary, resolvable claim.

    Excludes: ``ambiguous`` claims (the locked "or only ambiguous/unbound
    primaries" branch of Rule 1); and, defensively, any claim carrying an
    explicit auto-promotion marker (``auto_promoted`` truthy, or
    ``primary_source == "auto_promoted"``) — no such marker exists on a
    claim produced by the current ``repro_spec_extractor`` today, but if one
    is added later this keeps the "auto-promote is ignored for the verdict"
    rule honoured without another change here.
    """
    if claim.get("ambiguous"):
        return False
    if claim.get("auto_promoted"):
        return False
    if claim.get("primary_source") == "auto_promoted":
        return False
    return True


def _governing_rollup_key(genuine_primaries: list[dict[str, Any]]) -> str:
    """The most severe status among ``genuine_primaries``, as a
    ``_ROLLUP_ORDER`` key (``"contradicted"``/``"inconclusive"``/
    ``"replicated"``).

    Mirrors ``reproducibility_verdict._rollup_primaries``'s "the weakest
    decisive outcome governs" — implemented here by translating this
    module's ``pass``/``fail``/``unmeasured`` vocabulary through
    ``_STATUS_TO_ROLLUP_KEY`` and taking the ``min`` by ``_ROLLUP_ORDER``,
    the SAME severity table that module already ranks by.
    """

    def _rank(claim: dict[str, Any]) -> int:
        key = _STATUS_TO_ROLLUP_KEY.get(claim.get("status"), _UNKNOWN_STATUS_ROLLUP_KEY)
        return _ROLLUP_ORDER.get(key, 1)

    worst = min(genuine_primaries, key=_rank)
    return _STATUS_TO_ROLLUP_KEY.get(worst.get("status"), _UNKNOWN_STATUS_ROLLUP_KEY)


def _per_claim_list(result_fidelity: Any) -> list[dict[str, Any]]:
    """Defensively extract ``result_fidelity["per_claim"]`` as a list.

    A missing/empty/non-dict ``result_fidelity`` (the RDR/legacy path, or an
    arXiv run where extraction never produced a ruler) degrades to an empty
    list, which flows through the SAME "no genuine primaries" branch as a
    real-but-empty claim set — never a special case, never a fallback to
    any other verdict.
    """
    if not isinstance(result_fidelity, dict):
        return []
    per_claim = result_fidelity.get("per_claim")
    if not isinstance(per_claim, list):
        return []
    return [c for c in per_claim if isinstance(c, dict)]


_MISSING = object()


def _gate_satisfied(evidence_gate: Any) -> bool:
    """Truthy check for ``evidence_gate``.

    Accepts a plain ``bool`` (today's only producer — mirrors
    ``report._apply_evidence_gate``'s internal ``has_real_evidence``) or any
    struct exposing a boolean ``satisfied`` attribute/key, so a richer
    evidence-gate object can be threaded through later without a signature
    change. Deliberately does NOT fall back to blind non-emptiness for a
    dict/object that DOES carry a ``satisfied`` field — a struct whose
    ``satisfied`` is ``False`` must never read as truthy just because the
    struct itself has other keys/attributes set. Anything without a
    ``satisfied`` field falls back to ordinary Python truthiness. Never
    raises.
    """
    if isinstance(evidence_gate, dict):
        probe = evidence_gate.get("satisfied", _MISSING)
    else:
        probe = getattr(evidence_gate, "satisfied", _MISSING)
    try:
        return bool(evidence_gate) if probe is _MISSING else bool(probe)
    except Exception:  # noqa: BLE001 — a gate value must never raise the decision
        return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def decide(
    *,
    result_fidelity: dict[str, Any] | None,
    evidence_gate: Any,
    fidelity_certificate: Any,
    claim_gate_cap: str | None = None,
    ruler_quality: str | None = None,
) -> Verdict:
    """The single, last, grade-free reproduction-verdict decision (§4.3).

    See the module docstring for the full taxonomy + parameter contract.
    ``ruler_quality`` is accepted (``None`` reads as "trusted") but not
    otherwise acted on — it is Spec B's seam, not Spec B itself (design
    §4.6, explicitly out of scope for Track A). ``fidelity_certificate`` is
    likewise accepted but not read (§4.4 — it is an evidence-gate input
    upstream, never an independent cap here).
    """
    genuine_primaries = [
        c for c in _per_claim_list(result_fidelity) if c.get("is_primary") and _is_genuine_primary(c)
    ]
    if not genuine_primaries:
        return _apply_cap(_result("inconclusive", "no_measurable_target"), claim_gate_cap)

    rollup_key = _governing_rollup_key(genuine_primaries)
    if rollup_key == "contradicted":
        result = _result("contradicted", "primary_claim_failed")
    elif rollup_key == "replicated":
        result = (
            _result("reproduced", "all_primary_claims_pass")
            if _gate_satisfied(evidence_gate)
            else _result("partial", "evidence_gate_not_satisfied")
        )
    else:
        # "inconclusive" rollup bucket: at least one unmeasured primary (and
        # none failed), or an unrecognized status defensively treated the
        # same way — faithfully attempted, not yet decisive.
        result = _result("partial", "primary_claim_unmeasured")

    return _apply_cap(result, claim_gate_cap)


# ---------------------------------------------------------------------------
# freeze_contract — Spec-C seam (§4.6)
# ---------------------------------------------------------------------------

_CONTRACT_STATUS_FROZEN = "frozen"


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".repro_spec_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


def freeze_contract(run_dir: Path) -> Path:
    """Freeze ``<run_dir>/rlm_state/repro_spec.json`` and return its path.

    Track A calls this immediately after the ruler is written — auto-freeze,
    no human in the loop yet. Spec C interposes the async ApprovalService
    offer + a default-auto timeout on this SAME call, so no lifecycle
    restructuring is needed when that lands (§4.6).

    Idempotent: freezing an already-frozen spec is a no-op re-write-avoider
    (``contract_status`` stays ``"frozen"``, ``frozen_at`` is not clobbered
    on a second call). Never fabricates a contract: if ``repro_spec.json``
    does not exist yet (extraction never ran / is disabled / RDR-legacy
    path), or exists but is not a JSON object, this returns the path
    unchanged — there is nothing to freeze, and inventing content here would
    be exactly the kind of fabrication this harness's evidence layer exists
    to refuse elsewhere. A genuine write failure (disk full, permissions) on
    an existing, well-formed spec is allowed to raise — silently swallowing
    it would let a caller believe the contract froze when it did not.
    """
    spec_path = Path(run_dir) / "rlm_state" / "repro_spec.json"
    try:
        raw = spec_path.read_text(encoding="utf-8")
    except OSError:
        return spec_path
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return spec_path
    if not isinstance(data, dict):
        return spec_path

    if data.get("contract_status") == _CONTRACT_STATUS_FROZEN:
        return spec_path

    data["contract_status"] = _CONTRACT_STATUS_FROZEN
    data.setdefault("frozen_at", datetime.now(timezone.utc).isoformat())
    _atomic_write_json(spec_path, data)
    return spec_path
