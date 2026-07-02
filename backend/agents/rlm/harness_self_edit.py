"""Phase C -- the harness self-edit tier (spec S11.2, Codex F11/F12, S17).

The Self-Harness blueprint on the existing substrate, scoped to an explicit
WHITELIST of prompt-guidance blocks and bounded numeric retry/threshold keys
(``self_edit_surface.json`` -- data, never code). Everything else -- guards,
evidence predicates, rubric + rubric-gen, the validator, budget enforcement,
the admission gates, the whitelist file itself, and this module -- is the
FROZEN tier: structurally unreachable, not merely policy-excluded. DGM
deleted the markers its own reward function used to detect its fabrication;
:data:`FROZEN_TIER_MARKERS` is why that boundary is structural here rather
than a promise a proposal could talk its way past.

Four promotion stages, staged and strengthened (F12): **candidate**
(:func:`propose`, whitelist + bounds only) -> **shadow**
(:meth:`HarnessEditGate.shadow`, fail-closed replay over the
``HarnessReplayCase`` corpus harvested from real campaign terminals --
:func:`harvest_replay_cases`; any raising/stale/reconstruction-failing case
rejects the WHOLE proposal, never silently skipped, F11; a clean overlay run
that changes ANY case's output also rejects -- every replay is a negative
control by construction, since :mod:`campaign_policy`'s
``decide``/``directives_fingerprint`` read no environment at all) ->
**canary** (:meth:`HarnessEditGate.promote_to_canary`, operator-supplied
paired A/B: >=2 papers x >=2 seeds/paper, every report path verified to
exist on disk, pairing complete, mean improvement > the measured grader
sigma) -> **default** (:meth:`HarnessEditGate.apply_default`, requires the
LITERAL ``operator_confirmed=True`` plus prior status ``"canary"``, bounds
re-validated against the CURRENT whitelist at apply time; no other code path
reaches ``"default"`` -- no autonomous canary->default flip, ever, S17).

A DEDICATED gate, not a reuse of ``held_out_gate.admit`` (F11): that stays
advisory-lesson machinery; harness mechanics need executable replay +
fail-closed promotion. Rejections split in two: a PRIOR-STAGE-not-reached
rejection (``shadow_not_passed``, ``canary_not_reached``,
``operator_confirmation_required``) is recorded but does NOT downgrade the
canonical status -- an operator probing out of order never bricks a good
proposal. An EVIDENCE-based rejection (frozen tier, bounds, a stale/erroring
replay, a failed canary) is terminal.

Defense-in-depth (fix pass): the frozen tier and stage order are
re-validated at EVERY boundary, not merely once at ``propose()`` --
:meth:`HarnessEditGate.shadow`/:meth:`~HarnessEditGate.promote_to_canary`/
:meth:`~HarnessEditGate.apply_default` each re-derive
:func:`_frozen_tier_hit` straight from the persisted proposal, before any
other logic, on every call (a hit is exactly as terminal as at propose
time). ``shadow`` additionally refuses to advance any status other than the
exact string ``"candidate"`` (reason ``"stage_order"``) -- a proposal
rejected at propose (frozen tier or otherwise) can never climb back to
``"shadow_passed"``/``"canary"``/``"default"`` by being replayed directly on
its deterministic id, even under a compromised whitelist. All three
`HarnessEditGate` methods also return ``{"status": "disabled"}`` with zero
filesystem access when the flag is off. :func:`active_overrides` drops any
on-disk override key that newly hits the frozen tier, independent of its
bounds check.

Imports: stdlib + :mod:`campaign_policy` (``decide``, ``directives_fingerprint``,
the typed records ``decide`` needs) + :mod:`attempt_assessment`
(``AttemptAssessment.from_dict``). No LLM, no sockets, no subprocess -- and
deliberately not ``reproduction_campaign``/``campaign_composition``/
``campaign_directives`` (this module sits BELOW ``campaign_composition`` in
the campaign import DAG).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.agents.rlm.attempt_assessment import AttemptAssessment
from backend.agents.rlm.campaign_policy import (
    CampaignBudget,
    CampaignSpend,
    PolicyConfig,
    decide,
    directives_fingerprint,
)

__all__ = [
    "SELF_EDIT_ENV",
    "SURFACE_PATH",
    "PROPOSALS_DIRNAME",
    "REPLAY_DIRNAME",
    "OVERRIDES_FILENAME",
    "FROZEN_TIER_MARKERS",
    "SelfEditRejected",
    "HarnessEditProposal",
    "HarnessReplayCase",
    "HarnessEditGate",
    "self_edit_enabled",
    "load_surface",
    "propose",
    "harvest_replay_cases",
    "active_overrides",
]

SELF_EDIT_ENV = "OPENRESEARCH_SELF_EDIT"
SURFACE_PATH = Path(__file__).parent / "self_edit_surface.json"
PROPOSALS_DIRNAME = "_memory/harness_proposals"
REPLAY_DIRNAME = "_memory/replay"
OVERRIDES_FILENAME = "_memory/harness_overrides.json"

_TRUTHY = ("1", "true", "yes", "on")

#: Structurally frozen tier (spec S11.2 bullet 2; DGM lesson). A proposal
#: whose surface_key CONTAINS any of these markers as a substring -- exact
#: key, prefix, or filename anywhere in the string -- is rejected before any
#: other validation. Substring containment deliberately subsumes "exact key"
#: and "prefix" (a string always contains itself / its own prefix) and
#: "filename" (a marker matches wherever it appears in a path-shaped key);
#: broader containment is the "reject harder" direction, never the reverse.
#: Matching is case-INSENSITIVE (see :func:`_frozen_tier_hit`) -- required
#: for the four generic single-word catch-alls (``rubric``/``validator``/
#: ``evidence``/``budget``) to actually catch this repo's UPPERCASE
#: ``OPENRESEARCH_*`` env-key convention; a case-sensitive comparison left
#: them dead code against every real key shape in the system.
FROZEN_TIER_MARKERS: tuple[str, ...] = (
    "self_edit_surface.json", "harness_self_edit.py",
    "OPENRESEARCH_EVIDENCE_GATE", "OPENRESEARCH_EVIDENCE_AUDIT",
    "OPENRESEARCH_ZERO_METRICS_GUARD", "OPENRESEARCH_STUB_METRICS_GUARD",
    "OPENRESEARCH_EVAL_PROVENANCE_GUARD", "OPENRESEARCH_ENV_LIVENESS_GATE",
    "OPENRESEARCH_NO_LEARNING_SIGNAL_GATE", "OPENRESEARCH_PER_MODEL_STATUS_GATE",
    "OPENRESEARCH_EXTERNAL_VALIDATOR", "OPENRESEARCH_VALIDATOR_",
    "OPENRESEARCH_GRADER_", "OPENRESEARCH_CAMPAIGN_MAX_", "OPENRESEARCH_MAX_",
    "rubric", "validator", "evidence", "budget",
)

_EMPTY_SURFACE: dict[str, Any] = {"version": 0, "numeric_keys": {}, "guidance_blocks": {}}


class SelfEditRejected(Exception):
    """A self-edit proposal/promotion was refused. Carries ``.reason`` for
    callers that prefer exception-based flow over the returned status dict
    every public function here actually uses."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class HarnessEditProposal:
    surface_key: str  # a numeric_keys key, or "guidance:" + a guidance_blocks id
    delta: Any  # the proposed value (number or str)
    mined_from: tuple[str, ...]  # evidence refs (artifact paths)


@dataclass(frozen=True)
class HarnessReplayCase:
    case_id: str
    kind: str  # "decide_replay" | "fingerprint_replay"
    inputs: Mapping[str, Any]  # JSON-safe, recorded at harvest time
    expected: Mapping[str, Any]


# --------------------------------------------------------------------------- #
# Flag + whitelist                                                            #
# --------------------------------------------------------------------------- #


def self_edit_enabled() -> bool:
    """Read on every call (no import-time capture), default OFF."""
    return os.environ.get(SELF_EDIT_ENV, "").strip().lower() in _TRUTHY


def load_surface() -> dict:
    """The editable-surface whitelist. Fail-CLOSED: a missing/corrupt/non-dict
    whitelist file degrades to an EMPTY surface (every key becomes
    ``unknown_key``) rather than raising or -- far worse -- being treated as
    unbounded. The whitelist file is itself a frozen-tier member; this is the
    read-time backstop for that invariant."""
    try:
        data = json.loads(SURFACE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return dict(_EMPTY_SURFACE)
    return data if isinstance(data, dict) else dict(_EMPTY_SURFACE)


# --------------------------------------------------------------------------- #
# Small IO helpers (mirrors experience_memory.py / lesson_distiller.py)       #
# --------------------------------------------------------------------------- #


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _proposal_path(runs_root: Path, proposal_id: str) -> Path:
    return Path(runs_root) / PROPOSALS_DIRNAME / f"{proposal_id}.json"


def _proposal_payload(proposal: HarnessEditProposal) -> dict[str, Any]:
    return {"surface_key": proposal.surface_key, "delta": proposal.delta, "mined_from": list(proposal.mined_from)}


def _proposal_id(proposal: HarnessEditProposal) -> str:
    digest = hashlib.sha256(_canonical_json(_proposal_payload(proposal)).encode("utf-8")).hexdigest()
    return digest[:12]


# --------------------------------------------------------------------------- #
# Frozen tier + bounds validation (shared by propose() and apply_default())   #
# --------------------------------------------------------------------------- #


def _frozen_tier_hit(surface_key: str) -> str | None:
    """Case-INSENSITIVE substring match: every real ``OPENRESEARCH_*`` key in
    this repo is uppercase, but the generic catch-all markers
    (``rubric``/``validator``/``evidence``/``budget``) are lowercase words --
    without folding case they never fire against the only naming convention
    this system actually uses."""
    upper_key = surface_key.upper()
    for marker in FROZEN_TIER_MARKERS:
        if marker.upper() in upper_key:
            return marker
    return None


def _numeric_in_bounds(delta: Any, spec: Mapping[str, Any]) -> bool:
    kind = spec.get("kind")
    if kind == "int":
        if not isinstance(delta, int) or isinstance(delta, bool):
            return False
    elif kind == "float":
        if not isinstance(delta, (int, float)) or isinstance(delta, bool):
            return False
    else:
        return False  # unknown/unsupported kind -> fail closed
    lo, hi = spec.get("min"), spec.get("max")
    try:
        return lo <= delta <= hi
    except TypeError:
        return False


def _check_against_surface(surface_key: str, delta: Any, surface: Mapping[str, Any]) -> str | None:
    """``None`` on success; else one of ``"unknown_key"``/``"out_of_bounds"``/
    ``"over_cap"``. The ONE place bounds logic lives -- reused by
    :func:`propose`, :meth:`HarnessEditGate.apply_default`'s at-apply-time
    revalidation, and :func:`active_overrides`'s at-read-time revalidation."""
    if surface_key.startswith("guidance:"):
        block_id = surface_key[len("guidance:") :]
        block_spec = (surface.get("guidance_blocks") or {}).get(block_id)
        if not isinstance(block_spec, Mapping):
            return "unknown_key"
        if not isinstance(delta, str):
            return "out_of_bounds"
        max_chars = block_spec.get("max_chars")
        if not isinstance(max_chars, (int, float)) or isinstance(max_chars, bool) or len(delta) > max_chars:
            return "over_cap"
        return None

    key_spec = (surface.get("numeric_keys") or {}).get(surface_key)
    if not isinstance(key_spec, Mapping):
        return "unknown_key"
    if not _numeric_in_bounds(delta, key_spec):
        return "out_of_bounds"
    return None


# --------------------------------------------------------------------------- #
# propose()                                                                    #
# --------------------------------------------------------------------------- #


def _persist_propose(
    runs_root: Path, proposal_id: str, proposal: HarnessEditProposal, status: str, detail: Mapping[str, Any]
) -> None:
    path = _proposal_path(runs_root, proposal_id)
    record = _read_json_dict(path) or {"id": proposal_id, "proposal": _proposal_payload(proposal), "history": []}
    record["status"] = status
    record["history"] = list(record.get("history") or []) + [
        {"stage": "propose", "at": _now_iso(), "detail": dict(detail)}
    ]
    _atomic_write_json(path, record)


def propose(proposal: HarnessEditProposal, *, runs_root: Path) -> dict:
    """Validate ``proposal`` against the frozen tier then the whitelist, and
    persist the outcome (candidate OR rejected -- every outcome is auditable,
    never silently dropped) to ``runs_root/_memory/harness_proposals/<id>.json``.

    OFF -> ``{"status": "disabled"}`` with NO filesystem access at all (the
    overlay/whitelist machinery is never read while the flag is off).
    """
    if not self_edit_enabled():
        return {"status": "disabled"}

    runs_root = Path(runs_root)
    proposal_id = _proposal_id(proposal)

    frozen_hit = _frozen_tier_hit(proposal.surface_key)
    if frozen_hit is not None:
        result = {"status": "rejected", "reason": "frozen_tier"}
        _persist_propose(runs_root, proposal_id, proposal, "rejected", {**result, "marker": frozen_hit})
        return result

    reason = _check_against_surface(proposal.surface_key, proposal.delta, load_surface())
    result = {"status": "rejected", "reason": reason} if reason is not None else {"status": "candidate", "id": proposal_id}
    _persist_propose(runs_root, proposal_id, proposal, result["status"], dict(result))
    return result


# --------------------------------------------------------------------------- #
# harvest_replay_cases()                                                      #
# --------------------------------------------------------------------------- #


def _latest_by_status(rows: Sequence[Mapping[str, Any]], attempt_n: int) -> dict[str, dict[str, Any]]:
    """Mirrors ``reproduction_campaign.CampaignLedger.latest_by_status``,
    duplicated (not imported) to respect this module's import boundary."""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("attempt_n") == attempt_n and isinstance(row.get("status"), str):
            latest[row["status"]] = dict(row)
    return latest


def harvest_replay_cases(
    run_dir: Path, *, runs_root: Path, state: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> Path | None:
    """Build executable ``HarnessReplayCase``s from THIS campaign's own
    ledger + persisted directives, and write them to
    ``runs_root/_memory/replay/<project>_<launched_at>.json``.

    Fail-soft: any defect anywhere in reconstruction -> ``None``, never
    raises into the campaign terminal that calls this.
    """
    try:
        return _harvest_inner(Path(run_dir), runs_root=Path(runs_root), state=state, rows=rows)
    except Exception:
        return None


def _harvest_inner(
    run_dir: Path, *, runs_root: Path, state: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> Path | None:
    clean_rows = [dict(r) for r in rows if isinstance(r, Mapping)]
    attempt_ns = sorted({r["attempt_n"] for r in clean_rows if isinstance(r.get("attempt_n"), int)})
    if not attempt_ns:
        return None

    budget_raw = state.get("budget")
    if not isinstance(budget_raw, Mapping):
        return None

    by_attempt = {n: _latest_by_status(clean_rows, n) for n in attempt_ns}
    cases: list[dict[str, Any]] = []

    cases.extend(_harvest_decide_cases(attempt_ns, by_attempt, budget_raw))
    cases.extend(_harvest_fingerprint_cases(run_dir, attempt_ns, by_attempt))

    if not cases:
        return None

    project_id = state.get("project_id") or run_dir.name
    launched_ats = [
        float(r["launched_at"]) for r in clean_rows if r.get("status") == "launched" and "launched_at" in r
    ]
    stamp = max(launched_ats) if launched_ats else 0.0
    out_path = runs_root / REPLAY_DIRNAME / f"{project_id}_{stamp:.6f}.json"
    _atomic_write_json(
        out_path, {"project_id": project_id, "run_dir": str(run_dir), "harvested_at": _now_iso(), "cases": cases}
    )
    return out_path


def _harvest_decide_cases(
    attempt_ns: Sequence[int], by_attempt: Mapping[int, Mapping[str, Any]], budget_raw: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """One ``decide_replay`` case per attempt with both an assessed row and
    its own decided row. Reconstruction limitation (v1, honestly documented):
    ``PolicyConfig`` knobs (plateau_k/width/width_skip_score/ladder_len) and
    ``next_estimate`` are CLI/env config never persisted to disk, so this
    replays against the repo's documented defaults
    (``OPENRESEARCH_CAMPAIGN_PLATEAU_K``=2, ``_WIDTH``=1, single-rung ladder)
    and a zero next_estimate. A recorded ``"budget_floor"`` decision is
    SKIPPED for the same reason -- its original ``next_estimate`` came from
    ``opts.est_gpu_hours``, which is unrecoverable from disk, and a zero
    substitute could never reproduce that rule firing."""
    cases: list[dict[str, Any]] = []
    running_spend = {"llm_usd": 0.0, "gpu_usd": 0.0, "gpu_hours": 0.0, "wall_s": 0.0}
    scope_rung_by_attempt: dict[int, int] = {1: 0}
    lineage_by_attempt: dict[int, str] = {}
    assessments_so_far: list[dict[str, Any]] = []

    for n in attempt_ns:
        assessed_row = by_attempt[n].get("assessed")
        if not isinstance(assessed_row, Mapping) or not isinstance(assessed_row.get("assessment"), Mapping):
            continue
        assessment_dict = dict(assessed_row["assessment"])
        assessments_so_far.append(assessment_dict)
        cost = assessment_dict.get("cost") or {}
        for meter in running_spend:
            running_spend[meter] += float(cost.get(meter, 0.0) or 0.0)

        decided_row = by_attempt[n].get("decided")
        decision = decided_row.get("decision") if isinstance(decided_row, Mapping) else None
        if (
            isinstance(decision, Mapping)
            and isinstance(decision.get("kind"), str)
            and decision.get("rule") != "budget_floor"
        ):
            cases.append({
                "case_id": f"decide_{n}",
                "kind": "decide_replay",
                "inputs": {
                    "assessments": list(assessments_so_far),
                    "budget": dict(budget_raw),
                    "spent": dict(running_spend),
                    "config": {
                        "max_attempts": int(budget_raw.get("max_attempts", 6)),
                        "plateau_k": 2, "width": 1, "width_skip_score": 0.5, "ladder_len": 1,
                    },
                    "next_estimate": {"llm_usd": 0.0, "gpu_usd": 0.0, "gpu_hours": 0.0, "wall_s": 0.0},
                    "lineage_by_attempt": dict(lineage_by_attempt),
                    "scope_rung_by_attempt": dict(scope_rung_by_attempt),
                    "current_rung": scope_rung_by_attempt.get(n, 0),
                    "blocking_gap": None,
                },
                "expected": {"kind": decision["kind"], "rule": decision.get("rule")},
            })

        if isinstance(decision, Mapping) and decision.get("kind") == "CONTINUE":
            next_plan = decision.get("next_plan") or {}
            if isinstance(next_plan.get("lineage"), str):
                lineage_by_attempt[n + 1] = next_plan["lineage"]
            if isinstance(next_plan.get("scope_rung"), int):
                scope_rung_by_attempt[n + 1] = next_plan["scope_rung"]

    return cases


def _harvest_fingerprint_cases(
    run_dir: Path, attempt_ns: Sequence[int], by_attempt: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """One ``fingerprint_replay`` case per launched row with a persisted
    ``directives/<n>.json``. v1 reconstruction (matching
    ``campaign_composition``'s own documented simplification): the
    ``runs_dir_hint`` always keys the LATEST assessed attempt, so a non-
    "fresh" lineage's ``seed_attempt_n`` is always the immediately preceding
    attempt."""
    cases: list[dict[str, Any]] = []
    for n in attempt_ns:
        launched_row = by_attempt[n].get("launched")
        if not isinstance(launched_row, Mapping):
            continue
        expected_sha = launched_row.get("directives_sha256")
        if not isinstance(expected_sha, str) or not expected_sha:
            continue

        directives_data = _read_json_dict(run_dir / "campaign" / "directives" / f"{n}.json")
        if directives_data is None:
            continue
        seed_lineage = directives_data.get("seed_lineage")
        scope_rung = directives_data.get("scope_rung")
        envelope = directives_data.get("envelope")
        if not isinstance(seed_lineage, str) or not isinstance(scope_rung, int) or not isinstance(envelope, Mapping):
            continue

        repair_action_kinds: list[str] = []
        leaf_repair_plan = directives_data.get("leaf_repair_plan")
        if isinstance(leaf_repair_plan, Mapping) and isinstance(leaf_repair_plan.get("plan"), list):
            repair_action_kinds = [
                entry.get("repair_class")
                for entry in leaf_repair_plan["plan"]
                if isinstance(entry, Mapping) and isinstance(entry.get("repair_class"), str)
            ]

        failure_classes: list[str] = []
        prior_assessed = by_attempt.get(n - 1, {}).get("assessed") if n > 1 else None
        if isinstance(prior_assessed, Mapping) and isinstance(prior_assessed.get("assessment"), Mapping):
            fclass = prior_assessed["assessment"].get("failure_class")
            if isinstance(fclass, str):
                failure_classes = [fclass]

        seed_attempt_n = None if seed_lineage == "fresh" else (n - 1)
        composite_lineage = f"{seed_lineage}:{seed_attempt_n or 0}"

        cases.append({
            "case_id": f"fingerprint_{n}",
            "kind": "fingerprint_replay",
            "inputs": {
                "seed_lineage": composite_lineage,
                "scope_rung": scope_rung,
                "repair_action_kinds": repair_action_kinds,
                "failure_classes": failure_classes,
                "envelope": dict(envelope),
            },
            "expected": {"directives_sha256": expected_sha},
        })
    return cases


# --------------------------------------------------------------------------- #
# Case execution (the REAL functions, reconstructed from case.inputs)         #
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _env_overlay(overrides: Mapping[str, str]):
    """Monkeypatch-style ``os.environ`` overlay, scoped to the ``with``
    block only. Restores the exact prior state (present-with-value, or
    absent) on exit, even on an exception."""
    sentinel = object()
    saved: dict[str, Any] = {}
    for key, value in overrides.items():
        saved[key] = os.environ.get(key, sentinel)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, prior in saved.items():
            if prior is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


def _execute_decide_case(inputs: Mapping[str, Any]) -> dict[str, Any]:
    assessments = [AttemptAssessment.from_dict(a) for a in inputs["assessments"]]
    result = decide(
        assessments,
        budget=CampaignBudget(**dict(inputs["budget"])),
        spent=CampaignSpend(**dict(inputs["spent"])),
        config=PolicyConfig(**dict(inputs["config"])),
        next_estimate=CampaignSpend(**dict(inputs["next_estimate"])),
        lineage_by_attempt={int(k): v for k, v in dict(inputs["lineage_by_attempt"]).items()},
        scope_rung_by_attempt={int(k): int(v) for k, v in dict(inputs["scope_rung_by_attempt"]).items()},
        runs_dir_hint={},  # never affects kind/rule -- only next_plan's own pointer choice
        current_rung=int(inputs["current_rung"]),
        blocking_gap=inputs.get("blocking_gap"),
    )
    return {"kind": result.kind, "rule": result.rule}


def _execute_fingerprint_case(inputs: Mapping[str, Any]) -> dict[str, Any]:
    fp = directives_fingerprint(
        seed_lineage=str(inputs["seed_lineage"]),
        scope_rung=int(inputs["scope_rung"]),
        repair_action_kinds=list(inputs["repair_action_kinds"]),
        failure_classes=list(inputs["failure_classes"]),
        envelope=dict(inputs["envelope"]),
    )
    return {"directives_sha256": fp}


#: Case-kind -> executor registry. Module-level and monkeypatchable BY DESIGN
#: (not merely an implementation detail): the real executors above are
#: provably env-insensitive (``decide``/``directives_fingerprint`` read no
#: environment at all), so the negative-control rejection path can only be
#: exercised end-to-end by injecting a knob-sensitive fake executor here.
_CASE_EXECUTORS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "decide_replay": _execute_decide_case,
    "fingerprint_replay": _execute_fingerprint_case,
}


def _execute_case(case: HarnessReplayCase) -> dict[str, Any]:
    executor = _CASE_EXECUTORS.get(case.kind)
    if executor is None:
        raise ValueError(f"unknown replay case kind {case.kind!r}")
    return executor(case.inputs)


def _case_from_dict(raw: Any) -> HarnessReplayCase | None:
    if not isinstance(raw, Mapping):
        return None
    case_id, kind = raw.get("case_id"), raw.get("kind")
    inputs, expected = raw.get("inputs"), raw.get("expected")
    if not isinstance(case_id, str) or not isinstance(kind, str):
        return None
    if not isinstance(inputs, Mapping) or not isinstance(expected, Mapping):
        return None
    return HarnessReplayCase(case_id=case_id, kind=kind, inputs=dict(inputs), expected=dict(expected))


# --------------------------------------------------------------------------- #
# Canary A/B validation                                                       #
# --------------------------------------------------------------------------- #


def _validate_ab_reports(ab_reports: Sequence[Mapping[str, Any]], *, grader_sigma: float) -> str | None:
    """``None`` on success; else the specific rejection reason (F12)."""
    reports = [r for r in ab_reports if isinstance(r, Mapping)]
    if len(reports) != len(list(ab_reports)):
        return "malformed_report_row"

    by_paper: dict[Any, set[Any]] = {}
    for r in reports:
        paper_id, seed = r.get("paper_id"), r.get("seed")
        if paper_id is None or seed is None:
            return "malformed_report_row"
        by_paper.setdefault(paper_id, set()).add(seed)

    if len(by_paper) < 2:
        return "insufficient_papers"
    if any(len(seeds) < 2 for seeds in by_paper.values()):
        return "insufficient_seeds"

    for r in reports:
        path = r.get("report_path")
        if not path or not Path(path).exists():
            return f"fabricated_evidence:{path}"

    pairs: dict[tuple[Any, Any], dict[str, float]] = {}
    for r in reports:
        arm = r.get("arm")
        if arm not in ("edit", "control"):
            return "malformed_report_row"
        try:
            score = float(r.get("score"))
        except (TypeError, ValueError):
            return "malformed_report_row"
        pairs.setdefault((r.get("paper_id"), r.get("seed")), {})[arm] = score

    deltas: list[float] = []
    for (paper_id, seed), arms in pairs.items():
        if "edit" not in arms or "control" not in arms:
            return f"incomplete_pairing:{paper_id}:{seed}"
        deltas.append(arms["edit"] - arms["control"])

    if not deltas:
        return "incomplete_pairing"

    mean_delta = sum(deltas) / len(deltas)
    if not (mean_delta > grader_sigma):
        return "insufficient_improvement"

    return None


# --------------------------------------------------------------------------- #
# Overrides file                                                              #
# --------------------------------------------------------------------------- #


def _write_override(runs_root: Path, surface_key: str, delta: Any) -> None:
    path = Path(runs_root) / OVERRIDES_FILENAME
    current = _read_json_dict(path) or {}
    current[surface_key] = delta
    _atomic_write_json(path, current)


def active_overrides(runs_root: Path) -> dict:
    """``{}`` unless :func:`self_edit_enabled`. Bounds-revalidated on READ
    too: an override file edited out-of-band to an out-of-bounds/unknown
    value is dropped, with the dropped keys surfaced under ``"_dropped"`` --
    fail-closed consumption, never a silent stale value riding the child env.
    Also frozen-tier-revalidated on read (defense-in-depth, independent of
    the bounds check): a hand-edited overrides file cannot smuggle a frozen
    key into a child env even under a compromised whitelist that would
    otherwise accept it as an ordinary bounded numeric key.
    """
    if not self_edit_enabled():
        return {}
    raw = _read_json_dict(Path(runs_root) / OVERRIDES_FILENAME)
    if raw is None:
        return {}

    surface = load_surface()
    result: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in raw.items():
        if _frozen_tier_hit(key) is not None:
            dropped.append(key)
        elif _check_against_surface(key, value, surface) is None:
            result[key] = value
        else:
            dropped.append(key)
    if dropped:
        result["_dropped"] = dropped
    return result


# --------------------------------------------------------------------------- #
# HarnessEditGate                                                             #
# --------------------------------------------------------------------------- #


def _persist_stage(
    runs_root: Path, proposal_id: str, record: Mapping[str, Any], *, status: str, stage: str, detail: Mapping[str, Any]
) -> None:
    updated = dict(record)
    updated["status"] = status
    updated["history"] = list(updated.get("history") or []) + [
        {"stage": stage, "at": _now_iso(), "detail": dict(detail)}
    ]
    _atomic_write_json(_proposal_path(runs_root, proposal_id), updated)


def _reject_if_frozen(runs_root: Path, proposal_id: str, record: Mapping[str, Any], stage: str) -> dict | None:
    """Defense-in-depth layer, independent of the persisted status: re-derive
    the frozen-tier verdict straight from the record's OWN proposal payload,
    at every gate boundary, every call -- never trusts that ``propose()``'s
    original check (or the current whitelist) is still trustworthy. Returns
    the (already-persisted) terminal rejection dict, or ``None`` when the
    surface_key is not frozen. Mirrors :func:`propose`'s exact shape: the
    marker name is persisted for audit but not echoed in the return value.
    """
    proposal = record.get("proposal") or {}
    frozen_hit = _frozen_tier_hit(str(proposal.get("surface_key", "")))
    if frozen_hit is None:
        return None
    result = {"status": "rejected", "reason": "frozen_tier"}
    _persist_stage(runs_root, proposal_id, record, status="rejected", stage=stage, detail={**result, "marker": frozen_hit})
    return result


class HarnessEditGate:
    """Dedicated Phase-C gate (Codex F11 -- ``held_out_gate`` stays lesson-only)."""

    def __init__(self, *, runs_root: Path, grader_sigma: float = 0.0067) -> None:
        self.runs_root = Path(runs_root)
        self.grader_sigma = grader_sigma

    def _load(self, proposal_id: str) -> dict[str, Any] | None:
        return _read_json_dict(_proposal_path(self.runs_root, proposal_id))

    def _load_corpus(self) -> tuple[list[HarnessReplayCase], str | None]:
        replay_dir = self.runs_root / REPLAY_DIRNAME
        if not replay_dir.exists():
            return [], None
        cases: list[HarnessReplayCase] = []
        for path in sorted(replay_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return [], f"corpus:{path.stem}"
            if not isinstance(data, Mapping) or not isinstance(data.get("cases"), list):
                return [], f"corpus:{path.stem}"
            for raw_case in data["cases"]:
                case = _case_from_dict(raw_case)
                if case is None:
                    return [], f"corpus:{path.stem}"
                cases.append(case)
        return cases, None

    def shadow(self, proposal_id: str) -> dict:
        """Replay ``proposal_id`` against the ENTIRE replay corpus. Empty
        corpus -> stays candidate (fail-soft cold-start, spec risk 4). ANY
        case raising, failing to reconstruct, or no longer matching its
        recorded baseline rejects the whole proposal (fail-closed, F11).

        Two independent defenses gate entry, both checked before any replay
        logic runs: (1) the frozen tier is re-derived from the persisted
        proposal, never trusted from a prior verdict; (2) stage order is
        structural -- only a proposal whose persisted status is EXACTLY
        ``"candidate"`` may be shadowed at all, so a ``"rejected"`` (frozen
        tier or otherwise), missing, or already-further-along status is
        refused terminally (reason ``"stage_order"``) rather than silently
        laundered forward.
        """
        if not self_edit_enabled():
            return {"status": "disabled"}

        record = self._load(proposal_id)
        if record is None:
            return {"status": "rejected", "reason": "proposal_not_found"}

        frozen = _reject_if_frozen(self.runs_root, proposal_id, record, "shadow")
        if frozen is not None:
            return frozen

        def _finish(status: str, **detail: Any) -> dict:
            result = {"status": status, **detail}
            _persist_stage(self.runs_root, proposal_id, record, status=status, stage="shadow", detail=result)
            return result

        if record.get("status") != "candidate":
            return _finish("rejected", reason="stage_order")

        corpus, corpus_error = self._load_corpus()
        if corpus_error is not None:
            return _finish("rejected", reason=f"replay_error:{corpus_error}")
        if not corpus:
            return _finish("candidate", reason="replay_corpus_empty")

        proposal = record.get("proposal") or {}
        surface_key = str(proposal.get("surface_key", ""))
        overlay = {} if surface_key.startswith("guidance:") else {surface_key: str(proposal.get("delta"))}

        for case in corpus:
            try:
                baseline = _execute_case(case)
            except Exception:
                return _finish("rejected", reason=f"replay_error:{case.case_id}")

            if baseline != dict(case.expected):
                return _finish("rejected", reason=f"corpus_stale:{case.case_id}")

            try:
                with _env_overlay(overlay):
                    overlay_result = _execute_case(case)
            except Exception:
                return _finish("rejected", reason=f"replay_error:{case.case_id}")

            if overlay_result != baseline:
                return _finish("rejected", reason=f"negative_control_regression:{case.case_id}")

        return _finish("shadow_passed")

    def promote_to_canary(self, proposal_id: str, *, ab_reports: Sequence[Mapping[str, Any]]) -> dict:
        """F12 strengthened canary. Requires prior status ``"shadow_passed"``
        (stage order is structural). Re-derives the frozen tier from the
        persisted proposal FIRST, independent of that stage-order check --
        defense-in-depth against a compromised whitelist or a tampered
        record."""
        if not self_edit_enabled():
            return {"status": "disabled"}

        record = self._load(proposal_id)
        if record is None:
            return {"status": "rejected", "reason": "proposal_not_found"}

        frozen = _reject_if_frozen(self.runs_root, proposal_id, record, "canary")
        if frozen is not None:
            return frozen

        def _finish(status: str, *, persist_status: str | None = None, **detail: Any) -> dict:
            result = {"status": status, **detail}
            _persist_stage(
                self.runs_root, proposal_id, record,
                status=persist_status if persist_status is not None else status,
                stage="canary", detail=result,
            )
            return result

        if record.get("status") != "shadow_passed":
            # Prerequisite ordering, not evidence -- retryable.
            return _finish("rejected", persist_status=record.get("status"), reason="shadow_not_passed")

        reason = _validate_ab_reports(ab_reports, grader_sigma=self.grader_sigma)
        if reason is not None:
            return _finish("rejected", reason=reason)

        return _finish("canary")

    def apply_default(self, proposal_id: str, *, operator_confirmed: bool) -> dict:
        """S17: NEVER automatic. Refuses unless ``operator_confirmed is True``
        (literally, not merely truthy) AND the prior status is ``"canary"``;
        re-validates bounds against the CURRENT whitelist at apply time.
        Re-derives the frozen tier from the persisted proposal FIRST, before
        even the operator-confirmation check -- under a compromised
        whitelist, bounds revalidation alone is no defense; this is."""
        if not self_edit_enabled():
            return {"status": "disabled"}

        record = self._load(proposal_id)
        if record is None:
            return {"status": "rejected", "reason": "proposal_not_found"}

        frozen = _reject_if_frozen(self.runs_root, proposal_id, record, "apply_default")
        if frozen is not None:
            return frozen

        def _finish(status: str, *, persist_status: str | None = None, **detail: Any) -> dict:
            result = {"status": status, **detail}
            _persist_stage(
                self.runs_root, proposal_id, record,
                status=persist_status if persist_status is not None else status,
                stage="apply_default", detail=result,
            )
            return result

        if operator_confirmed is not True:
            # Prerequisite ordering, not evidence -- retryable.
            return _finish("rejected", persist_status=record.get("status"), reason="operator_confirmation_required")

        if record.get("status") != "canary":
            return _finish("rejected", persist_status=record.get("status"), reason="canary_not_reached")

        proposal = record.get("proposal") or {}
        surface_key, delta = proposal.get("surface_key"), proposal.get("delta")
        reason = _check_against_surface(str(surface_key), delta, load_surface())
        if reason is not None:
            return _finish("rejected", reason=reason)

        _write_override(self.runs_root, str(surface_key), delta)
        return _finish("default")
