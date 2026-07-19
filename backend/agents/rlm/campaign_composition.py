"""``build_campaign`` composition root (Unit 9a).

Wires every landed campaign module (``reproduction_campaign``,
``campaign_policy``, ``attempt_assessment``, ``campaign_directives``,
``attempt_driver``, ``understanding_gate``, ``campaign_report``,
``run_spec_contract``, ``attempt_isolation``, the DISTILL memory modules)
into one real ``CampaignStages`` instance and returns a runnable
``ReproductionCampaign``. Deliberately not in the spec's §15 file map --
pure glue; every non-trivial decision delegates to the module that owns it.

v1 simplifications:
  - ``understand``'s extractor is deterministic (no LLM), so the two
    double-extraction passes always agree -- UNDERSTAND never blocks
    attempt 1 in v1 (the gate machinery is live for a future span-grounded
    extractor).

``runs_dir_hint`` (the seed-pointer source for DECIDE/lineage_arms) resolves
EVERY assessed attempt, not just the latest. It used to carry only the latest,
on the reasoning that an attempt's ``code/`` is unambiguously identifiable only
while it is the current ``run_dir/code`` -- but that made a non-latest champion
structurally unseedable (``lineage_arms`` drops an arm with no pointer, so the
champion arm silently degraded to ``fresh``), and the champion is routinely not
the latest attempt. The driver now records each attempt's ``code/`` under its
attempt number in ``campaign/attempt_code_index.json`` as the pre-launch archive
moves it (``attempt_driver.rotate_attempt_code_index``), so an archived
attempt stays addressable; :func:`_gather_campaign_view` reads that index --
per attempt's OWN launch run dir, so width children (``<id>_w<k>``) resolve
correctly too -- and falls back to the live ``code/`` for campaigns that predate
the index. An attempt whose code is genuinely gone still has no hint and still
degrades to ``fresh``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.agents.rlm import external_validator, failure_attribution, lesson_distiller
from backend.agents.rlm.attempt_assessment import AttemptAssessment, assess_attempt, rubric_sha256
from backend.agents.rlm.attempt_driver import (
    AttemptDriver,
    AttemptHandle,
    LiveCliDriver,
    PairedDriver,
    UnifiedRunDriver,
    resolve_attempt_code_dir,
    rotate_attempt_code_index,
)
from backend.agents.rlm.campaign_directives import AttemptDirectives, synthesize_directives
from backend.agents.rlm.campaign_policy import (
    AttemptEnvelope,
    CampaignBudget,
    EnforceabilityError,
    EnforcementContext,
    EnforcementPlan,
    EnvelopeExhausted,
    NextAttemptPlan,
    PolicyConfig,
    attempt_estimate,
    campaign_floor,
    check_enforceability,
    decide as campaign_decide,
    derive_envelope,
    lineage_arms,
)
from backend.agents.rlm.campaign_report import write_campaign_report, write_plan_only_report
from backend.agents.rlm.campaign_types import CampaignSpend
from backend.agents.rlm.experience_memory import ExperienceMemory
from backend.agents.rlm.harness_self_edit import active_overrides, harvest_replay_cases, self_edit_enabled
from backend.agents.rlm.recipe_library import admit_recipe
from backend.agents.rlm.reproduction_campaign import (
    CampaignInitError,
    CampaignLedger,
    CampaignState,
    CampaignStages,
    InFlight,
    ReproductionCampaign,
    default_liveness_probe,
)
from backend.agents.rlm.run_spec_contract import (
    RunSpecValidationError,
    apply_run_spec,
    run_spec_key_applies,
    validate_run_spec_keys,
)
from backend.agents.rlm.understanding_gate import ExtractedField, run_understanding
from backend.services.runs.attempt_isolation import force_archive_incomplete

__all__ = ["CampaignOptions", "build_campaign", "mint_width_project_ids"]

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")
# Per-attempt driver-owned keys the canonical run-spec profile must never
# set (attempt_driver.build_attempt_env sets these per attempt).
_DRIVER_OWNED_ENV_KEYS: frozenset[str] = frozenset(
    {
        "OPENRESEARCH_BASELINE_EXTRA_GUIDANCE",
        "OPENRESEARCH_SEED_BEST_ATTEMPT",
        "OPENRESEARCH_TARGET_BEST_FLOOR",
        "OPENRESEARCH_MAX_RUN_GPU_USD",
    }
)


@dataclass(frozen=True)
class CampaignOptions:
    paper_ref: str
    runs_root: Path
    repo_root: Path
    max_llm_usd: float
    max_gpu_usd: float
    max_gpu_hours: float
    max_attempts: int
    wall_clock_s: float | None
    mode: str
    driver: str
    width: int
    plateau_k: int
    sandbox: str
    billing_sandbox: str | None
    gpu_usd_per_hr: float | None
    est_gpu_hours: float
    run_spec_path: str
    scope_spec: str | None
    scope_ladder: tuple[str, ...]
    paper_hint: str | None
    require_cpu_tier: bool
    arxiv_id: str | None
    paper_class: str
    resume: bool
    root_model: str | None = None
    execution_mode: str = "max"
    gpu_mode: str = "auto"
    minimize_compute: bool = False


# --- small stateless helpers ------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mint_width_project_ids(project_id: str, k: int) -> list[str]:
    """Child project ids for budget-gated width (spec §8.3, Codex F16): the
    campaign mints ``<project_id>_w1..k`` itself and passes them through the
    EXISTING ``--project-id`` override (``register_project`` already
    ingests a non-canonical id verbatim, per Unit 9a's mirror-guard check)."""
    return [f"{project_id}_w{i}" for i in range(1, k + 1)]


def _count_prior_width_launches(project_id: str, rows: Sequence[Mapping[str, Any]], *, attempt_n: int) -> int:
    """How many DIFFERENT prior attempts already launched under a
    ``<project_id>_w<n>`` child id -- excludes THIS ``attempt_n``'s own rows
    so a re-plan of the same attempt (the orphaned-intent-relaunch path,
    spec §5) recomputes the identical child id rather than minting a new
    one (mirrors ``_plan_attempt_impl``'s existing own-attempt exclusion
    for the novelty fingerprint domain)."""
    prefix = f"{project_id}_w"
    count = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "launched":
            continue
        if row.get("attempt_n") == attempt_n:
            continue
        candidate = row.get("project_id")
        if isinstance(candidate, str) and candidate.startswith(prefix) and candidate[len(prefix):].isdigit():
            count += 1
    return count


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- fail-soft artifact read: missing/corrupt -> empty
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _rubric_sha256_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    tree = _read_json_dict(path)
    return rubric_sha256(tree) if tree else None


def _gpu_plan_rate_and_count(run_dir: Path) -> tuple[float | None, int]:
    plan = _read_json_dict(run_dir / "rlm_state" / "gpu_plan.json")
    raw_count = plan.get("gpu_count")
    gpu_count = int(raw_count) if isinstance(raw_count, (int, float)) and raw_count >= 1 else 1
    total_rate = plan.get("total_usd_per_hr")
    if isinstance(total_rate, (int, float)):
        return float(total_rate), gpu_count
    sku_rate = plan.get("sku_usd_per_hr")
    if isinstance(sku_rate, (int, float)):
        return float(sku_rate) * gpu_count, gpu_count
    return None, gpu_count


def _enforcement_ctx(run_dir: Path, opts: CampaignOptions, mode: str) -> tuple[EnforcementContext, float | None]:
    rate, gpu_count = _gpu_plan_rate_and_count(run_dir)
    gpu_usd_per_hr = opts.gpu_usd_per_hr if opts.gpu_usd_per_hr is not None else rate
    ctx = EnforcementContext(
        driver_kind=opts.driver,
        sandbox=opts.billing_sandbox or opts.sandbox,
        mode=mode,
        tiering_strategy=None,  # LiveCliDriver has no VM tiering strategy
        max_gpu_count=gpu_count,
        gpu_usd_per_hr=gpu_usd_per_hr,
        require_cpu_tier=opts.require_cpu_tier,
    )
    return ctx, gpu_usd_per_hr


def _enforcement_mapping(
    plan: EnforcementPlan, *, paper_hint: str | None = None, sandbox: str | None = None
) -> dict[str, Any]:
    """Built strictly from ``EnforcementPlan`` fields -- ALWAYS
    ``plan.vm_ceiling_s`` (post-co-tightening), never
    ``AttemptEnvelope.vm_ceiling_s``. ``cli_args`` is list-of-lists so it
    round-trips through JSON byte-identically. When ``paper_hint`` is set,
    ``["--paper-hint", paper_hint]`` is appended to ``cli_args`` -- the
    driver's argv builder (``attempt_driver.build_reproduce_argv``) already
    emits every ``cli_args`` pair verbatim, so no driver-side change is
    needed for the child to receive it. ``sandbox`` rides the same channel:
    without it the child falls to the repo DEFAULT sandbox (runpod), not the
    campaign's -- the 2026-07-02 live SDAR launch died at RunPod preflight on
    a GCP VM exactly this way."""
    cli_args = [[flag, value] for flag, value in plan.cli_args]
    if paper_hint:
        cli_args.append(["--paper-hint", paper_hint])
    if sandbox:
        cli_args.append(["--sandbox", sandbox])
    return {
        "cli_args": cli_args,
        "env": dict(plan.env),
        "effective_wall_s": plan.effective_wall_s,
        "vm_ceiling_s": plan.vm_ceiling_s,
        "provision_charged_to_gpu": plan.provision_charged_to_gpu,
        "notes": list(plan.notes),
    }


# --- validate_init / understand ---------------------------------------------


def _validate_init_impl(opts: CampaignOptions) -> list[str]:
    try:
        raw_text = Path(opts.run_spec_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise CampaignInitError(f"cannot read campaign run-spec profile {opts.run_spec_path!r}: {exc}") from exc
    try:
        profile = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise CampaignInitError(f"campaign run-spec profile {opts.run_spec_path!r} is not valid JSON: {exc}") from exc
    if not isinstance(profile, dict):
        raise CampaignInitError(f"campaign run-spec profile {opts.run_spec_path!r} must be a JSON object")

    try:
        validate_run_spec_keys(set(profile) | _DRIVER_OWNED_ENV_KEYS)
    except RunSpecValidationError as exc:
        raise CampaignInitError(str(exc)) from exc

    overlap = sorted(set(profile) & _DRIVER_OWNED_ENV_KEYS)
    if overlap:
        # Fail-closed: the child applies --run-spec AFTER env inheritance, so
        # a profile value would clobber the driver's per-attempt env.
        raise CampaignInitError(
            f"campaign run-spec profile must not set per-attempt driver-owned keys {overlap}"
        )

    notices: list[str] = []
    validator_flag = str(profile.get("OPENRESEARCH_EXTERNAL_VALIDATOR", "")).strip().lower()
    if validator_flag not in _TRUTHY:
        notices.append("validator_not_configured: REPRODUCED is unreachable unattended (spec §8.2 rule 1)")
    billing_sandbox = opts.billing_sandbox or opts.sandbox
    if opts.gpu_usd_per_hr is None and billing_sandbox != "local":
        notices.append("no_gpu_rate: gpu_usd enforceability depends on gpu_plan.json at plan time")
    return notices


def _understand_impl(run_dir: Path, opts: CampaignOptions) -> dict[str, Any]:
    # Deterministic, LLM-free v1 extractor: identical output for every
    # variant, so the double-extraction diff never disagrees (spec F9 --
    # UNDERSTAND blocking is a structural no-op until a span-grounded
    # extractor lands).
    def _extract(_variant: str) -> dict[str, ExtractedField]:
        fields: dict[str, ExtractedField] = {
            "paper_ref": ExtractedField(name="paper_ref", value=opts.paper_ref, source_span=None),
            "rubric_present": ExtractedField(
                name="rubric_present", value=(run_dir / "generated_rubric.json").exists(), source_span=None
            ),
            "parsed_text_present": ExtractedField(
                name="parsed_text_present", value=(run_dir / "parsed_full_text.txt").exists(), source_span=None
            ),
        }
        if opts.scope_spec:
            fields["scope_spec"] = ExtractedField(name="scope_spec", value=opts.scope_spec, source_span=None)
        return fields

    result = run_understanding(
        extract=_extract, lint=None, probe_assets=None, out_path=run_dir / "campaign" / "understanding.json"
    )
    out: dict[str, Any] = {"sha256": result.sha256, "blocking": list(result.blocking)}
    observed = _rubric_sha256_if_exists(run_dir / "generated_rubric.json")
    if observed is not None:
        out["rubric_sha256"] = observed
    return out


# --- plan_attempt ------------------------------------------------------------


def _launch_run_dir_by_attempt(rows: Sequence[Mapping[str, Any]]) -> dict[int, Path]:
    """Each attempt's OWN launch run dir, off its ``launched`` ledger row.

    Usually the campaign's own run dir, but a budget-gated width attempt
    (spec §8.3) launches under ``<project_id>_w<k>`` and writes its ``code/``
    -- and its attempt-code index -- there, not under the campaign's
    top-level dir. Resolving an attempt's code against the campaign run dir
    would silently miss every width child's tree.
    """
    out: dict[int, Path] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "launched":
            continue
        attempt_n, raw_dir = row.get("attempt_n"), row.get("run_dir")
        if isinstance(attempt_n, int) and isinstance(raw_dir, str) and raw_dir:
            out[attempt_n] = Path(raw_dir)
    return out


def _gather_campaign_view(
    run_dir: Path, rows: Sequence[Mapping[str, Any]]
) -> tuple[list[AttemptAssessment], dict[int, str], dict[int, int], dict[int, str]]:
    """Rebuild assessments (latest-per-attempt), lineage_by_attempt /
    scope_rung_by_attempt (read off the PRIOR attempt's decided
    ``next_plan``; attempt 1 starts at rung 0), and ``runs_dir_hint`` (EVERY
    assessed attempt whose code still exists -- live or archived; see the
    module docstring)."""
    attempt_ns = sorted(
        {row.get("attempt_n") for row in rows if isinstance(row, dict) and isinstance(row.get("attempt_n"), int)}
    )
    by_attempt = {n: CampaignLedger.latest_by_status(rows, n) for n in attempt_ns}

    assessments: list[AttemptAssessment] = []
    for n in attempt_ns:
        assessed_row = by_attempt[n].get("assessed")
        if isinstance(assessed_row, dict) and isinstance(assessed_row.get("assessment"), dict):
            assessments.append(AttemptAssessment.from_dict(assessed_row["assessment"]))

    lineage_by_attempt: dict[int, str] = {}
    scope_rung_by_attempt: dict[int, int] = {1: 0}
    for n in attempt_ns:
        decided_row = by_attempt[n].get("decided")
        if not isinstance(decided_row, dict):
            continue
        decision = decided_row.get("decision")
        if not isinstance(decision, dict) or decision.get("kind") != "CONTINUE":
            continue
        next_plan = decision.get("next_plan") or {}
        if isinstance(next_plan.get("lineage"), str):
            lineage_by_attempt[n + 1] = next_plan["lineage"]
        if isinstance(next_plan.get("scope_rung"), int):
            scope_rung_by_attempt[n + 1] = next_plan["scope_rung"]

    # Seed pointers for EVERY assessed attempt, resolved through the
    # driver-maintained attempt-code index (live tree while it is live, the
    # archived tree afterwards). A champion that is not the latest assessed
    # attempt used to have no pointer at all, so lineage_arms dropped its arm
    # and campaign_policy's guard-filtered champion could never actually be
    # seeded -- every attempt cold-started.
    launch_dirs = _launch_run_dir_by_attempt(rows)
    latest_assessed_n = max((a.attempt_n for a in assessments), default=None)
    runs_dir_hint: dict[int, str] = {}
    for assessment in assessments:
        n = assessment.attempt_n
        resolved = resolve_attempt_code_dir(launch_dirs.get(n, run_dir), n)
        if resolved is not None:
            runs_dir_hint[n] = resolved

    # Back-compat: a campaign that launched before the index existed has no
    # index to read, but its latest assessed attempt's code is still live at
    # the un-archived run_dir/code -- exactly the one pointer the old
    # latest-only hint carried, so a resumed pre-index campaign keeps working.
    if latest_assessed_n is not None and latest_assessed_n not in runs_dir_hint:
        legacy_code = launch_dirs.get(latest_assessed_n, run_dir) / "code"
        if legacy_code.is_dir():
            runs_dir_hint[latest_assessed_n] = str(legacy_code)

    return assessments, lineage_by_attempt, scope_rung_by_attempt, runs_dir_hint


def _base_next_plan(rows: Sequence[Mapping[str, Any]], attempt_n: int) -> NextAttemptPlan:
    """Read back what DECIDE already committed for this attempt (the prior
    attempt's decided ``next_plan``), or a fresh plan for attempt 1 / no
    prior CONTINUE. PLAN never recomputes DECIDE's termination logic."""
    prior_n = attempt_n - 1
    decided_row = CampaignLedger.latest_by_status(rows, prior_n).get("decided") if prior_n >= 1 else None
    if isinstance(decided_row, dict):
        decision = decided_row.get("decision")
        if isinstance(decision, dict) and decision.get("kind") == "CONTINUE":
            next_plan = decision.get("next_plan") or {}
            return NextAttemptPlan(
                lineage=str(next_plan.get("lineage") or "fresh"),
                seed_attempt_n=next_plan.get("seed_attempt_n"),
                seed_pointer=next_plan.get("seed_pointer"),
                scope_rung=int(next_plan.get("scope_rung") or 0),
                width=int(next_plan.get("width") or 1),
            )
    return NextAttemptPlan(lineage="fresh", seed_attempt_n=None, seed_pointer=None, scope_rung=0, width=1)


def _attempt_context(
    run_dir: Path, opts: CampaignOptions, attempt_n: int, assessments: Sequence[AttemptAssessment]
) -> tuple[dict[str, str], str | None, list[str], list[str], list[str], list[str], list[str]]:
    """Everything ``synthesize_directives`` needs beyond plan/envelope.
    Every artifact ref is included ONLY when the file exists at plan time
    (the just-assessed attempt's artifacts are still at run_dir, pre-quarantine)."""
    prior_run_artifacts: dict[str, str] = {}
    for key, relpath in (
        ("leaf_triage", "rlm_state/leaf_triage.json"),
        ("failure_capsules", "rlm_state/failure_capsules.jsonl"),
        ("metrics", "code/metrics.json"),
        ("prior_report", "final_report.json"),
    ):
        candidate = run_dir / relpath
        if candidate.exists():
            prior_run_artifacts[key] = str(candidate)

    understanding_path = run_dir / "campaign" / "understanding.json"
    understanding_ref = str(understanding_path) if understanding_path.exists() else None

    # spec §6.3: still-unresolved UNDERSTAND fields ride into attempt-1
    # directives only -- UNDERSTAND runs once, not per attempt.
    unresolved_warnings: list[str] = []
    if attempt_n == 1 and understanding_path.exists():
        payload = _read_json_dict(understanding_path).get("payload")
        if isinstance(payload, dict):
            unresolved_warnings = [str(x) for x in (payload.get("unresolved") or [])]

    raw_improvements = _read_json_dict(run_dir / "final_report.json").get("improvements")
    improvement_notes = [str(x) for x in raw_improvements][:8] if isinstance(raw_improvements, list) else []

    suggested_fixes: list[str] = []
    if opts.arxiv_id:
        for lesson in lesson_distiller.active_lessons(opts.runs_root, opts.arxiv_id):
            fix = lesson.get("suggested_fix") if isinstance(lesson, dict) else None
            if fix:
                suggested_fixes.append(str(fix))
    infra_hints = ExperienceMemory(opts.runs_root).infra_hints()
    memory_hints = (suggested_fixes + infra_hints)[:5]
    # infra_hints() returns rendered strings, not raw store keys -- that
    # rendered text doubles as the "signature" here (audit-only field, never
    # a novelty-fingerprint input).
    injected_lesson_signatures = [h for h in infra_hints if h in memory_hints]

    # Self-edit guidance seam (Unit 11, flag-gated): a promoted
    # "guidance:campaign_preamble_suffix" override rides the tail of the
    # EXISTING capped hint channel -- campaign_directives.py is untouched.
    if self_edit_enabled():
        suffix = active_overrides(opts.runs_root).get("guidance:campaign_preamble_suffix")
        if isinstance(suffix, str) and suffix:
            memory_hints = [*memory_hints, suffix]

    failure_classes: list[str] = []
    if assessments:
        latest = max(assessments, key=lambda a: a.attempt_n)
        if latest.failure_class:
            failure_classes = [latest.failure_class]

    return (
        prior_run_artifacts,
        understanding_ref,
        unresolved_warnings,
        improvement_notes,
        memory_hints,
        injected_lesson_signatures,
        failure_classes,
    )


def _synthesize_with_novelty(
    *,
    run_dir: Path,
    opts: CampaignOptions,
    project_id: str,
    attempt_n: int,
    base_plan: NextAttemptPlan,
    envelope: AttemptEnvelope,
    enforcement_mapping: Mapping[str, Any],
    assessments: Sequence[AttemptAssessment],
    lineage_by_attempt: Mapping[int, str],
    runs_dir_hint: Mapping[int, str],
    prior_fingerprints: set[Any],
) -> tuple[AttemptDirectives | None, dict[str, Any] | None]:
    """Synthesize directives for ``base_plan``; on a novelty collision (F10)
    walk ``lineage_arms``' remaining arms -- scope_rung/width stay pinned
    (orthogonal to lineage) -- until one is novel or all arms are exhausted."""
    (
        prior_run_artifacts,
        understanding_ref,
        unresolved_warnings,
        improvement_notes,
        memory_hints,
        injected_lesson_signatures,
        failure_classes,
    ) = _attempt_context(run_dir, opts, attempt_n, assessments)
    target_floor = campaign_floor(assessments)

    def _make(plan: NextAttemptPlan) -> AttemptDirectives:
        return synthesize_directives(
            attempt_n=attempt_n,
            project_id=project_id,
            paper_ref=opts.paper_ref,
            plan=plan,
            envelope=envelope,
            enforcement=enforcement_mapping,
            run_spec_path=opts.run_spec_path,
            understanding_ref=understanding_ref,
            unresolved_warnings=unresolved_warnings,
            prior_run_artifacts=prior_run_artifacts,
            improvement_notes=improvement_notes,
            memory_hints=memory_hints,
            injected_lesson_signatures=injected_lesson_signatures,
            failure_classes=failure_classes,
            scope_spec=opts.scope_spec,
            target_floor=target_floor,
            out_dir=run_dir / "campaign",
            root_model=opts.root_model,
            execution_mode=opts.execution_mode,
            gpu_mode=opts.gpu_mode,
            minimize_compute=opts.minimize_compute,
        )

    directives = _make(base_plan)
    if directives.fingerprint not in prior_fingerprints:
        return directives, None

    arms = lineage_arms(assessments, runs_dir_hint=runs_dir_hint, lineage_by_attempt=lineage_by_attempt)
    for arm in arms[1:]:
        candidate_plan = NextAttemptPlan(
            lineage=arm.lineage,
            seed_attempt_n=arm.seed_attempt_n,
            seed_pointer=arm.seed_pointer,
            scope_rung=base_plan.scope_rung,
            width=base_plan.width,
        )
        directives = _make(candidate_plan)
        if directives.fingerprint not in prior_fingerprints:
            return directives, None

    return None, {"refusal": "no_novel_plan", "downgrade_to_checkpoint": False}


def _plan_attempt_impl(
    run_dir: Path, opts: CampaignOptions, project_id: str, state: CampaignState, rows: list
) -> dict[str, Any]:
    assessments, lineage_by_attempt, _scope_rung_by_attempt, runs_dir_hint = _gather_campaign_view(run_dir, rows)

    budget = CampaignBudget(**state.budget)
    spent = CampaignSpend.from_mapping(state.spent)
    try:
        envelope = derive_envelope(budget, spent, attempts_completed=len(assessments))
    except EnvelopeExhausted as exc:
        return {"refusal": f"budget_floor:{exc.meter}", "downgrade_to_checkpoint": False}

    ctx, _rate = _enforcement_ctx(run_dir, opts, state.mode)
    try:
        enforcement_plan = check_enforceability(envelope, ctx)
    except EnforceabilityError as exc:
        return {"refusal": "unenforceable:" + ";".join(exc.reasons), "downgrade_to_checkpoint": True}

    attempt_n = state.next_attempt_n
    base_plan = _base_next_plan(rows, attempt_n)
    # The §8.4 novelty domain is "any PRIOR attempt's plan" -- never this same
    # attempt's own write-ahead residue. Without the attempt_n exclusion, the
    # spec-§5-sanctioned orphaned-intent resume (crash between the intent
    # append and the in_flight write -> _resume_orphaned_intent ->
    # _loop(allow_supersede=True) -> plan_attempt for the SAME attempt_n)
    # resynthesizes identical directives, collides with its own just-written
    # intent row, and refuses "no_novel_plan" instead of legitimately
    # re-arming the same never-launched attempt.
    prior_fingerprints = {
        row.get("directives_sha256")
        for row in rows
        if isinstance(row, dict)
        and row.get("status") == "launched"
        and row.get("directives_sha256")
        and row.get("attempt_n") != attempt_n
    }

    # Budget-gated width (spec §8.3, F16): a width>1 next_plan mints THIS
    # attempt a distinct child project id `<project_id>_w<k>` (k counts up
    # across consecutive width attempts) instead of launching under the
    # campaign's own top-level id. v1 is deliberately simple (sequential,
    # not true parallel fan-out): the campaign's OWN bookkeeping (run_dir,
    # campaign/, the ledger) always stays anchored to the top-level
    # project_id -- only the LAUNCHED attempt's project_id/run_dir move.
    launch_project_id = project_id
    width_k: int | None = None
    if base_plan.width > 1:
        width_k = _count_prior_width_launches(project_id, rows, attempt_n=attempt_n) + 1
        launch_project_id = mint_width_project_ids(project_id, width_k)[-1]

    enforcement_mapping = _enforcement_mapping(
        enforcement_plan, paper_hint=opts.paper_hint, sandbox=opts.sandbox
    )
    # Self-edit env seam (Unit 11, flag-gated): promoted numeric knob
    # overrides ride the child env alongside the enforceability plan's own.
    # active_overrides() may also carry the diagnostic "_dropped" key (names
    # of out-of-bounds overrides dropped on read, see its docstring) and
    # "guidance:"-prefixed entries (consumed separately by _attempt_context,
    # above) -- neither is a real env var, so both are excluded here: only
    # keys the run-spec contract itself would actually apply survive the
    # merge into the child env.
    if self_edit_enabled():
        enforcement_mapping["env"].update(
            {
                k: str(v)
                for k, v in active_overrides(opts.runs_root).items()
                if run_spec_key_applies(k) and not k.startswith("guidance:")
            }
        )

    directives, refusal = _synthesize_with_novelty(
        run_dir=run_dir,
        opts=opts,
        project_id=launch_project_id,
        attempt_n=attempt_n,
        base_plan=base_plan,
        envelope=envelope,
        enforcement_mapping=enforcement_mapping,
        assessments=assessments,
        lineage_by_attempt=lineage_by_attempt,
        runs_dir_hint=runs_dir_hint,
        prior_fingerprints=prior_fingerprints,
    )
    if refusal is not None:
        return refusal
    assert directives is not None  # invariant: refusal is None iff directives is not None

    if width_k is not None:
        angle = f"[width] candidate angle {width_k}/{base_plan.width}: vary the implementation approach."
        guidance = f"{directives.extra_guidance}\n{angle}" if directives.extra_guidance else angle
        directives = replace(directives, extra_guidance=guidance)
        directives.persist(run_dir / "campaign" / "directives" / f"{attempt_n}.json")

    launch_run_dir = opts.runs_root / launch_project_id

    return {
        "attempt_n": attempt_n,
        "directives_sha256": directives.fingerprint,
        "envelope": envelope.to_dict(),
        "project_id": launch_project_id,
        "run_dir": str(launch_run_dir),
        "refusal": None,
        "downgrade_to_checkpoint": False,
        "launch_payload": directives,
    }


# --- launch / await_result ---------------------------------------------------


def _launch_impl(driver: AttemptDriver, directives: Any) -> dict[str, Any]:
    return asdict(driver.launch(directives))


def _await_result_impl(
    driver: AttemptDriver, opts: CampaignOptions, run_dir: Path, project_id: str, handle_dict: Mapping[str, Any]
) -> dict[str, Any]:
    # Tolerant reconstructor: the resume path's minimal {pid, run_dir,
    # lease_ref} dict carries no attempt_n/project_id/driver/launched_at.
    # LiveCliDriver.await_result only reads .run_dir/.pid/.await_deadline, so
    # the rest are safe closure-captured defaults, never consulted.
    handle = AttemptHandle(
        attempt_n=int(handle_dict.get("attempt_n") or 0),
        project_id=str(handle_dict.get("project_id") or project_id),
        run_dir=str(handle_dict.get("run_dir") or run_dir),
        driver=str(handle_dict.get("driver") or opts.driver),
        pid=handle_dict.get("pid"),
        launched_at=float(handle_dict.get("launched_at") or 0.0),
        lease_ref=handle_dict.get("lease_ref"),
        await_deadline=handle_dict.get("await_deadline"),
    )
    return asdict(driver.await_result(handle))


# --- assess / assess_from_disk ------------------------------------------------


def _current_rubric_pin(campaign_dir: Path) -> str | None:
    loaded = CampaignLedger(campaign_dir).load_state()
    return loaded.rubric_sha256 if loaded is not None else None


def _run_assessment(
    opts: CampaignOptions,
    attempt_run_dir: Path,
    attempt_n: int,
    project_id: str,
    directives_sha256: str,
    pinned: str | None,
) -> dict[str, Any]:
    assessment = assess_attempt(
        attempt_run_dir,
        attempt_n=attempt_n,
        driver=opts.driver,
        project_id=project_id,
        directives_sha256=directives_sha256,
        pinned_rubric_sha256=pinned,
        arxiv_id=opts.arxiv_id,
    )
    result = assessment.to_dict()
    # Pin-at-first-sight: computed here (never persisted from inside a
    # stage, which would race the state machine's own writes) -- the MACHINE
    # adopts it via the "rubric_sha256_observed" key.
    if pinned is None:
        observed = _rubric_sha256_if_exists(attempt_run_dir / "generated_rubric.json")
        if observed is not None:
            result["rubric_sha256_observed"] = observed
    return result


def _assess_impl(
    opts: CampaignOptions, project_id: str, campaign_dir: Path, raw: Mapping[str, Any], planned: Mapping[str, Any]
) -> dict[str, Any]:
    return _run_assessment(
        opts,
        Path(raw["run_dir"]),
        int(planned["attempt_n"]),
        str(planned.get("project_id") or project_id),
        str(planned["directives_sha256"]),
        _current_rubric_pin(campaign_dir),
    )


def _assess_from_disk_impl(
    opts: CampaignOptions, project_id: str, campaign_dir: Path, in_flight: InFlight
) -> dict[str, Any]:
    rows = CampaignLedger(campaign_dir).read_rows()
    launched_row = CampaignLedger.latest_by_status(rows, in_flight.attempt_n).get("launched") or {}
    return _run_assessment(
        opts,
        Path(in_flight.run_dir),
        in_flight.attempt_n,
        project_id,
        str(launched_row.get("directives_sha256") or ""),
        _current_rubric_pin(campaign_dir),
    )


def _quarantine_impl(opts: CampaignOptions, project_id: str, in_flight: InFlight) -> None:
    # force_archive_incomplete resolves run_dir from (project_id, runs_root).
    # The campaign's own top-level project_id is right for MOST callers, but
    # on the orphaned-intent / dead-in-flight resume paths the dead attempt
    # may be a WIDTH CHILD (in_flight.run_dir = runs_root/<project_id>_w<k>,
    # spec §8.3) -- quarantining under the top-level id would leave the
    # child's residue untouched. Derive the target from in_flight.run_dir's
    # basename ONLY when it resolves to a direct child of opts.runs_root;
    # otherwise (empty run_dir, or a run_dir outside runs_root) fail back to
    # the campaign project_id -- never quarantine outside runs_root.
    target_project_id = project_id
    if in_flight.run_dir:
        resolved_run_dir = Path(in_flight.run_dir).resolve()
        if resolved_run_dir.parent == opts.runs_root.resolve():
            target_project_id = resolved_run_dir.name
    archived = force_archive_incomplete(
        target_project_id, opts.runs_root, reason="campaign_resume_quarantine"
    )
    # Keep the attempt-code index truthful about the code this just MOVED.
    # The residue belongs to whichever attempt the index still calls "live"
    # -- NOT to in_flight.attempt_n, which on the orphaned-intent path is the
    # attempt whose launch never fired (its predecessor owns the residue).
    # new_live_attempt_n=None: nothing is live in that dir once it is
    # quarantined. Skipping this would strand the quarantined attempt's code
    # in an unindexed archive and make it unseedable.
    rotate_attempt_code_index(
        opts.runs_root / target_project_id, archived=archived, new_live_attempt_n=None
    )


# --- distill -------------------------------------------------------------------


def _distill_impl(run_dir: Path, opts: CampaignOptions, _assessment: Mapping[str, Any]) -> None:
    # Each miner independently guarded (spec §11.1): one raising must never
    # suppress the others. Admission gates stay owned by the miners themselves.
    try:
        lesson_distiller.mine_lessons(run_dir, opts.runs_root, opts.arxiv_id)
    except Exception:  # noqa: BLE001 -- fail-soft DISTILL call
        logger.warning("campaign_composition: mine_lessons failed", exc_info=True)

    try:
        admit_recipe(
            run_dir,
            opts.runs_root,
            report=_read_json_dict(run_dir / "final_report.json"),
            validator_verdict=external_validator.load_verdict(run_dir),
            paper_class=opts.paper_class,
        )
    except Exception:  # noqa: BLE001 -- fail-soft DISTILL call
        logger.warning("campaign_composition: admit_recipe failed", exc_info=True)

    try:
        exp_rows = _read_jsonl(run_dir / "experiment_runs.jsonl")
        if exp_rows:
            attribution = failure_attribution.attribute_failure(exp_rows[-1], arxiv_id=opts.arxiv_id)
            ExperienceMemory(opts.runs_root).record(attribution, arxiv_id=opts.arxiv_id)
    except Exception:  # noqa: BLE001 -- fail-soft DISTILL call
        logger.warning("campaign_composition: experience_memory record failed", exc_info=True)


# --- decide --------------------------------------------------------------------


def _maybe_attach_asha_advisory(
    result: dict[str, Any],
    assessments: list,
    current_rung: int,
    *,
    max_gpu_usd: float | None = None,
    gpu_usd_spent: float = 0.0,
    max_gpu_count: int | None = None,
) -> None:
    """SHADOW-MODE ASHA scheduler (``OPENRESEARCH_SCHEDULER_TREE``, default OFF).

    When on, attach the ASHA promote/freeze/kill advisory over the current
    assessments to ``result["asha_advisory"]`` — additive metadata that changes NO
    campaign decision, so an operator can compare the scheduler's advice against the
    live deterministic decision before the tree is ever made authoritative. Off ⇒
    no-op, byte-identical. Fail-soft: any error leaves ``result`` untouched (the
    advisory must NEVER perturb the fail-closed decide path).

    ``current_rung`` is the FIDELITY meter (optimizer-step ladder). The optional
    ``max_gpu_usd`` / ``gpu_usd_spent`` / ``max_gpu_count`` feed the *independent*
    WIDTH meter (GPU-$ budget + hard A100 cap) — the two are never merged. Absent ⇒
    the width meter degrades to the geometric ``eta`` fallback (prior behaviour), so
    every existing OFF/positional call stays byte-identical. Per-attempt GPU-$ comes
    from each deterministic ``AttemptAssessment.cost.gpu_usd``; it is used solely
    for the advisory's width meter, never a live decision. All the width arithmetic
    runs *after* the env-gate short-circuit, so OFF adds zero work to the fail-closed
    path."""
    import os

    if os.environ.get("OPENRESEARCH_SCHEDULER_TREE", "").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        return
    try:
        from backend.agents.rlm.asha_campaign_adapter import (
            asha_decide_for_assessments,
        )
        from backend.agents.rlm.asha_scheduler import RungConfig

        # WIDTH meter (independent of the fidelity rung): remaining GPU-$ budget +
        # hard A100 cap. Only built when a $ ceiling is supplied; else None ⇒ the
        # core falls back to the geometric eta width (unchanged legacy behaviour).
        gpu_usd_budget: float | None = None
        gpu_usd_by_attempt: dict[int, float] | None = None
        if max_gpu_usd is not None:
            gpu_usd_budget = max(0.0, float(max_gpu_usd) - float(gpu_usd_spent))
            gpu_usd_by_attempt = {}
            for assessment in assessments:
                try:
                    attempt_n = getattr(assessment, "attempt_n", None)
                    cost = getattr(assessment, "cost", None)
                    gpu_usd = getattr(cost, "gpu_usd", None)
                except Exception:  # noqa: BLE001 -- a bad cost must not drop the advisory
                    continue
                if (
                    not isinstance(attempt_n, int)
                    or isinstance(attempt_n, bool)
                    or not isinstance(gpu_usd, (int, float))
                    or isinstance(gpu_usd, bool)
                ):
                    continue
                observed_gpu_usd = float(gpu_usd)
                if math.isfinite(observed_gpu_usd) and observed_gpu_usd >= 0.0:
                    gpu_usd_by_attempt[attempt_n] = observed_gpu_usd

        decisions = asha_decide_for_assessments(
            assessments,
            RungConfig(
                rung=int(current_rung),
                higher_is_better=True,
                noise_floor=0.0067,
                gpu_usd_budget=gpu_usd_budget,
                a100_cap=max_gpu_count,
            ),
            gpu_usd_by_attempt=gpu_usd_by_attempt,
        )
        result["asha_advisory"] = {
            "rung": int(current_rung),
            "width_meter": {
                "gpu_usd_budget": gpu_usd_budget,
                "a100_cap": max_gpu_count,
                "gpu_usd_spent": (
                    float(gpu_usd_spent) if max_gpu_usd is not None else None
                ),
            },
            "decisions": [
                {"branch_id": d.branch_id, "action": d.action, "reason": d.reason}
                for d in decisions
            ],
        }
    except Exception:  # noqa: BLE001 — advisory must never break the fail-closed decide
        pass


def _decide_impl(run_dir: Path, opts: CampaignOptions, state: CampaignState, rows: list) -> dict[str, Any]:
    assessments, lineage_by_attempt, scope_rung_by_attempt, runs_dir_hint = _gather_campaign_view(run_dir, rows)
    budget = CampaignBudget(**state.budget)
    spent = CampaignSpend.from_mapping(state.spent)
    config = PolicyConfig(
        max_attempts=budget.max_attempts,
        plateau_k=opts.plateau_k,
        width=opts.width,
        ladder_len=len(opts.scope_ladder) or 1,
    )
    ctx, rate = _enforcement_ctx(run_dir, opts, state.mode)
    # ``rate`` is the PER-GPU $/hr (EnforcementContext.gpu_usd_per_hr — see
    # check_enforceability's own ``ctx.gpu_usd_per_hr * ctx.max_gpu_count``
    # and attempt_estimate's provisioning-overhead term, which both scale the
    # per-GPU rate by the GPU count). Omitting that factor here undercounted
    # the DECIDE budget-floor lookahead on every multi-GPU gpu_plan -- reuse
    # ctx.max_gpu_count (already resolved by _enforcement_ctx) rather than
    # re-deriving it.
    next_estimate = attempt_estimate(
        est_gpu_hours=opts.est_gpu_hours,
        est_usd=opts.est_gpu_hours * (rate or 0.0) * ctx.max_gpu_count,
        ctx=ctx,
    )
    decision = campaign_decide(
        assessments,
        budget=budget,
        spent=spent,
        config=config,
        next_estimate=next_estimate,
        lineage_by_attempt=lineage_by_attempt,
        scope_rung_by_attempt=scope_rung_by_attempt,
        runs_dir_hint=runs_dir_hint,
        current_rung=state.scope_rung,
        blocking_gap=None,  # no live probe-confirmed-gap source wired in v1
    )
    result = decision.to_dict()
    # Shadow advisory: fidelity meter = state.scope_rung; width meter = remaining
    # GPU-$ (opts.max_gpu_usd − spent.gpu_usd) + hard A100 cap (ctx.max_gpu_count,
    # already resolved by _enforcement_ctx). OFF ⇒ byte-identical (helper returns
    # before touching any of these).
    _maybe_attach_asha_advisory(
        result,
        assessments,
        state.scope_rung,
        max_gpu_usd=opts.max_gpu_usd,
        gpu_usd_spent=spent.gpu_usd,
        max_gpu_count=ctx.max_gpu_count,
    )
    return result


# --- write_reports ---------------------------------------------------------------


def _write_reports_impl(
    run_dir: Path, state: CampaignState, rows: list, decision: Mapping[str, Any], runs_root: Path | None = None
) -> None:
    write_campaign_report(run_dir, state=state.to_dict(), rows=rows)

    assessed_ns = sorted(
        {
            row.get("attempt_n")
            for row in rows
            if isinstance(row, dict) and row.get("status") == "assessed" and isinstance(row.get("attempt_n"), int)
        }
    )
    # F14: INFEASIBLE (or no attempt ever assessed, e.g. an INIT/plan-refusal
    # early exit) never terminates report-less.
    if decision.get("kind") == "INFEASIBLE" or not assessed_ns:
        stop_reason = decision.get("stop_reason")
        write_plan_only_report(
            run_dir,
            stop_reason=str(stop_reason or ""),
            what_would_unblock=[str(stop_reason)] if stop_reason else [],
            state=state.to_dict(),
        )

    champion_n = decision.get("champion_attempt_n")
    if champion_n is not None and assessed_ns and champion_n == assessed_ns[-1]:
        # Only the LATEST assessed attempt's report is still live at
        # run_dir top-level (an older champion's is already archived).
        campaign_dir = run_dir / "campaign"
        campaign_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("json", "md"):
            src = run_dir / f"final_report.{ext}"
            if src.exists():
                shutil.copyfile(src, campaign_dir / f"champion_final_report.{ext}")

    # Self-edit replay-harvest seam (Unit 11, flag-gated): an operator who
    # never opts in gets zero new stores under runs/_memory/ (runs_root=None
    # on every call this module makes outside build_campaign, and self-edit
    # off, are BOTH required -- either alone keeps this a no-op).
    if runs_root is not None and self_edit_enabled():
        try:
            harvest_replay_cases(run_dir, runs_root=runs_root, state=state.to_dict(), rows=rows)
        except Exception:  # noqa: BLE001 -- fail-soft harvest call
            logger.warning("campaign_composition: harvest_replay_cases failed", exc_info=True)


# --- emit_event / poll_steering ---------------------------------------------------


def _emit_event_impl(run_dir: Path, event_type: str, payload: Mapping[str, Any]) -> None:
    entry = {"event": event_type, "timestamp": _now_iso(), **dict(payload)}
    path = run_dir / "dashboard_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same direct-append pattern as backend/routes/messages.py.
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def _poll_steering_impl(run_dir: Path, state: CampaignState) -> None:
    path = run_dir / "campaign" / "user_messages.jsonl"
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    new_lines = lines[state.steering_cursor :]
    for line in new_lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("op") == "set_mode" and row.get("mode") in ("checkpoint", "unattended"):
            state.mode = row["mode"]
    # Malformed / non-actionable rows still advance the cursor.
    state.steering_cursor += len(new_lines)


# --- build_campaign ----------------------------------------------------------------


def _build_driver(opts: CampaignOptions) -> AttemptDriver:
    """Driver selection (spec §7). "paired" passes ``operator_ack=True``
    ONLY because it is reached exclusively via an explicit
    ``--campaign-driver paired`` CLI selection (or its env-default
    equivalent) -- there is no other construction path, so the ack is
    genuinely tied to the operator's own explicit choice (Codex F8)."""
    live = LiveCliDriver(runs_root=opts.runs_root, repo_root=opts.repo_root)
    if opts.driver == "live":
        return live
    if opts.driver == "unified":
        return UnifiedRunDriver(runs_root=opts.runs_root)
    if opts.driver == "paired":
        return PairedDriver(live=live, unified=UnifiedRunDriver(runs_root=opts.runs_root), operator_ack=True)
    raise ValueError(f"unknown campaign driver {opts.driver!r}")


def _apply_profile_env(opts: CampaignOptions, *, env: MutableMapping[str, str] | None = None) -> int:
    """Fill the CAMPAIGN process's own env from the run-spec profile.

    The launched child already receives the profile via ``--run-spec``
    (``attempt_driver``/``cli.py::_load_run_spec``), but the campaign-SIDE
    readers -- DISTILL's ``lesson_distiller``/``ExperienceMemory``, any
    ``OPENRESEARCH_*``-gated helper called directly in this process -- only
    ever saw whatever an operator hand-exported. This applies the SAME keys
    (via the canonical ``run_spec_contract.apply_run_spec``) into the
    campaign process's environment so both sides agree without duplicate
    operator setup.

    Explicit operator env wins: a key already present in ``env`` is left
    untouched -- this only fills gaps, never overrides (F15's INIT-time
    ``validate_run_spec_keys``/driver-owned-key rejection is unaffected,
    since it validates the PROFILE's own keys, not the process env).

    Fail-soft: an unreadable/malformed profile is a no-op here -- the
    user-facing error is raised by ``_validate_init_impl`` at the INIT
    stage; this convenience helper must never crash campaign construction.

    Returns the number of keys actually applied (observability/tests).
    """
    if env is None:
        env = os.environ
    try:
        profile = json.loads(Path(opts.run_spec_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(profile, dict):
        return 0
    staged: dict[str, str] = {}
    try:
        apply_run_spec(profile, staged)
    except Exception:  # noqa: BLE001 -- never block campaign construction
        return 0
    applied = 0
    for key, val in staged.items():
        if key in env:
            continue  # explicit operator env wins
        env[key] = val
        applied += 1
    return applied


def build_campaign(project_id: str, opts: CampaignOptions) -> ReproductionCampaign:
    _apply_profile_env(opts)
    run_dir = opts.runs_root / project_id
    campaign_dir = run_dir / "campaign"
    driver: AttemptDriver = _build_driver(opts)

    budget = {
        "max_llm_usd": opts.max_llm_usd,
        "max_gpu_usd": opts.max_gpu_usd,
        "max_gpu_hours": opts.max_gpu_hours,
        "max_attempts": opts.max_attempts,
        "max_wall_clock_s": opts.wall_clock_s,
    }

    stages = CampaignStages(
        validate_init=lambda: _validate_init_impl(opts),
        understand=lambda: _understand_impl(run_dir, opts),
        plan_attempt=lambda state, rows: _plan_attempt_impl(run_dir, opts, project_id, state, rows),
        launch=lambda payload: _launch_impl(driver, payload),
        await_result=lambda handle: _await_result_impl(driver, opts, run_dir, project_id, handle),
        assess=lambda raw, planned: _assess_impl(opts, project_id, campaign_dir, raw, planned),
        assess_from_disk=lambda in_flight: _assess_from_disk_impl(opts, project_id, campaign_dir, in_flight),
        quarantine=lambda in_flight: _quarantine_impl(opts, project_id, in_flight),
        distill=lambda assessment: _distill_impl(run_dir, opts, assessment),
        decide=lambda state, rows: _decide_impl(run_dir, opts, state, rows),
        write_reports=lambda state, rows, decision: _write_reports_impl(
            run_dir, state, rows, decision, opts.runs_root
        ),
        liveness_probe=default_liveness_probe,
        emit_event=lambda event_type, payload: _emit_event_impl(run_dir, event_type, payload),
        poll_steering=lambda state: _poll_steering_impl(run_dir, state),
    )

    return ReproductionCampaign(
        run_dir=run_dir,
        project_id=project_id,
        paper_ref=opts.paper_ref,
        budget=budget,
        mode=opts.mode,
        driver=opts.driver,
        stages=stages,
        resume=opts.resume,
    )
