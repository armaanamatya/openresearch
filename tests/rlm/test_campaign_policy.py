"""Tests for backend/agents/rlm/campaign_policy.py Part 1 — CampaignBudget /
CampaignSpend / AttemptEnvelope / enforceability mapping (Unit 2).

Pure-policy-math module: no I/O, no env reads, no backend imports — every
input is built directly in memory here. Covers spec §10.1 (money red lines)
+ §10.2 (headroom rule) and Codex resolutions F2 (``--max-usd`` is LLM-only
spend, never GPU) and F3 (``stage_on_gpu`` bills provisioning to the GPU
meters).
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.attempt_assessment import AttemptAssessment, ReportDigest, ValidatorStatus
from backend.agents.rlm.campaign_policy import (
    DEFAULT_ATTEMPT_WALL_S,
    ENVELOPE_FLOORS,
    FINALIZE_HEADROOM_S,
    AttemptEnvelope,
    CampaignBudget,
    CampaignSpend,
    Decision,
    EnforceabilityError,
    EnforcementContext,
    EnvelopeExhausted,
    NextAttemptPlan,
    PolicyConfig,
    attempt_estimate,
    campaign_floor,
    check_enforceability,
    decide,
    derive_envelope,
    directives_fingerprint,
    lineage_arms,
    next_scope_rung,
    seeding_pool,
    select_champion,
    width_for_next,
)

# The historical GCP VM control-plane ``max-run-duration`` default (28h) that
# the enforcement plan must never silently depend on (rule 5 / spec §10.2).
_GCP_28H_DEFAULT_S = 100800.0


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _budget(**overrides: object) -> CampaignBudget:
    defaults: dict[str, object] = dict(
        max_llm_usd=100.0, max_gpu_usd=50.0, max_gpu_hours=10.0, max_attempts=6
    )
    defaults.update(overrides)
    return CampaignBudget(**defaults)  # type: ignore[arg-type]


def _ctx(**overrides: object) -> EnforcementContext:
    defaults: dict[str, object] = dict(
        driver_kind="live",
        sandbox="local",
        mode="unattended",
        tiering_strategy=None,
        max_gpu_count=1,
        gpu_usd_per_hr=None,
        require_cpu_tier=False,
    )
    defaults.update(overrides)
    return EnforcementContext(**defaults)  # type: ignore[arg-type]


def _envelope(**overrides: object) -> AttemptEnvelope:
    defaults: dict[str, object] = dict(
        llm_usd=5.0,
        gpu_usd=10.0,
        gpu_hours=5.0,
        wall_s=DEFAULT_ATTEMPT_WALL_S,
        vm_ceiling_s=DEFAULT_ATTEMPT_WALL_S + FINALIZE_HEADROOM_S,
    )
    defaults.update(overrides)
    return AttemptEnvelope(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CampaignBudget validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["max_llm_usd", "max_gpu_usd", "max_gpu_hours"])
@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_budget_validates_positive_money_meters(field, bad_value):
    kwargs = dict(max_llm_usd=10.0, max_gpu_usd=10.0, max_gpu_hours=10.0, max_attempts=3)
    kwargs[field] = bad_value
    with pytest.raises(ValueError):
        CampaignBudget(**kwargs)


def test_budget_accepts_valid_meters():
    budget = _budget()
    assert budget.max_llm_usd == 100.0
    assert budget.max_gpu_usd == 50.0
    assert budget.max_gpu_hours == 10.0
    assert budget.max_attempts == 6
    assert budget.max_wall_clock_s is None


def test_budget_validates_max_attempts_at_least_one():
    with pytest.raises(ValueError):
        _budget(max_attempts=0)
    with pytest.raises(ValueError):
        _budget(max_attempts=-3)


def test_budget_validates_wall_clock_when_set():
    with pytest.raises(ValueError):
        _budget(max_wall_clock_s=0.0)
    with pytest.raises(ValueError):
        _budget(max_wall_clock_s=-100.0)
    with pytest.raises(ValueError):
        _budget(max_wall_clock_s=float("nan"))
    # unset is always fine
    _budget(max_wall_clock_s=None)


# ---------------------------------------------------------------------------
# CampaignSpend / AttemptEnvelope data-carrier behavior
# ---------------------------------------------------------------------------


def test_campaign_spend_add_and_dict_roundtrip():
    a = CampaignSpend(llm_usd=1.0, gpu_usd=2.0, gpu_hours=0.5, wall_s=100.0)
    b = CampaignSpend(llm_usd=0.5, gpu_usd=1.5, gpu_hours=0.25, wall_s=50.0)
    total = a.add(b)
    assert total == CampaignSpend(llm_usd=1.5, gpu_usd=3.5, gpu_hours=0.75, wall_s=150.0)
    roundtripped = CampaignSpend.from_mapping(total.to_dict())
    assert roundtripped == total


def test_campaign_spend_from_mapping_is_strict_on_missing_keys():
    with pytest.raises(KeyError):
        CampaignSpend.from_mapping({"llm_usd": 1.0, "gpu_usd": 0.0, "gpu_hours": 0.0})


def test_attempt_envelope_dict_roundtrip():
    envelope = _envelope()
    roundtripped = AttemptEnvelope.from_mapping(envelope.to_dict())
    assert roundtripped == envelope


def test_attempt_envelope_from_mapping_is_strict_on_missing_keys():
    with pytest.raises(KeyError):
        AttemptEnvelope.from_mapping({"llm_usd": 1.0, "gpu_usd": 1.0})


# ---------------------------------------------------------------------------
# derive_envelope
# ---------------------------------------------------------------------------


def test_derive_envelope_share_math():
    budget = _budget(
        max_llm_usd=100.0, max_gpu_usd=100.0, max_gpu_hours=100.0,
        max_attempts=6, max_wall_clock_s=36000.0,
    )
    spent = CampaignSpend(llm_usd=10.0, gpu_usd=10.0, gpu_hours=10.0, wall_s=0.0)
    envelope = derive_envelope(budget, spent, attempts_completed=1)

    # remaining=90 (money) / 36000 (wall); expected_remaining=max(1,6-1)=5
    # share = remaining/5 -- comfortably above each floor, so share wins.
    assert envelope.llm_usd == pytest.approx(90.0 / 5)
    assert envelope.gpu_usd == pytest.approx(90.0 / 5)
    assert envelope.gpu_hours == pytest.approx(90.0 / 5)
    assert envelope.wall_s == pytest.approx(36000.0 / 5)


def test_derive_envelope_share_uses_floor_when_share_below_floor():
    budget = _budget(max_llm_usd=100.0, max_gpu_usd=100.0, max_gpu_hours=100.0, max_attempts=50)
    spent = CampaignSpend(llm_usd=97.0, gpu_usd=97.0, gpu_hours=97.75, wall_s=0.0)
    envelope = derive_envelope(budget, spent, attempts_completed=1)

    # remaining=3.0 (llm/gpu_usd) or 2.25 (gpu_hours); expected_remaining=49
    # -> share = remaining/49 < floor for every meter, so the floor wins,
    # capped by min(remaining, floor).
    assert envelope.llm_usd == pytest.approx(min(3.0, ENVELOPE_FLOORS["llm_usd"]))
    assert envelope.gpu_usd == pytest.approx(min(3.0, ENVELOPE_FLOORS["gpu_usd"]))
    assert envelope.gpu_hours == pytest.approx(min(2.25, ENVELOPE_FLOORS["gpu_hours"]))


@pytest.mark.parametrize(
    "meter,budget_kwargs,spent_kwargs",
    [
        ("llm_usd", dict(max_llm_usd=10.0), dict(llm_usd=9.95)),
        ("gpu_usd", dict(max_gpu_usd=10.0), dict(gpu_usd=9.95)),
        ("gpu_hours", dict(max_gpu_hours=1.0), dict(gpu_hours=0.9)),
        ("wall_s", dict(max_wall_clock_s=2000.0), dict(wall_s=1000.0)),
    ],
)
def test_derive_envelope_raises_when_remaining_below_floor(meter, budget_kwargs, spent_kwargs):
    budget_defaults: dict[str, object] = dict(
        max_llm_usd=1000.0, max_gpu_usd=1000.0, max_gpu_hours=1000.0, max_attempts=6
    )
    budget_defaults.update(budget_kwargs)
    budget = CampaignBudget(**budget_defaults)  # type: ignore[arg-type]

    spend_defaults: dict[str, object] = dict(llm_usd=0.0, gpu_usd=0.0, gpu_hours=0.0, wall_s=0.0)
    spend_defaults.update(spent_kwargs)
    spent = CampaignSpend(**spend_defaults)  # type: ignore[arg-type]

    with pytest.raises(EnvelopeExhausted) as exc_info:
        derive_envelope(budget, spent, attempts_completed=0)
    assert exc_info.value.meter == meter
    assert exc_info.value.remaining < exc_info.value.floor


def test_envelope_never_exceeds_remaining():
    budget = _budget(
        max_llm_usd=1000.0, max_gpu_usd=1000.0, max_gpu_hours=1000.0,
        max_attempts=10, max_wall_clock_s=100000.0,
    )
    spent = CampaignSpend(llm_usd=250.0, gpu_usd=100.0, gpu_hours=50.0, wall_s=10000.0)

    for attempts_completed in range(0, budget.max_attempts):
        envelope = derive_envelope(budget, spent, attempts_completed=attempts_completed)
        assert envelope.llm_usd <= (budget.max_llm_usd - spent.llm_usd) + 1e-9
        assert envelope.gpu_usd <= (budget.max_gpu_usd - spent.gpu_usd) + 1e-9
        assert envelope.gpu_hours <= (budget.max_gpu_hours - spent.gpu_hours) + 1e-9
        assert envelope.wall_s <= (budget.max_wall_clock_s - spent.wall_s) + 1e-9

    # At the last attempt (expected_remaining == 1) share == remaining exactly.
    last = derive_envelope(budget, spent, attempts_completed=budget.max_attempts - 1)
    assert last.llm_usd == pytest.approx(budget.max_llm_usd - spent.llm_usd)


def test_default_attempt_wall_when_no_campaign_wall_meter():
    budget = _budget(max_wall_clock_s=None)
    spent = CampaignSpend()
    envelope = derive_envelope(budget, spent, attempts_completed=0)
    assert envelope.wall_s == DEFAULT_ATTEMPT_WALL_S

    # A late attempts_completed / partially-spent money meters must not
    # perturb the per-attempt-only wall default (stays comfortably above
    # every money floor so this isolates the wall behavior specifically).
    envelope2 = derive_envelope(
        budget, CampaignSpend(llm_usd=98.0, gpu_usd=48.0, gpu_hours=9.5), attempts_completed=5
    )
    assert envelope2.wall_s == DEFAULT_ATTEMPT_WALL_S


def test_vm_ceiling_is_wall_plus_headroom_never_28h_default():
    budget = _budget(max_wall_clock_s=None)
    spent = CampaignSpend()
    envelope = derive_envelope(budget, spent, attempts_completed=0)

    assert envelope.vm_ceiling_s == pytest.approx(envelope.wall_s + FINALIZE_HEADROOM_S)
    assert envelope.vm_ceiling_s != pytest.approx(_GCP_28H_DEFAULT_S)

    plan = check_enforceability(envelope, _ctx(sandbox="local"))
    assert plan.vm_ceiling_s == pytest.approx(plan.effective_wall_s + FINALIZE_HEADROOM_S)
    assert plan.vm_ceiling_s != pytest.approx(_GCP_28H_DEFAULT_S)


# ---------------------------------------------------------------------------
# check_enforceability
# ---------------------------------------------------------------------------


def test_max_usd_flag_carries_llm_meter_only():
    budget = _budget(max_llm_usd=120.0, max_gpu_usd=40.0, max_gpu_hours=8.0, max_attempts=1)
    spent = CampaignSpend()
    envelope = derive_envelope(budget, spent, attempts_completed=0)
    assert envelope.llm_usd != envelope.gpu_usd

    plan = check_enforceability(envelope, _ctx(sandbox="local"))
    flag_values = dict(plan.cli_args)
    assert float(flag_values["--max-usd"]) == pytest.approx(envelope.llm_usd)
    assert float(flag_values["--max-usd"]) != pytest.approx(envelope.gpu_usd)


def test_wall_cotightening_bounds_gpu_hours():
    envelope = _envelope(gpu_usd=1000.0, gpu_hours=2.0)
    ctx = _ctx(sandbox="local", tiering_strategy=None, max_gpu_count=2)
    plan = check_enforceability(envelope, ctx)

    hours_bound_s = envelope.gpu_hours / ctx.max_gpu_count * 3600.0
    assert plan.effective_wall_s == pytest.approx(hours_bound_s)
    assert plan.effective_wall_s < envelope.wall_s
    assert plan.effective_wall_s / 3600.0 * ctx.max_gpu_count <= envelope.gpu_hours + 1e-9


def test_wall_cotightening_bounds_gpu_usd_when_rate_known():
    envelope = _envelope(gpu_usd=4.0, gpu_hours=1000.0)
    ctx = _ctx(sandbox="gcp", tiering_strategy=None, max_gpu_count=2, gpu_usd_per_hr=2.0)
    plan = check_enforceability(envelope, ctx)

    usd_bound_s = envelope.gpu_usd / (ctx.gpu_usd_per_hr * ctx.max_gpu_count) * 3600.0
    assert plan.effective_wall_s == pytest.approx(usd_bound_s)
    assert plan.effective_wall_s < envelope.wall_s
    assert plan.effective_wall_s / 3600.0 * ctx.max_gpu_count * ctx.gpu_usd_per_hr <= envelope.gpu_usd + 1e-9


def test_stage_on_gpu_provision_subtracted_from_bounds_and_charged():
    envelope = _envelope(gpu_usd=10.0, gpu_hours=5.0)
    base = dict(sandbox="gcp", max_gpu_count=1, gpu_usd_per_hr=2.0, mode="checkpoint", require_cpu_tier=False)

    plan_without = check_enforceability(envelope, _ctx(tiering_strategy=None, **base))
    plan_with = check_enforceability(envelope, _ctx(tiering_strategy="stage_on_gpu", **base))

    assert plan_with.effective_wall_s == pytest.approx(plan_without.effective_wall_s - 900.0)
    assert plan_with.provision_charged_to_gpu is True
    assert "stage_on_gpu:provision_billed_to_gpu" in plan_with.notes
    assert plan_without.provision_charged_to_gpu is False
    assert "stage_on_gpu:provision_billed_to_gpu" not in plan_without.notes


def test_wall_clock_starved_raises():
    envelope = _envelope(gpu_usd=1000.0, gpu_hours=0.05)
    ctx = _ctx(sandbox="local", tiering_strategy=None, max_gpu_count=1)

    with pytest.raises(EnforceabilityError) as exc_info:
        check_enforceability(envelope, ctx)
    assert "wall_clock_starved" in exc_info.value.reasons


def test_cloud_without_rate_is_unenforceable_gpu_usd():
    envelope = _envelope(gpu_usd=10.0, gpu_hours=5.0)
    ctx = _ctx(sandbox="gcp", tiering_strategy=None, max_gpu_count=1, gpu_usd_per_hr=None)

    with pytest.raises(EnforceabilityError) as exc_info:
        check_enforceability(envelope, ctx)
    assert "gpu_usd_unenforceable:no_rate" in exc_info.value.reasons


def test_local_sandbox_gpu_usd_vacuous_note():
    envelope = _envelope(gpu_usd=10.0, gpu_hours=5.0)
    ctx = _ctx(sandbox="local", tiering_strategy=None, max_gpu_count=1, gpu_usd_per_hr=None)

    plan = check_enforceability(envelope, ctx)
    assert "gpu_usd_vacuous_local" in plan.notes
    assert all(name != "--max-run-gpu-usd" for name, _ in plan.cli_args)
    assert "OPENRESEARCH_MAX_RUN_GPU_USD" not in plan.env


def test_require_cpu_tier_refuses_stage_on_gpu_unattended_only():
    envelope = _envelope(gpu_usd=10.0, gpu_hours=5.0)
    unattended_ctx = _ctx(
        sandbox="gcp", tiering_strategy="stage_on_gpu", max_gpu_count=1,
        gpu_usd_per_hr=2.0, require_cpu_tier=True, mode="unattended",
    )
    with pytest.raises(EnforceabilityError) as exc_info:
        check_enforceability(envelope, unattended_ctx)
    assert "cpu_tier_required:stage_on_gpu" in exc_info.value.reasons

    checkpoint_ctx = _ctx(
        sandbox="gcp", tiering_strategy="stage_on_gpu", max_gpu_count=1,
        gpu_usd_per_hr=2.0, require_cpu_tier=True, mode="checkpoint",
    )
    plan = check_enforceability(envelope, checkpoint_ctx)
    assert plan.provision_charged_to_gpu is True


def test_require_cpu_tier_allows_satisfying_tiering_strategies_unattended():
    envelope = _envelope(gpu_usd=10.0, gpu_hours=5.0)
    for strategy in ("machine_type_flip", "cpu_warm_disk_then_gpu_attach"):
        ctx = _ctx(
            sandbox="gcp", tiering_strategy=strategy, max_gpu_count=1,
            gpu_usd_per_hr=2.0, require_cpu_tier=True, mode="unattended",
        )
        plan = check_enforceability(envelope, ctx)
        assert plan.provision_charged_to_gpu is False


def test_enforceability_error_aggregates_all_reasons():
    envelope = _envelope(gpu_usd=10.0, gpu_hours=5.0)
    ctx = _ctx(
        sandbox="gcp", tiering_strategy="stage_on_gpu", max_gpu_count=1,
        gpu_usd_per_hr=None, require_cpu_tier=True, mode="unattended",
    )
    with pytest.raises(EnforceabilityError) as exc_info:
        check_enforceability(envelope, ctx)
    reasons = set(exc_info.value.reasons)
    assert reasons == {"gpu_usd_unenforceable:no_rate", "cpu_tier_required:stage_on_gpu"}


def test_gpu_hours_unenforceable_when_gpu_count_unknown():
    envelope = _envelope(gpu_usd=10.0, gpu_hours=5.0)
    ctx = _ctx(sandbox="local", tiering_strategy=None, max_gpu_count=0)
    with pytest.raises(EnforceabilityError) as exc_info:
        check_enforceability(envelope, ctx)
    assert "gpu_hours_unenforceable:unknown_gpu_count" in exc_info.value.reasons


def test_enforcement_plan_cli_args_deterministic_order():
    envelope = AttemptEnvelope(
        llm_usd=12.5, gpu_usd=20.0, gpu_hours=5.0, wall_s=14400.0, vm_ceiling_s=16200.0,
    )
    ctx = _ctx(sandbox="gcp", tiering_strategy=None, max_gpu_count=1, gpu_usd_per_hr=5.0)

    plan_a = check_enforceability(envelope, ctx)
    plan_b = check_enforceability(envelope, ctx)

    names = [name for name, _ in plan_a.cli_args]
    assert names == ["--max-usd", "--max-wall-clock", "--max-run-gpu-usd"]
    assert plan_a.cli_args == plan_b.cli_args  # deterministic across repeated calls
    assert dict(plan_a.cli_args)["--max-usd"] == "12.5"
    assert dict(plan_a.cli_args)["--max-run-gpu-usd"] == "20"
    assert plan_a.env == {"OPENRESEARCH_MAX_RUN_GPU_USD": "20"}


def test_runpod_sandbox_appends_max_pod_seconds_after_gpu_usd():
    # The runpod control-plane ceiling analog (RunBudget.max_pod_seconds)
    # must be made explicit, deterministically ordered after --max-run-gpu-usd.
    envelope = _envelope(gpu_usd=10.0, gpu_hours=5.0)
    runpod_ctx = _ctx(sandbox="runpod", tiering_strategy=None, max_gpu_count=1, gpu_usd_per_hr=2.0)
    plan = check_enforceability(envelope, runpod_ctx)

    names = [name for name, _ in plan.cli_args]
    assert names == ["--max-usd", "--max-wall-clock", "--max-run-gpu-usd", "--max-pod-seconds"]
    assert dict(plan.cli_args)["--max-pod-seconds"] == "19800"
    assert plan.vm_ceiling_s == pytest.approx(19800.0)

    # --max-pod-seconds is a runpod-only knob -- absent on local and gcp.
    local_plan = check_enforceability(envelope, _ctx(sandbox="local", tiering_strategy=None, max_gpu_count=1))
    assert all(name != "--max-pod-seconds" for name, _ in local_plan.cli_args)

    gcp_plan = check_enforceability(
        envelope, _ctx(sandbox="gcp", tiering_strategy=None, max_gpu_count=1, gpu_usd_per_hr=2.0)
    )
    assert all(name != "--max-pod-seconds" for name, _ in gcp_plan.cli_args)


# ---------------------------------------------------------------------------
# attempt_estimate
# ---------------------------------------------------------------------------


def test_attempt_estimate_adds_provision_under_stage_on_gpu():
    ctx = _ctx(
        sandbox="gcp", tiering_strategy="stage_on_gpu", max_gpu_count=2,
        gpu_usd_per_hr=3.0, provision_overhead_s=600.0,
    )
    result = attempt_estimate(est_gpu_hours=4.0, est_usd=8.0, ctx=ctx)

    provision_hours = 600.0 / 3600.0 * 2
    assert result.gpu_hours == pytest.approx(4.0 + provision_hours)
    assert result.gpu_usd == pytest.approx(8.0 + provision_hours * 3.0)
    assert result.llm_usd == pytest.approx(ENVELOPE_FLOORS["llm_usd"])
    assert result.wall_s == 0.0


def test_attempt_estimate_without_stage_on_gpu_no_addition():
    ctx = _ctx(sandbox="gcp", tiering_strategy=None, max_gpu_count=2, gpu_usd_per_hr=3.0)
    result = attempt_estimate(est_gpu_hours=4.0, est_usd=8.0, ctx=ctx)
    assert result.gpu_hours == pytest.approx(4.0)
    assert result.gpu_usd == pytest.approx(8.0)


def test_attempt_estimate_stage_on_gpu_without_known_rate_no_addition():
    ctx = _ctx(sandbox="gcp", tiering_strategy="stage_on_gpu", max_gpu_count=2, gpu_usd_per_hr=None)
    result = attempt_estimate(est_gpu_hours=4.0, est_usd=8.0, ctx=ctx)
    assert result.gpu_hours == pytest.approx(4.0)
    assert result.gpu_usd == pytest.approx(8.0)


# ===========================================================================
# Part 2 (Unit 5) — DECIDE policy tests: terminal rules, guard-filtered
# champion selection (F5), lineage arms, scope ladder, plateau, and the
# typed prose-free novelty fingerprint (F10). Spec §8.1-§8.4.
# ===========================================================================

_UNSET = object()

_CLEAN_PREDICATES: dict = {
    "backed_by_ledger": True,
    "provenance_present": True,
    "metrics_non_degenerate": True,
    "metric_keys_real": True,
    "rerun_agrees": None,
    "run_level_clean": True,
}
_CLEAN_GUARDS: dict = {
    "fabrication": False,
    "all_models_failed": False,
    "env_unavailable": False,
    "no_learning_signal": False,
}
_CONTRADICTED_KWARGS: dict = dict(
    meets_target=False, implementation_verdict="faithful", replication_verdict="contradicted"
)
_BASE_ENVELOPE: dict = {
    "llm_usd": 5.0, "gpu_usd": 10.0, "gpu_hours": 2.0, "wall_s": 14400.0, "vm_ceiling_s": 16200.0,
}


def _predicates(true_count: int) -> dict:
    """An evidence_predicates dict whose ``evidence_count`` is exactly
    ``true_count`` (0-4): the first ``true_count`` of the four counted keys
    are True, the rest False. ``rerun_agrees``/``run_level_clean`` are
    excluded from the count by design (see ``evidence_count``'s docstring)
    so their values here never perturb it."""
    keys = ("backed_by_ledger", "provenance_present", "metrics_non_degenerate", "metric_keys_real")
    return {
        **{k: i < true_count for i, k in enumerate(keys)},
        "rerun_agrees": None,
        "run_level_clean": True,
    }


def _report(
    *,
    score: float | None = 0.7,
    target: float | None = 0.6,
    meets_target: bool | None = True,
    implementation_verdict: str | None = "faithful",
    replication_verdict: str | None = "reproduced",
    verdict: str | None = "reproduced",
    stop_reason: str | None = None,
    exclusions: tuple = (),
    path: str = "/runs/p/final_report.json",
) -> ReportDigest:
    return ReportDigest(
        score=score, target=target, meets_target=meets_target,
        implementation_verdict=implementation_verdict, replication_verdict=replication_verdict,
        verdict=verdict, stop_reason=stop_reason, exclusions=exclusions, path=path,
    )


def _assessment(
    attempt_n: int,
    *,
    driver: str = "live_cli",
    project_id: str = "proj-1",
    directives_sha256: str = "sha",
    final_report: object = _UNSET,
    evidence_predicates: dict | None = None,
    guard_flags: dict | None = None,
    validator: ValidatorStatus | None = None,
    leaf_pass_count: int | None = 5,
    leaf_vector_ref: str | None = "/runs/p/rubric_evaluation.json",
    failure_class: str | None = None,
    failure_signature: str | None = None,
    failure_scope: str | None = None,
    cost: CampaignSpend | None = None,
    rubric_sha256_ok: bool | None = None,
    hard_quarantined: bool = False,
    soft_quarantined: bool = False,
    quarantine_reasons: tuple = (),
    **report_overrides: object,
) -> AttemptAssessment:
    """Fixture factory building the REAL ``AttemptAssessment`` dataclass
    (never redefined here). Clean-by-default (``grade_usable_for_terminal``
    is True unless a quarantine flag is set). Pass ``final_report=None`` for
    a missing report, or any ``_report(...)`` kwarg (``score=``/``target=``/
    ``meets_target=``/``implementation_verdict=``/...) to steer the
    auto-built one via ``**report_overrides``."""
    if final_report is _UNSET:
        final_report = _report(**report_overrides)  # type: ignore[arg-type]
    return AttemptAssessment(
        attempt_n=attempt_n,
        driver=driver,
        project_id=project_id,
        directives_sha256=directives_sha256,
        final_report=final_report,  # type: ignore[arg-type]
        evidence_predicates=dict(evidence_predicates if evidence_predicates is not None else _CLEAN_PREDICATES),
        guard_flags=dict(guard_flags if guard_flags is not None else _CLEAN_GUARDS),
        validator=validator if validator is not None else ValidatorStatus(status="clean", fingerprint="fp", fresh=True),
        leaf_pass_count=leaf_pass_count,
        leaf_vector_ref=leaf_vector_ref,
        failure_class=failure_class,
        failure_signature=failure_signature,
        failure_scope=failure_scope,
        cost=cost if cost is not None else CampaignSpend(),
        rubric_sha256_ok=rubric_sha256_ok,
        hard_quarantined=hard_quarantined,
        soft_quarantined=soft_quarantined,
        quarantine_reasons=quarantine_reasons,
    )


def _config(**overrides: object) -> PolicyConfig:
    # plateau_k defaults to 99 (effectively disabled) so a test targeting
    # one EXHAUSTED sub-rule does not accidentally trip the plateau rule
    # too -- override it explicitly in tests that exercise plateau itself.
    defaults: dict[str, object] = dict(max_attempts=6, plateau_k=99, width=1, width_skip_score=0.5, ladder_len=1)
    defaults.update(overrides)
    return PolicyConfig(**defaults)  # type: ignore[arg-type]


def _tiny_estimate(**overrides: object) -> CampaignSpend:
    defaults: dict[str, object] = dict(llm_usd=0.01, gpu_usd=0.01, gpu_hours=0.01, wall_s=0.0)
    defaults.update(overrides)
    return CampaignSpend(**defaults)  # type: ignore[arg-type]


def _decide(
    assessments: tuple,
    *,
    budget: CampaignBudget | None = None,
    spent: CampaignSpend | None = None,
    config: PolicyConfig | None = None,
    next_estimate: CampaignSpend | None = None,
    lineage_by_attempt: dict | None = None,
    scope_rung_by_attempt: dict | None = None,
    runs_dir_hint: dict | None = None,
    current_rung: int = 0,
    blocking_gap: str | None = None,
) -> Decision:
    """Thin ``decide()`` wrapper with campaign-agnostic defaults: a huge,
    effectively-unconstrained budget and (via ``_config()``'s own default)
    plateau disabled, so a test targeting one rule does not accidentally
    trip another. Every given assessment defaults to scope rung 0 (the full
    rung under the default ``ladder_len=1``) unless overridden."""
    if scope_rung_by_attempt is None:
        scope_rung_by_attempt = {a.attempt_n: 0 for a in assessments}
    return decide(
        assessments,
        budget=budget if budget is not None else _budget(
            max_llm_usd=1000.0, max_gpu_usd=1000.0, max_gpu_hours=1000.0
        ),
        spent=spent if spent is not None else CampaignSpend(),
        config=config if config is not None else _config(),
        next_estimate=next_estimate if next_estimate is not None else _tiny_estimate(),
        lineage_by_attempt=lineage_by_attempt or {},
        scope_rung_by_attempt=scope_rung_by_attempt,
        runs_dir_hint=runs_dir_hint or {},
        current_rung=current_rung,
        blocking_gap=blocking_gap,
    )


# ---------------------------------------------------------------------------
# DECIDE — rule 1: REPRODUCED
# ---------------------------------------------------------------------------


def test_reproduced_requires_full_scope_rung():
    a = _assessment(1, meets_target=True)
    decision = _decide((a,), config=_config(ladder_len=3), scope_rung_by_attempt={1: 0}, current_rung=0)
    assert decision.kind == "CONTINUE"


def test_reproduced_blocked_by_soft_quarantine_validator_missing():
    a = _assessment(
        1, meets_target=True, soft_quarantined=True,
        validator=ValidatorStatus(status="missing", fingerprint=None, fresh=False),
    )
    decision = _decide((a,))
    assert decision.kind == "CONTINUE"


def test_reproduced_blocked_by_hard_quarantine_guard():
    a = _assessment(1, meets_target=True, hard_quarantined=True)
    decision = _decide((a,))
    assert decision.kind == "CONTINUE"


def test_reproduced_blocked_by_rubric_hash_mismatch():
    # hard_quarantined is left False so this cannot be passing merely
    # because of the guard test above -- decide() must consult
    # rubric_sha256_ok directly (rule 1's own "AND rubric_sha256_ok is not
    # False" clause), independent of the hard_quarantined flag.
    a = _assessment(1, meets_target=True, rubric_sha256_ok=False)
    decision = _decide((a,))
    assert decision.kind == "CONTINUE"


def test_reproduced_requires_run_level_clean():
    # hard_quarantined/soft_quarantined both False -- AttemptAssessment does
    # not derive either from evidence_predicates -- but run_level_clean
    # itself is False; rule 1's separate, explicit predicate check must
    # still block REPRODUCED.
    a = _assessment(1, meets_target=True, evidence_predicates={**_CLEAN_PREDICATES, "run_level_clean": False})
    decision = _decide((a,))
    assert decision.kind == "CONTINUE"


def test_reproduced_happy_path_sets_champion():
    a = _assessment(1, meets_target=True)
    decision = _decide((a,))
    assert decision.kind == "REPRODUCED"
    assert decision.rule == "reproduced"
    assert decision.stop_reason is None
    assert decision.next_plan is None
    assert select_champion((a,)) is a


# ---------------------------------------------------------------------------
# DECIDE — rule 2: CONTRADICTED
# ---------------------------------------------------------------------------


def test_contradicted_requires_two_different_lineages():
    one = (_assessment(1, **_CONTRADICTED_KWARGS),)
    assert _decide(one, lineage_by_attempt={1: "champion"}).kind != "CONTRADICTED"

    same_lineage = (_assessment(1, **_CONTRADICTED_KWARGS), _assessment(2, **_CONTRADICTED_KWARGS))
    assert _decide(same_lineage, lineage_by_attempt={1: "champion", 2: "champion"}).kind != "CONTRADICTED"

    diff_lineage = (_assessment(1, **_CONTRADICTED_KWARGS), _assessment(2, **_CONTRADICTED_KWARGS))
    decision = _decide(diff_lineage, lineage_by_attempt={1: "champion", 2: "runner_up"})
    assert decision.kind == "CONTRADICTED"
    assert decision.rule == "contradicted_two_lineages"
    assert decision.next_plan is None


def test_contradicted_requires_clean_envelope_on_both():
    tainted = (
        _assessment(1, hard_quarantined=True, **_CONTRADICTED_KWARGS),
        _assessment(2, **_CONTRADICTED_KWARGS),
    )
    decision = _decide(tainted, lineage_by_attempt={1: "champion", 2: "runner_up"})
    assert decision.kind != "CONTRADICTED"


# ---------------------------------------------------------------------------
# DECIDE — rule 3: INFEASIBLE
# ---------------------------------------------------------------------------


def test_infeasible_on_blocking_gap():
    decision = _decide((), blocking_gap="dataset_gated_no_license")
    assert decision.kind == "INFEASIBLE"
    assert decision.rule == "blocking_gap"
    assert decision.stop_reason == "infeasible:dataset_gated_no_license"
    assert decision.next_plan is None


# ---------------------------------------------------------------------------
# DECIDE — rule 4a: per-meter budget floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("meter", ["llm_usd", "gpu_usd", "gpu_hours"])
def test_budget_floor_per_meter_arithmetic(meter):
    budget = _budget(max_llm_usd=100.0, max_gpu_usd=100.0, max_gpu_hours=100.0, max_attempts=99)
    spent_kwargs = dict(llm_usd=90.0, gpu_usd=90.0, gpu_hours=90.0, wall_s=0.0)
    spent_kwargs[meter] = 95.0  # remaining = 5.0 on the tested meter
    spent = CampaignSpend(**spent_kwargs)

    boundary_kwargs = dict(llm_usd=0.01, gpu_usd=0.01, gpu_hours=0.01, wall_s=0.0)
    boundary_kwargs[meter] = 5.0  # remaining == estimate exactly -> NOT exhausted
    boundary = _decide((), budget=budget, spent=spent, next_estimate=CampaignSpend(**boundary_kwargs))
    assert boundary.kind != "EXHAUSTED"

    over_kwargs = dict(boundary_kwargs)
    over_kwargs[meter] = 5.01  # remaining < estimate -> exhausted
    over = _decide((), budget=budget, spent=spent, next_estimate=CampaignSpend(**over_kwargs))
    assert over.kind == "EXHAUSTED"
    assert over.rule == "budget_floor"
    assert over.stop_reason == f"budget_floor:{meter}"


def test_budget_floor_wall_meter_only_checked_when_campaign_wall_set():
    no_wall_budget = _budget(max_llm_usd=1000.0, max_gpu_usd=1000.0, max_gpu_hours=1000.0)
    decision = _decide((), budget=no_wall_budget, next_estimate=_tiny_estimate(wall_s=10_000_000.0))
    assert decision.kind != "EXHAUSTED"

    wall_budget = _budget(max_llm_usd=1000.0, max_gpu_usd=1000.0, max_gpu_hours=1000.0, max_wall_clock_s=100.0)
    decision2 = _decide((), budget=wall_budget, next_estimate=_tiny_estimate(wall_s=200.0))
    assert decision2.kind == "EXHAUSTED"
    assert decision2.stop_reason == "budget_floor:wall_s"


# ---------------------------------------------------------------------------
# DECIDE — rule 4b: max_attempts
# ---------------------------------------------------------------------------


def test_max_attempts():
    config = _config(max_attempts=2)

    one = (_assessment(1, meets_target=False),)
    assert _decide(one, config=config).kind == "CONTINUE"

    two = (_assessment(1, meets_target=False), _assessment(2, meets_target=False))
    decision = _decide(two, config=config)
    assert decision.kind == "EXHAUSTED"
    assert decision.rule == "max_attempts"
    assert decision.stop_reason == "max_attempts"


# ---------------------------------------------------------------------------
# DECIDE — rule 4c: plateau
# ---------------------------------------------------------------------------


def test_plateau_fires_at_k_consecutive_no_improvement_no_new_signature():
    assessments = (
        _assessment(1, meets_target=False, evidence_predicates=_predicates(3)),
        _assessment(2, meets_target=False, evidence_predicates=_predicates(3)),
        _assessment(3, meets_target=False, evidence_predicates=_predicates(3)),
    )
    decision = _decide(assessments, config=_config(plateau_k=2))
    assert decision.kind == "EXHAUSTED"
    assert decision.rule == "plateau"
    assert decision.stop_reason == "plateau"


def test_plateau_reset_by_evidence_improvement():
    assessments = (
        _assessment(1, meets_target=False, evidence_predicates=_predicates(2)),
        _assessment(2, meets_target=False, evidence_predicates=_predicates(2)),
        _assessment(3, meets_target=False, evidence_predicates=_predicates(3)),  # improved
    )
    decision = _decide(assessments, config=_config(plateau_k=2))
    assert decision.kind == "CONTINUE"


def test_plateau_reset_by_new_failure_signature():
    assessments = (
        _assessment(1, meets_target=False, evidence_predicates=_predicates(3), failure_signature="sig_a"),
        _assessment(2, meets_target=False, evidence_predicates=_predicates(3), failure_signature="sig_a"),
        _assessment(3, meets_target=False, evidence_predicates=_predicates(3), failure_signature="sig_b"),
    )
    decision = _decide(assessments, config=_config(plateau_k=2))
    assert decision.kind == "CONTINUE"


# ---------------------------------------------------------------------------
# DECIDE — rule 4d: repeated infra signature
# ---------------------------------------------------------------------------


def test_infra_signature_three_consecutive_stops():
    two = (
        _assessment(1, meets_target=False, failure_signature="infra_x", failure_scope="infra"),
        _assessment(2, meets_target=False, failure_signature="infra_x", failure_scope="infra"),
    )
    assert _decide(two).kind == "CONTINUE"

    three = two + (_assessment(3, meets_target=False, failure_signature="infra_x", failure_scope="infra"),)
    decision = _decide(three)
    assert decision.kind == "EXHAUSTED"
    assert decision.rule == "infra_signature_repeated"
    assert decision.stop_reason == "infra_signature_repeated"


# ---------------------------------------------------------------------------
# DECIDE — rule 4e: report_missing_twice
# ---------------------------------------------------------------------------


def test_report_missing_twice_unrepairable():
    once = (_assessment(1, final_report=None, failure_class="report_missing"),)
    assert _decide(once).kind == "CONTINUE"

    twice = once + (_assessment(2, final_report=None, failure_class="report_missing"),)
    decision = _decide(twice)
    assert decision.kind == "EXHAUSTED"
    assert decision.rule == "report_missing_twice"
    assert decision.stop_reason == "report_missing_twice"


# ---------------------------------------------------------------------------
# select_champion / seeding_pool / campaign_floor (F5)
# ---------------------------------------------------------------------------


def test_select_champion_filters_hard_and_soft_quarantine():
    hard = _assessment(1, hard_quarantined=True, evidence_predicates=_predicates(4))
    soft = _assessment(2, soft_quarantined=True, evidence_predicates=_predicates(4))
    clean = _assessment(3, evidence_predicates=_predicates(1))

    champion = select_champion((hard, soft, clean))
    assert champion is clean  # a HIGHER-scoring guard-tripped attempt never wins

    assert select_champion((hard, soft)) is None


def test_select_champion_ranking_tiebreaks():
    # evidence count dominates outright, even against a worse everything-else.
    low_evidence_best_else = _assessment(1, evidence_predicates=_predicates(2), score=0.9, target=0.6, leaf_pass_count=10)
    high_evidence_worst_else = _assessment(2, evidence_predicates=_predicates(3), score=0.1, target=0.9, leaf_pass_count=0)
    assert select_champion((low_evidence_best_else, high_evidence_worst_else)) is high_evidence_worst_else

    # tied evidence count -> smaller target distance wins.
    far = _assessment(3, evidence_predicates=_predicates(2), score=0.1, target=0.9)
    near = _assessment(4, evidence_predicates=_predicates(2), score=0.55, target=0.6)
    assert select_champion((far, near)) is near

    # tied evidence + tied distance -> higher leaf_pass_count wins (None == -1, worst).
    no_leaf = _assessment(5, evidence_predicates=_predicates(2), score=0.5, target=0.6, leaf_pass_count=None)
    some_leaf = _assessment(6, evidence_predicates=_predicates(2), score=0.5, target=0.6, leaf_pass_count=0)
    assert select_champion((no_leaf, some_leaf)) is some_leaf

    # tied evidence + tied distance + tied leaf count -> smaller attempt_n wins.
    later = _assessment(8, evidence_predicates=_predicates(2), score=0.5, target=0.6, leaf_pass_count=3)
    earlier = _assessment(7, evidence_predicates=_predicates(2), score=0.5, target=0.6, leaf_pass_count=3)
    assert select_champion((later, earlier)) is earlier


def test_seeding_pool_keeps_soft_quarantined_excludes_hard():
    hard = _assessment(1, hard_quarantined=True)
    soft = _assessment(2, soft_quarantined=True)
    clean = _assessment(3)

    pool = seeding_pool((hard, soft, clean))
    assert {a.attempt_n for a in pool} == {2, 3}


def test_campaign_floor_from_clean_only():
    hard = _assessment(1, hard_quarantined=True, score=0.99)
    soft = _assessment(2, soft_quarantined=True, score=0.95)
    clean = _assessment(3, score=0.9)

    assert campaign_floor((hard, soft, clean)) == pytest.approx(0.9)
    assert campaign_floor((hard, soft)) is None


# ---------------------------------------------------------------------------
# lineage_arms (§8.2)
# ---------------------------------------------------------------------------


def test_lineage_arms_fresh_first_attempt():
    arms = lineage_arms((), runs_dir_hint={})
    assert arms == (NextAttemptPlan(lineage="fresh", seed_attempt_n=None, seed_pointer=None, scope_rung=0, width=1),)


def test_lineage_arms_champion_seeded_by_default():
    a = _assessment(1, evidence_predicates=_predicates(3))
    arms = lineage_arms((a,), runs_dir_hint={1: "/runs/p/attempt_1/code"})
    assert arms[0].lineage == "champion"
    assert arms[0].seed_attempt_n == 1
    assert arms[0].seed_pointer == "/runs/p/attempt_1/code"
    assert arms[-1].lineage == "fresh"


def test_lineage_arms_switch_to_runner_up_after_champion_lineage_stalls_twice():
    a1 = _assessment(1, evidence_predicates=_predicates(4))  # the champion: best evidence
    a2 = _assessment(2, evidence_predicates=_predicates(2))  # seeded from champion, weaker
    a3 = _assessment(3, evidence_predicates=_predicates(3))  # seeded from champion, still <= 4

    runs_dir_hint = {1: "/runs/p/1/code", 2: "/runs/p/2/code", 3: "/runs/p/3/code"}
    lineage_by_attempt = {1: "fresh", 2: "champion", 3: "champion"}

    assert select_champion((a1, a2, a3)) is a1  # sanity: a1 is genuinely the champion

    arms = lineage_arms((a1, a2, a3), runs_dir_hint=runs_dir_hint, lineage_by_attempt=lineage_by_attempt)

    assert all(arm.seed_attempt_n != 1 for arm in arms)  # champion (a1) excluded -- stalled
    assert arms[0].lineage == "runner_up"
    assert arms[0].seed_attempt_n == 3  # a3 outranks a2 in the seeding-pool ranking
    assert arms[-1].lineage == "fresh"


def test_lineage_arms_end_with_fresh():
    assert lineage_arms((), runs_dir_hint={})[-1].lineage == "fresh"

    a = _assessment(1, evidence_predicates=_predicates(2))
    assert lineage_arms((a,), runs_dir_hint={1: "/runs/p/1/code"})[-1].lineage == "fresh"

    b = _assessment(2, evidence_predicates=_predicates(1))
    arms = lineage_arms((a, b), runs_dir_hint={1: "/runs/p/1/code", 2: "/runs/p/2/code"})
    assert arms[-1].lineage == "fresh"


# ---------------------------------------------------------------------------
# next_scope_rung (§8.2 scope ladder)
# ---------------------------------------------------------------------------


def test_next_scope_rung_reexpands_one_rung_on_green_only():
    green = _assessment(1, meets_target=True)
    not_meeting = _assessment(2, meets_target=False)
    quarantined_but_meeting = _assessment(3, meets_target=True, hard_quarantined=True)

    assert next_scope_rung(0, None, ladder_len=3) == 0  # no latest -> hold
    assert next_scope_rung(0, green, ladder_len=3) == 1  # green -> advance one rung
    assert next_scope_rung(0, not_meeting, ladder_len=3) == 0  # not meeting target -> hold
    assert next_scope_rung(0, quarantined_but_meeting, ladder_len=3) == 0  # quarantined -> hold

    assert next_scope_rung(2, green, ladder_len=3) == 2  # clamp: already full, stays full
    assert next_scope_rung(1, None, ladder_len=3) == 1  # hold never drops below current_rung


# ---------------------------------------------------------------------------
# width_for_next (§8.3)
# ---------------------------------------------------------------------------


def test_width_requires_weak_history_and_budget_on_every_meter():
    weak_history = (_assessment(1, score=0.2, target=0.9, meets_target=False),)
    strong_history = (_assessment(1, score=0.8, target=0.9, meets_target=False),)
    huge_budget = _budget(max_llm_usd=1000.0, max_gpu_usd=1000.0, max_gpu_hours=1000.0, max_attempts=99)
    tiny_estimate = _tiny_estimate()

    # width==1 configured -> always 1, regardless of everything else.
    assert width_for_next(
        weak_history, config=_config(width=1), budget=huge_budget, spent=CampaignSpend(), next_estimate=tiny_estimate
    ) == 1

    # width>1 but history is NOT weak (a prior score >= width_skip_score) -> 1.
    assert width_for_next(
        strong_history, config=_config(width=3, width_skip_score=0.5), budget=huge_budget,
        spent=CampaignSpend(), next_estimate=tiny_estimate,
    ) == 1

    # width>1, weak history, but one meter's remaining budget can't cover
    # width * estimate -> 1.
    starved_budget = _budget(max_llm_usd=5.0, max_gpu_usd=1000.0, max_gpu_hours=1000.0, max_attempts=99)
    assert width_for_next(
        weak_history, config=_config(width=3, width_skip_score=0.5), budget=starved_budget,
        spent=CampaignSpend(), next_estimate=_tiny_estimate(llm_usd=2.0),
    ) == 1

    # width>1, weak history, budget covers it, but too few attempts remain -> 1.
    assert width_for_next(
        weak_history, config=_config(width=3, width_skip_score=0.5, max_attempts=2), budget=huge_budget,
        spent=CampaignSpend(), next_estimate=tiny_estimate,
    ) == 1

    # everything satisfied -> width granted.
    assert width_for_next(
        weak_history, config=_config(width=3, width_skip_score=0.5), budget=huge_budget,
        spent=CampaignSpend(), next_estimate=tiny_estimate,
    ) == 3


# ---------------------------------------------------------------------------
# directives_fingerprint (§8.4, F10)
# ---------------------------------------------------------------------------


def test_fingerprint_prose_free_identical_typed_inputs_equal():
    # The signature has no parameter for prose at all -- there is no way to
    # even PASS propose_improvements text or a grader justification to it.
    # Two "attempts" that would differ only in such prose (simulated here as
    # local variables never fed into the call) hash identically, because the
    # call is byte-identical on every typed field the function DOES accept.
    attempt_a_prose = "propose_improvements: try a lower learning rate this time"
    attempt_b_prose = "propose_improvements: the grader thinks batch size is the issue"
    assert attempt_a_prose != attempt_b_prose  # the prose genuinely differs...

    kwargs = dict(
        seed_lineage="champion", scope_rung=0,
        repair_action_kinds=["protocol_gap", "render_artifact"],
        failure_classes=["fabrication_suspected"],
        envelope=_BASE_ENVELOPE,
    )
    fp_a = directives_fingerprint(**kwargs)
    fp_b = directives_fingerprint(**kwargs)
    assert fp_a == fp_b  # ...but it was never an input, so nothing changes.
    assert isinstance(fp_a, str) and len(fp_a) == 64  # sha256 hexdigest


def test_fingerprint_envelope_quantization():
    base = directives_fingerprint(
        seed_lineage="champion", scope_rung=0, repair_action_kinds=["protocol_gap"],
        failure_classes=["x"], envelope=_BASE_ENVELOPE,
    )

    # small drift within the same 1.0 / 0.25 / 900.0 buckets -> identical hash.
    drifted = dict(_BASE_ENVELOPE)
    drifted["llm_usd"] = 5.4  # rounds to 5.0 -- same bucket as base's 5.0
    drifted["gpu_hours"] = 2.1  # rounds to 2.0 (nearest 0.25) -- same bucket
    drifted["wall_s"] = 14550.0  # rounds to 14400.0 (nearest 900) -- same bucket
    same_hash = directives_fingerprint(
        seed_lineage="champion", scope_rung=0, repair_action_kinds=["protocol_gap"],
        failure_classes=["x"], envelope=drifted,
    )
    assert same_hash == base

    # crossing a bucket boundary -> different hash.
    crossed = dict(_BASE_ENVELOPE)
    crossed["llm_usd"] = 5.6  # rounds to 6.0 -- a different bucket
    different_hash = directives_fingerprint(
        seed_lineage="champion", scope_rung=0, repair_action_kinds=["protocol_gap"],
        failure_classes=["x"], envelope=crossed,
    )
    assert different_hash != base


def test_fingerprint_arm_and_rung_and_kinds_change_hash():
    base_kwargs = dict(
        seed_lineage="champion", scope_rung=0,
        repair_action_kinds=["protocol_gap"], failure_classes=["fabrication_suspected"],
        envelope=_BASE_ENVELOPE,
    )
    base = directives_fingerprint(**base_kwargs)

    assert directives_fingerprint(**{**base_kwargs, "seed_lineage": "runner_up"}) != base
    assert directives_fingerprint(**{**base_kwargs, "scope_rung": 1}) != base
    assert directives_fingerprint(**{**base_kwargs, "repair_action_kinds": ["render_artifact"]}) != base
    assert directives_fingerprint(**{**base_kwargs, "failure_classes": ["all_models_failed"]}) != base

    # kind ORDER and duplicates must not matter -- the hash sorts+sets them.
    assert directives_fingerprint(**{**base_kwargs, "repair_action_kinds": ["protocol_gap", "protocol_gap"]}) == base


# ---------------------------------------------------------------------------
# Controller-review fix pass: champion_attempt_n on every terminal (F5/§5/§12)
# + Decision.to_dict() + F4 seeding-starvation fix in lineage_arms (§8.1)
# ---------------------------------------------------------------------------


def _reproduced_carries_champion_case():
    a = _assessment(1, meets_target=True)
    return (a,), _config(), "REPRODUCED", 1


def _exhausted_max_attempts_with_champion_case():
    # attempt 1 has strictly more evidence than attempt 2 -- select_champion
    # must pick it unambiguously (evidence count dominates the rank key).
    strong = _assessment(1, meets_target=False, evidence_predicates=_predicates(3))
    weak = _assessment(2, meets_target=False, evidence_predicates=_predicates(1))
    return (strong, weak), _config(max_attempts=2), "EXHAUSTED", 1


def _exhausted_no_clean_attempt_case():
    # both attempts hard-quarantined -- select_champion has nothing to pick.
    a = _assessment(1, meets_target=False, hard_quarantined=True)
    b = _assessment(2, meets_target=False, hard_quarantined=True)
    return (a, b), _config(max_attempts=2), "EXHAUSTED", None


@pytest.mark.parametrize(
    "case_factory",
    [
        _reproduced_carries_champion_case,
        _exhausted_max_attempts_with_champion_case,
        _exhausted_no_clean_attempt_case,
    ],
    ids=["reproduced", "exhausted_max_attempts_with_champion", "exhausted_no_clean_attempt"],
)
def test_terminal_decisions_carry_champion_so_far(case_factory):
    assessments, config, expected_kind, expected_champion = case_factory()
    decision = _decide(assessments, config=config)
    assert decision.kind == expected_kind
    assert decision.champion_attempt_n == expected_champion


def test_decision_to_dict_roundtrips_next_plan():
    plan = NextAttemptPlan(
        lineage="champion", seed_attempt_n=3, seed_pointer="/runs/p/3/code", scope_rung=1, width=2
    )
    decision = Decision(
        kind="CONTINUE", rule="continue", stop_reason=None, next_plan=plan, champion_attempt_n=3
    )

    payload = decision.to_dict()

    assert payload == {
        "kind": "CONTINUE",
        "rule": "continue",
        "stop_reason": None,
        "next_plan": {
            "lineage": "champion",
            "seed_attempt_n": 3,
            "seed_pointer": "/runs/p/3/code",
            "scope_rung": 1,
            "width": 2,
        },
        "champion_attempt_n": 3,
    }
    # next_plan must be a real nested dict (asdict semantics), not the
    # NextAttemptPlan instance itself or some shallow vars() stand-in --
    # this is exactly the shape reproduction_campaign.py's
    # decision.get("next_plan").get("scope_rung") accesses need.
    assert isinstance(payload["next_plan"], dict)
    assert payload["next_plan"]["scope_rung"] == 1

    # a terminal Decision's next_plan=None survives the round trip untouched.
    terminal = Decision(
        kind="EXHAUSTED", rule="max_attempts", stop_reason="max_attempts", next_plan=None, champion_attempt_n=2
    )
    terminal_payload = terminal.to_dict()
    assert terminal_payload["next_plan"] is None
    assert terminal_payload["champion_attempt_n"] == 2


def test_scheduler_plan_defaults_preserve_legacy_decision_payload_and_reject_invalid_metadata():
    legacy_plan = NextAttemptPlan(
        lineage="fresh", seed_attempt_n=None, seed_pointer=None, scope_rung=0, width=1
    )
    legacy_payload = Decision(
        kind="CONTINUE", rule="continue", stop_reason=None, next_plan=legacy_plan
    ).to_dict()
    assert legacy_payload["next_plan"] == {
        "lineage": "fresh",
        "seed_attempt_n": None,
        "seed_pointer": None,
        "scope_rung": 0,
        "width": 1,
    }

    typed_plan = NextAttemptPlan(
        lineage="fresh", seed_attempt_n=None, seed_pointer=None, scope_rung=0, width=1,
        branch_type="ambiguity",
    )
    assert Decision(
        kind="CONTINUE", rule="continue", stop_reason=None, next_plan=typed_plan
    ).to_dict()["next_plan"]["branch_type"] == "ambiguity"

    with pytest.raises(ValueError):
        NextAttemptPlan(
            lineage="fresh", seed_attempt_n=None, seed_pointer=None, scope_rung=0, width=1,
            branch_type="free-text",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        NextAttemptPlan(
            lineage="fresh", seed_attempt_n=None, seed_pointer=None, scope_rung=0, width=1,
            branch_type="ambiguity", is_safety_bracket=True,
        )


def test_no_clean_champion_still_seeds_from_best_seedable():
    weak = _assessment(1, soft_quarantined=True, evidence_predicates=_predicates(1))
    strong = _assessment(2, soft_quarantined=True, evidence_predicates=_predicates(3))
    # scores/ranks best of the three but is HARD-quarantined -- must never
    # be chosen for seeding, unlike the soft-quarantined pair above.
    hard = _assessment(3, hard_quarantined=True, evidence_predicates=_predicates(4))

    assert select_champion((weak, strong, hard)) is None  # sanity: nothing guard-clean

    runs_dir_hint = {1: "/runs/p/1/code", 2: "/runs/p/2/code", 3: "/runs/p/3/code"}
    arms = lineage_arms((weak, strong, hard), runs_dir_hint=runs_dir_hint)

    assert len(arms) == 2
    assert arms[0].lineage == "runner_up"
    assert arms[0].seed_attempt_n == 2  # strong outranks weak in the seeding pool
    assert arms[0].seed_pointer == "/runs/p/2/code"
    assert all(arm.seed_attempt_n != 3 for arm in arms)  # hard-quarantined never chosen
    assert arms[-1].lineage == "fresh"

    # a missing runs_dir_hint pointer for the best-seedable attempt falls
    # through to fresh-only, matching the champion/runner_up branches' own
    # "skipped if absent" rule.
    no_pointer_arms = lineage_arms((weak, strong, hard), runs_dir_hint={})
    assert no_pointer_arms == (
        NextAttemptPlan(lineage="fresh", seed_attempt_n=None, seed_pointer=None, scope_rung=0, width=1),
    )


# ---------------------------------------------------------------------------
# Mutation-testing rigor pass (Unit 5 DECIDE): a mutation-testing review
# proved the implementation correct but these five behaviors unpinned --
# each test below fails against the specific mutant it targets and passes
# against the real implementation. See each docstring for the mutant pinned.
# ---------------------------------------------------------------------------


def test_rule_order_reproduced_wins_over_exhausted():
    """MAJOR: a REPRODUCED-qualifying attempt (clean, meets_target at the
    full rung, run_level_clean, rubric ok) coexists with
    len(assessments) >= config.max_attempts -- rule 1 (REPRODUCED) must be
    evaluated, and win, before rule 4b (max_attempts) ever gets a look. A
    mutant that checks max_attempts FIRST returns EXHAUSTED/max_attempts
    here instead -- this pins the rule-table ORDER (1 strictly before 4)."""
    a = _assessment(1, meets_target=True, evidence_predicates=_CLEAN_PREDICATES, rubric_sha256_ok=None)
    config = _config(max_attempts=1)
    assert len((a,)) >= config.max_attempts  # sanity: rule 4b's own condition genuinely holds too

    decision = _decide((a,), config=config)
    assert decision.kind == "REPRODUCED"
    assert decision.rule == "reproduced"
    assert decision.champion_attempt_n == 1


def test_plateau_mid_window_improvement_blocks():
    """MAJOR: evidence counts [3, 4, 3] across 3 attempts, every
    failure_signature previously seen, plateau_k=2 -- the window is
    [attempt2, attempt3]. attempt2's evidence (4) beats the best
    strictly-before-it (attempt1's 3): a genuine mid-window improvement
    that must block plateau. A mutant that inspects only the LAST window
    item (attempt3: evidence 3, not > the campaign-wide best-before-it of
    4) would call this plateaued -- this pins the per-window-MEMBER
    semantics (every window item is checked, not just the last)."""
    assessments = (
        _assessment(1, meets_target=False, evidence_predicates=_predicates(3), failure_signature="sig_a"),
        _assessment(2, meets_target=False, evidence_predicates=_predicates(4), failure_signature="sig_a"),
        _assessment(3, meets_target=False, evidence_predicates=_predicates(3), failure_signature="sig_a"),
    )
    decision = _decide(assessments, config=_config(plateau_k=2))
    assert decision.kind == "CONTINUE"


def test_plateau_fires_flat_evidence_seen_signature():
    """Companion to the above: evidence counts [3, 3, 3] (flat, no
    per-window improvement) with an explicit, previously-seen
    failure_signature on every attempt (not relying on the None-signature
    short-circuit the original fires-test used) -- plateau must still
    fire."""
    assessments = (
        _assessment(1, meets_target=False, evidence_predicates=_predicates(3), failure_signature="sig_a"),
        _assessment(2, meets_target=False, evidence_predicates=_predicates(3), failure_signature="sig_a"),
        _assessment(3, meets_target=False, evidence_predicates=_predicates(3), failure_signature="sig_a"),
    )
    decision = _decide(assessments, config=_config(plateau_k=2))
    assert decision.kind == "EXHAUSTED"
    assert decision.rule == "plateau"
    assert decision.stop_reason == "plateau"


def test_infra_repeat_mixed_scope_does_not_fire():
    """MINOR: 3 consecutive identical failure_signatures, but the MIDDLE
    attempt is scoped "method" not "infra" -- the rule requires ALL THREE
    window members scoped infra, so a mixed-scope run must not fire
    infra_signature_repeated."""
    assessments = (
        _assessment(1, meets_target=False, failure_signature="infra_x", failure_scope="infra"),
        _assessment(2, meets_target=False, failure_signature="infra_x", failure_scope="method"),
        _assessment(3, meets_target=False, failure_signature="infra_x", failure_scope="infra"),
    )
    decision = _decide(assessments)
    assert decision.kind == "CONTINUE"
    assert decision.rule != "infra_signature_repeated"


def test_contradicted_unlabeled_lineages_never_count():
    """MINOR: two clean faithful+contradicted attempts with an EMPTY
    lineage_by_attempt (nobody labeled), and then with only ONE of the two
    labeled -- neither must satisfy the ">=2 DIFFERENT labeled lineages"
    rule. An unlabeled/unknown lineage must never be treated as "differing"
    from another unlabeled one."""
    two = (_assessment(1, **_CONTRADICTED_KWARGS), _assessment(2, **_CONTRADICTED_KWARGS))

    empty = _decide(two, lineage_by_attempt={})
    assert empty.kind == "CONTINUE"

    one_labeled = _decide(two, lineage_by_attempt={1: "champion"})
    assert one_labeled.kind == "CONTINUE"


def test_fingerprint_permutation_independent():
    """MINOR: the same KINDS/CLASSES content in a different list order,
    hashed alongside the SAME envelope mapping built with a different
    key-insertion order, must hash identically -- pins that the fingerprint
    sorts the kind/class sets before hashing and that dict key order never
    leaks into the digest."""
    kinds_a = ["protocol_gap", "render_artifact", "aggregation_gap"]
    kinds_b = ["aggregation_gap", "protocol_gap", "render_artifact"]
    classes_a = ["fabrication_suspected", "all_models_failed"]
    classes_b = ["all_models_failed", "fabrication_suspected"]

    envelope_forward = dict(_BASE_ENVELOPE)
    envelope_reversed = dict(reversed(list(_BASE_ENVELOPE.items())))
    assert list(envelope_forward.keys()) != list(envelope_reversed.keys())  # genuinely different insertion order

    fp_a = directives_fingerprint(
        seed_lineage="champion", scope_rung=0,
        repair_action_kinds=kinds_a, failure_classes=classes_a, envelope=envelope_forward,
    )
    fp_b = directives_fingerprint(
        seed_lineage="champion", scope_rung=0,
        repair_action_kinds=kinds_b, failure_classes=classes_b, envelope=envelope_reversed,
    )
    assert fp_a == fp_b
