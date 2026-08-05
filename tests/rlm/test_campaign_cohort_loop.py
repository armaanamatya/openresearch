"""Focused unit for the receipt-gated ``_cohort_loop`` (Phase B3.2).

Drives a REAL ``SchedulerAuthorityController`` (3-branch faithful/ambiguity/
discovery portfolio, ``a100_cap=2``, ``rung_steps=(10, 50)`` -- mirrors the
canonical controller drive test's ``_spec``) from inside a ``ReproductionCampaign``
whose ``stages.launch``/``await_result``/``assess`` are CPU stubs that write the
exact on-disk evidence the harness receipt producer reads: per-branch
``metrics.json`` (so ``ambiguity`` underperforms) beside a planted
``final_report.score`` grade, a 5-file ``checkpoint_components/`` dir, and the
pinned dataset/run-spec manifests.

Asserts:
- ``_cohort_loop`` drives claim -> receipt -> decide_rung -> apply -> terminal;
- the low-metric branch (``ambiguity``) is FROZEN and the others PROMOTED;
- a ``revive`` re-queues the frozen branch;
- every receipt's metric came from ``metrics.json[metric_id]`` (never the
  planted ``0.01`` grade);
- byte-identical OFF: ``scheduler_controller=None`` dispatches ``_loop``, not
  ``_cohort_loop``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from backend.agents.rlm import cell_checkpoint
from backend.agents.rlm.campaign_directives import AttemptDirectives
from backend.agents.rlm.campaign_policy import AttemptEnvelope
from backend.agents.rlm.reproduction_campaign import (
    CampaignStages,
    CampaignState,
    ReproductionCampaign,
)
from backend.agents.rlm.scheduler_authority_controller import SchedulerAuthorityController
from backend.agents.rlm.scheduler_evidence import PaperStepLadder
from backend.agents.rlm.scheduler_runtime import (
    BranchTemplate,
    SchedulerAuthoritySpec,
)


def _f10(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _default_budget() -> dict:
    return {"max_llm_usd": 100.0, "max_gpu_usd": 100.0, "max_gpu_hours": 10.0, "max_attempts": 6}


def _zero_cost() -> dict:
    return {"llm_usd": 0.0, "gpu_usd": 0.0, "gpu_hours": 0.0, "wall_s": 0.0}


def _directives(project_id: str, paper_ref: str, attempt_n: int) -> AttemptDirectives:
    # A production-shaped AttemptDirectives -- the object the REAL driver
    # attribute-accesses. Required fields filled minimally; the rest None/
    # defaults. This is the regression fixture: the cohort loop must hand the
    # driver this OBJECT, never a dict (the stub read a dict, so the live
    # attribute-access crash was never exercised).
    return AttemptDirectives(
        attempt_n=attempt_n,
        project_id=project_id,
        paper_ref=paper_ref,
        enforcement={"cli_args": [], "env": {}},
        target_floor=None,
        understanding_ref=None,
        unresolved_warnings=(),
        leaf_repair_plan=None,
        improvement_notes=(),
        failure_capsules_ref=None,
        prior_evidence_ref=None,
        memory_hints=(),
        injected_lesson_signatures=(),
        scope_spec=None,
        scope_rung=0,
        seed_lineage="fresh",
        seed_pointer=None,
        envelope=AttemptEnvelope(
            llm_usd=1.0, gpu_usd=1.0, gpu_hours=0.1, wall_s=600.0, vm_ceiling_s=2400.0
        ),
        extra_guidance="",
        run_spec_path=None,
        fingerprint=f"dsha-{attempt_n}",
    )


def _spec() -> SchedulerAuthoritySpec:
    ladder = PaperStepLadder(
        paper_ref="2605.15155", metric_id="eval.accuracy", direction="maximize",
        r_max_steps=50, rung_steps=(10, 50), schedule_source_sha256="a" * 64,
    )
    return SchedulerAuthoritySpec(
        ladder=ladder,
        branches=(
            BranchTemplate("faithful", "faithful", _f10("faithful"), seed=1),
            BranchTemplate("ambiguity", "ambiguity", _f10("ambiguity"), seed=2),
            BranchTemplate("discovery", "discovery", _f10("discovery"), seed=3),
        ),
        gpu_usd_budget=100.0,
        discovery_gpu_usd_budget=100.0,
        a100_cap=2,
    )


# Per-branch ladder metric: ambiguity is the clear underperformer. The planted
# grade (final_report.score=0.01) is IDENTICAL for every branch and MUST never
# be the value the receipt records.
_BRANCH_METRIC = {"faithful": 0.90, "ambiguity": 0.10, "discovery": 0.50}


def _materialize_cell(cell_out: Path, *, metric_value: float) -> None:
    cell_out.mkdir(parents=True, exist_ok=True)
    (cell_out / "metrics.json").write_text(
        json.dumps({"eval.accuracy": metric_value, "final_report": {"score": 0.01}}),
        encoding="utf-8",
    )
    # Write the REAL trainer checkpoint layout: <cell_out>/checkpoints/step_<N>/
    # with the 5 components, exactly what gpu_cell_runner points
    # OPENRESEARCH_CELL_CHECKPOINT_DIR at and cell_checkpoint.latest_checkpoint_dir
    # resolves. The step number is cosmetic (the receipt's to_step comes from
    # the launch, not the checkpoint).
    cell_checkpoint.write_checkpoint(
        cell_out / "checkpoints",
        100,
        model=b"model-bytes",
        optimizer=b"optimizer-bytes",
        lr_scheduler=b"lr_scheduler-bytes",
        rng=b"rng-bytes",
        data_order=b"data_order-bytes",
    )
    (cell_out / "dataset_manifest.json").write_text('{"dataset":"pinned"}', encoding="utf-8")
    (cell_out / "run_spec.json").write_text('{"image":"pinned@sha256"}', encoding="utf-8")


class _CohortStages:
    """Stub stage set whose launch/await/assess materialize per-branch evidence
    under ``<branch_run_dir>/code`` and whose ``decide`` returns CONTINUE until
    both rungs resolve, then a terminal REPRODUCED."""

    def __init__(self, run_dir: Path, controller: Any = None) -> None:
        self.run_dir = Path(run_dir)
        self.controller = controller
        self.decide_calls = 0
        self.receipt_values: dict[str, list[float]] = {}
        # Every (branch_id, run_dir) the launch stage materialized -- used to
        # assert branches never collide on one run dir.
        self.launched_run_dirs: list[tuple[str, str]] = []

    # -- injected callables (bound methods) --------------------------------- #
    def validate_init(self) -> list:
        return []

    def understand(self) -> dict:
        return {"sha256": "understanding-sha", "blocking": []}

    def plan_attempt(self, state: CampaignState, rows: list) -> dict:
        n = state.next_attempt_n
        # launch_payload is a REAL AttemptDirectives (the production shape),
        # NOT a dict -- so the per-branch derivation + the driver hand-off are
        # exercised the way the live path runs them.
        return {
            "attempt_n": n,
            "directives_sha256": f"dsha-{n}",
            "envelope": {"llm_usd": 1.0, "gpu_usd": 1.0, "gpu_hours": 0.1, "wall_s": 600.0},
            "project_id": state.project_id,
            "run_dir": str(self.run_dir / f"attempt_{n}"),
            "refusal": None,
            "downgrade_to_checkpoint": False,
            "launch_payload": _directives(state.project_id, state.paper_ref, n),
        }

    def launch(self, directives: Any) -> dict:
        # REGRESSION GUARD: the cohort loop must hand the driver the per-branch
        # AttemptDirectives OBJECT, never the outer dict payload. Passing a dict
        # is exactly the live crash this test locks out.
        assert not isinstance(directives, dict) and hasattr(directives, "paper_ref"), (
            "cohort loop must pass an AttemptDirectives object to launch, not a dict"
        )
        # The branch id is recovered from the DISTINCT per-branch project_id the
        # cohort derived (``<project_id>__<sanitized_branch_id>``; the spec
        # branch ids are already [A-Za-z]+ so sanitize is identity). Materialize
        # each branch's cell under its own project_id-keyed run dir so
        # concurrent launches never collide.
        branch_id = str(directives.project_id).rsplit("__", 1)[-1]
        assert branch_id in _BRANCH_METRIC, f"unexpected branch id {branch_id!r}"
        branch_run_dir = self.run_dir / f"branch_{directives.project_id}"
        cell_out = branch_run_dir / "code"
        _materialize_cell(cell_out, metric_value=_BRANCH_METRIC[branch_id])
        self.launched_run_dirs.append((branch_id, str(branch_run_dir)))
        return {"pid": None, "run_dir": str(branch_run_dir), "lease_ref": None, "branch_id": branch_id}

    def await_result(self, handle: dict) -> dict:
        return {"run_dir": handle.get("run_dir"), "exit_condition": "exited:0"}

    def assess(self, raw_result: dict, payload: dict) -> dict:
        # Deterministic assessment: healthy exit, explicit per-branch GPU-$
        # (never read from a cost ledger). No divergence classifier trip.
        return {"score": 1.0, "cost": {**_zero_cost(), "gpu_usd": 1.0}}

    def assess_from_disk(self, in_flight: Any) -> dict:  # pragma: no cover - unused
        return {"score": 0.0, "cost": _zero_cost()}

    def quarantine(self, in_flight: Any) -> None:  # pragma: no cover - unused
        return None

    def distill(self, assessment: dict) -> None:
        return None

    def decide(self, state: CampaignState, rows: list) -> dict:
        self.decide_calls += 1
        # CONTINUE while any branch is still climbing (queued/launching/
        # awaiting_receipt), so authority is allowed to keep reordering the
        # cohort. Only once every branch has left the active pool (promoted to
        # the terminal rung + frozen/killed) does decide return a terminal
        # verdict -- terminal ALWAYS wins over the authority reordering.
        if self.controller is not None:
            active = any(
                b.state in {"queued", "launching", "awaiting_receipt"}
                for b in self.controller.branches.values()
            )
            if active:
                return {"kind": "CONTINUE", "rule": "climbing", "stop_reason": None, "next_plan": {}}
        return {"kind": "REPRODUCED", "rule": "r1", "stop_reason": None, "next_plan": None}

    def write_reports(self, state: CampaignState, rows: list, decision: dict) -> None:
        return None

    def liveness_probe(self, in_flight: Any) -> bool:  # pragma: no cover - unused
        return True

    def emit_event(self, name: str, payload: dict) -> None:
        return None

    def as_stages(self) -> CampaignStages:
        return CampaignStages(
            validate_init=self.validate_init,
            understand=self.understand,
            plan_attempt=self.plan_attempt,
            launch=self.launch,
            await_result=self.await_result,
            assess=self.assess,
            assess_from_disk=self.assess_from_disk,
            quarantine=self.quarantine,
            distill=self.distill,
            decide=self.decide,
            write_reports=self.write_reports,
            liveness_probe=self.liveness_probe,
            emit_event=self.emit_event,
        )


def _fresh_state(campaign: ReproductionCampaign) -> CampaignState:
    return CampaignState(
        project_id=campaign.project_id, paper_ref=campaign.paper_ref, state="attempt_loop",
        next_attempt_n=1, mode="unattended", driver="live_cli",
        budget=_default_budget(), spent=_zero_cost(), scope_rung=0,
        in_flight=None, understanding_sha256="understanding-sha", rubric_sha256=None,
        steering_cursor=0, pending_approval=None, warnings=[], terminal=None,
        created_at=1.0, updated_at=1.0,
    )


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #


def test_cohort_loop_drives_claim_receipt_decide_apply(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    controller = SchedulerAuthorityController(
        run_dir, campaign_id="proj_1", spec=_spec(),
    )
    assert controller.bootstrap() == 3
    stages_obj = _CohortStages(run_dir, controller=controller)

    campaign = ReproductionCampaign(
        run_dir=run_dir, project_id="proj_1", paper_ref="2605.15155",
        budget=_default_budget(), mode="unattended", driver="live_cli",
        stages=stages_obj.as_stages(), scheduler_controller=controller,
    )
    (run_dir / "campaign").mkdir(parents=True, exist_ok=True)

    state = _fresh_state(campaign)
    terminal = campaign._cohort_loop(state)

    # Terminal deterministic decision won (authority never flips it).
    assert terminal["kind"] == "REPRODUCED"

    # Authority applied: ambiguity underperformed and was FROZEN; the other two
    # were promoted (they climbed off rung 0 and left the frozen pool).
    assert controller.branches["ambiguity"].state == "frozen"
    assert controller.branches["faithful"].state != "frozen"
    assert controller.branches["discovery"].state != "frozen"

    # The freeze is reversible: revive re-queues the frozen branch.
    revived = campaign.scheduler_controller.revive("ambiguity")
    assert revived.state == "queued"
    assert controller.branches["ambiguity"].state == "queued"

    # The campaign's fail-closed ledger attested every verified receipt: each
    # attest row carries status="scheduler_receipt" + the branch id + receipt
    # sha (proving the ``attest`` callback was the real ledger appender, not a
    # worker-authored write).
    rows = campaign.ledger.read_rows()
    attest_rows = [r for r in rows if r.get("status") == "scheduler_receipt"]
    assert {r["branch_id"] for r in attest_rows} >= {"faithful", "ambiguity", "discovery"}

    # Every persisted receipt's metric came from metrics.json[metric_id], NEVER
    # the planted 0.01 grade. Read the verified receipt files on disk.
    receipt_dir = run_dir / "campaign" / "scheduler_receipts"
    seen: dict[str, float] = {}
    for path in receipt_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        bid = payload["branch_id"]
        value = payload["metric"]["value"]
        assert value != 0.01, "receipt metric must be metrics.json, never the grade"
        # keep the rung-0 (from_step 0) receipt for the per-branch ladder assert
        if payload["from_step"] == 0:
            seen[bid] = value
    assert seen["ambiguity"] == 0.10
    assert seen["faithful"] == 0.90
    assert seen["discovery"] == 0.50

    # Branches never collided on one run dir: each distinct branch launched into
    # a DISTINCT run dir (the per-branch ``__<branch>`` project_id derivation).
    by_branch: dict[str, set[str]] = {}
    for bid, rd in stages_obj.launched_run_dirs:
        by_branch.setdefault(bid, set()).add(rd)
    for bid in ("faithful", "ambiguity", "discovery"):
        assert by_branch.get(bid), f"branch {bid} never launched"
    all_dirs = [rd for _bid, rd in stages_obj.launched_run_dirs]
    # No two DIFFERENT branches share a run dir (a branch may relaunch at a
    # higher rung into its own dir, so compare across branches only).
    for a_bid, a_rd in stages_obj.launched_run_dirs:
        for b_bid, b_rd in stages_obj.launched_run_dirs:
            if a_bid != b_bid:
                assert a_rd != b_rd, f"branches {a_bid}/{b_bid} collided on {a_rd}"
    assert len(all_dirs) >= 3


def test_off_dispatches_serial_loop_byte_identical(tmp_path, monkeypatch):
    # scheduler_controller=None => the dispatch guard picks the serial _loop,
    # never _cohort_loop (byte-identical OFF).
    run_dir = tmp_path / "run"
    stages_obj = _CohortStages(run_dir)
    campaign = ReproductionCampaign(
        run_dir=run_dir, project_id="proj_1", paper_ref="2605.15155",
        budget=_default_budget(), mode="unattended", driver="live_cli",
        stages=stages_obj.as_stages(), scheduler_controller=None,
    )

    picked: list[str] = []
    real_loop = campaign._loop
    real_cohort = campaign._cohort_loop

    def _spy_loop(state, *, allow_supersede=False):
        picked.append("_loop")
        return {"kind": "REPRODUCED"}

    def _spy_cohort(state, *, allow_supersede=False):  # pragma: no cover - must not run
        picked.append("_cohort_loop")
        return {"kind": "REPRODUCED"}

    monkeypatch.setattr(campaign, "_loop", _spy_loop)
    monkeypatch.setattr(campaign, "_cohort_loop", _spy_cohort)

    campaign._drive(_fresh_state(campaign))
    assert picked == ["_loop"]
    # sanity: the real bound methods still exist (dispatch is the only change).
    assert callable(real_loop) and callable(real_cohort)


def test_run_entrypoint_reaches_cohort_loop_under_authority(tmp_path):
    """Phase B exit bar: a REAL campaign ``run()`` under a live authority
    controller routes through the campaign's own entrypoint
    (``_run_body`` -> ``_run_fresh`` -> ``_drive`` -> ``_cohort_loop``) -- NOT a
    throwaway harness -- and drives freeze/promote from VERIFIED on-disk
    receipts to a terminal the deterministic decision (never authority)
    renders. This is the same code path a flags-ON campaign takes; the
    both-flags+spec construction is covered by test_authority_controller_wiring."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    controller = SchedulerAuthorityController(run_dir, campaign_id="proj_1", spec=_spec())
    assert controller.bootstrap() == 3
    stages_obj = _CohortStages(run_dir, controller=controller)
    campaign = ReproductionCampaign(
        run_dir=run_dir, project_id="proj_1", paper_ref="2605.15155",
        budget=_default_budget(), mode="unattended", driver="live_cli",
        stages=stages_obj.as_stages(), scheduler_controller=controller,
    )
    (run_dir / "campaign").mkdir(parents=True, exist_ok=True)

    # Full entrypoint -- not _cohort_loop directly.
    terminal = campaign.run()
    assert terminal["kind"] == "REPRODUCED"

    # Authority applied THROUGH the campaign entrypoint: the underperformer was
    # frozen (reversible), the others promoted.
    assert controller.branches["ambiguity"].state == "frozen"
    assert controller.branches["faithful"].state != "frozen"
    assert controller.branches["discovery"].state != "frozen"

    # A frozen branch is revivable at its checkpoint (the revive substrate).
    revived = controller.revive("ambiguity")
    assert revived.state == "queued"

    # Every verified receipt persisted on disk carries the deterministic metric,
    # NEVER the planted 0.01 grade -- the evidence-not-grade red line, end to end
    # from the campaign entrypoint.
    receipts = sorted((run_dir / "campaign" / "scheduler_receipts").glob("*.json"))
    assert receipts, "an authority run must persist verified receipts"
    for rp in receipts:
        payload = json.loads(rp.read_text(encoding="utf-8"))
        assert payload["metric"]["value"] != 0.01


# --------------------------------------------------------------------------- #
# Rung-step -> trainer iteration-budget wire (Edit 1)                          #
#                                                                             #
# The per-branch launch payload must carry the ladder ``to_step`` into the    #
# child ``reproduce`` subprocess's env as ``OPENRESEARCH_CELL_ITER_BUDGET``   #
# (and ``_FROM`` = ``from_step``) via ``enforcement["env"]`` -- the existing  #
# ``attempt_driver.build_attempt_env`` forward path -- so the real trainer's  #
# cell ``iters`` is capped to the rung. Without it the child ignores the      #
# ladder and runs its full ``cells.json`` iters, and successive-halving-at-   #
# rungs never actually happens on real cells.                                  #
# --------------------------------------------------------------------------- #
from backend.agents.rlm.reproduction_campaign import (  # noqa: E402
    _branch_iter_budget_enforcement,
)
from backend.agents.rlm.scheduler_authority_controller import (  # noqa: E402
    SchedulerLaunch,
)


def _launch(branch_id: str, *, from_step: int, to_step: int,
            resume: str | None = None) -> SchedulerLaunch:
    return SchedulerLaunch(
        branch_id=branch_id,
        branch_type="faithful",
        hypothesis_fingerprint=_f10(branch_id),
        rung=1,
        from_step=from_step,
        to_step=to_step,
        seed=1,
        parent_branch_id=None,
        resume_checkpoint_path=resume,
        is_safety_bracket=False,
    )


def test_branch_iter_budget_enforcement_carries_to_step():
    # The factored env-merge: to_step -> OPENRESEARCH_CELL_ITER_BUDGET,
    # from_step -> OPENRESEARCH_CELL_ITER_FROM, merged onto existing env.
    base = {"cli_args": [("--x", "1")], "env": {"KEEP": "yes"}}
    launch = _launch("faithful", from_step=10, to_step=50)
    merged = _branch_iter_budget_enforcement(base, launch)
    env = merged["env"]
    assert env["OPENRESEARCH_CELL_ITER_BUDGET"] == "50"
    assert env["OPENRESEARCH_CELL_ITER_FROM"] == "10"
    # Existing env keys + non-env enforcement keys are preserved untouched.
    assert env["KEEP"] == "yes"
    assert merged["cli_args"] == [("--x", "1")]
    # The base enforcement mapping is not mutated in place.
    assert "OPENRESEARCH_CELL_ITER_BUDGET" not in base["env"]


def test_branch_iter_budget_enforcement_handles_missing_env_key():
    # A base enforcement with no ``env`` key still gets one carrying the budget.
    merged = _branch_iter_budget_enforcement({"cli_args": []}, _launch("d", from_step=0, to_step=10))
    assert merged["env"]["OPENRESEARCH_CELL_ITER_BUDGET"] == "10"
    assert merged["env"]["OPENRESEARCH_CELL_ITER_FROM"] == "0"


def test_cohort_launch_payload_threads_budget_into_branch_enforcement(tmp_path):
    # End to end through the real ``_cohort_launch_payload``: the per-branch
    # AttemptDirectives the driver receives carries the budget in its
    # enforcement env, and the base directives' env keys survive.
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    campaign = ReproductionCampaign(
        run_dir=run_dir, project_id="proj_1", paper_ref="2605.15155",
        budget=_default_budget(), mode="unattended", driver="live_cli",
        stages=_CohortStages(run_dir).as_stages(), scheduler_controller=None,
    )
    base_directives = AttemptDirectives(
        attempt_n=1,
        project_id="proj_1",
        paper_ref="2605.15155",
        enforcement={"cli_args": [], "env": {"PRESERVED": "1"}},
        target_floor=None,
        understanding_ref=None,
        unresolved_warnings=(),
        leaf_repair_plan=None,
        improvement_notes=(),
        failure_capsules_ref=None,
        prior_evidence_ref=None,
        memory_hints=(),
        injected_lesson_signatures=(),
        scope_spec=None,
        scope_rung=0,
        seed_lineage="fresh",
        seed_pointer=None,
        envelope=AttemptEnvelope(
            llm_usd=1.0, gpu_usd=1.0, gpu_hours=0.1, wall_s=600.0, vm_ceiling_s=2400.0
        ),
        extra_guidance="",
        run_spec_path=None,
        fingerprint="dsha-1",
    )
    planned = {"project_id": "proj_1", "launch_payload": base_directives}
    launch = _launch("faithful", from_step=10, to_step=50)
    payload = campaign._cohort_launch_payload(planned, launch)

    branch_lp = payload["launch_payload"]
    env = dict(branch_lp.enforcement["env"])
    assert env["OPENRESEARCH_CELL_ITER_BUDGET"] == "50"
    assert env["OPENRESEARCH_CELL_ITER_FROM"] == "10"
    assert env["PRESERVED"] == "1"
    # The base directives object was NOT mutated (per-branch replace only).
    assert "OPENRESEARCH_CELL_ITER_BUDGET" not in dict(base_directives.enforcement["env"])


def test_degrade_to_campaign_error_persists_traceback(tmp_path):
    # A campaign that degrades to a bare ``campaign_error:<ClassName>`` stop
    # reason must leave the FULL traceback on disk. Without it an
    # authority/dispatch-seam failure (e.g. the fail-closed no-checkpoint raise
    # at ``_cohort_loop`` line ~1118) is undiagnosable post-mortem -- the
    # summary line names only the exception class. This is exactly how the
    # first ResNet Tree-B GPU launch surfaced its driver mismatch.
    from backend.agents.rlm.scheduler_runtime import SchedulerRuntimeError

    run_dir = tmp_path / "run"
    campaign = ReproductionCampaign(
        run_dir=run_dir, project_id="proj_1", paper_ref="2605.15155",
        budget=_default_budget(), mode="unattended", driver="live_cli",
        stages=_CohortStages(run_dir).as_stages(), scheduler_controller=None,
    )
    (run_dir / "campaign").mkdir(parents=True, exist_ok=True)

    # Call the degrade path from inside a REAL except block -- mirrors run()'s
    # ``except Exception as exc: return self._degrade_to_campaign_error(exc)``
    # -- so ``traceback.format_exc()`` captures the live traceback. The report
    # write inside the degrade path may fail on the stub stages; we assert the
    # traceback file only (it is written BEFORE any report work).
    try:
        raise SchedulerRuntimeError(
            "cohort branch produced no resumable checkpoint -- refusing to fabricate a receipt"
        )
    except SchedulerRuntimeError as exc:
        try:
            campaign._degrade_to_campaign_error(exc)
        except Exception:  # noqa: BLE001 -- report writing is not under test here
            pass

    tb_path = run_dir / "campaign" / "campaign_error_traceback.txt"
    assert tb_path.exists(), "campaign_error traceback must be persisted to disk"
    text = tb_path.read_text(encoding="utf-8")
    assert "Traceback" in text
    assert "SchedulerRuntimeError" in text
    assert "no resumable checkpoint" in text
