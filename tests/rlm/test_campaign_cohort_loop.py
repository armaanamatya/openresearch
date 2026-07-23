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
    components = cell_out / "checkpoint_components"
    components.mkdir(exist_ok=True)
    for name in ("model", "optimizer", "lr_scheduler", "rng", "data_order"):
        (components / name).write_bytes(f"{name}-bytes".encode("utf-8"))
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

    # -- injected callables (bound methods) --------------------------------- #
    def validate_init(self) -> list:
        return []

    def understand(self) -> dict:
        return {"sha256": "understanding-sha", "blocking": []}

    def plan_attempt(self, state: CampaignState, rows: list) -> dict:
        n = state.next_attempt_n
        return {
            "attempt_n": n,
            "directives_sha256": f"dsha-{n}",
            "envelope": {"llm_usd": 1.0, "gpu_usd": 1.0, "gpu_hours": 0.1, "wall_s": 600.0},
            "project_id": state.project_id,
            "run_dir": str(self.run_dir / f"attempt_{n}"),
            "refusal": None,
            "downgrade_to_checkpoint": False,
            "launch_payload": {"attempt_n": n},
        }

    def launch(self, payload: dict) -> dict:
        # A per-branch run dir keyed to the branch id keeps two concurrent
        # cohort launches from clobbering each other's cell output.
        branch_id = payload["branch_id"]
        rung = payload["rung"]
        branch_run_dir = self.run_dir / f"branch_{branch_id}_rung_{rung}"
        cell_out = branch_run_dir / "code"
        _materialize_cell(cell_out, metric_value=_BRANCH_METRIC[branch_id])
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
