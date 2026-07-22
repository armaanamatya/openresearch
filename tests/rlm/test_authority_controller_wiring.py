"""Wiring tests for the SchedulerAuthorityController composition-root gate
(Phase B2.1 + B2.2).

The controller is constructed in ``campaign_composition.build_campaign`` ONLY
when BOTH ``OPENRESEARCH_SCHEDULER_TREE`` and
``OPENRESEARCH_SCHEDULER_AUTHORITATIVE`` are on AND ``authority_spec_path`` is
present. Constructing it side-effects the run dir at ``__init__`` (writes
``campaign/scheduler_ladder.json`` + ``campaign/scheduler_tree_state.json``),
so the RED LINE is byte-identical-OFF: any gate condition false ⇒ construct
NOTHING (no controller, no files).

Hermetic: tmp_path-only, no sockets, no LLM calls, no subprocess launches —
the campaign is constructed (not run), and the spec JSON is emitted to a tmp
path whose ladder paper_ref matches the campaign's ``_opts`` paper_ref.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from backend.agents.rlm.campaign_composition import CampaignOptions, build_campaign
from backend.agents.rlm.scheduler_authority_controller import SchedulerAuthorityController

_PAPER_REF = "2605.15155"
_LADDER_STATE_FILES = ("scheduler_ladder.json", "scheduler_tree_state.json")


@pytest.fixture(autouse=True)
def _clean_scheduler_env(monkeypatch):
    """Both scheduler flags start unset; a full env snapshot/restore guards the
    RAW ``os.environ`` writes ``_apply_profile_env`` performs (mirrors the
    composition suite's own fixture)."""
    monkeypatch.delenv("OPENRESEARCH_SCHEDULER_TREE", raising=False)
    monkeypatch.delenv("OPENRESEARCH_SCHEDULER_AUTHORITATIVE", raising=False)
    _snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(_snapshot)


def _write_profile(path: Path) -> Path:
    """A minimal run-spec profile whose PATH avoids the clean-context markers."""
    path.write_text(json.dumps({"OPENRESEARCH_REUSE_RUBRIC": "1"}), encoding="utf-8")
    return path


def _write_authority_spec(path: Path, *, paper_ref: str = _PAPER_REF) -> Path:
    """Emit a valid authority-spec JSON: ONE faithful, non-safety branch so the
    conditional ``safety_gpu_usd_budget``/``discovery_gpu_usd_budget`` validators
    never fire — ``width`` needs only ``gpu_usd_budget`` + ``a100_cap``."""
    spec = {
        "schema_version": 1,
        "ladder": {
            "paper_ref": paper_ref,
            "metric_id": "success_rate",
            "direction": "maximize",
            "r_max_steps": 50,
            "rung_steps": [10, 50],
            "schedule_source_sha256": "a" * 64,
        },
        "branches": [
            {
                "branch_id": "faithful",
                "branch_type": "faithful",
                "hypothesis_fingerprint": "b" * 64,
                "seed": 1,
                "parent_branch_id": None,
                "is_safety_bracket": False,
            }
        ],
        "width": {"gpu_usd_budget": 1.0, "a100_cap": 2},
    }
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def _opts(tmp_path: Path, **overrides) -> CampaignOptions:
    base = dict(
        paper_ref=_PAPER_REF,
        runs_root=tmp_path / "runs",
        repo_root=tmp_path,
        max_llm_usd=100.0,
        max_gpu_usd=100.0,
        max_gpu_hours=10.0,
        max_attempts=6,
        wall_clock_s=None,
        mode="unattended",
        driver="live",
        width=1,
        plateau_k=2,
        sandbox="local",
        billing_sandbox=None,
        gpu_usd_per_hr=None,
        est_gpu_hours=2.0,
        run_spec_path=str(_write_profile(tmp_path / "profile.json")),
        scope_spec=None,
        scope_ladder=("full",),
        paper_hint=None,
        require_cpu_tier=False,
        arxiv_id=_PAPER_REF,
        paper_class="generic",
        resume=False,
    )
    base.update(overrides)
    return CampaignOptions(**base)


def _run_dir(opts: CampaignOptions, project_id: str) -> Path:
    return opts.runs_root / project_id


def _no_scheduler_side_effects(run_dir: Path) -> bool:
    """No ladder/state file was written under campaign/ — the controller (and
    thus its runtime __init__) never constructed."""
    campaign_dir = run_dir / "campaign"
    return all(not (campaign_dir / name).exists() for name in _LADDER_STATE_FILES)


# --------------------------------------------------------------------------- #
# OFF — byte-identical: no controller, no run-dir side effects                 #
# --------------------------------------------------------------------------- #


def test_off_constructs_no_controller(tmp_path, monkeypatch):
    """Flags unset AND no spec ⇒ .scheduler_controller is None AND the run dir
    has NO scheduler_ladder.json / scheduler_tree_state.json. This is the
    byte-identical-OFF proof: construction never happened."""
    # Both flags unset (fixture already delenv'd), authority_spec_path=None.
    opts = _opts(tmp_path, authority_spec_path=None)

    campaign = build_campaign("prj_off", opts)

    assert campaign.scheduler_controller is None
    assert _no_scheduler_side_effects(_run_dir(opts, "prj_off"))


def test_flags_on_but_no_spec_no_side_effects(tmp_path, monkeypatch):
    """Even with both flags ON, an absent spec keeps the OFF invariant: the
    ``and opts.authority_spec_path`` gate short-circuits before any construct."""
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "1")
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_AUTHORITATIVE", "1")
    opts = _opts(tmp_path, authority_spec_path=None)

    campaign = build_campaign("prj_on_nospec", opts)

    assert campaign.scheduler_controller is None
    assert _no_scheduler_side_effects(_run_dir(opts, "prj_on_nospec"))


def test_spec_present_but_flags_off_no_side_effects(tmp_path, monkeypatch):
    """A spec path with the flags OFF must also construct nothing — every gate
    condition is ANDed, so any single false leaves the run dir untouched."""
    spec_path = _write_authority_spec(tmp_path / "authority_spec.json")
    opts = _opts(tmp_path, authority_spec_path=str(spec_path))

    campaign = build_campaign("prj_spec_off", opts)

    assert campaign.scheduler_controller is None
    assert _no_scheduler_side_effects(_run_dir(opts, "prj_spec_off"))


# --------------------------------------------------------------------------- #
# ON — both flags + a valid spec construct the controller                      #
# --------------------------------------------------------------------------- #


def test_on_with_spec_constructs_controller(tmp_path, monkeypatch):
    """Both flags set AND a valid authority_spec_path ⇒ .scheduler_controller
    is a SchedulerAuthorityController bound to project_id, and bootstrap()
    registered the spec's declared branch."""
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "1")
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_AUTHORITATIVE", "1")
    spec_path = _write_authority_spec(tmp_path / "authority_spec.json")
    opts = _opts(tmp_path, authority_spec_path=str(spec_path))

    campaign = build_campaign("prj_on", opts)

    controller = campaign.scheduler_controller
    assert isinstance(controller, SchedulerAuthorityController)
    # campaign_id lives on the runtime (the controller has no campaign_id attr).
    assert controller.runtime.campaign_id == "prj_on"
    # bootstrap() ran in build_campaign — the declared branch is registered.
    assert "faithful" in controller.branches
    # The construction did side-effect the run dir (the ON complement of the
    # byte-identical-OFF proof).
    campaign_dir = _run_dir(opts, "prj_on") / "campaign"
    assert (campaign_dir / "scheduler_ladder.json").exists()
    assert (campaign_dir / "scheduler_tree_state.json").exists()


def test_on_without_spec_no_controller(tmp_path, monkeypatch):
    """Both flags set but authority_spec_path=None ⇒ no controller, no crash
    (the spec gate is required even under both flags)."""
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "1")
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_AUTHORITATIVE", "1")
    opts = _opts(tmp_path, authority_spec_path=None)

    campaign = build_campaign("prj_on_none", opts)

    assert campaign.scheduler_controller is None
