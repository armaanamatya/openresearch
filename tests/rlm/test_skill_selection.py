"""Tests for relevance-gated, agent-selected skill activation
(OPENRESEARCH_SKILL_SELECT, 2026-07-06).

Covers the selection module (deterministic recall reuse + bounded LLM prune +
persistence + consumption helpers) and its three wiring sites: the
detect_environment trigger, the consult_skill index focus, and the verifier
grader-prompt injection.

Hermetic: real vendored catalog throughout; every LLM path uses a scripted
FakeLlmClient / capturing stub — no network. OFF-state assertions prove
byte-identical behaviour; ON-state assertions prove the new path fires.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from backend.agents.rlm import skill_selection as sel
from backend.agents.rlm.primitives import (
    PRIMITIVE_REGISTRY,
    consult_skill,
    detect_environment,
)
from backend.agents.rlm.skill_catalog import load_catalog
from backend.evals.paperbench.leaf_scorer import score_reproduction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sdar_claim_map() -> dict:
    return {
        "core_contribution": "GRPO reinforcement learning for agentic search reasoning (OPSD)",
        "model_architecture": "Qwen2.5-3B policy trained with GRPO",
        "training_recipe": {"optimizer": "AdamW"},
        "datasets": [{"name": "Search-QA"}, {"name": "HotpotQA"}],
        "metrics": [{"name": "success_rate"}],
        "claims": [{"method": "GRPO", "dataset": "Search-QA", "metric": "reward"}],
    }


def _sdar_env() -> dict:
    return {"framework": "pytorch", "pip_packages": {"vllm": "0.6.3", "verl": "0.1", "trl": "0.14"}}


class _ScriptedClient:
    """Returns one scripted response to every .complete() call."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self.response


class _RaisingClient:
    def complete(self, *, system: str, user: str) -> str:
        raise RuntimeError("transport down")


_TINY_TREE = {
    "id": "root",
    "requirements": "root",
    "weight": 1,
    "sub_tasks": [
        {"id": "leaf-a", "requirements": "leaf a", "weight": 1, "sub_tasks": []},
        {"id": "leaf-b", "requirements": "leaf b", "weight": 1, "sub_tasks": []},
    ],
}


class _CapturingGrader:
    """Records grader user prompts; returns a full-credit batch response."""

    def __init__(self) -> None:
        self.users: list[str] = []

    def complete(self, *, system: str, user: str) -> str:
        self.users.append(user)
        return json.dumps([
            {"leaf_id": "leaf-a", "score": 1.0, "justification": "ok"},
            {"leaf_id": "leaf-b", "score": 1.0, "justification": "ok"},
        ])


def _honest_run_dir(tmp: str) -> Path:
    run_dir = Path(tmp)
    (run_dir / "final_report.json").write_text(
        json.dumps({"reproduction_summary": "r", "baseline_metrics": {"success_rate": 0.5}}),
        encoding="utf-8",
    )
    return run_dir


# ---------------------------------------------------------------------------
# Flag gating — select_enabled requires BOTH flags
# ---------------------------------------------------------------------------

def test_select_enabled_requires_both_flags(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_SKILLS", raising=False)
    monkeypatch.delenv("OPENRESEARCH_SKILL_SELECT", raising=False)
    assert sel.select_enabled() is False

    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    assert sel.select_enabled() is False  # SELECT still off

    monkeypatch.setenv("OPENRESEARCH_SKILL_SELECT", "1")
    assert sel.select_enabled() is True

    monkeypatch.delenv("OPENRESEARCH_SKILLS", raising=False)
    assert sel.select_enabled() is False  # SKILLS master off dominates


# ---------------------------------------------------------------------------
# Deterministic recall (reuses skill_matcher; widened by pip/system packages)
# ---------------------------------------------------------------------------

def test_match_candidate_skills_recalls_sdar_domains():
    catalog = load_catalog()
    cands = sel.match_candidate_skills(_sdar_claim_map(), _sdar_env(), catalog)
    names = {c["name"] for c in cands}
    # The RL/GRPO training skill and the verl trainer must be recalled from the
    # claim map; the vLLM serving skill must be recalled from pip_packages (the
    # recall-widening fix — the shared matcher only reads env["framework"]).
    assert "grpo-rl-training" in names
    assert "verl-rl-training" in names
    assert "serving-llms-vllm" in names
    # Every candidate carries the shaped fields.
    for c in cands:
        assert set(c) >= {"name", "category", "description", "reason"}


def test_match_candidate_skills_deterministic():
    catalog = load_catalog()
    a = sel.match_candidate_skills(_sdar_claim_map(), _sdar_env(), catalog)
    b = sel.match_candidate_skills(_sdar_claim_map(), _sdar_env(), catalog)
    assert [c["name"] for c in a] == [c["name"] for c in b]


def test_match_candidate_skills_respects_candidates_max():
    catalog = load_catalog()
    cands = sel.match_candidate_skills(_sdar_claim_map(), _sdar_env(), catalog, candidates_max=3)
    assert len(cands) <= 3


# ---------------------------------------------------------------------------
# Selection orchestration — deterministic / LLM-prune / fallback
# ---------------------------------------------------------------------------

def test_select_active_skills_deterministic_only(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILL_SELECT_DETERMINISTIC", "1")
    catalog = load_catalog()
    # A raising client would break the run if consulted — deterministic mode must
    # never consult it.
    art = sel.select_active_skills(_sdar_claim_map(), _sdar_env(), catalog, llm_client=_RaisingClient())
    assert art["selector"] == "deterministic"
    assert art["selected"] == [c["name"] for c in art["candidates"]]
    assert "grpo-rl-training" in art["selected"]


def test_select_active_skills_llm_prune_picks_subset():
    catalog = load_catalog()
    client = _ScriptedClient(json.dumps({
        "selected": [
            {"name": "grpo-rl-training", "reason": "paper trains the policy with GRPO"},
            {"name": "serving-llms-vllm", "reason": "vLLM serves rollouts"},
        ]
    }))
    art = sel.select_active_skills(_sdar_claim_map(), _sdar_env(), catalog, llm_client=client)
    assert art["selector"] == "deterministic+llm"
    assert art["selected"] == ["grpo-rl-training", "serving-llms-vllm"]
    assert art["reasons"]["grpo-rl-training"] == "paper trains the policy with GRPO"
    assert client.calls, "the LLM prune must have been consulted"


def test_select_active_skills_llm_prune_preserves_candidate_order():
    catalog = load_catalog()
    # Return the two names in REVERSE candidate order; selected must follow the
    # deterministic candidate order, not the LLM's listing order.
    client = _ScriptedClient(json.dumps({
        "selected": [{"name": "serving-llms-vllm"}, {"name": "grpo-rl-training"}]
    }))
    art = sel.select_active_skills(_sdar_claim_map(), _sdar_env(), catalog, llm_client=client)
    cand_order = [c["name"] for c in art["candidates"]]
    assert art["selected"] == [n for n in cand_order if n in {"grpo-rl-training", "serving-llms-vllm"}]


def test_select_active_skills_falls_back_on_unparseable(monkeypatch):
    catalog = load_catalog()
    art = sel.select_active_skills(
        _sdar_claim_map(), _sdar_env(), catalog, llm_client=_ScriptedClient("not json at all")
    )
    assert art["selector"] == "deterministic"
    assert art["selected"] == [c["name"] for c in art["candidates"]]


def test_select_active_skills_falls_back_on_client_error():
    catalog = load_catalog()
    art = sel.select_active_skills(_sdar_claim_map(), _sdar_env(), catalog, llm_client=_RaisingClient())
    assert art["selector"] == "deterministic"
    assert art["selected"]  # non-empty deterministic floor


def test_llm_prune_restricts_to_candidate_names():
    catalog = load_catalog()
    # A hallucinated name not in the candidate list must be dropped.
    client = _ScriptedClient(json.dumps({
        "selected": [{"name": "totally-made-up-skill"}, {"name": "grpo-rl-training"}]
    }))
    art = sel.select_active_skills(_sdar_claim_map(), _sdar_env(), catalog, llm_client=client)
    assert "totally-made-up-skill" not in art["selected"]
    assert art["selected"] == ["grpo-rl-training"]
    assert art["selector"] == "deterministic+llm"


def test_select_active_skills_empty_subject_matter_never_raises():
    catalog = load_catalog()
    art = sel.select_active_skills({}, {}, catalog, llm_client=None)
    assert art["selected"] == []
    assert art["candidates"] == []
    assert art["selector"] == "deterministic"


def test_select_active_skills_empty_catalog():
    art = sel.select_active_skills(_sdar_claim_map(), _sdar_env(), {}, llm_client=None)
    assert art["selected"] == []


# ---------------------------------------------------------------------------
# Persistence roundtrip
# ---------------------------------------------------------------------------

def test_write_load_roundtrip(tmp_path):
    catalog = load_catalog()
    art = sel.select_active_skills(_sdar_claim_map(), _sdar_env(), catalog, llm_client=None)
    sel.write_active_skills(tmp_path, art)
    assert (tmp_path / "rlm_state" / "active_skills.json").is_file()
    loaded = sel.load_active_skills(tmp_path)
    assert loaded is not None
    assert loaded["selected"] == art["selected"]
    assert loaded["selector"] == art["selector"]


def test_load_active_skills_absent_returns_none(tmp_path):
    assert sel.load_active_skills(tmp_path) is None


# ---------------------------------------------------------------------------
# Verifier context builder
# ---------------------------------------------------------------------------

def test_build_verifier_skill_context_bounded():
    catalog = load_catalog()
    art = sel.select_active_skills(_sdar_claim_map(), _sdar_env(), catalog, llm_client=None)
    ctx_str = sel.build_verifier_skill_context(art, catalog, max_bodies=1)
    assert ctx_str is not None
    # Advisory framing + at least one selected name present.
    assert "Skill playbooks relevant to this paper" in ctx_str
    assert "score ONLY from the actual code" in ctx_str
    assert "grpo-rl-training" in ctx_str
    # At most `max_bodies` inlined playbook bodies.
    assert ctx_str.count("### Playbook:") <= 1


def test_build_verifier_skill_context_zero_bodies():
    catalog = load_catalog()
    art = sel.select_active_skills(_sdar_claim_map(), _sdar_env(), catalog, llm_client=None)
    ctx_str = sel.build_verifier_skill_context(art, catalog, max_bodies=0)
    assert ctx_str is not None
    assert "### Playbook:" not in ctx_str  # descriptions only


def test_build_verifier_skill_context_empty_artifact_returns_none():
    catalog = load_catalog()
    assert sel.build_verifier_skill_context({"selected": []}, catalog) is None


# ---------------------------------------------------------------------------
# Wiring 1 — detect_environment trigger (ON / OFF / idempotent)
# ---------------------------------------------------------------------------

def test_detect_environment_writes_active_skills_when_select_on(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    monkeypatch.setenv("OPENRESEARCH_SKILL_SELECT", "1")
    ctx = make_context(tmp_path)
    detect_environment(_sdar_claim_map(), ctx=ctx)

    active_path = ctx.project_dir / "rlm_state" / "active_skills.json"
    assert active_path.is_file()
    art = json.loads(active_path.read_text(encoding="utf-8"))
    assert art["selected"], "an SDAR-like paper must select at least one skill"
    assert "grpo-rl-training" in art["selected"]
    # The skills_selected event was emitted to the dashboard log.
    events = (ctx.project_dir / "dashboard_events.jsonl").read_text(encoding="utf-8")
    assert "skills_selected" in events


def test_detect_environment_no_active_skills_when_select_off(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    monkeypatch.delenv("OPENRESEARCH_SKILL_SELECT", raising=False)
    ctx = make_context(tmp_path)
    result = detect_environment(_sdar_claim_map(), ctx=ctx)

    # Deterministic skill_match (SKILLS layer) still runs; the SELECT layer does not.
    assert "skill_match" in (result.get("extra") or {})
    assert not (ctx.project_dir / "rlm_state" / "active_skills.json").exists()


def test_detect_environment_no_active_skills_when_skills_off(make_context, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_SKILLS", raising=False)
    monkeypatch.setenv("OPENRESEARCH_SKILL_SELECT", "1")  # master off dominates
    ctx = make_context(tmp_path)
    detect_environment(_sdar_claim_map(), ctx=ctx)
    assert not (ctx.project_dir / "rlm_state" / "active_skills.json").exists()


def test_detect_environment_selection_is_idempotent(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    monkeypatch.setenv("OPENRESEARCH_SKILL_SELECT", "1")
    ctx = make_context(tmp_path)
    state_dir = ctx.project_dir / "rlm_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    sentinel = {"selected": ["sentinel-skill"], "selector": "preexisting"}
    (state_dir / "active_skills.json").write_text(json.dumps(sentinel), encoding="utf-8")

    detect_environment(_sdar_claim_map(), ctx=ctx)

    # Pre-existing artifact must survive untouched (no redundant LLM call / overwrite).
    after = json.loads((state_dir / "active_skills.json").read_text(encoding="utf-8"))
    assert after == sentinel


# ---------------------------------------------------------------------------
# Wiring 2 — consult_skill index surfaces the active set (ON / OFF)
# ---------------------------------------------------------------------------

def test_consult_skill_index_surfaces_active_set(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    monkeypatch.setenv("OPENRESEARCH_SKILL_SELECT", "1")
    ctx = make_context(tmp_path)
    catalog = load_catalog()
    art = sel.select_active_skills(_sdar_claim_map(), _sdar_env(), catalog, llm_client=None)
    sel.write_active_skills(ctx.project_dir, art)

    result = consult_skill(ctx=ctx)
    assert result["kind"] == "index"
    assert "active" in result
    active_names = {e["name"] for e in result["active"]}
    assert "grpo-rl-training" in active_names
    assert result["active_note"]


def test_consult_skill_index_no_active_when_select_off(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    monkeypatch.delenv("OPENRESEARCH_SKILL_SELECT", raising=False)
    ctx = make_context(tmp_path)
    catalog = load_catalog()
    # Even if an artifact exists on disk, SELECT off => the index is byte-identical.
    sel.write_active_skills(
        ctx.project_dir,
        sel.select_active_skills(_sdar_claim_map(), _sdar_env(), catalog, llm_client=None),
    )
    result = consult_skill(ctx=ctx)
    assert result["kind"] == "index"
    assert "active" not in result


# ---------------------------------------------------------------------------
# Wiring 3 — verifier grader-prompt injection (ON / OFF, byte-identical off)
# ---------------------------------------------------------------------------

def test_grader_prompt_carries_skill_context_when_provided():
    grader = _CapturingGrader()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _honest_run_dir(tmp)
        score_reproduction(
            _TINY_TREE, run_dir, grader, skill_context="SKILL_CONTEXT_MARKER_XYZ"
        )
    assert grader.users, "the grader must have been called"
    assert all("SKILL_CONTEXT_MARKER_XYZ" in u for u in grader.users)


def test_grader_prompt_byte_identical_without_skill_context():
    grader = _CapturingGrader()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _honest_run_dir(tmp)
        score_reproduction(_TINY_TREE, run_dir, grader)  # no skill_context
    assert grader.users
    assert all("SKILL_CONTEXT_MARKER" not in u for u in grader.users)
    assert all("Skill playbooks relevant to this paper" not in u for u in grader.users)


# ---------------------------------------------------------------------------
# Primitive count unchanged — no new primitive was added
# ---------------------------------------------------------------------------

def test_primitive_count_unchanged():
    assert len(PRIMITIVE_REGISTRY) == 19
