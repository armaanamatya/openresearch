"""Tests for the consult_skill primitive and its OPENRESEARCH_SKILLS wiring
(Release-1 5.C): the primitive itself, the detect_environment matcher hook +
cache-key separation, the system-prompt catalog section, and the matched-
shortlist implementer-guidance block.

Hermetic: OFF-state assertions run with the flag deleted (byte-identical to
today); ON-state assertions use monkeypatch.setenv("OPENRESEARCH_SKILLS", "1").
Real vendored skill-library fixtures throughout (skill_catalog.load_catalog()
against backend/agents/rlm/skills/) — no mocks of our own modules.
"""
from __future__ import annotations

import json

from backend.agents.rlm.primitives import (
    PRIMITIVE_DESCRIPTIONS,
    PRIMITIVE_REGISTRY,
    consult_skill,
    detect_environment,
)
from backend.agents.rlm.system_prompt import build_system_prompt


def _ctx_meta():
    return {"context": {"type": "str", "length": 10}}


def _root_model():
    from backend.agents.rlm.models import ROOT_MODELS
    return ROOT_MODELS["gpt-5"]


def _write_skill_match_fixture(project_dir, skill_names=("verl-rl-training",)):
    state_dir = project_dir / "rlm_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "skill_match.json").write_text(
        json.dumps({"domain": "ml-rl", "skill_names": list(skill_names), "reasons": ["grpo"]}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Registry coverage (mirrors test_registry.py at the primitive level)
# ---------------------------------------------------------------------------


def test_registry_and_descriptions_include_consult_skill():
    assert "consult_skill" in PRIMITIVE_REGISTRY
    assert "consult_skill" in PRIMITIVE_DESCRIPTIONS
    assert "OPENRESEARCH_SKILLS" in PRIMITIVE_DESCRIPTIONS["consult_skill"]


# ---------------------------------------------------------------------------
# OFF state — byte-identical / inert
# ---------------------------------------------------------------------------


def test_consult_skill_disabled_when_flag_unset(make_context, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_SKILLS", raising=False)
    ctx = make_context(tmp_path)
    assert consult_skill(name="verl-rl-training", ctx=ctx) == {"status": "disabled"}


def test_consult_skill_disabled_with_no_args(make_context, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_SKILLS", raising=False)
    ctx = make_context(tmp_path)
    assert consult_skill(ctx=ctx) == {"status": "disabled"}


def test_system_prompt_omits_skill_catalog_when_flag_off(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_SKILLS", raising=False)
    prompt = build_system_prompt(context_metadata=_ctx_meta(), root_model=_root_model())
    assert "SKILL LIBRARY (consult_skill)" not in prompt
    assert prompt.count("{custom_tools_section}") == 1


def test_shortlist_block_empty_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_SKILLS", raising=False)
    from backend.agents.baseline_implementation import _skill_shortlist_block

    _write_skill_match_fixture(tmp_path)
    assert _skill_shortlist_block(tmp_path) == ""


def test_compute_constraint_guidance_no_shortlist_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_SKILLS", raising=False)
    from backend.agents.baseline_implementation import _compute_constraint_guidance

    _write_skill_match_fixture(tmp_path)
    guidance = _compute_constraint_guidance(
        sandbox_mode="local", gpu_mode="auto", project_dir=tmp_path,
    )
    assert "MATCHED SKILL PLAYBOOKS" not in guidance


def test_detect_environment_no_skill_match_when_flag_off(make_context, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_SKILLS", raising=False)
    ctx = make_context(tmp_path)
    method_spec = {"core_contribution": "GRPO reinforcement learning for LLM agents"}
    result = detect_environment(method_spec, ctx=ctx)
    assert "skill_match" not in (result.get("extra") or {})
    assert not (ctx.project_dir / "rlm_state" / "skill_match.json").exists()


# ---------------------------------------------------------------------------
# ON state — consult_skill primitive
# ---------------------------------------------------------------------------


def test_consult_skill_known_name_returns_body(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    ctx = make_context(tmp_path)
    result = consult_skill(name="verl-rl-training", ctx=ctx)
    assert result["status"] == "ok"
    assert result["name"] == "verl-rl-training"
    assert result["category"] == "ml-training"
    assert isinstance(result["tags"], list) and result["tags"]
    assert isinstance(result["body"], str) and len(result["body"]) > 0
    assert isinstance(result["references"], list)
    assert isinstance(result["scripts"], list)


def test_consult_skill_unknown_name_returns_did_you_mean(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    ctx = make_context(tmp_path)
    result = consult_skill(name="verl-rl-trainng", ctx=ctx)
    assert result["status"] == "not_found"
    assert len(result["did_you_mean"]) > 0
    assert "verl-rl-training" in result["did_you_mean"]


def test_consult_skill_category_browse(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    ctx = make_context(tmp_path)
    result = consult_skill(category="scholar-evaluation", ctx=ctx)
    assert result["status"] == "ok"
    assert result["kind"] == "category"
    names = {s["name"] for s in result["skills"]}
    assert "scholar-evaluation" in names
    assert all("description" in s and s["description"] for s in result["skills"])


def test_consult_skill_index_when_no_args(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    ctx = make_context(tmp_path)
    result = consult_skill(ctx=ctx)
    assert result["status"] == "ok"
    assert result["kind"] == "index"
    counts = {c["name"]: c["count"] for c in result["categories"]}
    assert "ml-training" in counts
    # Sorted by count descending: ml-training (27 in the vendored catalog) must
    # rank ahead of the single-skill scholar-evaluation category.
    assert result["categories"][0]["name"] == "ml-training"
    assert counts["ml-training"] >= counts.get("scholar-evaluation", 0)


def test_consult_skill_ships_scripts_into_code_dir(make_context, tmp_path, monkeypatch):
    """scholar-evaluation vendors a real scripts/ dir; when code/ exists the
    script must land at code/skill_scripts/<name>/ and its relpath is returned."""
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    ctx = make_context(tmp_path)
    (ctx.project_dir / "code").mkdir(parents=True)

    result = consult_skill(name="scholar-evaluation", ctx=ctx)

    assert result["status"] == "ok"
    assert result["scripts"], "scholar-evaluation ships a real scripts/ dir"
    dest_rel = result["scripts"][0]
    assert dest_rel.startswith("skill_scripts/scholar-evaluation/")
    assert (ctx.project_dir / "code" / dest_rel).is_file()


def test_consult_skill_skips_scripts_when_code_dir_absent(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    ctx = make_context(tmp_path)
    # No code/ directory created — must skip silently, not raise.
    result = consult_skill(name="scholar-evaluation", ctx=ctx)
    assert result["status"] == "ok"
    assert result["scripts"] == []


# ---------------------------------------------------------------------------
# ON state — detect_environment matcher hook + cache-key separation
# ---------------------------------------------------------------------------


def test_detect_environment_hook_persists_skill_match(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    ctx = make_context(tmp_path)
    method_spec = {
        "core_contribution": "GRPO reinforcement learning for LLM agents",
        "model_architecture": "Qwen",
        "datasets": ["ALFWorld"],
        "training_recipe": {"optimizer": "GRPO"},
    }
    result = detect_environment(method_spec, ctx=ctx)

    assert result["extra"]["skill_match"]["domain"] == "ml-rl"
    state_path = ctx.project_dir / "rlm_state" / "skill_match.json"
    assert state_path.is_file()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["domain"] == "ml-rl"
    assert isinstance(saved["skill_names"], list)


def test_detect_environment_cache_key_separates_skills_flag(make_context, tmp_path, monkeypatch):
    """Same ctx + method_spec, flag OFF then ON: the ON call must NOT reuse the
    OFF call's cached spec_dict (Deliverable 3's cache-key correctness rule —
    without the _payload["skills"] gate this would incorrectly cache-hit)."""
    monkeypatch.delenv("OPENRESEARCH_SKILLS", raising=False)
    ctx = make_context(tmp_path)
    method_spec = {"core_contribution": "GRPO reinforcement learning for LLM agents"}

    off_result = detect_environment(method_spec, ctx=ctx)
    assert "skill_match" not in (off_result.get("extra") or {})

    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    on_result = detect_environment(method_spec, ctx=ctx)
    assert "skill_match" in on_result["extra"]


# ---------------------------------------------------------------------------
# ON state — system prompt + implementer-guidance shortlist
# ---------------------------------------------------------------------------


def test_system_prompt_includes_skill_catalog_when_flag_on(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    prompt = build_system_prompt(context_metadata=_ctx_meta(), root_model=_root_model())
    assert "SKILL LIBRARY (consult_skill)" in prompt
    assert "consult_skill(name=" in prompt
    assert prompt.count("{custom_tools_section}") == 1


def test_shortlist_block_renders_when_flag_on_and_fixture_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    from backend.agents.baseline_implementation import _skill_shortlist_block

    _write_skill_match_fixture(tmp_path)
    block = _skill_shortlist_block(tmp_path)
    assert "MATCHED SKILL PLAYBOOKS" in block
    assert "verl-rl-training" in block
    assert "consult_skill(name=" in block


def test_shortlist_block_empty_when_no_state_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    from backend.agents.baseline_implementation import _skill_shortlist_block

    assert _skill_shortlist_block(tmp_path) == ""


def test_compute_constraint_guidance_includes_shortlist_when_flag_on(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    from backend.agents.baseline_implementation import _compute_constraint_guidance

    _write_skill_match_fixture(tmp_path)
    guidance = _compute_constraint_guidance(
        sandbox_mode="local", gpu_mode="auto", project_dir=tmp_path,
    )
    assert "MATCHED SKILL PLAYBOOKS" in guidance
    assert "verl-rl-training" in guidance


# ---------------------------------------------------------------------------
# ON state — relevance-gated SELECTED skill block (OPENRESEARCH_SKILL_SELECT)
#
# _active_skill_block reads rlm_state/active_skills.json (written by
# detect_environment's selection hook, see test_skill_selection.py) and is
# preferred over the raw _skill_shortlist_block above at the injection site
# (baseline_implementation._compute_constraint_guidance: `_active_skill_block(
# project_dir) or _skill_shortlist_block(project_dir)`).
# ---------------------------------------------------------------------------


def _write_active_skills_fixture(project_dir, selected=("grpo-rl-training",)):
    state_dir = project_dir / "rlm_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "active_skills.json").write_text(
        json.dumps({
            "selected": list(selected),
            "candidates": [],
            "domain": "ml-rl",
            "subject_matter_keys": {},
            "selector": "deterministic",
            "reasons": {},
        }),
        encoding="utf-8",
    )


def test_active_skill_block_uses_selected_set(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    monkeypatch.setenv("OPENRESEARCH_SKILL_SELECT", "1")
    from backend.agents.baseline_implementation import _active_skill_block

    _write_active_skills_fixture(tmp_path, selected=("grpo-rl-training",))
    block = _active_skill_block(tmp_path)
    assert "SELECTED SKILL PLAYBOOKS" in block
    assert "grpo-rl-training" in block


def test_active_skill_block_empty_when_select_off(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SKILLS", "1")
    monkeypatch.delenv("OPENRESEARCH_SKILL_SELECT", raising=False)
    from backend.agents.baseline_implementation import _active_skill_block

    # Even with a real artifact on disk, SELECT off => empty (byte-identical).
    _write_active_skills_fixture(tmp_path, selected=("grpo-rl-training",))
    assert _active_skill_block(tmp_path) == ""
