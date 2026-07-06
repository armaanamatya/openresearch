"""Tests for Task 8: wiring spec_validator into run_pipeline_rlm.

Structural sibling of test_run_validator_wiring.py (separation tier +
fail-closed transport) and test_report_validation_stamp.py (the report
stamp), plus a hook-level test suite mirroring test_spec_validator.py's
fake-completions pattern. Hermetic: no network, no real LLM calls;
``sample_completions`` is monkeypatched on the ``spec_validator`` module
directly (module-level import there is required for this to work — see that
module's own docstring).

Per the task-8 brief: prefer testing ``_run_spec_validator`` + the report
stamp directly (unit-level) over spinning up the whole ``run_pipeline_rlm``.
The "all 4 events fire in order" scenario is exercised by manually composing
the exact call-site sequence (the two call-site-owned events bracketing the
hook), calling the REAL production functions throughout — never spinning up
paper ingestion / the RLM engine / a subprocess.
"""

from __future__ import annotations

import json
import os

import pytest

from backend.agents.rlm import spec_validator as sv
from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm
from backend.agents.rlm.role_models import RoleSelection, RoleSpec
from backend.agents.rlm.run import (
    _resolve_spec_validator_transport,
    _run_spec_validator,
    _spec_validator_separation_tier,
    assert_no_foundry_oauth_coresidency,
)
from backend.agents.rlm.sse_bridge import (
    build_spec_generated_event,
    build_spec_generation_started_event,
)

# ---------------------------------------------------------------------------
# Fixtures (mirrors test_spec_validator.py's Task-6 fixture exactly)
# ---------------------------------------------------------------------------

_RUBRIC = {"leaves": [
    {"id": "L1", "requirement": "Report ALFWorld success rate near 84.4"},   # grounded
    {"id": "L2", "requirement": "Report ImageNet top-1 accuracy of 99.9"},   # hallucinated (absent)
]}
_PAPER = "SDAR improves over GRPO (+9.4% on ALFWorld ... 84.4 ...). Search-QA, WebShop."


class _FakeClient:
    """Marker object only — sample_completions is monkeypatched at the
    module level, so this object's methods are never actually invoked; it
    only needs to be non-None to select the "client selected" branch."""


def _emit_recorder():
    events: list[dict] = []

    def emit(ev: dict) -> None:
        events.append(ev)

    return emit, events


# ---------------------------------------------------------------------------
# _run_spec_validator — the pre-loop hook, unit-level
# ---------------------------------------------------------------------------


def test_hook_off_by_default_self_gates(monkeypatch, tmp_path):
    """OFF (OPENRESEARCH_SPEC_VALIDATOR unset): the hook self-gates — no
    events, returns None, no verdict persisted. Defense-in-depth: this holds
    even when called directly (not just via the call-site's own gate)."""
    monkeypatch.delenv("OPENRESEARCH_SPEC_VALIDATOR", raising=False)
    emit, events = _emit_recorder()

    result = _run_spec_validator(
        rubric=_RUBRIC, paper_text=_PAPER, project_dir=tmp_path,
        emit=emit, spec_validator_client=_FakeClient(), separation="independent",
        validator_model="test-model",
    )

    assert result is None
    assert events == []
    assert not (tmp_path / "rlm_state" / "spec_validation_verdict.json").exists()


def test_hook_missing_rubric_is_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR", "1")
    emit, events = _emit_recorder()

    result = _run_spec_validator(
        rubric=None, paper_text=_PAPER, project_dir=tmp_path,
        emit=emit, spec_validator_client=_FakeClient(), separation="independent",
        validator_model="test-model",
    )

    assert result is None
    assert events == []


def test_hook_missing_paper_text_is_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR", "1")
    emit, events = _emit_recorder()

    result = _run_spec_validator(
        rubric=_RUBRIC, paper_text="", project_dir=tmp_path,
        emit=emit, spec_validator_client=_FakeClient(), separation="independent",
        validator_model="test-model",
    )

    assert result is None
    assert events == []


def test_hook_client_none_emits_unavailable(monkeypatch, tmp_path):
    """Unselected/unavailable spec_validator_client: emits the started/
    validated pair with an "unavailable" verdict and never runs the panel —
    no client means no LLM nomination to machine-check, so never veto."""
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR", "1")
    emit, events = _emit_recorder()

    result = _run_spec_validator(
        rubric=_RUBRIC, paper_text=_PAPER, project_dir=tmp_path,
        emit=emit, spec_validator_client=None, separation="unavailable",
        validator_model="unavailable",
    )

    assert result is None
    assert [e["event"] for e in events] == ["spec_validation_started", "spec_validated"]
    assert events[0]["validator_model"] == "unavailable"
    assert events[1]["verdict"] == "unavailable"
    assert events[1]["flagged_leaves"] == []
    assert not (tmp_path / "rlm_state" / "spec_validation_verdict.json").exists()


def test_hook_on_flags_hallucinated_leaf_and_persists_verdict(monkeypatch, tmp_path):
    """ON, BLOCK off: the hallucinated leaf is flagged + persisted, but the
    returned rubric is None (nothing to write back -- BLOCK is disabled)."""
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR", "1")
    monkeypatch.delenv("OPENRESEARCH_SPEC_VALIDATOR_BLOCK", raising=False)
    monkeypatch.setattr(
        sv, "sample_completions",
        lambda *a, **k: ['[{"predicate":"hallucinated_leaf","leaf_id":"L2"}]'],
    )
    emit, events = _emit_recorder()

    result = _run_spec_validator(
        rubric=_RUBRIC, paper_text=_PAPER, project_dir=tmp_path,
        emit=emit, spec_validator_client=_FakeClient(), separation="independent",
        validator_model="grok",
    )

    assert result is None  # BLOCK is off -- nothing for the caller to write back
    assert [e["event"] for e in events] == ["spec_validation_started", "spec_validated"]
    assert events[1]["verdict"] == "flagged"
    assert events[1]["flagged_leaves"] == ["L2"]

    verdict_path = tmp_path / "rlm_state" / "spec_validation_verdict.json"
    assert verdict_path.exists()
    persisted = json.loads(verdict_path.read_text())
    assert persisted["status"] == "flagged"
    assert persisted["flagged_leaves"] == ["L2"]
    assert persisted["separation"] == "independent"
    assert persisted["panel_models"] == ["grok"]


def test_hook_missing_key_claim_paper_phrase_excluded_from_event_and_report_stamp(monkeypatch, tmp_path):
    """FIX regression test (whole-branch review Finding #1): a
    missing_key_claim suspicion whose leaf_id is a model-authored
    paper-phrase label (not a real rubric leaf id) must not leak into
    flagged_leaves anywhere downstream of the hook -- not the persisted
    verdict, not the spec_validated SSE event (corpus-free, no
    redact_corpus pass), and not the report.spec_validation stamp. A REAL
    hallucinated_leaf id must still ride all three (no regression)."""
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR", "1")
    phrase = "the paper's ImageNet ablation on distribution shift"
    monkeypatch.setattr(
        sv, "sample_completions",
        lambda *a, **k: [json.dumps([
            {"predicate": "missing_key_claim", "leaf_id": phrase},
            {"predicate": "hallucinated_leaf", "leaf_id": "L2"},
        ])],
    )
    emit, events = _emit_recorder()

    _run_spec_validator(
        rubric=_RUBRIC, paper_text=_PAPER, project_dir=tmp_path,
        emit=emit, spec_validator_client=_FakeClient(), separation="independent",
        validator_model="grok",
    )

    # 1. The spec_validated SSE event.
    validated_event = next(e for e in events if e["event"] == "spec_validated")
    assert phrase not in validated_event["flagged_leaves"]
    assert "L2" in validated_event["flagged_leaves"]

    # 2. The persisted verdict store (rlm_state/spec_validation_verdict.json).
    verdict_path = tmp_path / "rlm_state" / "spec_validation_verdict.json"
    persisted = json.loads(verdict_path.read_text())
    assert phrase not in persisted["flagged_leaves"]
    assert "L2" in persisted["flagged_leaves"]
    # The predicate itself is still server-side recorded for operator review
    # (never on the SSE wire or the report stamp, which read flagged_leaves
    # exclusively -- see report.py's spec_validation stamp block).
    assert any(
        p["predicate"] == "missing_key_claim" and p["leaf_id"] == phrase
        for p in persisted["predicates"]
    )

    # 3. The report.spec_validation stamp.
    (tmp_path / "generated_rubric.json").write_text(json.dumps(_RUBRIC))
    report = RLMFinalReport()
    write_final_report_rlm(report, tmp_path)
    result = json.loads((tmp_path / "final_report.json").read_text())
    stamp = result.get("spec_validation", {})
    assert phrase not in stamp.get("flagged_leaves", [])
    assert "L2" in stamp.get("flagged_leaves", [])


def test_hook_block_enabled_drops_confirmed_leaf(monkeypatch, tmp_path):
    """BLOCK ON: the confirmed hallucinated leaf is dropped from the
    RETURNED rubric -- AND the call-site write-back reassigns
    context_dict['rubric_spec'] to the cleaned rubric, so the RLM loop that
    reads context_dict['rubric_spec'] uses the dropped-leaf rubric before
    RLM(...) is constructed. The write-back leg is the one that actually
    protects the loop -- assert it, not just the hook's return value."""
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR", "1")
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_BLOCK", "1")
    monkeypatch.setattr(
        sv, "sample_completions",
        lambda *a, **k: ['[{"predicate":"hallucinated_leaf","leaf_id":"L2"}]'],
    )
    emit, events = _emit_recorder()

    # Mirror the run.py call site exactly: the loop consumes
    # context_dict["rubric_spec"], and the hook's return value is written back
    # into it when non-None.
    context_dict = {"rubric_spec": _RUBRIC, "paper_text": _PAPER}
    result = _run_spec_validator(
        rubric=context_dict["rubric_spec"],
        paper_text=context_dict["paper_text"],
        project_dir=tmp_path,
        emit=emit, spec_validator_client=_FakeClient(), separation="independent",
        validator_model="grok",
    )
    if result is not None:
        context_dict["rubric_spec"] = result

    assert result is not None
    assert [leaf["id"] for leaf in result["leaves"]] == ["L1"]
    # The rubric the RLM loop actually consumes now has the confirmed leaf gone.
    assert [leaf["id"] for leaf in context_dict["rubric_spec"]["leaves"]] == ["L1"]
    assert context_dict["rubric_spec"] is result
    assert events[-1]["verdict"] == "flagged"


def test_hook_block_enabled_but_clean_verdict_is_unchanged_noop(monkeypatch, tmp_path):
    """BLOCK ON but nothing confirmed: apply_block's own no-op contract
    (test_apply_block_noop_when_nothing_confirmed in test_spec_validator.py)
    returns the SAME rubric object unchanged rather than None -- so the hook
    returns that unchanged rubric too. The call site's write-back
    (context_dict["rubric_spec"] = result) is then a harmless no-op: same
    object, same content, identical leaves."""
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR", "1")
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_BLOCK", "1")
    monkeypatch.setattr(sv, "sample_completions", lambda *a, **k: ["[]"])
    emit, events = _emit_recorder()

    result = _run_spec_validator(
        rubric=_RUBRIC, paper_text=_PAPER, project_dir=tmp_path,
        emit=emit, spec_validator_client=_FakeClient(), separation="independent",
        validator_model="grok",
    )

    assert result is _RUBRIC
    assert [leaf["id"] for leaf in result["leaves"]] == ["L1", "L2"]
    assert events[-1]["verdict"] == "clean"


def test_hook_swallows_internal_errors(monkeypatch, tmp_path):
    """A spec-validator internal error must never crash the run: returns
    None, never raises. (run_spec_validation_panel itself already fails soft
    against a sample_completions error, so this injects the failure one layer
    further in -- inside the panel runner itself -- to exercise THIS
    function's own outer try/except backstop.)"""
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR", "1")

    def _boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(sv, "run_spec_validation_panel", _boom)
    emit, events = _emit_recorder()

    result = _run_spec_validator(
        rubric=_RUBRIC, paper_text=_PAPER, project_dir=tmp_path,
        emit=emit, spec_validator_client=_FakeClient(), separation="independent",
        validator_model="grok",
    )

    assert result is None
    # spec_validation_started already fired before the panel call raised;
    # spec_validated never fires -- the outer except is the last-resort net.
    assert [e["event"] for e in events] == ["spec_validation_started"]


# ---------------------------------------------------------------------------
# Full call-site sequence: all 4 events in order + verdict persisted +
# report.spec_validation stamped. Manually composes the exact sequence
# run_pipeline_rlm uses (see run.py around the rubric cascade), calling only
# real production functions -- no paper ingestion / RLM engine / subprocess.
# ---------------------------------------------------------------------------


def test_full_call_site_sequence_all_four_events_and_report_stamp(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR", "1")
    monkeypatch.setattr(sv, "sample_completions", lambda *a, **k: ["[]"])  # clean
    emit, events = _emit_recorder()
    (tmp_path / "rlm_state").mkdir()

    # Call-site step 1 (before the rubric cascade in run_pipeline_rlm).
    emit(build_spec_generation_started_event())
    # (The rubric cascade itself is untouched by Task 8 and is not exercised
    # here; context_dict["rubric_spec"] is already resolved to _RUBRIC by the
    # time the call site reaches step 2.)
    # Call-site step 2 (after the cascade).
    emit(build_spec_generated_event(leaf_count=len(_RUBRIC["leaves"])))
    blocked = _run_spec_validator(
        rubric=_RUBRIC, paper_text=_PAPER, project_dir=tmp_path,
        emit=emit, spec_validator_client=_FakeClient(), separation="independent",
        validator_model="grok",
    )

    assert [e["event"] for e in events] == [
        "spec_generation_started",
        "spec_generated",
        "spec_validation_started",
        "spec_validated",
    ]
    assert events[1]["leaf_count"] == 2
    assert events[3]["verdict"] == "clean"
    assert blocked is None  # nothing confirmed -- no block needed

    verdict_path = tmp_path / "rlm_state" / "spec_validation_verdict.json"
    assert verdict_path.exists()

    # report.py stamp: write_final_report_rlm loads the verdict keyed by the
    # rubric's OWN fingerprint.
    (tmp_path / "generated_rubric.json").write_text(json.dumps(_RUBRIC))
    report = RLMFinalReport()
    write_final_report_rlm(report, tmp_path)
    result = json.loads((tmp_path / "final_report.json").read_text())
    stamp = result.get("spec_validation", {})
    assert stamp.get("status") == "clean"
    assert stamp.get("panel_models") == ["grok"]
    assert stamp.get("separation") == "independent"
    assert stamp.get("rubric_fingerprint") == sv.rubric_fingerprint(_RUBRIC)


# ---------------------------------------------------------------------------
# report.py — spec_validation stamp (parallel to test_report_validation_stamp.py)
# ---------------------------------------------------------------------------


def _persist(tmp_path, rubric, **kwargs) -> sv.SpecValidatorVerdict:
    verdict = sv.SpecValidatorVerdict(
        status=kwargs.get("status", "clean"),
        flagged_leaves=kwargs.get("flagged_leaves", []),
        predicates=kwargs.get("predicates", []),
        panel_models=kwargs.get("panel_models", ["grok"]),
        separation=kwargs.get("separation", "independent"),
        rubric_fingerprint=sv.rubric_fingerprint(rubric),
    )
    sv.persist_spec_verdict(tmp_path, verdict)
    return verdict


def test_report_stamp_matching_fingerprint(tmp_path):
    _persist(tmp_path, _RUBRIC, status="flagged", flagged_leaves=["L2"])
    (tmp_path / "generated_rubric.json").write_text(json.dumps(_RUBRIC))

    report = RLMFinalReport()
    write_final_report_rlm(report, tmp_path)

    result = json.loads((tmp_path / "final_report.json").read_text())
    stamp = result.get("spec_validation", {})
    assert stamp.get("status") == "flagged"
    assert stamp.get("flagged_leaves") == ["L2"]


def test_report_stamp_stale_fingerprint_leaves_default(tmp_path):
    """A verdict for a DIFFERENT rubric (stale) must not be stamped."""
    other_rubric = {"leaves": [{"id": "X1", "requirement": "unrelated"}]}
    _persist(tmp_path, other_rubric, status="clean")
    (tmp_path / "generated_rubric.json").write_text(json.dumps(_RUBRIC))

    report = RLMFinalReport()
    write_final_report_rlm(report, tmp_path)

    result = json.loads((tmp_path / "final_report.json").read_text())
    assert result.get("spec_validation", {}) == {}


def test_report_stamp_no_verdict_no_rubric_file_default(tmp_path):
    """OFF / no verdict at all -- spec_validation stays at its default {},
    and the rest of the report is unaffected (byte-identical OFF state)."""
    report = RLMFinalReport()
    report.iterations = 3
    write_final_report_rlm(report, tmp_path)

    result = json.loads((tmp_path / "final_report.json").read_text())
    assert result.get("spec_validation", {}) == {}
    assert result["iterations"] == 3


def test_report_default_field_is_empty_dict():
    """Sanity: the bare model's spec_validation defaults to {} (mirrors
    validation's default) -- the OFF-state byte-identical invariant."""
    report = RLMFinalReport()
    assert report.spec_validation == {}


# ---------------------------------------------------------------------------
# _spec_validator_separation_tier
# ---------------------------------------------------------------------------


def _planner_spec(
    provider: str = "anthropic-oauth",
    model: str | None = "claude-sonnet-4-6",
    family: str | None = "claude",
    token: str = "claude-oauth",
) -> RoleSpec:
    return RoleSpec(role="planner", token=token, provider=provider, model=model, family=family)


def test_spec_tier_unavailable_when_backend_unset(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", raising=False)
    monkeypatch.delenv("OPENRESEARCH_VALIDATOR_BACKEND", raising=False)
    sel = RoleSelection(planner=_planner_spec(), executor=None, verifier=None, grader=None)
    assert _spec_validator_separation_tier(sel) == "unavailable"


def test_spec_tier_unavailable_when_role_selection_none(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", raising=False)
    monkeypatch.delenv("OPENRESEARCH_VALIDATOR_BACKEND", raising=False)
    assert _spec_validator_separation_tier(None) == "unavailable"


def test_spec_tier_falls_back_to_validator_backend_env(monkeypatch):
    """SPEC_VALIDATOR_BACKEND unset but VALIDATOR_BACKEND set: the spec tier
    still resolves (mirrors build_spec_validator_client's own fallback)."""
    monkeypatch.delenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", raising=False)
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "azure")
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "gpt-4o-valB")
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    sel = RoleSelection(planner=_planner_spec(), executor=None, verifier=None, grader=None)
    assert _spec_validator_separation_tier(sel) == "independent"


def test_spec_tier_compares_against_planner_not_executor(monkeypatch):
    """The rubric is authored by the planner, so independence for THIS panel
    is planner x spec_validator. A DIFFERENT (same-family-as-spec-validator)
    executor must be ignored -- if the tier wrongly compared against
    executor here it would read "degraded" instead of "independent"."""
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", "azure")
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_MODEL", "gpt-4o-sv")
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

    planner = _planner_spec()  # claude family -- independent from an azure spec_validator
    executor = RoleSpec(role="executor", token="azure", provider="azure", model="gpt-4o-sv", family="gpt")
    sel = RoleSelection(planner=planner, executor=executor, verifier=None, grader=None)
    assert _spec_validator_separation_tier(sel) == "independent"


def test_spec_tier_degraded_same_planner_family_and_model(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", "azure")
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_MODEL", "gpt-4o-shared")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-shared")

    planner = RoleSpec(role="planner", token="azure", provider="azure", model="gpt-4o-shared", family="gpt")
    sel = RoleSelection(planner=planner, executor=None, verifier=None, grader=None)
    assert _spec_validator_separation_tier(sel) == "degraded"


# ---------------------------------------------------------------------------
# _resolve_spec_validator_transport — the SETUP block, internally flag-gated.
# The OFF test is the lock for the reviewer's Important finding: with the
# spec-validator flag OFF but the EXTERNAL validator ON (VALIDATOR_BACKEND set)
# and even a spec_validator role selected, the setup block must mutate no env,
# build/log no client, emit no separation warning, and NEVER raise. Without the
# gate this exact input crashes (the bridge sets SPEC_VALIDATOR_BACKEND from the
# grok role -> the fail-closed build hits missing foundry creds -> RuntimeError)
# and/or leaks a spec_validator_separation_* run_warning off the VALIDATOR_*
# fallback -- so this test would FAIL on the pre-fix code.
# ---------------------------------------------------------------------------


def test_resolve_setup_off_is_inert_even_with_validator_backend_and_role(monkeypatch):
    """OFF ⇒ byte-identical: no env mutation, no emit, no raise, default return."""
    monkeypatch.delenv("OPENRESEARCH_SPEC_VALIDATOR", raising=False)
    # External validator ON (the setup block's client build + tier both fall
    # back to these) + no spec-validator vars pre-set.
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "azure")
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "gpt-4o-valB")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-planner")
    monkeypatch.delenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", raising=False)
    monkeypatch.delenv("OPENRESEARCH_SPEC_VALIDATOR_MODEL", raising=False)
    # No azure creds -> a build would fail-closed (crash) WITHOUT the gate.
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    # A selected spec_validator role (grok) — the bridge would mutate env +
    # trigger the fail-closed foundry build without the flag gate.
    planner = RoleSpec(role="planner", token="azure", provider="azure",
                       model="gpt-4o-planner", family="gpt")
    spec_validator = RoleSpec(role="spec_validator", token="grok",
                              provider="azure-foundry", model=None, family="grok")
    sel = RoleSelection(planner=planner, executor=None, verifier=None, grader=None,
                        spec_validator=spec_validator)

    emit, events = _emit_recorder()
    client, label, tier = _resolve_spec_validator_transport(
        role_selection=sel, llm_client=object(), provider_label="anthropic", emit=emit,
    )

    assert client is None
    assert label == "anthropic"        # the hoisted provider_label default
    assert tier == "unavailable"       # the hoisted default (no tier computed)
    assert events == []                # OFF ⇒ NO SSE event (the reviewer's Scenario A)
    # OFF ⇒ NO env mutation (the reviewer's Scenario B — the bridge is skipped).
    assert not os.environ.get("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", "").strip()


def test_resolve_setup_on_fail_closed_raises_on_missing_creds(monkeypatch):
    """ON + explicitly-selected backend with missing creds ⇒ RuntimeError
    (fail-closed preserved by the extraction — a misconfigured spec validator
    crashes an ON run rather than silently riding the planner client)."""
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR", "1")
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", "azure")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    sel = RoleSelection(planner=_planner_spec(), executor=None, verifier=None, grader=None)
    emit, _events = _emit_recorder()
    with pytest.raises(RuntimeError, match="spec validator setup failed"):
        _resolve_spec_validator_transport(
            role_selection=sel, llm_client=object(), provider_label="anthropic", emit=emit,
        )


def test_resolve_setup_on_builds_client_and_emits_weak(monkeypatch):
    """ON path intact after extraction: a successfully-built client + a
    planner×spec_validator "weak" tier emits the spec_validator_separation_weak
    advisory. build_spec_validator_client is monkeypatched to a sentinel so the
    test needs no real creds."""
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR", "1")
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_BACKEND", "azure")
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_MODEL", "gpt-4o-svB")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-plannerA")

    sentinel = object()
    from backend.agents.rlm import grader_transport as _gt
    monkeypatch.setattr(
        _gt, "build_spec_validator_client",
        lambda *, fallback_client, fallback_label="": (sentinel, "spec_validator:azure:gpt-4o-svB"),
    )

    planner = RoleSpec(role="planner", token="azure", provider="azure",
                       model="gpt-4o-plannerA", family="gpt")
    sel = RoleSelection(planner=planner, executor=None, verifier=None, grader=None)

    emit, events = _emit_recorder()
    client, label, tier = _resolve_spec_validator_transport(
        role_selection=sel, llm_client=object(), provider_label="anthropic", emit=emit,
    )

    assert client is sentinel
    assert label == "spec_validator:azure:gpt-4o-svB"
    assert tier == "weak"  # azure planner(deployA) x azure spec_validator(deployB)
    assert [e["event"] for e in events] == ["run_warning"]
    assert events[0]["code"] == "spec_validator_separation_weak"


# ---------------------------------------------------------------------------
# Coresidency guard (Landmine T4 -> T8): spec_validator=grok (azure-foundry)
# must not false-trip the anthropic-foundry / claude-oauth coresidency guard.
# This is EXACTLY the shape of the canonical autonomous run-spec
# (configs/autonomous_reproduction_run_spec.json): opus-foundry root +
# executor/grader/verifier=sonnet-foundry (anthropic-foundry) paired with
# spec_validator=grok (azure-foundry).
# ---------------------------------------------------------------------------


def test_coresidency_allows_spec_validator_grok_alongside_foundry_root_and_subroles():
    planner = RoleSpec(role="planner", token="opus-foundry", provider="anthropic-foundry",
                        model="claude-opus-4-8", family="claude")
    executor = RoleSpec(role="executor", token="sonnet-foundry", provider="anthropic-foundry",
                         model="claude-sonnet-5", family="claude")
    grader = RoleSpec(role="grader", token="sonnet-foundry", provider="anthropic-foundry",
                       model="claude-sonnet-5", family="claude")
    verifier = RoleSpec(role="verifier", token="sonnet-foundry", provider="anthropic-foundry",
                         model="claude-sonnet-5", family="claude")
    spec_validator = RoleSpec(role="spec_validator", token="grok", provider="azure-foundry",
                               model=None, family="grok")
    sel = RoleSelection(
        planner=planner, executor=executor, verifier=verifier, grader=grader,
        validator=None, spec_validator=spec_validator,
    )
    assert assert_no_foundry_oauth_coresidency("opus-foundry", sel) is None
