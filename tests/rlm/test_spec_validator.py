"""Tests for backend.agents.rlm.spec_validator (Task 6).

Structural sibling of test_external_validator.py: rubric-vs-paper (not
metrics-vs-disk), fired ONCE pre-loop. All tests are hermetic — no network
calls, no real LLM calls. ``sv.sample_completions`` is monkeypatched directly
on the module (it MUST be imported at module level for this to work — see
spec_validator.py's docstring on that point). pytest-socket blocks
non-loopback anyway.
"""

from __future__ import annotations

import json

from backend.agents.rlm import spec_validator as sv

# ---------------------------------------------------------------------------
# Fixtures (verbatim from the task-6 brief's Step 1)
# ---------------------------------------------------------------------------

_RUBRIC = {"leaves": [
    {"id": "L1", "requirement": "Report ALFWorld success rate near 84.4"},   # grounded
    {"id": "L2", "requirement": "Report ImageNet top-1 accuracy of 99.9"},   # hallucinated (absent)
]}
_PAPER = "SDAR improves over GRPO (+9.4% on ALFWorld ... 84.4 ...). Search-QA, WebShop."


class _FakeClient:  # sample_completions returns a JSON array of suspicions
    def __init__(self, arr):
        self._arr = arr


# ---------------------------------------------------------------------------
# Step 1 brief tests (verbatim)
# ---------------------------------------------------------------------------


def test_hallucinated_leaf_flagged(monkeypatch):
    monkeypatch.setattr(sv, "sample_completions",
        lambda *a, **k: ['[{"predicate":"hallucinated_leaf","leaf_id":"L2"}]'])
    v = sv.run_spec_validation_panel(spec_validator_client=_FakeClient(None),
        panel_models=["grok"], rubric=_RUBRIC, paper_text=_PAPER, separation="independent")
    assert v.status == "flagged" and "L2" in v.flagged_leaves and "L1" not in v.flagged_leaves


def test_grounded_leaf_not_flagged_even_if_llm_points_at_it(monkeypatch):
    # LLM opinion is never dispositive: machine-check clears L1 (grounded), so no veto.
    monkeypatch.setattr(sv, "sample_completions",
        lambda *a, **k: ['[{"predicate":"hallucinated_leaf","leaf_id":"L1"}]'])
    v = sv.run_spec_validation_panel(spec_validator_client=_FakeClient(None),
        panel_models=["grok"], rubric=_RUBRIC, paper_text=_PAPER, separation="independent")
    assert "L1" not in v.flagged_leaves
    assert v.status == "clean"


def test_apply_block_drops_confirmed_and_renormalizes():
    v = sv.SpecValidatorVerdict(status="flagged", flagged_leaves=["L2"],
        predicates=[sv.SpecPredicateVerdict("hallucinated_leaf", "L2", True, "absent")],
        panel_models=["grok"], separation="independent", rubric_fingerprint="x")
    out = sv.apply_block(_RUBRIC, v)
    assert [l["id"] for l in out["leaves"]] == ["L1"]


def test_disabled_is_noop(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_SPEC_VALIDATOR", raising=False)
    assert sv.spec_validator_enabled() is False


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------


def test_spec_validator_enabled_when_set(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR", "1")
    assert sv.spec_validator_enabled() is True


def test_spec_validator_panel_n_default():
    assert sv.spec_validator_panel_n() == 2


def test_spec_validator_panel_n_custom(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_PANEL_N", "4")
    assert sv.spec_validator_panel_n() == 4


def test_spec_validator_panel_n_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_PANEL_N", "notanint")
    assert sv.spec_validator_panel_n() == 2


def test_spec_validator_panel_n_min_1(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_PANEL_N", "0")
    assert sv.spec_validator_panel_n() >= 1


def test_spec_validator_block_enabled_default_false():
    assert sv.spec_validator_block_enabled() is False


def test_spec_validator_block_enabled_when_set(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_BLOCK", "1")
    assert sv.spec_validator_block_enabled() is True


# ---------------------------------------------------------------------------
# rubric_fingerprint
# ---------------------------------------------------------------------------


def test_rubric_fingerprint_stable():
    assert sv.rubric_fingerprint(_RUBRIC) == sv.rubric_fingerprint(_RUBRIC)


def test_rubric_fingerprint_canonical_key_order():
    """Same content, different key order -> same fingerprint (sort_keys=True)."""
    r1 = {"leaves": [{"id": "L1", "requirement": "x"}], "meta": {"a": 1, "b": 2}}
    r2 = {"meta": {"b": 2, "a": 1}, "leaves": [{"requirement": "x", "id": "L1"}]}
    assert sv.rubric_fingerprint(r1) == sv.rubric_fingerprint(r2)


def test_rubric_fingerprint_changes_with_content():
    other = {"leaves": [{"id": "L1", "requirement": "Report ALFWorld success rate near 84.4"},
                         {"id": "L2", "requirement": "something else entirely"}]}
    assert sv.rubric_fingerprint(_RUBRIC) != sv.rubric_fingerprint(other)


# ---------------------------------------------------------------------------
# run_spec_validation_panel — None client -> unavailable
# ---------------------------------------------------------------------------


def test_run_spec_validation_panel_none_client():
    v = sv.run_spec_validation_panel(
        spec_validator_client=None,
        panel_models=[],
        rubric=_RUBRIC,
        paper_text=_PAPER,
        separation="unavailable",
    )
    assert v.status == "unavailable"
    assert v.flagged_leaves == []
    assert v.predicates == []


# ---------------------------------------------------------------------------
# Panel call failure -> unavailable (fail-soft)
# ---------------------------------------------------------------------------


def test_run_spec_validation_panel_call_failure_returns_unavailable(monkeypatch):
    def _bad(*a, **k):
        raise RuntimeError("network error")

    monkeypatch.setattr(sv, "sample_completions", _bad)
    v = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["test-model"],
        rubric=_RUBRIC,
        paper_text=_PAPER,
        separation="independent",
    )
    assert v.status == "unavailable"


# ---------------------------------------------------------------------------
# Malformed / empty LLM responses handled gracefully
# ---------------------------------------------------------------------------


def test_malformed_json_from_panel_handled(monkeypatch):
    monkeypatch.setattr(sv, "sample_completions", lambda *a, **k: ["this is not json at all"])
    v = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["m"],
        rubric=_RUBRIC,
        paper_text=_PAPER,
        separation="independent",
    )
    assert v.status == "clean"
    assert v.predicates == []


def test_no_suspicions_is_clean(monkeypatch):
    monkeypatch.setattr(sv, "sample_completions", lambda *a, **k: [json.dumps([])])
    v = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["m"],
        rubric=_RUBRIC,
        paper_text=_PAPER,
        separation="independent",
    )
    assert v.status == "clean"
    assert v.flagged_leaves == []


def test_unknown_predicate_ignored(monkeypatch):
    monkeypatch.setattr(sv, "sample_completions",
        lambda *a, **k: ['[{"predicate":"made_up_predicate","leaf_id":"L1"}]'])
    v = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["m"],
        rubric=_RUBRIC,
        paper_text=_PAPER,
        separation="independent",
    )
    assert v.status == "clean"
    assert v.predicates == []


def test_partial_json_in_prose_handled(monkeypatch):
    """JSON array embedded in prose is extracted (fence/prose-tolerant parse)."""
    response = (
        'The rubric looks suspicious. Findings: '
        '[{"predicate": "hallucinated_leaf", "leaf_id": "L2"}] That is all.'
    )
    monkeypatch.setattr(sv, "sample_completions", lambda *a, **k: [response])
    v = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["m"],
        rubric=_RUBRIC,
        paper_text=_PAPER,
        separation="independent",
    )
    assert v.status == "flagged"
    assert "L2" in v.flagged_leaves


# ---------------------------------------------------------------------------
# Min-aggregation: one panelist violated + one clean -> flagged
# ---------------------------------------------------------------------------


def test_min_aggregation_one_panelist_flags(monkeypatch):
    responses = [
        '[{"predicate":"hallucinated_leaf","leaf_id":"L2"}]',
        "[]",
    ]

    def _fake(client, *, system, user, n, **kwargs):
        return responses[:n] if n <= len(responses) else responses + [responses[-1]] * (n - len(responses))

    monkeypatch.setattr(sv, "sample_completions", _fake)
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_PANEL_N", "2")
    v = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["model-a", "model-b"],
        rubric=_RUBRIC,
        paper_text=_PAPER,
        separation="weak",
    )
    assert v.status == "flagged"
    assert "L2" in v.flagged_leaves


def test_carries_separation_and_panel_models(monkeypatch):
    monkeypatch.setattr(sv, "sample_completions", lambda *a, **k: ["[]"])
    v = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["oauth-sonnet", "azure-gpt4o"],
        rubric=_RUBRIC,
        paper_text=_PAPER,
        separation="independent",
    )
    assert v.separation == "independent"
    assert v.panel_models == ["oauth-sonnet", "azure-gpt4o"]
    assert v.rubric_fingerprint == sv.rubric_fingerprint(_RUBRIC)


def test_leaf_id_not_in_rubric_is_fail_soft_not_violated(monkeypatch):
    """A hallucinated_leaf/wrong_target suspicion on an id absent from the
    rubric cannot be machine-verified -- fail-soft, never violated."""
    monkeypatch.setattr(sv, "sample_completions",
        lambda *a, **k: ['[{"predicate":"hallucinated_leaf","leaf_id":"L999"}]'])
    v = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["m"],
        rubric=_RUBRIC,
        paper_text=_PAPER,
        separation="independent",
    )
    assert v.status == "clean"
    assert "L999" not in v.flagged_leaves


# ---------------------------------------------------------------------------
# wrong_target predicate
# ---------------------------------------------------------------------------

_RUBRIC_WT = {"leaves": [
    {"id": "W1", "requirement": "Achieves ALFWorld success rate of 84.4"},
]}
_PAPER_WT_CONTRADICTS = "SDAR achieves ALFWorld success rate of 50.0 in our main experiments."
_PAPER_WT_AGREES = "SDAR achieves ALFWorld success rate of 84.4 in our main experiments."
_PAPER_WT_NO_CLAIM = "SDAR is a self-distilled agentic reinforcement learning method for ALFWorld."


def test_wrong_target_confirmed_when_both_sides_extract_and_disagree(monkeypatch):
    monkeypatch.setattr(sv, "sample_completions",
        lambda *a, **k: ['[{"predicate":"wrong_target","leaf_id":"W1"}]'])
    v = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["m"],
        rubric=_RUBRIC_WT,
        paper_text=_PAPER_WT_CONTRADICTS,
        separation="independent",
    )
    assert v.status == "flagged"
    assert "W1" in v.flagged_leaves
    pv = v.predicates[0]
    assert pv.predicate == "wrong_target"
    assert pv.violated is True


def test_wrong_target_clear_when_values_agree(monkeypatch):
    monkeypatch.setattr(sv, "sample_completions",
        lambda *a, **k: ['[{"predicate":"wrong_target","leaf_id":"W1"}]'])
    v = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["m"],
        rubric=_RUBRIC_WT,
        paper_text=_PAPER_WT_AGREES,
        separation="independent",
    )
    assert v.status == "clean"
    assert "W1" not in v.flagged_leaves


def test_wrong_target_fail_soft_when_paper_has_no_comparable_claim(monkeypatch):
    """Paper text carries no clean numeric claim for the metric -> fail-soft, not violated."""
    monkeypatch.setattr(sv, "sample_completions",
        lambda *a, **k: ['[{"predicate":"wrong_target","leaf_id":"W1"}]'])
    v = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["m"],
        rubric=_RUBRIC_WT,
        paper_text=_PAPER_WT_NO_CLAIM,
        separation="independent",
    )
    assert v.status == "clean"
    assert "W1" not in v.flagged_leaves


def test_check_wrong_target_direct():
    assert sv.check_wrong_target(
        "Achieves ALFWorld success rate of 84.4", _PAPER_WT_CONTRADICTS
    ) is False
    assert sv.check_wrong_target(
        "Achieves ALFWorld success rate of 84.4", _PAPER_WT_AGREES
    ) is True
    assert sv.check_wrong_target(
        "Achieves ALFWorld success rate of 84.4", _PAPER_WT_NO_CLAIM
    ) is True


# ---------------------------------------------------------------------------
# placeholder_leaf predicate
# ---------------------------------------------------------------------------

_RUBRIC_PLACEHOLDER = {"leaves": [
    {"id": "P1", "requirement": "The hyperparameters (, ) are correctly set as described."},
    {"id": "P2", "requirement": "Sets lambda=0.1 for the self-distillation loss weight."},
]}


def test_placeholder_leaf_confirmed(monkeypatch):
    monkeypatch.setattr(sv, "sample_completions",
        lambda *a, **k: ['[{"predicate":"placeholder_leaf","leaf_id":"P1"}]'])
    v = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["m"],
        rubric=_RUBRIC_PLACEHOLDER,
        paper_text=_PAPER,
        separation="independent",
    )
    assert v.status == "flagged"
    assert "P1" in v.flagged_leaves


def test_placeholder_leaf_not_confirmed_for_concrete_leaf(monkeypatch):
    monkeypatch.setattr(sv, "sample_completions",
        lambda *a, **k: ['[{"predicate":"placeholder_leaf","leaf_id":"P2"}]'])
    v = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["m"],
        rubric=_RUBRIC_PLACEHOLDER,
        paper_text=_PAPER,
        separation="independent",
    )
    assert v.status == "clean"
    assert "P2" not in v.flagged_leaves


def test_check_placeholder_leaf_direct():
    assert sv.check_placeholder_leaf("The hyperparameters (, ) are correctly set.") is False
    assert sv.check_placeholder_leaf("Sets lambda=0.1 for the loss weight.") is True


# ---------------------------------------------------------------------------
# missing_key_claim predicate — advisory, always confirmed, never blocked
# ---------------------------------------------------------------------------


def test_missing_key_claim_flagged_but_advisory(monkeypatch):
    monkeypatch.setattr(sv, "sample_completions",
        lambda *a, **k: ['[{"predicate":"missing_key_claim","leaf_id":"MISSING:webshop_result"}]'])
    v = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["m"],
        rubric=_RUBRIC,
        paper_text=_PAPER,
        separation="independent",
    )
    assert v.status == "flagged"
    assert "MISSING:webshop_result" in v.flagged_leaves
    pv = [p for p in v.predicates if p.predicate == "missing_key_claim"][0]
    assert pv.violated is True


def test_missing_key_claim_never_dropped_by_apply_block():
    v = sv.SpecValidatorVerdict(
        status="flagged",
        flagged_leaves=["L1"],
        predicates=[sv.SpecPredicateVerdict("missing_key_claim", "L1", True, "advisory gap")],
        panel_models=["m"],
        separation="independent",
        rubric_fingerprint=sv.rubric_fingerprint(_RUBRIC),
    )
    out = sv.apply_block(_RUBRIC, v)
    assert [l["id"] for l in out["leaves"]] == ["L1", "L2"]


# ---------------------------------------------------------------------------
# apply_block — additional coverage
# ---------------------------------------------------------------------------


def test_apply_block_noop_when_nothing_confirmed():
    v = sv.SpecValidatorVerdict(
        status="clean", flagged_leaves=[], predicates=[],
        panel_models=["m"], separation="independent", rubric_fingerprint="x",
    )
    out = sv.apply_block(_RUBRIC, v)
    assert [l["id"] for l in out["leaves"]] == ["L1", "L2"]


def test_apply_block_renormalizes_existing_weights():
    rubric = {"leaves": [
        {"id": "A", "requirement": "x", "weight": 0.2},
        {"id": "B", "requirement": "y", "weight": 0.3},
        {"id": "C", "requirement": "z", "weight": 0.5},
    ]}
    v = sv.SpecValidatorVerdict(
        status="flagged", flagged_leaves=["B"],
        predicates=[sv.SpecPredicateVerdict("hallucinated_leaf", "B", True, "absent")],
        panel_models=["m"], separation="independent", rubric_fingerprint="x",
    )
    out = sv.apply_block(rubric, v)
    ids = [l["id"] for l in out["leaves"]]
    assert ids == ["A", "C"]
    weights = [l["weight"] for l in out["leaves"]]
    assert abs(sum(weights) - 1.0) < 1e-9
    assert weights[0] < weights[1]  # 0.2 : 0.5 ratio preserved


def test_apply_block_never_raises_on_malformed_rubric():
    v = sv.SpecValidatorVerdict(
        status="flagged", flagged_leaves=["X"],
        predicates=[sv.SpecPredicateVerdict("hallucinated_leaf", "X", True, "absent")],
        panel_models=["m"], separation="independent", rubric_fingerprint="x",
    )
    assert sv.apply_block({}, v) == {}
    assert sv.apply_block(None, v) is None
    assert sv.apply_block({"leaves": "not-a-list"}, v) == {"leaves": "not-a-list"}


def test_apply_block_only_drops_hallucinated_and_wrong_target():
    """A confirmed placeholder_leaf predicate is NOT in apply_block's drop set
    (only hallucinated_leaf/wrong_target are droppable per the task spec)."""
    v = sv.SpecValidatorVerdict(
        status="flagged", flagged_leaves=["L1"],
        predicates=[sv.SpecPredicateVerdict("placeholder_leaf", "L1", True, "placeholder")],
        panel_models=["m"], separation="independent", rubric_fingerprint="x",
    )
    out = sv.apply_block(_RUBRIC, v)
    assert [l["id"] for l in out["leaves"]] == ["L1", "L2"]


# ---------------------------------------------------------------------------
# persist_spec_verdict / load_spec_verdict round-trip
# ---------------------------------------------------------------------------


def test_persist_and_load_round_trip(tmp_path):
    verdict = sv.SpecValidatorVerdict(
        status="flagged",
        flagged_leaves=["L2"],
        predicates=[sv.SpecPredicateVerdict("hallucinated_leaf", "L2", True, "absent")],
        panel_models=["grok"],
        separation="independent",
        rubric_fingerprint="abc123",
    )
    sv.persist_spec_verdict(tmp_path, verdict)
    loaded = sv.load_spec_verdict(tmp_path)
    assert loaded is not None
    assert loaded.status == "flagged"
    assert loaded.flagged_leaves == ["L2"]
    assert loaded.separation == "independent"
    assert loaded.rubric_fingerprint == "abc123"
    assert len(loaded.predicates) == 1
    pv = loaded.predicates[0]
    assert pv.predicate == "hallucinated_leaf"
    assert pv.violated is True
    assert pv.leaf_id == "L2"


def test_load_spec_verdict_returns_none_when_absent(tmp_path):
    assert sv.load_spec_verdict(tmp_path) is None


def test_load_spec_verdict_stale_fingerprint_ignored(tmp_path):
    verdict = sv.SpecValidatorVerdict(
        status="clean", flagged_leaves=[], predicates=[],
        panel_models=["m"], separation="independent",
        rubric_fingerprint="stored_fp",
    )
    sv.persist_spec_verdict(tmp_path, verdict)
    assert sv.load_spec_verdict(tmp_path, expect_fingerprint="different_fp") is None


def test_load_spec_verdict_matching_fingerprint_returned(tmp_path):
    verdict = sv.SpecValidatorVerdict(
        status="clean", flagged_leaves=[], predicates=[],
        panel_models=["m"], separation="weak",
        rubric_fingerprint="matching_fp",
    )
    sv.persist_spec_verdict(tmp_path, verdict)
    result = sv.load_spec_verdict(tmp_path, expect_fingerprint="matching_fp")
    assert result is not None
    assert result.rubric_fingerprint == "matching_fp"


def test_persist_spec_verdict_creates_dir(tmp_path):
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    verdict = sv.SpecValidatorVerdict(
        status="unavailable", flagged_leaves=[], predicates=[],
        panel_models=[], separation="unavailable", rubric_fingerprint="",
    )
    sv.persist_spec_verdict(project_dir, verdict)
    assert (project_dir / "rlm_state" / "spec_validation_verdict.json").exists()


def test_persisted_verdict_file_is_valid_json(tmp_path):
    verdict = sv.SpecValidatorVerdict(
        status="flagged", flagged_leaves=["L2"],
        predicates=[sv.SpecPredicateVerdict("hallucinated_leaf", "L2", True, "absent")],
        panel_models=["grok"], separation="independent", rubric_fingerprint="fp_xyz",
    )
    sv.persist_spec_verdict(tmp_path, verdict)
    target = tmp_path / "rlm_state" / "spec_validation_verdict.json"
    data = json.loads(target.read_text())
    assert data["status"] == "flagged"
    assert data["flagged_leaves"] == ["L2"]
    assert data["rubric_fingerprint"] == "fp_xyz"


def test_persisted_verdict_never_contains_paper_text(tmp_path, monkeypatch):
    """Corpus isolation: the persisted verdict file must never carry paper text."""
    marker = "UNIQUE_PAPER_MARKER_XYZ_NEVER_PERSISTED"
    paper_text = f"SDAR improves over GRPO on ALFWorld 84.4 {marker} extra prose here."
    monkeypatch.setattr(sv, "sample_completions",
        lambda *a, **k: ['[{"predicate":"hallucinated_leaf","leaf_id":"L2"}]'])
    verdict = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["grok"],
        rubric=_RUBRIC,
        paper_text=paper_text,
        separation="independent",
    )
    sv.persist_spec_verdict(tmp_path, verdict)
    target = tmp_path / "rlm_state" / "spec_validation_verdict.json"
    raw = target.read_text()
    # The planted marker and a distinctive paper-prose fragment must both be
    # absent — the persisted verdict carries only leaf ids / enums / details.
    assert marker not in raw
    assert "improves over GRPO" not in raw
    assert "extra prose here" not in raw


# ---------------------------------------------------------------------------
# Direct check_* unit coverage (mirrors external_validator's direct-check tests)
# ---------------------------------------------------------------------------


def test_check_hallucinated_leaf_direct():
    assert sv.check_hallucinated_leaf("Report ALFWorld success rate near 84.4", _PAPER) is True
    assert sv.check_hallucinated_leaf("Report ImageNet top-1 accuracy of 99.9", _PAPER) is False


def test_check_hallucinated_leaf_empty_requirement_is_healthy():
    """Nothing distinctive to check -> conservatively healthy (fail-soft)."""
    assert sv.check_hallucinated_leaf("", _PAPER) is True


# ---------------------------------------------------------------------------
# hallucinated_leaf hardening — numeric precision drift (must-fix boundary)
# ---------------------------------------------------------------------------

# A grounded number-only leaf: same value, different textual precision/format
# ("84.4" in the leaf vs "84.40" in the paper). Must NOT be flagged.
_LEAF_NUM_ONLY = "Report success rate of 84.4"
_PAPER_NUM_DRIFT = "SDAR reaches a success rate of 84.40 on the held-out benchmark."


def test_number_only_grounded_leaf_precision_drift_not_flagged():
    """84.4 vs 84.40 (precision/format drift) must NOT flag a number-only leaf.

    Before the fix this was a DESTRUCTIVE false-positive: exact \\b84\\.4\\b
    does not match inside '84.40', overlap 0.0 -> flagged -> droppable.
    """
    assert sv.check_hallucinated_leaf(_LEAF_NUM_ONLY, _PAPER_NUM_DRIFT) is True


def test_hallucinated_and_wrong_target_agree_on_precision_drift():
    """Internal consistency: the SAME 84.4-vs-84.40 pair that
    _wrong_target_violated CLEARS must not be flagged by check_hallucinated_leaf
    (both now share _WRONG_TARGET_REL_TOL)."""
    assert sv._wrong_target_violated(_LEAF_NUM_ONLY, _PAPER_NUM_DRIFT) is False
    assert sv.check_hallucinated_leaf(_LEAF_NUM_ONLY, _PAPER_NUM_DRIFT) is True


def test_number_only_absent_value_still_flagged():
    """A number-only leaf whose value is NOT in the paper (beyond tolerance) is
    still flagged — the tolerance grounds drift, not a genuinely absent number."""
    assert sv.check_hallucinated_leaf("Report success rate of 12.0", _PAPER_NUM_DRIFT) is False


# ---------------------------------------------------------------------------
# hallucinated_leaf — PRECISION-FIRST: only CamelCase/acronym + numbers veto
# ---------------------------------------------------------------------------


def test_procedural_leaves_never_flagged():
    """REGRESSION GUARD (the reviewer's approval condition): ordinary
    procedural leaves with NO CamelCase/acronym entity and NO number must
    never be machine-flagged as hallucinated — their distinctive set is empty
    → overlap 1.0 → grounded. Before the precision-first fix, the lowercase
    length>=4 branch flagged these ("normalization"/"regularizer"/
    "distributions" absent from the paper), which under BLOCK would drop a
    GROUNDED leaf — the destructive direction the brief prioritizes against."""
    for leaf in (
        "Verify correct normalization of inputs",
        "Model generalizes to unseen distributions",
        "Ablation removes the auxiliary regularizer",
    ):
        assert sv.check_hallucinated_leaf(leaf, _PAPER) is True, leaf


def test_lowercase_only_entity_is_accepted_false_negative():
    """A hallucination cited ONLY via lowercase prose entities (no CamelCase/
    acronym, no number) is NOT machine-flagged — an ACCEPTED, brief-sanctioned
    false-negative under precision-first (the LLM nomination + min-aggregation
    still surface it advisorily; a false-positive that drops a grounded leaf is
    worse). 'mujoco'/'halfcheetah' absent from _PAPER, but no strong anchor →
    grounded."""
    assert sv.check_hallucinated_leaf("Report mujoco halfcheetah reward", _PAPER) is True


def test_lowercase_grounded_entity_not_flagged():
    """Consistent with the above: a lowercase-only leaf whose entities DO
    appear in the paper is likewise not flagged (no CamelCase/acronym/number
    anchor to check → grounded)."""
    assert sv.check_hallucinated_leaf("Report webshop and search results", _PAPER) is True


def test_camelcase_entity_still_dominates_generic_prose():
    """A grounded CamelCase+numeric leaf (ALFWorld/84.4) stays grounded even
    surrounded by generic prose — only the strong anchors are checked, both
    present."""
    assert sv.check_hallucinated_leaf(
        "Verify that the reported ALFWorld success rate is approximately 84.4", _PAPER
    ) is True


def test_lowercase_only_entity_not_flagged_via_panel(monkeypatch):
    """End-to-end: even when a panel POINTS at a lowercase-only leaf, the
    machine-check clears it (precision-first) → status clean, no veto. The
    LLM's nomination is never dispositive."""
    rubric = {"leaves": [{"id": "E1", "requirement": "Report mujoco halfcheetah return"}]}
    monkeypatch.setattr(sv, "sample_completions",
        lambda *a, **k: ['[{"predicate":"hallucinated_leaf","leaf_id":"E1"}]'])
    v = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["m"],
        rubric=rubric,
        paper_text=_PAPER,
        separation="independent",
    )
    assert v.status == "clean"
    assert "E1" not in v.flagged_leaves


# ---------------------------------------------------------------------------
# Minor 2 — structural leaf_id cap (corpus isolation hardening)
# ---------------------------------------------------------------------------


def test_leaf_id_capped_in_verdict_and_persisted(tmp_path, monkeypatch):
    """An over-long LLM-echoed leaf_id is truncated to <= 64 chars at ingest,
    so no verdict/persisted field can carry a long model-emitted span."""
    long_id = "MISSING:" + "z" * 200
    monkeypatch.setattr(sv, "sample_completions",
        lambda *a, **k: [json.dumps([{"predicate": "missing_key_claim", "leaf_id": long_id}])])
    v = sv.run_spec_validation_panel(
        spec_validator_client=_FakeClient(None),
        panel_models=["m"],
        rubric=_RUBRIC,
        paper_text=_PAPER,
        separation="independent",
    )
    assert all(len(x) <= 64 for x in v.flagged_leaves)
    assert all(len(p.leaf_id) <= 64 for p in v.predicates)
    sv.persist_spec_verdict(tmp_path, v)
    raw = (tmp_path / "rlm_state" / "spec_validation_verdict.json").read_text()
    assert long_id not in raw  # the uncapped original never reaches disk


def test_check_missing_key_claim_always_false_stub():
    """Advisory-only stub: always returns False (never 'healthy') so the
    predicate is always recorded — apply_block is what keeps it non-destructive."""
    assert sv.check_missing_key_claim() is False
