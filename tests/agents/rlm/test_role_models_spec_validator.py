"""spec_validator sub-role — contract tests mirroring the ``validator`` role.

Task 4 (autonomous-upload-ui SDD): the ``spec_validator`` sub-role is the
resolver plumbing for a later rubric-vs-paper pre-loop validator (T6's
``spec_validator.py`` + T5's ``build_spec_validator_client`` transport + T8's
``run.py`` hook — none of that lands here). This module is pure resolution,
mirroring the existing ``validator`` sub-role (spec 2026-06-20 §7.2/§7.4,
``tests/agents/rlm/test_role_models_validator.py``) EXACTLY:

  * the sub-role resolves from the unified surface + a legacy
    ``spec_validator_model_setting`` feeder, and ``RoleSelection.stamp()``
    gains a ``spec_validator`` key;
  * the §4.7-style stamp preference — ``OPENRESEARCH_SPEC_VALIDATOR_MODEL``
    (the deployment T5's transport will actually target) takes precedence over
    the role's own ``model`` for a bridged azure-foundry pick;
  * cross-role leak guards: ``OPENRESEARCH_SPEC_VALIDATOR_MODEL`` must not
    leak onto sibling roles, and ``OPENRESEARCH_VALIDATOR_MODEL`` must not
    leak onto ``spec_validator`` (the two env-preference branches are
    independent ``if``/``elif`` arms keyed on ``self.role``);
  * no regression: adding ``spec_validator`` does not change
    grader/verifier/validator/executor behaviour, and it participates in the
    advisory fidelity-warning surface exactly like its sibling sub-roles.
"""
from __future__ import annotations

import pytest

from backend.agents.rlm.role_models import (
    PROVIDER_ANTHROPIC_OAUTH,
    PROVIDER_AZURE,
    PROVIDER_AZURE_FOUNDRY,
    RoleModelError,
    parse_model_spec,
    resolve_role_models,
)


# ---------------------------------------------------------------------------
# 1 (load-bearing membership, cheap direct guard). ``spec_validator`` is a
# known ROLE and a strict sub-role (unknown token raises) — the two structural
# edits (#1 ``ROLES`` tuple, #2 ``_SUBROLES`` frozenset) this whole file
# depends on.
# ---------------------------------------------------------------------------
def test_spec_validator_is_a_known_role():
    from backend.agents.rlm.role_models import ROLES

    assert "spec_validator" in ROLES


def test_spec_validator_is_a_subrole():
    # Strict-parse path: an unknown token for spec_validator must raise (same
    # contract as executor/verifier/grader/validator), proving it is a
    # sub-role, not silently falling through to the lenient planner passthrough.
    with pytest.raises(RoleModelError):
        parse_model_spec("not-a-real-token", role="spec_validator")


# ---------------------------------------------------------------------------
# 5. spec_validator sub-role resolution + stamp + legacy feeder.
# ---------------------------------------------------------------------------
def test_spec_validator_inherits_none_by_default():
    sel = resolve_role_models(planner_token="claude-oauth")
    assert sel.spec_validator is None
    assert sel.stamp()["spec_validator"] is None


def test_spec_validator_from_unified_surface():
    sel = resolve_role_models(
        planner_token="claude-oauth", cli_models="spec_validator=grok"
    )
    assert sel.spec_validator is not None
    assert sel.spec_validator.provider == PROVIDER_AZURE_FOUNDRY
    assert "spec_validator" in sel.stamp()
    assert sel.spec_validator in sel.explicit_subroles.values()


def test_spec_validator_legacy_model_setting_feeder():
    # OPENRESEARCH_SPEC_VALIDATOR_MODEL (threaded as spec_validator_model_setting)
    # sets it when the unified surface does not.
    sel = resolve_role_models(
        planner_token="claude-oauth", spec_validator_model_setting="sonnet"
    )
    assert sel.spec_validator is not None
    assert sel.spec_validator.provider == PROVIDER_ANTHROPIC_OAUTH
    assert sel.spec_validator.family == "claude"


@pytest.mark.parametrize("sv", ["", "   ", None])
def test_spec_validator_model_setting_blank_stays_none(sv):
    sel = resolve_role_models(
        planner_token="claude-oauth", spec_validator_model_setting=sv
    )
    assert sel.spec_validator is None


def test_spec_validator_unified_beats_legacy_setting():
    sel = resolve_role_models(
        planner_token="claude-oauth",
        cli_models="spec_validator=gpt-4o-azure",
        spec_validator_model_setting="sonnet",
    )
    # CLI surface wins over the legacy feeder.
    assert sel.spec_validator.provider == PROVIDER_AZURE


def test_spec_validator_stamp_present_in_full_shape():
    sel = resolve_role_models(
        planner_token="claude-oauth", cli_models="spec_validator=gpt-4o-azure"
    )
    stamp = sel.stamp()
    assert "spec_validator" in stamp
    assert stamp["spec_validator"] == "azure:gpt-4o"


# ---------------------------------------------------------------------------
# 5b. The stamp prefers OPENRESEARCH_SPEC_VALIDATOR_MODEL, mirroring the
# validator's §4.7 fix — and the two env preferences must not cross-leak.
# ---------------------------------------------------------------------------
def test_spec_validator_stamp_prefers_spec_validator_model_env(monkeypatch):
    # A bridged azure-foundry spec_validator (model None) would otherwise stamp
    # the global AZURE_FOUNDRY_DEPLOYMENT (the executor's model). With
    # OPENRESEARCH_SPEC_VALIDATOR_MODEL set, the stamp must name that model.
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_MODEL", "grok-4.3-specval")
    spec = parse_model_spec("grok", role="spec_validator")
    assert spec.provider == PROVIDER_AZURE_FOUNDRY
    assert spec.stamp == "azure-foundry:grok-4.3-specval"


def test_spec_validator_model_env_does_not_leak_to_other_roles(monkeypatch):
    # The env preference is keyed on role == "spec_validator" only — a
    # grader/validator foundry RoleSpec must NOT pick up
    # OPENRESEARCH_SPEC_VALIDATOR_MODEL.
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_MODEL", "grok-4.3-specval")
    grader = parse_model_spec("grok", role="grader")
    assert grader.role == "grader"
    assert grader.stamp != "azure-foundry:grok-4.3-specval"
    validator = parse_model_spec("grok", role="validator")
    assert validator.role == "validator"
    assert validator.stamp != "azure-foundry:grok-4.3-specval"


def test_validator_model_env_does_not_leak_to_spec_validator(monkeypatch):
    # The reverse leak: OPENRESEARCH_VALIDATOR_MODEL (the sibling validator's
    # own env preference) must NOT be picked up by spec_validator.
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "grok-4.3-val")
    spec_validator = parse_model_spec("grok", role="spec_validator")
    assert spec_validator.role == "spec_validator"
    assert spec_validator.stamp != "azure-foundry:grok-4.3-val"


def test_spec_validator_stamp_ignores_blank_spec_validator_model_env(monkeypatch):
    # A blank/whitespace OPENRESEARCH_SPEC_VALIDATOR_MODEL falls through to the
    # normal deployment-resolution path (no spurious "azure-foundry:" stamp).
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_MODEL", "   ")
    spec = parse_model_spec("gpt-4o-azure", role="spec_validator")
    # Concrete model present → normal stamp, env ignored.
    assert spec.stamp == "azure:gpt-4o"


def test_roleselection_spec_validator_stamp_inherits_env_preference(monkeypatch):
    # The aggregator delegates to RoleSpec.stamp, so the env preference flows
    # through RoleSelection.stamp() too.
    monkeypatch.setenv("OPENRESEARCH_SPEC_VALIDATOR_MODEL", "gpt-4o-specvalB")
    sel = resolve_role_models(
        planner_token="claude-oauth", cli_models="spec_validator=gpt-4o-azure"
    )
    assert sel.stamp()["spec_validator"] == "azure:gpt-4o-specvalB"


# ---------------------------------------------------------------------------
# 6. No regression — adding spec_validator must not change
# grader/verifier/executor/validator, and it joins the advisory fidelity
# surface like its sibling sub-roles.
# ---------------------------------------------------------------------------
def test_grader_verifier_validator_unchanged_when_no_spec_validator():
    sel = resolve_role_models(
        planner_token="claude-oauth",
        cli_models="verifier=sonnet,grader=gpt-4o-azure,validator=sonnet",
    )
    assert sel.verifier is not None and sel.verifier.provider == PROVIDER_ANTHROPIC_OAUTH
    assert sel.grader is not None and sel.grader.provider == PROVIDER_AZURE
    assert sel.validator is not None and sel.validator.provider == PROVIDER_ANTHROPIC_OAUTH
    assert sel.spec_validator is None


def test_back_compat_stamp_shape_now_carries_spec_validator_key():
    # The legacy 5-key stamp (planner/executor/verifier/grader/validator) gains
    # a 6th "spec_validator" key (None when unselected); the five existing keys
    # are byte-identical to before.
    sel = resolve_role_models(planner_token="claude-oauth")
    assert sel.stamp() == {
        "planner": "anthropic-oauth:claude-sonnet-4-6",
        "executor": None,
        "verifier": None,
        "grader": None,
        "validator": None,
        "spec_validator": None,
    }


def test_fidelity_warning_covers_a_non_claude_spec_validator():
    # spec_validator participates in the advisory fidelity surface like the
    # other sub-roles (explicit_subroles now includes it).
    sel = resolve_role_models(
        planner_token="claude-oauth", cli_models="spec_validator=gpt-4o-azure"
    )
    warnings = sel.fidelity_warnings(fidelity_critical=True)
    assert any("spec_validator" in w for w in warnings)
    # ... and is silent on a Claude spec_validator.
    sel_claude = resolve_role_models(
        planner_token="claude-oauth", cli_models="spec_validator=sonnet"
    )
    assert sel_claude.fidelity_warnings(fidelity_critical=True) == []
