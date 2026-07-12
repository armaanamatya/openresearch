"""Track E Task 7: skill-as-reference composition + its leniency guard.

The skill-select artifact (``rlm_state/active_skills.json``) already exists
to steer the LLM implementer/verifier (spec §6.4). This module reshapes it
into an eval-facing reference STRUCTURE
(``{expected_metric_families, standard_baselines, eval_protocol,
dataset_expectations}``) — never a pass/fail. The north-star invariant this
suite locks: **a skill can supply structure, never a pass** — a skill
reference can never flip a ``result_fidelity`` per-claim ``status`` (which
keys SOLELY on measured on-disk artifacts + a scope-verified bind; the only
three values are ``"pass" | "fail" | "unmeasured"``).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from backend.agents.rlm import result_fidelity
from backend.evals.reference_from_skills import compose_reference

_REPO_ROOT = Path(__file__).resolve().parents[1]

_REFERENCE_KEYS = (
    "expected_metric_families",
    "standard_baselines",
    "eval_protocol",
    "dataset_expectations",
)

_FIXTURE_ARTIFACT = {
    "selected": ["sdar-reproduction", "gcp-gke-reproduction"],
    "candidates": [
        {
            "name": "sdar-reproduction",
            "category": "paper-reproduction",
            "description": "SDAR self-distilled agentic RL playbook.",
            "reason": "domain match: agentic RL",
        },
        {
            "name": "gcp-gke-reproduction",
            "category": "cloud-compute",
            "description": "GKE cell-matrix operator preflight.",
            "reason": "infra skill for sandbox=gcp",
        },
    ],
    "domain": "rl-training",
    "subject_matter_keys": {
        "datasets": ["ALFWorld", "WebShop"],
        "metrics": ["success_rate", "reward"],
        "methods": ["GRPO", "OPSD"],
        "frameworks": ["vllm", "torch"],
    },
    "selector": "deterministic+llm",
    "reasons": {
        "sdar-reproduction": "matches agentic RL domain",
        "gcp-gke-reproduction": "infra skill for sandbox=gcp",
    },
}


def _write_active_skills(project_dir: Path, artifact: dict) -> None:
    rlm_state = project_dir / "rlm_state"
    rlm_state.mkdir(parents=True, exist_ok=True)
    (rlm_state / "active_skills.json").write_text(json.dumps(artifact), encoding="utf-8")


# ---------------------------------------------------------------------------
# compose_reference: structure derived from a fixture active_skills.json
# ---------------------------------------------------------------------------


def test_compose_reference_returns_structure_from_fixture(tmp_path):
    _write_active_skills(tmp_path, _FIXTURE_ARTIFACT)

    ref = compose_reference(tmp_path)

    assert set(ref.keys()) == set(_REFERENCE_KEYS)

    metric_values = {e["value"] for e in ref["expected_metric_families"]}
    assert metric_values == {"success_rate", "reward"}

    dataset_values = {e["value"] for e in ref["dataset_expectations"]}
    assert dataset_values == {"ALFWorld", "WebShop"}

    protocol_values = {e["value"] for e in ref["eval_protocol"]}
    assert protocol_values == {"GRPO", "OPSD"}

    baseline_values = {e["value"] for e in ref["standard_baselines"]}
    assert baseline_values == {"sdar-reproduction", "gcp-gke-reproduction"}


def test_compose_reference_every_entry_tagged_evaluator_computed(tmp_path):
    _write_active_skills(tmp_path, _FIXTURE_ARTIFACT)

    ref = compose_reference(tmp_path)

    all_entries = [entry for key in _REFERENCE_KEYS for entry in ref[key]]
    assert all_entries, "fixture must produce at least one entry per family"
    for entry in all_entries:
        assert entry["provenance"] == "evaluator_computed"


def test_compose_reference_standard_baselines_trace_their_source_skill(tmp_path):
    _write_active_skills(tmp_path, _FIXTURE_ARTIFACT)

    ref = compose_reference(tmp_path)

    by_value = {e["value"]: e for e in ref["standard_baselines"]}
    assert by_value["sdar-reproduction"]["source_skill"] == "sdar-reproduction"
    assert by_value["gcp-gke-reproduction"]["source_skill"] == "gcp-gke-reproduction"

    # Paper-subject-matter-derived families are not attributed to one skill.
    for entry in ref["expected_metric_families"] + ref["dataset_expectations"]:
        assert entry["source_skill"] is None


def test_compose_reference_absent_active_skills_is_well_formed_empty(tmp_path):
    # No rlm_state/ directory at all.
    ref = compose_reference(tmp_path)
    assert ref == {key: [] for key in _REFERENCE_KEYS}


def test_compose_reference_malformed_json_fail_soft(tmp_path):
    rlm_state = tmp_path / "rlm_state"
    rlm_state.mkdir(parents=True)
    (rlm_state / "active_skills.json").write_text("{not valid json", encoding="utf-8")

    ref = compose_reference(tmp_path)
    assert ref == {key: [] for key in _REFERENCE_KEYS}


def test_compose_reference_non_dict_json_fail_soft(tmp_path):
    rlm_state = tmp_path / "rlm_state"
    rlm_state.mkdir(parents=True)
    (rlm_state / "active_skills.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    ref = compose_reference(tmp_path)
    assert ref == {key: [] for key in _REFERENCE_KEYS}


def test_compose_reference_never_carries_a_status_or_verdict_key(tmp_path):
    # Structural shape check: every entry is exactly {value, provenance,
    # source_skill} — no status/pass/fail/verdict field ever sneaks in.
    _write_active_skills(tmp_path, _FIXTURE_ARTIFACT)

    ref = compose_reference(tmp_path)

    banned_keys = {"status", "pass", "fail", "verdict", "meets_target", "expected_status"}
    for key in _REFERENCE_KEYS:
        for entry in ref[key]:
            assert set(entry.keys()) == {"value", "provenance", "source_skill"}
            assert not (set(entry.keys()) & banned_keys)


# ---------------------------------------------------------------------------
# Static leniency guard: no import-level path from this module to any
# status-writing module (mirrors Track E Task 1's verdict-surface guard).
# Parses real `import`/`from ... import` statements via ``ast`` rather than a
# raw substring scan, so accurately documenting the invariant in this
# module's own docstring (which necessarily names "result_fidelity") can
# never itself trip the guard — only an actual import would.
# ---------------------------------------------------------------------------


def _imported_module_names(src: str) -> set[str]:
    tree = ast.parse(src)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            names.update(alias.name for alias in node.names)
    return names


def test_module_never_imports_result_fidelity_or_verdict_modules():
    src = (_REPO_ROOT / "backend" / "evals" / "reference_from_skills.py").read_text(
        encoding="utf-8"
    )
    imported = _imported_module_names(src)
    banned_tokens = ("result_fidelity", "verdict_authority", "campaign_policy")
    for name in imported:
        for token in banned_tokens:
            assert token not in name, (
                f"reference_from_skills.py must not import {name!r} (contains banned "
                f"token {token!r}) — a skill reference feeds only display/structure, "
                "never a claim status (spec §6.4 leniency guard)"
            )


# ---------------------------------------------------------------------------
# THE LOAD-BEARING TEST: a skill reference cannot flip an unmeasured claim
# to "pass". result_fidelity per-claim status is a 3-value taxonomy —
# "pass" | "fail" | "unmeasured" — and keys SOLELY on measured on-disk
# artifacts + a scope-verified bind.
# ---------------------------------------------------------------------------


def test_skill_reference_cannot_flip_qualitative_claim_to_pass(tmp_path):
    # An adversarial active_skills.json: a free-text "reason" that itself
    # asserts an expected pass. compose_reference must not propagate this
    # prose into the reference at all (it only reads name/list fields).
    artifact = dict(_FIXTURE_ARTIFACT)
    artifact["reasons"] = {"sdar-reproduction": "accuracy should pass at 0.99"}
    _write_active_skills(tmp_path, artifact)
    reference = compose_reference(tmp_path)

    run_dir = tmp_path / "run"
    (run_dir / "code").mkdir(parents=True)
    # No metrics.json at all — nothing is measurable regardless of claim kind.

    claim = {
        "claim_id": "c1",
        "kind": "qualitative",  # always unmeasured, checked before any bind/measure
        "metric_name": "accuracy",
        "is_primary": True,
        "claimed_effect": 0.99,
        "equivalence_margin": 0.01,
        # Adversarial injection: attach this module's own composed reference,
        # plus forged pass/status fields straight onto the claim — the exact
        # shape a poisoned consumer might expect result_fidelity to read.
        "skill_reference": reference,
        "expected_status": "pass",
        "status": "pass",
    }
    repro_spec = {"claims": [claim]}

    result = result_fidelity.evaluate(repro_spec, run_dir)

    assert len(result["per_claim"]) == 1
    per_claim = result["per_claim"][0]
    assert per_claim["status"] == "unmeasured"
    assert per_claim["status"] in ("pass", "fail", "unmeasured")  # the 3-value taxonomy
    assert per_claim["status"] != "pass"


def test_skill_reference_cannot_flip_unbound_numeric_claim_to_pass(tmp_path):
    # Second variant of the load-bearing test using an "unbound" (rather than
    # qualitative) claim: the metric genuinely cannot be bound to anything in
    # metrics.json, so status resolves to "unmeasured" post-bind.
    _write_active_skills(tmp_path, _FIXTURE_ARTIFACT)
    reference = compose_reference(tmp_path)

    run_dir = tmp_path / "run"
    code_dir = run_dir / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "metrics.json").write_text(
        json.dumps({"per_model": {"modelA": {"envA": {"accuracy": 0.5}}}}),
        encoding="utf-8",
    )

    claim = {
        "claim_id": "c2",
        "kind": "numeric",
        "metric_name": "totally_unrelated_metric_xyz",
        "scope": {},
        "is_primary": True,
        "claimed_effect": 0.99,
        "equivalence_margin": 0.01,
        "skill_reference": reference,
        "expected_status": "pass",
        "status": "pass",
    }
    repro_spec = {"claims": [claim]}

    result = result_fidelity.evaluate(repro_spec, run_dir)

    assert len(result["per_claim"]) == 1
    per_claim = result["per_claim"][0]
    assert per_claim["status"] == "unmeasured"
    assert per_claim["status"] != "pass"


def test_skill_reference_attached_at_repro_spec_level_still_cannot_flip_status(tmp_path):
    # Third variant: the reference lives at the repro_spec top level (rather
    # than nested onto the claim) — still no code path to per-claim status.
    _write_active_skills(tmp_path, _FIXTURE_ARTIFACT)
    reference = compose_reference(tmp_path)

    run_dir = tmp_path / "run"
    (run_dir / "code").mkdir(parents=True)

    repro_spec = {
        "skill_reference": reference,
        "claims": [
            {
                "claim_id": "c3",
                "kind": "qualitative",
                "metric_name": "reward",
                "is_primary": True,
                "claimed_effect": 1.0,
                "equivalence_margin": 0.0,
            }
        ],
    }

    result = result_fidelity.evaluate(repro_spec, run_dir)

    assert len(result["per_claim"]) == 1
    assert result["per_claim"][0]["status"] == "unmeasured"
