"""2026-06-09 recurring-issue fixes: preflight harness-import rule,
requirements sanitizer, dataset-coverage matching, grounding name extraction,
runtime-capacity env assumption, and agent-visible compute_scope feedback.

Each test pins one of the failure classes observed on every Adam/All-CNN
attempt (see docs/runbooks; the issues recurred across BOTH papers, i.e. they
are harness-generic, not paper-specific).
"""

from __future__ import annotations

import json
import textwrap

from backend.agents.paper_grounding import assert_paper_grounded
from backend.agents.rlm.env_pin import sanitize_requirements
from backend.agents.rlm.preflight_ast import scan_code_dir
from backend.agents.rlm.primitives import _validate_scope_metrics, plan_reproduction
from backend.agents.schemas import ScopeSpec


# ------------------------------------------------- preflight: harness import

def _scan(tmp_path, source: str):
    (tmp_path / "train.py").write_text(textwrap.dedent(source))
    return scan_code_dir(tmp_path)


def test_unguarded_backend_import_is_hard_violation(tmp_path):
    violations = _scan(tmp_path, """
        import backend.agents.rlm.rubric_guard as rg
        print(rg)
    """)
    assert any(
        v.severity == "hard" and "backend" in v.detail for v in violations
    ), violations


def test_unguarded_from_backend_import_is_flagged(tmp_path):
    violations = _scan(tmp_path, """
        from backend.agents.rlm.rubric_guard import assert_metrics_schema
    """)
    assert any("backend" in v.detail for v in violations)


def test_guarded_copy_helper_pattern_passes(tmp_path):
    # The sanctioned pattern used by every harness-copied helper.
    violations = _scan(tmp_path, """
        try:
            from rubric_guard import RubricGuardFailure
        except ImportError:
            from backend.agents.rlm.rubric_guard import RubricGuardFailure
    """)
    assert not [v for v in violations if "backend" in (v.detail or "")]


def test_unrelated_backend_like_names_not_flagged(tmp_path):
    violations = _scan(tmp_path, """
        import backends_db
        from my_backend import thing
    """)
    assert not [v for v in violations if "harness" in (v.suggested_fix or "")]


# ---------------------------------------------- requirements.txt sanitizer

def test_sanitize_drops_prose_keeps_valid():
    lines = [
        "torch==2.2.0",
        "(Section",                       # the 2026-06-07 Adam killer
        "torchvision>=0.17",
        "Based on the paper we need scipy",  # prose
        "numpy",
        "# a comment",
        "",
        "-r extra.txt",
        "git+https://github.com/x/y.git",
        "pkg @ https://example.com/pkg.whl",
        "scikit-learn[alldeps]~=1.4 ; python_version >= '3.10'",
    ]
    kept, invalid = sanitize_requirements(lines)
    assert invalid == ["(Section", "Based on the paper we need scipy"]
    assert "torch==2.2.0" in kept and "numpy" in kept
    assert "-r extra.txt" in kept and "git+https://github.com/x/y.git" in kept
    assert "pkg @ https://example.com/pkg.whl" in kept
    assert "scikit-learn[alldeps]~=1.4 ; python_version >= '3.10'" in kept
    assert "# a comment" in kept and "" in kept


# ------------------------------------------------ dataset-coverage matching

def test_env_keyed_dataset_coverage_accepted():
    # All-CNN/Adam shape: env keys carry the dataset name plus a qualifier.
    scope = ScopeSpec(datasets=["MNIST", "CIFAR-10"])
    metrics = {"per_dataset": {"mnist_mlp": {"acc": 0.99}, "cifar10_noaug": {"acc": 0.89}}}
    assert _validate_scope_metrics(scope, metrics) is None


def test_cifar100_does_not_satisfy_cifar10():
    scope = ScopeSpec(datasets=["CIFAR-10", "CIFAR-100"])
    metrics = {"per_dataset": {"cifar100": {"acc": 0.6}}}
    hint = _validate_scope_metrics(scope, metrics)
    assert hint is not None and "CIFAR-10" in hint


def test_multi_model_env_keys_cover_datasets():
    scope = ScopeSpec(models=["a", "b"], datasets=["MNIST", "IMDB"])
    metrics = {"per_model": {
        "a": {"mnist_logreg": {"acc": 0.9}, "imdb_bow": {"acc": 0.88}},
        "b": {"mnist": {"acc": 0.9}, "imdb": {"acc": 0.86}},
    }}
    assert _validate_scope_metrics(scope, metrics) is None


# ------------------------------------------------ grounding name extraction

_PAPER = "We evaluate Adam on MNIST and CIFAR-10 using logistic regression."


def test_dict_entries_reduce_to_name_field():
    cm = {"datasets": [{"name": "MNIST", "source": "torchvision"}]}
    assert assert_paper_grounded(cm, _PAPER) == []


def test_serialized_dict_string_reduces_to_name():
    cm = {"datasets": ["[{'name': 'CIFAR-10', 'source': 'torchvision'}]"]}
    assert assert_paper_grounded(cm, _PAPER) == []


def test_prose_values_are_skipped_not_flagged():
    cm = {"datasets": ["Based on the provided excerpt:\n\n**ML datasets**"]}
    assert assert_paper_grounded(cm, _PAPER) == []


def test_genuinely_unfounded_short_name_still_flagged():
    cm = {"datasets": ["ImageNet-21k"]}
    violations = assert_paper_grounded(cm, _PAPER)
    assert len(violations) == 1 and violations[0].value == "ImageNet-21k"


def test_claim_field_dict_value_extracted():
    cm = {"claims": [{"dataset": {"name": "MNIST"}, "method": "Adam", "metric": "nll"}]}
    violations = assert_paper_grounded(cm, _PAPER)
    assert all(v.field != "dataset" for v in violations)


# --------------------------------------- detect_environment runtime capacity

def test_detect_environment_appends_runtime_gpu_assumption(
    make_context, tmp_path, monkeypatch
):
    from backend.services.runtime import gpu_capacity as gc

    fake = gc.GpuCapacity(
        backend_kind="local", num_gpus=2, per_gpu_vram_gb=24.0,
        free_gpu_ids=("GPU-aaa", "GPU-bbb"), can_escalate=False,
        total_vram_gb=48.0,
    )
    monkeypatch.setattr(gc, "describe_capacity", lambda ctx: fake)

    from backend.agents.rlm.primitives import detect_environment

    ctx = make_context(tmp_path)
    method_spec = {"core_contribution": "A PyTorch CNN.", "claims": [],
                   "datasets": [], "metrics": []}
    result = detect_environment(method_spec, ctx=ctx)
    rt = [a for a in result["assumptions"] if a.get("assumption_id") == "ENV-RT1"]
    assert len(rt) == 1
    assert "2× CUDA GPU" in rt[0]["chosen_value"]
    assert "24 GB" in rt[0]["chosen_value"]
    # on-disk spec stays consistent with the returned dict
    on_disk = json.loads((ctx.project_dir / "environment_spec.json").read_text())
    assert any(a.get("assumption_id") == "ENV-RT1" for a in on_disk["assumptions"])


def test_detect_environment_no_gpus_no_annotation(make_context, tmp_path, monkeypatch):
    from backend.services.runtime import gpu_capacity as gc

    fake = gc.GpuCapacity(
        backend_kind="local", num_gpus=0, per_gpu_vram_gb=0.0,
        free_gpu_ids=(), can_escalate=False,
    )
    monkeypatch.setattr(gc, "describe_capacity", lambda ctx: fake)

    from backend.agents.rlm.primitives import detect_environment

    ctx = make_context(tmp_path)
    result = detect_environment(
        {"core_contribution": "X", "claims": [], "datasets": [], "metrics": []}, ctx=ctx
    )
    assert not [a for a in result["assumptions"] if a.get("assumption_id") == "ENV-RT1"]


# ------------------------------------------------ compute_scope feedback

_CONTRACT_WITH_PROSE_SCOPE = json.dumps({
    "reproduction_definition": "Same algorithm, same dataset.",
    "smoke_test_plan": "1000 steps.",
    "full_run_plan": "full.",
    "expected_outputs": ["metrics.json"],
    "evaluation_plan": "eval.",
    "compute_scope": "CPU-only (per ENV003), single machine, no GPU.",
})


def test_plan_reproduction_string_compute_scope_warns_the_agent(make_context, tmp_path):
    ctx = make_context(tmp_path, llm_responses=[_CONTRACT_WITH_PROSE_SCOPE])
    result = plan_reproduction(
        {"core_contribution": "X"}, {"framework": "pytorch"}, ctx=ctx
    )
    assert result["compute_scope"] is None
    warnings = result.get("warnings") or []
    assert any("compute_scope" in w and "is_clipped" in w for w in warnings)
