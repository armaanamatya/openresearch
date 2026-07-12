"""Track E Task 6: the 11-dimension scorecard dimension-mapper.

``build_scorecard`` maps each of the 11 evaluator dimensions (spec §6.1) onto a
:class:`~backend.evals.evaluation_report.ScorecardRow`, keyed ONLY on EXISTING
deterministic signals already produced elsewhere in the harness (never a new
detector, never a fabricated pass):

  * GATE rows (``gates=True``) key on artifact-anchored evidence and default
    to ``"unmeasured"`` (never auto-``"pass"``) when that artifact is absent:
    ``numerical_reproduction`` (result_fidelity/repro_spec.json),
    ``execution_completeness`` (``_has_experiment_evidence`` + ok-receipt
    count), ``environment_fidelity`` (env_health.jsonl exclusions),
    ``dataset_availability`` (data-unavailable leaves),
    ``tables_figures`` (``fig_*.json`` sidecars, GATE-lite -- never "fail").
  * DISPLAY rows (``gates=False``) are ALWAYS ``status="display"`` and carry
    an empty ``detail`` when their signal is absent -- they can never
    contribute a verdict gate cap (:meth:`EvaluationReport.gate_caps`):
    ``autonomy``, ``efficiency``, ``paper_understanding``, ``dag_planning``,
    ``debugging``, ``scientific_analysis``.

Every fixture in this file is a bare ``tmp_path`` -- no production ``runs/``
directory is ever read or mutated.
"""

import json

from backend.evals.evaluation_report import EvaluationReport
from backend.evals.scorecard import (
    DIMENSIONS,
    DISPLAY_DIMENSIONS,
    GATE_DIMENSIONS,
    build_scorecard,
)


def _row(rows, dimension):
    matches = [r for r in rows if r.dimension == dimension]
    assert len(matches) == 1, f"expected exactly one {dimension!r} row, got {len(matches)}"
    return matches[0]


# --------------------------------------------------------------------------- #
# Shape / structural invariants
# --------------------------------------------------------------------------- #

def test_build_scorecard_returns_all_11_dimensions_on_an_empty_run(tmp_path):
    rows = build_scorecard(tmp_path)
    assert len(rows) == 11 == len(DIMENSIONS)
    assert {r.dimension for r in rows} == set(DIMENSIONS)


def test_gate_rows_key_on_artifacts_display_rows_never_gate(tmp_path):
    rows = build_scorecard(tmp_path)
    for dim in GATE_DIMENSIONS:
        r = _row(rows, dim)
        assert r.gates is True, f"{dim} must be a GATE row"
    for dim in DISPLAY_DIMENSIONS:
        r = _row(rows, dim)
        assert r.gates is False, f"{dim} must never gate"
        assert r.status == "display", f"{dim} must always report status='display'"


def test_display_row_status_is_always_the_literal_string_display(tmp_path):
    # Even once its signal artifact IS present, a DISPLAY row's status stays
    # "display" (never "pass"/"fail") -- only detail/evidence_refs vary.
    (tmp_path / "human_interventions.jsonl").write_text(
        json.dumps(
            {
                "ts": "t",
                "kind": "credentials",
                "what": "x",
                "why": "",
                "artifact_diff": "",
                "blocking": True,
            }
        )
        + "\n"
    )
    r = _row(build_scorecard(tmp_path), "autonomy")
    assert r.status == "display" and r.gates is False
    assert r.detail != ""


# --------------------------------------------------------------------------- #
# GATE row: numerical_reproduction (result_fidelity / repro_spec.json)
# --------------------------------------------------------------------------- #

def _claim(**kw):
    base = {
        "claim_id": "primary_0",
        "is_primary": True,
        "kind": "numeric",
        "metric_name": "accuracy",
        "estimate_kind": "point",
        "claimed_effect": 0.99,
        "equivalence_margin": 0.01,
        "direction": "higher_is_better",
        "scope": {},
        "ambiguous": False,
    }
    base.update(kw)
    return base


def _write_repro_spec(project_dir, claims):
    state = project_dir / "rlm_state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "repro_spec.json").write_text(json.dumps({"claims": claims}))


def _write_metrics(project_dir, metrics):
    code = project_dir / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "metrics.json").write_text(json.dumps(metrics))


def test_numerical_reproduction_unmeasured_when_repro_spec_missing(tmp_path):
    r = _row(build_scorecard(tmp_path), "numerical_reproduction")
    assert r.status == "unmeasured"
    assert r.gates is True


def test_numerical_reproduction_pass_when_primary_claim_measures_within_margin(tmp_path):
    _write_metrics(tmp_path, {"accuracy": 0.991})
    _write_repro_spec(tmp_path, [_claim()])
    r = _row(build_scorecard(tmp_path), "numerical_reproduction")
    assert r.status == "pass"


def test_numerical_reproduction_fail_when_primary_claim_contradicted(tmp_path):
    _write_metrics(tmp_path, {"accuracy": 0.80})
    _write_repro_spec(tmp_path, [_claim()])
    r = _row(build_scorecard(tmp_path), "numerical_reproduction")
    assert r.status == "fail"


def test_numerical_reproduction_never_auto_passes_on_empty_claims(tmp_path):
    _write_repro_spec(tmp_path, [])
    r = _row(build_scorecard(tmp_path), "numerical_reproduction")
    assert r.status == "unmeasured"


# --------------------------------------------------------------------------- #
# GATE row: execution_completeness (_has_experiment_evidence + ok-receipt)
# --------------------------------------------------------------------------- #

def test_execution_completeness_unmeasured_when_no_experiment_runs(tmp_path):
    r = _row(build_scorecard(tmp_path), "execution_completeness")
    assert r.status == "unmeasured"


def test_execution_completeness_pass_on_clean_success_with_metrics(tmp_path):
    (tmp_path / "experiment_runs.jsonl").write_text(
        json.dumps({"success": True, "metrics": {"accuracy": 0.9}}) + "\n"
    )
    r = _row(build_scorecard(tmp_path), "execution_completeness")
    assert r.status == "pass"


def test_execution_completeness_fail_when_every_row_failed(tmp_path):
    (tmp_path / "experiment_runs.jsonl").write_text(
        json.dumps({"success": False, "metrics": {}}) + "\n"
    )
    r = _row(build_scorecard(tmp_path), "execution_completeness")
    assert r.status == "fail"


def test_execution_completeness_pass_via_ok_receipt_out_of_process_fallback(tmp_path):
    (tmp_path / "experiment_runs.jsonl").write_text(
        json.dumps({"success": False, "metrics": {}}) + "\n"
    )
    state = tmp_path / "rlm_state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "experiment_ok_receipts.jsonl").write_text(
        json.dumps(
            {"experiment_run_id": "e1", "ok": True, "metrics_sha256": "abc123", "ts": "t"}
        )
        + "\n"
    )
    r = _row(build_scorecard(tmp_path), "execution_completeness")
    assert r.status == "pass"


# --------------------------------------------------------------------------- #
# GATE row: environment_fidelity (env_health.jsonl exclusions)
# --------------------------------------------------------------------------- #

def _env_health_row(**kw):
    base = {
        "env": "alfworld",
        "n_turns": 3,
        "reward": 1.0,
        "unavailable": False,
        "served": True,
    }
    base.update(kw)
    return base


def test_environment_fidelity_unmeasured_when_no_env_health(tmp_path):
    r = _row(build_scorecard(tmp_path), "environment_fidelity")
    assert r.status == "unmeasured"


def test_environment_fidelity_pass_when_env_served_episodes(tmp_path):
    outdir = tmp_path / "code" / "outputs" / "run1"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "env_health.jsonl").write_text(json.dumps(_env_health_row()) + "\n")
    r = _row(build_scorecard(tmp_path), "environment_fidelity")
    assert r.status == "pass"


def test_environment_fidelity_excluded_when_env_never_served(tmp_path):
    outdir = tmp_path / "code" / "outputs" / "run1"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "env_health.jsonl").write_text(
        json.dumps(_env_health_row(unavailable=True, served=False, n_turns=0)) + "\n"
    )
    r = _row(build_scorecard(tmp_path), "environment_fidelity")
    assert r.status == "excluded"


# --------------------------------------------------------------------------- #
# GATE row: dataset_availability (data-unavailable leaves)
# --------------------------------------------------------------------------- #

def _leaf(leaf_id, requirements=""):
    return {"id": leaf_id, "requirements": requirements, "sub_tasks": []}


def test_dataset_availability_unmeasured_when_no_rubric_tree(tmp_path):
    r = _row(build_scorecard(tmp_path), "dataset_availability")
    assert r.status == "unmeasured"


def test_dataset_availability_pass_when_no_unavailable_signal(tmp_path):
    tree = {"id": "root", "sub_tasks": [_leaf("leaf_1", "Report accuracy on CIFAR-10")]}
    (tmp_path / "rubric_tree.json").write_text(json.dumps(tree))
    r = _row(build_scorecard(tmp_path), "dataset_availability")
    assert r.status == "pass"


def test_dataset_availability_excluded_when_dataset_verified_unavailable(tmp_path):
    tree = {"id": "root", "sub_tasks": [_leaf("leaf_1", "Report accuracy on FreyFace")]}
    (tmp_path / "rubric_tree.json").write_text(json.dumps(tree))
    _write_metrics(tmp_path, {"data_load_failures": [{"dataset": "FreyFace"}]})
    r = _row(build_scorecard(tmp_path), "dataset_availability")
    assert r.status == "excluded"


# --------------------------------------------------------------------------- #
# GATE-lite row: tables_figures (fig_*.json sidecars) -- never "fail"
# --------------------------------------------------------------------------- #

def test_tables_figures_unmeasured_when_no_code_dir(tmp_path):
    r = _row(build_scorecard(tmp_path), "tables_figures")
    assert r.status == "unmeasured"


def test_tables_figures_unmeasured_never_fail_when_code_dir_has_no_sidecar(tmp_path):
    (tmp_path / "code").mkdir(parents=True, exist_ok=True)
    r = _row(build_scorecard(tmp_path), "tables_figures")
    assert r.status == "unmeasured"


def test_tables_figures_pass_when_a_sidecar_exists(tmp_path):
    code = tmp_path / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "fig_auto_model_env.json").write_text(json.dumps({"figure": "fig_auto_model_env"}))
    r = _row(build_scorecard(tmp_path), "tables_figures")
    assert r.status == "pass"


# --------------------------------------------------------------------------- #
# DISPLAY rows -- always status="display"; empty detail when the signal is
# absent, populated detail when present. NEVER a cap (see gate_caps() below).
# --------------------------------------------------------------------------- #

def test_autonomy_display_empty_detail_when_absent(tmp_path):
    r = _row(build_scorecard(tmp_path), "autonomy")
    assert r.status == "display" and r.gates is False and r.detail == ""


def test_efficiency_display_populated_when_gpu_ledger_present(tmp_path):
    (tmp_path / "gpu_ledger.jsonl").write_text(
        json.dumps(
            {
                "experiment_run_id": "e1",
                "start_ts": "2026-07-10T00:00:00+00:00",
                "end_ts": "2026-07-10T01:00:00+00:00",
                "gpu_hours": 1.0,
                "provider": "gcp",
                "rate_usd_per_hr": 3.0,
                "est_cost_usd": 3.0,
            }
        )
        + "\n"
    )
    r = _row(build_scorecard(tmp_path), "efficiency")
    assert r.status == "display" and r.gates is False and r.detail != ""


def test_paper_understanding_display_empty_when_no_rubric_on_final_report(tmp_path):
    (tmp_path / "final_report.json").write_text(json.dumps({"verdict": "partial"}))
    r = _row(build_scorecard(tmp_path), "paper_understanding")
    assert r.status == "display" and r.detail == ""


def test_paper_understanding_display_populated_from_final_report_rubric(tmp_path):
    rubric = {"areas": [{"area": "Method fidelity", "score": 0.8, "weight": 1.0}]}
    (tmp_path / "final_report.json").write_text(
        json.dumps({"verdict": "partial", "rubric": rubric})
    )
    r = _row(build_scorecard(tmp_path), "paper_understanding")
    assert r.status == "display" and "fidelity_score_from_rubric" in r.detail


def test_dag_planning_display_populated_from_experiment_runs(tmp_path):
    (tmp_path / "experiment_runs.jsonl").write_text(
        json.dumps({"success": True, "metrics": {"a": 1}, "experiment_run_id": "e1"}) + "\n"
    )
    r = _row(build_scorecard(tmp_path), "dag_planning")
    assert r.status == "display" and r.detail != ""


def test_debugging_display_populated_from_failure_capsules(tmp_path):
    state = tmp_path / "rlm_state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "failure_capsules.jsonl").write_text(
        json.dumps({"schema": 1, "failure_class": "cuda_oom", "error_signature": "x"}) + "\n"
    )
    r = _row(build_scorecard(tmp_path), "debugging")
    assert r.status == "display" and "cuda_oom" in r.detail


def test_scientific_analysis_display_empty_when_no_candidate_events(tmp_path):
    r = _row(build_scorecard(tmp_path), "scientific_analysis")
    assert r.status == "display" and r.detail == ""


def test_scientific_analysis_display_populated_from_candidate_events(tmp_path):
    (tmp_path / "dashboard_events.jsonl").write_text(
        json.dumps({"ts": "t", "event": "candidate_proposed", "data": {}})
        + "\n"
        + json.dumps({"ts": "t", "event": "candidate_outcome", "data": {}})
        + "\n"
    )
    r = _row(build_scorecard(tmp_path), "scientific_analysis")
    assert r.status == "display" and r.detail != ""


# --------------------------------------------------------------------------- #
# A DISPLAY row contributes NO gate cap via EvaluationReport.gate_caps().
# --------------------------------------------------------------------------- #

def test_display_rows_contribute_no_gate_cap(tmp_path):
    (tmp_path / "final_report.json").write_text(json.dumps({"verdict": "reproduced"}))
    rows = build_scorecard(tmp_path)
    display_only = [r for r in rows if r.dimension in DISPLAY_DIMENSIONS]
    assert display_only and all(r.gates is False for r in display_only)
    er = EvaluationReport.from_run(tmp_path)
    er.scorecard = display_only
    assert er.gate_caps() is None
    assert er.verdict == "reproduced"  # untouched by gate_caps()


def test_gate_rows_can_cap_but_display_rows_riding_along_do_not_change_it(tmp_path):
    # An all-empty run: every GATE row is "unmeasured" (never "pass"), so the
    # cap is driven ENTIRELY by the 5 gate rows -- the 6 display rows riding
    # in the SAME scorecard list change nothing about the resulting cap.
    (tmp_path / "final_report.json").write_text(json.dumps({"verdict": "reproduced"}))
    rows = build_scorecard(tmp_path)
    er = EvaluationReport.from_run(tmp_path)
    er.scorecard = rows
    cap_with_display = er.gate_caps()
    er.scorecard = [r for r in rows if r.dimension in GATE_DIMENSIONS]
    cap_gate_only = er.gate_caps()
    assert cap_with_display == cap_gate_only
    assert cap_with_display is not None  # every gate row is unmeasured on an empty run
