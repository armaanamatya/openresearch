from backend.agents.rlm.conversion_guard import detect_projection_incoherence


def test_detects_empty_provenance_with_graded_metrics():
    report = {"baseline_metrics": {}, "experiment_run_id": None, "primitive_trace": {}}
    rubric = {"overall_score": 0.53, "leaf_scores": [{"id": "acc", "score": 0.53}],
              "evidence_cites_metrics": True}
    metrics_on_disk = {"cifar10_cnn": {"top1": 0.91}}
    issue = detect_projection_incoherence(report, rubric, metrics_on_disk)
    assert issue is not None
    assert issue["kind"] == "empty_provenance_with_graded_evidence"


def test_no_issue_when_coherent():
    report = {"baseline_metrics": {"cifar10_cnn": {"top1": 0.91}},
              "experiment_run_id": "run-1", "primitive_trace": {"run_experiment": 1}}
    rubric = {"overall_score": 0.53, "evidence_cites_metrics": True}
    assert detect_projection_incoherence(report, rubric, {"cifar10_cnn": {"top1": 0.91}}) is None


def test_no_issue_when_no_metrics_on_disk():
    report = {"baseline_metrics": {}, "experiment_run_id": None, "primitive_trace": {}}
    rubric = {"overall_score": 0.53, "evidence_cites_metrics": True}
    assert detect_projection_incoherence(report, rubric, None) is None
