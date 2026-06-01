from backend.agents.gepa.metrics.improvement_metrics import (
    weak_area_coverage, category_diversity, combined_improvement_score,
)

HYPOTHESES = [
    {"path_id": "p1", "title": "Lower LR", "category": "hyperparameter",
     "hypothesis": "Reduce lr", "rationale": "...", "expected_outcome": "..."},
    {"path_id": "p2", "title": "Add dropout", "category": "regularization",
     "hypothesis": "Add dropout", "rationale": "...", "expected_outcome": "..."},
    {"path_id": "p3", "title": "More data aug", "category": "data",
     "hypothesis": "More aug", "rationale": "...", "expected_outcome": "..."},
]


def test_weak_area_coverage_all_covered():
    weak_leaves = ["hyperparameter", "regularization", "data"]
    score, fb = weak_area_coverage(HYPOTHESES, weak_leaves)
    assert score == 1.0


def test_weak_area_coverage_partial():
    weak_leaves = ["hyperparameter", "architecture"]
    score, fb = weak_area_coverage(HYPOTHESES, weak_leaves)
    assert score == 0.5
    assert "architecture" in fb


def test_weak_area_coverage_no_leaves():
    score, fb = weak_area_coverage(HYPOTHESES, [])
    assert score == 1.0


def test_category_diversity_three_unique():
    score = category_diversity(HYPOTHESES)
    assert score == 1.0


def test_category_diversity_all_same():
    hyps = [{"category": "hyperparameter"} for _ in range(3)]
    score = category_diversity(hyps)
    assert score < 0.5


def test_category_diversity_empty():
    assert category_diversity([]) == 0.0


def test_combined_improvement_score():
    score, fb = combined_improvement_score(HYPOTHESES, ["hyperparameter"])
    assert 0.0 < score <= 1.0
