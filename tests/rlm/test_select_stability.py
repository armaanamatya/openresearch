from backend.agents.rlm.select_stability import stability_report


def test_stability_report_flip_rate_and_winprob():
    regrades = [
        {"a": 0.50, "b": 0.55, "c": 0.40},
        {"a": 0.56, "b": 0.54, "c": 0.41},
        {"a": 0.49, "b": 0.57, "c": 0.42},
        {"a": 0.48, "b": 0.58, "c": 0.39},
    ]
    rep = stability_report(regrades)
    assert rep["k"] == 4
    assert rep["top1_flip_rate"] == 0.25            # top-1 changed once across 4 (a then b,b,b)
    assert abs(rep["win_prob"]["b"] - 0.75) < 1e-9  # b top-1 in 3/4
    assert rep["margin_p50"] >= 0.0


def test_empty_regrades():
    rep = stability_report([])
    assert rep["k"] == 0
    assert rep["win_prob"] == {}
