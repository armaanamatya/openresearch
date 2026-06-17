from scripts.bes_a1_capture import summarize_regrades


def test_noisy_select_when_winner_flips_at_tiny_margin():
    # winner flips between re-grades AND margins are below sigma -> noise
    regrades = [{"a": 0.50, "b": 0.51}, {"a": 0.51, "b": 0.50}]
    out = summarize_regrades(regrades, repeatability_sigma=0.02)
    assert out["report"]["top1_flip_rate"] > 0.0
    assert out["report"]["margin_min"] <= 0.02
    assert out["verdict"] == "select_is_noise"


def test_stable_select_when_consistent_winner_with_wide_margin():
    regrades = [{"a": 0.50, "b": 0.70}, {"a": 0.51, "b": 0.71}]
    out = summarize_regrades(regrades, repeatability_sigma=0.02)
    assert out["report"]["top1_flip_rate"] == 0.0
    assert out["verdict"] == "select_stable"
