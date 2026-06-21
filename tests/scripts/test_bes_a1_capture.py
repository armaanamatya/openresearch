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


def test_regrade_candidates_with_injected_scorer(tmp_path):
    from scripts.bes_a1_capture import regrade_candidates, summarize_regrades

    # two candidate dirs + a rubric tree file
    for i in (0, 1):
        d = tmp_path / "candidates" / f"rlm_impl_{i}" / "code"
        d.mkdir(parents=True)
        (d / "train.py").write_text("# stub")
    (tmp_path / "generated_rubric.json").write_text("{}")

    # deterministic fake: candidate 1 always beats 0 by a wide margin (stable)
    scores = {"rlm_impl_0": 0.50, "rlm_impl_1": 0.70}

    def fake_score(code_dir, rubric_tree):
        cid = code_dir.parent.name  # rlm_impl_N
        return scores[cid]

    regrades = regrade_candidates(tmp_path, k=3, score_one=fake_score)
    assert len(regrades) == 3
    assert all("rlm_impl_0" in r and "rlm_impl_1" in r for r in regrades)
    out = summarize_regrades(regrades, repeatability_sigma=0.02)
    assert out["verdict"] == "select_stable"  # wide stable margin


def test_regrade_candidates_fail_soft_on_scorer_error(tmp_path):
    """A candidate that raises during scoring is skipped (warn, not crash)."""
    from scripts.bes_a1_capture import regrade_candidates

    for i in (0, 1):
        d = tmp_path / "candidates" / f"rlm_impl_{i}" / "code"
        d.mkdir(parents=True)
    (tmp_path / "generated_rubric.json").write_text("{}")

    def always_fail(code_dir, rubric_tree):
        raise RuntimeError("simulated scorer failure")

    regrades = regrade_candidates(tmp_path, k=2, score_one=always_fail)
    # K rounds still returned even when all candidates error
    assert len(regrades) == 2
    # Each round dict is empty (both candidates were skipped)
    assert all(r == {} for r in regrades)


def test_regrade_candidates_no_candidates_warns(tmp_path):
    """No candidates/ dir → warns and returns K empty dicts."""
    from scripts.bes_a1_capture import regrade_candidates
    import warnings

    (tmp_path / "generated_rubric.json").write_text("{}")

    def fake_score(code_dir, rubric_tree):
        return 0.5

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        regrades = regrade_candidates(tmp_path, k=2, score_one=fake_score)

    assert len(regrades) == 2
    assert all(r == {} for r in regrades)
    assert any("no candidates" in str(warning.message).lower() for warning in w)
