"""Best-attempt anti-regression rails (2026-06-11).

Pins the Adam regression pattern: 0.831 best, then seven attempts that
re-derived everything from scratch and landed at 0.69/0.0/0.736/0.762/0.151.
The rails make the best prior attempt the FLOOR of every new attempt: its code
is seeded as reference, its earned leaves are named, and the in-run target
refuses to finish below it.
"""

from __future__ import annotations

import json

from backend.agents.rlm.best_attempt import (
    REFERENCE_DIR_NAME,
    best_attempt_guidance_block,
    champion_ceiling,
    find_best_attempt,
    floored_target,
    leaf_champions,
    leaf_regressions,
    seed_reference_code,
)


def _attempt(root, name, score, leaves=None, code_files=None):
    d = root / "attempts" / name
    (d / "code").mkdir(parents=True, exist_ok=True)
    report = {
        "verdict": "reproduced",
        "rubric": {"overall_score": score,
                   "leaf_scores": leaves or []},
    }
    (d / "final_report.json").write_text(json.dumps(report))
    for fname, content in (code_files or {}).items():
        (d / "code" / fname).write_text(content)
    return d


def test_find_best_attempt_picks_max(tmp_path):
    _attempt(tmp_path, "20260607T000000-0-aa", 0.831)
    _attempt(tmp_path, "20260608T000000-0-bb", 0.69)
    _attempt(tmp_path, "20260609T000000-0-cc", 0.762)
    best = find_best_attempt(tmp_path)
    assert best is not None and best["score"] == 0.831
    assert best["dir"].name.startswith("20260607")


def test_find_best_attempt_none_when_unscored(tmp_path):
    (tmp_path / "attempts").mkdir()
    assert find_best_attempt(tmp_path) is None
    assert find_best_attempt(tmp_path / "nope") is None


def test_leaf_regressions_join_and_threshold():
    best = {"rubric": {"overall_score": 0.83, "leaf_scores": [
        {"id": "L1", "score": 1.0, "justification": "IMDB BoW ran all three optimizers"},
        {"id": "L2", "score": 0.8, "justification": "x"},
        {"id": "L3", "score": 0.5, "justification": "y"},
    ]}}
    latest = {"rubric": {"overall_score": 0.69, "leaf_scores": [
        {"id": "L1", "score": 0.2},   # big regression
        {"id": "L2", "score": 0.75},  # within noise (0.05 < 0.15)
        {"id": "L3", "score": 0.5},   # unchanged
    ]}}
    regs = leaf_regressions(best, latest)
    assert [r["id"] for r in regs] == ["L1"]
    assert regs[0]["best"] == 1.0 and regs[0]["latest"] == 0.2
    assert "IMDB" in regs[0]["evidence"]


def test_seed_reference_code_flag_and_content(tmp_path, monkeypatch):
    _attempt(tmp_path, "20260607T000000-0-aa", 0.831,
             code_files={"train.py": "best code", "metrics.json": "{}"})
    heavy = tmp_path / "attempts/20260607T000000-0-aa/code/outputs/r1"
    heavy.mkdir(parents=True)
    (heavy / "model.pt").write_text("weights")

    monkeypatch.delenv("OPENRESEARCH_SEED_BEST_ATTEMPT", raising=False)
    assert seed_reference_code(tmp_path) is None  # flag off

    monkeypatch.setenv("OPENRESEARCH_SEED_BEST_ATTEMPT", "1")
    rel = seed_reference_code(tmp_path)
    assert rel == f"code/{REFERENCE_DIR_NAME}"
    ref = tmp_path / "code" / REFERENCE_DIR_NAME
    assert (ref / "train.py").read_text() == "best code"
    assert not (ref / "outputs").exists()           # heavy dir skipped
    assert (ref / "_BEST_ATTEMPT_README.txt").exists()
    # idempotent re-seed
    assert seed_reference_code(tmp_path) == rel


def test_guidance_block_names_regressions(tmp_path, monkeypatch):
    _attempt(tmp_path, "20260607T000000-0-aa", 0.831, leaves=[
        {"id": "LEAFAAAA", "score": 1.0, "justification": "all six families measured"},
    ], code_files={"train.py": "x"})
    _attempt(tmp_path, "20260609T000000-0-bb", 0.69, leaves=[
        {"id": "LEAFAAAA", "score": 0.0},
    ])
    monkeypatch.setenv("OPENRESEARCH_SEED_BEST_ATTEMPT", "1")
    block = best_attempt_guidance_block(tmp_path)
    assert "BEST PRIOR ATTEMPT — rubric 0.831" in block
    assert REFERENCE_DIR_NAME in block
    assert "best 1.00 vs latest 0.00" in block
    assert "all six families measured" in block

    monkeypatch.delenv("OPENRESEARCH_SEED_BEST_ATTEMPT")
    assert best_attempt_guidance_block(tmp_path) == ""


def test_floored_target(tmp_path, monkeypatch):
    _attempt(tmp_path, "20260607T000000-0-aa", 0.831)
    monkeypatch.delenv("OPENRESEARCH_TARGET_BEST_FLOOR", raising=False)
    assert floored_target(tmp_path, 0.6) == 0.6  # flag off

    monkeypatch.setenv("OPENRESEARCH_TARGET_BEST_FLOOR", "1")
    assert floored_target(tmp_path, 0.6) == 0.831   # raised to best
    assert floored_target(tmp_path, 0.9) == 0.9     # higher target kept
    assert floored_target(tmp_path, None) == 0.831  # None -> floor
    assert floored_target(tmp_path / "fresh", 0.6) == 0.6  # no prior -> unchanged


def test_leaf_champions_span_attempts(tmp_path):
    from backend.agents.rlm.best_attempt import champion_ceiling, leaf_champions
    _attempt(tmp_path, "20260608T000000-0-aa", 0.70, leaves=[
        {"id": "L1", "score": 1.0, "justification": "base/strided stars"},
        {"id": "L2", "score": 0.2, "justification": "convpool dead"},
    ])
    _attempt(tmp_path, "20260610T000000-0-bb", 0.74, leaves=[
        {"id": "L1", "score": 0.5},
        {"id": "L2", "score": 0.9, "justification": "all families converged"},
    ])
    champs = leaf_champions(tmp_path)
    assert champs["L1"]["score"] == 1.0 and champs["L1"]["attempt"].startswith("20260608")
    assert champs["L2"]["score"] == 0.9 and champs["L2"]["attempt"].startswith("20260610")
    assert abs(champion_ceiling(tmp_path) - 0.95) < 1e-9  # mean(1.0, 0.9)


# ---------------------------------------------------------------------------
# Quarantine THEN rank — the guard-filtered selection pool
# ---------------------------------------------------------------------------


def _quarantine(attempt_dir, failure_class: str) -> None:
    """Trip a HARD quarantine guard on an attempt (attempt_assessment reads
    the LATEST experiment_runs.jsonl row's failure_class)."""
    (attempt_dir / "experiment_runs.jsonl").write_text(
        json.dumps({"success": True, "failure_class": failure_class,
                    "error": "all-zero metrics claim gpu training"}) + "\n"
    )


def test_fabricating_attempt_never_wins_champions_or_guidance(tmp_path, monkeypatch):
    """A fabrication-suspected attempt scores HIGH precisely because it
    fabricated. Ranking the raw LLM score would hand the implementer that
    fabrication as both the best-attempt anchor and the leaf champion to
    reproduce. The guard filter (campaign_policy.seeding_pool, the same gate
    select_champion applies) drops it BEFORE ranking — evidence, not grade."""
    fab = _attempt(tmp_path, "20260607T000000-0-fab", 0.95, leaves=[
        {"id": "LEAFAAAA", "score": 1.0, "justification": "fabricated: metrics never measured"},
    ], code_files={"train.py": "fabricated"})
    _quarantine(fab, "fabrication_suspected")

    _attempt(tmp_path, "20260609T000000-0-hon", 0.40, leaves=[
        {"id": "LEAFAAAA", "score": 0.30, "justification": "honestly measured"},
    ], code_files={"train.py": "honest"})

    # The RAW scan still sees the fabricator as "best" — proving the filter,
    # not a changed scan, is what protects selection.
    assert find_best_attempt(tmp_path)["score"] == 0.95

    champs = leaf_champions(tmp_path)
    assert champs["LEAFAAAA"]["score"] == 0.30           # never the fabricated 1.0
    assert champs["LEAFAAAA"]["attempt"].startswith("20260609")
    assert abs(champion_ceiling(tmp_path) - 0.30) < 1e-9

    monkeypatch.setenv("OPENRESEARCH_SEED_BEST_ATTEMPT", "1")
    block = best_attempt_guidance_block(tmp_path)
    assert "BEST PRIOR ATTEMPT — rubric 0.400" in block   # the honest attempt
    assert "0.95" not in block
    assert "fabricated" not in block
    assert "20260607T000000-0-fab" not in block


def test_all_models_failed_attempt_is_also_excluded(tmp_path, monkeypatch):
    """all_models_failed is the other hard-quarantine guard (the monolithic
    run_experiment fake-green): every model errored, metrics are 0.0, exit
    code 0. It must not anchor guidance either."""
    dead = _attempt(tmp_path, "20260607T000000-0-dead", 0.90, leaves=[
        {"id": "LEAFBBBB", "score": 1.0, "justification": "every model failed"},
    ], code_files={"train.py": "dead"})
    _quarantine(dead, "all_models_failed")

    _attempt(tmp_path, "20260609T000000-0-live", 0.35, leaves=[
        {"id": "LEAFBBBB", "score": 0.20, "justification": "real run"},
    ], code_files={"train.py": "live"})

    assert leaf_champions(tmp_path)["LEAFBBBB"]["score"] == 0.20
    monkeypatch.setenv("OPENRESEARCH_SEED_BEST_ATTEMPT", "1")
    assert "rubric 0.350" in best_attempt_guidance_block(tmp_path)


def test_every_attempt_quarantined_yields_no_guidance_rather_than_a_fabrication(
    tmp_path, monkeypatch
):
    """Fail CLOSED: when the only scored attempts are hard-quarantined the
    pool is empty and the rails go silent. An empty guidance block is the
    honest outcome; anchoring on the fabrication is not."""
    fab = _attempt(tmp_path, "20260607T000000-0-fab", 0.95, leaves=[
        {"id": "LEAFAAAA", "score": 1.0, "justification": "fabricated"},
    ], code_files={"train.py": "x"})
    _quarantine(fab, "fabrication_suspected")

    assert leaf_champions(tmp_path) == {}
    assert champion_ceiling(tmp_path) is None
    monkeypatch.setenv("OPENRESEARCH_SEED_BEST_ATTEMPT", "1")
    assert best_attempt_guidance_block(tmp_path) == ""


def test_soft_quarantined_attempts_stay_selectable(tmp_path, monkeypatch):
    """F4 seeding-starvation rule (spec §8.1): the external validator is
    default-OFF, so EVERY real attempt is soft-quarantined ("validator:
    missing"). Filtering on soft quarantine would make these rails inert on
    every real run — only HARD quarantine (fabrication family) excludes."""
    _attempt(tmp_path, "20260607T000000-0-aa", 0.72, leaves=[
        {"id": "LEAFCCCC", "score": 0.8, "justification": "no validator configured"},
    ], code_files={"train.py": "x"})

    assert leaf_champions(tmp_path)["LEAFCCCC"]["score"] == 0.8
    monkeypatch.setenv("OPENRESEARCH_SEED_BEST_ATTEMPT", "1")
    assert "rubric 0.720" in best_attempt_guidance_block(tmp_path)


def test_guidance_block_includes_crossover_targets(tmp_path, monkeypatch):
    # Best attempt is bb (0.74), but aa holds the L1 champion (1.0 vs bb's 0.5).
    _attempt(tmp_path, "20260608T000000-0-aa", 0.70, leaves=[
        {"id": "L1XXXXXX", "score": 1.0, "justification": "stars"},
    ], code_files={"train.py": "x"})
    _attempt(tmp_path, "20260610T000000-0-bb", 0.74, leaves=[
        {"id": "L1XXXXXX", "score": 0.5},
    ], code_files={"train.py": "y"})
    monkeypatch.setenv("OPENRESEARCH_SEED_BEST_ATTEMPT", "1")
    block = best_attempt_guidance_block(tmp_path)
    assert "CROSSOVER TARGETS" in block
    assert "champion 1.00 in 20260608T000000" in block
