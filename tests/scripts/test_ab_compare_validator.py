"""scripts/ab_compare.py — D1 validator mode + D3 BES default posture.

Validator mode (``--require-stamped`` / ``REPROLAB_REQUIRE_STAMPED_AB``,
default OFF) must refuse an apples-to-oranges Δ: it requires BOTH arms to be
stamped, their ``rubric_tree.json`` sha256 to match, and their recorded
``scope`` to match — selecting the BEST report per arm. Reporter mode (flag
off) must stay byte-for-byte unchanged.

D3 asserts the BES master gate (``Settings.bes_enabled`` /
``REPROLAB_BES_ENABLED``) defaults OFF — BES stays default-OFF per the
2026-06-16 posture (1 clean pair, the ≥3-paired-SDAR bar unmet).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "ab_compare", REPO_ROOT / "scripts" / "ab_compare.py"
)
ab_compare = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("ab_compare", ab_compare)
_SPEC.loader.exec_module(ab_compare)


# Canonical scope block both arms share unless a test deliberately diverges it.
_SCOPE = {
    "requested": "CIFAR-10: Models A/B/C",
    "ran": ["a_base", "a_strided", "c_allcnn"],
    "gaps": ["ImageNet not executed"],
}
# Canonical rubric_tree bytes both arms share unless a test diverges it.
_RUBRIC_TREE = {"leaves": [{"id": "leaf_a"}, {"id": "leaf_b"}], "version": 1}


def _write_arm(
    runs_root: Path,
    project_id: str,
    *,
    arm: str | None,
    score: float,
    pair_id: str | None = "ab-1",
    scope: dict | None = None,
    rubric_tree: dict | None = None,
    write_rubric_tree: bool = True,
    leaf_scores: list[dict] | None = None,
    paper_id: str = "1412.6806",
    completed_at: str = "2026-06-11T06:00:00+00:00",
) -> Path:
    """Write a run dir with final_report.json (+ optional rubric_tree.json)."""
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "paper": {"id": paper_id, "title": "All-CNN"},
        "verdict": "partial",
        "scope": _SCOPE if scope is None else scope,
        "rubric": {
            "overall_score": score,
            "meets_target": score >= 0.6,
            "areas": [],
            "leaf_scores": leaf_scores or [],
        },
        "cost": {"llm_usd": 2.0},
        "iterations": 3,
        "started_at": "2026-06-11T00:00:00+00:00",
        "completed_at": completed_at,
    }
    if arm is not None:
        report["experiment_arm"] = {
            "arm": arm,
            "ab_pair_id": pair_id,
            "bes": {"enabled": arm == "bes", "candidates_per_cluster": 2 if arm == "bes" else 1},
        }
    (run_dir / "final_report.json").write_text(json.dumps(report), encoding="utf-8")
    if write_rubric_tree:
        tree = _RUBRIC_TREE if rubric_tree is None else rubric_tree
        (run_dir / "rubric_tree.json").write_text(json.dumps(tree), encoding="utf-8")
    return run_dir


def _valid_pair(runs_root: Path) -> None:
    _write_arm(runs_root, "prj_ctl", arm="control", score=0.60,
               leaf_scores=[{"id": "leaf_a", "score": 0.5}])
    _write_arm(runs_root, "prj_bes", arm="bes", score=0.72,
               leaf_scores=[{"id": "leaf_a", "score": 0.9}])


# ---------------------------------------------------------------------------
# D1 — validator REJECTS bad pairs
# ---------------------------------------------------------------------------

def test_validator_rejects_unstamped_control(tmp_path: Path):
    # Under a PAPER selector, reporter mode admits an unstamped legacy run as
    # control; validator mode must NOT — that is exactly the apples-to-oranges
    # case it exists to refuse.
    _write_arm(tmp_path, "prj_legacy", arm=None, score=0.55, leaf_scores=[{"id": "leaf_a", "score": 0.5}])
    _write_arm(tmp_path, "prj_bes", arm="bes", score=0.6, leaf_scores=[{"id": "leaf_a", "score": 0.9}])

    # Sanity: reporter mode DOES admit the unstamped run as control.
    rep = ab_compare.build_comparison(tmp_path, paper="1412.6806")
    assert rep["control"]["project_id"] == "prj_legacy"
    assert rep["deltas"]  # reporter emits a Δ

    # Validator mode refuses: the unstamped run is dropped → no control arm.
    cmp = ab_compare.build_comparison(tmp_path, paper="1412.6806", require_stamped=True)
    assert cmp["validation_error"]
    assert "unstamped" in cmp["validation_error"].lower()
    assert cmp["control"] is None  # the legacy run was not admitted
    # Refusal blanks the Δ and the leaf moves.
    assert cmp["deltas"] == {}
    assert cmp["top_leaf_moves"] == []
    # The refusal is surfaced in the rendered markdown.
    md = ab_compare.render_markdown(cmp)
    assert "REFUSED" in md


def test_validator_rejects_rubric_sha_mismatch(tmp_path: Path):
    _write_arm(tmp_path, "prj_ctl", arm="control", score=0.6)
    _write_arm(tmp_path, "prj_bes", arm="bes", score=0.7,
               rubric_tree={"leaves": [{"id": "DIFFERENT"}], "version": 9})

    cmp = ab_compare.build_comparison(tmp_path, pair_id="ab-1", require_stamped=True)
    assert cmp["validation_error"]
    assert "rubric_tree" in cmp["validation_error"]
    assert cmp["deltas"] == {}


def test_validator_rejects_missing_rubric_tree(tmp_path: Path):
    # Control has no rubric_tree.json — cannot verify rubric equality.
    _write_arm(tmp_path, "prj_ctl", arm="control", score=0.6, write_rubric_tree=False)
    _write_arm(tmp_path, "prj_bes", arm="bes", score=0.7)

    cmp = ab_compare.build_comparison(tmp_path, pair_id="ab-1", require_stamped=True)
    assert cmp["validation_error"]
    assert "rubric_tree.json" in cmp["validation_error"]
    assert "prj_ctl" in cmp["validation_error"]
    assert cmp["deltas"] == {}


def test_validator_rejects_scope_mismatch(tmp_path: Path):
    _write_arm(tmp_path, "prj_ctl", arm="control", score=0.6)
    # Same rubric, but a different ran-list → different experiment matrix.
    _write_arm(tmp_path, "prj_bes", arm="bes", score=0.7,
               scope={"requested": "CIFAR-10: Models A/B/C",
                      "ran": ["a_base", "a_strided", "c_allcnn", "EXTRA_CELL"],
                      "gaps": []})

    cmp = ab_compare.build_comparison(tmp_path, pair_id="ab-1", require_stamped=True)
    assert cmp["validation_error"]
    assert "scope" in cmp["validation_error"]
    assert cmp["deltas"] == {}


def test_validator_rejects_when_an_arm_is_missing(tmp_path: Path):
    # Only a control arm exists — no bes arm to compare against.
    _write_arm(tmp_path, "prj_ctl", arm="control", score=0.6)

    cmp = ab_compare.build_comparison(tmp_path, pair_id="ab-1", require_stamped=True)
    assert cmp["validation_error"]
    assert cmp["bes"] is None
    assert cmp["deltas"] == {}


# ---------------------------------------------------------------------------
# D1 — validator ACCEPTS a valid pair and emits the Δ
# ---------------------------------------------------------------------------

def test_validator_accepts_valid_stamped_matched_pair(tmp_path: Path):
    _valid_pair(tmp_path)

    cmp = ab_compare.build_comparison(tmp_path, pair_id="ab-1", require_stamped=True)
    assert cmp["validation_error"] is None
    assert cmp["require_stamped"] is True
    assert cmp["control"]["project_id"] == "prj_ctl"
    assert cmp["bes"]["project_id"] == "prj_bes"
    # The Δ is emitted (bes 0.72 − control 0.60).
    assert cmp["deltas"]["overall_score"] == pytest.approx(0.12)
    # And the leaf move is computed.
    moves = {m["leaf"]: m["delta"] for m in cmp["top_leaf_moves"]}
    assert moves == {"leaf_a": pytest.approx(0.4)}
    md = ab_compare.render_markdown(cmp)
    assert "REFUSED" not in md
    assert "Δ (bes − control)" in md


def test_validator_json_artifact_is_serialisable_and_clean(tmp_path: Path):
    # The slim JSON must not leak the working-only keys (_run_dir is a Path).
    _valid_pair(tmp_path)
    out = tmp_path / "_out"
    cmp = ab_compare.build_comparison(tmp_path, pair_id="ab-1", require_stamped=True)
    out.mkdir()
    slim = {k: v for k, v in cmp.items()}
    for label in ("control", "bes"):
        if slim.get(label):
            slim[label] = {k: v for k, v in slim[label].items() if k not in ab_compare._SLIM_DROP}
    # Round-trips through JSON (would raise on a stray Path).
    text = json.dumps(slim, indent=2)
    reloaded = json.loads(text)
    assert "_run_dir" not in reloaded["control"]
    assert "_scope" not in reloaded["bes"]
    assert reloaded["validation_error"] is None


# ---------------------------------------------------------------------------
# D1 — select=best is the default in BOTH modes
# ---------------------------------------------------------------------------

def test_select_best_is_default_picks_higher_scoring_attempt(tmp_path: Path):
    # Two BES candidates; the lower-completed_at one scores higher. Default
    # select=best must pick the higher score, NOT the latest.
    _write_arm(tmp_path, "prj_bes_hi", arm="bes", score=0.80,
               completed_at="2026-06-10T00:00:00+00:00")
    _write_arm(tmp_path, "prj_bes_lo", arm="bes", score=0.40,
               completed_at="2026-06-11T00:00:00+00:00")  # newer but worse
    _write_arm(tmp_path, "prj_ctl", arm="control", score=0.30)

    # Reporter mode, default selector.
    cmp = ab_compare.build_comparison(tmp_path, paper="1412.6806")
    assert cmp["select"] == "best"
    assert cmp["bes"]["project_id"] == "prj_bes_hi"  # higher score wins

    # Validator mode, default selector — same best pick.
    cmp_v = ab_compare.build_comparison(tmp_path, pair_id="ab-1", require_stamped=True)
    assert cmp_v["bes"]["project_id"] == "prj_bes_hi"


# ---------------------------------------------------------------------------
# D1 — reporter mode (flag OFF) is unchanged on a basic pair
# ---------------------------------------------------------------------------

def test_reporter_mode_unchanged_no_validation_keys(tmp_path: Path):
    _valid_pair(tmp_path)
    # An unstamped run that reporter mode would still admit as control via the
    # paper selector — proving the flag-off path keeps prior leniency.
    _write_arm(tmp_path, "prj_legacy", arm=None, score=0.50)

    cmp = ab_compare.build_comparison(tmp_path, paper="1412.6806")  # require_stamped defaults OFF
    # No validator keys in reporter mode.
    assert "validation_error" not in cmp
    assert "require_stamped" not in cmp
    # The Δ is still emitted (reporter never refuses).
    assert cmp["deltas"]["overall_score"] == pytest.approx(0.12)
    md = ab_compare.render_markdown(cmp)
    assert "REFUSED" not in md


def test_reporter_mode_still_admits_unstamped_control(tmp_path: Path):
    # Flag OFF: an unstamped legacy run remains an admissible control under a
    # paper selector (byte-for-byte prior behaviour) — the case the validator
    # refuses.
    _write_arm(tmp_path, "prj_legacy", arm=None, score=0.55)
    _write_arm(tmp_path, "prj_bes", arm="bes", score=0.60)

    cmp = ab_compare.build_comparison(tmp_path, paper="1412.6806")
    assert cmp["control"]["project_id"] == "prj_legacy"
    assert cmp["arms_found"] == {"bes": 1, "unstamped": 1}
    assert "validation_error" not in cmp


def test_env_flag_enables_validator_via_main(tmp_path: Path, monkeypatch, capsys):
    # REPROLAB_REQUIRE_STAMPED_AB=1 turns on validator mode through main();
    # an unstamped control is refused with a nonzero exit + REFUSED on stderr.
    _write_arm(tmp_path, "prj_legacy", arm=None, score=0.55)
    _write_arm(tmp_path, "prj_bes", arm="bes", score=0.60)

    monkeypatch.setenv("REPROLAB_REQUIRE_STAMPED_AB", "1")
    monkeypatch.setattr(
        sys, "argv",
        ["ab_compare.py", "--pair-id", "ab-1", "--runs-root", str(tmp_path),
         "--out", str(tmp_path / "_ab")],
    )
    rc = ab_compare.main()
    assert rc == 3  # validator refusal exit code
    err = capsys.readouterr().err
    assert "REFUSED" in err


def test_reporter_mode_exit_zero_on_valid_pair_via_main(tmp_path: Path, monkeypatch, capsys):
    # Flag OFF (default): a basic valid pair exits 0, no REFUSED, unchanged.
    _valid_pair(tmp_path)
    monkeypatch.delenv("REPROLAB_REQUIRE_STAMPED_AB", raising=False)
    monkeypatch.setattr(
        sys, "argv",
        ["ab_compare.py", "--pair-id", "ab-1", "--runs-root", str(tmp_path),
         "--out", str(tmp_path / "_ab")],
    )
    rc = ab_compare.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "REFUSED" not in out


# ---------------------------------------------------------------------------
# D3 — BES master gate defaults OFF (posture assertion; does NOT change it)
# ---------------------------------------------------------------------------

def test_bes_master_gate_defaults_off():
    # The canonical default lives on Settings.bes_enabled (backend/config.py).
    # BES stays default-OFF per the 2026-06-16 posture until the validator
    # (D1) + smoke SELECT (D2) land and a ≥3-seed paired SDAR clears the bar.
    from backend.config import Settings

    s = Settings()
    assert s.bes_enabled is False
    # candidates-per-cluster default = 1 => parity (no competing pool) even if
    # the master gate were flipped without setting N>1.
    assert s.bes_candidates_per_cluster == 1


def test_bes_rlm_enabled_helper_is_false_by_default(monkeypatch):
    # The RLM-path gate (bes_rlm) reads the same Settings; with a clean env it
    # must report BES disabled (enabled AND n>1 are both required).
    from backend.config import Settings

    for var in ("REPROLAB_BES_ENABLED", "REPROLAB_BES_CANDIDATES_PER_CLUSTER"):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    bes_active = bool(s.bes_enabled) and int(s.bes_candidates_per_cluster) > 1
    assert bes_active is False
