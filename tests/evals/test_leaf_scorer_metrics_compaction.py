"""G1: the grader sees EVERY model's results on a wide grid, not just the first
one or two before the byte budget truncates (2026-06-14 fix).

Root cause: `_gather_evidence` serialized the full metrics.json and hard-truncated
it at 32 KB. A wide grid (SDAR 3x3, Adam 6x6, All-CNN 14 models) carries per-epoch
history arrays, so a single run is ~340 KB and the cut fell "inside a_strided" —
every later-sorted model (the paper's HEADLINE result, e.g. All-CNN-C) was invisible
to the grader, which then scored those leaves "central claim unverified" although the
cell had run. The fix compacts long numeric series to {len,first,last,min,max} while
preserving every scalar, key-name agnostically (general to any paper).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from backend.evals.paperbench.leaf_scorer import (
    _code_file_priority,
    _compact_metrics_for_grader,
    _gather_evidence,
    _summarize_long_list,
    _MAX_METRICS_BYTES,
)


def _write_metrics(run_dir: Path, run_id: str, payload: dict, mtime: float) -> Path:
    d = run_dir / "code" / "outputs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / "metrics.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(p, (mtime, mtime))
    return p


# --- the pure compactor -----------------------------------------------------

def test_summarize_long_numeric_list_collapses_to_series():
    out = _summarize_long_list([float(i) for i in range(350)])
    assert out == {"_series": {"len": 350, "first": 0.0, "last": 349.0, "min": 0.0, "max": 349.0}}


def test_summarize_short_list_passes_through():
    short = [0.1, 0.2, 0.3]
    assert _summarize_long_list(short) == short


def test_summarize_long_nonnumeric_list_keeps_head_and_tail():
    out = _summarize_long_list([f"step{i}" for i in range(100)])
    assert isinstance(out, list)
    assert out[0] == "step0" and out[-1] == "step99"
    assert any("more elided" in str(x) for x in out)


def test_bool_not_treated_as_numeric_series():
    # A long list of bools is non-numeric (bools are excluded), so it keeps head/tail.
    out = _summarize_long_list([True] * 50)
    assert isinstance(out, list)


def test_compact_preserves_scalars_summarizes_arrays():
    metrics = {
        "per_model": {
            "c_allcnn": {
                "cifar10_noaug": {
                    "allcnn": {
                        "test_error_pct": 14.81,
                        "best_lr": 0.05,
                        "epochs_run": 350,
                        "train_loss_history": [2.0 - i / 350 for i in range(350)],
                    }
                }
            }
        }
    }
    compact = _compact_metrics_for_grader(metrics)
    leaf = compact["per_model"]["c_allcnn"]["cifar10_noaug"]["allcnn"]
    assert leaf["test_error_pct"] == 14.81          # scalar preserved
    assert leaf["best_lr"] == 0.05
    assert leaf["epochs_run"] == 350
    assert leaf["train_loss_history"]["_series"]["len"] == 350  # array summarized


# --- the regression guard: end-to-end through _gather_evidence ---------------

def test_gather_evidence_surfaces_all_models_in_wide_grid(tmp_path: Path):
    """A 14-model grid whose RAW serialization far exceeds 32 KB: every model's
    scalar result must reach the grader, including the last-sorted ones that the
    old raw-truncation cut off."""
    (tmp_path / "code").mkdir(parents=True)
    models = (
        [f"a_{v}" for v in ("base", "strided", "convpool", "allcnn")]
        + [f"b_{v}" for v in ("base", "strided", "convpool", "allcnn")]
        + [f"c_{v}" for v in ("base", "strided", "convpool", "allcnn")]
        + ["c_allcnn_cifar100", "c_allcnn_aug"]
    )
    per_model = {}
    for i, m in enumerate(models):
        per_model[m] = {
            "env": {
                "baseline": {
                    "test_error_pct": 9.0 + i,                       # a distinct, findable scalar
                    "best_lr": 0.05,
                    "epochs_run": 350,
                    "train_loss_history": [2.0 - j / 400 for j in range(400)],
                    "test_acc_history": [0.4 + j / 1000 for j in range(400)],
                }
            }
        }
    payload = {"status": "complete", "per_model": per_model}
    # Sanity: raw is far over the old 32 KB cap, so the bug WOULD bite without compaction.
    assert len(json.dumps(payload, indent=2)) > 100_000
    _write_metrics(tmp_path, "exp_final", payload, mtime=2000)

    evidence = _gather_evidence(tmp_path)

    # Every model key AND its unique scalar must be present — including the last.
    for i, m in enumerate(models):
        assert f'"{m}"' in evidence, f"model {m} hidden from grader"
        assert str(9.0 + i) in evidence, f"{m} test_error_pct hidden"
    # The headline late-sorted models specifically (the ones the bug hid).
    assert '"c_allcnn"' in evidence
    assert '"c_allcnn_cifar100"' in evidence
    # The verbatim 400-element histories must NOT be dumped (that's the whole point).
    assert evidence.count('"train_loss_history"') <= len(models)  # one summary key each, no array spam
    assert "_series" in evidence


def test_gather_evidence_compaction_within_budget(tmp_path: Path):
    (tmp_path / "code").mkdir(parents=True)
    per_model = {
        f"m{i}": {"env": {"b": {"test_error_pct": float(i), "hist": list(range(500))}}}
        for i in range(20)
    }
    _write_metrics(tmp_path, "big", {"per_model": per_model}, mtime=2000)
    evidence = _gather_evidence(tmp_path)
    # The metrics block stays within the metrics budget after compaction.
    block = evidence.split("=== latest experiment metrics.json")[1]
    assert len(block) <= _MAX_METRICS_BYTES + 4096  # block label + slack
    assert '"m19"' in evidence  # last model still present


# --- G1-sibling: final_report.json baseline_metrics compaction ---------------

def test_gather_evidence_compacts_baseline_metrics_in_final_report(tmp_path: Path):
    """baseline_metrics mirrors the aggregate metrics (per_model + history arrays);
    fed verbatim it ~exhausts the 200 KB total budget. It must be compacted so it
    doesn't starve the code-file evidence."""
    (tmp_path / "code").mkdir(parents=True)
    big_bm = {
        "per_model": {
            f"m{i}": {"env": {"b": {"test_error_pct": float(i),
                                    "train_loss_history": list(range(500))}}}
            for i in range(10)
        }
    }
    (tmp_path / "final_report.json").write_text(
        json.dumps({"verdict": "reproduced", "baseline_metrics": big_bm,
                    "paper": {"id": "x"}, "reproduction_summary": "ok"}),
        encoding="utf-8",
    )
    evidence = _gather_evidence(tmp_path)
    report_block = evidence.split("=== latest experiment")[0]
    assert '"m9"' in report_block          # last model survives in the report snippet
    assert "_series" in report_block       # the 500-element history was summarized
    # The verbatim long array must NOT appear in the report snippet.
    assert '"train_loss_history": [\n' not in report_block


# --- Finding 1: code-file evidence ranked by load-bearing-ness ---------------

def test_code_file_priority_loadbearing_beats_lexicographic(tmp_path: Path):
    code = tmp_path / "code"
    code.mkdir()
    early_helper = code / "a_helper.py"      # sorts FIRST alphabetically, non-bearing
    late_model = code / "z_model.py"         # sorts LAST alphabetically, load-bearing
    early_helper.write_text("x" * 100)
    late_model.write_text("x" * 100)
    ranked = sorted([early_helper, late_model], key=_code_file_priority)
    assert ranked[0] == late_model           # content beats alphabetical position


def test_code_file_priority_smaller_first_within_tier(tmp_path: Path):
    code = tmp_path / "code"
    code.mkdir()
    big = code / "model_big.py"
    small = code / "model_small.py"
    big.write_text("x" * 5000)
    small.write_text("x" * 100)
    ranked = sorted([big, small], key=_code_file_priority)
    assert ranked[0] == small                # smaller-first so more distinct files fit


def test_gather_evidence_includes_loadbearing_code_file(tmp_path: Path):
    """A load-bearing file that sorts late must reach the grader."""
    code = tmp_path / "code"
    code.mkdir(parents=True)
    (code / "zzz_optimizers.py").write_text("class Adam: pass\n", encoding="utf-8")
    (code / "aaa_notes.txt").write_text("misc\n", encoding="utf-8")
    evidence = _gather_evidence(tmp_path)
    assert "zzz_optimizers.py" in evidence
    assert "class Adam" in evidence
