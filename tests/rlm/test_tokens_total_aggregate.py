"""Tests for tokens_total.json aggregate — PR-α.5.

Covers:
1. Given a fixture cost_ledger.jsonl with multiple primitives × calls × models,
   tokens_total.json has correct by_primitive, by_model, grand_total sums.
2. Empty ledger produces zero grand_total.
3. Missing cost_ledger.jsonl produces zero grand_total.
4. write_tokens_total uses atomic write (tmp + os.replace).
5. schema_version is 1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


from backend.agents.rlm.report import _aggregate_tokens_total, write_tokens_total


def _write_ledger(run_dir: Path, rows: list[dict]) -> None:
    """Write rows to cost_ledger.jsonl in a run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(r) for r in rows)
    (run_dir / "cost_ledger.jsonl").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_ROWS = [
    # understand_section × 2 calls, both on sonnet
    {
        "primitive": "understand_section",
        "model": "claude-sonnet-4-6",
        "input_tokens": 2000, "output_tokens": 300,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 50,
    },
    {
        "primitive": "understand_section",
        "model": "claude-sonnet-4-6",
        "input_tokens": 3000, "output_tokens": 400,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 100,
    },
    # implement_baseline × 2 calls, both on opus
    {
        "primitive": "implement_baseline",
        "model": "claude-opus-4-7",
        "input_tokens": 10000, "output_tokens": 5000,
        "cache_creation_input_tokens": 200, "cache_read_input_tokens": 0,
    },
    {
        "primitive": "implement_baseline",
        "model": "claude-opus-4-7",
        "input_tokens": 12000, "output_tokens": 6000,
        "cache_creation_input_tokens": 300, "cache_read_input_tokens": 0,
    },
    # verify_against_rubric × 2 calls: one sonnet, one opus
    {
        "primitive": "verify_against_rubric",
        "model": "claude-sonnet-4-6",
        "input_tokens": 4000, "output_tokens": 800,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
    },
    {
        "primitive": "verify_against_rubric",
        "model": "claude-opus-4-7",
        "input_tokens": 5000, "output_tokens": 1000,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
    },
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_by_primitive_sums(tmp_path):
    run_dir = tmp_path / "run1"
    _write_ledger(run_dir, FIXTURE_ROWS)
    data = _aggregate_tokens_total(run_dir)

    bp = data["by_primitive"]

    assert bp["understand_section"]["input_tokens"] == 5000   # 2000 + 3000
    assert bp["understand_section"]["output_tokens"] == 700   # 300 + 400
    assert bp["understand_section"]["calls"] == 2

    assert bp["implement_baseline"]["input_tokens"] == 22000  # 10000 + 12000
    assert bp["implement_baseline"]["output_tokens"] == 11000 # 5000 + 6000
    assert bp["implement_baseline"]["calls"] == 2

    assert bp["verify_against_rubric"]["input_tokens"] == 9000  # 4000 + 5000
    assert bp["verify_against_rubric"]["calls"] == 2


def test_by_model_sums(tmp_path):
    run_dir = tmp_path / "run1"
    _write_ledger(run_dir, FIXTURE_ROWS)
    data = _aggregate_tokens_total(run_dir)

    bm = data["by_model"]

    # sonnet: understand_section (both) + verify_against_rubric (first)
    # input: 2000 + 3000 + 4000 = 9000
    # output: 300 + 400 + 800 = 1500
    assert bm["claude-sonnet-4-6"]["input_tokens"] == 9000
    assert bm["claude-sonnet-4-6"]["output_tokens"] == 1500

    # opus: implement_baseline (both) + verify_against_rubric (second)
    # input: 10000 + 12000 + 5000 = 27000
    # output: 5000 + 6000 + 1000 = 12000
    assert bm["claude-opus-4-7"]["input_tokens"] == 27000
    assert bm["claude-opus-4-7"]["output_tokens"] == 12000


def test_grand_total(tmp_path):
    run_dir = tmp_path / "run1"
    _write_ledger(run_dir, FIXTURE_ROWS)
    data = _aggregate_tokens_total(run_dir)

    gt = data["grand_total"]
    assert gt["input_tokens"] == 36000  # 5000+22000+9000
    assert gt["output_tokens"] == 13500  # 700+11000+1800 = 13500
    assert gt["calls"] == 6
    assert gt["cache_creation_input_tokens"] == 500  # 200 + 300
    assert gt["cache_read_input_tokens"] == 150       # 50 + 100


def test_schema_version(tmp_path):
    run_dir = tmp_path / "run1"
    _write_ledger(run_dir, FIXTURE_ROWS)
    data = _aggregate_tokens_total(run_dir)
    assert data["schema_version"] == 1
    assert "computed_at_utc" in data


def test_empty_ledger_produces_zeros(tmp_path):
    run_dir = tmp_path / "run_empty"
    _write_ledger(run_dir, [])
    data = _aggregate_tokens_total(run_dir)
    assert data["grand_total"]["input_tokens"] == 0
    assert data["grand_total"]["calls"] == 0
    assert data["by_primitive"] == {}
    assert data["by_model"] == {}


def test_missing_ledger_produces_zeros(tmp_path):
    run_dir = tmp_path / "run_no_ledger"
    run_dir.mkdir()
    # No cost_ledger.jsonl created
    data = _aggregate_tokens_total(run_dir)
    assert data["grand_total"]["input_tokens"] == 0
    assert data["grand_total"]["calls"] == 0


def test_write_tokens_total_atomic(tmp_path, monkeypatch):
    """Ensures tokens_total.json is written via tmp + os.replace."""
    run_dir = tmp_path / "run1"
    _write_ledger(run_dir, FIXTURE_ROWS)

    replaces = []
    real_replace = os.replace
    monkeypatch.setattr(
        os, "replace",
        lambda src, dst: (replaces.append(str(src)), real_replace(src, dst))[1],
    )

    path = write_tokens_total(run_dir)
    assert path.exists()
    assert any(".tmp" in r for r in replaces), "expected .tmp in os.replace call"


def test_write_tokens_total_content(tmp_path):
    run_dir = tmp_path / "run1"
    _write_ledger(run_dir, FIXTURE_ROWS)
    path = write_tokens_total(run_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["grand_total"]["calls"] == 6
    assert "by_primitive" in data
    assert "by_model" in data


def test_alias_fields_supported(tmp_path):
    """cost_ledger rows may use 'agent_id' instead of 'primitive', or 'tokens_in'/'tokens_out'."""
    run_dir = tmp_path / "run1"
    _write_ledger(run_dir, [
        # agent_id alias
        {"agent_id": "detect_environment", "model": "gpt-5",
         "tokens_in": 1000, "tokens_out": 200,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    ])
    data = _aggregate_tokens_total(run_dir)
    bp = data["by_primitive"]
    assert "detect_environment" in bp
    assert bp["detect_environment"]["input_tokens"] == 1000
    assert bp["detect_environment"]["output_tokens"] == 200


def test_malformed_line_is_skipped(tmp_path):
    """Malformed JSONL lines are silently skipped."""
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "cost_ledger.jsonl").write_text(
        "not json\n"
        + json.dumps({"primitive": "plan_reproduction", "model": "gpt-5",
                      "input_tokens": 300, "output_tokens": 60}) + "\n",
        encoding="utf-8",
    )
    data = _aggregate_tokens_total(run_dir)
    assert data["grand_total"]["calls"] == 1
    assert data["by_primitive"]["plan_reproduction"]["input_tokens"] == 300
