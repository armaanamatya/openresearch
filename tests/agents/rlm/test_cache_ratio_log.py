"""Phase 6.1 — per-run prompt-cache hit-ratio measurement (opt #6).

Survey evidence: cache_read_input_tokens = 0 across every run in runs/ — the 32KB
system prompt is re-billed each iteration. This read-only helper surfaces the ratio
so the leak is measurable and the 6.2 fix (OAuth cache_control breakpoint) can be
decided on data, not guesses.
"""
from __future__ import annotations

import json

from backend.agents.rlm.run import _cache_hit_ratio


def _write_ledger(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_cache_hit_ratio_zero_when_no_reads(tmp_path):
    _write_ledger(
        tmp_path / "cost_ledger.jsonl",
        [
            {"input_tokens": 1000, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 500},
            {"input_tokens": 2000, "cache_read_input_tokens": 0},
        ],
    )
    assert _cache_hit_ratio(tmp_path) == 0.0


def test_cache_hit_ratio_fraction(tmp_path):
    _write_ledger(
        tmp_path / "cost_ledger.jsonl",
        [
            {"input_tokens": 1000, "cache_read_input_tokens": 3000},
        ],
    )
    # 3000 / (3000 + 1000) = 0.75
    assert _cache_hit_ratio(tmp_path) == 0.75


def test_cache_hit_ratio_none_when_no_llm_tokens(tmp_path):
    _write_ledger(
        tmp_path / "cost_ledger.jsonl",
        [{"input_tokens": 0, "cache_read_input_tokens": 0}],  # pure file-IO primitive
    )
    assert _cache_hit_ratio(tmp_path) is None


def test_cache_hit_ratio_missing_file_is_none(tmp_path):
    assert _cache_hit_ratio(tmp_path) is None


def test_cache_hit_ratio_tolerates_corrupt_lines(tmp_path):
    (tmp_path / "cost_ledger.jsonl").write_text(
        "{ not json\n" + json.dumps({"input_tokens": 100, "cache_read_input_tokens": 100}) + "\n"
    )
    assert _cache_hit_ratio(tmp_path) == 0.5
