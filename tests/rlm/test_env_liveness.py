"""
Tests for backend/agents/rlm/env_liveness.py (F2 env-liveness gate)
and the _append_env_health producer in agentic_rollout.py.

All tests are hermetic — no network, no filesystem I/O outside tmp_path.
pytest-socket blocks non-loopback; monkeypatch.setenv/delenv controls flags
and cell output dir.

Test cases:
  1. Default OFF → dead_envs / env_liveness_scope_gaps return [] even with a
     dead env present.
  2. Dead env: all episodes served=False, unavailable=True → flagged.
  3. Live env: at least one episode served=True → NOT flagged.
  4. No health data → NOT flagged (conservative).
  5. Producer round-trip: _append_env_health writes readable jsonl; unset dir
     → no-op.
  6. read_env_health aggregation correctness (counts/mean_turns).
  7. env_liveness_scope_gaps returns correct kind='env_setup_failed'.
"""

from __future__ import annotations

import json

import pytest

from backend.agents.rlm.env_liveness import (
    dead_envs,
    env_liveness_gate_enabled,
    env_liveness_scope_gaps,
    read_env_health,
)
from backend.agents.rlm.agentic_rollout import _append_env_health


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_health(path, records):
    """Write a list of dicts as a jsonl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _make_health_file(tmp_path, records, subdir="outputs/run_01/cell_ws"):
    """Write env_health.jsonl under tmp_path/<subdir>/env_health.jsonl."""
    health_dir = tmp_path / subdir
    health_dir.mkdir(parents=True, exist_ok=True)
    health_path = health_dir / "env_health.jsonl"
    _write_health(health_path, records)
    return health_path


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

class TestEnvLivenessGateEnabled:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_ENV_LIVENESS_GATE", raising=False)
        assert env_liveness_gate_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "YES", "ON", " 1 "])
    def test_enabled_values(self, monkeypatch, val):
        monkeypatch.setenv("OPENRESEARCH_ENV_LIVENESS_GATE", val)
        assert env_liveness_gate_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "2"])
    def test_disabled_values(self, monkeypatch, val):
        monkeypatch.setenv("OPENRESEARCH_ENV_LIVENESS_GATE", val)
        assert env_liveness_gate_enabled() is False


# ---------------------------------------------------------------------------
# Test 1: Default OFF — byte-identical invariant
# ---------------------------------------------------------------------------

class TestDefaultOff:
    def test_dead_envs_off_even_with_dead_env(self, tmp_path, monkeypatch):
        """Gate disabled: dead_envs returns [] even when a dead env is present."""
        monkeypatch.delenv("OPENRESEARCH_ENV_LIVENESS_GATE", raising=False)
        _make_health_file(tmp_path, [
            {"env": "webshop", "n_turns": 0, "reward": 0.0,
             "unavailable": True, "served": False},
        ])
        assert dead_envs(tmp_path) == []

    def test_scope_gaps_off_even_with_dead_env(self, tmp_path, monkeypatch):
        """Gate disabled: env_liveness_scope_gaps returns []."""
        monkeypatch.delenv("OPENRESEARCH_ENV_LIVENESS_GATE", raising=False)
        _make_health_file(tmp_path, [
            {"env": "webshop", "n_turns": 0, "reward": 0.0,
             "unavailable": True, "served": False},
        ])
        assert env_liveness_scope_gaps(tmp_path) == []


# ---------------------------------------------------------------------------
# Test 2: Dead env — all episodes unavailable → flagged
# ---------------------------------------------------------------------------

class TestDeadEnv:
    def test_all_unavailable_flagged(self, tmp_path, monkeypatch):
        """An env where every episode was unavailable is returned by dead_envs."""
        monkeypatch.setenv("OPENRESEARCH_ENV_LIVENESS_GATE", "1")
        _make_health_file(tmp_path, [
            {"env": "webshop", "n_turns": 0, "reward": 0.0,
             "unavailable": True, "served": False},
            {"env": "webshop", "n_turns": 0, "reward": 0.0,
             "unavailable": True, "served": False},
        ])
        result = dead_envs(tmp_path)
        assert len(result) == 1
        env_name, reason = result[0]
        assert env_name == "webshop"
        assert "webshop" in reason

    def test_all_zero_turn_flagged(self, tmp_path, monkeypatch):
        """An env where every episode is zero-turn (served=False, unavailable=False)
        is also flagged as dead."""
        monkeypatch.setenv("OPENRESEARCH_ENV_LIVENESS_GATE", "1")
        _make_health_file(tmp_path, [
            {"env": "alfworld", "n_turns": 0, "reward": 0.0,
             "unavailable": False, "served": False},
        ])
        result = dead_envs(tmp_path)
        assert len(result) == 1
        env_name, _reason = result[0]
        assert env_name == "alfworld"

    def test_scope_gaps_has_kind_env_setup_failed(self, tmp_path, monkeypatch):
        """env_liveness_scope_gaps returns the canonical exclusion kind field."""
        monkeypatch.setenv("OPENRESEARCH_ENV_LIVENESS_GATE", "1")
        _make_health_file(tmp_path, [
            {"env": "webshop", "n_turns": 0, "reward": 0.0,
             "unavailable": True, "served": False},
        ])
        gaps = env_liveness_scope_gaps(tmp_path)
        assert len(gaps) == 1
        gap = gaps[0]
        assert gap["kind"] == "env_setup_failed"
        assert gap["item"] == "webshop"
        assert "reason" in gap


# ---------------------------------------------------------------------------
# Test 3: Live env — at least one served episode → NOT flagged
# ---------------------------------------------------------------------------

class TestLiveEnv:
    def test_one_served_not_flagged(self, tmp_path, monkeypatch):
        """An env with even one served episode is never flagged."""
        monkeypatch.setenv("OPENRESEARCH_ENV_LIVENESS_GATE", "1")
        _make_health_file(tmp_path, [
            {"env": "webshop", "n_turns": 0, "reward": 0.0,
             "unavailable": True, "served": False},
            {"env": "webshop", "n_turns": 5, "reward": 0.3,
             "unavailable": False, "served": True},
        ])
        assert dead_envs(tmp_path) == []

    def test_all_served_not_flagged(self, tmp_path, monkeypatch):
        """A fully healthy env is not flagged."""
        monkeypatch.setenv("OPENRESEARCH_ENV_LIVENESS_GATE", "1")
        _make_health_file(tmp_path, [
            {"env": "alfworld", "n_turns": 10, "reward": 1.0,
             "unavailable": False, "served": True},
            {"env": "alfworld", "n_turns": 8, "reward": 0.5,
             "unavailable": False, "served": True},
        ])
        assert dead_envs(tmp_path) == []


# ---------------------------------------------------------------------------
# Test 4: No health data → NOT flagged (conservative)
# ---------------------------------------------------------------------------

class TestNoHealthData:
    def test_no_health_files_not_flagged(self, tmp_path, monkeypatch):
        """When there are no env_health.jsonl files, nothing is flagged."""
        monkeypatch.setenv("OPENRESEARCH_ENV_LIVENESS_GATE", "1")
        assert dead_envs(tmp_path) == []

    def test_health_outside_outputs_not_flagged(self, tmp_path, monkeypatch):
        """A health file NOT under outputs/ is ignored (conservative)."""
        monkeypatch.setenv("OPENRESEARCH_ENV_LIVENESS_GATE", "1")
        # Place health file outside any outputs/ subtree
        top_level = tmp_path / "env_health.jsonl"
        _write_health(top_level, [
            {"env": "webshop", "n_turns": 0, "reward": 0.0,
             "unavailable": True, "served": False},
        ])
        assert dead_envs(tmp_path) == []

    def test_empty_dir_not_flagged(self, tmp_path, monkeypatch):
        """An empty code_dir returns no dead envs."""
        monkeypatch.setenv("OPENRESEARCH_ENV_LIVENESS_GATE", "1")
        assert dead_envs(tmp_path) == []


# ---------------------------------------------------------------------------
# Test 5: Producer round-trip — _append_env_health
# ---------------------------------------------------------------------------

class TestProducerRoundTrip:
    def test_write_and_read(self, tmp_path, monkeypatch):
        """_append_env_health writes a readable jsonl; read_env_health picks it up."""
        monkeypatch.setenv("OPENRESEARCH_ENV_LIVENESS_GATE", "1")
        out_dir = tmp_path / "outputs" / "run_01" / "cell_ws"
        out_dir.mkdir(parents=True)
        monkeypatch.setenv("OPENRESEARCH_CELL_OUTPUT_DIR", str(out_dir))

        _append_env_health({
            "env": "webshop",
            "n_turns": 0,
            "reward": 0.0,
            "unavailable": True,
            "served": False,
        })

        # The file must exist and be readable.
        health_path = out_dir / "env_health.jsonl"
        assert health_path.exists(), "env_health.jsonl was not created"
        lines = health_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["env"] == "webshop"
        assert rec["unavailable"] is True
        assert rec["served"] is False

        # read_env_health must aggregate this record.
        health = read_env_health(tmp_path)
        assert "webshop" in health
        assert health["webshop"]["episodes_total"] == 1
        assert health["webshop"]["episodes_served"] == 0
        assert health["webshop"]["episodes_unavailable"] == 1

    def test_unset_env_var_noop(self, tmp_path, monkeypatch):
        """_append_env_health is a complete no-op when OPENRESEARCH_CELL_OUTPUT_DIR is unset."""
        monkeypatch.delenv("OPENRESEARCH_CELL_OUTPUT_DIR", raising=False)
        # Should not raise and should not create any file.
        _append_env_health({"env": "webshop", "n_turns": 0, "reward": 0.0,
                             "unavailable": True, "served": False})
        # Confirm nothing was written anywhere under tmp_path.
        written = list(tmp_path.rglob("env_health.jsonl"))
        assert written == [], f"unexpected files created: {written}"

    def test_multiple_appends(self, tmp_path, monkeypatch):
        """Multiple _append_env_health calls append distinct lines."""
        out_dir = tmp_path / "outputs" / "run_01" / "cell_ws"
        out_dir.mkdir(parents=True)
        monkeypatch.setenv("OPENRESEARCH_CELL_OUTPUT_DIR", str(out_dir))

        for i in range(3):
            _append_env_health({
                "env": "alfworld",
                "n_turns": i + 1,
                "reward": float(i) * 0.1,
                "unavailable": False,
                "served": True,
            })

        lines = (out_dir / "env_health.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# Test 6: read_env_health aggregation correctness
# ---------------------------------------------------------------------------

class TestReadEnvHealthAggregation:
    def test_count_aggregation(self, tmp_path):
        """read_env_health correctly aggregates episodes_total/served/unavailable."""
        records = [
            {"env": "webshop", "n_turns": 0, "reward": 0.0,
             "unavailable": True, "served": False},
            {"env": "webshop", "n_turns": 0, "reward": 0.0,
             "unavailable": True, "served": False},
            {"env": "webshop", "n_turns": 7, "reward": 0.8,
             "unavailable": False, "served": True},
            {"env": "alfworld", "n_turns": 5, "reward": 1.0,
             "unavailable": False, "served": True},
        ]
        _make_health_file(tmp_path, records)
        health = read_env_health(tmp_path)

        assert "webshop" in health
        ws = health["webshop"]
        assert ws["episodes_total"] == 3
        assert ws["episodes_served"] == 1
        assert ws["episodes_unavailable"] == 2

        assert "alfworld" in health
        al = health["alfworld"]
        assert al["episodes_total"] == 1
        assert al["episodes_served"] == 1
        assert al["episodes_unavailable"] == 0

    def test_mean_turns(self, tmp_path):
        """read_env_health computes mean_turns correctly."""
        records = [
            {"env": "webshop", "n_turns": 2, "reward": 0.0,
             "unavailable": False, "served": True},
            {"env": "webshop", "n_turns": 4, "reward": 0.0,
             "unavailable": False, "served": True},
        ]
        _make_health_file(tmp_path, records)
        health = read_env_health(tmp_path)
        assert abs(health["webshop"]["mean_turns"] - 3.0) < 1e-6

    def test_multiple_health_files_same_env(self, tmp_path):
        """Health records from different cell dirs are merged by env name."""
        _make_health_file(tmp_path, [
            {"env": "webshop", "n_turns": 0, "reward": 0.0,
             "unavailable": True, "served": False},
        ], subdir="outputs/run_01/cell_ws_seed0")
        _make_health_file(tmp_path, [
            {"env": "webshop", "n_turns": 0, "reward": 0.0,
             "unavailable": True, "served": False},
        ], subdir="outputs/run_01/cell_ws_seed1")
        health = read_env_health(tmp_path)
        assert health["webshop"]["episodes_total"] == 2

    def test_missing_dir_returns_empty(self, tmp_path):
        """read_env_health on a non-existent directory returns {}."""
        assert read_env_health(tmp_path / "does_not_exist") == {}

    def test_malformed_lines_skipped(self, tmp_path):
        """Malformed JSON lines are silently skipped; valid lines still aggregated."""
        health_dir = tmp_path / "outputs" / "run_01" / "cell"
        health_dir.mkdir(parents=True)
        (health_dir / "env_health.jsonl").write_text(
            'not-json\n'
            '{"env": "webshop", "n_turns": 3, "reward": 0.5, "unavailable": false, "served": true}\n'
            '{broken\n',
            encoding="utf-8",
        )
        health = read_env_health(tmp_path)
        assert health["webshop"]["episodes_total"] == 1
        assert health["webshop"]["episodes_served"] == 1
