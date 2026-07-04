"""
Tests for WS-C env-level health recording — backend/agents/rlm/sdar_env_base.py.

``agentic_rollout.rollout_episode`` already writes one env_health.jsonl row per
episode it drives, but an agent-written trainer that steps a concrete ``*Env``
directly (never calling ``rollout_episode``) previously produced NO health rows
at all, leaving the F2 env-liveness gate (env_liveness.py) with no data. This
module verifies ``AgenticEnv.__init_subclass__`` transparently wraps every
concrete subclass's ``reset``/``step`` so a health row is emitted regardless of
which code drives the env, and that ``rollout_episode``'s ownership claim
prevents double-counting when it DOES drive the episode.

All tests are hermetic — no network, no filesystem I/O outside tmp_path.
"""

from __future__ import annotations

import json

from backend.agents.rlm.agentic_rollout import rollout_episode
from backend.agents.rlm.env_liveness import read_env_health
from backend.agents.rlm.sdar_env_base import AgenticEnv, StepResult


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class ManualEnv(AgenticEnv):
    """A minimal concrete AgenticEnv the tests drive directly (not via
    ``rollout_episode``).  ``step`` finishes the episode (reward=1.0, no
    ``unavailable``) whenever the action is the literal string ``"done"``;
    otherwise it records the turn and continues.
    """

    max_turns = 10
    env_name = "manual_demo"

    def reset(self, *, seed=None, task=None) -> str:
        self._start_episode(system="sys")
        obs = "obs0"
        self._record_obs(obs)
        return obs

    def step(self, action: str) -> StepResult:
        self._record_act(action)
        if action == "done":
            self._finish(1.0, info={"success": True})
            return StepResult(observation="fin", reward=1.0, done=True, info={"success": True})
        obs = f"obs{self._turns_taken}"
        self._record_obs(obs)
        return StepResult(observation=obs, reward=0.0, done=False)


class UnavailableOnResetEnv(AgenticEnv):
    """Mirrors WebShopEnv's finish-during-reset contract: a dead server
    finishes the episode with ``unavailable=True`` DURING ``reset()``, before
    any ``step()`` is ever called.
    """

    max_turns = 5
    env_name = "unavailable_demo"

    def reset(self, *, seed=None, task=None) -> str:
        self._start_episode(system="sys")
        self._finish(0.0, info={"unavailable": True, "reason": "server down"})
        return "[unavailable]"

    def step(self, action: str) -> StepResult:  # pragma: no cover - never reached
        self._record_act(action)
        return StepResult(observation="unreachable", reward=0.0, done=True)


class CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]


def _read_rows(out_dir) -> list[dict]:
    path = out_dir / "env_health.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 1. Manual drive: reset -> 3 steps -> done -> exactly one row
# ---------------------------------------------------------------------------

def test_manual_drive_three_steps_then_done_writes_one_row(tmp_path, monkeypatch):
    out_dir = tmp_path / "outputs" / "run_01" / "cell"
    out_dir.mkdir(parents=True)
    monkeypatch.setenv("OPENRESEARCH_CELL_OUTPUT_DIR", str(out_dir))

    env = ManualEnv()
    env.reset()
    env.step("a")
    env.step("b")
    env.step("done")

    rows = _read_rows(out_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["env"] == "manual_demo"
    assert row["n_turns"] == 3
    assert row["served"] is True
    assert row["source"] == "env"


# ---------------------------------------------------------------------------
# 2. Abandoned episode: reset -> 1 step (not done) -> reset again
# ---------------------------------------------------------------------------

def test_abandoned_episode_flushed_on_next_reset(tmp_path, monkeypatch):
    out_dir = tmp_path / "outputs" / "run_01" / "cell"
    out_dir.mkdir(parents=True)
    monkeypatch.setenv("OPENRESEARCH_CELL_OUTPUT_DIR", str(out_dir))

    env = ManualEnv()
    env.reset()
    env.step("a")  # not done -- nothing flushed yet
    assert _read_rows(out_dir) == []

    env.reset()  # abandons the first episode -> flushed as n_turns=1
    rows = _read_rows(out_dir)
    assert len(rows) == 1
    assert rows[0]["n_turns"] == 1
    assert rows[0]["served"] is True

    # The second episode is tracked fresh, independent of the first.
    env.step("done")
    rows2 = _read_rows(out_dir)
    assert len(rows2) == 2
    assert rows2[1]["n_turns"] == 1
    assert rows2[1]["served"] is True


# ---------------------------------------------------------------------------
# 3. Finish-during-reset unavailable (n_turns=0 case)
# ---------------------------------------------------------------------------

def test_finish_during_reset_unavailable_flushed_on_next_reset(tmp_path, monkeypatch):
    out_dir = tmp_path / "outputs" / "run_01" / "cell"
    out_dir.mkdir(parents=True)
    monkeypatch.setenv("OPENRESEARCH_CELL_OUTPUT_DIR", str(out_dir))

    env = UnavailableOnResetEnv()
    env.reset()  # finishes with unavailable=True during reset, 0 turns
    assert _read_rows(out_dir) == []  # pending -- not flushed yet

    env.reset()  # the NEXT reset flushes the pending (abandoned) episode
    rows = _read_rows(out_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["n_turns"] == 0
    assert row["unavailable"] is True
    assert row["served"] is False
    assert row["source"] == "env"


def test_finish_during_reset_unavailable_flushed_by_explicit_helper(tmp_path, monkeypatch):
    """Same scenario, flushed via the module's flush mechanism directly instead
    of a second ``reset()`` call (the "explicit module flush helper" escape
    hatch mentioned in the design)."""
    out_dir = tmp_path / "outputs" / "run_01" / "cell"
    out_dir.mkdir(parents=True)
    monkeypatch.setenv("OPENRESEARCH_CELL_OUTPUT_DIR", str(out_dir))

    env = UnavailableOnResetEnv()
    env.reset()
    env._flush_episode()

    rows = _read_rows(out_dir)
    assert len(rows) == 1
    assert rows[0]["n_turns"] == 0
    assert rows[0]["unavailable"] is True
    assert rows[0]["served"] is False


# ---------------------------------------------------------------------------
# 4. Rollout ownership: rollout_episode drives it -> exactly one row, source=="rollout"
# ---------------------------------------------------------------------------

def test_rollout_ownership_writes_exactly_one_row_sourced_rollout(tmp_path, monkeypatch):
    out_dir = tmp_path / "outputs" / "run_01" / "cell"
    out_dir.mkdir(parents=True)
    monkeypatch.setenv("OPENRESEARCH_CELL_OUTPUT_DIR", str(out_dir))

    tok = CharTokenizer()

    def generate(prompt_text):
        text = "done"
        return text, tok.encode(text)

    env = ManualEnv()
    env.reset(seed=0, task=None)  # caller resets; rollout_episode does not
    rollout_episode(env, generate=generate, tokenizer=tok)

    rows = _read_rows(out_dir)
    assert len(rows) == 1
    assert rows[0]["source"] == "rollout"
    assert rows[0]["env"] == "manual_demo"


def test_rollout_ownership_survives_finish_during_reset(tmp_path, monkeypatch):
    """A rollout_episode call over an env that already finished DURING reset()
    (0 turns, unavailable) still writes exactly one row (the rollout's own)."""
    out_dir = tmp_path / "outputs" / "run_01" / "cell"
    out_dir.mkdir(parents=True)
    monkeypatch.setenv("OPENRESEARCH_CELL_OUTPUT_DIR", str(out_dir))

    tok = CharTokenizer()

    def generate(prompt_text):  # pragma: no cover - must never be called
        raise AssertionError("generate must not run on an already-done env")

    env = UnavailableOnResetEnv()
    env.reset(seed=0, task=None)  # finishes during reset (unavailable, 0 turns)
    rollout_episode(env, generate=generate, tokenizer=tok)

    rows = _read_rows(out_dir)
    assert len(rows) == 1
    assert rows[0]["source"] == "rollout"
    assert rows[0]["n_turns"] == 0
    assert rows[0]["unavailable"] is True
    assert rows[0]["served"] is False


# ---------------------------------------------------------------------------
# 5. No env var -> no file
# ---------------------------------------------------------------------------

def test_no_env_var_writes_no_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_CELL_OUTPUT_DIR", raising=False)

    env = ManualEnv()
    env.reset()
    env.step("done")

    assert list(tmp_path.rglob("env_health.jsonl")) == []


# ---------------------------------------------------------------------------
# 6. env var set, writer path unwritable -> no raise
# ---------------------------------------------------------------------------

def test_unwritable_output_dir_does_not_raise(tmp_path, monkeypatch):
    # Point at a path whose parent is itself a *file* -- "<file>/env_health.jsonl"
    # can never be opened, so this reliably exercises the write-failure path.
    blocking_file = tmp_path / "not_a_dir"
    blocking_file.write_text("x")
    monkeypatch.setenv("OPENRESEARCH_CELL_OUTPUT_DIR", str(blocking_file))

    env = ManualEnv()
    env.reset()
    env.step("done")  # must not raise despite the unwritable target


# ---------------------------------------------------------------------------
# 7. read_env_health aggregates rows carrying the extra "source" field
# ---------------------------------------------------------------------------

def test_read_env_health_round_trip_with_source_field(tmp_path, monkeypatch):
    out_dir = tmp_path / "outputs" / "run_01" / "cell"
    out_dir.mkdir(parents=True)
    monkeypatch.setenv("OPENRESEARCH_CELL_OUTPUT_DIR", str(out_dir))

    env = ManualEnv()
    env.reset()
    env.step("done")

    rows = _read_rows(out_dir)
    assert rows[0]["source"] == "env"  # sanity: the raw row does carry it

    health = read_env_health(tmp_path)
    assert "manual_demo" in health
    assert health["manual_demo"]["episodes_total"] == 1
    assert health["manual_demo"]["episodes_served"] == 1


# ---------------------------------------------------------------------------
# 8. Inheritance: subclass-of-subclass overriding step wrapped exactly once
# ---------------------------------------------------------------------------

def test_grandchild_override_wrapped_once_no_double_rows(tmp_path, monkeypatch):
    out_dir = tmp_path / "outputs" / "run_01" / "cell"
    out_dir.mkdir(parents=True)
    monkeypatch.setenv("OPENRESEARCH_CELL_OUTPUT_DIR", str(out_dir))

    class MiddleEnv(ManualEnv):
        """Inherits reset/step verbatim -- must NOT be re-wrapped."""

    class GrandchildEnv(MiddleEnv):
        """Overrides step with its own full implementation -- gets its own wrap."""

        env_name = "grandchild_demo"

        def step(self, action: str) -> StepResult:
            self._record_act(action)
            self._finish(0.5, info={})
            return StepResult(observation="fin2", reward=0.5, done=True, info={})

    # MiddleEnv redefined neither -> nothing new in its own __dict__ to wrap.
    assert "reset" not in MiddleEnv.__dict__
    assert "step" not in MiddleEnv.__dict__

    # GrandchildEnv's own step is wrapped exactly once: a marker is present and
    # it is a DISTINCT function object from ManualEnv's already-wrapped step.
    assert "step" in GrandchildEnv.__dict__
    assert getattr(GrandchildEnv.__dict__["step"], "_health_wrapped", False) is True
    assert GrandchildEnv.__dict__["step"] is not ManualEnv.__dict__["step"]

    env = GrandchildEnv()
    env.reset()
    env.step("go")  # GrandchildEnv.step finishes immediately

    rows = _read_rows(out_dir)
    assert len(rows) == 1  # exactly one row -- no double count from re-wrapping
    assert rows[0]["env"] == "grandchild_demo"
    assert rows[0]["n_turns"] == 1
    assert rows[0]["served"] is True
