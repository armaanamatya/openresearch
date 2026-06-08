"""Tests for the real ALFWorld TextWorld agentic env (`alfworld_env.py`).

The 2026-05-31 run de-scoped ALFWorld entirely; this module restores it as a
real ``alfworld`` TextWorld environment.  The base venv has ``alfworld`` +
``textworld`` installed but **not** the downloaded game data (``ALFWORLD_DATA``
unset), so the real game path cannot run in CI.  These tests therefore:

  * assert ``available()`` reports ``(False, <data reason>)`` when the data is
    absent — without raising (the fail-soft availability contract);
  * drive a **full episode against an injected FAKE TextWorld env** (the
    ``tw_env_factory`` / ``_env`` seam) — no alfworld data, no GPU, no network —
    proving reset → step → terminal ``won=True`` yields reward 1.0 / done /
    ``info.success`` and that the ``> `` / ``action:`` scaffolding is stripped;
  * assert a bad action against the fake env returns an observation, never raises.

Imported via the FLAT copyable-module path (``sys.path.insert`` →
``import alfworld_env``) — the same way the agent's generated ``code/`` imports
it after ``run_with_sdk`` copies the helper in.
"""

from __future__ import annotations

import sys

import pytest

# The module is a copyable helper that uses a FLAT ``from sdar_env_base import``;
# make both it and its base importable as top-level modules (mirrors how the
# agent sandbox imports them out of ``code/``). Remove the flat dir right after
# import so it does not leak into the rest of the session (a lingering entry gives
# package modules like rubric_guard a second identity, breaking unrelated tests).
_RLM_DIR = "/home/sww35/openresearch/backend/agents/rlm"
sys.path.insert(0, _RLM_DIR)
# Drop any stale bare-name identity a sibling test cached from another worktree, so
# this loads THIS repo's modules. (The merge brought alfworld_env.py into this repo;
# the old hard-coded openresearch-fullscope path tested a stale pre-merge copy.)
for _stale in ("alfworld_env", "sdar_env_base"):
    sys.modules.pop(_stale, None)
try:
    import alfworld_env  # noqa: E402
    from alfworld_env import ALFWorldEnv  # noqa: E402
finally:
    while _RLM_DIR in sys.path:
        sys.path.remove(_RLM_DIR)


# ---------------------------------------------------------------------------
# A fake TextWorld batched gym env (the seam the real get_environment returns).
# Mirrors the real API: reset() -> (obs_list, infos_dict);
# step(cmds_list) -> (obs_list, scores_list, dones_list, infos_dict). All BATCHED.
# ---------------------------------------------------------------------------


class _FakeTWEnv:
    """A scripted ALFWorld game: reach the fridge, then take the apple to win.

    Returns values in the *real* batched shape (length-1 lists + dict-of-lists)
    so the env's unwrapping is exercised exactly as it would be in production.
    """

    def __init__(self) -> None:
        self.seeded: int | None = None
        self._opened = False
        self._won = False
        self._reset_count = 0
        self.commands_seen: list[str] = []

    def seed(self, seed: int) -> list[int]:
        self.seeded = seed
        return [seed]

    def reset(self):
        self._reset_count += 1
        self._opened = False
        self._won = False
        obs = (
            "-= Welcome to TextWorld =-\nYou are in the kitchen.\n"
            "Your task is to: put a cool apple in the countertop."
        )
        infos = {"won": [False], "admissible_commands": [["go to fridge 1", "look"]]}
        return [obs], infos

    def step(self, commands):
        # The real batched env asserts a list/tuple of commands.
        assert isinstance(commands, (list, tuple)), "commands must be batched"
        cmd = commands[0]
        self.commands_seen.append(cmd)

        if cmd == "go to fridge 1":
            obs = "You arrive at fridge 1. On the fridge 1 you see an apple 1."
            return [obs], [0.0], [False], {"won": [False]}

        if cmd == "take apple 1 from fridge 1":
            self._won = True
            obs = "You take the apple 1 from the fridge 1. Task complete!"
            # Terminal: won=True, done=True, score 1.0 — all batched.
            return [obs], [1.0], [True], {"won": [True]}

        # Anything else: a valid step that makes no progress.
        obs = "Nothing happens."
        return [obs], [0.0], [False], {"won": [False]}


class _RaisingTWEnv(_FakeTWEnv):
    """A fake whose step() raises on a specific command (fail-soft probe)."""

    def step(self, commands):
        if commands[0] == "explode":
            raise RuntimeError("boom from the underlying env")
        return super().step(commands)


# ---------------------------------------------------------------------------
# available() — fail-soft when game data is absent
# ---------------------------------------------------------------------------


def test_available_false_when_alfworld_data_unset(monkeypatch):
    """No ALFWORLD_DATA → (False, reason mentioning data), no raise."""
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    ok, reason = ALFWorldEnv.available()
    assert ok is False
    assert isinstance(reason, str) and reason
    assert "data" in reason.lower()
    assert "alfworld_data" in reason.lower()


def test_available_false_when_data_dir_missing(monkeypatch, tmp_path):
    """A pointed-at dir that doesn't exist → (False, reason), no raise."""
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    missing = tmp_path / "no_such_alfworld_data"
    ok, reason = ALFWorldEnv.available(data_dir=str(missing))
    assert ok is False
    assert "does not exist" in reason or "data" in reason.lower()


def test_available_false_when_dir_has_no_games(monkeypatch, tmp_path):
    """An existing but empty data dir (no traj_data.json) → (False, reason)."""
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    empty = tmp_path / "alfworld_data"
    empty.mkdir()
    ok, reason = ALFWorldEnv.available(data_dir=str(empty))
    assert ok is False
    assert "no alfworld games" in reason.lower() or "traj_data" in reason.lower()


def test_available_never_raises(monkeypatch):
    """Contract: available() is total — returns a bool+str, never raises."""
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    result = ALFWorldEnv.available()
    assert isinstance(result, tuple) and len(result) == 2
    assert isinstance(result[0], bool) and isinstance(result[1], str)


# ---------------------------------------------------------------------------
# Full episode against an injected FAKE TextWorld env (no data / GPU / network)
# ---------------------------------------------------------------------------


def test_full_episode_to_win_via_injected_factory(monkeypatch):
    """reset → '> go to fridge 1' → 'take apple 1 from fridge 1' → won=True.

    Uses the ``tw_env_factory`` seam so the real alfworld data is never touched.
    Asserts the terminal reward is 1.0, done, info.success True, and that the
    leading ``> `` scaffolding was stripped before reaching the env.
    """
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    fake = _FakeTWEnv()
    env = ALFWorldEnv(tw_env_factory=lambda seed: fake)

    obs0 = env.reset(seed=7, task=None)
    assert "kitchen" in obs0.lower()
    assert fake.seeded == 7  # determinism: the seed was threaded through
    assert env.done is False

    # First action carries a leading "> " scaffolding which MUST be stripped.
    res1 = env.step("> go to fridge 1")
    assert res1.done is False
    assert res1.reward == 0.0
    assert "fridge 1" in res1.observation.lower()
    assert fake.commands_seen[-1] == "go to fridge 1"  # "> " stripped

    # Second action wins the episode.
    res2 = env.step("action: take apple 1 from fridge 1")
    assert fake.commands_seen[-1] == "take apple 1 from fridge 1"  # "action:" stripped
    assert res2.done is True
    assert res2.reward == 1.0
    assert res2.info["success"] is True
    assert res2.info["won"] is True
    assert res2.info["steps"] == 2

    # Episode accessors agree with the terminal StepResult.
    assert env.done is True
    assert env.episode_reward() == 1.0
    assert env.last_info["success"] is True


def test_full_episode_via_low_level_env_seam(monkeypatch):
    """The lower-level ``_env`` seam (pre-built env) also drives a win."""
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    fake = _FakeTWEnv()
    env = ALFWorldEnv(_env=fake)

    env.reset(seed=1)
    env.step("go to fridge 1")
    res = env.step("take apple 1 from fridge 1")
    assert res.done is True and res.reward == 1.0
    assert res.info["success"] is True


def test_initial_observation_recorded_in_transcript(monkeypatch):
    """reset records the room+goal observation and surfaces the goal in prompt."""
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    env = ALFWorldEnv(_env=_FakeTWEnv())
    obs = env.reset(seed=0)
    # The returned observation is recorded; the student prompt renders it.
    prompt = env.build_student_prompt()
    assert "kitchen" in prompt.lower()
    # The goal line was lifted into the system header.
    assert "cool apple" in prompt.lower()
    assert obs in prompt


# ---------------------------------------------------------------------------
# Admissible commands are surfaced in the recorded observation (2026-05-31 HIGH)
# ---------------------------------------------------------------------------


def test_reset_surfaces_admissible_commands_in_observation(monkeypatch):
    """reset() must append the legal commands the env reports, not just the room.

    The fake's reset infos carry ``admissible_commands=[["go to fridge 1",
    "look"]]``; the recorded observation (and the student prompt) must list them
    so the policy can choose from the legal set — the system prompt promises this.
    """
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    env = ALFWorldEnv(_env=_FakeTWEnv())
    obs = env.reset(seed=0)
    assert "admissible commands" in obs.lower()
    assert "go to fridge 1" in obs
    assert "look" in obs
    # The augmented observation is what was recorded into the transcript.
    assert obs in env.build_student_prompt()


def test_step_surfaces_admissible_commands_in_observation(monkeypatch):
    """step() observations also carry the admissible-command block."""
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)

    class _AdmissibleStepEnv(_FakeTWEnv):
        def step(self, commands):
            assert isinstance(commands, (list, tuple))
            self.commands_seen.append(commands[0])
            obs = "You arrive at fridge 1."
            infos = {"won": [False], "admissible_commands": [["open fridge 1", "look"]]}
            return [obs], [0.0], [False], infos

    env = ALFWorldEnv(_env=_AdmissibleStepEnv())
    env.reset(seed=0)
    res = env.step("go to fridge 1")
    assert res.done is False
    # Both the returned StepResult and the recorded transcript carry the block.
    assert "admissible commands" in res.observation.lower()
    assert "open fridge 1" in res.observation
    assert res.observation in env.build_student_prompt()


def test_admissible_commands_block_is_bounded(monkeypatch):
    """A huge admissible-command list is capped (count + chars) so the prompt
    can't blow the token budget."""
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)

    big_list = [f"go to location {i}" for i in range(500)]

    class _BigAdmissibleEnv(_FakeTWEnv):
        def reset(self):
            obs = "You are in the kitchen.\nYour task is to: find a thing."
            return [obs], {"won": [False], "admissible_commands": [list(big_list)]}

    env = ALFWorldEnv(_env=_BigAdmissibleEnv())
    obs = env.reset(seed=0)
    assert "admissible commands" in obs.lower()
    # Far fewer than 500 commands survive, and the block is char-bounded.
    block = obs.split("Admissible commands:", 1)[1]
    assert block.count("go to location") <= alfworld_env._MAX_ADMISSIBLE_COMMANDS
    assert len(block) <= alfworld_env._MAX_ADMISSIBLE_CHARS + 16  # + "…" suffix slack
    assert "…" in obs  # truncation marker present


def test_missing_admissible_commands_records_obs_unchanged(monkeypatch):
    """When infos has no admissible_commands the observation is recorded as-is
    (no block, no crash)."""
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)

    class _NoAdmissibleEnv(_FakeTWEnv):
        def reset(self):
            obs = "You are in the kitchen.\nYour task is to: find a thing."
            return [obs], {"won": [False]}  # no admissible_commands key

    env = ALFWorldEnv(_env=_NoAdmissibleEnv())
    obs = env.reset(seed=0)
    assert "admissible commands" not in obs.lower()
    assert "kitchen" in obs.lower()


# ---------------------------------------------------------------------------
# Fail-soft behaviour: bad action / raising env / unavailable backend
# ---------------------------------------------------------------------------


def test_bad_action_against_fake_returns_obs_no_raise(monkeypatch):
    """A nonsense (but parseable) command → a real obs from the env, not done."""
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    env = ALFWorldEnv(_env=_FakeTWEnv())
    env.reset(seed=0)
    res = env.step("frobnicate the quux")  # valid grammar, no progress
    assert res.done is False
    assert res.reward == 0.0
    assert isinstance(res.observation, str) and res.observation


def test_empty_action_nudges_without_advancing_env(monkeypatch):
    """An empty/blank action wastes a turn with a nudge; the env isn't stepped."""
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    fake = _FakeTWEnv()
    env = ALFWorldEnv(_env=fake)
    env.reset(seed=0)
    res = env.step("   ")  # blank → unparseable
    assert res.done is False
    assert "command" in res.observation.lower()
    assert fake.commands_seen == []  # env was NOT advanced


def test_env_step_exception_is_failsoft(monkeypatch):
    """If the underlying env.step() raises, step() degrades — never propagates."""
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    env = ALFWorldEnv(_env=_RaisingTWEnv())
    env.reset(seed=0)
    res = env.step("explode")  # the fake raises on this command
    assert res.done is False
    assert res.reward == 0.0
    assert isinstance(res.observation, str) and res.observation


def test_unavailable_backend_first_step_is_terminal_zero(monkeypatch):
    """No data + no injected env → reset gives a short obs, first step ends 0.0.

    This is the in-cell safety net: an unavailable env yields a clean
    zero-reward terminal step flagged ``info["unavailable"]`` instead of raising.
    """
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    env = ALFWorldEnv()  # real path, but data absent → unavailable

    obs = env.reset(seed=0)
    assert "unavailable" in obs.lower()
    assert env.done is False  # not terminal until the first step

    res = env.step("go to fridge 1")
    assert res.done is True
    assert res.reward == 0.0
    assert res.info["unavailable"] is True
    assert isinstance(res.info.get("reason"), str) and res.info["reason"]
    assert res.info["success"] is False


def test_turn_cap_terminates_episode(monkeypatch):
    """Reaching max_turns without winning → reward 0.0, done (no infinite loop)."""
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    env = ALFWorldEnv(_env=_FakeTWEnv())
    env.max_turns = 3  # shrink for the test
    env.reset(seed=0)

    res = None
    for _ in range(3):
        res = env.step("look")  # never wins
    assert res is not None
    assert res.done is True
    assert res.reward == 0.0
    assert res.info["success"] is False
    assert env.turns_taken == 3


def test_reset_with_garbage_seed_does_not_raise(monkeypatch):
    """A non-int/garbage seed must coerce to the default (0), never raise.

    ``reset(seed=...)`` coerces inside a try/except so a bad seed degrades to the
    deterministic default instead of bubbling a ValueError/TypeError out of the
    fail-soft env. Covers a string, a float, an unconvertible object, and None.
    """
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)

    class _Unconvertible:
        def __int__(self):  # pragma: no cover - exercised via int() raising
            raise ValueError("nope")

    for bad in ("not-a-seed", 3.7, object(), _Unconvertible(), [1, 2], None):
        fake = _FakeTWEnv()
        env = ALFWorldEnv(_env=fake)
        obs = env.reset(seed=bad)  # type: ignore[arg-type]  # must NOT raise
        assert isinstance(obs, str) and obs
        assert env.done is False
        # A garbage seed still yields a usable episode (drive it to a win).
        env.step("go to fridge 1")
        res = env.step("take apple 1 from fridge 1")
        assert res.done is True and res.reward == 1.0


def test_reset_is_idempotent_across_episodes(monkeypatch):
    """A second reset clears terminal state — back-to-back episodes are clean."""
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    fake = _FakeTWEnv()
    env = ALFWorldEnv(_env=fake)

    env.reset(seed=0)
    env.step("go to fridge 1")
    env.step("take apple 1 from fridge 1")
    assert env.done is True

    # Re-reset: state cleared, transcript fresh, not done.
    env.reset(seed=0)
    assert env.done is False
    assert env.turns_taken == 0
    assert env.episode_reward() == 0.0


# ---------------------------------------------------------------------------
# Module hygiene: imports cleanly, action cleaner is robust
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("> go to fridge 1", "go to fridge 1"),
        ("action: take apple 1", "take apple 1"),
        ("Action: open drawer 2", "open drawer 2"),
        (">> look", "look"),
        ("```\ngo north\n```", "go north"),
        ('"inventory"', "inventory"),
        ("go to sink 1\nThen I will wash the mug.", "go to sink 1"),
        ("", ""),
        (None, ""),
        ("   ", ""),
    ],
)
def test_clean_action_strips_scaffolding(raw, expected):
    assert alfworld_env._clean_action(raw) == expected


def test_module_has_thirty_turn_cap():
    """ALFWorld episodes are long-horizon: the spec mandates max_turns=30."""
    assert ALFWorldEnv.max_turns == 30
