"""Tests for BES Phase 4A — ALFWorld env-reuse (A1) + reward shaping (A2).

Spec: ``docs/superpowers/specs/2026-06-07-bes-integration/phase-4-frontier-and-honesty-guard.md`` §2.

Two GATED, default-OFF behaviours are added to ``alfworld_env.ALFWorldEnv``:

  * **A1 (``REPROLAB_ALFWORLD_ENV_REUSE``)** — construct the underlying TextWorld
    env once per instance and ``reset()`` it in place across episodes instead of
    rebuilding it every episode (the ~82× reload tax). Flag OFF ⇒ rebuild every
    episode (byte-identical to the original).
  * **A2 (``REPROLAB_ALFWORLD_SHAPING``)** — emit dense INTERMEDIATE sub-goal
    progress credit on non-terminal steps, while the terminal ``float(won)`` /
    ``info["won"]`` stay the SEPARATE authoritative signal. Flag OFF ⇒ every
    intermediate step's reward is exactly ``0.0`` (byte-identical to the original
    sparse terminal-only reward).

The CRITICAL INVARIANT under test: with BOTH flags UNSET, behaviour is
byte-identical to today. The flag-off parity tests assert that directly.

The base venv has ``alfworld`` + ``textworld`` installed but NOT the downloaded
game data (``ALFWORLD_DATA`` unset), so the real game path cannot run in CI.
These tests therefore drive everything through the injected ``tw_env_factory`` /
``_env`` seam (a scripted fake) and unit-test the shaping helpers in isolation —
no alfworld data, no GPU, no network. Imported via the FLAT copyable-module path
from THIS repo's ``backend/agents/rlm`` (so the edits under test are exercised).
"""

from __future__ import annotations

import sys

import pytest

# Import the copyable helper + base from THIS repo (the files this change edits).
# Strip the flat dir immediately after import so it does not leak a second
# identity for package modules into the rest of the session.
_RLM_DIR = "/home/sww35/openresearch/backend/agents/rlm"
sys.path.insert(0, _RLM_DIR)
try:
    import alfworld_env  # noqa: E402
    from alfworld_env import ALFWorldEnv  # noqa: E402
finally:
    while _RLM_DIR in sys.path:
        sys.path.remove(_RLM_DIR)


# ---------------------------------------------------------------------------
# A scripted fake TextWorld batched gym env with a "reach the apple, open the
# fridge, take the apple to win" arc. Returns the real batched shape (length-1
# lists + dict-of-lists) so the env's unwrapping runs exactly as in production.
# ---------------------------------------------------------------------------


class _ShapingTWEnv:
    """Goal: put a cool apple in the countertop. Arc exercised by the tests:

      * ``go to fridge 1``      → obs mentions 'apple' (reach_target sub-goal)
      * ``open fridge 1``       → obs 'You open the fridge 1' (open_receptacle)
      * ``look``                → 'Nothing happens.' (no sub-goal — no progress)
      * ``take apple 1 ...``    → terminal won=True (authoritative success)
    """

    def __init__(self) -> None:
        self.seeded: int | None = None
        self.reset_count = 0
        self.commands_seen: list[str] = []

    def seed(self, seed: int) -> list[int]:
        self.seeded = seed
        return [seed]

    def reset(self):
        self.reset_count += 1
        obs = (
            "-= Welcome to TextWorld =-\nYou are in the kitchen.\n"
            "Your task is to: put a cool apple in the countertop."
        )
        infos = {"won": [False], "admissible_commands": [["go to fridge 1", "look"]]}
        return [obs], infos

    def step(self, commands):
        assert isinstance(commands, (list, tuple)), "commands must be batched"
        cmd = commands[0]
        self.commands_seen.append(cmd)

        if cmd == "go to fridge 1":
            # Reaching the receptacle that holds the goal object (apple visible).
            obs = "You arrive at fridge 1. On the fridge 1 you see an apple 1."
            return [obs], [0.0], [False], {"won": [False]}
        if cmd == "open fridge 1":
            obs = "You open the fridge 1. The fridge 1 contains an apple 1."
            return [obs], [0.0], [False], {"won": [False]}
        if cmd == "take apple 1 from fridge 1":
            obs = "You take the apple 1 from the fridge 1. Task complete!"
            return [obs], [1.0], [True], {"won": [True]}
        # Anything else: a valid step that makes no progress toward a sub-goal.
        obs = "Nothing happens."
        return [obs], [0.0], [False], {"won": [False]}


@pytest.fixture(autouse=True)
def _clear_phase4_flags(monkeypatch):
    """Every test starts with both Phase-4A flags UNSET (the default-OFF baseline)."""
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    monkeypatch.delenv(alfworld_env._SHAPING_FLAG, raising=False)
    monkeypatch.delenv(alfworld_env._ENV_REUSE_FLAG, raising=False)


# ---------------------------------------------------------------------------
# (1) Flag OFF → byte-parity: intermediate reward == 0.0, terminal == float(won)
# ---------------------------------------------------------------------------


def test_shaping_off_intermediate_reward_is_zero():
    """With REPROLAB_ALFWORLD_SHAPING unset, every non-terminal step rewards 0.0.

    This is the byte-identical baseline: even a step that WOULD earn shaped credit
    (reaching the apple, opening the fridge) returns exactly 0.0 when the flag is
    off — proving shaping is fully gated.
    """
    env = ALFWorldEnv(_env=_ShapingTWEnv())
    env.reset(seed=0)

    res_reach = env.step("go to fridge 1")  # would be a sub-goal if shaping on
    assert res_reach.done is False
    assert res_reach.reward == 0.0
    assert res_reach.info == {}  # no "shaped" key when off (original info shape)

    res_open = env.step("open fridge 1")  # would be a sub-goal if shaping on
    assert res_open.done is False
    assert res_open.reward == 0.0
    assert res_open.info == {}


def test_shaping_off_terminal_reward_is_float_won():
    """Flag OFF: the terminal step still carries reward == float(won) and info.won.

    A winning episode → reward 1.0; the terminal contract is unchanged by the
    (disabled) shaping path.
    """
    env = ALFWorldEnv(_env=_ShapingTWEnv())
    env.reset(seed=0)
    env.step("go to fridge 1")
    env.step("open fridge 1")
    res = env.step("take apple 1 from fridge 1")
    assert res.done is True
    assert res.reward == 1.0  # == float(won)
    assert res.info["won"] is True
    assert res.info["success"] is True
    assert env.episode_reward() == 1.0


def test_shaping_off_terminal_reward_is_zero_on_loss():
    """Flag OFF: a non-winning terminal (turn cap) → reward float(False) == 0.0."""
    env = ALFWorldEnv(_env=_ShapingTWEnv())
    env.max_turns = 2
    env.reset(seed=0)
    env.step("look")
    res = env.step("look")  # hits the turn cap without winning
    assert res.done is True
    assert res.reward == 0.0  # == float(won == False)
    assert res.info["won"] is False


# ---------------------------------------------------------------------------
# (2) Flag ON → dense intermediate credit; terminal won unchanged + separate
# ---------------------------------------------------------------------------


def test_shaping_on_progress_step_rewards_positive(monkeypatch):
    """Flag ON: reaching the target object yields reward > 0 on a non-terminal step."""
    monkeypatch.setenv(alfworld_env._SHAPING_FLAG, "1")
    env = ALFWorldEnv(_env=_ShapingTWEnv())
    env.reset(seed=0)

    res = env.step("go to fridge 1")  # obs mentions 'apple' → reach_target sub-goal
    assert res.done is False
    assert res.reward > 0.0
    assert res.reward == pytest.approx(alfworld_env._SHAPE_REACH_TARGET)
    assert res.info.get("shaped") == pytest.approx(alfworld_env._SHAPE_REACH_TARGET)
    # Shaping is intermediate ONLY — it never claims terminal success.
    assert res.info.get("won") is False
    assert res.info.get("success") is False


def test_shaping_on_no_progress_step_rewards_zero(monkeypatch):
    """Flag ON: a step that makes no sub-goal progress still rewards exactly 0.0."""
    monkeypatch.setenv(alfworld_env._SHAPING_FLAG, "1")
    env = ALFWorldEnv(_env=_ShapingTWEnv())
    env.reset(seed=0)

    res = env.step("look")  # 'Nothing happens.' — no sub-goal reached
    assert res.done is False
    assert res.reward == 0.0


def test_shaping_on_subgoal_credited_once(monkeypatch):
    """Flag ON: each sub-goal pays out at most once per episode (no farming)."""
    monkeypatch.setenv(alfworld_env._SHAPING_FLAG, "1")
    env = ALFWorldEnv(_env=_ShapingTWEnv())
    env.reset(seed=0)

    first = env.step("go to fridge 1")
    assert first.reward == pytest.approx(alfworld_env._SHAPE_REACH_TARGET)
    # Re-reaching the same object earns nothing further.
    again = env.step("go to fridge 1")
    assert again.reward == 0.0


def test_shaping_on_terminal_won_unchanged_and_separate(monkeypatch):
    """Flag ON: shaped intermediate credit accrues, but the terminal reward is
    STILL exactly float(won) and info["won"] is the authoritative signal.

    Proves the Codex constraint: shaping does not replace or contaminate the
    terminal success signal — a held-out terminal-success eval is unaffected.
    """
    monkeypatch.setenv(alfworld_env._SHAPING_FLAG, "1")
    env = ALFWorldEnv(_env=_ShapingTWEnv())
    env.reset(seed=0)

    r_reach = env.step("go to fridge 1")
    assert r_reach.reward == pytest.approx(alfworld_env._SHAPE_REACH_TARGET)
    r_open = env.step("open fridge 1")
    assert r_open.reward == pytest.approx(alfworld_env._SHAPE_OPEN_RECEPTACLE)

    term = env.step("take apple 1 from fridge 1")
    assert term.done is True
    # Terminal reward is float(won) EXACTLY — not the shaped value, not a sum.
    assert term.reward == 1.0
    assert term.reward == float(term.info["won"])
    assert term.info["won"] is True
    assert term.info["success"] is True
    # The terminal info carries the authoritative success, never a "shaped" key.
    assert "shaped" not in term.info
    # episode_reward() is the terminal scalar (float(won)), unaffected by shaping.
    assert env.episode_reward() == 1.0


def test_shaping_on_terminal_won_false_still_float_won(monkeypatch):
    """Flag ON but the episode loses: terminal reward is float(False)==0.0 even
    though intermediate shaped credit was emitted earlier — terminal stays the
    separate authoritative signal."""
    monkeypatch.setenv(alfworld_env._SHAPING_FLAG, "1")
    env = ALFWorldEnv(_env=_ShapingTWEnv())
    env.max_turns = 3
    env.reset(seed=0)

    r = env.step("go to fridge 1")  # earns shaped credit
    assert r.reward > 0.0
    env.step("look")
    term = env.step("look")  # turn cap → terminal, did NOT win
    assert term.done is True
    assert term.reward == 0.0  # float(won == False)
    assert term.info["won"] is False


# ---------------------------------------------------------------------------
# Shaping helpers in isolation (no env construction at all)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "goal,expected",
    [
        ("Your task is to: put a cool apple in the countertop.", "apple"),
        ("put a cool apple in the countertop", "apple"),
        ("heat some egg and put it in the fridge", "egg"),
        ("find two mug and put them in the cabinet", "mug"),
        ("clean a fork and put it in the drawer", "fork"),
        ("", ""),
    ],
)
def test_extract_target_noun(goal, expected):
    assert alfworld_env._extract_target_noun(goal) == expected


def test_flag_on_helper_truthy_values(monkeypatch):
    for truthy in ("1", "true", "TRUE", "yes", "on", "On"):
        monkeypatch.setenv("REPROLAB_TEST_FLAG_X", truthy)
        assert alfworld_env._flag_on("REPROLAB_TEST_FLAG_X") is True
    for falsy in ("0", "false", "no", "off", "", "  "):
        monkeypatch.setenv("REPROLAB_TEST_FLAG_X", falsy)
        assert alfworld_env._flag_on("REPROLAB_TEST_FLAG_X") is False
    monkeypatch.delenv("REPROLAB_TEST_FLAG_X", raising=False)
    assert alfworld_env._flag_on("REPROLAB_TEST_FLAG_X") is False


def test_shaped_step_reward_is_zero_when_flag_off():
    """The shaping function itself returns 0.0 with the flag off — total parity at
    the seam, independent of any observation content."""
    env = ALFWorldEnv(_env=_ShapingTWEnv())
    env.reset(seed=0)
    env._target_noun = "apple"
    # Even with a clearly-progressing action+obs, flag-off → 0.0.
    assert env._shaped_step_reward("go to fridge 1", "you see an apple 1") == 0.0


def test_shaped_step_reward_open_receptacle_only(monkeypatch):
    """The shaping function awards the open-receptacle sub-goal independently."""
    monkeypatch.setenv(alfworld_env._SHAPING_FLAG, "1")
    env = ALFWorldEnv(_env=_ShapingTWEnv())
    env.reset(seed=0)
    env._target_noun = "nonexistent"  # so reach_target never fires
    env._credited = set()
    val = env._shaped_step_reward("open fridge 1", "You open the fridge 1.")
    assert val == pytest.approx(alfworld_env._SHAPE_OPEN_RECEPTACLE)


# ---------------------------------------------------------------------------
# (A1) Construct-once env reuse — gated, default OFF
# ---------------------------------------------------------------------------


def test_env_reuse_off_rebuilds_every_episode():
    """Flag OFF (default): the factory is invoked on EVERY reset — the original
    rebuild-per-episode behaviour (byte-identical baseline)."""
    builds: list[int] = []

    def factory(seed):
        builds.append(seed)
        return _ShapingTWEnv()

    env = ALFWorldEnv(tw_env_factory=factory)
    env.reset(seed=0)
    env.reset(seed=1)
    env.reset(seed=2)
    # The factory is invoked once per episode (the byte-parity rebuild proof).
    assert len(builds) == 3
    assert builds == [0, 1, 2]  # each episode's seed threaded through
    assert env._tw_cache is None  # nothing cached when reuse is off


def test_env_reuse_on_constructs_once_across_episodes(monkeypatch):
    """Flag ON: the env object is built ONCE and reset() in place across episodes
    (the ~82× reload-tax fix). Same underlying env instance is reused; only the
    construction counter proves single-build."""
    monkeypatch.setenv(alfworld_env._ENV_REUSE_FLAG, "1")
    builds: list[int] = []
    made: list[_ShapingTWEnv] = []

    def factory(seed):
        builds.append(seed)
        e = _ShapingTWEnv()
        made.append(e)
        return e

    env = ALFWorldEnv(tw_env_factory=factory)
    env.reset(seed=0)
    env.reset(seed=1)
    env.reset(seed=2)

    assert len(builds) == 1  # constructed exactly once
    assert env._tw_cache_built == 1
    assert env._tw_cache is made[0]  # the cached object is the one built first
    # reset() was called in place on the SAME env object for each episode (3×).
    assert made[0].reset_count == 3
    # Determinism: later resets re-seed the cached env with each episode's seed.
    assert made[0].seeded == 2


def test_env_reuse_on_still_drives_a_win(monkeypatch):
    """Flag ON: a reused env still plays a full episode correctly to a terminal win."""
    monkeypatch.setenv(alfworld_env._ENV_REUSE_FLAG, "1")
    env = ALFWorldEnv(tw_env_factory=lambda seed: _ShapingTWEnv())

    env.reset(seed=0)
    env.step("go to fridge 1")
    env.step("open fridge 1")
    res = env.step("take apple 1 from fridge 1")
    assert res.done is True and res.reward == 1.0

    # A second episode on the reused env starts clean and can win again.
    env.reset(seed=1)
    assert env.done is False
    env.step("go to fridge 1")
    env.step("open fridge 1")
    res2 = env.step("take apple 1 from fridge 1")
    assert res2.done is True and res2.reward == 1.0


def test_both_flags_off_is_byte_identical_baseline():
    """Belt-and-braces: with BOTH flags off, a full episode matches the original
    contract — intermediate rewards 0.0, terminal == float(won), one build per
    reset, nothing cached, no 'shaped' info key anywhere."""
    builds: list[int] = []
    env = ALFWorldEnv(tw_env_factory=lambda seed: (builds.append(seed), _ShapingTWEnv())[1])

    env.reset(seed=0)
    r1 = env.step("go to fridge 1")
    r2 = env.step("open fridge 1")
    r3 = env.step("take apple 1 from fridge 1")

    assert r1.reward == 0.0 and r1.info == {}
    assert r2.reward == 0.0 and r2.info == {}
    assert r3.done is True and r3.reward == 1.0 and r3.info["won"] is True
    assert "shaped" not in r3.info
    assert len(builds) == 1  # one build for the single reset
    assert env._tw_cache is None
