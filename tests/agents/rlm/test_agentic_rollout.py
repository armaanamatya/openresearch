"""Tests for ``agentic_rollout`` — the multi-turn → flat-token + response-mask
conversion shared by every SDAR env and the agent-generated trainer.

These run on the **base venv with zero heavy deps**: ``agentic_rollout`` operates
on an *injected* ``generate`` + ``tokenizer``, so the suite injects a char-ord
fake tokenizer, a scripted *real* :class:`AgenticEnv` subclass (so the actual
transcript/prompt machinery is exercised, not a mock of it), and a fake
``generate``.  The module is imported the way the copyable helper is imported
inside an agent sandbox — top-level via ``sys.path``, not as a package — to prove
it stands alone.

The invariants asserted here are the whole reason the module exists: an
off-by-one mask silently zeros the OPSD/GRPO loss, so we pin
``len(mask) == len(sequence)``, ``sum(mask) == total response tokens``, the
mask-1 runs == each turn's ``response_ids`` in order, the turn cap, the reward
pass-through, the ``info`` merge, and the empty-generate fail-soft path.
"""

from __future__ import annotations

import sys

# Import the copyable helpers the way the agent sandbox does: flat on sys.path.
# ``agentic_rollout`` is dep-free; ``sdar_env_base`` is too, so the scripted fake
# env can be a *real* AgenticEnv subclass exercising the genuine interface.
_RLM_DIR = "/home/sww35/openresearch-fullscope/backend/agents/rlm"
sys.path.insert(0, _RLM_DIR)
try:
    import agentic_rollout as ar  # noqa: E402
    from sdar_env_base import AgenticEnv, StepResult  # noqa: E402
finally:
    # Don't leak the flat dir into the session — a lingering entry gives package
    # modules (rubric_guard, …) a second identity under a bare import, breaking
    # unrelated tests (rl_scaffold).
    while _RLM_DIR in sys.path:
        sys.path.remove(_RLM_DIR)


# --------------------------------------------------------------------------- #
# Fakes                                                                         #
# --------------------------------------------------------------------------- #
class CharTokenizer:
    """Deterministic char-ord encoder: ``len(encode(text)) == len(text)``.

    Making token count equal string length lets the tests assert exact mask sums
    and exact mask-1 spans without depending on a real BPE tokenizer.
    """

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]


class CallTokenizer:
    """A tokenizer with no ``.encode`` — only the ``tokenizer(text)["input_ids"]``
    HF call surface — to prove ``_encode`` handles both shapes."""

    def __call__(self, text: str) -> dict:
        return {"input_ids": [ord(c) for c in text]}


class ScriptedEnv(AgenticEnv):
    """A real multi-turn AgenticEnv: end when an action contains the magic word.

    Drives a genuine transcript (``_record_obs`` / ``_record_act``) so
    ``build_student_prompt`` returns the actual growing transcript the rollout
    must delta-encode.  Terminal reward is fixed (1.0) on success; if the turn
    cap is hit first the env leaves reward at 0.0 (matching the spec's
    max_turns-exhaustion contract).
    """

    max_turns = 4

    def __init__(self, magic: str = "answer", reward: float = 1.0) -> None:
        super().__init__()
        self._magic = magic
        self._reward = reward

    def reset(self, *, seed: int | None = None, task=None) -> str:
        self._start_episode(system="SYS: say the magic word.")
        obs = "OBS0: think."
        self._record_obs(obs)
        return obs

    def step(self, action: str) -> StepResult:
        self._record_act(action)
        if self._magic in (action or ""):
            obs = "OBS: solved."
            self._record_obs(obs)
            self._finish(self._reward, info={"success": True, "magic": self._magic})
            return StepResult(observation=obs, reward=self._reward, done=True,
                              info={"success": True, "magic": self._magic})
        obs = f"OBS{self._turns_taken}: keep going."
        self._record_obs(obs)
        return StepResult(observation=obs, reward=0.0, done=False)


def _runs_of_ones(mask: list[int]) -> list[list[int]]:
    """Split index positions of the mask into contiguous runs of 1s.

    Returns a list (one per run) of the *positions* that are set, so a test can
    map them back onto ``sequence_ids`` and compare against each turn's
    ``response_ids``.
    """
    runs: list[list[int]] = []
    cur: list[int] = []
    for i, m in enumerate(mask):
        if m == 1:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _assert_core_invariants(traj: ar.Trajectory) -> None:
    """The four cross-cutting invariants every trajectory must satisfy."""
    # 1. mask and sequence are the same length.
    assert len(traj.response_mask) == len(traj.sequence_ids)
    # 2. mask sum == total generated tokens across turns.
    total_resp = sum(len(t.response_ids) for t in traj.turns)
    assert sum(traj.response_mask) == total_resp
    # 3. the mask-1 runs correspond, in order, to each turn's response_ids.
    #    (Turns whose generate produced 0 tokens contribute no run — filtered out.)
    runs = _runs_of_ones(traj.response_mask)
    nonempty = [t.response_ids for t in traj.turns if t.response_ids]
    assert len(runs) == len(nonempty)
    for positions, resp_ids in zip(runs, nonempty):
        assert [traj.sequence_ids[p] for p in positions] == resp_ids
    # 4. every mask value is binary.
    assert set(traj.response_mask) <= {0, 1}


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #
def test_single_turn_solves_immediately():
    tok = CharTokenizer()

    def generate(prompt_text):
        # First (and only) turn already says the magic word → episode ends.
        text = "I think the answer is 7"
        return text, tok.encode(text)

    env = ScriptedEnv()
    env.reset(seed=0, task=None)  # caller resets (rollout does NOT)
    traj = ar.rollout_episode(env, generate=generate, tokenizer=tok)

    _assert_core_invariants(traj)
    assert len(traj.turns) == 1
    assert traj.reward == 1.0
    assert traj.info["n_turns"] == 1
    assert traj.info["success"] is True            # env.last_info merged through
    assert traj.info["magic"] == "answer"
    assert traj.info["response_lengths"] == [len("I think the answer is 7")]
    # The first delta is the WHOLE opening prompt (prev_prompt_ids was empty),
    # so sequence == encode(prompt0) + response_ids.
    expected_prompt0 = tok.encode(traj.turns[0].prompt_text)
    expected_resp0 = tok.encode("I think the answer is 7")
    assert traj.sequence_ids == expected_prompt0 + expected_resp0
    assert traj.response_mask == [0] * len(expected_prompt0) + [1] * len(expected_resp0)


def test_multi_turn_no_action_duplicated_in_context():
    """Two non-answering turns then a solving turn — the meat of the module, and
    the regression test for the 2026-06-01 BLOCKER.

    The fix: an action's tokens appear ONLY as that turn's mask-1 ``response_ids``
    and NEVER re-enter as mask-0 context (the old transcript-delta re-tokenised the
    prior recorded action back in as context, duplicating it).  We prove it with
    sentinel response ids that no char-ord encode of any text could produce, then
    assert those ids land only on mask-1 positions and nowhere in the context —
    while the env's observations DO appear in the context.
    """
    tok = CharTokenizer()
    # Distinct sentinel id-blocks per turn; values no short char-ord encode yields.
    sentinels = [[90001, 90002], [90011, 90012, 90013], [90021]]
    actions = ["let me look around", "still thinking", "the answer is 42"]  # last = magic
    calls = {"i": 0}

    def generate(prompt_text):
        i = calls["i"]
        calls["i"] += 1
        return actions[i], list(sentinels[i])

    env = ScriptedEnv()
    env.reset(seed=1, task=None)
    traj = ar.rollout_episode(env, generate=generate, tokenizer=tok)

    _assert_core_invariants(traj)
    assert len(traj.turns) == 3
    assert traj.reward == 1.0
    assert traj.info["n_turns"] == 3
    assert calls["i"] == 3  # generate called exactly once per turn

    # The prompts are still the full append-only transcript (used for generation),
    # so prefix-extension still holds even though the SEQUENCE is built obs-by-obs.
    p0, p1, p2 = (t.prompt_ids for t in traj.turns)
    assert p1[:len(p0)] == p0
    assert p2[:len(p1)] == p1

    # Each turn's sentinel block is exactly that turn's mask-1 run, in order.
    runs = _runs_of_ones(traj.response_mask)
    assert [[traj.sequence_ids[p] for p in run] for run in runs] == sentinels

    # THE fix: no generated/action token appears at a mask-0 (context) position.
    context_ids = [traj.sequence_ids[i] for i, m in enumerate(traj.response_mask) if m == 0]
    all_sentinels = {s for block in sentinels for s in block}
    assert not (all_sentinels & set(context_ids)), "an action token leaked into the context"

    # The env's intermediate observations DID enter the context (obs-delta), so the
    # model still conditions on what it saw. ScriptedEnv emits "OBS1/OBS2: keep going."
    ctx_ord_set = set(context_ids)
    for obs_snippet in ("OBS1: keep going.", "OBS2: keep going."):
        assert set(tok.encode(obs_snippet)) <= ctx_ord_set

    # Sanity: context excludes the actions, so it is strictly shorter than encoding
    # the full final transcript (which contains both actions and observations).
    assert (len(traj.response_mask) - sum(traj.response_mask)) < len(p2)


def test_generated_ids_are_appended_verbatim_not_retokenized():
    """The mask-1 positions must be EXACTLY the ids ``generate`` returned, even
    when those ids bear no relation to re-tokenizing the action text.

    This proves the prompt-delta arithmetic keeps generated ids out-of-band: a
    tokenizer that would re-encode the recorded action differently can never
    corrupt the response span.
    """
    tok = CharTokenizer()
    sentinel = [90001, 90002, 90003]  # ids no char-ord encode would ever produce

    def generate(prompt_text):
        return "the answer is here", list(sentinel)  # text records the magic word

    env = ScriptedEnv()
    env.reset(seed=2, task=None)
    traj = ar.rollout_episode(env, generate=generate, tokenizer=tok)

    _assert_core_invariants(traj)
    assert len(traj.turns) == 1
    runs = _runs_of_ones(traj.response_mask)
    assert len(runs) == 1
    assert [traj.sequence_ids[p] for p in runs[0]] == sentinel
    # And the sentinel ids appear nowhere in the (char-ord) context positions.
    context_ids = [traj.sequence_ids[i] for i, m in enumerate(traj.response_mask) if m == 0]
    assert not (set(sentinel) & set(context_ids))


def test_max_turns_exhaustion_zero_reward():
    """Never-answering generate → episode runs to the cap, reward 0.0, done loop
    stops at max_turns (not forever)."""
    tok = CharTokenizer()

    def generate(prompt_text):
        text = "nope"
        return text, tok.encode(text)

    env = ScriptedEnv()  # max_turns = 4
    env.reset(seed=3, task=None)
    traj = ar.rollout_episode(env, generate=generate, tokenizer=tok)

    _assert_core_invariants(traj)
    assert len(traj.turns) == 4               # exactly the cap
    assert traj.info["n_turns"] == 4
    assert traj.reward == 0.0                 # never solved
    assert sum(traj.response_mask) == 4 * len("nope")


def test_max_turns_override_caps_below_env_default():
    """An explicit ``max_turns`` arg overrides the env's larger ``max_turns``."""
    tok = CharTokenizer()

    def generate(prompt_text):
        return "nope", tok.encode("nope")

    env = ScriptedEnv()  # env.max_turns == 4
    env.reset(seed=4, task=None)
    traj = ar.rollout_episode(env, generate=generate, tokenizer=tok, max_turns=2)

    _assert_core_invariants(traj)
    assert len(traj.turns) == 2
    assert traj.info["n_turns"] == 2


def test_empty_generate_advances_turn_without_crashing():
    """A ``generate`` returning ("", []) must still count as a turn and not crash;
    its turn contributes prompt-delta context but zero response tokens."""
    tok = CharTokenizer()

    def generate(prompt_text):
        return "", []

    env = ScriptedEnv()  # max_turns = 4; empty action never matches magic word
    env.reset(seed=5, task=None)
    traj = ar.rollout_episode(env, generate=generate, tokenizer=tok)

    _assert_core_invariants(traj)
    assert len(traj.turns) == 4               # empty turns still advance to the cap
    assert sum(traj.response_mask) == 0       # no generated tokens anywhere
    assert all(len(t.response_ids) == 0 for t in traj.turns)
    assert traj.reward == 0.0


def test_call_surface_tokenizer_without_encode():
    """The other HF tokenizer surface — ``tok(text)["input_ids"]`` — works too."""
    tok = CallTokenizer()

    def generate(prompt_text):
        text = "the answer"
        return text, [ord(c) for c in text]

    env = ScriptedEnv()
    env.reset(seed=6, task=None)
    traj = ar.rollout_episode(env, generate=generate, tokenizer=tok)

    _assert_core_invariants(traj)
    assert len(traj.turns) == 1
    assert traj.reward == 1.0


def test_caller_resets_rollout_does_not():
    """Pin the settled contract: rollout_episode does NOT reset the env.

    If the caller forgot to reset, ``env.done`` is its default (False here) and
    the loop still runs — but the point is that rollout never *calls* reset, so a
    seed/task is never re-applied behind the caller's back.  We assert reset was
    invoked exactly once (by us) before the rollout, never again.
    """
    tok = CharTokenizer()
    reset_calls = {"n": 0}

    class CountingEnv(ScriptedEnv):
        def reset(self, *, seed=None, task=None) -> str:
            reset_calls["n"] += 1
            return super().reset(seed=seed, task=task)

    def generate(prompt_text):
        return "the answer", [ord(c) for c in "the answer"]

    env = CountingEnv()
    env.reset(seed=7, task=None)
    assert reset_calls["n"] == 1
    ar.rollout_episode(env, generate=generate, tokenizer=tok)
    assert reset_calls["n"] == 1  # rollout_episode did NOT call reset


def test_already_done_env_yields_empty_trajectory():
    """Defensive: an already-terminal env (done before entry) produces an empty
    trajectory with the env's reward, no turns, and does not crash."""
    tok = CharTokenizer()

    def generate(prompt_text):  # pragma: no cover - must never be called
        raise AssertionError("generate must not run on an already-done env")

    env = ScriptedEnv()
    env.reset(seed=8, task=None)
    env.step("the answer")  # terminal: env.done is now True, reward 1.0
    assert env.done is True

    traj = ar.rollout_episode(env, generate=generate, tokenizer=tok)
    assert traj.turns == []
    assert traj.sequence_ids == []
    assert traj.response_mask == []
    assert traj.reward == 1.0       # carries the env's terminal reward through
    assert traj.info["n_turns"] == 0


def test_malformed_generate_return_is_fail_soft():
    """A ``generate`` returning a non-(text, ids) shape degrades to an empty
    response (wastes a turn) rather than crashing — spec §0.3 fail-soft."""
    tok = CharTokenizer()

    def generate(prompt_text):
        return None  # not a (text, ids) tuple

    env = ScriptedEnv()
    env.reset(seed=9, task=None)
    traj = ar.rollout_episode(env, generate=generate, tokenizer=tok)

    _assert_core_invariants(traj)
    assert len(traj.turns) == 4         # ran to cap with empty responses
    assert sum(traj.response_mask) == 0
    assert traj.reward == 0.0


def test_trajectory_and_turn_dataclass_shape():
    """The dataclasses expose exactly the fields the spec mandates."""
    assert set(ar.Turn.__dataclass_fields__) == {"prompt_text", "prompt_ids", "response_ids"}
    assert set(ar.Trajectory.__dataclass_fields__) == {
        "turns", "sequence_ids", "response_mask", "reward", "info",
    }
