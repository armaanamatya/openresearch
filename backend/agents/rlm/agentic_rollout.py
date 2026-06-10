"""Multi-turn episode → flat (token_ids, response_mask) conversion for SDAR RL.

**Why this module exists.**  The single most bug-prone part of agentic RL is
turning a multi-turn rollout — ``[system][obs][act][obs][act]…`` — into the flat
token sequence + per-position *response mask* the GRPO / OPSD loss consumes.  The
mask must be ``1`` on exactly the positions the *student model generated* and
``0`` everywhere else (system prompt, observations, the re-rendered transcript
tail).  An off-by-one — masking one context token, or dropping the first token of
a response — silently *zeros or corrupts the policy-gradient signal*: the run
trains, burns GPU hours, and learns nothing, with no error to point at.  In the
2026-05-31 SDAR collapse the agent-written trainer reimplemented this per env and
got it subtly wrong each time.  Centralising it here — one implementation, fully
unit-tested against the invariants — means every env (ALFWorld, WebShop,
Search-QA) and the agent-generated trainer share one correct conversion.

**What it does NOT depend on.**  The conversion operates purely on two *injected*
callables — ``generate`` (the trainer wraps its HF ``model.generate``) and a
``tokenizer`` (only ``.encode`` / ``tokenizer(text)["input_ids"]`` are used).  So
this module has **zero** torch / transformers / numpy dependency and imports on
the bare base venv; the unit test drives it with a char-ord fake tokenizer and a
scripted fake env.  Copyable helper — mirror of the ``gpu_cell_runner.py`` /
``sdar_env_base.py`` pattern (``run_with_sdk`` copies it into every run's
``code/`` dir), so it stays stdlib-only and the agent sandbox can ``import
agentic_rollout`` directly.

**The rollout contract (settled).**  ``rollout_episode`` does **not** call
``env.reset`` — the *caller* resets (it holds the episode-specific ``seed`` /
``task`` and, for GRPO, rolls a group of ``G`` episodes off the same reset).
``rollout_episode`` owns only the turn loop: per turn it reads the running
transcript via ``env.build_student_prompt()``, feeds that text to ``generate``,
``env.step``s the returned action, and repeats until ``env.done`` or the turn cap.
This matches the §1 protocol (``env.reset(...)`` then loop) and keeps the module's
sole responsibility the error-prone token/mask conversion.

**The mask-alignment rule (the crux).**  The training sequence is assembled
*observation-by-observation*, NOT by diffing the rendered transcript.  Each turn
contributes, in order:

    [context_ids]      (mask 0)   then   [response_ids]   (mask 1)

where ``context_ids`` is the WHOLE opening prompt on turn 0 (system + initial
observation) and ONLY the observation the previous ``env.step`` returned on every
later turn.  The model's action enters the sequence *exactly once* — as that
turn's ``response_ids`` (mask 1), appended verbatim (never re-tokenised).  It is
never folded back in as context.

This is deliberate.  An earlier design diffed the full transcript
(``current_prompt_ids[len(prev_prompt_ids):]``) and so re-tokenised the prior
action — which the env had already recorded, often *cleaned* (e.g. a stripped
``> ``) — as mask-0 context, even though that action was already present as
mask-1 ``response_ids``.  That duplicated every action in the sequence and
silently corrupted the GRPO/OPSD loss context (the 2026-06-01 review BLOCKER).
Taking context only from the env's returned observations removes the action from
the context path entirely.  Generation still conditions on the full
``build_student_prompt()`` render (the env owns its system header + formatting);
only the *training* sequence is built obs-by-obs.  The invariants this guarantees
— proven in the test — are ``len(response_mask) == len(sequence_ids)``,
``sum(response_mask) == total response tokens across turns``, the mask-``1`` runs
equal each turn's ``response_ids`` in order, and no prior action's tokens ever
appear at a mask-``0`` position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, Tuple

if TYPE_CHECKING:  # hints only — never imported at runtime (keeps the module dep-free)
    from sdar_env_base import AgenticEnv  # noqa: F401

__all__ = ["Turn", "Trajectory", "rollout_episode"]


# A tokenizer is anything exposing ``.encode(text) -> list[int]`` OR being callable
# as ``tok(text)["input_ids"] -> list[int]`` (the two HF surfaces).  We duck-type it
# via a Protocol for hints only; ``_encode`` below tolerates either shape at runtime.
class _TokenizerLike(Protocol):
    def encode(self, text: str) -> Any: ...  # pragma: no cover - structural typing only


#: ``generate(prompt_text) -> (response_text, response_token_ids)`` — injected by the
#: trainer (it wraps HF ``model.generate`` + decode).  ``response_token_ids`` are the
#: exact ids the model produced; they are what we mark with ``1`` in the mask.
GenerateFn = Callable[[str], Tuple[str, "list[int]"]]


@dataclass
class Turn:
    """One model turn within an episode.

    ``prompt_text`` is the full transcript shown to the student this turn (the
    return of ``env.build_student_prompt()``).  ``prompt_ids`` is its full token
    encoding (handy for debugging / re-deriving the delta).  ``response_ids`` are
    the model-generated token ids for this turn — exactly the positions that get
    a ``1`` in :attr:`Trajectory.response_mask`.
    """

    prompt_text: str
    prompt_ids: list[int]
    response_ids: list[int]


@dataclass
class Trajectory:
    """The flat result of one rolled-out episode, ready for the RL loss.

    ``sequence_ids`` is the interleaved ``[delta-prompt][response][delta-prompt]
    [response]…`` token stream; ``response_mask`` is the same length, ``1`` on the
    student-generated positions and ``0`` elsewhere.  ``reward`` is the env's
    terminal scalar (``env.episode_reward()``); ``info`` is ``env.last_info``
    merged with rollout stats (``n_turns`` plus the per-turn response lengths).
    """

    turns: list[Turn]
    sequence_ids: list[int]
    response_mask: list[int]
    reward: float
    info: dict[str, Any] = field(default_factory=dict)


def _encode(tokenizer: _TokenizerLike, text: str) -> list[int]:
    """Encode ``text`` to a list of int ids via whichever HF surface exists.

    Tolerates the two common shapes — ``tokenizer.encode(text)`` and
    ``tokenizer(text)["input_ids"]`` — and an empty string (returns ``[]``).  Any
    array-like return (e.g. a numpy/torch tensor) is coerced to a python ``list``
    so the module never imports those libraries.  Fail-soft: a tokenizer that
    raises on a given input degrades to ``[]`` rather than killing the rollout.
    """
    if not text:
        return []
    raw: Any
    try:
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            raw = encode(text)
        else:
            # ``tokenizer(text)`` → BatchEncoding / dict with "input_ids".
            out = tokenizer(text)  # type: ignore[operator]
            raw = out["input_ids"] if isinstance(out, dict) else out
    except Exception:  # pragma: no cover - defensive; real tokenizers don't raise here
        return []
    return _to_int_list(raw)


def _to_int_list(raw: Any) -> list[int]:
    """Coerce a tokenizer return (list / tuple / tensor / nested list) to ``list[int]``."""
    if raw is None:
        return []
    # ``.tolist()`` covers numpy arrays and torch tensors without importing either.
    tolist = getattr(raw, "tolist", None)
    if callable(tolist):
        raw = tolist()
    if isinstance(raw, (list, tuple)):
        seq = list(raw)
        # Some HF tokenizers return a batch [[...]] even for a single string.
        if len(seq) == 1 and isinstance(seq[0], (list, tuple)):
            seq = list(seq[0])
        return [int(x) for x in seq]
    # Scalar id (rare) — wrap it.
    try:
        return [int(raw)]
    except (TypeError, ValueError):  # pragma: no cover - truly unexpected shape
        return []


def rollout_episode(
    env: "AgenticEnv",
    *,
    generate: GenerateFn,
    tokenizer: _TokenizerLike,
    max_turns: int | None = None,
    max_new_tokens: int = 64,
) -> Trajectory:
    """Drive ONE multi-turn episode and return its flat :class:`Trajectory`.

    Contract: the **caller** has already called ``env.reset(seed=..., task=...)``
    (it owns the episode-specific seed/task and, under GRPO, the group of rollouts
    sharing one reset).  This function owns only the turn loop.

    Per turn:

    1. ``prompt = env.build_student_prompt()`` — the full running transcript.
    2. ``response_text, response_ids = generate(prompt)`` — one model turn.  The
       injected ``generate`` is responsible for honouring ``max_new_tokens``; we
       pass it through for callers that read it off the kwargs but do not re-cap
       here (the returned ``response_ids`` are taken verbatim).
    3. ``env.step(response_text)`` — advances the env, appends the action + the
       new observation to the transcript (so the *next* prompt grows).
    4. Stop at ``env.done`` or when ``max_turns`` turns have been taken.

    Sequence/mask are built with the observation-delta rule documented at module
    top: turn 0 contributes the whole opening prompt (mask 0) then its
    ``response_ids`` (mask 1); each later turn contributes ONLY the prior step's
    observation (mask 0) then its ``response_ids`` (mask 1).  An action is in the
    sequence exactly once — as response tokens — never re-tokenised as context.

    Robustness: a ``generate`` returning ``("", [])`` still *counts as a turn*
    (the env is stepped with the empty action, may nudge or waste the turn) and
    never crashes — the turn contributes its context delta and zero response
    tokens.  ``max_new_tokens`` is accepted for API symmetry with the trainer's
    ``generate`` wrapper.
    """
    # Resolve the turn cap: explicit arg > env's declared cap > 1 (single-turn).
    cap = max_turns if max_turns is not None else getattr(env, "max_turns", 1)
    try:
        cap = int(cap)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        cap = 1
    if cap < 0:
        cap = 0

    turns: list[Turn] = []
    sequence_ids: list[int] = []
    response_mask: list[int] = []
    response_lengths: list[int] = []

    # Context is appended to the TRAINING sequence exactly once per source: the
    # opening transcript (system + initial observation) at turn 0, then ONLY the
    # observation the env returns from each step thereafter.  The model's action
    # NEVER re-enters as context — it lives in the sequence solely as that turn's
    # ``response_ids`` (mask 1).  Generation still conditions on the FULL running
    # transcript (``build_student_prompt`` — the env owns its system header +
    # formatting), but the sequence is assembled obs-by-obs so a prior action the
    # env recorded (possibly cleaned) is never re-tokenised back in as mask-0
    # context.  That re-tokenisation was the 2026-06-01 review BLOCKER: it
    # duplicated each action (already present as mask-1 ``response_ids``) and
    # silently corrupted the GRPO/OPSD loss context.
    #
    # ``next_context_text is None`` is the turn-0 sentinel → use the whole opening
    # prompt; on later turns it is the prior step's observation (possibly "").
    next_context_text: str | None = None
    # Phase 4A (Codex blocker): accumulate per-step INTERMEDIATE shaped rewards
    # (0.0 unless REPROLAB_ALFWORLD_SHAPING is on) so reward shaping reaches the GRPO
    # training return assembled below. Terminal reward/success stay separate (info).
    shaped_sum: float = 0.0

    for _turn_idx in range(cap):
        # Defensive: if the env is already terminal (caller stepped it, or a prior
        # step ended the episode without us breaking), stop before generating.
        if getattr(env, "done", False):
            break

        # Generation conditions on the full running transcript (best fidelity).
        prompt_text = env.build_student_prompt()
        if prompt_text is None:
            prompt_text = ""
        prompt_ids = _encode(tokenizer, prompt_text)

        # --- context for the TRAINING sequence (mask 0) ---
        # turn 0: the whole opening prompt; later turns: ONLY the new observation
        # the previous step returned (never the action — see the note above).
        if next_context_text is None:
            context_ids = prompt_ids
        else:
            context_ids = _encode(tokenizer, next_context_text)
        sequence_ids.extend(context_ids)
        response_mask.extend(0 for _ in context_ids)

        # One model turn.  ``generate`` is injected; tolerate a bad return shape.
        try:
            gen = generate(prompt_text)
        except Exception:  # pragma: no cover - defensive; injected fn shouldn't raise
            gen = ("", [])
        response_text, response_ids = _coerce_generate_result(gen)

        # The model-generated positions — the ONLY ones marked 1.  Appended
        # verbatim (not re-tokenized) so the mask lines up with what the policy
        # actually produced, which is what the loss differentiates.
        sequence_ids.extend(response_ids)
        response_mask.extend(1 for _ in response_ids)

        turns.append(
            Turn(prompt_text=prompt_text, prompt_ids=prompt_ids, response_ids=response_ids)
        )
        response_lengths.append(len(response_ids))

        # Advance the env with the model's TEXT action.  The env records the action
        # + new observation onto its transcript; we take the RETURNED observation
        # as the next turn's context delta (never the action).
        try:
            result = env.step(response_text)
        except Exception:  # pragma: no cover - AgenticEnv.step contracts to never raise
            break

        obs = getattr(result, "observation", "")
        next_context_text = obs if isinstance(obs, str) else ("" if obs is None else str(obs))

        _done = getattr(result, "done", False) or getattr(env, "done", False)
        if not _done:
            # Non-terminal shaped credit only — the terminal scalar is added once via
            # episode_reward() below, so this never double-counts the terminal reward.
            shaped_sum += float(getattr(result, "reward", 0.0) or 0.0)
        if _done:
            break

    # --- assemble the trajectory --------------------------------------------
    # Reward is the env's terminal scalar (``0.0`` if the episode never finished —
    # e.g. it ran out of turns; matches the §2 max_turns-exhaustion contract when
    # the env's own ``step`` left reward at 0).
    try:
        terminal_reward = float(env.episode_reward())
    except Exception:  # pragma: no cover - defensive
        terminal_reward = 0.0
    # Phase 4A (Codex blocker): the TRAINING return folds in the per-step shaped
    # sub-goal rewards so shaping actually reaches GRPO. With shaping OFF every
    # StepResult.reward is 0.0 ⇒ shaped_sum == 0.0 ⇒ reward == terminal_reward
    # (byte-identical parity). The terminal scalar + success stay SEPARATE in info
    # for the held-out terminal-success eval — shaped reward is never read as success.
    reward = terminal_reward + shaped_sum

    # info = env.last_info merged with rollout stats.  Rollout stats are namespaced
    # plainly (``n_turns``) but env keys take precedence is avoided: we start from
    # the env's info then overlay our stats, so the caller always sees fresh turn
    # counts even if an env happened to stash a stale ``n_turns``.
    env_info = getattr(env, "last_info", {}) or {}
    info: dict[str, Any] = dict(env_info)
    info["n_turns"] = len(turns)
    info["response_lengths"] = response_lengths
    info["terminal_reward"] = terminal_reward
    info["shaped_reward"] = shaped_sum

    return Trajectory(
        turns=turns,
        sequence_ids=sequence_ids,
        response_mask=response_mask,
        reward=reward,
        info=info,
    )


def _coerce_generate_result(gen: Any) -> Tuple[str, list[int]]:
    """Normalise a ``generate`` return into ``(response_text, response_ids)``.

    Accepts the canonical ``(text, ids)`` tuple; degrades any other shape to
    ``("", [])`` so a misbehaving injected ``generate`` wastes a turn instead of
    crashing the rollout (the fail-soft rule from spec §0.3).
    """
    if isinstance(gen, (tuple, list)) and len(gen) >= 2:
        text, ids = gen[0], gen[1]
        text = "" if text is None else str(text)
        return text, _to_int_list(ids)
    # A lone string (no ids) — keep the text, no generated-token positions.
    if isinstance(gen, str):
        return gen, []
    return "", []
