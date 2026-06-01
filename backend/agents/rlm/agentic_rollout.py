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

**The mask-alignment rule (the crux).**  Each turn ``env.build_student_prompt()``
returns the *full* running transcript, which only ever grows (``_record_*`` only
appends).  We therefore track the previous turn's full-prompt token ids and take

    delta_ids = current_full_prompt_ids[len(prev_full_prompt_ids):]

— the new transcript *tail* since the previous turn (system header + the prior
action as the env recorded it + the new observation).  The sequence for the turn
is then ``delta_ids`` (mask ``0`` — context) followed by the turn's
``response_ids`` (mask ``1`` — generated).  We compare *prompt-to-prompt* (not
prompt-to-sequence): the generated ``response_ids`` are appended out-of-band and
are deliberately kept out of the prompt-delta arithmetic, so a tokenizer that
re-tokenizes the recorded action differently from what the model emitted can
never desync the delta.  The invariants this guarantees — proven in the test —
are ``len(response_mask) == len(sequence_ids)``,
``sum(response_mask) == total response tokens across turns``, and the mask-``1``
runs equal each turn's ``response_ids`` in order.
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

    Sequence/mask are built with the prompt-delta rule documented at module top:
    each turn contributes ``current_prompt_ids[len(prev_prompt_ids):]`` (mask 0)
    then ``response_ids`` (mask 1).

    Robustness: a ``generate`` returning ``("", [])`` still *counts as a turn*
    (the env is stepped with the empty action, may nudge or waste the turn) and
    never crashes — the turn simply contributes its prompt-delta and zero response
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

    # The previous turn's FULL prompt token ids.  Empty at turn 0, so the first
    # delta is the whole opening prompt (system + initial observation).  Because
    # the transcript only ever grows (``_record_*`` append-only), comparing the
    # current full prompt against the previous full prompt yields a clean,
    # monotonic tail — and keeps the model-generated ``response_ids`` (appended
    # below, out of band) from ever entering the prompt-delta arithmetic.
    prev_prompt_ids: list[int] = []

    for _turn_idx in range(cap):
        # Defensive: if the env is already terminal (caller stepped it, or a prior
        # step ended the episode without us breaking), stop before generating.
        if getattr(env, "done", False):
            break

        prompt_text = env.build_student_prompt()
        if prompt_text is None:
            prompt_text = ""
        prompt_ids = _encode(tokenizer, prompt_text)

        # --- mask-alignment: the new transcript tail since the previous turn ---
        # delta = the suffix of this turn's prompt beyond the previous prompt's
        # length.  These are pure CONTEXT tokens (header on turn 0; the prior
        # recorded action + the new observation on later turns) → mask 0.
        delta_ids = prompt_ids[len(prev_prompt_ids):]
        sequence_ids.extend(delta_ids)
        response_mask.extend(0 for _ in delta_ids)

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

        # Advance the env with the model's TEXT action.  The env records the
        # action + new observation onto the transcript, so the next prompt grows.
        try:
            result = env.step(response_text)
        except Exception:  # pragma: no cover - AgenticEnv.step contracts to never raise
            break

        prev_prompt_ids = prompt_ids

        if getattr(result, "done", False) or getattr(env, "done", False):
            break

    # --- assemble the trajectory --------------------------------------------
    # Reward is the env's terminal scalar (``0.0`` if the episode never finished —
    # e.g. it ran out of turns; matches the §2 max_turns-exhaustion contract when
    # the env's own ``step`` left reward at 0).
    try:
        reward = float(env.episode_reward())
    except Exception:  # pragma: no cover - defensive
        reward = 0.0

    # info = env.last_info merged with rollout stats.  Rollout stats are namespaced
    # plainly (``n_turns``) but env keys take precedence is avoided: we start from
    # the env's info then overlay our stats, so the caller always sees fresh turn
    # counts even if an env happened to stash a stale ``n_turns``.
    env_info = getattr(env, "last_info", {}) or {}
    info: dict[str, Any] = dict(env_info)
    info["n_turns"] = len(turns)
    info["response_lengths"] = response_lengths

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
