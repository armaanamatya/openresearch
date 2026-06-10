"""Copyable base classes for SDAR-style teacher/student environments.

The 2026-05-31 failure (`prj_09047604e591d969`).  Every `alfworld` cell of the
SDAR matrix died with::

    AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'

The agent's trainer (`code/sdar/train.py`) called ``env.build_student_prompt(...)``
on an ``ALFWorldEnv`` that simply never defined it.  An ``AttributeError`` raised
*mid-grid*, after model load + a rollout had already burned GPU minutes, is the
worst possible place to discover a one-line interface gap: 18 cells, zero metrics,
rubric 0.0.

This module turns that runtime ``AttributeError`` into a **construction-time
``TypeError``**.  Any environment that subclasses :class:`BaseEnv` but forgets
``build_student_prompt`` / ``build_teacher_prompt`` cannot even be instantiated —
Python's ``ABCMeta`` refuses ``ALFWorldEnv()`` with::

    TypeError: Can't instantiate abstract class ALFWorldEnv with abstract
    methods build_student_prompt, build_teacher_prompt

…which surfaces at env-construction (cell start), before the model loads, and
names the exact missing method.  Paired with the AST pre-flight backstop
(`preflight_ast._check_env_interface_contract`) that catches a non-subclassing
``*Env`` *before* the grid runs at all.

Two interfaces live here:

* :class:`BaseEnv` — the original **single-turn** contract (one prompt → one
  answer → reward).  Closed-book Search-QA used this.
* :class:`AgenticEnv` — the **multi-turn agentic** contract (2026-06-01).
  ALFWorld, WebShop and retrieval-augmented Search-QA are agentic: the policy
  emits an action, the environment returns an observation, and this repeats for
  several turns before a terminal reward.  ``BaseEnv`` cannot express that loop —
  which is *why* the 2026-05-31 run had to fake ALFWorld as closed-book QA and
  Search-QA as parametric recall (rubric floor ~0.05).  ``AgenticEnv`` adds
  ``reset`` / ``step`` / ``episode_reward`` and ships working transcript-rendering
  defaults for the two prompt builders, so a concrete env implements only the two
  abstract methods and still satisfies the construction-time guarantee.

Copyable helper — mirror of the ``gpu_cell_runner.py`` / ``rubric_guard.py``
pattern.  ``run_with_sdk`` copies this file into ``code/sdar_env_base.py`` and the
``implement_baseline`` prompt instructs the agent to::

    from sdar_env_base import AgenticEnv, StepResult

    class ALFWorldEnv(AgenticEnv):
        max_turns = 30
        def reset(self, *, seed=None, task=None) -> str: ...
        def step(self, action: str) -> StepResult: ...

Zero non-stdlib dependencies, so the copy-and-paste route always works inside an
agent sandbox.  The heavy per-env libraries (rank_bm25 / sentence-transformers /
faiss / alfworld / requests) live in the concrete subclass files, never here.
Auth-agnostic by construction (no provider branching, no LLM calls).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

__all__ = ["BaseEnv", "AgenticEnv", "StepResult"]


class BaseEnv(ABC):
    """Interface every SDAR teacher/student environment must satisfy.

    The signatures are deliberately permissive (``*args, **kwargs``): each
    environment (ALFWorld, WebShop, Search-QA, …) builds its prompts from
    whatever observation/state shape fits its own data.  The *only* contract
    this base enforces is that the two methods the SDAR trainer calls **exist**,
    so a missing one fails loudly at construction rather than mid-rollout.

    Subclasses are free to add ``reset`` / ``step`` / any env-specific surface;
    those are not abstracted here because they are not the cross-cutting
    invariant that broke the 2026-05-31 run.  Multi-turn agentic environments
    should subclass :class:`AgenticEnv` (below) instead, which formalises the
    rollout loop and provides the two prompt builders for free.
    """

    @abstractmethod
    def build_student_prompt(self, *args: Any, **kwargs: Any) -> str:
        """Return the prompt shown to the *student* policy for one turn/episode.

        Called by the SDAR trainer on every rollout.  Must return a ``str``.
        """
        raise NotImplementedError

    @abstractmethod
    def build_teacher_prompt(self, *args: Any, **kwargs: Any) -> str:
        """Return the prompt shown to the *teacher* policy for self-distillation.

        Called by the SDAR trainer when computing the teacher/student gap.  Must
        return a ``str``.
        """
        raise NotImplementedError


@dataclass
class StepResult:
    """Outcome of one :meth:`AgenticEnv.step` — one model action against the env.

    ``observation`` is the text the environment returns after the action (a
    search result, an ALFWorld room description, a WebShop page); the trainer
    feeds it into the next student prompt.  ``reward`` is the *incremental*
    reward for this step — most agentic envs are terminal-reward-only, so it
    stays ``0.0`` until the episode ends, at which point ``done`` is ``True`` and
    ``reward`` (== :meth:`AgenticEnv.episode_reward`) carries the episode score.

    **Dense intermediate reward (BES Phase 4A, opt-in).** An env MAY emit a
    positive ``reward`` on a *non-terminal* step as sub-goal-progress shaping (see
    ``ALFWorldEnv`` behind ``REPROLAB_ALFWORLD_SHAPING``). When it does, the
    contract is: the **terminal** step's ``reward`` and ``info["won"]`` remain the
    SEPARATE authoritative success signal — shaping is intermediate credit only
    and must never replace or contaminate the terminal ``float(won)``. A held-out
    evaluation therefore still measures real terminal success, not shaped reward.
    By default (flag off) the intermediate ``reward`` is exactly ``0.0`` and this
    field behaves identically to the terminal-only contract above.

    ``info`` carries env-specific diagnostics (``{"success": True, "f1": 0.83,
    "n_search": 2}``) that the trainer folds into the cell's ``metrics.json``; a
    shaped non-terminal step additionally carries ``{"shaped": <credit>}`` so
    shaped credit is never mistaken for terminal success.
    """

    observation: str
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class AgenticEnv(BaseEnv):
    """Multi-turn agentic environment for SDAR rollouts.

    The SDAR trainer drives exactly one episode like this::

        env = SearchQAEnv(...)
        env.reset(seed=42, task=example)            # load one task, clear transcript
        for _ in range(env.max_turns):
            prompt = env.build_student_prompt()      # render the running transcript
            action = student.generate(prompt)        # ONE model turn
            res = env.step(action)                   # advance env, append to transcript
            if res.done:
                break
        reward = env.episode_reward()                # terminal scalar, ideally in [0, 1]

    A group of ``G`` such episodes (same task, different sampling seeds) feeds the
    GRPO advantage; the OPSD gate is computed token-wise over the *student*
    response spans of the rolled-out sequence.  The companion
    ``agentic_rollout.rollout_episode`` consumes this interface and returns the
    flat ``(token_ids, response_mask, reward)`` trajectory — so the error-prone
    multi-turn → sequence/mask conversion lives in one tested place, not in every
    agent-generated trainer.

    **What a subclass implements.** Only :meth:`reset` and :meth:`step` (kept
    ``@abstractmethod`` so a missing one still fails at *construction*, preserving
    the BaseEnv guarantee).  :meth:`build_student_prompt` /
    :meth:`build_teacher_prompt` have working defaults that render the transcript
    accumulated by the ``_record_*`` helpers, so a subclass gets a correct prompt
    for free and overrides only to add a system header or a teacher oracle.

    **Robustness contract.** :meth:`step` must never raise on a malformed action —
    parse defensively and return an observation that nudges the policy.  A bad
    action wastes a turn; it does not crash the cell.  The whole point of the
    construction-time ABC + the AST pre-flight is that *interface* errors surface
    before the grid; *behavioural* robustness (defensive parsing, turn caps) is
    the subclass's job and is what keeps a live grid from dying mid-rollout.
    """

    #: Hard cap on model turns per episode (subclasses override; ``1`` == single-turn).
    max_turns: int = 1

    def __init__(self) -> None:
        # (role, text) pairs; role in {"system", "obs", "act"}.
        self._transcript: list[tuple[str, str]] = []
        self._episode_reward: float = 0.0
        self._done: bool = False
        self._turns_taken: int = 0
        self._last_info: dict[str, Any] = {}

    # --- the contract subclasses MUST satisfy --------------------------------

    @abstractmethod
    def reset(self, *, seed: int | None = None, task: Any = None) -> str:
        """Start a new episode; return the initial observation text.

        Implementations clear/seed env state (call :meth:`_start_episode`),
        record the opening observation via :meth:`_record_obs`, and return it.
        Must be deterministic given ``seed`` + ``task`` (same cell → same
        rollout → reproducible metrics).
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str) -> StepResult:
        """Apply one model ``action``; return the resulting :class:`StepResult`.

        Implementations record the action (:meth:`_record_act`), advance the
        underlying environment, record the new observation (:meth:`_record_obs`),
        and on episode end set the terminal reward via :meth:`_finish`.  Returns
        a :class:`StepResult` mirroring that state.  **Never raises** on a bad
        action — parse defensively (see the class docstring).
        """
        raise NotImplementedError

    # --- transcript bookkeeping (subclasses call these from reset/step) ------

    def _start_episode(self, system: str = "") -> None:
        """Reset transcript + reward state at the top of :meth:`reset`."""
        self._transcript = []
        if system:
            self._transcript.append(("system", system))
        self._episode_reward = 0.0
        self._done = False
        self._turns_taken = 0
        self._last_info = {}

    def _record_obs(self, text: str) -> None:
        """Append an environment observation to the transcript."""
        self._transcript.append(("obs", "" if text is None else str(text)))

    def _record_act(self, text: str) -> None:
        """Append a model action to the transcript (and count the turn)."""
        self._transcript.append(("act", "" if text is None else str(text)))
        self._turns_taken += 1

    def _finish(self, reward: float, info: dict[str, Any] | None = None) -> None:
        """Mark the episode terminal with a scalar reward (+ optional info)."""
        self._episode_reward = float(reward)
        self._done = True
        if info:
            self._last_info = dict(info)

    # --- prompt rendering (sensible defaults; override to taste) -------------

    def render_transcript(self) -> str:
        """Render system + interleaved observations/actions into one prompt.

        Plain concatenation — observations and actions are recorded already
        formatted by the subclass (e.g. an action recorded as ``"> go to fridge"``
        or ``"search(quantum hall effect)"``).  Override to impose a different
        chat/template structure.
        """
        parts = [text for _role, text in self._transcript if text]
        return ("\n".join(parts)).strip() + "\n"

    def build_student_prompt(self, *args: Any, **kwargs: Any) -> str:
        return self.render_transcript()

    def build_teacher_prompt(self, *args: Any, **kwargs: Any) -> str:
        # Default: identical to the student view.  OPSD's teacher is the same
        # policy (the gap is token-level and detached); a subclass with a
        # stronger teacher or an oracle hint overrides this.
        return self.render_transcript()

    # --- episode accessors ---------------------------------------------------

    def episode_reward(self) -> float:
        """Terminal scalar reward for the episode just rolled out."""
        return float(self._episode_reward)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def turns_taken(self) -> int:
        return self._turns_taken

    @property
    def last_info(self) -> dict[str, Any]:
        return dict(self._last_info)
