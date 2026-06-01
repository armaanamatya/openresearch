"""Real ALFWorld TextWorld agentic environment for SDAR rollouts.

The 2026-05-31 run (`prj_09047604e591d969`) **de-scoped ALFWorld entirely**:
the long-horizon embodied-planning environment that the SDAR paper trains on was
faked as closed-book QA, so the rubric leaves that inspect for real ALFWorld
episodes (admissible-action navigation, multi-step pick/place/clean/heat tasks)
could never pass.  This module restores ALFWorld as what the paper actually
uses — a **real** TextWorld interactive-fiction environment driven through the
installed ``alfworld`` package (``alfworld.agents.environment.get_environment``).

Why TextWorld, not THOR.  ALFWorld ships two backends: ``AlfredThorEnv`` (the
AI2-THOR 3-D simulator, needs a GPU display / X server / Unity) and
``AlfredTWEnv`` (a pure-text TextWorld game compiled from the same ALFRED task).
SDAR's agent is a *language* policy that emits text commands (``go to fridge 1``,
``take apple 1 from countertop 2``) and reads text observations — so the
TextWorld variant is the right and the *headless* one.  THOR would force a
display dependency into every GPU cell for no benefit.

The ALFWorld API this binds to (discovered from the installed 0.4.2 source,
``alfworld/agents/environment/alfred_tw_env.py`` + the TextWorld batched gym
env it builds)::

    from alfworld.agents.environment import get_environment
    AlfredTWEnv = get_environment("AlfredTWEnv")            # -> the class
    tw = AlfredTWEnv(config, train_eval="eval_out_of_distribution")
    tw.seed(seed)                                          # shuffles game pool
    env = tw.init_env(batch_size=1)                        # TextworldBatchGymEnv
    obs, infos = env.reset()                               # obs:[str], infos:{k:[v]}
    obs, scores, dones, infos = env.step(["go to fridge 1"])  # all BATCHED lists

``init_env`` registers the games with ``request_infos = EnvInfos(won=True,
admissible_commands=True, ...)``, so ``infos["won"][0]`` is the success signal
and ``infos["admissible_commands"][0]`` is the legal-action list for the turn.
Everything is **batched** (batch_size=1 here): observations/scores/dones are
length-1 lists, infos is a dict of length-1 lists — this env unwraps index 0.

Copyable helper — mirror of the ``gpu_cell_runner.py`` / ``sdar_env_base.py``
pattern.  ``run_with_sdk`` copies this file into the run's ``code/`` dir and the
``implement_baseline`` prompt instructs the agent to::

    from alfworld_env import ALFWorldEnv          # FLAT import inside code/
    env = ALFWorldEnv()                            # real get_environment path
    obs = env.reset(seed=0, task=None)

``alfworld`` is **lazy-imported inside** :meth:`reset` / :meth:`available` so this
module imports cleanly on a host where the package is present but the game
**data is not downloaded** (``ALFWORLD_DATA`` unset) — the common CI state.  When
the data is genuinely missing the env degrades to a clean zero-reward terminal
step (see :meth:`available`); it never raises and kills the grid.

The real ``get_environment`` path is the default; a ``tw_env_factory`` /
``_env`` seam (mirroring ``env_cache``'s injected ``downloader`` / ``probe``)
lets tests drive a fake TextWorld env with no alfworld data and no GPU.

Determinism: ``reset(seed=s)`` calls ``tw.seed(s)`` (which seeds the game-pool
shuffle) before ``init_env``/``reset``, so the same ``seed`` always loads the
same game in the same order — identical cells produce identical rollouts.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from sdar_env_base import AgenticEnv, StepResult

__all__ = ["ALFWorldEnv"]


#: Standard ALFWorld on-disk layout under ``$ALFWORLD_DATA`` (the
#: ``alfworld-download`` console script writes ``json_2.1.1/{train,valid_*}``).
#: ``$ALFWORLD_DATA`` is expanded by ``AlfredTWEnv`` itself via
#: ``os.path.expandvars`` at ``collect_game_files`` time.
_DATA_SUBPATH_TRAIN = "$ALFWORLD_DATA/json_2.1.1/train"
_DATA_SUBPATH_EVAL_ID = "$ALFWORLD_DATA/json_2.1.1/valid_seen"
_DATA_SUBPATH_EVAL_OOD = "$ALFWORLD_DATA/json_2.1.1/valid_unseen"

#: The six ALFRED task types (all of them — the paper trains across the full set).
_ALL_TASK_TYPE_IDS = [1, 2, 3, 4, 5, 6]

#: Bounds on the admissible-command block appended to each observation. The real
#: TextWorld env can report dozens of legal commands per turn; without a cap the
#: block would inflate the student prompt and blow the token budget on long
#: rollouts. Cap the count first, then hard-truncate the joined string.
_MAX_ADMISSIBLE_COMMANDS = 50
_MAX_ADMISSIBLE_CHARS = 1500

#: Guidance prepended to the system prompt so the policy emits clean TextWorld
#: commands instead of chat scaffolding (``> ...`` / ``action: ...``), and knows
#: it may inspect the admissible-action list the env reports each turn.
_ACTION_GUIDANCE = (
    "You are an embodied agent in a text-based household environment (ALFWorld). "
    "Each turn, respond with exactly ONE natural-language command and nothing "
    "else — no leading '>' and no 'action:' prefix. Valid commands look like: "
    "'go to fridge 1', 'open fridge 1', 'take apple 1 from countertop 2', "
    "'put apple 1 in/on countertop 1', 'heat apple 1 with microwave 1', "
    "'cool apple 1 with fridge 1', 'clean apple 1 with sinkbasin 1', "
    "'examine drawer 1', 'inventory', 'look'. The environment will report the "
    "admissible commands; choose from them to make progress toward the goal."
)


def _minimal_alfworld_config(data_dir: str) -> dict[str, Any]:
    """Build the smallest config ``AlfredTWEnv`` accepts for headless TW eval.

    ``AlfredTWEnv.__init__`` → ``collect_game_files`` reads
    ``dataset.{data_path,eval_id_data_path,eval_ood_data_path,num_*_games}``,
    ``env.{task_types,goal_desc_human_anns_prob}``; ``init_env`` additionally
    reads ``env.{domain_randomization,expert_type}`` and
    ``general.training_method`` (we use ``"dqn"`` so only
    ``rl.training.max_nb_steps_per_episode`` is needed — the ``dagger`` branch
    would pull in the expert planner).  ``data_dir`` is interpolated as the
    ``$ALFWORLD_DATA`` root; the package expands ``$ALFWORLD_DATA`` itself, so we
    also export it (best-effort) for any sub-path that reads the raw env var.
    """
    return {
        "dataset": {
            "data_path": _DATA_SUBPATH_TRAIN,
            "eval_id_data_path": _DATA_SUBPATH_EVAL_ID,
            "eval_ood_data_path": _DATA_SUBPATH_EVAL_OOD,
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "env": {
            "type": "AlfredTWEnv",
            "task_types": list(_ALL_TASK_TYPE_IDS),
            "goal_desc_human_anns_prob": 0.0,
            "domain_randomization": False,
            "expert_type": "handcoded",
        },
        "general": {
            "training_method": "dqn",
        },
        "rl": {
            "training": {
                # Per-game TextWorld step cap; we bound turns ourselves via
                # ``max_turns`` but the registration needs a positive value.
                "max_nb_steps_per_episode": 50,
            },
        },
        # ``data_dir`` is the resolved ALFWORLD_DATA root, kept for callers that
        # want to know where games were sought (not consumed by AlfredTWEnv).
        "_resolved_alfworld_data": data_dir,
    }


def _extract_goal(observation: str) -> str:
    """Pull the 'Your task is to ...' goal line out of a TextWorld observation.

    ALFWorld's initial observation is the room description followed by a
    ``Your task is to: <goal>`` line.  We surface it verbatim in the system
    prompt so the policy sees the goal up front; if absent, return "".
    """
    if not observation:
        return ""
    for line in str(observation).splitlines():
        low = line.strip().lower()
        if low.startswith("your task is to") or low.startswith("task:"):
            return line.strip()
    return ""


def _clean_action(action: str) -> str:
    """Strip model scaffolding from a raw generated action.

    The policy is told to emit a bare command, but models routinely prepend
    ``> `` (interactive-fiction prompt echo) or ``action:`` / ``Action:``.  We
    strip a single leading marker of each kind and collapse to the first
    non-empty line (a chat model may add a trailing explanation).  Defensive:
    never raises; returns ``""`` for ``None``/blank.
    """
    if not action:
        return ""
    text = str(action).strip()
    # Take the first non-empty line — discard any trailing reasoning.
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Strip a leading interactive-fiction prompt: "> go to fridge 1".
        while line.startswith(">"):
            line = line[1:].lstrip()
        # Strip a leading "action:" / "Action:" / "command:" label.
        low = line.lower()
        for prefix in ("action:", "command:", "act:"):
            if low.startswith(prefix):
                line = line[len(prefix):].lstrip()
                low = line.lower()
        # Strip wrapping backticks/quotes a model may add.
        line = line.strip("`").strip().strip('"').strip("'").strip()
        if line:
            return line
    return ""


def _first(value: Any, default: Any = None) -> Any:
    """Unwrap a batched (length-1 list) value, tolerating a scalar.

    The TextWorld batched gym env returns lists (batch_size=1 → length 1);
    a fake env in a test may return scalars.  This accepts either shape.
    """
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


def _format_admissible(commands: Any) -> str:
    """Render a bounded "Admissible commands: …" line from a command list.

    The system prompt promises the policy that the env reports the legal
    commands each turn; without this the policy never actually sees them (the
    2026-05-31 HIGH finding).  ``commands`` is the *already-unwrapped* value for
    this turn (a list of command strings) — callers pass
    ``infos["admissible_commands"]`` after :meth:`_unwrap_infos`.  Returns "" for
    a missing/empty/non-list value (no block, no crash), else a single line
    capped at :data:`_MAX_ADMISSIBLE_COMMANDS` commands and
    :data:`_MAX_ADMISSIBLE_CHARS` characters so it can't blow the token budget.
    """
    if not isinstance(commands, (list, tuple)) or not commands:
        return ""
    items: list[str] = []
    for cmd in commands[:_MAX_ADMISSIBLE_COMMANDS]:
        if cmd is None:
            continue
        text = str(cmd).strip()
        if text:
            items.append(text)
    if not items:
        return ""
    truncated = len(commands) > len(items)
    joined = ", ".join(items)
    if len(joined) > _MAX_ADMISSIBLE_CHARS:
        joined = joined[:_MAX_ADMISSIBLE_CHARS].rstrip(", ")
        truncated = True
    suffix = ", …" if truncated else ""
    return f"Admissible commands: {joined}{suffix}"


class ALFWorldEnv(AgenticEnv):
    """Real long-horizon ALFWorld TextWorld environment (SDAR full-scope).

    One episode = one ALFRED household task played as interactive fiction: the
    policy navigates rooms and manipulates objects via natural-language commands
    over up to :attr:`max_turns` turns, earning reward 1.0 iff it satisfies the
    goal condition (``infos["won"]``) and 0.0 otherwise — the sparse,
    long-horizon reward the SDAR self-distillation is designed to handle.

    **Backend.** Default is the real ``alfworld`` ``AlfredTWEnv`` (TextWorld,
    headless).  A ``tw_env_factory`` callable or a pre-built ``_env`` object can
    be injected for tests / alternative provisioning — the factory takes
    ``(seed)`` and returns an object exposing the TextWorld batched-gym API
    (``reset() -> (obs_list, infos)``, ``step(cmds_list) -> (obs, scores, dones,
    infos)``).

    **Fail-soft.** If ALFWorld data is missing (``ALFWORLD_DATA`` unset / no
    games) or the package can't be constructed, :meth:`available` returns
    ``(False, reason)`` and the env, if used anyway, yields a short observation
    and a single zero-reward terminal step marked ``info["unavailable"]`` — it
    never raises and never kills the GPU cell.
    """

    max_turns: int = 30

    def __init__(
        self,
        *,
        train_eval: str = "eval_out_of_distribution",
        tw_env_factory: Callable[[int], Any] | None = None,
        _env: Any | None = None,
        data_dir: str | None = None,
    ) -> None:
        """Construct the env (no game is loaded until :meth:`reset`).

        Args:
            train_eval: which ALFWorld split to draw games from. Default is the
                out-of-distribution validation split (the paper's eval setting);
                ``"train"`` / ``"eval_in_distribution"`` also accepted.
            tw_env_factory: optional seam — ``factory(seed) -> batched-gym env``.
                When given, it fully replaces the real ``get_environment`` path
                (used by tests to inject a fake without alfworld data).
            _env: optional pre-built batched-gym env to use directly (an even
                lower-level seam than ``tw_env_factory``; ``reset`` will call
                ``.reset()`` on it).
            data_dir: explicit ALFWORLD_DATA root; defaults to the env var.
        """
        super().__init__()
        self._train_eval = train_eval or "eval_out_of_distribution"
        self._tw_env_factory = tw_env_factory
        self._injected_env = _env
        # Snapshot the *user-provided* data dir at construction, BEFORE any
        # alfworld import can default ALFWORLD_DATA to ~/.cache/alfworld.
        self._data_dir = data_dir if data_dir is not None else os.environ.get("ALFWORLD_DATA")

        # Live episode state.
        self._env: Any | None = None
        self._infos: dict[str, Any] = {}
        self._unavailable_reason: str | None = None
        self._won: bool = False

    # --- availability ---------------------------------------------------------

    @classmethod
    def available(cls, *, data_dir: str | None = None) -> tuple[bool, str]:
        """Return ``(usable, reason)`` for the real ALFWorld backend.

        Checks, in order and all fail-soft (never raises):

        1. ``ALFWORLD_DATA`` is set (or ``data_dir`` passed) and the directory
           exists — the game data must have been downloaded.
        2. Some ``traj_data.json`` games actually exist under the data root
           (a populated install, not an empty cache dir).
        3. ``alfworld.agents.environment.get_environment`` imports and yields
           the ``AlfredTWEnv`` class.

        On any miss returns ``(False, <reason mentioning data/package>)`` so the
        caller (trainer / ``env_cache``) can record a verified Exclusion instead
        of crashing.  Reading the env var FIRST matters: importing alfworld has
        the side effect of defaulting ``ALFWORLD_DATA`` to ``~/.cache/alfworld``,
        so we must inspect the *pre-import* value to honor a genuinely-unset var.
        """
        # 1. Data root presence — read the env var BEFORE importing alfworld.
        root = data_dir if data_dir is not None else os.environ.get("ALFWORLD_DATA")
        if not root:
            return (
                False,
                "ALFWORLD_DATA is unset; ALFWorld game data is not downloaded "
                "(run alfworld-download into the shared cache).",
            )
        if not os.path.isdir(root):
            return (
                False,
                f"ALFWORLD_DATA={root!r} does not exist; ALFWorld game data is "
                "not downloaded.",
            )

        # 2. Are there any games on disk? (cheap bounded walk).
        if not cls._has_any_games(root):
            return (
                False,
                f"no ALFWorld games (traj_data.json) found under ALFWORLD_DATA="
                f"{root!r}; data appears not downloaded.",
            )

        # 3. Can the package be imported?
        #    Hard-assign ALFWORLD_DATA to the resolved root BEFORE importing —
        #    importing alfworld defaults ALFWORLD_DATA to ~/.cache/alfworld via a
        #    setdefault, which would then ignore an explicit ``data_dir``. Assign
        #    (not setdefault) so the explicit path always wins. ``root`` is the
        #    verified existing directory from step 1.
        try:
            os.environ["ALFWORLD_DATA"] = str(root)
            from alfworld.agents.environment import get_environment  # noqa: WPS433

            get_environment("AlfredTWEnv")
        except Exception as exc:  # pragma: no cover - exercised only with pkg issues
            return (False, f"alfworld import/construction failed: {exc!r}")

        return (True, "ok")

    @staticmethod
    def _has_any_games(root: str, *, max_scan: int = 200000) -> bool:
        """Return True iff at least one ``traj_data.json`` exists under ``root``.

        Bounded walk (``max_scan`` dir entries) so a pathological tree can't hang
        the check.  Fail-soft: any OS error → treat as 'no games'.
        """
        scanned = 0
        try:
            for _dirpath, _dirnames, filenames in os.walk(root):
                if "traj_data.json" in filenames:
                    return True
                scanned += len(filenames) + 1
                if scanned >= max_scan:
                    break
        except OSError:
            return False
        return False

    # --- the AgenticEnv contract ---------------------------------------------

    def reset(self, *, seed: int | None = None, task: Any = None) -> str:
        """Load one ALFWorld game and return the initial observation.

        Seeds the game-pool shuffle with ``seed`` (determinism), builds the
        minimal TextWorld config, constructs the batched gym env (or the injected
        fake), resets it, and records the opening room+goal observation.

        Fail-soft: if the backend is unavailable / construction fails, starts an
        episode whose first :meth:`step` terminates with a zero reward and an
        ``info["unavailable"]`` flag — never raises.
        """
        self._infos = {}
        self._won = False
        self._unavailable_reason = None
        self._env = None
        # Coerce the seed defensively: a bad seed (str/float/garbage) must never
        # raise out of reset() — the AgenticEnv contract is fail-soft. Bad input
        # degrades to the deterministic default (0), not a crash.
        try:
            seed_val = 0 if seed is None else int(seed)
        except (TypeError, ValueError):
            seed_val = 0

        system = _ACTION_GUIDANCE

        # Build/obtain the underlying TextWorld batched env via the seam ladder:
        # explicit _env  >  injected factory  >  real get_environment.
        try:
            tw_env = self._build_tw_env(seed_val)
        except Exception as exc:  # noqa: BLE001 - fail-soft, record + degrade
            self._unavailable_reason = f"ALFWorld env construction failed: {exc!r}"
            tw_env = None

        if tw_env is None:
            # Unavailable path: record the reason, start a stub episode.
            if self._unavailable_reason is None:
                ok, reason = self.available(data_dir=self._data_dir)
                self._unavailable_reason = reason if not ok else "ALFWorld env unavailable"
            self._start_episode(system=system)
            obs = (
                "[ALFWorld unavailable] "
                + self._unavailable_reason
                + " This episode will end without reward."
            )
            self._record_obs(obs)
            return obs

        self._env = tw_env

        # Reset the batched env → (obs_list, infos). Tolerate a scalar fake.
        try:
            reset_out = tw_env.reset()
        except Exception as exc:  # noqa: BLE001 - fail-soft
            self._unavailable_reason = f"ALFWorld reset failed: {exc!r}"
            self._env = None
            self._start_episode(system=system)
            obs = "[ALFWorld unavailable] " + self._unavailable_reason
            self._record_obs(obs)
            return obs

        obs, infos = self._unpack_reset(reset_out)
        self._infos = infos

        goal = _extract_goal(obs)
        full_system = system if not goal else f"{system}\n\nGoal: {goal}"
        self._start_episode(system=full_system)

        observation = obs if obs else "You are in a room."
        observation = self._record_obs_with_admissible(observation, infos)
        return observation

    def step(self, action: str) -> StepResult:
        """Apply one cleaned natural-language command to the TextWorld env.

        Strips ``> `` / ``action:`` scaffolding from the model output, sends the
        bare command to the env as a length-1 batch, unwraps the batched
        ``(obs, score, done, info)``, and records the observation.  On terminal
        (env ``done`` or last turn) calls :meth:`_finish` with ``float(won)``.

        Never raises: a backend error or a malformed action yields a degraded
        observation; the unavailable backend yields a zero-reward terminal step.
        """
        cleaned = _clean_action(action)
        # Record the cleaned action (this is also what the transcript shows).
        self._record_act(cleaned)

        # Unavailable backend → single zero-reward terminal step.
        if self._env is None:
            reason = self._unavailable_reason or "ALFWorld backend unavailable"
            info = {"unavailable": True, "reason": reason, "success": False, "steps": self.turns_taken}
            self._finish(0.0, info=info)
            obs = "[ALFWorld unavailable] " + reason
            self._record_obs(obs)
            return StepResult(observation=obs, reward=0.0, done=True, info=info)

        # Empty/unparseable action → nudge, waste a turn, do not advance the env.
        if not cleaned:
            obs = (
                "No command parsed. Respond with one command, e.g. 'look', "
                "'go to fridge 1', or 'take apple 1 from countertop 2'."
            )
            self._record_obs(obs)
            if self.turns_taken >= self.max_turns:
                info = {"success": False, "steps": self.turns_taken, "won": False}
                self._finish(0.0, info=info)
                return StepResult(observation=obs, reward=0.0, done=True, info=info)
            return StepResult(observation=obs, reward=0.0, done=False)

        # Drive the real env. Commands MUST be a list (batched API asserts it).
        try:
            step_out = self._env.step([cleaned])
        except Exception as exc:  # noqa: BLE001 - fail-soft on any env error
            obs = f"That command could not be executed ({exc!r}). Try another command."
            self._record_obs(obs)
            if self.turns_taken >= self.max_turns:
                info = {"success": False, "steps": self.turns_taken, "won": False}
                self._finish(0.0, info=info)
                return StepResult(observation=obs, reward=0.0, done=True, info=info)
            return StepResult(observation=obs, reward=0.0, done=False)

        obs, score, done, infos = self._unpack_step(step_out)
        self._infos = infos
        observation = obs if obs else "(no change)"
        observation = self._record_obs_with_admissible(observation, infos)

        won = self._won_from(infos, score)
        # Terminal if the env says done, the goal is met, or we hit the turn cap.
        terminal = bool(done) or won or self.turns_taken >= self.max_turns

        if terminal:
            self._won = won
            reward = float(won)
            info = {
                "success": bool(won),
                "steps": self.turns_taken,
                "won": bool(won),
                "score": float(score) if score is not None else 0.0,
            }
            self._finish(reward, info=info)
            return StepResult(observation=observation, reward=reward, done=True, info=info)

        return StepResult(observation=observation, reward=0.0, done=False)

    # --- observation recording (with admissible-command surfacing) -----------

    def _record_obs_with_admissible(self, observation: str, infos: dict[str, Any]) -> str:
        """Record ``observation`` with a bounded admissible-command block appended.

        The env requests ``EnvInfos(admissible_commands=True)`` and the system
        prompt tells the policy the env reports the legal commands each turn — but
        before the 2026-05-31 fix only the raw room text was recorded, so the
        policy never actually saw them.  This appends a single bounded
        ``Admissible commands: …`` line derived from ``infos["admissible_commands"]``
        (already unwrapped to a list by :meth:`_unwrap_infos`).  When the key is
        missing/empty the observation is recorded unchanged (no crash).  Returns
        the (possibly augmented) text so the caller can mirror it into StepResult.
        """
        block = ""
        if isinstance(infos, dict):
            block = _format_admissible(infos.get("admissible_commands"))
        full = f"{observation}\n{block}" if block else observation
        self._record_obs(full)
        return full

    # --- backend construction (the seam ladder) ------------------------------

    def _build_tw_env(self, seed: int) -> Any | None:
        """Return the batched TextWorld env (injected seam or real alfworld).

        Order: explicit ``_env`` > injected ``tw_env_factory(seed)`` > real
        ``get_environment("AlfredTWEnv")``.  Returns ``None`` when the real
        backend is unavailable (data missing) so :meth:`reset` degrades cleanly.
        """
        if self._injected_env is not None:
            return self._maybe_seed(self._injected_env, seed)

        if self._tw_env_factory is not None:
            # The factory receives the seed (so it can build a seeded env); we
            # also best-effort ``.seed()`` the result so determinism is uniform
            # across both seams and the real path below.
            return self._maybe_seed(self._tw_env_factory(seed), seed)

        # Real path. Gate on availability (reads the pre-import env var).
        ok, reason = self.available(data_dir=self._data_dir)
        if not ok:
            self._unavailable_reason = reason
            return None

        return self._construct_real_tw_env(seed)

    @staticmethod
    def _maybe_seed(env: Any, seed: int) -> Any:
        """Best-effort ``env.seed(seed)`` — fakes/real envs need not implement it."""
        seed_fn = getattr(env, "seed", None)
        if callable(seed_fn):
            try:
                seed_fn(seed)
            except Exception:  # noqa: BLE001 - seeding is best-effort
                pass
        return env

    def _construct_real_tw_env(self, seed: int) -> Any:
        """Construct the genuine ``AlfredTWEnv`` and return its batched gym env.

        Lazy-imports alfworld here so the module imports without the package
        side-effects firing at import time.  ``init_env(batch_size=1)`` returns a
        ``TextworldBatchGymEnv``.

        Two correctness fixes (2026-05-31 Codex review):

        * **ALFWORLD_DATA must be hard-assigned BEFORE the import.** Importing
          ``alfworld`` has the side effect of defaulting ``ALFWORLD_DATA`` to
          ``~/.cache/alfworld``; a post-import ``setdefault`` then silently keeps
          that default and *ignores* the explicit ``data_dir``.  When the caller
          supplied an explicit root we therefore assign ``os.environ`` (not
          setdefault) *before* importing the package so the config's literal
          ``$ALFWORLD_DATA`` resolves to the intended path.
        * **Seed the gym env, not the AlfredTWEnv.** ``AlfredTWEnv`` (alfworld
          0.4.2) has no ``seed()`` method; the seedable object is the
          ``TextworldBatchGymEnv`` returned by ``init_env``.  Seeding ``alfred``
          was a silent no-op, so we seed the gym env after building it.
        """
        data_dir = self._data_dir or os.environ.get("ALFWORLD_DATA")
        if self._data_dir:
            # Explicit root → hard-assign BEFORE importing alfworld so the
            # package's import-time default can't win and the config's literal
            # "$ALFWORLD_DATA" resolves to the intended path.
            os.environ["ALFWORLD_DATA"] = str(self._data_dir)

        from alfworld.agents.environment import get_environment  # noqa: WPS433

        if data_dir and not os.environ.get("ALFWORLD_DATA"):
            # No explicit root, but we resolved one from the env var before the
            # import wiped it: restore it so the config still interpolates.
            os.environ["ALFWORLD_DATA"] = str(data_dir)

        config = _minimal_alfworld_config(data_dir or "")
        env_cls = get_environment("AlfredTWEnv")
        alfred = env_cls(config, train_eval=self._train_eval)

        # Build the batched gym env, THEN seed it for determinism. The seedable
        # object is the TextworldBatchGymEnv (AlfredTWEnv has no seed()).
        env = alfred.init_env(batch_size=1)
        return self._maybe_seed(env, seed)

    # --- batched-output unwrapping -------------------------------------------

    @staticmethod
    def _unpack_reset(reset_out: Any) -> tuple[str, dict[str, Any]]:
        """Normalize a TextWorld ``reset()`` return into ``(obs_str, infos)``.

        Real shape: ``(obs_list, infos_dict)``.  Tolerates a bare obs (no infos)
        or scalar obs from a fake.  Returns infos with batched values UNWRAPPED
        to scalars (``{"won": [False]}`` → ``{"won": False}``) for convenience.
        """
        if isinstance(reset_out, tuple):
            if len(reset_out) >= 2:
                raw_obs, raw_infos = reset_out[0], reset_out[1]
            elif len(reset_out) == 1:
                raw_obs, raw_infos = reset_out[0], {}
            else:
                raw_obs, raw_infos = "", {}
        else:
            raw_obs, raw_infos = reset_out, {}
        obs = _first(raw_obs, default="")
        return ("" if obs is None else str(obs)), ALFWorldEnv._unwrap_infos(raw_infos)

    @staticmethod
    def _unpack_step(step_out: Any) -> tuple[str, float, bool, dict[str, Any]]:
        """Normalize a TextWorld ``step()`` return into scalars.

        Real shape: ``(obs_list, scores_list, dones_list, infos_dict)``.
        Tolerates shorter tuples from a minimal fake (``(obs, reward, done)`` or
        ``(obs, reward, done, info)``) — the common gym 4-tuple — by inferring
        the batched-vs-scalar shape per element.
        """
        if not isinstance(step_out, tuple):
            return ("" if step_out is None else str(step_out)), 0.0, False, {}

        obs = _first(step_out[0], default="") if len(step_out) >= 1 else ""
        score = _first(step_out[1], default=0.0) if len(step_out) >= 2 else 0.0
        done = _first(step_out[2], default=False) if len(step_out) >= 3 else False
        infos = step_out[3] if len(step_out) >= 4 else {}

        obs_str = "" if obs is None else str(obs)
        try:
            score_f = float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            score_f = 0.0
        return obs_str, score_f, bool(done), ALFWorldEnv._unwrap_infos(infos)

    @staticmethod
    def _unwrap_infos(infos: Any) -> dict[str, Any]:
        """Unwrap a batched infos dict (``{k: [v]}``) to ``{k: v}``; pass dicts.

        TextWorld's batched env returns each info key as a length-batch list;
        with batch_size=1 we take index 0.  A fake may already pass scalars.
        """
        if not isinstance(infos, dict):
            return {}
        out: dict[str, Any] = {}
        for key, value in infos.items():
            out[key] = _first(value, default=value) if isinstance(value, (list, tuple)) else value
        return out

    @staticmethod
    def _won_from(infos: dict[str, Any], score: float | None) -> bool:
        """Derive the episode success signal from infos / score.

        Primary signal is ``infos["won"]`` (TextWorld's goal-condition flag,
        requested via ``EnvInfos(won=True)``).  Falls back to ``infos["success"]``
        or a positive score for fakes that don't surface ``won``.
        """
        if isinstance(infos, dict):
            if "won" in infos:
                return bool(infos["won"])
            if "success" in infos:
                return bool(infos["success"])
        try:
            return bool(score) and float(score) > 0.0
        except (TypeError, ValueError):
            return False
