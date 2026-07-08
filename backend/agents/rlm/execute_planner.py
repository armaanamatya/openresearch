"""Deterministic execute-mode planning — turn a cloned repo into an ``ExecuteSpec``.

Pure/deterministic: NO LLM, NO network, NO GPU. Given a framework detected by
``framework_detector.detect_framework`` (verl only, in Increment 1), enumerate
the repo's own launch scripts, extract the authors' own entrypoint + hydra
overrides VERBATIM, and produce a downscaled-but-faithful ``ExecuteSpec`` the
generic ``execute_cell_synth`` shim can run. Contracts mirror
``docs/superpowers/specs/2026-07-07-deterministic-any-paper-execute-mode-design.md``
§4, and the downscale numbers mirror the hand-authored, GPU-proven
``scripts/ucpo_execute_cell/train_cell.py`` (UCPO, real reward 0.25).

CRITICAL invariant (do not "fix" this): ``data.max_response_length`` is NEVER
downscaled. Truncating it shrinks the token budget available for a model's
chain-of-thought before its boxed answer, which silently collapses the reward
to ~0 — a false degenerate-training signal, not a real failure. The authors'
own value for that key always survives verbatim in ``launch.command``.

Shell-variable resolution (deterministic, NO shell execution): the real
``run_ucpo_1.5b.sh`` sets its hydra args via shell variables
(``data.train_files=$TRAIN_DATADIR`` et al). ``_collect_shell_vars`` scans the
launch script for SIMPLE literal ``VAR=<literal>`` assignments — including
nested ones like ``SAVE_DIR=../ds_${EXP_NAME}/`` resolved by bounded iterative
substitution — and those values are substituted into the extracted argument
tokens purely, via regex, never a shell. Known limitation (by design, not a
bug): only genuinely dynamic assignments — command substitution
(``$(...)``/backticks) or a value with shell metacharacters (``| & ; > <``) —
stay unresolved, and such a token is left verbatim rather than guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from backend.agents.rlm.framework_detector import detect_framework

# --- contracts (design §4) ---------------------------------------------------


@dataclass(frozen=True)
class LaunchSpec:
    kind: str            # "module" | "script" | "shell"
    command: str         # "python -m ucpo.main_run <authors' args>"
    cwd: str             # relative to code/ ("" | "verl")
    overrides: dict       # framework knob -> downscaled value (applied deterministically)


@dataclass(frozen=True)
class RewardSpec:
    kind: str                 # "verl" | "hf_trainer" | "json_file" | "log_regex"
    keys: tuple               # TRAIN reward-key candidates (fallback) -> reward_mean
    log_glob: str             # "$OUTPUT_DIR/*.log"
    metrics_file: str | None  # for json_file: "all_results.json"
    eval_keys: tuple = ()     # HELD-OUT val-key candidates (preferred) -> success_rate


@dataclass(frozen=True)
class ExecuteSpec:
    framework: str            # verl | hf_trainer | accelerate | python | bash
    setup: tuple               # ("pip install -e verl --no-deps --no-build-isolation",)
    launch: LaunchSpec
    reward: RewardSpec
    image_key: str            # "verl" | "base"
    est_vram_gb: float
    confidence: float
    source: str                # "deterministic" | "hybrid" | "llm"
    reason: str
    data_slice: dict | None = None   # {"train_file": "...parquet", "slice_rows": 32} | None


# --- launch extraction --------------------------------------------------------

_MODULE_INVOCATION_RE = re.compile(r"python3?\s+-m\s+(\S+)")
_ENTRYPOINT_SUFFIXES: tuple[str, ...] = ("main_run", "main_ppo", "main", "train")
_VERL_SIGNATURE_TOKENS: tuple[str, ...] = ("actor_rollout_ref", "algorithm.adv_estimator")

# Shell-variable resolution (deterministic, no shell execution).
# A simple ``[export ]NAME=<rhs>`` assignment; ``$VAR``/``${VAR}`` references.
_SHELL_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_]\w*)=(.*)$")
_VAR_REF_RE = re.compile(r"\$\{?(\w+)\}?")
# An RHS carrying any of these is dynamic (command substitution / pipes / redirects /
# chaining) — never resolved to a literal, always left verbatim.
_DYNAMIC_RHS_MARKERS: tuple[str, ...] = ("$(", "`", "|", "&", ";", ">", "<")
# Shell arg-passthrough tokens that are never a valid hydra override.
_ARG_PASSTHROUGH_TOKENS = frozenset(("$@", '"$@"', "'$@'"))
_MAX_VAR_RESOLVE_PASSES = 10

# Enumeration order matters: the FIRST matching script/line wins (deterministic,
# no ambiguity resolution in Increment 1 — that is the design's bounded-LLM
# tie-break, deliberately out of scope here).
_SCRIPT_GLOB_PATTERNS: tuple[str, ...] = (
    "scripts*/**/*.sh",
    "run*.sh",
    "train*.sh",
    "examples/**/*.sh",
)

_SLICE_ROWS = 32

# Estimated VRAM for the downscaled single-GPU execute cell. MUST stay <= 64 so
# the shared capacity_gate (cell_matrix.py: est_vram_gb * 1.25 headroom vs the
# 80GB A100-80 budget) does not drop the cell BEFORE dispatch — 60 * 1.25 = 75 < 80.
# The downscale (param+optimizer offload, gpu_memory_utilization=0.6, batch 8, a
# 1.5B model) fits well under 80GB; the prior 70.0 was a pre-gate estimate that the
# direct-run_matrix proof tolerated but the full harness path (which applies the
# gate) rejected as capacity_exhausted. Tunable per model/GPU via
# OPENRESEARCH_EXECUTE_SYNTH_VRAM_GB at the synth layer (execute_cell_synth).
_EXECUTE_CELL_VRAM_GB = 60.0

# The verl-adapter downscale: fit 1xA100 + a fast non-zero-reward proof,
# applied AFTER the authors' own args (hydra last-wins). Mirrors the flags in
# scripts/ucpo_execute_cell/train_cell.py. `data.max_response_length` is
# deliberately absent — see the module docstring.
_DOWNSCALE_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("trainer.n_gpus_per_node", "1"),
    ("trainer.nnodes", "1"),
    ("trainer.save_freq", "-1"),
    ("trainer.test_freq", "-1"),
    ("trainer.total_epochs", "1"),
    ("trainer.logger", "[console]"),
    ("data.train_batch_size", "8"),
    ("actor_rollout_ref.actor.ppo_mini_batch_size", "8"),
    ("actor_rollout_ref.rollout.tensor_model_parallel_size", "1"),
    ("actor_rollout_ref.rollout.gpu_memory_utilization", "0.6"),
    ("actor_rollout_ref.model.enable_gradient_checkpointing", "True"),
    ("actor_rollout_ref.actor.fsdp_config.param_offload", "True"),
    ("actor_rollout_ref.actor.fsdp_config.optimizer_offload", "True"),
    ("actor_rollout_ref.ref.fsdp_config.param_offload", "True"),
)

# TRAIN reward keys verl logs, most-specific first (mirrors
# scripts/ucpo_execute_cell/train_cell.py's _REWARD_KEYS). These are a TRAINING
# signal, not a held-out measurement — the shim writes them as ``reward_mean``
# (never ``success_rate``), so the eval-provenance guard treats the cell as
# RL-reward-only (exempt) instead of vetoing a mislabeled success rate.
_REWARD_KEYS: tuple[str, ...] = (
    "critic/rewards/mean",
    "critic/score/mean",
    "reward/mean",
    "critic/rewards/mean/all",
)

# HELD-OUT validation keys verl logs, preferred over the train reward — THIS is
# the paper's actual claimed result (e.g. UCPO's math-benchmark accuracy). The
# shim writes the first that resolves as ``success_rate`` with aggregate
# provenance. A key ending in ``/`` is a per-data_source PREFIX (verl's
# reward-manager val logs ``val/test_score/<data_source>`` with no bare
# aggregate) resolved to its single distinct concrete sub-key. Only tried when
# the repo actually ships a held-out val set (see ``plan``).
_EVAL_KEYS: tuple[str, ...] = (
    "val/success_rate",   # agentic (SDAR): verl logs the flat aggregate directly
    "val/test_score/",    # verl reward-manager val, one key per data_source (math, ...)
    "val/acc/mean",
    "val/score/mean",
)

# Validate every step so a real ``val/*`` score is logged for the shim to bridge.
# The val slice is tiny (``_VAL_SLICE_ROWS``) so the extra passes are cheap; the
# adapter keeps the LAST logged value (verl's final/most-converged val point).
_VAL_TEST_FREQ = 1
_VAL_SLICE_ROWS = 8


def _strip_one_quote_layer(value: str) -> str:
    """Strip ONE layer of surrounding matched single/double quotes."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _substitute_vars(text: str, var_map: dict[str, str]) -> str:
    """Replace ``$VAR`` / ``${VAR}`` with ``var_map[VAR]`` where known; an
    unknown var reference is left verbatim (never guessed)."""
    def _repl(match: re.Match) -> str:
        name = match.group(1)
        return var_map[name] if name in var_map else match.group(0)

    return _VAR_REF_RE.sub(_repl, text)


def _collect_shell_vars(script_text: str) -> dict[str, str]:
    """Resolve SIMPLE literal ``VAR=<literal>`` assignments from a launch
    script into a ``{name: value}`` map — purely, with no shell execution.

    An RHS containing command substitution or shell metacharacters is skipped
    (left for the token to keep verbatim). One layer of surrounding quotes is
    stripped. Known-literal ``$VAR``/``${VAR}`` references appearing inside RHS
    values are substituted in bounded iterative passes (so a nested
    ``SAVE_DIR=../ds_${EXP_NAME}/`` resolves once ``EXP_NAME`` is known); a
    value with an unresolved ``$VAR`` after the passes stays as-is.
    """
    raw: dict[str, str] = {}
    for line in script_text.splitlines():
        match = _SHELL_ASSIGN_RE.match(line)
        if not match:
            continue
        name, rhs = match.group(1), match.group(2).strip()
        if any(marker in rhs for marker in _DYNAMIC_RHS_MARKERS):
            continue  # dynamic RHS — never resolve to a literal
        raw[name] = _strip_one_quote_layer(rhs)

    resolved: dict[str, str] = dict(raw)
    for _ in range(_MAX_VAR_RESOLVE_PASSES):
        changed = False
        for name, value in list(resolved.items()):
            new_value = _substitute_vars(value, resolved)
            if new_value != value:
                resolved[name] = new_value
                changed = True
        if not changed:
            break
    return resolved


def _resolve_arg_tokens(tokens: list[str], var_map: dict[str, str]) -> list[str]:
    """Substitute resolved shell vars into each arg token and drop shell
    arg-passthrough (``$@``) tokens (empty here; verl chokes on a literal)."""
    resolved: list[str] = []
    for token in tokens:
        if token in _ARG_PASSTHROUGH_TOKENS:
            continue
        resolved.append(_substitute_vars(token, var_map))
    return resolved


def _join_continuations(text: str) -> list[str]:
    """Collapse backslash line-continuations into single logical lines.

    A launch script's real invocation is typically one shell statement spread
    over many ``\\``-continued lines (hydra CLI overrides, one per line) — the
    module regex only ever sees a full logical line, never a fragment.
    """
    logical: list[str] = []
    buf: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.rstrip()
        if stripped.endswith("\\"):
            buf.append(stripped[:-1])
            continue
        buf.append(stripped)
        logical.append(" ".join(part.strip() for part in buf if part.strip()))
        buf = []
    if buf:
        logical.append(" ".join(part.strip() for part in buf if part.strip()))
    return logical


def _iter_candidate_scripts(code_dir: Path) -> list[Path]:
    ordered: list[Path] = []
    seen: set[Path] = set()
    for pattern in _SCRIPT_GLOB_PATTERNS:
        for path in sorted(code_dir.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                ordered.append(path)
    return ordered


def _is_entrypoint_module(module: str) -> bool:
    return any(
        module == suffix or module.endswith(f".{suffix}") for suffix in _ENTRYPOINT_SUFFIXES
    )


def _find_launch_invocation(code_dir: Path) -> tuple[Path, str, list[str]] | None:
    """Return ``(script_path, module, hydra_args)`` for the first matching
    ``python -m <module> <args>`` invocation, or ``None``."""
    for script_path in _iter_candidate_scripts(code_dir):
        try:
            text = script_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        var_map = _collect_shell_vars(text)
        for line in _join_continuations(text):
            match = _MODULE_INVOCATION_RE.search(line)
            if not match:
                continue
            module = match.group(1)
            after = line[match.end():].strip()
            if not (_is_entrypoint_module(module) or any(tok in after for tok in _VERL_SIGNATURE_TOKENS)):
                continue
            raw_args = after.split() if after else []
            args = _resolve_arg_tokens(raw_args, var_map)
            return script_path, module, args
    return None


def _extract_train_file(code_dir: Path, args: list[str]) -> str | None:
    """Return the authors' ``data.train_files=`` value if it resolves to a
    real ``.parquet`` file under ``code_dir``, else ``None``."""
    resolved_root = code_dir.resolve()
    for token in args:
        if not token.startswith("data.train_files="):
            continue
        value = token.split("=", 1)[1]
        if not value.endswith(".parquet"):
            continue
        candidate = (code_dir / value).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            continue  # escapes code/ — never trust a path traversal
        if candidate.is_file():
            return value
    return None


def _extract_val_file(code_dir: Path, args: list[str]) -> str | None:
    """Return the authors' ``data.val_files=`` value if it resolves to a real
    ``.parquet`` file under ``code_dir``, else ``None`` (mirror of
    ``_extract_train_file`` — a held-out val set is what makes verl log a real
    ``val/*`` score; without one the cell honestly falls back to the train
    reward)."""
    resolved_root = code_dir.resolve()
    for token in args:
        if not token.startswith("data.val_files="):
            continue
        value = token.split("=", 1)[1]
        if not value.endswith(".parquet"):
            continue
        candidate = (code_dir / value).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            continue  # escapes code/ — never trust a path traversal
        if candidate.is_file():
            return value
    return None


def _detect_verl_dirname(code_dir: Path) -> str:
    """The bundled verl project's directory name — "usually 'verl'"."""
    candidate = code_dir / "verl"
    if candidate.is_dir() and (
        (candidate / "setup.py").is_file() or (candidate / "pyproject.toml").is_file()
    ):
        return "verl"
    return "verl"


def plan(code_path: str | Path) -> ExecuteSpec | None:
    """Deterministically plan an execute-mode cell for a cloned repo.

    Returns ``None`` (a fall-through signal, never an exception) when the
    framework isn't confidently verl, or no launch entrypoint can be found —
    the caller should fall back to today's LLM execute path.
    """
    code_dir = Path(code_path)
    framework, confidence, _evidence = detect_framework(code_dir)
    if framework != "verl" or confidence < 0.7:
        return None

    found = _find_launch_invocation(code_dir)
    if found is None:
        return None
    script_path, module, args = found

    overrides = dict(_DOWNSCALE_OVERRIDES)  # test_freq=-1 default (validation off)

    data_slice: dict | None = None
    train_file = _extract_train_file(code_dir, args)
    if train_file is not None:
        data_slice = {"train_file": train_file, "slice_rows": _SLICE_ROWS}

    # Held-out validation: only when the repo actually ships a val set. Enabling
    # test_freq with no val_files would make verl error; without a val set we
    # honestly fall back to the train reward (eval_keys stays empty).
    eval_keys: tuple = ()
    val_file = _extract_val_file(code_dir, args)
    if val_file is not None:
        overrides["trainer.test_freq"] = str(_VAL_TEST_FREQ)  # hydra last-wins over the -1 default
        data_slice = dict(data_slice or {})
        data_slice["val_file"] = val_file
        data_slice["val_rows"] = _VAL_SLICE_ROWS
        eval_keys = _EVAL_KEYS

    launch = LaunchSpec(
        kind="module",
        command=f"python -m {module} " + " ".join(args),
        cwd="",
        overrides=overrides,
    )

    reward = RewardSpec(
        kind="verl",
        keys=_REWARD_KEYS,
        log_glob="$OUTPUT_DIR/*.log",
        metrics_file=None,
        eval_keys=eval_keys,
    )

    verl_dirname = _detect_verl_dirname(code_dir)
    setup = (f"pip install -e {verl_dirname} --no-deps --no-build-isolation",)

    try:
        rel_script = script_path.relative_to(code_dir)
    except ValueError:
        rel_script = script_path

    return ExecuteSpec(
        framework="verl",
        setup=setup,
        launch=launch,
        reward=reward,
        image_key="verl",
        est_vram_gb=_EXECUTE_CELL_VRAM_GB,
        confidence=confidence,
        source="deterministic",
        reason=f"verl launch entrypoint '{module}' found in {rel_script}",
        data_slice=data_slice,
    )
