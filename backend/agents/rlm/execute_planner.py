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
    keys: tuple               # ordered reward-key candidates, first found wins
    log_glob: str             # "$OUTPUT_DIR/*.log"
    metrics_file: str | None  # for json_file: "all_results.json"


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

# Candidate reward keys verl logs, most-specific first (mirrors
# scripts/ucpo_execute_cell/train_cell.py's _REWARD_KEYS).
_REWARD_KEYS: tuple[str, ...] = (
    "critic/rewards/mean",
    "critic/score/mean",
    "reward/mean",
    "critic/rewards/mean/all",
)


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

    launch = LaunchSpec(
        kind="module",
        command=f"python -m {module} " + " ".join(args),
        cwd="",
        overrides=dict(_DOWNSCALE_OVERRIDES),
    )

    data_slice: dict | None = None
    train_file = _extract_train_file(code_dir, args)
    if train_file is not None:
        data_slice = {"train_file": train_file, "slice_rows": _SLICE_ROWS}

    reward = RewardSpec(
        kind="verl",
        keys=_REWARD_KEYS,
        log_glob="$OUTPUT_DIR/*.log",
        metrics_file=None,
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
        est_vram_gb=70.0,
        confidence=confidence,
        source="deterministic",
        reason=f"verl launch entrypoint '{module}' found in {rel_script}",
        data_slice=data_slice,
    )
