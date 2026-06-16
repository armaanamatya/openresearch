"""Per-paper invariant loader — declarative pre-flight checks driven by
``docs/papers/<arxiv_id>.yaml``.

The 2026-05-25 codex review surfaced that the SDAR paper (arXiv 2605.15155)
needs three pre-flight invariants the harness was missing: (a) the
sigmoid-gated OPSD gate must have ``stop_gradient`` (a ``.detach()`` call
or ``torch.no_grad()`` context); (b) the agent must load real Qwen
weights via ``AutoModelForCausalLM.from_pretrained("Qwen/...")``, not an
``nn.Linear(d, V)`` surrogate; (c) the per-experiment metrics dict must
use canonical model keys (``qwen3_1_7b``) so the rubric grader can map
them back to the paper's declared variants.

Rather than hardcode SDAR-specific Python in pre_flight_validator.py,
this module reads invariants declaratively from the paper-hint YAML and
generates the AST checks generically. Any paper can declare:

  algorithm_invariants:
    stop_gradient_on_gate: true
    stop_gradient_variables: [gate, g_t, gate_t]
    real_model_required: true

  models_in_paper:
    qwen3_1_7b: Qwen/Qwen3-1.7B-Instruct
    qwen2_5_3b: Qwen/Qwen2.5-3B-Instruct

  paper_targets:
    variants_required: [qwen3_1_7b, qwen2_5_3b, qwen2_5_7b, grpo, opsd, ...]

…and pre-flight enforces. New papers add their YAML; no code change needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Common variable-name heuristic for the "sigmoid gate" pattern across RL
# distillation papers (SDAR, OPSD, DPO-style gating). When a paper declares
# `stop_gradient_on_gate: true` without an explicit list of variable names,
# we fall back to this set.
_DEFAULT_GATE_VARIABLE_NAMES = (
    "gate", "g_t", "gate_t", "g", "alpha_t", "weight_t",
)


# Common surrogate patterns. If a paper requires real model weights but
# train.py only contains these, that's a surrogate. The check is opt-in
# per paper (via `real_model_required: true`).
_SURROGATE_MODEL_TOKENS = (
    "nn.Linear",
    "nn.Embedding",
    "torch.nn.Linear",
    "torch.nn.Embedding",
)


# Registry of well-known training-algorithm token sets. When a paper YAML
# declares a `loss` like ``"L = L_GRPO + lambda * L_OPSD"``, we parse the
# algorithm names (case-insensitive substring) and pull the canonical
# token set each algorithm's train.py implementation should reference.
# Pre-flight then verifies the agent's train.py actually mentions those
# tokens — catches "agent dropped GRPO entirely and used vanilla CE"
# class regressions. Refresh as new RL/distillation algorithms land.
_ALGORITHM_TOKEN_PATTERNS: dict[str, tuple[str, ...]] = {
    # RL — policy gradient family
    "grpo":      ("logprobs", "advantages", "ratio", "clip"),
    "ppo":       ("logprobs", "advantages", "ratio", "clip"),
    "trpo":      ("logprobs", "advantages", "kl"),
    "reinforce": ("logprobs", "rewards"),
    "dpo":       ("logprobs", "ref_logprobs", "beta"),
    # Self-distillation family
    "opsd":      ("teacher", "student", "sigmoid"),
    "rlsd":     ("teacher", "student", "kl"),
    "skill-sd":  ("teacher", "student", "skill"),
    "kd":        ("teacher", "student", "kl"),
    # Standard supervised
    "ce":        ("cross_entropy",),
    "mle":       ("log",),
}


@dataclass
class LossInvariant:
    """One algorithmic loss term that train.py must reference.

    Extracted heuristically from ``algorithm_invariants.loss`` in the
    paper YAML. The pre-flight check verifies at least
    ``min_matching_tokens`` of ``required_tokens`` appear in the source
    text of train.py. Tolerant of paraphrasing — the agent's variable
    names may differ slightly from the canonical reference."""

    name: str
    required_tokens: tuple[str, ...]
    min_matching_tokens: int = 2


def _extract_loss_invariants(loss_str: str) -> tuple[LossInvariant, ...]:
    """Heuristically parse ``"L = L_GRPO + lambda * L_OPSD"``-style strings.

    Walks the string for any algorithm name in ``_ALGORITHM_TOKEN_PATTERNS``;
    returns one ``LossInvariant`` per match. Case-insensitive substring,
    word-boundary aware so ``"reinforce-style"`` matches but ``"reinforcement"``
    does NOT (avoids over-matching).
    """
    if not loss_str:
        return ()
    import re as _re
    out: list[LossInvariant] = []
    for algo, tokens in _ALGORITHM_TOKEN_PATTERNS.items():
        # Match algorithm name with non-alphanumeric boundaries — allows
        # underscore-prefixed forms like ``L_GRPO`` (common in paper loss
        # expressions) but rejects substring matches like ``REINFORCEment``
        # (where the char after is alphabetic).
        pattern = rf"(?<![A-Za-z0-9]){_re.escape(algo)}(?![A-Za-z0-9])"
        if _re.search(pattern, loss_str, flags=_re.IGNORECASE):
            # min_matching_tokens scales with the token set size: a 4-token
            # algorithm requires 2 matches (50%); a 2-token algorithm
            # requires 1.
            mm = max(1, len(tokens) // 2)
            out.append(LossInvariant(
                name=algo.upper(),
                required_tokens=tokens,
                min_matching_tokens=mm,
            ))
    return tuple(out)


@dataclass
class AlgorithmInvariant:
    """Per-paper algorithmic invariants the agent's train.py must satisfy."""

    stop_gradient_variables: tuple[str, ...] = ()
    """Variable names whose assignment RHS must be wrapped in ``.detach()``
    or inside a ``torch.no_grad()`` block. The pre-flight AST scan
    walks every ``target = expr`` statement and flags violations."""

    real_model_required: bool = False
    """When True, train.py must call ``AutoModelForCausalLM.from_pretrained(<canonical_path>)``
    where the path matches one of ``ModelInvariants.canonical_models`` values.
    Surrogate models (bare ``nn.Linear`` / ``nn.Embedding`` LM heads) are blocked."""

    rl_rollout_centric: bool = False
    """When True, the training loop is rollout-based (GRPO, PPO, REINFORCE)
    not epoch-based. Affects prompt text: per-epoch print cadence is
    replaced with per-rollout / per-policy-update cadence."""

    loss_invariants: tuple[LossInvariant, ...] = ()
    """Algorithm-specific token sets that train.py must reference. Derived
    from the paper YAML's ``algorithm_invariants.loss`` field. The
    pre-flight check verifies at least ``min_matching_tokens`` appear
    in train.py source for each invariant; catches "agent dropped the
    paper's algorithm and used vanilla CE" regressions."""


@dataclass
class ModelInvariants:
    """Per-paper model identity invariants."""

    canonical_models: dict[str, str] = field(default_factory=dict)
    """Mapping ``short_name -> HF_path``. The HF path is what the agent's
    ``from_pretrained`` call MUST reference; the short_name is what
    ``metrics.json.per_model`` keys MUST canonicalize to."""

    variants_required: tuple[str, ...] = ()
    """Variants the paper compares (model sizes + baseline algorithms).
    Each must appear in ``metrics.json.per_model`` OR ``metrics.json.omitted``
    with a reason. Surfaced in the agent's prompt so the agent knows the
    canonical key set up-front."""


@dataclass
class PaperInvariants:
    """Aggregated invariants for one paper, loaded from its YAML hint."""

    arxiv_id: str
    algorithm: AlgorithmInvariant | None = None
    models: ModelInvariants | None = None
    multi_env: tuple[str, ...] = ()
    """Environment names for multi-env papers (SDAR: alfworld + search_qa + webshop).
    Drives the ``per_model.per_dataset`` nesting requirement in the prompt."""


def _short_to_pyident(name: str) -> str:
    """Canonicalize a model short name to a Python-identifier safe form.

    ``qwen3-1.7b`` → ``qwen3_1_7b``, ``Qwen/Qwen3-1.7B-Instruct`` → ``qwen3_1_7b_instruct``.
    Used to canonicalize ``metrics.json.per_model`` keys + the rubric-grader
    cross-check. Idempotent.
    """
    out = name.lower()
    for ch in ("/", "-", ".", " ", ":"):
        out = out.replace(ch, "_")
    # Collapse multiple underscores.
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def canonical_model_key(name: str) -> str:
    """Canonical comparison key for a model id, robust to display-vs-metrics drift.

    Builds on :func:`_short_to_pyident` and additionally strips a trailing chat/
    instruct variant suffix so a scope-spec display name and an agent's
    ``metrics.json`` per_model key compare equal::

        canonical_model_key("Qwen3-1.7B-Instruct")      == "qwen3_1_7b"
        canonical_model_key("Qwen/Qwen2.5-3B-Instruct")  == "qwen2_5_3b"
        canonical_model_key("qwen3_1_7b")                == "qwen3_1_7b"   # idempotent

    Used by the scope-metrics validator so a correctly-run model is never flagged
    ``per_model_incomplete`` purely because of name formatting.
    """
    s = _short_to_pyident(name.rsplit("/", 1)[-1])  # drop HF org prefix (Qwen/…)
    for suffix in ("_instruct", "_chat", "_it", "_base"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s.strip("_")


def load_paper_invariants(arxiv_id: str, repo_root: Path | None = None) -> PaperInvariants | None:
    """Load ``docs/papers/<arxiv_id>.yaml`` and extract invariants. Fail-soft.

    Returns ``None`` when:
      * ``arxiv_id`` is empty / None
      * the yaml file doesn't exist for this arxiv id
      * PyYAML can't parse it (returns None so pre-flight skips silently)
      * no invariant fields are declared

    Returns a populated ``PaperInvariants`` when at least one of
    ``algorithm_invariants`` / ``models_in_paper`` / ``paper_targets.variants_required``
    is present in the yaml.
    """
    if not arxiv_id:
        return None
    if repo_root is None:
        # Default: the repo root is two parents up from this file
        # (backend/agents/rlm/paper_invariants.py → backend/agents → backend → repo).
        repo_root = Path(__file__).resolve().parents[3]
    yaml_path = repo_root / "docs" / "papers" / f"{arxiv_id}.yaml"
    if not yaml_path.exists():
        return None
    try:
        import yaml as _yaml
        data = _yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — pre-flight observability must never raise
        return None
    if not isinstance(data, dict):
        return None

    inv = PaperInvariants(arxiv_id=arxiv_id)

    # --- algorithm_invariants ---
    algo_block = data.get("algorithm_invariants") or {}
    if isinstance(algo_block, dict) and algo_block:
        sg_vars: tuple[str, ...] = ()
        if algo_block.get("stop_gradient_on_gate"):
            # Explicit override list, or the default gate-variable heuristic.
            declared = algo_block.get("stop_gradient_variables")
            if isinstance(declared, list) and declared:
                sg_vars = tuple(str(v).strip() for v in declared if str(v).strip())
            else:
                sg_vars = _DEFAULT_GATE_VARIABLE_NAMES
        rl_rollout = bool(algo_block.get("rl_rollout_centric", False))
        # Heuristic auto-detect: paper YAML names a GRPO / PPO loss term?
        loss_str = str(algo_block.get("loss", ""))
        if not rl_rollout:
            ll = loss_str.lower()
            if "grpo" in ll or "ppo" in ll or "reinforce" in ll:
                rl_rollout = True
        # Parse loss invariants from the loss expression — catches the
        # GRPO+OPSD class of regressions where the agent silently drops
        # the paper's algorithm and substitutes vanilla CE.
        loss_invariants = _extract_loss_invariants(loss_str)
        inv.algorithm = AlgorithmInvariant(
            stop_gradient_variables=sg_vars,
            real_model_required=bool(algo_block.get("real_model_required", False)),
            rl_rollout_centric=rl_rollout,
            loss_invariants=loss_invariants,
        )

    # --- models_in_paper ---
    models_block = data.get("models_in_paper") or {}
    variants_required: tuple[str, ...] = ()
    pt = data.get("paper_targets") or {}
    if isinstance(pt, dict):
        vr = pt.get("variants_required") or []
        if isinstance(vr, list):
            variants_required = tuple(str(v).strip() for v in vr if str(v).strip())
    if (isinstance(models_block, dict) and models_block) or variants_required:
        canonical: dict[str, str] = {}
        if isinstance(models_block, dict):
            for k, v in models_block.items():
                if isinstance(k, str) and isinstance(v, str):
                    canonical[_short_to_pyident(k)] = v.strip()
        inv.models = ModelInvariants(
            canonical_models=canonical,
            variants_required=variants_required,
        )
        # If models are declared but the algorithm-level flag isn't set,
        # default to requiring real models — declaring an HF path means
        # "this is the model that needs to load".
        if inv.algorithm is None:
            inv.algorithm = AlgorithmInvariant()
        if canonical and not inv.algorithm.real_model_required:
            inv.algorithm.real_model_required = True

    # --- multi_env ---
    datasets_block = data.get("datasets") or {}
    if isinstance(datasets_block, dict) and len(datasets_block) > 1:
        inv.multi_env = tuple(k for k in datasets_block.keys() if isinstance(k, str))

    if inv.algorithm is None and inv.models is None and not inv.multi_env:
        return None
    return inv


__all__ = [
    "AlgorithmInvariant",
    "LossInvariant",
    "ModelInvariants",
    "PaperInvariants",
    "load_paper_invariants",
    "_short_to_pyident",
    "canonical_model_key",
    "_extract_loss_invariants",
    "_ALGORITHM_TOKEN_PATTERNS",
    "_DEFAULT_GATE_VARIABLE_NAMES",
    "_SURROGATE_MODEL_TOKENS",
]
