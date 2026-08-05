"""rubric_gen.py — self-generate a PaperBench-shaped rubric tree from paper text.

For arXiv runs that arrive without a vendored rubric.json, this module derives
a structurally compatible rubric from the paper itself so the run is scorable
by ``backend.evals.paperbench.leaf_scorer`` (flatten_leaves / roll_up).
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# --- Evaluator-lockdown canary leaf (spec §10.4) ----------------------------

#: Flag gate for the ONE call site inside generate_rubric_tree; default OFF.
RUBRIC_CANARY_ENV = "OPENRESEARCH_RUBRIC_CANARY"
_TRUTHY = ("1", "true", "yes", "on")

CANARY_LEAF_ID = "canary_nonexistent_artifact"
_CANARY_REQUIREMENTS = (
    "Verify the artifact 'artifacts/psi_calibration_manifest.bin' exists and "
    "its recorded psi-divergence metric is reported."
)


class LlmClient(Protocol):
    def complete(self, *, system: str, user: str) -> str: ...


# --- Deterministic-leaf annotation (grader-fidelity A2) ----------------------
#
# WHY THIS EXISTS
# ---------------
# ``deterministic_leaf_checker`` can grade a leaf in pure Python — a
# hyperparameter against ``provenance.json``, an artifact against the filesystem,
# a numeric result against ``metrics.json`` — but it only ever fires on a leaf
# carrying a ``check_kind`` + ``assertion`` annotation. Until this module emitted
# one, NO generated rubric ever did, so the entire deterministic path was dead
# code and every leaf of every arXiv paper was graded by LLM opinion — exactly
# what the project's "evidence, not grade" red line forbids.
#
# THE BIAS: NO ANNOTATION BEATS A GUESSED ONE
# -------------------------------------------
# An annotation is written at rubric-gen time — BEFORE the run — so it can only
# *predict* what the artifacts will contain. A wrong prediction is not a harmless
# miss: it deterministically FAILS a faithful reproduction, and for a triage
# product a false negative is the expensive error (cf. learn.md 2026-07-07, where
# an over-broad LR guard hard-blocked a faithful alpha=0.0 ablation and two SDAR
# "surrogate" verdicts were overturned as faithful). So every gate below is
# one-directional: it can only ever REFUSE to annotate. When any check is
# unsatisfied the leaf is left un-annotated and the LLM grades it — the exact
# behavior we have today. Annotation only ever ADDS determinism where the claim
# is provably grounded; it never subtracts coverage.
#
# The four gates a proposed annotation must clear:
#   1. STRUCTURE   — the kind/assertion must match the checker's contract.
#   2. GROUNDING   — the asserted value must actually occur in the PAPER TEXT
#                    *and* in the leaf's own requirements string. A value the
#                    paper never states is a hallucination; drop it.
#   3. VOCABULARY  — a hparam ``field`` must be one the provenance producers
#                    ACTUALLY write (see _PROVENANCE_HPARAM_FIELDS). Asserting a
#                    field nobody emits is a guaranteed 0.0 on a faithful run.
#   4. SPECIFICITY — the assertion must not UNDER-specify the leaf. A leaf that
#                    pins two numbers, or whose substance is algorithmic fidelity
#                    ("implements the gate g_t=σ(β·Δ_t)"), cannot be reduced to a
#                    single scalar/file-exists check without over-crediting a
#                    stub. Those stay with the LLM, which is correct.
#
# Gated by the SAME flag that gates the routing (OPENRESEARCH_DETERMINISTIC_LEAVES).
# Unset ⇒ the prompt, the tree, and therefore generated_rubric.json are all
# byte-identical to before.

#: Flag gate — shared with ``leaf_scorer._deterministic_leaves_enabled``.
DETERMINISTIC_LEAVES_ENV = "OPENRESEARCH_DETERMINISTIC_LEAVES"

#: The three kinds, imported from the checker so the contract has ONE source of
#: truth. If the checker's vocabulary ever changes, this import breaks loudly
#: rather than silently emitting annotations nothing routes.
from backend.evals.paperbench.deterministic_leaf_checker import (  # noqa: E402
    COEFFICIENTS_KEY,
    DETERMINISTIC_CHECK_KINDS,
)

CHECK_HPARAM = "deterministic:hparam"
CHECK_ARTIFACT = "deterministic:artifact"
CHECK_NUMERIC = "deterministic:numeric"

#: Hyperparameter fields the provenance PRODUCERS actually write. This is a
#: whitelist, not a guess — every entry is emitted by at least one of:
#:   * the harness-owned cell route (``provenance._PROVENANCE_PARAM_KEYS``:
#:     lr / weight_decay / momentum / dropout / epochs / seed / batch_size / …)
#:   * the agent-facing contract shown in the implementer prompt
#:     (``baseline_implementation._PROVENANCE_BLOCK``: model_key / baseline /
#:     seed / epochs / steps / batch_size / per_optimizer{opt:{lr,…}} / …)
#:
#: These are BOOKKEEPING knobs — the training recipe. The paper's ALGORITHMIC
#: constants (β, λ, τ, a temperature, a clip ε) live in their own namespace; see
#: _COEFFICIENT_NAMES below.
_PROVENANCE_HPARAM_FIELDS: frozenset[str] = frozenset({
    "lr", "batch_size", "epochs", "steps", "seed",
    "weight_decay", "momentum", "dropout", "num_classes",
})

# --------------------------------------------------------------------------- #
# PAPER-DECLARED COEFFICIENTS — the second namespace (2026-07-13).
#
# Until the provenance contract carried them, ``beta``/``lambda``/``alpha``/…
# were DELIBERATELY refused here: no producer wrote a field by those names, so
# asserting one resolved to ``provenance_missing`` and would have hard-failed a
# faithful run. That is now fixed at the source — ``emit_provenance(...,
# coefficients={...})`` writes them and the implementer is instructed (from THIS
# rubric's own annotations) to emit every one the contract names. So the
# VOCABULARY gate widens to admit them.
#
# WIDENING THE VOCABULARY IS NOT LOWERING THE BAR. All four original gates still
# apply to a coefficient leaf, unchanged — structure, grounding, vocabulary,
# specificity — and a coefficient must additionally clear TWO gates a plain
# hyperparameter does not:
#
#   5. ROLE      — the leaf text must NAME the symbol it pins (``λ`` or
#                  ``lambda``). Without this, an LLM could hang ``field: "beta"``
#                  on a leaf about a temperature; grounding would not catch it
#                  (the value does occur in the paper) and the implementer would
#                  be told to emit the wrong symbol. This is the coefficient
#                  analogue of the artifact gate's "the file must be named in the
#                  leaf."
#   6. CONTESTED — if the operator's per-paper registry (``configs/papers/<id>.yaml``
#                  → ``paper_invariants.declared_coefficients``) records a value
#                  that DISAGREES with the paper text, the symbol is contested and
#                  no machine check on it is sound. Drop it → LLM. SDAR is exactly
#                  this: the text prints β=10/λ=0.1, the authors' released scripts
#                  use β=5/λ=0.01, and a run faithfully executing the authors' code
#                  would be scored 0.0 by a `beta ~= 10` assertion. See
#                  ``AlgorithmInvariant.coefficients``.
#
# Values are NEVER range-checked. ``0.0`` (an ablation) and ``10`` are equally
# legitimate; a guard that keyed on the ambiguous NAME ``alpha`` rather than its
# role once hard-blocked a faithful ``alpha=0.0`` ablation (learn.md 2026-07-07).
# The only question ever asked is "does it equal what the paper declared HERE?".
# --------------------------------------------------------------------------- #

#: The canonical coefficient vocabulary. Greek letters (papers name algorithmic
#: constants with them) plus the named constants that recur across RL /
#: distillation / decoding methods. Deliberately DISJOINT from
#: _PROVENANCE_HPARAM_FIELDS — a symbol has exactly one home.
_COEFFICIENT_NAMES: frozenset[str] = frozenset({
    # Greek symbols — role is whatever the PAPER declares, never inferred here.
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "rho", "sigma", "tau",
    "upsilon", "phi", "chi", "psi", "omega",
    # Named algorithmic constants.
    "temperature", "top_k", "top_p", "clip_eps", "kl_coef", "entropy_coef",
    "loss_weight", "gate_threshold", "threshold", "margin", "label_smoothing",
    "ema_decay", "discount", "gae_lambda", "kl_target", "reward_scale",
    "group_size", "lora_rank", "lora_alpha", "warmup_ratio", "gradient_clip",
})

#: Spelling variants → the canonical name. SPELLINGS ONLY — never a semantic
#: remap. ``lambda`` is never rewritten to ``loss_weight``: that would be guessing
#: the symbol's ROLE, the precise mistake learn.md 2026-07-07 records. (Greek
#: glyph → ASCII word is handled by ``provenance.canonical_coefficient_name``.)
_COEFFICIENT_ALIASES: dict[str, str] = {
    "topk": "top_k", "top-k": "top_k", "top_k": "top_k",
    "topp": "top_p", "top-p": "top_p", "top_p": "top_p",
    "eps_clip": "clip_eps", "clip_epsilon": "clip_eps", "clip_ratio": "clip_eps",
    "cliprange": "clip_eps", "clip_range": "clip_eps",
    "kl_coefficient": "kl_coef", "kl_penalty_coef": "kl_coef",
    "entropy_coefficient": "entropy_coef", "entropy_bonus": "entropy_coef",
    "discount_factor": "discount",
    "temp": "temperature", "sampling_temperature": "temperature",
    "grad_clip": "gradient_clip", "max_grad_norm": "gradient_clip",
    "label_smoothing_eps": "label_smoothing",
}

#: Paper phrasings → the canonical provenance field name. An LLM-proposed field
#: outside this map is dropped (never coerced by fuzzy match — a near-miss guess
#: is exactly the failure mode we are refusing).
_HPARAM_FIELD_ALIASES: dict[str, str] = {
    "lr": "lr", "learning_rate": "lr", "learningrate": "lr", "step_size": "lr",
    "batch_size": "batch_size", "batchsize": "batch_size",
    "minibatch_size": "batch_size", "mini_batch_size": "batch_size",
    "epochs": "epochs", "num_epochs": "epochs", "n_epochs": "epochs",
    "max_epochs": "epochs", "training_epochs": "epochs",
    "steps": "steps", "num_steps": "steps", "training_steps": "steps",
    "max_steps": "steps",
    "seed": "seed", "random_seed": "seed",
    "weight_decay": "weight_decay", "wd": "weight_decay",
    "momentum": "momentum",
    "dropout": "dropout", "dropout_p": "dropout", "dropout_rate": "dropout",
    "num_classes": "num_classes", "n_classes": "num_classes",
}

#: Leaves whose SUBSTANCE is algorithmic fidelity. A scalar or file-exists check
#: cannot stand in for "implements the sigmoid gate with a stop-gradient" — a stub
#: with an empty train.py would score 1.0 on the paper's core invariant. These
#: always stay with the LLM, which is the whole point of keeping it in the loop.
_ALGORITHMIC_TOKENS: tuple[str, ...] = (
    "implement", "equation", "formula", "derivation", "stop-gradient",
    "stop gradient", "detach", "architecture", "loss function",
    "objective function", "algorithm", "gate", "backbone", "encoder",
    "decoder", "forward pass", "backward pass", "gradient",
)

#: The SPECIFICITY gate for a COEFFICIENT leaf. Same bar as _ALGORITHMIC_TOKENS —
#: "a scalar check must never stand in for an algorithmic claim" — enforced by a
#: sharper instrument, because the blunt one misfires on exactly the leaves this
#: path exists to catch.
#:
#: The problem: _ALGORITHMIC_TOKENS refuses any leaf containing a mechanism NOUN
#: ("gate", "encoder"). But "The gate sharpening coefficient beta is set to 10" is
#: a pure VALUE claim that merely names the mechanism the constant belongs to —
#: naming it is *desirable* (the ROLE gate wants the symbol identified). Refusing
#: it sent SDAR's β=10, the single most important constant in the canonical test
#: paper, back to LLM opinion.
#:
#: The property that actually distinguishes a fidelity leaf is an IMPLEMENTATION
#: CLAIM: a verb ("implements"), a named structure ("loss function"), or a restated
#: FORMULA ("g_t = σ(β·Δ_t)"). So a coefficient leaf is refused when it makes such a
#: claim (tokens below) OR when it restates a formula (_contains_formula) — and the
#: formula test is what closes the hole that dropping the bare nouns would open:
#:
#:   "train.py implements the sigmoid gate ... with beta=10"   -> "implement"   REFUSED
#:   "The gate is computed as g_t = sigma(beta*Delta_t), beta=10" -> formula     REFUSED
#:   "The gate sharpening coefficient beta is set to 10"       -> neither       allowed
#:
#: Net: strictly MORE discriminating than a noun blacklist, not less. The mechanism
#: still gets its own LLM-graded fidelity leaf (the prompt now demands the split),
#: so a stub that hardcodes beta=10 and implements nothing still loses that leaf.
#: The standard-hyperparameter path keeps the original _ALGORITHMIC_TOKENS untouched.
_COEFFICIENT_CLAIM_TOKENS: tuple[str, ...] = (
    "implement", "equation", "formula", "derivation", "stop-gradient",
    "stop gradient", "detach", "loss function", "objective function",
    "algorithm", "forward pass", "backward pass", "correctly", "faithfully",
    "is computed as", "is defined as", "is given by",
)

#: Matches an ``=`` (not ``==`` / ``>=`` / ``<=`` / ``!=``) and captures what follows,
#: so a binding whose right-hand side is NOT a bare number can be recognized as a
#: restated formula rather than a value declaration.
_BINDING_RE = re.compile(r"(?<![=<>!~])=(?!=)\s*(\S+)")

#: A bare numeric right-hand side (sci-notation / sign / trailing punctuation or %).
_RHS_NUMBER_RE = re.compile(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?[%,.;:)]?$")


def _contains_formula(requirements: str) -> bool:
    """True iff the leaf restates a FORMULA (an ``=`` whose RHS is not a number).

    ``"lambda = 0.1"`` binds a value → False.  ``"g_t = sigma(beta * Delta_t)"``
    restates the mechanism → True.  A leaf with no ``=`` at all → False.
    """
    text = _normalize_numeric_text(requirements)
    for m in _BINDING_RE.finditer(text):
        if not _RHS_NUMBER_RE.match(m.group(1)):
            return True
    return False


#: An artifact leaf must actually be ASKING about existence.
_EXISTENCE_TOKENS: tuple[str, ...] = (
    "exist", "is produced", "are produced", "is written", "are written",
    "is saved", "are saved", "is emitted", "are emitted", "is generated",
    "are generated", "produces", "writes", "saves", "emits", "contains",
    "includes", "is present", "are present",
)

#: Metric polarity. Used to pick higher_better vs lower_better; a metric matching
#: BOTH lists or NEITHER is unclassifiable → no annotation.
_HIGHER_BETTER_TOKENS: tuple[str, ...] = (
    "accuracy", "acc", "success rate", "success_rate", "f1", "exact match",
    "em", "score", "reward", "win rate", "win_rate", "auc", "bleu", "rouge",
    "recall", "precision", "pass@", "top-1", "top1", "top-5", "top5", "mrr",
)
_LOWER_BETTER_TOKENS: tuple[str, ...] = (
    "loss", "error", "perplexity", "ppl", "rmse", "mae", "mse", "fid",
    "wer", "cer", "regret",
)

#: Relative tolerance when matching an asserted value against a paper number.
#: Tight — this is an identity check ("did the paper really say 1e-4?"), not a
#: similarity check.
_GROUNDING_REL_TOL = 1e-6

#: Result-match tolerance for a numeric leaf. Mirrors
#: ``rubric_contract._RESULT_MATCH_TOLERANCE`` (10% relative): a faithful
#: reproduction lands near, not exactly on, the paper's number, and the checker
#: grades trend/threshold satisfaction rather than exact magnitude.
_NUMERIC_REL_TOLERANCE = 0.10

_UNICODE_MINUS = "−"

#: Cross-references whose numbers are NOT pinned quantities ("Section 4.1",
#: "Table 2"). Stripped before counting how many numbers a leaf pins.
_XREF_RE = re.compile(
    r"\b(?:section|sec|table|tab|figure|fig|appendix|app|equation|eq|algorithm|alg)"
    r"\.?\s*[a-z]?\.?\s*\d+(?:\.\d+)*",
    re.IGNORECASE,
)

#: A standalone number — not glued to letters, so model names ("Qwen2.5-7B") and
#: identifiers do not read as pinned hyperparameters.
_STANDALONE_NUM_RE = re.compile(
    r"(?<![A-Za-z\d.\-])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?![A-Za-z\d])"
)


# ---------------------------------------------------------------------------
# System prompt — instructs the LLM to produce the six-category rubric JSON.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a research-reproduction rubric author for ReproLab.

You are given the full text of a research paper. Produce a PaperBench-style
weighted rubric that a grader will use to score an attempted reproduction of
that paper. The rubric grades only concrete reproduction artifacts — source
code, the environment, executed runs, produced metrics and plots — never
process, effort, or how the reproduction was carried out.

BEFORE writing any leaf, mentally extract from the paper text:
  • Every named algorithm, method variant, and baseline (e.g. "GRPO", "OPSD",
    "SDAR", "Skill-SD") — use their exact names in leaf text.
  • Every equation-level detail the code must implement (e.g. "g_t = σ(β·Δ_t)",
    "stop-gradient on the gate", "token-level KL divergence").
  • Every exact numeric hyperparameter and its value (e.g. "β=10", "λ=0.1",
    "learning rate 1e-4", "batch size 64", "hidden size 256").
  • Every exact model name (e.g. "Qwen2.5-7B-Instruct", "Qwen3-1.7B") and
    dataset name (e.g. "ALFWorld", "WebShop", "Search-QA").
  • Every reported numeric result, BOUND TO THE EXACT CONFIGURATION THAT
    PRODUCED IT — the (model/architecture, dataset, augmentation/setting) triple
    (e.g. "WRN-28-10 + Cutout on CIFAR-10 = 3.08%", not a bare "3.08%"). Results
    tables have ONE ROW PER ARCHITECTURE; the headline/best number in a paper's
    title or abstract usually belongs to a DIFFERENT (often larger/stronger) model
    than the one being reproduced. A "Result match" leaf MUST name the exact model
    + dataset it targets and use THAT row's value — NEVER carry a number from one
    architecture's row into a result-match leaf for a different architecture.
Use only specifics found in the paper text; do NOT invent values.

Organize the rubric under these six categories. The weight of each category
should fall in the range shown (weights are relative — they need not sum to
exactly 1):

  Method and code fidelity to the paper             0.30 - 0.45
  Data and preprocessing fidelity                   0.10 - 0.20
  Experiment execution and reproducibility          0.15 - 0.25
  Evaluation protocol and metric correctness        0.15 - 0.25
  Result match versus the paper's reported targets  0.15 - 0.30
  Artifact completeness and provenance              0.05 - 0.10

For each category write 2 to 5 leaf criteria. Each leaf MUST:
  1. Name the EXACT paper-specific item it checks (algorithm, equation,
     hyperparameter value, model name, dataset name, or numeric result).
  2. Quote the section number where it is described (e.g. "Section 3.2").
  3. Be independently checkable from artifacts alone.

STRICT PROHIBITION — a leaf requirement string must NEVER contain:
  • Empty parentheses or placeholders: "(, )", "( )", "(β, λ)", "(, λ)",
    "(α, β, learning rate, etc.)" or any other unfilled template.
  • Vague phrases with no values: "the hyperparameters are correctly set",
    "the model is implemented correctly", "the training follows the paper".
  Every such leaf is INVALID and must be rewritten with the actual values
  extracted from the paper before you produce the JSON.

GOOD leaf examples (these show the required level of specificity):
  "train.py implements the sigmoid gate g_t = σ(β·Δ_t) with β=10 and a
   stop-gradient applied to the gate, as described in Section 3.3."
  "The GRPO and OPSD baselines are re-implemented with the same Qwen2.5-7B-
   Instruct backbone as the proposed SDAR model (Section 4.1)."
  "Sets λ=0.1 for the self-distillation loss weight and batch size 64 in
   train.py, matching Section 4.1 Table 2 hyper-parameters."
  "train.py implements the two-layer bidirectional GRU encoder with
   hidden size 256 described in Section 3.1."
  "WRN-28-10 with Cutout (length 16) reaches 3.08% test error on CIFAR-10
   (Table 1, WRN-28-10 row) — NOT the 2.56% Shake-Shake row, which is a
   different architecture."

WEAK leaf examples (NEVER produce these):
  "The model is implemented correctly."
  "The hyperparameters (, ) are correctly set as described in Section 4.1."
  "Training follows the methodology described in the paper."

Give every leaf a relative weight within its category.

Return ONLY this JSON object and nothing else:

{
  "categories": [
    {
      "name": "Method and code fidelity to the paper",
      "weight": 0.40,
      "leaves": [
        {"requirements": "<concrete paper-specific criterion with exact values>", "weight": 0.3}
      ]
    }
  ]
}"""


# ---------------------------------------------------------------------------
# Optional prompt suffix — asks for the machine-checkable annotation.
# Appended ONLY when OPENRESEARCH_DETERMINISTIC_LEAVES is on, so an OFF run sends
# the byte-identical prompt it sends today (a changed prompt is a changed rubric,
# even if no annotation is emitted).
#
# The model PROPOSES; Python DISPOSES. Every field below is re-derived, grounded
# against the paper text, and whitelisted on our side before it can reach disk —
# so a wrong or invented proposal costs a dropped annotation, never a wrong grade.
# ---------------------------------------------------------------------------
_CHECK_ANNOTATION_PROMPT = """

ONE HYPERPARAMETER PER LEAF (atomicity):

Write hyperparameter leaves ATOMICALLY — one leaf per hyperparameter. Do NOT pack
several settings into one leaf. Instead of

  BAD:  "Training uses mini-batch size 256, learning rate 0.1, weight decay 0.0001
         and momentum 0.9 (Section 3.4)."

emit four separate leaves, each pinning exactly one value:

  GOOD: "Training uses a mini-batch size of 256 (Section 3.4)."
        "Training uses an initial learning rate of 0.1 (Section 3.4)."
        "Training uses a weight decay of 0.0001 (Section 3.4)."
        "Training uses a momentum of 0.9 (Section 3.4)."

A packed leaf is all-or-nothing: it cannot distinguish a run that got the batch
size right and the weight decay wrong, so it is graded by a coarse LLM judgment
instead of an exact check. Splitting costs nothing and each atomic leaf is then
verified exactly. Give each split leaf a proportionally smaller weight so the
category's total weight is unchanged. This applies to hyperparameter leaves only —
do NOT split a leaf whose substance is one algorithmic idea.

SPLIT THE MECHANISM FROM ITS CONSTANT (this is the important one):

A paper's algorithmic coefficients — β, λ, α, τ, a temperature, a clipping ε — are
BOTH an algorithmic idea AND an exact number. Write them as TWO leaves, never one:

  BAD:  "train.py implements the sigmoid gate g_t = sigma(beta * Delta_t) with
         beta=10 and a stop-gradient on the gate (Section 3.3)."

  GOOD: "train.py implements the sigmoid gate g_t = sigma(beta * Delta_t) with a
         stop-gradient applied to the gate (Section 3.3)."      <- no "check"
        "The gate sharpening coefficient beta is set to 10 (Section 3.3)."
                                                                <- "check" on beta

The first leaf is a fidelity judgment and must stay with the human-like grader: a
stub that merely hardcodes beta=10 while implementing nothing would pass any check
on the number alone. The second is a pure value claim and is verified exactly. Write
the value leaf as a plain statement of the constant — name the symbol, give the
number, and do NOT restate how the mechanism works, or it becomes a judgment leaf
again and the check is discarded.

MACHINE-CHECKABLE ANNOTATION (optional per leaf, key "check"):

Some leaves can be graded by a program instead of a human reader. For those ONLY,
add a "check" object to the leaf. A leaf with no "check" key is graded by an LLM,
which is the correct and expected outcome for most leaves.

Emit "check" in exactly one of these three shapes:

  {"kind": "deterministic:hparam",   "field": "<a field name, see below>",
      "value": <number>}
    Use ONLY when the leaf pins exactly ONE named quantity to ONE number stated in
    the paper. Two families of field name are recorded by the run's provenance
    manifest, and ONLY these:
      (a) training hyperparameters —
          lr, batch_size, epochs, steps, seed, weight_decay, momentum, dropout,
          num_classes
      (b) paper-declared algorithmic coefficients — the constants that define the
          METHOD, recorded under their own name exactly as the paper writes them:
          alpha, beta, gamma, delta, epsilon, zeta, eta, theta, kappa, lambda, mu,
          nu, xi, rho, sigma, tau, phi, chi, psi, omega, temperature, top_k, top_p,
          clip_eps, kl_coef, entropy_coef, loss_weight, gate_threshold, threshold,
          margin, label_smoothing, ema_decay, discount, gae_lambda, kl_target,
          reward_scale, group_size, lora_rank, lora_alpha, warmup_ratio,
          gradient_clip
    For (b) use the SYMBOL the paper uses ("beta" for a paper's β), and make sure
    the leaf text itself names that symbol.

  {"kind": "deterministic:artifact", "glob": "<filename or glob>"}
    Use ONLY when the leaf asks whether a FILE EXISTS (a checkpoint, a metrics
    file, a produced figure). The filename must appear in the leaf's own text.

  {"kind": "deterministic:numeric",  "metric_key": "<metric name>", "target": <number>}
    Use ONLY when the leaf compares a produced metric against a number the paper
    REPORTS as a result. The metric must be one whose direction is obvious
    (accuracy/F1/reward = higher is better; loss/error/perplexity = lower).

HARD RULES — a violated rule makes the annotation useless or harmful:
  • OMIT "check" WHENEVER YOU ARE NOT CERTAIN. A missing annotation costs nothing
    (an LLM grades the leaf). A WRONG annotation deterministically FAILS a correct
    reproduction. When in doubt, leave it out.
  • NEVER invent a value. Every "value"/"target" must be a number printed in the
    paper text you were given. If the paper does not state it, omit "check".
  • NEVER annotate a JUDGMENT leaf — anything about whether the method is
    correctly/faithfully implemented, an equation, an architecture, a loss
    function, or code quality. Those MUST be LLM-graded. A file-existence check
    can never stand in for "implements the algorithm correctly", and a check on a
    coefficient's VALUE can never stand in for "implements the mechanism that
    coefficient belongs to".
  • NEVER annotate a leaf that pins two or more numbers (e.g. "lambda=0.1 and
    batch size 64"). One check cannot verify both, and a partial check would
    credit a run that got the other one wrong. Split it, per the rules above.
  • The leaf must NAME the field it checks. A "check" on "beta" belongs only on a
    leaf that actually mentions beta (or β)."""


def _build_system_prompt() -> str:
    """The rubric-author prompt; the annotation section is flag-gated."""
    if deterministic_leaves_enabled():
        return _SYSTEM_PROMPT + _CHECK_ANNOTATION_PROMPT
    return _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_rubric_tree(
    paper_text: str,
    llm_client: LlmClient,
    *,
    paper_title: str = "",
    max_attempts: int = 3,
    max_paper_chars: int = 48000,
    project_dir: Path | None = None,
    emit_warning: Any | None = None,
) -> dict | None:
    """Derive a PaperBench-shaped rubric tree from a paper's full text.

    Returns a rubric dict compatible with ``flatten_leaves`` / ``roll_up``, or
    ``None`` if the paper is too short to derive a rubric from, or if all LLM
    attempts fail (honest degradation — the run proceeds rubric-less).

    ``project_dir``/``emit_warning`` are optional and ONLY feed the advisory,
    flag-gated literature claim-grounding hook (``OPENRESEARCH_LITERATURE_CLAIM_GATE``,
    see :mod:`backend.agents.rlm.literature_claim_gate`) run just before
    returning a successfully-built tree — they never affect the tree itself.
    """
    if len(paper_text.strip()) < 500:
        logger.warning(
            "generate_rubric_tree: paper text too short (%d chars) — skipping rubric generation",
            len(paper_text.strip()),
        )
        return None

    user_msg = (
        f"Paper title: {paper_title}\n\nPaper text:\n\n{paper_text[:max_paper_chars]}"
    )

    last_error: str = "unknown"
    for attempt in range(1, max_attempts + 1):
        try:
            raw = llm_client.complete(system=_build_system_prompt(), user=user_msg)
        except Exception as exc:
            last_error = f"LLM exception on attempt {attempt}: {exc}"
            logger.warning("generate_rubric_tree: %s", last_error)
            if attempt < max_attempts:
                time.sleep(min(2 ** attempt, 30))
            continue

        parsed = _extract_json_object(raw)
        if parsed is None:
            last_error = f"unparseable JSON on attempt {attempt}"
            logger.warning("generate_rubric_tree: %s", last_error)
            continue

        categories = _clean_categories(parsed.get("categories") or [])
        if not categories or sum(len(c["leaves"]) for c in categories) == 0:
            last_error = f"empty categories/leaves after cleaning on attempt {attempt}"
            logger.warning("generate_rubric_tree: %s", last_error)
            continue

        tree = _build_tree(
            categories,
            paper_title,
            paper_text=paper_text,
            contested=_contested_coefficients(project_dir),
        )
        leaf_count = sum(len(c["leaves"]) for c in categories)
        logger.info(
            "generate_rubric_tree: built rubric — %d leaves across %d categories",
            leaf_count,
            len(categories),
        )
        if deterministic_leaves_enabled():
            # Pre-persist validation: a malformed annotation is dropped here, never
            # written into generated_rubric.json (the caller json.dumps this tree
            # straight to disk). A dropped leaf simply falls through to the LLM.
            strip_invalid_annotations(tree)
            coverage = annotation_coverage(tree)
            logger.info(
                "generate_rubric_tree: deterministic-leaf coverage — %d/%d leaves "
                "(%.0f%%) graded by the pure-Python checker, %d by the LLM; by kind: %s",
                coverage["deterministic"], coverage["total"],
                100.0 * coverage["fraction"], coverage["llm"], coverage["by_kind"] or "{}",
            )
        if _rubric_canary_enabled():
            # NB the canary leaf is appended AFTER annotation and carries no
            # check_kind — deliberately. It must stay LLM-graded: its whole purpose
            # is to catch a grader that credits an artifact honest work cannot
            # produce, and a deterministic 0.0 would neutralize the trap.
            append_canary_leaf(tree)
        _apply_literature_claim_gate(paper_text, project_dir, emit_warning)
        return tree

    logger.warning(
        "generate_rubric_tree: all %d attempts failed — last: %s",
        max_attempts,
        last_error,
    )
    return None


def _apply_literature_claim_gate(
    paper_text: str,
    project_dir: Path | None,
    emit_warning: Any | None,
) -> None:
    """Advisory-only rubric-INPUT grounding hook (``OPENRESEARCH_LITERATURE_CLAIM_GATE``).

    Fail-soft: any import/network/logic failure is swallowed here (on top of
    ``run_literature_claim_gate``'s own internal fail-soft handling) — this
    NEVER changes the rubric tree, only (optionally) emits a ``run_warning``
    through the caller-supplied ``emit_warning`` callable. Byte-identical to
    a no-op call when the flag is unset (the callee returns immediately).
    """
    try:
        from backend.agents.rlm.literature_claim_gate import run_literature_claim_gate

        run_literature_claim_gate(paper_text, project_dir=project_dir, emit_warning=emit_warning)
    except Exception:  # noqa: BLE001 — fail-soft, never break rubric generation
        logger.debug("generate_rubric_tree: literature claim gate skipped due to exception", exc_info=True)


def _rubric_canary_enabled() -> bool:
    return os.environ.get(RUBRIC_CANARY_ENV, "").strip().lower() in _TRUTHY


def append_canary_leaf(tree: dict) -> dict:
    """Append one weight-0 canary leaf to the tree's first category (spec
    §10.4). The leaf references an artifact + metric that honest
    reproduction work cannot produce; any grader credit > 0 on
    :data:`CANARY_LEAF_ID` is a campaign fabrication signal (see
    :func:`canary_tripped`). ``weight=0.0`` keeps it OUT of the score
    denominator -- ``roll_up`` weighted-averages by weight, so a 0-weight
    leaf never moves the score regardless of how it is graded.

    Returns the SAME ``tree`` object, mutated in place. A tree with no
    categories (``sub_tasks`` missing, not a list, or empty) is returned
    unchanged -- there is no "first category" to attach to.
    """
    categories = tree.get("sub_tasks")
    if not isinstance(categories, list) or not categories:
        return tree
    first_category = categories[0]
    if not isinstance(first_category, dict):
        return tree

    leaf = {
        "id": CANARY_LEAF_ID,
        "requirements": _CANARY_REQUIREMENTS,
        "weight": 0.0,
        "task_category": first_category.get("requirements"),
        "finegrained_task_category": None,
        "sub_tasks": [],
    }
    sub_tasks = first_category.get("sub_tasks")
    if not isinstance(sub_tasks, list):
        sub_tasks = []
        first_category["sub_tasks"] = sub_tasks
    sub_tasks.append(leaf)
    return tree


def canary_tripped(rubric_evaluation: Mapping) -> bool:
    """True iff a ``rubric_evaluation``-shaped payload (``leaf_scores``: a
    list of ``{"id": ..., "score": ...}`` records, per
    ``leaf_scorer.score_reproduction``) scored :data:`CANARY_LEAF_ID` with
    ``score > 0``. Any positive credit at all is a fabrication signal,
    since the canary references an artifact/metric honest work cannot
    produce (spec §10.4).

    ASSESS wiring of this signal into ``guard_flags`` is a documented
    follow-up, NOT part of this unit -- this helper ships pure and tested
    so that wiring is a one-line addition later.
    """
    leaf_scores = rubric_evaluation.get("leaf_scores") if isinstance(rubric_evaluation, Mapping) else None
    if not isinstance(leaf_scores, list):
        return False
    for entry in leaf_scores:
        if not isinstance(entry, dict) or entry.get("id") != CANARY_LEAF_ID:
            continue
        score = entry.get("score")
        if isinstance(score, (int, float)) and score > 0:
            return True
    return False


# ---------------------------------------------------------------------------
# Deterministic-leaf annotation — grounding + validation
#
# Every function here is pure, deterministic, and fail-soft. Each returns
# "no annotation" on ANY doubt; none can raise into rubric generation.
# ---------------------------------------------------------------------------


def deterministic_leaves_enabled() -> bool:
    """Is the deterministic-leaf feature on? (Same flag the router reads.)"""
    return os.environ.get(DETERMINISTIC_LEAVES_ENV, "").strip().lower() in _TRUTHY


def _contested_coefficients(project_dir: Path | None) -> dict[str, float]:
    """Operator-declared coefficient values for this run's paper (``{}`` if none).

    Feeds the CONTESTED gate. Fail-soft in every direction: no project_dir, no
    arXiv id, no ``configs/papers/<id>.yaml``, or an import error all yield ``{}``,
    which means "no veto" — the paper-text-grounded annotation stands unchanged.
    """
    try:
        from backend.agents.rlm.paper_invariants import declared_coefficients

        return declared_coefficients(project_dir)
    except Exception:  # noqa: BLE001 — a veto lookup must never break rubric-gen.
        logger.debug("rubric_gen: contested-coefficient lookup failed", exc_info=True)
        return {}


def coefficient_fields(tree: Any) -> dict[str, float]:
    """The paper-declared coefficients THIS rubric will machine-check.

    ``{"beta": 10.0, "lambda": 0.1}`` — read straight off the persisted tree's
    ``coefficients.*`` assertions. This is the DETERMINISTIC source for the
    implementer's emit-contract: ``baseline_implementation`` calls it on
    ``generated_rubric.json`` (written before the RLM loop starts, so it is always
    on disk by the time ``implement_baseline`` runs) and instructs the agent to emit
    exactly these names. The implementer therefore never improvises the list — the
    set of coefficients it is told to record is, by construction, the set it will be
    graded on. Fail-soft: any malformed tree yields ``{}``.
    """
    out: dict[str, float] = {}

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("check_kind") == CHECK_HPARAM:
            assertion = node.get("assertion")
            if isinstance(assertion, dict):
                field = assertion.get("field")
                value = assertion.get("value")
                if (
                    isinstance(field, str)
                    and field.startswith(f"{COEFFICIENTS_KEY}.")
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    out[field.split(".", 1)[1]] = float(value)
        for child in node.get("sub_tasks") or []:
            _walk(child)

    try:
        _walk(tree)
    except Exception:  # noqa: BLE001 — fail-soft; an unreadable tree yields no contract.
        logger.debug("rubric_gen: coefficient_fields walk failed", exc_info=True)
    return out


def _normalize_numeric_text(text: str) -> str:
    """Rewrite LaTeX/unicode scientific notation into plain ``e`` notation.

    A paper writes ``1 × 10^{-4}`` / ``1 \\times 10^-4`` / ``10^{-4}``; a rubric
    asserts ``0.0001``. Without this, grounding would reject a value the paper
    plainly states — and a rejected grounding means a leaf silently loses its
    deterministic check. Normalizing costs nothing and only ever ADDS matches.
    """
    t = text.replace(_UNICODE_MINUS, "-")
    # "1.0 × 10^{-4}" / "1.0 \times 10^-4" / "3 * 10^4"  →  "1.0e-4" / "3e4"
    t = re.sub(
        r"(\d(?:\.\d+)?)\s*(?:×|x|\*|\\times|\\cdot|·)\s*10\s*\^?\s*\{?\s*(-?\d+)\s*\}?",
        r"\1e\2", t, flags=re.IGNORECASE,
    )
    # standalone "10^{-4}"  →  "1e-4"
    t = re.sub(r"(?<![\d.])10\s*\^\s*\{?\s*(-?\d+)\s*\}?", r"1e\1", t)
    return t


def _numbers_in(text: str) -> list[float]:
    """Every standalone numeric literal in ``text`` (sci-notation normalized)."""
    out: list[float] = []
    for m in _STANDALONE_NUM_RE.finditer(_normalize_numeric_text(text)):
        try:
            out.append(float(m.group()))
        except ValueError:
            continue
    return out


def _numbers_close(a: float, b: float) -> bool:
    """Identity-grade numeric equality (relative, so 1e-4 == 0.0001)."""
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return abs(a - b) < 1e-12
    return abs(a - b) / scale <= _GROUNDING_REL_TOL


def _pinned_numbers(requirements: str) -> list[float]:
    """Distinct numeric quantities a leaf PINS, ignoring cross-references.

    "Sets λ=0.1 for the loss weight and batch size 64 (Section 4.1, Table 2)"
    pins {0.1, 64} — two quantities — so no single-field assertion can check it
    without over-crediting a run that got one right and the other wrong. The
    Section/Table numbers are references, not quantities, and are stripped first.
    """
    stripped = _XREF_RE.sub(" ", requirements)
    seen: list[float] = []
    for n in _numbers_in(stripped):
        if not any(_numbers_close(n, s) for s in seen):
            seen.append(n)
    return seen


def _has_token(text: str, tokens: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(tok in low for tok in tokens)


def _value_grounded(value: Any, text: str, *, allow_percent_scaling: bool = False) -> bool:
    """Is ``value`` actually stated in ``text``?

    Numeric values are matched against every numeric literal in the text (so
    ``1e-4`` grounds against ``0.0001``); strings by case-insensitive substring.
    ``allow_percent_scaling`` additionally accepts the ×100 / ÷100 forms, because
    a paper reporting "72.3%" may be asserted as either ``72.3`` or ``0.723``.

    NOTE the honest limit: this proves the value OCCURS in the paper, not that it
    occurs in the role the leaf claims. It is a necessary, not sufficient,
    condition — its job is to kill values the paper never mentions at all (the
    hallucination case). Specificity gates below carry the rest.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        candidates = [float(value)]
        if allow_percent_scaling:
            candidates += [float(value) * 100.0, float(value) / 100.0]
        haystack = _numbers_in(text)
        return any(
            _numbers_close(c, n) for c in candidates for n in haystack
        )
    if isinstance(value, str):
        v = value.strip().lower()
        return bool(v) and v in text.lower()
    return False


def _is_known_hparam_field(field: str) -> bool:
    """Is ``field`` addressable in provenance? The two namespaces, and only those.

    A bare name must be a bookkeeping hyperparameter a producer writes; a dotted
    name must be ``coefficients.<known symbol>``. Fail-closed on anything else, so
    a hallucinated field can never be persisted into ``generated_rubric.json`` —
    which is hashed, pinned across a campaign, AND (now) the source the implementer
    reads to learn which coefficients to emit. Garbage here would propagate.
    """
    if field in _PROVENANCE_HPARAM_FIELDS:
        return True
    namespace, dot, name = field.partition(".")
    return bool(dot) and namespace == COEFFICIENTS_KEY and name in _COEFFICIENT_NAMES


def _validate_annotation(kind: Any, assertion: Any) -> bool:
    """Structural gate — does this annotation match the checker's contract?

    Run immediately before the tree is returned (and therefore before
    ``generated_rubric.json`` is written), so a malformed annotation is DROPPED
    rather than persisted. A malformed annotation can never hard-fail a good run:
    the checker would route it to the LLM anyway, but persisting garbage into the
    rubric — which is hashed, pinned across a campaign, and shown to the
    implementer — is its own bug. Fail-closed on anything unrecognized.
    """
    if kind not in DETERMINISTIC_CHECK_KINDS:
        return False
    if not isinstance(assertion, dict) or not assertion:
        return False

    if kind == CHECK_HPARAM:
        field = assertion.get("field")
        if not isinstance(field, str) or not _is_known_hparam_field(field):
            return False
        if assertion.get("op") not in ("==", "~=", ">=", "<=", "!="):
            return False
        if isinstance(assertion.get("value"), bool):
            return False
        return isinstance(assertion.get("value"), (int, float, str))

    if kind == CHECK_ARTIFACT:
        globs = assertion.get("glob")
        if isinstance(globs, str):
            globs = [globs]
        if not isinstance(globs, list) or not globs:
            return False
        return all(
            isinstance(g, str) and g.strip() and ".." not in g and not g.startswith("/")
            for g in globs
        )

    if kind == CHECK_NUMERIC:
        if not isinstance(assertion.get("metric_key"), str) or not assertion["metric_key"]:
            return False
        if assertion.get("direction") not in ("higher_better", "lower_better", "within"):
            return False
        target = assertion.get("target")
        return isinstance(target, (int, float)) and not isinstance(target, bool)

    return False


def _canonical_coefficient(raw_field: str) -> str | None:
    """Canonical coefficient name for an LLM-proposed field, or ``None``.

    Runs the proposal through the manifest's OWN normalizer (so ``β`` and
    ``\\lambda`` land on the same key the producer writes), applies the
    spelling-alias map, then enforces the whitelist. A field outside the
    vocabulary is dropped — never fuzzy-matched onto a near neighbour, since a
    near-miss guess is exactly the failure mode this path refuses.
    """
    from backend.agents.rlm.provenance import canonical_coefficient_name

    name = canonical_coefficient_name(raw_field)
    if not name:
        return None
    name = _COEFFICIENT_ALIASES.get(name, name)
    return name if name in _COEFFICIENT_NAMES else None


def _coefficient_surface_forms(name: str) -> tuple[str, ...]:
    """Every spelling of ``name`` a paper/leaf might print (for the ROLE gate)."""
    from backend.agents.rlm.provenance import _GREEK_TO_ASCII

    forms = {name, name.replace("_", "-"), name.replace("_", " ")}
    # the Greek glyph(s) whose ASCII word is this name — papers print "λ", not "lambda".
    forms.update(glyph for glyph, word in _GREEK_TO_ASCII.items() if word == name)
    # spelling aliases that normalize onto this name ("eps_clip" for "clip_eps").
    forms.update(alias for alias, canon in _COEFFICIENT_ALIASES.items() if canon == name)
    return tuple(sorted(forms))


def _coefficient_named_in(requirements: str, name: str) -> bool:
    """ROLE gate: does the leaf actually NAME the symbol it claims to pin?

    An LLM can hang ``field: "beta"`` on a leaf that is really about a
    temperature. GROUNDING would not catch it (the value does occur in the paper),
    so the assertion would compare the right number against the wrong symbol AND
    push the wrong symbol into the implementer's emit-contract. Requiring the leaf
    to print the symbol — ``λ``, ``lambda``, ``top-k`` — is cheap and closes it.

    Mirrors the artifact gate's "the filename must appear in the leaf text" rule.
    ASCII forms need a word boundary so ``beta`` does not match inside ``betas``
    (Adam's tuple); a Greek glyph is matched as a plain substring.
    """
    low = requirements.lower()
    for form in _coefficient_surface_forms(name):
        if form.isascii():
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(form)}(?![A-Za-z0-9_])", low):
                return True
        elif form in low:  # a Greek glyph — no ASCII word boundary applies.
            return True
    return False


def _annotate_coefficient(
    raw_field: str,
    requirements: str,
    raw: dict,
    paper_text: str,
    contested: Mapping | None,
) -> dict | None:
    """Build a grounded ``deterministic:hparam`` assertion on ``coefficients.<name>``.

    Returns ``None`` — leaf stays LLM-graded — on ANY doubt. Six gates, every one
    one-directional (each can only ever REFUSE): vocabulary, value-shape,
    specificity, ROLE, CONTESTED, grounding.
    """
    # 1. VOCABULARY — a known algorithmic-coefficient symbol, or nothing.
    name = _canonical_coefficient(raw_field)
    if name is None:
        logger.debug(
            "rubric_gen: dropping annotation — field %r is neither a provenance "
            "hyperparameter nor a known paper coefficient; leaf → LLM", raw_field,
        )
        return None

    # 2. VALUE SHAPE — only a real number is mechanically comparable. NB `0.0` is a
    #    perfectly legitimate declared value (an ablation): test the TYPE, never the
    #    truthiness, and never the range.
    value = raw.get("value")
    if isinstance(value, str):
        coerced = _numbers_in(value)
        value = coerced[0] if len(coerced) == 1 else value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    # 3. SPECIFICITY — same bar, sharper instrument (see _COEFFICIENT_CLAIM_TOKENS).
    #    A leaf that makes an IMPLEMENTATION CLAIM ("implements the sigmoid gate with
    #    a stop-gradient") or restates a FORMULA ("g_t = sigma(beta*Delta_t)") must
    #    NOT collapse to `beta == 10`: a stub that hardcodes beta=10 and implements
    #    nothing would score 1.0 on the paper's core invariant. Those stay with the
    #    LLM — that is the whole point of keeping it in the loop. The prompt demands a
    #    SEPARATE atomic value leaf ("the gate sharpening coefficient beta is 10"),
    #    and it is that leaf which lands here.
    if _has_token(requirements, _COEFFICIENT_CLAIM_TOKENS) or _contains_formula(requirements):
        return None
    pinned = _pinned_numbers(requirements)
    if len(pinned) != 1 or not _numbers_close(float(value), pinned[0]):
        return None

    # 4. ROLE — the leaf must name the symbol (see _coefficient_named_in).
    if not _coefficient_named_in(requirements, name):
        logger.debug(
            "rubric_gen: dropping coefficient annotation %s=%r — the leaf never names "
            "the symbol, so the field may not be the one this leaf is about; leaf → LLM",
            name, value,
        )
        return None

    # 5. CONTESTED — the operator's per-paper registry disagrees with the paper text
    #    about this symbol (SDAR: text β=10, authors' released scripts β=5). Then no
    #    machine check is sound: a run faithfully executing the authors' code emits
    #    the other value and would be graded 0.0. Hand it to the LLM, which can read
    #    both the code and the paper.
    if isinstance(contested, Mapping) and name in contested:
        declared = contested.get(name)
        if isinstance(declared, (int, float)) and not isinstance(declared, bool):
            if not _numbers_close(float(value), float(declared)):
                logger.warning(
                    "rubric_gen: dropping CONTESTED coefficient annotation %s — paper "
                    "text says %r but configs/papers/<id>.yaml declares %r (e.g. the "
                    "authors' released scripts). A deterministic check would fail a run "
                    "that faithfully reproduced either one; leaf → LLM.",
                    name, value, declared,
                )
                return None

    # 6. GROUNDING — the value must be stated in the paper, not merely asserted.
    if not _value_grounded(value, paper_text):
        logger.warning(
            "rubric_gen: dropping UNGROUNDED coefficient annotation %s=%r — value does "
            "not occur in the paper text; leaf → LLM", name, value,
        )
        return None

    from backend.agents.rlm.provenance import coefficient_field

    return {
        "field": coefficient_field(name),
        # One rule for every coefficient, int or float: a declared constant is an
        # IDENTITY check under the checker's existing `~=` tolerance semantics.
        # β=10 agrees with 10.0 (JSON round-trip / int-vs-float); β=9.999 is a
        # DIFFERENT constant and β=1.0 is a surrogate — both fail, deterministically.
        # No new tolerance semantics are invented; this is the same formula the
        # float branch of _annotate_hparam already uses.
        "op": "~=",
        "value": float(value),
        "tolerance": max(abs(float(value)) * 1e-6, 1e-12),
        # THE FALSE-NEGATIVE VALVE — the load-bearing knob. emit_provenance is
        # fail-soft and OPTIONAL, so a faithful run may simply carry no manifest, or
        # a manifest without this symbol. That must route to the LLM (which can read
        # `beta = 10` straight out of train.py), NEVER grade 0.0. Only a coefficient
        # that is LOCATED and WRONG fails deterministically — which is the point.
        "on_missing": "llm",
    }


def _annotate_hparam(
    requirements: str,
    raw: dict,
    paper_text: str,
    contested: Mapping | None = None,
) -> dict | None:
    """Build a grounded ``deterministic:hparam`` assertion, or ``None``.

    Two namespaces, one entry point: a BOOKKEEPING hyperparameter
    (``lr``/``epochs``/``batch_size``…) asserts a bare field; a paper-declared
    algorithmic COEFFICIENT (β, λ, τ…) asserts ``coefficients.<name>``. A field in
    neither vocabulary is refused and the leaf is graded by the LLM.
    """
    # VOCABULARY, namespace 1 — a field the provenance producers already write.
    raw_field = str(raw.get("field", "")).strip().lower().replace(" ", "_")
    field = _HPARAM_FIELD_ALIASES.get(raw_field)
    if field is None or field not in _PROVENANCE_HPARAM_FIELDS:
        # VOCABULARY, namespace 2 — a paper-declared coefficient.
        return _annotate_coefficient(raw_field, requirements, raw, paper_text, contested)

    value = raw.get("value")
    if isinstance(value, str):
        coerced = _numbers_in(value)
        value = coerced[0] if len(coerced) == 1 else value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None  # only numeric hyperparameters are mechanically checkable.

    # SPECIFICITY — a leaf whose substance is algorithmic fidelity, or which pins
    # more than one quantity, cannot be reduced to one scalar compare.
    if _has_token(requirements, _ALGORITHMIC_TOKENS):
        return None
    pinned = _pinned_numbers(requirements)
    if len(pinned) != 1 or not _numbers_close(float(value), pinned[0]):
        return None

    # GROUNDING — the value must be stated in the paper itself, not just asserted.
    if not _value_grounded(value, paper_text):
        logger.warning(
            "rubric_gen: dropping UNGROUNDED hparam annotation %s=%r — value does not "
            "occur in the paper text; leaf → LLM (a guessed assertion would "
            "deterministically fail a correct reproduction)",
            field, value,
        )
        return None

    # An integer-valued knob compares exactly; a float carries a relative tolerance
    # so 1e-4 vs 0.0001 vs 0.00010000001 all agree.
    if float(value).is_integer() and field in ("epochs", "steps", "batch_size", "seed", "num_classes"):
        assertion: dict[str, Any] = {"field": field, "op": "==", "value": int(value)}
    else:
        assertion = {
            "field": field,
            "op": "~=",
            "value": float(value),
            "tolerance": max(abs(float(value)) * 1e-6, 1e-12),
        }
    # THE FALSE-NEGATIVE VALVE. emit_provenance is fail-soft and optional, so a
    # faithful run may simply have no manifest — that must route to the LLM (which
    # can read lr=1e-4 out of train.py), never grade 0.0.
    assertion["on_missing"] = "llm"
    return assertion


def _annotate_artifact(requirements: str, raw: dict) -> dict | None:
    """Build a grounded ``deterministic:artifact`` assertion, or ``None``."""
    # SPECIFICITY — never let a file-exists check stand in for a fidelity claim.
    # "train.py implements the sigmoid gate" must NOT become "does train.py exist".
    if _has_token(requirements, _ALGORITHMIC_TOKENS):
        return None
    if not _has_token(requirements, _EXISTENCE_TOKENS):
        return None

    raw_globs = raw.get("glob", raw.get("globs"))
    if isinstance(raw_globs, str):
        raw_globs = [raw_globs]
    if not isinstance(raw_globs, list):
        return None

    globs: list[str] = []
    for g in raw_globs:
        if not isinstance(g, str):
            continue
        g = g.strip()
        if not g or ".." in g or g.startswith("/"):
            continue  # path escape / absolute → never persist.
        # GROUNDING — the artifact must be NAMED in the leaf. An invented filename
        # is a guaranteed 0.0 against a run that named it anything else.
        stem = re.sub(r"[*?\[\]]", "", Path(g).name).strip()
        if not stem or stem.lower() not in requirements.lower():
            logger.debug(
                "rubric_gen: dropping artifact glob %r — not named in the leaf text", g
            )
            continue
        globs.append(g)

    if not globs:
        return None
    # No on_missing valve here, deliberately: for an EXISTENCE leaf, "the file is
    # absent" is precisely the question being asked, so a missing file is a real
    # verdict (0.0), not an unresolvable lookup.
    return {"glob": globs}


def _annotate_numeric(requirements: str, raw: dict, paper_text: str) -> dict | None:
    """Build a grounded ``deterministic:numeric`` assertion, or ``None``."""
    metric_key = str(raw.get("metric_key", "")).strip()
    if not metric_key:
        return None

    target = raw.get("target")
    if isinstance(target, str):
        coerced = _numbers_in(target)
        target = coerced[0] if len(coerced) == 1 else None
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        return None

    # SPECIFICITY — a result leaf that also asserts an algorithmic invariant is a
    # fidelity leaf wearing a number; leave it to the LLM. And a leaf citing more
    # than one figure (e.g. "72.3 vs the 63.9 baseline") is not checkable by a
    # single threshold without crediting a run that missed the other one.
    if _has_token(requirements, _ALGORITHMIC_TOKENS):
        return None
    pinned = _pinned_numbers(requirements)
    if len(pinned) != 1 or not _numbers_close(float(target), pinned[0]):
        return None

    # POLARITY — an unclassifiable metric cannot be graded as a threshold at all
    # (grading a loss with higher_better would invert the verdict).
    probe = f"{metric_key} {requirements}"
    higher = _has_token(probe, _HIGHER_BETTER_TOKENS)
    lower = _has_token(probe, _LOWER_BETTER_TOKENS)
    if higher == lower:  # both or neither → ambiguous.
        return None
    direction = "higher_better" if higher else "lower_better"

    # GROUNDING — the target must be a number the paper actually reports.
    if not _value_grounded(target, paper_text, allow_percent_scaling=True):
        logger.warning(
            "rubric_gen: dropping UNGROUNDED numeric annotation %s=%r — target does not "
            "occur in the paper text; leaf → LLM",
            metric_key, target,
        )
        return None

    return {
        "metric_key": metric_key,
        "target": float(target),
        # Threshold satisfaction, not exact magnitude — a faithful reproduction
        # lands NEAR the paper's number, and must not be failed for missing it by
        # a hair. Mirrors rubric_contract._RESULT_MATCH_TOLERANCE.
        "tolerance": abs(float(target)) * _NUMERIC_REL_TOLERANCE,
        "direction": direction,
        # THE FALSE-NEGATIVE VALVE. The canonical metrics shape is
        # per_model[m][env][baseline]={"metric": …}, so a predicted key name may
        # simply not exist even though the result does. That is a NAMING mismatch,
        # not missing evidence — route it to the LLM instead of grading 0.0.
        # (Fabrication in the other direction stays covered by the A7 evidence gate.)
        "on_missing": "llm",
    }


def _annotate_leaf(
    leaf: dict, paper_text: str, contested: Mapping | None = None
) -> tuple[str, dict] | None:
    """Derive a validated ``(check_kind, assertion)`` for one leaf, or ``None``.

    ``None`` — a leaf we decline to annotate — is the SAFE, common, and correct
    outcome: the leaf is graded by the LLM exactly as it is today. Judgment leaves
    ("is the method faithfully described") must land here, and so must any leaf
    whose value we cannot ground.

    ``contested`` is the operator's per-paper coefficient registry; a symbol whose
    declared value disagrees with the paper text is never machine-checked.
    """
    try:
        raw = leaf.get("check")
        if not isinstance(raw, dict):
            return None
        kind = str(raw.get("kind", "")).strip()
        requirements = str(leaf.get("requirements", ""))
        if not requirements:
            return None

        if kind == CHECK_HPARAM:
            assertion = _annotate_hparam(requirements, raw, paper_text, contested)
        elif kind == CHECK_ARTIFACT:
            assertion = _annotate_artifact(requirements, raw)
        elif kind == CHECK_NUMERIC:
            assertion = _annotate_numeric(requirements, raw, paper_text)
        else:
            return None  # unknown/absent kind (incl. an explicit "judgment") → LLM.

        if assertion is None:
            return None
        # Final structural gate before this can reach disk.
        if not _validate_annotation(kind, assertion):
            logger.warning(
                "rubric_gen: dropping structurally-invalid %s annotation %r", kind, assertion
            )
            return None
        return (kind, assertion)
    except Exception:  # noqa: BLE001 — annotation is a bonus; never break rubric-gen.
        logger.debug("rubric_gen: leaf annotation failed — leaf → LLM", exc_info=True)
        return None


def strip_invalid_annotations(tree: dict) -> int:
    """Drop any leaf annotation that fails the structural gate. Returns #dropped.

    Belt-and-suspenders pass over the FINAL tree, immediately before it is
    returned (and thus before ``run.py`` persists ``generated_rubric.json``): a
    malformed ``check_kind``/``assertion`` is removed and logged rather than
    written to disk. Mutates ``tree`` in place; fail-soft.
    """
    dropped = 0

    def _walk(node: Any) -> None:
        nonlocal dropped
        if not isinstance(node, dict):
            return
        if "check_kind" in node or "assertion" in node:
            if not _validate_annotation(node.get("check_kind"), node.get("assertion")):
                logger.warning(
                    "rubric_gen: dropping invalid annotation from leaf %r "
                    "(check_kind=%r) — NOT persisted; leaf falls through to the LLM",
                    node.get("id"), node.get("check_kind"),
                )
                node.pop("check_kind", None)
                node.pop("assertion", None)
                dropped += 1
        for child in node.get("sub_tasks") or []:
            _walk(child)

    try:
        _walk(tree)
    except Exception:  # noqa: BLE001 — fail-soft; a persist-guard must never raise.
        logger.debug("rubric_gen: strip_invalid_annotations failed", exc_info=True)
    return dropped


def annotation_coverage(tree: dict) -> dict[str, Any]:
    """Leaf counts by grading route — the metric this whole feature exists to move.

    Returns ``{"total", "deterministic", "llm", "by_kind", "fraction"}``.
    """
    total = 0
    by_kind: dict[str, int] = {}

    def _walk(node: Any) -> None:
        nonlocal total
        if not isinstance(node, dict):
            return
        children = [c for c in (node.get("sub_tasks") or []) if isinstance(c, dict)]
        if not children:
            total += 1
            kind = node.get("check_kind")
            if kind in DETERMINISTIC_CHECK_KINDS:
                by_kind[kind] = by_kind.get(kind, 0) + 1
            return
        for child in children:
            _walk(child)

    _walk(tree)
    deterministic = sum(by_kind.values())
    return {
        "total": total,
        "deterministic": deterministic,
        "llm": total - deterministic,
        "by_kind": by_kind,
        "fraction": (deterministic / total) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_json_object(raw: str) -> dict | None:
    """Extract the first JSON object from a string (reuses primitives._extract_json — review M3 / T26)."""
    from backend.agents.rlm.primitives import _extract_json
    try:
        return _extract_json(raw)
    except ValueError:
        return None


def _is_placeholder_requirement(req: str) -> bool:
    """Return True only for a genuinely empty / comma-only parenthetical.

    This regex is the *last-resort* net for a truly empty template the model
    forgot to fill — "(, )", "( )", "(,)". The primary defense against vague
    leaves is the system prompt's concrete-value requirement; this net must
    never over-drop a concrete leaf.

    The earlier net ``\\(\\s*[^)0-9A-Za-z"\\']*\\s*\\)`` over-dropped real
    metric/equation leaves whose parenthetical merely lacked an ASCII char —
    "success rate (%)", "(gate.detach())" (inner ()), "r_t(θ)" (Greek) — which
    stripped the SDAR rubric invariants from the tree (F-32). So it fires only
    on an empty/comma-only paren, and never on a method-call paren (one
    immediately preceded by a word char, e.g. ``detach()``).
    """
    if re.search(r'(?<!\w)\(\s*(?:,\s*)*\)', req):
        return True
    return False


def _clean_categories(raw_categories: list) -> list[dict]:
    """Drop malformed categories and leaves; return a clean list."""
    cleaned: list[dict] = []
    for cat in raw_categories:
        if not isinstance(cat, dict):
            continue
        name = cat.get("name", "")
        if not isinstance(name, str) or not name.strip():
            continue
        raw_leaves = cat.get("leaves") or []
        good_leaves = []
        for lf in raw_leaves:
            if not isinstance(lf, dict):
                continue
            req = lf.get("requirements", "")
            if not isinstance(req, str) or not req.strip():
                continue
            if _is_placeholder_requirement(req):
                logger.warning(
                    "generate_rubric_tree: dropped placeholder leaf: %r", req[:120]
                )
                continue
            good_leaves.append(lf)
        if not good_leaves:
            continue
        cleaned.append({"name": name.strip(), "weight": cat.get("weight"), "leaves": good_leaves})
    return cleaned


def _normalize_weights(weights: list) -> list[float]:
    """Normalize raw weights to sum to 1.0.

    A weight that is None, <= 0, or non-numeric is filled with the **mean of the
    valid weights** in the level — so a leaf with a missing weight still counts,
    rather than silently dropping to weight 0 and being excluded from `roll_up`.
    If no weight in the level is valid, every entry gets an equal share.
    """
    coerced: list[float | None] = []
    for w in weights:
        try:
            v = float(w)
        except (TypeError, ValueError):
            v = None
        coerced.append(v if (v is not None and v > 0.0) else None)

    valid = [v for v in coerced if v is not None]
    if not valid:
        n = len(coerced)
        return [1.0 / n] * n if n else []

    fill = sum(valid) / len(valid)
    filled = [v if v is not None else fill for v in coerced]
    total = sum(filled)
    return [v / total for v in filled]


def _build_tree(
    categories: list[dict],
    paper_title: str,
    *,
    paper_text: str = "",
    contested: Mapping | None = None,
) -> dict:
    """Build the rubric tree from cleaned categories.

    ``paper_text`` is used ONLY to ground deterministic-leaf annotations (a value
    the paper does not state is never asserted). It is absent by default so every
    existing caller — and every run with the flag off — behaves identically.

    ``contested`` is the operator-declared coefficient registry for this paper (see
    ``_annotate_coefficient``'s CONTESTED gate); absent by default, and absent means
    "no veto".
    """
    annotate = deterministic_leaves_enabled() and bool(paper_text)

    cat_weights_raw = [c.get("weight") for c in categories]
    cat_weights = _normalize_weights(cat_weights_raw)

    category_nodes: list[dict] = []
    for cat, cat_w in zip(categories, cat_weights):
        leaf_weights_raw = [lf.get("weight") for lf in cat["leaves"]]
        leaf_weights = _normalize_weights(leaf_weights_raw)

        leaf_nodes: list[dict] = []
        for lf, lw in zip(cat["leaves"], leaf_weights):
            node: dict[str, Any] = {
                "id": uuid.uuid4().hex,
                "requirements": lf["requirements"].strip(),
                "weight": lw,
                "task_category": cat["name"],
                "finegrained_task_category": None,
                "sub_tasks": [],
            }
            if annotate:
                annotated = _annotate_leaf(lf, paper_text, contested)
                if annotated is not None:
                    node["check_kind"], node["assertion"] = annotated
            leaf_nodes.append(node)

        category_nodes.append({
            "id": uuid.uuid4().hex,
            "requirements": cat["name"],
            "weight": cat_w,
            "task_category": None,
            "finegrained_task_category": None,
            "sub_tasks": leaf_nodes,
        })

    return {
        "id": uuid.uuid4().hex,
        "requirements": f"Reproduce: {paper_title or 'the paper'}",
        "weight": 1.0,
        "task_category": None,
        "finegrained_task_category": None,
        "sub_tasks": category_nodes,
    }
