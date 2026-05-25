"""Pre-flight validator — static AST + grep checks BEFORE dispatching commands.

The agent's ``implement_baseline`` step often produces a ``train.py`` that
silently:

  * Subsamples datasets (e.g., 4 K MNIST instead of 60 K).
  * Skips paper variants (e.g., runs 2 of the 5 model variants the paper
    compares).
  * Substitutes surrogate models (e.g., writes ``class TinyMLP`` instead
    of the paper's actual architecture).

The post-run rubric-contract validator (:mod:`rubric_contract`) catches these
shortcuts AFTER the pod has burned ~10 minutes of compute. This module catches
them BEFORE :func:`run_experiment` dispatches ``commands.json`` to the
sandbox, by performing cheap static checks on every ``.py`` file under
``code_dir``.

Design contract:

  * Pure function: no I/O side effects beyond reading the code directory.
  * Fail-soft: any unexpected shape returns an empty list (or a single
    syntax-error violation) rather than raising; the validator MUST NOT
    block a run on an internal error of its own.
  * Synchronous and fast: every check is AST/regex-only, no LLM calls,
    no network — must complete in <500 ms on a typical paper-sized
    code directory.
  * One violation per concrete issue, with a ``hint`` field that maps
    directly to a fix the agent can perform on its next iteration.

Severity policy:

  * ``hard``  — block dispatch; the run cannot succeed against the rubric
    contract with this code (missing required variant, surrogate model,
    detectable dataset subsetting).
  * ``soft``  — surface a warning; the dispatch proceeds (the agent may
    be writing the key dynamically at runtime).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

# ---------------------------------------------------------------------------
# Surrogate-model substring vocabulary. Case-insensitive substring match on
# every ``class <Name>`` AST node. Each entry signals "this is a stand-in for
# the real architecture" — the rubric grader will fail on any of these.
# ---------------------------------------------------------------------------
_SURROGATE_TOKENS: tuple[str, ...] = (
    "tiny",
    "mock",
    "dummy",
    "smoke",
    "fake",
    "stub",
    "toy",
    "surrogate",
)

# Floor below which a detected dataset subset is considered a scope violation:
# subsetting to less than 90 % of the paper's declared full size.
_DATASET_SUBSET_FLOOR: float = 0.90

# Hard cap on per-validator work so a pathological input cannot blow the
# 500 ms budget. Files larger than this are still parsed for class names and
# variant tokens, but the AST-walk for numeric literal subsets is bounded.
_MAX_AST_NODES_PER_FILE: int = 50_000


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PreFlightViolation:
    """A single pre-flight violation against the paper's declared contract.

    Attributes
    ----------
    severity : "hard" | "soft"
        ``hard`` blocks dispatch and fails the run with
        ``pre_flight_blocked=True``.  ``soft`` is logged but the run proceeds.
    area : str
        Rubric area name this violation maps to (one of the PaperBench five).
    detail : str
        Human-readable description of WHAT is wrong.
    hint : str
        Concrete actionable suggestion the agent can apply on its next
        ``implement_baseline`` iteration.
    """

    severity: Literal["hard", "soft"]
    area: str
    detail: str
    hint: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "area": self.area,
            "detail": self.detail,
            "hint": self.hint,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_python_files(code_dir: Path) -> list[Path]:
    """Return the .py files this validator should inspect.

    Scope:
      * ``train.py`` at code_dir root if present.
      * Every ``exp_*.py`` at code_dir root if present.
      * Falls back to every top-level ``.py`` when neither shape exists, so
        an agent that emits ``main.py`` instead is not silently skipped.

    Sub-directories are NOT walked — the validator targets the entry
    points the agent writes, not vendored library code.
    """
    if not code_dir.exists() or not code_dir.is_dir():
        return []

    candidates: list[Path] = []
    train = code_dir / "train.py"
    if train.is_file():
        candidates.append(train)
    for exp in sorted(code_dir.glob("exp_*.py")):
        if exp.is_file():
            candidates.append(exp)

    if candidates:
        return candidates

    # Fallback: scan every top-level .py the agent emitted.
    return sorted(p for p in code_dir.glob("*.py") if p.is_file())


def _variant_tokens(variant: str) -> tuple[str, ...]:
    """Expand a variant id into the case-insensitive tokens we accept.

    e.g. ``qwen3_1_7b`` →
      ("qwen3_1_7b", "qwen3-1.7b", "qwen3-1_7b", "qwen3_1.7b").

    The grader checks per_model OR omitted by exact key; the pre-flight is
    more permissive because the agent's code may name the model with hyphens
    + dots (matching the HF repo id) but the YAML uses underscores (Python
    identifier-safe). We accept both.
    """
    base = variant.strip()
    if not base:
        return ()
    tokens = {base, base.lower()}

    # Common transforms: underscores ↔ hyphens, and the "_N_M_" → "-N.M-"
    # pattern that appears in Qwen / Llama / Mistral model ids.
    snake = base.replace("-", "_").replace(".", "_")
    kebab = base.replace("_", "-").replace(".", "-")
    dotted = re.sub(r"_(\d)_(\d)", r"_\1.\2", snake)
    dotted_kebab = re.sub(r"-(\d)-(\d)", r"-\1.\2", kebab)
    tokens.update({snake, kebab, dotted, dotted_kebab})

    # Pre-lowercased forms so the substring match below is O(file_len * tokens).
    return tuple(sorted({t.lower() for t in tokens if t}))


def _file_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _all_text(code_dir: Path) -> str:
    """Concatenate every top-level .py file's text (lowercased) once.

    Used by the variant + required-key + required-artifact checks, which are
    pure string-grep. We do this once per call so a 5-variant paper with
    a 200 KB train.py is still O(file_size), not O(file_size * variants).
    """
    parts: list[str] = []
    for path in _iter_python_files(code_dir):
        parts.append(_file_text(path))
    return "\n".join(parts).lower()


def _parse_or_violation(path: Path) -> tuple[ast.AST | None, PreFlightViolation | None]:
    """Parse ``path`` to an AST; on SyntaxError return a hard violation.

    A train.py that does not parse will fail the run anyway; surfacing it as
    a pre-flight violation lets the agent's repair_context see the exact line
    rather than a generic "command failed" error after the pod boot.
    """
    text = _file_text(path)
    if not text:
        return None, None
    try:
        return ast.parse(text, filename=str(path)), None
    except SyntaxError as exc:
        return None, PreFlightViolation(
            severity="hard",
            area="Experiment execution and reproducibility",
            detail=f"{path.name} has SyntaxError at line {exc.lineno}: {exc.msg}",
            hint=(
                f"fix the syntax error in {path.name} before re-dispatching — "
                f"the sandbox cannot execute a file that does not parse."
            ),
        )


# ---------------------------------------------------------------------------
# Per-check implementations
# ---------------------------------------------------------------------------


def _check_variants(
    paper_targets: dict,
    code_text_lower: str,
    out: list[PreFlightViolation],
) -> None:
    """Each variant in ``variants_required`` must appear in code OR be honestly
    declared as omitted via ``metrics["omitted"]["<variant>"]``."""
    variants = paper_targets.get("variants_required") or []
    if not isinstance(variants, (list, tuple)) or not variants:
        return

    for variant in variants:
        if not isinstance(variant, str) or not variant.strip():
            continue
        tokens = _variant_tokens(variant)
        if not tokens:
            continue

        # Two ways to satisfy the variant:
        #   (a) the variant token appears anywhere in the code (case-insensitive)
        #   (b) the code declares it omitted: metrics["omitted"]["<variant>"]
        present_in_code = any(tok in code_text_lower for tok in tokens)
        # Check honest-omit patterns: metrics["omitted"]["<variant>"] OR
        # metrics["omitted"]={..., "<variant>": ...}. The grader requires a
        # string key match in either dict-literal form.
        omit_patterns = (
            f'"omitted"]["{variant.lower()}"]',
            f"'omitted']['{variant.lower()}']",
            f'"omitted"]["{variant}"]'.lower(),
            f"'omitted']['{variant}']".lower(),
            f'"{variant.lower()}":',  # dict key in a literal omitted={...}
            f"'{variant.lower()}':",
        )
        # Per-key heuristic: the literal "omitted" plus the variant id within
        # 200 chars of each other → almost certainly an honest-omit declaration.
        # The substring tests above already cover that; this is the gate.
        honestly_omitted = (
            "omitted" in code_text_lower
            and any(p in code_text_lower for p in omit_patterns)
        )

        if present_in_code or honestly_omitted:
            continue

        out.append(PreFlightViolation(
            severity="hard",
            area="Experiment execution and reproducibility",
            detail=(
                f"variant {variant!r} missing from code AND not declared as "
                f"omitted in metrics.json"
            ),
            hint=(
                f"either reference {variant!r} in your training/eval code "
                f"(e.g. as a config key, model id, or per_model branch), "
                f"OR declare it omitted via "
                f"metrics['omitted'][{variant!r}] = '<one-line reason>'"
            ),
        ))


# Numeric-literal patterns we'll treat as candidate dataset subsetters.
# Order matters for readability only — each matcher returns the integer N
# that bounds the slice.
def _walk_for_subset_sizes(tree: ast.AST) -> Iterable[int]:
    """Yield numeric N for every ``range(0, N)`` / ``[:N]`` / ``range(N)``
    pattern we find. Other call shapes (``range(low, high, step)``) are
    ignored — they're rarely used to subset a dataset.

    Also yields N for keyword forms like ``train_size=N`` and
    ``num_samples=N`` so a literal kwarg on dataset constructors is caught.
    """
    node_count = 0
    for node in ast.walk(tree):
        node_count += 1
        if node_count > _MAX_AST_NODES_PER_FILE:
            return

        # range(N) and range(0, N)
        if isinstance(node, ast.Call):
            func_name = (
                node.func.id if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute)
                else None
            )
            if func_name == "range" and node.args:
                if len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, int):
                        yield node.args[0].value
                elif (
                    len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[0].value == 0
                    and isinstance(node.args[1].value, int)
                ):
                    yield node.args[1].value

            # Keyword args on calls: train_size=N, num_samples=N, n_samples=N,
            # max_samples=N, subset_size=N.
            for kw in node.keywords:
                if kw.arg in (
                    "train_size", "num_samples", "n_samples",
                    "max_samples", "subset_size", "subset",
                ) and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, int):
                        yield kw.value.value

        # x[:N] slicing
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            sl = node.slice
            # Only treat [:N] (lower is None or 0) as a subset signal — [N:]
            # / [N:M] are common indexing patterns and produce false positives.
            lower_ok = sl.lower is None or (
                isinstance(sl.lower, ast.Constant) and sl.lower.value in (0, None)
            )
            if lower_ok and isinstance(sl.upper, ast.Constant) and isinstance(sl.upper.value, int):
                yield sl.upper.value


def _check_dataset_size(
    paper_targets: dict,
    files: list[Path],
    trees: dict[Path, ast.AST],
    out: list[PreFlightViolation],
) -> None:
    """If any AST literal subset size is < 90 % of the paper's declared full
    size, raise a hard violation. The agent should either use the full dataset
    or declare the subset in metrics.json["omitted"] with a reason."""
    full_size = paper_targets.get("train_size_full")
    try:
        full_size_int = int(full_size) if full_size is not None else None
    except (TypeError, ValueError):
        full_size_int = None
    if not full_size_int or full_size_int <= 0:
        return  # no contract → nothing to check

    floor = int(full_size_int * _DATASET_SUBSET_FLOOR)
    # Track the smallest subset we found so the violation message is concrete.
    found_subsets: list[tuple[Path, int]] = []
    for path in files:
        tree = trees.get(path)
        if tree is None:
            continue
        for n in _walk_for_subset_sizes(tree):
            # We only care about numbers in the "dataset-size shaped" range:
            # > 100 (rule out batch-size, dim-size literals) and < full_size.
            if 100 < n < floor:
                found_subsets.append((path, n))

    if not found_subsets:
        return

    # Surface the smallest detected subset as the canonical violation — the
    # agent fixes one place at a time and any single < 90 % subset fails the
    # contract anyway.
    path, smallest = min(found_subsets, key=lambda x: x[1])
    out.append(PreFlightViolation(
        severity="hard",
        area="Data fidelity and preparation",
        detail=(
            f"{path.name} subsets a dataset to {smallest} samples; paper's "
            f"train_size_full is {full_size_int} "
            f"(< {int(_DATASET_SUBSET_FLOOR * 100)}% threshold)"
        ),
        hint=(
            f"use the full training set ({full_size_int} samples). If a subset is "
            f"unavoidable for compute reasons, declare it honestly in "
            f"metrics['omitted']['dataset_subset'] = '<reason>' instead of "
            f"silently slicing."
        ),
    ))


def _check_surrogate_models(
    files: list[Path],
    trees: dict[Path, ast.AST],
    out: list[PreFlightViolation],
) -> None:
    """Flag every ``class <Name>`` whose name contains a surrogate token."""
    for path in files:
        tree = trees.get(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            name_lower = node.name.lower()
            for token in _SURROGATE_TOKENS:
                if token in name_lower:
                    out.append(PreFlightViolation(
                        severity="hard",
                        area="Method fidelity to the paper",
                        detail=(
                            f"{path.name}: surrogate model class "
                            f"{node.name!r} detected (contains {token!r})"
                        ),
                        hint=(
                            f"the paper requires the real architecture, not a "
                            f"stand-in.  Remove {node.name!r} and import / "
                            f"instantiate the paper's actual model."
                        ),
                    ))
                    break  # one violation per class is enough


def _check_required_metrics_keys(
    paper_targets: dict,
    code_text_lower: str,
    out: list[PreFlightViolation],
) -> None:
    """Soft check: the literal metric key string appears somewhere in code.

    The agent may build keys dynamically (``metrics[f"acc_{name}"] = ...``);
    that's why this is a soft violation. The hint nudges the agent to add an
    explicit assignment so the rubric grader can find the key reliably.
    """
    required = paper_targets.get("required_metrics_keys") or []
    if not isinstance(required, (list, tuple)):
        return
    for key in required:
        if not isinstance(key, str) or not key:
            continue
        if key.lower() in code_text_lower:
            continue
        out.append(PreFlightViolation(
            severity="soft",
            area="Evaluation protocol and metric correctness",
            detail=(
                f"required metric key {key!r} not found as a literal string "
                f"in any code file"
            ),
            hint=(
                f"add an explicit assignment such as metrics[{key!r}] = ... "
                f"so the grader can locate the key without depending on "
                f"runtime string interpolation."
            ),
        ))


def _artifact_search_token(name: str) -> str:
    """Reduce a required artifact glob to the substring we expect to see.

    e.g. ``fig_*.png`` → ``fig_``; ``README.md`` → ``readme.md``;
    ``training_curves.json`` → ``training_curves.json``.

    Returns the lowercased token; empty string means "skip".
    """
    if not name:
        return ""
    # Take the prefix up to the first glob metacharacter — that's the literal
    # piece every matching file must share.
    for meta in ("*", "?", "["):
        idx = name.find(meta)
        if idx != -1:
            name = name[:idx]
            break
    return name.strip().lower()


def _check_required_artifacts(
    paper_targets: dict,
    code_text_lower: str,
    out: list[PreFlightViolation],
) -> None:
    """Soft check: each required artifact name appears as a literal string."""
    required = paper_targets.get("required_artifacts") or []
    if not isinstance(required, (list, tuple)):
        return
    for raw in required:
        if not isinstance(raw, str):
            continue
        token = _artifact_search_token(raw)
        if not token:
            continue
        if token in code_text_lower:
            continue
        out.append(PreFlightViolation(
            severity="soft",
            area="Artifact completeness and provenance",
            detail=(
                f"required artifact {raw!r} not referenced as a literal "
                f"string in any code file"
            ),
            hint=(
                f"emit {raw!r} into $OUTPUT_DIR — reference the filename "
                f"explicitly in your code so it's easy to audit which "
                f"path-write corresponds to which rubric artifact."
            ),
        ))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _check_tensor_device_mismatch(
    trees: dict[Path, ast.AST],
    out: list[PreFlightViolation],
) -> None:
    """Block dispatch if model.to(device) appears inside a function that
    takes ``optimizer`` as a parameter.

    Symptom this catches: the 2026-05-24 Dropout Exp 1 + Exp 3 crash —

        ``RuntimeError: Expected all tensors to be on the same device,
        but found at least two devices, cuda:0 and cpu``

    Root cause: the agent constructed ``AdamOptimizer(model.parameters(),
    ...)`` BEFORE moving the model to the GPU. The optimizer's stored
    state tensors (``self.m``, ``self.v`` allocated via ``zeros_like(p)``)
    are on CPU; the gradient ``g = p.grad.data`` comes from the model's
    GPU params after a later ``model.to(device)`` (often inside
    ``run_epochs(model, loader, optimizer, ...)``). The ``self.m[i] = ...``
    update mixes CPU and GPU tensors → crash.

    The Pythonic fix is: ``model.to(device)`` THEN ``Optimizer(model.parameters(),
    ...)``. The agent's repair_context surfaces the hint so the next
    ``implement_baseline`` iteration writes correct code.
    """
    # Per-file pass: identify which names had ``.parameters()`` called on
    # them (those are the model objects the optimizer was built around).
    # ``xb.to(device)`` on a batch tensor is a false positive; ``model.to(device)``
    # is the real bug. We only flag the .to() target if the same name has
    # .parameters() somewhere else in the same file.
    for path, tree in trees.items():
        names_with_params: set[str] = set()
        for n in ast.walk(tree):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "parameters"
                and isinstance(n.func.value, ast.Name)
            ):
                names_with_params.add(n.func.value.id)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Function must accept ``optimizer`` as a positional param —
            # that's the smoking-gun shape for "optimizer constructed
            # elsewhere, model moved here".
            arg_names = {a.arg for a in node.args.args}
            arg_names |= {a.arg for a in getattr(node.args, "kwonlyargs", [])}
            if "optimizer" not in arg_names and "opt" not in arg_names:
                continue
            # Walk the function body for ``something.to(<device_expr>)`` or
            # the ``something.cuda()`` shorthand (same bug, different syntax).
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Call):
                    continue
                if not isinstance(inner.func, ast.Attribute):
                    continue
                attr = inner.func.attr
                if attr not in {"to", "cuda"}:
                    continue
                # Target must look like a model (parameters() called on it
                # elsewhere in the same file) — skip batch tensors xb/yb.
                target = ""
                if isinstance(inner.func.value, ast.Name):
                    target = inner.func.value.id
                if target not in names_with_params:
                    continue
                # ``.cuda()`` with no args is unambiguous — flag.
                # ``.to(...)`` needs a device-shaped first arg.
                if attr == "to":
                    if not inner.args:
                        continue
                    first = inner.args[0]
                    device_like = False
                    if isinstance(first, ast.Name) and first.id in {"device", "DEVICE"}:
                        device_like = True
                    elif isinstance(first, ast.Constant) and isinstance(first.value, str):
                        if first.value.startswith("cuda") or first.value == "cpu":
                            device_like = True
                    elif isinstance(first, ast.Call) and isinstance(first.func, ast.Attribute):
                        # torch.device("cuda") pattern
                        if first.func.attr == "device":
                            device_like = True
                    if not device_like:
                        continue
                # We have: function-with-optimizer-param containing
                # model.to(device) or model.cuda() where model.parameters() exists.
                call_shape = f"{target or '<expr>'}.{attr}(...)"
                out.append(PreFlightViolation(
                    severity="hard",
                    area="Result match",
                    detail=(
                        f"{path.name}:{inner.lineno}: `{call_shape}` "
                        f"inside function `{node.name}` that takes `optimizer` "
                        f"as a parameter. The optimizer was constructed BEFORE "
                        f"this function ran, capturing CPU-tensor refs; its "
                        f"state tensors (self.m, self.v from zeros_like) stay "
                        f"on CPU while the model moves to GPU. The next "
                        f"optimizer.step() mixes CPU + GPU tensors and crashes "
                        f"with `RuntimeError: tensors on cuda:0 and cpu`."
                    ),
                    hint=(
                        f"Move the `.to(device)` call BEFORE constructing the "
                        f"optimizer — the standard PyTorch idiom: "
                        f"\n    model = MyModel().to(device)\n"
                        f"    optimizer = AdamOptimizer(model.parameters(), ...)\n"
                        f"NOT inside a function that receives `optimizer` as "
                        f"an argument."
                    ),
                ))


def _check_absurd_learning_rate(
    trees: dict[Path, ast.AST],
    out: list[PreFlightViolation],
) -> None:
    """Block dispatch when an LR literal is outside the sane training range.

    The 2026-05-25 Dropout regression: the agent extracted MNIST hyperparameters
    and picked ``lr=10.0`` (the paper's Section 2 mentioned a scaling factor of
    10 that the agent confused for the base learning rate). Result: train_loss
    immediately became NaN, training churned for 30+ minutes with zero useful
    output, watchdog killed.

    Sane training-LR range for SGD/Adam-class optimizers in practice:
      * lower: 1e-7 (below this is effectively no learning)
      * upper: 1.0  (above this is divergent for all standard architectures)

    This check looks for assignments like ``lr=10.0``, ``learning_rate = 5``,
    ``Optimizer(lr=20)``, and any obvious-literal pattern. AST visitor walks
    keyword args + simple assignments. False positives on non-LR variables
    named ``lr_something`` are avoided by checking the exact key.
    """
    LR_NAMES = {
        "lr", "learning_rate", "alpha", "base_lr", "max_lr",
        "init_lr", "initial_lr",
    }
    LOWER_BOUND = 1e-7
    UPPER_BOUND = 1.0

    def _flag(path: Path, lineno: int, key: str, value: float) -> None:
        out.append(PreFlightViolation(
            severity="hard",
            area="Experiment execution and reproducibility",
            detail=(
                f"{path.name}:{lineno}: `{key}={value}` is outside the sane "
                f"training range [{LOWER_BOUND}, {UPPER_BOUND}]. Standard "
                f"optimizers (SGD/Momentum/Adam) diverge with lr > 1.0 — "
                f"the 2026-05-25 Dropout run hit lr=10.0 and produced "
                f"train_loss=NaN from epoch 1, churning for 30 min before "
                f"watchdog kill."
            ),
            hint=(
                f"Set {key} to a value in {{1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}} "
                f"depending on optimizer/architecture (Adam: ~1e-3 typical, "
                f"SGD+momentum: ~1e-2 typical). If the paper genuinely uses "
                f"the requested literal as a *scale factor* applied to a base, "
                f"reify that in code (`base_lr * scale`) rather than as the lr."
            ),
        ))

    def _maybe_float(node: ast.AST) -> float | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        # Handle unary-minus literals like ``-1e-3``
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant)
                and isinstance(node.operand.value, (int, float))):
            return float(-node.operand.value)
        return None

    for path, tree in trees.items():
        for node in ast.walk(tree):
            # Pattern A: assignments — ``lr = 10`` / ``learning_rate = 5.0``
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    if target.id not in LR_NAMES:
                        continue
                    val = _maybe_float(node.value)
                    if val is None:
                        continue
                    if val < LOWER_BOUND or val > UPPER_BOUND:
                        _flag(path, node.lineno, target.id, val)
            # Pattern B: keyword args — ``Optimizer(lr=10)`` / ``Adam(lr=5)``
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg not in LR_NAMES:
                        continue
                    val = _maybe_float(kw.value)
                    if val is None:
                        continue
                    if val < LOWER_BOUND or val > UPPER_BOUND:
                        _flag(path, node.lineno, kw.arg, val)
            # Pattern C: dict literals — ``CONFIG = {"lr": 10.0}`` /
            # ``MLP_CONFIGS = [{"lr": 10}, ...]`` — covers Dropout's case
            # where lr lived in a list of config dicts, not a top-level assign.
            if isinstance(node, ast.Dict):
                for key_node, val_node in zip(node.keys, node.values):
                    if not isinstance(key_node, ast.Constant):
                        continue
                    if not isinstance(key_node.value, str):
                        continue
                    if key_node.value not in LR_NAMES:
                        continue
                    val = _maybe_float(val_node)
                    if val is None:
                        continue
                    if val < LOWER_BOUND or val > UPPER_BOUND:
                        _flag(path, node.lineno, key_node.value, val)


def _check_requirements_torch_redundancy(
    code_dir: Path,
    base_image: str | None,
    out: list[PreFlightViolation],
) -> None:
    """Block dispatch if requirements.txt re-installs torch on a runpod/pytorch base.

    The official RunPod PyTorch images (``runpod/pytorch:*-py*-cuda*-*``)
    ship with torch / torchvision / torchaudio pre-installed.  Putting them
    in ``requirements.txt`` triggers a 755 MB+ re-download that has hit a
    ~50% mid-stream-truncation rate on every recent v10-class run.  Hard
    violation — the auto-derive should have stripped these, but the agent
    can still write its own requirements.txt and bypass auto-derive.
    """
    if not base_image:
        return
    bi = base_image.lower()
    if not (bi.startswith("runpod/pytorch") or "runpod/pytorch" in bi):
        return
    req_path = code_dir / "requirements.txt"
    if not req_path.exists():
        return
    try:
        text = req_path.read_text(encoding="utf-8")
    except OSError:
        return
    import re as _re
    offenders: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _re.match(r"([A-Za-z0-9_\-\.]+)", stripped)
        if not m:
            continue
        name = m.group(1).lower()
        if name in {"torch", "torchvision", "torchaudio"}:
            offenders.append(stripped)
    for off in offenders:
        out.append(PreFlightViolation(
            severity="hard",
            area="Experiment execution and reproducibility",
            detail=(
                f"requirements.txt contains {off!r} but base image "
                f"{base_image} already has it pre-installed — re-installation "
                f"costs ~10 min + ~755 MB download which has dropped mid-stream "
                f"~50% of recent runs."
            ),
            hint=(
                f"Remove {off!r} (and any of torch/torchvision/torchaudio) "
                f"from requirements.txt.  The base image already provides them."
            ),
        ))


def validate_code_pre_flight(
    code_dir: Path,
    paper_targets: dict | None,
    *,
    arxiv_id: str | None = None,  # noqa: ARG001 — accepted for future hooks
    base_image: str | None = None,
) -> list[PreFlightViolation]:
    """Static AST + grep checks on train.py + every exp_*.py BEFORE dispatch.

    Parameters
    ----------
    code_dir : Path
        Directory containing the agent's ``train.py`` (and any ``exp_*.py``).
    paper_targets : dict | None
        The ``paper_targets`` section from the YAML override.  When None or
        empty, returns an empty list (no contract → nothing to validate).
    arxiv_id : str | None
        Accepted for future per-paper overrides; currently unused.

    Returns
    -------
    list[PreFlightViolation]
        One entry per concrete issue.  Empty list = compliant OR no
        contract to validate against.

    Notes
    -----
    Fail-soft: any internal exception swallows and returns whatever
    violations were collected so far.  Pre-flight observability MUST NOT
    block a run on its own bug.
    """
    code_dir = Path(code_dir)
    violations: list[PreFlightViolation] = []

    # The torch-redundancy check runs even without a paper_targets contract —
    # it's a base-image invariant, not a paper-specific one.
    try:
        _check_requirements_torch_redundancy(code_dir, base_image, violations)
    except Exception:  # noqa: BLE001
        pass

    # Parse every Python file once. Both the paper-independent
    # tensor-device-mismatch check AND the paper-specific checks need
    # ASTs; sharing the parse is cheap and keeps line-number reporting
    # consistent across all violations.
    files = _iter_python_files(code_dir)
    trees: dict[Path, ast.AST] = {}
    if files:
        try:
            for path in files:
                tree, syn_err = _parse_or_violation(path)
                if syn_err is not None:
                    violations.append(syn_err)
                if tree is not None:
                    trees[path] = tree
        except Exception:  # noqa: BLE001 — never raise from pre-flight
            return violations

    # Tensor-device-mismatch is a runtime-correctness invariant (catches the
    # 2026-05-24 Dropout Exp 1+3 crash); runs regardless of paper_targets.
    try:
        _check_tensor_device_mismatch(trees, violations)
    except Exception:  # noqa: BLE001
        pass

    # Absurd-LR check — runs regardless of paper_targets (2026-05-25 Dropout
    # regression: lr=10.0 produced train_loss=NaN for the entire run).
    try:
        _check_absurd_learning_rate(trees, violations)
    except Exception:  # noqa: BLE001
        pass

    if not isinstance(paper_targets, dict) or not paper_targets:
        return violations

    if not files:
        # No files to check — leave the variant / surrogate checks silent.
        # rubric_contract.py will catch the missing-artifact failure post-run.
        return violations

    # Build the lowercased corpus once. Used by all string-grep checks.
    try:
        code_text_lower = _all_text(code_dir)
    except Exception:  # noqa: BLE001
        return violations

    try:
        _check_variants(paper_targets, code_text_lower, violations)
        _check_dataset_size(paper_targets, files, trees, violations)
        _check_surrogate_models(files, trees, violations)
        _check_required_metrics_keys(paper_targets, code_text_lower, violations)
        _check_required_artifacts(paper_targets, code_text_lower, violations)
    except Exception:  # noqa: BLE001 — observability must never block the run
        return violations

    return violations


__all__ = [
    "PreFlightViolation",
    "validate_code_pre_flight",
]
