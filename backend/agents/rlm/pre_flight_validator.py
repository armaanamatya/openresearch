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


def validate_code_pre_flight(
    code_dir: Path,
    paper_targets: dict | None,
    *,
    arxiv_id: str | None = None,  # noqa: ARG001 — accepted for future hooks
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
    if not isinstance(paper_targets, dict) or not paper_targets:
        return []

    code_dir = Path(code_dir)
    violations: list[PreFlightViolation] = []

    files = _iter_python_files(code_dir)
    if not files:
        # No files to check — leave the variant / surrogate checks silent.
        # rubric_contract.py will catch the missing-artifact failure post-run.
        return []

    # Pre-parse every file once; remember syntax errors as hard violations
    # but keep going with the rest of the files. A file that doesn't parse
    # contributes nothing to the text-grep either (its text WAS read but the
    # AST-only checks below skip it).
    trees: dict[Path, ast.AST] = {}
    try:
        for path in files:
            tree, syn_err = _parse_or_violation(path)
            if syn_err is not None:
                violations.append(syn_err)
            if tree is not None:
                trees[path] = tree
    except Exception:  # noqa: BLE001 — never raise from pre-flight
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
