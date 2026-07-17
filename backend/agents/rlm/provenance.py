"""Execution-provenance emitter — copyable helper (Lane D2a).

The grader is text-only and truncates code; it never sees the PNGs a run
produces, and the per-cell ``training_curves.json`` / ``config_used.json``
written into per-cell subdirs never reach the aggregated level the judge
reads.  The result is a faithful run that LOSES rubric points on
*evidence-visibility* — "45-epoch not confirmed," "log-scale axis not
verifiable," "batch=128 only an assumption" — even though the work was done.

This module gives the agent's own code a way to **emit machine-readable
execution evidence** the grader can read at face value:

- :func:`emit_provenance` writes ``provenance.json`` — a structured record of
  what each experiment actually ran (model, env, baseline, seed, epochs,
  steps, batch size, per-optimizer hyperparameters, hardware, framework
  versions, a convergence series, and the paper-declared **coefficients**).  A
  long convergence series is stored as a compact
  ``{len, first, last, min, max, sampled}`` **summary** so it never blows the
  grader's evidence byte-cap.

  The ``coefficients`` section (``{"beta": 10, "lambda": 0.1}``) is the one that
  carries METHOD FIDELITY rather than bookkeeping: it records the paper's
  algorithmic constants as the code actually used them, so a rubric leaf pinning
  ``β=10`` is settled by a comparison instead of an LLM's opinion.  See the
  extended note beside :data:`COEFFICIENTS_KEY`.
- :func:`emit_figure_sidecar` writes a ``<png_stem>.json`` next to each PNG so
  the figure-blind grader can read the axes (``scale:"log"`` answers
  *"log-scale axis not verifiable"*) and the series without seeing the image.
- :func:`assert_provenance` is the self-validation hook (mirror of
  ``rubric_guard.assert_metrics_schema``): the agent calls it at the end of
  ``train_cell.py`` and a missing manifest / missing figure sidecar raises
  :class:`RubricGuardFailure` whose JSON-shaped message rides the next
  iteration's ``repair_context`` channel.

Like ``gpu_cell_runner.py`` and ``rubric_guard.py``, this file is copied
**flat** into the sandbox ``code/`` directory, so it has **zero non-stdlib
dependencies** (``json`` / ``pathlib`` / ``os`` / ``typing`` only) and runs
standalone.  Auth-agnostic by construction — no provider branching, no LLM
calls, no clock reads that affect output by default (``generated_at`` is an
argument).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:  # sandbox-flat: rubric_guard.py is copied next to this file.
    from rubric_guard import RubricGuardFailure
except ImportError:  # in-repo import path.
    from backend.agents.rlm.rubric_guard import RubricGuardFailure


# The grader caps evidence per file; a convergence array longer than this is
# stored as a compact summary instead of the raw list so a long training curve
# never blows the cap.  Kept well under the per-file budget.
_MAX_SERIES_LEN = 32
# Number of evenly-spaced points retained in a summary's ``sampled`` field.
_MAX_SAMPLED_POINTS = 20

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------- #
# Paper-declared COEFFICIENTS — a namespace of their own (2026-07-13).
#
# WHY A SEPARATE NAMESPACE, AND WHY IT IS NOT OPTIONAL
# ----------------------------------------------------
# The fields above (``lr`` / ``epochs`` / ``batch_size`` / ``per_optimizer.*``)
# are *bookkeeping* hyperparameters — the training recipe. They are NOT the
# thing a surrogate gets wrong. What a surrogate gets wrong is the paper's
# ALGORITHMIC CONSTANTS: SDAR's ``g_t = σ(β·Δ_t)`` with ``β=10``, its
# ``λ=0.1`` distillation weight, a temperature, a gate threshold, a clip ε.
# Those ARE the method. Until this section existed no provenance producer wrote
# a field by any of those names, so a rubric leaf pinning ``β=10`` could only be
# graded by an LLM's opinion of the code — precisely what the project's
# "evidence, not grade" red line forbids.
#
# ROLE-SCOPED, NEVER NAME-GUESSED (learn.md 2026-07-07)
# ----------------------------------------------------
# ``alpha`` / ``beta`` / ``lambda`` / ``tau`` are AMBIGUOUS: the same glyph is a
# learning-rate-ish knob in one paper and a loss coefficient in the next, and
# ``0.0`` (an ablation) and ``>1.0`` (a weight) are both legitimate values. An
# over-broad LR guard that keyed on the NAME ``alpha`` hard-blocked a faithful
# ``alpha=0.0`` ablation (prj_618). So a coefficient is addressed by its ROLE —
# the namespace says "this is a constant the PAPER DECLARED", nothing more —
# and NOTHING in this file, or in any consumer of it, may range-check, sanity-
# check, or otherwise interpret a coefficient's value. The only legal question
# is "does it equal the value the paper declared for this leaf?".
#
# The dotted address ``coefficients.<name>`` is what a rubric leaf asserts, and
# it resolves through ``deterministic_leaf_checker``'s EXISTING dotted-field
# traversal (top level first, then each ``experiments[*]`` record) — the same
# resolution style that makes a bare ``lr`` find ``per_optimizer.adam.lr``. No
# second lookup mechanism is introduced.
# --------------------------------------------------------------------------- #

#: The namespace key. A leaf asserts ``field="coefficients.beta"``.
COEFFICIENTS_KEY = "coefficients"

#: Greek glyphs → the ASCII word provenance stores. A paper prints ``β``; a
#: manifest key must be a plain identifier. Spelling normalization ONLY — never a
#: semantic remap (``lambda`` is never rewritten to ``loss_weight``: that would be
#: guessing the symbol's role, the exact mistake learn.md 2026-07-07 records).
_GREEK_TO_ASCII: dict[str, str] = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho",
    "σ": "sigma", "τ": "tau", "υ": "upsilon", "φ": "phi", "χ": "chi",
    "ψ": "psi", "ω": "omega",
}


def canonical_coefficient_name(name: Any) -> str:
    """Canonical manifest key for a paper-declared coefficient. Idempotent.

    ``"β"`` → ``"beta"``; ``"\\beta"`` → ``"beta"``; ``"Top-K"`` → ``"top_k"``;
    ``"beta"`` → ``"beta"``. Returns ``""`` for anything unusable — callers drop
    an empty name rather than inventing one.

    Pure spelling normalization. It never maps one symbol onto another.
    """
    s = str(name).strip()
    if not s:
        return ""
    # A bare Greek glyph (possibly LaTeX-escaped) → its ASCII word.
    stripped = s.lstrip("\\").strip()
    if stripped in _GREEK_TO_ASCII:
        return _GREEK_TO_ASCII[stripped]
    s = stripped.lower()
    if s in _GREEK_TO_ASCII:  # already-lowered glyph
        return _GREEK_TO_ASCII[s]
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in ("_", "-", " ", "."):
            out.append("_")
        # anything else (parens, math ops) is dropped.
    joined = "".join(out)
    while "__" in joined:
        joined = joined.replace("__", "_")
    return joined.strip("_")


def coefficient_field(name: Any) -> str:
    """The dotted provenance field a rubric leaf asserts for a coefficient.

    ``coefficient_field("β") == "coefficients.beta"``. This is the ONE place the
    address is constructed — ``rubric_gen`` (which writes the assertion) and the
    checker (which resolves it) both key off this, so the contract cannot drift.
    Returns ``""`` when the name is unusable.
    """
    canon = canonical_coefficient_name(name)
    return f"{COEFFICIENTS_KEY}.{canon}" if canon else ""


def normalize_coefficients(coefficients: Any) -> dict[str, Any]:
    """Canonicalize a ``{name: value}`` coefficient mapping. Fail-soft.

    Keys are canonicalized (``β`` → ``beta``); values are kept ONLY when they are
    a real number (int/float, bool excluded — a bool is not a declared constant).
    A non-numeric or unnamed entry is dropped rather than written, because a
    coefficient that cannot be compared is worse than absent: absent routes to the
    LLM, garbage would be compared and could fail a faithful run.

    NOTE what is deliberately NOT here: any range/plausibility check. ``0.0`` and
    ``1e4`` are both legitimate declared values (learn.md 2026-07-07).
    """
    out: dict[str, Any] = {}
    if not isinstance(coefficients, dict):
        return out
    try:
        for raw_name, value in coefficients.items():
            name = canonical_coefficient_name(raw_name)
            if not name or not _is_number(value):
                continue
            out[name] = value
    except Exception:  # noqa: BLE001 — fail-soft; never break the training run.
        return {}
    return out


def _is_number(x: Any) -> bool:
    """True for an int/float that is not a bool (bools are ints in Python)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _evenly_spaced(seq: list[Any], k: int) -> list[Any]:
    """Return up to ``k`` evenly-spaced elements of ``seq`` (endpoints kept).

    For ``len(seq) <= k`` returns ``list(seq)``.  Otherwise samples ``k``
    indices spread across ``[0, len-1]`` so the first and last entries are
    always present — the shape of a curve survives the downsample.
    """
    n = len(seq)
    if k <= 0:
        return []
    if n <= k:
        return list(seq)
    if k == 1:
        return [seq[0]]
    # k indices from 0..n-1 inclusive, evenly spaced.
    step = (n - 1) / (k - 1)
    idxs = sorted({int(round(i * step)) for i in range(k)})
    return [seq[i] for i in idxs]


def _summarize_series(values: Any) -> Any:
    """Summarize a long numeric series; pass short / non-list values through.

    A list longer than :data:`_MAX_SERIES_LEN` becomes a compact dict::

        {"len": N, "first": v0, "last": vN-1, "min": .., "max": ..,
         "sampled": [<= _MAX_SAMPLED_POINTS evenly-spaced points]}

    ``min``/``max`` are computed only over the numeric members (a series with
    a stray ``None`` or string still summarizes without raising).  Anything
    that is not an over-length list is returned unchanged.
    """
    if not isinstance(values, list) or len(values) <= _MAX_SERIES_LEN:
        return values
    numeric = [v for v in values if _is_number(v)]
    summary: dict[str, Any] = {
        "len": len(values),
        "first": values[0],
        "last": values[-1],
        "min": min(numeric) if numeric else None,
        "max": max(numeric) if numeric else None,
        "sampled": _evenly_spaced(values, _MAX_SAMPLED_POINTS),
    }
    return summary


def _is_series_summary(obj: Any) -> bool:
    """True iff ``obj`` is a summary dict produced by :func:`_summarize_series`."""
    return isinstance(obj, dict) and "len" in obj and "sampled" in obj


def _summarize_convergence(convergence: Any) -> Any:
    """Apply :func:`_summarize_series` to every axis of a convergence mapping.

    ``convergence`` is expected to be ``{axis_name: [values...]}``; each axis
    array is summarized independently.  A non-dict is returned unchanged.
    """
    if not isinstance(convergence, dict):
        return convergence
    return {axis: _summarize_series(arr) for axis, arr in convergence.items()}


def _summarize_experiment(exp: Any) -> Any:
    """Return a shallow copy of one experiment with its convergence summarized.

    A per-experiment ``coefficients`` sub-dict is canonicalized in place (``β`` →
    ``beta``) so a cell that OVERRIDES a run-global coefficient — an ablation
    sweeping λ — is addressable at ``coefficients.<name>`` inside that record,
    exactly as it is at the manifest top level.
    """
    if not isinstance(exp, dict):
        return exp
    out = dict(exp)
    if "convergence" in out:
        out["convergence"] = _summarize_convergence(out["convergence"])
    if COEFFICIENTS_KEY in out:
        normalized = normalize_coefficients(out[COEFFICIENTS_KEY])
        if normalized:
            out[COEFFICIENTS_KEY] = normalized
        else:
            out.pop(COEFFICIENTS_KEY, None)
    return out


def _series_is_nonempty(convergence: Any) -> bool:
    """True iff ``convergence`` has at least one axis with data.

    Handles both the raw-list form and the summarized form: a summary with
    ``len > 0`` counts, and a raw list with any element counts.
    """
    if not isinstance(convergence, dict):
        return False
    for arr in convergence.values():
        if _is_series_summary(arr):
            if arr.get("len", 0):
                return True
        elif isinstance(arr, list):
            if len(arr) > 0:
                return True
    return False


def emit_provenance(
    output_dir: str | Path,
    *,
    experiments: dict,
    coefficients: dict | None = None,
    run_id: str | None = None,
    generated_at: str | None = None,
) -> Path:
    """Write ``<output_dir>/provenance.json`` describing what each run did.

    The ``experiments`` mapping is keyed by experiment id; each value carries
    the per-experiment fields the rubric's evidence-visibility leaves want to
    confirm (``model_key``, ``env``, ``baseline``, ``seed``, ``epochs``,
    ``steps``, ``batch_size``, ``per_optimizer``, ``hardware``,
    ``framework_versions``, ``convergence``).  Any ``convergence`` axis longer
    than 32 entries is stored as a compact summary (see
    :func:`_summarize_series`) so the grader's evidence byte-cap is never
    blown by a long training curve.

    ``coefficients`` is the run-global mapping of PAPER-DECLARED ALGORITHMIC
    CONSTANTS actually used by the code — ``{"beta": 10, "lambda": 0.1}`` for
    SDAR's ``g_t = σ(β·Δ_t)`` and its distillation weight.  These are the
    algorithmic invariants a surrogate gets wrong, so recording them is what
    lets a rubric leaf that pins ``β=10`` be graded by a string compare instead
    of an LLM's opinion.  Emit the value your code ACTUALLY USES (pass the same
    Python variable the loss reads) — this manifest is evidence, not a
    restatement of the paper.  A cell that overrides a run-global coefficient
    (an ablation sweeping λ) puts its own value in that experiment's
    ``coefficients`` sub-dict; both addresses resolve.

    Args:
        output_dir:    Directory to write ``provenance.json`` into (created if
                       absent).
        experiments:   ``{exp_id: {...}}`` execution record.  Per-experiment
                       ``convergence`` arrays are summarized in place.
        coefficients:  Optional ``{name: number}`` paper-declared constants.
                       Keys are canonicalized (``"β"`` → ``"beta"``); non-numeric
                       entries are dropped.  Omitted from the manifest entirely
                       when empty, so a caller that passes nothing writes the
                       byte-identical file it wrote before this section existed.
        run_id:        Optional run identifier stamped into the manifest.
        generated_at:  Optional timestamp string.  Left ``None`` by default so
                       the function is deterministic — the caller supplies a
                       clock value when one is wanted.

    Returns:
        The path to the written ``provenance.json`` (returned even on a
        serialization/write error — this helper is **fail-soft** and must
        never raise from the agent's training script).
    """
    out_dir = Path(output_dir)
    target = out_dir / "provenance.json"

    summarized: dict[str, Any] = {}
    figures: list[Any] = []
    try:
        if isinstance(experiments, dict):
            for exp_id, exp in experiments.items():
                summarized[str(exp_id)] = _summarize_experiment(exp)
                # Allow an experiment to carry its own figure descriptors.
                if isinstance(exp, dict):
                    exp_figs = exp.get("figures")
                    if isinstance(exp_figs, list):
                        figures.extend(exp_figs)
    except Exception:  # noqa: BLE001 — fail-soft; never break the training run.
        summarized = {}

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "experiments": summarized,
        "figures": figures,
    }
    # Key omitted when empty — an agent that emits no coefficients writes exactly
    # the manifest it wrote before this section existed.
    normalized_coefficients = normalize_coefficients(coefficients)
    if normalized_coefficients:
        payload[COEFFICIENTS_KEY] = normalized_coefficients

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001 — fail-soft: return the path regardless.
        pass
    return target


# ---------------------------------------------------------------------------
# Issue #3 (2026-06-15): HARNESS-OWNED provenance producer for the cell route.
# emit_provenance (above) is AGENT-facing — the agent must call it. When the agent
# doesn't (the observed case: All-CNN/Adam leaves stuck at 0.7, "weight_decay stated
# not verifiable", "no artifacts confirm the lr search"), the recipe never reaches
# the grader. This builds provenance.json from the MECHANICAL facts the harness
# already controls — the emitted cells (cells.json) + the actual per-cell params in
# the aggregated metrics.json + the staged-search grid — so the eval-protocol leaves
# can confirm the recipe even when the agent skips emit_provenance. Merges (never
# overwrites) an agent-emitted file: agent semantic fields win, the harness fills the
# mechanical params the agent omitted. Fail-soft; stdlib-only.
# ---------------------------------------------------------------------------
_PROVENANCE_PARAM_KEYS: tuple[str, ...] = (
    "lr", "best_lr", "weight_decay", "dropout", "dropout_p", "momentum", "epochs",
    "epochs_total", "epochs_run", "seed", "batch_size", "augment", "use_zca", "num_classes",
)


def _safe_load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — missing/malformed → None (fail-soft).
        return None


def _latest_metrics(code_dir: Path) -> dict | None:
    """Newest-by-mtime aggregated metrics.json under code/ (or code/outputs/*/)."""
    cands: list[Path] = list((code_dir / "outputs").rglob("metrics.json")) if (code_dir / "outputs").is_dir() else []
    top = code_dir / "metrics.json"
    if top.is_file():
        cands.append(top)
    best: Path | None = None
    best_mtime = -1.0
    for p in cands:
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        if mt > best_mtime:
            best_mtime, best = mt, p
    if best is None:
        return None
    d = _safe_load_json(best)
    return d if isinstance(d, dict) else None


def _lookup_cell_metric(per_model: Any, mk: Any, env: Any, baseline: Any) -> dict | None:
    """per_model[mk][env][baseline] when all three axes resolve, else None."""
    try:
        node = per_model[mk][env][baseline]
        return node if isinstance(node, dict) else None
    except (KeyError, TypeError):
        return None


def _summarize_lr_search(search: Any, per_model: Any) -> dict | None:
    """Record the searched grid per group (the eval-protocol leaf wants the SEARCH
    confirmed, not just the winning lr)."""
    if not isinstance(search, list) or not search:
        return None
    grid: set[float] = set()
    groups: list[dict] = []
    for g in search:
        if not isinstance(g, dict):
            continue
        gid = g.get("group") or (g.get("promote") or {}).get("id")
        cand_lrs: list[float] = []
        for c in (g.get("candidates") or []):
            if not isinstance(c, dict):
                continue
            v = (c.get("params") or {}).get("lr", c.get("lr"))
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                cand_lrs.append(float(v))
                grid.add(float(v))
        if gid is not None:
            groups.append({"group": str(gid), "searched_lr": sorted(set(cand_lrs))})
    return {"grid": sorted(grid), "groups": groups} if grid else None


def build_cell_provenance(
    code_dir: str | Path,
    *,
    run_id: str | None = None,
    generated_at: str | None = None,
) -> Path:
    """Write a HARNESS-owned ``provenance.json`` from cells.json + metrics.json.

    Returns the path (even on error — fail-soft). See the module-level note above.
    """
    code = Path(code_dir)
    cells_doc = _safe_load_json(code / "cells.json")
    if isinstance(cells_doc, dict):
        cells = cells_doc.get("cells")
        search = cells_doc.get("search")
    elif isinstance(cells_doc, list):
        cells, search = cells_doc, None
    else:
        cells, search = None, None
    metrics = _latest_metrics(code)
    per_model = metrics.get("per_model") if isinstance(metrics, dict) else None

    experiments: dict[str, Any] = {}
    for cell in (cells or []):
        if not isinstance(cell, dict) or not cell.get("id"):
            continue
        cid = str(cell["id"])
        rec: dict[str, Any] = {}
        for axis in ("model_key", "env", "baseline"):
            if cell.get(axis) is not None:
                rec[axis] = cell[axis]
        flat = {
            **(cell.get("params") if isinstance(cell.get("params"), dict) else {}),
            **{k: v for k, v in cell.items() if k != "params"},
        }
        for k in _PROVENANCE_PARAM_KEYS:
            if k in flat:
                rec[k] = flat[k]
        mrec = _lookup_cell_metric(per_model, rec.get("model_key"), rec.get("env"), rec.get("baseline"))
        if isinstance(mrec, dict):
            for k in _PROVENANCE_PARAM_KEYS:
                if k in mrec and k not in rec:
                    rec[k] = mrec[k]
        experiments[cid] = rec

    # Merge an agent-emitted provenance.json (agent semantic fields preserved).
    existing = _safe_load_json(code / "provenance.json")
    if isinstance(existing, dict) and isinstance(existing.get("experiments"), dict):
        for eid, arec in existing["experiments"].items():
            if isinstance(arec, dict):
                experiments[str(eid)] = {**experiments.get(str(eid), {}), **arec}

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": "harness_cell_provenance",
        "run_id": run_id,
        "generated_at": generated_at,
        "experiments": {k: _summarize_experiment(v) for k, v in experiments.items()},
    }

    # PRESERVE the agent's top-level coefficients block. This function OVERWRITES
    # code/provenance.json, and the harness cannot re-derive a paper-declared
    # coefficient from cells.json/metrics.json (nothing mechanical knows what β is).
    # Without this the cell route would silently DESTROY the only record of the
    # algorithmic constants the agent emitted — the rubric leaf would then resolve
    # to `provenance_missing` and fall back to the LLM, quietly deleting exactly the
    # evidence this contract exists to capture.
    if isinstance(existing, dict):
        preserved = normalize_coefficients(existing.get(COEFFICIENTS_KEY))
        if preserved:
            payload[COEFFICIENTS_KEY] = preserved

    lr_search = _summarize_lr_search(search, per_model)
    if lr_search:
        payload["lr_search"] = lr_search

    target = code / "provenance.json"
    try:
        code.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001 — fail-soft: never break the run.
        pass
    return target


def emit_figure_sidecar(
    png_path: str | Path,
    *,
    shows: str,
    axis: dict,
    series: dict,
) -> Path:
    """Write a ``<png_stem>.json`` sidecar next to ``png_path``.

    The figure-blind grader reads this instead of the image.  ``axis`` is
    normalized to ``{x:{label,scale}, y:{label,scale}}`` and any over-length
    array in ``series`` is summarized (see :func:`_summarize_series`) so a
    long curve does not blow the evidence cap.  The ``axis.scale:"log"`` field
    is exactly what answers *"log-scale axis not verifiable."*

    Args:
        png_path:  Path to the PNG the sidecar describes.  The sidecar is
                   written next to it with the same stem and a ``.json`` suffix.
        shows:     One-line description of what the figure plots.
        axis:      ``{x:{label,scale}, y:{label,scale}}`` (passed through as
                   given; callers should supply both axes).
        series:    ``{name: [values...]}`` plotted series; long arrays are
                   summarized.

    Returns:
        The path to the written sidecar JSON (returned even on error —
        **fail-soft**).
    """
    png = Path(png_path)
    target = png.with_suffix(".json")

    safe_series: dict[str, Any] = {}
    try:
        if isinstance(series, dict):
            for name, arr in series.items():
                safe_series[str(name)] = _summarize_series(arr)
    except Exception:  # noqa: BLE001 — fail-soft.
        safe_series = {}

    payload: dict[str, Any] = {
        "shows": shows,
        "axis": axis if isinstance(axis, dict) else {},
        "series": safe_series,
    }

    try:
        png.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001 — fail-soft.
        pass
    return target


def assert_provenance(output_dir: str | Path, *, require_series: bool = False) -> None:
    """Raise :class:`RubricGuardFailure` if execution provenance is incomplete.

    Self-validation hook for the agent's ``train_cell.py`` (mirror of
    ``rubric_guard.assert_metrics_schema``).  The raised message is a
    JSON-shaped payload so it rides the next iteration's ``repair_context``
    channel.

    Fails when **any** of:

    - ``<output_dir>/provenance.json`` is absent.
    - ``require_series`` is True and **no** experiment has a non-empty
      ``convergence`` series.
    - any ``fig_*.png`` in ``output_dir`` lacks its ``<stem>.json`` sidecar.

    When ``require_series`` is False and the manifest is present (and every
    figure has a sidecar) this is a no-op — light papers that declare no curve
    requirement are best-effort, never hard-failed.

    Args:
        output_dir:     Directory expected to contain ``provenance.json`` and
                        any ``fig_*.png`` + sidecars.
        require_series: When True, at least one experiment must carry a
                        non-empty convergence series (flipped on only by a
                        paper-level curve-requirement signal).

    Raises:
        RubricGuardFailure: With a JSON-shaped message naming the concrete gap.
    """
    out_dir = Path(output_dir)
    manifest = out_dir / "provenance.json"

    if not manifest.is_file():
        raise RubricGuardFailure(
            json.dumps({
                "provenance_guard": "manifest_missing",
                "expected_path": str(manifest),
                "hint": (
                    "Call emit_provenance(output_dir, experiments={...}) at the "
                    "end of the training script so the grader can read execution "
                    "evidence (epochs, batch size, hardware, convergence). "
                    "Without it the evidence-visibility rubric leaves score low."
                ),
            })
        )

    # Read the manifest defensively — a corrupt manifest is itself a violation.
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RubricGuardFailure(
            json.dumps({
                "provenance_guard": "manifest_unreadable",
                "path": str(manifest),
                "error": str(exc),
                "hint": (
                    "provenance.json exists but is not valid JSON. Re-emit it "
                    "with emit_provenance(...) before the script exits."
                ),
            })
        ) from exc

    if require_series:
        experiments = payload.get("experiments") if isinstance(payload, dict) else None
        has_series = False
        if isinstance(experiments, dict):
            for exp in experiments.values():
                if isinstance(exp, dict) and _series_is_nonempty(exp.get("convergence")):
                    has_series = True
                    break
        if not has_series:
            raise RubricGuardFailure(
                json.dumps({
                    "provenance_guard": "series_missing",
                    "hint": (
                        "This paper declares convergence-curve requirements, but "
                        "no experiment in provenance.json carries a non-empty "
                        "convergence series. Record per-step/per-epoch metrics "
                        "(e.g. convergence={'iteration': [...], 'loss': [...]}) "
                        "and pass them through emit_provenance(...)."
                    ),
                })
            )

    # Every fig_*.png must have a <stem>.json sidecar next to it.
    missing_sidecars: list[str] = []
    try:
        pngs = sorted(out_dir.glob("fig_*.png"))
    except OSError:
        pngs = []
    for png in pngs:
        if not png.with_suffix(".json").is_file():
            missing_sidecars.append(png.name)

    if missing_sidecars:
        raise RubricGuardFailure(
            json.dumps({
                "provenance_guard": "figure_sidecar_missing",
                "missing_sidecars": missing_sidecars,
                "hint": (
                    "Each fig_*.png must have a machine-readable <stem>.json "
                    "sidecar (the grader is figure-blind). Call "
                    "emit_figure_sidecar(png_path, shows=..., axis=..., "
                    "series=...) for every figure you save."
                ),
            })
        )


__all__ = [
    "RubricGuardFailure",
    "emit_provenance",
    "emit_figure_sidecar",
    "assert_provenance",
    "build_cell_provenance",
    "SCHEMA_VERSION",
    # Paper-declared coefficients — the namespace + its one address builder.
    "COEFFICIENTS_KEY",
    "canonical_coefficient_name",
    "coefficient_field",
    "normalize_coefficients",
]
