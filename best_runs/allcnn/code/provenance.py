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
  versions, and a convergence series).  A long convergence series is stored as
  a compact ``{len, first, last, min, max, sampled}`` **summary** so it never
  blows the grader's evidence byte-cap.
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
import os
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
    """Return a shallow copy of one experiment with its convergence summarized."""
    if not isinstance(exp, dict):
        return exp
    out = dict(exp)
    if "convergence" in out:
        out["convergence"] = _summarize_convergence(out["convergence"])
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

    Args:
        output_dir:    Directory to write ``provenance.json`` into (created if
                       absent).
        experiments:   ``{exp_id: {...}}`` execution record.  Per-experiment
                       ``convergence`` arrays are summarized in place.
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

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001 — fail-soft: return the path regardless.
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
    "SCHEMA_VERSION",
]
