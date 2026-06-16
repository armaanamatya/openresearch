#!/usr/bin/env python3
"""Regrade-backfill lane (Q6 of the 2026-06-16 grader-fidelity remediation).

Re-grade *all* historical scored runs under a runs root through the post-A
leaf scorer (:func:`backend.evals.paperbench.leaf_scorer.score_reproduction`) so
the leaderboard is uniform under a single ``grader_version``. Once the grader's
nondeterministic noise is denoised (A5 transport + A1 median-of-N) and the
upward-biased best-of-run MAX floor is retired (A3/A4), the *historical* scores
on disk were produced by the OLD (v0) grader and are no longer comparable to a
freshly-graded run. This tool re-grades each saved run under the current grader
and stamps provenance so a leaderboard can honestly compare runs graded the same
way.

What it does, per run dir (and per ``attempts/<ts>/`` sub-dir):

  1. **Skip gracefully** any dir that lacks a rubric tree
     (``rubric_tree.json`` / ``generated_rubric.json``) OR lacks a scored artifact
     (``rubric_evaluation.json`` / ``final_report.json`` carrying a rubric block) —
     there is nothing to re-grade, and that is a normal outcome (interrupted runs,
     ingest-only dirs), not an error.
  2. **Re-grade** the run's COMPLETE on-disk evidence via
     ``score_reproduction(rubric_tree, run_dir, llm_client, degraded=False)`` —
     ``degraded=False`` is passed explicitly so we measure the live, non-degraded
     grading path uniformly (matching ``scripts/calibrate_grader.py``).
  3. **Preserve** the prior v0 artifact **verbatim** in an archive sidecar
     (``rubric_evaluation.v0.json`` / ``final_report.v0.json``) before any write,
     written exactly once (never clobbered on a re-run) — prior data is never
     mutated in place or destroyed.
  4. **Stamp + adopt** the fresh grade onto the rubric block:
       * ``final_report.json``  → fields land on ``report["rubric"]`` (a nested dict).
       * ``rubric_evaluation.json`` → fields land at the top level (flat dict — it
         *is* the enriched ``score_reproduction`` result).
     The stamp is ``grader_version`` (default ``"v1"``), ``grader_samples``
     (``REPROLAB_GRADER_SAMPLES`` or 1), ``grader_temperature`` (0). The fresh
     scoring values (``overall_score``, ``target_score``, ``meets_target`` derived as
     ``overall >= target``, ``leaf_count``, ``graded``, ``coverage_pct``,
     ``compute_adjusted_score`` mirrored from ``overall_score``, ``leaf_scores``,
     ``degraded``, ``rubric_source``) are written so the leaderboard reads the
     uniform v1 number. ``meets_target`` mirrors ``binding.py``'s derivation; we do
     NOT invent a ``compute_scope``.

Safety / determinism
---------------------
* **Dry-run by default.** Nothing is written without ``--apply`` — a bare invocation
  reports exactly what *would* change (which dirs, which artifacts, old->new score)
  and touches no file. This is the operator's "show me first" pass.
* **Injectable grader client.** ``backfill_run`` / ``backfill_root`` take an
  ``llm_client`` so the unit test drives a deterministic stub with zero network.
  The real CLI builds the client the same way ``scripts/calibrate_grader.py`` does
  (Featherless Qwen via ``resolve_root_model`` + ``OpenAILlmClient``); that builder
  and ``score_reproduction`` are imported **lazily** so ``--help`` and the tests
  touch no credentials/network.
* We re-use the published score-extraction helper
  (``backend.services.runs.report_resolution.extract_scores``) read-only to report
  the prior score for the dry-run diff; we never edit any existing file.

Usage
-----
    python -m scripts.regrade_backfill                       # DRY-RUN over runs/
    python -m scripts.regrade_backfill --apply               # write the v1 backfill
    python -m scripts.regrade_backfill --only prj_abc123 --apply
    python -m scripts.regrade_backfill --runs-root /path/to/runs --grader-version v1 --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

# Filenames tried (in order) when resolving the rubric tree for a run dir.
_RUBRIC_CANDIDATES = ("rubric_tree.json", "generated_rubric.json")

# The two scored-artifact shapes we stamp, with the JSON path their rubric block
# lives at: final_report.json nests under ["rubric"]; rubric_evaluation.json is
# itself the flat enriched score dict.
_FINAL_REPORT = "final_report.json"
_RUBRIC_EVAL = "rubric_evaluation.json"

# Default grader-provenance stamp values.
_DEFAULT_GRADER_VERSION = "v1"
_DEFAULT_GRADER_TEMPERATURE = 0


class _LlmClient(Protocol):
    """Grader transport contract (mirror of leaf_scorer.LlmClient).

    Re-declared locally so this module imports without dragging the scorer in at
    import time (``score_reproduction`` is imported lazily). The unit-test stub
    satisfies this.
    """

    def complete(self, *, system: str, user: str) -> str: ...


# ---------------------------------------------------------------------------
# Provenance / value helpers (pure)
# ---------------------------------------------------------------------------


def _grader_samples_from_env() -> int:
    """Number of grader samples to record, from ``REPROLAB_GRADER_SAMPLES`` (default 1).

    Pure read of the env at call time; an unparseable / non-positive value falls
    back to 1 rather than raising (the stamp is provenance, not a control knob).
    """
    raw = os.environ.get("REPROLAB_GRADER_SAMPLES", "")
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        return 1
    return n if n >= 1 else 1


def build_stamp(
    grader_version: str,
    *,
    grader_samples: Optional[int] = None,
    grader_temperature: int = _DEFAULT_GRADER_TEMPERATURE,
) -> dict[str, Any]:
    """Return the provenance stamp dict applied to a rubric block.

    ``grader_samples`` defaults to :func:`_grader_samples_from_env` when omitted so
    the CLI honours ``REPROLAB_GRADER_SAMPLES`` while tests can pin it explicitly.
    """
    return {
        "grader_version": grader_version,
        "grader_samples": (
            grader_samples if grader_samples is not None else _grader_samples_from_env()
        ),
        "grader_temperature": grader_temperature,
    }


def _derive_meets_target(overall: Any, target: Any) -> Optional[bool]:
    """Mirror binding.py: meets_target = overall >= target, None if target unknown."""
    if target is None:
        return None
    try:
        return bool(float(overall) >= float(target))
    except (TypeError, ValueError):
        return None


def _scoring_fields_from_result(result: dict[str, Any]) -> dict[str, Any]:
    """Map a ``score_reproduction`` result to the rubric-block scoring fields.

    Mirrors the field set ``binding.py`` persists into ``rubric_evaluation.json``
    (and that ``report.py`` merges into ``final_report.json``'s rubric block), plus
    the two derived values: ``meets_target`` (overall >= target) and
    ``compute_adjusted_score`` (mirrors ``overall_score`` — the scorer returns no
    separate adjusted value, and the leaderboard falls back overall->adjusted).
    """
    overall = result.get("overall_score")
    target = result.get("target_score")
    fields: dict[str, Any] = {
        "overall_score": overall,
        "target_score": target,
        "meets_target": _derive_meets_target(overall, target),
        "leaf_count": result.get("leaf_count"),
        "graded": result.get("graded"),
        "coverage_pct": result.get("coverage_pct"),
        "degraded": result.get("degraded"),
        "rubric_source": result.get("rubric_source"),
        "compute_adjusted_score": overall,
        "leaf_scores": result.get("leaf_scores", []),
    }
    return fields


def apply_to_final_report(
    report: dict[str, Any], result: dict[str, Any], stamp: dict[str, Any]
) -> dict[str, Any]:
    """Return ``report`` with the fresh grade + stamp merged onto ``report["rubric"]``.

    Mutates and returns the same dict (the caller has already deep-loaded it from
    disk; the v0 copy is archived separately before this runs). A missing/non-dict
    ``rubric`` block is created so a flat-schema report still receives the stamp.
    """
    rubric = report.get("rubric")
    if not isinstance(rubric, dict):
        rubric = {}
        report["rubric"] = rubric
    rubric.update(_scoring_fields_from_result(result))
    rubric.update(stamp)
    return report


def apply_to_rubric_eval(
    payload: dict[str, Any], result: dict[str, Any], stamp: dict[str, Any]
) -> dict[str, Any]:
    """Return ``payload`` with the fresh grade + stamp merged at the TOP level.

    ``rubric_evaluation.json`` is the flat enriched ``score_reproduction`` result,
    so scoring fields and the stamp both land at the top level (no nesting).
    Preserves any non-scoring keys already present (e.g. ``iteration``,
    ``compute_scope``, ``areas``, ``weak_leaves``).
    """
    payload.update(_scoring_fields_from_result(result))
    payload.update(stamp)
    return payload


# ---------------------------------------------------------------------------
# Resolution / IO helpers
# ---------------------------------------------------------------------------


def resolve_rubric_tree(run_dir: Path) -> Optional[dict[str, Any]]:
    """Load the run's rubric tree, or ``None`` if neither candidate file exists.

    Returns the parsed dict so the caller re-grades against it. A present-but-corrupt
    rubric file returns ``None`` (treated as un-gradeable / skip) rather than raising
    — the backfill must never abort the whole sweep on one bad dir.
    """
    for name in _RUBRIC_CANDIDATES:
        candidate = run_dir / name
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
            if isinstance(data, dict):
                return data
            return None
    return None


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    """Parse a JSON object file, or ``None`` if missing/corrupt/not-a-dict."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _has_scored_rubric_block(report: Optional[dict[str, Any]]) -> bool:
    """True if a final_report dict carries a rubric block with a numeric score.

    Uses the published ``extract_scores`` so we match the leaderboard's own notion
    of "this report has a score" (handles nested ``rubric`` + flat ``rubric_score``).
    """
    if not report:
        return False
    from backend.services.runs.report_resolution import extract_scores

    overall, adjusted = extract_scores(report)
    return overall is not None or adjusted is not None


def _v0_sidecar_path(artifact_path: Path) -> Path:
    """``foo.json`` -> ``foo.v0.json`` (the archive of the prior, pre-backfill value)."""
    return artifact_path.with_suffix(".v0" + artifact_path.suffix)


def _archive_v0_once(artifact_path: Path) -> bool:
    """Copy ``artifact_path`` verbatim to its ``.v0`` sidecar iff not already archived.

    Byte-for-byte copy (so the historical value is preserved exactly, including any
    fields we don't re-derive). Returns True if a new sidecar was created, False if
    one already existed (idempotent — a second backfill never overwrites the v0
    snapshot, which would let a v1 number masquerade as the original).
    """
    sidecar = _v0_sidecar_path(artifact_path)
    if sidecar.exists():
        return False
    sidecar.write_bytes(artifact_path.read_bytes())
    return True


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically persist ``data`` as pretty JSON (tmp + replace, default=str)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Per-run result records
# ---------------------------------------------------------------------------


@dataclass
class ArtifactOutcome:
    """What happened (or would happen) to one scored artifact in a run dir."""

    name: str  # "final_report.json" / "rubric_evaluation.json"
    path: Path
    old_score: Optional[float]
    new_score: Optional[float]
    v0_archived: bool  # a .v0 sidecar was (or would be) newly created
    written: bool  # the artifact was stamped+rewritten (False in dry-run)


@dataclass
class RunOutcome:
    """The backfill outcome for a single run dir (skipped or regraded)."""

    run_dir: Path
    status: str  # "regraded" | "skipped"
    reason: str = ""  # populated when skipped
    artifacts: list[ArtifactOutcome] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core: backfill one run dir
# ---------------------------------------------------------------------------


def backfill_run(
    run_dir: Path,
    llm_client: _LlmClient,
    *,
    grader_version: str = _DEFAULT_GRADER_VERSION,
    grader_samples: Optional[int] = None,
    grader_temperature: int = _DEFAULT_GRADER_TEMPERATURE,
    apply: bool = False,
    score_fn: Optional[Callable[..., dict[str, Any]]] = None,
) -> RunOutcome:
    """Re-grade a single run dir and stamp ``grader_version`` onto its rubric blocks.

    Skips gracefully (``status="skipped"``) when the dir has no rubric tree or no
    scored artifact. Otherwise re-grades ONCE via ``score_fn`` (default the real
    ``score_reproduction``, ``degraded=False``) and applies the fresh grade + stamp
    to whichever of ``final_report.json`` / ``rubric_evaluation.json`` carry a score.

    Honours ``apply``: when False (the default), nothing is written — each
    :class:`ArtifactOutcome` reports the old->new score and ``written=False`` so the
    caller can print a dry-run diff. The grader IS invoked even in dry-run (that is
    how we compute the would-be new score); pass a stub client in tests.
    """
    rubric_tree = resolve_rubric_tree(run_dir)
    if rubric_tree is None:
        return RunOutcome(run_dir, "skipped", reason="no rubric tree")

    final_report = _load_json(run_dir / _FINAL_REPORT)
    rubric_eval = _load_json(run_dir / _RUBRIC_EVAL)

    has_final = _has_scored_rubric_block(final_report)
    has_eval = rubric_eval is not None and rubric_eval.get("overall_score") is not None
    if not has_final and not has_eval:
        return RunOutcome(run_dir, "skipped", reason="no scored artifact")

    if score_fn is None:
        from backend.evals.paperbench.leaf_scorer import (  # type: ignore
            score_reproduction as score_fn,
        )

    # Infer rubric_source from which rubric file is present, mirroring
    # calibrate_grader: generated_rubric.json => "generated", else bundle default.
    rubric_source = (
        "generated" if (run_dir / "generated_rubric.json").exists() else "paperbench_bundle"
    )

    result = score_fn(
        rubric_tree,
        run_dir,
        llm_client,
        rubric_source=rubric_source,
        degraded=False,
    )
    new_score = result.get("overall_score")
    stamp = build_stamp(
        grader_version,
        grader_samples=grader_samples,
        grader_temperature=grader_temperature,
    )

    outcomes: list[ArtifactOutcome] = []

    if has_final:
        from backend.services.runs.report_resolution import extract_scores

        old_overall, _ = extract_scores(final_report or {})
        path = run_dir / _FINAL_REPORT
        archived = False
        written = False
        if apply:
            archived = _archive_v0_once(path)
            apply_to_final_report(final_report, result, stamp)  # type: ignore[arg-type]
            _write_json(path, final_report)  # type: ignore[arg-type]
            written = True
        else:
            # Report whether a v0 archive WOULD be created (none exists yet).
            archived = not _v0_sidecar_path(path).exists()
        outcomes.append(
            ArtifactOutcome(_FINAL_REPORT, path, old_overall, new_score, archived, written)
        )

    if has_eval:
        old_overall = rubric_eval.get("overall_score") if rubric_eval else None  # type: ignore[union-attr]
        path = run_dir / _RUBRIC_EVAL
        archived = False
        written = False
        if apply:
            archived = _archive_v0_once(path)
            apply_to_rubric_eval(rubric_eval, result, stamp)  # type: ignore[arg-type]
            _write_json(path, rubric_eval)  # type: ignore[arg-type]
            written = True
        else:
            archived = not _v0_sidecar_path(path).exists()
        outcomes.append(
            ArtifactOutcome(_RUBRIC_EVAL, path, old_overall, new_score, archived, written)
        )

    return RunOutcome(run_dir, "regraded", artifacts=outcomes)


# ---------------------------------------------------------------------------
# Sweep: discover candidate run dirs under a root
# ---------------------------------------------------------------------------


def discover_run_dirs(runs_root: Path, only: Optional[str] = None) -> list[Path]:
    """Return candidate run dirs: each top-level run + each ``attempts/<ts>/`` sub-dir.

    A run's best-scoring attempt may live in ``attempts/*`` (attempt isolation), and
    each attempt carries its own ``rubric_evaluation.json`` / ``final_report.json``
    (``attempt_isolation.PER_ATTEMPT_SIDECARS``), so we stamp those too for a uniform
    leaderboard. ``only`` filters to the run dir whose name equals the given run id,
    INCLUDING all of its ``attempts/*`` sub-dirs. Dirs are returned sorted for
    deterministic output.
    """
    if not runs_root.exists():
        return []
    dirs: list[Path] = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        # `only` scopes to one run id (the top-level dir name); its attempts ride along.
        if only is not None and child.name != only:
            continue
        # Top-level run dir.
        dirs.append(child)
        # Attempt sub-dirs (one level under <run>/attempts/).
        attempts = child / "attempts"
        if attempts.is_dir():
            for att in sorted(attempts.iterdir()):
                if att.is_dir():
                    dirs.append(att)
    return dirs


def backfill_root(
    runs_root: Path,
    llm_client: _LlmClient,
    *,
    grader_version: str = _DEFAULT_GRADER_VERSION,
    grader_samples: Optional[int] = None,
    grader_temperature: int = _DEFAULT_GRADER_TEMPERATURE,
    apply: bool = False,
    only: Optional[str] = None,
    score_fn: Optional[Callable[..., dict[str, Any]]] = None,
) -> list[RunOutcome]:
    """Backfill every discovered run dir under ``runs_root``; return all outcomes."""
    outcomes: list[RunOutcome] = []
    for run_dir in discover_run_dirs(runs_root, only=only):
        outcomes.append(
            backfill_run(
                run_dir,
                llm_client,
                grader_version=grader_version,
                grader_samples=grader_samples,
                grader_temperature=grader_temperature,
                apply=apply,
                score_fn=score_fn,
            )
        )
    return outcomes


# ---------------------------------------------------------------------------
# Real grader client (lazy; identical pattern to scripts/calibrate_grader.py)
# ---------------------------------------------------------------------------


def _build_real_llm_client() -> _LlmClient:
    """Build the grading client from env, identical to scripts/calibrate_grader.py.

    Featherless Qwen root via ``resolve_root_model`` + ``OpenAILlmClient`` (same
    backend a normal scored run uses; needs only ``FEATHERLESS_API_KEY``). Imported
    lazily so ``--help`` and unit tests never touch network/credential code. We do
    NOT edit the client builder — we reuse the existing public constructors.
    """
    from backend.agents.rlm.models import resolve_root_model
    from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

    root = resolve_root_model("qwen3-coder-featherless")
    bk = root.backend_kwargs
    return OpenAILlmClient(
        model=bk["model_name"], api_key=bk["api_key"], base_url=bk["base_url"]
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fmt_score(v: Optional[float]) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "  n/a "


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.regrade_backfill",
        description=(
            "Re-grade all historical scored runs under a runs root to a uniform "
            "grader_version (default DRY-RUN; pass --apply to write)."
        ),
    )
    p.add_argument(
        "--runs-root",
        default="runs",
        help="root directory of run dirs to backfill (default: runs)",
    )
    p.add_argument(
        "--grader-version",
        default=_DEFAULT_GRADER_VERSION,
        help=f"value stamped as grader_version (default {_DEFAULT_GRADER_VERSION!r})",
    )
    p.add_argument(
        "--only",
        default=None,
        help="restrict to a single run id (covers its attempts/* too)",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="WRITE the backfill (stamp + adopt fresh scores, archiving v0 sidecars). "
        "Without this flag the tool is a dry-run and writes nothing.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="explicit no-write mode (the default); accepted for clarity. "
        "If both --apply and --dry-run are given, --dry-run wins (safe).",
    )
    return p


def _print_outcomes(outcomes: list[RunOutcome], *, apply: bool) -> None:
    mode = "APPLY" if apply else "DRY-RUN (no files written)"
    print(f"regrade_backfill [{mode}]")
    regraded = [o for o in outcomes if o.status == "regraded"]
    skipped = [o for o in outcomes if o.status == "skipped"]
    for o in regraded:
        print(f"\n  {o.run_dir}")
        for a in o.artifacts:
            arrow = f"{_fmt_score(a.old_score)} -> {_fmt_score(a.new_score)}"
            v0 = "archive v0" if a.v0_archived else "v0 exists"
            act = "written" if a.written else "would write"
            print(f"      {a.name:<24} {arrow}   [{v0}; {act}]")
    print(
        f"\nsummary: {len(regraded)} regraded, {len(skipped)} skipped "
        f"(of {len(outcomes)} dirs scanned)"
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    # Dry-run is the default and wins any ambiguity (fail safe).
    apply = bool(args.apply) and not bool(args.dry_run)

    runs_root = Path(args.runs_root).resolve()
    if not runs_root.exists():
        print(f"error: runs root does not exist: {runs_root}", file=sys.stderr)
        return 1

    # .env is only needed for the real grader client; load it lazily so --help is free.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # pragma: no cover - dotenv always present in this repo
        pass

    llm_client = _build_real_llm_client()

    outcomes = backfill_root(
        runs_root,
        llm_client,
        grader_version=args.grader_version,
        apply=apply,
        only=args.only,
    )
    _print_outcomes(outcomes, apply=apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
