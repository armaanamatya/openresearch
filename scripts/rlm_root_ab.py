"""A/B measurement harness for the OpenResearch RLM root-orchestration loop.

Measures the effect of enabling ``OPENRESEARCH_ARG_CONTRACTS`` +
``OPENRESEARCH_STUB_METRICS_GUARD`` on the root loop health metrics derived
from ``dashboard_events.jsonl`` + ``final_report.json``.

**Credentials / cost note**
Both env vars are consumed by the live CLI driver (``main()``), which calls
``python -m backend.cli reproduce ...`` as a subprocess.  The default root
model is ``gpt-chat-latest``, which requires:

  AZURE_FOUNDRY_ENDPOINT   – the ``…/openai/v1`` base URL
  AZURE_FOUNDRY_DEPLOYMENT – the deployment name (e.g. ``gpt-chat-latest``)
  AZURE_FOUNDRY_API_KEY    – Bearer-auth API key

Each ``--trials N`` × 2 arms run costs a small amount (CPU root, no GPU).
This is the **Tier-B validation tool** that gates any default-flip of the
guard flags (alongside the repo's A/B-≥3-paired-runs rule).

The "executor-stubbed CPU-only" ideal is a future refinement (would need a
partial-primitive-stub mode — today it runs the real pipeline on the chosen
paper).

**Importing this module never spawns subprocesses.**  Live execution is
gated behind the ``main()`` CLI entrypoint only.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import uuid
from collections import Counter
from typing import Any


# ---------------------------------------------------------------------------
# parse_run_metrics — pure, fail-soft
# ---------------------------------------------------------------------------

def parse_run_metrics(run_dir: pathlib.Path) -> dict:
    """Parse one run's dashboard_events.jsonl + final_report.json into
    root-orchestration metrics.

    Pure + fail-soft: missing files, garbled lines, and unexpected shapes all
    degrade gracefully (missing counts become 0, missing scalars become None).

    Event shapes observed in real runs
    -----------------------------------
    There are **three** ``run_warning`` wire shapes:

    1. Top-level ``code``  (newer harness events):
       ``{"event": "run_warning", "code": "<code>", "message": "...", ...}``

    2. Nested ``data.code``  (older harness events):
       ``{"event": "run_warning", "data": {"code": "<code>", "message": "..."}}``

    3. Nested ``data.reason`` (infrastructure/watchdog events, no ``code``):
       ``{"event": "run_warning", "data": {"reason": "stale_run", ...}}``

    All three are tallied by their ``code`` (shapes 1+2) or their ``reason``
    (shape 3, stored under the ``reason`` key so they don't crowd the code
    namespace).

    ``primitive_call`` events carry:
      ``{"event": "primitive_call", "primitive": "<name>", "status": "start"|"ok"|"error", ...}``

    ``experiment_completed`` events carry a ``data.failure_class`` field that
    surfaces ``"arg_contract"`` when the arg-contracts guard fires.

    ``run_complete`` carries: ``{"event": "run_complete", "iterations": N, ...}``

    ``repl_iteration`` events (``iteration`` field) mark root iterations.

    ``final_report.json`` schema:
      Top level: ``verdict``, ``iterations``
      Nested:    ``rubric.overall_score``, ``rubric.meets_target``
      (``overall_score``/``meets_target`` at top-level are usually ``None``.)
    """
    run_dir = pathlib.Path(run_dir)

    run_warning_counts: Counter = Counter()
    arg_contract_blocks: int = 0
    fabrication_suspected: int = 0
    max_repl_iteration: int = 0
    run_complete_iterations: int | None = None
    first_implement_baseline_iteration: int | None = None
    repl_iter_at_first_impl: int = 0  # running repl_iteration counter up to first impl

    events_path = run_dir / "dashboard_events.jsonl"
    if events_path.exists():
        try:
            with events_path.open(encoding="utf-8", errors="replace") as fh:
                repl_iter_count = 0
                seen_first_impl = False
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj: dict = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    event_type: str = obj.get("event") or obj.get("type") or ""

                    # --- repl_iteration: track running count ---
                    if event_type == "repl_iteration":
                        it = obj.get("iteration")
                        if isinstance(it, int):
                            if it > max_repl_iteration:
                                max_repl_iteration = it
                        repl_iter_count += 1

                    # --- run_warning: tally by code (three wire shapes) ---
                    elif event_type == "run_warning":
                        # Shape 1: code at top level
                        code = obj.get("code")
                        if code:
                            run_warning_counts[str(code)] += 1
                            if "fabrication" in str(code):
                                fabrication_suspected += 1
                        else:
                            # Shapes 2 + 3: code/reason inside 'data'
                            data = obj.get("data")
                            if isinstance(data, dict):
                                inner_code = data.get("code")
                                inner_reason = data.get("reason")
                                if inner_code:
                                    run_warning_counts[str(inner_code)] += 1
                                    if "fabrication" in str(inner_code):
                                        fabrication_suspected += 1
                                elif inner_reason:
                                    run_warning_counts[str(inner_reason)] += 1

                    # --- primitive_call: detect first implement_baseline ---
                    elif event_type == "primitive_call":
                        prim = obj.get("primitive", "")
                        status = obj.get("status", "")
                        if (
                            prim == "implement_baseline"
                            and status == "start"
                            and not seen_first_impl
                        ):
                            seen_first_impl = True
                            # Record the repl_iteration count at this point
                            repl_iter_at_first_impl = repl_iter_count
                            # Use max observed iteration field (may be None for
                            # most runs; fall back to ordinal event count)
                            first_implement_baseline_iteration = (
                                max_repl_iteration if max_repl_iteration else repl_iter_count
                            )

                    # --- experiment_completed: surface failure_class ---
                    elif event_type == "experiment_completed":
                        data = obj.get("data", {})
                        if isinstance(data, dict):
                            fc = data.get("failure_class", "")
                            if fc == "arg_contract":
                                arg_contract_blocks += 1
                            if "fabrication" in str(fc):
                                fabrication_suspected += 1

                    # --- run_complete: authoritative iterations count ---
                    elif event_type == "run_complete":
                        iters = obj.get("iterations")
                        if isinstance(iters, int):
                            run_complete_iterations = iters

        except OSError:
            pass

    # Reconcile iteration count: run_complete.iterations is authoritative;
    # fall back to max repl_iteration field seen; then to ordinal event count.
    if run_complete_iterations is not None:
        iterations: int | None = run_complete_iterations
    elif max_repl_iteration > 0:
        iterations = max_repl_iteration
    else:
        iterations = None

    # --- final_report.json ---
    verdict: str | None = None
    overall_score: float | None = None
    meets_target: bool | None = None

    report_path = run_dir / "final_report.json"
    if report_path.exists():
        try:
            with report_path.open(encoding="utf-8", errors="replace") as fh:
                report: dict = json.load(fh)
            verdict = report.get("verdict")
            # Top-level overall_score / meets_target are usually None;
            # the canonical values live in report.rubric.
            overall_score = report.get("overall_score")
            meets_target = report.get("meets_target")
            rubric = report.get("rubric")
            if isinstance(rubric, dict):
                if overall_score is None:
                    overall_score = rubric.get("overall_score")
                if meets_target is None:
                    meets_target = rubric.get("meets_target")
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    return {
        "run_warning_counts": dict(run_warning_counts),
        "arg_contract_blocks": arg_contract_blocks,
        "fabrication_suspected": fabrication_suspected,
        "iterations": iterations,
        "first_implement_baseline_iteration": first_implement_baseline_iteration,
        "verdict": verdict,
        "overall_score": overall_score,
        "meets_target": meets_target,
    }


# ---------------------------------------------------------------------------
# aggregate — mean/sum per metric across an arm's trials
# ---------------------------------------------------------------------------

def aggregate(run_metrics_list: list[dict]) -> dict:
    """Aggregate metrics across multiple trials of one arm.

    Counts → mean; overall_score → mean (None excluded);
    verdict → distribution dict; run_warning_counts → mean per code;
    iterations / first_implement_baseline_iteration → mean (None excluded).
    """
    if not run_metrics_list:
        return {}

    n = len(run_metrics_list)

    # Verdict distribution
    verdicts: Counter = Counter()
    for m in run_metrics_list:
        v = m.get("verdict")
        if v is not None:
            verdicts[v] += 1

    # Scalar means (skip None)
    def _mean(key: str) -> float | None:
        vals = [m[key] for m in run_metrics_list if m.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    # Warning code means
    all_codes: set[str] = set()
    for m in run_metrics_list:
        all_codes.update((m.get("run_warning_counts") or {}).keys())

    warning_means: dict[str, float] = {}
    for code in sorted(all_codes):
        total = sum(
            (m.get("run_warning_counts") or {}).get(code, 0)
            for m in run_metrics_list
        )
        warning_means[code] = total / n

    return {
        "n": n,
        "run_warning_counts_mean": warning_means,
        "arg_contract_blocks_mean": sum(m.get("arg_contract_blocks", 0) for m in run_metrics_list) / n,
        "fabrication_suspected_mean": sum(m.get("fabrication_suspected", 0) for m in run_metrics_list) / n,
        "iterations_mean": _mean("iterations"),
        "first_implement_baseline_iteration_mean": _mean("first_implement_baseline_iteration"),
        "overall_score_mean": _mean("overall_score"),
        "verdict_distribution": dict(verdicts),
        "meets_target_count": sum(1 for m in run_metrics_list if m.get("meets_target")),
    }


# ---------------------------------------------------------------------------
# _print_table — console summary
# ---------------------------------------------------------------------------

def _print_table(arm_results: dict[str, dict]) -> None:
    """Print a simple before/after comparison table."""
    arms = list(arm_results.keys())
    metrics_order = [
        "n",
        "iterations_mean",
        "first_implement_baseline_iteration_mean",
        "overall_score_mean",
        "meets_target_count",
        "arg_contract_blocks_mean",
        "fabrication_suspected_mean",
    ]
    col_w = 22
    header = f"{'metric':<40}" + "".join(f"{a:>{col_w}}" for a in arms)
    print(header)
    print("-" * len(header))
    for key in metrics_order:
        row = f"{key:<40}"
        for arm in arms:
            val = arm_results[arm].get(key)
            if val is None:
                row += f"{'—':>{col_w}}"
            elif isinstance(val, float):
                row += f"{val:>{col_w}.3f}"
            else:
                row += f"{str(val):>{col_w}}"
        print(row)

    # Warning codes: all codes across all arms
    all_codes: set[str] = set()
    for agg in arm_results.values():
        all_codes.update((agg.get("run_warning_counts_mean") or {}).keys())
    if all_codes:
        print()
        print("run_warning_counts_mean:")
        for code in sorted(all_codes):
            row = f"  {code:<38}"
            for arm in arms:
                val = (arm_results[arm].get("run_warning_counts_mean") or {}).get(code, 0.0)
                row += f"{val:>{col_w}.2f}"
            print(row)

    print()
    for arm in arms:
        print(f"  {arm} verdict_distribution: {arm_results[arm].get('verdict_distribution', {})}")


# ---------------------------------------------------------------------------
# _write_report — JSON + markdown
# ---------------------------------------------------------------------------

def _write_report(out_dir: pathlib.Path, paper: str, arm_results: dict[str, dict], model: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "paper": paper,
        "model": model,
        "arms": arm_results,
    }
    (out_dir / "root_ab_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    arms = list(arm_results.keys())
    lines = [
        "# RLM Root A/B Report",
        "",
        f"**Paper:** {paper}  **Model:** {model}",
        "",
        "| metric | " + " | ".join(arms) + " |",
        "|--------|" + "|".join(["-----"] * len(arms)) + "|",
    ]
    scalar_keys = [
        "n", "iterations_mean", "first_implement_baseline_iteration_mean",
        "overall_score_mean", "meets_target_count",
        "arg_contract_blocks_mean", "fabrication_suspected_mean",
    ]
    for key in scalar_keys:
        cells = []
        for arm in arms:
            val = arm_results[arm].get(key)
            cells.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        lines.append(f"| {key} | " + " | ".join(cells) + " |")

    lines += ["", "## Verdict distributions", ""]
    for arm in arms:
        lines.append(f"- **{arm}**: {arm_results[arm].get('verdict_distribution', {})}")

    lines += ["", "## Warning code means", ""]
    all_codes: set[str] = set()
    for agg in arm_results.values():
        all_codes.update((agg.get("run_warning_counts_mean") or {}).keys())
    if all_codes:
        lines.append("| code | " + " | ".join(arms) + " |")
        lines.append("|------|" + "|".join(["----"] * len(arms)) + "|")
        for code in sorted(all_codes):
            cells = [
                f"{(arm_results[arm].get('run_warning_counts_mean') or {}).get(code, 0.0):.2f}"
                for arm in arms
            ]
            lines.append(f"| {code} | " + " | ".join(cells) + " |")

    (out_dir / "root_ab_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI driver — only executes inside main(); importing is always safe
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            A/B measurement harness for the OpenResearch RLM root loop.

            Runs two arms (control vs guarded) each --trials times, parses
            the resulting run dirs, and writes a comparison report.

            Requires:
              AZURE_FOUNDRY_ENDPOINT / AZURE_FOUNDRY_DEPLOYMENT / AZURE_FOUNDRY_API_KEY
              (or whichever credentials match --model).
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--paper", required=True,
                        help="arXiv ID (e.g. 2605.15155) or path to PDF")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per arm (default: 3)")
    parser.add_argument("--model", default="gpt-chat-latest",
                        help="Root model token (default: gpt-chat-latest)")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: runs/_ab/<paper>-<uuid8>)")
    args = parser.parse_args(argv)

    paper: str = args.paper
    trials: int = args.trials
    model: str = args.model

    # Resolve output dir
    key = f"{paper.replace('/', '-')}-{uuid.uuid4().hex[:8]}"
    out_dir = pathlib.Path(args.out) if args.out else pathlib.Path("runs/_ab") / key

    # Arm definitions: env-var overlays on top of the current environment
    arm_env_overrides: dict[str, dict[str, str]] = {
        "control": {
            # Guards off: unset the flag vars by removing them from env
            # (we explicitly delete them below when building the subprocess env)
        },
        "guarded": {
            "OPENRESEARCH_ARG_CONTRACTS": "1",
            "OPENRESEARCH_STUB_METRICS_GUARD": "1",
        },
    }

    runs_root = pathlib.Path("runs")
    arm_results: dict[str, Any] = {}

    for arm_name, overrides in arm_env_overrides.items():
        print(f"\n=== Arm: {arm_name} ({trials} trial(s)) ===")
        trial_metrics: list[dict] = []

        for trial_idx in range(trials):
            project_id = f"ab_{arm_name}_{uuid.uuid4().hex[:12]}"
            print(f"  Trial {trial_idx + 1}/{trials}  project_id={project_id}")

            # Build subprocess environment
            env = os.environ.copy()
            # Control arm: strip both guard flags so they're truly off
            for key_to_strip in ("OPENRESEARCH_ARG_CONTRACTS", "OPENRESEARCH_STUB_METRICS_GUARD"):
                env.pop(key_to_strip, None)
            # Apply arm-specific overrides
            env.update(overrides)

            cmd = [
                sys.executable, "-m", "backend.cli", "reproduce", paper,
                "--model", model,
                "--sandbox", "local",
                "--project-id", project_id,
            ]
            try:
                result = subprocess.run(cmd, env=env, check=False)
                if result.returncode != 0:
                    print(f"    WARNING: cli exited with code {result.returncode}")
            except Exception as exc:
                print(f"    ERROR: subprocess failed: {exc}")

            run_dir = runs_root / project_id
            metrics = parse_run_metrics(run_dir)
            trial_metrics.append(metrics)
            print(f"    verdict={metrics.get('verdict')}  "
                  f"score={metrics.get('overall_score')}  "
                  f"iters={metrics.get('iterations')}")

        arm_results[arm_name] = aggregate(trial_metrics)

    print("\n=== A/B Summary ===\n")
    _print_table(arm_results)

    _write_report(out_dir, paper=paper, arm_results=arm_results, model=model)
    print(f"\nReport written to: {out_dir}/root_ab_report.{{json,md}}")


if __name__ == "__main__":
    main()
