#!/usr/bin/env python3
"""Programmatic trace miner — Phase 1 of the optimization platform.

Reads every ``runs/<id>/`` directory, aggregates structured stats, and
emits ``findings/<date>-stats-<n>.md``. NO LLM calls — pure Python
aggregation over JSON / JSONL / text artifacts.

The output is a stats document; a narrative-findings section is
appended separately by a human reviewer (or by the current Claude
session) after reading the stats. See
``docs/superpowers/specs/2026-05-27-optimization-platform-design.md``
§3 Phase 1.

Usage:
    python scripts/mine_traces.py
    python scripts/mine_traces.py --runs-dir runs/ --out findings/2026-05-27-stats-001.md
    python scripts/mine_traces.py --top-k 15 --cluster-threshold 0.80
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ── Data shapes ──────────────────────────────────────────────────────────────


@dataclass
class RunSummary:
    run_id: str
    path: Path
    paper_title: str | None = None
    paper_source: str | None = None
    status: str | None = None
    verdict: str | None = None
    rubric_overall: float | None = None
    cost_usd: float | None = None
    wall_clock_s: float | None = None
    iterations: int | None = None
    exit_reason: str | None = None
    error_text: str | None = None
    mode: str | None = None
    sandbox: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    event_count: int = 0
    primitive_counter: Counter[str] = field(default_factory=Counter)
    experiment_success_count: int = 0
    experiment_failure_count: int = 0
    repair_loops: dict[str, int] = field(default_factory=dict)
    forced_iteration_warnings: int = 0
    user_messages: int = 0
    stderr_tail: str | None = None


@dataclass
class ErrorCluster:
    representative: str
    members: list[tuple[str, str]]  # (run_id, error_snippet)

    @property
    def count(self) -> int:
        return len(self.members)


# ── Parsing ──────────────────────────────────────────────────────────────────


def _safe_load_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _safe_iter_jsonl(p: Path):
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    except OSError:
        return


def _wall_clock(started: str | None, completed: str | None) -> float | None:
    if not started or not completed:
        return None
    try:
        s = datetime.fromisoformat(started.replace("Z", "+00:00"))
        c = datetime.fromisoformat(completed.replace("Z", "+00:00"))
        return (c - s).total_seconds()
    except (ValueError, TypeError):
        return None


def _classify_exit(status: str | None, error: str | None, stderr_tail: str) -> str:
    if status == "completed":
        return "completed"
    text = (error or "") + " " + (stderr_tail or "")
    if "exit code -9" in text or "SIGKILL" in text or "out of memory" in text.lower():
        return "oom_killed"
    if "wall_clock" in text.lower() or "timeout" in text.lower():
        return "wall_clock_exceeded"
    if "budget" in text.lower() and "exceed" in text.lower():
        return "budget_exceeded"
    if "Pipeline exited with status 3" in text:
        return "pipeline_status_3"
    if "Pipeline exited with status" in text:
        m = re.search(r"Pipeline exited with status (\d+)", text)
        return f"pipeline_status_{m.group(1)}" if m else "pipeline_failed"
    if status == "failed":
        return "failed_other"
    return status or "unknown"


def parse_run(run_dir: Path) -> RunSummary:
    rs = RunSummary(run_id=run_dir.name, path=run_dir)

    # demo_status.json — UI-facing snapshot, most reliable for timing + source
    status = _safe_load_json(run_dir / "demo_status.json") or {}
    rs.status = status.get("status")
    rs.started_at = status.get("startedAt")
    rs.completed_at = status.get("completedAt") or status.get("updatedAt")
    rs.sandbox = status.get("sandboxMode")
    rs.mode = status.get("runMode")
    src = status.get("sourcePdf") or {}
    rs.paper_title = src.get("title") or status.get("sourceLabel")
    rs.paper_source = status.get("sourceKind")
    rs.error_text = status.get("error")
    rs.wall_clock_s = _wall_clock(rs.started_at, rs.completed_at)

    # final_report.json — authoritative outcome + scores
    fr = _safe_load_json(run_dir / "final_report.json") or {}
    rs.verdict = fr.get("verdict")
    rubric = fr.get("rubric") or {}
    rs.rubric_overall = rubric.get("overall_score")
    rs.iterations = fr.get("iterations")
    cost = fr.get("cost") or {}
    rs.cost_usd = (cost.get("llm_usd") or 0.0) + (cost.get("primitives") or 0.0)
    if not rs.mode:
        rs.mode = fr.get("mode")

    # dashboard_events.jsonl — event stream
    for ev in _safe_iter_jsonl(run_dir / "dashboard_events.jsonl"):
        rs.event_count += 1
        kind = ev.get("event")
        if kind == "primitive_call" and ev.get("status") == "ok":
            prim = ev.get("primitive")
            if prim:
                rs.primitive_counter[prim] += 1
        elif kind == "run_warning":
            if ev.get("code") == "forced_iteration":
                rs.forced_iteration_warnings += 1
        elif kind == "user_message":
            rs.user_messages += 1

    # experiment_runs.jsonl — run_experiment outcomes
    for ev in _safe_iter_jsonl(run_dir / "experiment_runs.jsonl"):
        if ev.get("success"):
            rs.experiment_success_count += 1
        else:
            rs.experiment_failure_count += 1

    # stderr tail for error classification (last 4KB)
    stderr_path = run_dir / "runner.stderr.log"
    if stderr_path.exists():
        try:
            text = stderr_path.read_text(errors="replace")
            rs.stderr_tail = text[-4000:] if len(text) > 4000 else text
        except OSError:
            pass

    rs.exit_reason = _classify_exit(rs.status, rs.error_text, rs.stderr_tail or "")
    return rs


# ── Aggregation ──────────────────────────────────────────────────────────────


_ERROR_LINE_RE = re.compile(
    r"^(?:Error|Exception|Traceback|RuntimeError|ValueError|TypeError|KeyError|"
    r"ImportError|ModuleNotFoundError|AttributeError|AssertionError|OSError|FileNotFoundError|"
    r"\w*Error[\w.]*|asyncgen|Loop <)",
    re.MULTILINE,
)


def extract_error_lines(text: str | None) -> list[str]:
    if not text:
        return []
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _ERROR_LINE_RE.match(s) or "Error" in s[:60] or "Exception" in s[:60]:
            # truncate long lines
            lines.append(s[:200])
    return lines


def cluster_errors(
    all_errors: list[tuple[str, str]],  # (run_id, line)
    threshold: float = 0.85,
) -> list[ErrorCluster]:
    """Cluster similar error lines via greedy difflib matching."""
    clusters: list[ErrorCluster] = []
    for run_id, err in all_errors:
        placed = False
        for c in clusters:
            if SequenceMatcher(None, err, c.representative).ratio() >= threshold:
                c.members.append((run_id, err))
                placed = True
                break
        if not placed:
            clusters.append(ErrorCluster(representative=err, members=[(run_id, err)]))
    clusters.sort(key=lambda c: c.count, reverse=True)
    return clusters


# ── Rendering ────────────────────────────────────────────────────────────────


def _fmt_score(v: float | None) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"


def _fmt_cost(v: float | None) -> str:
    return f"${v:.2f}" if isinstance(v, (int, float)) else "—"


def _fmt_dur(v: float | None) -> str:
    if not isinstance(v, (int, float)):
        return "—"
    if v < 60:
        return f"{v:.0f}s"
    if v < 3600:
        return f"{v / 60:.1f}m"
    return f"{v / 3600:.2f}h"


def render_report(
    runs: list[RunSummary],
    error_clusters: list[ErrorCluster],
    top_k: int,
) -> str:
    out = []
    out.append("# Trace mining stats — 2026-05-27")
    out.append("")
    out.append(
        "Generated by `scripts/mine_traces.py` — Phase 1 of the optimization "
        "platform (see `docs/superpowers/specs/2026-05-27-optimization-platform-design.md`)."
    )
    out.append("")
    out.append(f"Corpus: {len(runs)} runs from `runs/`.")
    out.append("")

    # ── §1 per-run table ────
    out.append("## 1. Per-run summary")
    out.append("")
    out.append(
        "| run_id | paper | mode | sandbox | status | verdict | Hermes | cost | wall | iters | exit_reason |"
    )
    out.append(
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for r in runs:
        out.append(
            f"| `{r.run_id[:24]}` | {(r.paper_title or '—')[:40]} | {r.mode or '—'} | "
            f"{r.sandbox or '—'} | {r.status or '—'} | {r.verdict or '—'} | "
            f"{_fmt_score(r.rubric_overall)} | {_fmt_cost(r.cost_usd)} | "
            f"{_fmt_dur(r.wall_clock_s)} | {r.iterations if r.iterations is not None else '—'} | "
            f"{r.exit_reason} |"
        )
    out.append("")

    # ── §2 exit reason distribution ────
    exit_counter: Counter[str] = Counter(r.exit_reason for r in runs)
    out.append("## 2. Exit reason distribution")
    out.append("")
    out.append("| exit_reason | count | fraction |")
    out.append("|---|---|---|")
    total = len(runs) or 1
    for reason, count in exit_counter.most_common():
        out.append(f"| `{reason}` | {count} | {count / total:.0%} |")
    out.append("")

    # ── §3 primitive call frequency ────
    global_prim: Counter[str] = Counter()
    for r in runs:
        global_prim.update(r.primitive_counter)
    out.append("## 3. Primitive call frequency (corpus-wide)")
    out.append("")
    if global_prim:
        out.append("| primitive | total calls | runs touched |")
        out.append("|---|---|---|")
        for prim, count in global_prim.most_common():
            touched = sum(1 for r in runs if r.primitive_counter.get(prim, 0) > 0)
            out.append(f"| `{prim}` | {count} | {touched}/{len(runs)} |")
    else:
        out.append("_No `primitive_call` events recorded in any run._")
    out.append("")

    # ── §4 experiment success rate ────
    total_exp = sum(r.experiment_success_count + r.experiment_failure_count for r in runs)
    total_succ = sum(r.experiment_success_count for r in runs)
    out.append("## 4. `run_experiment` outcomes")
    out.append("")
    if total_exp:
        out.append(
            f"- Total `run_experiment` calls across corpus: **{total_exp}**"
        )
        out.append(
            f"- Successful: **{total_succ}** ({total_succ / total_exp:.0%})"
        )
        out.append(
            f"- Failed: **{total_exp - total_succ}** ({(total_exp - total_succ) / total_exp:.0%})"
        )
        out.append(
            f"- Runs containing ≥ 1 `run_experiment` call: "
            f"{sum(1 for r in runs if (r.experiment_success_count + r.experiment_failure_count) > 0)}/{len(runs)}"
        )
    else:
        out.append("_No `experiment_runs.jsonl` data found — most runs never reached `run_experiment`._")
    out.append("")

    # ── §5 forced-iteration + user-message activity ────
    fi_total = sum(r.forced_iteration_warnings for r in runs)
    um_total = sum(r.user_messages for r in runs)
    out.append("## 5. Steering signals")
    out.append("")
    out.append(f"- Forced-iteration warnings emitted: **{fi_total}** across {sum(1 for r in runs if r.forced_iteration_warnings > 0)} runs.")
    out.append(f"- User chat messages received: **{um_total}** across {sum(1 for r in runs if r.user_messages > 0)} runs.")
    out.append("")

    # ── §6 error clusters ────
    out.append(f"## 6. Top {min(top_k, len(error_clusters))} error clusters")
    out.append("")
    if error_clusters:
        out.append("Clustered by `difflib.SequenceMatcher` with threshold 0.85.")
        out.append("")
        for i, c in enumerate(error_clusters[:top_k], 1):
            distinct_runs = sorted({m[0] for m in c.members})
            out.append(
                f"### Cluster {i}: {c.count} occurrences across {len(distinct_runs)} runs"
            )
            out.append("")
            out.append(f"**Representative line:** `{c.representative}`")
            out.append("")
            out.append(f"**Runs affected:** {', '.join(f'`{rid[:20]}`' for rid in distinct_runs[:8])}")
            if len(distinct_runs) > 8:
                out.append(f"_... and {len(distinct_runs) - 8} more_")
            out.append("")
    else:
        out.append("_No error lines extracted from stderr logs._")
    out.append("")

    # ── §7 cost + wall-clock rollups ────
    costs = [r.cost_usd for r in runs if isinstance(r.cost_usd, (int, float)) and r.cost_usd > 0]
    walls = [r.wall_clock_s for r in runs if isinstance(r.wall_clock_s, (int, float))]
    out.append("## 7. Cost & wall-clock rollup")
    out.append("")
    if costs:
        out.append(f"- Total spend: **${sum(costs):.2f}** across {len(costs)} runs with cost > 0")
        out.append(f"- Mean cost: ${sum(costs) / len(costs):.2f}; max ${max(costs):.2f}; min ${min(costs):.2f}")
    else:
        out.append("- No runs report cost > 0.")
    if walls:
        out.append(f"- Mean wall-clock: {_fmt_dur(sum(walls) / len(walls))}; max {_fmt_dur(max(walls))}; min {_fmt_dur(min(walls))}")
    out.append("")

    out.append("---")
    out.append("")
    out.append("## 8. Narrative findings (appended by reviewer)")
    out.append("")
    out.append("_To be filled in after reviewing the stats above._")
    out.append("")
    return "\n".join(out)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("findings/2026-05-27-stats-001.md"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cluster-threshold", type=float, default=0.85)
    args = parser.parse_args()

    if not args.runs_dir.exists():
        parser.error(f"--runs-dir {args.runs_dir} does not exist")

    run_dirs = sorted(
        p
        for p in args.runs_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
        and any((p / f).exists() for f in ("demo_status.json", "final_report.json", "dashboard_events.jsonl"))
    )
    if not run_dirs:
        parser.error(f"no run directories found under {args.runs_dir}")

    runs = [parse_run(p) for p in run_dirs]
    runs.sort(key=lambda r: r.started_at or "")

    all_errors: list[tuple[str, str]] = []
    for r in runs:
        for line in extract_error_lines(r.stderr_tail):
            all_errors.append((r.run_id, line))
    clusters = cluster_errors(all_errors, threshold=args.cluster_threshold)

    report = render_report(runs, clusters, top_k=args.top_k)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    sys.stdout.write(f"wrote {args.out} ({len(report):,} chars)\n")
    sys.stdout.write(f"  runs={len(runs)}  events={sum(r.event_count for r in runs)}  ")
    sys.stdout.write(f"errors={len(all_errors)}  clusters={len(clusters)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
