#!/usr/bin/env python3
"""Pair with/without-BES runs of a paper and emit an A/B comparison report.

Deterministic, no LLM calls. Reads each run's best final_report.json (via
``report_resolution.resolve_best_report``), groups by the report's
``experiment_arm`` stamp (written by ``report.write_final_report_rlm``;
reports predating the stamp are labelled ``unstamped``), picks one run per
arm, and writes ``runs/_ab/<key>/ab_report.{md,json}`` with overall/adjusted
score deltas, wall-clock + cost deltas, the BES candidate pool, and the
top leaf-level score moves.

Usage:
    python scripts/ab_compare.py --paper 1412.6806
    python scripts/ab_compare.py --pair-id allcnn-ab-20260611
    python scripts/ab_compare.py --paper 1412.6980 --select best
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_candidates(runs_root: Path) -> list[tuple[Path, dict]]:
    """All (run_dir, report) pairs under runs_root.

    Prefers the run's OWN top-level final_report.json over best-attempt
    resolution: A/B arm dirs carry the seeded ancestor history as attempts
    (allcnn-ab-20260611: the 0.7395 ancestor out-scored the arm's 0.7378),
    so the resolver would surface the UNSTAMPED ancestor and the pair filter
    would drop the arm. The arm's own outcome IS the A/B measurement;
    attempts are only a fallback for dirs without a top-level report.
    """
    from backend.services.runs.report_resolution import resolve_best_report

    out: list[tuple[Path, dict]] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith(("_", ".")):
            continue
        report: dict | None = None
        top = run_dir / "final_report.json"
        if top.is_file():
            try:
                loaded = json.loads(top.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and loaded:
                    report = loaded
            except (json.JSONDecodeError, OSError):
                report = None
        if report is None:
            try:
                resolved = resolve_best_report(run_dir)
            except Exception:  # noqa: BLE001 — one bad dir never kills the report
                continue
            report = resolved.report
        if report is None:
            continue
        out.append((run_dir, report))
    return out


def _arm_of(report: dict) -> str:
    arm = report.get("experiment_arm")
    if isinstance(arm, dict) and arm.get("arm"):
        return str(arm["arm"])
    return "unstamped"


def _paper_id(report: dict) -> str | None:
    paper = report.get("paper")
    if isinstance(paper, dict) and paper.get("id"):
        return str(paper["id"])
    return None


def _scores(report: dict) -> tuple[float | None, float | None]:
    from backend.services.runs.report_resolution import extract_scores

    try:
        overall, adjusted = extract_scores(report)
    except Exception:  # noqa: BLE001
        overall, adjusted = None, None
    return overall, adjusted


def _wall_clock_s(report: dict) -> float | None:
    started, completed = report.get("started_at"), report.get("completed_at")
    if not (started and completed):
        return None
    try:
        return (
            datetime.fromisoformat(str(completed)) - datetime.fromisoformat(str(started))
        ).total_seconds()
    except ValueError:
        return None


def _cost_usd(report: dict, run_dir: Path) -> float | None:
    cost = report.get("cost")
    if isinstance(cost, dict) and cost.get("llm_usd") is not None:
        try:
            return float(cost["llm_usd"])
        except (TypeError, ValueError):
            pass
    ledger = run_dir / "cost_ledger.jsonl"
    if not ledger.is_file():
        return None
    total = 0.0
    seen = False
    try:
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("usd", "cost_usd"):
                if row.get(key) is not None:
                    total += float(row[key])
                    seen = True
                    break
    except OSError:
        return None
    return total if seen else None


def _leaf_scores(report: dict) -> dict[str, float]:
    rubric = report.get("rubric")
    leaves = (rubric or {}).get("leaf_scores") if isinstance(rubric, dict) else None
    out: dict[str, float] = {}
    for leaf in leaves or []:
        if not isinstance(leaf, dict):
            continue
        lid = str(leaf.get("id") or leaf.get("leaf_id") or "")
        if not lid:
            continue
        try:
            out[lid] = float(leaf.get("score") or 0.0)
        except (TypeError, ValueError):
            continue
    return out


def _summarise(run_dir: Path, report: dict) -> dict[str, Any]:
    overall, adjusted = _scores(report)
    arm_stamp = report.get("experiment_arm") if isinstance(report.get("experiment_arm"), dict) else {}
    rubric = report.get("rubric") if isinstance(report.get("rubric"), dict) else {}
    return {
        "project_id": run_dir.name,
        "arm": _arm_of(report),
        "ab_pair_id": (arm_stamp or {}).get("ab_pair_id"),
        "overall_score": overall,
        "compute_adjusted_score": adjusted,
        "verdict": report.get("verdict"),
        "meets_target": bool(rubric.get("meets_target") or False),
        "iterations": report.get("iterations"),
        "wall_clock_s": _wall_clock_s(report),
        "cost_usd": _cost_usd(report, run_dir),
        "completed_at": report.get("completed_at"),
        "bes": (arm_stamp or {}).get("bes") or {},
        "leaf_scores": _leaf_scores(report),
    }


def _pick(rows: list[dict], select: str) -> dict | None:
    if not rows:
        return None
    if select == "best":
        return max(rows, key=lambda r: (r["overall_score"] is not None, r["overall_score"] or -1.0))
    return max(rows, key=lambda r: str(r.get("completed_at") or ""))


def _fmt(v: Any, suffix: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}{suffix}"
    return f"{v}{suffix}"


def build_comparison(
    runs_root: Path,
    *,
    paper: str | None = None,
    pair_id: str | None = None,
    select: str = "latest",
) -> dict[str, Any]:
    """The full comparison payload (also the JSON artifact)."""
    rows: list[dict] = []
    for run_dir, report in _load_candidates(runs_root):
        if paper and _paper_id(report) != paper:
            continue
        if pair_id:
            stamp = report.get("experiment_arm")
            if not (isinstance(stamp, dict) and stamp.get("ab_pair_id") == pair_id):
                continue
        rows.append(_summarise(run_dir, report))

    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)

    control = _pick(by_arm.get("control", []) + ([] if pair_id else by_arm.get("unstamped", [])), select)
    bes = _pick(by_arm.get("bes", []), select)

    deltas: dict[str, float | None] = {}
    if control and bes:
        for key in ("overall_score", "compute_adjusted_score", "wall_clock_s", "cost_usd"):
            a, b = control.get(key), bes.get(key)
            deltas[key] = (b - a) if (a is not None and b is not None) else None

    leaf_moves: list[dict] = []
    if control and bes:
        all_ids = set(control["leaf_scores"]) | set(bes["leaf_scores"])
        for lid in all_ids:
            c = control["leaf_scores"].get(lid)
            v = bes["leaf_scores"].get(lid)
            if c is None or v is None or abs(v - c) < 1e-9:
                continue
            leaf_moves.append({"leaf": lid, "control": c, "bes": v, "delta": v - c})
        leaf_moves.sort(key=lambda m: -abs(m["delta"]))
        leaf_moves = leaf_moves[:8]

    return {
        "key": pair_id or paper or "all",
        "paper": paper,
        "pair_id": pair_id,
        "select": select,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arms_found": {arm: len(items) for arm, items in sorted(by_arm.items())},
        "control": control,
        "bes": bes,
        "deltas": deltas,
        "top_leaf_moves": leaf_moves,
    }


def render_markdown(cmp: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# A/B report — {cmp['key']}")
    lines.append("")
    lines.append(f"Generated {cmp['generated_at']} · select={cmp['select']} · arms found: "
                 + ", ".join(f"{a}×{n}" for a, n in cmp["arms_found"].items()))
    lines.append("")
    lines.append("| arm | project | score | adj | verdict | meets target | iters | wall | cost |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for label in ("control", "bes"):
        r = cmp.get(label)
        if not r:
            lines.append(f"| {label} | _missing_ | — | — | — | — | — | — | — |")
            continue
        wall = r.get("wall_clock_s")
        wall_h = f"{wall / 3600:.1f}h" if wall is not None else "—"
        lines.append(
            f"| {label} | {r['project_id']} | {_fmt(r['overall_score'])} | "
            f"{_fmt(r['compute_adjusted_score'])} | {r.get('verdict') or '—'} | "
            f"{'yes' if r.get('meets_target') else 'no'} | {_fmt(r.get('iterations'))} | "
            f"{wall_h} | {_fmt(r.get('cost_usd'), ' USD')} |"
        )
    lines.append("")
    if cmp["deltas"]:
        d = cmp["deltas"]
        lines.append("## Δ (bes − control)")
        lines.append("")
        lines.append(f"- overall_score: **{_fmt(d.get('overall_score'))}**")
        lines.append(f"- compute_adjusted_score: {_fmt(d.get('compute_adjusted_score'))}")
        w = d.get("wall_clock_s")
        lines.append(f"- wall_clock: {_fmt(w / 3600 if w is not None else None, 'h')}")
        lines.append(f"- cost: {_fmt(d.get('cost_usd'), ' USD')}")
        lines.append("")
    bes = cmp.get("bes") or {}
    pool = (bes.get("bes") or {}).get("pool") or []
    if pool:
        lines.append("## BES candidate pool (static SELECT scores)")
        lines.append("")
        lines.append("| candidate | ok | static score |")
        lines.append("|---|---|---|")
        winner = (bes.get("bes") or {}).get("winner")
        for c in pool:
            mark = " ← selected" if c.get("candidate_id") == winner else ""
            lines.append(f"| {c.get('candidate_id')}{mark} | {c.get('ok')} | {_fmt(c.get('score'))} |")
        lines.append("")
    if cmp["top_leaf_moves"]:
        lines.append("## Top leaf-level moves")
        lines.append("")
        lines.append("| leaf | control | bes | Δ |")
        lines.append("|---|---|---|---|")
        for m in cmp["top_leaf_moves"]:
            lines.append(f"| {m['leaf'][:48]} | {_fmt(m['control'])} | {_fmt(m['bes'])} | {_fmt(m['delta'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", default=None, help="Paper id to pair (final_report.json paper.id).")
    ap.add_argument("--pair-id", default=None, help="Explicit OPENRESEARCH_AB_PAIR_ID to pair on.")
    ap.add_argument("--runs-root", default="runs", help="Runs root directory.")
    ap.add_argument("--out", default=None, help="Output dir (default <runs-root>/_ab/<key>/).")
    ap.add_argument("--select", choices=["latest", "best"], default="latest",
                    help="Which run to take per arm when several match.")
    args = ap.parse_args()

    if not args.paper and not args.pair_id:
        ap.error("one of --paper / --pair-id is required")

    runs_root = Path(args.runs_root)
    cmp = build_comparison(runs_root, paper=args.paper, pair_id=args.pair_id, select=args.select)

    out_dir = Path(args.out) if args.out else runs_root / "_ab" / str(cmp["key"]).replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ab_report.json"
    md_path = out_dir / "ab_report.md"
    slim = {k: v for k, v in cmp.items()}
    for label in ("control", "bes"):
        if slim.get(label):
            slim[label] = {k: v for k, v in slim[label].items() if k != "leaf_scores"}
    json_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    md = render_markdown(cmp)
    md_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"[ab_compare] wrote {md_path} and {json_path}")
    if not cmp.get("control") or not cmp.get("bes"):
        print("[ab_compare] WARNING: missing an arm — comparison incomplete", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
