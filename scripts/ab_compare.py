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
    python scripts/ab_compare.py --pair-id allcnn-ab-20260611 --require-stamped

Validator mode (``--require-stamped`` / ``OPENRESEARCH_REQUIRE_STAMPED_AB=1``,
default OFF = reporter behaviour, byte-for-byte): refuses to emit a Δ unless
BOTH arms carry an ``experiment_arm`` stamp, their ``rubric_tree.json`` sha256
match, and their recorded ``scope`` match — so a reported Δ can never be
apples-to-oranges. See spec D1 (``…2026-06-16-grader-fidelity-…``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _flag_on(name: str) -> bool:
    """Env-flag truthiness (matches the repo-wide pattern)."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


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


def _is_stamped(report: dict) -> bool:
    """True iff the report carries a real ``experiment_arm`` stamp.

    A stamp is the ``{arm, ab_pair_id, ...}`` block written by
    ``report.write_final_report_rlm`` (via ``bes_rlm.experiment_arm_stamp``).
    A report without it is labelled ``unstamped`` by ``_arm_of`` and is NOT a
    valid arm in validator mode.
    """
    arm = report.get("experiment_arm")
    return isinstance(arm, dict) and bool(arm.get("arm")) and str(arm.get("arm")) != "unstamped"


def _rubric_tree_sha256(run_dir: Path) -> str | None:
    """sha256 of ``<run_dir>/rubric_tree.json`` bytes, or None if absent."""
    p = run_dir / "rubric_tree.json"
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def _scope_key(scope: Any) -> str:
    """Canonical, order-stable JSON of the recorded ``scope`` for equality.

    ``scope`` is whatever the report records (today a free-text
    ``{requested, ran, gaps}`` block — see report.py). ``sort_keys`` makes
    dict-key order irrelevant; lists stay order-sensitive (a different ``ran``
    list IS a different scope — exactly the apples-to-oranges case to catch).
    """
    return json.dumps(scope, sort_keys=True, ensure_ascii=True, default=str)


def validate_stamped_pair(control: dict, bes: dict) -> str | None:
    """Refuse-reason for a pair under ``--require-stamped``, or None if valid.

    Enforced contract (the "stamped arm" contract):
      1. BOTH arms carry an ``experiment_arm`` stamp (no ``unstamped`` control);
      2. each arm's run-dir archive is complete (all BES required artifacts present);
      3. each arm's ``rubric_tree.json`` exists and their sha256 are identical;
      4. the two arms' recorded ``scope`` are equal.
    """
    from backend.agents.rlm.archive_completeness import check_bes_archive

    if not control.get("_stamped"):
        return (
            "control arm is unstamped (no experiment_arm.arm) — re-run the control "
            "with OPENRESEARCH_AB_ARM=control so the pair is explicitly labelled"
        )
    if not bes.get("_stamped"):
        return (
            "bes arm is unstamped (no experiment_arm.arm) — re-run the bes arm "
            "with OPENRESEARCH_AB_ARM=bes so the pair is explicitly labelled"
        )

    # Archive-completeness gate: a BES efficacy delta requires a complete archive
    # for each arm (the Adam lesson — an incomplete archive makes the Δ folklore).
    c_archive = check_bes_archive(Path(control["_run_dir"]))
    if not c_archive.complete:
        return (
            f"control arm {control['project_id']} has an incomplete run-dir archive "
            f"(missing: {', '.join(c_archive.missing)}) — a BES efficacy delta "
            "requires all required artifacts to be present"
        )
    b_archive = check_bes_archive(Path(bes["_run_dir"]))
    if not b_archive.complete:
        return (
            f"bes arm {bes['project_id']} has an incomplete run-dir archive "
            f"(missing: {', '.join(b_archive.missing)}) — a BES efficacy delta "
            "requires all required artifacts to be present"
        )

    c_sha = _rubric_tree_sha256(control["_run_dir"])
    b_sha = _rubric_tree_sha256(bes["_run_dir"])
    if c_sha is None:
        return f"control arm {control['project_id']} has no rubric_tree.json — cannot verify rubric equality"
    if b_sha is None:
        return f"bes arm {bes['project_id']} has no rubric_tree.json — cannot verify rubric equality"
    if c_sha != b_sha:
        return (
            "rubric_tree.json differs between arms "
            f"(control {c_sha[:12]} != bes {b_sha[:12]}) — the arms were graded "
            "against different rubrics; pin one with OPENRESEARCH_REUSE_RUBRIC=1"
        )

    if _scope_key(control["_scope"]) != _scope_key(bes["_scope"]):
        return (
            "scope differs between arms (models/datasets/seeds the report records "
            "as scope) — the arms did not run the same experiment matrix"
        )
    return None


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
        # _run_dir / _scope / leaf_scores are validator/working-only fields,
        # stripped from the slim JSON artifact (see main()) so reporter-mode
        # output stays byte-for-byte identical when --require-stamped is off.
        "_run_dir": run_dir,
        "_scope": report.get("scope"),
        "_stamped": _is_stamped(report),
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
    select: str = "best",
    require_stamped: bool = False,
) -> dict[str, Any]:
    """The full comparison payload (also the JSON artifact).

    ``require_stamped`` (validator mode, default OFF): when True, an unstamped
    run is NEVER admitted as control and ``validate_stamped_pair`` must pass
    (matching ``rubric_tree.json`` sha + scope) before any Δ / leaf-move is
    emitted; on failure the payload carries ``validation_error`` and empty
    ``deltas``/``top_leaf_moves``. When False, behaviour is byte-for-byte the
    prior reporter.
    """
    rows: list[dict] = []
    for run_dir, report in _load_candidates(runs_root):
        # Matching: with BOTH selectors, a run matches on EITHER — the
        # 1412.6980 pair's control predates the stamp (matches by paper only)
        # while its BES arm shipped paper.id=None on the override lineage
        # (matches by pair only). AND-semantics would pair neither.
        paper_ok = bool(paper) and _paper_id(report) == paper
        stamp = report.get("experiment_arm")
        pair_ok = bool(pair_id) and (
            isinstance(stamp, dict) and stamp.get("ab_pair_id") == pair_id
        )
        if paper and pair_id:
            if not (paper_ok or pair_ok):
                continue
        elif paper and not paper_ok:
            continue
        elif pair_id and not pair_ok:
            continue
        rows.append(_summarise(run_dir, report))

    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)

    # Unstamped runs may stand in as control when pairing involves a paper
    # selector (pre-stamp reports are the historical controls). Only a PURE
    # pair-id pairing excludes them — there an unstamped run can't be tied to
    # the pair at all. Validator mode (require_stamped) NEVER admits an
    # unstamped run as control: an apples-to-oranges Δ is the whole thing it
    # exists to refuse.
    allow_unstamped = (not require_stamped) and (not pair_id or bool(paper))
    control = _pick(
        by_arm.get("control", [])
        + (by_arm.get("unstamped", []) if allow_unstamped else []),
        select,
    )
    bes = _pick(by_arm.get("bes", []), select)

    # Validator gate: when both arms resolve, assert the stamped contract
    # (stamp present + matching rubric_tree.json sha + matching scope) BEFORE
    # any Δ is computed. A refusal blanks deltas/leaf_moves and carries the
    # reason. Reporter mode (require_stamped=False) skips this entirely.
    validation_error: str | None = None
    if require_stamped:
        if not (control and bes):
            found = ", ".join(f"{a}×{len(items)}" for a, items in sorted(by_arm.items())) or "none"
            # In validator mode unstamped runs are NOT admitted as control, so
            # name them explicitly when their exclusion is why an arm is empty
            # — the refusal stays actionable (re-run the arm with OPENRESEARCH_AB_ARM).
            unstamped_note = (
                " (unstamped runs are not admissible as control in validator mode — "
                "re-run the control with OPENRESEARCH_AB_ARM=control)"
                if by_arm.get("unstamped") and not control
                else ""
            )
            validation_error = (
                "validator mode requires both a stamped control and a stamped bes arm; "
                f"found arms: {found}{unstamped_note}"
            )
        else:
            validation_error = validate_stamped_pair(control, bes)

    deltas: dict[str, float | None] = {}
    if control and bes and not validation_error:
        for key in ("overall_score", "compute_adjusted_score", "wall_clock_s", "cost_usd"):
            a, b = control.get(key), bes.get(key)
            deltas[key] = (b - a) if (a is not None and b is not None) else None

    leaf_moves: list[dict] = []
    if control and bes and not validation_error:
        all_ids = set(control["leaf_scores"]) | set(bes["leaf_scores"])
        for lid in all_ids:
            c = control["leaf_scores"].get(lid)
            v = bes["leaf_scores"].get(lid)
            if c is None or v is None or abs(v - c) < 1e-9:
                continue
            leaf_moves.append({"leaf": lid, "control": c, "bes": v, "delta": v - c})
        leaf_moves.sort(key=lambda m: -abs(m["delta"]))
        leaf_moves = leaf_moves[:8]

    payload: dict[str, Any] = {
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
    # Validator-only keys: omitted entirely in reporter mode so the existing
    # artifact stays byte-for-byte identical when --require-stamped is off.
    if require_stamped:
        payload["require_stamped"] = True
        payload["validation_error"] = validation_error
    return payload


def render_markdown(cmp: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# A/B report — {cmp['key']}")
    lines.append("")
    lines.append(f"Generated {cmp['generated_at']} · select={cmp['select']} · arms found: "
                 + ", ".join(f"{a}×{n}" for a, n in cmp["arms_found"].items()))
    lines.append("")
    # Validator refusal banner (validator mode only — the key is absent in
    # reporter mode, so this block never renders there).
    if cmp.get("validation_error"):
        lines.append(f"> **REFUSED (validator mode):** {cmp['validation_error']}")
        lines.append(">")
        lines.append("> No Δ emitted — the pair is not a like-for-like comparison.")
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


# Working-only summary keys (a Path + raw scope) — never JSON-serialized and
# never present in the prior artifact; stripped from the slim JSON in BOTH
# modes so reporter output stays byte-for-byte identical.
_SLIM_DROP = ("leaf_scores", "_run_dir", "_scope", "_stamped")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", default=None, help="Paper id to pair (final_report.json paper.id).")
    ap.add_argument("--pair-id", default=None, help="Explicit OPENRESEARCH_AB_PAIR_ID to pair on.")
    ap.add_argument("--runs-root", default="runs", help="Runs root directory.")
    ap.add_argument("--out", default=None, help="Output dir (default <runs-root>/_ab/<key>/).")
    ap.add_argument("--select", choices=["latest", "best"], default="best",
                    help="Which run to take per arm when several match (default: best).")
    ap.add_argument("--require-stamped", action="store_true", default=False,
                    help="Validator mode: refuse a Δ unless both arms are stamped and "
                         "their rubric_tree.json sha + scope match "
                         "(also via OPENRESEARCH_REQUIRE_STAMPED_AB=1).")
    args = ap.parse_args()

    if not args.paper and not args.pair_id:
        ap.error("one of --paper / --pair-id is required")

    require_stamped = args.require_stamped or _flag_on("OPENRESEARCH_REQUIRE_STAMPED_AB")

    runs_root = Path(args.runs_root)
    cmp = build_comparison(
        runs_root,
        paper=args.paper,
        pair_id=args.pair_id,
        select=args.select,
        require_stamped=require_stamped,
    )

    out_dir = Path(args.out) if args.out else runs_root / "_ab" / str(cmp["key"]).replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ab_report.json"
    md_path = out_dir / "ab_report.md"
    slim = {k: v for k, v in cmp.items()}
    for label in ("control", "bes"):
        if slim.get(label):
            slim[label] = {k: v for k, v in slim[label].items() if k not in _SLIM_DROP}
    json_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    md = render_markdown(cmp)
    md_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"[ab_compare] wrote {md_path} and {json_path}")
    if cmp.get("validation_error"):
        print(f"[ab_compare] REFUSED (validator mode): {cmp['validation_error']}", file=sys.stderr)
        return 3
    if not cmp.get("control") or not cmp.get("bes"):
        print("[ab_compare] WARNING: missing an arm — comparison incomplete", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
