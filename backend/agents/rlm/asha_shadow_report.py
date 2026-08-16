"""Offline shadow-analysis for the ASHA scheduler advisory.

The shadow advisory (``OPENRESEARCH_SCHEDULER_TREE=1``) is attached to every
campaign ``decided`` row's ``decision["asha_advisory"]`` (see
``campaign_composition._maybe_attach_asha_advisory``) and persisted verbatim to
``runs/<project_id>/campaign/attempts.jsonl`` — but it is deliberately kept OUT of
the live SSE stream so it can never be mistaken for a real decision. This module is
the *consumer*: it reads those persisted rows back and shows, per decide point and
as a run-level rollup, what the ASHA tree WOULD have done next to the blind serial
loop that actually ran.

It is the evidence an operator needs before the authority flip: how many branches
ASHA would have **true-killed for provable breakage** (the deterministic-evidence
kill the serial loop ignores) or **frozen** (halved below top-k) — i.e. the GPU-$
the scheduler would have saved — without ANY code yet keying a live decision on it.

Pure + stdlib-only; no import of the campaign loop, so reading a run can never
perturb it. CLI: ``python -m backend.agents.rlm.asha_shadow_report <run_dir|ledger>``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

# Advisory actions the ASHA core emits (see asha_scheduler.SchedulerDecision).
_KILL = "kill"
_FREEZE = "freeze"
_PROMOTE = "promote"


@dataclass(frozen=True)
class DecidePoint:
    """One campaign ``decided`` row, paired with its shadow advisory (if any)."""

    attempt_n: int | None
    live_kind: str | None  # the decision the loop ACTUALLY took (CONTINUE / STOP / ...)
    live_rule: str | None
    live_stop_reason: str | None
    rung: int | None  # fidelity meter at this decide point
    gpu_usd_budget: float | None  # width meter: remaining GPU-$ (None ⇒ eta fallback)
    a100_cap: int | None  # width meter: hard concurrency cap
    # (branch_id, action, reason) for every branch the advisory ranked here.
    actions: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class ShadowReport:
    points: tuple[DecidePoint, ...]
    total_decided: int
    with_advisory: int  # coverage: decided rows that carried an advisory
    # Rollup from the LAST advisory-bearing decide point (the most complete cohort
    # view). Branch ids the tree would, at that final point, kill / freeze / promote.
    final_killed: tuple[str, ...]
    final_frozen: tuple[str, ...]
    final_promoted: tuple[str, ...]


def _advisory_of(decision: Mapping) -> Mapping | None:
    adv = decision.get("asha_advisory") if isinstance(decision, Mapping) else None
    return adv if isinstance(adv, Mapping) else None


def analyze_shadow_rows(rows: Sequence[Mapping]) -> ShadowReport:
    """Fold the campaign ledger rows into a shadow report. Pure.

    Reads only ``status == "decided"`` rows; every other row (planned/launched/
    assessed) is ignored. Fail-soft on shape drift — a malformed advisory degrades
    to "no advisory for this point", never an exception.
    """
    points: list[DecidePoint] = []
    with_advisory = 0
    last_actions: tuple[tuple[str, str, str], ...] = ()

    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "decided":
            continue
        decision = row.get("decision")
        decision = decision if isinstance(decision, Mapping) else {}
        adv = _advisory_of(decision)

        rung = gpu_usd_budget = a100_cap = None
        actions: tuple[tuple[str, str, str], ...] = ()
        if adv is not None:
            with_advisory += 1
            rung = adv.get("rung")
            wm = adv.get("width_meter")
            if isinstance(wm, Mapping):
                gpu_usd_budget = wm.get("gpu_usd_budget")
                a100_cap = wm.get("a100_cap")
            decisions = adv.get("decisions")
            if isinstance(decisions, Sequence):
                actions = tuple(
                    (str(d.get("branch_id")), str(d.get("action")), str(d.get("reason")))
                    for d in decisions
                    if isinstance(d, Mapping)
                )
            last_actions = actions  # most-recent advisory wins the rollup

        points.append(
            DecidePoint(
                attempt_n=row.get("attempt_n"),
                live_kind=decision.get("kind"),
                live_rule=decision.get("rule"),
                live_stop_reason=decision.get("stop_reason"),
                rung=rung,
                gpu_usd_budget=gpu_usd_budget,
                a100_cap=a100_cap,
                actions=actions,
            )
        )

    killed = tuple(b for b, a, _ in last_actions if a == _KILL)
    frozen = tuple(b for b, a, _ in last_actions if a == _FREEZE)
    promoted = tuple(b for b, a, _ in last_actions if a == _PROMOTE)
    return ShadowReport(
        points=tuple(points),
        total_decided=len(points),
        with_advisory=with_advisory,
        final_killed=killed,
        final_frozen=frozen,
        final_promoted=promoted,
    )


def render_report(report: ShadowReport) -> str:
    """Human-readable table + rollup for the terminal."""
    out: list[str] = []
    out.append("ASHA shadow advisory — offline analysis")
    out.append(
        f"decided rows: {report.total_decided}   "
        f"with advisory: {report.with_advisory}"
    )
    if report.with_advisory == 0:
        out.append(
            "  (no advisory recorded — the run was NOT under "
            "OPENRESEARCH_SCHEDULER_TREE=1; re-run with the flag to populate this)"
        )
        return "\n".join(out)

    out.append("")
    out.append(f"  {'att':>4}  {'live':<10} {'rung':>4} {'budget$':>9} {'cap':>4}  branches")
    for p in report.points:
        if not p.actions:
            continue
        brs = " ".join(f"{b}:{a}" for b, a, _ in p.actions)
        budget = "-" if p.gpu_usd_budget is None else f"{p.gpu_usd_budget:.2f}"
        cap = "-" if p.a100_cap is None else str(p.a100_cap)
        rung = "-" if p.rung is None else str(p.rung)
        live = (p.live_kind or "?")[:10]
        out.append(f"  {str(p.attempt_n):>4}  {live:<10} {rung:>4} {budget:>9} {cap:>4}  {brs}")

    out.append("")
    out.append("final cohort view (last advisory):")
    out.append(f"  true-kill (provable breakage): {list(report.final_killed) or '—'}")
    out.append(f"  freeze (halved below top-k):   {list(report.final_frozen) or '—'}")
    out.append(f"  promote (keep investing):      {list(report.final_promoted) or '—'}")
    if report.final_killed:
        out.append(
            f"  → {len(report.final_killed)} branch(es) ASHA would have abandoned for "
            "PROVABLE breakage that the serial loop retried — the GPU-$ a tree saves."
        )
    return "\n".join(out)


def _resolve_ledger(path: Path) -> Path:
    """Accept a ledger file directly, a run dir, or a campaign dir."""
    if path.is_file():
        return path
    for candidate in (path / "campaign" / "attempts.jsonl", path / "attempts.jsonl"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no attempts.jsonl found at or under {path}")


def _read_rows(ledger: Path) -> list[dict]:
    rows: list[dict] = []
    for line in ledger.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a torn final line (matches CampaignLedger.read_rows)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline analysis of the ASHA shadow advisory in a campaign run."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="run dir (runs/<id>), its campaign/ dir, or an attempts.jsonl file",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the report as JSON instead of a table"
    )
    args = parser.parse_args(argv)

    try:
        ledger = _resolve_ledger(args.path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = analyze_shadow_rows(_read_rows(ledger))
    if args.json:
        print(json.dumps(asdict(report), indent=2, default=str))
    else:
        print(render_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
