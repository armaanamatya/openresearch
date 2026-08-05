"""Campaign report renderer + plan-only report writer (Unit 8).

Deterministic, LLM-free rendering of the ReproductionCampaign's ledger state
into the two deliverables required by the design spec (spec §12
"Observability + the deliverable", `docs/history/specs/
2026-07-01-reproduction-campaign-and-self-improving-harness-design.md`):

- ``write_campaign_report`` writes ``campaign_report.md`` (attempt table,
  evidence trajectory, exclusions, claims-vs-measured, champion, budget)
  purely from the campaign's ``attempts.jsonl`` rows and ``campaign.json``
  state snapshot -- formatting of decisions already made deterministically
  upstream, no LLM.
- ``write_plan_only_report`` is Codex review finding F14: ``ReproductionRun``
  returns ``report=None`` on its ``PLAN_ONLY`` outcome, so an INFEASIBLE (or
  UNDERSTAND-gate-blocked) campaign would otherwise terminate report-less.
  Writes a minimal ``final_report.json``/``final_report.md`` pair, but only
  when no real attempt already produced one (never clobbers a genuine
  report).

Design constraint (Unit 8 brief): this module consumes PLAIN DICTS only (the
campaign ledger rows + the ``campaign.json`` state mapping) and imports
NOTHING from sibling campaign modules -- those are implemented in parallel
units. All field access is defensive (``.get`` with fallbacks) so a missing
or partially populated field renders as ``"—"`` instead of raising.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MISSING = "—"  # em dash, used for every "no data" render slot
_LEDGER_STATUSES = ("launched", "assessed", "decided")
_BUDGET_METERS = ("llm_usd", "gpu_usd", "gpu_hours")
_ATTEMPT_HEADER = (
    "| n | driver | score/target | meets | verdicts (impl/repl) | trust | "
    "evidence | leaves | cost (llm$/gpu$/gpu-h/wall-h) | decision rule |"
)
_ATTEMPT_SEP = "|---|---|---|---|---|---|---|---|---|---|"
_ATTEMPT_EMPTY_ROW = (
    f"| {_MISSING} | {_MISSING} | no attempts recorded | {_MISSING} | {_MISSING} | "
    f"{_MISSING} | {_MISSING} | {_MISSING} | {_MISSING} | {_MISSING} |"
)


def write_campaign_report(
    run_dir: Path,
    *,
    state: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    now: Callable[[], float] = time.time,
) -> Path:
    """Render ``<run_dir>/campaign_report.md`` and return its path.

    Identical ``(state, rows)`` plus a fixed ``now`` always produces
    byte-identical output -- there is no wall-clock or ordering dependence
    beyond what the caller supplies.
    """
    run_dir = Path(run_dir)
    grouped = _group_rows(rows)
    attempt_ns = sorted(grouped)

    sections = [
        _render_header(state),
        _render_budget_table(state),
        _render_attempt_table(grouped, attempt_ns),
        _render_evidence_trajectory(grouped, attempt_ns),
    ]
    # Emitted ONLY when at least one attempt was actually killed, so a campaign
    # that never cut its losses (incl. every campaign with the flag off)
    # renders a byte-identical report to before this feature existed.
    stopped_early = _render_stopped_early(grouped, attempt_ns)
    if stopped_early is not None:
        sections.append(stopped_early)
    sections += [
        _render_exclusions(rows),
        _render_claims_vs_measured(rows),
        _render_champion(state, grouped),
        _render_footer(now),
    ]
    text = "\n\n".join(sections) + "\n"

    path = run_dir / "campaign_report.md"
    _atomic_write_text(path, text)
    return path


def write_plan_only_report(
    run_dir: Path,
    *,
    stop_reason: str,
    what_would_unblock: Sequence[str],
    state: Mapping[str, Any],
    now: Callable[[], float] = time.time,
) -> tuple[Path, Path]:
    """F14: write a minimal ``final_report.{json,md}`` pair for a campaign
    that never launched a real attempt (INFEASIBLE / UNDERSTAND-gate block).

    Never clobbers a real attempt's report: the json claim is an atomic
    ``O_CREAT | O_EXCL`` open, not an exists()-check-then-write -- two
    concurrent callers racing on the same ``run_dir`` can both reach this
    function, and only one may win the write. The md is written AFTER the
    json claim succeeds. A crash between those two steps strands a
    ``final_report.json`` with no ``final_report.md`` sibling; the next call
    detects that shape (json exists, md missing) and recovers ONLY the md,
    regenerated from the json's own persisted content (never from this
    call's arguments, which may not match what was actually claimed) --
    see :func:`_recover_stranded_md`. A real attempt's report (any other
    verdict) always wins; when the json claim loses to one, neither file is
    touched. Always returns the two paths, written or not, so the caller can
    log/link them unconditionally.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "final_report.json"
    md_path = run_dir / "final_report.md"

    generated_at = _iso(now())
    payload = {
        "verdict": "plan_only",
        "stop_reason": stop_reason,
        "what_would_unblock": list(what_would_unblock),
        "paper": {"ref": state.get("paper_ref")},
        "campaign": {
            "project_id": state.get("project_id"),
            "terminal": state.get("terminal"),
            "spent": state.get("spent"),
        },
        "generated_at": generated_at,
    }

    try:
        fd = os.open(json_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        if not md_path.exists():
            _recover_stranded_md(json_path, md_path)
        return json_path, md_path

    try:
        os.write(fd, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)

    _atomic_write_text(
        md_path,
        _plan_only_md_text(
            paper_ref=state.get("paper_ref"),
            project_id=state.get("project_id"),
            stop_reason=stop_reason,
            what_would_unblock=what_would_unblock,
            generated_at=generated_at,
        ),
    )

    return json_path, md_path


def _plan_only_md_text(
    *,
    paper_ref: Any,
    project_id: Any,
    stop_reason: Any,
    what_would_unblock: Sequence[Any],
    generated_at: str,
) -> str:
    md_lines = [
        "# Reproduction plan (no attempt launched)",
        "",
        f"- Paper: {_disp(paper_ref)}",
        f"- Project: {_disp(project_id)}",
        "- Verdict: plan_only",
        f"- Stop reason: {_disp(stop_reason)}",
        "",
        "## What would unblock this",
        "",
    ]
    if what_would_unblock:
        md_lines.extend(f"- {item}" for item in what_would_unblock)
    else:
        md_lines.append("- (none recorded)")
    md_lines.extend(["", f"Generated at: {generated_at}", ""])
    return "\n".join(md_lines)


def _recover_stranded_md(json_path: Path, md_path: Path) -> None:
    """F14 crash recovery: a prior call claimed ``final_report.json`` with the
    ``O_EXCL`` open (below) but crashed/was killed before the md write that
    follows it, stranding a plan-only json with no md sibling. Regenerate the
    md from the json's OWN persisted content. A REAL attempt report (any
    verdict other than ``plan_only``, or a json that fails to parse) is left
    strictly alone -- writing its md is not this function's job.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(data, dict) or data.get("verdict") != "plan_only":
        return
    paper = data.get("paper") if isinstance(data.get("paper"), dict) else {}
    campaign = data.get("campaign") if isinstance(data.get("campaign"), dict) else {}
    _atomic_write_text(
        md_path,
        _plan_only_md_text(
            paper_ref=paper.get("ref"),
            project_id=campaign.get("project_id"),
            stop_reason=data.get("stop_reason"),
            what_would_unblock=data.get("what_would_unblock") or [],
            generated_at=data.get("generated_at") or _MISSING,
        ),
    )


# Ledger grouping ------------------------------------------------------------


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Mapping[str, Any]]]:
    """Group ledger rows by ``attempt_n``, keeping only the LAST row seen for
    each ``status`` (last-writer-wins on read, per the design context's
    "Ledger row vocabulary" -- superseding rows are all retained on disk but
    only the latest of each status is rendered here).
    """
    grouped: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        attempt_n = row.get("attempt_n")
        status = row.get("status")
        if not isinstance(attempt_n, int) or status not in _LEDGER_STATUSES:
            continue
        grouped.setdefault(attempt_n, {})[status] = row
    return grouped


def _assessment_of(entry: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any] | None:
    assessed_row = entry.get("assessed")
    if not isinstance(assessed_row, Mapping):
        return None
    assessment = assessed_row.get("assessment")
    return assessment if isinstance(assessment, Mapping) else None


def _final_report_of(assessment: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if assessment is None:
        return None
    final_report = assessment.get("final_report")
    return final_report if isinstance(final_report, Mapping) else None


# Header / budget -------------------------------------------------------------


def _render_header(state: Mapping[str, Any]) -> str:
    terminal = state.get("terminal")
    if isinstance(terminal, Mapping):
        terminal_line = (
            f"- Terminal: {_disp(terminal.get('kind'))} "
            f"(rule={_disp(terminal.get('rule'))}, "
            f"stop_reason={_disp(terminal.get('stop_reason'))})"
        )
    else:
        terminal_line = "- Terminal: in progress"
    lines = [
        "# Campaign Report",
        "",
        f"- Paper: {_disp(state.get('paper_ref'))}",
        f"- Project: {_disp(state.get('project_id'))}",
        f"- Mode: {_disp(state.get('mode'))}",
        f"- Driver: {_disp(state.get('driver'))}",
        terminal_line,
    ]
    return "\n".join(lines)


def _render_budget_table(state: Mapping[str, Any]) -> str:
    budget = state.get("budget")
    spent = state.get("spent")
    budget = budget if isinstance(budget, Mapping) else {}
    spent = spent if isinstance(spent, Mapping) else {}

    lines = ["## Budget", "", "| meter | budget | spent |", "|---|---|---|"]
    for meter in _BUDGET_METERS:
        lines.append(f"| {meter} | {_fmt(budget.get(meter), 2)} | {_fmt(spent.get(meter), 2)} |")
    budget_wall_h = _fmt(_seconds_to_hours(budget.get("wall_s")), 1)
    spent_wall_h = _fmt(_seconds_to_hours(spent.get("wall_s")), 1)
    lines.append(f"| wall_clock_h | {budget_wall_h} | {spent_wall_h} |")
    return "\n".join(lines)


# Attempt table ---------------------------------------------------------------


def _render_attempt_table(
    grouped: Mapping[int, Mapping[str, Mapping[str, Any]]],
    attempt_ns: Sequence[int],
) -> str:
    lines = ["## Attempts", "", _ATTEMPT_HEADER, _ATTEMPT_SEP]
    if not attempt_ns:
        lines.append(_ATTEMPT_EMPTY_ROW)
        return "\n".join(lines)

    for n in attempt_ns:
        lines.append(_render_attempt_row(n, grouped[n]))
    return "\n".join(lines)


def _render_attempt_row(n: int, entry: Mapping[str, Mapping[str, Any]]) -> str:
    assessment = _assessment_of(entry)
    launched_row = entry.get("launched")
    if assessment is not None:
        driver = _disp(assessment.get("driver"))
    elif isinstance(launched_row, Mapping):
        driver = _disp(launched_row.get("driver"))
    else:
        driver = _MISSING

    if assessment is None:
        rest = " | ".join([_MISSING] * 7)  # meets, verdicts, trust, evidence, leaves, cost, rule
        return f"| {n} | {driver} | in-flight/unassessed | {rest} |"

    final_report = _final_report_of(assessment)
    meets, verdicts = _meets_and_verdicts(final_report)
    pred_pass, pred_total = _predicate_pass_total(assessment.get("evidence_predicates"))
    decision_rule = _MISSING
    decided_row = entry.get("decided")
    if isinstance(decided_row, Mapping):
        decision = decided_row.get("decision")
        if isinstance(decision, Mapping):
            decision_rule = _disp(decision.get("rule"))

    cells = [
        str(n),
        driver,
        _score_target(final_report),
        meets,
        verdicts,
        _trust_str(assessment),
        _MISSING if pred_total == 0 else f"{pred_pass}/{pred_total}",
        _leaves_str(assessment.get("leaf_pass_count")),
        _cost_str(assessment.get("cost")),
        decision_rule,
    ]
    return "| " + " | ".join(cells) + " |"


def _score_target(final_report: Mapping[str, Any] | None) -> str:
    if final_report is None:
        return f"{_MISSING}/{_MISSING}"
    return f"{_fmt(final_report.get('score'), 3)}/{_fmt(final_report.get('target'), 3)}"


def _meets_and_verdicts(final_report: Mapping[str, Any] | None) -> tuple[str, str]:
    """Return (``meets`` yes/no/—, ``impl/repl`` verdict pair) for one attempt row."""
    if final_report is None:
        return _MISSING, f"{_MISSING}/{_MISSING}"
    meets_raw = final_report.get("meets_target")
    meets = {True: "yes", False: "no"}.get(meets_raw, _MISSING)
    verdicts = (
        f"{_disp(final_report.get('implementation_verdict'))}/"
        f"{_disp(final_report.get('replication_verdict'))}"
    )
    return meets, verdicts


def _doomed_kill_of(assessment: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The doomed-kill evidence packet stamped onto a killed attempt's
    assessment by ``reproduction_campaign._apply_doomed_kill``, if any."""
    if not isinstance(assessment, Mapping):
        return None
    kill = assessment.get("doomed_kill")
    return kill if isinstance(kill, Mapping) else None


def _trust_str(assessment: Mapping[str, Any]) -> str:
    reasons = assessment.get("quarantine_reasons")
    first_reason = reasons[0] if isinstance(reasons, (list, tuple)) and reasons else None
    # Checked BEFORE the quarantine flags (a kill also sets hard_quarantined, to
    # keep the attempt out of champion/seeding). Rendering it as
    # "hard-quarantined" would say the attempt is UNTRUSTWORTHY, when the truth
    # is that we never let it finish — "we stopped paying" and "the science
    # failed" must never read the same in the deliverable.
    if _doomed_kill_of(assessment) is not None:
        return "stopped-early (doomed)"
    if assessment.get("hard_quarantined"):
        return f"hard-quarantined: {first_reason}" if first_reason else "hard-quarantined"
    if assessment.get("soft_quarantined"):
        return f"soft-quarantined: {first_reason}" if first_reason else "soft-quarantined"
    return "clean"


def _predicate_pass_total(predicates: Any) -> tuple[int, int]:
    if not isinstance(predicates, Mapping):
        return 0, 0
    return sum(1 for v in predicates.values() if v), len(predicates)


def _leaves_str(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        return _MISSING
    return str(value)


def _cost_str(cost: Any) -> str:
    cost = cost if isinstance(cost, Mapping) else {}
    wall_h = _fmt(_seconds_to_hours(cost.get("wall_s")), 1)
    return (
        f"${_fmt(cost.get('llm_usd'), 2)}/${_fmt(cost.get('gpu_usd'), 2)}/"
        f"{_fmt(cost.get('gpu_hours'), 2)}h/{wall_h}h"
    )


# Evidence trajectory ----------------------------------------------------------


def _render_evidence_trajectory(
    grouped: Mapping[int, Mapping[str, Mapping[str, Any]]],
    attempt_ns: Sequence[int],
) -> str:
    lines = ["## Evidence Trajectory", ""]
    if not attempt_ns:
        lines.append("- no attempts recorded")
        return "\n".join(lines)

    best_pred: int | None = None
    best_leaves: int | None = None
    for n in attempt_ns:
        assessment = _assessment_of(grouped[n])
        if assessment is None:
            lines.append(f"- attempt {n}: unassessed")
            continue

        pred_pass, pred_total = _predicate_pass_total(assessment.get("evidence_predicates"))
        leaf_raw = assessment.get("leaf_pass_count")
        leaf_val = leaf_raw if isinstance(leaf_raw, int) and not isinstance(leaf_raw, bool) else None

        improved = best_pred is not None and (
            pred_pass > best_pred
            or (leaf_val is not None and best_leaves is not None and leaf_val > best_leaves)
        )
        flag = " +" if improved else ""
        lines.append(
            f"- attempt {n}: predicates {pred_pass}/{pred_total}, "
            f"leaves {_leaves_str(leaf_raw)}{flag}"
        )

        best_pred = pred_pass if best_pred is None else max(best_pred, pred_pass)
        if leaf_val is not None:
            best_leaves = leaf_val if best_leaves is None else max(best_leaves, leaf_val)
    return "\n".join(lines)


# Stopped early (doomed-run early kill, spec §10.3) ----------------------------


def _render_stopped_early(
    grouped: Mapping[int, Mapping[str, Mapping[str, Any]]],
    attempt_ns: Sequence[int],
) -> str | None:
    """The cut-losses disclosure: which attempts the campaign KILLED, on what
    measured evidence, and what it cost. ``None`` (section omitted entirely)
    when nothing was killed.

    This exists because a killed attempt must never be silently indistinguish-
    able from a failed one. A reader of this report — human or downstream
    triage consumer — has to be able to tell "we chose to stop paying for this"
    apart from "this paper did not reproduce", because only one of those is
    evidence about the paper.
    """
    killed: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]] = []
    for n in attempt_ns:
        assessment = _assessment_of(grouped[n])
        kill = _doomed_kill_of(assessment)
        if kill is not None and assessment is not None:
            killed.append((n, kill, assessment))
    if not killed:
        return None

    lines = [
        "## Stopped early (doomed)",
        "",
        "These attempts were killed mid-flight to cut losses — the campaign stopped paying.",
        "They are NOT science failures: they are excluded from the champion, from lineage",
        "seeding, and from the plateau/exclusion counters, and no lesson was distilled from",
        "them. Their real spend is still charged to the budget above.",
        "",
    ]
    for n, kill, assessment in killed:
        baseline = kill.get("baseline_attempt_n")
        lines.append(
            f"- **Attempt {n}** — {_disp(kill.get('reason'))} vs attempt {_disp(baseline)} "
            f"(cost {_cost_str(assessment.get('cost'))})"
        )
        margin, polls = kill.get("margin"), kill.get("polls_required")
        if margin is not None or polls is not None:
            lines.append(
                f"  - thresholds: margin {_fmt(margin, 2)}, "
                f"{_disp(polls)} consecutive advancing polls, "
                f"progress floor {_fmt(kill.get('min_progress'), 2)}; "
                f"{_disp(kill.get('observations'))} observations taken"
            )
        detail = kill.get("detail")
        if isinstance(detail, (list, tuple)):
            lines.extend(f"  - evidence: {item}" for item in detail)
    return "\n".join(lines)


# Exclusions / claims-vs-measured / champion / footer --------------------------


def _latest_assessed_final_report(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The LATEST ``status=="assessed"`` row's ``assessment.final_report``.

    "Latest" means the last such row in append order (the most recently
    issued assessment across the whole campaign, not per attempt) -- the
    most up-to-date picture of the campaign's evidence. ``None`` when there
    is no assessed row, or its ``final_report`` isn't a mapping (e.g. the
    latest assessment's report was missing).
    """
    last_assessed: Mapping[str, Any] | None = None
    for row in rows:
        if isinstance(row, Mapping) and row.get("status") == "assessed":
            last_assessed = row
    if not isinstance(last_assessed, Mapping):
        return None
    assessment = last_assessed.get("assessment")
    final_report = assessment.get("final_report") if isinstance(assessment, Mapping) else None
    return final_report if isinstance(final_report, Mapping) else None


def _render_exclusions(rows: Sequence[Mapping[str, Any]]) -> str:
    """Declared exclusions with verification status (spec §12 locked
    decision 9), from the LATEST assessed row's ``final_report``.

    Prefers the structured ``exclusions_detail`` (axis/kind/verified/reason,
    sourced from ``code/metrics.json::scope.exclusions``) when present;
    falls back to the plain ``exclusions`` strings (from
    ``final_report.json::scope.gaps``) otherwise, so an older/simpler
    assessment (no ``exclusions_detail``) still renders something.
    """
    final_report = _latest_assessed_final_report(rows)

    detail_lines: list[str] = []
    raw_detail = final_report.get("exclusions_detail") if isinstance(final_report, Mapping) else None
    if isinstance(raw_detail, (list, tuple)):
        for entry in raw_detail:
            if not isinstance(entry, Mapping):
                continue
            status = "verified" if entry.get("verified") else "UNVERIFIED"
            detail_lines.append(
                f"- {_disp(entry.get('item'))} [{_disp(entry.get('axis'))}/{_disp(entry.get('kind'))}] "
                f"— {status}: {_disp(entry.get('reason'))}"
            )

    lines = ["## Exclusions", ""]
    if detail_lines:
        lines.extend(detail_lines)
    else:
        raw = final_report.get("exclusions") if isinstance(final_report, Mapping) else None
        exclusions = [str(x) for x in raw] if isinstance(raw, (list, tuple)) else []
        if exclusions:
            lines.extend(f"- {item}" for item in exclusions)
        else:
            lines.append("- none declared")
    return "\n".join(lines)


def _render_claims_vs_measured(rows: Sequence[Mapping[str, Any]]) -> str:
    """Claims-vs-measured deltas from the existing rubric/two-axis machinery
    (spec §12 locked decision 9): the LATEST assessed row's
    ``final_report.per_claim`` (``reproducibility.per_claim`` in
    ``final_report.json``) as a table.
    """
    final_report = _latest_assessed_final_report(rows)
    per_claim = final_report.get("per_claim") if isinstance(final_report, Mapping) else None

    lines = ["## Claims vs measured", ""]
    if not isinstance(per_claim, (list, tuple)) or not per_claim:
        lines.append("- no per-claim data recorded")
        return "\n".join(lines)

    lines.append("| claim | status | credit | measured | CI |")
    lines.append("|---|---|---|---|---|")
    for entry in per_claim:
        if not isinstance(entry, Mapping):
            continue
        ci_low, ci_high = entry.get("ci_low"), entry.get("ci_high")
        ci = _MISSING if ci_low is None and ci_high is None else f"[{_fmt(ci_low, 3)}, {_fmt(ci_high, 3)}]"
        cells = [
            _disp(entry.get("claim_id")),
            _disp(entry.get("status")),
            _fmt(entry.get("credit"), 3),
            _fmt(entry.get("measured_mean"), 3),
            ci,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_champion(
    state: Mapping[str, Any],
    grouped: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> str:
    terminal = state.get("terminal")
    champion_n = terminal.get("champion_attempt_n") if isinstance(terminal, Mapping) else None

    lines = ["## Champion", ""]
    if champion_n is None:
        lines.append("none (no guard-clean attempt)")
        return "\n".join(lines)

    entry = grouped.get(champion_n, {})
    assessment = _assessment_of(entry)
    if assessment is None:
        lines.append(f"Attempt {champion_n} (assessment unavailable)")
        return "\n".join(lines)

    final_report = _final_report_of(assessment)
    pred_pass, pred_total = _predicate_pass_total(assessment.get("evidence_predicates"))
    lines.append(
        f"Attempt {champion_n}: score {_score_target(final_report)}, "
        f"evidence {pred_pass}/{pred_total} predicates, "
        f"{_leaves_str(assessment.get('leaf_pass_count'))} leaves"
    )
    return "\n".join(lines)


def _render_footer(now: Callable[[], float]) -> str:
    return f"---\nGenerated at: {_iso(now())}"


# Formatting + I/O helpers -----------------------------------------------------


def _disp(value: Any) -> str:
    if value is None or value == "":
        return _MISSING
    return str(value)


def _fmt(value: Any, decimals: int) -> str:
    if value is None or isinstance(value, bool):
        return _MISSING
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return _MISSING


def _seconds_to_hours(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value) / 3600.0
    except (TypeError, ValueError):
        return None


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # PID-namespaced so two processes racing on the same `path` never share
    # (and clobber) one tmp file before the rename.
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
