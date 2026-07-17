"""Best-attempt anti-regression rails — attempts improve on the best, never restart.

The Adam regression pattern (0.831 best on 2026-06-07, then 0.69 / 0.0 / 0.736
/ 0.762 / 0.151 across seven attempts): every attempt re-derived the entire
implementation from scratch, so the run that hit the sweet spot was treated as
disposable history. Its working code, its earned rubric leaves, and its target
were all forgotten — each new attempt explored freely and routinely landed
below the proven baseline.

Three rails, each flag-gated and fail-soft:

* ``seed_reference_code`` (``OPENRESEARCH_SEED_BEST_ATTEMPT``) — copy the best
  prior attempt's ``code/`` into ``code/_best_attempt/`` at run start, so the
  implementer can start FROM the proven solution instead of from zero.
* ``best_attempt_guidance_block`` (same flag) — implementer-prompt block:
  the best score, the pointer to the seeded code, and a leaf-level regression
  list (which rubric leaves the best attempt earned that the latest attempt
  lost, with the best run's evidence summaries) — protect earned leaves before
  chasing new ones.
* ``floored_target`` (``OPENRESEARCH_TARGET_BEST_FLOOR``) — raise the in-run
  ``target_score`` to the best prior attempt's score, so the forced-iteration
  policy refuses to finish below the proven baseline while budget remains
  (wall-clock bypass unchanged).

Leaf ids are stable across attempts of one paper (``generated_rubric.json`` is
a paper-level artifact, never archived), so leaf-level joins are exact.

**Quarantine THEN rank.** ``leaf_champions`` / ``best_attempt_guidance_block``
select over the GUARD-FILTERED pool (:func:`_guard_clean_scored_attempts`),
never over a raw LLM score. A fabrication-suspected attempt can score high
precisely BECAUSE it fabricated, so ranking a raw score would let the
implementer's guidance -- and the leaf-champion crossover targets -- anchor on
the fabrication and propagate it forward. The filter is
``campaign_policy.seeding_pool`` itself (imported, never reimplemented), the
same hard-quarantine gate the campaign's own ``select_champion`` applies before
it ranks: fabrication / all-models-failed / tripped-canary attempts are dropped,
while a merely soft-quarantined one (e.g. no external validator configured --
the default) stays eligible, per spec §8.1/F4. Evidence, not grade.

Never raises: the guard filter fails CLOSED (an attempt it cannot assess is
dropped from the pool, not admitted to it).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_SEED_FLAG = "OPENRESEARCH_SEED_BEST_ATTEMPT"
ENV_TARGET_FLOOR_FLAG = "OPENRESEARCH_TARGET_BEST_FLOOR"

REFERENCE_DIR_NAME = "_best_attempt"

# Campaign seed-marker seam (Codex F5 delivery half). The campaign-owned
# guard-filtered champion selection (spec §8.2) writes this file to hand an
# EXPLICIT seed pointer + target floor to the run subprocess it launches —
# under ``campaign/``, a directory attempt_isolation never archives, so the
# marker survives the run's own ``maybe_archive_prior_attempt`` call at
# start-of-run. When present it is authoritative: the raw score-ranked scan
# below (``find_best_attempt``) is a legacy/non-campaign fallback only and is
# never consulted once a campaign is driving.
CAMPAIGN_SEED_MARKER = "campaign/seed_staging.json"

# Heavy / recomputable artifacts never copied into the reference seed.
_SEED_SKIP_DIRS = frozenset({"outputs", "__pycache__", "datasets", REFERENCE_DIR_NAME})
_SEED_SKIP_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".npz")


def _flag_on(name: str) -> bool:
    val = os.environ.get(name, "").strip().lower()
    return bool(val) and val not in ("0", "false", "off")


def _read_seed_marker(project_dir: Path) -> dict[str, Any] | None:
    """Read+parse ``campaign/seed_staging.json``; ``None`` if absent/invalid.

    Never raises (mirrors the rest of this module's fail-soft contract) —
    a torn/partial marker is treated identically to "no marker".
    """
    path = project_dir / CAMPAIGN_SEED_MARKER
    try:
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001 — a torn marker never breaks the rail
        logger.debug("best_attempt: seed marker unreadable at %s", path, exc_info=True)
        return None


def _copy_code_tree(src: Path, dst: Path) -> int:
    """Copy *src* into *dst*, skipping heavy/recomputable artifacts.

    Shared by the marker-driven (campaign) and score-ranked (legacy) seeding
    paths so the skip-set lives in exactly one place. Idempotent: *dst* is
    wiped and recreated fresh each call.
    """
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for child in src.iterdir():
        if child.name in _SEED_SKIP_DIRS:
            continue
        if child.is_file():
            if child.suffix in _SEED_SKIP_SUFFIXES:
                continue
            shutil.copy2(child, dst / child.name)
            copied += 1
        elif child.is_dir():
            shutil.copytree(
                child, dst / child.name,
                ignore=shutil.ignore_patterns(
                    "outputs", "__pycache__", "datasets",
                    *(f"*{s}" for s in _SEED_SKIP_SUFFIXES),
                ),
                dirs_exist_ok=True,
            )
            copied += 1
    return copied


def _read_report(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a torn report never breaks the rail
        return None


def _score_of(report: dict[str, Any] | None) -> float | None:
    if not isinstance(report, dict):
        return None
    rub = report.get("rubric")
    raw = (rub or {}).get("overall_score") if isinstance(rub, dict) else None
    if raw is None:
        raw = report.get("overall_score")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def find_best_attempt(project_dir: Path | str) -> dict[str, Any] | None:
    """Best-scoring PRIOR attempt of this paper, or None when nothing scored.

    Scans ``attempts/*/final_report.json`` (the current in-flight attempt has
    no final report yet, so it is never self-referenced). Returns
    ``{"dir", "score", "report"}`` with ``dir`` as a Path.
    """
    try:
        attempts_root = Path(project_dir) / "attempts"
        if not attempts_root.is_dir():
            return None
        best: dict[str, Any] | None = None
        for attempt in sorted(p for p in attempts_root.iterdir() if p.is_dir()):
            report = _read_report(attempt / "final_report.json")
            score = _score_of(report)
            if score is None:
                continue
            if best is None or score > best["score"]:
                best = {"dir": attempt, "score": score, "report": report}
        return best
    except Exception:  # noqa: BLE001
        logger.debug("best_attempt: scan failed", exc_info=True)
        return None


def _leaves(report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    rub = (report or {}).get("rubric") or {}
    out: dict[str, dict[str, Any]] = {}
    for leaf in rub.get("leaf_scores") or []:
        if isinstance(leaf, dict) and leaf.get("id") is not None and leaf.get("score") is not None:
            out[str(leaf["id"])] = leaf
    return out


def leaf_regressions(
    best_report: dict[str, Any] | None,
    latest_report: dict[str, Any] | None,
    *,
    top_n: int = 8,
    min_delta: float = 0.15,
) -> list[dict[str, Any]]:
    """Leaves the best attempt earned that the latest attempt lost.

    Joined on the stable leaf id; sorted by lost points descending; only
    deltas ≥ ``min_delta`` (grader noise stays out of the list).
    """
    best_leaves = _leaves(best_report)
    latest_leaves = _leaves(latest_report)
    rows: list[dict[str, Any]] = []
    for lid, bleaf in best_leaves.items():
        lleaf = latest_leaves.get(lid)
        if lleaf is None:
            continue
        try:
            delta = float(bleaf["score"]) - float(lleaf["score"])
        except (TypeError, ValueError):
            continue
        if delta < min_delta:
            continue
        rows.append({
            "id": lid,
            "best": float(bleaf["score"]),
            "latest": float(lleaf["score"]),
            "evidence": str(bleaf.get("justification") or "")[:200],
        })
    rows.sort(key=lambda r: r["best"] - r["latest"], reverse=True)
    return rows[:top_n]


def seed_reference_code(project_dir: Path | str) -> str | None:
    """Copy the campaign-selected or best prior attempt's ``code/`` into
    ``code/_best_attempt/``.

    FIRST checks ``OPENRESEARCH_SEED_BEST_ATTEMPT``: off means no campaign
    env is driving this run, so a stale ``campaign/seed_staging.json`` left
    over from an earlier campaign must be inert — "no campaign env set =>
    byte-identical behavior" for a manual run. Only once the flag is on does
    the campaign seed marker (``CAMPAIGN_SEED_MARKER``) get consulted: when
    present, its ``source_code_dir`` is staged verbatim and the score-ranked
    scan below is never consulted (Codex F5 — campaign-owned, guard-filtered
    selection must never be second-guessed by a raw score). A marker whose
    ``source_code_dir`` is missing — or which names a source holding nothing
    seedable — fails CLOSED (returns None; no fallback to the scan) rather
    than silently reverting to score-ranked seeding under a campaign, and
    rather than leaving behind an empty ``code/_best_attempt/`` whose README
    would claim to be "the COMPLETE working code" of a seed that never
    materialized. The driver stages the marker at the seed's POST-archive
    path (``attempt_driver._prepare_launch``), so under a healthy campaign
    the source is the archived ``attempts/<ts>/code`` tree, not the live
    ``code/`` the pre-launch archive just emptied.

    With no marker (flag still on), behavior is unchanged: reference
    material for the implementer (it copies what it wants) — heavy artifacts
    skipped, idempotent (re-seeds fresh each call). Returns the relative
    path seeded, or None.
    """
    try:
        project_dir = Path(project_dir)

        if not _flag_on(ENV_SEED_FLAG):
            return None

        marker = _read_seed_marker(project_dir)
        if marker is not None:
            raw_src = marker.get("source_code_dir")
            src = Path(raw_src) if raw_src else None
            if src is None or not src.is_dir():
                logger.debug(
                    "best_attempt: seed marker present but source_code_dir "
                    "missing/invalid: %r", raw_src,
                )
                return None
            dst = project_dir / "code" / REFERENCE_DIR_NAME
            copied = _copy_code_tree(src, dst)
            if copied == 0:
                # Fail CLOSED, same as a missing source: an empty seed is not
                # a seed. (Pre-fix, a marker pointing at the freshly-emptied
                # live code/ could land here and ship a README-only "seed".)
                shutil.rmtree(dst, ignore_errors=True)
                logger.warning(
                    "best_attempt: seed marker source %s held nothing seedable "
                    "— failing closed (no reference staged)", src,
                )
                return None
            (dst / "_BEST_ATTEMPT_README.txt").write_text(
                f"Campaign-selected seed: attempt {marker.get('attempt_n')} "
                f"({marker.get('lineage')}) — source {src}.\n"
                "This is the COMPLETE working code of the campaign-selected seed.\n"
                "Start from it; copy files verbatim and change only what your\n"
                "directives/guidance names.\n",
                encoding="utf-8",
            )
            logger.info(
                "best_attempt: seeded %d item(s) from campaign marker "
                "(attempt %s, lineage %s) into code/%s",
                copied, marker.get("attempt_n"), marker.get("lineage"),
                REFERENCE_DIR_NAME,
            )
            return f"code/{REFERENCE_DIR_NAME}"

        best = find_best_attempt(project_dir)
        if best is None:
            return None
        src = best["dir"] / "code"
        if not src.is_dir():
            return None
        dst = project_dir / "code" / REFERENCE_DIR_NAME
        copied = _copy_code_tree(src, dst)
        (dst / "_BEST_ATTEMPT_README.txt").write_text(
            f"Best prior attempt: {best['dir'].name} — rubric {best['score']:.4f}.\n"
            "This is the COMPLETE working code of the best-scoring prior attempt.\n"
            "Start from it; copy files verbatim and change only what the\n"
            "regression list in your guidance names.\n",
            encoding="utf-8",
        )
        logger.info(
            "best_attempt: seeded %d item(s) from %s (score %.4f) into code/%s",
            copied, best["dir"].name, best["score"], REFERENCE_DIR_NAME,
        )
        return f"code/{REFERENCE_DIR_NAME}"
    except Exception:  # noqa: BLE001 — seeding must never block a run
        logger.debug("best_attempt: seeding failed", exc_info=True)
        return None


def _all_scored_attempts(project_dir: Path) -> list[dict[str, Any]]:
    """Every scored prior attempt, chronologically (dir names are timestamps).

    RAW — no trust filter. Only :func:`_guard_clean_scored_attempts` should
    feed selection; this is its unfiltered input.
    """
    try:
        attempts_root = Path(project_dir) / "attempts"
        if not attempts_root.is_dir():
            return []
        out = []
        for attempt in sorted(p for p in attempts_root.iterdir() if p.is_dir()):
            report = _read_report(attempt / "final_report.json")
            if _score_of(report) is not None:
                out.append({"dir": attempt, "score": _score_of(report), "report": report})
        return out
    except Exception:  # noqa: BLE001
        return []


def _guard_clean_scored_attempts(project_dir: Path) -> list[dict[str, Any]]:
    """:func:`_all_scored_attempts` minus every HARD-quarantined attempt.

    The trust gate is ``campaign_policy.seeding_pool`` over assessments built
    by ``attempt_assessment.assess_attempt`` — the campaign's own machinery,
    imported rather than reimplemented so this rail can never drift from the
    quarantine rules ``select_champion`` enforces. Hard quarantine =
    fabrication guard tripped / all-models-failed / rubric canary tripped;
    soft quarantine (validator missing or stale) deliberately stays SEEDABLE
    (spec §8.1/F4 — with the external validator default-OFF, filtering on it
    would starve the rail on every real run).

    ``pinned_rubric_sha256=None``: the campaign owns rubric pinning, and
    ``generated_rubric.json`` is paper-level (never archived), so it is
    absent from an archived attempt dir — pinning here would mismatch every
    attempt into hard quarantine.

    Imports are local: this module is loaded on hot child-run paths and must
    not pull the campaign import chain in at module scope. Fail-CLOSED — an
    attempt whose assessment cannot be computed is DROPPED, never admitted.
    """
    attempts = _all_scored_attempts(project_dir)
    if not attempts:
        return []
    try:
        from backend.agents.rlm.attempt_assessment import assess_attempt
        from backend.agents.rlm.campaign_policy import seeding_pool
    except Exception:  # noqa: BLE001 — no guard filter => no selection (fail closed)
        logger.debug("best_attempt: guard filter unavailable; pool empty", exc_info=True)
        return []

    by_n: dict[int, dict[str, Any]] = {}
    assessments = []
    for i, att in enumerate(attempts, start=1):
        try:
            assessments.append(
                assess_attempt(
                    att["dir"],
                    attempt_n=i,
                    driver="",
                    project_id="",
                    directives_sha256="",
                    pinned_rubric_sha256=None,
                )
            )
        except Exception:  # noqa: BLE001 — unassessable => quarantined by default
            logger.debug("best_attempt: assess failed for %s", att["dir"], exc_info=True)
            continue
        by_n[i] = att
    return [by_n[a.attempt_n] for a in seeding_pool(assessments) if a.attempt_n in by_n]


def _best_of(pool: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Highest-scoring attempt WITHIN an already guard-filtered pool."""
    return max(pool, key=lambda a: a["score"]) if pool else None


def _leaf_champions_from(pool: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    champs: dict[str, dict[str, Any]] = {}
    for att in pool:
        for lid, leaf in _leaves(att["report"]).items():
            try:
                sc = float(leaf["score"])
            except (TypeError, ValueError):
                continue
            cur = champs.get(lid)
            if cur is None or sc > cur["score"]:
                champs[lid] = {
                    "score": sc,
                    "evidence": str(leaf.get("justification") or "")[:160],
                    "attempt": att["dir"].name,
                }
    return champs


def _ceiling_of(champs: dict[str, dict[str, Any]]) -> float | None:
    if not champs:
        return None
    vals = [c["score"] for c in champs.values()]
    return sum(vals) / len(vals)


def leaf_champions(project_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Per-leaf CHAMPION across every GUARD-CLEAN scored attempt (forward-search
    crossover).

    The best single attempt is not the ceiling — champions are scattered
    across attempts (All-CNN: attempt 2 held the base/strided stars, attempt 3
    the only converged convpool/all-conv cells). Joined on the stable leaf id;
    returns ``{leaf_id: {score, evidence, attempt}}``.

    Hard-quarantined attempts are excluded (see the module docstring): a
    fabricated leaf routinely scores 1.0, and crediting it as the champion
    would hand the implementer a fabrication to reproduce.
    """
    return _leaf_champions_from(_guard_clean_scored_attempts(Path(project_dir)))


def champion_ceiling(project_dir: Path | str) -> float | None:
    """Unweighted mean of per-leaf champions — a ROUGH crossover ceiling.

    'If one run reproduced every leaf at its best-ever level simultaneously.'
    Indicative only (real roll-up is weighted); None when no champions exist.
    """
    return _ceiling_of(leaf_champions(project_dir))


def best_attempt_guidance_block(project_dir: Path | str, *, max_chars: int = 2400) -> str:
    """Implementer-prompt block: best score + seeded-code pointer + regressions.

    Every attempt named here — the best, the latest it is compared against, and
    the crossover champions — comes from the GUARD-FILTERED pool, never the raw
    score-ranked scan. Anchoring the implementer on a fabrication-quarantined
    attempt would launder the fabrication into the next attempt's code, and the
    fitness signal is the deterministic evidence layer, never the LLM grade.
    """
    if not _flag_on(ENV_SEED_FLAG):
        return ""
    try:
        project_dir = Path(project_dir)
        # One scan+assessment pass; best/latest/champions all derive from it.
        pool = _guard_clean_scored_attempts(project_dir)
        best = _best_of(pool)
        if best is None:
            return ""
        latest = pool[-1]  # pool is chronological (attempts/ dirs are timestamps)
        lines = [
            "",
            "",
            f"BEST PRIOR ATTEMPT — rubric {best['score']:.3f} "
            f"({best['dir'].name}). Its COMPLETE working code is seeded at "
            f"code/{REFERENCE_DIR_NAME}/ — START FROM IT: copy files verbatim and "
            "modify only what is necessary. Do NOT rewrite the solution from "
            "scratch; unforced rewrites are how previous attempts regressed "
            "below this baseline. Protect already-earned rubric leaves before "
            "chasing new ones.",
        ]
        if latest["dir"] != best["dir"]:
            regs = leaf_regressions(best["report"], latest["report"])
            if regs:
                lines.append(
                    f"LEAVES THE BEST ATTEMPT EARNED THAT THE LATEST "
                    f"({latest['score']:.3f}) LOST — restore these specifically:"
                )
                for r in regs:
                    lines.append(
                        f"  - leaf {r['id'][:8]}: best {r['best']:.2f} vs latest "
                        f"{r['latest']:.2f} — best-run evidence: {r['evidence']}"
                    )
        # Forward-search crossover: leaves where some OTHER attempt beat the
        # best attempt — no single run is the ceiling; combine the champions.
        champs = _leaf_champions_from(pool)
        best_leaves = _leaves(best["report"])
        cross = []
        for lid, ch in champs.items():
            bleaf = best_leaves.get(lid)
            bscore = float(bleaf["score"]) if bleaf and bleaf.get("score") is not None else 0.0
            if ch["score"] - bscore >= 0.15 and ch["attempt"] != best["dir"].name:
                cross.append((lid, ch, bscore))
        if cross:
            ceiling = _ceiling_of(champs)
            lines.append(
                "CROSSOVER TARGETS — leaves where a DIFFERENT attempt beat the "
                "best one (no single prior run is the ceiling; reproduce ALL "
                "champions simultaneously"
                + (f"; rough combined ceiling ≈ {ceiling:.2f}" if ceiling else "")
                + "):"
            )
            for lid, ch, bscore in sorted(cross, key=lambda t: t[1]["score"] - t[2], reverse=True)[:6]:
                lines.append(
                    f"  - leaf {lid[:8]}: champion {ch['score']:.2f} in "
                    f"{ch['attempt'][:15]} (best attempt had {bscore:.2f}) — "
                    f"champion evidence: {ch['evidence']}"
                )
        # Forward-search population step: the cell grid IS a population — for
        # configs with no proven champion, run small candidate populations
        # instead of betting one guess per cell (the uniform-lr bet is how the
        # All-CNN stars regressed while the dead families were being revived).
        lines.append(
            "POPULATION RULE for cells without a paper-grade champion: emit "
            "2 candidate cells with distinct hyperparameters (suffix the ids, "
            "e.g. _lr005/_lr001) instead of betting on one guess — the grid "
            "runs them in parallel, and the best result becomes the champion "
            "automatically. Cells WITH a champion keep its exact config."
        )
        block = "\n".join(lines)
        if len(block) > max_chars:
            block = block[: max_chars - 15].rstrip() + "\n  (truncated)"
        return block + "\n"
    except Exception:  # noqa: BLE001
        logger.debug("best_attempt: guidance block failed", exc_info=True)
        return ""


def floored_target(project_dir: Path | str, target: float | None) -> float | None:
    """Raise ``target_score`` to the campaign seed marker's floor, else the
    best prior attempt's score (flag-gated).

    FIRST checks ``OPENRESEARCH_TARGET_BEST_FLOOR``: off means no campaign
    env is driving this run, so a stale ``campaign/seed_staging.json`` left
    over from an earlier campaign must be inert (``target`` returned
    unchanged, never even reading the marker) — "no campaign env set =>
    byte-identical behavior" for a manual run.

    Only once the flag is on does the campaign seed marker (Codex F5) get
    consulted: when present, its numeric ``target_floor`` (if any) is
    applied via ``max(target, floor)`` WITHOUT scanning ``attempts/`` — the
    campaign, not a raw score-ranked scan, owns the floor once it is
    driving. A marker without a numeric ``target_floor`` leaves ``target``
    unchanged (still without scanning).

    With no marker (flag still on), behavior is unchanged: raises
    ``target_score`` to the best prior attempt's score. The forced-iteration
    policy then refuses FINAL_VAR below the proven baseline while
    iterations/budget remain — the wall-clock bypass is untouched, so an
    honest partial still ships at the deadline.
    """
    if not _flag_on(ENV_TARGET_FLOOR_FLAG):
        return target

    try:
        marker = _read_seed_marker(Path(project_dir))
    except Exception:  # noqa: BLE001
        marker = None
    if marker is not None:
        floor = marker.get("target_floor")
        if isinstance(floor, (int, float)) and not isinstance(floor, bool):
            floor = float(floor)
            return floor if (target is None or floor > target) else target
        return target

    try:
        best = find_best_attempt(project_dir)
        if best is None:
            return target
        floor = float(best["score"])
        if target is None or floor > float(target):
            return floor
        return target
    except Exception:  # noqa: BLE001
        return target


__all__ = [
    "CAMPAIGN_SEED_MARKER",
    "ENV_SEED_FLAG",
    "ENV_TARGET_FLOOR_FLAG",
    "REFERENCE_DIR_NAME",
    "best_attempt_guidance_block",
    "champion_ceiling",
    "find_best_attempt",
    "floored_target",
    "leaf_champions",
    "leaf_regressions",
    "seed_reference_code",
]
