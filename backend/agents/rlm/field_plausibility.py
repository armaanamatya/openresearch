"""Advisory field-plausibility band over the literature corpus — Phase 4 of
the literature-corpus plan (2026-08-03).

Compares the run's PROVENANCED reproduced metrics (the latest successful
``experiment_runs.jsonl`` row — the same canonical record the final report's
provenance back-link derives from, falling back to ``code/metrics.json``)
against ``lit_results`` rows in the global corpus (``runs/_corpus/corpus.db``)
for the same (dataset, metric), and emits an advisory
``metric_outside_field_band`` ``run_warning`` when a reproduced value is a
strong outlier against >=3 field values.

Modeled on :mod:`backend.agents.rlm.literature_claim_gate`'s advisory
contract:
  - NEVER mutates the rubric tree, any score, or the verdict;
  - fail-soft everywhere — any exception collapses to an empty findings list
    (this must never break report finalization);
  - bounded lookups (row cap, findings cap).

ADVISORY FOREVER: the band is derived from the network-sourced corpus, and
``lit_results`` rows may be LLM-extracted (Lane B — may route/rank, never
sole provenance). Per the evidence doctrine it may warn, never gate — no
future A/B result makes this blocking. Absence of corroboration is never
flagged: novel results are legal, and fewer than ``MIN_FIELD_VALUES`` field
values means no band exists.

Outlier rule (deliberately TIGHTER than the plan's "|z|>3 OR outside
min–max±20%"): a value is flagged only when it is outside the ±20% band AND
|z| > 3 (when a nonzero σ exists; σ==0 degrades to band-only). The plan's
``or`` false-positives on tightly-clustered field values — e.g. field
[71.0, 71.1, 71.2] gives σ≈0.1, so a perfectly plausible 72.0 has |z|≈25;
requiring both conditions keeps the signal precise, which matters more than
recall for an advisory warning. Scale ambiguity (percent vs fraction) is
resolved in the run's favor: if ANY of {v, v*100, v/100} sits inside the
band, nothing is flagged.
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes")

#: Minimum number of literature values for a (dataset, metric) pair before a
#: band exists at all — below this, novel/thin coverage is never flagged.
MIN_FIELD_VALUES = 3
#: |z| threshold (sample stdev) — both this AND the band must agree.
Z_THRESHOLD = 3.0
#: The min–max band is widened by this fraction on each side.
BAND_MARGIN = 0.20
#: Bound the advisory output — this is a signal, not an exhaustive audit.
MAX_FINDINGS = 8
#: Bound the lit_results scan (a corpus is <=25 papers * <=20 rows anyway).
MAX_FIELD_ROWS = 5000
#: Corroborating values shown in the warning message.
MAX_VALUES_SHOWN = 6

_NORM_RE = re.compile(r"[^a-z0-9]+")


def field_plausibility_enabled() -> bool:
    """True iff ``OPENRESEARCH_FIELD_PLAUSIBILITY`` is truthy."""
    return os.environ.get("OPENRESEARCH_FIELD_PLAUSIBILITY", "").strip().lower() in _TRUTHY


@dataclass(frozen=True)
class PlausibilityFinding:
    """One advisory "outside the field band" result for a reproduced metric."""

    dataset: str
    metric: str
    reproduced_value: float
    field_values: tuple[float, ...]
    band: tuple[float, float]
    z: float | None
    metric_path: str

    @property
    def message(self) -> str:
        shown = ", ".join(f"{v:g}" for v in self.field_values[:MAX_VALUES_SHOWN])
        more = len(self.field_values) - MAX_VALUES_SHOWN
        suffix = f" (+{more} more)" if more > 0 else ""
        z_part = f", |z|={abs(self.z):.1f}" if self.z is not None else ""
        return (
            f"reproduced {self.metric}={self.reproduced_value:g} on {self.dataset!r} "
            f"({self.metric_path}) is outside the field band "
            f"[{self.band[0]:g}, {self.band[1]:g}] from {len(self.field_values)} "
            f"literature values: {shown}{suffix}{z_part}; advisory only — the band "
            f"is corpus-derived and never gates the verdict"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "metric": self.metric,
            "reproduced_value": self.reproduced_value,
            "field_values": list(self.field_values[:MAX_VALUES_SHOWN]),
            "n_field_values": len(self.field_values),
            "band": list(self.band),
            "z": self.z,
            "metric_path": self.metric_path,
        }


def _norm_name(text: str) -> str:
    return _NORM_RE.sub(" ", (text or "").lower()).strip()


def _dataset_terms(dataset: str) -> set[str]:
    """Normalized names for a dataset: itself + curated aliases (both ways)."""
    from backend.services.knowledge.corpus.extraction import DATASET_ALIASES

    norm = _norm_name(dataset)
    terms = {norm} if norm else set()
    for canonical, aliases in DATASET_ALIASES.items():
        family = {_norm_name(canonical)} | {_norm_name(a) for a in aliases}
        if norm in family:
            terms |= family
    return {t for t in terms if t}


def _walk_metric_leaves(
    obj: object, path: tuple[str, ...], out: list[tuple[tuple[str, ...], float]]
) -> None:
    """Collect (path, value) for every FINITE numeric leaf; non-finite values
    are the fabrication guards' jurisdiction, not a band question."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk_metric_leaves(v, (*path, str(k)), out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        fv = float(obj)
        if fv == fv and fv not in (float("inf"), float("-inf")):
            out.append((path, fv))


def _leaf_matches(
    path: tuple[str, ...], dataset_terms: set[str], metric_norm: str
) -> bool:
    """True iff this metric leaf plausibly reports ``metric`` on ``dataset``.

    Two shapes are recognized, both conservative exact-ish matches:
      - nested: some path component names the dataset AND the leaf key
        normalizes exactly to the metric name;
      - flat: the leaf key itself is "<dataset> <metric>" normalized
        (covers ``alfworld_success_rate``-style keys).
    """
    if not path or not metric_norm:
        return False
    leaf_norm = _norm_name(path[-1])
    for term in dataset_terms:
        if leaf_norm == f"{term} {metric_norm}":
            return True
    if leaf_norm != metric_norm:
        return False
    for component in path[:-1]:
        comp_norm = _norm_name(component)
        for term in dataset_terms:
            if comp_norm == term or re.search(
                rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", comp_norm
            ):
                return True
    return False


def _band(values: list[float]) -> tuple[float, float]:
    """min–max widened by ``BAND_MARGIN`` on each side (sign-aware)."""
    lo, hi = min(values), max(values)
    band_lo = lo * (1 - BAND_MARGIN) if lo >= 0 else lo * (1 + BAND_MARGIN)
    band_hi = hi * (1 + BAND_MARGIN) if hi >= 0 else hi * (1 - BAND_MARGIN)
    return band_lo, band_hi


def _anomalous(value: float, values: list[float]) -> tuple[bool, float | None]:
    """(is_outlier, z) for one interpretation of the reproduced value."""
    band_lo, band_hi = _band(values)
    outside = value < band_lo or value > band_hi
    try:
        sigma = statistics.stdev(values)
    except statistics.StatisticsError:
        sigma = 0.0
    if sigma > 0:
        z = (value - statistics.mean(values)) / sigma
        return outside and abs(z) > Z_THRESHOLD, z
    return outside, None  # σ==0: identical field values — band-only


def gather_plausibility_findings(
    metrics: object,
    field_rows: dict[tuple[str, str], list[float]],
) -> list[PlausibilityFinding]:
    """Pure comparison core: reproduced metric leaves vs field value groups.

    ``field_rows`` maps (dataset, metric) — raw corpus strings — to the
    literature values for that pair. Never raises; any failure -> [].
    """
    try:
        leaves: list[tuple[tuple[str, ...], float]] = []
        _walk_metric_leaves(metrics, (), leaves)
        if not leaves:
            return []

        findings: list[PlausibilityFinding] = []
        for (dataset, metric), values in sorted(field_rows.items()):
            if len(findings) >= MAX_FINDINGS:
                break
            clean = [v for v in values if isinstance(v, (int, float)) and v == v]
            if len(clean) < MIN_FIELD_VALUES:
                continue
            terms = _dataset_terms(dataset)
            metric_norm = _norm_name(metric)
            if not terms or not metric_norm:
                continue
            for path, value in leaves:
                if len(findings) >= MAX_FINDINGS:
                    break
                if not _leaf_matches(path, terms, metric_norm):
                    continue
                # Scale ambiguity (percent vs fraction) resolved in the run's
                # favor: flag only when EVERY interpretation is an outlier.
                interpretations = [value, value * 100.0, value / 100.0]
                verdicts = [_anomalous(v, clean) for v in interpretations]
                if all(outlier for outlier, _z in verdicts):
                    findings.append(
                        PlausibilityFinding(
                            dataset=dataset,
                            metric=metric,
                            reproduced_value=value,
                            field_values=tuple(sorted(clean)),
                            band=_band(clean),
                            z=verdicts[0][1],
                            metric_path=".".join(path),
                        )
                    )
        return findings
    except Exception:  # noqa: BLE001 — advisory, never break finalization
        logger.debug("field_plausibility: gather failed", exc_info=True)
        return []


def _reproduced_metrics(project_dir: Path) -> object:
    """The run's provenanced metrics: latest successful ``experiment_runs.jsonl``
    row's ``metrics`` (the canonical record), else ``code/metrics.json``."""
    chosen: object = None
    exp_log = project_dir / "experiment_runs.jsonl"
    try:
        if exp_log.exists():
            for line in exp_log.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and rec.get("success") and rec.get("metrics"):
                    chosen = rec["metrics"]  # keep the latest successful
    except OSError:
        chosen = None
    if chosen is not None:
        return chosen
    try:
        mpath = project_dir / "code" / "metrics.json"
        if mpath.is_file():
            return json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return None


def _load_field_rows(store: Any) -> dict[tuple[str, str], list[float]]:
    """(dataset, metric) -> values from ``lit_results``, bounded."""
    rows: dict[tuple[str, str], list[float]] = {}
    cursor = store.connection.execute(
        "SELECT dataset, metric, value FROM lit_results"
        " WHERE dataset != '' AND metric != ''"
        f" ORDER BY dataset, metric LIMIT {MAX_FIELD_ROWS}"
    )
    for row in cursor.fetchall():
        try:
            rows.setdefault((row["dataset"], row["metric"]), []).append(float(row["value"]))
        except (TypeError, ValueError):
            continue
    return rows


def run_field_plausibility(
    project_dir: Path,
    *,
    emit_warning: Callable[[str, str], None] | None = None,
    store: Any | None = None,
    runs_root: Path | None = None,
) -> list[PlausibilityFinding]:
    """Top-level advisory hook — called from the report write chokepoint.

    Returns ``[]`` immediately (no disk/corpus access at all) when
    ``OPENRESEARCH_FIELD_PLAUSIBILITY`` is unset — safe to call
    unconditionally. NEVER mutates anything; ``emit_warning``, if given, is
    called once per finding as ``emit_warning(code, message)`` — the
    ``literature_claim_gate``/``report_claim_gate`` optional-callable
    convention. Fail-soft throughout.
    """
    try:
        if not field_plausibility_enabled():
            return []
        metrics = _reproduced_metrics(Path(project_dir))
        if metrics is None:
            return []

        opened_here = False
        if store is None:
            from backend.services.knowledge.corpus.store import CorpusStore, corpus_root

            root = corpus_root(Path(runs_root) if runs_root else Path(project_dir).parent)
            if not (root / "corpus.db").is_file():
                return []  # no corpus built — never create one at finalize
            store = CorpusStore(root)
            opened_here = True
        try:
            field_rows = _load_field_rows(store)
        finally:
            if opened_here:
                try:
                    store.close()
                except Exception:  # noqa: BLE001
                    pass
        if not field_rows:
            return []

        findings = gather_plausibility_findings(metrics, field_rows)
        if findings and emit_warning is not None:
            for finding in findings:
                try:
                    emit_warning("metric_outside_field_band", finding.message)
                except Exception:  # noqa: BLE001
                    pass
        return findings
    except Exception:  # noqa: BLE001 — fail-soft, never break the report write
        logger.debug("field_plausibility: run_field_plausibility failed", exc_info=True)
        return []


__all__ = [
    "PlausibilityFinding",
    "field_plausibility_enabled",
    "gather_plausibility_findings",
    "run_field_plausibility",
]
