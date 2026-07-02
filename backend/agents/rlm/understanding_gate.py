"""UNDERSTAND gate: double-extraction cross-verification (spec §6, F9 §20).

Two extraction passes are deterministically diffed; disagreements get one
targeted third pass and majority-resolve (2-of-3) or stay unresolved (never
guessed, excluded from ``fields``). Blocking is tiered (F9): only a lint
failure on a span-grounded cross-verified field, or a probe-confirmed asset
gap, may block attempt 1 -- every LLM-only/unresolved field is advisory
forever, no matter what a lint check says about it.

Pure stdlib and extractor-agnostic: built against injected ``extract``/
``lint``/``probe_assets`` callables so a future span-grounded live extractor
slots in without touching this module.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

__all__ = [
    "AssetGap",
    "ExtractedField",
    "Extractor",
    "LintFinding",
    "UnderstandingResult",
    "run_understanding",
]


@dataclass(frozen=True)
class ExtractedField:
    """One extractor-reported field.

    ``source_span`` is ``None`` for an LLM-only field carrying no grounding
    into the paper text -- such a field can never become blocking (F9).
    """

    name: str
    value: Any  # numbers, strings, enums, or lists of identity strings
    source_span: str | None = None


@dataclass(frozen=True)
class LintFinding:
    field: str
    reason: str  # machine string, e.g. "metric_range_invalid"


@dataclass(frozen=True)
class AssetGap:
    asset: str
    kind: str  # "gated" | "missing" | "unreachable"
    probe_confirmed: bool
    detail: str


@dataclass(frozen=True)
class UnderstandingResult:
    fields: Mapping[str, Any]  # resolved name -> value
    source_spans: Mapping[str, str]  # only span-grounded fields appear here
    cross_verified: frozenset[str]  # fields where the two passes agreed (or majority-resolved)
    unresolved: tuple[str, ...]  # disagreed even after the targeted third pass
    lint_findings: tuple[LintFinding, ...]
    asset_gaps: tuple[AssetGap, ...]
    blocking: tuple[str, ...]  # machine reasons that may block attempt 1
    advisory: tuple[str, ...]  # everything else, incl. every unresolved field
    sha256: str  # canonical-JSON hash of the persisted stable payload


# Called with variant tag: "a", "b", then (on disagreement) one targeted pass
# "targeted:<comma-joined sorted field names>".
Extractor = Callable[[str], Mapping[str, ExtractedField]]


def _values_equal(a: Any, b: Any) -> bool:
    """Deterministic per-field diff rule (spec §6.3).

    Lists (identity lists) compare as sets of their string form, so element
    order alone is never a disagreement. Everything else -- numbers, strings,
    enums, bools -- compares by plain ``==`` (so ``42 == 42.0``; a bool rides
    the same operator rather than being coerced into any numeric-tolerance
    special case).
    """
    a_is_list, b_is_list = isinstance(a, list), isinstance(b, list)
    if a_is_list or b_is_list:
        return a_is_list and b_is_list and {str(x) for x in a} == {str(x) for x in b}
    return a == b


def run_understanding(
    *,
    extract: Extractor,
    lint: Callable[[Mapping[str, Any]], Iterable[LintFinding]] | None = None,
    probe_assets: Callable[[], Iterable[AssetGap]] | None = None,
    out_path: Path,
) -> UnderstandingResult:
    """Run the double-extraction UNDERSTAND gate and persist it atomically.

    Exceptions raised by ``extract``/``lint``/``probe_assets`` are not caught
    here -- the campaign's never-raises degradation wrapper owns that; this
    module stays pure.
    """
    pass_a = extract("a")
    pass_b = extract("b")
    all_names = sorted(set(pass_a) | set(pass_b))

    resolved: dict[str, Any] = {}
    span_pair: dict[str, tuple[ExtractedField, ExtractedField]] = {}
    cross_verified: set[str] = set()
    disagreed: list[str] = []

    for name in all_names:
        fa, fb = pass_a.get(name), pass_b.get(name)
        if fa is not None and fb is not None and _values_equal(fa.value, fb.value):
            resolved[name] = fa.value
            span_pair[name] = (fa, fb)
            cross_verified.add(name)
        else:
            disagreed.append(name)

    pass_t: Mapping[str, ExtractedField] = {}
    if disagreed:
        pass_t = extract("targeted:" + ",".join(sorted(disagreed)))

    unresolved: list[str] = []
    for name in disagreed:
        ft = pass_t.get(name)
        fa, fb = pass_a.get(name), pass_b.get(name)
        preferred = None
        if ft is not None:
            if fa is not None and _values_equal(ft.value, fa.value):
                preferred = fa
            elif fb is not None and _values_equal(ft.value, fb.value):
                preferred = fb
        if preferred is not None:
            resolved[name] = ft.value
            # Keep pass-a's span text when pass-a is one of the two agreeing
            # extractions (majority-adoption generalization of the direct-
            # agreement rule); otherwise fall back to pass-b, the other
            # ORIGINAL contributor -- never the third pass itself.
            span_pair[name] = (preferred, ft)
            cross_verified.add(name)
        else:
            unresolved.append(name)

    unresolved_sorted = tuple(sorted(unresolved))
    fields = {name: resolved[name] for name in sorted(resolved)}
    source_spans = {
        name: span_pair[name][0].source_span
        for name in sorted(resolved)
        if span_pair[name][0].source_span and span_pair[name][1].source_span
    }

    lint_findings = tuple(lint(fields)) if lint is not None else ()
    asset_gaps = tuple(probe_assets()) if probe_assets is not None else ()
    # Canonical order: two logically-identical inputs supplied in
    # caller-swapped order must hash identically (F9 determinism fix).
    lint_findings = tuple(
        sorted(lint_findings, key=lambda finding: (finding.field, finding.reason))
    )
    asset_gaps = tuple(
        sorted(asset_gaps, key=lambda gap: (gap.asset, gap.kind, gap.detail))
    )

    blocking: list[str] = []
    advisory: list[str] = []
    for finding in lint_findings:
        reason = f"lint:{finding.reason}:{finding.field}"
        blocked = finding.field in cross_verified and finding.field in source_spans
        (blocking if blocked else advisory).append(reason)
    for gap in asset_gaps:
        reason = f"asset:{gap.kind}:{gap.asset}"
        (blocking if gap.probe_confirmed else advisory).append(reason)
    for name in unresolved_sorted:
        advisory.append(f"unresolved:{name}")
    # Canonical order (same determinism guarantee as above): plain
    # lexicographic sort of the reason strings.
    blocking = sorted(blocking)
    advisory = sorted(advisory)

    payload = {
        "fields": fields,
        "source_spans": source_spans,
        "cross_verified": sorted(cross_verified),
        "unresolved": list(unresolved_sorted),
        "lint_findings": [asdict(f) for f in lint_findings],
        "asset_gaps": [asdict(g) for g in asset_gaps],
        "blocking": blocking,
        "advisory": advisory,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # sha256 covers ONLY `payload`; hashing the wrapper that also carries the
    # hash would make the hash cover itself.
    target_path = Path(out_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps({"payload": payload, "sha256": sha256}, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, target_path)

    return UnderstandingResult(
        fields=fields,
        source_spans=source_spans,
        cross_verified=frozenset(cross_verified),
        unresolved=unresolved_sorted,
        lint_findings=lint_findings,
        asset_gaps=asset_gaps,
        blocking=tuple(blocking),
        advisory=tuple(advisory),
        sha256=sha256,
    )
