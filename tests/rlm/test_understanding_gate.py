"""Unit tests for backend.agents.rlm.understanding_gate (Campaign Unit 7).

Hermetic, tmp_path-only, pure stdlib — no network, no backend fixtures.
Covers (spec §6, Codex F9 §20):
  * double-extraction diff determinism (agree / disagree / missing-in-one-pass)
  * single targeted third pass, sorted field names, majority (2-of-3) adoption
  * still-unresolved fields excluded from ``fields``, never guessed
  * span-groundedness requires spans on BOTH agreeing passes
  * tiered blocking: only span-grounded+cross-verified lint failures and
    probe-confirmed asset gaps may block; everything else is advisory forever
  * persisted-payload hash determinism/stability and atomic-write round-trip
"""

from __future__ import annotations

import hashlib
import json

from backend.agents.rlm.understanding_gate import (
    AssetGap,
    ExtractedField,
    LintFinding,
    run_understanding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ScriptedExtractor:
    """Records every ``variant`` it is called with and replays a canned reply.

    ``responses`` maps a variant tag ("a", "b", or the exact "targeted:..."
    string) to the ``Mapping[str, ExtractedField]`` that call should return.
    An unscripted variant returns an empty mapping.
    """

    def __init__(self, responses: dict[str, dict[str, ExtractedField]]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self, variant: str) -> dict[str, ExtractedField]:
        self.calls.append(variant)
        return self.responses.get(variant, {})


# ---------------------------------------------------------------------------
# Diff determinism
# ---------------------------------------------------------------------------


def test_agreeing_passes_no_third_call(tmp_path):
    extractor = ScriptedExtractor(
        {
            "a": {"lr": ExtractedField("lr", 0.001, "sec 4.2")},
            "b": {"lr": ExtractedField("lr", 0.001, "sec 4.2")},
        }
    )
    result = run_understanding(extract=extractor, out_path=tmp_path / "u.json")
    assert extractor.calls == ["a", "b"]
    assert result.fields == {"lr": 0.001}
    assert result.cross_verified == frozenset({"lr"})


def test_disagreement_triggers_single_targeted_pass_with_sorted_fields(tmp_path):
    extractor = ScriptedExtractor(
        {
            "a": {
                "zeta": ExtractedField("zeta", 1, None),
                "alpha": ExtractedField("alpha", 1, None),
            },
            "b": {
                "zeta": ExtractedField("zeta", 2, None),
                "alpha": ExtractedField("alpha", 2, None),
            },
            "targeted:alpha,zeta": {
                "zeta": ExtractedField("zeta", 1, None),
                "alpha": ExtractedField("alpha", 1, None),
            },
        }
    )
    run_understanding(extract=extractor, out_path=tmp_path / "u.json")
    assert extractor.calls == ["a", "b", "targeted:alpha,zeta"]


def test_majority_adoption_marks_cross_verified(tmp_path):
    extractor = ScriptedExtractor(
        {
            "a": {"seed": ExtractedField("seed", 42, "s1")},
            "b": {"seed": ExtractedField("seed", 7, "s2")},
            "targeted:seed": {"seed": ExtractedField("seed", 42, "s3")},
        }
    )
    result = run_understanding(extract=extractor, out_path=tmp_path / "u.json")
    assert result.fields["seed"] == 42
    assert "seed" in result.cross_verified
    assert result.unresolved == ()


def test_majority_adoption_matching_pass_b_uses_pass_b_span(tmp_path):
    """Coverage gap: the third pass can majority-agree with pass-b instead of
    pass-a. The adopted value and span-grounding must follow pass-b in that
    branch -- span-grounding still requires BOTH pass-b's and the third
    pass's source_span to be non-empty (never the third pass's own span
    text, never pass-a's, since pass-a was never part of the agreement)."""
    extractor = ScriptedExtractor(
        {
            "a": {"seed": ExtractedField("seed", 99, "span-a")},
            "b": {"seed": ExtractedField("seed", 7, "span-b")},
            "targeted:seed": {"seed": ExtractedField("seed", 7, "span-t")},
        }
    )
    result = run_understanding(extract=extractor, out_path=tmp_path / "u.json")
    assert result.fields["seed"] == 7  # adopted value is pass-b's, not pass-a's
    assert "seed" in result.cross_verified
    assert result.unresolved == ()
    assert result.source_spans["seed"] == "span-b"


def test_still_unresolved_goes_advisory_and_excluded_from_fields(tmp_path):
    extractor = ScriptedExtractor(
        {
            "a": {"batch_size": ExtractedField("batch_size", 16, None)},
            "b": {"batch_size": ExtractedField("batch_size", 32, None)},
            "targeted:batch_size": {"batch_size": ExtractedField("batch_size", 64, None)},
        }
    )
    result = run_understanding(extract=extractor, out_path=tmp_path / "u.json")
    assert "batch_size" not in result.fields
    assert result.unresolved == ("batch_size",)
    assert "unresolved:batch_size" in result.advisory
    assert "batch_size" not in result.cross_verified
    # An unresolved field must NEVER contribute to blocking -- assert the
    # FULL tuple, not membership (a mutation appending unresolved names to
    # blocking must be caught here).
    assert result.blocking == ()


def test_numeric_exact_and_list_set_compare(tmp_path):
    extractor = ScriptedExtractor(
        {
            "a": {
                "epochs": ExtractedField("epochs", 42, None),
                "datasets": ExtractedField("datasets", ["a", "b"], None),
            },
            "b": {
                "epochs": ExtractedField("epochs", 42.0, None),
                "datasets": ExtractedField("datasets", ["b", "a"], None),
            },
        }
    )
    result = run_understanding(extract=extractor, out_path=tmp_path / "u.json")
    assert extractor.calls == ["a", "b"]  # no disagreement -> no third pass
    assert result.fields["epochs"] == 42
    assert result.fields["datasets"] == ["a", "b"]
    assert result.cross_verified == frozenset({"epochs", "datasets"})


def test_missing_in_one_pass_is_disagreement(tmp_path):
    extractor = ScriptedExtractor(
        {
            "a": {"lr": ExtractedField("lr", 0.01, "span")},
            "b": {},
            "targeted:lr": {"lr": ExtractedField("lr", 0.01, "span2")},
        }
    )
    result = run_understanding(extract=extractor, out_path=tmp_path / "u.json")
    assert extractor.calls == ["a", "b", "targeted:lr"]
    assert result.fields["lr"] == 0.01
    assert "lr" in result.cross_verified


# ---------------------------------------------------------------------------
# Span-groundedness
# ---------------------------------------------------------------------------


def test_span_grounded_requires_spans_on_agreeing_passes(tmp_path):
    extractor = ScriptedExtractor(
        {
            "a": {
                "lr": ExtractedField("lr", 0.01, "sec 3.1"),
                "seed": ExtractedField("seed", 42, "sec 3.2"),
            },
            "b": {
                "lr": ExtractedField("lr", 0.01, None),
                "seed": ExtractedField("seed", 42, "sec 3.2b"),
            },
        }
    )
    result = run_understanding(extract=extractor, out_path=tmp_path / "u.json")
    assert "lr" in result.cross_verified
    assert "lr" not in result.source_spans
    assert result.source_spans["seed"] == "sec 3.2"


# ---------------------------------------------------------------------------
# Tiered blocking (F9)
# ---------------------------------------------------------------------------


def test_lint_on_span_grounded_cross_verified_blocks(tmp_path):
    extractor = ScriptedExtractor(
        {
            "a": {"metric_range": ExtractedField("metric_range", 1.5, "table 2")},
            "b": {"metric_range": ExtractedField("metric_range", 1.5, "table 2b")},
        }
    )

    def lint(fields):
        return [LintFinding(field="metric_range", reason="metric_range_invalid")]

    result = run_understanding(extract=extractor, lint=lint, out_path=tmp_path / "u.json")
    assert "lint:metric_range_invalid:metric_range" in result.blocking
    assert "lint:metric_range_invalid:metric_range" not in result.advisory


def test_lint_on_llm_only_field_never_blocks_even_when_cross_verified(tmp_path):
    """THE F9 test: a span-less field is advisory forever, even cross-verified."""
    extractor = ScriptedExtractor(
        {
            "a": {"claim": ExtractedField("claim", "sota", None)},
            "b": {"claim": ExtractedField("claim", "sota", None)},
        }
    )

    def lint(fields):
        return [LintFinding(field="claim", reason="claim_unsupported")]

    result = run_understanding(extract=extractor, lint=lint, out_path=tmp_path / "u.json")
    assert "claim" in result.cross_verified
    assert "claim" not in result.source_spans
    assert "lint:claim_unsupported:claim" in result.advisory
    # Full-tuple equality, not membership (see the two sibling tests above).
    assert result.blocking == ()


def test_lint_on_unresolved_field_never_blocks(tmp_path):
    extractor = ScriptedExtractor(
        {
            "a": {"opt": ExtractedField("opt", "adam", "s1")},
            "b": {"opt": ExtractedField("opt", "sgd", "s2")},
            "targeted:opt": {"opt": ExtractedField("opt", "rmsprop", "s3")},
        }
    )

    def lint(fields):
        return [LintFinding(field="opt", reason="optimizer_unknown")]

    result = run_understanding(extract=extractor, lint=lint, out_path=tmp_path / "u.json")
    assert "opt" in result.unresolved
    assert "lint:optimizer_unknown:opt" in result.advisory
    # Full-tuple equality, not membership: catches a mutation that appends
    # an unresolved-field reason to blocking under a different string too.
    assert result.blocking == ()


def test_probe_confirmed_gap_blocks_unconfirmed_advisory(tmp_path):
    extractor = ScriptedExtractor({"a": {}, "b": {}})

    def probe():
        return [
            AssetGap(asset="hf:foo/bar", kind="missing", probe_confirmed=True, detail="404"),
            AssetGap(asset="hf:baz/qux", kind="gated", probe_confirmed=False, detail="unchecked"),
        ]

    result = run_understanding(extract=extractor, probe_assets=probe, out_path=tmp_path / "u.json")
    assert "asset:missing:hf:foo/bar" in result.blocking
    assert "asset:gated:hf:baz/qux" in result.advisory
    assert "asset:gated:hf:baz/qux" not in result.blocking


# ---------------------------------------------------------------------------
# Persisted payload: hash determinism + atomic write
# ---------------------------------------------------------------------------


def test_persisted_payload_hash_is_deterministic_and_stable_across_key_order(tmp_path):
    extractor1 = ScriptedExtractor(
        {
            "a": {
                "lr": ExtractedField("lr", 0.01, "s"),
                "seed": ExtractedField("seed", 42, "s2"),
            },
            "b": {
                "seed": ExtractedField("seed", 42, "s2b"),
                "lr": ExtractedField("lr", 0.01, "sb"),
            },
        }
    )
    extractor2 = ScriptedExtractor(
        {
            "a": {
                "seed": ExtractedField("seed", 42, "s2"),
                "lr": ExtractedField("lr", 0.01, "s"),
            },
            "b": {
                "lr": ExtractedField("lr", 0.01, "sb"),
                "seed": ExtractedField("seed", 42, "s2b"),
            },
        }
    )
    r1 = run_understanding(extract=extractor1, out_path=tmp_path / "u1.json")
    r2 = run_understanding(extract=extractor2, out_path=tmp_path / "u2.json")
    assert r1.sha256 == r2.sha256
    assert r1.fields == r2.fields


def test_persisted_payload_hash_stable_across_lint_finding_order(tmp_path):
    """Reviewer scenario (Important fix): the SAME two lint findings returned
    in swapped caller order must canonicalize identically -- sorted by
    (field, reason), not caller-supplied order."""
    extractor1 = ScriptedExtractor(
        {
            "a": {"lr": ExtractedField("lr", 0.01, "s")},
            "b": {"lr": ExtractedField("lr", 0.01, "sb")},
        }
    )
    extractor2 = ScriptedExtractor(
        {
            "a": {"lr": ExtractedField("lr", 0.01, "s")},
            "b": {"lr": ExtractedField("lr", 0.01, "sb")},
        }
    )
    finding_alpha = LintFinding(field="alpha", reason="metric_range_invalid")
    finding_zeta = LintFinding(field="zeta", reason="optimizer_unknown")

    r1 = run_understanding(
        extract=extractor1,
        lint=lambda fields: [finding_alpha, finding_zeta],
        out_path=tmp_path / "forward.json",
    )
    r2 = run_understanding(
        extract=extractor2,
        lint=lambda fields: [finding_zeta, finding_alpha],
        out_path=tmp_path / "reversed.json",
    )

    assert r1.sha256 == r2.sha256
    assert r1.lint_findings == r2.lint_findings == (finding_alpha, finding_zeta)
    assert r1.advisory == r2.advisory


def test_persisted_payload_hash_stable_across_asset_gap_order(tmp_path):
    """Same scenario for asset gaps -- sorted by (asset, kind, detail)."""
    extractor1 = ScriptedExtractor(
        {
            "a": {"lr": ExtractedField("lr", 0.01, "s")},
            "b": {"lr": ExtractedField("lr", 0.01, "sb")},
        }
    )
    extractor2 = ScriptedExtractor(
        {
            "a": {"lr": ExtractedField("lr", 0.01, "s")},
            "b": {"lr": ExtractedField("lr", 0.01, "sb")},
        }
    )
    gap_a = AssetGap(asset="a-asset", kind="missing", probe_confirmed=True, detail="404")
    gap_b = AssetGap(asset="b-asset", kind="gated", probe_confirmed=False, detail="unchecked")

    r1 = run_understanding(
        extract=extractor1,
        probe_assets=lambda: [gap_a, gap_b],
        out_path=tmp_path / "forward.json",
    )
    r2 = run_understanding(
        extract=extractor2,
        probe_assets=lambda: [gap_b, gap_a],
        out_path=tmp_path / "reversed.json",
    )

    assert r1.sha256 == r2.sha256
    assert r1.asset_gaps == r2.asset_gaps == (gap_a, gap_b)
    assert r1.blocking == r2.blocking == ("asset:missing:a-asset",)
    assert r1.advisory == r2.advisory == ("asset:gated:b-asset",)


def test_atomic_write_and_result_roundtrip(tmp_path):
    extractor = ScriptedExtractor(
        {
            "a": {"lr": ExtractedField("lr", 0.01, "s1")},
            "b": {"lr": ExtractedField("lr", 0.01, "s1b")},
        }
    )
    out_path = tmp_path / "campaign" / "understanding.json"
    result = run_understanding(extract=extractor, out_path=out_path)

    assert out_path.exists()
    tmp_marker = out_path.with_suffix(out_path.suffix + ".tmp")
    assert not tmp_marker.exists()

    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["sha256"] == result.sha256
    canonical = json.dumps(on_disk["payload"], sort_keys=True, separators=(",", ":"))
    recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert recomputed == result.sha256
    assert on_disk["payload"]["fields"] == {"lr": 0.01}
