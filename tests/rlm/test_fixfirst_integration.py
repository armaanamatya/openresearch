"""Integration: the deterministic floor feeds the bounded fix-first repair loop.

Hermetic — no GPU, no LLM, no sandbox. Composes the Tier-1 zero-metrics veto
(``zero_metrics_detection``) with the repair loop (``ForcedIterationPolicy``)
over a planted v6-shaped fake metrics.json, proving the spec §8 behaviors at the
seam level:

  * a planted fake that is never really fixed → the loop refuses, then stops
    HONESTLY as repair_exhausted within budget (never refuses forever, never
    ships the fake silently);
  * a planted fake that IS really fixed (real metrics replace the zeros) → the
    floor stops vetoing, the success clears the trigger, and FINAL_VAR is accepted.

This is the end-to-end loop a real run drives — minus the LLM/GPU it would call.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.agents.rlm.forced_iteration import ForcedIterationPolicy
from backend.agents.rlm.zero_metrics_detection import zero_metrics_should_veto


def _write_metrics(project_dir: Path, metrics: dict) -> None:
    code = project_dir / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def _read_metrics(project_dir: Path) -> dict:
    mp = project_dir / "code" / "metrics.json"
    return json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}


def _fingerprint(project_dir: Path) -> str:
    from backend.agents.rlm.external_validator import evidence_fingerprint
    return evidence_fingerprint(_read_metrics(project_dir))


# The v6 shape: real GPU claim + real wall-time, but every result metric is 0.0,
# and no provenance.json links the metric to a real output.
_FAKE = {"loss": 0.0, "reward": 0.0, "accuracy": 0.0, "device": "cuda", "wall_time_s": 500.0}
_REAL = {"loss": 1.23, "reward": 0.42, "accuracy": 0.71, "device": "cuda", "wall_time_s": 510.0}


def test_planted_fake_repairs_then_honest_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ZERO_METRICS_GUARD", "1")
    _write_metrics(tmp_path, _FAKE)

    # Tier-1 floor: GPU-claim + all-zero result values + no provenance → veto.
    assert zero_metrics_should_veto(
        _read_metrics(tmp_path), gpu_claim=True, provenance_present=False
    ) is True

    # The veto degrades to fabrication_suspected, which drives the repair loop.
    policy = ForcedIterationPolicy(
        min_iterations=0,
        evidence_fingerprint=lambda: _fingerprint(tmp_path),
        current_iteration=lambda: 1,
        remaining_s=lambda: 9999.0,
    )
    policy._total_run_experiments = 1

    refusals = []
    for _ in range(5):
        # Each repair attempt leaves the evidence unchanged (the executor failed
        # to wire training to real outputs) → no fingerprint progress.
        policy.record_repair_attempt("fabrication_suspected")
        refusals.append(policy.should_refuse()[0])

    # Bounded + honest: stops as repair_exhausted within budget, never forever.
    assert policy._terminal_failure_class == "repair_exhausted"
    assert refusals[0] is True          # first attempts refuse (fix-first)
    assert refusals[-1] is False        # then ships the honest report
    assert True in refusals             # it really did force at least one repair


def test_planted_fake_then_real_repair_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ZERO_METRICS_GUARD", "1")
    _write_metrics(tmp_path, _FAKE)

    policy = ForcedIterationPolicy(
        min_iterations=0,
        evidence_fingerprint=lambda: _fingerprint(tmp_path),
        current_iteration=lambda: 1,
        remaining_s=lambda: 9999.0,
    )
    policy._total_run_experiments = 1

    policy.record_repair_attempt("fabrication_suspected")
    assert policy.should_refuse()[0] is True  # refuses the fake

    # A REAL repair: the executor rewires training so the metrics become real.
    _write_metrics(tmp_path, _REAL)
    assert zero_metrics_should_veto(
        _read_metrics(tmp_path), gpu_claim=True, provenance_present=False
    ) is False  # the floor no longer vetoes — the evidence is real

    policy.clear_repair_trigger()  # the success run_experiment clears the trigger
    assert policy.should_refuse()[0] is False  # accepts — real evidence ships


def test_default_off_no_veto_no_loop(tmp_path, monkeypatch):
    # Flag off: the floor never vetoes and the loop never engages (byte-identical).
    monkeypatch.delenv("OPENRESEARCH_ZERO_METRICS_GUARD", raising=False)
    _write_metrics(tmp_path, _FAKE)
    assert zero_metrics_should_veto(
        _read_metrics(tmp_path), gpu_claim=True, provenance_present=False
    ) is False
    policy = ForcedIterationPolicy(min_iterations=0)  # evidence_fingerprint unset
    assert policy.evidence_fingerprint is None
