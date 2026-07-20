"""Hermetic evidence tests for the offline ASHA authority A/B gate."""
from __future__ import annotations

import json
from pathlib import Path

from backend.agents.rlm.asha_authority_gate import evaluate_manifest, main


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _campaign_state(*, terminal_kind: str = "EXHAUSTED", paper: str = "2605.15155") -> dict:
    return {
        "project_id": "fixture",
        "paper_ref": paper,
        "state": "terminal",
        "rubric_sha256": "a" * 64,
        "terminal": {"kind": terminal_kind, "rule": "budget", "stop_reason": "cap"},
    }


def _advisory(*, action: str = "promote", branch_id: str = "1") -> dict:
    return {
        "rung": 0,
        "width_meter": {"gpu_usd_budget": 5.0, "a100_cap": 1},
        "decisions": [
            {"branch_id": branch_id, "action": action, "reason": "deterministic_rank"}
        ],
    }


def _audit(
    *,
    applied: bool = False,
    action: str = "continue",
    basis: str = "no deterministic metric and optimizer-step lineage",
    failure_class: str | None = None,
) -> dict:
    out = {
        "enabled": True,
        "applied": applied,
        "action": action,
        "deterministic_evidence_basis": basis,
    }
    if failure_class is not None:
        out["failure_class"] = failure_class
    return out


def _calibration(scores: list[float], *, run_id: str) -> dict:
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
    return {
        "run_id": run_id,
        "k": len(scores),
        "overall": {
            "n": len(scores),
            "scores": scores,
            "mean": mean,
            "min": min(scores),
            "max": max(scores),
            "stdev": variance**0.5,
        },
    }


def _run(
    root: Path,
    name: str,
    *,
    authoritative: bool,
    score: float = 0.8,
    terminal_kind: str = "EXHAUSTED",
    advisory: dict | None = None,
    audit: dict | None = None,
    failure_class: str | None = None,
) -> Path:
    run = root / name
    _json(run / "campaign" / "campaign.json", _campaign_state(terminal_kind=terminal_kind))
    decision = {
        "kind": "CONTINUE",
        "rule": "continue",
        "stop_reason": None,
        "next_plan": {"scope_rung": 0},
        "champion_attempt_n": None,
        "asha_advisory": advisory or _advisory(),
    }
    if authoritative:
        decision["asha_authority_audit"] = audit or _audit()
    rows = [
        {
            "attempt_n": 1,
            "status": "assessed",
            "assessment": {"failure_class": failure_class},
        },
        {"attempt_n": 1, "status": "decided", "decision": decision},
    ]
    (run / "campaign" / "attempts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    _json(run / "final_report.json", {"rubric": {"overall_score": score}})
    _json(
        run / "tokens_total.json",
        {"schema_version": 1, "run_id": name, "grand_total": {"calls": 1}},
    )
    _json(run / "calibration.json", _calibration([0.79, 0.80, 0.81, 0.80, 0.80], run_id=name))
    return run


def _cost(run: Path, usd: float) -> dict:
    record_id = f"provider-{run.name}"
    _json(
        run / "gpu_observation.json",
        {
            "schema_version": 1,
            "source": "provider_bill_export",
            "records": [{
                "record_id": record_id,
                "run_id": run.name,
                "provider_record_id": f"bill-{run.name}",
                "gpu_sku": "NVIDIA-A100",
                "cost_usd": usd,
            }],
        },
    )
    return {
        "gpu_usd": usd,
        "run_id": run.name,
        "tokens_total_path": "tokens_total.json",
        "gpu_evidence_path": "gpu_observation.json",
        "source": "provider_bill_export",
        "evidence_record_id": record_id,
    }


def _pair(pair_id: str, shadow: Path, authority: Path, *, shadow_usd: float = 10.0, authority_usd: float = 8.0) -> dict:
    return {
        "pair_id": pair_id,
        "shadow": str(shadow),
        "authoritative": str(authority),
        "calibrations": {
            "shadow": "calibration.json",
            "authoritative": "calibration.json",
        },
        "verified_gpu_costs": {
            "shadow": _cost(shadow, shadow_usd),
            "authoritative": _cost(authority, authority_usd),
        },
    }


def _manifest(*pairs: dict) -> dict:
    return {"schema_version": 1, "pairs": list(pairs)}


def test_rejects_self_authored_applied_audit_even_with_bound_costs(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"shadow-{n}", authoritative=False)
        authority = _run(
            tmp_path,
            f"authority-{n}",
            authoritative=True,
            audit=_audit(applied=n == 0, action="promote" if n == 0 else "continue"),
        )
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)

    assert report.eligible_for_operator_review is False
    assert report.applied_action_count == 0
    assert report.total_verified_gpu_savings_usd == 6.0
    assert "authoritative_evidence_receipt_unavailable" in report.failure_codes


def test_rejects_noop_audit_pasted_onto_an_incomplete_decision(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"shadow-{n}", authoritative=False)
        authority = _run(
            tmp_path,
            f"authority-{n}",
            authoritative=True,
            audit=_audit(applied=n == 0, action="promote" if n == 0 else "continue"),
        )
        if n == 0:
            ledger = authority / "campaign" / "attempts.jsonl"
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            rows[-1]["decision"].pop("next_plan")
            rows[-1]["decision"].pop("champion_attempt_n")
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)

    assert report.eligible_for_operator_review is False
    assert "decision_contract_missing" in report.failure_codes
    assert "unlinked_authority_audit" in report.failure_codes


def test_rejects_fewer_than_three_pairs(tmp_path):
    pairs = [
        _pair(f"p{n}", _run(tmp_path, f"s{n}", authoritative=False), _run(tmp_path, f"a{n}", authoritative=True))
        for n in range(2)
    ]
    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "minimum_pair_count" in report.failure_codes


def test_rejects_missing_shadow_advisory_coverage(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False)
        authority = _run(tmp_path, f"a{n}", authoritative=True)
        if n == 0:
            ledger = shadow / "campaign" / "attempts.jsonl"
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            rows[-1]["decision"].pop("asha_advisory")
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "shadow_advisory_coverage" in report.failure_codes


def test_rejects_unpersisted_or_unapplied_authority_audit(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False)
        authority = _run(tmp_path, f"a{n}", authoritative=True)
        if n == 0:
            ledger = authority / "campaign" / "attempts.jsonl"
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            rows[-1]["decision"].pop("asha_authority_audit")
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "authority_audit_missing" in report.failure_codes
    assert "meaningful_authority_action" in report.failure_codes


def test_rejects_any_nonliteral_training_diverged_kill(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False)
        authority = _run(tmp_path, f"a{n}", authoritative=True)
        if n == 0:
            authority = _run(
                tmp_path,
                f"a{n}",
                authoritative=True,
                audit=_audit(applied=True, action="kill", failure_class="cell_execution_error"),
            )
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "invalid_true_kill" in report.failure_codes


def test_rejects_inadequate_or_noisy_calibration(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False)
        authority = _run(tmp_path, f"a{n}", authoritative=True)
        if n == 0:
            _json(authority / "calibration.json", _calibration([0.5, 0.7, 0.5, 0.7, 0.5], run_id=authority.name))
        if n == 1:
            _json(shadow / "calibration.json", _calibration([0.8, 0.8, 0.8, 0.8], run_id=shadow.name))
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "grader_sigma" in report.failure_codes
    assert "grader_calibration_k" in report.failure_codes


def test_rejects_score_degradation_beyond_pair_sigma(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False, score=0.80)
        authority = _run(tmp_path, f"a{n}", authoritative=True, score=0.75 if n == 0 else 0.80)
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "score_degradation" in report.failure_codes


def test_rejects_ledger_only_or_nonpositive_cost_claims(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False)
        authority = _run(tmp_path, f"a{n}", authoritative=True)
        pair = _pair(f"p{n}", shadow, authority, shadow_usd=8.0, authority_usd=8.0)
        if n == 0:
            pair["verified_gpu_costs"]["shadow"] = {
                "gpu_usd": 10.0,
                "run_id": shadow.name,
                "source": "cost_ledger",
                "tokens_total_path": "tokens_total.json",
                "gpu_evidence_path": "cost_ledger.jsonl",
            }
            (shadow / "cost_ledger.jsonl").write_text("{}\n", encoding="utf-8")
        pairs.append(pair)

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "verified_gpu_cost" in report.failure_codes
    assert "positive_verified_gpu_saving" in report.failure_codes


def test_rejects_empty_or_unbound_cost_evidence_even_when_manifest_claims_saving(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False)
        authority = _run(tmp_path, f"a{n}", authoritative=True)
        pair = _pair(f"p{n}", shadow, authority)
        if n == 0:
            pair["verified_gpu_costs"]["shadow"]["evidence_record_id"] = "absent"
            _json(shadow / "gpu_observation.json", {"schema_version": 1, "source": "provider_bill_export", "records": []})
        pairs.append(pair)

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "verified_gpu_cost" in report.failure_codes


def test_rejects_unbound_direct_grader_calibration(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False)
        authority = _run(tmp_path, f"a{n}", authoritative=True)
        if n == 0:
            calibration = _calibration([0.79, 0.80, 0.81, 0.80, 0.80], run_id="other-run")
            _json(authority / "calibration.json", calibration)
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "unbound_grader_calibration" in report.failure_codes


def test_rejects_mismatched_terminal_evidence_and_malformed_ledger(tmp_path):
    pairs = []
    for n in range(3):
        shadow = _run(tmp_path, f"s{n}", authoritative=False)
        authority = _run(
            tmp_path,
            f"a{n}",
            authoritative=True,
            terminal_kind="CONTRADICTED" if n == 0 else "EXHAUSTED",
        )
        if n == 1:
            with (shadow / "campaign" / "attempts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{ broken json\n")
        pairs.append(_pair(f"p{n}", shadow, authority))

    report = evaluate_manifest(_manifest(*pairs), manifest_dir=tmp_path)
    assert report.eligible_for_operator_review is False
    assert "terminal_evidence_mismatch" in report.failure_codes
    assert "malformed_campaign_ledger" in report.failure_codes


def test_cli_writes_machine_readable_rejection(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    _json(manifest_path, _manifest())
    assert main([str(manifest_path), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["eligible_for_operator_review"] is False
    assert "minimum_pair_count" in payload["failure_codes"]
