"""Offline ASHA shadow-advisory analysis
(`backend.agents.rlm.asha_shadow_report`): folds campaign ledger `decided` rows
into a coverage + kill/freeze/promote rollup, fail-soft on shape drift, with a CLI
that accepts a run dir / campaign dir / ledger file."""
import json

from backend.agents.rlm.asha_shadow_report import (
    analyze_shadow_rows,
    main,
    render_report,
)


def _decided(attempt_n, kind="CONTINUE", advisory=None, rule="r"):
    decision = {"kind": kind, "rule": rule, "stop_reason": None}
    if advisory is not None:
        decision["asha_advisory"] = advisory
    return {"attempt_n": attempt_n, "status": "decided", "decision": decision}


def _adv(rung, actions, *, budget=6.0, cap=8):
    return {
        "rung": rung,
        "width_meter": {"gpu_usd_budget": budget, "a100_cap": cap, "gpu_usd_spent": 4.0},
        "decisions": [
            {"branch_id": b, "action": a, "reason": rsn} for b, a, rsn in actions
        ],
    }


def test_ignores_non_decided_rows():
    rows = [
        {"attempt_n": 1, "status": "planned"},
        {"attempt_n": 1, "status": "launched"},
        {"attempt_n": 1, "status": "assessed", "assessment": {}},
        _decided(1),
    ]
    report = analyze_shadow_rows(rows)
    assert report.total_decided == 1  # only the "decided" row counts


def test_coverage_counts_advisory_rows():
    rows = [
        _decided(1, advisory=None),  # flag was off for this decide
        _decided(2, advisory=_adv(0, [("1", "promote", "top_k_by_score")])),
    ]
    report = analyze_shadow_rows(rows)
    assert report.total_decided == 2
    assert report.with_advisory == 1


def test_rollup_uses_last_advisory():
    # Two advisory points; the rollup must reflect the LAST (most complete cohort).
    rows = [
        _decided(2, advisory=_adv(0, [("1", "promote", "top_k_by_score")])),
        _decided(
            3,
            advisory=_adv(
                1,
                [
                    ("1", "promote", "top_k_by_score"),
                    ("2", "freeze", "halved_below_topk"),
                    ("3", "kill", "breakage_true_kill"),
                ],
            ),
        ),
    ]
    report = analyze_shadow_rows(rows)
    assert report.final_promoted == ("1",)
    assert report.final_frozen == ("2",)
    assert report.final_killed == ("3",)  # provable-breakage kill surfaced


def test_fail_soft_on_malformed_advisory():
    rows = [
        {"attempt_n": 1, "status": "decided", "decision": {"kind": "CONTINUE"}},  # no adv
        {"attempt_n": 2, "status": "decided", "decision": {"asha_advisory": "boom"}},  # wrong type
        {"attempt_n": 3, "status": "decided"},  # no decision at all
    ]
    report = analyze_shadow_rows(rows)  # must not raise
    assert report.total_decided == 3
    assert report.with_advisory == 0
    assert report.final_killed == ()


def test_render_flag_off_message():
    report = analyze_shadow_rows([_decided(1, advisory=None)])
    text = render_report(report)
    assert "OPENRESEARCH_SCHEDULER_TREE=1" in text  # tells the operator how to populate it


def test_render_surfaces_kill_savings_line():
    rows = [_decided(1, advisory=_adv(0, [("3", "kill", "breakage_true_kill")]))]
    text = render_report(analyze_shadow_rows(rows))
    assert "PROVABLE breakage" in text
    assert "true-kill" in text


def test_main_reads_run_dir_and_renders(tmp_path, capsys):
    # Mirror the real layout: runs/<id>/campaign/attempts.jsonl
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    rows = [
        _decided(1, advisory=_adv(0, [("1", "promote", "top_k_by_score"),
                                      ("2", "freeze", "halved_below_topk")])),
    ]
    (campaign / "attempts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    rc = main([str(tmp_path)])  # pass the RUN dir; tool finds campaign/attempts.jsonl
    assert rc == 0
    out = capsys.readouterr().out
    assert "with advisory: 1" in out
    assert "1:promote" in out


def test_main_json_mode(tmp_path, capsys):
    ledger = tmp_path / "attempts.jsonl"
    rows = [_decided(1, advisory=_adv(0, [("3", "kill", "breakage_true_kill")]))]
    ledger.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    rc = main([str(ledger), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["with_advisory"] == 1
    assert payload["final_killed"] == ["3"]


def test_main_missing_ledger_errors(tmp_path, capsys):
    rc = main([str(tmp_path / "nonexistent")])
    assert rc == 2  # clean non-zero, not a traceback
    assert "error:" in capsys.readouterr().err


def test_main_tolerates_torn_final_line(tmp_path, capsys):
    ledger = tmp_path / "attempts.jsonl"
    good = json.dumps(_decided(1, advisory=_adv(0, [("1", "promote", "x")])))
    ledger.write_text(good + "\n{ torn partial line", encoding="utf-8")  # crash-mid-append
    rc = main([str(ledger)])
    assert rc == 0  # the torn tail is skipped, not fatal
    assert "with advisory: 1" in capsys.readouterr().out
