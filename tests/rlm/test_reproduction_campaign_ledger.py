"""Fail-closed spend-ledger tests for ``ReproductionCampaign`` (Unit 1).

Covers ``CampaignLedger``'s atomic+fsync durability guarantees in isolation
from the state machine: atomic snapshot writes, append-only newline-guarded
rows, corrupt-tail tolerance vs. mid-file-corruption halt, and unwritable-dir
fail-closed behavior. Spec: docs/history/specs/2026-07-01-reproduction-
campaign-and-self-improving-harness-design.md §5, F1, F7.
"""

from __future__ import annotations

import json
import os

import pytest

from backend.agents.rlm.reproduction_campaign import (
    CampaignLedger,
    CampaignLedgerError,
    CampaignState,
)


def _state(**overrides) -> CampaignState:
    base = dict(
        project_id="proj_1",
        paper_ref="2605.15155",
        state="attempt_loop",
        next_attempt_n=1,
        mode="unattended",
        driver="live_cli",
        budget={"max_llm_usd": 10.0, "max_gpu_usd": 10.0, "max_gpu_hours": 2.0, "max_attempts": 6},
        spent={"llm_usd": 0.0, "gpu_usd": 0.0, "gpu_hours": 0.0, "wall_s": 0.0},
        scope_rung=0,
        in_flight=None,
        understanding_sha256=None,
        rubric_sha256=None,
        steering_cursor=0,
        pending_approval=None,
        warnings=[],
        terminal=None,
        created_at=1000.0,
        updated_at=1000.0,
    )
    base.update(overrides)
    return CampaignState(**base)


def test_write_state_is_atomic_and_fsynced(tmp_path, monkeypatch):
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    ledger = CampaignLedger(campaign_dir)

    ledger.write_state(_state())
    before = (campaign_dir / "campaign.json").read_text(encoding="utf-8")
    assert json.loads(before)["project_id"] == "proj_1"

    import backend.agents.rlm.reproduction_campaign as mod

    def _raise_replace(_src, _dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(mod.os, "replace", _raise_replace)

    with pytest.raises(CampaignLedgerError):
        ledger.write_state(_state(next_attempt_n=2))

    after = (campaign_dir / "campaign.json").read_text(encoding="utf-8")
    assert after == before
    assert json.loads(after)["next_attempt_n"] == 1
    # No leftover temp file from the failed swap.
    assert [p for p in campaign_dir.iterdir() if p.name.startswith(".campaign_json_")] == []


def test_append_row_fsyncs_and_repairs_torn_tail(tmp_path, monkeypatch):
    """append_row fsyncs each row and REPAIRS (never entombs) a torn tail.

    A crash mid-append can leave a partial, newline-less JSON fragment as the
    last bytes on disk. The next append_row must discard exactly that
    fragment -- never a previously-fsynced complete row -- so read_rows never
    trips over what would otherwise become a corrupt MID-FILE line (the
    tail-poisoning bug: a naive newline-PREFIX would entomb the fragment
    mid-file and read_rows would then raise CampaignLedgerError forever).
    """
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    ledger = CampaignLedger(campaign_dir)
    attempts_path = campaign_dir / "attempts.jsonl"

    ledger.append_row({"attempt_n": 1, "status": "launched"})
    ledger.append_row({"attempt_n": 1, "status": "assessed"})

    # Simulate a crash mid-write: a partial JSON fragment with no trailing
    # newline, appended directly (bypassing the ledger, the way a torn write
    # would actually leave the file).
    with attempts_path.open("ab") as fh:
        fh.write(b'{"attempt_n": 2, "status": "TORN_FRAGMENT_MARKER')
    content_before_repair = attempts_path.read_text(encoding="utf-8")
    assert "TORN_FRAGMENT_MARKER" in content_before_repair  # sanity: torn tail present
    assert not content_before_repair.endswith("\n")

    import backend.agents.rlm.reproduction_campaign as mod

    real_fsync = os.fsync
    fsync_calls = {"n": 0}

    def _counting_fsync(fd):
        fsync_calls["n"] += 1
        return real_fsync(fd)

    monkeypatch.setattr(mod.os, "fsync", _counting_fsync)

    ledger.append_row({"attempt_n": 2, "status": "launched"})  # must not raise

    assert fsync_calls["n"] >= 1

    content = attempts_path.read_text(encoding="utf-8")
    assert "TORN_FRAGMENT_MARKER" not in content  # fragment repaired away, not entombed
    assert content.endswith("\n")

    rows = ledger.read_rows()  # must not raise -- no mid-file corruption survives
    assert rows == [
        {"attempt_n": 1, "status": "launched"},
        {"attempt_n": 1, "status": "assessed"},
        {"attempt_n": 2, "status": "launched"},
    ]


def test_torn_tail_repair_preserves_prior_rows_byte_identically(tmp_path):
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    ledger = CampaignLedger(campaign_dir)
    attempts_path = campaign_dir / "attempts.jsonl"

    ledger.append_row({"attempt_n": 1, "status": "launched"})
    ledger.append_row({"attempt_n": 1, "status": "assessed"})
    prior_bytes = attempts_path.read_bytes()

    with attempts_path.open("ab") as fh:
        fh.write(b'{"attempt_n": 2, "status": "torn_and_incomplete_fragmen')

    ledger.append_row({"attempt_n": 2, "status": "launched"})

    final_bytes = attempts_path.read_bytes()
    # The repair truncates ONLY the torn fragment -- the two prior,
    # already-fsynced rows must survive byte-for-byte, untouched.
    assert final_bytes.startswith(prior_bytes)
    appended_bytes = final_bytes[len(prior_bytes):]
    assert appended_bytes == (
        json.dumps({"attempt_n": 2, "status": "launched"}, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")


def test_read_rows_tolerates_corrupt_final_line_only(tmp_path):
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    ledger = CampaignLedger(campaign_dir)
    attempts_path = campaign_dir / "attempts.jsonl"
    attempts_path.write_text(
        json.dumps({"attempt_n": 1, "status": "launched"}) + "\n"
        + '{"attempt_n": 1, "status": "assessed", BROKEN',
        encoding="utf-8",
    )

    rows = ledger.read_rows()
    assert len(rows) == 1
    assert rows[0]["status"] == "launched"


def test_read_rows_halts_on_mid_file_corruption(tmp_path):
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    ledger = CampaignLedger(campaign_dir)
    attempts_path = campaign_dir / "attempts.jsonl"
    attempts_path.write_text(
        '{"attempt_n": 1, BROKEN\n' + json.dumps({"attempt_n": 1, "status": "assessed"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CampaignLedgerError):
        ledger.read_rows()


def test_unwritable_dir_raises_ledger_error(tmp_path):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root bypasses directory permission checks")
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    campaign_dir.chmod(0o500)
    ledger = CampaignLedger(campaign_dir)
    try:
        with pytest.raises(CampaignLedgerError):
            ledger.write_state(_state())
        with pytest.raises(CampaignLedgerError):
            ledger.append_row({"attempt_n": 1, "status": "launched"})
    finally:
        campaign_dir.chmod(0o700)


def test_latest_by_status_last_writer_wins_rows_retained(tmp_path):
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    ledger = CampaignLedger(campaign_dir)
    ledger.append_row({"attempt_n": 1, "status": "launched", "v": 1})
    ledger.append_row({"attempt_n": 1, "status": "assessed", "v": 1})
    ledger.append_row({"attempt_n": 1, "status": "assessed", "v": 2})

    rows = ledger.read_rows()
    assert len(rows) == 3  # all rows retained on disk

    latest = CampaignLedger.latest_by_status(rows, 1)
    assert latest["launched"]["v"] == 1
    assert latest["assessed"]["v"] == 2
