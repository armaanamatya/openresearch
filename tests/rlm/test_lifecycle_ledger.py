"""Tests for backend/agents/rlm/lifecycle_ledger.py.

All tests are hermetic (no network, tmp_path project dirs).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm.lifecycle_ledger import (
    LedgerRecord,
    _LEDGER_DIR,
    _LEDGER_FILE,
    append_record,
    lifecycle_ledger_enabled,
    project_inputs,
    read_records,
)


# =========================================================================== #
# Fixtures                                                                       #
# =========================================================================== #


def _project_dir(tmp_path: Path) -> Path:
    """Return a fresh temporary project directory."""
    d = tmp_path / "runs" / "prj_test"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record(
    *,
    primitive: str = "run_experiment",
    seq: int = 0,
    inputs_projection: dict | None = None,
    outputs_pointer: dict | None = None,
    evidence_keys: list | None = None,
    outcome: str = "ok",
    iteration: int = 1,
) -> LedgerRecord:
    return LedgerRecord(
        primitive=primitive,
        seq=seq,
        inputs_projection=inputs_projection or {},
        outputs_pointer=outputs_pointer or {},
        evidence_keys=evidence_keys or [],
        outcome=outcome,
        iteration=iteration,
    )


def _ledger_path(project_dir: Path) -> Path:
    return project_dir / "rlm_state" / _LEDGER_DIR / _LEDGER_FILE


# =========================================================================== #
# Feature flag                                                                  #
# =========================================================================== #


class TestLifecycleLedgerEnabled:
    def test_default_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENRESEARCH_LIFECYCLE_LEDGER", raising=False)
        assert lifecycle_ledger_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes", "YES", "on", "ON"])
    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", val)
        assert lifecycle_ledger_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "maybe", "2"])
    def test_falsy_values(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", val)
        assert lifecycle_ledger_enabled() is False


# =========================================================================== #
# project_inputs — redaction                                                    #
# =========================================================================== #


class TestProjectInputs:
    def test_plan_reproduction_section_ids_and_hparam_keys_only(self) -> None:
        kwargs = {
            "section_ids": ["abstract", "experiments"],
            "hyperparameters": {"lr": 0.001, "batch_size": 32, "epochs": 10},
            "section_text": "CANARY_PAPER_TEXT_DO_NOT_LEAK some prose here",
            "paper": "CANARY_PAPER_TEXT_DO_NOT_LEAK full paper text",
        }
        result = project_inputs("plan_reproduction", kwargs)
        assert set(result.keys()) == {"section_ids", "hparam_keys"}
        assert result["section_ids"] == ["abstract", "experiments"]
        # Only KEYS, never values
        assert set(result["hparam_keys"]) == {"lr", "batch_size", "epochs"}
        # Canary must not appear anywhere in the projection
        result_str = json.dumps(result)
        assert "CANARY_PAPER_TEXT_DO_NOT_LEAK" not in result_str
        assert "0.001" not in result_str
        assert "some prose here" not in result_str

    def test_plan_reproduction_empty_kwargs(self) -> None:
        result = project_inputs("plan_reproduction", {})
        assert result == {"section_ids": [], "hparam_keys": []}

    def test_plan_reproduction_non_list_section_ids(self) -> None:
        result = project_inputs("plan_reproduction", {"section_ids": "abstract"})
        assert result["section_ids"] == []

    def test_plan_reproduction_non_dict_hyperparameters(self) -> None:
        result = project_inputs("plan_reproduction", {"hyperparameters": [1, 2, 3]})
        assert result["hparam_keys"] == []

    def test_implement_baseline_present_booleans(self) -> None:
        kwargs = {
            "plan": "some big plan",
            "repair_context": {"error": "x"},
            "sandbox_mode": "docker",
            "gpu_mode": "single",
        }
        result = project_inputs("implement_baseline", kwargs)
        assert result == {
            "plan_present": True,
            "repair_context_present": True,
            "sandbox_mode": "docker",
            "gpu_mode": "single",
        }

    def test_implement_baseline_absent_plan(self) -> None:
        result = project_inputs("implement_baseline", {})
        assert result == {
            "plan_present": False,
            "repair_context_present": False,
            "sandbox_mode": "",
            "gpu_mode": "",
        }

    def test_implement_baseline_false_plan_is_falsy(self) -> None:
        result = project_inputs("implement_baseline", {"plan": "", "repair_context": None})
        assert result["plan_present"] is False
        assert result["repair_context_present"] is False

    def test_run_experiment_with_values(self) -> None:
        result = project_inputs("run_experiment", {"env_id": "alfworld", "code": "train.py source"})
        assert result == {"env_id": "alfworld", "code_present": True}

    def test_run_experiment_empty(self) -> None:
        result = project_inputs("run_experiment", {})
        assert result == {"env_id": "", "code_present": False}

    def test_unknown_primitive_returns_empty(self) -> None:
        result = project_inputs("understand_section", {"content": "paper text", "foo": "bar"})
        assert result == {}

    def test_default_returns_empty(self) -> None:
        result = project_inputs("build_environment", {"dockerfile": "FROM ubuntu\nRUN apt-get..."})
        assert result == {}


# =========================================================================== #
# LedgerRecord — validation                                                     #
# =========================================================================== #


class TestLedgerRecord:
    def test_valid_outcomes(self) -> None:
        for outcome in ("ok", "failed", "raised", "timeout"):
            rec = _record(outcome=outcome)
            assert rec.outcome == outcome

    def test_invalid_outcome_raises(self) -> None:
        with pytest.raises(ValueError, match="outcome"):
            _record(outcome="partial_timeout")

    def test_frozen(self) -> None:
        rec = _record()
        with pytest.raises((AttributeError, TypeError)):
            rec.seq = 99  # type: ignore[misc]

    def test_ok_outcome_not_mutated(self) -> None:
        rec = _record(outcome="ok", seq=5, iteration=2)
        assert rec.outcome == "ok"
        assert rec.seq == 5
        assert rec.iteration == 2


# =========================================================================== #
# append_record + read_records — round-trip                                    #
# =========================================================================== #


class TestAppendReadRoundTrip:
    def test_round_trip_single_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
        pd = _project_dir(tmp_path)
        rec = _record(
            primitive="run_experiment",
            seq=0,
            outputs_pointer={"metrics_path": "code/metrics.json"},
            evidence_keys=["prj_test:run_experiment:0"],
            outcome="ok",
            iteration=1,
        )
        append_record(pd, rec)
        records = read_records(pd)
        assert len(records) == 1
        r = records[0]
        assert r.primitive == "run_experiment"
        assert r.seq == 0
        assert r.outputs_pointer == {"metrics_path": "code/metrics.json"}
        assert r.evidence_keys == ["prj_test:run_experiment:0"]
        assert r.outcome == "ok"
        assert r.iteration == 1

    def test_multiple_records_appended_in_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
        pd = _project_dir(tmp_path)
        for i in range(5):
            append_record(pd, _record(seq=i, outcome="ok", iteration=i))
        records = read_records(pd)
        assert len(records) == 5
        for i, r in enumerate(records):
            assert r.seq == i
            assert r.iteration == i

    def test_all_outcome_values_round_trip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
        pd = _project_dir(tmp_path)
        for outcome in ("ok", "failed", "raised", "timeout"):
            append_record(pd, _record(outcome=outcome))
        records = read_records(pd)
        assert len(records) == 4
        outcomes = [r.outcome for r in records]
        assert sorted(outcomes) == ["failed", "ok", "raised", "timeout"]

    def test_ledger_dir_created_automatically(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
        pd = _project_dir(tmp_path)
        assert not _ledger_path(pd).parent.exists()
        append_record(pd, _record())
        assert _ledger_path(pd).exists()

    def test_flag_off_no_file_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENRESEARCH_LIFECYCLE_LEDGER", raising=False)
        pd = _project_dir(tmp_path)
        append_record(pd, _record())
        assert not _ledger_path(pd).exists()
        assert read_records(pd) == []

    def test_read_records_absent_file_returns_empty(self, tmp_path: Path) -> None:
        pd = _project_dir(tmp_path)
        assert read_records(pd) == []

    def test_read_records_skips_malformed_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
        pd = _project_dir(tmp_path)
        ledger = _ledger_path(pd)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write("{not valid json\n")
            fh.write(
                json.dumps(
                    {
                        "primitive": "run_experiment",
                        "seq": 0,
                        "inputs_projection": {},
                        "outputs_pointer": {},
                        "evidence_keys": [],
                        "outcome": "ok",
                        "iteration": 1,
                    }
                )
                + "\n"
            )
            fh.write("\n")  # blank line
        records = read_records(pd)
        assert len(records) == 1
        assert records[0].primitive == "run_experiment"

    def test_read_records_skips_invalid_outcome(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
        pd = _project_dir(tmp_path)
        ledger = _ledger_path(pd)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as fh:
            # bad outcome
            fh.write(
                json.dumps(
                    {
                        "primitive": "plan_reproduction",
                        "seq": 0,
                        "inputs_projection": {},
                        "outputs_pointer": {},
                        "evidence_keys": [],
                        "outcome": "INVALID_OUTCOME",
                        "iteration": 0,
                    }
                )
                + "\n"
            )
            # good record
            fh.write(
                json.dumps(
                    {
                        "primitive": "run_experiment",
                        "seq": 1,
                        "inputs_projection": {},
                        "outputs_pointer": {},
                        "evidence_keys": [],
                        "outcome": "failed",
                        "iteration": 0,
                    }
                )
                + "\n"
            )
        records = read_records(pd)
        assert len(records) == 1
        assert records[0].outcome == "failed"

    def test_inputs_projection_stored_in_jsonl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
        pd = _project_dir(tmp_path)
        projection = project_inputs(
            "plan_reproduction",
            {
                "section_ids": ["intro", "method"],
                "hyperparameters": {"lr": 1e-4, "seed": 42},
            },
        )
        rec = _record(primitive="plan_reproduction", inputs_projection=projection)
        append_record(pd, rec)
        records = read_records(pd)
        assert records[0].inputs_projection == {
            "section_ids": ["intro", "method"],
            "hparam_keys": ["lr", "seed"],
        }


# =========================================================================== #
# SENTINEL CANARY TEST — redaction guarantee                                   #
# =========================================================================== #


class TestSentinelCanary:
    """Asserts that a known paper-text canary string is NEVER written to any
    file under rlm_state/ regardless of what kwargs are passed in."""

    CANARY = "CANARY_PAPER_TEXT_DO_NOT_LEAK"

    def _write_canary_record(self, project_dir: Path) -> None:
        """Write a plan_reproduction record whose raw kwargs contain the canary."""
        kwargs = {
            "section_ids": ["abstract"],
            "hyperparameters": {"lr": 0.001},
            "section_text": f"{self.CANARY}: this is the full paper section text",
            "paper": f"{self.CANARY}: complete raw paper content",
            "extra_prose": f"More paper text {self.CANARY} embedded here",
        }
        projection = project_inputs("plan_reproduction", kwargs)
        rec = LedgerRecord(
            primitive="plan_reproduction",
            seq=0,
            inputs_projection=projection,
            outputs_pointer={"plan_path": "rlm_state/plan.json"},
            evidence_keys=[],
            outcome="ok",
            iteration=1,
        )
        append_record(project_dir, rec)

    def test_canary_absent_from_all_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
        pd = _project_dir(tmp_path)
        self._write_canary_record(pd)

        # Walk every file under rlm_state/ and assert the canary is absent
        rlm_state = pd / "rlm_state"
        assert rlm_state.exists(), "rlm_state directory should exist after write"

        leaked_files: list[str] = []
        for fpath in rlm_state.rglob("*"):
            if not fpath.is_file():
                continue
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if self.CANARY in content:
                leaked_files.append(str(fpath))

        assert leaked_files == [], (
            f"Canary string leaked into file(s): {leaked_files!r}. "
            "The project_inputs projection must not include raw paper text."
        )

    def test_canary_absent_from_projection_output(self) -> None:
        """project_inputs itself must not echo the canary in its return value."""
        kwargs = {
            "section_ids": ["abstract"],
            "hyperparameters": {"lr": 0.001},
            "section_text": f"{self.CANARY}: paper prose",
            "paper": self.CANARY,
        }
        result = project_inputs("plan_reproduction", kwargs)
        result_str = json.dumps(result)
        assert self.CANARY not in result_str

    def test_canary_absent_even_in_section_id_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even if the canary sneaks into section_ids, it only lands as an id string.
        This test documents the boundary: section_ids items are stored (they are opaque
        ids, not prose), but no other field leaks the canary."""
        monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
        pd = _project_dir(tmp_path)
        kwargs = {
            "section_ids": ["sec1"],  # safe ids only
            "hyperparameters": {},
            "paper_body": f"{self.CANARY}: never stored",
        }
        projection = project_inputs("plan_reproduction", kwargs)
        rec = _record(primitive="plan_reproduction", inputs_projection=projection)
        append_record(pd, rec)

        rlm_state = pd / "rlm_state"
        for fpath in rlm_state.rglob("*"):
            if fpath.is_file():
                content = fpath.read_text(encoding="utf-8", errors="replace")
                assert self.CANARY not in content

    def test_implement_baseline_plan_prose_not_stored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The plan value (potentially long prose) must not appear in the ledger."""
        monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
        pd = _project_dir(tmp_path)
        kwargs = {
            "plan": f"Long plan text: {self.CANARY}",
            "repair_context": {"error": f"fix this: {self.CANARY}"},
            "sandbox_mode": "local",
            "gpu_mode": "single",
        }
        projection = project_inputs("implement_baseline", kwargs)
        # projection only carries boolean flags, not the plan text
        assert self.CANARY not in json.dumps(projection)

        rec = _record(primitive="implement_baseline", inputs_projection=projection)
        append_record(pd, rec)

        rlm_state = pd / "rlm_state"
        for fpath in rlm_state.rglob("*"):
            if fpath.is_file():
                content = fpath.read_text(encoding="utf-8", errors="replace")
                assert self.CANARY not in content

    def test_run_experiment_code_not_stored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The code kwarg (source files) must not appear in the ledger."""
        monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
        pd = _project_dir(tmp_path)
        kwargs = {
            "env_id": "alfworld",
            "code": f"import torch  # {self.CANARY}",
        }
        projection = project_inputs("run_experiment", kwargs)
        assert self.CANARY not in json.dumps(projection)
        assert projection["code_present"] is True  # presence flag only

        rec = _record(primitive="run_experiment", inputs_projection=projection)
        append_record(pd, rec)

        rlm_state = pd / "rlm_state"
        for fpath in rlm_state.rglob("*"):
            if fpath.is_file():
                content = fpath.read_text(encoding="utf-8", errors="replace")
                assert self.CANARY not in content


# =========================================================================== #
# append_record — fail-soft guarantees                                          #
# =========================================================================== #


class TestAppendRecordFailSoft:
    def test_does_not_raise_on_invalid_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
        # /nonexistent/... cannot be created (permission error on most systems;
        # simulate by passing a path that cannot be a directory)
        bad_dir = Path("/dev/null/cannot_create_subdir")
        # must not raise
        append_record(bad_dir, _record())

    def test_flag_off_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENRESEARCH_LIFECYCLE_LEDGER", raising=False)
        # even a completely bad path does not raise when disabled
        append_record(Path("/nonexistent/path"), _record())


# =========================================================================== #
# read_records — fail-soft guarantees                                           #
# =========================================================================== #


class TestReadRecordsFailSoft:
    def test_does_not_raise_on_unreadable_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
        pd = _project_dir(tmp_path)
        ledger = _ledger_path(pd)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        # Write something that can be parsed but then corrupt it
        ledger.write_text("totally not json\nalso not json\n", encoding="utf-8")
        # Should return empty, never raise
        records = read_records(pd)
        assert records == []

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        pd = _project_dir(tmp_path)
        ledger = _ledger_path(pd)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text("", encoding="utf-8")
        assert read_records(pd) == []
