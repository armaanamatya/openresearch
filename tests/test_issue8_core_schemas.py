"""
Tests for Issue #8: Core schemas for tasks, events, artifacts, and verification records.
Run: pytest tests/test_issue8_core_schemas.py -v
"""
import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError


# -- 1. AgentTask schema ----------------------------------------------------

class TestAgentTaskSchema:
    def test_valid_task_creation(self):
        from backend.schemas.tasks import AgentTask
        task = AgentTask(
            task_id="task_001",
            agent_type="paper_understanding",
            status="created",
            parent_task_id=None,
        )
        assert task.status == "created"
        assert task.parent_task_id is None

    def test_task_rejects_invalid_status(self):
        from backend.schemas.tasks import AgentTask
        with pytest.raises(ValidationError):
            AgentTask(
                task_id="task_002",
                agent_type="paper_understanding",
                status="nonexistent_status",
            )

    def test_task_status_enum_has_all_lifecycle_states(self):
        from backend.schemas.tasks import TaskStatus
        required = {
            "created", "context_prepared", "running",
            "artifact_submitted", "verification_pending",
            "verified", "failed", "blocked_requires_human",
        }
        actual = {s.value for s in TaskStatus}
        assert required.issubset(actual), f"Missing: {required - actual}"

    def test_failure_substatus_enum_coverage(self):
        from backend.schemas.tasks import FailureSubstatus
        required_substates = {
            "failed_install", "failed_dependency_resolution",
            "failed_docker_build", "failed_training",
            "timeout", "out_of_memory",
            "blocked_approval", "blocked_license",
        }
        actual = {s.value for s in FailureSubstatus}
        assert required_substates.issubset(actual), f"Missing: {required_substates - actual}"


# -- 2. AgentMessage schema -------------------------------------------------

class TestAgentMessageSchema:
    def test_valid_message(self):
        from backend.schemas.messages import AgentMessage
        msg = AgentMessage(
            message_id="msg_001",
            agent_id="paper_understanding",
            content="Extraction complete",
            structured_outputs={"claims": []},
        )
        assert msg.agent_id == "paper_understanding"

    def test_message_serializes_to_json(self):
        from backend.schemas.messages import AgentMessage
        msg = AgentMessage(
            message_id="msg_002",
            agent_id="environment_detective",
            content="CUDA 11.3 detected",
            structured_outputs={},
        )
        data = json.loads(msg.model_dump_json())
        assert "agent_id" in data
        assert "timestamp" in data or "message_id" in data


# -- 3. Run schema ----------------------------------------------------------

class TestRunSchema:
    def test_baseline_run(self):
        from backend.schemas.runs import Run
        run = Run(
            run_id="baseline_001",
            run_type="baseline",
            status="created",
            task_id="task_001",
        )
        assert run.run_type == "baseline"

    def test_improvement_run(self):
        from backend.schemas.runs import Run
        run = Run(
            run_id="improvement_001",
            run_type="improvement",
            status="created",
            task_id="task_002",
            parent_run_id="baseline_001",
        )
        assert run.parent_run_id == "baseline_001"


# -- 4. Artifact schema -----------------------------------------------------

class TestArtifactSchema:
    def test_valid_artifact(self):
        from backend.schemas.artifacts import Artifact
        art = Artifact(
            artifact_id="art_001",
            artifact_type="metrics",
            run_id="baseline_001",
            file_path="runs/baseline/metrics.json",
        )
        assert art.artifact_type == "metrics"

    def test_artifact_types_include_prd_set(self):
        from backend.schemas.artifacts import ArtifactType
        required = {"metrics", "logs", "plots", "dockerfile", "diff", "report"}
        actual = {t.value for t in ArtifactType}
        assert required.issubset(actual), f"Missing: {required - actual}"


# -- 5. Verification schema -------------------------------------------------

class TestVerificationSchema:
    def test_valid_verification_record(self):
        from backend.schemas.verifications import VerificationRecord
        v = VerificationRecord(
            verification_id="ver_001",
            run_id="baseline_001",
            verifier_type="method_fidelity",
            status="verified_with_caveats",
            method_fidelity_score=86,
            environment_recovery_score=91,
            data_pipeline_confidence=88,
            artifact_completeness_score=96,
        )
        assert v.status == "verified_with_caveats"

    def test_verification_status_enum(self):
        from backend.schemas.verifications import VerificationStatus
        required = {
            "verified", "verified_with_caveats",
            "partial_reproduction", "failed_reproduction",
            "blocked_requires_human", "invalid_claim",
        }
        actual = {s.value for s in VerificationStatus}
        assert required.issubset(actual), f"Missing: {required - actual}"


# -- 6. Event payload schema ------------------------------------------------

class TestEventPayloadSchema:
    def test_event_types_defined(self):
        from backend.schemas.events import EventType
        required = {
            "agent_started", "agent_completed", "agent_failed",
            "agent_reasoning_step", "verification_gate_result",
            "shared_state_updated", "context_enrichment",
        }
        actual = {t.value for t in EventType}
        assert required.issubset(actual), f"Missing: {required - actual}"

    def test_event_payload_roundtrip(self):
        from backend.schemas.events import EventPayload
        evt = EventPayload(
            event="agent_started",
            agent_id="paper_understanding",
            data={"task_id": "task_001"},
        )
        raw = evt.model_dump_json()
        restored = EventPayload.model_validate_json(raw)
        assert restored.event == evt.event
        assert restored.agent_id == evt.agent_id

    def test_reasoning_step_event(self):
        from backend.schemas.events import EventPayload
        evt = EventPayload(
            event="agent_reasoning_step",
            agent_id="environment_detective",
            data={
                "step_type": "rlm_query",
                "query": "What CUDA version?",
                "result": "CUDA 11.3",
                "citations": [{"source_id": "src_042", "trust_level": "secondary"}],
            },
        )
        assert evt.data["step_type"] == "rlm_query"


# -- 7. Cross-schema consistency --------------------------------------------

def test_task_status_matches_event_expectations():
    """Task statuses should map cleanly to event types."""
    from backend.schemas.tasks import TaskStatus
    from backend.schemas.events import EventType
    assert "verified" in {s.value for s in TaskStatus}
    assert "verification_gate_result" in {t.value for t in EventType}
