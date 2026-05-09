"""
Tests for Issue #10: Agent task lifecycle service and state transitions.
Run: pytest tests/test_issue10_task_lifecycle.py -v
"""
import pytest


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    from backend.persistence.database import Database
    database = Database(f"sqlite:///{tmp_path / 'test.db'}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def lifecycle_service(db):
    from backend.services.orchestration.task_lifecycle import TaskLifecycleService
    return TaskLifecycleService(db)


@pytest.fixture
def created_task(lifecycle_service):
    """Return a task in 'created' state."""
    return lifecycle_service.create_task(
        agent_type="paper_understanding",
        parent_task_id=None,
    )


# -- 1. Task creation -------------------------------------------------------

class TestTaskCreation:
    def test_create_task_returns_task_with_id(self, lifecycle_service):
        task = lifecycle_service.create_task(agent_type="runner")
        assert task.task_id is not None
        assert task.status == "created"

    def test_create_child_task(self, lifecycle_service, created_task):
        child = lifecycle_service.create_task(
            agent_type="runner",
            parent_task_id=created_task.task_id,
        )
        assert child.parent_task_id == created_task.task_id


# -- 2. Valid state transitions (happy path) ---------------------------------

HAPPY_PATH = [
    ("created", "context_prepared"),
    ("context_prepared", "running"),
    ("running", "artifact_submitted"),
    ("artifact_submitted", "verification_pending"),
    ("verification_pending", "verified"),
]


class TestValidTransitions:
    @pytest.mark.parametrize("from_status, to_status", HAPPY_PATH)
    def test_happy_path_transition(self, lifecycle_service, from_status, to_status):
        task = lifecycle_service.create_task(agent_type="test_agent")
        # Walk the task to from_status
        for step_from, step_to in HAPPY_PATH:
            if step_from == from_status:
                break
            lifecycle_service.transition(task.task_id, step_to)
        # Now do the actual transition under test
        updated = lifecycle_service.transition(task.task_id, to_status)
        assert updated.status == to_status

    def test_running_to_failed(self, lifecycle_service):
        task = lifecycle_service.create_task(agent_type="runner")
        lifecycle_service.transition(task.task_id, "context_prepared")
        lifecycle_service.transition(task.task_id, "running")
        updated = lifecycle_service.transition(
            task.task_id, "failed",
            failure_substatus="failed_training",
        )
        assert updated.status == "failed"
        assert updated.failure_substatus == "failed_training"

    def test_verification_pending_to_blocked(self, lifecycle_service):
        task = lifecycle_service.create_task(agent_type="verifier")
        lifecycle_service.transition(task.task_id, "context_prepared")
        lifecycle_service.transition(task.task_id, "running")
        lifecycle_service.transition(task.task_id, "artifact_submitted")
        lifecycle_service.transition(task.task_id, "verification_pending")
        updated = lifecycle_service.transition(task.task_id, "blocked_requires_human")
        assert updated.status == "blocked_requires_human"


# -- 3. Invalid state transitions -------------------------------------------

INVALID_TRANSITIONS = [
    ("created", "running"),            # must go through context_prepared
    ("created", "verified"),           # cannot skip to end
    ("verified", "running"),           # terminal to active
    ("failed", "running"),             # terminal to active
    ("running", "created"),            # backwards
    ("artifact_submitted", "created"), # backwards
]


class TestInvalidTransitions:
    @pytest.mark.parametrize("from_status, to_status", INVALID_TRANSITIONS)
    def test_invalid_transition_raises(self, lifecycle_service, from_status, to_status):
        from backend.services.orchestration.task_lifecycle import InvalidTransitionError
        task = lifecycle_service.create_task(agent_type="test_agent")
        # Walk task to from_status
        for step_from, step_to in HAPPY_PATH:
            if step_from == from_status:
                break
            lifecycle_service.transition(task.task_id, step_to)
        with pytest.raises(InvalidTransitionError):
            lifecycle_service.transition(task.task_id, to_status)


# -- 4. Failure substatus ---------------------------------------------------

class TestFailureSubstatus:
    def test_failed_with_substatus(self, lifecycle_service):
        """Transitioning to failed records the substatus."""
        task = lifecycle_service.create_task(agent_type="runner")
        lifecycle_service.transition(task.task_id, "context_prepared")
        lifecycle_service.transition(task.task_id, "running")
        updated = lifecycle_service.transition(
            task.task_id, "failed",
            failure_substatus="timeout",
        )
        assert updated.failure_substatus == "timeout"


# -- 5. Event emission ------------------------------------------------------

class TestEventEmission:
    def test_transition_emits_event(self, lifecycle_service):
        """Each transition should produce a structured event."""
        events = []
        lifecycle_service.on_event(lambda e: events.append(e))

        task = lifecycle_service.create_task(agent_type="runner")
        lifecycle_service.transition(task.task_id, "context_prepared")

        assert len(events) >= 1
        last_event = events[-1]
        assert last_event.event in ("task_status_changed", "agent_started", "agent_completed")
        assert last_event.data["task_id"] == task.task_id
        assert last_event.data["new_status"] == "context_prepared"

    def test_creation_emits_event(self, lifecycle_service):
        events = []
        lifecycle_service.on_event(lambda e: events.append(e))
        task = lifecycle_service.create_task(agent_type="runner")
        assert any(e.data.get("task_id") == task.task_id for e in events)


# -- 6. Transition history --------------------------------------------------

class TestTransitionHistory:
    def test_task_records_transition_timestamps(self, lifecycle_service):
        task = lifecycle_service.create_task(agent_type="runner")
        lifecycle_service.transition(task.task_id, "context_prepared")
        lifecycle_service.transition(task.task_id, "running")
        history = lifecycle_service.get_transition_history(task.task_id)
        assert len(history) >= 2
        # Timestamps should be monotonically increasing
        for i in range(1, len(history)):
            assert history[i].timestamp >= history[i - 1].timestamp
