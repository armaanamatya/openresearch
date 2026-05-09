"""
Tests for Issue #9: SQLite persistence and repository layer for core MVP entities.
Run: pytest tests/test_issue9_persistence.py -v
"""
import os
import pytest


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Provide a fresh SQLite database for each test."""
    from backend.persistence.database import Database
    db_path = str(tmp_path / "test.db")
    database = Database(f"sqlite:///{db_path}")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def task_repo(db):
    from backend.persistence.repositories.task_repository import TaskRepository
    return TaskRepository(db)


@pytest.fixture
def message_repo(db):
    from backend.persistence.repositories.message_repository import MessageRepository
    return MessageRepository(db)


@pytest.fixture
def artifact_repo(db):
    from backend.persistence.repositories.artifact_repository import ArtifactRepository
    return ArtifactRepository(db)


@pytest.fixture
def run_repo(db):
    from backend.persistence.repositories.run_repository import RunRepository
    return RunRepository(db)


@pytest.fixture
def verification_repo(db):
    from backend.persistence.repositories.verification_repository import VerificationRepository
    return VerificationRepository(db)


# -- 1. Database bootstrap ---------------------------------------------------

class TestDatabaseBootstrap:
    def test_database_creates_file(self, tmp_path):
        from backend.persistence.database import Database
        db_path = str(tmp_path / "new.db")
        database = Database(f"sqlite:///{db_path}")
        database.initialize()
        assert os.path.isfile(db_path)
        database.close()

    def test_initialize_is_idempotent(self, tmp_path):
        """Calling initialize() twice should not raise or corrupt data."""
        from backend.persistence.database import Database
        db_path = str(tmp_path / "idem.db")
        database = Database(f"sqlite:///{db_path}")
        database.initialize()
        database.initialize()  # second call must not raise
        database.close()

    def test_tables_created(self, db):
        """All core entity tables exist after initialization."""
        tables = db.list_tables()
        required = {"agent_tasks", "agent_messages", "artifacts", "runs", "verifications"}
        assert required.issubset(set(tables)), f"Missing tables: {required - set(tables)}"


# -- 2. Task repository CRUD ------------------------------------------------

class TestTaskRepository:
    def test_create_and_get(self, task_repo):
        from backend.schemas.tasks import AgentTask
        task = AgentTask(
            task_id="t1", agent_type="paper_understanding", status="created",
        )
        task_repo.create(task)
        retrieved = task_repo.get_by_id("t1")
        assert retrieved is not None
        assert retrieved.task_id == "t1"
        assert retrieved.status == "created"

    def test_update_status(self, task_repo):
        from backend.schemas.tasks import AgentTask
        task = AgentTask(task_id="t2", agent_type="runner", status="created")
        task_repo.create(task)
        task_repo.update_status("t2", "running")
        updated = task_repo.get_by_id("t2")
        assert updated.status == "running"

    def test_list_by_status(self, task_repo):
        from backend.schemas.tasks import AgentTask
        task_repo.create(AgentTask(task_id="t3", agent_type="a", status="created"))
        task_repo.create(AgentTask(task_id="t4", agent_type="b", status="running"))
        task_repo.create(AgentTask(task_id="t5", agent_type="c", status="created"))
        created = task_repo.list_by_status("created")
        assert len(created) == 2

    def test_get_nonexistent_returns_none(self, task_repo):
        assert task_repo.get_by_id("nonexistent") is None

    def test_list_by_parent(self, task_repo):
        from backend.schemas.tasks import AgentTask
        task_repo.create(AgentTask(task_id="parent", agent_type="orchestrator", status="running"))
        task_repo.create(AgentTask(task_id="child1", agent_type="runner", status="created", parent_task_id="parent"))
        task_repo.create(AgentTask(task_id="child2", agent_type="runner", status="created", parent_task_id="parent"))
        children = task_repo.list_by_parent("parent")
        assert len(children) == 2


# -- 3. Message repository --------------------------------------------------

class TestMessageRepository:
    def test_create_and_list(self, message_repo):
        from backend.schemas.messages import AgentMessage
        message_repo.create(AgentMessage(
            message_id="m1", agent_id="paper_understanding",
            content="done", structured_outputs={},
        ))
        messages = message_repo.list_by_agent("paper_understanding")
        assert len(messages) == 1


# -- 4. Artifact repository -------------------------------------------------

class TestArtifactRepository:
    def test_create_and_list_by_run(self, artifact_repo):
        from backend.schemas.artifacts import Artifact
        artifact_repo.create(Artifact(
            artifact_id="a1", artifact_type="metrics",
            run_id="run1", file_path="runs/baseline/metrics.json",
        ))
        artifact_repo.create(Artifact(
            artifact_id="a2", artifact_type="logs",
            run_id="run1", file_path="runs/baseline/logs/train.log",
        ))
        arts = artifact_repo.list_by_run("run1")
        assert len(arts) == 2


# -- 5. Run repository ------------------------------------------------------

class TestRunRepository:
    def test_create_and_get(self, run_repo):
        from backend.schemas.runs import Run
        run_repo.create(Run(
            run_id="r1", run_type="baseline", status="created", task_id="t1",
        ))
        run = run_repo.get_by_id("r1")
        assert run.run_type == "baseline"


# -- 6. Verification repository ---------------------------------------------

class TestVerificationRepository:
    def test_create_and_get(self, verification_repo):
        from backend.schemas.verifications import VerificationRecord
        verification_repo.create(VerificationRecord(
            verification_id="v1", run_id="r1",
            verifier_type="method_fidelity", status="verified",
        ))
        record = verification_repo.get_by_id("v1")
        assert record.status == "verified"

    def test_list_by_run(self, verification_repo):
        from backend.schemas.verifications import VerificationRecord
        verification_repo.create(VerificationRecord(
            verification_id="v2", run_id="r1",
            verifier_type="environment_execution", status="verified",
        ))
        verification_repo.create(VerificationRecord(
            verification_id="v3", run_id="r1",
            verifier_type="data_metrics", status="verified_with_caveats",
        ))
        records = verification_repo.list_by_run("r1")
        assert len(records) == 2


# -- 7. Migration idempotency -----------------------------------------------

def test_migration_runs_on_empty_db(tmp_path):
    from backend.persistence.database import Database
    db = Database(f"sqlite:///{tmp_path / 'fresh.db'}")
    db.initialize()
    assert "agent_tasks" in db.list_tables()
    db.close()
