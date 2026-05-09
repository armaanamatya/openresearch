"""
Tests for Issue #11: Blackboard records, delegation tree tracking, and spawn-policy guards.
Run: pytest tests/test_issue11_blackboard.py -v
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
def blackboard_service(db):
    from backend.services.orchestration.blackboard import BlackboardService
    return BlackboardService(db)


@pytest.fixture
def delegation_service(db):
    from backend.services.orchestration.delegation import DelegationService
    return DelegationService(db)


@pytest.fixture
def spawn_policy(db):
    from backend.services.orchestration.spawn_policy import SpawnPolicy
    return SpawnPolicy(db, max_depth=3, max_fanout=5)


# -- 1. Blackboard record storage -------------------------------------------

class TestBlackboardRecords:
    def test_write_and_read_record(self, blackboard_service):
        blackboard_service.write(
            key="cuda_version",
            value="11.3",
            scope="branch_shared",
            owner_task_id="task_env_detective",
        )
        record = blackboard_service.read("cuda_version")
        assert record.value == "11.3"
        assert record.scope == "branch_shared"

    def test_overwrite_record(self, blackboard_service):
        blackboard_service.write(key="lr", value="0.001", scope="private_to_parent", owner_task_id="t1")
        blackboard_service.write(key="lr", value="0.0001", scope="private_to_parent", owner_task_id="t1")
        record = blackboard_service.read("lr")
        assert record.value == "0.0001"

    def test_read_nonexistent_returns_none(self, blackboard_service):
        assert blackboard_service.read("nonexistent") is None


# -- 2. Scoped visibility ---------------------------------------------------

class TestScopedVisibility:
    def test_private_to_parent_not_visible_to_sibling(self, blackboard_service):
        blackboard_service.write(
            key="secret", value="hidden", scope="private_to_parent",
            owner_task_id="child_a",
        )
        # Sibling should not see it
        visible = blackboard_service.read_visible(
            key="secret", reader_task_id="child_b", reader_lineage=["parent"]
        )
        assert visible is None

    def test_branch_shared_visible_to_descendants(self, blackboard_service):
        blackboard_service.write(
            key="dataset_path", value="/data/cifar10", scope="branch_shared",
            owner_task_id="parent",
        )
        visible = blackboard_service.read_visible(
            key="dataset_path", reader_task_id="grandchild",
            reader_lineage=["parent", "child"],
        )
        assert visible is not None
        assert visible.value == "/data/cifar10"

    def test_global_verified_visible_to_all(self, blackboard_service):
        blackboard_service.write(
            key="paper_claim_map", value='{"claims": []}', scope="global_verified",
            owner_task_id="paper_agent",
        )
        visible = blackboard_service.read_visible(
            key="paper_claim_map", reader_task_id="unrelated_agent",
            reader_lineage=[],
        )
        assert visible is not None


# -- 3. Delegation tree tracking ---------------------------------------------

class TestDelegationTree:
    def test_register_parent_child(self, delegation_service):
        delegation_service.register("parent_task", child_id="child_task_1")
        delegation_service.register("parent_task", child_id="child_task_2")
        children = delegation_service.get_children("parent_task")
        assert len(children) == 2

    def test_get_lineage(self, delegation_service):
        delegation_service.register("root", child_id="level1")
        delegation_service.register("level1", child_id="level2")
        delegation_service.register("level2", child_id="level3")
        lineage = delegation_service.get_lineage("level3")
        assert lineage == ["root", "level1", "level2", "level3"]

    def test_get_depth(self, delegation_service):
        delegation_service.register("root", child_id="l1")
        delegation_service.register("l1", child_id="l2")
        delegation_service.register("l2", child_id="l3")
        assert delegation_service.get_depth("l3") == 3
        assert delegation_service.get_depth("root") == 0

    def test_root_has_no_parent(self, delegation_service):
        delegation_service.register("root", child_id="child")
        assert delegation_service.get_parent("root") is None

    def test_lineage_queryable_for_dashboard(self, delegation_service):
        """The full tree should be serializable for the frontend dashboard."""
        delegation_service.register("root", child_id="a")
        delegation_service.register("root", child_id="b")
        delegation_service.register("a", child_id="a1")
        tree = delegation_service.get_tree("root")
        assert tree["task_id"] == "root"
        assert len(tree["children"]) == 2


# -- 4. Spawn-policy guards -------------------------------------------------

class TestSpawnPolicy:
    def test_allows_spawn_within_limits(self, spawn_policy, delegation_service):
        delegation_service.register("root", child_id="c1")
        allowed = spawn_policy.check_can_spawn(
            parent_task_id="root",
            delegation_service=delegation_service,
        )
        assert allowed is True

    def test_blocks_spawn_at_max_depth(self, spawn_policy, delegation_service):
        delegation_service.register("root", child_id="l1")
        delegation_service.register("l1", child_id="l2")
        delegation_service.register("l2", child_id="l3")
        # l3 is at depth 3, which equals max_depth
        allowed = spawn_policy.check_can_spawn(
            parent_task_id="l3",
            delegation_service=delegation_service,
        )
        assert allowed is False

    def test_blocks_spawn_at_max_fanout(self, spawn_policy, delegation_service):
        for i in range(5):
            delegation_service.register("root", child_id=f"child_{i}")
        # root already has 5 children (max_fanout=5)
        allowed = spawn_policy.check_can_spawn(
            parent_task_id="root",
            delegation_service=delegation_service,
        )
        assert allowed is False

    def test_spawn_policy_returns_reason_on_block(self, spawn_policy, delegation_service):
        delegation_service.register("root", child_id="l1")
        delegation_service.register("l1", child_id="l2")
        delegation_service.register("l2", child_id="l3")
        result = spawn_policy.check_can_spawn_detailed(
            parent_task_id="l3",
            delegation_service=delegation_service,
        )
        assert result.allowed is False
        assert "depth" in result.reason.lower()


# -- 5. Escalation conditions -----------------------------------------------

class TestEscalation:
    def test_escalation_triggered_on_repeated_failure(self, spawn_policy):
        """After N child failures, escalation should be flagged."""
        result = spawn_policy.check_escalation(
            parent_task_id="root",
            child_failure_count=3,
        )
        assert result.should_escalate is True

    def test_no_escalation_on_first_failure(self, spawn_policy):
        result = spawn_policy.check_escalation(
            parent_task_id="root",
            child_failure_count=1,
        )
        assert result.should_escalate is False
