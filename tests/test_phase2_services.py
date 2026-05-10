"""Phase 2 foundations: graph, memory, comparisons, and worktrees."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from backend.persistence.database import Database


@pytest.fixture
def db(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'phase2.db'}")
    database.initialize()
    yield database
    database.close()


def test_python_ast_graph_ingests_and_queries_calls_and_imports(db, tmp_path: Path):
    from backend.services.context.graph import KnowledgeGraphService

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text(
        """
import torch

def train():
    return 1

def main():
    train()
    torch.manual_seed(7)
""".strip()
    )

    graph = KnowledgeGraphService(db)
    node_count, edge_count = graph.ingest_python_repo(project_id="prj_graph", repo_root=repo)

    assert node_count >= 4
    assert edge_count >= 3

    callers = graph.query("function", project_id="prj_graph", calls="train").nodes
    assert [node.name for node in callers] == ["main"]

    torch_modules = graph.query("module", project_id="prj_graph", imports="torch").nodes
    assert [node.path for node in torch_modules] == ["train.py"]


def test_graph_query_tool_returns_cited_results(db, tmp_path: Path):
    from backend.services.context.graph import KnowledgeGraphService
    from backend.services.context.workspace.tools.graph_query import GraphQueryTool

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "model.py").write_text(
        """
def train():
    return 1

def evaluate():
    train()
""".strip()
    )

    graph = KnowledgeGraphService(db)
    graph.ingest_python_repo(project_id="prj_tool", repo_root=repo)

    result = GraphQueryTool(graph).call(
        workspace_id="ws_1",
        project_id="prj_tool",
        entity_type="function",
        calls="train",
    )

    assert result.value["results"][0]["name"] == "evaluate"
    assert result.citations[0].source_id == "knowledge_graph:prj_tool"


def test_cross_project_memory_records_and_searches(db):
    from backend.services.context.memory import CrossProjectMemoryService

    memory = CrossProjectMemoryService(db)
    record = memory.remember_environment_recipe(
        source_project_id="ppo_cartpole",
        paper_id="ppo",
        title="PPO CartPole CPU environment",
        summary="Python 3.11 with torch 2.2 and gymnasium 0.29 works for CartPole.",
        packages={"torch": "2.2.0", "gymnasium": "0.29.1"},
        evidence_refs=("runs/ppo/baseline/provenance.json",),
        confidence=0.9,
    )

    loaded = memory.get(record.id)
    assert loaded is not None
    assert loaded.metrics["packages"]["torch"] == "2.2.0"

    hits = memory.search("torch gymnasium cartpole", kind="environment_recipe")
    assert hits
    assert hits[0].record.source_project_id == "ppo_cartpole"
    assert "torch" in hits[0].matched_terms


def test_multi_paper_comparison_groups_verified_comparable_runs(db):
    from backend.services.comparison import MultiPaperComparisonService, PaperRunSummary

    service = MultiPaperComparisonService(db)
    report = service.compare(
        [
            PaperRunSummary(
                project_id="ppo_a",
                paper_title="PPO A",
                method_name="PPO",
                dataset="CartPole-v1",
                split="eval",
                metric_name="mean_reward",
                metric_value=475.0,
                status="verified",
                assumptions=("A001",),
            ),
            PaperRunSummary(
                project_id="ppo_b",
                paper_title="PPO B",
                method_name="PPO tuned",
                dataset="CartPole-v1",
                split="eval",
                metric_name="mean_reward",
                metric_value=492.0,
                status="verified_with_caveats",
                assumptions=("A001", "A002"),
            ),
            PaperRunSummary(
                project_id="mixmatch_reduced",
                paper_title="MixMatch",
                dataset="CIFAR-10",
                split="test",
                metric_name="accuracy",
                metric_value=0.72,
                status="verified",
                reduced_run=True,
            ),
        ]
    )

    assert report.status == "partial"
    assert report.groups[0].best_project_id == "ppo_b"
    assert report.shared_assumptions == ("A001",)
    assert report.incomparable_runs[0].reason == "reduced run must be compared separately"

    persisted = service.get(report.comparison_id)
    assert persisted is not None
    assert persisted.comparison_id == report.comparison_id


def test_git_worktree_manager_creates_isolated_improvement_branch(tmp_path: Path):
    if shutil.which("git") is None:
        pytest.skip("git is not available")

    from backend.services.worktrees import GitWorktreeManager

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("root\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")

    manager = GitWorktreeManager(worktrees_root=tmp_path / "worktrees")
    spec = manager.spec_for(
        project_id="prj_1",
        path_id="path_001",
        slug="entropy sweep",
    )
    info = manager.create(repo_root=repo, spec=spec)

    assert info.branch == "improvement/path_001-entropy-sweep"
    assert (spec.worktree_path / "README.md").read_text() == "root\n"
    assert spec.worktree_path.resolve() != repo.resolve()

    worktrees = manager.list(repo_root=repo)
    assert any(item.path.resolve() == spec.worktree_path.resolve() for item in worktrees)

    manager.remove(repo_root=repo, path=spec.worktree_path, force=True)
    assert not spec.worktree_path.exists()


def _git(repo: Path, *args: str) -> None:
    result = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
