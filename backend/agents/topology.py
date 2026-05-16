"""Canonical pipeline topology — single source of truth for what the
frontend renders.

The orchestrator's PipelineStage enum + _pipeline_steps list already
encode the stage progression; this module collects the metadata that
the UI needs (node ids, display names, role descriptions, terminal
kinds, gate positions in the stage sequence) into a serialisable
shape.

Keep this file decoupled from React / pixel coordinates. It's pure
domain data. Frontend layout (x/y) is computed in lib/pipeline/layout.ts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


NodeKind = Literal["source", "agent", "improvement_path", "audit", "report"]
Tone = Literal["accent", "hermes", "info", "neutral"]


class PipelineNode(BaseModel):
    """One renderable node in the workflow graph.

    `agent_ids` is the list of backend agent IDs this node represents
    (a single UI node can summarise several backend agents — e.g. the
    `env` node covers both `environment-detective` and `environment-
    verifier`). Telemetry filtering uses this.

    `tour_caption` is optional. When present, the /demo tour overlay
    includes this node as a step with the caption text.
    """
    id: str
    kind: NodeKind
    internal_label: str
    demo_label: str
    step: str
    role: str
    detail: str
    icon: str  # icon key, matches frontend ICONS dict
    tone: Tone
    agent_ids: list[str] = Field(default_factory=list)
    tour_caption: str | None = None


class PipelineEdge(BaseModel):
    source: str
    target: str


class PipelineGate(BaseModel):
    """A verification gate sits between two nodes. The frontend draws
    a chip on the edge midpoint."""
    id: str
    before_node: str
    after_node: str
    label: str


class PipelineStage(BaseModel):
    """One step in the orchestrator's state machine. The frontend uses
    these for the progress bar (stage index / total)."""
    id: str
    order: int


class PipelineTopology(BaseModel):
    nodes: list[PipelineNode]
    edges: list[PipelineEdge]
    gates: list[PipelineGate]
    stages: list[PipelineStage]
    improvement_path_ids: list[str]
    """Node IDs that represent parallel improvement explorations. The
    UI uses this list to render the 'Improvement sub-agents' rollup
    and to read run.payload.pathStates."""


def default_topology() -> PipelineTopology:
    """Return the canonical 12-node / 14-stage / 3-gate pipeline.

    This mirrors what backend/agents/orchestrator.py drives today. A
    future per-paper customisation lives in
    LiveRunState.payload.pipeline (override) — this function is the
    fallback when no override is set.
    """
    nodes = [
        PipelineNode(id="src", kind="source", internal_label="Source", demo_label="Paper",
                     step="Source intake", role="Receives the source artifact",
                     detail="This is the paper or workspace input that starts the run.",
                     icon="doc", tone="neutral", agent_ids=[]),
        PipelineNode(id="read", kind="agent", internal_label="paper-understanding", demo_label="Reader",
                     step="Paper understanding", role="Extracts claims, metrics, and assumptions",
                     detail="Parses the paper and turns benchmarks and assumptions into a runnable plan.",
                     icon="brain", tone="info",
                     agent_ids=["paper-understanding", "artifact-discovery"],
                     tour_caption="Reader extracts the paper's claims and metrics."),
        PipelineNode(id="env", kind="agent", internal_label="environment-detective", demo_label="Forge",
                     step="Environment", role="Rebuilds the runtime environment",
                     detail="Resolves dependencies and creates the isolated execution environment.",
                     icon="beaker", tone="info",
                     agent_ids=["environment-detective", "environment-verifier"],
                     tour_caption="Forge rebuilds the runtime environment from scratch."),
        PipelineNode(id="plan", kind="agent", internal_label="reproduction-planner", demo_label="Architect",
                     step="Reproduction plan", role="Defines the verification contract",
                     detail="Maps paper claims to experiments and checkpoints.",
                     icon="doc", tone="info",
                     agent_ids=["reproduction-planner", "root-orchestrator"]),
        PipelineNode(id="impl", kind="agent", internal_label="baseline-implementation", demo_label="Builder",
                     step="Baseline implementation", role="Builds and runs the baseline",
                     detail="Produces the baseline implementation and records first metrics.",
                     icon="zap", tone="accent",
                     agent_ids=["baseline-implementation", "experiment-runner",
                                "method-fidelity-verifier", "data-metrics-verifier",
                                "artifact-diff-verifier"],
                     tour_caption="Builder runs the baseline implementation."),
        PipelineNode(id="opt", kind="improvement_path", internal_label="optimizer-path", demo_label="Vesta",
                     step="Optimizer path", role="Explores optimizer changes",
                     detail="Tests alternative optimizers and schedules.",
                     icon="spark", tone="info",
                     agent_ids=["improvement-orchestrator", "improvement-path"],
                     tour_caption="Five parallel improvement paths explore the design space."),
        PipelineNode(id="bb", kind="improvement_path", internal_label="backbone-path", demo_label="Athena",
                     step="Backbone path", role="Tests representation swaps",
                     detail="Evaluates backbone changes.",
                     icon="copy", tone="info",
                     agent_ids=["improvement-orchestrator", "improvement-path"]),
        PipelineNode(id="aug", kind="improvement_path", internal_label="augmentation-path", demo_label="Orion",
                     step="Augmentation path", role="Explores robustness changes",
                     detail="Sweeps augmentation strategies.",
                     icon="graph", tone="info",
                     agent_ids=["improvement-orchestrator", "improvement-path"]),
        PipelineNode(id="hor", kind="improvement_path", internal_label="horizon-path", demo_label="Lyra",
                     step="Horizon path", role="Extends planning horizon",
                     detail="Tests longer-horizon variants.",
                     icon="flag", tone="info",
                     agent_ids=["improvement-orchestrator", "improvement-path"]),
        PipelineNode(id="div", kind="improvement_path", internal_label="diffusion-path", demo_label="Pyxis",
                     step="Diffusion path", role="Sweeps diffusion settings",
                     detail="Compares DDIM and related inference-time changes.",
                     icon="compute", tone="info",
                     agent_ids=["improvement-orchestrator", "improvement-path"]),
        PipelineNode(id="audit", kind="audit", internal_label="supervisor-verifier", demo_label="Hermes",
                     step="Result audit", role="Verifies claims against the run",
                     detail="Checks whether claimed results are grounded in the run outputs.",
                     icon="shield", tone="hermes",
                     agent_ids=["supervisor-verifier", "verifier", "hermes"],
                     tour_caption="Hermes audits whether the claimed result is supported."),
        PipelineNode(id="report", kind="report", internal_label="report-generator", demo_label="Scribe",
                     step="Final report", role="Packages the reproducibility output",
                     detail="Compiles manifests, logs, checkpoints, and the audit trail.",
                     icon="flag", tone="neutral",
                     agent_ids=["supervisor-verifier", "root-orchestrator"],
                     tour_caption="Scribe packages the final reproducibility report."),
    ]

    edges = [
        PipelineEdge(source="src", target="read"),
        PipelineEdge(source="read", target="env"),
        PipelineEdge(source="read", target="plan"),
        PipelineEdge(source="env", target="impl"),
        PipelineEdge(source="plan", target="impl"),
        PipelineEdge(source="impl", target="opt"),
        PipelineEdge(source="impl", target="bb"),
        PipelineEdge(source="impl", target="aug"),
        PipelineEdge(source="impl", target="hor"),
        PipelineEdge(source="impl", target="div"),
        PipelineEdge(source="opt", target="audit"),
        PipelineEdge(source="bb", target="audit"),
        PipelineEdge(source="aug", target="audit"),
        PipelineEdge(source="hor", target="audit"),
        PipelineEdge(source="div", target="audit"),
        PipelineEdge(source="audit", target="report"),
    ]

    gates = [
        PipelineGate(id="gate_1", before_node="plan", after_node="impl", label="Gate 1"),
        PipelineGate(id="gate_2", before_node="impl", after_node="bb", label="Gate 2"),
        PipelineGate(id="gate_3", before_node="bb", after_node="audit", label="Gate 3"),
    ]

    stages = [
        PipelineStage(id="ingested", order=0),
        PipelineStage(id="paper_understood", order=1),
        PipelineStage(id="artifacts_discovered", order=2),
        PipelineStage(id="environment_built", order=3),
        PipelineStage(id="plan_created", order=4),
        PipelineStage(id="gate_1_passed", order=5),
        PipelineStage(id="baseline_implemented", order=6),
        PipelineStage(id="baseline_run", order=7),
        PipelineStage(id="gate_2_passed", order=8),
        PipelineStage(id="improvements_selected", order=9),
        PipelineStage(id="improvements_run", order=10),
        PipelineStage(id="gate_3_passed", order=11),
        PipelineStage(id="research_map_generated", order=12),
        PipelineStage(id="complete", order=13),
    ]

    return PipelineTopology(
        nodes=nodes,
        edges=edges,
        gates=gates,
        stages=stages,
        improvement_path_ids=["opt", "bb", "aug", "hor", "div"],
    )
