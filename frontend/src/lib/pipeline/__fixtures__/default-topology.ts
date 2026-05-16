import type { PipelineTopology } from "../topology";

/**
 * Frontend mirror of `backend/agents/topology.py::default_topology()`.
 *
 * Used by component tests that render the workflow view without a live
 * backend. The actual production topology is fetched from
 * GET /pipeline/topology; this fixture exists so unit tests don't have
 * to stub fetch for an SSR'd dependency.
 */
export const defaultTopologyFixture: PipelineTopology = {
  nodes: [
    {
      id: "src",
      kind: "source",
      internal_label: "Source",
      demo_label: "Paper",
      step: "Source intake",
      role: "Receives the source artifact",
      detail: "This is the paper or workspace input that starts the run.",
      icon: "doc",
      tone: "neutral",
      agent_ids: []
    },
    {
      id: "read",
      kind: "agent",
      internal_label: "paper-understanding",
      demo_label: "Reader",
      step: "Paper understanding",
      role: "Extracts claims, metrics, and assumptions",
      detail: "Parses the paper and turns benchmarks and assumptions into a runnable plan.",
      icon: "brain",
      tone: "info",
      agent_ids: ["paper-understanding", "artifact-discovery"]
    },
    {
      id: "env",
      kind: "agent",
      internal_label: "environment-detective",
      demo_label: "Forge",
      step: "Environment",
      role: "Rebuilds the runtime environment",
      detail: "Resolves dependencies and creates the isolated execution environment.",
      icon: "beaker",
      tone: "info",
      agent_ids: ["environment-detective", "environment-verifier"]
    },
    {
      id: "plan",
      kind: "agent",
      internal_label: "reproduction-planner",
      demo_label: "Architect",
      step: "Reproduction plan",
      role: "Defines the verification contract",
      detail: "Maps paper claims to experiments and checkpoints.",
      icon: "doc",
      tone: "info",
      agent_ids: ["reproduction-planner", "root-orchestrator"]
    },
    {
      id: "impl",
      kind: "agent",
      internal_label: "baseline-implementation",
      demo_label: "Builder",
      step: "Baseline implementation",
      role: "Builds and runs the baseline",
      detail: "Produces the baseline implementation and records first metrics.",
      icon: "zap",
      tone: "accent",
      agent_ids: [
        "baseline-implementation",
        "experiment-runner",
        "method-fidelity-verifier",
        "data-metrics-verifier",
        "artifact-diff-verifier"
      ]
    },
    {
      id: "opt",
      kind: "improvement_path",
      internal_label: "optimizer-path",
      demo_label: "Vesta",
      step: "Optimizer path",
      role: "Explores optimizer changes",
      detail: "Tests alternative optimizers and schedules.",
      icon: "spark",
      tone: "info",
      agent_ids: ["improvement-orchestrator", "improvement-path"]
    },
    {
      id: "bb",
      kind: "improvement_path",
      internal_label: "backbone-path",
      demo_label: "Athena",
      step: "Backbone path",
      role: "Tests representation swaps",
      detail: "Evaluates backbone changes.",
      icon: "copy",
      tone: "info",
      agent_ids: ["improvement-orchestrator", "improvement-path"]
    },
    {
      id: "aug",
      kind: "improvement_path",
      internal_label: "augmentation-path",
      demo_label: "Orion",
      step: "Augmentation path",
      role: "Explores robustness changes",
      detail: "Sweeps augmentation strategies.",
      icon: "graph",
      tone: "info",
      agent_ids: ["improvement-orchestrator", "improvement-path"]
    },
    {
      id: "hor",
      kind: "improvement_path",
      internal_label: "horizon-path",
      demo_label: "Lyra",
      step: "Horizon path",
      role: "Extends planning horizon",
      detail: "Tests longer-horizon variants.",
      icon: "flag",
      tone: "info",
      agent_ids: ["improvement-orchestrator", "improvement-path"]
    },
    {
      id: "div",
      kind: "improvement_path",
      internal_label: "diffusion-path",
      demo_label: "Pyxis",
      step: "Diffusion path",
      role: "Sweeps diffusion settings",
      detail: "Compares DDIM and related inference-time changes.",
      icon: "compute",
      tone: "info",
      agent_ids: ["improvement-orchestrator", "improvement-path"]
    },
    {
      id: "audit",
      kind: "audit",
      internal_label: "supervisor-verifier",
      demo_label: "Hermes",
      step: "Result audit",
      role: "Verifies claims against the run",
      detail: "Checks whether claimed results are grounded in the run outputs.",
      icon: "shield",
      tone: "hermes",
      agent_ids: ["supervisor-verifier", "verifier", "hermes"]
    },
    {
      id: "report",
      kind: "report",
      internal_label: "report-generator",
      demo_label: "Scribe",
      step: "Final report",
      role: "Packages the reproducibility output",
      detail: "Compiles manifests, logs, checkpoints, and the audit trail.",
      icon: "flag",
      tone: "neutral",
      agent_ids: ["supervisor-verifier", "root-orchestrator"]
    }
  ],
  edges: [
    { source: "src", target: "read" },
    { source: "read", target: "env" },
    { source: "read", target: "plan" },
    { source: "env", target: "impl" },
    { source: "plan", target: "impl" },
    { source: "impl", target: "opt" },
    { source: "impl", target: "bb" },
    { source: "impl", target: "aug" },
    { source: "impl", target: "hor" },
    { source: "impl", target: "div" },
    { source: "opt", target: "audit" },
    { source: "bb", target: "audit" },
    { source: "aug", target: "audit" },
    { source: "hor", target: "audit" },
    { source: "div", target: "audit" },
    { source: "audit", target: "report" }
  ],
  gates: [
    { id: "gate_1", before_node: "plan", after_node: "impl", label: "Gate 1" },
    { id: "gate_2", before_node: "impl", after_node: "bb", label: "Gate 2" },
    { id: "gate_3", before_node: "bb", after_node: "audit", label: "Gate 3" }
  ],
  stages: [
    { id: "ingested", order: 0 },
    { id: "paper_understood", order: 1 },
    { id: "artifacts_discovered", order: 2 },
    { id: "environment_built", order: 3 },
    { id: "plan_created", order: 4 },
    { id: "gate_1_passed", order: 5 },
    { id: "baseline_implemented", order: 6 },
    { id: "baseline_run", order: 7 },
    { id: "gate_2_passed", order: 8 },
    { id: "improvements_selected", order: 9 },
    { id: "improvements_run", order: 10 },
    { id: "gate_3_passed", order: 11 },
    { id: "research_map_generated", order: 12 },
    { id: "complete", order: 13 }
  ],
  improvement_path_ids: ["opt", "bb", "aug", "hor", "div"]
};
