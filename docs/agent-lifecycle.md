# Agent Lifecycle

This is the operational shape of a ReproLab run: which agents execute, what
state they move through, and where Phase 2 services attach.

## Pipeline Stages

1. `created` - project workspace exists, paper input is registered, policy is loaded.
2. `paper_ingested` - paper text, metadata, and initial artifacts are available.
3. `paper_understood` - Paper Understanding Agent produces the claim map, dataset and metric requirements, and ambiguity list.
4. `artifacts_discovered` - Artifact Discovery Agent records repos, datasets, dependency clues, source confidence, and contradiction candidates.
5. `environment_ready` - Environment Detective Agent creates the Dockerfile, environment lock, compatibility notes, and environment assumptions.
6. `plan_ready` - Reproduction Planner creates the reproduction contract, smoke test, run plan, evaluation plan, and verification checklist.
7. `gate_1_passed` - verifier team checks the plan before implementation.
8. `baseline_implemented` - Baseline Implementation Agent adapts an existing repo or generates code from the paper contract.
9. `baseline_run` - Experiment Runner writes logs, metrics, plots, command history, assumptions, and provenance.
10. `gate_2_passed` - verifier team audits the baseline package.
11. `improvements_planned` - Improvement Orchestrator selects three evidence-backed hypotheses.
12. `improvements_run` - Improvement Path Agents execute in isolated branches or worktrees.
13. `gate_3_passed` - verifier team audits each improvement path.
14. `final_report_ready` - Supervisor Verification Agent publishes the final status, caveats, and research map.

Pipeline stages are resumable checkpoints. A restart should continue from the
last completed stage rather than repeating earlier work.

## Agent Task States

Each concrete agent task has a narrower lifecycle:

1. `created` - task record is created with agent id, parent task id, budget, timeout, read scope, and write scope.
2. `context_prepared` - orchestrator loads scoped Context REPL variables, graph tools, semantic index handles, citations, and artifact pointers.
3. `running` - agent is active and may emit reasoning, tool-use, and progress events.
4. `artifact_submitted` - agent returns structured output and writes owned artifacts.
5. `verification_pending` - output is waiting for verifier or supervisor review.
6. `verified` - supervisor accepted the output as usable for downstream stages.
7. `failed` - task ended with a classified failure.
8. `blocked_requires_human` - task cannot proceed without user approval.

Agents never mutate global ledgers directly. They submit structured records to
the orchestrator, which serializes writes to the blackboard, assumption ledger,
experiment ledger, decision log, and provenance records.

## Runtime Statuses

The dashboard and event stream use these live statuses:

- `idle` - registered and available, not currently assigned.
- `queued` - task exists but is waiting on dependencies, policy, or capacity.
- `running` - actively using tools or reasoning.
- `waiting` - paused for another agent, verifier result, sandbox operation, or approval.
- `completed` - task finished and returned structured output.
- `failed` - task ended with a non-retryable error or exhausted retries.
- `cancelled` - orchestrator or user stopped the task.

Runtime failures are classified with substatus values such as
`failed_install`, `failed_dependency_resolution`, `failed_data_download`,
`failed_dataset_validation`, `failed_docker_build`, `failed_smoke_test`,
`failed_training`, `failed_evaluation`, `failed_metric_validation`,
`failed_plot_generation`, `timeout`, `out_of_memory`, `out_of_disk`,
`blocked_approval`, `blocked_license`, `blocked_credentials`,
`blocked_unavailable_dataset`, and `inconclusive_budget`.

## Verification Decisions

Verifier agents produce advisory reports. The Supervisor Verification Agent is
the final authority and records one binding decision:

- `verified`
- `verified_with_caveats`
- `partial_reproduction`
- `failed_reproduction`
- `blocked_requires_human`
- `invalid_claim`

The four verifier scopes are method fidelity, environment and execution, data
and metrics, and artifact and diff. Supervisor disagreements are resolved by
reading the verifier evidence, not by vote counting.

## Phase 2 Attach Points

- Knowledge graph service builds AST-backed nodes and edges for source files.
- `graph_query()` exposes structural navigation inside the Context REPL.
- Cross-project memory stores reusable environment, dataset, failure, and result lessons.
- Git worktree manager isolates competing improvement branches when the baseline is a Git repo.
- Multi-paper comparison service summarizes shared datasets, metrics, rankings, and research map items.
- Dataset cache, approval, diagnostics, and scoring services add persistent workspace state around the agent pipeline.

These services are deterministic support layers. They do not replace the agent
pipeline; they make agent decisions more inspectable, reusable, and safer.
