import { describe, expect, it } from "vitest";

import { buildLiveDemoDashboard, pathStateMap } from "./pipeline-dashboard";

describe("buildLiveDemoDashboard", () => {
  it("converts a completed pipeline state into replayable dashboard data", () => {
    const data = buildLiveDemoDashboard(
      {
        project_id: "ui_demo_123",
        stage: "complete",
        paper_claim_map: {
          core_contribution: "Proximal Policy Optimization for CartPole-v1",
          datasets: [{ name: "CartPole-v1" }],
          metrics: [{ name: "mean_reward", definition: "Mean reward over 100 episodes" }],
          ambiguities: [
            {
              assumption_id: "A001",
              detail: "Adam epsilon is not specified",
              evidence: ["Section 4.1"],
              risk: "medium"
            }
          ]
        },
        environment_spec: {
          python_version: "3.11",
          framework: "torch",
          framework_version: "2.5.1",
          assumptions: [
            {
              assumption_id: "A009",
              detail: "CPU-only runtime is acceptable for the demo",
              chosen_value: "cpu",
              risk: "low"
            }
          ]
        },
        baseline_result: {
          mode: "implement_from_paper",
          assumptions_applied: ["A001", "A009"]
        },
        experiment_artifacts: {
          success: true,
          metrics: {
            mean_reward: 487,
            improvement: 0
          },
          plots: ["reward_curve.png"],
          log_path: "runs/ui_demo_123/baseline/run.log",
          commands_log_path: "runs/ui_demo_123/baseline/commands.log",
          provenance_path: "runs/ui_demo_123/baseline/provenance.json"
        },
        gate_1: { passed: true, status: "verified" },
        gate_2: { passed: true, status: "verified_with_caveats" },
        gate_3: { passed: true, status: "verified" },
        path_results: [
          {
            path_id: "path_1",
            hypothesis: "Increase learning rate slightly",
            success: true,
            metrics: { mean_reward: 501, improvement: 14 }
          },
          {
            path_id: "path_2",
            hypothesis: "Anneal entropy coefficient",
            success: false,
            metrics: {},
            failure_notes: "Reward collapsed after early gains"
          }
        ],
        research_map: {
          promising_directions: ["path_1 improved reward to 501"],
          dead_ends: ["path_2 regressed after reward collapse"],
          next_experiments: ["Combine path_1 with longer training"]
        },
        assumption_ledger: [{ assumption_id: "A001" }, { assumption_id: "A009" }],
        decision_log: [
          "gate_1: verified",
          "gate_2: verified_with_caveats",
          "gate_3: verified"
        ]
      },
      {
        projectId: "ui_demo_123",
        outputDir: "runs/ui_demo_123",
        sourceKind: "workspace_fixture",
        runMode: "sdk",
        sourceLabel: "In-repo PPO workspace fixture",
        sourceNote: "The repo does not currently contain a checked-in PDF."
      }
    );

    expect(data.initialSnapshot.agents[0]?.id).toBe("root-orchestrator");
    expect(data.initialSnapshot.progress).toEqual([
      {
        stage: "plan",
        status: "passed",
        detail: "Gate 1 passed. The plan is ready for baseline work."
      },
      {
        stage: "baseline",
        status: "caveat",
        detail: "Gate 2 passed. The baseline unlocked improvement work."
      },
      {
        stage: "improvement",
        status: "passed",
        detail: "Gate 3 passed. The research map is ready."
      }
    ]);

    expect(
      data.events.some(
        (event) =>
          event.event === "verification_gate_result" &&
          event.stage === "baseline" &&
          event.status === "caveat"
      )
    ).toBe(true);

    expect(
      data.events.some(
        (event) =>
          event.event === "agent_completed" && event.agent.id === "path-path_1"
      )
    ).toBe(true);

    expect(data.summary.meanReward).toBe(487);
    expect(data.summary.improvementCount).toBe(2);
    expect(data.summary.sourceLabel).toBe("In-repo PPO workspace fixture");
    expect(data.runMode).toBe("sdk");
    expect(data.summary.runModeLabel).toBe("SDK");
    expect(data.initialSnapshot.hermesPanel?.title).toBe("Hermes Verification");
    expect(data.initialSnapshot.conceptCard?.title).toContain("Proximal Policy Optimization");
  });

  it("marks unfinished stages as pending for in-flight runs", () => {
    const data = buildLiveDemoDashboard(
      {
        project_id: "ui_sdk_demo_456",
        stage: "environment_built",
        paper_claim_map: {
          core_contribution: "Partial run",
          datasets: [{ name: "CartPole-v1" }],
          metrics: [{ name: "mean_reward" }],
          ambiguities: []
        },
        environment_spec: {
          python_version: "3.11",
          framework: "torch",
          framework_version: "2.5.1",
          assumptions: []
        }
      },
      {
        projectId: "ui_sdk_demo_456",
        outputDir: "runs/ui_sdk_demo_456",
        sourceKind: "workspace_fixture",
        runMode: "sdk",
        sourceLabel: "In-repo PPO workspace fixture",
        sourceNote: "Fixture"
      }
    );

    expect(data.initialSnapshot.progress).toEqual([
      { stage: "plan", status: "pending", detail: "Waiting for Gate 1 verification." },
      { stage: "baseline", status: "pending", detail: "Waiting for Gate 2 verification." },
      { stage: "improvement", status: "pending", detail: "Waiting for Gate 3 verification." }
    ]);
  });

  it("labels SDK provider when metadata includes one", () => {
    const data = buildLiveDemoDashboard(
      {
        project_id: "ui_sdk_openai_demo_456",
        stage: "ingested"
      },
      {
        projectId: "ui_sdk_openai_demo_456",
        outputDir: "runs/ui_sdk_openai_demo_456",
        sourceKind: "workspace_fixture",
        runMode: "sdk",
        llmProvider: "openai",
        sourceLabel: "In-repo PPO workspace fixture",
        sourceNote: "Fixture"
      }
    );

    expect(data.summary.runModeLabel).toBe("SDK: OpenAI");
    expect(data.summary.llmProvider).toBe("openai");
  });

  it("prefers real backend Hermes audit reports when available", () => {
    const data = buildLiveDemoDashboard(
      {
        project_id: "ui_sdk_demo_789",
        stage: "complete",
        paper_claim_map: {
          core_contribution: "Trust-region style PPO variant"
        },
        hermes_step_reports: {
          "paper-understanding": [
            {
              target: "paper-understanding",
              scope: "step",
              status: "grounded",
              summary: "Paper concept is grounded in the extracted claim map.",
              evidence_refs: [{ path: "runs/ui_sdk_demo_789/paper_claim_map.json" }]
            }
          ],
          "baseline-implementation": [
            {
              target: "baseline-implementation",
              scope: "step",
              status: "caveat",
              summary: "Implementation mostly matches the concept but one assumption stayed implicit.",
              recommended_intervention: "request_evidence"
            }
          ]
        },
        hermes_checkpoint_reports: {
          gate_2: [
            {
              target: "gate_2",
              scope: "checkpoint",
              status: "unsupported",
              summary: "One reported baseline claim is not backed by artifact evidence.",
              unsupported_claims: ["Baseline summary overstates the validated reward delta."],
              recommended_intervention: "downgrade_claim",
              evidence_refs: [{ path: "runs/ui_sdk_demo_789/baseline/metrics.json" }]
            }
          ]
        },
        hermes_interventions: [
          {
            target: "gate_2",
            scope: "checkpoint",
            action: "downgrade_claim",
            reason: "Unsupported baseline summary",
            status: "unsupported"
          }
        ]
      },
      {
        projectId: "ui_sdk_demo_789",
        outputDir: "runs/ui_sdk_demo_789",
        sourceKind: "workspace_fixture",
        runMode: "sdk",
        sourceLabel: "In-repo PPO workspace fixture",
        sourceNote: "Fixture"
      }
    );

    expect(data.initialSnapshot.hermesPanel?.overallStatus).toBe("unsupported");
    expect(data.initialSnapshot.hermesPanel?.summary).toContain("1 intervention");
    expect(data.initialSnapshot.hermesPanel?.checks[0]?.detail).toContain("grounded");
    expect(data.initialSnapshot.hermesPanel?.checks[3]?.detail).toContain(
      "not backed by artifact evidence"
    );
  });

  it("surfaces uploaded-paper metadata in the source summary", () => {
    const data = buildLiveDemoDashboard(
      {
        project_id: "prj_upload",
        stage: "complete",
        paper_claim_map: {
          core_contribution: "Uploaded PPO paper"
        }
      },
      {
        projectId: "prj_upload",
        outputDir: "runs/prj_upload",
        sourceKind: "uploaded_pdf",
        runMode: "offline",
        sourceLabel: "ppo-paper.pdf",
        sourceNote: "Uploaded from the lab page."
      }
    );

    expect(data.summary.sourceLabel).toBe("ppo-paper.pdf");
    expect(data.initialSnapshot.dataPanels[1]?.items).toContain("Uploaded from the lab page.");
  });
});

describe("pathStateMap", () => {
  it("routes a path to its keyword bucket via hypothesis text", () => {
    const map = pathStateMap(
      [{ path_id: "p1", hypothesis: "swap optimizer for AdamW", success: true }],
      "improvements_run"
    );
    expect(map.opt).toBe("done");
    expect(map.bb).toBe("upcoming");
  });

  it("marks a failed path as attention", () => {
    const map = pathStateMap(
      [{ path_id: "p1", hypothesis: "diffusion sampler swap", success: false }],
      "improvements_run"
    );
    expect(map.div).toBe("attention");
  });

  it("falls back to round-robin when no keywords match", () => {
    const map = pathStateMap(
      [{ path_id: "p1", hypothesis: "novel hand-crafted reward shaping" }],
      "improvements_run"
    );
    // first non-matching slot in display order: opt
    expect(map.opt).toBe("running");
  });

  it("marks unmatched nodes as skipped after the round finishes", () => {
    const map = pathStateMap(
      [{ path_id: "p1", hypothesis: "backbone resnet swap", success: true }],
      "complete"
    );
    expect(map.bb).toBe("done");
    expect(map.opt).toBe("skipped");
    expect(map.aug).toBe("skipped");
  });

  it("returns running while a path has no success field yet", () => {
    const map = pathStateMap(
      [{ path_id: "p1", hypothesis: "augmentation dropout sweep" }],
      "improvements_run"
    );
    expect(map.aug).toBe("running");
  });
});
