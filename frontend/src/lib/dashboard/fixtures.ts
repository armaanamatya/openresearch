import type { DashboardSnapshot } from "./contracts";

export const sampleDashboardSnapshot: DashboardSnapshot = {
  run_id: "run_ppo_cartpole",
  project_name: "PPO CartPole Reproduction",
  current_stage: "improvement",
  stage_summary:
    "Baseline artifacts are verified. Improvement agents are exploring follow-up paths while the supervisor tracks lineage and evidence coverage.",
  tasks: [
    {
      task_id: "task_001",
      parent_task_id: null,
      root_run_id: "run_ppo_cartpole",
      agent_id: "orchestrator",
      title: "Coordinate reproduction pipeline",
      summary: "Tracks the end-to-end run, publishes shared state, and delegates downstream work.",
      status: "running",
      stage: "plan",
      scope: "global_verified",
      delegation_depth: 0,
      assigned_budget: "overall run budget",
      assigned_timeout: "30m",
      write_scope: "project-wide ledgers",
      output_record_ids: ["record_root_summary"],
      artifacts: ["runs/root/report.md"],
      structured_outputs: {
        active_stage: "improvement",
        pending_children: 2
      },
      citations: [
        {
          source_id: "src_001",
          trust_level: "primary",
          content: "PPO method and evaluation target are defined in the paper's experiment section.",
          source_kind: "paper_section"
        }
      ],
      published_at: "2026-05-09T02:05:00Z"
    },
    {
      task_id: "task_017",
      parent_task_id: "task_001",
      root_run_id: "run_ppo_cartpole",
      agent_id: "environment_detective",
      title: "Recover baseline environment",
      summary: "Resolve runtime versions and container requirements before the baseline run.",
      status: "verified",
      stage: "baseline",
      scope: "global_verified",
      delegation_depth: 1,
      assigned_budget: "build-only",
      assigned_timeout: "8m",
      write_scope: "runs/baseline/",
      output_record_ids: ["record_env_spec"],
      artifacts: ["runs/baseline/Dockerfile", "runs/baseline/environment.lock"],
      structured_outputs: {
        python: "3.11",
        torch: "cpu-only",
        gymnasium: "confirmed"
      },
      citations: [
        {
          source_id: "src_042",
          trust_level: "secondary",
          content: "Issue #14 confirms CUDA is unnecessary for the PPO demo baseline.",
          source_kind: "github_issue"
        }
      ],
      published_at: "2026-05-09T02:12:00Z"
    },
    {
      task_id: "task_042",
      parent_task_id: "task_017",
      root_run_id: "run_ppo_cartpole",
      agent_id: "failure-analysis-child",
      title: "Inspect reward-collapse regression",
      summary: "Analyze why one improvement path collapsed after initial gains.",
      status: "completed",
      stage: "improvement",
      scope: "branch_shared",
      delegation_depth: 2,
      assigned_budget: "path_2 analysis",
      assigned_timeout: "5m",
      write_scope: "runs/improvements/path_2/",
      output_record_ids: ["record_failure_report"],
      artifacts: ["runs/improvements/path_2/failure_report.json"],
      structured_outputs: {
        root_causes: ["reward collapse after epoch 9"],
        recommended_next_action: "reduce entropy coefficient"
      },
      citations: [
        {
          source_id: "src_011",
          trust_level: "primary",
          content: "Paper notes entropy tuning as a stability lever for PPO.",
          source_kind: "paper_section"
        },
        {
          source_id: "src_045",
          trust_level: "strong_secondary",
          content: "Run logs show policy entropy spikes immediately before regression.",
          source_kind: "run_log"
        }
      ],
      published_at: "2026-05-09T02:23:00Z"
    },
    {
      task_id: "task_051",
      parent_task_id: "task_001",
      root_run_id: "run_ppo_cartpole",
      agent_id: "supervisor_verifier",
      title: "Review improvement comparability",
      summary: "Ensure path agents preserved dataset, metric, and seed policy before promoting results.",
      status: "verification_pending",
      stage: "verification",
      scope: "global_verified",
      delegation_depth: 1,
      assigned_budget: "verification window",
      assigned_timeout: "6m",
      write_scope: "runs/improvements/",
      output_record_ids: ["record_verifier_pending"],
      artifacts: ["runs/improvements/verification.json"],
      structured_outputs: {
        open_caveats: ["Need one final comparability check for path_3"]
      },
      citations: [
        {
          source_id: "src_083",
          trust_level: "primary",
          content: "Comparability contract requires the same environment, benchmark, and metric as the baseline.",
          source_kind: "paper_section"
        }
      ],
      published_at: "2026-05-09T02:28:00Z"
    }
  ],
  events: [
    {
      event: "agent_started",
      agent_id: "environment_detective",
      task_id: "task_017",
      parent_task_id: "task_001",
      timestamp: "2026-05-09T02:08:00Z",
      status: "running",
      summary: "Environment detective started runtime recovery."
    },
    {
      event: "agent_reasoning_step",
      agent_id: "environment_detective",
      task_id: "task_017",
      parent_task_id: "task_001",
      timestamp: "2026-05-09T02:09:10Z",
      step_type: "rlm_query",
      query: "What runtime hints exist for the baseline demo?",
      context_segment: "github_issues[14]",
      result: "CPU-only baseline is supported for the CartPole demo path.",
      citations: [
        {
          source_id: "src_042",
          trust_level: "secondary",
          content: "Issue #14 confirms CUDA is unnecessary for the demo baseline.",
          source_kind: "github_issue"
        }
      ]
    },
    {
      event: "shared_state_updated",
      agent_id: "failure-analysis-child",
      task_id: "task_042",
      parent_task_id: "task_017",
      timestamp: "2026-05-09T02:23:00Z",
      record_type: "task_result",
      scope: "branch_shared",
      status: "completed",
      structured_outputs: {
        root_causes: ["reward collapse after epoch 9"],
        recommended_next_action: "reduce entropy coefficient"
      },
      citations: [
        {
          source_id: "src_011",
          trust_level: "primary",
          content: "Paper notes entropy tuning as a stability lever for PPO.",
          source_kind: "paper_section"
        },
        {
          source_id: "src_045",
          trust_level: "strong_secondary",
          content: "Run logs show policy entropy spikes immediately before regression.",
          source_kind: "run_log"
        }
      ]
    },
    {
      event: "context_enrichment",
      agent_id: "orchestrator",
      task_id: "task_001",
      parent_task_id: null,
      timestamp: "2026-05-09T02:24:00Z",
      variable_name: "failure_report_path_2",
      visibility: "branch_shared",
      details: "Failure analysis output promoted for sibling comparison inside the same improvement branch."
    },
    {
      event: "verification_gate_result",
      agent_id: "supervisor_verifier",
      task_id: "task_051",
      parent_task_id: "task_001",
      timestamp: "2026-05-09T02:29:00Z",
      gate: "improvement",
      outcome: "caveat",
      details: "Path_3 still needs a final comparability review before promotion.",
      citations: [
        {
          source_id: "src_083",
          trust_level: "primary",
          content: "Comparability contract requires the same environment, benchmark, and metric as the baseline.",
          source_kind: "paper_section"
        }
      ]
    }
  ]
};
