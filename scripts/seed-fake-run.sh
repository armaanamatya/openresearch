#!/usr/bin/env bash
# seed-fake-run.sh — create a fake run with worker reports for testing.
# Usage: ./scripts/seed-fake-run.sh [project_id]
# Respects REPROLAB_RUNS_ROOT (defaults to ./runs).

set -euo pipefail

PROJECT_ID="${1:-seed_reports_test}"
RUNS_ROOT="${REPROLAB_RUNS_ROOT:-./runs}"
RUN_DIR="${RUNS_ROOT}/${PROJECT_ID}"

mkdir -p "${RUN_DIR}/reports/worker_reports"
mkdir -p "${RUN_DIR}/code"

NOW=$(date -u +"%Y-%m-%dT%H:%M:%S+00:00")

# demo_status.json
cat > "${RUN_DIR}/demo_status.json" <<EOF
{
  "projectId": "${PROJECT_ID}",
  "outputDir": "${RUN_DIR}",
  "runMode": "rlm",
  "llmProvider": "anthropic",
  "status": "completed",
  "startedAt": "${NOW}",
  "updatedAt": "${NOW}",
  "sourceLabel": "Seeded test run"
}
EOF

# Worker 1: succeeded rdr_cluster
cat > "${RUN_DIR}/reports/worker_reports/worker-success-1.json" <<EOF
{
  "report_id": "wr-success-1",
  "worker_id": "worker-success-1",
  "worker_type": "rdr_cluster",
  "agent_id": "baseline-implementation",
  "status": "completed",
  "started_at": "${NOW}",
  "finished_at": "${NOW}",
  "model": "claude-sonnet-4-20250514",
  "provider": "anthropic",
  "cluster_id": "cluster-env-setup",
  "duration_ms": 45000,
  "implemented": ["Environment setup", "Dockerfile creation"],
  "left_undone": [],
  "commands": [
    {"command": "pip install torch", "exit_code": 0, "source": "worker_report"},
    {"command": "python -c 'import torch; print(torch.__version__)'", "exit_code": 0, "source": "tool_call"}
  ],
  "issues": [],
  "procedures_followed": true,
  "procedure_notes": "All steps followed",
  "error": null,
  "assignment": {
    "summary": "Set up the training environment for SDAR reproduction",
    "expected_outputs": ["Dockerfile", "requirements.txt"],
    "constraints": ["Must use PyTorch 2.1+"]
  },
  "execution_summary": {
    "concise_summary": "Created Dockerfile and requirements.txt for SDAR training",
    "implemented": ["Dockerfile", "requirements.txt"],
    "created_files": ["Dockerfile", "requirements.txt", "setup.py"]
  },
  "blockers": [],
  "errors": [],
  "artifacts": [
    {"path": "Dockerfile", "type": "config", "description": "Training container"},
    {"path": "requirements.txt", "type": "config", "description": "Python deps"}
  ],
  "tests": [],
  "next_actions": []
}
EOF

# Worker 2: failed rdr_cluster with SDK blocker
cat > "${RUN_DIR}/reports/worker_reports/worker-failed-1.json" <<EOF
{
  "report_id": "wr-failed-1",
  "worker_id": "worker-failed-1",
  "worker_type": "rdr_cluster",
  "agent_id": "baseline-implementation",
  "status": "failed",
  "started_at": "${NOW}",
  "finished_at": "${NOW}",
  "model": "claude-sonnet-4-20250514",
  "provider": "anthropic",
  "cluster_id": "cluster-training",
  "duration_ms": 12000,
  "implemented": [],
  "left_undone": ["Training loop", "Metrics collection"],
  "commands": [
    {"command": "python train.py", "exit_code": 1, "source": "tool_call"}
  ],
  "issues": ["Exception: Claude Code returned an error result: success"],
  "procedures_followed": null,
  "procedure_notes": "",
  "error": "Exception: Claude Code returned an error result: success",
  "assignment": {
    "summary": "Implement SDAR training loop for Qwen3-1.7B",
    "expected_outputs": ["train.py", "metrics.json"],
    "constraints": []
  },
  "execution_summary": null,
  "blockers": [
    {
      "title": "SDK success-with-no-text",
      "description": "Claude Code returned exit status 'success' but produced no text output.",
      "severity": "critical",
      "source": "claude_agent_sdk",
      "suggested_fix": "Check the agent prompt contract."
    }
  ],
  "errors": [
    {
      "message": "Exception: Claude Code returned an error result: success",
      "stack_or_trace": null,
      "source_file": "backend/agents/rdr/agent.py",
      "recoverable": false
    }
  ],
  "artifacts": [],
  "tests": [],
  "next_actions": [
    {
      "priority": "high",
      "action": "Investigate SDK success-with-no-text pattern",
      "owner_or_component": "claude_agent_sdk",
      "rationale": "Blocking all cluster completions"
    }
  ]
}
EOF

# Worker 3: blocked rlm_primitive
cat > "${RUN_DIR}/reports/worker_reports/worker-blocked-1.json" <<EOF
{
  "report_id": "wr-blocked-1",
  "worker_id": "worker-blocked-1",
  "worker_type": "rlm_primitive",
  "agent_id": "run_experiment",
  "status": "blocked",
  "started_at": "${NOW}",
  "finished_at": "${NOW}",
  "model": "gpt-5",
  "provider": "openai",
  "duration_ms": 3000,
  "implemented": [],
  "left_undone": ["Experiment execution"],
  "commands": [
    {"command": "docker run experiment", "exit_code": 137, "source": "worker_report"}
  ],
  "issues": ["OOM killed"],
  "procedures_followed": null,
  "procedure_notes": "",
  "error": "Container was OOM-killed (exit 137)",
  "assignment": {
    "summary": "Execute training experiment in Docker container"
  },
  "blockers": [
    {
      "title": "OOM Kill",
      "description": "Docker container killed due to insufficient memory",
      "severity": "high",
      "source": "docker",
      "suggested_fix": "Increase container memory limit or reduce batch size"
    }
  ],
  "errors": [
    {
      "message": "Container was OOM-killed (exit 137)",
      "recoverable": true
    }
  ],
  "artifacts": [
    {"path": "logs/experiment.log", "type": "log", "description": "Experiment stderr"}
  ],
  "tests": [],
  "next_actions": []
}
EOF

# summary_report.json
cat > "${RUN_DIR}/reports/summary_report.json" <<EOF
{
  "total_workers": 3,
  "by_status": {"completed": 1, "failed": 1, "blocked": 1},
  "critical_blockers": [
    {
      "title": "SDK success-with-no-text",
      "description": "Claude Code returned exit status 'success' but produced no text output.",
      "severity": "critical",
      "source": "claude_agent_sdk"
    }
  ],
  "files_changed": ["Dockerfile", "requirements.txt", "setup.py"],
  "commands_run": 4,
  "failed_commands": 2,
  "tests_summary": {"passed": 0, "failed": 0},
  "final_run_status": "completed",
  "top_next_actions": [
    {
      "priority": "high",
      "action": "Investigate SDK success-with-no-text pattern",
      "owner_or_component": "claude_agent_sdk"
    }
  ],
  "generated_at": "${NOW}"
}
EOF

# Also write the legacy worker_reports.jsonl for backward compat
echo '{"report_id":"wr-success-1","agent_id":"baseline-implementation","status":"completed","implemented":["Environment setup"],"commands":[{"command":"pip install torch","exit_code":0}],"issues":[],"procedures_followed":true}' > "${RUN_DIR}/worker_reports.jsonl"
echo '{"report_id":"wr-failed-1","agent_id":"baseline-implementation","status":"failed","implemented":[],"commands":[{"command":"python train.py","exit_code":1}],"issues":["Exception: Claude Code returned an error result: success"],"error":"Exception: Claude Code returned an error result: success"}' >> "${RUN_DIR}/worker_reports.jsonl"

echo "Seeded fake run at ${RUN_DIR}"
echo "  3 worker reports (1 succeeded, 1 failed, 1 blocked)"
echo "  summary_report.json present"
echo ""
echo "View at: http://localhost:3000/lab?projectId=${PROJECT_ID}"
