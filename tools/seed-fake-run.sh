#!/usr/bin/env bash
# Seed a fake in-flight run for UI audits.
#
# Writes runs/<project_id>/{demo_status.json,agent_telemetry.jsonl}
# so /lab?projectId=<id> renders the workflow view, /library shows
# the run, and the telemetry strip has non-zero numbers.
#
# Usage:
#   tools/seed-fake-run.sh                 # default project id
#   tools/seed-fake-run.sh prj_my_smoke    # custom project id
#   tools/seed-fake-run.sh --clean         # remove all prj_*_smoke fixtures

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--clean" ]]; then
    find runs -maxdepth 1 -type d -name "prj_*_smoke" -exec rm -r {} +
    echo "removed fake fixtures"
    exit 0
fi

project_id="${1:-prj_diffusion_smoke}"
run_dir="runs/${project_id}"
mkdir -p "${run_dir}"

cat > "${run_dir}/demo_status.json" <<JSON
{
  "projectId": "${project_id}",
  "outputDir": "${run_dir}",
  "runMode": "sdk",
  "llmProvider": "anthropic",
  "executionMode": "efficient",
  "sandboxMode": "runpod",
  "gpuMode": "auto",
  "model": "sonnet",
  "status": "running",
  "sourceKind": "uploaded_pdf",
  "sourceLabel": "Diffusion Policy reproduction",
  "sourceNote": "Smoke fixture",
  "startedAt": "2026-05-14T15:00:00Z",
  "updatedAt": "2026-05-14T15:08:00Z",
  "pid": 1,
  "payload": {
    "summary": { "stage": "baseline_implemented", "doneCount": 6, "totalCount": 14 },
    "decisionLog": ["gate_1: verified", "Baseline: CartPole-v1 selected"],
    "gates": {
      "gate_1": { "passed": true, "status": "verified", "chipStatus": "passed", "detail": "Plan OK" },
      "gate_2": { "passed": false, "chipStatus": "running" },
      "gate_3": { "passed": false, "chipStatus": "pending" }
    },
    "pathStates": { "opt": "upcoming", "bb": "upcoming", "aug": "upcoming", "hor": "upcoming", "div": "upcoming" }
  },
  "log": "[15:00] paper-understanding started\n[15:01] paper-understanding completed\n[15:02] environment-detective started\n[15:04] reproduction-planner: contract written\n[15:04] Gate 1: verified\n[15:05] baseline-implementation started",
  "telemetry": []
}
JSON

cat > "${run_dir}/agent_telemetry.jsonl" <<JSON
{"agent_id":"paper-understanding","model":"claude-sonnet-4-6","duration_seconds":41.2,"message_count":12,"output_chars":8420,"success":true}
{"agent_id":"environment-detective","model":"claude-sonnet-4-6","duration_seconds":18.7,"message_count":6,"output_chars":2150,"success":true}
{"agent_id":"reproduction-planner","model":"claude-sonnet-4-6","duration_seconds":25.3,"message_count":8,"output_chars":3890,"success":true}
{"agent_id":"baseline-implementation","model":"claude-sonnet-4-6","duration_seconds":240.0,"message_count":31,"output_chars":18250}
JSON

echo "seeded ${project_id}"
echo "open: http://127.0.0.1:3001/lab?projectId=${project_id}"
