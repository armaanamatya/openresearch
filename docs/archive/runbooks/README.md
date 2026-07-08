# Archived runbooks

Point-in-time session handoffs and run logs, archived 2026-07-01 (repo cleanup).
They describe the state of the world on their date — do not treat as current.
Evergreen operator docs and CLAUDE.md-cited canonical runbooks stay in `docs/runbooks/`.

- [2026-05-23-e2e-rlmpaper-run-findings.md](2026-05-23-e2e-rlmpaper-run-findings.md) — Findings - E2E localhost run of the RLM paper (2026-05-23)
- [2026-05-23-ftrl-rdr-run-postmortem.md](2026-05-23-ftrl-rdr-run-postmortem.md) — Postmortem — FTRL RDR run scoring 3.2% (2026-05-23)
- [2026-05-24-runpod-dependency-map.md](2026-05-24-runpod-dependency-map.md) — RunPod dependency map — what the agent installs, why it sometimes fails, how to keep it working
- [2026-05-26-3day-run-audit.md](2026-05-26-3day-run-audit.md) — 3-Day Run Audit — 2026-05-24 through 2026-05-26
- [2026-05-27-rlm-reproduction-stability-plan.md](2026-05-27-rlm-reproduction-stability-plan.md) — 2026-05-27 RLM reproduction stability plan
- [2026-05-27-sdar-run-issues.md](2026-05-27-sdar-run-issues.md) — 2026-05-27 SDAR run issues + bug log
- [2026-05-31-sdar-full-reproduction-resource-map.md](2026-05-31-sdar-full-reproduction-resource-map.md) — SDAR Full Reproduction — Resource Map (2026-05-31)
- [2026-06-01-backend-core-merge-and-continuation-handoff.md](2026-06-01-backend-core-merge-and-continuation-handoff.md) — Backend-Core Remediation — Merge & Continuation Handoff (2026-06-01)
- [2026-06-01-harness-breakdown.md](2026-06-01-harness-breakdown.md) — Harness Breakdown & Recent Changelog (snapshot: 2026-06-01)
- [2026-06-01-sdar-fullscope-rerun-handoff.md](2026-06-01-sdar-fullscope-rerun-handoff.md) — SDAR full-scope rerun — handoff (2026-06-01)
- [2026-06-01-session-handoff.md](2026-06-01-session-handoff.md) — SESSION HANDOFF — 2026-06-01 (SDAR remediation + harness fairness/full-scope)
- [2026-06-02-merge-all-worktrees-onto-sdar.md](2026-06-02-merge-all-worktrees-onto-sdar.md) — Merge all feature worktrees onto `5.30.26_sdar` — handoff (2026-06-02)
- [2026-06-02-worktree-consolidation-handoff.md](2026-06-02-worktree-consolidation-handoff.md) — Worktree consolidation → one merge-ready branch for `main`
- [2026-06-03-azure-aks-gpu-backend-handoff.md](2026-06-03-azure-aks-gpu-backend-handoff.md) — Azure AKS GPU backend — standup runbook & handoff
- [2026-06-07-azure-k8s-gpu-implementation-prompt.md](2026-06-07-azure-k8s-gpu-implementation-prompt.md) — Azure / Kubernetes GPU Backend — Implementation Handoff Prompt
- [2026-06-08-agent-codegen-tdd-hardening-handoff.md](2026-06-08-agent-codegen-tdd-hardening-handoff.md) — Two-Axis Reproducibility Verdict + Pre-Training Fidelity Gate (handoff)
- [2026-06-08-merge-clean-branch-to-main-handoff.md](2026-06-08-merge-clean-branch-to-main-handoff.md) — Handoff — Merge the clean Azure line → `origin/main` (2026-06-08)
- [2026-06-09-recurring-failures-remediation.md](2026-06-09-recurring-failures-remediation.md) — 2026-06-09 — Recurring-failure remediation (Adam + All-CNN forensics)
- [2026-06-11-omnizip-opus-run.md](2026-06-11-omnizip-opus-run.md) — OmniZip (2511.14582) end-to-end run — Opus quality routing + sharded 7B hosting
- [2026-06-13-azure-aionic-deploy-handoff.md](2026-06-13-azure-aionic-deploy-handoff.md) — Azure AIONIC L1 deploy — handoff & redeploy runbook
- [2026-06-14-sdar-on-azure-live-test-handoff.md](2026-06-14-sdar-on-azure-live-test-handoff.md) — SDAR-on-Azure — LIVE end-to-end test handoff (post-hardening)
- [2026-06-14-sdar-on-azure-run.md](2026-06-14-sdar-on-azure-run.md) — SDAR on Azure — operator run playbook
- [2026-06-14-sdar-on-azure-session-handoff.md](2026-06-14-sdar-on-azure-session-handoff.md) — SDAR-on-Azure — session handoff
- [2026-06-16-grader-fidelity-remediation-handoff.md](2026-06-16-grader-fidelity-remediation-handoff.md) — Grader-Fidelity & Harness Remediation — New-Session Handoff
- [2026-06-16-sdar-on-gcp-a100-vm.md](2026-06-16-sdar-on-gcp-a100-vm.md) — SDAR on a GCP A100 VM — launch runbook (2026-06-16)
- [2026-06-19-sdar-gcp-e2e-guardrails-handoff.md](2026-06-19-sdar-gcp-e2e-guardrails-handoff.md) — SDAR end-to-end on GCP — guarded root + validated executor (handoff, 2026-06-19)
- [2026-06-20-grounded-self-improvement-gcp-test-handoff.md](2026-06-20-grounded-self-improvement-gcp-test-handoff.md) — Grounded self-improvement — GCP end-to-end test handoff (2026-06-20)
- [2026-06-20-sdar-gcp-actor-critic-run-handoff.md](2026-06-20-sdar-gcp-actor-critic-run-handoff.md) — SDAR-on-GCP with the Actor–Critic Evidence Layer — New-Session Handoff
- [2026-06-20-sdar-harness-refactor-and-external-validation-handoff.md](2026-06-20-sdar-harness-refactor-and-external-validation-handoff.md) — SDAR harness reliability + external-validation refactor — handoff (2026-06-20)
- [2026-06-20-validation-coverage-and-capped-rerun-handoff.md](2026-06-20-validation-coverage-and-capped-rerun-handoff.md) — Validation-coverage improvements + capped SDAR re-run — handoff (2026-06-20)
- [2026-06-21-cleanup-phase-a-handoff.md](2026-06-21-cleanup-phase-a-handoff.md) — Cleanup Phase A — ready-to-run handoff (OWNER actions)
- [2026-06-21-smoke-crash-surfacing-debug-handoff.md](2026-06-21-smoke-crash-surfacing-debug-handoff.md) — Pre-GPU smoke crash-surfacing — debug handoff (why SDAR keeps dying at the smoke + the crash never reaches the executor)
- [2026-06-22-lifecycle-driver-and-feedback-fix-handoff.md](2026-06-22-lifecycle-driver-and-feedback-fix-handoff.md) — Lifecycle Driver + Forced-Iteration Feedback Fix — Handoff (2026-06-22)
- [2026-06-22-sdar-gcp-e2e-and-rl-smoke-fix-handoff.md](2026-06-22-sdar-gcp-e2e-and-rl-smoke-fix-handoff.md) — SDAR-on-GCP e2e + the RL-aware smoke fix — handoff (read first, self-contained)
