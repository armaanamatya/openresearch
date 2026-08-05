<!-- doc-meta: status=current; last-verified=2026-08-01 -->
# Open issues — the honest ledger

One line per genuinely unresolved issue. When you fix one, delete its row in the same
commit. History/narrative lives in the periods dossiers, not here.

| Issue | Since | Status | Pointer |
|---|---|---|---|
| Tier-3 Phase C: scheduler FREEZE/PROMOTE/REVIVE never triggered on a real GPU run — two billed A100 runs (treat2_grok, treat3_gpt) both fail-closed at receipt production because no trainer emitted the 5-component checkpoint contract; blocker is harness-forced checkpoint emission (pre-scaffolded `train_cell.py` or `OPENRESEARCH_GKE_SYNTH_CELL`), not GPU budget (already spent ~$5–6); default-ON authority flip still needs ≥3 paired A/B runs + grader-σ gate + operator sign-off | 2026-07-22 | ⛔ blocked on harness-forced checkpoint emission | `docs/progress/2026-07-22-tier3-adam-progress.md` |
| Ablation grid ops G2–G5 never fire (only G1 branching has ever fired) | 2026-07-31 | open — limits ablation campaign coverage | `docs/superpowers/plans/2026-07-31-feature-ablation-campaign-plan.md` §grid |
| `grok` is not a validated executor (emits no cell/commands manifest → evidence gate fails the run) | 2026-07-22 | guarded 2026-08-03 — fail-fast at launch: a grok token/deployment resolving into executor/verifier/grader raises before any spend (`role_models.check_grok_execution_roles`, wired via `run.py::_enforce_grok_executor_guard`; escape hatch `OPENRESEARCH_ALLOW_GROK_EXECUTOR` → loud run_warning; sanctioned executor = `sonnet-foundry`). Validation as a production executor still NOT done — grok remains root-only | `backend/agents/rlm/role_models.py` guard + `docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md` troubleshooting table |
| SDAR baseline: headline +7% never reproduced (best partial 0.363/0.600; Track B authors-trainer 0.456 verified) | 2026-06 | open — aspirational, canonical stress test | `best_runs/sdar/README.md` |
| Parser idempotence keys on the event store, not run-dir files — stale state needs a fresh `--project-id` | 2026-07-08 | open (by-design; documented workaround) | `docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md` |
| No funded `ANTHROPIC_API_KEY`; `OPENAI_API_KEY` dead (401). Root-model selection prefers gpt-5 whenever `OPENAI_API_KEY` is present (`resolve_root_model`) — a dead key silently hijacks the root | 2026-06 | open (by-design selection; operative posture: Foundry keys) | `docs/runbooks/2026-08-01-remote-run-llm-auth.md` |
| `docs/policies/artifacts.md`: best_runs history frozen "pending a separately reviewed curation pass" that never happened | 2026-07-20 | open — schedule or drop | `docs/policies/artifacts.md` |

**Resolved recently (kept ~30 days to kill stale memory, then delete):** 18 collection errors +
20 test failures — fixed in `954e3a8b` (suite now collects 10.2k+ clean). k8s 409 retry
collision — fixed in `5c026301` (PR #12). Cutout FALSE-"failed" Tier-1 fixes — merged via the
consolidation trunk. GKE IAM grants + train-scope blockers — moot (GKE is NOT USED). OAuth
sub-agent SDK flakiness — moot (⛔ OAuth forbidden, 2026-08-01). Scripted GCP VM path
SDAR-hardcoded (no arbitrary-paper launcher) — resolved 2026-08-03 by the generic
`scripts/vm_paper_run.sh` (caveat: `VmComputeProvider.launch`'s campaign auto-launch leg still
invokes the SDAR-pinned `scripts/sdar_gcp_run.sh`; see the 2026-07-22 runbook's direct-recipe
note). Foundry LLM cost blindness —
resolved 2026-08-03: Foundry-Claude prices via `pricing.py` aliases + the root usage drain
(`run.py::_drain_foundry_root_usage_to_ledger`, landed `8cafa936`), and unknown-rate models now
write explicit `"unpriced": true` ledger rows surfaced in `demo_status.cost_summary` /
`final_report.cost` / `tokens_total.json` (`cost_confidence: "partial"`); idle-GPU-node time
remains blind (see root `CLAUDE.md` → "Cost visibility").
