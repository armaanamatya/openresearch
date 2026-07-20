# Backend-Core Opportunity Audit — Handoff / Launch Spec

Date: 2026-05-31 · Branch context: `harden/root-harness` (worktree `/home/sww35/openresearch-harden`)
Status: **NOT STARTED — this is the launch spec.** Scoped via grill: *opportunity audit (analysis-only → prioritized backlog), backend core, multi-agent.*

## 0. Goal & non-goals
**Goal:** a comprehensive, evidence-grounded **opportunity audit** of the OpenResearch *backend core* — break it into subsystems, fan out analysis agents, and produce a **prioritized backlog** of improvements (correctness, cost/perf, simplification, test gaps, security, architecture) ranked by **value-vs-effort**.

**Non-goals (this pass):**
- **No code changes.** This produces a *backlog doc*, not commits. The user triages what to actually do.
- **Don't pollute `harden/root-harness`.** The hardening track has a tight thesis (tighten existing logic, no new deps). Audit findings are a *separate* artifact; implementation, if any, goes on its own branch.
- **Frontend out of scope** (skip `frontend/`); focus where the value + risk live.

## 1. Subsystems to audit (backend core)
| Subsystem | Path | Focus |
|---|---|---|
| RLM orchestrator | `backend/agents/rlm/` (run, primitives, report, context, system_prompt, sse_bridge) | reliability, dead code, the dual RLM/RDR paths, prompt bloat |
| RDR controller | `backend/agents/rdr/` | overlap with RLM, the deterministic cluster path |
| Provider runtime | `backend/agents/runtime/` (claude/openai adapters, registry, invoke, base) | provider parity (post-P1), the deferred per-role model picker, the aclose-race surface |
| Sandbox + budget | `backend/services/runtime/` (local_process, local_docker, runpod, brev, gpu_catalog, local_gpu_allocator, budget, artifacts) | **cost/perf** (GPU efficiency), the P4 exec template-method, log truncation, budget enforcement gaps |
| Ingestion | `backend/services/ingestion/` (resolving_parser, arxiv fetcher, OCR, discovery) | parse fidelity (see W-1), caching |
| Scoring | `backend/evals/paperbench/` (leaf_scorer, bundle, rubric generation) | scoring fidelity (see W-2), the §5c/B2 work |
| HTTP + persistence | `backend/routes/`, SQLite projections | API surface, leaderboard |
| Batch / scripts | `scripts/` (batch_reproduce, GPU leasing) | the local multi-GPU path |

## 2. Audit axes (score every subsystem against these)
1. **Correctness / reliability** — wedges, races, silent-inert guards, degenerate-run laundering.
2. **Cost & latency** — GPU efficiency, token burn, caching, log-handling. (Highest leverage for a GPU harness.)
3. **Simplification / dead code** — unused helpers, the RLM↔RDR duplication, prompt bloat.
4. **Test-coverage gaps** — what's under-tested (esp. the run_experiment escalation loop, the sandbox backends).
5. **Security** — secret handling (the `.env` is plaintext live keys), the exfil surface (parry-guard C1), MCP.
6. **Architecture / coupling** — what's hard to change/test; where seams are missing.

## 3. KNOWN WEAKNESSES — evidence-grounded seed (do NOT re-discover these)
Found by reading the live run `runs/prj_09047604e591d969` (2026-05-31) + the 14-launch SDAR debug history in memory `[[sdar-local-baseline-status]]`. Severity = impact on the harness's core promise (honest, faithful reproduction). "Status" notes what's already addressed so the audit doesn't re-propose fixed work.

### Scoring integrity (HIGH — this is the harness's whole purpose)
- **W-2 · Rubric generator drops truncated/"placeholder" leaves.** The live run's `batch_child.log` shows `generate_rubric_tree: dropped placeholder leaf` **×5** — including the stop-gradient gate, GRPO baseline, and the ALFWorld / Search-QA / WebShop eval leaves. The auto-generated rubric is silently *losing core requirements*, so the run is graded against an incomplete rubric. **Status: OPEN.** Likely the leaf-text truncation/placeholder heuristic is over-aggressive. High priority — it directly corrupts the score.
- **W-3 · Masked code-bugs laundered into `data_load_failures` → degenerate runs shipped as "success".** Recurring across launches; partly enforced by `a0b4c4d` (`_reclassify_masked_code_bugs`, `_degenerate_training_violation`). **Status: PARTIAL** — verify the enforcement actually fires on a fresh run; the pattern recurred repeatedly.

### Paper fidelity (HIGH)
- **W-1 · Parser fails → silent lossy fallback.** Live run: `parsed_full_text.txt missing — parser likely failed; falling back to workspace variable (lossy)`; the paper title degraded to the literal placeholder `"paper_text"`. The reproduction proceeded on **degraded paper input**. **Status: PARTIAL** — P0's ar5iv fallback helps arXiv HTML, but the parser-failure → silent-lossy-text path itself remains. The title-placeholder is a visible symptom worth tracing.

### Reliability (HIGH — runs wedge/crash)
- **W-4 · `claude-agent-sdk` aclose async-gen race.** Live run: `Loop <_UnixSelectorEventLoop ...> that handles pid ... is closed`. **Status: PARTIAL** — `150bf37` routed the root path through `run_isolated` with retry; the warning still appears. Quantify residual impact.
- **W-5 · Distributed-training fragility:** OOM-on-backward, torchrun setup-duplication deadlock, FSDP rank-0 NCCL 600s timeout. **Status: PARTIAL** (`a86f7d5` torchrun toggle, `a86290d` collective-discipline guidance) — but it's *guidance*, agent-compliance-dependent.
- **W-6 · ALFWorld multi-turn blocker chain** (API → config → game-count → reward-parse — one new bug per launch). **Status: WORKED-AROUND** (pivoted to Search-QA-only). The multi-turn envs remain brittle.
- **W-9 · "Guard wired but silently not firing" class.** The forced-iteration policy was inert for an unknown duration (thread-local on the wrong thread, fixed `6990d56`). **Audit action:** sweep for other guards that may not actually fire (thread/async boundaries, env-var gating).

### Cost / latency (MED — high leverage)
- **W-7 · Rollout configs 45–80 min/step** (unbatched HF generate; `AlfredTWEnv.init_env()` scanning all ~3500 games). **Status: PARTIAL** (guidance caps), still agent-dependent.
- **W-8 · Token burn / retry bursts** — a sub-agent cost-reduction spec exists (`2026-05-28-subscription-cost-reduction-design.md`); measure actual burn.
- Sequential per-model training leaves GPUs idle (parallel-per-model is a noted future improvement).

### Observability / data-quality (LOW–MED)
- **W-10 · Provider mislabel:** worker_reports.jsonl records `provider:"openai"` for a `claude-oauth` model. Wrong provenance in the report.
- **W-11 · Empty `build_environment` worker report** under `local` (1ms no-op, all fields empty) — uninformative.
- **W-12 · `compute_scope_invalid`** event in the live run's dashboard_events — a scope-validation issue surfacing; trace it.
- **W-13 · scope-validator key-normalization** (canonical model names vs sanitized metric keys). **Status: PARTIAL** (`1deb0fb`).

## 4. How to run the audit (multi-agent, backend-core)
1. **Discover** the work-list inline first: list the subsystem entry points + rough LOC per area (so the fan-out is sized to reality).
2. **Fan out** ~8–12 analysis agents (one per subsystem, the big ones split) — each: read its subsystem, score against the §2 axes, and return a structured finding list `{title, subsystem, axis, severity, evidence (file:line), proposed_change, value, effort}`. Seed each with the relevant W-items so they verify/extend rather than re-find.
3. **Synthesize:** dedupe across agents, drop anything already fixed (check git log / the Status notes), and **rank by value-vs-effort** into the backlog. A completeness-critic pass: "what subsystem/axis got thin coverage?"
4. Run it as a **Workflow** (the user opted into the multi-agent sweep) — pipeline `discover → analyze-per-subsystem → synthesize`.

## 5. Output
A single ranked backlog doc: `docs/audits/2026-05-31-backend-core-opportunity-backlog.md` — table of findings sorted by value-vs-effort, each with evidence anchors + a one-line proposed change + a suggested home (hardening phase, a new branch, or "won't do"). The user triages; nothing is implemented from this pass without a follow-up decision.

## 6. Relationship to the hardening track
The hardening phases (P1 done, P2 done, P3 partial, P4–P6 + P1.5 pending) are a *curated subset* of "make the harness trustworthy." This audit is the *broad net* — it will surface items that map into the existing phases (e.g., W-7/W-8 → P4 cost work; W-4 → P1.5 reliability) and items that are net-new. Keep the two artifacts distinct: the design doc is the committed plan; this backlog is the candidate pool.
