# Backend-Core Remediation — New-Session Handoff & Execution Prompt

> **You are picking up a remediation track for the OpenResearch backend.** This doc is self-contained: read it top-to-bottom and you can execute every phase without prior context. It is the *execution prompt*; the **evidence** for each finding (`F-NN`) lives in the companion backlog `docs/audits/2026-05-31-backend-core-opportunity-backlog.md` (and `.json`). Read a finding's backlog entry — which carries the verified `file:line` + proposed change — before you touch it.

Date: 2026-05-31 · Worktree: `/home/sww35/openresearch-harden` · Branch: `harden/root-harness`
Source: a 9-subsystem, 18-agent **audit with an adversarial verify pass** (every finding independently confirmed still-present in current code; 2 candidates dropped as already-fixed). 47 confirmed-unfixed findings.

---

## 0. Mission
Burn down the 47 confirmed backend weaknesses **in value order**, one tiny tested commit at a time, **without regressions**, so the OpenResearch reproduction harness grades *honestly* (faithful paper + complete rubric + measured-not-claimed scores) and runs *cheaply and reliably*.

## 1. Cold-start orientation
- **What OpenResearch is:** an agent that reproduces research papers end-to-end and *scores* the reproduction against a rubric. The harness is the trusted referee; the agents are workers.
- **Already shipped on this branch** (don't redo): **P0** ar5iv fallback (`5df6d19`); **P1** RuntimeGuard blocklist activation + Claude/OpenAI `allowed_tools` parity + hermetic SDK config (`647ea37`/`83413f9`/`c2e527b`/`6d0695e`); **P2** experiment manifest + `final_report` back-link (`b733b30`/`0031398`/`f86e84f`); **P3** metric-projection §5b + adversarial-grader stance §8a-B1 (`7152a78`/`2cbdf97`). Full suite was green (3371) at handoff.
- **Read first:** this doc → the backlog (`docs/audits/2026-05-31-backend-core-opportunity-backlog.md`) → the hardening design doc (`docs/superpowers/specs/2026-05-31-root-harness-hardening-design.md`, for the 8 invariants + the still-pending P1.5/P4–P6) → memory `[[root-harness-hardening]]`, `[[harden-borrow-provenance]]`, `[[sdar-local-baseline-status]]`.

## 2. Critical gotchas (these cost time — heed them)
- **Bash cwd resets to `/home/sww35/openresearch` between calls.** Prefix every command: `cd /home/sww35/openresearch-harden && …`.
- **The worktree has no venv.** Run tests with the main checkout's interpreter from the worktree cwd: `cd /home/sww35/openresearch-harden && /home/sww35/openresearch/.venv/bin/python -m pytest …` (cwd wins, so `backend` imports from the worktree).
- **Commit as `lolout1`, NO co-author trailer** (overrides the harness default — see `[[commit-attribution-preference]]`).
- **Anchors drift.** The audit's `file:line` were correct at this HEAD, but always **re-Read the cited region right before editing** and match `Edit` on unique content, not line numbers.
- **The audit ran on this worktree's HEAD** (post P0–P3), so these are genuinely *post-hardening* gaps — but still re-confirm each finding's evidence (and `git log -- <file>`) before fixing; don't trust the title alone.

## 3. Guardrails (non-negotiable — same discipline that kept P0–P3 clean)
1. **No hallucination.** Every change cites a real `file:line` you read. If a finding's evidence doesn't reproduce, mark it dropped and move on — don't invent.
2. **Only tighten existing logic.** No new dependencies, no rewrites, no new frameworks. Narrow fixes.
3. **Behavioral changes ship default-on behind a `OPENRESEARCH_*` env hatch**; anything that can move a rubric score ships **observe-first** (log the would-change delta, enforcement OFF) and is flipped only after one real SDAR run confirms no spurious loss.
4. **One finding (or one tight cluster) per commit**, each with its test, so a regression is attributable.
5. **Phase exit gate = new tests green AND `pytest tests/ -q` full-suite zero-regression.** Ship a phase's tests before starting the next.

## 4. How to run / verify
```bash
cd /home/sww35/openresearch-harden
/home/sww35/openresearch/.venv/bin/python -m pytest tests/ -k "<area>" -q     # focused
/home/sww35/openresearch/.venv/bin/python -m pytest tests/ -q                 # full-suite gate (~70s, expect 6 env-skips)
```
To validate an observe-first scoring/fidelity change end-to-end, launch the canonical SDAR baseline (see `[[sdar-local-baseline-status]]` for the proven 2-GPU recipe) and inspect `runs/<id>/final_report.json` + `dashboard_events.jsonl`.

---

## 5. Phases (value-ordered). `sev/val/eff` = severity / value / effort.

### Phase A — Fidelity: stop grading garbage (HIGHEST — 7 findings)
**Why:** the live SDAR run was graded against degraded paper text *and* an incomplete rubric. No downstream scoring honesty matters if the inputs are corrupt. This is the highest-leverage phase.
| F | sev/val/eff | finding |
|---|---|---|
| **F-27** | h/h/sm | HTML parser drops every `<math>` equation → strips the SDAR rubric invariants (`g_t=σ(β·Δ_t)`, λ=0.1) from the paper text |
| **F-32** | h/h/sm | rubric-gen placeholder regex over-drops concrete metric/equation leaves (the 5 dropped leaves in the live run) |
| **F-34** | m/h/sm | the placeholder test only exercises patterns that pass by luck — none of the 3 real failures (pairs with F-32) |
| **F-28** | h/h/sm | arXiv HTML fetch is single-shot (no retry) while PDF got 3-attempt retry — transient failure → lossy run |
| **F-29** | m/h/sm | parsed-paper precondition gate defaults `allow_lossy=True` — a parser failure still ships a degraded run silently |
| **F-30** | m/m/sm | corpus title degrades to the literal placeholder `'paper_text'` (visible W-1 symptom) |
| **F-31** | l/m/me | `raw_paper.{html,pdf}` re-fetched on every ingest — no validated-cache reuse |
**Approach:** F-27 + F-32/F-34 first (they directly restored the SDAR invariants into paper+rubric). F-29 should flip to fail-closed-on-lossy behind a hatch (observe-first if it changes run outcomes). **Exit gate:** ingestion + scoring + rubric tests green; ideally one SDAR ingest shows `parsed_full_text.txt` written + the 5 leaves retained.

### Phase B — Scoring & guard correctness (HIGH — 9 findings)
**Why:** the grader + the "guard wired but silently not firing" class (the same bug shape as the forced-iteration `6990d56` fix).
| F | sev/val/eff | finding |
|---|---|---|
| **F-06** | h/h/sm | two-experiment `FINAL_VAR` guard never resets per turn → false refusals (W-9 sibling) |
| **F-33** | m/h/me | grader evidence reads only the first 6 KB *head* of each code file alphabetically — misses later code |
| **F-03** | m/m/sm | `_data_load_failure_is_code_bug` phrases false-positive on genuine data-unavailability ("no such file", "errno 2", "has no attribute") |
| **F-07** | m/m/sm | `failure_classifier`: declared `disk_exhausted` class has no inline detector — real ENOSPC → "unknown" |
| **F-08** | m/m/sm | `failure_classifier`: HF gated-repo 401/403 + NCCL timeout misclassified as "unknown" |
| **F-04** | l/l/sm | escalation-loop OOM detection misses watchdog-killed OOM (no exit_code/infra flag) despite the inline comment |
| **F-05** | l/l/sm | masked-code-bug reclassification + other success-gated postflight guards never run on a *failed* run |
| **F-11** | l/l/sm | `build_final_report` reconciles verdict against the root's self-reported score *before* the best-of-run floor |
| **F-35** | l/m/me | §5c programmatic citation-clamp + B2 spec-gate short-circuit (**the known-pending P3 remainder** — observe-first, needs a SDAR run) |
**Approach:** F-06 is a W-9 guard-correctness bug — fix + add a "guard fires on the worker thread" regression like the `6990d56` pattern. F-03/F-07/F-08 are tightening the classifier phrase/detector tables (deterministic, easy tests). F-35 is already scheduled — do it last, observe-first. **Exit gate:** forced-iteration + failure_classifier + leaf_scorer + report tests green.

### Phase C — Cost / perf (HIGH leverage, mostly small — 7 findings)
**Why:** a GPU+LLM harness; these are cheap wins with real $ / latency impact.
| F | sev/val/eff | finding |
|---|---|---|
| **F-01** | h/h/sm | `verify_against_rubric` cache key SHA-256-hashes the **entire `code/` tree incl. multi-GB checkpoints** every call (the leaf scorer is already capped; this path wasn't) |
| **F-02** | l/l/sm | dead/wrong metrics-invalidation bit in that same cache key (`code/metrics.json` is never written) — fix with F-01 |
| **F-09** | m/m/sm | per-paper context metadata at prompt position 2 busts cross-run prompt-cache |
| **F-18** | m/m/sm | OpenAI `reasoning_tokens` dropped in `collect_agent_text` usage accumulation → ledger under-counts |
| **F-39** | m/m/sm | leaderboard cache grows unbounded + re-stats every run dir on every request |
| **F-12** | h/h/sm | RDR cluster-agent SDK cost written to `code/cost_ledger.jsonl` not the run root → final-report under-reports cost |
| **F-17** | l/l/sm | RDR final-report cost double-keys `total_usd` into both `llm_usd` and `primitives` |
**Approach:** F-01+F-02 together (one cache-key fix). The rest are independent small fixes. **Exit gate:** primitives + runtime + leaderboard + rdr cost tests green.

### Phase D — Budget & runtime enforcement (this IS design §6 / P4 — 8 findings)
**Why:** the design already planned a `RuntimeBackend.exec` template-method to centralize budget+watchdog+log-handling; the audit confirms the gaps and gives the exact list. **Do this as the design's P4** — these findings ARE its content.
| F | sev/val/eff | finding |
|---|---|---|
| **F-22** | h/h/sm | `BrevBackend` has zero budget enforcement despite being a paid GPU sandbox |
| **F-23** | m/h/sm | GPU/pod budget enforced only in-arrears at `exec()` boundaries — no upfront/mid-exec cap (design #5) |
| **F-15** | m/m/me | `run_budget` caps (`--max-usd`/`--max-pod-seconds`/`--max-run-gpu-usd`) not enforced in the RDR path |
| **F-42** | m/h/sm | `--max-run-gpu-usd` run-level cap silently dropped on the rdr + batch paths |
| **F-26** | m/h/me | `RuntimeBackend.exec` is a bare abstractmethod — budget/watchdog logic duplicated-by-omission across backends (the template-method) |
| **F-24** | m/m/sm | `LocalProcessBackend.exec` output fully uncapped in memory (no `[-32000:]` tail) — the M4 log-truncation family (also `runpod_backend.py:1323` + `local_docker.py:351`) |
| **F-25** | l/l/sm | `artifacts.py` command-log/provenance writers are dead code (the **`commands.log` revival** re-scoped here from P2) |
| **F-14** | h/h/me | RDR has no process-level wall-clock watchdog — a wedged scorer/env-build/SDK-aclose deadlocks the run (design "watchdog unify") |
**Approach:** build the `exec` template-method (F-26) first — it's the seam the others hang off (F-22/F-23/F-24/F-25 all become "implement the hook once"). Then watchdog-unify (F-14). **Exit gate:** the design's P4 conformance test (parametrized over all backends: budget enforcement + log truncation + commands.log emission) + watchdog-on-RDR test.

### Phase E — Security, egress & resume safety (HIGH where it bites — 5 findings)
| F | sev/val/eff | finding |
|---|---|---|
| **F-37** | h/h/me | `POST /runs/arxiv` is a server-side SSRF when the demo gate is unset (the local-dev default) |
| **F-40** | m/m/sm | `post_message` has no PII/secret scrubber despite the "defense-in-depth" contract (design borrow #6) |
| **F-36** | h/h/sm | SSE byte-offset reader drops events torn across poll reads (partial-line bug) |
| **F-20** | m/m/sm | `_resolve_mcp_servers` has zero tests — the Bearer-header injection + enabled-agents CSV-split (design P5 test) |
| **F-41** | l/l/sm | `resume_run` re-spawns the orchestrator without validating any persisted checkpoint exists (design borrow #10) |
**Approach:** F-37 (SSRF) is the only true security risk — validate/allowlist the arxiv id → URL even when the demo gate is off. F-40/F-20/F-41 are design-borrow items (#6/#10 + the MCP test) — fold them here. **Exit gate:** route + sse_bridge + mcp + resume tests green; an SSRF negative test.

### Phase F — Cleanup, dead code, provider, tests (LOW-risk, do last / opportunistic — 11 findings)
| F | sev/val/eff | finding |
|---|---|---|
| **F-13** | l/l/sm | dead `_ClusterWatchdog` class (63 lines) — delete (design says so) |
| **F-44** | l/l/sm | `_read_last_rubric` duplicate path in the fallback tuple + misses the flat layout |
| **F-46** | l/m/me | `cmd_reproduce` is a ~456-line god-function — extract the env-var bridging / ingest / title steps |
| **F-43** | m/m/sm | `batch_reproduce` docstring advertises duplicate-paper usage the code can't support |
| **F-45** | m/m/me | scripts-cli helper coverage gap (`_canonical_project_id`, `_extract_score`, …) |
| **F-47** | l/l/sm | `batch_reproduce` `live_procs` module-global mutated by N worker threads with no lock |
| **F-16** | l/l/sm | RDR `total_agent_dispatches` ("iterations") overcounts on resume |
| **F-19** | l/l/sm | `collect_agent_text` labels ledger by the runtime's hardcoded `provider_name` (W-10 provider mislabel) |
| **F-10** | l/l/sm | `include_hints=False` cost lever exists but is never exercised |
| **F-38** | m/m/me | chat-message + RDR-introspection routers ignore the injected `run_service.runs_root` (latent path bug) |
| **F-21** | l/l/me | per-role provider selection unimplemented (the deferred per-role model picker) |
**Approach:** batch the trivial dead-code/doc fixes; F-46/F-45/F-38 are larger refactors — gate on whether they're worth it. **Exit gate:** full suite green; no behavior change for the pure-cleanup items.

---

## 6. Reconciliation with the hardening design doc
- **Phase D ≈ design P4** (budget/runtime template-method + watchdog unify + commands.log). Treat them as the same work; the audit just supplied the exact finding list.
- **F-35** = the **P3 remainder** (§5c citation-clamp + B2). **F-40/F-20/F-41** = design borrows **#6/#10** + the P5 MCP test. **F-24/F-25** = M4 + the P2-deferred commands.log.
- **P1.5** (preventive `can_use_tool` rework + parry-guard exfil-taint C1) is **separate and still pending** — not in this backlog (it's a known scheduled phase, higher risk). Sequence it independently.
- Net-new from the audit (not previously tracked): **F-01, F-06, F-12, F-27, F-32, F-36, F-37** — the highest-value surprises; prioritize them.

## 7. Suggested order
**A → B → C** are the highest value-per-effort (fidelity + scoring honesty + cheap perf wins, almost all small-effort). **D** is the meaty refactor (= P4). **E** has the one real security item (F-37). **F** is opportunistic cleanup. Within each phase, do the `h/h/sm` items first. One tested commit each; full-suite gate before advancing.
