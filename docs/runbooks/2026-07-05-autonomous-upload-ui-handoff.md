# Autonomous Upload → Live Reproduction UI — Execution Handoff (2026-07-05)

**Read this first, then open the plan and start Task 1.** This is a clean context-clear point: everything a
fresh session needs is committed to the branch or noted below. No re-derivation required.

## What to build

Spec: `docs/superpowers/specs/2026-07-05-autonomous-upload-ui-and-live-reproduction-design.md`
Plan: `docs/superpowers/plans/2026-07-05-autonomous-upload-ui-implementation-plan.md` (14 tasks, TDD, exact
`file:line` anchors — authored from a full backend + frontend recon this session; you do NOT need to re-run
Explore).

One-liner: PDF upload / arXiv id → opt-in `autonomous` → GKE + Opus-4.8-Foundry root reproduction → visible
spec-generation + external-validation phase → seamless redirect to a live agentic-reasoning session view, in
the alphaXiv maroon / neo-brutalist aesthetic (reference: `alphaxiv_{1..5,autoresearc}.png` in the primary
worktree root `/home/abheekp/openresearch/`).

## Environment (already provisioned — do not redo)

- **Worktree:** `/home/abheekp/openresearch-autonomous-ui`, branch `feat/autonomous-upload-ui`.
- **Base commit:** `ed086edb` — "Add the Anthropic-Foundry provider + lifecycle-primary/execute-mode
  foundation". This is the committed foundation this feature CONSUMES (`opus-foundry`/`sonnet-foundry` tokens,
  `run_lifecycle_primary`, execute-mode). The UI branch is already rebased onto it. Do NOT rewrite below it.
- **Python:** `.venv` (full `backend/requirements.txt` + `requirements-dev.txt`, editable-installed to THIS
  worktree). Run tests with `.venv/bin/python -m pytest …` (socket-hermetic).
- **Node:** `node_modules` installed via `npm ci`. Use `source ~/.nvm/nvm.sh && nvm use v22.14.0` (system Node
  is broken for vitest). `npm test` (vitest), `npx playwright test` (no npm alias), lab suite
  `--no-file-parallelism`.
- **`.env`:** copied from the primary worktree (funded `AZURE_FOUNDRY_API_KEY` = Opus 4.8 / Sonnet 5; gitignored,
  will NOT be committed). Foundation tests are green with it present.

## Execution model

- `superpowers:subagent-driven-development`: fresh Sonnet-max implementer per task; **Opus reviews every diff**
  (implementer self-verifies, then a fresh mutation-armed reviewer). TDD; commit per task; descriptive
  present-tense messages; **no AI-attribution trailers**. Push to `deepinvent` only on explicit request.
- Backend tasks (T1-T8, T3b) are Python + hermetic — parallelizable with frontend Phase 0 (T9a/T9b).
  Frontend flow tasks (T10-T13) depend on the backend request/SSE contracts (see the plan's dependency graph).
- Exit bar per task: touched tests green; `uvx ruff@0.15.16 check .` clean (Python) / `npx tsc --noEmit` clean
  (TS); NO NEW failures beyond the baseline set below.

## Baseline known-failing tests (exit bar = no NEW failures)

`test_accelerator`, `test_external_validator`, `test_report_validation_stamp`,
`test_gcp_orchestrator_settings::test_claude_code_oauth_token_prefixed_env_override`. With `.env` present in
this worktree they should pass; if one fails it is pre-existing (env-dependent), not yours.

## Hard invariants (test-enforced)

- Opt-in / default-OFF is **byte-identical** to today. `autonomous=False` + `OPENRESEARCH_SPEC_VALIDATOR` unset
  ⇒ every existing path unchanged. Each backend task ships an OFF-state test.
- **Corpus isolation:** the 4 new SSE events + the session view carry ids/counts/enums/leaf-ids ONLY — never
  paper text. Route through `sse_bridge`; render only sanitized fields.
- **Secrets server-side:** Foundry key + cloud creds never reach the browser or a run-spec on the wire.
- **Auth before spend (D4):** the GPU-spending launch is gated by a confirm dialog + auth gate + the profile's
  `OPENRESEARCH_MAX_RUN_GPU_USD`.

## Do NOT touch

- The **WS-F external-monitor stream** (`backend/services/external_monitor/`, `backend/routes/external_runs.py`,
  `frontend/src/**/external-runs/`, the `app.py`/`config.py` poller wiring, `lab-sidebar.tsx` "GPU Monitor"
  nav) is a separate in-flight stream, UNCOMMITTED in the *primary* worktree only. It is file-disjoint from
  this feature. Leave it alone.

## Operator-owned (NOT code tasks in this plan)

- **GKE cluster production-readiness** (`infra/gcp/` + the IaC-gap audit) is the live-spend precondition for D2.
  Until ready, exercise the autonomous launch against the mocked/local SSE path (T14). A real GPU run needs the
  cluster.
- **Default-flip** (`autonomous` becomes every upload's default) is gated on the SDAR Phase-1 PASS + canary
  (§8). This plan ships `autonomous` default-OFF only.
- **Org/billing/compute console** (`_4` sidebar) = Phase 4, deferred. A minimal session shell suffices here.

## Start here

1. Open the plan. Confirm the dependency graph.
2. Dispatch Task 1 (backend §3.1 forwarding fix) to a fresh Sonnet implementer, TDD.
3. Review the diff (Opus). Then Task 2, … Backend T1-T8 first (or parallel with T9a/T9b), then frontend flow.
