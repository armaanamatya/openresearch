# Autonomous Upload → Live Reproduction UI — Design Spec

> **Status:** APPROVED (direction) 2026-07-05 — decisions locked below. Implementation is executed
> by a fresh session against this spec + a task-by-task plan (authored via `superpowers:writing-plans`
> from this spec, before coding).
>
> ## Decisions (locked 2026-07-05)
> - **D1 Scope = Core flow + aesthetic.** Phases 0-3 (upload → landing → repo-confirm → launch →
>   visible spec-validation → live reasoning view) + the alphaXiv design system. Auth **stubbed**
>   (single-user/dev). Org / billing / compute-console (the `_4` sidebar) **deferred** (Phase 4, later).
> - **D2 Live transport = Backend-orchestrated + GKE dispatch.** The root loop runs backend-side (SSE
>   reasoning streams via the existing lab, unchanged) and dispatches GPU cells to GKE (`--sandbox gke`).
>   Dependency: the GKE cluster must be production-ready (`infra/gcp/`).
> - **D3 Validator = funded grok / azure-foundry**, cross-family vs the Opus/Sonnet reproduction stack;
>   pluggable via `SPEC_VALIDATOR_BACKEND`/`_MODEL` (a real GPT key can be dropped in later).
> - **D4 Launch cost guard = ON (default):** a confirm dialog + a hard per-run `MAX_RUN_GPU_USD` from
>   the autonomous profile + an auth gate before the GPU-spending spawn.
> **Worktree:** `feat/autonomous-upload-ui` at `/home/abheekp/openresearch-autonomous-ui` (isolated
> from the in-flight SDAR foundation work on `reconcile/grounded-self-improvement-on-main`).
> **Author:** Opus (planning). Implementation to be executed by a fresh Sonnet-max session against
> this spec + the accompanying plan, Opus reviewing every diff.
> **Reference UX:** `alphaxiv_1..5.png` + `alphaxiv_autoresearc.png` in the repo root.

## 0. One-liner

Upload a PDF (or paste an arXiv id) in the web UI → OpenResearch resolves the paper + its author
repo → an Opus-4.8 root drives an autonomous reproduction on GCP → the LLM generates a reproduction
**spec** (rubric + procedure) that an **independent external validator agent** checks → the user is
**seamlessly redirected to a live agentic-reasoning view** streaming the real cloud run — all in the
alphaXiv-inspired maroon / neo-brutalist aesthetic of the reference screenshots.

This is a **product-assembly** spec: the reproduction *engine* (Opus-root + lifecycle driver +
execute-mode + evidence guards + checkpoint-resume) is the separately-built SDAR foundation. This
spec wires that engine to the UI and adds the missing product surfaces.

## 1. Goals / Non-goals

**Goals**
- A PDF upload (and arXiv id) in the UI that **actually triggers an autonomous cloud reproduction**
  (today the upload path silently drops the autonomous config and the advanced GPU fields — see §3.1).
- The 5-screen flow: paper landing (`_3`) → sign-in → repo-confirm + launch (`_4`) → **visible
  spec-generation + external-validation** phase → **live agentic-reasoning session view** (`_5`).
- A cohesive alphaXiv-inspired **design system** (maroon accent `#8B1E2D`-family, neo-brutalist hard
  drop-shadow cards, paper-centric layout).
- The external **spec validator** ("like GPT") surfaced as a gating, visible pre-run check.

**Non-goals (this spec)**
- The full multi-tenant **billing / compute-management console** (the `_4` sidebar's
  Compute/Instances/Storage/Billing) — scoped to a **minimal project/session shell** here; the full
  cloud-console is a later product track (§4.5).
- New reproduction-engine capability — the engine is the SDAR foundation (Opus root, driver, guards,
  resume), specced + built elsewhere. This spec consumes it, does not extend it.
- Changing the deterministic evidence/fitness signal (the red line: evidence, not the LLM grade).

## 2. The flow, screen by screen (mapped to the reference UX)

| # | Screen | URL (target) | Backend seam |
|---|---|---|---|
| A | **Paper landing** (`_3`) — paper card, "OpenResearch deploys an agent to build a minimal reproduction… → Sign in to start" | `openresearch.sh/abs/<arxiv_id>` | ingest/resolve paper metadata (title, id); no run yet |
| B | **Upload / arXiv entry** — drop a PDF or paste an id | (landing CTA) | `POST /runs/upload` (multipart) / `POST /runs/arxiv` (§3.1) |
| C | **Repo-confirm + launch** (`_4`) — "We found the code for this paper. Confirm the repository… `ZJU-REAL/SDAR` … 🚀 Start autoresearch" | `openresearch.sh/orgs/<org>/abs/<id>` | repo-first resolver (`OPENRESEARCH_USE_AUTHOR_REPO`) surfaces the resolved repo; the launch applies the **autonomous profile** (§3.2) |
| D | **Spec generation + external validation** (NEW, visible) — "Analyzing paper → Generating reproduction spec → Validating spec with `<validator>` → ✓ spec validated" | run starts; `rubric_gen` → `spec_validator` (§3.3) emit new SSE events |
| E | **Live agentic-reasoning session** (`_5`) — real root reasoning + tool-call chips streaming live; left rail = project + "New agent" + session list | `openresearch.sh/…/sessions/<run_id>` | existing SSE `/runs/<id>/events`, re-skinned (§3.4) |

The transition **D → E is the "seamless redirect"** the operator asked for: once the spec is
generated and the external validator returns (no blocking veto), the UI auto-navigates to the live
session view, which is already streaming the root's orientation reasoning.

## 3. Architecture — reuse map + the new seams

**Reuse (already exists; recon-verified this session):**
- **Upload path** — `POST /runs/upload`→`FileLiveRunService.start_uploaded_run`→`_start_python_run`
  →`cmd_reproduce` (`backend/services/events/live_runs.py`, `backend/cli.py:1573`). arXiv is symmetric.
- **Repo-first** — `OPENRESEARCH_USE_AUTHOR_REPO` + `RepoResolver`/`RepoProvisioner` resolve + clone
  the author repo into `runs/<id>/repo/`, persist `rlm_state/repo_spec.json` (the `_4` "we found the
  code" data).
- **SSE lab** — `/runs/<id>/events` streams `repl_iteration`/`primitive_call`/`rubric_score`/… from
  `dashboard_events.jsonl`, egress-sanitized by `sse_bridge.sanitize_iteration` (corpus never leaks).
  The RLM lab already renders this; `_5` is a **re-skin**, not a new pipeline.
- **Run-spec env-sink** — `cmd_reproduce` loads `--run-spec <json>` via
  `run_spec_contract.apply_run_spec` into the child `os.environ` before flag resolution (the
  canonical way to apply the autonomous profile).

### 3.1 Fix the upload path (real bug found)

The `/runs/upload` (`app.py:700-715`) and `/runs/arxiv` (`app.py:662-674`) constructors **do not
forward** `dynamic_gpu / force_single_gpu / gpu_parallelism / accelerator / max_gpu_usd_per_hour /
vram_gb / root_provider / subagent_auth` even though `StartRunRequest` supports them and the frontend
sends them → those "Advanced options" are **silent no-ops for every real paper today**. Fix: forward
them on both routes (both sides required).

### 3.2 The autonomous profile (the "use GCP + Opus root" switch)

Add `apply_autonomous_profile_override(request) -> StartRunRequest`, a sibling of
`apply_sandbox_override`/`apply_provider_override` (`live_runs.py:451-474`, called at 785-786). When a
new opt-in `autonomous: bool` field is set, `model_copy(update=...)` forces:
- `sandbox = "gke"` (see §3.5 — orchestrator backend-side, GPU cells to GKE, so the SSE reasoning
  streams directly);
- `model = "opus-foundry"` (root = Opus 4.8 via the Foundry Anthropic endpoint);
- `run_spec = <server-fixed canonical path>` → new `configs/autonomous_reproduction_run_spec.json`.

**Critical (recon-verified):** `model` + `sandbox` must be **explicit request overrides**, not run-spec
env keys — the uploaded-paper `Namespace` always passes an explicit `model` (default `sonnet`) that
wins over `OPENRESEARCH_RLM_ROOT_MODEL`, and `sandbox` is a direct field with no env indirection. Thread
`run_spec` into the `common` dict + the uploaded-paper `Namespace` (mirror `repo_url`).

`configs/autonomous_reproduction_run_spec.json` (paper-agnostic): `LIFECYCLE_PRIMARY=1`,
`CELL_RESUME_AUTO=1`, `USE_AUTHOR_REPO=1` + `REPRODUCTION_MODE=execute` (fail-soft to adapt/scratch when
no repo resolves), the evidence-guard set (`ZERO_METRICS_GUARD`, `EXTERNAL_VALIDATOR`, `EVIDENCE_GATE`,
`EVIDENCE_AUDIT`, `ENV_LIVENESS_GATE`, `NO_LEARNING_SIGNAL_GATE`, `EVAL_PROVENANCE_GUARD`,
`STUB_METRICS_GUARD`, `TWO_AXIS_VERDICT`, `SPEC_VALIDATOR`), `ROLE_MODELS={executor,grader,verifier:
sonnet-foundry}` — WITHOUT the SDAR-specific `REPO_LOCAL_PATH`/`REPO_COMMIT`/`HF_HOME` pins.

### 3.3 Spec generation + external validation (visible, gating)

The engine already generates the reproduction spec (rubric tree via `rubric_gen.generate_rubric_tree`;
`paper_claim_map`/`reproduction_contract` mid-loop). The **external spec validator** is the
`spec_validator.py` module designed this session (a sibling of `external_validator.py`): after the
rubric is generated and **before any primitive/GPU spend** (`run_pipeline_rlm`, right after rubric
resolution), a **separate-model** (cross-family — planner vs validator) adversarial agent machine-checks
each rubric leaf against the FULL paper text (`hallucinated_leaf` / `wrong_target` / `missing_key_claim`
/ `placeholder_leaf`). Advisory by default; opt-in `SPEC_VALIDATOR_BLOCK` drops machine-confirmed bad
leaves. See the sibling spec `.superpowers/sdd/productization-designs.md` §B for the full module design.

**New for the UI:** surface this phase live. Emit new SSE events at the chokepoints:
`spec_generation_started` → `spec_generated{leaf_count}` → `spec_validation_started{validator_model}` →
`spec_validated{verdict, flagged_leaves}` (or `spec_validation_flagged`). The frontend renders a compact
"Analyzing → Generating spec → Validating with `<model>` → ✓" stepper on screen D, then redirects to E.
All fail-soft; corpus-free (leaf ids + counts only, never paper text).

### 3.4 The live agentic-reasoning view (`_5`) — a re-skin, not a rebuild

`_5` is the existing SSE stream rendered as a clean **session log**: interleaved root reasoning
(`repl_iteration` sanitized text) + **tool-call chips** (`primitive_call` → "List project runs",
"Fetch paper report", "Check baseline run command", …) + rubric-climb + steering chat. Build a new
`SessionReasoningView` component (alphaXiv-styled) consuming the SAME `/api/demo/events` proxy the lab
uses; left rail = project name + "New agent" + session list (`localStorage`/leaderboard-backed). The
egress sanitizer already guarantees no corpus leaks — the view MUST render only sanitized SSE fields.

### 3.5 The live-reasoning transport (DECISION D2)

The rich reasoning reaches the browser only if the backend serving `/runs/<id>/events` can read the
run's `dashboard_events.jsonl`. Two models:

- **(Recommended) Backend-orchestrated + GKE Job dispatch.** The backend spawns the orchestrator
  (root loop) locally and it dispatches GPU cells to GKE (`--sandbox gke`, `GkeJobBackend` exists). The
  root reasoning streams from the backend's own `dashboard_events.jsonl` → the existing SSE lab works
  **unchanged**; only a re-skin (§3.4) is needed. Dependency: the GKE cluster must be production-ready
  (`infra/gcp/` + the IaC-gap audit — a real but bounded infra task).
- **(Interim) Single-VM + reasoning-tail.** Keep the proven single-VM path (orchestrator ON the GPU
  VM) and stream its `dashboard_events.jsonl` to the browser by extending the WS-F external monitor to
  tail the reasoning JSONL (not just parsed progress). Reuses current VM infra; more plumbing + latency.

**Recommendation:** ship the product on **backend-orchestrated + GKE**; use the single-VM path only for
the near-term manual SDAR proof runs (already in flight).

## 4. Component plan (phased)

**Phase 0 — Design system.** Tailwind tokens for the maroon palette + neo-brutalist card (hard offset
shadow, thick border), the paper-centric header (Paper/…/Autoresearch-style tabs), typography. A small
component kit (Card, PrimaryButton, RepoField, Stepper, ReasoningChip, SessionRail). Storybook/visual
snapshot optional.

**Phase 1 — Upload → landing → repo-confirm → launch.** §3.1 fix + §3.2 autonomous profile + the `_3`
landing + `_4` repo-confirm card + the `autonomous` opt-in (default OFF until the SDAR Phase-1 proof;
then flip default per §8). A **cost-guard confirmation** on launch (§5 D4).

**Phase 2 — Spec-gen + external-validation phase (D).** §3.3 — the `spec_validator` module (from the
sibling design) + the 4 new SSE events + the screen-D stepper. Gating = advisory by default (show the
verdict, allow proceed); blocking opt-in.

**Phase 3 — Live reasoning session view (E).** §3.4 re-skin + the `_5` session rail. This is the
centerpiece deliverable ("show real reasoning from the live run").

**Phase 4 — (future) org/compute/billing shell.** The `_4` sidebar's full console
(Compute/Instances/Storage/Billing/orgs). Deferred; a minimal project/session shell (Phase 3) suffices
for the core flow.

## 5. Decision points (need operator sign-off)

- **D1 — Product scope.** *Recommended:* core flow (Phases 0-3) + alphaXiv aesthetic, **auth stubbed**
  (single-user/dev), org/billing deferred. Alternatives: (b) + real auth + minimal org shell;
  (c) full SaaS (auth+orgs+billing+compute console).
- **D2 — Live-reasoning transport.** *Recommended:* backend-orchestrated + GKE dispatch (§3.5).
  Alternative: single-VM reasoning-tail (interim).
- **D3 — External validator model.** *Recommended:* funded **grok / azure-foundry** (works today,
  keyless-to-OpenAI, a genuine cross-family check vs the Opus/Sonnet reproduction stack). Alternative:
  a real OpenAI GPT key (currently dead per project memory) — pluggable via `SPEC_VALIDATOR_BACKEND`.
- **D4 — Launch cost guard.** A UI "🚀 Start autoresearch" click **spends GPU money**. *Recommended:*
  a confirm dialog showing the estimated cap + a hard per-run `MAX_RUN_GPU_USD` from the autonomous
  profile + an auth gate before spawn. (The `_4` screen already shows a balance/credits affordance.)

## 6. Security / privacy

- **Corpus isolation:** the reasoning view (§3.4) and the new spec-phase events (§3.3) MUST route only
  through the existing egress sanitizer / carry only ids+counts — never paper text. This is the
  system's hard invariant.
- **Secrets:** the Foundry key + cloud creds stay server-side (the `CredentialBroker` redaction layer);
  never sent to the browser or embedded in a run-spec on the wire.
- **Auth before spend:** even a stubbed auth must gate the GPU-spending launch (D4).
- **`configs/external_runs.json`-class configs:** any live-infra config stays gitignored (mirrors the
  WS-F fix).

## 7. Testing

- **Backend:** the upload-autonomous seam (autonomous ON → sandbox/model/run_spec forwarded; OFF →
  byte-identical); the new SSE spec-phase events (hermetic); the `spec_validator` gate (ON+OFF); the
  advanced-field-forwarding fix.
- **Frontend:** vitest for the components + the flow state machine; **Playwright e2e** for
  upload→landing→repo-confirm→(mocked run)→reasoning-view; visual snapshot of the design system.
  (Node: nvm v22.14.0; lab suite `--no-file-parallelism`.)
- **Contract:** the SSE payload shapes the reasoning view consumes must match the backend emitters
  (avoid the WS-F-class drift).

## 8. Rollout

Phase 0 → 1 → 2 → 3 behind an **`autonomous` opt-in toggle (default OFF)**. The **default-flip**
(every upload autonomously runs on GCP) is gated on the **SDAR Phase-1 PASS + the canary** — i.e. the
engine must be proven before the UI makes it the default (same discipline as `LIFECYCLE_PRIMARY`).
Phase 4 (billing/console) is a separate track.

## 9. Execution notes (for the implementing session)

- Work in the `feat/autonomous-upload-ui` worktree. **Rebase onto
  `reconcile/grounded-self-improvement-on-main` once the SDAR foundation (Anthropic-Foundry provider +
  lifecycle-primary + execute-mode + spec_validator) is committed** — this feature consumes those.
- Sonnet-max implementers, file-disjoint, TDD; Opus reviews every diff. Commit at phase milestones;
  push `deepinvent` only on request.
- Companion designs already written: `.superpowers/sdd/productization-designs.md` (§A UI-upload seam,
  §B spec-validator) and `.superpowers/sdd/wsf-external-monitor-fix-plan.md` (the live-log monitor).
- A task-by-task **implementation plan** should be authored (via `superpowers:writing-plans`) from this
  spec after approval, before coding.
