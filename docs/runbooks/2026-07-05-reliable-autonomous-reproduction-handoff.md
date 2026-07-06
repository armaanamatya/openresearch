# Reliable Autonomous Reproduction — SESSION HANDOFF (2026-07-05)

> **Purpose:** a fresh session (zero prior context) must **finish all SDAR fixes + the
> fully-autonomous / dynamic / any-paper reproduction changes, then run SDAR Phase-1 end-to-end.**
> Everything is already **designed + planned + committed** — this session investigated, verified the
> fix live, wrote the spec + the 16-task plan, and was about to execute. Read this, then the two
> committed artifacts it points at. **Do NOT re-derive; execute.**
>
> - **Spec** (approved): `docs/superpowers/specs/2026-07-05-reliable-autonomous-reproduction-design.md` — commit `a9cbb32b`
> - **Plan** (16 TDD tasks): `docs/superpowers/plans/2026-07-05-reliable-autonomous-reproduction-foundation.md` — commit `a96911e5`
> - **Ledger:** `.superpowers/sdd/progress.md` → section "Reliable Autonomous Reproduction — Foundation"
> - **Memory:** `reference-azure-foundry-anthropic-endpoint`, `project-reliable-autonomous-reproduction`, `project_lifecycle_driver`, `project_sdar_gcp_rl_smoke_fix`, `feedback_*`.

---

## 0. TL;DR + first action

**First action:** invoke `superpowers:subagent-driven-development` and execute the plan
(`docs/superpowers/plans/2026-07-05-reliable-autonomous-reproduction-foundation.md`) from **Task 1**,
base commit **`a96911e5`**. Sonnet implementers, file-disjoint, **synchronous**; **you (Opus) review
every diff inline** — never a Sonnet reviewer. The Task-1 brief is already generated at
`.superpowers/sdd/task-1-brief.md`.

**The one-line story:** both SDAR Phase-1 execute runs failed (root churned + execute mode was never
wired — *not* "proven" as the older handoff claims). The fix is a **reliable Opus-4.8 root + a
harness-owned deterministic driver** — both verified feasible live. Land the foundation (WS-A/B/C),
then run the ~$30 Phase-1 gate, then generalize to any paper (WS-D) and fold in SOTA (WS-E).

---

## 1. State — what is committed, what is next

| Item | Status |
|---|---|
| Investigation of the two failed SDAR runs | ✅ done (§2) |
| Live verification of the Opus/Sonnet Foundry Anthropic endpoint | ✅ done (both HTTP 200) |
| Approved design spec (`a9cbb32b`) | ✅ committed |
| 16-task foundation plan (`a96911e5`) | ✅ committed |
| **Foundation implementation (WS-A/B/C, Tasks 1-16)** | ⏳ **NOT STARTED — execute this** |
| **SDAR Phase-1 run (~$30, pre-authorized)** | ⏳ gated on the foundation landing |
| WS-D (any-paper generalization) / WS-E (SOTA E3/E5/E6/E7) / WS-F (external-runs monitor) | ⏳ follow-on plans, after Phase-1 |
| Coworker Foundry/GCP/Claude-Code onboarding + README | ⏳ separate, lower-priority (may already be committed — check `git log`) |

Branch: `reconcile/grounded-self-improvement-on-main`. **Push `deepinvent` only.**

---

## 2. The two bugs + the fix

Both SDAR execute-mode Phase-1 runs (`sdar_exec_phase1_1783279253`, `_1783280123`, in
`gs://deepinvent-ext-ut-sdar-runs/`) **failed at ~5 min / $0**, `verdict=failed`. Two **independent**
root causes (verified by tracing the report writer + repo resolver):

- **B1 — root-model unreliability (dominant).** The `azure-foundry`/grok root emitted REPL code that
  never called `implement_baseline`/`run_experiment`, then produced an unparseable 292-char
  `FINAL_VAR` → `report.py::_parse_response` fails → failed report at $0. The degenerate detectors did
  not fire (grok *was* emitting code, just useless code). No keyless paper-validated root existed
  (gpt-5/claude API keys dead, oauth degenerates, grok churns).
- **B2 — execute mode never wired (orthogonal config bug).** `reproduction.mode` is fixed at **setup
  time** in `rlm_state/repo_spec.json` by `run.py::_resolve_and_clone_repo` (line ~598), driven only
  by `OPENRESEARCH_REPRODUCTION_MODE` (default `adapt`). The report block exists (clone succeeded +
  `USE_AUTHOR_REPO` on) yet mode=`adapt` → `REPRODUCTION_MODE=execute` was never set in the driver's
  `.cache` run-spec (`gcp_sdar_preflight.sh:618` hardcodes `:-adapt`). `execution.ran=false` is the B1
  symptom; `mode=adapt` is the B2 symptom. **Fixing one does not fix the other.**

**The fix (Approach 3, approved — a reliable brain + a deterministic skeleton):**
1. **Route root = `claude-opus-4-8`, executor/grader/verifier = `claude-sonnet-5`** via the funded
   Azure Foundry **Anthropic** endpoint (§4) — a paper-capable, reliable orchestrator. Scoped
   **per-client** (root `get_client` patch + `AnthropicMessagesClient` `base_url` + executor
   `ClaudeAgentOptions.env`); **no global `ANTHROPIC_BASE_URL` leak** into `claude-oauth`; a hard
   co-residency guard.
2. **Promote `run_lifecycle_primary`** (`lifecycle_driver.py:418-574`, *already exists*, **bypasses
   the flaky root loop**) to the **default deterministic backbone** (plan→implement→run→verify→repair).
   Load-bearing change: harden `_synth_result_from_summary` (`run.py:1072`) to project an honest report.
3. **Fix B2** — make `REPRODUCTION_MODE=execute` authoritative + **fail-loud** when execute is
   requested but `adapt` is stamped.

**SOTA scan validated this pivot:** AIDE (MLE-bench's *best* scaffold) is a hard-coded controller;
PaperBench removed its submit tool because the free agent quit early (= B1); Opus-lead/Sonnet-worker is
Anthropic's own +90.2% pattern (spec §5.E).

---

## 3. Task inventory — execute IN ORDER

All in the plan (`docs/superpowers/plans/2026-07-05-reliable-autonomous-reproduction-foundation.md`),
each a TDD task with real code + tests. Every change **default-OFF / byte-identical** when its
flag/field is absent.

**A. Foundation (Tasks 1-16) — do first, gets to Phase-1:**
- **WS-A — Anthropic-Foundry provider (T1-7):** `foundry_anthropic.py` resolver → `AnthropicMessagesClient`
  `base_url` param → `grader_transport` `anthropic-foundry` branch → `models.py` `opus-foundry`/`sonnet-foundry`
  roots + `_anthropic_foundry_patch.py` (rlm `get_client` patch) → `role_models.py` tokens → `run.py`
  wiring + co-residency guard → live smoke script.
- **WS-B — deterministic driver default (T8-10):** harden `_synth_result_from_summary` (`run.py:1072`,
  load-bearing) → primary-path input guard → driver edge-case hardening (re-drive a repairable
  `implement_baseline`; persist a pre-committed ordered plan [SOTA E2]; honest `repair_exhausted`).
- **WS-C — execute-mode wiring (T11-14):** fail-loud `assert_execute_mode_stamped` → extend
  `configs/sdar_execute_run_spec.json` (root `opus-foundry`, roles `sonnet-foundry`,
  `LIFECYCLE_PRIMARY=1`, `REPRODUCTION_MODE=execute`, guards) → driver honors the run-spec mode (drop
  the hardcoded `adapt`) → Phase-1 kickoff wrapper.
- **T16 — CLAUDE.md rule block** (the Anthropic-Foundry provider; `LIFECYCLE_PRIMARY`; the execute-mode
  wiring rule).

**B. SDAR Phase-1 run (Task 15, ~$30 PRE-AUTHORIZED) — the proof.** See §7.

**C. Follow-ons (own spec/plan each, after Phase-1 is green):**
- **WS-D — fully autonomous / dynamic / any-paper generalization** (the north star, §8).
- **WS-E — SOTA adoptions** E3 (blind-typed grading) / E5 (multi-seed repro-lock) / E6 (action-stream
  monitor) / E7 (JudgeEval).
- **WS-F — finish + commit the external-runs monitor** (its pytest currently **HANGS** on 2 broken SSE
  tests; poll-loop untested; a backend↔frontend contract drift; gate the always-on poller behind
  `OPENRESEARCH_EXTERNAL_MONITOR` + gitignore the live-infra `configs/external_runs.json`). Commit
  **isolated** from the foundation.

**D. Coworker onboarding + README** (lower priority; privacy-aware — team Foundry key for Azure root +
GCP + Claude Code; `.env` stays gitignored, key shared out-of-band). May already be committed this
session — check `git log`.

---

## 4. Verified facts + key coordinates (do NOT re-derive)

- **Foundry Anthropic endpoint (VERIFIED live 2026-07-05):**
  `https://appradhann-4738-resource.services.ai.azure.com/anthropic/v1/messages`, auth
  `x-api-key: $AZURE_FOUNDRY_API_KEY` (already in `.env`), `anthropic-version: 2023-06-01`. Models
  `claude-opus-4-8` + `claude-sonnet-5` (only these two; `opus-4-1` → 404). **Same resource + key** as
  the existing OpenAI-compat `azure-foundry`/grok endpoint (`.../openai/v1`) — but that one is
  DISTINCT (OpenAI SDK). No new Settings fields needed (`azure_foundry_api_key`/`_endpoint` exist at
  `config.py:202/195`).
- **rlm `AnthropicClient`** (`.venv/.../rlm/clients/anthropic.py:23-24`) + **`AnthropicMessagesClient`**
  (`backend/services/context/workspace/tools/anthropic_messages_client.py:100`) both build
  `anthropic.Anthropic(api_key=…)` with **no** `base_url` → per-client scoping needed (patch / new
  param). The `claude` CLI (executor via `claude_runtime.py`) honors `ANTHROPIC_BASE_URL` and
  `ClaudeAgentOptions.env` (`claude_agent_sdk/types.py:607/1721`) — set it **per-subprocess**.
- **Driver:** `run_lifecycle_primary` `lifecycle_driver.py:418-574`; gate `_lifecycle_primary_enabled`
  `run.py:1052`; `_synth_result_from_summary` `run.py:1072`; primary branch `run.py:3862-3882`.
- **Mode:** set at `run.py:598` (`_resolve_and_clone_repo`) → `repo_spec.json`; report block
  `report.py:1456` (`_build_reproduction_block`); driver hardcode `gcp_sdar_preflight.sh:618`.
- **GCP VM:** `sdar-2model-a`, zone `us-central1-a`, project `deepinvent-ext-ut`, `a2-ultragpu-4g`
  (4×A100-80GB), **STOPPED** (restart to use). `export CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud`.
  Cache disk `sdar-cache-a` → `/mnt/sdar-cache` (persists: conda envs `sdar`/`retriever`, SDAR repo
  pinned, HF weights, the 64 GB Search index). GCS `gs://deepinvent-ext-ut-sdar-runs/`. Restart:
  `gcloud compute instances start sdar-2model-a --zone us-central1-a --project deepinvent-ext-ut`.
  Stop: `… stop … --discard-local-ssd=false`.
- **Only proven signal:** the authors' verl Search-QA-3B = `val/success_rate` **0.456** @ 150 steps.

---

## 5. Working discipline (all prompt instructions, condensed)

- **Roles:** Opus authors/plans + **reviews EVERY diff** (verify the diff, not the summary); **Sonnet
  executes** (impl code included) against a tight, file-disjoint spec, dispatched **SYNCHRONOUSLY**
  (background subagents were lost on a mid-turn process exit). **Never a Sonnet reviewer** — Opus
  reviews inline (Codex if available). Delegate to Opus/Sonnet; Fable only for analysis. `/implement`
  for implementation (never `/implement_codex` or `codex:*`).
- **Quality:** every change **default-OFF / byte-identical** when its flag/field is absent; hermetic
  ON+OFF tests; **TDD** (failing test first). Root-level elegant solutions (one canonical abstraction
  + a guard test), not scattered patches.
- **Verify:** `.venv/bin/python -m pytest <path> -v`; `uvx ruff@0.15.16 check <files>` (Python 3.12 via
  `uv sync --frozen`). **Baseline known-failing (exit bar = no NEW failures):** `test_accelerator`,
  `test_external_validator`, `test_report_validation_stamp`,
  `test_gcp_orchestrator_settings::test_claude_code_oauth_token_prefixed_env_override`.
- **Git:** commit **infrequently at milestones**; descriptive present-tense headlines
  (what+symptom+resolution); **no Conventional-Commit prefixes; no Co-Authored-By / AI trailer**;
  identity `lolout1` / `appradhann@gmail.com`
  (`git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit`); **push `deepinvent` only**
  (`git@github.com:Deepinvent/scientific_article_generator.git`) — never origin/openresearch/replix.
  Isolate foundation commits from the uncommitted **external-monitor** feature (`git add` exact paths).
- **Money/irreversible:** checkpoint the operator before any GPU spend **except** the pre-authorized
  ~$30 Phase-1 slice; **autostop ON** always; never leave a 4×A100 idle.

---

## 6. How to resume the plan (Subagent-Driven Development)

1. `cat .superpowers/sdd/progress.md` → the "Reliable Autonomous Reproduction — Foundation" section is
   the checklist; tasks marked `[x]` are DONE (do not re-dispatch). Base commit `a96911e5`.
2. Per task: `scripts/task-brief docs/superpowers/plans/2026-07-05-reliable-autonomous-reproduction-foundation.md N`
   (helper at the subagent-driven-development skill dir) → dispatch a **Sonnet** implementer with the
   brief path + report path + the interfaces from earlier tasks. Implementer does TDD + commits.
3. **Review each diff inline (Opus):** `scripts/review-package <BASE> <HEAD>` → read the printed file →
   adjudicate; dispatch a fix subagent for Critical/Important. Append `Task N: complete …` to the ledger.
4. After Task 16: a broad whole-branch review (Opus), then `superpowers:finishing-a-development-branch`.
5. The foundation files (`models.py`, `role_models.py`, `run.py`, `grader_transport.py`,
   `lifecycle_driver.py`, `anthropic_messages_client.py`, `claude_runtime.py`, `configs/`, `scripts/`)
   are all **clean** (disjoint from the uncommitted external-monitor tree).

---

## 7. SDAR Phase-1 validation (Task 15 — ~$30, PRE-AUTHORIZED)

- **Pre-run (no spend):** confirm `OPENRESEARCH_REPO_COMMIT` in the run-spec matches the VM's STEP-2
  SHA; reconcile the verl `eval_provenance.json` schema edge with `EVAL_PROVENANCE_GUARD` (spec §6);
  run the OFF-state regression (no new failures).
- **Run:** restart `sdar-2model-a`; `scripts/foundry_anthropic_smoke.py` (opus + sonnet OK); launch
  `configs/sdar_execute_cells_phase1.json` (Search-3B) with the run-spec + `--model opus-foundry`,
  **autostop ON** → GCS.
- **PASS gate (all of):** harness-driven `val/success_rate` **≥ 0.40** (target 0.456); guards clean
  (zero-metrics / eval-provenance / env-liveness / no-learning); external-validator no-veto;
  `code/metrics.json` real value + `eval_provenance.json` (`provenance_kind:"aggregate"`);
  `final_report.reproduction.mode=="execute"` AND `execution.ran==true`; verdict ≠ `failed`.
- **On PASS** → checkpoint the operator for the **$400 grid** (stays checkpointed). **On MISS** → debug
  on the ~$30 evidence; do NOT spend on the grid.

---

## 8. Fully autonomous, dynamic, across-papers (the north star — WS-D)

The foundation is **paper-agnostic by construction** — the "any paper" work is generalization, not a
rewrite:
- The **deterministic driver** carries no paper-specific logic; it drives
  understand→plan→implement→run→verify→repair for any paper, and **circumvents edge cases** (repairable
  re-drives, partial-evidence rescue, evidence-fingerprint repair loop, honest `repair_exhausted`).
- The **Anthropic-Foundry provider** + role routing are paper-agnostic.
- **Execute mode + the operator manifest seam** (`OPENRESEARCH_CELLS_SEED_PATH`) + local-repo pin work
  for **any** paper whose repo ships a runnable pipeline (declare the grid once; the harness guarantees
  it behind a launcher + a value-preserving metrics adapter). `metrics_source.kind` (currently `verl`)
  is an extensible per-framework adapter registry — a new framework is a new adapter, not a harness
  edit. A paper with **no** author repo drives the from-scratch backbone (adapt/scratch) unchanged.
- The **evidence guards** (zero-metrics / eval-provenance / env-liveness / no-learning / evidence-gate)
  are the deterministic, paper-agnostic fitness signal (the red line: evidence, not grade).
- `--paper-hint` remains the optional per-paper invariants surface.

WS-D makes this explicit + tested (adapter dispatch data-driven; no-repo paper drives from-scratch;
any-paper end-to-end smoke). Its own spec/plan comes after Phase-1 proves the mechanism on SDAR.

---

## 9. Reconstitution (if the VM is stopped/rebuilt)

The staged `sdar-cache-a` disk (`/mnt/sdar-cache`, `--discard-local-ssd=false` preserves it) holds
conda + envs + the SDAR repo + the index + weights. The VM's `/home/abheekp/openresearch` is a non-git
rsync copy — sync changed `backend/…`/`configs/…`/`scripts/…` by `tar` + `gcloud compute scp` (a full
tar times out; sync only changed files). STEP-2 verl edits live in the VM's `/mnt/sdar-cache/SDAR` git
repo (committed, so the pin seeds them). git identity on that repo: `git config user.email
abheek@deepinvent.ai; user.name lolout1`.
