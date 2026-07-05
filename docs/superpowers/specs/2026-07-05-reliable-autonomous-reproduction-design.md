# Reliable Autonomous Paper Reproduction — Opus-driven, harness-owned — DESIGN

> **Status:** DRAFT (brainstorming → spec). Author: Opus (design + review). Date: 2026-07-05.
> Supersedes the optimistic "chain proven" claim in
> [`2026-07-05-sdar-execute-autonomous-run-and-services-handoff.md`](../../runbooks/2026-07-05-sdar-execute-autonomous-run-and-services-handoff.md).
> Companion evidence: [`2026-07-04-sdar-gcp-runs-log-analysis.md`](../../audits/2026-07-04-sdar-gcp-runs-log-analysis.md).
> Memory: [[reference-azure-foundry-anthropic-endpoint]], [[project_sdar_execute_mode_reproduction]],
> [[project_lifecycle_driver]], [[project_sdar_gcp_rl_smoke_fix]].

## 1. Summary

Make the paper-reproduction harness **reliably autonomous on any paper** by removing the flaky LLM
"root" from the orchestration driver's seat. Two complementary reliability levers, layered
(the approved **Approach 3**):

1. **A reliable reasoning brain.** Route the RLM root/driver to **Claude Opus 4.8** and the
   executor/grader/verifier to **Claude Sonnet 5**, both served by the funded Azure Foundry
   **Anthropic-compatible** endpoint (verified live 2026-07-05). This replaces the grok/foundry
   and oauth roots that churn / degenerate / emit unparseable output.
2. **A harness-owned deterministic backbone.** Promote the existing `run_lifecycle_primary`
   (`backend/agents/rlm/lifecycle_driver.py`) to the **default** reproduction control flow — a
   state machine that owns `understand → plan → implement → run → verify → repair → finalize` and
   calls the LLM only for bounded sub-tasks. Completion no longer depends on the model *choosing*
   to call the primitives.

The concrete proof is a **passing SDAR Search-QA-3B Phase-1 run** (harness-driven
`val/success_rate` ≥ 0.40, target 0.456, evidence guards clean), pre-authorized at ~$30 GPU. The
same mechanisms generalize to any paper; the spec also folds in SOTA best-practice patterns, makes
the driver + seams fully paper-agnostic, and finishes the **external-runs monitor** as the
"inspect runs / steer the agent" surface (matching the alphaXiv Autoresearch loop).

All changes ship **default-OFF / byte-identical when their flag/field is absent**, TDD, hermetic
ON+OFF tests, `ruff` clean — the standing discipline.

## 2. Problem statement + evidence

Both SDAR execute-mode Phase-1 runs (2026-07-05) **failed identically**, and *not* for the reason
the session-2 handoff implies (it was written mid-run and never reconciled):

| Run | Verdict | Iters | Wall | Cost | `reproduction.mode` | `execution.ran` | Signature |
|---|---|---|---|---|---|---|---|
| `sdar_exec_phase1_1783279253` | failed | 21 | ~5 min | $0 | adapt | false | root churned |
| `sdar_exec_phase1_1783280123` | failed | 21 | ~4 min | $0 | adapt | false | `could not parse RLM response (len=292)` |

**Two independent, confirmed root causes** (verified by tracing the report writer + repo resolver):

- **B1 — Root-model unreliability (the dominant blocker).** The `azure-foundry`/grok root emits REPL
  code blocks that never call `implement_baseline`/`run_experiment`, then produces an unparseable
  292-char `FINAL_VAR` → `report.py::_parse_response` fails → "failed" report at $0. The two
  degenerate detectors did **not** fire: the churning-root detector
  (`run.py::_register_iteration_progress`) resets on *any* code block (grok emitted code, just
  useless code), and the FINAL_VAR-refusal detector never saw a refusal because the loop ended by
  the root "finalizing" with garbage. This is the twice-confirmed pattern from
  [[project_sdar_gcp_rl_smoke_fix]]: reproduction is gated by **root reliability**, and no keyless
  paper-validated root existed (gpt-5/claude API keys dead, oauth degenerates, grok churns).
- **B2 — Execute mode was never wired (an orthogonal config-default bug).** `reproduction.mode` is
  fixed at **setup time** in `rlm_state/repo_spec.json` by `run.py::_resolve_and_clone_repo`
  (line ~598), driven **only** by `OPENRESEARCH_REPRODUCTION_MODE` (default `"adapt"`;
  `provisioner`/`resolver.py:71-72` collapses anything not `execute`/`reference` to `adapt`). The
  report block *exists* (so the clone succeeded and `USE_AUTHOR_REPO` was on) yet mode is `"adapt"`
  → **`REPRODUCTION_MODE=execute` was simply never set** in the `.cache` run-spec the driver built
  (the preflight's hardcoded `${OPENRESEARCH_REPRODUCTION_MODE:-adapt}` at
  `gcp_sdar_preflight.sh:618`). The repo config `configs/sdar_execute_run_spec.json` *does* set it,
  but the driver ignored that file. `execution.ran=false` is the B1 symptom (no successful
  `run_experiment`); `mode="adapt"` is the B2 wiring symptom. **Fixing one does not fix the other.**

The execute-mode seams built the prior session (command cells, verl metrics adapter, GPU partition,
cells-seed authority) are **built + unit-green but were never exercised end-to-end**, because the
root never drove them to a scored experiment.

## 3. Goals / non-goals

**Goals**
- G1. A reliable root/driver: Opus 4.8 reasoning + a harness-owned deterministic backbone that
  completes the reproduction lifecycle regardless of any single model's momentary behavior.
- G2. Fix B2: execute mode (and its cells-route/GPU-partition chain) actually stamps + engages.
- G3. A **passing SDAR Phase-1** run: harness-driven `val/success_rate` ≥ 0.40 (target 0.456),
  evidence guards clean, external-validator no-veto — the concrete proof.
- G4. Paper-agnostic: the driver + provider + seams work for an arbitrary paper, not SDAR-special.
- G5. Fold in SOTA orchestration/reliability best practices (2026).
- G6. Finish + commit the external-runs monitor as the inspect/steer surface.
- G7. Document the resulting rules in `CLAUDE.md`.

**Non-goals**
- The full SDAR grid ($400) — stays a checkpointed follow-on after Phase-1 passes.
- Replacing the `rlms` library or the 18-primitive surface.
- Making WebShop actually serve (separate server-provisioning task); the env-liveness gate keeps a
  dead env *honest*, which is sufficient here.
- Autonomous harness self-edit (Phase C) — out of scope.

## 4. Approach (approved: Approach 3, layered)

Opus 4.8 is a top-tier reasoning model → a reliable orchestrator that emits structured reports.
The deterministic backbone is the guarantee-of-completion backstop that also owns the repair loop.
Together: a capable brain **and** a deterministic skeleton = reliable on any paper/root. A flaky
moment or a novel paper degrades to "the harness drives it anyway," not "21 iterations of garbage."

**Model routing (approved):** root/driver = `claude-opus-4-8`; executor (`implement_baseline`) +
grader + verifier = `claude-sonnet-5`. Both via the Foundry Anthropic endpoint.

## 5. Architecture — workstreams

Each workstream is default-OFF / byte-identical when its flag/field is unset, with hermetic ON+OFF
tests. Ordered by dependency; WS-A + WS-B + WS-C are the reliability foundation that gates Phase-1.

### WS-A — Anthropic-Foundry provider (Opus root / Sonnet exec+grader)

**Verified facts (2026-07-05):** the Azure resource `appradhann-4738-resource` serves
`claude-opus-4-8` and `claude-sonnet-5` on `https://appradhann-4738-resource.services.ai.azure.com/anthropic/v1/messages`,
auth `x-api-key: $AZURE_FOUNDRY_API_KEY` (the key already in `.env`), `anthropic-version: 2023-06-01`.
Both return HTTP 200. The `rlm` lib's `AnthropicClient` builds `anthropic.Anthropic(api_key=…)`
**without** `base_url`, but the Anthropic SDK **and** the `claude` CLI (claude-agent-sdk) natively
honor the `ANTHROPIC_BASE_URL` env var.

**Design — one canonical resolver + strictly per-client scoping (mirrors `foundry_endpoint.py`; no
new Settings fields — `azure_foundry_api_key`/`_endpoint`/`_deployment` at `config.py:195/202/209`
already back both resolvers):**

1. **`backend/agents/runtime/foundry_anthropic.py`** — `resolve_foundry_anthropic_credentials()`
   returning `(base_url=…/anthropic/v1, api_key, {opus: "claude-opus-4-8", sonnet: "claude-sonnet-5"})`,
   read from `AZURE_FOUNDRY_ANTHROPIC_ENDPOINT` (default: derive `.../anthropic/v1` from the host of
   `AZURE_FOUNDRY_ENDPOINT`) + `AZURE_FOUNDRY_API_KEY` (os.environ → Settings, mirroring the existing
   OpenAI-compat `foundry_endpoint._env_or_settings`). Model names overridable via
   `AZURE_FOUNDRY_ANTHROPIC_OPUS`/`_SONNET`.
2. **Root registry (`models.py`):** new entries `opus-foundry` (→ `claude-opus-4-8`) and
   `sonnet-foundry` (→ `claude-sonnet-5`), `rlm_backend="anthropic"`, `paper_validated=True` (Opus 4.8
   is paper-capable), `api_key_env` sourced from the resolver. Aliases `opus4.8`/`opus-4-8` → the
   opus entry. `resolve_root_model` selection unchanged.
3. **Per-client `base_url` — NOT a global env shim (the crux; the OAuth-leak fix).** The mapper
   confirmed that `ANTHROPIC_BASE_URL` in `os.environ` is **process-global** and would silently
   redirect *any* `claude-oauth` Claude call in the same process to Foundry. So we scope per client,
   never globally:
   - **Root** (`rlm` `AnthropicClient`, no `base_url` kwarg): register a new `anthropic-foundry`
     `get_client` monkeypatch **alongside** `apply_oauth_backend_patch`/`apply_anthropic_caching_patch`
     (`run.py:123-124`) that constructs `anthropic.Anthropic(api_key=$AZURE_FOUNDRY_API_KEY,
     base_url=…/anthropic/v1)` for the `anthropic-foundry` backend literal. No global env. (Include it
     in the caching-patch wrap so it isn't bypassed.)
   - **Grader + verifier** (`AnthropicMessagesClient`, no `base_url` kwarg today): add an explicit
     `base_url` param to `AnthropicMessagesClient.__init__` and pass it from a new
     `build_transport_client("anthropic-foundry", …)` branch. No global env.
   - **Executor** (`ClaudeAgentRuntime` → `claude` CLI subprocess): confirmed clean —
     `ClaudeAgentOptions` exposes an **`env: dict[str,str]`** field (`claude_agent_sdk/types.py:607,1721`)
     that the subprocess transport merges into the CLI's environment
     (`_internal/transport/subprocess_cli.py:430-459`). Pass `{"ANTHROPIC_BASE_URL": …,
     "ANTHROPIC_API_KEY": …}` there, in the executor's options only — **per-subprocess, no
     process-global mutation.** So all three tiers are scoped per-client/per-subprocess.
   - **Co-residency guard (hard, tested invariant):** when any tier resolves to `anthropic-foundry`,
     assert no tier resolves to `claude-oauth`, and vice-versa — categorically prevents the leak.
   Unset provider ⇒ no patch registered, no env mutation ⇒ byte-identical.
4. **Role-model routing (`role_models.py`):** tokens `opus-foundry`/`sonnet-foundry` (or reuse
   `opus`/`sonnet` de-collapsed when the Anthropic-Foundry provider is active) route
   executor/verifier/grader to Sonnet 5 via the same endpoint. Executor uses the existing
   `anthropic` runtime; verifier/grader use `grader_transport` (below).
5. **Grader/validator transport (`grader_transport.py`):** an `anthropic-foundry` backend beside the
   existing `azure-foundry` (OpenAI-compat) one — builds an `AnthropicMessagesClient` pointed at the
   resolver's base_url + key + `claude-sonnet-5`. `complete_samples` (median-of-N) already supported.
6. **Precise integration coordinates:** see §5.A-coords (filled from the reliability-foundation
   mapper) for the exact file:line touch points and the confirmed non-leak guard.

**Auth-header note:** the endpoint expects Anthropic's `x-api-key` (verified). No `api-version`
query param is needed on the `/anthropic/v1` path (verified). The `claude` CLI executor honors
`ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY`; the SDK isolation invariant (`setting_sources=[]`,
explicit `mcp_servers`, non-plan `permission_mode`) is preserved.

### WS-B — Deterministic driver as the default reproduction backbone

`run_lifecycle_primary` (`lifecycle_driver.py:418-574`) already implements the full proactive
backbone: `drive_lifecycle_chain(need_baseline)` runs the fixed 6-step spine (understand → detect →
plan → implement → run → [repair loop] → verify), then a bounded improvement climb
(`propose_improvements` → implement → re-drive → best-of-climb). It **bypasses `rlm.completion`
entirely** (`run.py:3862-3882`) — which is exactly why it is immune to the B1 churn (there is no
root loop to churn). Today it is gated default-OFF behind `OPENRESEARCH_LIFECYCLE_PRIMARY`.

**Changes to make it the reliable default (bounded, ~4 items):**
1. **Gate default flip** at `run.py::_lifecycle_primary_enabled()` (lines 1052-1063): default-ON for
   reproduction runs with an opt-OUT escape hatch (`OPENRESEARCH_LIFECYCLE_PRIMARY=0`). *Kept behind
   a run-spec toggle for the SDAR run first; the global default flip is a separate step gated on the
   ≥3 paired-A/B + grader-σ rule, so unset stays byte-identical until validated.*
2. **Harden `_synth_result_from_summary`** (`run.py:1072-1121`) — the load-bearing fix. Today it
   returns `None` when `rubric_score is None`, forcing `run_failed=True` and a "failed" shell. As the
   default path it must project a real report from the driver summary (verdict/scope/baseline_metrics
   from the driven `verify_result` + the evidence bundle), so a completed-but-unscored backbone still
   ships an honest report, and a genuinely-empty run still fails honestly.
3. **Guarantee the driver's inputs are always populated** for non-degenerate runs: `custom_tools`
   must be the **wrapped** dict (as at `run.py:3635`), and `paper_text`/`rubric_spec` sourced from
   `context_dict` — confirm on the primary path, not only the reactive one.
4. **Reconcile the forced-iteration interceptor:** it is entered only in
   `_run_completion_on_worker` (which primary skips), so under primary the FINAL_VAR refusal
   machinery is inert by construction. Document this; ensure the evidence-gate + honesty guards still
   run on the driver's finalize path (they do — finalize is shared).

**Driver hardening for edge cases (the "circumvent any edge case" ask):** the driver already handles
repairable-vs-fatal, partial-evidence rescue (grade a 3-of-6-cell partial grid instead of `None`),
and wall-clock budget. Add: (a) a bounded re-drive when `implement_baseline` itself returns a
repairable contract-guard failure (not only `run_experiment`); (b) fold the fabrication/evidence
guards' veto into the repair trigger keyed on the **evidence fingerprint changing** (reuse
`OPENRESEARCH_REPAIR_MAX_ITERATIONS`), so a veto drives a fix-first loop, not a false success; (c)
honest terminal `repair_exhausted` when the fingerprint stops changing; (d) **SOTA E1 —
reason-free-form-then-forced-action:** the driver already calls each primitive directly (no free-form
root step to mis-emit), which *is* the forced-action pattern; extend it so any residual root/LLM
sub-call that must return a decision does so via a tool-forced/constrained emit, never a parseable-or-
not blob; (e) **SOTA E2 — pre-committed ordered plan:** persist `plan_reproduction`'s output as an
ordered step list in `rlm_state/` that the driver dispatches in order (the executor cannot reorder
it), turning "what next?" into deterministic dispatch. These make the B1 unparseable-churn class
*structurally impossible* on the default path.

### WS-C — Execute-mode + driver wiring fix (B2)

The `mode="adapt"` bug is a wiring default, fixed by making the operator's intent authoritative:
1. **Ship an authoritative run-spec** that sets `OPENRESEARCH_REPRODUCTION_MODE=execute` +
   `OPENRESEARCH_USE_AUTHOR_REPO=1` + the repo local-path/commit pin + the guards + the
   Anthropic-Foundry root/role selection + `OPENRESEARCH_LIFECYCLE_PRIMARY=1`. This is
   `configs/sdar_execute_run_spec.json` extended (WS-A/WS-B keys) and **actually passed** to the run.
2. **Close the driver default-override:** `gcp_sdar_preflight.sh:618` hardcodes
   `${OPENRESEARCH_REPRODUCTION_MODE:-adapt}` and builds its own `.cache` spec. Fix the driver to
   honor a passed `--run-spec` verbatim (respect the override / drop the hardcoded `adapt`), OR route
   the run through `--run-spec configs/sdar_execute_run_spec.json` end-to-end so the mode is set. The
   run must **verify** at startup that `repo_spec.json.mode=="execute"` (fail-loud if not).
3. **No new engagement logic needed:** the mapper confirmed the cells-route/GPU-partition chain
   engages once (a) mode is `execute`, (b) `implement_baseline` runs once to seed repo→`code/`, and
   (c) `code/cells.json` is present with all-command cells (the seeded phase-1 manifest). Under WS-B
   the driver *guarantees* `implement_baseline` runs — closing the B1 gap that left the chain
   unexercised.

### WS-D — Paper-agnostic generalization

The foundation is already generic; this workstream makes "any paper" explicit and removes
SDAR-special assumptions:
- The **operator manifest seam** (`OPENRESEARCH_CELLS_SEED_PATH`) + execute mode + local-repo pin are
  paper-agnostic: for any paper whose repo ships a runnable pipeline, the operator declares the grid
  once and the harness guarantees it behind a launcher, with a value-preserving metrics adapter. The
  `metrics_source.kind` adapter set (currently `verl`) is extensible per framework.
- The **driver + provider** carry no paper-specific logic; `--paper-hint` remains the optional
  per-paper invariants surface. For a paper with **no** author repo, execute mode is skipped and the
  driver drives the from-scratch backbone (adapt/scratch) unchanged.
- Add a small **framework-adapter registry** note: `metrics_source.kind ∈ {verl, …}`; a new framework
  is a new adapter, not a harness edit (mirrors the env-adapter seam). Concrete new adapters are
  out-of-scope here (added per-paper), but the extension point is specified.

### WS-E — SOTA best-practice fold-in

The web-research scan (PaperBench, AIDE/MLE-bench, AI Scientist v2, PaperCoder, SWE-agent/OpenHands,
LangGraph, Anthropic's *Building Effective Agents*) is **complete** — see **§5.E** for the sourced
findings, the "already strong / keep" list, the validations of this spec's choices, and the **7 new
adoptions (E1–E7)** with their landing points. Headline: the scan **validates the pivot** (deterministic
controller + bounded LLM operators; Opus-planner + Sonnet-worker) rather than revealing a gap. E1
(reason-then-forced-action) and E2 (pre-committed ordered plan) fold into **WS-B**; E3/E5/E6/E7 are
default-OFF follow-ons sequenced **after** SDAR Phase-1 is green so they never delay the concrete proof.

### WS-F — External-runs monitor (inspect/steer surface)

*(Finish-checklist from the external-monitor assessment; see §5.F.)* The uncommitted subsystem
(`backend/services/external_monitor/`, `backend/routes/external_runs.py`, `frontend/.../external-runs/`,
`app.py` lifespan wiring) SSH-polls hand-launched remote VM runs and republishes their progress as SSE
+ persisted JSONL — the "inspect runs, watch live logs, steer" surface matching the alphaXiv
Autoresearch loop. The assessment found it well-designed, modular, and **already fail-soft on boot**,
but **not commit-ready**: the pytest command **hangs** (2 broken SSE tests), the poll-loop state
machine is untested, there's a backend↔frontend contract drift, and the committed config points at
live infra with an always-on poller. Close the §5.F blocking fixes + add an enable gate + hardening,
then commit **isolated** from the harness work (do not sweep it into a foundation commit).

## 6. SDAR Phase-1 validation (the ~$30 gate — pre-authorized)

**Pre-authorized:** restart `sdar-2model-a` (us-central1-a, 4×A100-80GB) and run the Phase-1
Search-QA-3B slice (~$30, autostop ON) without a further checkpoint. The **$400 full grid stays
checkpointed.**

**Run:** `configs/sdar_execute_cells_phase1.json` (one cell: `search_qa_3b`, retrieval service on 1
GPU + training on 3, verl `val/success_rate` adapter) with the WS-A/B/C run-spec (Opus root, Sonnet
exec+grader, `LIFECYCLE_PRIMARY=1`, `REPRODUCTION_MODE=execute`, guards ON). Autostop uploads
`runs/<pid>/` to GCS and self-stops.

**PASS gate:** harness-driven `val/success_rate` **≥ 0.40** (target 0.456) **AND** the deterministic
guards clean (`ZERO_METRICS_GUARD` / `EVAL_PROVENANCE_GUARD` / `ENV_LIVENESS_GATE` /
`NO_LEARNING_SIGNAL_GATE`) **AND** `code/metrics.json` shows a real measured value with an
`eval_provenance.json` (`provenance_kind:"aggregate"`) sidecar **AND** external-validator no-veto
**AND** `final_report.reproduction.mode=="execute"` with `execution.ran==true`.

**On miss:** the seams/driver are debugged on the ~$30 evidence before any grid spend. Reconcile the
one known open edge from the prior handoff: the verl adapter's `eval_provenance.json` has a simpler
(aggregate-only) schema than `eval_provenance.py` expects with `EVAL_PROVENANCE_GUARD` on — either
exempt verl-sourced cells or write a `records`-shaped sidecar (decide before the run).

## 7. Testing strategy

- **WS-A:** hermetic unit tests for `resolve_foundry_anthropic_credentials` (env→settings, URL
  derivation), the scoped env shim (ON sets `ANTHROPIC_BASE_URL`; OAuth path does NOT; unset =
  byte-identical), the registry entries, and a **live smoke** `scripts/foundry_anthropic_smoke.py`
  (opus + sonnet ping, opt-in, not in CI).
- **WS-B:** extend `tests/rlm/test_run_lifecycle_primary.py` — primary-as-default routes to the
  backbone; `_synth_result_from_summary` projects a real report from a scored summary AND fails
  honestly on an empty one; the wrapped-tools/paper_text/rubric_spec inputs are asserted present;
  OFF (opt-out) = byte-identical.
- **WS-C:** a test asserting `REPRODUCTION_MODE=execute` in the shipped run-spec round-trips to
  `repo_spec.json.mode=="execute"`; a startup fail-loud when execute was requested but mode≠execute.
- **WS-D:** `metrics_source.kind` adapter dispatch is data-driven + tested; a no-repo paper drives
  the from-scratch backbone (existing suites).
- **WS-E:** each adopted pattern ships its own hermetic ON+OFF test.
- **WS-F:** the existing `tests/routes/test_external_runs_http.py` +
  `tests/services/external_monitor/` must pass; add redaction + fail-soft-boot + parse-edge tests as
  the assessment dictates.
- Keep the cells-route / execute-mode / role-model / forced-iteration / report OFF-state suites green
  throughout. `.venv/bin/python -m pytest`; `uvx ruff@0.15.16 check` (Python 3.12).

## 8. Rollout / phasing

1. **Foundation (WS-A + WS-B + WS-C)** — land + unit-green + ruff clean; commit the milestone
   (isolated from WS-F).
2. **Phase-1 validation** — restart VM, run the ~$30 Search-3B slice, evaluate the PASS gate. Iterate
   on the ~$30 evidence until green.
3. **WS-D + WS-E** — generalization + SOTA fold-in (may interleave with #2's debugging).
4. **WS-F** — finish + commit the external-runs monitor (parallel, independent).
5. **CLAUDE.md** (G7) + memory updates.
6. **Grid ($400)** — checkpointed; only after Phase-1 passes.

## 9. Risks

- **R1 — a global `ANTHROPIC_BASE_URL` would hijack `claude-oauth` (process-global).** Mitigation
  (WS-A item 3 + §5.A-coords): scope **per client** — a root `get_client` patch + an
  `AnthropicMessagesClient` `base_url` param — so no global env is needed for root/grader/verifier;
  the executor CLI's env is set per-subprocess where possible; and a **hard, tested co-residency
  guard** forbids `anthropic-foundry` and `claude-oauth` in the same run. Our approved routing has no
  OAuth tier, so the invariant holds by construction for the SDAR run.
- **R2 — `_synth_result_from_summary` under-projects** and ships a thin report as the default path.
  Mitigation: WS-B item 2 projects through the evidence bundle + verify_result; tested both ways.
- **R3 — Opus 4.8 as RLM root is not SDAR-paper-validated in *our* harness** (only reachability is
  proven). Mitigation: this is the Phase-1 run's purpose; the deterministic driver de-risks it (the
  backbone runs even if Opus's free-form driving is imperfect). Rate limits: 1M TPM / 1000 RPM
  (ample).
- **R4 — verl eval-provenance schema mismatch** with `EVAL_PROVENANCE_GUARD` (§6). Mitigation:
  reconcile before the run.
- **R5 — Money.** Autostop ON always; the ~$30 slice is the only pre-authorized spend; the grid stays
  checkpointed.

## 10. Working discipline

Opus authors + reviews **every** diff (verify the diff, not the summary); **Sonnet executes** against
tight, file-disjoint specs, dispatched **synchronously** for critical work (background subagents were
lost on a mid-turn process exit). TDD; every change **default-OFF / byte-identical** when its
flag/field is absent, hermetic ON+OFF tests; `ruff` clean; commit **infrequently at milestones**
(descriptive present-tense headlines: what+symptom+resolution); **push `deepinvent` only**; identity
`lolout1`; **no Co-Authored-By trailer**. Money/irreversible ⇒ checkpoint (except the pre-authorized
~$30 slice). `/implement` for implementation.

## 11. CLAUDE.md update (G7)

Add a concise rule block under the feature-flags / RLM-auth sections capturing: (a) the
Anthropic-Foundry provider (Opus 4.8 root / Sonnet 5 exec+grader via `.../anthropic/v1`,
`x-api-key=$AZURE_FOUNDRY_API_KEY`, the scoped-shim non-leak invariant); (b) `LIFECYCLE_PRIMARY` as
the harness-owned deterministic backbone (what it guarantees, the opt-out); (c) the `mode="adapt"`
wiring rule (`REPRODUCTION_MODE` is authoritative + set at setup, fail-loud when execute requested).
Keep the *rule*, not the incident narrative (which lives in this spec + the runbook).

---

### §5.A-coords — Anthropic-Foundry integration coordinates (mapped)

Confirmed premise: both Anthropic clients build `anthropic.Anthropic(api_key=…)` with **no**
`base_url` kwarg — `rlm/clients/anthropic.py:23-24` (root) and
`backend/services/context/workspace/tools/anthropic_messages_client.py:100-102` (grader/verifier).
Hence per-client scoping requires either a `get_client` patch (root) or a new `base_url` param
(grader/verifier); a global `ANTHROPIC_BASE_URL` is the only lever for the executor CLI.

**`models.py` (root registry):** `RootModel`+`api_key_env` `50-88`; `cred_provider` `90-111` (add
`opus-foundry`/`sonnet-foundry`→`"azure-foundry"`); clone the **`claude` entry `219-227`** for
`opus-foundry`(`claude-opus-4-8`)/`sonnet-foundry`(`claude-sonnet-5`) with a new
`rlm_backend="anthropic-foundry"` literal + `api_key_env="AZURE_FOUNDRY_API_KEY"`; register the
literal in `_VALID_RLM_BACKENDS 334-347` + `_BACKEND_ENV_KEY 356-361`; `_credential_value` already
resolves `AZURE_FOUNDRY_API_KEY` (`400`); add aliases in `_MODEL_ALIASES` foundry block `631-642`
(keep distinct from `opus`/`sonnet`→oauth at `601/603`); base_url injection branch at the
`azure-foundry` spot `770-774`; resolver `resolve_root_model 651-776` (avoid the anthropic-oauth
early-return `707-734`; API-key loop `745-755` validates `api_key_env`).

**`role_models.py` (sub-roles):** add `PROVIDER_ANTHROPIC_FOUNDRY` near `58-71`; add to
`SUBROLE_PROVIDERS 76-84` + `_VALIDATED_SUBROLE_PROVIDERS 88-90` (suppresses the fidelity warning —
it IS Claude); add `"sonnet-foundry"`/`"opus-foundry"` tokens to `_ROLE_VOCAB 103-142`;
`_classify_model_family 145-195` → map to `"claude"` (~`173`); `parse_model_spec` strict gate
`412-416` needs the provider in the set; `resolve_role_models 476-559` flows automatically.

**`grader_transport.py`:** clone the **anthropic branch `195-204`** into an `"anthropic-foundry"`
branch passing `AnthropicMessagesClient(model="claude-sonnet-5", api_key=$AZURE_FOUNDRY_API_KEY,
base_url=…/anthropic/v1)` (needs the new `base_url` param); `build_grader_client 292-322` +
`build_validator_client 325-387` flow through. `resolve_anthropic_subrole_backend 41-79` must NOT
be allowed to map a foundry sub-role to `oauth`/`anthropic` — special-case or bypass.

**`run.py`:** register the root `get_client` patch at the import chokepoint `123-124`; executor
provider map `_resolve_agent_runtime 429-441` (add `"anthropic-foundry":"anthropic"`,
`agent_model=spec.model`); `_subrole_backend`/`resolve_anthropic_subrole_backend 2908-2911` guard;
verifier client `2913-2922`; grader env bridge `2983-2986`; **RLM root construction `3433-3451`** (the
patch makes the `anthropic-foundry` backend build the Foundry client — no env needed).

**`foundry_endpoint.py` (mirror):** `resolve_foundry_credentials 68-77` + `_env_or_settings 55-65`
are the template; the new `resolve_foundry_anthropic_credentials` reuses `_env_or_settings` but
normalizes to `…/anthropic/v1` (do NOT reuse `normalize_foundry_base_url 36-52`, which forces
`/openai/v1`).

**Executor runtime:** `ClaudeAgentRuntime` (`claude_runtime.py:51-133`) + `make_runtime("anthropic")`
(`factory.py:569-572`) need **no new class** — only `agent_model="claude-sonnet-5"` + the CLI env.

**OAuth non-leak (R1):** the co-residency guard lives where `resolve_role_models` + root selection
are reconciled (`run.py ~2884-2911`): if any tier is `anthropic-foundry`, assert none is
`claude-oauth`. `llm_auth_strategy` gate (`factory.py:257-313`, `config.py:695-703`) is unaffected.

### §5.E — SOTA patterns cross-check + adoptions (sourced)

The web scan (primary sources via WebFetch; WebSearch was unavailable on the runner; the three most
decision-relevant claims re-verified verbatim) **validates the pivot rather than revealing a large
gap.** Headline: the benchmark *reference* agents that look most LLM-as-driver (PaperBench
`IterativeAgent`, AI Scientist v2) are deliberately minimal *to measure the model, not the scaffold*;
every *reliability-oriented* production system converges on **deterministic controllers that call the
LLM only for bounded sub-tasks** — our WS-B. The strongest datapoint: MLE-bench's best scaffold is
**AIDE**, whose controller is a *hard-coded* tree-search policy, not a free LLM loop
([arxiv 2410.07095](https://arxiv.org/abs/2410.07095), [AIDE 2502.13138](https://arxiv.org/html/2502.13138v1)).

**Already strong (keep):** evaluator-optimizer (`verify`/`propose_improvements`/climb), staged
pipeline, Reflexion-style failure memory (`failure_capsules`/`negative_lessons`), independent
verifier (`external_validator`), the fabrication/evidence guards (the red line), budget governance,
hierarchical weighted rubric leaves, typed dimensions (`ScopeSpec`/`cell_matrix`), evidence-bundle
receipt. The scan confirms each is a recognized best practice.

**Validations of this spec's choices:**
- **Deterministic controller + bounded LLM operators** (AIDE; PaperCoder's *fixed* Planning→Analysis→Coding,
  strict ordered file list, [2504.17192](https://arxiv.org/html/2504.17192v1); Anthropic *Building
  Effective Agents* — "workflows = predefined code paths", [anthropic.com](https://www.anthropic.com/engineering/building-effective-agents))
  → **WS-B**. PaperBench removed its submit tool because the free agent quit early
  ([2504.01848](https://arxiv.org/html/2504.01848v1)) — literally B1.
- **Opus-4.8 planner + Sonnet-5 workers** (OpenAI "o-series are the planners, GPT the workhorses",
  [reasoning-best-practices](https://developers.openai.com/api/docs/guides/reasoning-best-practices);
  Anthropic Opus-4-lead + Sonnet-4-subagents, **+90.2%**, [multi-agent-research-system](https://www.anthropic.com/engineering/multi-agent-research-system))
  → **WS-A routing** validated.

**New adoptions we do NOT yet do (each → a default-OFF flag + test):**

| # | Adoption (source) | Where | Rationale |
|---|---|---|---|
| E1 | **Reason free-form, then emit a *forced/constrained* action** (Claude tool-use `strict`/`tool_choice`; Outlines FSM [2307.09702]; but forcing JSON degrades reasoning [2408.02442] → keep reasoning separate) | WS-B (driver's per-step emit) | Makes the "unparseable FINAL_VAR" (B1) *structurally impossible* without hurting reasoning. The single most on-point fix. |
| E2 | **Pre-committed ordered plan artifact the executor cannot reorder** (PaperCoder) | WS-B (`plan_reproduction` → persisted ordered step list the driver dispatches) | Turns "what next?" from an LLM guess into deterministic dispatch. |
| E3 | **Type each rubric leaf + *blind*, evidence-scoped grading** (PaperBench Result-Match/Execution/Code-Dev: Result nodes see outputs+logs but NOT source; Code nodes see source but NOT logs) | new default-OFF flag over the leaf scorer | Upgrades the evidence gate; stops the judge crediting fabricated results from the wrong evidence. |
| E4 | **Metrics produced by execution, never the root** (MLE-bench: agent forbidden to write predictions) | strengthen `evidence_gate` + WS-B report projection | Closes the fabrication class the zero-metrics/eval-provenance guards already chase. |
| E5 | **Fresh-sandbox re-execution receipt + ≥3 seeds + error bars** (PaperBench fresh-VM `reproduce.sh`; MLE-bench ≥3 seeds; ML Repro Checklist) | repro-lock receipt (extends `CANONICAL_EVIDENCE_BUNDLE`) | Distinguishes real reproduction from hard-coded / one-lucky-seed. |
| E6 | **Cheap independent monitor over the root's action stream** (CoT reward-hacking monitor [2503.11926] — a weaker model reliably catches a stronger one) | new default-OFF monitor (grok/gpt tier) | Catches churn/reward-hacking early + cheaply; complements the degenerate detector. |
| E7 | **JudgeEval-style judge-accuracy suite vs gold labels** (PaperBench JudgeEval F1) | tooling (`scripts/`) beside `calibrate_grader.py` | Measures the grader's *accuracy*, not just its σ-noise — the fitness signal's fitness. |

**Prioritization for THIS spec:** E1 + E2 land inside WS-B (they *are* the deterministic-driver
hardening and directly kill B1); E4 is already largely present (reinforce). E3, E5, E6, E7 are
higher-value-but-larger → specified here as default-OFF follow-ons, flagged for the writing-plans
phase to sequence after SDAR Phase-1 is green (they must not delay the concrete proof). Every item
is byte-identical when its flag is unset.

### §5.F — External-runs monitor finish checklist (assessed)

**Assessment:** the subsystem (SSH-polls hand-launched remote VM runs → SSE + persisted
`runs/_external/<id>/events.jsonl`; Next.js page + proxies + sparkline/feed UI; nav entry) is
**well-designed, cleanly modular, fail-soft on boot** (the poller can never block/crash app boot —
verified), with **no stubs**. But it is **NOT commit-ready** — the specified pytest command **hangs
indefinitely** (two broken SSE tests). Finish to production quality, then commit **isolated** from the
harness work. `backend/services/external_monitor/` unit tests pass 34/34; the HTTP list/404 tests pass;
the two SSE-replay tests hang.

**Blocking fixes (must clear before commit):**
1. **Fix the hanging SSE tests** (`tests/routes/test_external_runs_http.py::test_events_route_replays_*`):
   the endpoint generator is an infinite `while True` keepalive (`external_runs.py:88-96`) → Starlette
   `TestClient.stream()` deadlocks at `__enter__`. Drive it with `httpx.AsyncClient(ASGITransport)`
   reading exactly N events under `asyncio.wait_for`, or a test-only terminating condition. This is the
   only coverage of the streaming path and it's broken.
2. **Contract drift:** backend sends `last_validation={metric,value,step}` (`poller.py:288`); frontend
   `ExternalRunValidation` type says `{metric,value,raw}` (`lib/external-runs/types.ts:20`). Align +
   add a serializer contract test.
3. **Nullable `value`/`step`** (parse-failure → `None`) not modeled: TS types declare non-null;
   `formatMetric`/`Sparkline`/progress math would render `null`. Make types `number | null` + guard, or
   drop null events server-side.

**Tests to add:** the entire `_tick`/`_poll_loop`/`_ssh_check` state machine is **untested** (backoff,
`_UNREACHABLE_THRESHOLD=3` + recovery, progress/heartbeat throttle, dedup, terminal→stop) — cover with
a fake `_ssh_check`; a working streaming-endpoint test; the two Node proxy routes (502 on backend down,
SSE passthrough). Run the existing thorough `external-runs-view.test.tsx` in CI.

**Hardening:** move `_persist_and_broadcast`'s blocking file I/O off the event loop (`asyncio.to_thread`);
bound subscriber queues (drop-oldest — a stalled SSE client currently grows memory unbounded); close the
replay→subscribe race (`external_runs.py:68` reads file then subscribes — an event in that window is
lost); `shlex.quote` `spec.process_pattern` (`parsing.py:51`, currently interpolated unquoted — a
malicious/typo'd registry entry could inject remote shell) or validate the registry as trusted-only;
basic **redaction** of remote log lines before persist/stream (a secret in a remote log is currently
echoed verbatim to the UI + JSONL).

**Config / deploy gate (important):** `configs/external_runs.json` pins **live infra**
(`deepinvent-ext-ut` / `sdar-2model-a` / real paths) and the poller starts **unconditionally** when the
registry loads → CI and every other deploy would background-`gcloud ssh` that VM on boot. Ship an
**empty/example** committed config + gitignore the real one, **and** gate the poller behind an explicit
enable flag (e.g. `OPENRESEARCH_EXTERNAL_MONITOR=1`, default OFF) so unset = no background SSH =
byte-identical. Document the ambient `gcloud`/host-key auth prerequisite (add non-interactive /
`--tunnel-through-iap` flags as needed).

This is the "inspect runs / watch live logs / steer" surface (the alphaXiv Autoresearch loop); once the
blocking fixes + the enable gate land, it commits cleanly and independently of WS-A/B/C.
