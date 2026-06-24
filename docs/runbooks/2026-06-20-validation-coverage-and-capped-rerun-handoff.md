# Validation-coverage improvements + capped SDAR re-run — handoff (2026-06-20)

> **For the NEXT (fresh) session.** Self-contained: assumes no prior context.
>
> **What happened:** the grounded-self-improvement test (root=gpt-chat-latest via
> Azure Foundry, executor=Sonnet/OAuth, validator=grok-4.3 via Azure Foundry) ran
> end-to-end on SDAR (arXiv 2605.15155) on GCP 8×A100 spot. It **validated the hard
> parts** (keyless GPT root drives the loop, Sonnet writes real multi-file SDAR code,
> real GPU training, grok validator transport, Tier-1 lifecycle ledger redaction) but
> **could not reach the validator panel / fix-first loop** because of a timeout trap
> (§2). The VM is **STOPPED** (`sdar-a100-8g`, TERMINATED, $0; boot disk + artifacts
> persist).
>
> **This session's job (two parts):**
> 1. **Build A/B/C** — extend validation coverage to the **root's report** (not just the
>    executor's metrics) + deterministic claim↔evidence guardrails (§3). All flag-gated,
>    default-OFF, with tests. (Opus writes/reviews; Sonnet implements.)
> 2. **One capped re-run** (§4) with `OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S=1800` so the
>    run actually reaches finalize and exercises the **full** Tier-1/2/3 pipeline + A/B/C
>    in ~1h / ~$12.
>
> Branch: `feat/grounded-self-improvement-harness-reliability` (NOT pushed; push to
> **deepinvent** only, on request). Prior handoff: `2026-06-20-grounded-self-improvement-gcp-test-handoff.md`.

---

## 1. What the 2026-06-20 run established (`sdar_gcp_grounded_gptchat_grok_20260620`)

The run config is the proven baseline — reuse it verbatim for the re-run (§4).

**VALIDATED (high confidence):**
- **Keyless `gpt-chat-latest` (Azure Foundry) drives the RLM root loop** — 8 `repl_iteration`,
  49 `primitive_call`, NO degeneration. Settles the "no reliable keyless root" blocker:
  a strong GPT-chat root works where Sonnet-OAuth-as-root risks the degenerate loop.
- **Sonnet (OAuth) executor writes REAL multi-file SDAR code** — 13 files incl
  `alfworld_env.py`, `search_qa_env.py`, `agentic_rollout.py`, `provenance.py`,
  `cells.json`, `train_cell.py`. Not a stub. (gpt-chat-latest stubs as executor — keep it
  root-only; Sonnet/gpt-5 remain the validated executor.)
- **Real GPU training, evolving SDAR-specific metrics** — 8 cells, real VRAM (7–36 GB).
  SDAR cell `teacher_gap_mean` −0.79→**−1.31**, `gate_activation_ratio` 0.22→**0.27** over
  steps 5→14, real non-zero step-varying loss. OPSD surrogate + sigmoid gate genuinely wired.
- **grok-4.3 validator transport confirmed live** — `build_validator_client` →
  `OpenAILlmClient(model="grok-4.3")`; separation `independent` (grok ≠ Sonnet executor).
- **Tier-1 lifecycle ledger: recording + correctly REDACTED** — 24 entries,
  `inputs_projection:{}`, `outputs_pointer:{code_dir}`, NO paper text. Redaction requirement holds.
- **Arg-contract guard fired** — `compute_scope_invalid` when the root passed a string for a
  dict arg (a root-hallucination caught deterministically).

**NOT reached this run (the whole reason for the capped re-run):** the validator **panel
running** at finalize, the zero-metrics veto, the Tier-3 fix-first loop, the honest terminal
report. These require `run_experiment` to **return** → root `verify` → `FINAL_VAR`.

**Minor bug found (fix opportunistically):** the `role_model_fidelity` advisory stamps the
validator as `azure-foundry:gpt-chat-latest (token 'grok-4.3')` — the **bridged validator
RoleSpec resolves its deployment from the global `AZURE_FOUNDRY_DEPLOYMENT` instead of
`OPENRESEARCH_VALIDATOR_MODEL`**. The actual transport + separation are correct (grok); only
the stamp is wrong. Fix: in the validator-role bridge (run.py / role_models.py), stamp the
RoleSpec model from `OPENRESEARCH_VALIDATOR_MODEL` when set. Verify `report.models`/`validation`
shows grok after the re-run.

---

## 2. THE TIMEOUT TRAP (root cause of "validator never ran") — the must-fix for the re-run

Everything converges at **6h**, so the run can't finalize before the outer wall:

| Knob | Value | Source |
|---|---|---|
| `EXPERIMENT_TIMEOUT_BY_MODE["max"]` | 21600 (6h) | `primitives.py` — per `run_experiment` call, becomes the **per-cell timeout** |
| Outer wall (`OPENRESEARCH_SDAR_OUTER_WALL_S`) | 21600 (6h) | `sdar_gcp_run.sh` `timeout --signal=TERM` |
| Step rate | ~2.5 min/step (search_qa); **~20–30 min/step (alfworld)** | empirical |
| 150 steps | ~6–7h search_qa; **50+ h alfworld** | derived |

At ~6h the **outer-wall SIGTERM** fires first → `run.py::_install_sigterm_finalizer` →
`_hard_stop_with_report`, which **does NOT call `run_validation_panel`** (the validator runs
ONLY in `_validator_gate` (run.py:~3071) and `_finalize` (run.py:~3560), both reached only on a
natural `FINAL_VAR`). So a timeout/SIGTERM/watchdog stop salvages a partial report but **skips
Tier-2/Tier-3 entirely.**

**Fix (the re-run must include):** cap the experiment timeout so cells get SIGKILLed early,
partials are collected by `gpu_cell_runner._load_cell_metrics`, `run_experiment` RETURNS, and
the root proceeds to verify → `FINAL_VAR` → validator → finalize:

```
OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S=1800   # 30 min/experiment; cells finalize on partial-but-real metrics
```

(Do **not** "just raise the outer wall" — alfworld would need 50+h; you'd pay ~6h×8×A100 ≈ $60+
to test soon-to-be-superseded code. The cap is ~$12 and tests the full pipeline.)

---

## 3. Build A/B/C — extend validation to the ROOT + deterministic guardrails

**Design rationale (the red line):** fitness = the deterministic evidence layer, never the LLM
grade, never the root's narrative. Today the grok validator + Tier-1 guards check the
**executor's metrics**; the **root's report narrative is unchecked** (it could overclaim numbers
absent from `metrics.json`). Don't add a second validator — give the existing one a second
*target* with a deterministic backstop. All knobs default-OFF ⇒ byte-identical to today.

### A — Report-claim ↔ evidence deterministic gate  (`OPENRESEARCH_REPORT_CLAIM_GATE`, default OFF)
- **New module** `backend/agents/rlm/report_claim_gate.py`.
- **Hook**: `report.py::write_final_report_rlm`, after the report dict is assembled, before write.
- **Logic**: extract quantitative *result claims* from the report's results/claims narrative
  (numbers adjacent to result terms — `accuracy|success|reward|f1|score|rate|em` — NOT
  hyperparameters like `lr/beta/lambda/seed`). For each, check for a matching value in the
  on-disk evidence (`code/metrics.json` + per-cell) within a relative tolerance, reusing
  `zero_metrics_detection.normalize_metric_values` + `evidence_gate` metrics reading.
- **Output**: `report["claim_grounding"] = {grounded:[...], ungrounded:[{claim,value,context}]}`;
  emit `run_warning code="report_claim_ungrounded"` listing unsupported claims. Conservative
  (only result-context numbers; hyperparameters exempt). Fail-soft: off/unparseable → no-op.
- **Test** `tests/rlm/test_report_claim_gate.py`: ungrounded claim flagged; grounded claim +
  hyperparameter not flagged; off ⇒ no `claim_grounding` key.

### B — grok validator's 2nd target = the root's report  (`OPENRESEARCH_VALIDATOR_CHECK_REPORT`, default OFF; requires `OPENRESEARCH_EXTERNAL_VALIDATOR=1`)
- **Hook**: `backend/agents/rlm/external_validator.py`.
- Add a **5th typed predicate** `report_claims_grounded` whose harness-side machine-check IS
  module A's claim↔metrics logic (the validator only *points*; A *verifies* — same contract as
  the existing `provenance_present`/`not_all_constant`/`gpu_claim_plausible`/`rerun_agrees`).
- Thread the report's claims/results narrative into `run_validation_panel(... report_claims=...)`
  (from `run.py::_finalize`, which has the assembled report) and into `_ADVERSARIAL_USER_TEMPLATE`
  so grok flags suspicious report claims. Min-aggregation veto extends to report-claim
  `metric_ref`s; `status="vetoed"` drives the Tier-3 fix-first loop (root must fix report OR
  evidence). Keep the predicate list valid in the prompt's JSON contract.
- **Test** `tests/rlm/test_external_validator_report.py`: report with an ungrounded claim →
  predicate machine-verifies → veto; grounded report → clean.

### C (optional) — metric-semantics guards  (`OPENRESEARCH_METRIC_SEMANTICS_GUARD`, default OFF)
- **New module** `backend/agents/rlm/metric_semantics.py` (sibling to `zero_metrics_detection.py`).
- Result-claiming rate metrics (`accuracy|success_rate|f1|precision|recall`) must be in **[0,1]**;
  loss/reward finite. A clear violation (e.g. accuracy 1.7 or −0.3) → degrade to the repairable
  `failure_class="fabrication_suspected"`, wired in `run_experiment` beside the zero-metrics +
  stub guards. Conservative — a legit 0.0 reward is in-range (NOT flagged); only out-of-range
  fires (calibration-sensitive, hence optional / behind its own flag).
- **Test** `tests/rlm/test_metric_semantics.py`: accuracy 1.7 → veto; 0.4 → pass; reward 0.0 → pass.

**Verified hooks (2026-06-20):** `evidence_gate.py` → `leaf_claims_measured_result`/`gate_decision`;
`external_validator.py` → `PredicateVerdict`/`ValidatorVerdict`/`check_*`/`_ADVERSARIAL_USER_TEMPLATE`/
`run_validation_panel`; `zero_metrics_detection.py` → `normalize_metric_values`/`looks_like_zero_metrics`/
`zero_metrics_should_veto`; `report.py` → `write_final_report_rlm` + verdict-downgrade honesty checks.

---

## 4. The capped validation re-run (turnkey)

The proven launch path needs **no script edits** — config flows through the VM's
`runs/.cache/sdar_gcp.env` (sourced by `sdar_gcp_run.sh` with `set -a` → inherited by the CLI).

### 4.1 Bring up the VM (it's TERMINATED on a CPU machine type after the stop)
```bash
export CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud
P=deepinvent-ext-ut; Z=us-central1-c; I=sdar-a100-8g
# inspect/sync on CPU first (cheap), then preflight launch flips to a2-highgpu-8g + gates + launches.
gcloud --project $P compute instances start $I --zone $Z          # starts on current (CPU) type
bash scripts/gcp_sdar_preflight.sh sync                            # push THIS branch's code (incl A/B/C) to the VM
```

### 4.2 Inject the env block (the proven config + the timeout cap + A/B/C flags)
The block from the prior run is saved at `runs/.cache/grounded_test_env_block.sh`. Append it to
the VM's `runs/.cache/sdar_gcp.env` (idempotent via the `>>> grounded-self-improvement E2E test`
markers), then **add the new lines**:
```bash
# new for this re-run (append inside the marker block, or as a second block):
export OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S="1800"      # THE FIX — cells finalize on partial metrics
export OPENRESEARCH_REPORT_CLAIM_GATE="1"                # A
export OPENRESEARCH_EXTERNAL_VALIDATOR="1"               # (already on)
export OPENRESEARCH_VALIDATOR_CHECK_REPORT="1"           # B
export OPENRESEARCH_METRIC_SEMANTICS_GUARD="1"           # C (if built)
export OPENRESEARCH_SDAR_PROJECT_ID="sdar_gcp_grounded_v2_20260620"   # fresh id
```
The rest of the proven block (verbatim): root `AZURE_FOUNDRY_DEPLOYMENT=gpt-chat-latest`,
`OPENRESEARCH_SDAR_ROOT=foundry`, `OPENRESEARCH_SDAR_MODELS=executor=sonnet,grader=sonnet,verifier=sonnet`,
`OPENRESEARCH_LLM_AUTH_STRATEGY=oauth_only`, `OPENRESEARCH_ZERO_METRICS_GUARD=1`,
`OPENRESEARCH_LIFECYCLE_LEDGER=1`, `OPENRESEARCH_STUB_METRICS_GUARD=1`, `OPENRESEARCH_EVIDENCE_GATE=1`,
`OPENRESEARCH_ARG_CONTRACTS=1`, `OPENRESEARCH_VALIDATOR_BACKEND=azure-foundry`,
`OPENRESEARCH_VALIDATOR_MODEL=grok-4.3`, `OPENRESEARCH_VALIDATOR_PANEL_N=2`,
`OPENRESEARCH_REPAIR_MAX_ITERATIONS=4`, `OPENRESEARCH_POSITIVE_RECIPES=1`, `OPENRESEARCH_GRADER_SAMPLES=1`,
`OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD=3`, `OPENRESEARCH_SDAR_NO_AUTOSTOP=1`.
Guidance file `runs/.cache/sdar_scope_guidance.txt` (smallest-two: Qwen3-1.7B + Qwen2.5-3B,
ALFWorld+Search-QA, anti-fab rails) is already staged — keep it.

### 4.3 Pre-launch creds probe (CPU, free) + launch
```bash
# OAuth live? both Foundry transports live?
bash scripts/gcp_sdar_preflight.sh check       # asset GREEN (sourced)
# (optional) re-run the foundry transport probe from this session's notes (gpt-chat-latest + grok → pong)
bash scripts/gcp_sdar_preflight.sh launch       # flips to a2-highgpu-8g, GPU gate, detached run
```

### 4.4 Success checks (the point of the re-run)
Watch `runs/<id>/dashboard_events.jsonl` + `final_report.json` + sidecars.
- [ ] **Reaches a natural finalize** (not a SIGTERM hard-stop): `run_experiment` returns ~30 min in
      on partial metrics; root calls `verify_against_rubric` then `FINAL_VAR`.
- [ ] **grok validator panel RUNS** — `rlm_state/validation_verdict.json` written;
      `final_report.json.validation` carries `{status, veto_set, separation:"independent"}`; the
      model stamp shows **grok-4.3** (confirms the §1 stamp-bug fix).
- [ ] **Tier-1** — real cells keep non-zero loss/teacher_gap/gate (no false zero-metrics veto);
      the WebShop `None`-metric cells handled (not credited); lifecycle ledger grows, still redacted.
- [ ] **A — report-claim gate** — `final_report.json.claim_grounding` present; any ungrounded number
      flagged (`report_claim_ungrounded`). Construct/observe one ungrounded claim to confirm it fires.
- [ ] **B — validator judges the report** — a `report_claims_grounded` predicate appears; an
      ungrounded claim → `validation.status="vetoed"` → Tier-3 repair driven.
- [ ] **C (if built)** — an out-of-range metric → `fabrication_suspected`; in-range untouched.
- [ ] **Tier-3** — a veto REFUSES `FINAL_VAR`, drives a bounded repair → honest `repair_exhausted`
      (NOT `root_degenerate_loop`, NOT a shipped fake) OR a real fix accepted.
- [ ] **Honest terminal** — real reproduction OR honest `degraded`/`failed`/`repair_exhausted` with a
      cited reason. Never a silently-shipped all-zero/overclaimed fake.

### 4.5 After: STOP the VM
`OPENRESEARCH_SDAR_NO_AUTOSTOP=1` keeps it up for sidecar inspection — **`gcloud compute instances
stop sdar-a100-8g --zone us-central1-c` as soon as results are captured** (8×A100 burns ~$10–12/h).

---

## 5. Gotchas / caveats
- **The timeout cap is mandatory** (§2) or the validator never runs. This is the #1 lesson.
- **Executor must be Sonnet/gpt-5**, never gpt-chat-latest (it stubs as executor; fine as root).
- **No Foundry deployment clash**: root reads `AZURE_FOUNDRY_DEPLOYMENT` (=gpt-chat-latest), validator
  reads `OPENRESEARCH_VALIDATOR_MODEL` (=grok-4.3) — same endpoint+key, distinct deployments.
- **gpt-chat-latest is reasoning-class**: returns empty below ~32 output tokens; the code's
  `_is_reasoning_model` (covers `gpt-chat`) correctly omits temperature + uses `max_completion_tokens`.
  grok-4.3 takes standard params (Foundry chat/completions) — both confirmed live this session.
- **Executor adds WebShop cells** despite "skip WebShop" guidance; they produce `None` metrics
  (server gap) and are handled — harmless, isolated.
- **alfworld cells are very slow** (~20–30 min/step) — the cap is what makes them tractable.
- **Branch not pushed**; push to **deepinvent** only, on request.

## 6. Pointers
- Working config block: `runs/.cache/grounded_test_env_block.sh`; guidance: `runs/.cache/sdar_scope_guidance.txt`.
- Prior-run artifacts (reference): `runs/sdar_gcp_grounded_gptchat_grok_20260620/` on the VM boot disk.
- Launch path: `scripts/gcp_sdar_preflight.sh` (lifecycle) + `scripts/sdar_gcp_run.sh` (the CLI invocation; sources `sdar_gcp.env`).
- Build hooks: `backend/agents/rlm/{evidence_gate,external_validator,zero_metrics_detection,report,run}.py`.
- Prior handoff + spec: `docs/runbooks/2026-06-20-grounded-self-improvement-gcp-test-handoff.md`,
  `docs/superpowers/specs/2026-06-20-grounded-self-improvement-and-harness-reliability-redesign-design.md`.
