# Grounded self-improvement — GCP end-to-end test handoff (2026-06-20)

> **For the NEXT (fresh) session.** Self-contained: assumes no prior context. The
> grounded-self-improvement + harness-reliability redesign is **implemented,
> Codex-reviewed, and committed** on branch `feat/grounded-self-improvement-harness-reliability`
> (10 commits `ebdbb61f`..`18d3fe5e`, **NOT pushed**). Full hermetic suite: **6587
> passed, 0 regressions** (15 failures are pre-existing/environmental — OCR data,
> azure-foundry pricing gap, creds/sandbox-config tests — none in touched files).
>
> **This session's job: run the first GCP end-to-end SDAR test that EXERCISES all
> the new guards**, with root = **Opus via OAuth**, executor = **GPT-5.5-thinking
> via Azure OpenAI**, validator = **OAuth-Claude** (cross-family from the GPT executor).
>
> Authoritative refs: spec `docs/superpowers/specs/2026-06-20-grounded-self-improvement-and-harness-reliability-redesign-design.md`,
> plan `docs/superpowers/plans/2026-06-20-grounded-self-improvement-implementation-plan.md`,
> operator checklist `docs/runbooks/2026-06-20-grounded-self-improvement-operator-checklist.md`.

---

## 0. What's new (the flags this test exercises)

All default-OFF; this test turns them ON. Fitness = the deterministic evidence
layer, never the LLM grade.

| Flag | Tier | What it does |
|---|---|---|
| `OPENRESEARCH_ZERO_METRICS_GUARD=1` | 1 | Vetoes an all-zero/constant `metrics.json` that CLAIMS gpu training but has no `provenance.json` → `fabrication_suspected`. **This is the v6 fix.** |
| `OPENRESEARCH_LIFECYCLE_LEDGER=1` | 1 | Append-only redacted per-primitive audit at `rlm_state/lifecycle/ledger.jsonl`. |
| `OPENRESEARCH_EXTERNAL_VALIDATOR=1` | 2 | Separate-model adversarial panel; typed-predicate machine-checks; min-aggregation veto; fingerprint-keyed verdict. Drives the fix-first loop. |
| `OPENRESEARCH_VALIDATOR_BACKEND=oauth` | 2 | Validator transport = OAuth-Claude (Sonnet). **Cross-family from the GPT executor → `independent` panel.** |
| `OPENRESEARCH_VALIDATOR_PANEL_N=2` | 2 | Panel sample count (keep small for cost). |
| `OPENRESEARCH_REPAIR_MAX_ITERATIONS=4` | 3 | Fix-first repair ceiling → honest `repair_exhausted`. |
| `OPENRESEARCH_POSITIVE_RECIPES=1` | 3 | Cross-run recipes. **Caveat (§5): admits only with a deterministic measured-vs-claimed target, which nothing produces yet → effectively a no-op this run. Enable to confirm it's harmless.** |

Keep the existing proven guards ON too: `OPENRESEARCH_STUB_METRICS_GUARD=1`,
`OPENRESEARCH_EVIDENCE_GATE=1`, `OPENRESEARCH_ARG_CONTRACTS=1`,
`OPENRESEARCH_LLM_AUTH_STRATEGY=oauth_only`.

---

## 1. The model config (the heart of this test)

| Role | Model | Transport | Why |
|---|---|---|---|
| **Root** (planner) | **Opus** | OAuth (`CLAUDE_CODE_OAUTH_TOKEN`) | The keyless-root experiment. `claude-oauth`=**Sonnet** is documented `RISK_DEGENERATE_LOOP` as root; **Opus is a stronger model that may NOT degenerate** — if it drives reliably this solves the "no reliable keyless root" blocker. |
| **Executor** (`implement_baseline`) | **GPT-5.5-thinking** | Azure OpenAI (`AZURE_OPENAI_*`) | EXPERIMENTAL. The validated executor is Sonnet/gpt-5; a strong *reasoning* model should write real multi-file code (unlike `gpt-chat-latest` which stubbed). This run reveals whether it writes real SDAR code or stubs. |
| **Validator** (NEW) | Sonnet | OAuth | **Cross-family from the GPT executor** → `independent` panel (the strongest separation). |
| **Grader / verifier** | Sonnet | OAuth | Default; quality-critical, stays on the proven OAuth-Sonnet. |

### ⚠️ Two config mechanisms to VERIFY first (I could not test these end-to-end)

1. **Opus-as-OAuth-root.** `claude-oauth` resolves to **Sonnet** by default (the
   root alias table collapses opus→sonnet). To pin **Opus** as the root, try
   `--models planner=opus` (with `--model` unset, `resolve_root_model` uses the
   planner token). **VERIFY it actually ran Opus**: check `final_report.json`
   `models.planner` stamps an Opus model id, and the live events. If it silently
   ran Sonnet, you need a `models.py` opus-oauth root entry or an
   `OPENRESEARCH_RLM_ROOT_MODEL` override — flag this and fix minimally.
2. **GPT-5.5-thinking executor deployment.** `OPENRESEARCH_EXECUTOR=azure` routes
   `implement_baseline` to `AZURE_OPENAI_DEPLOYMENT`. Point that at the
   GPT-5.5-thinking deployment. Confirm `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT`
   / `AZURE_OPENAI_DEPLOYMENT` are set for that deployment on the GCP box.

No deployment clash: the **validator is OAuth** (not Azure), so it does not share
`AZURE_OPENAI_DEPLOYMENT` with the executor.

---

## 2. Preflight (on the GCP box, before launch)

```bash
export CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud
P=deepinvent-ext-ut; Z=us-central1-c; I=sdar-a100-8g   # 8×A100 spot; currently TERMINATED ($0)
# Bring it up (machine-type flip needs TERMINATED):
gcloud compute instances set-machine-type $I --zone $Z --machine-type a2-highgpu-8g  # while TERMINATED
gcloud compute instances start $I --zone $Z --project $P
gcloud compute ssh abheekp@$I --zone $Z --project $P
```

On the box, `cd /home/abheekp/openresearch`, then:
- `git fetch && git checkout feat/grounded-self-improvement-harness-reliability` (the branch is NOT pushed — if the remote lacks it, `scp`/sync the local branch or `git push` to deepinvent first per repo policy).
- Confirm `.env`: `CLAUDE_CODE_OAUTH_TOKEN` LIVE (`claude --print "ping"`→`pong`), `AZURE_OPENAI_API_KEY`/`_ENDPOINT`/`_DEPLOYMENT`(=gpt-5.5-thinking) set.
- Caches warm (Qwen weights, ALFWorld + Search-QA; WebShop is a documented gap → scope to ALFWorld + Search-QA).

---

## 3. The run spec (uses the NEW `--run-spec`, P0 §10)

Write `runs/.cache/grounded_test_runspec.json` (one file instead of the 12-var
env whitelist):

```json
{
  "OPENRESEARCH_ZERO_METRICS_GUARD": "1",
  "OPENRESEARCH_LIFECYCLE_LEDGER": "1",
  "OPENRESEARCH_EXTERNAL_VALIDATOR": "1",
  "OPENRESEARCH_VALIDATOR_BACKEND": "oauth",
  "OPENRESEARCH_VALIDATOR_PANEL_N": "2",
  "OPENRESEARCH_REPAIR_MAX_ITERATIONS": "4",
  "OPENRESEARCH_POSITIVE_RECIPES": "1",
  "OPENRESEARCH_STUB_METRICS_GUARD": "1",
  "OPENRESEARCH_EVIDENCE_GATE": "1",
  "OPENRESEARCH_ARG_CONTRACTS": "1",
  "OPENRESEARCH_LLM_AUTH_STRATEGY": "oauth_only",
  "OPENRESEARCH_EXECUTOR": "azure",
  "OPENRESEARCH_GRADER_SAMPLES": "1",
  "OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD": "3",
  "models": "validator=sonnet,grader=sonnet,verifier=sonnet",
  "baseline_extra_guidance": "SDAR smallest-two scope: Qwen3-1.7B + Qwen2.5-3B on ALFWorld + Search-QA only (WebShop is a known env gap). Wire training loss to backprop from the model, reward to read real env outcomes, eval to score against gold — DO NOT write zero or placeholder metrics."
}
```

> Notes: `OPENRESEARCH_LLM_AUTH_STRATEGY=oauth_only` forces the Claude sub-roles
> (validator/grader/verifier) onto the OAuth token even though an `ANTHROPIC_API_KEY`
> may be present — so the validator is genuinely OAuth-Sonnet. The executor uses the
> single documented switch `OPENRESEARCH_EXECUTOR=azure`, which routes
> `implement_baseline` to the Azure OpenAI deployment named by `AZURE_OPENAI_DEPLOYMENT`
> (= gpt-5.5-thinking) — so `executor` is deliberately NOT in the `models` string
> (don't double-configure it). Set the **root** on the command line (§4) since the
> root is not a sub-role token.

---

## 4. Launch

The repo's GCP launcher (`scripts/gcp_sdar_preflight.sh`, now `--run-spec`-aware
via this session's work) drives a detached spot run. Confirm how `launch` consumes
the root model + the run-spec in `scripts/sdar_gcp_run.sh`; the shape is:

```bash
export CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud
# Root = Opus via OAuth (VERIFY it pins Opus — see §1 caveat 1):
OPENRESEARCH_SDAR_ROOT="claude-oauth" \
OPENRESEARCH_SDAR_ROOT_MODELS="planner=opus" \
OPENRESEARCH_SDAR_RUNSPEC="runs/.cache/grounded_test_runspec.json" \
OPENRESEARCH_SDAR_PROJECT_ID="sdar_gcp_grounded_opus_gpt55_$(date +%Y%m%d)" \
bash scripts/gcp_sdar_preflight.sh launch
```

If the launcher doesn't yet thread `--run-spec`/the root-models through, run the
CLI directly on the box (the canonical form — adapt to `sdar_gcp_run.sh`):

```bash
python -m backend.cli reproduce 2605.15155 \
  --sandbox local --gpus-per-run auto \
  --models planner=opus,validator=sonnet,grader=sonnet,verifier=sonnet \
  --run-spec runs/.cache/grounded_test_runspec.json \
  --project-id sdar_gcp_grounded_opus_gpt55_$(date +%Y%m%d)
# (executor=Azure comes from OPENRESEARCH_EXECUTOR=azure in the run-spec, not --models)
```

8×A100 spot ≈ cents–dollars/run; the validator panel + repair loop add a few LLM
calls. Budget caps (`OPENRESEARCH_MAX_RUN_GPU_USD`) still apply.

---

## 5. What to verify (the point of the test)

Watch `runs/<id>/dashboard_events.jsonl` + `final_report.json` + the new sidecars.

**Root reliability (the keyless-root experiment):**
- [ ] `final_report.json.models.planner` is an **Opus** id (not Sonnet) — else §1 caveat 1.
- [ ] The root does NOT degenerate at iter 0 (no `root_degenerate_refusal_loop` early-abort). If it DOES, Opus-oauth is also degenerate-as-root → run the §5.2 precondition experiment (`OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD=16`) to see if it self-recovers with more rope.

**Executor capability (Azure GPT-5.5-thinking):**
- [ ] `implement_baseline` writes **real multi-file SDAR code** (env files, `train_cell.py`, `cells.json`) — not a stub. If it stubs, `STUB_METRICS_GUARD` should catch it (`fabrication_suspected`).

**Tier 1 — zero-metrics floor (the v6 fix):**
- [ ] If any cell writes all-0.0 metrics after GPU training with no provenance → a `run_warning code="fabrication_suspected"` fires and the result degrades (NOT shipped as success). This is the headline check.
- [ ] `rlm_state/lifecycle/ledger.jsonl` exists and records per-primitive outcomes (ok/failed/timeout/raised), with NO paper text in it.

**Tier 2 — validator:**
- [ ] On a `FINAL_VAR`-attempt, the panel runs ONCE per evidence state; `rlm_state/validation_verdict.json` is written; `final_report.json.validation` carries `{status, veto_set, separation}` with **`separation="independent"`** (Opus-root is irrelevant; separation is executor-GPT vs validator-Claude).
- [ ] If the metrics are fabricated, `validation.status="vetoed"` and a `validator_veto` repair is driven (not a clean finalize).

**Tier 3 — fix-first loop:**
- [ ] A fabrication veto (Tier 1 or 2) REFUSES `FINAL_VAR` and drives a repair; if the executor keeps producing the same fake → bounded by `REPAIR_MAX_ITERATIONS` and stops as `failure_class="repair_exhausted"` (an HONEST failed/degraded report with a cited reason), NOT `root_degenerate_loop`, NOT a shipped fake.
- [ ] If the executor FIXES it (real metrics) → accepted, real score.
- [ ] `POSITIVE_RECIPES`: expect NO recipe admitted (no claimed-target signal) — confirm it's a harmless no-op (`runs/_recipes/` empty/absent).

**Honest outcome either way:** the run should END in one of {real reproduction, honest `degraded`/`failed` with a cited unfixed reason, honest `repair_exhausted`} — never a silently-shipped all-zero fake. That is the whole point.

---

## 6. Known caveats / risks (read before launching)

- **Opus-oauth-root is unverified** (caveat §1.1). It may run Sonnet silently, or degenerate. Both are informative outcomes — record which.
- **GPT-5.5-thinking executor is unvalidated** for the executor tier. It may write real code (good — a new validated executor!) or stub (then the guards catch it). Either way the test is informative.
- **POSITIVE_RECIPES won't admit** (no deterministic target signal is produced) — it's the most incomplete piece; treat as "verify harmless," not "verify admits."
- **Validator-rerun-on-suspicion (§7.5)** is a P2 stub (returns skipped) — no GPU re-materialization yet.
- **The cited-nudge-to-ledger** detector polish was documented as a remaining follow-on (the nudge names the stage but doesn't yet cite the ledger record path).
- **All flags are default-OFF.** If any new guard misbehaves, drop it from the run-spec and the harness reverts to the prior baseline — diagnose in isolation.
- **Cost.** 8×A100 spot + the panel/repair LLM calls. Preemption is handled (cell-fingerprint resume + `primitive_cache`); relaunch reuses cached primitives.

## 7. After the test

- Record outcomes in a short follow-up (which guards fired, did Opus-root drive, did GPT-5.5 executor write real code, did the loop repair-or-honest-fail).
- If a guard's default-ON flip is warranted, that needs ≥3 paired SDAR A/B runs first (operator checklist §4) — do NOT flip a default from one run.
- Branch is **not pushed**; push to **deepinvent** only, on request (repo policy).
- The "claude-oauth worked before but fails now" regression investigation remains open (git-bisect candidate) — orthogonal to this test but related.

## 8. Pointers

- Implementation: `backend/agents/rlm/{zero_metrics_detection,lifecycle_ledger,external_validator,recipe_library,forced_iteration}.py` + wires in `{primitives,binding,run,report,role_models,grader_transport}.py` + `cli.py` `--run-spec`.
- Spec / plan / operator checklist: see the header.
- Canonical SDAR run: `docs/runbooks/2026-05-23-sdar-baseline-handoff.md`, `2026-06-16-sdar-on-gcp-a100-vm.md`.
- Superseded post-mortem: `docs/runbooks/2026-06-20-sdar-harness-refactor-and-external-validation-handoff.md`.
