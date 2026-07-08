# SDAR end-to-end on GCP — guarded root + validated executor (handoff, 2026-06-19)

> **Status:** READY to run next session. All VMs are STOPPED ($0 compute now). Code is on
> `deepinvent` `feat/azure-bicep-canonical-aoai-hardening` @ **94c177e8** (the reasoning-chat-root
> guardrails). This handoff executes the **deferred end-to-end lever** recorded in
> `docs/local/2026-06-19-kimi-sdar-run-handoff.md` (a local working note; §"End-to-end lever, USER
> DEFERRED") — now that the guardrails make it viable.

## 1. Goal + success criteria

First **successful** end-to-end SDAR (arXiv 2605.15155) reproduction on GCP with **root =
gpt-chat-latest** (Foundry, OAuth-free) + a **validated executor = Sonnet**, the new orchestration
guardrails ON. Prove the three invariants hold for a chat root:

1. **Real code** — `run_experiment` does real GPU training (peak VRAM > 0; no `fabrication_suspected`
   veto on the *final* result); `experiment_runs.jsonl` has `success=True` with the paper's REAL
   metric keys (`success_rate`/`accuracy`/reward), never `total_length`/`chunk_count`.
2. **Full paper** — the requested scope runs (smallest-two first, then full), with honest
   `scope.gaps` only for genuinely-blocked pieces (WebShop is install-risk → documented gap, not a
   crash).
3. **Terminate correctly** — `FINAL_VAR` fires cleanly; `verdict ∈ {reproduced, partial}` with a
   non-zero rubric; NOT a `root_degenerate_loop` abort.

## 2. Why this should now work (what changed since the v5 stub)

Run `sdar_gcp_gptchat_v5_20260619` (all roles gpt-chat-latest) **stubbed at `implement_baseline`**
(0-GPU, placeholder `total_length`/`chunk_count`, empty `cells.json`) and the v5 thesis was
confirmed: *an unvalidated coder can drive the loop but cannot faithfully implement.* This session
added (commit 94c177e8, all **default-OFF** — enable via the flags below):

- **G2 `OPENRESEARCH_STUB_METRICS_GUARD`** — a placeholder-only-metrics `success` is now degraded to
  the repairable `fabrication_suspected`, so the root **re-implements instead of finalizing on a
  stub** (the v5 stub would now be caught route-agnostically — the VRAM verdict missed it because it
  never *claimed* gpu training).
- **G1 `OPENRESEARCH_ARG_CONTRACTS`** — placeholder args (`'unknown'`) are blocked before the
  primitive runs with a crisp repair.
- **P1-P3** (always-on in the `azure-foundry` addendum) — grounding / full-paper persistence +
  honest-failure / stub→re-drive guidance.
- **Executor = Sonnet** (validated) does the actual implementation; the guards backstop the
  gpt-chat *root*. Confirmed this session: the non-Claude executor already runs on the OpenAI Agents
  SDK with full tool parity — so a stub is model-bound, which is exactly why executor=Sonnet is the
  lever, not a harness change.

## 3. Procedure

**Mechanical setup (VM start, preflight, OAuth) — follow the canonical runbook:**
`docs/runbooks/2026-06-16-sdar-on-gcp-a100-vm.md`. Key points: VM `sdar-a100-8g`
(`a2-highgpu-8g` = 8×A100-40GB, us-central1-c, SPOT, 500GB disk) is **TERMINATED** — start it via
`scripts/gcp_sdar_preflight.sh start`; run `scripts/gcp_sdar_preflight.sh prepare` (warms HF
weights + ALFWorld/WebShop/Search-QA, writes `runs/.cache/sdar_gcp.env`, verifies 8 GPUs);
`scripts/gcp_sdar_preflight.sh check` must be GREEN before any paid run.

**Executor auth (the one new prerequisite):** the Sonnet executor needs a Claude credential on the
VM — `claude setup-token` → export `CLAUDE_CODE_OAUTH_TOKEN` (long-lived, headless-safe; see memory
`reference_claude_setup_token_headless`). The gpt-chat-latest root/grader/verifier use
`AZURE_FOUNDRY_*` (already in `.env`). Unset any no-credit `ANTHROPIC_API_KEY`.

**Launch (root = gpt-chat-latest, executor = Sonnet, guards ON):**

```bash
# on the VM, after `prepare` + `check` GREEN + claude setup-token
AZURE_FOUNDRY_DEPLOYMENT=gpt-chat-latest \
OPENRESEARCH_SDAR_MODELS=executor=sonnet,grader=foundry,verifier=foundry \
OPENRESEARCH_ARG_CONTRACTS=1 \
OPENRESEARCH_STUB_METRICS_GUARD=1 \
OPENRESEARCH_GRADER_SAMPLES=3 \
OPENRESEARCH_EVIDENCE_GATE=1 \
OPENRESEARCH_SDAR_PROJECT_ID=sdar_gcp_gptchat_guarded_v6_20260619 \
bash scripts/gcp_sdar_preflight.sh launch
```

Start **smallest-two** (Qwen3-1.7B + Qwen2.5-3B) — export the runbook §5
`OPENRESEARCH_BASELINE_EXTRA_GUIDANCE` scope string before launch; drop it for the full 3-model
matrix once smoke passes.

**Alternatives:**
- **executor = gpt-5** instead of Sonnet: `OPENRESEARCH_SDAR_MODELS=executor=gpt-5,grader=foundry,verifier=foundry` + a live `OPENAI_API_KEY`.
- **Cheapest harness-proof ($0 LLM, all-Sonnet):** runbook §5 `--model claude-oauth` with
  `OPENRESEARCH_ARG_CONTRACTS=1 OPENRESEARCH_STUB_METRICS_GUARD=1` exported — proves the GPU/scope
  path without the foundry root.

## 4. What to watch (the new guard signals)

`tail -f runs/<id>/code/.exec_live.log` (live training) + the `run_warning` codes in
`runs/<id>/dashboard_events.jsonl`:

- `fabrication_suspected` (G2) / `arg_contract` (G1) — **expected to fire only if** the executor
  stubs / the root passes a placeholder, each followed by a **re-drive** (NOT a finalize). On a
  healthy Sonnet-executor run they should be absent on the final result.
- `forced_iteration` — fine in moderation (it's blocking premature finalize); a
  `root_degenerate_refusal_loop` abort is a FAILURE of the run.
- The win condition: the root recognizes any weak `run_experiment` result and re-drives
  `implement_baseline` instead of shipping it.

## 5. Optional — Tier-B A/B (gates flipping the guard defaults ON)

```bash
.venv/bin/python scripts/rlm_root_ab.py --paper 2605.15155 --trials 3
```
Runs guards-off vs guards-on arms and writes `runs/_ab/<key>/root_ab_report.{json,md}`. Combine
with the repo's ≥3-paired-SDAR-run rule before changing any guard default from OFF.

## 6. Cost + cleanup

8×A100-40GB SPOT bills **only while RUNNING**. Stop the VM the moment the run finishes or you step
away:

```bash
scripts/gcp_sdar_preflight.sh stop        # or: gcloud compute instances stop sdar-a100-8g --zone us-central1-c
```

VM is **currently STOPPED** (TERMINATED). No GKE clusters; no Azure VMs/VMSS/AKS running. Residual
cost is only the 500GB boot disk (preserves the v5 run evidence) — left intact deliberately.

## 7. Pointers

- Canonical procedure: `docs/runbooks/2026-06-16-sdar-on-gcp-a100-vm.md`
- Prior run + deferred lever: `docs/local/2026-06-19-kimi-sdar-run-handoff.md` (local working note)
- Guardrails design + status: `docs/superpowers/plans/2026-06-19-gptchat-rlm-root-optimization.md`,
  memory `reasoning-chat-root-guardrails`, `foundry-gptchat-root-not-executor`
- SDAR baseline (scope, invariants): `docs/runbooks/2026-05-23-sdar-baseline-handoff.md`
