<!-- doc-meta: status=live-journal; last-verified=2026-07-22 -->
# Tier-3 + ADAM A/B — live progress journal

> **Live journal, updated ≥ every 30 min while this effort is active** (see the GCP-run
> journaling rule in `backend/services/runtime/CLAUDE.md`). Newest entry on top. Each entry
> is append-only; never rewrite history — correct later entries instead.
>
> **Scope:** Tier-3 program (base pipeline green → scheduler applied → ADAM fixed-budget A/B
> on GCP single-VM, no GKE). Design spec:
> `docs/superpowers/specs/2026-07-22-tier3-scheduler-adam-ab-design.md`.

### 2026-07-23 07:3x UTC — QUIESCENT status check: no run active, VM TERMINATED, docs consistent; effort paused at the precisely-characterized Phase-C blocker
- **Progress:** Post-terminal housekeeping done. Confirmed ground truth: `gcloud compute instances list` shows the
  ONLY instance `adam-tier3-vm` = **TERMINATED** (no stray running VMs); no active campaign/reproduce process; no
  background poller/monitor/workflow running (all exited on the treat3 MILESTONE last turn). Docs brought consistent
  across the three status-bearing surfaces: this journal + the design spec
  (`docs/superpowers/specs/2026-07-22-tier3-scheduler-adam-ab-design.md`, new STATUS banner, last-verified→07-23) +
  memory `tier3-phaseab-scheduler-applies-2026-07-22` (model-swap hypothesis recorded as tested-live-and-rejected).
- **Worked:** end-to-end system exercised with all tools on the live GCP A100 path — RLM pipeline + ASHA scheduler
  (spawn→claim→launch) + Azure/`gpt-chat-latest` reasoning credential + evidence receipt gate. Fail-closed behavior is
  correct (red line held live, twice). Artifacts preserved in `runs_logs/treat3_gpt_20260723/` (scheduler actions,
  tokens_total, metrics).
- **Failures/stalls:** the Phase-C live A/B remains BLOCKED — no verified receipt was ever produced because the branch
  reproduction never emits `checkpoints/step_*`. Root cause is NOT model quality (gpt-chat-latest ran the full loop per
  `runs_logs/treat3_gpt_20260723/tokens_total.json`: 289,870 in / 5,089 out, 39 calls) and NOT a config typo — it is
  the model authoring a monolithic scaffold trainer instead of the cell-matrix contract. Error surfaced as
  `campaign_error:SchedulerRuntimeError` from the receipt producer (`backend/agents/rlm/scheduler_receipt_producer.py`
  `build_raw_receipt` → `cell_checkpoint.latest_checkpoint_dir(...)==None`). Secondary: that exception propagates
  through `_cohort_loop` and aborts the whole campaign (only 5 scheduler actions ever written — never launched
  branches 2-4 / attempts 2-3).
- **Infra/GCP:** 0 running instances (cost capped). No open pollers/agents. Nothing to tear down.
- **Cost:** ledger $0 for the run (fail-closed pre-receipt; foundry spend ledger-blind). Real LLM use =
  289,870+5,089 tokens gpt-chat-latest (per `tokens_total.json`). Session A100 total ~1.5 h ≈ $5-6; now $0/hr (stopped).
- **Improvements landed this session:** confirmed the reasoning-model transport needs NO code change
  (`openai_client._is_reasoning_model`→`max_completion_tokens`, verified via live smoke test); docs/memory realigned so
  the recorded blocker no longer misattributes the gap to executor credential.
- **Next action (future session, no work in-flight now):** implement HARNESS-FORCED checkpoint emission so any model
  necessarily produces `checkpoints/step_*` → verified receipt → first real freeze/promote A/B: either scaffold a
  `train_cell.py` stub wired to `cell_checkpoint.write_checkpoint(...)` at the rung steps, or enable
  `OPENRESEARCH_GKE_SYNTH_CELL`. Optionally harden `_cohort_loop` for per-branch fail-isolation. 12 commits remain
  UNPUSHED (push only when the operator asks).

### 2026-07-23 07:0x UTC — `adam_treat3_gpt` TERMINAL: EXHAUSTED/fail-closed — gpt-chat-latest RAN (confirmed via token ledger) but STILL didn't emit the checkpoint contract → model-swap hypothesis NOT borne out; gap is harness-forced checkpointing, not model choice. VM STOPPED.
- **Result:** Cohort `adam_treat3_gpt_20260723` reached TERMINAL at 07:02 (poller bib1oatp0 poll 2): **EXHAUSTED,
  `campaign_error:SchedulerRuntimeError`**, 0 receipts, no checkpoints — the SAME fail-closed shape as treat2_grok.
  BUT the diagnostics make this a MORE informative result than grok's:
  1. **gpt-chat-latest was genuinely the executor** — `runs_logs/treat3_gpt_20260723/tokens_total.json` shows
     `gpt-chat-latest: 289,870 input / 5,089 output` across **39 primitive calls** (rlm_root 266k in, propose_improvements,
     plan_reproduction, etc.). The Azure-key reasoning-model routing (root via OpenAILlmClient `max_completion_tokens`;
     no code change) worked end-to-end on the live GCP path. `run_config.model=grok` is only the credential label; the
     served deployment was gpt-chat-latest per the ledger.
  2. **It STILL did not emit the 5-component checkpoint contract.** Branch `code/` has the scaffold helpers
     (`cell_checkpoint.py`, `cell_scheduler.py`, `gpu_cell_runner.py`) but **no `train_cell.py`, no `cells.json`, no
     `checkpoints/` dir, no `step_*`**. It wrote a monolithic `train.py` and a template `metrics.json`
     (`{"return":2.0, per_dataset MNIST/IMDB/CIFAR}` — leftover SDAR-workspace scaffold, NOT a real ADAM `train_loss`).
     So `cell_checkpoint.latest_checkpoint_dir(<branch>/code/checkpoints)` returned None and the receipt producer
     FAIL-CLOSED with `SchedulerRuntimeError` (evidence-not-grade red line HELD, live, again).
  - **Decisive finding:** swapping grok→gpt-chat-latest (a reasoning model that follows the contract in an isolated
    preflight prompt) did NOT make the *autonomous full-pipeline* run emit the contract. Guidance-in-the-prompt alone
    is insufficient; the reproduction reused the SDAR scaffold and never built the cell-matrix trainer. **The real gap
    for a live scheduler A/B is harness-FORCED checkpoint emission** — a pre-scaffolded `train_cell.py` the model only
    fills in, or the `OPENRESEARCH_GKE_SYNTH_CELL` synthesis path — not a better model.
- **Worked:** full scheduler machinery again proven live (4×`branch_registered` + `launch_claimed`, real isolated
  branch reproduction); Azure/gpt-chat-latest credential + reasoning transport validated end-to-end on GCP; fail-closed
  is correct (not a crash). Artifacts saved to `runs_logs/treat3_gpt_20260723/` (scheduler actions, tokens_total,
  metrics).
- **Failures/stalls:** the awaited checkpoint milestone did NOT occur — root cause is the model authoring a monolithic
  scaffold trainer instead of the cell-matrix contract; no `train_cell.py` produced despite `BASELINE_EXTRA_GUIDANCE`.
- **Infra/GCP:** `adam-tier3-vm` **TERMINATED** (verified `status=TERMINATED`) — cost capped. Poller bib1oatp0 exited
  on MILESTONE; monitoring loop ENDED (no reschedule).
- **Cost:** ledger $0 (fail-closed pre-receipt; foundry spend ledger-blind — real LLM use was 289,870+5,089 tokens
  gpt-chat-latest per the run's tokens_total). Session A100 ~30 min this run; cumulative ~1.5 h ≈ $5-6.
- **Next action (for a future session):** to get the FIRST real freeze/promote A/B, force the checkpoint contract at
  the harness level (scaffolded `train_cell.py` stub wired to `cell_checkpoint.write_checkpoint` at rung steps, or
  enable `OPENRESEARCH_GKE_SYNTH_CELL`) so any model necessarily emits `checkpoints/step_*` → verified receipt →
  scheduler decide_rung. Model choice is NOT the lever; the emission scaffold is.

### 2026-07-23 06:4x UTC — NEW RUN `adam_treat3_gpt` (gpt-chat-latest via Azure key): scheduler-ON, the checkpoint-emission test — LIVE, branch reproducing
- **Progress:** Launched the decisive follow-up to treat2's fail-close: same full scheduler-ON machinery, but the
  executor/all-roles model is now **`gpt-chat-latest`** (Azure Foundry deployment on the operator's standing Azure key)
  instead of grok — a reasoning model that (preflight-verified) follows the 5-component checkpoint contract verbatim,
  which grok would not. Project `adam_treat3_gpt_20260723` (campaign pid 1518), `--sandbox local --campaign-driver
  live --root-model grok` + `OPENRESEARCH_ROLE_MODELS={executor,verifier,grader:grok}` all resolving to the foundry
  endpoint (`AZURE_FOUNDRY_DEPLOYMENT=gpt-chat-latest`), scheduler TREE+AUTHORITATIVE ON, `--authority-spec
  configs/adam_authority_spec.json` (4 branches, rungs 100/300/1000), caps `--max-llm-usd 15 --max-gpu-usd 12
  --max-gpu-hours 3 --max-attempts 3`. **Ground-truth (live SSH, campaign etime 06:39):** `state=attempt_loop
  terminal=None`; scheduler already fired **5 actions** ending at `launch_claimed` for `adam-ambiguity-beta1-0p9`; its
  per-branch reproduction (pid 1520, distinct `__adam_ambiguity_beta1_0p9` project) is deep in the RLM loop (rubric
  generated, `iterations/`, `code/`, `environment_spec.json` on disk). `scheduler_receipts/`=0, **no `checkpoints/step_*`
  yet** (GPU 0% — understand/plan phase, same pre-training pattern).
- **Worked:** gpt-chat-latest confirmed live on the Azure key and follows the checkpoint instruction verbatim in
  preflight; the reasoning-model transport is **already supported in code** (`openai_client._is_reasoning_model` →
  `max_completion_tokens`; executor Agents-SDK path native) — **no code change needed**, verified by real-client smoke
  test. Scheduler spawn→claim→launch firing identically on the live path with the new model.
- **Failures/stalls:** none. The decisive moment is upcoming and identical to before: when branch 1 finishes, does
  gpt-chat-latest **emit `checkpoints/step_*`** (→ verified receipt → scheduler decide_rung/freeze/promote, the first
  real live A/B leg) or fail-close at receipt like grok? This run exists to answer exactly that.
- **Infra/GCP:** 1 A100 RUNNING (us-central1-f). Poller `beavyjjuc` **exited** — it was tracking the OLD `treat2_grok`
  run (it reported that run's `EXHAUSTED` at poll 11, NOT treat3); re-arming a fresh poller pointed at treat3 that
  breaks on `checkpoints/step_*` / receipts>0 / freeze|promote|revive|receipt_verified / terminal.
- **Cost:** foundry LLM spend is **ledger-blind** (`cost_ledger.jsonl`=$0 is not $0); A100 ~$3.67/hr while running;
  token truth lands in the branch's `tokens_total.json` at completion. Cumulative session A100 well under the caps.
- **Next action:** poll to the checkpoint/receipt milestone (or fail-close); on terminal, pull
  `scheduler_tree_actions.jsonl` + receipts + reports to `runs_logs/`, determine freeze/promote-vs-fail-close, **STOP
  the A100** to cap cost, journal + memory + notify.

### 2026-07-23 06:2x UTC — FULL SYSTEM RUN COMPLETE: launch fix works, scheduler engaged, fail-closed at checkpoint (as designed) + start.sh UI bug FIXED
- **Progress (result of the fully-built-out run):** Treatment cohort `adam_treat2_grok_20260723` reached TERMINAL at
  06:22 (poll 11): **EXHAUSTED, `campaign_error:SchedulerRuntimeError`**, 0 receipts, ledger $0. This is the PREDICTED,
  CORRECT outcome — the complete pipeline ran end-to-end:
  1. Launch-adapter fix (382bae7b) WORKED — the scheduler claimed `adam-ambiguity-beta1-0p9` and ran a REAL branch
     reproduction in its own isolated run dir (grok wrote `metrics.json` + `final_report.json`).
  2. At receipt time, `cell_checkpoint.latest_checkpoint_dir(<branch>/code/checkpoints)` returned None — **verified:
     NO `checkpoints/` dir exists** in the branch (grok's shallow monolithic `train.py` never called
     `cell_checkpoint.write_checkpoint`, despite the guidance). So the checkpoint fix (d52777aa) FAIL-CLOSED with
     `SchedulerRuntimeError` — refusing to fabricate a receipt from a missing checkpoint (evidence-not-grade red line
     HELD, live). Campaign terminated EXHAUSTED.
  - **Net:** scheduler LOGIC + launch adapter proven live; the sole remaining gap for a real A/B is empirically
    confirmed = the reproduction must EMIT the 5-component checkpoint. grok doesn't → needs a validated executor
    (gpt-5/Sonnet) or harness-forced checkpoint emission.
- **UI BUG FIXED (operator: "fix the ui, never face this again"):** `start.sh:186` used `wait -n` (bash 4.3+), but
  macOS ships bash 3.2 → under `set -euo pipefail` it failed instantly ("wait: -n: invalid option"), tearing down both
  servers before they served. Replaced with the bash-3.2-safe `kill -0` poll watchdog (mirrors the pattern
  `scripts/dev.sh:244` already uses) + reap-the-dead-child for its exit code. Syntax-checked under `/bin/bash` 3.2;
  live-verified: `./start.sh` now boots backend+frontend and STAYS UP (watchdog blocks correctly), serving the UI incl.
  detail views. TO COMMIT.
- **Worked:** the fully-built system executed the whole loop (spawn→claim→launch→reproduce→receipt-gate); fail-closed
  is CORRECT behavior, not a crash of the scheduler; UI now launches cleanly on macOS.
- **Failures/stalls:** none unexpected — the SchedulerRuntimeError IS the designed fail-closed. Leftover "backend.cli"
  proc in the check was my own SSH bash, not a lingering campaign.
- **Infra/GCP:** `adam-tier3-vm` **TERMINATED** (cost capped). Stale pollers b088yr5s0/beavyjjuc finished.
- **Cost:** treatment run ledger $0 (fail-closed before receipts; grok reproduction tokens ledger-blind). Cumulative
  session A100 ~65 min ≈ $4-5.
- **Next action:** commit the start.sh fix; the 30-min monitoring wakeup will fire once more, see terminal, and end.
  A real live scheduler A/B remains gated on a checkpoint-emitting reproduction (validated executor credential).

### 2026-07-23 06:4x UTC — Cohort running branch 1 (ambiguity); scheduler at launch_claimed, awaiting first receipt
- **Progress:** Treatment cohort `adam_treat2_grok_20260723` healthy and advancing. The scheduler claimed
  `adam-ambiguity-beta1-0p9` first; its reproduction (child `reproduce … --project-id …__adam_ambiguity_beta1_0p9`)
  is in the grok understand→implement phase (GPU idle 0% — same pre-training pattern as the control run). Scheduler
  actions still at seq 5 (`launch_claimed`); `campaign/scheduler_receipts/` = 0 (expected — branch not finished).
  campaign pid 1459 alive; per-branch run dir isolated on disk.
- **Worked:** the fixed cohort→driver adapter is stable ~8 min into a live branch reproduction; no crash.
- **Failures/stalls:** none. The decisive moment is upcoming: when branch 1 finishes, does it emit a 5-component
  checkpoint (→ verified receipt → scheduler decide_rung/freeze/promote) or fail closed at receipt (the known gap)?
- **Infra/GCP:** 1 A100 RUNNING (us-central1-f); stale control-run poller `b0xxpdm7r` finished (harmless); active
  poller `beavyjjuc` tracks scheduler action count + terminal.
- **Cost:** live grok reproduction accruing under `--max-llm-usd 10`; GPU ~$3.67/hr while running. Ledger-blind;
  token truth at completion in the run's `tokens_total.json`.
- **Next action:** keep polling; on the first receipt/decide (or fail-closed), capture the scheduler_tree_actions +
  per-branch outcome, report, and STOP the VM at cohort terminal to cap cost.

### 2026-07-23 06:3x UTC — FULL SYSTEM RUNNING: launch fix VALIDATED LIVE; cohort now running real branch reproductions
- **Progress:** Operator: "run the fully built out system." Re-synced current code (382bae7b, with the launch fix) to
  the VM, restarted it (us-central1-f RUNNING), and launched the FULL treatment arm `adam_treat2_grok_20260723`
  (scheduler ON: `OPENRESEARCH_SCHEDULER_TREE=1` + `OPENRESEARCH_SCHEDULER_AUTHORITATIVE=1` + `--authority-spec
  configs/adam_authority_spec.json` + grok all roles + checkpoint-emit guidance + `OPENRESEARCH_CELL_RESUME_AUTO=1`;
  caps `--max-llm-usd 10 --max-gpu-usd 10 --max-gpu-hours 2 --max-attempts 3`). **The launch-adapter fix WORKS live:**
  the run got PAST the prior AttributeError crash — the scheduler registered all 4 branches, claimed
  `adam-ambiguity-beta1-0p9`, and spawned a REAL per-branch reproduction with the exact distinct project_id my fix
  produces: `reproduce 1412.6980 --project-id adam_treat2_grok_20260723__adam_ambiguity_beta1_0p9 --model grok` (pid
  1464), in its own run dir (no collision). campaign pid 1459, state=attempt_loop.
- **Worked:** cohort→LiveCliDriver adapter now functions end-to-end on a real run; per-branch project_id/run-dir
  isolation confirmed on disk. Scheduler at `launch_claimed` (seq 5); next actions (receipt→decide_rung→freeze/promote)
  fire after this branch finishes (~15-20 min grok).
- **Failures/stalls:** none yet. WATCHING for the known checkpoint/receipt gap — if grok's trainer doesn't emit the
  5-component checkpoint, `build_raw_receipt` will fail closed at receipt time (correctly). Poller `beavyjjuc` tracks
  scheduler action count + terminal.
- **Infra/GCP:** 1 A100 RUNNING (us-central1-f), 2.5h backstop; billing ~$3.67/hr while the cohort runs (intended).
- **Cost:** live run under `--max-llm-usd 10`/`--max-gpu-usd 10`; grok ledger-blind (truth = the run's
  `tokens_total.json` at completion). Cumulative session A100 climbing (~$4-6 est by completion).
- **Improve/Next action:** monitor the cohort — capture whether the scheduler FREEZES/PROMOTES from real receipts
  (full live A/B demonstration) or fail-closes at the checkpoint gap; pull the scheduler_tree_actions + any per-branch
  scores; STOP the VM at terminal to cap cost.

### 2026-07-23 06:1x UTC — Quiescent: session banked, awaiting operator (idle checkpoint)
- **Progress:** No new work since the 06:0x entry — session is at a clean stopping point per the operator's
  "land launch fix, bank result" choice. This is a heartbeat/idle checkpoint (cron cadence).
- **State verified now:** GCP `adam-tier3-vm` **TERMINATED** (us-central1-f, reusable, $0 GPU). No active
  workflow/agent/GCP run. Git HEAD `382bae7b` (+ `d52777aa`, `0746590e`) — 3 commits, **unpushed**, on
  `integrate/degke-runpod-on-trunk`. Tests re-checked green: `test_campaign_cohort_loop` + `test_scheduler_runtime`
  = 12 passed. Untracked: `docs/progress/`, `runs_logs/`, `docs/superpowers/specs/2026-07-22-tier3-scheduler-adam-ab-design.md`.
- **Worked / Failures / Stalls:** none new (idle).
- **Infra/GCP:** clean — 1 VM, terminated. No stray clusters/instances.
- **Cost:** $0 since last entry. Session cumulative A100 ~50 min ≈ $3–4; grok LLM modest (ledger-blind; token truth in
  `runs_logs/adam_ctrl_grok_20260723/tokens_total.json`).
- **Next action (unchanged, operator-gated):** to reach a real live scheduler A/B, supply a validated executor
  credential (gpt-5/Sonnet — current keys 401) for a deeper, checkpoint-emitting reproduction; then re-run the
  treatment arm (launch adapter now fixed). Optional: push the 3 commits; land the `_code_tree_bytes`
  `checkpoint_components` name nit.

### 2026-07-23 06:0x UTC — Launch-adapter bug FIXED + committed (382bae7b); progress banked per operator choice
- **Progress:** Operator chose "land launch fix, bank result." Fixed the cohort-loop live launch adapter:
  `reproduction_campaign.py:1055` now passes `payload["launch_payload"]` (a per-branch `AttemptDirectives` OBJECT
  with a distinct `project_id=<base>__<branch_id>` so branches don't collide) to `stages.launch`, instead of the raw
  dict — mirroring the serial `_loop:1277`. `_cohort_launch_payload` rewritten to build that branch directives via
  `dataclasses.replace` (guarding the `is_safety_bracket`/`branch_type` `__post_init__` contract). Commit `382bae7b`.
- **Worked:** the test stub was tightened from reading a dict to asserting a real directives object
  (`assert not isinstance(x, dict) and hasattr(x, "paper_ref")`) — a REGRESSION GUARD proven live: reverting only the
  line-1055 fix fails 2/3 cohort tests. Validation: 55 passed (targeted) / 481 passed (`-k cohort|scheduler|authority|
  campaign`) / ruff clean; re-verified locally (37 passed). Downstream check: `await_result(dict(handle))`/`assess`
  are NOT new crashes — `handle` is a dict on the serial path too (`_launch_impl`=`asdict(...)`), so they're symmetric;
  `launch` was the ONLY object-consumer being fed a dict. No second cohort-loop crash remains.
- **Failures/stalls:** two implementer subagents hit infra stream-idle timeouts; first left only a stray runbook edit
  (reverted), second (fresh, full investigation pre-loaded) completed cleanly. No code lost.
- **Infra/GCP:** `adam-tier3-vm` TERMINATED (us-central1-f, reusable). No live run. GCP fully idle.
- **Cost:** $0 this segment (all local code work). Session A100 total ~50 min ≈ $3-4; grok LLM modest (ledger-blind).
- **Improve/Next action (banked handoff):** the ONLY remaining blocker to a real live A/B is that each branch cell's
  trainer must emit the 5-component checkpoint (`checkpoints/step_N/*`) + pinned manifests for a verified receipt —
  grok's shallow monolithic `train.py` doesn't, so treatment fail-closes at receipts (correctly). Needs either a
  validated executor credential (gpt-5/Sonnet — current keys 401) that reliably emits the cell route + checkpoints,
  or harness work to force checkpoint emission. Scheduler logic itself is DONE + proven (hermetic + live-fired).
  Follow-up nit: `scheduler_receipt_producer._code_tree_bytes` still excludes the stale name `checkpoint_components`.

### 2026-07-23 05:3x UTC — TREATMENT ARM: scheduler FIRED LIVE (4 branches spawned + launch claimed), then a launch-adapter bug
- **Progress (big):** Operator: "test that now with EVERYTHING we implemented." Ran BOTH proofs of the freezing:
  **(1) HERMETIC** — `tests/rlm/test_campaign_cohort_loop.py` all 3 pass, incl.
  `test_cohort_loop_drives_claim_receipt_decide_apply` (a real `campaign.run()` that freezes the underperformer,
  promotes the others, revives the frozen one from VERIFIED receipts). **(2) LIVE on GCP A100** — synced current code
  (d52777aa) + launched `adam_treat_grok_20260723` with `OPENRESEARCH_SCHEDULER_TREE=1` +
  `OPENRESEARCH_SCHEDULER_AUTHORITATIVE=1` + `--authority-spec configs/adam_authority_spec.json`. The **scheduler
  ENGAGED end-to-end**: `scheduler_ladder.json` + `scheduler_tree_state.json` + `scheduler_tree_actions.jsonl`
  written; the controller **registered all 4 branches** (`branch_spawned` seq 1-4: adam-faithful-lr1e-3[safety],
  adam-ambiguity-beta1-0p9, adam-discovery-lr1e-4, adam-discovery-eps1e-8, queued at rung 0) and **atomically
  claimed a launch** (`launch_claimed` seq 5 → adam-ambiguity-beta1-0p9, state=launching). The ASHA authority
  machinery works on a real run.
- **Failure (root-caused, file:line):** right after the claim the campaign crashed `campaign_error:AttributeError`,
  `$0` spent (before any reproduction). **Root cause:** `reproduction_campaign.py:1046` `_cohort_loop` calls
  `self.stages.launch(payload)` passing the WHOLE dict, but `stages.launch(X)`→`driver.launch(X)` and the real
  `LiveCliDriver` attribute-accesses a **directives object** (`directives.paper_ref`, via `build_reproduce_argv`) →
  `AttributeError: 'dict' has no attribute ...`. The serial `_loop:1277` correctly passes `planned["launch_payload"]`
  (the directives object, `campaign_composition.py:770`). The cohort path was ONLY ever tested against a stub
  `_CohortStages.launch` that read the dict — so the live-driver adapter was never exercised. Also needs per-branch
  `project_id`/run-dir (like width's `launch_project_id`) so branches don't collide.
- **Worked:** the scheduler-logic layer is proven twice (hermetic + live spawn/claim). Traceback not persisted (campaign
  catches → records only the type); root-caused from the code + the tree-actions log.
- **Stalls/bottlenecks:** the live cohort→LiveCliDriver adapter is under-exercised; fixing to a full live A/B may be an
  iterative debug loop (fix launch → likely surface the next stub-vs-real gap). VM STOPPED (crash was $0; no idle bill).
- **Infra/GCP:** `adam-tier3-vm` TERMINATED. Both control (0.203) + treatment (crashed) artifacts on the VM +
  control pulled to `runs_logs/`.
- **Cost:** treatment run $0 (crashed pre-reproduction). Cumulative A100 ~50 min ≈ $3-4; grok LLM modest.
- **Improve/Next action:** fix `_cohort_loop` launch adapter — build a per-branch directives object (branch
  project_id/run-dir + seed/branch_type/is_safety_bracket via `dataclasses.replace`) and pass `payload["launch_payload"]`
  to `stages.launch` (mirror serial:1277); update the stub test to pass a directives OBJECT (regression guard); keep
  serial byte-identical. Then re-run the live treatment arm and see how much further the cohort gets.

### 2026-07-23 05:2x UTC — CONTROL ARM COMPLETE: score 0.203 (weak/shallow); VM stopped; A/B feasibility re-assessed
- **Progress:** The scheduler-OFF control arm (`adam_ctrl_grok_20260723`) finished: campaign `state=terminal`,
  `demo=completed`, `final_report.json` written and pulled to `runs_logs/adam_ctrl_grok_20260723/`. **Result:
  overall rubric 0.203 / target 0.60, verdict INCONCLUSIVE** (impl=partial, repl=inconclusive; 22/22 PaperBench-bundle
  leaves graded). Models all grok (`executor/verifier/grader = azure-foundry:grok-4.3`). **VM STOPPED** (TERMINATED)
  immediately after to cap cost.
- **Why weak (from final_report.md weakest leaves):** grok's `train.py` implemented ONLY MNIST logistic regression
  (Adam/Adagrad/RMSProp/SGD-Nesterov/Adadelta, 1 epoch) — MISSING the IMDB BoW model, the MLP (hidden layers/dropout),
  the CIFAR-10 CNN, a real `adam.py` (Algorithm 1 — it just used `torch.optim.Adam`), and figure generation. Several
  leaves 0.00 incl. "fabricated IMDB keys". Only 1 `baseline-implementation` call + 2 experiment runs; `--max-attempts 2`
  and `--max-llm-usd 6` bounded it. So it's a thin slice, not a faithful full reproduction.
- **Worked:** END-TO-END GCP reproduction pipeline now proven — the whole original blocker is cleared. Grok/Foundry
  drove understand→implement→run→grade cleanly; real rubric grades produced; artifacts pulled; VM lifecycle clean.
- **Failures/notes:** report title "Unknown Paper" (paper-name resolution miss, cosmetic). Token oddity:
  `by_primitive.baseline-implementation` attributed to `gpt-4o` (113,838 in / 1,344 out) despite executor=grok — likely
  a token-counter mislabel (the dead OpenAI key can't serve gpt-4o); grok-4.3 = 408,752 in / 4,187 out. Not blocking.
- **A/B feasibility (honest):** with grok + these bounds, the reproduction is shallow (~0.20). The scheduler promotes
  good BRANCHES; it cannot manufacture reproduction quality, so a grok treatment arm would likely also land ~0.20 →
  A/B delta ≈ noise. WORSE: a REAL scheduler-ON run needs each branch cell's trainer to emit the 5-component checkpoint
  (`cell_checkpoint.write_checkpoint`→`checkpoints/step_N/*`) + pinned manifests, or `build_raw_receipt` fails CLOSED
  (now correctly, post line-1081 fix) — grok's shallow `train.py` almost certainly does NOT, so a treatment run would
  likely fail-closed at receipt production. The credential wall (only grok valid; gpt-5/Sonnet 401) caps quality.
- **Infra/GCP:** `adam-tier3-vm` TERMINATED (us-central1-f, reusable). No other GCP.
- **Cost:** A100 cumulative ~45–50 min running ≈ **$3–4** (setup + dead-key run + control). LLM = grok via Foundry,
  **LEDGER-BLIND**; token truth = `runs_logs/adam_ctrl_grok_20260723/tokens_total.json` (grok 408.8K in/4.2K out +
  gpt-4o-labeled 113.8K in/1.3K out); USD via the Foundry portal (grok-4.3 pricing), modest (~$1–2 est).
- **Improve/Next action:** decision point for the operator — (A) run treatment-as-pipeline-validation now (expect
  fail-closed at receipts or ~0.20, documenting the exact final gap), (B) accept control-baseline + treatment-code-landed
  as the deliverable and defer a meaningful A/B until a validated-executor credential (gpt-5/Sonnet) exists, or
  (C) strengthen the control reproduction first (more attempts/budget/guidance) for a solid baseline. Also: the
  code_tree_bytes `checkpoint_components` exclusion-name follow-up.

### 2026-07-23 05:1x UTC — Treatment-arm recon done (in parallel while control trains); real scope clarified
- **Progress:** While the grok control arm trains, ran a READ-ONLY recon subagent to map the scheduler-ON treatment
  arm exactly. Findings: **(a)** `SchedulerAuthoritySpec` JSON schema is clear — `ladder{paper_ref,metric_id,
  direction,r_max_steps,rung_steps,schedule_source_sha256}` + `width{gpu_usd_budget,a100_cap,discovery_gpu_usd_budget,
  eta,noise_floor}` + `branches[{branch_id,branch_type,hypothesis_fingerprint(64hex),seed,is_safety_bracket}]`, loaded
  via `scheduler_runtime.load_authority_spec(path, paper_ref)`; **(b)** flags = `OPENRESEARCH_SCHEDULER_TREE=1` +
  `OPENRESEARCH_SCHEDULER_AUTHORITATIVE=1` + `--authority-spec <path>` (routes `build_campaign`→controller→
  `_cohort_loop`); **(c)** GAP #2 is a REAL bug — `reproduction_campaign.py:1081` hardcodes
  `checkpoint_components_dir=cell_out/"checkpoint_components"` (a CPU-stub layout) instead of resolving a real cell's
  checkpoint via `cell_checkpoint.latest_checkpoint_dir(cell_out/"checkpoints")` (trainer writes to
  `OPENRESEARCH_CELL_CHECKPOINT_DIR=<out>/checkpoints/step_<N>/{model,optimizer,lr_scheduler,rng,data_order}`,
  `gpu_cell_runner.py:804`) — 2-line fix + test.
- **Real scope surfaced (honest):** the full treatment run additionally requires EACH branch cell's (LLM-generated)
  trainer to actually EMIT the 5-component checkpoint + pinned `metrics.json`/`dataset_manifest.json`/`run_spec.json`,
  else `build_raw_receipt` fails closed (evidence-not-grade red line — no receipt, no ASHA decision). The harness copies
  `cell_checkpoint.py` into `code/` + the implementer guidance names the 5 components, but whether grok's trainer
  complies is the fragile, largely-unexercised part. So "scheduler ON" is spec + wiring-fix + a real receipt-producing
  reproduction — NOT just spec+2-lines. Matches the Phase-C "gated, none-autonomous" caution.
- **Worked:** recon precise + file:line-cited; two clean wins identified (author `configs/adam_authority_spec.json`;
  fix line 1081) — landing them now via an implementer subagent, TDD + review, without launching the full arm yet.
- **Failures/stalls:** none (recon read-only).
- **Infra/GCP:** control arm (`adam_ctrl_grok_20260723`) training on the A100; poller `b088yr5s0` active.
- **Cost:** unchanged trajectory (control run under `--max-llm-usd 6`; treatment code work is $0 local).
- **Improve/Next action:** land the ADAM spec + the line-1081 checkpoint-wiring fix (committed, tests green); pull the
  control score when it lands; then attempt the treatment arm and honestly report how far the receipt-producing path
  gets (likely surfacing the trainer-emits-real-checkpoints integration as the final gap).

### 2026-07-23 05:0x UTC — Control run was DEAD on a bad OpenAI key; fixed → relaunched on grok, now genuinely RUNNING
- **Progress:** Caught that the first grok... no — the first control run (`adam_ctrl_gcp_20260723`, gpt-5) was NOT
  progressing: `demo_status=queued` for ~8 min with no events/GPU because **credential preflight FAILED** —
  `attempt_1.log`: `OPENAI_API_KEY rejected (HTTP 401): Incorrect API key provided: sk-svcac…`. The key I synced from
  my LOCAL `.env` is invalid/revoked (confirmed: `api.openai.com/v1/models` → 401). Probed all available creds:
  **Anthropic 401, OpenAI(local & VM) 401, but AZURE_FOUNDRY grok-4.3 → HTTP 200** (the docs' OAuth-free "reliable
  root," and what SDAR GCP runs actually used). **Fix:** neutralized the dead `OPENAI_API_KEY`, relaunched both-arms
  standard on **grok** (`--root-model grok`, `OPENRESEARCH_ROLE_MODELS` all grok, `OPENRESEARCH_SKIP_CRED_PREFLIGHT=1`
  since Foundry was independently verified 200) as `adam_ctrl_grok_20260723` (pid 2725). Now: `demo=running`, RLM
  pipeline started ("Workspace ready — Starting agent pipeline"). Both A/B arms will use grok → internally
  consistent comparison (grok isn't SDAR-validated → advisory `root_model_unvalidated` warning; fine for easy ADAM).
- **Worked:** the VM's baked Foundry creds are valid + already known-good for this repo; switching root+roles to grok
  cleared the auth wall; the run advanced past credential into the agent loop.
- **Failures (with locus):** `attempt_1.log` (`runs/adam_ctrl_gcp_20260723/campaign/attempt_1.log`) — cred preflight
  401 on the synced OpenAI key. Root cause: synced a stale/invalid local `OPENAI_API_KEY`. Lesson: validate the LLM
  credential BEFORE launching a billed run (a 30-sec `curl /v1/models` probe would have caught it pre-launch).
- **Stalls/bottlenecks:** ~8 min lost to the dead-key stall (VM idle-billed, no LLM/GPU spend since preflight
  rejects before real calls). Resolved.
- **Infra/GCP:** 1 A100 RUNNING (us-central1-f). Old gpt-5 campaign self-exhausted (2 failed-preflight attempts).
- **Cost:** GPU ~$0.55 setup + ~15 min idle/ingest (~$0.9). LLM: $0 on the dead run (no real calls); grok run now
  accruing under `--max-llm-usd 6` (Foundry spend is LEDGER-BLIND — truth via `tokens_total.json` + Foundry portal).
- **Improve/Next action:** new poller for `adam_ctrl_grok_20260723`; on terminal → pull score, STOP VM. Add a
  pre-launch cred-probe to the runbook. Then treatment arm (grok + scheduler ON + ADAM authority spec + checkpoints).

### 2026-07-23 04:5x UTC — Control arm in attempt_loop (LLM implement phase, pre-training)
- **Progress:** Control run healthy and advancing. Campaign `STATE=attempt_loop` (attempt 1 of 2); child reproduce
  `demo_status=queued`→starting; GPU idle (0% / 0 MiB) — expected: gpt-5 is in the understand→plan→implement phase
  writing the ADAM reproduction code before any training kicks off. No `experiment_runs.jsonl` yet (no training cell
  has run). Ingest fully complete (6/6). Background poller `b0xxpdm7r` polling every 90s for a terminal state.
- **Worked:** run stable ~2 min in; no crashes/import errors; the synced current code drives the campaign fine on
  the VM.
- **Failures:** none. (`demo_status=queued` is normal early-attempt state, not a stall — watching for the
  running→training transition on the next polls.)
- **Stalls/bottlenecks:** none yet. If `demo_status` stays `queued` past ~10 min, would investigate the child
  reproduce subprocess launch.
- **Infra/GCP:** 1 A100 RUNNING (us-central1-f), GPU currently idle (pre-training). 2.5h auto-STOP backstop armed.
- **Cost:** GPU live (~$3.67/hr, ~$0.55 setup + a few min of run). LLM accruing under the `--max-llm-usd 6` cap
  (gpt-5 implement phase); exact figure from the VM's `tokens_total.json` at completion, not the ledger.
- **Improve/Next action:** keep polling; on `final_report.json` pull it + record control-arm rubric score, STOP the
  VM to cap cost, then G3 treatment arm (ADAM `SchedulerAuthoritySpec` + `_cohort_loop` checkpoint wiring → A/B).

### 2026-07-23 04:4x UTC — CONTROL ARM LAUNCHED on GCP A100 and progressing (first real Tier-3 run!)
- **Progress:** Operator chose **GCP A100 (reliable env)**. Restarted VM (us-central1-f, 2.5h backstop), synced current
  code + refreshed key, launched the ADAM control arm. Bootstrap clean: code overlaid (git HEAD `0746590e`),
  `OPENAI_API_KEY` refreshed (1 line), `pip install` no errors, **`import backend.cli OK; torch 2.12.1+cu130 cuda True`**
  (env_pin left default-ON → detected coherent cu130 ≥12.1 and SKIPPED the cu121 reinstall, preserving the CUDA build).
  Campaign **pid 1742** launched detached → `runs/adam_ctrl_gcp_20260723.out`. Confirmed progressing: ingest 1/6→6/6
  done (register→fetch→parse→discover→index→workspace), attempt 1 started (`campaign.json`, `understanding.json`,
  `directives/`, `attempt_1.log` written). This is the FIRST real Tier-3 reproduction run to actually launch.
- **Command:** `campaign 1412.6980 --campaign-driver live --sandbox local --root-model gpt-5 --project-id
  adam_ctrl_gcp_20260723 --max-attempts 2 --max-llm-usd 6 --max-gpu-usd 6 --max-gpu-hours 1.5`; env
  `OPENRESEARCH_ROLE_MODELS={executor,verifier,grader all gpt-5}`, `OPENRESEARCH_GPU_DEVICE_IDS=0`. Scheduler OFF.
- **Worked:** VM-as-host approach validated — the proven local campaign path runs unmodified on the remote A100;
  git-archive-overlay + key-refresh + incremental pip all clean; torch/CUDA intact; harness ingested ADAM fine.
- **Failures:** two earlier scp-loop scripting bugs (pipe masked scp exit code; grep matched the filename inside an
  error string → skipped retry) — cosmetic, fixed by a clean scp; files confirmed on VM (tarball 21M, key 183B).
- **Stalls/bottlenecks:** none — run in flight. Expected wall-clock ~20–60 min (LLM loop: understand→implement→
  train ADAM on MNIST/CIFAR→verify).
- **Infra/GCP:** 1 A100 RUNNING (us-central1-f). Billing ~$3.67/hr WHILE the run is active (intended now).
- **Cost:** GPU cumulative ≈ $0.55 setup + live run (~$2–4 est). LLM bounded by `--max-llm-usd 6`. Truth via
  `gcloud` + the VM's `runs/adam_ctrl_gcp_20260723/tokens_total.json` (check on completion), not the ledger.
- **Improve/Next action:** monitor `runs/adam_ctrl_gcp_20260723.out` + `demo_status.json`; on `final_report.json`
  pull it down + record the control-arm rubric score; STOP the VM immediately after to cap cost; then G3 treatment
  arm (author ADAM `SchedulerAuthoritySpec` + `_cohort_loop` checkpoint wiring, scheduler ON) → A/B scorecard.

### 2026-07-23 03:2x UTC — Operator interrupted launch ("analyze again"); VM STOPPED to deliberate; key re-analysis
- **Progress:** Sync assets staged (`git archive HEAD` tarball 21M at `/tmp/or_src.tgz`; `OPENAI_API_KEY` snippet
  ready, 167 chars). Operator interrupted right before launch and asked to re-analyze → **STOPPED the VM**
  (TERMINATED, no idle A100 billing) to deliberate rather than burn ~$3.67/hr.
- **Key re-analysis (honest):** the "VM-as-host" pivot means **GCP and local now CONVERGE** — both just run
  `campaign … --sandbox local` as subprocesses; the ONLY difference is the host (Mac-CPU-no-CUDA vs GCP-A100-Linux).
  For **ADAM specifically (CPU-tiny MNIST/CIFAR), the A100 provides ZERO compute benefit** — the paper doesn't need a
  GPU. So GCP's real (and legitimate) value here is NOT speed but a **validated Linux+CUDA environment** where the
  harness's GPU-centric machinery is known-good, vs this Mac (darwin/MPS, no CUDA) where env_pin/cell-runner are
  UNVALIDATED and may fight the run. Trade: GCP ≈ $5–10 for a reliable A/B; local = $0 GPU but harness-on-Mac risk.
  Remaining treatment-arm code (ADAM authority spec + `_cohort_loop` checkpoints) is identical on either host.
- **Worked / Failures:** none new (no run launched). VM stop clean.
- **Stalls/bottlenecks:** awaiting operator steer on platform given the convergence insight.
- **Infra/GCP:** `adam-tier3-vm` TERMINATED (us-central1-f, reusable, boot disk persists; ~pennies/day). No other GCP.
- **Cost:** cumulative GPU ≈ $0.55 (G1 + two short RUNNING windows). LLM $0. No run spend.
- **Improve/Next action:** present the convergence re-analysis to the operator; on their steer either (a) restart VM
  + sync + launch control arm on GCP-A100 (reliable), or (b) attempt the control arm locally on the Mac ($0 GPU,
  harness-risk). Either way then do the treatment-arm code + A/B scorecard.

### 2026-07-23 03:0x UTC — VM restarted with 2.5h backstop; about to sync code + launch control arm
- **Progress:** G2 control-arm setup. VM `adam-tier3-vm` (us-central1-f) STOPPED→rescheduled→STARTED, now **RUNNING**
  with `maxRunDuration=9000s` (2.5h) STOP backstop (the original 1h was too tight for the LLM reproduction loop; a
  running instance can't change scheduling, so a stop/reschedule/start cycle was required). Next: overlay current
  code (`git archive HEAD`, HEAD=`0746590e`) + refresh `OPENAI_API_KEY`, then SSH-launch the scheduler-OFF ADAM
  campaign detached.
- **Worked:** set-scheduling to 9000s while TERMINATED; restart in us-central1-f succeeded (no re-stockout).
- **Failures:** `gcloud compute instances set-scheduling … --max-run-duration` **cannot run while RUNNING**
  (`Max run duration cannot be changed while the instance is running`) — must be TERMINATED first. Also an earlier
  `instances update --max-run-duration` printed help (wrong subcommand; the correct one is `set-scheduling`).
- **Stalls/bottlenecks:** none active. A stale local `runs/adam_ctrl_dbg_20260723/demo_status.json` = `queued`
  (never launched, from a pre-pivot debug attempt) — ignore.
- **Infra/GCP:** 1 instance RUNNING (`adam-tier3-vm`, us-central1-f, A100-40GB). No GKE/other resources.
- **Cost:** G1 ≈ $0.40. Current: A100 RUNNING again (~$3.67/hr) — will launch the run immediately to avoid idle.
  LLM $0 so far. (Ledger blind to Foundry/idle; truth = `gcloud` + `tokens_total.json`.)
- **Improve/Next action:** scp `git archive HEAD` tarball → untar over `/home/abheekp/openresearch` → refresh
  `OPENAI_API_KEY` in `.env` → `.venv/bin/python -m backend.cli campaign 1412.6980 --campaign-driver live
  --sandbox local --root-model gpt-5 --max-llm-usd 6 --max-gpu-usd 6 --max-gpu-hours 1.5 --gpu-usd-per-hr 3.67`
  (scheduler OFF) detached; smoke-tail first minutes; poll + pull `final_report.json`.

### 2026-07-23 02:4x UTC — G1 de-risk DONE on A100 (us-central1-f); operator chose GCP; leaner "VM-as-host" plan
- **Progress:** Operator chose **Invest in GCP path**. Ran the G1 de-risk: quota `NVIDIA_A100_GPUS limit=16 usage=0`
  (feasible). First create in **us-central1-b STOCKED OUT** (`STOCKOUT`, GCP suggested `us-central1-f`); retried in
  **us-central1-f → RUNNING**. This also VALIDATES gap-#1's fix: `a2-highgpu-1g` boots the A100-native machine image
  where `g2-standard-8` failed. SSH-inspected the image, then **STOPPED** the VM (TERMINATED, no idle GPU billing;
  boot disk persists, reusable). G1 cost ≈ $0.40.
- **G1 inspection findings (`adam-tier3-vm`, us-central1-f):** user `abheekp`, repo at `/home/abheekp/openresearch`;
  **venv healthy** (Python 3.12.3, **torch 2.12.1+cu130, CUDA True**, A100-SXM4-40GB); `.env` has 7 baked keys (may be
  stale → refresh at run time); `import backend.cli` OK. **Key constraint:** the baked repo is a **non-git snapshot**
  (`fatal: not a git repository`) from ~2026-06-20 → `git pull` impossible; must overlay current code via a
  `git archive HEAD` tarball + incremental `pip install` (torch already coherent cu130 ≥12.1, so env_pin skips it).
- **PLAN PIVOT (leaner, lower-risk, same goal):** do NOT rewrite the SDAR-hardwired `VmComputeProvider` stage/launch.
  Use the A100 VM as a plain remote GPU host and run the **proven local path inside it via SSH**:
  `campaign 1412.6980 --campaign-driver live --sandbox local` (LocalProcessBackend → the VM's A100). Both A/B arms run
  the SAME current code; scheduler OFF = control, scheduler ON = treatment. Avoids the fragile orchestration rewrite;
  the only remaining CODE work is the (unavoidable) treatment-arm ADAM `SchedulerAuthoritySpec` + `_cohort_loop`
  checkpoint wiring. Control arm needs ZERO new code.
- **Worked:** capacity found (us-central1-f), machine-type fix validated, baked venv/torch/CUDA all healthy.
- **Failures:** us-central1-b A100 stockout (worked around by zone).
- **Cost:** G1 ≈ $0.40 (A100 ~7 min). VM now TERMINATED ($0 GPU; ~pennies/day disk). LLM $0.
- **Next action:** G2 control arm — START VM (us-central1-f), overlay current code (`git archive HEAD` tarball),
  refresh `OPENAI_API_KEY`, SSH-launch the scheduler-OFF ADAM campaign detached with hard cost caps
  (`--max-gpu-usd 6 --gpu-usd-per-hr 3.67 --max-gpu-hours 1.5 --max-llm-usd 6`), smoke-tail the first minutes, then
  poll + pull `final_report.json`. Journal ≥ every 30 min while running.

### 2026-07-23 02:0x UTC — ROOT CAUSE found: GCP single-VM path is SDAR-hardwired, cannot run ADAM; local path is the answer
- **Progress:** Read the actual `provision_cpu` / `stage` / `launch` code + `sdar_gcp_run.sh` end-to-end and the
  gcloud state. The `provision_cpu` abort is NOT the real story — there are **two** integration gaps, one fundamental:
  1. **Machine-type ⇄ machine-image mismatch (config):** the only machine image `sdar-mi-20260620` was baked from an
     **`a2-highgpu-1g` (1×A100-40GB)** instance (`gcloud compute machine-images describe`), but the run config forced
     `--machine-type g2-standard-8` (L4). Creating a g2/L4 VM from an A100-native machine image fails at
     `gcloud compute instances create` (baked A100 accelerator config is incompatible with the g2 family) → the
     swallowed non-zero return → `SandboxRuntimeError` → `_safe_call` → `_abort_before_gpu`. THIS is the $0/no-VM abort.
  2. **The GCP single-VM path is SDAR-ONLY as wired (fundamental).** `VmComputeProvider.launch` (`vm_compute_provider.py:613`)
     unconditionally runs `scripts/sdar_gcp_run.sh`, which is **hard-wired to SDAR (arXiv 2605.15155)**: it `exit 1`s
     unless `runs/.cache/sdar_gcp.env` exists, hardcodes the SDAR paper + the SDAR implementer guidance heredoc +
     `PROJECT_ID=sdar_gcp_20260618`, and never reads a paper id. Worse, `stage`/`launch` are handed a `RunPlan`
     object (not a dict) → coerced to `{}` (`vm_compute_provider.py:521,625`) → the shipped `run_spec.json` is EMPTY
     and paper/model default to SDAR/foundry. So even with the machine type fixed, this path would reproduce **SDAR,
     not ADAM**. Making it paper-agnostic is a FEATURE (thread a real dict run_spec + a generic run script), not a
     config fix — above the "figure out the run config" scope.
- **Worked (the pivot):** The A/B does **not need GCP at all.** `campaign … --campaign-driver live --sandbox local`
  uses `LiveCliDriver` (`campaign_composition.py:1228`), which spawns `python -m backend.cli reproduce` as a **local
  subprocess** — no VM, $0 GPU. The Phase-B scheduler-apply `_cohort_loop` calls `self.stages.launch()`
  **driver-agnostically** (`reproduction_campaign.py:1045`), so it drives `LiveCliDriver` too. ADAM (1412.6980) is a
  CPU-tiny paper (MNIST/CIFAR logistic-regression + small nets) → runs on this Mac's CPU/MPS in minutes. So the whole
  Tier-3 A/B is achievable **locally, $0 GPU**, sidestepping every GCP gap. Scheduler-apply is already local + hermetic.
- **Failures:** none new — the 4 prior $0 aborts are now fully explained (gap #1).
- **Stalls / bottlenecks:** Decision surfaced to the operator (local vs invest-in-GCP): "fix all and run" on GCP means
  building paper-agnostic VM orchestration on an SDAR-tuned image — a week, not an afternoon. Local control arm is an
  afternoon; the treatment arm still needs the ADAM `SchedulerAuthoritySpec` + `_cohort_loop` checkpoint wiring
  (Phase C remaining) on ANY platform.
- **Infra / GCP issues:** GCP clean (0 clusters, 0 VMs). Auth OK (`aayush@deepinvent.ai`, project `deepinvent-ext-ut`).
  Only one machine image exists (A100-native); no L4-native or generic image.
- **Cost:** Tier-3 GPU **$0**, LLM **$0** (all aborts pre-LLM). No spend this session.
- **Improve / Next action:** await operator choice. If LOCAL (recommended): control arm =
  `python -m backend.cli reproduce 1412.6980 --sandbox local` (or `campaign … --campaign-driver live --sandbox local`),
  `OPENRESEARCH_DISABLE_ENV_PIN=1` (no CUDA on Mac). Then author the ADAM authority spec + wire `_cohort_loop`
  checkpoints for the treatment arm. Also still worth landing: the diagnostics fix (persist `outcome.reasons` +
  thread through `AttemptRawResult`) so a future VM abort is never again invisible.

### 2026-07-23 01:1x UTC — ADAM control run BLOCKED at provision_cpu (integration gap); $0 spent, GCP clean
- **Progress:** Launched the ADAM control arm 4× (all `$0`, NO VM ever created, zero waste). Recovered the
  full run config from past runs (SSH user `abheekp`, image `sdar-mi-20260620`, L4 `g2-standard-8`).
  Fixed 3 CLI/config issues in sequence, each surfaced by a fast $0 fail:
  1. `--model`/`--models` → the campaign uses `--root-model` + `OPENRESEARCH_ROLE_MODELS` env (argparse-2).
  2. `unenforceable:gpu_usd_no_rate` → added `--gpu-usd-per-hr 0.75` (the meter refuses-unattended without a rate).
  3. **`VmSpec` built with no machine image** → `attempt_driver.py:646` now threads `OPENRESEARCH_GCP_MACHINE_IMAGE`
     / `_IMAGE_FAMILY` into the VmSpec (commit `0746590e`, byte-identical off; verified the L4 create argv builds).
- **Worked:** All fail-safes worked perfectly — the campaign refuses-unattended on an unenforceable meter, and
  every abort is pre-provision so **cost stayed $0** across all attempts. GCP fully clean (0 clusters, 0 VMs).
  Verified at $0: preflight returns `available=True`; `provision_cpu` builds the correct `g2-standard-8` +
  `--source-machine-image sdar-mi-20260620` create argv with a fake success runner.
- **Failures (the wall):** Even with the image fix, the run still `EXHAUSTED / report_missing_twice`, $0, no VM.
  Localized precisely: the in-process reproduction reaches `decision=PROCEED` then aborts at
  **`provision_cpu`** (`reproduction_run.py:232` → `_abort_before_gpu`, BEFORE `self._lease_ref` is set at :236)
  — so `provision_cpu` returned None or raised **before any gcloud create ran** ($0/no-VM). `preflight` passes.
  **Diagnostics gap:** `reproduction_run.json` persists only `{state,decision,lease_ref}` — it DROPS
  `outcome.reasons`, and the error is swallowed by `_safe_call` (not logged even at `OPENRESEARCH_LOG_LEVEL=DEBUG`),
  so the actual cause is invisible. Suspect: the provider's default `_default_subprocess_runner`
  ("only runs outside tests", `vm_compute_provider.py:8,151`) may be suppressed in this context, OR a real
  gcloud create failed silently.
- **Stalls / bottlenecks:** Blocked on the swallowed abort reason — can't fix `provision_cpu` without seeing
  `outcome.reasons`. This is a real integration gap: the unified single-VM campaign path was never run to
  completion, so `provision_cpu` failure + the reasons-not-persisted gap were never exercised.
- **Infra / GCP issues:** Stray `gke-ltx` cluster fully DELETED (0 clusters/instances, $0 ongoing). No Tier-3
  VM ever provisioned.
- **Cost:** Tier-3 GPU **$0** (no VM). LLM **$0** (aborts before any LLM call). Stray-node total ~$3 (stopped).
- **Improve (the fix path):** (1) DIAGNOSTICS FIRST — persist `outcome.reasons` into `reproduction_run.json`
  (+ log the `_abort_before_gpu` reason) so the next run is diagnosable; (2) then re-run to read the real
  `provision_cpu` reason and fix it. Do this as a focused systematic-debugging session, not more blind retries.
- **Next:** small diagnostics fix (persist/log the abort reason) → one instrumented $0 re-run to read the
  `provision_cpu` cause → fix → the control-arm baseline. Then Phase C treatment prep + the A/B.

## ⚡ LIVE GCP RUN — ADAM control arm (attempted 2026-07-23 ~00:50–01:10 UTC — BLOCKED at provision_cpu, $0)
- **Run:** `adam_ctrl_20260723` (ADAM arXiv 1412.6980, **scheduler OFF = baseline "no-tree" score**).
- **Config (recovered from past runs per operator "figure it out"):** GCP single-VM, `g2-standard-8`
  1×L4-24GB (~$0.70/hr — verified the provider resolves L4 not the $14/hr A100 default), zone
  `us-central1-b`, instance `adam-ctrl-vm-20260723`, SSH user `abheekp` (provider default), project
  `deepinvent-ext-ut`. Root `gpt-5` (OpenAI) + sub-roles `executor/verifier/grader=grok` (Foundry;
  no Anthropic creds). Caps: `--max-llm-usd 5 --max-gpu-usd 5 --max-gpu-hours 2`.
- **Command:** `campaign 1412.6980 --campaign-driver unified --sandbox local --billing-sandbox gcp
  --model gpt-5 --models executor=grok,verifier=grok,grader=grok --max-llm-usd 5 --max-gpu-usd 5
  --max-gpu-hours 2 --project-id adam_ctrl_20260723`. Background task `bk42xjgcn`.
- **Monitoring:** `runs/adam_ctrl_20260723/{demo_status.json,dashboard_events.jsonl}` +
  `gcloud compute instances list` (watch the VM machine type on create; kill if it shows a2-highgpu-4g).
- **Watch items:** (1) create uses `g2-standard-8` not A100; (2) SSH as `abheekp` connects (if os-login
  blocks aayush-as-abheekp, provisioning fails — retry with aayush's os-login user); (3) L4 zonal stockout
  in us-central1-b → retry another zone; (4) teardown fires (no stray VM after terminal).

## Entry template (copy for each update)

```
### <UTC timestamp> — <one-line status>
- **Progress:** what advanced since last entry (tasks/commits/runs)
- **Worked:** what went right / is validated
- **Failures:** what broke, with the error + file:line or log path
- **Stalls / bottlenecks:** where time is being lost; blocked-on-X
- **Infra / GCP issues:** VM provisioning, GPU stockout, IAM, quota, teardown, stray billing
- **Cost:** GPU$ + LLM$ so far (from tokens_total.json + gcloud, NOT the ledger alone)
- **Improve:** what we'd change; follow-ups
- **Next:** the immediate next action
```

---

### 2026-07-23 00:43 UTC — Operator authorized billed run + node teardown; GCP preflight green; awaiting SSH user + go
- **Progress:** Operator decisions: (1) authorize the billed ADAM run, (2) tear down the stray node.
  Executed the teardown — `gcloud container clusters delete gke-ltx-20260722 --location us-central1-c`
  (bg `blv23e1va`): the node is GONE (`gcloud compute instances list` now empty → node billing STOPPED),
  control plane `STOPPING`. Ran read-only GCP preflight for the single-VM path (the GKE Workload-Identity
  blockers in [[dynamic-reproduction-scheduler]] do NOT apply — the compute-VM path uses `gcloud compute`
  + scp, not GKE/WIF).
- **Worked:** Preflight green: active account `aayush@deepinvent.ai`, project `deepinvent-ext-ut`
  (billing-enabled); GPU quota L4=8 / A100=8 / A100-80=4, all usage 0; root creds `OPENAI_API_KEY` (gpt-5)
  + `AZURE_FOUNDRY_*` (grok/Foundry) present. Cost estimate for the CONTROL arm (ADAM, scheduler OFF):
  **~$3–6 hard-capped** — 1×L4-24GB (`g2-standard-8`) @ ~$0.70/hr × ≤2h + ≤$5 LLM, via
  `--max-gpu-usd 5 --max-llm-usd 5 --max-gpu-hours 2`.
- **Failures / stalls:** **Blocked on operator input before provisioning (money gate):** (1) `gcloud
  compute os-login describe-profile` returned EMPTY → `OPENRESEARCH_GCP_SSH_USER` unresolved (needed for
  scp/ssh into the VM); (2) awaiting the explicit final go (a wrong machine-type = 20× cost). No Anthropic
  key / no Claude OAuth → Sonnet sub-agents MUST route to Foundry (`--models executor=grok,verifier=grok,
  grader=grok`) or the run dies at the first sub-call.
- **Infra / GCP issues:** **COST GOTCHA (loud):** the campaign single-VM path's DEFAULT machine is a
  4×A100-80 (`a2-highgpu-4g`, ~$14/hr) — MUST `export OPENRESEARCH_GCP_GPU_MACHINE_TYPE=g2-standard-8`
  (1×L4, ~$0.70/hr) or it bills 20× (`docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md:111-120`). No
  Tier-3 VM provisioned yet.
- **Cost:** Stray-node billing STOPPED (cluster deleted, ~$3 total, no longer accruing). Tier-3 GPU **$0**
  (no VM yet). Pending control-run est ~$3–6. Verified via `gcloud` (node gone) + no on-disk tokens ledger.
- **Improve:** Make the $14/hr-default machine-type a REQUIRED explicit flag / loud preflight warning in the
  campaign VM path so no operator can accidentally bill 20× — currently only a runbook note.
- **Next:** on operator SSH user + go → provision the L4 VM + run ADAM **control arm** (scheduler OFF) in
  the background with monitoring + the mandatory GCP-run journal; then Phase C **treatment** prep (author
  the ADAM `SchedulerAuthoritySpec` + wire `_cohort_loop` to real-contract checkpoints) for the ON arm.

### 2026-07-23 00:4x UTC — Phase C prereq DONE: real 5-field checkpoint contract (local, $0)
- **Progress:** Full `tests/rlm/` confirmed green (bg `b4n5w0872` exit 0; the tail was harmless asyncio
  teardown noise, not a failure). Then built the Phase C prerequisite — the real 5-field checkpoint
  contract:
  - `03bee5a0` — `backend/agents/rlm/cell_checkpoint.py` (stdlib-only, atomic, `LATEST` written last for
    preemption safety): `write_checkpoint(dir, step, *, model/optimizer/lr_scheduler/rng/data_order bytes)`
    → `step_<n>/` components dir; `latest_checkpoint_dir`/`read_checkpoint` for resume. Trainer guidance
    (`baseline_implementation.py`) now names all 5 components. Load-bearing test
    `test_written_checkpoint_is_receipt_ready` proves a contract-written checkpoint round-trips through
    `write_verified_receipt` → verifiable receipt.
  - `0c13d606` — added `cell_checkpoint.py` to `_HARNESS_CODE_HELPERS` so it's copied into the cell
    sandbox (a generated trainer can `import cell_checkpoint`). Covered by the existing helper-copy test.
- **Worked:** 19 tests green (checkpoint + receipt producer + helper-copy). The checkpoint substrate that
  a REAL cell needs to emit receipts is now in place + runnable-in-cell + trainer-guided. Memory recorded
  (`tier3-phaseAB-scheduler-applies-2026-07-22`); supersedes the "no producer, audit-only" state for the
  LOCAL path.
- **Failures / stalls:** None. (Earlier B3 subagent infra timeouts already recovered.)
- **Infra / GCP issues:** Stray `gke-ltx` CPU node STILL up (~$3+, not ours) — still awaiting operator
  teardown call.
- **Cost:** Tier-3 GPU **$0** (all local/hermetic). Phase C-prereq cost = LLM orchestration only.
- **Improve:** Cohort loop still resolves `checkpoint_components_dir` from the CPU-stub layout
  (`cell_out/checkpoint_components`); a REAL cell writes to `CELL_CHECKPOINT_DIR/step_<n>/` — wiring
  `_cohort_loop` to resolve via `cell_checkpoint.latest_checkpoint_dir(...)` is the next local step.
- **Next (all GATED on operator input — clean stopping point for autonomous work):** (1) wire cohort loop
  to real-contract checkpoints; (2) author the ADAM `SchedulerAuthoritySpec` (ladder + hyperparam branches);
  (3) the billed ADAM A/B on real GCP — needs **operator GPU-spend authorization** + the **stray-node
  teardown call**. Default-ON authority flip stays operator-only (≥3 paired A/B + grader-σ + sign-off).

### 2026-07-23 00:21 UTC — Loop check: Phase B committed & green; full tests/rlm/ still confirming
- **Progress:** No new code since `4aebdb81` (Phase B B3.3 exit test). Phase B is complete (8 tasks, 6
  commits `db7801ed`→`4aebdb81`). HEAD = `4aebdb81`.
- **Worked:** 506-test scheduler+campaign+cohort+receipt+authority gate green (7.24s). All Phase B
  red-line tests pass (evidence-not-grade end-to-end, terminal-wins, byte-identical-OFF).
- **Failures / stalls:** None. The full `tests/rlm/` regression (background task `b4n5w0872`) is STILL
  RUNNING — pytest process 71829 alive; the `| tail -6` pipe buffers so there's no interim output yet
  (not hung, just a large socket-hermetic suite). Final confirmation pending.
- **Infra / GCP issues:** Stray `gke-ltx-20260722` CPU node (`e2-standard-4`, us-central1-c) STILL
  RUNNING — now ~24h up, still not ours, still awaiting the operator teardown call. `gcloud` shows
  exactly 1 running instance (no stray A100s).
- **Cost:** Tier-3 GPU **$0** (no run launched). Stray CPU node ≈ `$0.134/hr × ~24h ≈ ~$3.2` to date.
  No Tier-3 `tokens_total.json` (none launched); verified via `gcloud` + on-disk artifacts.
- **Improve:** For long regression runs, avoid `| tail` in a backgrounded command (it hides interim
  progress) — write to a file and poll, or run foreground with a bounded `-x`/first-failure.
- **Next:** confirm the full `tests/rlm/` result, then Phase C prep — the real trainer must emit the
  5-field checkpoint contract (local, $0) before the ADAM A/B; operator GPU-spend authorization + the
  stray-node teardown are the gates for the billed run.

### 2026-07-23 00:xx UTC — PHASE B DONE — scheduler applies freeze/promote/revive from verified receipts (local, $0)
- **Progress:** All 8 Phase B tasks landed, committed, green. B3 (the integration crux that timed out
  twice as a heavy subagent) recovered + completed:
  - `afc0194d`/`6178d817` B3.2 — flag-gated `_cohort_loop` (via a single `_drive` dispatch seam; serial
    `_loop` untouched OFF) drives `claim_launches → run branch cell → build_raw_receipt →
    record_cell_receipt → decide_rung → apply (promote/freeze/kill) → revive → reconcile`. Terminal
    deterministic decision ALWAYS wins; provider GPU-$ from deterministic assessment (never the ledger);
    fail-closed on an incomplete cohort. (Recovered from a dropped subagent's uncommitted work +
    committed; fixed the B2.3 lineage test whose `run()`-expectation the new dispatch invalidated.)
  - `4aebdb81` B3.3 — the exit test: a REAL `campaign.run()` under a live controller routes through the
    campaign's own entrypoint (`_run_body → _run_fresh → _drive → _cohort_loop`), freezes the
    underperformer, promotes the others, and a `revive` re-queues the frozen branch — all from VERIFIED
    on-disk receipts whose metric is `metrics.json[metric_id]`, never the planted `0.01` grade.
- **Worked:** **Phase B exit bar met** — freeze/promote/revive from verified receipts, reachable from a
  real campaign entrypoint, `OPENRESEARCH_SCHEDULER_AUTHORITATIVE` still default-OFF + byte-identical
  when off. **506 tests green** across scheduler+campaign+cohort+receipt+authority (full `tests/rlm/`
  confirmation running). Full commit chain: `db7801ed 93b75470 2c2c9ee8 c1501a2b 6178d817 4aebdb81`.
- **Failures / stalls:** Two heavy B3 subagents hit infra timeouts (stream-idle 333s, socket-close 715s)
  — but the second left a high-quality uncommitted `_cohort_loop` + 296-line test that I recovered,
  reviewed, and committed. Lesson: decompose the hardest integration into smaller subagent tasks; a
  read+design+implement mega-task idles too long and drops.
- **Infra / GCP issues:** Stray `gke-ltx` CPU cluster still up (unchanged, awaiting operator teardown).
- **Cost:** Tier-3 GPU **$0** (all local/hermetic). Phase B cost = LLM subagent orchestration only.
- **Improve:** The 5-field checkpoint substrate is real for the CPU-stub/demo; a REAL ADAM run (Phase C)
  needs the actual trainer to write the 5 checkpoint components per contract — a Phase C prerequisite.
  Also: default-ON flip of authority stays operator-only (≥3 paired A/B + grader-σ + sign-off).
- **Next:** **Phase C** — the ADAM fixed-budget A/B on real GCP single-VM (control OFF vs treatment ON),
  scorecard comparing best rubric score + GPU$ + wall-clock. Prereq: the real trainer emits the 5-field
  checkpoint contract; operator GPU-spend authorization; the stray-node teardown call.

### 2026-07-22 23:3x UTC — Phase B: B1+B2 done & green (5/8 tasks); B3 integration is the remaining crux
- **Progress:** Phase B plan authored
  (`docs/superpowers/plans/2026-07-22-tier3-subproject-b-scheduler-apply.md`) and B1+B2 executed
  subagent-driven, TDD, per-task commits with full red-line review:
  - `db7801ed` B1.1 — harness-owned 5-field checkpoint materializer.
  - `93b75470` B1.2 — receipt producer builds the verified `raw_receipt` from a cell's deterministic
    `metrics.json[metric_id]` (NEVER the grade); reviewed → fixed NaN/Inf fail-closed + code-only
    digest (amended). 9 tests green.
  - `2c2c9ee8` B2.1+B2.2 — `CampaignOptions.authority_spec_path` + `--authority-spec`;
    `build_campaign` constructs `SchedulerAuthorityController` (both-flags+spec gated, lazy imports),
    injects into `ReproductionCampaign`. Byte-identical-OFF proven (no `scheduler_ladder.json`/
    `scheduler_tree_state.json` across 3 OFF branches). 150+ tests green.
  - `c1501a2b` B2.3 — controller is sole `branch-tree:<id>` lineage writer under authority (serial
    branch-spawn emit suppressed); non-vacuity verified.
- **Worked:** The receipt producer round-trips through the real `write_verified_receipt`/
  `load_verified_receipt` (shape proven), evidence-not-grade guard passes for the right reason, and
  every OFF invariant is tested. B1 (producer + checkpoint substrate) + B2 (construct/inject/lineage)
  are the machinery + wiring — all green.
- **Failures / stalls:** **B3 (the cohort loop + campaign-reachable exit test) is the integration
  crux and did NOT land.** A one-shot capable-subagent attempt (design+B3.2+B3.3) hit a stream idle
  timeout at 333s after 12 read-only tool uses — it committed nothing. Root cause: B3 has a real
  design fork that needs the campaign internals read first — (a) how `_loop` runs a single branch's
  training cell (the cell-execution injection point for a CPU stub), and (b) whether a full
  `campaign` is hermetically drivable without a live LLM root (likely the honest interpretation is:
  invoke the SAME `_cohort_loop` the real campaign takes under authority, with only the per-branch
  CELL execution stubbed — not a throwaway harness). Decomposing B3 into: recon `_loop` cell-exec →
  tight `_cohort_loop` (emission in-loop, controller in scope) → exit test.
- **Infra / GCP issues:** Stray `gke-ltx` CPU cluster still up (unchanged, awaiting operator call).
- **Cost:** Tier-3 GPU **$0**. B1+B2 cost = LLM subagent orchestration only.
- **Improve:** Emission belongs IN `_cohort_loop` (controller in scope), NOT threaded into
  `_execute_cell_matrix` — cleaner + testable. Update the B3 plan section to reflect that.
- **Next:** focused recon of `_loop`'s per-branch cell execution → implement `_cohort_loop` +
  in-loop receipt emission → the campaign-reachable freeze/promote/revive exit test (= Phase B done).

### 2026-07-22 23:08 UTC — Phase B recon landed; APPLY gap confirmed = 3 layers, receipt is the missing gate
- **Progress:** Phase B recon workflow `wjk2xcjou` (4 parallel readers: decide-seam / authority
  controller+runtime / receipt schema / cell-runner emission) **completed** (~294s, 4 agents,
  ~484k subagent tokens). HEAD unchanged at `b409104f` (no new code this tick — recon only).
- **Worked:** Clear, actionable map for the B plan. Key finding: the receipt-gated authority tree
  is **fully built + hermetically tested but has ZERO campaign call sites** —
  `SchedulerAuthorityController` (`scheduler_authority_controller.py:48`) exposes the whole
  promote/freeze/kill/revive surface (`bootstrap`/`claim_launches`/`record_cell_receipt`/
  `decide_rung`/`revive`/`reconcile_lineage`) but is constructed only in its own test. Today's
  `_maybe_apply_asha_authority` (`campaign_composition.py:1038`) holds ONLY the result dict and
  writes an additive `applied:false` audit — structurally cannot apply.
- **The APPLY gap is 3 distinct layers (not one edit):** (1) CONSTRUCT a campaign-lifetime
  controller in `build_campaign` (`campaign_composition.py:1272`), gated; (2) turn the serial
  `_loop` (`reproduction_campaign.py:949`) into a cohort driver (claim→run→receipt→decide_rung→
  promote/freeze/kill/revive); (3) emit a controller-attestable optimizer-step/checkpoint
  **receipt** from the local cell runner — **this is THE missing gate, the honest reason
  `applied:false` stands today.** Budget/a100_cap/gpu-$ data is already present at the seam
  (`campaign_composition.py:1114-1121`); what's missing is a receipt-bound deterministic metric +
  paper-step ladder (`scheduler_evidence.py:7,33` — must never fall back to `final_report.score`
  or `scope_rung`).
- **Failures:** None. (Recon read-only.)
- **Stalls / bottlenecks:** None. The 5-field checkpoint substrate (deferred from A) is confirmed
  as B's to build — it's the receipt's `checkpoint_path`+sha requirement.
- **Infra / GCP issues:** Stray GKE cluster `gke-ltx-20260722` still RUNNING (`us-central1-c`,
  1× `e2-standard-4` CPU, no GPU) — unchanged, still not ours, awaiting operator teardown call.
  `gcloud compute instances list` → exactly 1 running instance (no stray A100s).
- **Cost:** Tier-3 GPU spend **$0** (no run). Stray CPU node now ≈ `$0.134/hr × ~14.5h ≈ ~$1.9`.
  Recon cost = LLM only (~484k subagent tokens). Verified via `gcloud` + on-disk artifacts.
- **Improve:** The controller is injection-ready (`reproduction_campaign.py:404-416` shows the
  `branch_tree_event_store=None` optional-injection precedent to mirror). Watch the aggregate-id
  collision: `_maybe_emit_root_branch_spawned` already writes to `branch-tree:<campaign_id>`
  (`branch_lineage.py:29`) — the controller owns that aggregate, so wiring must reconcile, not
  double-write.
- **Next:** author Phase B's TDD implementation plan from the recon (CONSTRUCT + cohort driver +
  local checkpoint/receipt substrate + local freeze/promote/revive demo), all default-OFF /
  byte-identical-off, evidence-red-line guarded. Then execute local-first ($0).

### 2026-07-22 22:57 UTC — Loop check: A stable & green; stray GKE cluster found (CPU, ~$2)
- **Progress:** No new code since `b409104f` (HEAD unchanged). This is a scheduled loop-tick state
  gather. Sub-project A build remains complete + green (Tasks 1–4, 6). Task 5 deferred; B not yet
  started. 6 uncommitted/untracked entries (session artifacts: this journal, the plan, the spec,
  the runbook edit, `.demo_backups/`, `runs_logs/`) — intentionally uncommitted.
- **Worked:** Green-gate holds (1042 targeted tests + doc-fidelity 8, ruff clean, per prior entry).
- **Failures:** None new. (No live Tier-3 run to fail.)
- **Stalls / bottlenecks:** None.
- **Infra / GCP issues:** **Stray GKE cluster running.** `gcloud compute instances list` →
  exactly **1** running instance: `gke-gke-ltx-20260722-default-pool-0c4fc60c-k77q`
  (`us-central1-c`), part of cluster **`gke-ltx-20260722`** (1 node). Machine `e2-standard-4`,
  **no GPU accelerator**, non-preemptible, created `2026-07-22T00:40 PT`. NOT from Tier-3 / this
  work (I launched nothing on GCP) — the `ltx` name points to a separate project. **No stray
  A100s** (the cost-visibility check that matters). Surfaced to the operator; **NOT deleted**
  (unknown ownership — deleting a running cloud resource is destructive + not mine to reap).
- **Cost:** This session's Tier-3 GPU spend = **$0** (no run launched). The stray CPU node ≈
  `$0.134/hr × ~14h ≈ ~$1.9` to date (e2-standard-4, not GPU — modest). No `tokens_total.json`
  exists for any Tier-3 run (none launched); only historical `cutout_val1` / `prj_resnetgcp12`
  carry token ledgers. Verified via `gcloud` + on-disk `tokens_total.json`, not the cost ledger.
- **Improve:** Tear down `gke-ltx-20260722` if it's not needed (operator call). It also validates
  the no-GKE guard's purpose — nothing in Tier-3 should ever stand up a GKE cluster; this one came
  from outside the guarded path.
- **Next:** author sub-project B's implementation plan (local-first: `SchedulerAuthorityController`
  in the campaign decide seam + receipts from `gpu_cell_runner` + ASHA freeze/promote/revive), OR
  execute A's ADAM cloud run on operator GPU-spend authorization.

### 2026-07-22 22:51 UTC — Sub-project A implemented (5 commits, suite green); no GCP run yet
- **Progress:** Sub-project A implementation plan authored
  (`docs/superpowers/plans/2026-07-22-tier3-subproject-a-clean-gcp-vm-run.md`) and executed
  subagent-driven, TDD, per-task commits on `integrate/degke-runpod-on-trunk`:
  - `30ab843b` Task 1 — VM `watch()` detects the in-VM `final_report.json` sentinel → graceful
    `FINALIZE` (stops idle-GPU billing to the max-run-duration ceiling); + `completed→FINALIZE`
    regression test proving it doesn't misroute to `RECOVERED`.
  - `043a0933` Task 2 — VM `collect()` tar allow-list widened to bring back evidence-audit
    artifacts (`generated_rubric.json`, `rubric_tree.json`, `rlm_state/evidence_bundle.json`,
    `rlm_state/validation_verdict.json`).
  - `bf3a937e` Task 3 — GKE guard message reworded PARKED→"not used" (revive branch kept inert);
    pinned the `OPENRESEARCH_CLOUD_FAILOVER` path fail-closed (verified the guard's plain
    `RuntimeError` is outside `select_backend_with_failover`'s `try/except SandboxRuntimeError`,
    so it propagates terminally — no bypass hole).
  - `698e82ce` Task 4 — "GKE is not used" across authoritative docs (README, architecture,
    operations, engineering-guide, root+runtime CLAUDE.md); dated runbooks/periods left untouched.
  - `b409104f` Task 6 — armed `OPENRESEARCH_CELL_ERROR_SALVAGE` in the canonical campaign run-spec
    (verdict-only, downgrade-only, receipt-gated; inert on non-cloud runs).
- **Worked:** Every task went red→green under TDD. Green-gate: **1042 targeted tests passed**
  (runtime + reproduction_run + cell_error_salvage + claude_md_fidelity), ruff clean, doc-fidelity
  8 passed. Two-stage review on Task 1 caught a missing state-machine regression test + a weak
  comment — both fixed and amended.
- **Failures:** None in the build.
- **Stalls / bottlenecks:** None. `SendMessage` (continue-same-subagent) unavailable in this
  environment → review-fix used a fresh subagent with full context (no quality loss).
- **Infra / GCP issues:** No live GCP run yet. The ADAM cloud A/B is deferred operator-money work,
  gated on the preflight cost estimate + operator checkpoint.
- **Cost:** $0 GPU (no run). LLM cost = subagent orchestration only.
- **Improve:** (1) 5-field checkpoint/resume was moved from A → sub-project B (it's the freeze/revive
  substrate; a first clean run within budget doesn't need it) — B's plan MUST absorb it as a stated
  prerequisite. (2) **Autonomous-profile routing (Task 5) is a surfaced operator decision, NOT
  implemented** — the profile still routes to `sandbox='gcp'` which the guard now fail-closes; the
  naive `local` default may be actively wrong (backend host may have no GPU). Deferred per operator.
- **Next:** author sub-project B's implementation plan (construct `SchedulerAuthorityController` in
  the campaign decide seam + emit receipts from `gpu_cell_runner` + let ASHA apply freeze/promote/
  revive — validated purely locally, zero cloud cost), OR execute A's ADAM cloud run when the
  operator authorizes GPU spend. **Tripwire:** give A's run generous `--max-gpu-hours` so ADAM
  finishes in one shot; if it repeatedly STOPs mid-run, pull checkpoint/resume forward from B.

### 2026-07-22 (session start) — Planning phase; no GCP run live yet
- **Progress:** End-to-end system analysis complete. Tier-3 scope locked (ADAM A/B,
  best-score-at-fixed-budget, GCP single-VM both arms, **no GKE**). Base-pipeline root cause
  diagnosed. Design spec written + GKE directive recorded to memory. Sub-project A recon
  workflow (`wvqqudk20`, 5 parallel readers) launched.
- **Worked:** Test collection is clean (10,203 tests, 0 collection errors — the old ~18
  baseline errors are gone). Existing GCP-without-GKE spec found and reused.
- **Failures:** No live run failures yet. Historical: `cutout_val1` + `resnetgcp*` all
  verdict=failed because cells never trained to completion (`status` stuck `running`) and
  there was no cell-resume → each retry restarted the full grid; run SIGTERM'd after ~9.6h.
- **Stalls / bottlenecks:** None yet (planning). Known future bottleneck: VM rung-receipt
  wiring (sub-project B).
- **Infra / GCP issues:** None active. Known: GKE IAM was blocked (why we use single-VM);
  GKE now permanently out.
- **Cost:** $0 (no runs).
- **Improve:** Land the cell-completion + resume fixes before any multi-hour GCP run.
- **Next:** Ingest recon findings → author sub-project A implementation plan.
