# SDAR harness reliability + external-validation refactor — handoff (2026-06-20)

> **Status:** All GCP VMs STOPPED ($0). Branch `feat/azure-bicep-canonical-aoai-hardening`
> @ `fbf581ac` + **uncommitted working-tree changes** (this session — see §9). This handoff is
> the post-mortem of a multi-attempt SDAR-on-GCP run (v6→v9) and the **brief for the next
> session**, whose goals are: (1) figure out **why it worked on the local university cluster via
> claude-oauth but fails now**, (2) add **external-agent validation to reduce hallucinations**,
> (3) **refactor/optimize the harness + logic**.

---

## 0. The single most important clue (start here)

**"Everything worked when we ran it via `claude oauth` on the local university cluster, but it
fails now."** — the operator.

This is the highest-value lead and the next session should treat it as the primary
investigation. What we observed *now* (2026-06-19/20, GCP):

- `claude-oauth` (Sonnet via OAuth) **as the RLM ROOT reliably DEGENERATES** — reads the paper,
  then calls `FINAL_VAR` 3× without ever calling `implement_baseline`, tripping the
  degenerate-loop detector at **iter 0** (run `v9`). The harness even emits a baked-in warning:
  *"root model 'claude-oauth' is an UNRELIABLE RLM root … Recommended: --model gpt-5."*
- This is **classified in code**: `backend/agents/rlm/root_validation.py:67` →
  `rlm_backend == "anthropic-oauth"` ⇒ `RISK_DEGENERATE_LOOP`.

If claude-oauth-as-root genuinely worked on the university cluster, then **something regressed or
the environments differ**. Hypotheses to test (in rough priority):

1. **The forced-iteration / degenerate-loop detector is newer than the working runs and is now
   aborting a root that previously self-recovered.** Forced-iteration landed 2026-05-24
   (`forced_iteration.py`), the degenerate-loop detector 2026-06-17 (`root_progress.py` +
   `_make_degenerate_loop_callback` in `run.py`). If the university-cluster runs predate these,
   claude-oauth may have called `FINAL_VAR` early, been allowed to continue, *and then done the
   work* — whereas now the detector hard-aborts at 3 refusals. **Test:** `git log --oneline`
   those files; try a run with `OPENRESEARCH_MIN_RUBRIC_ITERATIONS=0` (disables forced-iteration)
   and `OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD` raised high, and see if claude-oauth recovers.
2. **OAuth transport difference: env-token vs credentials-file.** The cluster likely used
   `~/.claude/.credentials.json` (from `claude login`); GCP uses `CLAUDE_CODE_OAUTH_TOKEN`. The
   root drives via `claude_agent_sdk.query(tools=[])` (`claude_oauth_client.py`). Verify the
   driving behavior is identical under both credential shapes (auth is fine either way — the
   degeneration is *behavioral*, but confirm the SDK isn't silently degrading, e.g. truncating,
   under the token).
3. **System-prompt / paper-context drift.** The root degenerates by deciding it has "nothing to
   run". Compare the current `system_prompt.py` + the offloaded `context` metadata to whatever
   the cluster ran. A subtle prompt change could flip a borderline-cooperative root into a
   refuser.
4. **Model drift.** `claude-oauth` resolves to whatever Sonnet the subscription serves today
   (`claude-sonnet-4-6` in v9 events). If the cluster ran an earlier Sonnet, model-version
   behavior differs.

**Bisect target:** find a commit/date where claude-oauth-as-root drove SDAR to a real result,
then `git bisect` against the degeneration. The fix may be "make the degenerate detector
recovery-aware for oauth roots" rather than "never use oauth root".

---

## 1. What we were trying to do

First **successful, completed** SDAR (arXiv 2605.15155) reproduction on a GCP 8×A100 VM, fully
**keyless** (no paid API keys): an OAuth-free chat root + a validated Sonnet executor + the
anti-fabrication guards, producing a terminal rubric verdict. Canonical procedure:
`docs/runbooks/2026-06-16-sdar-on-gcp-a100-vm.md`; the guardrails handoff this session executed:
`docs/runbooks/2026-06-19-sdar-gcp-e2e-guardrails-handoff.md`.

---

## 2. The run journey this session (v6 → v9) — what each proved

| Run | Config | Outcome | Lesson |
|-----|--------|---------|--------|
| **v6** `sdar_gcp_gptchat_guarded_v6_20260619` | root=gpt-chat-latest, exec=Sonnet, guards on, smallest-two | **Reached REAL 8-GPU training** (model loads, 150 steps, ~30 GB/card) — a *first*. BUT cells wrote **all-zero metrics**; then **spot-preempted** ~80 min in, before verify/repair. | The validated Sonnet executor writes **real, executing, multi-file code** (no stub). Two new failures surfaced: **zero-metrics hallucination** (§3A) and **spot preemption**. OAuth + guards + cells route all worked. |
| **v7** (smallest-two, +anti-zero guidance) | same, fresh id | Killed seconds in to switch scope. | n/a |
| **v8** `…_gptchat_guarded_v8_20260619` | root=gpt-chat-latest, **3-model + 7B FSDP** | **gpt-chat root DEGENERATED at iter 2.** | gpt-chat-latest root is **non-deterministic** (drove fine in v6, degenerated here). |
| **v9** `sdar_gcp_sonnet_guarded_v9_20260619` | **root=claude-oauth (all-Sonnet)**, 3-model | **Root DEGENERATED at iter 0** (FINAL_VAR ×3, signature `no_experiment`). | `claude-oauth` is a **documented degenerate root** (`RISK_DEGENERATE_LOOP`). See §0. |

**Evidence location:** v6's real-training + zero-metrics artifacts live on the **GCP boot disk**
(`runs/sdar_gcp_gptchat_guarded_v6_20260619/` on `sdar-a100-8g`), not on the local box. To inspect:
flip the VM to `e2-standard-16`, `start`, SSH in (see §8).

---

## 3. The two core problems to fix

### A. Zero-metrics hallucination (the motivating case for external validation)

A v6 cell (`grpo__qwen2.5_3b__search_qa__s0`) ran 150 real steps over ~8 min of real GPU compute
(`elapsed_sec≈501`, `status:"completed"`) but wrote a `metrics.json` where **every** value is
exactly `0.0`: `loss`, `l_grpo`, `mean_reward`, `accuracy_avg`, `f1_avg`, `teacher_gap_mean`,
`gate_activation_ratio` — and all 150 history entries are `0.0`. The executor's code **runs and
consumes GPU but the training/reward/eval are not wired to the real model outputs** (loss
disconnected from the graph; reward not reading env outcomes; eval not scoring against gold).

**Why every existing guard missed it:**
- `STUB_METRICS_GUARD` — only fires when the metric *keys* are all placeholders
  (`total_length`/`chunk_count`); here the keys are *real* (`loss`/`reward`/`accuracy`), just
  zero-valued.
- VRAM antifab — only fires on ~0 GPU memory; here GPU *was* used (~30 GB).
- `EVIDENCE_GATE` / leaf grader — would likely have caught it, but the run was **preempted before
  it reached `verify_against_rubric`**.

**This is the canonical "plausible-but-fake result" hallucination** the operator wants to kill.
This session added a **detection-only** signal (`runs/.cache/monitor_snap.sh::realmetrics` — flags
"completed cells but all key metrics ≡ 0"), and stronger anti-zero implementer guidance
(`runs/.cache/sdar_scope_guidance.txt`). **Neither is in the harness yet** — promoting a
deterministic all-zero-metrics veto into the run-time path is a concrete first task.

### B. No reliable + keyless RLM root

The root-model matrix (verified in `backend/agents/rlm/models.py` + `root_validation.py`):

| Root | `rlm_backend` | `paper_validated` | Risk | Auth needed | Verdict |
|------|---------------|-------------------|------|-------------|---------|
| `gpt-5` | `openai` | **True** | `RISK_NONE` | live `OPENAI_API_KEY` (**dead/401**) | reliable, but no key |
| `claude` | `anthropic` | False | `RISK_UNVALIDATED` (fidelity advisory only) | funded `ANTHROPIC_API_KEY` (**empty**) | **reliable driver** (standard rlms client), needs key |
| `claude-oauth` | `anthropic-oauth` | False | **`RISK_DEGENERATE_LOOP`** | OAuth token (have it) | **degenerates as root** |
| `gpt-chat-latest` (foundry) | `openai`(custom endpoint) | False | `RISK_UNVALIDATED` | `AZURE_FOUNDRY_*` (have it) | **non-deterministic** (v6 ok, v8 degenerate) |

**Key insight:** `claude` (API-key transport) is a *reliable driver* — only `claude-oauth`
(OAuth-token transport via `claude_agent_sdk`) is the degenerate one. **Both are Sonnet; the
transport is what matters.** So "Sonnet for everything" *reliably* requires a funded
`ANTHROPIC_API_KEY` for the root (sub-roles stay on the free OAuth token). The operator was asked
to provide one; **pending** as of this handoff.

---

## 4. What is PROVEN to work (do not regress these)

- **OAuth executor/sub-roles** — `CLAUDE_CODE_OAUTH_TOKEN` + `claude_agent_sdk` authenticates
  flawlessly (zero auth errors across all of v6's sub-calls; `claude --print "ping"` → `pong`).
- **Sonnet executor writes real code** — full multi-file SDAR impl (`alfworld_env.py` 42 KB,
  `search_qa_env.py` 39 KB, `train_cell.py` 46 KB, `agentic_rollout.py`, `skillbank.py`, real
  Qwen load, `cells.json`). The validated-executor thesis holds.
- **Real GPU training + cells route** — one-GPU-per-cell `run_matrix` ran 8 cells in parallel on
  8×A100; 7B-FSDP path designed (`gpus:2`, `device_map="auto"`).
- **Guards** — preflight AST caught a real swallowed-OOM (`silent_oom`) bug and forced the
  executor to fix it (impl 3→4 in v6).
- **GCP infra path** — `prepare`→GPU-gate→`launch`→detached run→self-stop/preemption-handling all
  work. Assets warm (3× Qwen weights, ALFWorld, Search-QA; WebShop is a documented gap).
- **OpenAI Agents SDK is already integrated** — `OpenAiAgentRuntime` + `AzureOpenAiAgentRuntime`
  + `AzureFoundryAgentRuntime` (all subclasses); ChatGPT/gpt-5/gpt-chat all execute through it
  with full Read/Write/Edit/Bash parity. **No SDK work is needed** to support OpenAI models — the
  bottleneck is model *capability* (gpt-chat stubs; gpt-5 is validated) and **auth keys**, not the
  SDK.

---

## 5. Goal: external-agent validation to reduce hallucinations

The zero-metrics case (§3A) is the design driver. Current anti-fabrication is **deterministic
guards** (good — the guard itself can't hallucinate) but they have blind spots (real-keys/zero-
values; semantic correctness of the algorithm). Proposed direction for the next session to design
(use the brainstorming skill + write a spec under `docs/superpowers/specs/`):

- **Deterministic first (cheap, no new hallucination surface):** promote the `realmetrics`
  all-zero-metrics check into the harness as a run-time veto (a completed cell whose result-
  claiming key metrics are all ≡ 0, or constant across all steps, is `degraded`/
  `fabrication_suspected` and repairable). This alone would have caught v6.
- **External *agent* validator (the new ask):** a **separate** LLM agent (distinct from the
  executor — different model and/or fresh context) that independently audits the run's claims
  against the on-disk evidence and the paper. It should answer questions deterministic checks
  can't: *does this `loss` actually backprop from the model? does the reward read real env
  outcomes? does `train.py` implement SDAR's OPSD stop-gradient gate as the paper specifies?* —
  and emit a structured verdict the harness can gate on. Design tensions to resolve:
  - The validator is itself an LLM → it can hallucinate. Mitigate with: force it to **cite the
    exact file:line / metric value** it bases each verdict on; make it adversarial ("find the
    fakery") rather than confirmatory; require ≥N independent validators to agree (the codebase
    already has the multi-sample / refute-panel pattern — see `grader_transport.complete_samples`,
    the BES adversarial-verify idea).
  - Keep it **separate from the executor** so a model can't bless its own work.
  - It complements, never replaces, the deterministic gates (`EVIDENCE_GATE`,
    `deterministic_leaf_checker`, the all-zero veto).
- **Reuse what exists:** `backend/agents/rlm/evidence_gate.py`, `leaf_scorer.py`,
  `deterministic_leaf_checker.py`, `two_axis_report.py`, the `OPENRESEARCH_GRADER_*` sampler
  transport. The external validator is arguably a **new role** alongside grader/verifier in the
  per-role model picker (`role_models.py`).

---

## 6. Goal: harness refactor / optimization leads

Concrete, observed-this-session opportunities (not exhaustive — the operator wants a broader
pass):

- **Root-driving reliability (highest leverage; see §0).** Either make the degenerate detector
  *recovery-aware* for oauth roots (don't hard-abort a root that's one refusal from doing real
  work — the experimental `OPENRESEARCH_OAUTH_AUTODRIVE` is the seed but its v1 "issues a directive
  rather than truly executing" caveat needs the **lifecycle-state-machine refactor** the code TODO
  references), or document gpt-5/claude-API as the only supported roots and fail-fast clearly.
- **Cross-process metrics contract.** The zero-metrics bug is partly a *contract* gap: the
  per-cell `train_cell.py` (agent-written) can write a `status:"completed"` `metrics.json` with
  dead values and nothing rejects it before aggregation. Tighten the cell→aggregate contract
  (`cell_matrix.py`, `gpu_cell_runner.py`) with a non-zero/variance assertion.
- **Spot-preemption resilience.** `us-central1-c` 8×A100 spot preempted ≥2× this session. The
  root loop is **not** resumed on relaunch (only completed cells fingerprint-resume), so each
  relaunch re-runs ~20 min of root overhead. Either (a) checkpoint/resume the root loop, or (b)
  document on-demand (capacity-permitting) for runs that must complete. The auto-relaunch
  babysitter built this session (`runs/.cache/autorelaunch.sh`) is a stop-gap, not a fix.
- **`.env` sync footgun (FIXED this session, keep it).** `gcp_sdar_preflight.sh::sync_repo` scp'd
  `"$stage"/*` which **skips dotfiles**, so `.env` silently never synced — the renamed OAuth token
  stayed local. Fixed with `shopt -s dotglob`. Audit for other dotfile-drop spots.
- **Run-shaping env forwarding.** The `launch` whitelist dropped `OPENRESEARCH_ARG_CONTRACTS` /
  `STUB_METRICS_GUARD` / `LLM_AUTH_STRATEGY` (so the handoff's "guards on" command silently ran
  with guards OFF). Fixed this session. The multi-line `BASELINE_EXTRA_GUIDANCE` can't go through
  the `env $REMOTE_ENV` word-split path at all — now staged as a file (`launch` scp's
  `runs/.cache/sdar_scope_guidance.txt`). Consider a cleaner config-file-based run spec instead of
  the env-var whitelist.

---

## 7. Infrastructure state

| Thing | Value |
|-------|-------|
| GCP project / zone | `deepinvent-ext-ut` / `us-central1-c` |
| VM | `sdar-a100-8g` — `a2-highgpu-8g` (8×A100-40GB) when running, **SPOT**, **currently TERMINATED on `e2-standard-16` ($0)** |
| gcloud | `export CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud` for every call; account `abheek@deepinvent.ai` |
| Caches | warm on the boot disk (Qwen weights, ALFWorld + Search-QA, venvs, WebShop venv); `runs/.cache/sdar_gcp.env` written |
| WebShop | server doesn't come up → **documented gap** (uses ALFWorld + Search-QA) |
| `.env` creds | `CLAUDE_CODE_OAUTH_TOKEN` LIVE (`claude --print ping`→`pong`); `OPENAI_API_KEY` **dead (401)**; `ANTHROPIC_API_KEY` **empty**; `AZURE_FOUNDRY_*` set (deployment defaults grok-4.3) |
| Spend this session | ~$30–35 (mostly v6's ~80 min on a2; degenerate attempts ~$2 each) |

**Lifecycle commands** (machine-type flip needs the VM TERMINATED):
```bash
export CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud
P=deepinvent-ext-ut; Z=us-central1-c; I=sdar-a100-8g
gcloud compute instances list --project $P --format='table(name,zone.basename(),machineType.basename(),status)'
# cheap CPU debug (read the boot disk, no GPU billing):
gcloud compute instances set-machine-type $I --zone $Z --machine-type e2-standard-16   # while TERMINATED
gcloud compute instances start $I --zone $Z --project $P
gcloud compute ssh abheekp@$I --zone $Z --project $P --command 'cd /home/abheekp/openresearch && ...'
gcloud compute instances stop $I --zone $Z --project $P
```

---

## 8. Helper scripts built this session (gitignored, under `runs/.cache/`)

- `runs/.cache/autorelaunch.sh` — auto-relaunch babysitter: launches a config on spot, relaunches
  on preemption (≤5), and **exits + wakes the operator** on completion / degenerate /
  `realmetrics=no` (all-zero recurrence). **Edit the `launch_v7()` env block + the `PID`** to
  re-target. Runs in the background; **to kill it, never `pkill -f autorelaunch.sh`** (it matches
  its own argv and kills the killer) — use `for pid in $(ps -eo pid,args | grep autorelaunch.sh |
  grep -v grep | awk '{print $1}'); do kill -9 $pid; done`. **Kill it BEFORE stopping the VM**, or
  it relaunches.
- `runs/.cache/monitor_snap.sh` — one-line run snapshot (iters/impl/runexp/succ/cells/maxgpuMiB/
  **realmetrics**/report). Hardcodes the project id `P=`.
- `runs/.cache/sdar_scope_guidance.txt` — implementer guidance (currently: full 3-model +
  7B-FSDP + **anti-zero-metrics block**). `sdar_gcp_run.sh` loads it; `launch` stages it to the VM.

Launch shape that was being used (root TBD pending key):
```bash
OPENRESEARCH_SDAR_ROOT=<claude|gpt-5|foundry> \
OPENRESEARCH_SDAR_MODELS=executor=sonnet,grader=sonnet,verifier=sonnet \
OPENRESEARCH_ARG_CONTRACTS=1 OPENRESEARCH_STUB_METRICS_GUARD=1 \
OPENRESEARCH_LLM_AUTH_STRATEGY=oauth_only \
OPENRESEARCH_GRADER_SAMPLES=1 OPENRESEARCH_EVIDENCE_GATE=1 \
OPENRESEARCH_SDAR_PROJECT_ID=sdar_gcp_<...>_v10_<date> \
bash scripts/gcp_sdar_preflight.sh launch
```

---

## 9. Uncommitted working-tree changes (this session)

Tracked (in `git status`):
- **`scripts/gcp_sdar_preflight.sh`** — (a) launch whitelist += `OPENRESEARCH_ARG_CONTRACTS`
  `OPENRESEARCH_STUB_METRICS_GUARD` `OPENRESEARCH_LLM_AUTH_STRATEGY`; (b) `sync_repo` dotglob fix
  (so `.env` actually syncs); (c) `launch` now scp's `runs/.cache/sdar_scope_guidance.txt`.
- **`scripts/sdar_gcp_run.sh`** — (a) exports `CLAUDE_CODE_OAUTH_TOKEN` from `.env` into
  `os.environ` (subshell-sourced, quote-safe; the CLI path doesn't `load_dotenv`); (b) loads a
  staged scope-guidance file if present.

Untracked / gitignored:
- `.env` — renamed `CLAUDE_OAUTH_TOKEN` → `CLAUDE_CODE_OAUTH_TOKEN` (the only name the code reads).
- `runs/.cache/{autorelaunch,monitor_snap}.sh`, `runs/.cache/sdar_scope_guidance.txt`.

**Decision for next session:** these script changes are correct, general improvements (worth
committing); the `runs/.cache/` helpers are run-operational (keep as-is). Nothing has been
committed or pushed yet (push target is **deepinvent only**, per repo policy).

---

## 10. Recommended first moves for the next session

1. **Resolve the root** (unblocks everything): get a funded `ANTHROPIC_API_KEY` (→ reliable
   all-Sonnet via `root=claude`) **or** a live `OPENAI_API_KEY` (→ `root=gpt-5`, the only
   `RISK_NONE` root). Wire it like the OAuth token (note: `sdar_gcp_run.sh` does
   `env -u ANTHROPIC_API_KEY` — that must be conditioned/removed for `root=claude`, and the key
   exported the same way the OAuth token is).
2. **Investigate §0** (worked-then / fails-now) in parallel — it may reveal a regression that
   makes the OAuth root usable again and is the cheapest path to "keyless + reliable".
3. **Promote the all-zero-metrics veto** into the harness (deterministic; would have caught v6).
4. **Brainstorm + spec the external-agent validator** (§5) under `docs/superpowers/specs/`.
5. Re-run SDAR with the resolved root once (1) lands; the executor/guards/GPU path is proven, so a
   completed run is mostly gated on the root + the zero-metrics fix.

---

## 11. Pointers

- Canonical run procedure: `docs/runbooks/2026-06-16-sdar-on-gcp-a100-vm.md`
- The guardrails handoff this session ran: `docs/runbooks/2026-06-19-sdar-gcp-e2e-guardrails-handoff.md`
- Root validation/classification: `backend/agents/rlm/root_validation.py`, `models.py` (`ROOT_MODELS`)
- OAuth root client: `backend/agents/rlm/claude_oauth_client.py`, `_oauth_backend_patch.py`
- Degenerate detector / forced-iteration: `run.py` (`_make_degenerate_loop_callback`,
  `_default_degenerate_threshold`), `forced_iteration.py`, `root_progress.py`
- Anti-fabrication today: `evidence_gate.py`, `stub_detection.py`, `leaf_scorer.py`,
  `deterministic_leaf_checker.py`, `cell_matrix.py`, `gpu_cell_runner.py`
- Executor runtimes (OpenAI Agents SDK + Claude SDK): `backend/agents/runtime/{openai_runtime,
  azure_openai_runtime,azure_foundry_runtime,claude_runtime,factory}.py`
- SDAR paper specifics / invariants: `docs/runbooks/2026-05-23-sdar-baseline-handoff.md`,
  `backend/agents/prompts/paper_hints.py`
- Relevant memories: `oauth-root-reliability…`, `foundry-gptchat-root-not-executor`,
  `reasoning-chat-root-guardrails`, `reference_claude_setup_token_headless`,
  `project_gcp_gke_backend`

---

## 12. One-paragraph summary for the impatient

The harness's hard parts work on GCP: a Sonnet executor (OAuth) writes real multi-file SDAR code
and reaches real 8×A100 training — the first time any run got there. Two things block a *completed*
run: (A) a **zero-metrics hallucination** — the executor's training/reward/eval can run on GPU yet
emit an all-`0.0` `metrics.json` that slips past every existing guard (caught only by a new
detection-only check), and (B) **no reliable keyless root** — `gpt-5`/`claude` (validated/reliable)
need paid keys we lack, `claude-oauth` reliably degenerates as a root, and `gpt-chat-latest` is
non-deterministic. The operator reports it **worked via claude-oauth on the local university
cluster**, so the degeneration is likely a regression or environment difference worth bisecting.
Next session: resolve the root (funded `ANTHROPIC_API_KEY` → all-Sonnet, or investigate the oauth-
root regression), promote the all-zero veto, and design an **external adversarial validator agent**
(separate model, must cite evidence) to kill plausible-but-fake results.
