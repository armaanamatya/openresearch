# Phase-3 (WS3 durability + WS2 guard) — Implementation Result + Follow-Up Designs

- **Date:** 2026-07-12
- **Branch:** `feat/gke-gpu-path-reproduction-reliability`
- **Predecessor:** `docs/runbooks/2026-07-11-unified-platform-phase1-2-implementation-handoff.md` (the Phase-3 fan-out playbook this executed)
- **Status:** WS3 fencing/adopt/deadline + reaper + WS2 guard + controller-launchability **DONE, committed, verified** (all flag-gated `OPENRESEARCH_DURABLE_CONTROLLER`, byte-identical off, verdict-inert). Two verdict-adjacent lead-only items (WS1-H1, WS-Ext) **deliberately deferred** with precise designs below.

---

## 0. How this was run
`ultracode` Workflows. A read-only recon Workflow verified every anchor + extracted the helper contracts
(→ `.history/sdd/phase3-recon/*.md`); the lead authored anchored briefs (→ `.history/sdd/phase3-*.md`
+ `phase3-contract.md`); a Wave-B Workflow ran 5 concurrent **disjoint-file** owner chains
(implement→adversarial-review→bounded-fix, Sonnet); the lead (Opus) reviewed **every diff**, re-ran the
baseline, and **owns all commits** (subagents never touched git). SDD ledger: `.history/sdd/progress.md`.

## 1. What landed (6 commits)
| Commit | What |
|---|---|
| `205ca38e` | `deadline.py` — pure absolute-epoch deadline helper (make/remaining/is_expired/serialize/parse), clock-injected; 18 tests |
| `f75c30a4` | **cell-runner fencing** (`k8s_job_cell_runner.py`): fenced Job name + `gen-<gen>/cells` blob prefix + `reprolab-run-id`/`reprolab-generation` labels via the `bind_run_context(fence_generation=…)` seam; **adopt-on-409** (duck-typed, skip/adopt/submit via `job_fence.adopt_or_submit`); **persisted-remaining deadline** on adopt |
| `ed040b02` | **reaper** (`blob_lease.py`): `reap_older_generations(run_id, token, *, list_jobs, delete_job)` — pure, injected-I/O, fail-soft at both levels; module stays SDK-free |
| `e24ce9cb` | **WS2 guard** (`k8s_job_backend.py`): `_command_needs_staged_code` + `exec()` fail-loud `monolithic_exec_unstaged` on durable+gcp code-dependent commands |
| `5f4bf21d` | **controller launchability** (`run_controller.py`+`cli.py`): `classify_controller_exit(3) → "money_halt"` (was "crash"); `campaign --project-id` + `cmd_campaign` threading (mirrors `reproduce`) — fixes the latent `build_controller_command --project-id` argparse bug |
| `f46bec9b` | **SaaS seam** (`live_runs.py`): flag+gcp-gated divert to an injectable `_submit_durable_controller` stub; off = byte-identical `Popen` |

**Verification:** combined-tree full suite = **20 failed / 9505 passed**; the only new failure vs the pre-edit
19-failure baseline was the *expected* obsolete `TestReapStub`, now updated (→ `TestReapRequiresInjectedIO`) →
back to the **19 known env failures** (`/tmp/phase3_baseline_failures.txt`: oauth/keychain/OCR/accelerator/
demo-gate/repo-hygiene), **+105** new Wave-B tests all green. Focused re-run (verdict guard + `test_registry`
+ every owner suite) = **334 passed**. `assert_verdict_surface_unchanged` green; `PRIMITIVE_REGISTRY` still **19**;
off-state byte-identical proven per owner.

## 2. Implementer deviations the lead SIGNED OFF (correct, and off the literal brief)
1. **Thread-safety (cell-runner):** the fence generation is read **once in `run_matrix`'s main thread** and
   threaded down as an explicit param — a `ContextVar` is invisible in the worker threads `run_matrix`
   spawns, so re-reading `_get_fence_generation()` deeper would silently see `None`. Mirrors the file's
   existing `_SETTINGS_PREFIX_CTX` re-pin precedent. **Correct.**
2. **`already_succeeded` shape (cell-runner adopt):** probes `outcome == "ok" or exit_code == 0` (the real
   per-cell `status.json` sentinels `_process_cell` already uses), **not** the brief's literal
   `result["status"]` (which never exists → would make the skip branch dead code). **Correct.**
3. **WS2 `-m` denylist:** `python -m pip install …` (the dominant real `-m` shape) is excluded from the
   staged-code predicate so the guard doesn't false-block environment bootstrap. **Correct.**

## 3. DEFERRED lead-only work — precise designs (the actionable follow-up)

### 3a. WS1-H1 — demo_status stale-republish guard (`run.py`)
**Finding (recon-verified):** the two `status="running"` writes at `run.py` ~:3310 (run-start snapshot) and
~:3399 (root-validation stamp) are **provably safe already** — they run once at process top, and
`attempt_isolation._reset_demo_status` resets any stale terminal file to `"queued"` before they fire. The
**real** exposure is `_update_cost_summary_loop` (~:2627): a 30 s background daemon does a non-atomic
whole-file read-modify-write of `demo_status.json`. `_cost_stop_event.set(); join(timeout=5.0)` runs before
`_finalize`, but the timeout does **not** guarantee a slow in-flight iteration's write landed — a write that
read `"running"` before the terminal write can land *after* it, republishing a stale `status`.
`assert_verdict_surface_unchanged` will **not** catch this (only `verdict`/`implementation_verdict`/
`replication_verdict` are surface keys; `status`/`process_status` are not).

**Design:** in `_update_cost_summary_loop`, right before the atomic `os.replace`, **re-read the live
on-disk `status`; if it is terminal (`completed`/`failed`/`stopped`), discard the tmp file and skip/return**
— the daemon must never republish a stale non-terminal snapshot over a terminal one. This does **not** touch
`_write_demo_status`, `_reset_demo_status`, or any verdict logic (so it can't break the legitimate
terminal→`queued` reset, and is verdict-surface-protective). **Open decision to make deliberately:** ship it
as a flag-gated capability (`OPENRESEARCH_DEMO_STATUS_TERMINAL_GUARD`, default-OFF, + `docs/reference/flags.md`
+ `gen_flag_registry --check`) vs. a pure correctness bugfix (no flag, ships with its regression test). Given
it is verdict-adjacent, the conservative flag-gated default-OFF is the safer default. **Acceptance:** a unit
test that seeds a terminal on-disk status then drives one loop iteration and asserts the terminal status
survives; `test_single_verdict_authority_guard` still green; registry still 19.

### 3b. WS-Ext — make `result_fidelity` actually measure (`repro_spec_extractor.py`)
**Finding (recon-verified):** every real claim has `kind == ""` → `result_fidelity._evaluate_claim` falls to
`missing_kind` → **always `unmeasured`** → the deterministic verdict can never be `reproduced`/`contradicted`
(Adam/every paper stays `inconclusive`/`partial`). `_normalize_claim_from_llm` **drops** `baseline_value`/
`proposed_value` (the LLM already emits them per `_EXTRACTOR_SYSTEM`). The A6a cross-check
(`_reconcile_with_blinded`) treats a **one-sided `None`** as a disagreement (over-conservative → false
ambiguous).

**Design (do all THREE together, with matched conservatism):**
1. **Explicit `kind`, never inferred.** Add a `kind ∈ {numeric, relative, trend, qualitative}` field to
   `_EXTRACTOR_SYSTEM` (with crisp definitions; "when uncertain → qualitative"), read it in
   `_normalize_claim_from_llm` **only if it is one of those four literals**, else leave it absent
   (→ `missing_kind` → `unmeasured`, the safe fall-through). **Do NOT infer `kind`** from `estimate_kind`
   (a unit, not a test kind) or from `baseline_value` presence — recon gotcha 2: a relative claim mis-typed
   as numeric compares a measured absolute against a claimed delta → **false `contradicted`**, the worst error.
2. **Thread `baseline_value`/`proposed_value`** through `_normalize_claim_from_llm` → `build_repro_spec`'s
   `comparison` dict → the `ComparisonSpec` dataclass (additive fields, defaults, so the frozen schema stays
   back-compatible). A `relative` claim needs a finite `baseline_value` to measure (it currently can't).
3. **Relax A6a:** in `_reconcile_with_blinded._cmp`, a one-sided `None` (first pass has a value, the blinded
   pass — given a shorter span — doesn't restate it) should **not** count as a disagreement; only an actual
   numeric conflict (both present, beyond tolerance) should.
**Gate:** flag-gated default-OFF (extractor emits no `kind` when off → byte-identical `unmeasured`).
**Acceptance (the whole point):** a real non-ambiguous numeric claim **measures** (pass/fail); the frozen
`runs/prj_adam_local_1` primary (genuinely ambiguous) **stays `inconclusive`**; **no false `contradicted`**;
the existing `test_normalize_lifts_..._real_adam` (asserts `"kind" not in` the *frozen* artifact) stays green
(the frozen artifact is untouched — only new extractions gain `kind`). Recon:
`.history/sdd/phase3-recon/run_py_repro_spec_extractor_py.md`.

## 4. Drill-gated / operator items (an agent cannot do these — real GKE + money)
- **The WS3 durability drill (headline validation):** `sandbox=gcp` run → kill the controller pod
  mid-training → a successor adopts the in-flight fenced Job by name, resumes from the persisted deadline +
  ledger, finalizes a real metric. Plus split-brain (stale token refused) + reaper (no orphaned A100 bills).
- **Reaper real adapter:** the drill-time controller supplies `list_jobs`/`delete_job` to
  `reap_older_generations` from a `BatchV1Api` (`list_namespaced_job(label_selector="reprolab-run-id=<run>")`
  → parse `reprolab-generation` label → int; delete via the `_delete_job_quietly` pattern).
- **`build_controller_command` budget-meters:** it still omits the REQUIRED `--max-llm-usd/--max-gpu-usd/
  --max-gpu-hours` — decide the values.yaml→controller budget flow at drill time.
- **Helm `reproduce`→`campaign`:** rewrite `orchestrator-deployment.yaml`/`orchestrator-cronjob.yaml`
  commands to `campaign --project-id <stable> --resume` + add the campaign values keys (untestable
  hermetically; `orchestrator.enabled:false` today).
- **Deadline bucket-consistency:** verify the cell-runner deadline **write** (`gcs_blob.upload_bytes` with
  `_cloud_setting("gcs_bucket")`) and **read** (`_blob_download_bytes`) resolve to the SAME GCS bucket — if
  not, adopt silently falls back to a fresh deadline (fail-soft, not broken, but the remaining-budget
  inheritance would be inert). Confirm in the drill.
- **Escalation label consideration:** an OOM-escalated cell's `reprolab-run-id` label carries the `-eN`
  suffix (matches its fenced Job name); a reaper querying the base run id would miss it. Fine while the
  reaper is drill-deferred; decide base-vs-suffixed when wiring the real adapter.

## 5. Deferred (design/infra not ready — NOT rushed)
- **Owner-6 GCS-mirror ledger + resume-promotion** (`reproduction_campaign.py` GCS-mirror + promote GCS
  `status.json` ahead of local `cell_manifest.json` in `cell_scheduler.should_skip_cell`): the riskiest piece
  (resume-skip correctness = lost work / GPU cost); the safe framing is "GCS status can only ADD skips when it
  confirms `ok`, never skip-without-evidence." Drill-gated anyway.
- **WS4 CPU-class lane** (`gpu_count=0`): no WS4 design spec, no CPU node pool, no CPU-capacity model — the
  three `max(1,…)` floors + the unconditional `nvidia.com/gpu` block + the parallelism divisor all assume ≥1.
  Premature; needs its own design + infra.

## 6. Where the material lives
- Recon (verbatim anchors + helper contracts): `.history/sdd/phase3-recon/*.md`
- Owner briefs + shared contract: `.history/sdd/phase3-*.md`
- SDD progress ledger: `.history/sdd/progress.md` (Phase-3 section at the end)
- WS3 design: `docs/history/specs/2026-07-10-durable-cloud-native-orchestration-ws3-design.md`
