# Pre-GPU smoke crash-surfacing — debug handoff (why SDAR keeps dying at the smoke + the crash never reaches the executor)

> **For the NEXT session. Self-contained.** Authored 2026-06-21 after a successful actor-critic
> harness validation run that nonetheless produced an honest `failed` because SDAR reproductions
> keep dying at the **pre-GPU smoke** with `smoke_metrics_unreal`, and the **fix-first repair loop
> cannot converge because the crashing cell's traceback is captured then thrown away.** This handoff
> roots both symptoms ("the smokes are so bad" + "the crash isn't surfacing") in **one code gap** and
> specs the robust fix + a reproduction + a verification plan.
>
> **Branch:** `feat/grounded-self-improvement-harness-reliability`. **Read alongside:** memory
> `[[gcp-a100-capacity-quota]]` (turnkey 4×A100 on-demand re-run), and
> `2026-06-20-validation-coverage-and-capped-rerun-handoff.md` (the timeout-trap fix, now confirmed working).

---

## GOAL & STATUS (read first) — run SDAR e2e on GCP with ALL session fixes

**Your job:** commit the session fixes, push them to the GCP 4×A100 on-demand instance, run SDAR
end-to-end, and confirm the harness now (a) **surfaces the crash** to the repair loop and (b) either
reaches **real GPU training → an honest scored report**, or fails honestly with the crash finally
**visible**. The actor-critic harness honesty itself is already validated (see §1) — this run is about
getting *past the smoke* to real training.

**Fixes from this session (all in the working tree — COMMIT them, then `sync` to the VM):**

| Fix | Files | Status |
|---|---|---|
| **Smoke crash-surfacing** (this handoff's subject) | `metric_reality_smoke.py` + `tests/rlm/test_metric_reality_smoke.py` | ✅ IMPLEMENTED (Fix A + tests; §3) |
| Cred-provider (azure-foundry → AZURE_FOUNDRY_API_KEY, not OPENAI) | `models.py` (`cred_provider`) + `cli.py` + `tests/rlm/test_models.py` | ✅ done, verified live |
| Dynamic on-demand provisioning switch | `scripts/gcp_sdar_preflight.sh` (`ensure_provisioning_model` + STANDARD `remote_prepare` guard + rsync/trap fixes) | ✅ done, verified live |
| `.env` line-18 quote (`"NVIDIA L40S"`) | `.env` | ✅ done |

**First action:** commit the tracked fixes (do NOT use `-A` — the `runs/` tree is untracked noise;
`.env` is gitignored, its line-18 fix reaches the VM via `sync`, not git):
```
git add backend/agents/rlm/metric_reality_smoke.py backend/agents/rlm/models.py backend/cli.py \
  scripts/gcp_sdar_preflight.sh tests/rlm/test_metric_reality_smoke.py tests/rlm/test_models.py \
  docs/runbooks/2026-06-21-smoke-crash-surfacing-debug-handoff.md && git commit
```
Then follow §6 to sync + launch. §4 (reproduce the alfworld crash) is optional — the e2e run surfaces
it automatically now that Fix A is in. Sanity before launch: `.venv/bin/python -m pytest tests/rlm/test_metric_reality_smoke.py tests/rlm/test_models.py -q` (expect 25 + 49 green).

---

## 0. TL;DR — one root cause, two symptoms

**`metric_reality_smoke._run_one_smoke_cell` captures the cell's crash output (`stderr=STDOUT, stdout=PIPE`) but DISCARDS it** (`proc.communicate(timeout=…)`'s return value is ignored) and returns only `(trace, peak_vram, launched, timed_out)` — **no output, no exit code.** So when a cell **crashes early** (the SDAR alfworld cell did — GPU util never left 0%), the smoke sees `trace=None` + a natural (non-timeout) exit and emits the dead-end message:

```
[qwen2_5_3b__sdar__alfworld__s0] executor exited without writing any per-step trace —
write metrics.json/smoke_trace.json incrementally per step
```

That string becomes the `repair_context` for the next `implement_baseline`. The executor (Sonnet) keeps adding trace-writing code but **never sees the actual crash** → 3 identical `smoke_metrics_unreal` rejections → `repair_exhausted` → honest `failed`.

- **The smoke is NOT mis-calibrated.** The 2026-06-17 `1fd6812e` calibration (timeout→inconclusive, accept ≥1 step) is correct. The smoke is *correctly* catching a crashing cell — it just **cannot tell anyone why.**
- **The fix is to surface the cell's stderr + exit code into the failure message** so the repair loop (and you) can see the crash. Plus a design question about the all-or-nothing smoke gate.

---

## 1. Evidence (the run that motivated this)

- **Run dir:** `runs/sdar_gcp_actorcritic_od_20260620/` on instance `sdar-a100-od` (us-central1-b, **STOPPED** — disk persists; start to inspect).
- **`final_report.json`** (the harness honesty IS proven — this is a validation success, not a harness bug):
  - `verdict=failed`, `rubric.overall_score=0.360`, `meets_target=False`, `target=0.6` (consistent — no fake green).
  - `validation={status:clean, separation:independent, panel_models:["grok-4.3"]}` — **the grok validator ran** (cross-family, independent). The prior 6h-trap run never reached it.
  - `stop_reason=None` — **natural `_finalize`** (the `OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S=1800` trap fix worked).
  - `models`: planner=`gpt-chat-latest`, executor=`anthropic-oauth:claude-sonnet-4-6`, grader/verifier=Sonnet.
- **`experiment_runs.jsonl`** (4 rows — the repair trajectory):
  1. `code_review_rejected` — `missing_teacher_model` (no teacher loaded for OPSD). Executor **FIXED it** (failure class changed → real progress).
  2. `smoke_metrics_unreal` — alfworld cell "exited without writing any per-step trace".
  3. `smoke_metrics_unreal` — **same**.
  4. `smoke_metrics_unreal` — **same**. GPU stayed at **0%** across all → the cell crashes **before model load**, i.e. an early crash, not a slow-rollout timeout (timeouts go *inconclusive*; this is a *natural exit*, correctly judged).

---

## 2. Root cause, grounded in code (`backend/agents/rlm/metric_reality_smoke.py`)

### Symptom A — "the crash isn't surfacing"
- **`_run_one_smoke_cell` (def L309, runner L355–401):**
  - L356–364: `subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, …)` — the crash traceback **is captured** (stderr merged into the stdout pipe).
  - L373: `proc.communicate(timeout=timeout_s)` — **the return value (the captured stdout+stderr) is discarded.**
  - L401: `return trace, peak_vram_gib, True, timed_out` — **no `proc.returncode`, no output.** The crash text is unrecoverable past this point.
- **`run_metric_reality_smoke` (def L424):**
  - L511–517 (natural-exit, no-records branch): appends only `f"[{cell_id}] executor exited without writing any per-step trace — write metrics.json/smoke_trace.json incrementally per step"`. **No traceback, no rc.**
  - L534–537: `detail = "smoke_metrics_unreal: " + "; ".join(failures[:3])` → `{"ok": False, "failure_class": "smoke_metrics_unreal", "detail": detail}`.
- **The flow to the executor:** that `detail` is consumed by `run_experiment` in `primitives.py` (smoke is wired there; `smoke_metrics_unreal` is a repairable class — see `primitives.py` `_RUN_EXPERIMENT_REPAIRABLE_FAILURES` / `_METRICS_BEARING_REPAIRABLE_FAILURES` and the repair-context construction) and becomes the next `implement_baseline` prompt's repair context. **By the time it reaches the executor, the crash is already gone.**

### Symptom B — "the smokes are so bad"
Not a calibration bug. Two compounding factors:
1. **The surfacing gap above** — the repair loop is *blind* (no traceback) so it can't converge.
2. **All-or-nothing gate** — `_smoke_max_cells` default **4** (L53/L62), and `if failures: return not-ok` (L534): **any single failing cell fails the entire smoke.** One crashing alfworld cell blocks the whole grid even if the search_qa cells would have passed.

---

## 3. The robust fix (the actual ask)

### Fix A — surface the cell stderr + exit code (the core fix)
In `_run_one_smoke_cell`:
1. Capture the output: `captured, _ = proc.communicate(timeout=timeout_s)` (`captured` = stdout+stderr merged; on `TimeoutExpired`, capture whatever the post-kill `communicate(timeout=5)` returns).
2. Return `proc.returncode` and a **bounded tail** of `captured` (e.g. last ~4000 chars) — extend the return tuple or (cleaner) return a small dataclass/dict `{trace, peak_vram_gb, launched, timed_out, returncode, output_tail}`.

In `run_metric_reality_smoke` (natural-exit branch, L511–517):
3. Build an actionable message that distinguishes **crash vs no-log**:
   - `rc != 0` → it crashed: `f"[{cell_id}] crashed (exit {rc}) before writing a per-step trace. Last output:\n{output_tail}"` (the tail carries the traceback).
   - `rc == 0` → ran but didn't log: keep the existing "write metrics.json/smoke_trace.json incrementally per step" hint.
4. The `detail` now carries the traceback → `repair_context` → the executor sees the real error and can fix it (or you can read it directly).

**Bounding/safety:** cap the tail (~4000 chars, keep the *end* where the traceback is). This text goes into the repair_context (not the SSE stream), but keep it path-clean; the corpus is not involved (cell output is the executor's own training stdout).

### Fix B — per-cell smoke vs all-or-nothing (design decision, do deliberately)
- **Option 1 (keep all-or-nothing):** a crash usually signals a *systemic* bug (all cells share `train_cell.py`), so blocking the whole grid is defensible. Minimum bar: just do Fix A so the one crash is actionable.
- **Option 2 (per-cell):** run all representatives; drop a *confirmed-crashing* cell to `scope.gaps` (mirror `cell_matrix.capacity_gate`'s dropped-cell handling) and let runnable cells proceed. Better when one env (alfworld) is broken but another (search_qa) works. Risk: a shared-code crash would let a half-working grid through.
- **Recommendation:** ship Fix A first (it's the real blocker). Treat Fix B as a follow-up once you see, with tracebacks surfaced, whether crashes are per-cell or systemic.

### Fix C — tests (`tests/rlm/test_metric_reality_smoke.py`)
- A `train_cell.py` stub that `raise RuntimeError("BOOM-<marker>")` → assert the verdict `detail` contains the marker **and** `exit`/`rc` (the traceback surfaces).
- A stub that exits 0 but writes no trace → assert the "write … per-step trace" hint (the rc==0 path).
- Keep the existing timeout→inconclusive and accept-1-step assertions green (don't regress `1fd6812e`).

---

## 4. Reproduce the specific alfworld crash (to confirm Fix A + diagnose the executor bug)

The smoke runs each cell in a `TemporaryDirectory` that is **cleaned up**, so the original traceback is gone — but the run's `code/train_cell.py` persists on the (stopped) instance disk. Run the alfworld cell **exactly as the smoke does** to print the real error:

```bash
export CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud
gcloud --project deepinvent-ext-ut compute instances start sdar-a100-od --zone us-central1-b   # 4xA100 on-demand
gcloud compute ssh abheekp@sdar-a100-od --zone us-central1-b --project deepinvent-ext-ut --command '
cd /home/abheekp/openresearch; d=runs/sdar_gcp_actorcritic_od_20260620
CID=$(python3 -c "import json;print([c[\"id\"] for c in json.load(open(\"$d/code/cells.json\"))[\"cells\"] if \"alfworld\" in c.get(\"id\",\"\")][0])")
export CUDA_VISIBLE_DEVICES=0 OPENRESEARCH_CELL_MAX_STEPS=2 OPENRESEARCH_CELL_TINY_SLICE=1 OPENRESEARCH_CELL_OUTPUT_DIR=/tmp/smoke_repro
export OPENRESEARCH_CELL_PARAMS="$(python3 -c "import json;print(json.dumps([c for c in json.load(open(\"$d/code/cells.json\"))[\"cells\"] if c[\"id\"]==\"'\"$CID\"'\"][0]))")"
mkdir -p /tmp/smoke_repro
.venv/bin/python $d/code/train_cell.py --cell-id="$CID" --output-dir=/tmp/smoke_repro 2>&1 | tail -80; echo "rc=$?"'
# stop when done:
gcloud --project deepinvent-ext-ut compute instances stop sdar-a100-od --zone us-central1-b
```

Classify the traceback: import error / ALFWorld env setup (TextWorld, game files) / teacher-model OOM (two 3B models on one 40GB A100?) / a logic bug (exits before the step loop). Then decide: a `PAPER_HINTS`/`BASELINE_EXTRA_GUIDANCE` nudge, **or** just rely on Fix A (the executor self-fixes once the traceback is in the repair context). Note: `train_cell.py` is executor-*generated* and non-deterministic — a fresh run may crash differently, which is exactly why the **systemic** surfacing fix matters more than this one bug.

---

## 5. Verification plan
1. Apply Fix A (+ Fix C tests). `pytest tests/rlm/test_metric_reality_smoke.py -q` green; full `pytest tests/rlm -q` no regressions.
2. Run §4 → confirm the traceback now appears in the smoke `detail` (manually, or via a quick `run_metric_reality_smoke` harness call against the broken cell).
3. Re-run SDAR on the 4×A100 (turnkey, §6) → confirm the `repair_context` now carries the crash, and either the executor converges (smoke passes → **real GPU training** → scored report) or you get an honest outcome with the crash finally **visible**.

---

## 6. GCP context — turnkey re-run (everything is set up)
- **Instance:** `sdar-a100-od`, **us-central1-b**, `a2-highgpu-4g` (4×A100), `STANDARD`/on-demand (no preemption), warm Qwen/ALFWorld caches (migrated via machine image `sdar-mi-20260620`). **STOPPED** — start to use.
- **Quota:** `A2_CPUS=48` in us-central1 (raised 12→48 this session; auto-granted). Enough for one `a2-highgpu-4g`.
- **Drive it** (the preflight reads these overrides):
  ```
  OPENRESEARCH_GCP_ZONE=us-central1-b OPENRESEARCH_GCP_INSTANCE=sdar-a100-od \
  OPENRESEARCH_GCP_GPU_MACHINE_TYPE=a2-highgpu-4g OPENRESEARCH_SDAR_MIN_GPUS=4 \
  OPENRESEARCH_GCP_PROVISIONING_MODEL=STANDARD bash scripts/gcp_sdar_preflight.sh sync   # push fixed code
  …same env… bash scripts/gcp_sdar_preflight.sh launch                                    # GPU gate + run
  ```
- **Env block:** `runs/.cache/grounded_test_env_block.sh` (no SKIP, fresh id; injected into the VM's `runs/.cache/sdar_gcp.env`). Full capacity/quota notes in memory `[[gcp-a100-capacity-quota]]`.
- **Uncommitted working-tree fixes from this session** (commit them): cred-provider (`models.py` `cred_provider` + `cli.py` call site + `tests/rlm/test_models.py`), provisioning switch (`scripts/gcp_sdar_preflight.sh` `ensure_provisioning_model` + on-demand `remote_prepare` guard), `.env` line-18 quote.

---

## 7. Files to touch
- `backend/agents/rlm/metric_reality_smoke.py` — `_run_one_smoke_cell` (L309–401, capture+return output/rc), `run_metric_reality_smoke` (L424–543, surface in the natural-exit branch L511–517 and the `detail` at L534–537).
- `backend/agents/rlm/primitives.py` — `run_experiment` smoke wiring + repair_context (confirm the new `detail` flows through unchanged; it should — it's pass-through).
- `tests/rlm/test_metric_reality_smoke.py` — the crash-surfacing tests (Fix C).

---

## 8. RESOLVED (2026-06-21) — Fix A confirmed + smoke made production-grade for varied papers + cred preflight hardened

A fresh session re-verified Fix A, then a Codex review surfaced two real issues that
were fixed. All changes are flag-gated/default-OFF-neutral, surgical, and green
(`pytest tests/rlm` clean; `uvx ruff` clean). The SDAR e2e run is the only remaining
step (gated on operator go — money/GPU-hours).

**Confirmed (no change needed):**
- **Fix A is correct and correctly wired.** The crash `detail` flows
  `metric_reality_smoke.run_metric_reality_smoke` → `primitives.py::run_experiment`
  `repair_context={"smoke_detail": …}` → next `implement_baseline`. `smoke_metrics_unreal`
  is in `_RUN_EXPERIMENT_REPAIRABLE_FAILURES`; the fix-first repair-exhaustion terminal is sound.
- **The smoke is NOT miscalibrated for SDAR** — the one local smoke-enabled run
  (`runs/_v3_capture/`) was a *true* rejection (no incremental trace; corroborated by the
  P1 code-review finding hardcoded `0.0`). SDAR logs `loss`/`l_grpo`, which the matcher
  already recognized — so the smoke was never the SDAR *content* blocker; the surfacing
  gap (Fix A) was.

**Shipped this session (`metric_reality_smoke.py`):**
- **Token-aware primary-loss detection** (`_is_primary_loss_key`). The old exact-tuple +
  `\b…\b` regex matched `loss`/`pg_loss` but NOT `train_loss`/`ce_loss`/`mse_loss`/
  `reconstruction_loss` (a word-boundary won't fire across `_`, a word char) — so a
  perfectly real supervised paper logging `{"train_loss": 2.3}` was wrongly rejected as
  `no loss-like key`. Now tokenized on `[^a-z0-9]+`; a key is the optimised loss iff a
  token is `loss` AND none is in `_NON_TRAINING_LOSS_TOKENS` (`val`/`valid`/`validation`/
  `eval`/`test`/`dev`/`holdout`/`scale`) — so `val_loss`/`eval_loss`/`loss_scale` do NOT
  false-pass (Codex's catch). Teeth preserved: an all-zero/constant `*_loss` still fails.
- **Grad-evidence sufficiency** (FIX 2). When NO training-loss key exists, the smoke now
  passes on positive backprop proof (finite `>0` `grad_norm`/`param_delta`) instead of a
  blanket rejection; with neither loss nor grad it rejects with an actionable, paper-
  agnostic hint. SDAR-neutral (SDAR logs a loss).
- **NamedTuple** `_SmokeCellResult` for the 6-field cell result (field-order safety).

**Shipped this session (`pre_flight_validator.py`) — Codex 6a/6b, harness-robustness:**
- The azure-foundry cred preflight read `os.environ` directly while the run reads
  `os.environ → Settings/.env`; a foundry key present only in `.env` (not shell-exported)
  would false-ABORT the run at preflight. `sdar_gcp_run.sh` had hand-lifted exactly this for
  `CLAUDE_CODE_OAUTH_TOKEN` but never generalized it. Fixed at the root: the branch now
  resolves via `foundry_endpoint.resolve_foundry_credentials()` (the run's single source of
  truth, normalized endpoint) — fixes the false-abort AND the raw-endpoint probe-shape gap.

**Tests:** `TestCrashDetailPropagation` (end-to-end: crash marker + `exit N` reach the
verdict `detail` — the actual point of Fix A, previously untested), `TestTokenAwareLoss`,
`TestGradEvidenceSufficiency`, `test_pre_flight_foundry_cred.py`; `test_cred_preflight.py::TestAzureFoundry`
re-pointed at the new single source of truth (contracts preserved, hermetic vs a live `.env`).

**Deliberately deferred (Codex agreed; rationale, not omission):**
- **Per-cell smoke gate** (vs all-or-nothing): a crash usually means a shared-`train_cell.py`
  systemic bug, so blocking the whole grid is defensible; the `scope.gaps` machinery exists
  if we later want it.
- **Single-step `loss>0`-no-grad false-positive tightening:** requiring grad would add NEW
  false-negatives for real trainers that don't log it; the post-run zero-metrics + evidence
  gates backstop fabricated FINAL metrics.
- **VRAM-bypass when sampling fails / inconclusive-timeout pass:** tightening either adds
  false-negatives; both are bounded by downstream guards + the wall-clock cap.
