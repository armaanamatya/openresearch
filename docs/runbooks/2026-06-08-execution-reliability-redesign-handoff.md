# Execution-Reliability Redesign — Handoff (2026-06-08, **v2 — grilled & resolved**)

**Mission:** implement a 5-pillar execution-reliability redesign of `run_experiment` (so a long
GPU training run is *observable while it runs* and its *completed work survives a timeout*), then
**restart two papers** to validation: Adam (`1412.6980`) and All-Conv-Net (`1412.6806`).

Decisions locked by the operator: **build all 5 pillars**; **all timeouts/watchdogs generous**;
local sandbox, 8×RTX 4090; **skip Codex review this round** (re-offer before ship).

> **v2 note.** v1 was grilled section-by-section against the real code on 2026-06-08. Several v1
> claims were wrong or stale; the corrections + resolved design decisions are in §A below and folded
> into every section. Read §A first — it changes where Pillar 3 is wired and what "liveness" means.

---

## A. Corrections to v1 + resolved design decisions (grilling 2026-06-08)

**Corrections (v1 was wrong/stale):**
- **`exec_stalled` is already in the enum.** `interface.py:24` has it (uncommitted, `git diff` = +1
  line). The tree is **not** clean. v1's "§0 step 1: add the enum" is **done**.
- **Pillar 3 was aimed at the wrong line.** Adam did **not** die at a backend `exec_timeout`. It
  died at the **outer experiment-level pool timeout** — `pool.submit(asyncio.run,
  _execute_in_sandbox(...)).result(timeout=timeout)` raising `concurrent.futures.TimeoutError` at
  `primitives.py:4705`, which sets `result={"success":False,"metrics":{},...}`. The inner
  `local_process.exec` never returned `exec_timeout` because the outer timeout (`8835 s`, the mode
  timeout clamped to remaining wall-clock) is **smaller** than the inner per-command cap
  (`_EXEC_TIMEOUT_SECONDS=14400`, `primitives.py:3426`/`:589`), so it pre-empts the inner. Finalize
  must be wired at **both** the outer handler and the inner return.
- **`leaf_scorer._latest_metrics_path` ranks by `(has_results, mtime)`** (per_model/comparison
  non-empty), not bare mtime (`leaf_scorer.py:195`). This is *why* a populated-partial gets picked
  up over an empty placeholder — and it means the agent-half (incremental metrics) is what makes the
  partial scorable; the harness-half makes it *not look like a hard failure*.
- **The Lane-E watchdog was fooled by a dumb heartbeat daemon.** `heartbeat_daemon_command` is
  injected on **local too** (`primitives.py:3066`/`:3070`) and touches `.heartbeat` every 30 s
  "whether the agent remembers or not." That is why Adam's watchdog stayed silent for 2.5 h despite
  zero real progress. Only a **computation** signal (GPU/CPU util) detects an in-process hang.
- **`run_experiment` is absent from the watchdog's per-primitive idle table** (`run_watchdog.py`
  `PRIMITIVE_IDLE_BASELINE_S`: `implement_baseline`=4 h, default 1800 s). So the longest-running
  primitive falls to a ~25–30 min idle kill — *shorter* than the new stall window. Must fix.

**Resolved decisions (all confirmed in grilling):**

| # | Decision |
|---|---|
| 1 | **Finalize-on-timeout at BOTH** the outer `concurrent.futures.TimeoutError` handler (`primitives.py:4705`) **and** the inner `exec_timeout`/`exec_stalled` return. Load latest on-disk metrics via `_latest_metrics_path`; accept when `_per_model_has_measured_value` ≥ 1 family; tag `partial_timeout` (new, **repairable**) with metrics attached; the forced-iteration ≤60 s wall-clock bypass ships the partial. |
| 2 | **Generous + unified liveness.** One signal feeds *both* the stall check and the Lane-E watchdog: **OR of {new stdout/stderr line, ckpt/`metrics.json` mtime bump under `project_root`, GPU-util > ~5% on the pinned *physical* ids, CPU-util (process-tree CPU-time delta)}**. A kill needs *all four* quiet for the full window. Stall window **60 min** (`REPROLAB_EXPERIMENT_STALL_S=3600`, `0` disables). `run_experiment` watchdog idle bumped to **≥ stall (4 h)**. The dumb heartbeat daemon no longer the sole signal. All fail-soft. |
| 3 | **GPU-util polls the *physical* device ids** (`sandbox.config.gpu_device_ids`), never the remapped `cuda:0..N` — concurrent batch runs lease disjoint cards, so a remapped index reads a neighbor's idle card → false-kill. `nvidia-smi -i <physical-id>`. |
| 4 | **CPU-util** = process-tree `utime+stime` delta over the poll interval (stdlib `/proc/<pid>/stat` primary, `psutil.children(recursive=True)` if importable, fail-soft). Covers dataset-download / tokenize / preprocess / CPU-eval phases that sit at 0 % GPU. |
| 5 | **Inner exec owns the deadline** (local-scoped): `run_experiment` passes the resolved experiment timeout as the per-command exec deadline so the **inner** fires first and `kill()`s the subprocess cleanly (no GPU-burning orphan); the outer pool `.result(timeout=…)` becomes a generous backstop (`resolved + buffer`). **runpod/docker exec paths byte-for-byte untouched** (they keep `_EXEC_TIMEOUT_SECONDS` + `outer == resolved`). |
| 6 | **Recovery = repairable `partial_timeout` + Adam-scoped bounding.** A retry only helps if the *structure* changes, so the **Adam `PAPER_HINTS`** entry caps the ~21-config VAE bias-correction sweep and **structures it as `cells.json`** (the harness-reliable form of incremental metrics) — **not** a global cell-route default flip. |
| 7 | **Pillar 5**: honest attribution message (this-run footprint vs free → "shared disk full — GC `runs/.cache/data`" when the run is small) + pre-download dataset-size guard. **Keep the hard block; no auto-delete** — the recipe's manual `rm -rf runs/.cache/{data,envs}` is the unblock (auto-deleting shared re-downloadable caches mid-flight could bite a concurrent run; leave as a deferred opt-in flag). |
| 8 | **Stop condition = reliability bar, not a score threshold.** PASS = Adam writes a populated `metrics.json` that **scores** (not `degraded_no_metrics=0.0`) for the families it completed, **and** the live log shows continuous progress (no multi-hour dead gap), **and** All-CNN runs with **no libcupti death** + non-zero score. 14 h/run; ≤ 2 restarts/paper for a *new* failure. (A rubric number conflates the fix with agent/rubric stochasticity — the scoring-fairness trap.) |
| 9 | **Fan-out**: own **A (`local_process`) + C (`primitives`/`run_watchdog`/`failure_classifier`/`sse_bridge`)** yourself — the timeout seam spans A↔C. Delegate only **B (guidance/hints)**. Preflight smoke set **per-run** in the recipe (don't flip the global default this round). |

**Residual risk (named, accepted):** on the **monolithic** path, recovery still depends on the agent
writing `metrics.json` incrementally — *guidance, not a guarantee* (Adam ignored it once). The only
**harness-reliable** incremental form is `cells.json` (the cell runner writes per-cell metrics
deterministically). That is why decision #6 structures Adam as cells. Reliability is highest on the
cell route, best-effort on the monolith.

---

## 0. First actions (do these in order)

1. **Confirm state:** `git branch --show-current` → `feat/azure-aks-gpu` at `bde6bcf`. `exec_stalled`
   is **already added** (uncommitted). The first *new* edit is Unit A (§4 Pillar 1/2).
2. **Read §A + §2** so you don't conflate the two root causes or re-aim Pillar 3 at the inner path.
3. **Implement** §4 via the §5 partition (own A+C; delegate B).
4. **Test** (§6), then **GC disk + restart** both papers (§7) — confirm GPU-hours with the operator first.
5. Do **not** commit/push unless the operator asks (standing rule §8).

---

## 1. Current state

| Item | Value |
|---|---|
| Branch / HEAD | `feat/azure-aks-gpu` @ `bde6bcf` |
| Uncommitted | `interface.py` (+1: `exec_stalled` enum). Tree NOT clean. |
| Disk | `/home` 99 % (shared, 11T) but **132 GB free** — restart NOT disk-blocked. `runs/.cache/data`=22 GB, `runs/.cache/envs`=2.3 GB — both GC-able (§7). |
| GPUs | 8×RTX 4090 24 GB; 1,3,4,5,6 idle; 0/2/7 light leftover. |
| Python | `/home/sww35/openresearch/.venv/bin/python` — **no `python` on PATH**. `psutil` 7.2.2 importable; `nvidia-smi` present. |

`bde6bcf` already shipped (DONE — do not redo): env_pin, `_venv_cuda_lib_dirs` libcupti fix,
`cuda_shlib_load` class, `cell_scheduler.py` into `code/`, opt-in preflight import smoke. See memory
`harness-env-reliability-fixes.md`.

---

## 2. Why the last two runs failed — TWO DISTINCT root causes

### All-Conv-Net (`prj_0a3202fc187bb692`, 1412.6806) — libcupti, pre-fix → ALREADY FIXED by `bde6bcf`
- exp-2 died at `import torch`: `libcupti.so.12: cannot open shared object file`. The agent's
  `requirements.txt` re-pinned `torch==2.2.0`, downgrading the coherent `2.5.1+cu121` stack.
- **Ran on PRE-`bde6bcf` code** (run finished 01:03 UTC; `bde6bcf` committed 02:18). A clean re-run on
  the current branch strips the re-pin → no downgrade → no libcupti death.
- **Action: none beyond `bde6bcf`.** Restart validates the env fix.

### Adam (`prj_6d41d2f09c026403`, 1412.6980) — outer timeout ate completed work → NEEDS THE NEW FIX
- **It actually trained.** 4 of 5 families finished with real numbers (MNIST-MLP test error **1.69 %**,
  logreg NLL **0.2395**, IMDB, CIFAR10). NOT an environment bug.
- One monolithic `python train.py` packed all 5 families, with an **unbounded ~21-config VAE
  bias-correction sweep** (β₂×lr×optimizer × 20 epochs) last, and wrote `metrics.json` **only at the end**.
- The **outer experiment-level pool timeout** (`8835 s` = `max`-mode 21600 clamped to remaining
  wall-clock) fired **mid-VAE** at `primitives.py:4705` → the 4 completed families' in-memory results
  evaporated → on-disk `metrics.json` frozen at `status:"running"` / empty `per_model` → all 23
  rubric leaves scored `degraded_no_metrics = 0.0`. The inner `exec_timeout` **never fired** (it caps
  at 14400 > 8835). The outer handler also tried to recover logs from `outputs/<run_id>/exec.log` — a
  path `LocalProcessBackend` never writes — so logs were empty too.
- Secondary `disk_exhausted` was **environmental** (shared `/home` full of *other* runs' caches;
  Adam's own outputs = 332 KB). The floor check fired correctly but **mis-attributed** it.
- **No SDK wedge.** The "dead gap" in `dashboard_events.jsonl` is just the `run_experiment` window
  (no events while `train.py` runs). The dumb heartbeat daemon kept the watchdog asleep (§A).

**This is the entire motivation:** completed GPU work must survive a timeout, and a long run must be
observable while it runs.

---

## 3. The two questions that drove the design

- **"Why is there even a timeout?"** Budget (GPU-hours cost; `--max-wall-clock`), genuine hangs
  (deadlock/frozen `load_dataset`), and `run_experiment` is a blocking primitive the root waits on.
  We keep it. The bug: **one fixed wall-clock number did two jobs** (detect-stuck *and* enforce-budget)
  and **discarded work** when it fired. Split liveness from budget; never discard.
- **"Why no debug logs showing it's ongoing?"** `LocalProcessBackend.exec` uses
  `await asyncio.wait_for(process.communicate(), timeout=…)` (`local_process.py:131`).
  `communicate()` **buffers** and returns only at exit — 2.5 h of silence. Fix: **stream**.

---

## 4. The design — 5 pillars (flag-gated, `local`-scoped, fail-soft)

Design principles: (1) **separate liveness from budget**; (2) runtime backend **SSE-agnostic**
(streaming/stall internal; progress *events* at the orchestrator via a sidecar tailer); (3)
**completed work is always scorable**; (4) every change **flag-gated + `local`-scoped** —
runpod/docker `exec` byte-for-byte untouched; (5) **fail-soft everywhere** (an augmentation bug must
never harden into an exec failure).

### Pillar 1 — Live output streaming + heartbeats *(file: `local_process.py`)*
Replace `communicate()` with a draining loop over both pipes (chunk read, split on `\n` **and** `\r`
so tqdm bars count). Per line: append to a **live log** `<project_root>/.exec_live.log` (append mode,
one `=== exec <iso> :: <cmd> ===` header per call) + update a **heartbeat sidecar**
`<project_root>/.exec_heartbeat.json` `{last_output_at, last_line, lines, pid, command}`. Still
accumulate stdout/stderr for `ExecResult` but **cap retained text** head+tail ≤ ~256 KB (live log
keeps everything). `tail -f code/.exec_live.log` answers "is it ongoing."

### Pillar 2 — Stall timeout + GPU/CPU liveness *(file: `local_process.py`; enum done)*
Two independent limits inside `exec`:
- **Hard budget cap** = the `timeout` param → `cause_kind=exec_timeout` (now the resolved experiment
  timeout, decision #5).
- **Stall window** = `REPROLAB_EXPERIMENT_STALL_S` (**default 3600 s**, `0` disables), read from the
  merged env. If **no liveness** for the window → kill → `cause_kind=exec_stalled`.

**Liveness = ANY of (decisions #2–#4):** new stdout/stderr line · ckpt/`metrics.json` mtime bump under
`project_root` (`*.pt`/`*.ckpt`/`metrics.json`) · **GPU-util > ~5 %** on the pinned **physical**
`gpu_device_ids` (`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i <physical-id>`,
poll ~30 s, fail-soft, toggle `REPROLAB_EXPERIMENT_GPU_LIVENESS` default on) · **CPU-util** =
process-tree `utime+stime` delta (`/proc` primary, `psutil` if importable, fail-soft). GPU+CPU are
the load-bearing signals: a real hang sits at ~0 % both; a quiet-but-computing phase (kernel compile,
big matmul, slow dataloader, CPU preprocess) does not. That is why a *generous* window + util is
trustworthy where a fixed wall is not.

### Pillar 2b — Unify with the Lane-E watchdog *(file: `run_watchdog.py` + `primitives.py` wiring)*
Decisions #2/§A: add `run_experiment` to `PRIMITIVE_IDLE_BASELINE_S` at **≥ stall window (4 h)**; feed
GPU+CPU-util into the watchdog liveness so an in-process hang at 0 % util is caught **even though the
dumb `.heartbeat` daemon keeps touching the file** (supersede/gate the daemon when Pillar 1 is active,
or have the watchdog consult the new heartbeat + util). No liveness-kill fires below the stall window.

### Pillar 3 — Finalize-on-timeout + incremental atomic metrics *(THE Adam fix)*
**Agent half (guidance — B):** `train.py` MUST write the canonical `metrics.json` **atomically
(`tmp` + `os.replace`) after EACH family**, populating `per_model` incrementally; `status` stays
non-terminal until the end but *results are already on disk*. Never write metrics only at the end.
(The leaf scorer reads newest-by-`(has_results, mtime)` — `leaf_scorer.py:195` — so each rewrite is
picked up.)

**Harness half (backstop — C, decisions #1/#5):** finalize at **both** timeout layers:
- the **outer** `concurrent.futures.TimeoutError` handler (`primitives.py:4705`) — the path that
  actually fired for Adam, and
- the **inner** `exec_timeout`/`exec_stalled` return (now the primary path once the inner owns the
  deadline).

In each: load the latest on-disk `metrics.json` (reuse `_latest_metrics_path`); if
`_per_model_has_measured_value` is true for ≥ 1 family → attach it, `failure_class="partial_timeout"`
(repairable), `repair_context` naming the done vs timed-out families, route through normal
scoring/aggregation instead of returning zero; else keep the empty-fail. Distinguish "populated
partial" from "empty placeholder" by per_model **measured-value** non-emptiness, **not** by `status`.
Watch the guards at `primitives.py:2538` (`incomplete_metrics`) / `:4912` (placeholder detector) —
they are gated on `success` so they won't re-zero a `success:False` partial, but verify.

### Pillar 4 — Bound the work *(guidance + paper-hint; decision #6)*
Guidance (B) forbids packing N independent families into one un-checkpointed `python train.py`; cap or
stream any sweep; **prefer `cells.json` + `train_cell.py`** so the one-GPU-per-cell route
(`gpu_cell_runner.run_matrix`, gated `primitives.py:4532`) bounds each config with its own timeout.
Add an **Adam `PAPER_HINTS`** entry: cap the VAE bias-correction sweep (the long pole, ~21 configs ×
20 epochs), smallest-config-first, **structured as `cells.json`** (harness-reliable incremental
metrics). Do **not** flip the cell route to unconditional default this round (higher blast radius; the
Pillar 3 net makes it non-urgent).

### Pillar 5 — Disk honesty + pre-download guard + GC *(file: `primitives.py` + ops; decision #7)*
Fix `_disk_floor_violation` (`primitives.py:3791`, called `:4642`/`:4977`): compare **this run's own
footprint** (`code/outputs` + per-run HF/torch cache) against free space; when the run is small but
the volume is full, emit "shared disk full — GC `runs/.cache/data`" instead of blaming this run, and
drop the hard-coded `natural_questions` advice unless attributable. Add a **pre-download dataset-size
guard** near `:4642`. Keep the hard block; **no auto-delete**. **Ops:** `rm -rf runs/.cache/data
runs/.cache/envs` before restart (frees ~24 GB).

---

## 5. Modular implementation plan (fan-out partition — decision #9)

| Unit | Files | Pillars | Owner |
|---|---|---|---|
| **A — Execution layer** | `local_process.py` (enum done) | 1, 2 | **You** (seam spans A↔C). |
| **C — Orchestrator + watchdog + disk + classify + SSE** | `primitives.py`, `run_watchdog.py`, `failure_classifier.py`, `sse_bridge.py`, `.env.example`, `CLAUDE.md` | 2b, 3-harness, 5, cross-cutting | **You.** |
| **B — Guidance** | `baseline_implementation.py`, `system_prompt.py`, `paper_hints.py` | 3-agent, 4 | **Delegate** (isolated text/config). |

**Exec contract (A↔C):** keep `RuntimeBackend.exec(sandbox, command, timeout)` **stable** (runpod/
docker untouched). `LocalProcessBackend` implements streaming + 4-signal stall internally, reads
config from the merged env, writes the live log + heartbeat sidecar under `project_root`, stays
SSE-agnostic. **Inner owns the deadline (local only):** `run_experiment` passes the resolved
experiment timeout as the per-command exec deadline; the outer pool `.result()` uses `resolved +
buffer`. **Progress→SSE** (C): an async tailer polls `.exec_heartbeat.json` ~30 s and emits a
sanitized `experiment_progress` through `sse_bridge.sanitize_iteration` (add to the allowlist).

**Metrics contract (B↔C):** agent writes `metrics.json` atomically after each family (B);
finalize-on-timeout reads newest-by-`(has_results,mtime)` and scores populated-partial (C). Both
reference `leaf_scorer.py:195`.

Each pillar independently flag-gated + unit-tested → ship/rollback piecemeal.

---

## 6. Testing

Beside `tests/rlm/test_env_reliability_fixes.py` and `tests/services/runtime/`:
- `exec` streams to live log + heartbeat sidecar; caps retained stdout; returns `exec_stalled` when a
  fake command emits nothing / no util past the stall window; returns `exec_timeout` at the hard cap;
  **GPU and CPU** liveness fail-soft when `nvidia-smi`/`/proc` absent.
- finalize-on-timeout at **both** layers: a populated-but-`status:running` `metrics.json` after the
  outer `TimeoutError` **and** after an inner `exec_timeout`/`exec_stalled` → scored partial, not
  zeroed; a truly empty placeholder → still rejected.
- watchdog: `run_experiment` idle threshold ≥ stall window; GPU/CPU liveness consulted; the dumb
  daemon alone does not keep a 0-util hang alive.
- `_disk_floor_violation`: small-run-on-full-volume → "shared disk full" message, not "this run's
  download."
- `failure_classifier` maps `exec_stalled` **and** `partial_timeout` to repairable classes with a
  suggested_fix.
- Regression gate: `.venv/bin/python -m pytest tests/services/runtime tests/rlm -q` (bde6bcf scope
  baseline **1686 / 0-regress**). Run full `tests/` before any restart.

---

## 7. Restart recipe (after fixes land + tests green; **confirm GPU-hours with operator first**)

**GC first (frees ~24 GB, exercises Pillar 5's premise):**
```bash
rm -rf runs/.cache/data runs/.cache/envs
```
**Launch both small papers (local, OAuth = $0, preflight smoke ON, 14 h cap, generous stall).** Verify
flags via `scripts/batch_reproduce.py --help` + `backend.cli reproduce --help` first; the `env -u`
prefix prevents a stale shell key from shadowing `.env` (§8):
```bash
# Option A — batch scheduler (leases disjoint GPUs, per-run venv + shared HF_HOME):
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY REPROLAB_PREFLIGHT_SMOKE=1 \
  REPROLAB_EXPERIMENT_STALL_S=3600 \
  .venv/bin/python scripts/batch_reproduce.py 1412.6980 1412.6806 \
    --gpus-per-run 2 --model claude-oauth --max-wall-clock 50400

# Option B — single CLI per paper:
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY REPROLAB_PREFLIGHT_SMOKE=1 \
  REPROLAB_EXPERIMENT_STALL_S=3600 \
  .venv/bin/python -m backend.cli reproduce 1412.6980 \
    --sandbox local --mode rlm --model claude-oauth --max-wall-clock 50400
```
CIFAR/MNIST-scale → 1–2 GPUs each; can run concurrently on the 8×4090.
**Watch it run:** `tail -f runs/<id>/code/.exec_live.log` and the `experiment_progress` SSE /
`dashboard_events.jsonl`.

**Done-criteria (reliability bar, decision #8):** both runs emit a **populated, scorable**
`metrics.json` (not `degraded_no_metrics=0.0`) and the live log shows continuous progress (no dead
multi-hour gap). All-CNN validates `bde6bcf` (no libcupti); Adam validates the redesign (timeout no
longer zeroes completed families). 14 h/run; ≤ 2 restarts/paper for a *new* failure.

---

## 8. Standing constraints (non-negotiable — from operator memory)

- **Commit as `lolout1`, NO `Co-Authored-By: Claude` trailer.** Commit/push **only when explicitly
  asked** (this redesign is uncommitted until the operator says so).
- **Long reproduction runs cap at 14 h (`--max-wall-clock 50400`)**, not 8 h.
- **Shell env shadows `.env`** → prefix `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY ...`.
- **Cheapest local:** `--model claude-oauth` ($0 both surfaces). No `python` on PATH → `.venv/bin/python`.
- **Workflow:** fan out sub-agents, parallel, efficient without sacrificing quality; operator usually
  wants **Codex adversarial review + a plan at the end** (waived for *this* build round — **re-offer
  before ship/merge**).
- `local`/runpod/docker parity: all changes `local`-scoped + flag-gated; do not perturb runpod/docker exec.

---

## 9. Appendix — grounded code anchors (verified 2026-06-08)

```
backend/services/runtime/interface.py
  :19  RuntimeCauseKind  · :24 exec_stalled (ALREADY ADDED, uncommitted)
  :79  ExecResult (frozen: stdout/stderr/cause_kind/timed_out/succeeded)
  :116 RuntimeBackend.exec(sandbox, command, timeout) (abstract; keep stable)

backend/services/runtime/local_process.py
  :25  _venv_cuda_lib_dirs (bde6bcf libcupti fix)
  :102 LocalProcessBackend.exec  · :123 create_subprocess_shell  · :131 communicate()+wait_for (REWRITE: stream + 4-signal stall)
  :135 TimeoutError -> kill + exec_timeout (add exec_stalled branch + finalize note)

backend/agents/rlm/primitives.py
  :452 EXPERIMENT_TIMEOUT_BY_MODE (efficient 7200 / max 21600)  · :459 resolve_experiment_timeout_s  · :589 _EXEC_TIMEOUT_SECONDS=14400
  :2507 _NON_TERMINAL_STATUSES  · :2512 _per_model_has_measured_value  · :2538 _metrics_completeness_violation (incomplete_metrics; gated on success)
  :3066 heartbeat_daemon_command injection (:3070 local hb dir)  · :3237 watchdog poll  · :3402 _run_watchdog start  · :3426 inner per-command exec timeout (=_EXEC_TIMEOUT_SECONDS)
  :3791 _disk_floor_violation (Pillar 5)
  :4452 def run_experiment  · :4532 cells.json gate  · :4619 timeout=resolve_experiment_timeout_s  · :4642 disk pre-check
  :4674 _execute_cell_matrix (cell route)  · :4704 outer pool .result(timeout)  · :4705 TimeoutError -> metrics={} (FINALIZE HERE)  · :4912 placeholder detector  · :4977 disk post-check

backend/evals/paperbench/leaf_scorer.py
  :195 _latest_metrics_path — ranks by (has_results, mtime); has_results = per_model OR comparison

backend/agents/rlm/run_watchdog.py
  PRIMITIVE_IDLE_BASELINE_S (implement_baseline=14400; default 1800; ADD run_experiment>=stall)
  WatchdogConfig warn=600/kill=1500; watches exec.log/.heartbeat/dashboard_events (ADD GPU+CPU util)

backend/agents/rlm/failure_classifier.py  — add exec_stalled + partial_timeout (repairable, mirror exec_timeout)
backend/agents/rlm/sse_bridge.py          — allowlist experiment_progress through sanitize_iteration
backend/agents/rlm/baseline_implementation.py · system_prompt.py · backend/agents/prompts/paper_hints.py  (Unit B)
```

**Failed run dirs (evidence):** `runs/prj_6d41d2f09c026403` (Adam),
`runs/prj_0a3202fc187bb692` (All-CNN). Adam's timed-out train.py:
`runs/prj_6d41d2f09c026403/attempts/20260607T230239-304380-f28fad/code/train.py` (VAE sweep ~706-767).

**Related memory:** `execution-reliability-redesign.md`, `harness-env-reliability-fixes.md`,
`scoring-fairness-spec.md`, `run-walltime-preference.md`, `commit-attribution-preference.md`.

---
*v1 authored 2026-06-08. v2 (grilled & resolved) 2026-06-08. Start at §0; read §A first.*
