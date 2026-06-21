# Workstream C — Hallucination Diagnosis (READ-ONLY)

> Date: 2026-06-20 · Auditor pass · Scope: make the "agents are hallucinating"
> complaint **falsifiable + concrete**. No production code written. Every claim
> cites a `file:line` and/or an on-disk run-artifact path.

---

## 1. Executive summary + verdict

**Headline (the falsifiable answer to "agents are hallucinating"):** On every
run artifact present on disk, the *shipped report* is **honest**. All 8 runs
under `runs/` returned `verdict ∈ {failed, partial}` with truthful low scores
(0.0–0.188); none claimed a success it did not earn. The strong-form complaint —
*fabricated results in a shipped report that passed* — is **NOT substantiated by
the available data**. The report-level evidence gate (`report.py:1496`, default
**ON**) and the `degraded_no_metrics` zeroing path are doing their job.

What I *did* find are weaker, latent, and grader-leniency issues:

- **One concrete latent defect (HARNESS-TRUST):** `run_experiment` records
  `success=true, status=complete` for a run where **every model errored**
  (`primitives.py:3982` — `success = all(r.succeeded ...)` checks subprocess exit
  codes, never `per_model` cell status). On the one observed instance it did
  **not** propagate to a green headline (the grader independently scored the run
  0.188 → `failed`), so today it is a *latent* greenlight, dangerous only if
  grader credit ever clears target on a run whose models all silently failed.
- **One config defect (HARNESS-TRUST):** the per-leaf evidence veto that would
  catch a credited-but-unmeasured result leaf is **default OFF**
  (`evidence_gate.py:48` / `leaf_scorer.py:291`), while the identically-named
  *verdict*-level gate is default ON (`report.py:1496`). Same env var, opposite
  defaults, different granularity — confusing and under-protective at the leaf.
- **Soft grader leniency (model-quality-adjacent, NOT fabrication):** the grader
  credits "Method/code fidelity" leaves 0.5–0.8 from reading docstrings/declared
  constants while hedging ("only partial code is visible"). This is *by design*
  (method leaves are exempt from the result veto, `evidence_gate.py:56-82`); it
  is leniency, not a hallucination.

**Verdict: lean HARNESS-TRUST, LOW confidence.** The one concrete actionable
defect is a harness aggregation/gating bug (the greenlight), not the model
inventing numbers in a report. I explicitly do **not** claim systemic
model-quality fabrication — the on-disk reports are honest, and the data is too
sparse (and too uniformly *failed*) to charge the models with fabricating
shipped results. The opposite-fix dichotomy resolves toward **tightening the
evidence gates** (flip the greenlight check + the leaf veto), not provider/model
routing — but the confidence is low because no run produced a fabricated *headline*
to diagnose, only a latent path to one.

---

## 2. Data availability note

| Run dir | verdict | Has full evidence chain? | Used? |
|---|---|---|---|
| `prj_09047604e591d969` | failed (0.188) | rubric_evaluation + experiment_runs + 15 graded leaves; **no `code/` retained** | **YES (primary)** |
| `prj_e2d9aebb05d4340f` | failed (0.0) | rate-limited ("hit your limit"), all `degraded_no_metrics` | context only |
| `pb_mechanistic-understanding_1780068083` | partial (0.0) | rubric_evaluation + `code/metrics.json` placeholder; all `degraded_no_metrics` | context only |
| `pb_mechanistic-understanding_1780068784` | partial (0.0) | no `code/`, score 0.0 | context only |
| `bes_a1_oauth_clean1` | failed (None) | SIGTERM partial; `bes_winner.json` present | secondary (BES note) |
| `bes_a1_oauth_runpod_1` | failed | tiny (419-byte ledger), barely started | no |
| `bes_a1_1412_6980_1781707862` | failed | `code/` present but no metrics/rubric | no |
| `bes_a1_1412_6980_1781710336` | failed | minimal | no |

**Severe data scarcity.** 8 run dirs total; **all 8 are `failed`/`partial`** —
there is **zero** completed-and-passing run to inspect for a fabricated headline.
Most are interrupted (SIGTERM / rate-limit / barely-started). Exactly one run
(`prj_09047604e591d969`) carries a real graded leaf set + a populated
`experiment_runs.jsonl`, so it is the load-bearing artifact for this audit. Per
the task's explicit license, I report **2 bulletproof examples + this scarcity
note** rather than inventing a third.

---

## 3. Concrete examples

| # | Run id | Hallucinator | Kind | Claim artifact:line | Refuting artifact:line | Should-have-caught backstop | Class |
|---|---|---|---|---|---|---|---|
| 1 | `prj_09047604e591d969` | **harness** (`run_experiment` aggregator) | (c) failed experiment reported as success | `runs/prj_09047604e591d969/experiment_runs.jsonl` row 0: `"success": true`, `metrics.status:"complete"` | same file, same row `metrics.per_model`: `qwen3_1_7b → {error:"ValueError", "...does not recognize this architecture"}`, `qwen2_5_3b → {error:"TypeError","...unexpected keyword argument 'dtype'"}` — **0 models trained**, all metrics `0.0`. Aggregation logic: `backend/agents/rlm/primitives.py:3982` `"success": all(r.succeeded for r in results)` (subprocess-exit only) | `dead_training_guard.py` / a per_model-status success check | **Class 2** (latent gap) |
| 2 | `prj_09047604e591d969` | **executor sub-agent** (`claude-sonnet-4-6`) | (d) fabricated library/API usage | refuting+claim co-located: `experiment_runs.jsonl` row 0 `metrics.per_model.qwen2_5_3b.detail`: `"__init__() got an unexpected keyword argument 'dtype'"` — the executor's model-loading call passed a kwarg the installed Transformers does not accept | same row: the call raised at load (`error:"TypeError"`), so the API usage is provably invalid against the runtime. **Note:** `runs/prj_09047604e591d969/code/` is **not retained** (GC'd) so the exact `from_pretrained(...)` source line cannot be cited — the error string in `experiment_runs.jsonl` is the only surviving on-disk proof. | `preflight_ast` / `rubric_guard.assert_metrics_schema` (neither catches a wrong kwarg to a 3rd-party API) | **Class 3** |

### Why the two contestable candidates were dropped (rigor note)
- **Grader credits method leaves 0.7/0.8** (`rubric_evaluation` leaves
  `2570128b…`=0.7, `7b327d40…`=0.8): the grader read the code and judged
  `β=10`/`λ=0.1` present — those constants *are* declared, so the credit is a
  true judgment, and method-fidelity leaves are *intentionally exempt* from the
  result veto (`evidence_gate.py:56-82`). This is **grader leniency**, not a
  hallucination. Recorded in §5 (Class-2 tuning), not the table.
- **BES static SELECT score 0.701** (`bes_a1_oauth_clean1/rlm_state/bes_winner.json`):
  static code-only grading is the *documented* SELECT mechanism, and
  `CLAUDE.md` + the 2026-06-17 BES spec already state the SELECT margin "sits
  inside grader noise" (the reason the evolutionary redesign was cut).
  "Unsubstantiated by execution" is tautological — SELECT is pre-execution by
  design. **Not a hallucination**; dropped.

---

## 4. Backstop inventory

| Backstop (file) | Flag | Default | What it checks | What it does NOT check |
|---|---|---|---|---|
| **Per-leaf evidence veto** `evidence_gate.py` + `leaf_scorer.py::_result_leaf_substantiated` (`leaf_scorer.py:708`) | `OPENRESEARCH_EVIDENCE_GATE` | **OFF** (`evidence_gate.py:48`, `leaf_scorer.py:291`) | Vetoes to 0.0 a leaf the grader credited (>0) that *claims a measured result* (`leaf_claims_measured_result`, category `result`/`metric`) when its `per_model` cell has no successful on-disk cell. | Method/code-fidelity, judgment, theory leaves (exempt by `leaf_claims_measured_result`, `evidence_gate.py:56-82`). So Example-style code-description credit is never vetoed. |
| **Verdict-level evidence gate** `report.py::_apply_evidence_gate` (`report.py:1496`) | `OPENRESEARCH_EVIDENCE_GATE` (**same var, opposite default**) | **ON** (`report.py:1496` defaults `"1"`) | Downgrades a `reproduced`/`partial` verdict to `failed` when an on-disk success+metrics row is not backed by ≥1 success-compatible in-process `run_experiment` ledger call (forge defense). | Per-leaf scores; the `success=True` aggregation itself; whether `per_model` cells actually succeeded (it trusts the row's success bool + ledger outcome, not cell status). |
| **`run_experiment` success aggregation** `primitives.py:3982` | — (always on) | — | `success = all(r.succeeded for r in results)` — every subprocess exited 0. | **Whether any `per_model` cell trained.** All-models-errored-but-exit-0 ⇒ `success=true` (Example 1). The cells-route aggregator `cell_matrix.aggregate_cell_metrics` (`cell_matrix.py:822-827`) DOES set `failed` when no cell is ok — but the **monolithic path** does not, and row 0 took the monolithic path. |
| **dead_training_guard** `dead_training_guard.py` | `OPENRESEARCH_DEAD_LOSS_EARLYSTOP` | **OFF** (`dead_training_guard.py:67`) | Flat/non-descending loss curve ⇒ early-stop "dead training". | A run that never produced a loss curve at all (load-time crash, Example 1) — there are no loss lines to observe. |
| **champion_artifact** `champion_artifact.py` | `OPENRESEARCH_CHAMPION_ARTIFACT` | OFF | Snapshots `code/` per verify keyed by `evidence_key`; restores highest-median at finalize. | Whether the champion's metrics were measured vs fabricated (it preserves whatever was graded). |
| **deterministic_leaf_checker** `leaf_scorer` | `OPENRESEARCH_DETERMINISTIC_LEAVES` | OFF | Routes leaves with `check_kind`+`assertion` through pure-Python checks (hparam vs `provenance.json`, artifact vs FS, numeric trend vs `metrics.json`). | Un-annotated/judgment leaves (the majority of an auto-generated rubric). |
| **rubric_guard** `rubric_guard.py::assert_metrics_schema` | — (emitted into agent train.py) | active when emitted | End-of-script: required metric keys/artifacts present ⇒ else `RubricGuardFailure`. | Whether the *values* are real or whether a wrong 3rd-party API kwarg was used (Example 2). |
| **two-axis upgrade clamp** `two_axis_report.py` | `OPENRESEARCH_TWO_AXIS_VERDICT` | OFF | A verdict UPGRADE needs ≥1 success-compatible `run_experiment` ledger call (forged green cert can't lift). | Downgrades (free); per-leaf credit. |
| **`degraded_no_metrics` zeroing** (leaf_scorer) | — (always on) | — | No metrics on disk ⇒ all leaves 0.0, `degraded:true`. **This is the workhorse keeping the 8 observed reports honest.** | A run that DID write a partial/placeholder `metrics.json` (e.g. `1780068083` placeholder still scored 0.0 correctly here, but a *populated-yet-fabricated* metrics.json would pass this gate and rely on the per-leaf veto, which is OFF). |

---

## 5. The three classes + minimal-closure suggestions

### Class 1 — backstop EXISTS but is OFF by default
- **Per-leaf evidence veto** — `OPENRESEARCH_EVIDENCE_GATE` default **OFF**
  (`evidence_gate.py:48`). Closure (prefer tightening existing gate): flip the
  *leaf-veto* consumer to default ON to match its verdict-level twin, OR rename
  one of the two consumers so the opposite defaults stop sharing a var name. The
  σ-gate precondition (grader noise ≤0.02) is already documented as MET in
  CLAUDE.md, so the flag is *eligible* to flip — it just needs the per-step A/B
  sanity run (§6).

### Class 2 — backstop EXISTS / should apply but has a GAP
- **`run_experiment` greenlight** (`primitives.py:3982`). Gap: monolithic-path
  `success` ignores `per_model` cell status; an all-models-errored run is
  `success=true`. Minimal closure (tighten, no new machinery): after building
  `metrics`, downgrade `success=False` (or `status="failed"`) when `per_model`
  is non-empty and **no** cell has an ok status — mirror the cells-route rule
  already in `cell_matrix.py:822-827`. Fail-closed, default-ON candidate (it can
  only turn a fake-green into an honest-red).
- **Grader leniency on description-only leaves** — not a code gap so much as a
  prompt/scoring-policy choice; lower priority. If pursued, cap method-fidelity
  credit when the justification self-reports "only partial code visible" — but
  this risks rejecting good runs and should stay behind a flag.

### Class 3 — NO backstop exists
- **Fabricated 3rd-party API usage** (Example 2: wrong `from_pretrained` kwarg).
  No gate validates the executor's generated calls against the *installed*
  library surface before run. Sketch of an evidence-first, deterministic,
  fail-closed, **default-OFF** check: a pre-run AST/import lint of `code/` that,
  for a curated allowlist of hot APIs (`AutoModel*.from_pretrained`,
  `Trainer(...)`), checks kwargs against the installed package's signature via
  `inspect.signature` and emits a repairable `failure_class=invalid_api_kwarg`
  *before* the GPU spend — zero paid GPU, pure local. Default OFF until A/B
  shows it doesn't false-positive on legitimate `**kwargs`-forwarding APIs.

**Prioritization:** (1) Class 2 greenlight fix — highest value, fail-closed,
smallest diff, directly closes the one concrete defect. (2) Class 1 leaf-veto
default flip — cheap, eligible, defends the *populated-yet-fabricated metrics.json*
shape the `degraded_no_metrics` workhorse does NOT cover. (3) Class 3 API-lint —
new machinery, lowest priority, default-OFF.

---

## 6. A/B validation plan stubs (repo policy: ≥3 paired A/B runs before any default flip; ZERO paid GPU)

**Fix 1 — greenlight (`primitives.py:3982`):**
- *Unit/replay (no GPU):* feed the saved
  `runs/prj_09047604e591d969/experiment_runs.jsonl` row 0 through the aggregation
  function; assert post-fix `success=False` / `status=failed` given all-errored
  `per_model`. Add a positive case (≥1 ok cell ⇒ still success). Pure replay.
- *A/B:* 3 paired runs (control vs patched) on a paper where models reliably
  load (positive) + 1 deliberately-broken-loader paper (negative); confirm the
  patched arm flips only the all-failed case and is byte-identical otherwise.
  Local sandbox, no paid GPU.

**Fix 2 — leaf-veto default flip (`evidence_gate.py:48`):**
- *Calibrate grader σ first* (policy): `scripts/calibrate_grader.py` K=3 on a
  saved run → confirm per-leaf σ ≤ 0.02 (CLAUDE.md records σ=0.0067 already; re-
  confirm on this artifact set) before flipping, so a noisy grade doesn't trip
  the veto.
- *Replay:* re-score `prj_09047604e591d969`'s rubric_evaluation with the flag
  ON vs OFF over the on-disk metrics; assert only genuinely-unsubstantiated
  result leaves move, method/judgment leaves unchanged. No GPU.
- *A/B:* ≥3 paired runs; verify overall-score delta is within grader σ on honest
  runs (no regression) and strictly down on a fabricated-metrics fixture.

**Fix 3 — API-kwarg lint (new, default-OFF):**
- *Unit only:* synthetic `code/` fixtures — one with `from_pretrained(dtype=...)`
  (the Example-2 shape) ⇒ must flag; one with legitimate `**kwargs` forwarding ⇒
  must NOT flag. No run, no GPU. Keep default OFF pending false-positive A/B.

---

### Appendix — load-bearing artifact paths
- `runs/prj_09047604e591d969/experiment_runs.jsonl` (rows: `success=true/status=complete` with errored `per_model`)
- `runs/prj_09047604e591d969/final_report.json` (verdict `failed`, 0.188; `baseline_metrics.status:"running"`)
- `runs/prj_09047604e591d969/rubric_evaluation.json` (leaf credits 0.5–0.8 on method leaves)
- `runs/bes_a1_oauth_clean1/rlm_state/bes_winner.json` (static SELECT scores — documented, not a hallucination)
- `backend/agents/rlm/primitives.py:3982` · `backend/agents/rlm/evidence_gate.py:48,56-82` · `backend/evals/paperbench/leaf_scorer.py:291,708` · `backend/agents/rlm/report.py:1496` · `backend/agents/rlm/cell_matrix.py:822-827` · `backend/agents/rlm/dead_training_guard.py:67`
