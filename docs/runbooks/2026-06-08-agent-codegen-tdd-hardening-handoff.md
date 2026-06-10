# Two-Axis Reproducibility Verdict + Pre-Training Fidelity Gate (handoff)

**Date:** 2026-06-08 · **Branch:** `feat/azure-aks-gpu` (same tree as the execution-reliability redesign) · **Status:** DESIGN LOCKED (grilled — round 2 expanded the mission), NOT YET IMPLEMENTED. This doc is self-contained — a fresh session can execute it without the originating conversation.

> **Scope note (2026-06-08, grilling round 2):** the original mission below ("catch codegen bugs before GPU burn") is now understood as a *subset* of a larger one. The TDD gate produces a **fidelity certificate**; that certificate exists to license an honest **two-axis reproducibility verdict** (did *we* build it faithfully ⟂ does the *paper's* claim hold). **Read Part A first** — it is the framing the rest of this doc serves.

> Companion docs: `2026-06-08-execution-reliability-redesign-handoff.md` (the *runtime* reliability work that just shipped @ `662e018`; this is the *codegen-correctness* sibling). Memory: `[[execution-reliability-redesign]]`, `[[scoring-fairness-spec]]`, `[[harness-env-reliability-fixes]]`.

---

## Part A — The reproducibility verdict this gate exists to serve (LOCKED 2026-06-08, grilling round 2)

**Why this section leads.** This system is built on **PaperBench** (`backend/evals/paperbench/{leaf_scorer,score,bundle,submission}.py`, `third_party/paperbench/`, the `task_category` taxonomy: Code Development / Execution / Result match / subtree). PaperBench grades **agents** on a *curated set of known-reproducible papers* (rubrics built with the original authors) and rolls all leaves into one "replication score" — so a Result-Match miss is, by construction, *the agent's* failure. **We point the same machinery at arbitrary arXiv papers whose reproducibility is UNKNOWN.** That breaks the assumption: a Result-Match miss is now ambiguous — *our* bug, or the *paper's* claim failing to replicate? A single score cannot say. The two-axis verdict is the adaptation that resolves it, and the TDD gate (Parts §1–§12) is what makes the resolution *trustworthy*.

**The two axes** (they map onto the existing rubric *areas* — this is surfacing + gating, **not** new scoring math):

| Axis | Values | The question | Source |
|---|---|---|---|
| `implementation_verdict` (fidelity) | faithful / partial / broken | Did **we** build the paper's method right? | "Method and code fidelity" area **+ the executable invariant certificate** |
| `replication_verdict` | replicated / partially-replicated / contradicted / inconclusive | Given a faithful build, did the **paper's** claims hold? | "Result match" area, **graded** (decision 2) |

**Locked decisions (grilling round 2):**

1. **Gating — the keystone.** `replication_verdict ∈ {replicated, contradicted}` is emittable **ONLY** when the fidelity certificate is green **AND ≥2 seeds agree**. No certificate, or <2 seeds → `inconclusive`. A contradiction is a strong scientific claim ("this published paper does not replicate") and must be *earned* — never asserted off an uncertified or single-seed run. This is *why* the gate (below) is load-bearing, not a cost optimization.

2. **Graded replication credit.** Not binary, not direction-only. Magnitude matches the claim → ~1.0 (`replicated`). Right direction but magnitude far off → **~0.75** (`partially replicated` — significant credit, not full). Direction inverts → low / `contradicted`. Today's D5 (`leaf_scorer.py:906` (`_SYSTEM_PROMPT` RESULT-MATCH block)) gives direction-only *full* credit; this refines it. (Rationale: "94.8% ≈ 95% = effective" only because *magnitude* is close, not merely "above a baseline.")

3. **Seeds.** User-configurable; **≥2 seeds minimum to declare `contradicted`** (one run cannot out-vote variance). The seed basis is shown plainly in the UI (decision 6).

4. **Executable spine, not LLM panels** (the "good-practice" correction — you do not manufacture ground truth by stacking LLM judges). The certificate rests on **executable invariant tests**: *run* a forward pass and assert `g == σ(10·Δ)` to tolerance — deterministic, green-or-not. **Mutation** (Tier 4) confirms the tests actually *bite* (defeats tautological tests). The **LLM consensus/refute pass is a thin LAST layer**, only for what genuinely cannot execute (does the paper's *prose* claim a *direction* the measured trend matches?). Certificate confidence = (executable invariants passed + provenance quality), **NOT** hand-curated hints.

5. **No hints — dynamic for ANY paper.** A **role-separated Extractor** mines the paper's falsifiable quantitative claims (full mining, ranked by centrality, **top-K** tested), each tied to a **paper-span citation**, and freezes a **`ReproSpec`** (claims + invariants + *claimed magnitudes*) **BEFORE the implementer runs**. `PAPER_HINTS` is demoted to an **optional cache/override** — never load-bearing. **Anti-circularity (writer-of-code ≠ definer-of-truth):** the spec is frozen by the Extractor and checked by an *independent* verifier, so a β=10→β=1 misread cannot certify itself (the same agent can't write impl + test + constant to its own wrong value and pass). **Claimed-magnitude extraction is its OWN unit with its OWN fixtures** — it is the fragile surface that decision 2's graded score depends on.

6. **UI/UX (`/lab`).** A two-axis **verdict card** (e.g. `fidelity 🟢 faithful · replication: contradicted · 0.75 · 2 seeds`) + a **measured-vs-claimed graph** per metric + plain-language "what this means." User-useful info (verdict, confidence, seed basis, friendly errors) is surfaced; **developer internals (raw logs, failure classes, raw events) stay behind a collapsed "developer details" panel**, enforced at the `sse_bridge` egress (extend its sanitization into an *audience* tag: user vs developer). Well-formatted, aesthetic.

7. **Leaderboard / schema.** **Additive** — keep `verdict`/`overall_score` for back-compat (`report_resolution.py`, leaderboard, `best_runs` don't break), ADD the two axes. Leaderboard **ranks by fidelity** (quality of the reproduction job); replication is a **labeled badge, NOT a rank penalty**. A faithful-but-contradicted run ranks HIGH and visibly flags the paper as non-replicating.

8. **Validation.** **Synthetic golden fixtures** assert every (fidelity × replication) quadrant lands in the right cell — faithful+refuting → `(faithful, contradicted)` **not** `failed`; buggy/surrogate → `(broken, inconclusive)`. This deterministic quadrant rail is the **merge gate for the full fan-out** (decision below): build all units in parallel, but **nothing merges until the rail is green**.

**Litmus test for the whole design:** a faithful reproduction that finds *SDAR does not beat GRPO* must emit `(implementation: faithful, replication: contradicted)` and rank as an **excellent reproduction carrying a negative finding** — never `failed`. If that case scores 0.0, the architecture has regressed to PaperBench's single-score assumption.

### Part A.1 — Codex adversarial-review resolution (LOCKED 2026-06-08)

Codex reviewed Part A (6-min run; every cited `file:line` was **directly verified** against the code — `leaf_scorer.py:906`, `report_resolution.py:44/217`, `leaderboard.py:200`, `primitives.py:2547` all confirmed; NOT a hallucination). The review materially hardened the design. **Unifying fix (Codex's #1 risk): a contradiction is a CLAIM-LEVEL claim under a typed eligibility contract — never a paper-level "green cert + 2 seeds" shortcut.** Adopted:

- **A1 — Typed `ComparisonSpec` per claim** (was U14 "extract a number"). Freeze: estimate kind (percentage-points vs relative-%), unit/scale, metric direction (higher/lower-better), baseline/comparator, model+dataset+split, aggregation, uncertainty, exact table coordinates, headline-vs-ablation. **Any ambiguity → `inconclusive`.** (Findings 1,3,10.)
- **A2 — Claim-scope eligibility gate (BLOCKER).** Paper-level `contradicted` only for a *predeclared primary* claim at *matching* scope. A result from the cost-bounded smallest-two models (SDAR 1.7B+3B) can **never** contradict a 7B-specific claim → claim-level result + paper-level `inconclusive`. (Finding 2 — the worst false-contradiction path in THIS codebase, given `REPROLAB_BASELINE_EXTRA_GUIDANCE` smallest-two scoping.)
- **A3 — Seed bundle with CI + equivalence region** (was "≥2 seeds agree"). Contradiction requires a seed-bundle artifact (per-seed metrics + verified-independent RNG) whose effect **CI excludes a claim-specific equivalence region**. Two agreeing seeds = necessary, not sufficient. Replication credit is **continuous by recovered-effect-fraction with uncertainty bands** (replaces the flat ~0.75 floor; label decoupled from leaf pass thresholds). (Findings 3,10.)
- **A4 — Schema-versioned verdict; legacy projected from FIDELITY (BLOCKER — refutes "additive is clean").** `reconcile_verdict_with_score` (`report.py:577`) downgrades by the *blended* `overall_score`, so a faithful-contradicted run whose result-match leaves drag the score down collapses to legacy `failed`. Fix: add `schema_version`; two-axis reports **skip** the aggregate-score reconcile; legacy `verdict` projection derives from the **fidelity axis only**. Historical reports keep current logic. (Finding 6.)
- **A5 — Fidelity-aware best-attempt + leaderboard rank.** Preserve `extract_scores` (`report_resolution.py:44`) tuple (callers unbroken); ADD a schema-aware fidelity rank used in BOTH `resolve_best_report._rank` (`:217`) AND `leaderboard._sort_key` (`:200`) — else a broken-high-score attempt is picked over the faithful-negative one and the latter never reaches the board. (Finding 7.)
- **A6 — Real independence, not just role-separation (BLOCKER).** (a) A **blinded verifier** re-extracts each claim from the *raw cited paper spans* — not from the Extractor's conclusions or the implementation; semantic disagreement → `inconclusive`. (b) Invariant tests are **verifier-owned**, generated from the frozen spec, invoke the **actual production train/eval entry points**, mutate **production** code, and require **execution-trace evidence** the asserted path ran — closing the "implementer tests a decoy" hole. Fix `must_match` OR-semantics where the spec needs BOTH terms (`leaf_scorer.py:~1016`; SDAR GRPO+distill). (Findings 4,5.)
- **A7 — Certificate obligation profiles** {static, forward-pass, multi-step, trace, end-to-end} scale the certificate to the claim type, so deterministic/eval-only/official-single-seed papers aren't needlessly blocked and long-horizon invariants aren't "certified" by a 1-step smoke. Single-run *positive* verdict only for demonstrably deterministic claims; *contradiction* always needs A3. (Finding 8 — also satisfies the original "paper-type-agnostic" mandate.)
- **A8 — Lighter metric-honesty gate (finding 9 scoped, NOT gold-plated).** Full independent metric recomputation (harness re-evaluates from persisted predictions/checkpoint/split) is its **own future epic** — flagged, not built here. For now the contradiction path requires: held-out-split assertion (no train leakage), non-constant/non-degenerate check, checkpoint+split provenance in the seed bundle, persisted prediction artifacts so recompute is *possible later*.
- **A9 — Adversarial INPUT fixtures (U18 expanded).** The rail tests dangerous *inputs*, not just clean output cells: relative/absolute ambiguity, comparator swap, lower-is-better metric, 7B-claim-on-3B scope mismatch, duplicated seeds, CI-crossing-zero, decoy/test-only implementation, legacy-schema report, competing attempts — each must resolve to `inconclusive` or the correct quadrant, **never a false `contradicted`**. (Finding 11.)

**Deferred (named, not dropped):** full independent metric recompute (A8); execution-trace tooling depth (A6b) beyond a first cut; cross-paper claim corpus. Next iteration after the deterministic spine is green.

---

**Build strategy (locked):** **full fan-out** — all units (Part A reproducibility-verdict units U11–U18 below + the Part §1–§12 gate units) built in parallel via sub-agents, **integration gated on decision 8's golden-fixture quadrant rail** (now expanded per A9). Deterministic spine FIRST (verdict logic + eligibility shape + schema versioning + the adversarial rail), then the LLM/GPU-heavy depth (blinded re-extraction, execution-trace tests, live seed-CI, UI graph). `local`-first; runpod/docker exec paths byte-for-byte unchanged unless explicitly extended; every new behavior behind a default-OFF escape-hatch flag.

---

## 0. TL;DR / mission — the fidelity-certificate layer (a subset of Part A)

Make the reproduction agent **stop making frequent codegen errors (major and minor)** by catching buggy `train.py`/`train_cell.py`/env code **before it burns GPU**, via a layered **pre-training TDD gate** (tests on a tiny/synthetic config; nothing expensive runs until green). The gate routes each failure's exact cause into `repair_context` so the next iteration fixes it. The empirical #1 error is **the agent writing buggy reproduction code**; this turns "burn the whole grid → fail → 0.0" into "fail in seconds → repair → retry."

**Iterate contract (set up front, per `/iterate`):**
- **Goal:** seeded codegen bugs are caught at the gate (0 escape to GPU); live Adam + All-CNN reach *scored* metrics instead of `cell_execution_error`/`preflight_blocked` zeros.
- **Success check (deterministic):** a new guardrail-regression harness seeds the *real* observed bugs and asserts each is caught at the right tier with the right repair hint, before any GPU command. Target **100% caught / 0 escaped**. (Live re-run is the secondary, stochastic signal.)
- **Max iterations:** 5 (build loop). Stop at success OR cap; report honestly at cap.

---

## 1. Why — empirical grounding (the real failures)

Surveyed `runs/*/experiment_runs.jsonl`, `final_report.json`, `dashboard_events.jsonl`. **Agent-written-code bugs dominate**, and the two live validation runs hit exactly this class:

| Failure | Where (evidence) | What the agent did wrong | Caught today by |
|---|---|---|---|
| `preflight_blocked` (duplicate kwarg) | `runs/prj_6d41d2f09c026403/experiment_runs.jsonl` (Adam) | `train.py:184 make_optimizer(..., lr=<v>, **kwargs)` passes `lr` twice → `TypeError: multiple values for 'lr'` | pre_flight_validator (late) |
| `cell_execution_error` (17/17 cells) | `runs/prj_0a3202fc187bb692/experiment_runs.jsonl` (All-CNN) | non-OOM code bug in `train_cell.py` → every cell dies, 0 metrics → 0.0 | only AFTER burning all 17 cells |
| `missing_module: backend` | `runs/prj_6d41d2f09c026403/attempts/.../experiment_runs.jsonl` | `import backend` (or `from backend.agents...`) inside the FLAT sandbox where the repo isn't importable | import smoke (but it's OFF) |
| `cell_execution_error` (4 cells) | `runs/prj_09047604e591d969__20260531-235743/` (SDAR) | same class, smaller grid | after burn |
| `torch_redundancy`, `disk_exhausted`, `exec_timeout 8835s` | Adam attempts | minor/infra, mostly handled by env_pin + the reliability redesign | various |

**Observed success rate ≈ 30%** of runs; failures universally produce `0.0` rubric (`degraded_no_metrics`) because cell/preflight failures yield no measured metrics. **The lever:** catch the codegen bug on cell 1 / before the grid, in seconds, deterministically.

---

## 2. Grilling decisions (LOCKED)

1. **Q1 — Target surface:** *Agent codegen correctness* first (train.py/train_cell.py/env bugs), reasoning fixes folded in second.
2. **Q2 — Primary lever:** *Detect-early gate, default-ON* (cheap deterministic pre-exec gate; flip the two gated-OFF smokes ON; route exact error into repair_context). Prevent + repair layered after.
3. **Q3 — TDD gate design:** operator said **"we can do all"** → implement the **full stack**: harness-owned deterministic smoke **+** agent-authored paper-invariant tests (red→green) **+** mutation-testing-lite (catch tautological tests). Hard-gate the real grid until green, with a wall-clock-floor bypass.
4. **Codex review plan required** (operator: "have codex review plan too") — §7.
5. **Model split** (operator): Opus = planning/design (this doc); Sonnet = implementation/execution (fan-out); Codex = adversarial review.
6. **Generalize past training** (operator: "or whatever contents paper are") — the gate is **paper-type-agnostic** via the smoke contract: training → 1 optimizer step; eval-only → tiny eval on a few examples; analysis → smoke the pipeline. Harness asserts the *contract*, not "training happened."

---

## 3. The existing surface this builds on (DO NOT rebuild)

The pipeline already has a deep guardrail surface. **Crucial discovery: the gate is half-built and even self-describes as "preflight TDD."** Full inventory by stage:

**Pre-flight**
- `preflight_ast.py:scan_code_dir` — static AST: syntax, undefined names, missing-attr/method on local classes, bad local imports. **Always-on.** KEEP as Tier 0.
- `preflight_smoke.py` (`is_enabled`/`emit`/`smoke_command`, `MARKER`, env `REPROLAB_PREFLIGHT_SMOKE`) — **IMPORT smoke**: AST-collects third-party dep roots, imports each in the sandbox (GPU hidden), exits 3 on any miss; flags `import backend` with the exact "use a bare import of the copied helper" fix. Zero-false-positive (a missing import IS a bug). **DEFAULT OFF.** → Tier 1, **flip ON**.
- `execution_smoke.py` (`smoke_command`/`interpret_exit`, `MARKER`, env `REPROLAB_EXECUTION_SMOKE`, `REPROLAB_SMOKE_STEPS`) — **1-step EXECUTION smoke**: runs the entry script for N=1 steps on tiny data with `CUDA_LAUNCH_BLOCKING=1` under `timeout`. Exit 0→ok, 124→`not_honored` (SOFT pass, skip), other→`crash` (BLOCKING). Catches device-side-assert, the duplicate-kwarg `TypeError`, the All-CNN cell bug. **DEFAULT OFF.** → Tier 2, **flip ON + harden** (see §4).
- `pre_flight_validator.py:validate_code_pre_flight` — surrogate/tiny-model + dataset-subsample + missing-variant detector. Hard violations block. Always-on.
- `env_pin.py:harden_requirements` — torch cu121 coherence (the libcupti fix). Always-on (local).
- `cell_matrix.py:capacity_gate` / `dataset_url_preflight` — VRAM + dead-dataset preflight. Always-on, fail-soft.

**During / post / finalize** (KEEP; the gate complements these): `local_process.exec` streaming+stall (just shipped), `run_watchdog.py`, `safe_builtins_patch`, `rubric_guard.py:assert_metrics_schema` (agent self-validates metrics shape end-of-train), `rubric_contract.py:validate_contract`, `failure_classifier.py`, `scope_classifier.py`, `report.py:_reconcile_verdict_against_evidence`, `forced_iteration.py` (wall-clock-floor bypass pattern to MIRROR), `sse_bridge.redact_corpus`.

**Wiring already present** (`backend/agents/rlm/primitives.py`):
- `_execute_in_sandbox` builds `bootstrap_commands`; appends import-smoke (≈3211–3221) then execution-smoke (≈3225–3242), each `is_enabled()`-gated.
- Short-circuit on a blocking smoke failure: ≈3447–3461 (reads `preflight_smoke_result.json`, skips remaining training commands).
- Cell path: `run_matrix` dispatch ≈4571–4584 (k8s + gpu_cell_runner), gated on `cells.json`+`train_cell.py` present ≈4733/4886/4918.

**Net-new (not present today):** agent-authored `test_reproduction.py` convention; mutation check; cell-aware execution smoke (run ONE cell before the grid); numeric-sanity (NaN/inf) leaf assertion; the regression harness; default-ON flip + prevention prompt/template changes.

---

## 4. The design — layered pre-training TDD gate

Insertion point: inside `run_experiment` (primitives.py), the gate runs **before** the expensive path — i.e. before `run_matrix` (cell path) and before the full training command (monolithic path). Each tier is **fail-LOUD on a real code bug** (block + repair_context) and **fail-SOFT on gate-infra trouble** (never block a legit run on our own flakiness). Wall-clock-floor bypass mirrors `forced_iteration._WALL_CLOCK_FLOOR_S` (≤60 s remaining → skip the gate, let work ship).

### Tier 0 — static AST (always-on, unchanged)
`preflight_ast.scan_code_dir`. Syntax / undefined name / missing-attr. Free, deterministic.

### Tier 1 — import smoke (flip DEFAULT-ON)
`preflight_smoke`. Every declared dep resolves in the flat sandbox; catches `missing_module` incl. `import backend` with the precise bare-import fix. Change: `is_enabled()` defaults TRUE (escape hatch `REPROLAB_PREFLIGHT_SMOKE=0`).

### Tier 2 — execution smoke (flip DEFAULT-ON + harden) — *highest ROI*
`execution_smoke`, extended:
- **Cell-aware:** when `cells.json`+`train_cell.py` exist, run `train_cell.py` on the **smallest cell** (capacity-gate already ranks by est_vram) with `REPROLAB_SMOKE_STEPS=1` + a harness-pinned tiny `--output-dir`, **before** `run_matrix`. This catches the All-CNN 17-cell bug on **cell 1**. Monolithic path keeps the existing entry-script smoke.
- **Close the soft-pass loophole:** today exit 124 (`not_honored`) is a soft pass — a script that ignores `REPROLAB_SMOKE_STEPS` is silently skipped, so the gate is bypassable. Harden: the harness **controls the tiny input** (inject a synthetic/sliced dataset + a hard step cap it owns), so honoring is not purely on trust. If still not honored, downgrade to soft pass **but emit a `run_warning`** so it's observable (don't fail silently).
- **Metrics-shape + numeric sanity assertion:** after the 1-step smoke, assert the contract-shaped `metrics.json` was written and leaves are finite (no `NaN`/`inf`/placeholder). Reuses `rubric_guard.assert_metrics_schema` logic. Catches `incomplete_metrics`/`scope_shape_violation` *before* the grid.
- **Static duplicate-kwarg lint** (cheap, deterministic): a small AST check for `f(x=…, **kw)` where `kw` provably contains `x` (the Adam bug). Belongs in `preflight_ast` (Tier 0) if statically decidable; otherwise Tier 2's 1-step run catches it at runtime anyway.

### Tier 3 — agent-authored invariant tests (NEW; red→green)
The agent writes `code/test_reproduction.py` asserting **paper-specific invariants** (shapes, loss components, the algorithm's defining equation — e.g. SDAR `g_t=σ(β·Δ_t)`, stop-grad on the gate, λ=0.1, β=10). Seeded from `paper_invariants.py` + the paper's `PAPER_HINTS` entry where present; otherwise the agent derives them. Harness runs them on the tiny config (CPU/1-GPU, seconds); **must be green before the grid**. Failure → `repair_context` names the failing assertion.
- Scaffolding: system_prompt + `baseline_implementation.py` template instruct the agent to (a) write `test_reproduction.py` first, (b) honor `REPROLAB_SMOKE_STEPS`, (c) use **bare** imports of copied helpers (never `from backend…`), (d) never pass a kwarg both explicitly and via `**kw`.

### Tier 4 — mutation check (NEW; catches tautological tests) — *iteration 2*
For each registered invariant constant (from `paper_invariants.py`), perturb it (e.g. β=10→β=11) and **confirm the agent's `test_reproduction.py` FAILS**. A test that still passes under mutation is tautological → reject + `repair_context`: "your invariant test does not actually exercise <constant>." Bounded to registered constants (deterministic, cheap).

### Prevention + repair (layered)
- **Prevention:** the §Tier-3 scaffolding changes (system_prompt.py, baseline_implementation.py) + Adam/All-CNN `PAPER_HINTS` invariants.
- **Repair:** add failure class(es) `tdd_gate_failed` (or reuse `preflight_blocked`) + `invariant_test_failed` to `failure_classifier.py` with tier-specific `suggested_fix`; the `MARKER` short-circuit already threads the failing artifact into the next iteration.

---

## 5. Phased implementation plan (fan-out map)

Each **Unit** is independent + Sonnet-sized (parallelizable via Agent tool or a Workflow). Opus owns sequencing + final synthesis; Codex reviews (§7).

**Iteration 1 — detect-early gate default-ON (the win):**
- **U1** Flip Tier 1 import smoke default-ON (`preflight_smoke.is_enabled`) + escape hatch + tests.
- **U2** Flip Tier 2 execution smoke default-ON + **cell-aware smoke** (run smallest cell before `run_matrix`) + close soft-pass loophole (harness-owned tiny input + `run_warning`) + tests. *(Highest ROI — owns the All-CNN/Adam fix.)*
- **U3** Tier 2 metrics-shape + numeric-sanity assertion (reuse rubric_guard) + duplicate-kwarg static lint in preflight_ast + tests.
- **U4** Failure-classifier classes (`tdd_gate_failed`/`invariant_test_failed`) + repair_context plumbing + tests.
- **U5** **Regression harness** (the success check, §6) — seed the real bugs, assert caught/0-escape.

**Iteration 2 — prevention + invariant TDD:**
- **U6** Tier 3 agent-authored `test_reproduction.py` convention: system_prompt.py + baseline_implementation.py scaffolding + harness runner + tests.
- **U7** Adam (`1412.6980`) + All-CNN (`1412.6806`) `PAPER_HINTS` invariants.
- **U8** Tier 4 mutation check against `paper_invariants.py` + tests.

**Iteration 3 — validate + decide defaults:**
- **U9** Live re-run Adam + All-CNN with the gate ON; confirm scored (not zeroed). A/B vs gate-OFF if needed.
- **U10** Decide which flips stay default-ON (CLAUDE.md: A/B ≥3 paired runs before flipping a default — the gate is a strict improvement, but honor the discipline / keep escape hatches).

**Part A — two-axis reproducibility verdict (full fan-out, golden-fixture-gated; supersedes the strict Iteration 1/2/3 ordering above):**
- **U11** Two-axis verdict in `report.py`: add `implementation_verdict` + `replication_verdict` derived from the existing fidelity / result-match rubric areas; keep `verdict`/`overall_score` **additive** (back-compat). Gating logic: certificate-green AND ≥2-seeds → {replicated|contradicted}, else `inconclusive` (Part A decisions 1, 3, 7).
- **U12** Graded replication credit in the leaf scorer / D5 path (`leaf_scorer.py:906` (`_SYSTEM_PROMPT` RESULT-MATCH block)): magnitude-aware `1.0 / ~0.75 / contradicted`, replacing direction-only full credit (decision 2).
- **U13** Role-separated **Extractor** → frozen `ReproSpec` (claims + invariants): full claim mining, ranked top-K, each evidence-gated to a paper span; spec frozen *before* implement (anti-circular, decision 5).
- **U14** **Claimed-magnitude extractor** — its own unit + own fixtures: pull claimed effect sizes from tables/abstract with provenance (the input to U12; the fragile surface).
- **U15** **≥2-seed contradiction gate** — user-configurable seed count wired into the matrix + the U11 gating logic (decision 3).
- **U16** **Executable invariant certificate** — promote Tier 3 `test_reproduction.py` to the *spine*; Tier 4 mutation confirms bite; thin LLM refute last (decision 4). Reframes/absorbs U6/U8 — they are no longer "Iteration 2 polish," they are the certificate.
- **U17** **UI verdict card** + measured-vs-claimed graph in `/lab`; `sse_bridge` **audience tag** (user vs developer); collapsed developer panel (decision 6).
- **U18** **Golden-fixture quadrant rail** — the deterministic merge gate: assert every (fidelity × replication) cell (decision 8). Extends the §6 regression harness; this is the **stop condition for the whole fan-out.**

**File-overlap (extend §8 map):** U11/U12→`report.py`+`leaf_scorer.py` (serialize); U13/U14→new `extractor`/`repro_spec` modules + `rubric_gen.py`; U16→`execution_smoke.py`/`preflight_ast.py`+system_prompt (serialize w/ U2/U3); U17→frontend `/lab` + `sse_bridge.py`; U18→tests only.

**Each unit:** read the cited file:symbol, make the smallest change, add/adjust tests, run the gate suite (`tests/services/runtime` + `tests/rlm`), 0-regress. One change per iteration so a pass/fail flip is attributable.

---

## 6. Success metric / stop condition — the deterministic rail

Per the **scoring-fairness lesson** (`[[scoring-fairness-spec]]`: "clean A/B = the deterministic rail; end-to-end re-runs conflate change + stochasticity"), the PRIMARY success check is a fast deterministic harness, NOT a live run:

`tests/rlm/test_codegen_tdd_gate.py` (new) seeds the **real** observed bad outputs and asserts each is caught at the right tier with the right repair hint, **before any GPU command**:
1. `train.py` with `make_optimizer(..., lr=x, **kw)` (kw has lr) → Tier 0 lint or Tier 2 → `repair_context` names the duplicate kwarg. *(Adam)*
2. `train_cell.py` raising AttributeError/TypeError on construction → Tier 2 **cell-smoke** catches on cell 1, grid never dispatched. *(All-CNN)*
3. `import backend` / `from backend.agents…` → Tier 1 → bare-import fix. *(missing_module)*
4. metrics.json with `NaN`/placeholder/empty `per_model` → Tier 2 numeric-sanity. *(incomplete_metrics)*
5. tautological `test_reproduction.py` → Tier 4 mutation check. *(iteration 2)*

**Target: 100% caught at the gate, 0 escaped to GPU.** Secondary (stochastic): Adam/All-CNN re-runs reach scored metrics.

**Part A quadrant rail (U18) — the verdict stop condition.** A second fixture set, `tests/rlm/test_reproducibility_verdict.py` (new), asserts every (fidelity × replication) cell lands correctly — *deterministically, no GPU*:
6. faithful build + green invariant certificate + metrics that **match** the claim (≥2 seeds) → `(faithful, replicated)`.
7. faithful build + green certificate + metrics that **invert** the claim (≥2 seeds) → **`(faithful, contradicted)` — NOT `failed`** *(the litmus test; the whole point)*.
8. faithful build + green certificate + right-direction-magnitude-far-off → `(faithful, partially-replicated, ~0.75)` (decision 2).
9. broken build (cells died / certificate red) **or** <2 seeds, regardless of metrics → `(broken|partial, inconclusive)` — never `contradicted` (decisions 1, 3).
10. leaderboard sort: a case-7 run ranks **above** a `(broken, inconclusive)` run (fidelity-ranked, decision 7).

**Verdict target: every quadrant cell correct, and case 7 never collapses to `failed`.** If it does, the design regressed to the single-score assumption.

---

## 7. Codex review plan (required)

Use `codex:codex-rescue` (the Codex shared-runtime agent) as the adversarial reviewer at two checkpoints:
1. **Design review (before build):** hand Codex THIS doc + the four key files (`preflight_smoke.py`, `execution_smoke.py`, `primitives.py` run_experiment region, `preflight_ast.py`). Ask: "Where will this gate produce false positives that block legit runs? Where is it bypassable? What codegen bug class does it miss?"
2. **Diff review (before commit/merge):** hand Codex the actual diff + the regression harness. Ask it to find an agent-output that escapes the gate.

**⚠ Lesson (`[[scoring-fairness-spec]]`):** a prior Codex review with ~1 tool_use **hallucinated** its file:line blockers. Mitigations: (a) give Codex real tool budget and the actual files, (b) **verify every Codex claim directly** against the code before acting, (c) treat its output as leads, not findings.

---

## 8. Model split + fan-out mechanics

- **Opus (planning):** this design, per-unit specs, sequencing, final synthesis, adjudicating Codex.
- **Sonnet (execution):** each Unit U1–U10 as an independent sub-agent (Agent tool, parallel where no file overlap; serialize units that touch the same file — e.g. U2/U3 both touch execution_smoke/preflight_ast). For a structured run, a `Workflow` pipeline (one stage per unit, adversarial-verify stage) is appropriate — but only if the operator explicitly opts into multi-agent orchestration ("use a workflow"/ultracode); otherwise plain parallel Agent calls.
- **Codex (review):** §7.
- **File-overlap map** (avoid parallel write conflicts): U2+U3→execution_smoke.py/preflight_ast.py (serialize); U1→preflight_smoke.py; U4→failure_classifier.py; U6→system_prompt.py+baseline_implementation.py; U5+regression→tests only.

---

## 9. Constraints + gotchas (from memory / CLAUDE.md — DO NOT violate)

- **Commit as `lolout1`, NO `Co-Authored-By: Claude` trailer** (`[[commit-attribution-preference]]`). Commit/push **only when explicitly asked.**
- `.venv/bin/python` (no bare `python` on PATH). Run tests: `.venv/bin/python -m pytest tests/ -n auto`.
- **Shell env shadows `.env`** → prefix runs with `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY` (BUG-LR-014).
- Root model `--model claude-oauth` ($0 both surfaces); sub-agents = `claude-sonnet-4-6`. (Optimal/validated root = GPT-5 ~$1/run; not used here.)
- Long reproduction runs `--max-wall-clock 50400` (14h); via `batch_reproduce.py` it's `--extra "--max-wall-clock 50400"` (it is NOT a batch flag — doc bug noted in the reliability handoff §7).
- **Default-flip discipline:** CLAUDE.md requires A/B ≥3 paired runs before flipping a default. The gate is a strict improvement, but keep every flip behind an escape-hatch env var and validate on the live re-runs.
- **Best practices June 2026:** test-time verification / tests-as-gate, structured-output validators, LLM-as-judge gates, mutation testing to defeat tautological tests, plan-then-execute with an Opus planner. (Web-research these per-tier if depth is wanted; scope to what each tier needs.)
- Keep Adam/All-CNN evidence run dirs (`runs/prj_6d41d2f09c026403`, `runs/prj_0a3202fc187bb692`).

---

## 10. Live run status (validating the *reliability* redesign, in parallel)

The execution-reliability redesign (`662e018`) is being validated live — independent of this codegen track, but the same Adam/All-CNN runs will exercise BOTH once the gate lands:
- **Adam** `1412.6980` → `prj_6d41d2f09c026403`, GPUs [1,3].
- **All-CNN** `1412.6806` → `prj_0a3202fc187bb692`, GPUs [4,5].
- Root `claude-oauth`, sub-agents `claude-sonnet-4-6`. Launched via the batch scheduler.
- **Watch for:** `.exec_live.log` streaming (reliability redesign working), and whether they hit the SAME codegen bugs again (which is the motivation for THIS work). If they re-hit `preflight_blocked`/`cell_execution_error`, that's the baseline the gate must eliminate.

---

## 11. Open decisions / deferred

- **Default-ON vs A/B-gated** for the flips (recommend: ship default-ON + escape hatch; honor A/B before declaring it permanent). — operator to confirm.
- **`test_reproduction.py` authorship** when no `PAPER_HINTS` entry exists: agent-derived invariants are weaker; mutation check (Tier 4) is the backstop.
- **Cell-smoke cost:** running 1 cell before the grid adds seconds–minutes; for a huge grid it's pure win, for a 1-cell grid it's redundant (skip when cells==1).
- **Backend scope:** gate logic is backend-agnostic (pure/deterministic); recommend ON for local first, flag controls all backends. runpod/docker exec paths must stay byte-for-byte unless explicitly extended.
- **Codex design review** not yet run (§7 checkpoint 1) — re-offer before build.

---

## 12. Quick-start for a fresh session

```bash
cd /home/sww35/openresearch
# 1. Read this doc + the four key files
#    backend/agents/rlm/{preflight_smoke,execution_smoke,preflight_ast,primitives}.py
# 2. Iteration 1 = U1..U5 (§5). Start with U2 (cell-aware execution smoke) — highest ROI.
# 3. Success check (build it first if not present):
.venv/bin/python -m pytest tests/rlm/test_codegen_tdd_gate.py -x   # the regression rail (§6)
.venv/bin/python -m pytest tests/services/runtime tests/rlm -n auto # 0-regress gate
# 4. Codex design review BEFORE coding (§7), verify its claims directly.
# 5. Commit as lolout1, no trailer, ONLY when the operator asks.
```

**One-line state:** design locked, nothing built; the gate is mostly *flip-ON + extend* of the existing "preflight TDD" smokes; start at U2; the deterministic regression harness (§6) is the stop condition.
