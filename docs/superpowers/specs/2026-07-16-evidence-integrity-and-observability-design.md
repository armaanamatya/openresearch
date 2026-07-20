# Evidence-Integrity & Observability — Design Spec

- **Date:** 2026-07-16
- **Status:** Partially implemented on branch `feat/evidence-integrity-w1` (W1-M1, W2, eval-coverage
  floor + observability shipped, default-OFF/tested; W1-M2 access-audit + W3/W5 not yet built).
  Live implementation status + verified pre-existing-mechanism findings:
  `docs/superpowers/plans/2026-07-16-evidence-integrity-loop-progress.md`. Flag catalog:
  `backend/agents/rlm/CLAUDE.md` → "Evidence-integrity + observability". Every flag in `docs/reference/flags.md`.
- **Author:** Aayush Baniya (via Claude)
- **Scope:** 5 workstreams (W1–W5). Single mega-spec by request; each workstream ships independently.
- **Related:** `backend/agents/rlm/CLAUDE.md` (evidence gate, fabrication guards), `learn.md`,
  Phase-1/2 hyperanalysis (this session), external sources cited inline.

> **Verification status:** File/symbol references below are grounded in a Phase-1 read of the
> codebase; those marked `⟨verify⟩` were inferred and MUST be confirmed against source before
> implementation (Phase B of the loop). Line numbers are deliberately avoided (they drift); we
> anchor on module + symbol names.

---

## 1. Motivation

The Phase-2 landscape scan (2025–2026) produced three load-bearing, primary-source-verified findings:

1. **The field is converging on our invariant.** RE-Bench (arXiv 2411.15114) and MLE-bench
   (arXiv 2410.07095) — the two most credible ML-research benchmarks — grade on **deterministic
   reference solutions / real leaderboards, never an LLM judge**. GroundEval (arXiv 2606.22737)
   quantified the failure we already assume: two LLM judges scored state-invalid code 0.85–0.90
   where a deterministic verifier scored it **0.000**.
2. **The threat model has widened from *faked metrics* to *evaluator tampering + data leakage*.**
   RewardHackingAgents (arXiv 2603.11337) detects reward-hacking via **evaluator-hash integrity +
   file-access logs + reported-vs-true metric disagreement**. Our current fabrication guards catch
   zero/stub metrics but NOT tampering of the grader itself or train/test leakage.
3. **Cost/observability is a solved commodity everywhere but here.** Our `cost_ledger.jsonl` is
   blind to Foundry-routed LLM spend (logs `$0`) and idle-GPU time by construction.

This spec turns those findings into shippable, evidence-first mechanisms that **deepen** the
north-star invariant rather than dilute it.

### North-star invariant (the red line — unchanged)
> Verdicts, trust gates, and self-improvement key on the **deterministic on-disk evidence layer,
> never a scalar LLM grade.** Every mechanism in this spec keys on measured artifacts; none routes
> a verdict through an LLM's judgment.

---

## 2. Global constraints (apply to every workstream)

- **G1 — Default-OFF flag-gated.** Each mechanism sits behind an `OPENRESEARCH_*` flag using the
  canonical idiom `os.environ.get("FLAG","").strip().lower() in ("1","true","yes")`, default-OFF,
  **byte-identical behavior when off**. New flags added to `docs/reference/flags.md` via
  `python scripts/gen_flag_registry.py` (freshness test guards it).
- **G2 — Evidence, not grade.** No new mechanism may credit or veto a verdict using an LLM score.
  All new gates evaluate typed predicates over on-disk artifacts. (W3's SimpleJudge baseline is the
  sole LLM-grade path and exists ONLY as a comparison baseline, never wired into a live verdict.)
- **G3 — Fail-closed.** New gates degrade a suspicious result to *repairable* / *failed*, never
  silently upgrade. Ambiguity resolves against crediting.
- **G4 — Hermetic tests.** Each mechanism ships with (a) a **positive** pytest that builds a fake
  run dir with tampered/leaked/incoherent artifacts and asserts the gate fires, and (b) an
  **A/B flag-OFF** pytest asserting byte-identical output vs baseline. Socket-hermetic (pytest-socket).
- **G5 — No new hard cloud dependency.** External observability (OTel/Datadog) is a *pluggable sink*;
  the file-backed ledger remains source of truth. The harness must run fully offline.

---

## 3. Unifying architecture — the Evidence Integrity Layer (EIL)

Today two primitives carry the evidence layer:
- `backend/agents/rlm/evidence_bundle.py` — mints the immutable receipt
  `{attempt, ledger_sequence, metrics_sha256, code_tree_digest, artifact_dir, coordinates}`
  (behind `OPENRESEARCH_CANONICAL_EVIDENCE_BUNDLE`).
- `backend/agents/rlm/evidence_gate.py` — `gate_decision(score, claims_result, has_disk_evidence)`
  vetoes a credited result lacking on-disk evidence.

The EIL extends **the same receipt + the same gate** along two new axes, so W1/W2 are not bolt-ons:

```
Evidence receipt (evidence_bundle)         Gate (evidence_gate + leaf_scorer)
├─ metrics_sha256        (existing)        ├─ has_disk_evidence veto          (existing)
├─ code_tree_digest      (existing)        ├─ zero/stub-metrics guards        (existing)
├─ grader_digest         (W1 — new)   ──►  ├─ integrity gate (tamper/leak)    (W1 — new)
├─ testset_digest        (W1 — new)   ──►  ├─ state-contract gate (coherence) (W2 — new)
└─ access_log_digest     (W1 — new)        └─ (min-aggregation veto, existing pattern)
```

- **W1 (Integrity)** answers: *was the grader/rubric/test-split un-tampered, and did the code avoid
  the held-out data?*
- **W2 (Coherence)** answers: *does the execution trace actually support the credited metric?*
- **W3** consumes the EIL to produce the LLM-judge-vs-evidence-gate scorecard.
- **W4** is orthogonal (cost) — shares nothing but the flag discipline.
- **W5** provides the *strong-form* backing for W1/W2: an out-of-process, read-only-mounted grader
  that makes tampering physically impossible (fork-option B), plus microVM isolation for paper code.

**Fork decision (recorded): C — detect-now / prevent-later.** W1 ships in-process detection
(fingerprint + access audit + metric cross-check); W5 later upgrades to out-of-process prevention.
Rationale: W1 is cheap and catches most tampering after the fact; W5's isolation is structural but
sandbox-sized. Layered defense; W1 need not be redone when W5 lands.

---

## 4. W1 — Grader-tampering + leakage detection  *(detailed)*

**Flag:** `OPENRESEARCH_GRADER_INTEGRITY` (default OFF).
**Files:** `evidence_bundle.py` (extend receipt), new `grader_integrity.py`, `evidence_gate.py`
(new gate outcomes), `binding.py` ⟨verify⟩ (access-audit hook around `run_experiment`),
`leaf_scorer.py` (consume gate).

### 4.1 Threat model
An agent that writes Python over the 19 primitives can, in principle:
1. **Tamper the grader/rubric** — mutate `leaf_scorer` inputs, `rubric_tree.json`, or scorer config
   so weak work grades high.
2. **Leak the held-out split** — read test/eval data during training, or copy labels into training
   inputs, inflating measured metrics honestly-computed-on-contaminated-data.
3. **Fake `metrics.json`** — write a metrics file whose numbers don't match the raw predictions/logs
   on disk (partially covered today by zero/stub guards; not covered for *plausible-but-wrong* numbers).

### 4.2 Mechanisms (all deterministic, on-disk)

**M1 — Grader/rubric fingerprint.** Extend the evidence receipt with:
- `grader_digest = sha256(scorer_source_bytes ‖ rubric_tree.json ‖ scorer_config)` computed at the
  moment of grading. `scorer_source_bytes` = the bytes of the scoring module(s) actually imported.
- `testset_digest = sha256(sorted(test_split_file_bytes))`.
The receipt is minted once and compared at the single write chokepoint. A mismatch between the
grader that *should* run (canonical repo bytes) and what *did* run → `evidence_tampered`.
> Note: because the grader runs in-process today, M1 is *tamper-evident*, not *tamper-proof*.
> W5 makes it tamper-proof. This is the explicit fork-C tradeoff.

**M2 — File-access audit around `run_experiment`.** Record a deterministic log of paths
opened/read/written by training+eval code, written to `runs/<id>/rlm_state/access_log.jsonl`:
- **local/docker:** an `open`/`os.open` audit hook installed in the sandbox entrypoint, OR a
  pre/post filesystem `mtime`+`atime` snapshot diff over the run dir (fallback when hooks unavailable).
- **GKE/cell-matrix:** emit the access log from the cell entrypoint
  (`docker/gke-cell-base/gke_cell_entrypoint.py` ⟨verify⟩), shipped back with per-cell results.
- `access_log_digest` added to the receipt.

**M3 — Leakage check** (consumes M2): assert no path under the held-out test-split directory was
read by a training-phase process, and (cheap) that test-example IDs/hashes do not appear in the
training input manifest. Violation → `leakage_suspected`.

**M4 — Reported-vs-true metric cross-check.** Where raw model outputs exist on disk (predictions,
eval logs), **independently re-derive** the headline metric with a deterministic recomputation and
compare to `metrics.json`. Disagreement beyond `OPENRESEARCH_METRIC_RECHECK_TOL` (default 1e-3
relative) → `metric_mismatch`. When raw outputs are absent, M4 is a no-op (does not fabricate a
comparison; absence is handled by the existing degrade path).

### 4.3 Gate wiring
`grader_integrity.evaluate(run_dir, receipt) -> IntegrityVerdict` returns a min-aggregated result
over M1/M3/M4 (M2 is data, not a verdict). Outcomes feed `evidence_gate` as new fail-closed classes:
`evidence_tampered`, `leakage_suspected`, `metric_mismatch`. Each maps into the existing
`leaf_triage` repair taxonomy (`provenance_gap` for M1, `protocol_gap` for M3, `result_quality`
for M4) so the repair loop and `leaf_repair_plan` are reused unchanged. When the flag is OFF,
`evaluate` is never called and the receipt omits the new fields → byte-identical.

### 4.4 Testing
- Positive: fabricated run dir with (a) mutated `rubric_tree.json`, (b) an access-log entry showing
  a training process read the test split, (c) a `metrics.json` inconsistent with raw predictions —
  assert each fires its class.
- A/B: flag OFF ⇒ receipt + verdict byte-identical to a golden baseline.

---

## 5. W2 — GroundEval state-contracts *(coherence)*

**Flag:** `OPENRESEARCH_STATE_CONTRACTS` (default OFF).
**Files:** new `backend/agents/rlm/state_contracts.py`, `leaf_scorer.py` (consume), `evidence_gate.py`.

**Idea (from GroundEval, arXiv 2606.22737):** artifact *existence* (today's `_gather_evidence`
⟨verify⟩) is weaker than trace *coherence*. A result-claiming leaf must satisfy a set of typed
predicates over the execution trace to be credited.

**Contract predicates** (deterministic, evaluated over `experiment_runs.jsonl` + cell metrics +
W1's `access_log.jsonl` + artifact timestamps):
- `checkpoint_exists` — a model checkpoint artifact is present.
- `checkpoint_after_train_start` — checkpoint mtime > train-start marker.
- `eval_loaded_checkpoint` — eval process read the checkpoint path (from access log).
- `eval_covered_full_testset` — eval row-count == expected test-split size (± tolerance).
- `metric_provenance_present` — the credited metric traces to a specific eval-run artifact.

Contracts are declared per leaf-type (classification/generation/RL/etc.) in a small registry;
unknown leaf-types get a permissive default (no new veto) to preserve G1 when data is thin.
Aggregation is **min-veto** (any failed hard predicate caps the leaf), mirroring `external_validator`.
Failures classify into existing `leaf_repair_plan` buckets. Flag OFF ⇒ contracts never evaluated.

**Testing:** positive fixture where checkpoint mtime precedes train-start (incoherent) and eval
row-count < test size ⇒ leaf vetoed; A/B flag-OFF byte-identical.

---

## 6. W3 — PaperBench head-to-head scorecard *(proof, not a runtime gate)*

**Surface:** new CLI subcommand `python -m backend.cli paperbench-scorecard` (operator-invoked;
expensive). **No change to any live verdict path.**
**Files:** `backend/evals/paperbench/` (harness + `scorecard.py`), `scripts/gen_paperbench_scorecard.py`
(mirrors existing `scripts/gen_leaderboard_readme.py`).

**What it does:** runs the reproduction pipeline over PaperBench's 20 ICML-2024 papers (arXiv
2504.01848) and, **on the same run artifacts**, emits two verdicts per paper:
1. **SimpleJudge baseline** — a pure-LLM rubric grade (mimics PaperBench's grader), isolated in a
   `grade_mode="llm_only"` path. This is the ONLY LLM-grade path in the spec; it never touches a
   live verdict (G2 preserved).
2. **Evidence-gated verdict** — our normal EIL-backed result.

Output: a disagreement table (paper × {llm_pass, evidence_verdict, divergence_reason}) + JSON/md
report — the public demonstration that our gate catches what an LLM judge credits.

**Constraints:** PaperBench-set download + GPU + LLM cost are all operator-gated; the subcommand
refuses to run without explicit money-cap flags (mirrors `campaign`'s required money meters).

**Testing:** the scorecard *assembler* is unit-tested with synthetic per-paper verdicts (no network,
no GPU); the end-to-end run is manual/operator-only and excluded from the hermetic suite.

---

## 7. W4 — Cost observability

**Flag:** `OPENRESEARCH_COST_OBSERVABILITY` (default OFF for new accounting fields).
**Files:** `backend/agents/resilience/cost.py` ⟨verify⟩ (`CostLedgerEntry`), `pricing.py` ⟨verify⟩,
`demo_status.json` writer (`live_runs.py` ⟨verify⟩), optional `cost_otel.py`.

**F1 — Close the Foundry $0 blind spot.** Add pricing entries for `opus-foundry`, `sonnet-foundry`,
`grok`, `azure-foundry` to `pricing.py`, and make the Foundry client path populate
`input_tokens`/`output_tokens` on `CostLedgerEntry` (today logs 0/0). When Foundry does not return
token counts, estimate from request/response sizes and stamp `cost_confidence="estimated"` with
provenance (never silently report a real dollar figure as measured).

**F2 — Idle-GPU accounting.** Track node/pod `provision → teardown` wall-clock separately from
active exec; add `idle_gpu_usd = max(0, node_uptime_s − active_exec_s) × sku_hourly/3600`. Sources:
RunPod pod lifecycle timestamps; GKE node uptime (kubectl / node-controller events). Recorded as
distinct ledger rows tagged `kind="idle_gpu"` so cost math stays auditable.

**F3 — Surface + optional OTel sink.** Extend `demo_status.json.cost_summary` with `foundry_usd`,
`idle_gpu_usd`, `cost_confidence`. Add an optional OpenTelemetry exporter (`cost_otel.py`, behind
the flag) that emits token/GPU spend as OTel metrics — pluggable sink only; file ledger stays SoT (G5).

**Testing:** synthetic ledger with Foundry rows (0/0 today) ⇒ assert non-zero estimated USD +
`cost_confidence="estimated"`; idle-GPU derivation from fabricated lifecycle timestamps; A/B flag-OFF
byte-identical `cost_summary`.

---

## 8. W5 — Sandbox-ingress hardening + out-of-process grader

**Flag:** `OPENRESEARCH_HARDENED_SANDBOX` (selects a hardened backend; default OFF → matrix unchanged).
**Files:** `backend/services/runtime/` (new backend class + `SandboxMode` entry), grader-isolation shim.

**H1 — Hardened execution backend.** Add a sandbox backend that runs untrusted paper code under
stronger isolation for the local/docker path. Candidates (decide in plan/Phase B):
Anthropic `sandbox-runtime` (OS-primitive; explicitly "not a complete security boundary"), gVisor,
or firecracker-microVM. Integrated as a new entry in the sandbox matrix; **default selection
unchanged** (G1).

**H2 — Out-of-process grader (fork-option B, the prevent half).** Run `leaf_scorer` in a separate
process with a **read-only** mount of the artifact dir, so agent-written code physically cannot
mutate the grader, rubric, or test split. This turns W1's *tamper-evident* fingerprint into
*tamper-proof*. Wired so that when `OPENRESEARCH_HARDENED_SANDBOX` is on, grading routes through the
isolated process; when off, in-process grading (with W1 detection) is used.

**Testing:** grader-isolation smoke (a fixture that attempts to write the rubric from within the
sandboxed process fails on the read-only mount); backend-selection A/B (flag OFF ⇒ same backend chosen).

---

## 9. Sequencing & dependencies

```
W1 ──► W2        (W2 reuses W1's access_log + receipt fields)
        │
W1,W2 ─┴─► W3    (scorecard is only meaningful once the gate is real)
W4  (independent, can run in parallel with W1/W2)
W5  (independent; consumes W1 to deliver the prevent-half H2)
```

Recommended build order: **W1 → W2 → W4 → W3 → W5** (risk-reduction first; W3 is the payoff demo;
W5 is the heaviest and backstops W1/W2).

---

## 10. Cross-cutting changes

- **Evidence receipt schema** gains `grader_digest`, `testset_digest`, `access_log_digest`
  (all optional; absent when flags off).
- **New flags** (all default-OFF): `OPENRESEARCH_GRADER_INTEGRITY`, `OPENRESEARCH_METRIC_RECHECK_TOL`,
  `OPENRESEARCH_STATE_CONTRACTS`, `OPENRESEARCH_COST_OBSERVABILITY`, `OPENRESEARCH_HARDENED_SANDBOX`.
  Register via `scripts/gen_flag_registry.py`; freshness test must pass.
- **New gate outcome classes:** `evidence_tampered`, `leakage_suspected`, `metric_mismatch`,
  `state_contract_failed` — all mapped into the existing `leaf_triage` taxonomy.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| New gate false-positives kill valid runs | Default-OFF; conservative predicates; only result-claiming leaves eligible; min-veto tuned on real runs before any default-flip (needs ≥3 paired A/B + grader-σ gate per repo policy). |
| In-process grader still tamperable (fork C) | Explicitly acknowledged; W5/H2 closes it structurally; W1 is detection layer. |
| Access-audit hook unavailable on some sandboxes | mtime/atime snapshot-diff fallback; GKE emits from cell entrypoint. |
| Foundry token estimation inaccurate | Stamp `cost_confidence="estimated"`, never present as measured; reconcile against cloud billing. |
| Metric re-derivation can't reproduce agent's metric | M4 no-ops when raw outputs absent; never fabricates a comparison. |
| Spec cites stale symbols | Phase-B verification pass confirms every `⟨verify⟩` before implementation. |

---

## 12. Out of scope (YAGNI)

- Rewriting the grader as a formal proof system (state-contracts are typed predicates, not a solver).
- A general-purpose FinOps dashboard (W4 surfaces fields + an OTel sink; UI is separate).
- Replacing the sandbox matrix (W5 *adds* a hardened option; it does not remove existing backends).
- Any change that flips a flag default (all mechanisms ship OFF; default-flips are a later, gated decision).

---

## 13. Open decisions (to resolve in Phase B / plan)

1. Access-audit implementation: `open`-hook vs mtime/atime snapshot vs LD_PRELOAD/strace-lite — pick per sandbox.
2. W5 isolation tech: sandbox-runtime vs gVisor vs firecracker — evaluate against setup cost + macOS-dev parity.
3. Whether M4's metric re-derivation is generic or per-leaf-type (start per-type; generalize later).

---

## 14. POST-VERIFICATION REVISIONS (2026-07-16, iter 2 — supersedes conflicting text above)

A read-only Phase-B pass verified every `⟨verify⟩` symbol against source. Result: core claims
confirmed; the following **corrections and reconciliations are authoritative** where they conflict
with §§3–8.

### 14.1 Symbol corrections (confirmed against code)
- `evidence_bundle.py`: minter is `mint_bundle(project_dir)`, reader `resolve_bundle(project_dir)`;
  field is `attempt_id` (not `attempt`); bundle also carries `coherent`/`status`/`incoherence_reason`.
  Flag `OPENRESEARCH_CANONICAL_EVIDENCE_BUNDLE` confirmed.
- `evidence_gate.py`: `gate_decision(*, score, claims_result, has_disk_evidence) -> (score, vetoed)`
  confirmed. **Two distinct flags:** `OPENRESEARCH_LEAF_EVIDENCE_GATE` (per-leaf, default-OFF, this
  module) vs `OPENRESEARCH_EVIDENCE_GATE` (verdict-level, default-ON, in `report.py`). Spec §3 must
  reference both correctly.
- `leaf_triage.py`: **6** categories, not 5 — the existing `cell_failure` joins render_artifact /
  provenance_gap / aggregation_gap / protocol_gap / result_quality. New W1/W2 outcomes map onto these
  6; do not invent a 7th unless required.
- `CostLedgerEntry` (`backend/agents/resilience/cost.py`, frozen): also has `attempt_index`,
  `cache_read_input_tokens`, `cache_creation_input_tokens`, `reasoning_tokens`. W4 adds a row-type tag.
- `pricing.py` (`backend/agents/resilience/pricing.py`): `PRICING: dict[str, ModelPricing]`,
  `estimate_cost_usd(model, usage)`. Foundry blind spot **confirmed** — no `opus-foundry`/
  `sonnet-foundry`/`grok`/`azure-foundry` entries (the file even documents the $0 gap in a comment).
- `demo_status.json` cost_summary is written by `backend/agents/rlm/run.py` (`_compute_cost_summary`
  + `_update_cost_summary_loop`), **NOT** `live_runs.py`; it already includes `usd_per_iter_p50`.
- `SandboxMode` enum lives in `backend/agents/execution.py` (members auto/azure/brev/docker/gcp/
  local/runpod/simulate); **`gke` is an alias to `gcp` via `_missing_`**, not a member. W5 adds a
  member/alias for the hardened backend the same way.
- CLI (`backend/cli.py`): argparse subparsers, `sub.add_parser(...)` + `set_defaults(func=...)`;
  `campaign` requires all three money-cap args. W3's `paperbench-scorecard` mirrors this exactly.
- Test patterns to mirror: `tests/evals/test_leaf_scorer_feasibility_scope.py` (flag OFF-vs-ON A/B),
  `tests/test_cli_sanity_mode.py` (fixture-dir + gate-fires).

### 14.2 W1 reconciliation — *grading-input* integrity (not grader-code)
The root writes Python in a **sandboxed** REPL; it cannot mutate the host's `leaf_scorer.py`. The real
attack surface is the **grading-input artifacts the agent legitimately writes into the run dir**:
`rubric_tree.json`, `provenance.json`, `metrics.json`. W1 is therefore renamed **grading-input
integrity** and its mechanisms sharpen to:
- **M1 — Rubric pinning.** At rubric-gen, mint `rubric_tree_sha256` into a sidecar / the evidence
  bundle. At grade time, recompute; a post-gen change to `rubric_tree.json` → `evidence_tampered`.
- **M4 — Reported-vs-true metric cross-check** (unchanged intent): re-derive the headline metric
  from raw on-disk outputs; disagreement beyond tol → `metric_mismatch`. No-op when raw outputs absent.
- **M2/M3 — Access-audit + leakage** (unchanged).
- **Naming:** new module `backend/evals/paperbench/grading_input_integrity.py` (sibling to
  `deterministic_leaf_checker.py`); **must not** be named `grader_digest*` (that name is taken by the
  A6 metrics-compaction module). Flag `OPENRESEARCH_GRADER_INTEGRITY` unchanged.

### 14.3 W2 reconciliation — extend `deterministic_leaf_checker.py`, don't fork it
`deterministic_leaf_checker.py` already implements the exact pattern W2 wants: typed, pure, fail-soft
predicates over on-disk evidence, annotated per-leaf (`check_kind` + `assertion`), routed via
`check_leaf(leaf, run_dir)` with a backwards-compat fall-through to the LLM. **W2 adds one new
`check_kind = "deterministic:state_contract"`** whose `assertion` declares the GroundEval-style trace
predicates (checkpoint_exists / checkpoint_after_train_start / eval_loaded_checkpoint /
eval_covered_full_testset / metric_provenance_present), evaluated over `experiment_runs.jsonl` +
W1's access log + artifact mtimes. This inherits the module's fail-soft + backwards-compat guarantees
for free. The standalone `state_contracts.py` in §5 is **withdrawn** in favor of this extension.
Flag `OPENRESEARCH_STATE_CONTRACTS` still gates whether the router honors the new kind.

### 14.4 Revised first-implementation slice (TDD target)
Start with **W1-M1 rubric pinning** as a pure, self-contained, unit-testable module — highest
value-to-risk: it closes a concrete hole (agent weakening `rubric_tree.json` after gen), is a pure
hash+compare (ideal for TDD), and touches no live verdict path when the flag is OFF.

