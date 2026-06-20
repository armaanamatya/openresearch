# Pre-GPU code review + report-claim validation — design (2026-06-20)

> **Status:** proposed · supersedes the A/B/C sketch in
> `docs/runbooks/2026-06-20-validation-coverage-and-capped-rerun-handoff.md` (§3) with a
> grounded, recon-verified design. Builds on the grounded-self-improvement three-tier model
> (`2026-06-20-grounded-self-improvement-and-harness-reliability-redesign-design.md`).
> All knobs **default-OFF ⇒ byte-identical baseline.**

## 1. Problem

Two failure modes have stopped SDAR (and any paper) from reproducing honestly on GCP:

1. **Wasted GPU on agent-code bugs.** The executor writes training code that *runs* on the
   8×A100 grid for hours but the result is wrong/fake — the v6 case: real training, all-0.0
   metrics because the loss was never wired to the model graph. The bug is only discovered
   *after* the spend. We want to catch executor errors/hallucinations **before** the grid.
2. **Unchecked report narrative.** The grok validator + deterministic guards check the
   **executor's metrics**; the **root's report narrative** is unchecked — it can claim numbers
   absent from `metrics.json`.

The **red line** (inherited): fitness = the deterministic evidence layer, never the LLM grade,
never a model's narrative. Every LLM judgment in this design is backed by a deterministic or
empirical check.

## 2. What already exists (recon-verified, do NOT rebuild)

| Layer | File | Default | Stage |
|---|---|---|---|
| AST anti-fab scan (`torch.randn`→logprobs, hardcoded metrics, stub model, env contract) | `preflight_ast.py` (`OPENRESEARCH_ANTIFAB_GUARD`) | **ON** | pre-grid |
| Import preflight | `preflight_smoke.py` (`OPENRESEARCH_PREFLIGHT_SMOKE`) | **ON** | pre-grid |
| 1-step crash smoke (exit-code only) | `execution_smoke.py` (`OPENRESEARCH_EXECUTION_SMOKE`) | OFF | pre-train (in-sandbox) |
| Adversarial validator panel on metrics | `external_validator.py` (`OPENRESEARCH_EXTERNAL_VALIDATOR`) | OFF | in-loop / finalize |
| zero/constant-metrics veto | `zero_metrics_detection.py` (`OPENRESEARCH_ZERO_METRICS_GUARD`) | OFF | post-run |
| VRAM anti-fab verdict | `gpu_cell_runner.py` (`OPENRESEARCH_ANTIFAB_GUARD`) | ON | post-run |
| Tier-3 fix-first repair loop | `forced_iteration.py` (engaged by zero-metrics OR validator) | — | in-loop |

**Gaps:** (a) no *separate-model LLM code review* pre-GPU; (b) the smoke is crash-only, not
"metrics are real"; (c) no report-claim validation.

**The seam** for a new pre-GPU gate: `primitives.py::run_experiment`, **after** `validate_code_pre_flight`
(the AST scan, ~L5944-5959) and **before** the escalation loop / cell route (~L5960-6000). A gate
here returns a `repairable` error dict → the executor re-implements on the next iteration with
**zero GPU spent**. `ctx.validator_client` (the grok transport) is already built when
`OPENRESEARCH_EXTERNAL_VALIDATOR=1`.

## 3. The unifying frame

One **separate validator agent** (grok, cross-family from the Sonnet executor) and one
**deterministic verifier**, applied at three stages. Each LLM *points*; each deterministic check
*verifies*.

| Stage | Validator (points) | Deterministic verifier | This spec builds |
|---|---|---|---|
| **Code** (pre-GPU) | **P1** code-review gate | `preflight_ast` (exists) + **P2** metric-reality smoke | P1, P2 |
| **Metrics** (post-GPU) | external_validator panel (exists) | zero_metrics + evidence_gate (exists) | — (timeout cap to reach it) |
| **Report** (finalize) | **B** report predicate | **A** claim↔evidence teeth | A, B |

Plus **C** (metric-semantics guard), the validator **stamp fix**, and the **timeout cap +
incremental-metrics guidance**.

## 4. Components

### 4.0 Shared term vocabulary + result-only normalizer (fix codex-1, the latent P0)

The existing `zero_metrics_detection.normalize_metric_values` excludes *structural* keys
(`status`/`count`/`seed`/…) but **not hyperparameter/config keys** (`lr`, `learning_rate`, `beta`,
`lambda`, `temperature`, …). So `{"loss":0.0,"accuracy":0.0,"learning_rate":1e-5}` is **not**
"all zero" (the `1e-5` is a nonzero leaf) and the existing zero-metrics guard **fails to veto** —
a latent hole in the v6 fix itself, inherited by P2 and the claim engine which reuse it.

**Fix (one canonical vocabulary).** Add a shared `_CONFIG_TERMS` frozenset
(`lr|learning_rate|beta|lambda|weight_decay|momentum|gamma|eps|clip|warmup|temperature|seed|
gpu|vram|gb|hour|layer|dim|hidden|token|max_len|num_*|*_size`) in `zero_metrics_detection.py` and
exclude any key matching it from `normalize_metric_values`. The SDAR result keys (`loss`,
`l_grpo`, `mean_reward`, `accuracy_avg`, `f1_avg`, `teacher_gap_mean`, `gate_activation_ratio`)
stay included; configs drop out. This **strengthens the existing guard** (now vetoes the
results-all-zero-with-hparams shape) — update `test_zero_metrics_detection.py` accordingly. The
claim engine imports the same `_CONFIG_TERMS`.

### 4.0b Shared engine — `claim_grounding.py` (NEW, pure, single source of truth)

Used by A, B, and the validator's report predicate. No LLM. Matches by **metric identity**, not
bare float (fix codex-7).

```
@dataclass(frozen=True)
class Claim: value: float; term: str; context: str   # term = normalized result word, e.g. "success_rate"

def extract_result_claims(text: str) -> list[Claim]
    # A number within ~40 chars of a RESULT term; the matched result word is captured as `term`.
    #   _RESULT_TERMS = accuracy|success|success_rate|reward|f1|precision|recall|score|em|exact_match|return|win
    # EXCLUDING numbers adjacent to a _CONFIG_TERMS word (shared with 4.0). Parses ints/floats/percents.
    #   Fail-soft -> [].

def flatten_measured_values(project_dir: Path) -> list[tuple[str, float]]
    # (term, value) pairs from code/metrics.json + every outputs/**/metrics.json, via the
    # result-only normalizer (4.0). `term` derived from the leaf key path (e.g. ".success_rate"
    # -> "success_rate"). Fail-soft -> [].

def check_claims_grounded(claims, measured, *, rel_tol=0.05) -> ClaimGrounding
    # ClaimGrounding = {grounded:[Claim], ungrounded:[Claim]}
    # A claim is grounded iff some measured (term', value') has COMPATIBLE term (claim.term and
    #   term' map to the same canonical metric, via a small synonym map e.g. success~success_rate,
    #   em~exact_match) AND value within rel_tol (trying value and value/100 for percent<->fraction).
    # Empty `measured` -> ALL claims "unverifiable" (NOT ungrounded): A/B no-op, evidence_gate owns
    #   the no-evidence case.
```

**`paper_claims` are excluded** from achieved-result extraction (fix codex-7): they are the
paper's *target* numbers, not the reproduction's results. A reads claims only from
`reproduction_summary` + `reported_metrics` (the root's *achieved* assertions).

`test_claim_grounding.py`: `success_rate=0.84` NOT grounded by a measured `loss=0.84` (identity
mismatch); grounded by a measured `success_rate≈0.84`; hyperparameter number exempt; `84%` matches
`0.84`; empty-measured ⇒ no "ungrounded"; paper_claims numbers never extracted as achieved.

### 4.1 P1 — code-review gate · `code_review_gate.py` (NEW)
**Flag** `OPENRESEARCH_CODE_REVIEW_GATE` (default OFF; requires `OPENRESEARCH_EXTERNAL_VALIDATOR=1`).
**Hook** `run_experiment` seam (after AST scan, before grid).

```
def code_review_gate_enabled() -> bool   # flag AND external_validator_enabled()

def review_executor_code(*, validator_client, code_dir, method_context: str) -> CodeReviewVerdict
    # CodeReviewVerdict = {blocking: bool, findings:[{file,line,severity,anti_pattern,detail}], raw:str}
```

- **Input**: the executor's training files (`train.py`/`train_cell.py`/env modules) + a
  `method_context` string built from the rubric / paper hint (paper-agnostic — the method spec is
  injected, the checklist is generic).
- **Checklist (generic anti-fab + method fidelity)**: does the loss trace to a real forward pass
  over sampled rollouts? are rewards from real env outcomes (not constants/`randn`)? is the
  teacher loaded and the gap computed from real logprobs? does eval score against gold labels?
  are any metrics hardcoded? is the model a real `from_pretrained`, not a stub?
- **Output contract**: grok returns JSON findings, each with `severity ∈
  {will_produce_fake_metrics, will_produce_wrong_metrics, style, uncertain}` and a `file:line`.
- **Grounding / conservatism**: a finding is **blocking** only if
  `severity ∈ {will_produce_fake_metrics, will_produce_wrong_metrics}` **and** the cited `file:line`
  exists in `code_dir`. Lower severities are advisory (attached, never block). This keeps the
  cheap LLM filter from false-blocking; P2 (smoke) is the empirical backstop for what it misses.
- **On blocking** → `run_experiment` returns `{outcome:"repairable", failure_class:"code_review_rejected",
  repair_context: <cited findings>}` — **no GPU spent**; feeds the existing repair loop
  (bounded by `REPAIR_MAX` → honest `repair_exhausted`).
- **Fail-OPEN at runtime**: validator unavailable / timeout / parse-error ⇒ non-blocking, proceed
  (the grid + P2 + post-run guards remain the backstop; a transient grok hiccup must not wedge a
  run). FAIL-CLOSED stays at client *build* time (existing `build_validator_client`).

`test_code_review_gate.py`: blocking finding ⇒ repairable dict; advisory-only ⇒ proceed;
client None / raises ⇒ proceed (fail-open); flag off ⇒ gate not invoked.

### 4.2 P2 — metric-reality smoke · `metric_reality_smoke.py` (NEW)
**Flag** `OPENRESEARCH_METRIC_REALITY_SMOKE` (default OFF). **Hook** `run_experiment` seam (after P1,
before the full grid).

```
def metric_reality_smoke_enabled() -> bool
def run_metric_reality_smoke(*, ctx, code_dir, cells: list|None) -> SmokeVerdict
    # SmokeVerdict = {ok: bool, failure_class:str|None, detail:str}
```

- **Coverage = one representative per group** (fix codex-3): group `cells.json` by
  `(model_id, eval_env, script/family, gpus)`; smoke one cell per group, capped at
  `OPENRESEARCH_SMOKE_MAX_CELLS` (default 4), run in parallel across GPUs (~2-3 min wall-clock).
  Smoking only the smallest cell would let a 3B/alfworld/multi-GPU-only bug through to the full grid.
  Each representative runs ~2 steps on a tiny slice (`OPENRESEARCH_CELL_MAX_STEPS=2` + tiny-slice env
  + `CUDA_VISIBLE_DEVICES=<one>`); bounded by `OPENRESEARCH_SMOKE_TIMEOUT_S` (default 300s).
  **Monolithic route** -> run the entry script 2 steps.
- **Assertions demand POSITIVE proof of real training** (fix codex-2 — a single nonzero aggregate
  must NOT pass). Per representative, REQUIRE a per-step trace (`metrics.json`/`smoke_trace.json`)
  with **>=2 step records**, and assert: a loss-like key present, **>0 and varied** across steps (a
  disconnected/constant loss fails); reward present and not constant-zero; peak VRAM > floor
  (`vram_evidence_verdict`). When the trace carries graph-evidence fields (guidance requests them):
  `grad_norm` finite & >0, `param_delta_norm` >0. Uses the result-only normalizer (4.0) so a nonzero
  hparam cannot satisfy "non-zero".
- **Cheap method-fidelity contract checks** (fix codex-5, partial): when the trace provides them,
  assert optimizer `lr`/`beta`/`lambda` match the expected hparams (paper hint / extracted contract)
  and teacher/student logprobs are finite where the method requires a teacher. Catches *wrong-but-sane*
  hparams a range-check misses. **Honest scope:** this *mitigates* subtle method errors; it does NOT
  fully close them (deeper objective correctness is P1's qualitative job; train/eval split-leak
  detection via disjoint split hashes is a documented future extension, see §10).
- **On any assertion failure OR "executor ran but produced no valid trace"** ->
  `{outcome:"repairable", failure_class:"smoke_metrics_unreal", repair_context:<which assertion>}`
  (fix codex-4 — "ran, no trace" is a broken-trainer signal, NOT a soft pass). Feeds the repair loop.
- **Fail-OPEN only when the smoke cannot launch at all** (no GPU visible, subprocess won't spawn) —
  a harness problem, not an executor problem; proceed (post-run guards backstop). Any executor-side
  outcome is judged, never soft-passed.

`test_metric_reality_smoke.py` (mock the cell run): all-zero trace ⇒ repairable; single nonzero
aggregate w/o >=2 step records ⇒ repairable; real varying loss + grad + VRAM ⇒ ok; one group fails
⇒ repairable; launch-failure ⇒ proceed (fail-open); flag off ⇒ not invoked.

### 4.3 A — report-claim teeth · `report_claim_gate.py` (NEW, deterministic, universal)
**Flag** `OPENRESEARCH_REPORT_CLAIM_GATE` (default OFF). **Hook** `report.py::write_final_report_rlm`
as the **FINAL verdict mutation** — **after** the best-of-run floor (~L1859) and the two-axis attach,
immediately before `model_dump_json` (fix codex-6: at ~L1830 the floor would *undo* the downgrade).
**Runs on ALL three finalize paths** (single write chokepoint) — the universal backstop covering the
SIGTERM/watchdog salvage path the LLM validator never reaches.

- Extract achieved claims from `report.reproduction_summary` + `report.reported_metrics` **only**
  (NOT `report.paper_claims`, which are paper *targets* — fix codex-7); `check_claims_grounded` (4.0b,
  identity-matched) vs `flatten_measured_values`.
- **Teeth (verdict CAP, applied last so nothing can re-upgrade it — fix codex-6):** if there is ≥1
  ungrounded **result** claim **and** measured evidence is non-empty (else no-op; evidence_gate owns
  no-evidence) **and** the verdict is currently `reproduced`/`partial` → **cap** the verdict at
  `partial` (conservative; never `failed` on a narrative-only mismatch, never an upgrade) and append a
  **cited** note to `reproduction_summary` (`[harness] report claimed <v> (<term>); code/metrics.json
  has no matching measured <term> within 5% → verdict capped at partial`). Also cap the serialized
  two-axis `reproducibility`/`implementation_verdict` dict and recompute `meets_target`/top-level
  mirror from the capped verdict. On the natural path B has already forced reconciliation, so A
  typically finds nothing; A's teeth are the salvage / `repair_exhausted` backstop.
- **Byte-identical-off (fix codex-8):** do NOT add a `claim_grounding` field to `RLMFinalReport`.
  Stamp `"claim_grounding"` into the **serialized dict** only when the flag is on (mirror the existing
  conditional-stamp pattern); when off, no key is emitted and the verdict is untouched. Emit
  `run_warning code="report_claim_ungrounded"`. Fail-soft: any exception ⇒ skip (never break the write).

`test_report_claim_gate.py`: ungrounded result claim + measured present ⇒ cap to partial + cite +
`claim_grounding` key; a floor-upgrade afterward does NOT lift it (codex-6 regression); grounded
report ⇒ unchanged; `loss=0.84` does NOT ground `success_rate=0.84` (codex-7); hyperparameter number
⇒ not flagged; no measured evidence ⇒ no-op; flag off ⇒ no `claim_grounding` key + byte-identical verdict.

### 4.4 B — in-loop report-fix gate (the aggressive path)
**Flag** `OPENRESEARCH_REPORT_CLAIM_GATE` (same flag as A) **and** the fix-first loop engaged.
**Mechanism**: capture the `FINAL_VAR` value in the `forced_iteration` interceptor and check its
claims **before** the report is accepted, so an overclaiming report drives a Tier-3 repair.

- `forced_iteration.py::_intercepted_final_var(self, variable_name)`: resolve the variable's value
  in the REPL namespace (`self`'s locals), stash it on the policy as `policy.pending_final_value`
  (a stringifiable snapshot). Best-effort; resolution failure ⇒ leave `None` (B no-ops, A still
  backstops at the chokepoint).
- New `policy.claim_gate` callback (sibling to `validator_gate`), wired in `run.py` only when the
  fix-first loop is engaged AND `OPENRESEARCH_REPORT_CLAIM_GATE=1`. It stringifies
  `pending_final_value`, `extract_result_claims` + `check_claims_grounded` vs on-disk measured; if
  ungrounded result claims exist → return `(refuse, directive)`.
- `should_refuse` consults `claim_gate` next to `validator_gate` (same `repair_floor` signature →
  `record_repair_attempt("report_claim")` → bounded by `REPAIR_MAX` → honest `repair_exhausted`,
  excluded from `root_degenerate_loop`). Directive: "your report claims <v> but metrics.json has no
  matching value; either run_experiment to produce that evidence or correct the report, then
  FINAL_VAR again."
- `_fixfirst_loop_engaged()` extended: also True when `OPENRESEARCH_REPORT_CLAIM_GATE`,
  `OPENRESEARCH_CODE_REVIEW_GATE`, or `OPENRESEARCH_METRIC_REALITY_SMOKE` is on (so the loop drives
  these refusals). Off ⇒ unchanged.

`test_report_claim_refuse.py`: pending value with an ungrounded claim ⇒ refuse + directive;
grounded ⇒ accept; bounded to `REPAIR_MAX` then accepts (`repair_exhausted`); value-resolution
failure ⇒ no refuse.

### 4.5 Validator report predicate (defense-in-depth) · `external_validator.py`
**Flag** `OPENRESEARCH_VALIDATOR_CHECK_REPORT` (default OFF; requires `EXTERNAL_VALIDATOR=1`).
Add a 5th predicate `report_claims_grounded` (5 touch points: `_VALID_PREDICATES`,
`_ADVERSARIAL_SYSTEM`, `_ADVERSARIAL_USER_TEMPLATE`, a `check_report_claims_grounded` whose
machine-check is `claim_grounding.check_claims_grounded`, and the `_machine_check` dispatch).
Thread the report's claims into the `_finalize` offline panel (`run.py` ~L3560) via a new
`report_claims=` arg to `run_validation_panel`. Min-aggregation veto unchanged. This is the grok
agent *also pointing* at the report; A is the teeth.

`test_external_validator_report.py`: report with an ungrounded claim ⇒ predicate machine-verifies
⇒ veto; grounded ⇒ clean; flag off ⇒ predicate absent.

### 4.6 C — metric-semantics guard · `metric_semantics.py` (NEW)
**Flag** `OPENRESEARCH_METRIC_SEMANTICS_GUARD` (default OFF). **Hook** `run_experiment` after the
zero-metrics guard (~L6511).

```
def metric_semantics_violation(metrics) -> str | None
    # rate-named keys (accuracy|success_rate|f1|precision|recall) must be in [0, 1+eps];
    # loss/reward must be finite. Returns a detail string on a CLEAR violation else None.
    # Conservative: 0.0 is in range; only out-of-range fires.
```
On violation → `failure_class="fabrication_suspected"` repair dict (beside the zero-metrics/stub
guards). `test_metric_semantics.py`: accuracy 1.7 ⇒ veto; 0.4 ⇒ pass; reward 0.0 ⇒ pass; off ⇒ no-op.

### 4.7 Validator stamp fix
The bridged validator `RoleSpec` (azure-foundry) stamps the global `AZURE_FOUNDRY_DEPLOYMENT`
instead of `OPENRESEARCH_VALIDATOR_MODEL`. Fix the validator-role stamp path to prefer
`OPENRESEARCH_VALIDATOR_MODEL` when set (transport already correct; this corrects `report.models`/
`validation`). `test_validator_stamp.py`.

### 4.8 Config — timeout cap + incremental-metrics guidance
- `OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S=1800` (already wired: `resolve_experiment_timeout_s` →
  `per_cell_timeout_s`). 30 min/cell ⇒ cells finalize on partials ⇒ `run_experiment` returns ⇒
  natural `FINAL_VAR` ⇒ validator + finalize reached (the §2 timeout-trap fix).
- Add to `runs/.cache/sdar_scope_guidance.txt`: "**write `metrics.json` incrementally after every
  training step** (not only at the end) as a list/array of per-step records — each record carrying at
  least `step`, `loss`, `mean_reward`, and (when available) `grad_norm` and `param_delta_norm` — so a
  wall-clock cap captures real partials AND the pre-GPU smoke can prove the loss backprops. Honor
  `OPENRESEARCH_CELL_MAX_STEPS` when set (the smoke runs you for ~2 steps). Run a per-cell self-check
  that `loss>0` and reward/accuracy are not all-zero before exit; an all-0.0 trace is a
  fabrication-equivalent failure."

## 5. Flag table (all default-OFF ⇒ byte-identical)

| Flag | Component | Requires |
|---|---|---|
| `OPENRESEARCH_CODE_REVIEW_GATE` | P1 | `EXTERNAL_VALIDATOR=1` |
| `OPENRESEARCH_METRIC_REALITY_SMOKE` | P2 | — |
| `OPENRESEARCH_REPORT_CLAIM_GATE` | A (+B when loop engaged) | — |
| `OPENRESEARCH_VALIDATOR_CHECK_REPORT` | validator predicate | `EXTERNAL_VALIDATOR=1` |
| `OPENRESEARCH_METRIC_SEMANTICS_GUARD` | C | — |
| `OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S` | timeout cap | — (config) |

## 6. Byte-identical-when-off contract

Each module's `*_enabled()` returns early when its flag is off; the `run_experiment` seam calls
P1/P2 only when enabled; `write_final_report_rlm` calls A only when enabled; the `FINAL_VAR`
interceptor stashes/checks only when B is enabled; `_fixfirst_loop_engaged()` adds the new flags as
OR terms only (off ⇒ unchanged). Held by the existing default-OFF regression suite + a new
`test_validation_layers_off_noop.py`.

## 7. Failure-class summary (all repairable, bounded by `REPAIR_MAX`)

`code_review_rejected` (P1) · `smoke_metrics_unreal` (P2) · `fabrication_suspected` (C, shared) ·
`report_claim` refusal (B, `repair_floor` signature). All feed the existing fix-first loop → honest
`repair_exhausted` when the executor can't reconcile (never `root_degenerate_loop`, never a shipped fake).

## 8. SDAR re-run config (after build + codex review)

Proven base block (`runs/.cache/grounded_test_env_block.sh`) + new lines:
```
OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S=1800
OPENRESEARCH_CODE_REVIEW_GATE=1
OPENRESEARCH_METRIC_REALITY_SMOKE=1
OPENRESEARCH_REPORT_CLAIM_GATE=1
OPENRESEARCH_VALIDATOR_CHECK_REPORT=1
OPENRESEARCH_METRIC_SEMANTICS_GUARD=1
OPENRESEARCH_SDAR_PROJECT_ID=sdar_gcp_grounded_v2_20260620
```
root=gpt-chat-latest (Foundry), executor=Sonnet/OAuth, validator=grok-4.3 (Foundry, separation
`independent`). Success = pre-GPU gates fire on a planted bug, validator panel runs at finalize,
honest terminal (real reproduction OR cited `degraded`/`failed`/`repair_exhausted`), never a
silently-shipped fake. Then STOP the VM.

## 9. Risks

- **P1 false-block** → conservative severity gate + fail-open + P2 backstop; bounded by `REPAIR_MAX`.
- **P2 partial-metrics thinness at 2 steps** → assert *presence + variation*, not magnitude; the
  incremental-metrics guidance guarantees per-step writes.
- **A false-downgrade from parser miss** → conservative extraction (result-term-adjacent, hparam
  exempt, percent/rounding tolerance), downgrade-only, cited/auditable, no-op when no measured
  evidence.
- **gpt-chat-latest root non-determinism** (v8 degenerated) → degenerate detector aborts fast;
  probe creds at launch; prefer a funded reliable root (gpt-5/claude-API) if available.
- **claude-oauth-as-root regression** → out of scope (this run uses gpt-chat-latest); separate bisect.

## 10. Honest scope — what this closes vs. mitigates

Stated plainly so we don't overclaim (the codex review's overall verdict): no validator fully
proves a reproduction is *scientifically correct* without re-deriving it. This design draws a clear
line:

- **Closed robustly (the fabrication class):** fake/zero/constant metrics (strengthened normalizer +
  P2 empirical smoke + post-run zero-metrics guard), blatant code anti-patterns (existing
  `preflight_ast`), GPU-claimed-but-no-VRAM (existing antifab), crash/import failures (existing
  smokes), per-group scale/env crashes (P2 per-group), out-of-range metrics (C), and report claims
  with no measured backing (A teeth + B in-loop). A variant of the v6 all-0.0 failure must now evade
  *five* independent gates.
- **Mitigated, not fully closed (the subtle-method class):** a *live but scientifically wrong*
  reproduction — wrong objective wired correctly enough to train, a data-split leak within sane
  ranges, an off-by-sign surrogate. P1 (LLM code review) and P2's hparam/logprob contract checks
  reduce this; they do not eliminate it. **Documented future extensions:** train/eval split-hash
  disjointness, objective-term contract assertions, a reference-trajectory check. We surface these as
  warnings where detectable rather than pretending to certify correctness.

The red line holds throughout: every teeth/veto decision is a deterministic or empirical check; the
LLM only *points*.
