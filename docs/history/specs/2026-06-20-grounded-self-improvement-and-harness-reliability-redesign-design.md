# Grounded Self-Improvement + Harness Reliability Redesign — design

> **Doc status:** Draft (rev. 2 — post Codex adversarial review + code verification) · spec (design)
> · authored 2026-06-20 · source-of-truth tier 3 (design intent until implemented). Supersedes the
> next-session brief in
> `docs/runbooks/2026-06-20-sdar-harness-refactor-and-external-validation-handoff.md`.
> Policy: [`docs/policies/documentation.md`](../../policies/documentation.md).
>
> **Decisions locked (brainstorming, 2026-06-20):** scope = one coordinated (umbrella) redesign,
> per-phase implementation sub-plans · root-auth = support ALL transports (reliability must be
> transport-agnostic) · degenerate-recovery = recovery-aware detect+nudge+clean-abort, **no harness
> synthesis** · spot-resilience = **reuse + verify the existing memoization** (not a new cache) ·
> self-improvement scope = **A+B** (in-run grounded repair + cross-run positive recipes; recursive
> harness-evolution **deferred**) · validator = separate-model adversarial panel, **min-aggregation
> veto (remedial, not terminal)**, deterministic floor per-experiment + LLM panel at
> `FINAL_VAR`-attempt, **typed-predicate machine-checks** + re-run-on-suspicion · continue-policy =
> **unified into `ForcedIterationPolicy`** with a **distinct repair-refusal class** · error handling =
> **one fix-first / fail-honest error→repair→re-validate loop for all repairable errors.**
>
> **Review status:** every Codex blocker + major resolved; see the **§18 review-resolution log**.

---

## 1. Problem & context

A multi-attempt SDAR-on-GCP run (`v6`→`v9`, 8×A100) proved the harness's hard parts work — a Sonnet
executor (OAuth) writes real multi-file SDAR code and reached real 8-GPU training for the first time
— but exposed two blockers to a *completed* run, a reliability gap, and an operator ask:

- **A. Zero-metrics hallucination (motivating case).** A `v6` cell ran 150 real GPU steps
  (`status:"completed"`, ~30 GB VRAM) yet wrote a `metrics.json` whose every value is exactly `0.0`.
  Every existing guard missed it: `STUB_METRICS_GUARD` keys on placeholder *keys* (here keys are
  real), VRAM antifab keys on ~0 GPU (GPU was used), and `EVIDENCE_GATE`/leaf-grader fire only
  *after* `verify_against_rubric`, which the spot-preempted run never reached.

- **B. No reliable + keyless RLM root.** `gpt-5` (only `RISK_NONE` paper-validated root) needs a live
  `OPENAI_API_KEY` (dead); `claude` (API-key, reliable driver) needs a funded `ANTHROPIC_API_KEY`
  (empty); `claude-oauth` is `RISK_DEGENERATE_LOOP` (`root_validation.py:67`); `gpt-chat-latest`
  (Foundry) is non-deterministic. Operator reports it **worked via `claude-oauth` on a university
  cluster but fails now** — the degenerate detector + abort landed 2026-06-17, days before.

- **C. Spot-preemption fragility.** Each relaunch re-runs root overhead. (Reframed below: cacheable
  primitives already memoize; this is smaller than the handoff implied.)

- **D. Operator ask.** External-agent validation, **integrate SOTA self-improvement** (mid-2026
  literature), refactor/optimize.

### 1.1 Ground-truth recon (verified against code, 2026-06-20; all Codex claims confirmed)

| Surface | Finding (file:line) |
|---|---|
| Zero-metrics insertion | `STUB_METRICS_GUARD` at `primitives.py:6442-6464` inside `run_experiment` post-processing (gated `OPENRESEARCH_STUB_METRICS_GUARD`), inspecting metric **keys** only. Sibling insertion point `primitives.py:6465`. |
| Cell → aggregate | `_load_metrics` **def** `gpu_cell_runner.py:267`, primary **call** `:913` (also `:801/:815/:985/:1013`). `aggregate_cell_metrics` `cell_matrix.py:713-802`, **two** call sites: `primitives.py:5663` (success) and `:5450` (all-cells-dropped terminal). **No numeric-value validation anywhere.** |
| **Metric shapes (critical)** | Real `metrics.json` are **flat scalars with NO per-step history** (e.g. `{"loss":0.0,"return":31.1,…}`). Nested `per_model[model][env][baseline]` appears in **only 8/148** files (cell-aggregated `code/outputs/*`); monolithic runs are flat. `aggregate_cell_metrics` emits flat-scalar leaves, **neither requires nor emits history**. `experiment_runs.jsonl` = `{timestamp,success,metrics,logs(str),artifact_dir}` — **zero** per-step series anywhere. **⇒ no raw series to "recompute from."** |
| Provenance (the usable floor signal) | `OPENRESEARCH_METRIC_PROVENANCE` **default ON** (`report.py:1285`) writes a `provenance.json` manifest; `deterministic_leaf_checker` already checks hparam-vs-provenance. This is the artifact the floor + validator key on. |
| Evidence gate timing | Two gates, opposite defaults: **report-level** `_apply_evidence_gate` **default ON** (`report.py:1496`); **leaf-scorer veto (A7)** **default OFF** (`evidence_gate.py:48`, `leaf_scorer.py:291`), called from `leaf_scorer.py:1991` *after* grading. A preempted-before-verify run reaches neither. |
| **Existing memoization (reframes resume)** | `primitive_cache.py` is a **separate on-disk JSONL cache** (`rlm_state/primitive_cache.jsonl`), key `version:primitive:sha256[:32]` (`:189-198`), `CACHEABLE_PRIMITIVES = {understand_section, extract_hyperparameters, detect_environment, plan_reproduction, verify_against_rubric, implement_baseline}` (`:66-78`), **explicitly excludes** `run_experiment` (real-world-state) + `build_environment` (Docker-cached) (`:40-45`). Called via `maybe_get`/`put` from `primitives.py`, **not** in `wrap_primitive`. `run_experiment` resumes via cell-fingerprint (`cell_scheduler.should_skip_cell` `:159-182`). |
| Degenerate machinery | `register_refusal` (`forced_iteration.py:584-622`); reset **only** via `record_state_change` (`:572-582`, fired by `run_experiment`/`implement_baseline`/`build_environment`); trip → `on_degenerate_refusal_loop` → terminal `root_degenerate_loop` (`run.py:980-1108`). Default threshold 3 (`forced_iteration.py:79-87`). |
| Role/transport | `RoleSpec(role,token,provider,model)` — **no family/lineage field** (`role_models.py:146-158`; only an `is_claude` property). `ROLES`=(planner,executor,verifier,grader) `:91`; `_SUBROLES` `:94`; `RoleSelection` `:191`; `resolve_role_models` `:340`; `_resolve_subrole` `:381`. `grader_transport.build_transport_client` (`:116`) **silently falls back to the caller's client** on missing/unknown/error config — *"we NEVER raise"* (`:245-261`). `sample_completions` `:80`. |
| Champion | `champion_artifact.best_champion` selects `max(key=(median_score, seq))` — **grader-derived** (`:178-199`); snapshot is **source-only** (+`rubric_block.json`), strips datasets/outputs/weights (`:47-50`). |

---

## 2. Goals / non-goals

**Goals.** (1) Kill the plausible-but-fake-result class. (2) Make *any* root usable (recovery +
honest fail) across *all* transports. (3) Integrate self-improvement that is provably safe under the
2026 reward-hacking evidence. (4) Reduce spot-preemption rework (reuse existing memoization). (5)
Replace the brittle env-whitelist launch path.

**Non-goals.** Recursive harness self-modification (Tier C — deferred). Forking `rlms` to resume its
engine. Re-litigating the existing grade-based *reporting* mechanisms (champion / best-of-run — §3.1).
A new raw-per-step-metrics schema as a *dependency* (offered as an optional future signal, §6.3).
Paid-key dependencies (all transports optional).

**Universal constraint (scoped — §11.2).** Every NEW flag is default-OFF; with all NEW flags unset the
harness is byte-identical to **its current baseline** — which already includes default-ON rails
(`REPROLAB_FINALIZE_REGRADE`, `REPROLAB_LEAF_TRIAGE`, `OPENRESEARCH_METRIC_PROVENANCE`, the
report-level evidence gate). "Byte-identical to today" means *to that baseline*, not to a
no-rails harness.

---

## 3. The spine — one principle

> **A self-improvement loop's fitness signal is the deterministic evidence layer — never the LLM
> grader or `overall_score`.** The external validator is the *grounding signal*; the deterministic
> gates are the *floor*. The system is *fix-first, fail-honest.*

Forced by the mid-2026 literature (§16): intrinsic self-correction without an external signal is
flat-to-negative and *hardened* through June 2026; self-improvement pointed at a noisy proxy
reward-hacks 47–74% of the time; LLM judges carry self-preference + consensus-collapse biases.
Three-tier trust model, each tier pointed only at the tier beneath:

```
 Tier 3  Self-improvement loops (A in-run repair · B cross-run recipe)
         fitness = Tier 1 predicates (+ Tier 2 veto); NEVER the grade
 Tier 2  External adversarial validator — separate model · adversarial · TYPED-PREDICATE machine-checks
 Tier 1  Deterministic floor — zero/constant-metrics veto · provenance check · evidence_gate ·
         deterministic_leaf_checker · evidence-fingerprint
```

### 3.1 The red line — precise scope (resolves the champion/grade conflict)
The red line governs **NEW Tier-3 decisions only**: *what to repair*, *whether a repair is accepted*,
*whether a recipe is admitted*. These must use Tier-1 predicates (+ the Tier-2 veto), never a
grade-derived score. **Existing grade-based *reporting* is explicitly out of scope** — `champion_artifact`
(selects by `median_score`) and the best-of-run floor *report* the best-graded artifact; they do not
*train* against the grade, and changing them is a separate concern. Distinction: **reporting a grade ≠
optimizing against it.** Guard: the new modules (`recipe_library`, the repair-acceptance path) must
not import or read grade-derived score fields except to copy them into the report.

---

## 4. Keystone — the evidence/lifecycle **ledger** (audit + provenance; record-only)

The recon collapses the keystone into its *honest* role. Memoized resume **already exists**
(`primitive_cache.py` + cell-fingerprint + Docker cache); the genuinely-missing piece is an
**auditable, evidence-fingerprinted record of every primitive's outputs** — the validator's
claims-bundle and recipe provenance.

### 4.1 What it is — `backend/agents/rlm/lifecycle_ledger.py` (new; stdlib-only, fail-soft)
A typed, append-only, **record-only** sidecar under `runs/<id>/rlm_state/lifecycle/`:

```python
@dataclass(frozen=True)
class LedgerRecord:
    primitive: str          # all lifecycle primitives (incl. run_experiment — recorded, not memoized)
    seq: int
    inputs_projection: dict # per-primitive REDACTED projection (4.3) — never raw paper text
    outputs_pointer: dict   # disk paths produced (code/, metrics.json, env id, plan path, provenance.json)
    evidence_keys: list[str]
    outcome: str            # ok | failed | raised | timeout
    iteration: int
```

**Record-only — there is NO memoization short-circuit in the ledger.** This is deliberate: a
`wrap_primitive` short-circuit would skip the side-effects that already happen around
`binding.py:383` (supplemental evidence emission, context-map updates, worker-report handling,
cost-ledger outcome). Memoization stays where it lives today (§5.4).

### 4.2 Write path + atomicity
Written by the existing `binding.wrap_primitive` interceptor (beside the cost-ledger stamp), gated
`OPENRESEARCH_LIFECYCLE_LEDGER`. **Write only after result validation + artifact checks**, via
temp-file + atomic `os.replace`. **Never record `ok` for a `timeout`/`raised`/retryable envelope** —
`wrap_primitive` runs primitives in daemon threads that can return a timeout while abandoned work
still mutates disk, so the ledger records the *validated* outcome, not the envelope.

### 4.3 Redaction (per-primitive projections)
`inputs_projection` is built by a **per-primitive projection function** (not the value-light wrapper
summary, which is not a redaction scheme). Each projection emits only bounded, non-corpus fields
(e.g. plan = section ids + hyperparameter keys, never paper prose). Guard: a **sentinel test** writes
a run with a known paper-text canary and asserts no ledger/cache file contains it.

### 4.4 Three consumers
1. **Validator claims-bundle (§7)** — the ledger + `evidence_keys` + `provenance.json` is the on-disk
   evidence the panel audits, available even on a run preempted before `verify_against_rubric`.
2. **Recipe provenance (§9.2)** — a gate-passed run's records supply the recipe's evidence fingerprint
   (the recipe body is a structured pattern, not a champion copy — §9.2).
3. **Resume reference (§5.4)** — the ledger does *not* implement resume; it points at what the
   existing caches already memoized.

---

## 5. Reliability layer (transport-agnostic · recovery-aware · no synthesis)

### 5.1 Recovery-aware detector (minimal integration lands in P3, not P5)
Upgrade the detector from *detect→abort* to *detect→nudge→(rope)→clean-abort*: on a no-progress
refusal, emit a **stage-specific cited nudge** (`root_progress.infer_required_stage` names the missing
primitive, now pointing at the ledger record, e.g. *"plan at `rlm_state/lifecycle/plan_reproduction.0.json`;
call `implement_baseline(plan=…)`"*). The harness **never synthesizes the root's calls** (grill #2).
Transport-agnostic. The *minimal* detector/continue-policy integration ships in **P3** (with the
repair loop), so no phase wires a veto whose recovery is absent (resolves the P5-too-late finding).

### 5.2 Precondition experiment (operator-run, ~$2)
Run SDAR once with `OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD=16` (≈ pre-2026-06-17). Self-recovers ⇒
detector regressed (generous default rope, nudge-first); still loops ⇒ OAuth root genuinely
degenerate (tight default, lean on keyed/Foundry for completion). Outcome recorded, not guessed.

### 5.3 Unified continue-policy with a **distinct repair-refusal class** (resolves B1)
`ForcedIterationPolicy` becomes the single authority over continue-vs-`FINAL_VAR`. The validator/floor
repair refusal is a **distinct refusal class**, NOT a root no-progress signature:
- A real repair re-runs `implement_baseline`/`run_experiment`, which **already calls
  `record_state_change` and resets the degenerate counter** — so the healthy fix-first path is
  byte-safe today.
- A repair refusal's **"progress" is keyed to the evidence fingerprint changing** between attempts. It
  carries its own budget (`OPENRESEARCH_REPAIR_MAX_ITERATIONS`) and is **excluded from the
  `root_degenerate_loop` signature**.
- If the root refuses to actually repair (no state change, evidence unchanged after the nudge), the
  policy stops with **`failure_class="repair_exhausted"` (honest)** — a clear, distinct outcome, never
  the confusing `root_degenerate_loop`.

Migration safety: the validator-input path is behind `OPENRESEARCH_EXTERNAL_VALIDATOR`; OFF ⇒
byte-identical to the hardened 2026-06-17 policy (contract test).

### 5.4 Spot resilience — reuse + verify, don't reinvent (resolves B2)
Resume already exists in three forms: `primitive_cache.py` (cacheable LLM-derivation primitives, on-disk,
content-hash keyed), cell-fingerprint (`run_experiment`), Docker cache (`build_environment`). **Work
here = verify + extend, not a new cache:**
- Verify `primitive_cache.jsonl` is enabled by default for the SDAR path and that its content-hash key
  is stable across a same-`--project-id` relaunch (so a fresh root's `implement_baseline` cache-hits).
- Extend coverage/instrumentation only where a gap is found.
- The root's *re-reasoning* between cached calls (rlms restarts at iteration 0) is **inherent and out
  of scope** — the expensive primitive bodies are what we avoid re-running.

---

## 6. Deterministic floor (Tier 1 — the fitness function)

### 6.1 Zero/constant-metrics veto — works on the REAL flat shape (resolves B4)
New `backend/agents/rlm/zero_metrics_detection.py` (stdlib-only, mirroring `stub_detection.py`),
sibling guard at `primitives.py:6465`, gated `OPENRESEARCH_ZERO_METRICS_GUARD`. **Normalizes BOTH
shapes first** (flat scalar — the common case — and nested `per_model`), then:
- **Result-claiming keys all zero/constant** (loss AND reward AND accuracy ≡ 0.0, or all bit-identical
  across the run's cells) **AND a GPU-training claim** (`gpu_cell_runner.metrics_claim_gpu_training`)
  **AND `provenance.json` absent or inconsistent** (no manifest linking the metric to a real output)
  ⇒ degrade to `failure_class="fabrication_suspected"`, emit `run_warning`, fail-soft.
- **False-positive guard:** a flat all-zero metric **with valid provenance** (a real eval that scored
  0 — a legitimately-failing baseline) is **NOT** vetoed. Provenance presence is the discriminator
  between "fake 0" and "real 0," since no per-step series exists to check variance.
- When a per-step history array *does* exist (rare today, see §6.3), zero-variance-across-history is an
  additional stronger signal.

### 6.2 (removed) "recompute-from-raw"
Dropped as a primary mechanism: **no raw per-step series exists** anywhere (`metrics.json` flat;
`experiment_runs.jsonl` text logs). Its intent is served by the provenance + constancy + GPU-claim
checks in §6.1 and the rerun-on-suspicion in §7.5.

### 6.3 Optional future signal — history schema contract (not a dependency)
A soft contract nudging `train_cell.py` to emit a per-step `history` array (via implementer guidance +
a non-fatal `contract_warning`), so the stronger variance signal becomes available over time. **The
floor must not depend on it** (it would no-op on every existing artifact).

### 6.4 Reuse
`evidence_gate` (report-level, default-ON; leaf veto A7 default-OFF), evidence-fingerprint,
`deterministic_leaf_checker` — unchanged, now joined by §6.1 as the *pre-grade* deterministic layer.

---

## 7. External adversarial validator (Tier 2 — grounding signal)

### 7.1 Module & seam
New `backend/agents/rlm/external_validator.py`. Panel **invoked once** by the unified continue-policy at
a `FINAL_VAR`-attempt; verdict **persisted** (§7.6) and **consumed (not re-run)** by `run.py::_finalize`
(report stamp + recipe admission) and the hard-stop salvage. Hard-stop with no panel run ⇒ Tier-1 floor
is the honest backstop (`validation: floor_only`). Reuses `sample_completions` + the `leaf_scorer`
evidence-gather; a thin **`validator` role** (§7.4) picks the model.

### 7.2 The panel — separate model, **fail-closed** (resolves B3-separation + convention conflict)
- **`validator_family ≠ executor_family`.** *Family* = model lineage (Sonnet / GPT / grok / Qwen),
  **not** transport: `claude` and `claude-oauth` are the same family. Requires a **new family
  classifier** (`RoleSpec` has only provider/model today) and a **new `RoleSpec.family` field**.
- **Fail-closed construction.** The validator must NOT reuse `grader_transport`'s silent fallback
  (*"we NEVER raise"*). A missing/unconstructable validator transport ⇒ a **`degraded`/`unavailable`
  verdict**, never a silent fall-through to the caller's (executor-family) client. If separation is
  impossible (only one lineage funded), degrade to same-family-different-seed + a loud
  `validator_separation_degraded` warning, and downstream policy treats it as *non-independent*.
- **Adversarial** ("find the fakery"), adversarially misaligned from the generator (anti-collapse).

### 7.3 Verdict = **typed-predicate machine-checks**, not citation-existence (resolves B3)
A panelist locates suspicion and asserts a **typed predicate**; the **harness machine-checks the
predicate against the artifact** — citation existence is necessary-not-sufficient (a fake all-0.0 file
cites `0.0` correctly). Predicate vocabulary (all deterministic, grounded in real artifacts):
- `provenance_present(metric)` — a `provenance.json` entry links the metric to a real output.
- `not_all_constant(result_keys)` — the §6.1 constancy check.
- `gpu_claim_plausible` — VRAM/step evidence consistent with the claimed compute.
- `rerun_agrees(metric)` — the §7.5 perturbed re-run reproduces the claim.

A leaf is vetoed only when a panelist's predicate **machine-verifies as violated**. This makes the
veto rest on the same deterministic predicates as the floor; the LLM only chooses which/where.

### 7.4 `validator` role surface (full file map — resolves the under-scoped finding)
Adding the role touches: `ROLES`/`_SUBROLES` (`role_models.py:91/94`), `RoleSpec` (+`family` field +
classifier), `RoleSelection` (fields + `stamp` + `explicit_subroles` + `fidelity_warnings`),
`resolve_role_models` (`:340`) + `_resolve_subrole` (`:381`), and transport/runtime wiring in `run.py`
+ a fail-closed builder (not `grader_transport`'s fallback path). Tests cover each surface.

### 7.5 Re-run-on-suspicion (cache-bust + isolation — resolves the finding)
On a flagged result-claim only (`OPENRESEARCH_VALIDATOR_RERUN_ON_SUSPICION`, `RunBudget`-gated):
re-materialize **one** cell with a **perturbed fingerprint** (so cell-resume cannot reuse the suspect
artifact) into an **isolated output dir** (never merged into the main aggregate), and compare. Near
wall-clock ⇒ skip + record `recheck_skipped_budget`.

### 7.6 Verdict store (resolves the persistence finding)
Persist `rlm_state/validation_verdict.json` (atomic temp+rename) keyed by **evidence fingerprint**;
`_finalize` consumes **only a verdict whose fingerprint matches the shipped evidence** (a stale verdict
is ignored). `final_report.json.validation` stamps panel model(s), per-leaf predicate verdicts, the
min-aggregated veto set, re-check outcomes, and any `validator_separation_degraded`.

### 7.7 Aggregation — min-aggregation veto, **remedial**
Any panelist whose predicate machine-verifies as violated ⇒ veto (MaxProof pessimistic min-aggregation;
`2606.13473`). **No majority vote** (consensus collapse; `2602.09341`). The veto **feeds §8 (repair),
not an immediate fail.** False-rejects are bounded because the veto requires a machine-verified
predicate, not an LLM opinion.

### 7.8 Cadence
Deterministic floor (Tier 1): **per `run_experiment`** (drives live repair). LLM panel (Tier 2): **at
each `FINAL_VAR`-attempt**, re-entrant into repair — sparing, avoiding the over-frequent-judging decay
(`2605.06635`).

---

## 8. The unified error → repair → re-validate loop (fix-first, fail-honest)

One canonical error-handling abstraction owned by the unified continue-policy (§5.3). **Every
repairable trigger feeds the same loop:** Tier-1 veto (zero/constant-metrics, provenance gap,
evidence_gate), Tier-2 predicate veto, execution/build/contract-guard errors, single OOM.

Pipeline: classify `failure_class` → build a **cited, structured repair directive** → inject via the
`leaf_triage → next-implementer-prompt` channel → re-implement (cache-bust) → re-run → re-validate →
the continue-policy decides accept / repeat / honest-stop.

- **Fix-first.** A trigger refuses `FINAL_VAR` and drives a repair iteration. "Fix" = **real
  re-implementation so the evidence becomes real** — the validator/floor standard is fixed; only the
  code changes. Never relax the bar (the consensus-trap reward-hack the red line forbids).
- **Generous but bounded.** Do not fail on the first errors: up to
  `OPENRESEARCH_REPAIR_MAX_ITERATIONS` (default generous, e.g. 4) within wall-clock, atop the existing
  `OPENRESEARCH_MIN_REPAIR_ITERATIONS` floor. Stop condition = **no-progress** (same `failure_class` +
  unchanged **evidence fingerprint** across attempts), not attempt-count-of-1.
- **Repairable vs. terminal.** Only repairable classes enter the loop (existing
  `_RUN_EXPERIMENT_REPAIRABLE_FAILURES` + new `fabrication_suspected` + validator predicate vetoes).
  Terminal classes — `capacity_exhausted`, `oom_shrink_exhausted`, dead-credit/auth,
  `root_degenerate_loop`, `repair_exhausted` — fail fast; retrying cannot fix them.
- **Honest failure is the floor.** Budget/wall-clock/no-progress exhausted with evidence still
  fake/erroring ⇒ ship `failed`/`degraded` with the cited unfixed reason (extends
  `_hard_stop_with_report`). Never a shipped fake, never a silent pass.

---

## 9. Self-improvement loops — A + B (Tier 3, gated on Tier 1/2 only)

### 9.1 Tier A — in-run grounded repair
The §8 loop *is* Tier A: verdict → structured, cited, taxonomy-categorized repair feedback
(DeepVerifier mold; `2601.15808`) through the existing `leaf_triage` channel. This **upgrades the
*content*** of the loop you already have; no parallel loop.

### 9.2 Tier B — cross-run positive recipes (principle, not champion-copy; resolves two findings)
New `backend/agents/rlm/recipe_library.py` + a `_finalize` admission hook (mirroring
`lesson_distiller.mine_lessons`):
- **Recipe body = a bounded structured pattern + evidence fingerprint** — `{problem_sig: model/dataset/
  task axes; solution_sig: key hyperparameters + `cells.json` shape + a ≤200-char technique summary +
  a code-path pointer; evidence_key}`. **Not** a raw `champion_artifact` snapshot copy (the champion is
  source-only, grade-selected, and detached from heavy evidence — principle-level memory, per
  `2606.04703`).
- **Keyed by paper-class** (via `PAPER_HINTS`/rubric shape), not `arxiv_id` (transfer).
- **Admission gate = Tier-1 + validator, NOT the grade (red line).** Admit only if: report-level
  `evidence_gate` passed **AND** a `success=True` `run_experiment` ledger row exists **AND** the
  validator verdict is clean **AND** a **deterministic `meets_target`** predicate (measured metric vs
  the paper's claimed target) holds. **Never** the champion's `median_score` rank, never agent prose
  (poison-proof; OWASP ASI06).
- **Inject top-1** by problem-sig similarity at the implementer-prompt site, **hard-capped (≤1–2)** +
  novelty-dedup + staleness retirement (skill-shadowing −21%; `2605.24050`). Gated
  `OPENRESEARCH_POSITIVE_RECIPES`.

---

## 10. Cheap win — config-file run spec

Add `--run-spec <path.json>` to the `reproduce` subparser; `cli.py` loads it early into the same
`os.environ["OPENRESEARCH_*"]` sink it already uses (`cli.py:1566-1697`), explicit flags overriding.
The GCP launcher `scp`s one spec instead of the 12-var whitelist (`gcp_sdar_preflight.sh:304`) + the
staged guidance file. Low risk — the env layer is already the single sink; multi-line guidance lives
in the JSON.

---

## 11. Invariants + guard tests — interfaces first (resolves B5)

### 11.1 Define the small testable interfaces first, then test the seam
| Interface (new) | Guard test (seam-scoped) |
|---|---|
| `ValidatorVerdict` (+ `validation_verdict.json` store, fingerprint-keyed) | `_finalize` ignores a fingerprint-mismatched verdict; stale-verdict unit test |
| `RoleSpec.family` + family classifier | family-collision ⇒ degraded mode + warning, never silent same-family |
| `RepairState` (distinct refusal class, evidence-fingerprint progress) | a floor/validator refusal never increments the `root_degenerate_loop` counter; stuck-repair ⇒ `repair_exhausted` |
| `LedgerRecord` + per-primitive projection | sentinel test: no paper-text canary in any ledger/cache file; ledger never records `ok` on timeout/raised |
| `recipe_admission(predicates)` | admission reads only Tier-1 + validator inputs; static import check that `recipe_library` + repair-acceptance never read grade fields except to copy into the report |
| zero/constant-metrics veto | fixture suite: v6 flat-all-zero + GPU-claim + no-provenance **fires**; flat-0 **with** provenance (legit baseline) does **not**; converged-near-0 with provenance does **not** |

The over-broad "no path consults `overall_score`" is replaced by the scoped static-import + admission
unit tests above.

### 11.2 "Default-OFF == baseline" (scoped)
Applies to **new flags only**. Compatibility tests explicitly list the existing default-ON rails
(`REPROLAB_FINALIZE_REGRADE`, `REPROLAB_LEAF_TRIAGE`, `OPENRESEARCH_METRIC_PROVENANCE`, report-level
`_apply_evidence_gate`) as the baseline, and assert each new flag OFF ⇒ no new behavior.

### 11.3 Flag naming
New flags use canonical `OPENRESEARCH_*` (CLAUDE.md). Note the bridge: `config.py::_apply_legacy_env_aliases`
maps legacy `REPROLAB_*` → `OPENRESEARCH_*` at import; existing rails still read `REPROLAB_*`. The spec
documents both; no migration of existing names.

---

## 12. Module / file map

**New:** `lifecycle_ledger.py` · `zero_metrics_detection.py` · `external_validator.py` ·
`recipe_library.py` · run-spec loader in `cli.py`.

**Touched:** `forced_iteration.py` (unified policy + distinct repair-refusal class) · `run.py`
(ledger write, validator invocation at `FINAL_VAR`-attempt, `_finalize` verdict-consume + recipe
admission, `_hard_stop_with_report` cited reason) · `binding.py` (record-only ledger sidecar at
`:383`, atomic, post-validation) · `role_models.py` (`validator` role + `RoleSpec.family` + classifier
+ all §7.4 surfaces) · `primitives.py:6465` (zero/constant-metrics guard) · `primitive_cache.py`
(verify/extend resume coverage — **not** a rewrite) · `root_progress.py` (stage-specific cited nudges)
· `leaf_triage.py` (validator-sourced structured repair content).

**Reused (unchanged):** `evidence_gate.py` · `leaf_scorer.py` · `grader_transport.sample_completions`
· `deterministic_leaf_checker.py` · `provenance` (`report.py` manifest) · `cell_scheduler.py`
(fingerprint resume) · `lesson_distiller.py` (pattern). **Explicitly not reused for the validator:**
`grader_transport.build_transport_client`'s silent-fallback path (§7.2 fail-closed).

---

## 13. Build sequence (umbrella spec; per-phase implementation sub-plans; each flag-gated + guard-tested + ≥3 paired SDAR runs before any default-flip)

- **P0 — cheap + fail-honest (no repair loop yet):** config run-spec · zero/constant-metrics veto as a
  **report-only / fail-honest** degrade (`fabrication_suspected` → honest `degraded`, no repair
  expectation) · run the $2 `THRESHOLD=16` precondition experiment. *Would have flagged `v6`.*
- **P1 — evidence ledger (record/provenance)** + **verify** `primitive_cache` cross-restart resume.
- **P2 — deterministic floor consolidation + validator as OFFLINE report-stamping only** (no
  continue-policy dependency yet; fail-closed transport + family classifier + verdict store).
- **P3 — unified continue-policy + the §8 fix-first loop + minimal degenerate-recovery.** Here the P0
  veto + P2 validator **upgrade from fail-honest to fix-first** (wired into the repair loop).
- **P4 — Tier B cross-run recipes.**
- **P5 — recovery-aware detector polish** (informed by P0's experiment) + resume polish.

One umbrella design (your "coordinated redesign"); the writing-plans step emits **one implementation
sub-plan per phase**, each with its own acceptance tests. (Resolves "too large for one unit" without
splitting the *design*.)

---

## 14. Testing strategy

Hermetic unit tests for every Tier-1 predicate, the redaction sentinel, the validator aggregation
(stubbed `sample_completions` → veto set), the fail-closed transport degrade, recipe admission, the
repair-refusal/degenerate separation, and the scoped default-OFF contract. The v6 flat-all-zero
`metrics.json` becomes a regression fixture. A/B via `scripts/ab_compare.py` + `experiment_arm` per
new flag. Integration: an end-to-end `--sandbox local` dry-run with a planted flat-all-zero,
no-provenance cell asserting the §8 loop repairs-then-honest-fails within budget.

---

## 15. Risks + open questions

1. **Validator model availability (resolved).** Prefer Foundry/grok or a keyed model for the panel; if
   only one lineage is funded, degrade to same-family-different-seed + loud warning, treated as
   non-independent — never block, never silent fallback.
2. **Phasing vs. atomic (resolved).** One umbrella spec, phased P0→P5 delivery + per-phase sub-plans.
3. **`primitive_cache` cross-restart resume (verify in P1).** It exists + is on-disk + content-hash
   keyed; P1 confirms it's enabled for SDAR and stable across a same-`--project-id` relaunch before we
   claim spot-resume is "solved."
4. **Provenance is the load-bearing floor signal** (no raw series exists). If `provenance.json` is
   absent on a given route, the flat-zero veto degrades to *escalate-to-validator* rather than hard-veto
   (avoids false-positive); §6.3's optional history contract is the long-term upgrade.
5. **Re-run-on-suspicion GPU budget** must respect `RunBudget`; near wall-clock ⇒ `recheck_skipped_budget`.
6. **Recipe paper-class granularity** — start from `PAPER_HINTS` classes; treat as a tunable (too fine =
   no transfer, too coarse = negative transfer).
7. **Validator cost** — bounded by cadence (panel only at `FINAL_VAR`-attempt) + `OPENRESEARCH_VALIDATOR_PANEL_N` (small default).

---

## 16. Research grounding (mid-2026 literature → design decisions)

All IDs verified findable on arXiv (2026-06-20).

| Decision | Evidence |
|---|---|
| External signal load-bearing; intrinsic self-correction fails | `2310.01798`, `2406.01297`, `2601.00828`, FlipFlop `2311.08596`→`2606.16011` |
| Verifier-guided repair as the loop (Tier A) | DeepVerifier `2601.15808`; co-evolving coder/tester `2506.03136` |
| Validator model-separate (family ≠ executor) | self-preference `2604.22891` (β≤0.307), `2506.02592` |
| Adversarial + anti-collapse, not collaborative | CoVerRL `2603.17775`; PROClaim `2603.28488` |
| Min-aggregation veto, not majority vote | AgentAuditor `2602.09341`; MaxProof `2606.13473` |
| Typed-predicate machine-check, not citation-existence | "Cited but Not Verified" `2605.06635`; "LLMs Gaming Verifiers" `2604.15149` |
| Sparse panel cadence | `2605.06635` (79%→17% under fan-out) |
| Red line: never self-improve against the proxy grade | ICLR 2026 RSI "Reward Hacking in Self-Improving Code Agents" (73.8%/46.8%); SpecBench `2605.21384`; `2605.02964`; Alignment Tipping `2510.04860`; Agent Drift `2601.04170` |
| Tier C deferred | DGM `2505.22954`; AlphaEvolve `2506.13131`; Live-SWE-agent `2511.13646`; Hyperagents `2603.19461`; SICA `2504.15228` |
| Recipes: principles not traces, gated, capped | `2606.04703`; CBR R&D-Agent `2606.05250`; skill-shadowing `2605.24050`; surveys `2603.07670`, `2504.06943` |
| Memory poisoning ⇒ evidence-gated admission only | OWASP ASI06 / Microsoft taxonomy v2.0 (2026-06-04) |

---

## 17. Pointers

- Handoff (superseded): `docs/runbooks/2026-06-20-sdar-harness-refactor-and-external-validation-handoff.md`
- Canonical SDAR run: `docs/runbooks/2026-05-23-sdar-baseline-handoff.md`, `2026-06-16-sdar-on-gcp-a100-vm.md`
- Anti-fabrication today: `evidence_gate.py`, `stub_detection.py`, `leaf_scorer.py`,
  `deterministic_leaf_checker.py`, `cell_matrix.py`, `gpu_cell_runner.py`, `primitive_cache.py`,
  `champion_artifact.py`
- Reliability: `forced_iteration.py`, `root_progress.py`, `root_validation.py`, `run.py`
- Per-role / transport: `role_models.py`, `grader_transport.py`, `backend/agents/runtime/*`
- Related specs: `2026-06-16-grader-fidelity-and-harness-remediation-design.md`,
  `2026-06-17-bes-evidence-first-and-conversion-remediation-design.md`

---

## 18. Review-resolution log (Codex adversarial review, 2026-06-20)

All claims were verified against source before resolution. **B = blocker, M = major, m = minor.**

| # | Finding | Resolution |
|---|---|---|
| B1 | Unified policy can trip `root_degenerate_loop` mid-repair | §5.3 distinct repair-refusal class, evidence-fingerprint progress, excluded from degenerate signature; stuck ⇒ `repair_exhausted` |
| B2 | Ledger-resume infeasible (`run_experiment`/`build_environment` excluded from caching) | §4 ledger is **record-only** (audit/provenance); §5.4 resume **reuses existing** `primitive_cache` + cell-fingerprint + Docker cache |
| B3 | "Cite + machine-check" insufficient (fake 0.0 cites correctly) | §7.3 **typed-predicate** machine-checks (provenance/constancy/gpu-plausibility/rerun); citation necessary-not-sufficient |
| B4 | Zero-metrics veto no-ops on real flat-no-history shape | §6.1 flat+nested normalization; signal = all-zero/constant + GPU-claim + **provenance absent/inconsistent**; legit-0 has provenance |
| B5 | §11 invariants unenforceable (interfaces absent) | §11.1 define small interfaces first, seam-scoped tests + static-import backstop |
| B6 | P0 veto before P3 repair loop | §13 P0 veto is **fail-honest report-only**; upgrades to fix-first in P3 |
| M | "beat target" / champion select by grade conflicts with red line | §3.1 red line scoped to NEW Tier-3 selection; §9.2 admission via Tier-1 `meets_target`, not champion grade |
| M | Degenerate recovery deferred to P5 | §5.1/§13 minimal integration pulled into P3 |
| M | validator role under-scoped | §7.4 full surface + new `RoleSpec.family` |
| M | wrap_primitive timeout → misleading ledger | §4.2 write post-validation, atomic, never `ok` on timeout/raised |
| M | inputs redaction underspecified | §4.3 per-primitive projections + sentinel canary test |
| M | wrapper memoization skips side-effects | §4.1 record-only; memoization stays in `primitive_cache` |
| M | family separation not implementable | §7.2/§7.4 new family field + classifier |
| M | validator silent fallback | §7.2 **fail-closed**; degraded/unavailable verdict, not grader fallback |
| M | re-run lacks cache-bust | §7.5 perturbed fingerprint + isolated output dir |
| M | verdict persistence vague | §7.6 `validation_verdict.json`, fingerprint-keyed, atomic |
| M | recompute-from-raw aspirational | §6.2 removed (no raw series); replaced by provenance/constancy |
| M | per_model not universal | §6.1 normalize flat + nested |
| M | "default-OFF==today" too strong | §11.2 scoped to new flags; existing default-ON rails listed |
| M | too large for one unit | §13 umbrella + per-phase sub-plans |
| M | recipes riskier than negatives / champion-detached | §9.2 structured principle + evidence fingerprint, gated admission |
| m | `_load_metrics` call vs def lines | §1.1 def 267 / call 913 |
| m | two `aggregate_cell_metrics` sites | §1.1 5663 + 5450 |
| m | flag naming `OPENRESEARCH_*` vs `REPROLAB_*` | §11.3 documented bridge |
