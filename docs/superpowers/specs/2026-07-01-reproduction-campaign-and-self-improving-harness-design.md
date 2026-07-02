# Reproduction Campaign + Self-Improving Harness — Design

> **Doc status:** Design spec · **v2 (2026-07-01)** · brainstormed + approved
> section-by-section (terminal contract, driver order, autonomy, self-edit scope,
> width, campaign home, validation, deliverable all operator-locked), then
> **adversarially reviewed by Codex and reworked — all 16 findings resolved (§20)**.
> Builds directly on
> `2026-07-01-paper-agnostic-multicloud-reproduction-and-self-improvement-design.md`
> (Phases 1a–1f, code-complete) and the Phase-1f cutover runbook
> `docs/runbooks/2026-07-01-sdar-unified-run-cutover.md`.

## 1. Context & goal

The frontier loop this repo exists for is:

> read paper → plan reproduction → write implementation → run experiment →
> analyze logs → fix code → run again → compare with paper → repeat until the
> paper is reproduced — and let the harness learn from every attempt.

Everything below the attempt boundary exists today. The RLM root loop runs one
attempt end-to-end with forced iteration, bounded in-run repair, fabrication
guards, and honest reporting. The single-attempt cloud lifecycle
(`ReproductionRun`, Phases 1a–1f) is code-complete behind
`OPENRESEARCH_UNIFIED_RUN`. Cross-attempt carry-over exists as flag-gated rails
(`best_attempt.py`, `prior_attempt_evidence.py`, `attempt_isolation.py`,
`champion_artifact.py`). Cross-run learning exists as advisory memory
(`failure_attribution.py`, `experience_memory.py`, `lesson_distiller.py`,
`recipe_library.py`, `held_out_gate.py`).

**What does not exist is the loop itself.** Today the "repeat until success"
brain is a human (or a Claude session) babysitting `scripts/loops/
kill_and_restart.sh` — whose own header says the attempt cap is "enforced by
the caller (the autonomous loop in the active Claude Code session)." This spec
replaces that session with a deterministic, checkpointed, budget-bounded
campaign controller, and — as its second phase — closes the self-improvement
loop with a staged, evidence-gated harness self-edit tier.

## 2. Locked decisions (the design contract)

| # | Decision | Choice |
|---|---|---|
| 1 | Terminal contract | **Honest terminal set**: `REPRODUCED` / `CONTRADICTED` / `INFEASIBLE` / `EXHAUSTED{stop_reason}`; retry only while the failure class is repairable AND budget remains AND the next plan is novel; plateau stall-stopper included. |
| 2 | Campaign target | **The full paper's rubric.** Scope-narrowing (triage DOWN_SCOPE) is an attempt-level *tactic*; the campaign schedules re-expansion once narrow scope is green. `REPRODUCED` = full in-budget rubric target met, remaining exclusions declared and verified. |
| 3 | Execution drivers | **Both, behind one seam**: `LiveCliDriver` (today's proven `backend.cli reproduce` path) is the default; `UnifiedRunDriver` (`unified_run.build_reproduction_run`) behind `OPENRESEARCH_UNIFIED_RUN`. Paired-driver campaigns are the *vehicle for conducting* the Phase-1f A/B under operator approval (§16, F8). |
| 4 | Autonomy | **Dynamic**: `unattended` (default; overnight OK within budget) and `checkpoint` (pause-and-notify after each DECIDE) — switchable mid-campaign via the campaign steering channel (§10.6). |
| 5 | Self-improvement | **Both tiers in scope, design now, implement in order**: Phase B = campaign + advisory memory wired into every attempt; Phase C = staged harness self-edit (whitelisted surface, frozen evaluator tier, dedicated `HarnessEditGate`, shadow→canary→operator-confirmed default). |
| 6 | Attempt width | **Budget-gated width** (`K` parallel candidate attempts on weak-history papers when budget and capacity allow); sequential otherwise. |
| 7 | Campaign home | **Checkpoint+resume by design**; runnable identically from laptop `nohup` or the CPU orchestrator VM. No in-cluster service (north-star unchanged). |
| 8 | Validation | Cheap fast paper first (loop mechanics), then **SDAR smallest-two on GCP, ≥3 paired runs** (the repo's mandated discipline). Campaign validation budget ≈ $300. |
| 9 | Deliverable | **The final report** (champion attempt's `final_report.{json,md}` + `campaign_report.md`). No user-facing "suggested next experiment" section; improvement proposals only fuel the next attempt's directives. |
| 10 | UNDERSTAND hardening | A thorough campaign-level understanding stage runs before any GPU spend — double-extraction cross-verification + deterministic contract lint — to eliminate transcription-class trivial bugs. |
| 11 | Red line (unchanged) | **Evidence, not grade.** The campaign layer adds NO LLM judgment of its own: every decision (retry, terminal, selection, memory admission, self-edit promotion) is a deterministic function of recorded artifacts. The recorded rubric grade (LLM-produced upstream) is one such artifact, but it is trusted only inside its guard envelope — fabrication guards + validator + rubric-integrity must all be clean before a grade can contribute to a terminal. An LLM never owns termination, budget, or blame. |

## 3. Research grounding (July 2026 SOTA — the levers this design adopts)

Each mechanism below is adopted *because* it has published evidence; the design
does not adopt anything that exists only as a vendor claim.

| Mechanism adopted | Evidence | Where it lands here |
|---|---|---|
| Deterministic controller owns the loop; LLM never owns termination/grading | 2026 consensus (plan-execute vs ReAct); MLR-Bench ~80% fabricated results under error pressure (2505.19955) | `ReproductionCampaign` (§5) |
| Clean-context retries seeded only with structured artifacts | CCRM (2605.08563): contaminated retries cascade 7.1×; clean restarts resolve ~21% more at equal budget | Directive synthesis contract (§9) + forced quarantine of incomplete attempts (§7, F6) |
| Forced continuation beats quit-early scaffolds | PaperBench: same o1, 13.2% (Basic) vs 24.4% (Iterative) (2504.01848) | Existing `forced_iteration` per attempt; campaign extends it across attempts |
| Fine-grained binary criteria as the repair driver, not scalar grades | RePro 62.6% Code-Dev SOTA (2508.16671); MLE-STAR ablation-localized refinement (2506.15692) | Leaf-level deltas in directives (§9); existing `leaf_triage` |
| Harness-owned evaluation the agent cannot write | RewardHackingAgents (2603.11337): agents attempt evaluator tampering in ~50% of episodes; locked evaluators → 0% compromise at ~2% overhead. AIRA2 hidden/decoupled eval split worth +13–18 pts (2603.26499) | Evaluator lockdown + rubric hash pin + canary leaf (§10.4) |
| Parallel candidates in isolated workspaces, deterministically selected | CAID +26.7pp absolute on PaperBench (2603.21489); AIRA2 async width | Budget-gated attempt width (§8.3) |
| Retry allocation as exploration/exploitation over lineages | REx (NeurIPS 2024): bandit over which prior program to refine beats breadth/depth-first at equal calls | Lineage policy v1 rule table + bandit upgrade path (§8.2) |
| Kill doomed runs early on deterministic signals | LLMZero (2606.18388): curve-vs-best comparator killed 62% of nodes, saved 40–60% GPU; BAGEN (2606.00198) 28–64% token savings | `doomed_run_comparator` in WATCH, flag-gated (§10.3) |
| Fixability-classified retries with per-class caps | OpenSkill (2606.06741) SELF-FIXABLE vs NEEDS-KNOWLEDGE routing; LLM blame assignment tops out at 53.5%/14.2% (Who&When 2505.00212) | DECIDE policy keyed on `FailureAttribution`, never LLM opinion (§8.1) |
| Tiered, distilled, recurrence-gated memory; never raw-log retention or LLM rewrite consolidation | ML-Master 2.0 cognitive accumulation 56.4% MLE-bench SOTA (2601.10402); STALE (2605.06527): <10% of frameworks detect invalidated memories; "Useful Memories Become Faulty" (2605.12978) | DISTILL stage guarantees existing miners run every attempt (§11.1) |
| Regression + held-out gated harness self-edit, staged promotion, frozen evaluator | Self-Harness (2606.09498) +14–21pp Terminal-Bench-2; DGM's marker-deletion incident (sakana.ai/dgm); SIA co-evolutionary Goodhart (2605.27276) | Self-edit tier (§11.2), frozen tier constitutional, dedicated `HarnessEditGate` |
| Maximize what the agent is given (repo-first, pinned envs, warm caches) | SocSci-Repro 93.4% with materials (2606.11447) vs PaperBench ~21–27% from paper alone | UNDERSTAND stage turns on repo-first + contract (§6) |
| Guard against confirmatory specification search | SocSci-Repro: prompt nudges induce tuning-toward-paper-numbers | §10.5 |
| Verifier quality bounds retry value; static verifiers decay | Inference Scaling fLaws (2411.17501); The Verification Horizon (2606.26300); CapCode capped-canary checks (2606.07379) | Canary leaf + evidence-predicate investment ordering (§10.4, §14) |

## 4. Current state (grounded inventory)

What the campaign composes, all existing:

- **Single attempt:** RLM root loop (`run.py`), 18 primitives, `forced_iteration`
  (+ degenerate-loop detector + `lifecycle_driver`), bounded fix-first repair
  (`OPENRESEARCH_REPAIR_MAX_ITERATIONS`, evidence-fingerprint-keyed), fabrication
  guard suite, `EvidenceAudit`, external validator, two-axis verdict,
  finalize-regrade, hard-stop salvage.
- **Cloud lifecycle:** `ReproductionRun` (checkpointed, fail-soft, budget-gated
  `ACQUIRE_GPU`, `recover()`), `unified_run.build_reproduction_run` (Phase 1f
  composition root, flag-gated, not yet in the live path). NOTE (F7):
  `ReproductionRun` checkpoints its state but exposes **no load/re-enter API**
  today — campaign resume semantics for unified attempts are therefore
  assess-from-disk, not re-attach (§5, Resume).
- **Cross-attempt rails (flag-gated today):** `attempt_isolation` (archive prior
  attempt; warm-retry detect — a hole the campaign must close, F6),
  `best_attempt` (seed code + guidance + floored target — **score-ranked**, F5),
  `prior_attempt_evidence` (measured per-cell results forward),
  `champion_artifact` (best-evidence snapshot — **median-score-keyed**, F5).
- **Memory substrate (flag-gated today):** `failure_capsule`,
  `failure_attribution` (stable signature + infra/method scope),
  `experience_memory` (global-infra store + per-paper wrap), `lesson_distiller`
  (recurrence-gated), `recipe_library` (evidence-gated), `held_out_gate`
  (EvidenceVector non-regression admission for **advisory lessons only** — not
  a mechanics gate, F11).
- **Understanding substrate:** `resolving_parser`, `semantic_contract.py`
  (`OPENRESEARCH_REPRO_CONTRACT`, source-linked fields), claim map, `rubric_gen`,
  `paper_hints`, `paper_invariants`, repo-first (#62).
- **Ops:** `--run-spec` env-sink loading (applies `OPENRESEARCH_*`/`REPROLAB_*`
  keys + `models` + `baseline_extra_guidance` — exact key names required, F15),
  `ab_compare.py`, `--project-id` override with ingest-on-noncanonical, SSE
  bridge, leaderboard best-attempt resolution.
- **Money reality (F2, F3):** CLI `--max-usd` caps **LLM/SDK spend**; GPU spend
  is governed separately (`--max-run-gpu-usd`, `OPENRESEARCH_MAX_RUN_GPU_USD`,
  GPU-hours); the GCP VM control-plane ceiling (`max-run-duration`) defaults to
  28h if not set explicitly. Under `stage_on_gpu` (the GCP on-demand A100
  default), **GPU billing starts at `provision_cpu`**, not `acquire_gpu` — the
  Phase-1 spec's §5.5 honesty note.
- **The gap:** `scripts/loops/kill_and_restart.sh` + a human/Claude session as
  the outer brain. No component decides retry/stop, synthesizes the next
  attempt, or guarantees memory distillation when a run dies at the wall clock.

## 5. Architecture — `ReproductionCampaign`

```
ReproductionCampaign  (NEW — deterministic outer state machine)
  INIT → UNDERSTAND → ┌─────────────── per-attempt loop ────────────────┐ → TERMINAL
                      │ PLAN_ATTEMPT → LAUNCH → AWAIT → ASSESS → DISTILL → DECIDE │
                      └──────────────────────────────────────────────────┘
                               │ AttemptDriver seam (NEW)
                               ├─ LiveCliDriver (default)      — backend.cli reproduce (proven path)
                               └─ UnifiedRunDriver (flag)      — build_reproduction_run().run()
Layer B (RLM root loop, primitives, guards) and ReproductionRun are unchanged underneath.
```

- **State (fail-CLOSED — the deliberate inversion of `ReproductionRun`'s
  fail-soft checkpoint, F1):** `runs/<project>/campaign/campaign.json` +
  `attempts.jsonl` are a **spend ledger, not an observability aid**. All writes
  are atomic (tempfile + `fsync` + `os.replace`). **Write-ahead invariant: no
  LAUNCH without a durably recorded intent row** — PLAN_ATTEMPT appends
  `{attempt_n, directives_sha256, envelope, status:"launched"}` and fsyncs
  BEFORE `driver.launch`; ASSESS updates the row with the assessment; terminal
  decisions are recorded before being acted on. A failed ledger write **halts
  the campaign** (no new attempt is ever launched on an unwritable ledger; the
  in-flight attempt, if any, finishes under its own cloud-side ceilings and is
  assessed on the next resume). `ReproductionRun`'s own internal checkpoint
  stays fail-soft — the two files have different jobs.
- **Resume protocol (F7):** `campaign.json.in_flight` records
  `{attempt_n, driver, run_dir, pid?, lease_ref?, launched_at}`. On `--resume`:
  no in-flight → re-enter at the checkpointed state. In-flight + liveness probe
  positive (pid alive / `demo_status.json` running) → re-attach and AWAIT.
  In-flight + dead → **assess-from-disk**: the existing hard-stop salvage means
  a killed live-path attempt still has an honest report; a dead unified attempt
  is assessed from its synced artifacts + `provider.recover()` semantics.
  `ReproductionRun` has no re-enter API (v1 limitation, stated honestly);
  adding one is north-star work. Transition replay is idempotent: `attempts.jsonl`
  rows are keyed by `attempt_n`; a resumed ASSESS overwrites nothing — it
  appends a superseding row referencing the same `attempt_n` (last-writer-wins
  on read; both rows retained for audit). No double-spend: LAUNCH refuses to
  fire for an `attempt_n` that already has a `status:"launched"` row unless the
  liveness probe says dead AND the quarantine step (§7) has archived its state.
- **Never raises:** `run()` wraps everything; an orchestration bug degrades to
  `EXHAUSTED{stop_reason="campaign_error:<...>"}` with the champion-so-far
  shipped — except the unwritable-ledger halt above, which stops *before*
  money moves rather than after.
- **Home:** a plain process — `nohup python -m backend.cli campaign ...` on the
  laptop or the CPU orchestrator VM. No daemon, no service (non-goal §17).

## 6. UNDERSTAND — the anti-trivial-bug stage (runs once, CPU-only)

Runs before any GPU dollar; produces the paper-level artifacts every attempt
cites. All steps are library calls (parser, contract extractor, rubric gen,
repo resolver) — no full RLM loop, no new agent surface.

1. **Repo-first resolution** — `OPENRESEARCH_USE_AUTHOR_REPO=1` in the campaign
   profile: resolve + clone the official repo when linked (pristine,
   SHA-pinned). The single biggest published lever (93% with materials vs
   ~21–27% from paper alone).
2. **Semantic contract** — build/refresh the `SemanticReproductionContract`
   (`OPENRESEARCH_REPRO_CONTRACT=1`): resource identities, capability profile,
   metric contracts (unit / direction / valid range / split).
3. **Double-extraction cross-verification** — hyperparameters, invariants, and
   protocol are extracted **twice independently** (prompt-variant passes);
   the two structured outputs are **deterministically diffed** (exact compare
   on numerics/enums, set-compare on identity lists). Disagreements trigger one
   targeted third pass constrained to the disagreed fields; still-unresolved
   fields are recorded in `understanding.json.unresolved[]` and injected into
   attempt-1 directives as explicit warnings.
4. **Deterministic contract lint** — dataset/model identifiers resolve against
   HF/catalog/`dataset_recipes`; metric ranges sane and direction-consistent;
   scope grid enumerable; invariant regexes compile; claim-map ↔ rubric leaves
   join. Pure checks, no LLM.
5. **Understanding gate — blocking authority is tiered (F9):** only
   **deterministically confirmed** gaps may block attempt 1: a lint failure on
   a *source-span-grounded, cross-verified* field (both extraction passes
   agree AND the field carries a paper source span) or a *probe-confirmed*
   asset gap (the Phase-1b reachability probe returns gated/missing). Blocking
   routes to `INFEASIBLE` (honest plan-only report, §12) instead of burning
   GPU on a garbage plan. LLM-only extracted fields (no source span, or
   unresolved disagreement) are **advisory-only — they can never block**; they
   ride into directives as warnings.

Output: `campaign/understanding.json`, sha256-stamped into `campaign.json`.
Paper-level artifacts (`generated_rubric.json`, contract, repo spec) remain
un-archived across attempts — consistent with `attempt_isolation`'s existing
stable-artifact list.

## 7. The `AttemptDriver` seam

```python
class AttemptDriver(ABC):
    def launch(self, directives: AttemptDirectives) -> AttemptHandle: ...
    def await_result(self, handle: AttemptHandle) -> AttemptRawResult: ...
    def abort(self, handle: AttemptHandle, *, reason: str) -> None: ...
```

- **`LiveCliDriver` (default):** spawns `backend.cli reproduce` on the SAME
  project id, passing the **campaign run-spec** via the existing `--run-spec`
  mechanism. The checked-in canonical profile (`configs/campaign_run_spec.json`)
  uses **exact env-sink key names** (F15) — e.g.
  `OPENRESEARCH_SEED_BEST_ATTEMPT`, `OPENRESEARCH_TARGET_BEST_FLOOR`,
  `OPENRESEARCH_PRIOR_ATTEMPT_EVIDENCE`, `OPENRESEARCH_FAILURE_CAPSULES`,
  `OPENRESEARCH_NEGATIVE_LESSONS`, `OPENRESEARCH_EXPERIENCE_MEMORY`,
  `OPENRESEARCH_EVIDENCE_GATE`, `OPENRESEARCH_REPRO_CONTRACT`,
  `OPENRESEARCH_USE_AUTHOR_REPO`, `OPENRESEARCH_REUSE_RUBRIC` — and campaign
  INIT **validates by round-trip**: after `_load_run_spec` application, every
  intended key must be present in the env sink, else fail-fast at $0
  (`campaign_error:run_spec_key_rejected`).
- **Clean-context enforcement (F6):** before every launch the driver runs a
  **force-quarantine** step: if the run dir holds attempt residue (`code/`
  etc.) without the campaign having recorded a completed assessment for it,
  the residue is explicitly archived to `attempts/<ts>_incomplete/` (a new
  explicit-archive entry point on `attempt_isolation` — the existing
  warm-retry heuristic, which would silently reuse `code/`, is **never
  exercised under a campaign**).
- **`UnifiedRunDriver`:** `build_reproduction_run(...).run()` under
  `OPENRESEARCH_UNIFIED_RUN=1`. Same directives, same assessment; resume =
  assess-from-disk (§5).
- **`AttemptRawResult`** is a *pointer set* (run dir, report path, ledger
  paths, exit condition), not parsed content — parsing/judgment belongs to
  ASSESS so drivers stay dumb and swappable.
- **Paired mode and Phase 1f (F8):** `--driver paired` alternates drivers on
  matched directives and feeds `ab_compare.py`. This **conducts** the Phase-1f
  step-3 A/B — it does not declare it satisfied: paired campaigns that spend
  GPU require explicit operator initiation (checkpoint mode or the CLI
  invocation itself), the cutover runbook's ≥3-paired parity judgment remains
  an operator sign-off, and `LiveCliDriver` stays the default until that
  sign-off. No default-flip is automated by this spec.

## 8. ASSESS + DECIDE — the deterministic retry policy

### 8.1 AttemptAssessment (pure read, no LLM)

One row per attempt appended to `attempts.jsonl`:
`{attempt_n, driver, directives_sha256, final_report{score, target, meets_target,
verdicts, stop_reason, exclusions}, evidence_vector (EvidenceAudit predicates),
guard_flags (zero-metrics / stub / eval-provenance / env-liveness /
no-learning-signal), validator{status, fingerprint, fresh}, leaf_vector_ref
(per-leaf pass/fail snapshot), failure_attribution{class, signature, scope,
confidence}, cost{llm_usd, gpu_usd, gpu_hours, wall_s}, rubric_sha256_check}`.
Prose is never read. Recorded grades enter the row as artifacts whose trust is
established by the accompanying guard/validator predicates (locked decision
11) — a grade with any guard tripped is quarantined, never consumed by DECIDE
rule 1. **Validator absence is quarantine, not clearance (F4):** a validator
status of `unavailable`/missing/stale-fingerprint quarantines the grade for
terminal purposes exactly like a tripped guard (the attempt remains usable for
lineage seeding and repair planning). A missing/corrupt report assesses as
`failure_class="report_missing"` (repairable once, then unrepairable).

### 8.2 DECIDE — rule table (v1, deterministic; bandit is a documented upgrade)

Evaluated in order:

1. `REPRODUCED` — `meets_target` on the full in-scope rubric AND every
   fabrication guard clean AND **`validator.status == "clean"` on a fresh
   evidence fingerprint** (F4) AND rubric integrity intact (hash match) AND
   the evidence audit run-level clean. If the campaign is configured without a
   reachable validator panel, INIT says so up front and `REPRODUCED` is
   unreachable in unattended mode — never silently downgraded (fail-closed,
   mirroring `build_validator_client`).
2. `CONTRADICTED` — two-axis verdict says implementation-faithful +
   replication-contradicted, with guards clean + validator clean, on **≥2
   attempts from different lineages/seeds**. One contradicting run is never
   terminal.
3. `INFEASIBLE` — a deterministically confirmed blocking gap (probe-confirmed
   gated/missing asset, unwritten-adapter class, compute ≫ budget even
   minimal). Produces the campaign-level plan-only report (§12, F14).
4. `EXHAUSTED` — budget floor breached on ANY meter (`remaining <
   conservative next-attempt estimate`), or `attempt_n == max_attempts`, or
   plateau (K consecutive attempts with no EvidenceVector improvement AND no
   new failure signature; K = `OPENRESEARCH_CAMPAIGN_PLATEAU_K`, default 2),
   or novelty exhausted (§8.4), or failure class unrepairable / per-class cap
   hit (same infra signature twice → must route through its distilled infra
   lesson or stop).
5. Else **CONTINUE** → lineage + scope + width selection:
   - **Champion selection is campaign-owned and guard-filtered (F5):**
     `select_champion(assessments)` first filters to attempts whose guard
     envelope is fully clean (guards + validator + rubric integrity), then
     ranks by EvidenceVector predicate count, then measured `meets_target`
     distance, then leaf-pass count. The score-ranked rails (`best_attempt`'s
     `overall_score` scan, `champion_artifact`'s median ranking) **never
     choose the campaign seed**: the campaign passes an explicit seed pointer
     (the selected attempt's archived `code/`) into the directive, and the
     driver stages it into the `code/_best_attempt/` slot itself. The floored
     target is likewise computed from guard-clean attempts only.
   - **Lineage (v1 rules):** attempt 1 = fresh (from UNDERSTAND artifacts).
     Attempt N+1 = champion-seeded, unless the champion lineage produced no
     evidence improvement twice running → switch to best-runner-up seed, then
     fresh-angle. (REx-style Thompson sampling over these three arms is the
     documented upgrade; v1 stays a rule table — 6-attempt campaigns are too
     short for a bandit to matter.)
   - **Scope ladder:** adopt triage DOWN_SCOPE for the attempt when forced;
     record `scope_delta_to_full`; once the narrow scope is green, the next
     directive re-expands one rung toward the full grid. The campaign target
     never shrinks (locked decision 2).
   - **Width (§8.3)** when eligible.

### 8.3 Budget-gated width

`OPENRESEARCH_CAMPAIGN_WIDTH=K>1` and history-weak (no attempt yet ≥
`OPENRESEARCH_CAMPAIGN_WIDTH_SKIP_SCORE`, default 0.5 — mirrors
`OPENRESEARCH_BES_ADAPTIVE_SKIP_SCORE`) and
`remaining_budget ≥ K × conservative estimate` on EVERY meter and capacity
supports it → launch K candidate attempts with per-candidate angle directives
in isolated lineages. **Lineage minting (F16):** the campaign mints child
project ids itself and passes them via the existing `--project-id` override
(`<campaign_project>_w<k>` — `register_project(project_id_override=...)`
already ingests non-canonical ids); `--project-id-suffix` remains
`batch_reproduce.py` machinery and is not claimed here. Assess all K, adopt
the guard-filtered EvidenceVector-best as champion (§8.2); the others'
capsules/lessons are distilled, artifacts retained for runner-up seeding. On
single-VM GCP, candidates run sequentially (still valuable: plan diversity);
true parallel width needs the local multi-GPU host or K VMs (operator opt-in —
cost multiplies).

### 8.4 Novelty gate (typed, prose-free — F10)

`directives_sha256` hashes **only the deterministic action schema**: seed
lineage id, scope ladder position, the *typed* per-leaf repair-action kinds
from `leaf_triage` (`render_artifact`/`protocol_gap`/... — never justification
text), the failure-class set being addressed, and the attempt envelope. LLM
prose (`propose_improvements` output, grader justifications) rides along as
*context* but is **excluded from the hash** — two attempts differing only in
prose are NOT novel. A synthesized directive set whose hash equals ANY prior
attempt's is refused: the policy forces the next lineage arm; if all arms
exhaust, terminal `EXHAUSTED{no_novel_plan}`.

## 9. PLAN_ATTEMPT — directive synthesis (clean-context contract)

Deterministic assembly, no new LLM surface. `directives/<n>.json` contains,
and ONLY contains, structured artifacts:

- understanding digest ref + unresolved-field warnings (§6.3)
- leaf-level repair plan (existing `leaf_triage` output of attempt N)
- recorded `propose_improvements` output of attempt N (an artifact, not a
  transcript; context-only w.r.t. novelty, §8.4)
- failure capsules (bounded, redacted — existing format)
- prior-attempt measured evidence block (existing `prior_attempt_evidence`)
- memory hints (lessons / recipes / infra, existing caps ≤5 / ≤200c)
- scope directive (ladder position) + explicit seed pointer (champion /
  runner-up / fresh — campaign-selected, §8.2)
- the attempt envelope (§10.1)

**Prohibited by construction:** raw transcripts, REPL history, prior prompts.
(CCRM: contaminated retries cascade 7.1×; clean restarts +21% at equal
budget.) Delivery uses only existing surfaces: the run-spec file,
`OPENRESEARCH_BASELINE_EXTRA_GUIDANCE`, `--scope-spec`, the staged seed slot.

## 10. Cost & safety rails

### 10.1 Campaign budget — split meters, enforceable-or-refuse (F2, F3)

`CampaignBudget{max_llm_usd, max_gpu_usd, max_gpu_hours, max_attempts,
max_wall_clock_s}` fixed at INIT — the CLI requires the money meters
explicitly (no defaults for dollars). Each attempt gets an
**`AttemptEnvelope{llm_usd, gpu_usd, gpu_hours, wall_s, vm_ceiling_s}`**
derived as `max(floor, remaining/expected_remaining_attempts)` per meter and
mapped onto the REAL enforcement knobs: `--max-usd` (LLM/SDK spend — that is
what it caps), `--max-run-gpu-usd`/`OPENRESEARCH_MAX_RUN_GPU_USD` +
`max_gpu_hours` (GPU spend), `--max-wall-clock`, and an **explicitly set**
cloud control-plane ceiling (`max-run-duration=vm_ceiling_s` — never the 28h
default). **Enforceability is checked at PLAN_ATTEMPT: a meter that cannot be
enforced on the chosen driver/backend (no cost ledger, no control-plane
ceiling) fails closed** — the campaign refuses to launch that attempt
unattended (it downgrades to `checkpoint` mode with an explicit reason, or
stops). **`stage_on_gpu` accounting (F3):** when the provider's tiering
strategy is `stage_on_gpu`, GPU billing starts at provisioning — the envelope
charges provision/stage/green time against `gpu_usd`/`gpu_hours` (estimate up
front, reconcile from the cost ledger at ASSESS), so "gates before GPU
dollars" is accounted honestly rather than asserted falsely. Operators who
want a hard guarantee set `OPENRESEARCH_CAMPAIGN_REQUIRE_CPU_TIER=1`:
unattended attempts then require a validated real-CPU-tier strategy
(`machine_type_flip` / `cpu_warm_disk_then_gpu_attach`) and otherwise refuse.
Spend is read back from the attempt's cost ledger + cloud billing fields at
ASSESS (assessed, not assumed); DECIDE's budget floor uses the conservative
next-attempt estimate (`estimate_scope_cost`) per meter.

### 10.2 The SIGTERM-trap rule
Per-attempt wall-clock is set with **finalize headroom** (≥30 min below the
attempt's `vm_ceiling_s`), and the campaign profile carries the
run-experiment timeout guidance — a hard stop must still reach the validator +
finalize path (the 2026-06-20 6h-trap lesson). ASSESS treats a hard-stopped
attempt's salvage report as first-class evidence.

### 10.3 Doomed-attempt early kill (`OPENRESEARCH_DOOMED_KILL`, default OFF)
`doomed_run_comparator.py`: during WATCH, compare the live training curve
(persisted metrics/heartbeat the stall guard already reads) against the best
COMPLETED attempt's curve at the same step. Two-signal discipline like the
stall guard, with conservative defaults: fires only when (a) the live headline
metric is worse than the best-completed curve's value at the same step by a
relative margin ≥ `DOOMED_MARGIN` (default 30%) for ≥ `DOOMED_POLLS`
consecutive polls (default 6), AND (b) training has passed a step-fraction
floor `DOOMED_MIN_PROGRESS` (default 0.2 of planned steps). Never fires on the
first attempt (no baseline), never on missing/unaligned data, never on a
metric whose direction is unknown to the metric contract. Abort classifies as
`failure_class="doomed_by_comparison"` (repairable — informs the next
directive). (LLMZero: 62% early kills, 40–60% GPU saved.)

### 10.4 Evaluator lockdown (`OPENRESEARCH_EVAL_LOCKDOWN`, default OFF)
- `generated_rubric.json` sha256 pinned in `campaign.json` at UNDERSTAND;
  ASSESS re-verifies before trusting any grade. Mismatch →
  `rubric_integrity_violation`: the attempt's grade is quarantined
  (fabrication-suspected repair), never a terminal `REPRODUCED`.
- Campaign state (`campaign/`, `attempts.jsonl`) lives outside the
  agent-writable `code/` tree; cells continue to receive copies, never
  originals.
- **Canary leaf** (`OPENRESEARCH_RUBRIC_CANARY`, default OFF): rubric-gen
  appends one leaf referencing an artifact/metric that honest work cannot
  produce; it carries **weight 0** (excluded from the score denominator) but is
  graded — any credit > 0 raises a campaign fabrication alarm feeding the
  validator-veto path. (CapCode capped-canary pattern; RewardHackingAgents:
  locked evaluation → 0% compromise.)

### 10.5 Anti-p-hacking
Directives carry protocol/invariant requirements, never "reach value X":
proximity-to-paper-numbers is not a reward anywhere in DECIDE (the two-axis
verdict + eval-provenance + protocol checks stay authoritative). Guards
against the SocSci confirmatory-specification-search failure mode.

### 10.6 Autonomy modes + campaign steering channel (F13)
`unattended` (default): runs to terminal within budget, overnight OK.
`checkpoint`: after each DECIDE, emit `campaign_awaiting_operator` (SSE +
ledger) with the next-attempt plan + cost estimate and block for approval.
**Steering lives in `runs/<project>/campaign/user_messages.jsonl`** — the
`campaign/` directory is never touched by `attempt_isolation` (the top-level
`user_messages.jsonl` is archived per attempt, so it cannot carry
cross-attempt operator state). Messages carry ids; the consumed-cursor is
checkpointed in `campaign.json`. A thin route addition
(`POST /runs/<id>/campaign/messages`) mirrors the existing steering endpoint.
Mode is switchable mid-campaign via the same channel.

## 11. Self-improvement

### 11.1 DISTILL (Phase B — every attempt, guaranteed)
After ASSESS, the campaign itself invokes the existing miners as library calls
on the attempt's run dir: `lesson_distiller.mine_lessons`, evidence-gated
recipe admission, `experience_memory.record(attribution)` (infra/method
routing invariant intact). Campaign-side invocation closes today's gap where a
wall-clock-killed run skips finalize hooks and loses its lessons. All
admission gates unchanged (recurrence ≥2, caps, staleness, EvidenceVector for
recipes) — the campaign guarantees *invocation*, never bypasses *admission*.

### 11.2 Self-edit tier (Phase C — `OPENRESEARCH_SELF_EDIT`, default OFF)
The Self-Harness blueprint on the existing substrate:

- **Editable surface = an explicit whitelist** (`backend/agents/rlm/
  self_edit_surface.json`): named prompt-guidance blocks (registry ids) and
  bounded numeric retry/threshold config keys (each with min/max). Nothing
  else.
- **Frozen tier is constitutional (structurally enforced, not policy):**
  fabrication guards, evidence predicates, rubric + rubric-gen, external
  validator, budget enforcement, the admission gates, the whitelist file
  itself, and this module. `harness_self_edit.py` rejects any proposal
  touching a path/key outside the whitelist; a unit test proves frozen-tier
  targets are rejected. (DGM deleted the markers its reward function used to
  detect its own fabrication — the proof this boundary must be structural.)
- **A dedicated `HarnessEditGate` — NOT `held_out_gate.admit` (F11):**
  `held_out_gate` is advisory-lesson machinery and stays that way. Phase C
  ships its own gate with the same EvidenceVector non-regression *principle*
  but mechanics-grade artifacts: `HarnessEditProposal{surface_key, delta,
  mined_from}`, executable `HarnessReplayCase`s (re-run the affected harness
  path on CPU against frozen evaluator inputs), fail-closed promotion
  semantics (any replay error = reject, never skip).
- **Staged promotion — strengthened (F12):** candidate → **shadow** (replay
  over the `HarnessReplayCase` corpus; validator-veto absolute; any held-out
  predicate regression rejects) → **canary** (paired A/B attempts on ≥2
  held-out papers × ≥2 seeds, improvement must exceed the measured grader σ,
  plus a negative-control replay: applied to unrelated cases the proposal
  must change nothing) → **default = operator-confirmed, never automatic**.
  Auto-apply exists only at the canary stage and only inside the whitelist;
  the canary→default flip is a human decision with the full lineage in front
  of them (`runs/_memory/harness_proposals/<id>.json`: mined-from, replay
  results, canary A/B, negative control, status).
- **ReplayCase harvesting:** every campaign terminal writes replay cases
  (evidence-vector snapshots + minimal artifact refs) to
  `runs/_memory/replay/`. Until the corpus is non-trivial, proposals stay
  candidates (fail-soft, same rule as Phase 1e).
- **Goodhart controls:** exactly one editable lever (the whitelist), a frozen
  rotating verifier (cases re-split per admission), negative controls, and a
  human default-flip — the SIA co-evolutionary failure mode has no coupled
  second lever to exploit.

## 12. Observability + the deliverable

- SSE: `campaign_started`, `attempt_started`, `attempt_assessed` (assessment
  row), `campaign_decision` (rule that fired + directive fingerprint),
  `campaign_awaiting_operator`, `campaign_terminal`. All flow through the
  existing sanitizer chokepoint; assessments carry no corpus text.
- Leaderboard: existing best-attempt resolution already rolls up attempts; a
  campaign column (attempts, terminal, spend) reads `campaign.json` at request
  time (no new projection).
- **Deliverable (locked decision 9):** the champion attempt's
  `final_report.{json,md}` + `campaign_report.md` — attempt table (per-attempt
  score / evidence vector / cost / decision), evidence trajectory, terminal +
  stop_reason, declared exclusions with verification status, claims-vs-measured
  deltas from the existing rubric/two-axis machinery.
- **Plan-only writer (F14):** `INFEASIBLE` (and UNDERSTAND-gate blocks) produce
  a campaign-authored deliverable — `campaign_report.md` + a minimal
  `final_report.json{verdict:"plan_only", stop_reason, what_would_unblock}` —
  because `ReproductionRun`'s `PLAN_ONLY` outcome carries `report=None` today;
  the campaign never terminates report-less.

## 13. Error-handling philosophy

Extends the two-regime rule one level up:

- **Attempt-level failure** → assessed, classified, distilled; repairable
  classes fuel the next directive; unrepairable classes terminate honestly.
- **Campaign-level failure** (driver crash, budget read failure) → fail-soft
  to `EXHAUSTED{campaign_error}` shipping the champion-so-far; **except** an
  unwritable spend ledger, which halts *before* launching anything new (§5).
  Never a stranded VM (the attempt's own teardown/watchdog ceilings are
  unchanged and cloud-side).
- **Fabrication anywhere** → quarantined grade, alarm, repair directive —
  never terminal `REPRODUCED` (guards remain the backstop; the campaign adds
  rubric-integrity + canary alarms + validator-absence quarantine on top).
- All new flags default-OFF; the campaign path itself is opt-in via the
  subcommand — **no behavior change for any existing entrypoint** (`reproduce`,
  batch, UI) until an operator runs `campaign`.

## 14. Testing strategy

Hermetic (zero cloud spend, socket-hermetic like the existing suites):

- `FakeAttemptDriver` scripted sequences: success-first · repair-then-success ·
  plateau at K · unrepairable class · same-infra-signature-twice cap ·
  mid-attempt budget kill · novelty exhaustion (typed-domain: prose-only change
  is NOT novel) · contradicted-requires-two · report-missing ·
  rubric-hash-mutation (grade quarantined, no REPRODUCED) ·
  **validator-unavailable quarantine (no REPRODUCED)** ·
  **envelope-unenforceable → refuse-unattended** ·
  **stage_on_gpu accounting (provision time charged to GPU meters)**.
- **Spend-ledger properties:** write-ahead intent before launch (kill between
  intent and launch → resume does not double-launch); unwritable ledger halts
  before LAUNCH; kill at EVERY state → resume → identical terminal + no
  double-spend.
- **Warm-retry quarantine:** a run dir with `code/` and no recorded assessment
  is force-archived before relaunch; the warm-retry heuristic is never hit
  under a campaign.
- DECIDE table tests: every rule row both ways; per-meter budget-floor
  arithmetic; width eligibility; scope-ladder re-expansion; guard-filtered
  champion selection (a higher-scoring guard-tripped attempt never seeds).
- Directive synthesis: clean-context contract (a transcript path in any input
  artifact fails the build), typed-fingerprint normalization.
- UNDERSTAND: double-extraction diff determinism; tiered blocking (span-
  grounded lint failure blocks; LLM-only field never blocks); run-spec
  round-trip validation (a rejected key fails INIT at $0).
- Self-edit: frozen-tier proposal rejected · out-of-bounds numeric rejected ·
  fabricated-evidence proposal rejected · negative-control regression rejects ·
  canary→default requires operator flag · promotion lineage recorded ·
  whitelist-file edit proposal rejected.
- Doomed-kill comparator: never fires on attempt 1 / missing data / unknown
  metric direction; fires on sustained dominance-gap fixtures.

Live (operator-gated, never CI): cheap paper campaign end-to-end (loop
mechanics, ~$10s), then SDAR smallest-two on GCP — ≥3 paired campaigns; run
with `--driver paired` to **conduct** the Phase-1f A/B under the runbook's
operator sign-off discipline. Validation budget ≈ $300 (locked decision 8).

## 15. Component → file map

| Component | New/extends | Where |
|---|---|---|
| `ReproductionCampaign` state machine + fail-closed spend ledger | NEW | `backend/agents/rlm/reproduction_campaign.py` |
| `AttemptDriver` + `LiveCliDriver` + `UnifiedRunDriver` + force-quarantine | NEW (+ explicit-archive entry point on `attempt_isolation`) | `backend/agents/rlm/attempt_driver.py` |
| `AttemptAssessment` (deterministic read) | NEW | `backend/agents/rlm/attempt_assessment.py` |
| DECIDE policy + guard-filtered champion + lineage rules + typed novelty + plateau | NEW | `backend/agents/rlm/campaign_policy.py` |
| Directive synthesis | NEW | `backend/agents/rlm/campaign_directives.py` |
| UNDERSTAND gate (double-extraction diff + tiered lint) | NEW | `backend/agents/rlm/understanding_gate.py` |
| Campaign report + plan-only writer | NEW | `backend/agents/rlm/campaign_report.py` |
| `AttemptEnvelope` + enforceability check | NEW (extends `budget.py` consumption) | `campaign_policy.py` |
| Doomed-run comparator | NEW (flag) | `backend/agents/rlm/doomed_run_comparator.py` |
| Evaluator lockdown (hash pin + canary wiring) | extends | `campaign_policy.py` + `rubric_gen.py` (canary leaf, flag) |
| Self-edit tier + `HarnessEditGate` | NEW (flag) | `backend/agents/rlm/harness_self_edit.py` + `self_edit_surface.json` |
| Replay-case harvest | NEW store | `runs/_memory/replay/` (writer in `reproduction_campaign.py`) |
| Campaign steering channel + route | NEW (thin) | `campaign/user_messages.jsonl` + `backend/routes/messages.py` sibling |
| CLI `campaign` subcommand (+ `--resume`, `--driver`, `--mode`) | extends | `backend/cli.py` |
| Campaign run-spec profile (exact `OPENRESEARCH_*` keys) | NEW (config) | `configs/campaign_run_spec.json` |
| SSE event types (6) | extends | `sse_bridge.py` vocabulary + frontend event map |

Flags (all default-OFF; campaign path opt-in by subcommand):
`OPENRESEARCH_CAMPAIGN_MAX_ATTEMPTS` (default 6) ·
`OPENRESEARCH_CAMPAIGN_MAX_LLM_USD` / `OPENRESEARCH_CAMPAIGN_MAX_GPU_USD` /
`OPENRESEARCH_CAMPAIGN_MAX_GPU_HOURS` (CLI-required, no dollar defaults) ·
`OPENRESEARCH_CAMPAIGN_WALL_CLOCK_S` · `OPENRESEARCH_CAMPAIGN_WIDTH` (1) ·
`OPENRESEARCH_CAMPAIGN_MODE` (unattended) ·
`OPENRESEARCH_CAMPAIGN_PLATEAU_K` (2) ·
`OPENRESEARCH_CAMPAIGN_DRIVER` (`live` | `unified` | `paired`; default `live`) ·
`OPENRESEARCH_CAMPAIGN_WIDTH_SKIP_SCORE` (0.5) ·
`OPENRESEARCH_CAMPAIGN_REQUIRE_CPU_TIER` (off — on: unattended requires a real
CPU tier) ·
`OPENRESEARCH_DOOMED_KILL` (+ `DOOMED_MARGIN`/`DOOMED_POLLS`/`DOOMED_MIN_PROGRESS`) ·
`OPENRESEARCH_EVAL_LOCKDOWN` · `OPENRESEARCH_RUBRIC_CANARY` ·
`OPENRESEARCH_SELF_EDIT`.

## 16. Phasing

- **Phase B (implement first):** §5–§10 + §11.1 + §12 + hermetic §14. Lands as
  one reviewable unit; live validation per §14 before any default-flip
  discussion. Paired-driver campaigns *conduct* Phase-1f step 3 under operator
  sign-off (F8) — this spec automates the procedure, not the judgment.
- **Phase C (implement second, same spec):** §11.2 self-edit tier +
  `HarnessEditGate` + replay harvest + canary/lockdown default-flip candidacy —
  each behind its flag, each subject to the ≥3-paired-run discipline, the
  canary→default flip always operator-confirmed.

## 17. Non-goals

- No in-cluster campaign service (process + checkpoint only; north-star
  unchanged).
- No mid-attempt cloud migration (provision-time failover exists; campaign
  retries are whole attempts).
- No new LLM surfaces in the campaign layer (synthesis is assembly of recorded
  artifacts).
- No self-edit outside the whitelist, ever; no editing the frozen tier, ever;
  **no autonomous canary→default flip, ever** (operator-confirmed only).
- No `ReproductionRun` re-enter API in Phase B (resume of unified attempts =
  assess-from-disk; the API is north-star).
- No UI work beyond the SSE events + leaderboard column + the thin steering
  route.
- No change to any existing entrypoint's behavior.

## 18. Open questions / risks

1. **Attempt-cost estimation fidelity** — per-attempt envelopes derive from
   `estimate_scope_cost` (conservative by design); a badly wrong estimate
   wastes at most one attempt (the live WATCH budget check + the explicit
   cloud ceiling remain the backstops).
2. **CONTRADICTED confidence** — two confirming attempts is a floor, not a
   proof; the report language must present it as "measured non-replication
   under a faithful implementation," never "the paper is wrong."
3. **Width on single-VM GCP** — sequential-only until multi-VM width is
   operator-approved (cost multiplies linearly).
4. **Replay-corpus cold-start** — until campaigns accumulate cases, the
   self-edit gate keeps everything candidate-only (fail-soft, same as Phase
   1e).
5. **Rubric drift across attempts** — pinned by `REUSE_RUBRIC` + hash; a
   deliberate operator rubric change mid-campaign requires a new campaign.
6. **Checkpoint-mode latency** — a paused campaign holds no lease (attempts
   are whole units; pause points are between attempts), so cost-of-waiting is
   zero by construction.
7. **Validator dependency for `REPRODUCED`** — fail-closed means a campaign
   without a configured validator panel cannot reach `REPRODUCED` unattended;
   INIT surfaces this up front so the operator funds a panel or accepts
   best-partial terminals (honest, and consistent with the grounded-SI spec's
   fail-closed validator transport).

## 19. References

- Phase-1 spec: `2026-07-01-paper-agnostic-multicloud-reproduction-and-self-improvement-design.md`; cutover runbook `2026-07-01-sdar-unified-run-cutover.md`.
- CCRM / context contamination: arXiv 2605.08563 · RewardHackingAgents: 2603.11337 · AIRA2: 2603.26499 · RePro: 2508.16671 · CAID: 2603.21489 · ML-Master 2.0: 2601.10402 · Self-Harness: 2606.09498 · LLMZero: 2606.18388 · BAGEN: 2606.00198 · OpenSkill: 2606.06741 · Who&When: 2505.00212 · STALE: 2605.06527 · memory-drift: 2605.12978 · PaperBench: 2504.01848 · MLR-Bench: 2505.19955 · SocSci-Repro: 2606.11447 · Inference Scaling fLaws: 2411.17501 · Verification Horizon: 2606.26300 · CapCode: 2606.07379 · MLE-STAR: 2506.15692 · REx: NeurIPS 2024 · DGM: sakana.ai/dgm · SIA: 2605.27276 · EurekAgent: 2606.13662.
- Repo red-line lineage: `2026-06-20-grounded-self-improvement-and-harness-reliability-redesign-design.md` (evidence-not-grade), `recipe_library.py:424` (multi-predicate admission).

## 20. Review resolution (Codex adversarial review, 2026-07-01)

All 16 findings resolved in v2.

| # | Sev | Finding (Codex, verified against code) | Resolution |
|---|---|---|---|
| F1 | BLOCKER | Campaign checkpoint specified fail-soft (mirroring `reproduction_run.py:139`) while resume/no-double-spend depend on it | Campaign ledger is **fail-closed**: atomic+fsync writes, write-ahead intent row before LAUNCH, unwritable ledger halts before money moves; `ReproductionRun`'s internal checkpoint stays fail-soft (§5) |
| F2 | BLOCKER | `--max-usd` is LLM/SDK spend (`cli.py:2392`), not GPU; GCP VM ceiling defaults 28h (`vm_compute_provider.py:115`) | Split meters (`max_llm_usd`/`max_gpu_usd`/`max_gpu_hours`/wall-clock) + `AttemptEnvelope` mapped to the real knobs + explicit `max-run-duration`; **enforceability checked per attempt, fail-closed to checkpoint-mode/stop** (§10.1) |
| F3 | BLOCKER | `stage_on_gpu` bills from `provision_cpu` (`vm_compute_provider.py:27,339`) — "no GPU before gates" false as stated | Envelope charges provision/stage/green to the GPU meters under `stage_on_gpu` (honest accounting); `OPENRESEARCH_CAMPAIGN_REQUIRE_CPU_TIER=1` for the hard guarantee (§10.1) |
| F4 | BLOCKER | Validator `unavailable` (`external_validator.py:474`, `run.py:3567`) could read as "non-veto" for terminals | `REPRODUCED` requires `validator.status=="clean"` on a fresh fingerprint; absence/staleness quarantines the grade; validator-less campaigns say so at INIT (§8.1, §8.2, risk 7) |
| F5 | BLOCKER | `best_attempt.py:63` / `champion_artifact.py:178` rank by score — red-line violation if they pick the seed | Champion selection is **campaign-owned and guard-filtered** (guards+validator+integrity clean first, EvidenceVector rank second); explicit seed pointer; floor from guard-clean attempts only (§8.2) |
| F6 | BLOCKER | `attempt_isolation.py:256` warm-retry heuristic silently reuses `code/` when a killed attempt left no report | **Force-quarantine before every launch** (explicit-archive entry point); warm retry never exercised under a campaign (§7) |
| F7 | MAJOR | `ReproductionRun` has no re-enter API; resume protocol underspecified | Concrete resume protocol: `in_flight` handle + liveness probe → re-attach or assess-from-disk; idempotent ledger replay; unified re-enter API declared north-star (§5, §17) |
| F8 | MAJOR | Spec claimed paired campaigns "are" Phase-1f A/B; runbook forbids pre-sign-off flips | Paired campaigns **conduct** the A/B under operator initiation + sign-off; no automated default-flip (§7, §16) |
| F9 | MAJOR | UNDERSTAND blocking on LLM-extracted fields contradicts "no new LLM judgment" | Tiered blocking authority: only span-grounded cross-verified lint failures or probe-confirmed asset gaps block; LLM-only fields advisory forever (§6.5) |
| F10 | MAJOR | Novelty hash over LLM prose (`propose_improvements`, grader justifications) is gameable/noisy | Typed novelty domain: seed lineage + scope + typed repair-action kinds + failure classes + envelope; prose excluded (§8.4) |
| F11 | MAJOR | `held_out_gate` is advisory-lesson-only (`held_out_gate.py:25`); reusing it for mechanics edits misuses the contract | Dedicated **`HarnessEditGate`** with executable `HarnessReplayCase`s + fail-closed promotion; `held_out_gate` untouched (§11.2) |
| F12 | MAJOR | One-canary→default too weak, Goodhart-prone | Canary = ≥2 held-out papers × ≥2 seeds + grader-σ bound + negative control; **default flip operator-confirmed, never automatic** (§11.2, §17) |
| F13 | MAJOR | Top-level `user_messages.jsonl` is archived per attempt (`attempt_isolation.py:85`) — steering channel would vanish | Campaign steering at `campaign/user_messages.jsonl` + message ids + checkpointed cursor + thin route (§10.6) |
| F14 | MAJOR | `INFEASIBLE` promised a report; `ReproductionRun` `PLAN_ONLY` returns `report=None` (`reproduction_run.py:202`) | Campaign-authored plan-only writer (`campaign_report.py`); the campaign never terminates report-less (§12) |
| F15 | MAJOR | Run-spec keys as written would be ignored by `_load_run_spec` (`cli.py:828`) | Canonical profile uses exact `OPENRESEARCH_*` keys; INIT round-trip validation fails at $0 on any rejected key (§7) |
| F16 | MINOR | `--project-id-suffix` is `batch_reproduce.py` machinery, not `cli reproduce` | Width lineages minted via `--project-id <campaign>_w<k>` (existing override + ingest path) (§8.3) |
