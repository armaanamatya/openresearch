# System improvement opportunities — workflow deep-research synthesis (2026-06-21)

> Output of a 5-facet parallel deep-research pass over the reproduce-a-paper pipeline
> (understanding → implementation → execution → scoring → improvement loop). Each idea
> is grounded in code (file:line). This is the **menu** for "how do we make the system
> better" — ranked by impact-per-effort, grouped by the cross-cutting theme. Pick a
> theme to turn into a spec + plan.

## The 7 cross-cutting themes (what the research actually found)

**T1 — The biggest wins are FLIPPING DARK MACHINERY ON, not building new.** Across
execution and scoring, the expensive safety-nets and cost-savers are already built but
ship **default-OFF or gated on a setting that defaults empty**. This is the single
highest-leverage, lowest-effort cluster.

**T2 — The auto-rubric + invariants have NO grounding pass and cover only ~4–5 papers.**
The rubric is one unverified LLM call; paper-specific correctness gates are hand-authored
YAML for 4 papers. For arbitrary arXiv papers the strongest fidelity guards are dark.

**T3 — Tables/equations are flattened before extraction**, so the hyperparameters the
whole run depends on are lost at the source (PDF `get_text("text")`, no `find_tables`;
equations survive only the HTML path).

**T4 — The improvement loop is "advisory, not productive."** The plateau/decline/best-so-far
machinery exists but is wired as *notes*, not as inputs to the next proposal or as gates —
so the loop re-proposes just-failed paths and churns until the iteration cap.

**T5 — Determinism is coupled to honesty.** Expanding deterministic leaf checks that read
agent-written `provenance.json` would *launder* fabrications; provenance must be grounded
first. And the evidence gate checks existence, not VALUE (the `claim_grounding` engine is
underused).

**T6 — Cost: spot GPUs + cache persistence + cell-resume are the biggest $ levers**, mostly
S/M effort, and the resume scaffolding already exists.

**T7 — No labeled honest/fab eval corpus exists** — the meta-blocker for every default-flip
decision (validator, evidence_audit, grader samples all rest on one un-labeled run).

---

## Do-first cluster (S-effort, high impact) — "flip the dark switches" (T1/T6)

These are mostly default flips, each S-effort, validated via the existing A/B harness:

| Idea | What | Lever | Anchor |
|---|---|---|---|
| **Persist HF/pip/dataset cache by default** | RunPod cache gated on `network_volume_id` which defaults `""` → every pod re-pulls ~2–5 GB | **$ (biggest single leak)** | `runpod_backend.py:709`, `config.py:282` |
| **Enable cell-resume by default** | `should_skip_cell` skips passed cells but `OPENRESEARCH_RESUME_CELLS` is OFF → reruns pay for all 9 cells to fix 1 | **$ (80–90% of rerun cost)** | `cell_scheduler.py:235` |
| **Enable orphan-guard by default** | abandoned training holds VRAM, starves the retry; `OPENRESEARCH_ORPHAN_GUARD` OFF | reliability + $ | `orphan_guard.py:34` |
| **Enable dead-training early-stop by default** | SIGKILL a provably-stuck cell vs burn the budget; `OPENRESEARCH_DEAD_LOSS_EARLYSTOP` OFF | $ + actionable signal | `dead_training_guard.py:65` |
| **Enable OOM hard-memcap on retries** | batch-scale is advisory; the `set_per_process_memory_fraction` shim enforces it; `OPENRESEARCH_OOM_ENFORCE` OFF | salvages OOM cells | `gpu_cell_runner.py:102` |
| **Default-on import preflight-smoke on cost sandboxes** | catches the whole `ModuleNotFoundError` class on CPU before a GPU pod boots; `OPENRESEARCH_PREFLIGHT_SMOKE` OFF | $ | `preflight_smoke.py:19` |
| **Re-run AST preflight after a patch** | patch-mode writes `train.py` with no re-validation → burns a GPU to rediscover the unfixed bug | $ + churn | `primitives.py:2082` |

**Plus the M-effort $ giant:** provision **RunPod spot/interruptible** (~50–70% cheaper;
resume scaffolding already exists at `gpu_cell_runner.py:483`) — `runpod_backend.py:727`.

> Caveat (T7): the *scoring* default-flips (external_validator, evidence_audit) need the
> labeled corpus first; the *execution* flips above are safe to A/B and ship now.

---

## Fidelity cluster (M-effort, raises the score ceiling) — T2/T3

1. **Ground every rubric leaf's numbers/sections against the paper text** before accepting
   it (drop hallucinated `β=5` leaves). `rubric_gen._clean_categories` (`rubric_gen.py:209`).
2. **Auto-derive `PaperInvariants`/`InvariantSpec` from the generated rubric** when no YAML
   exists → the hard-gate enforcement scales from 4 papers to *every* run.
   `paper_invariants.load_paper_invariants:222`, `schemas.py:1100`.
3. **Structured table + equation extraction** (root cause): add `find_tables()`/`pdfplumber`;
   carry `<math>` alttext; this fixes the hyperparameter source data that 1, 2, and value-
   grounding all depend on. `pymupdf_parser.py:127`, `html_parser.py`.
4. **Verify regex-extracted hyperparameters** with a focused LLM/source-span cross-check +
   confidence. `paper_understanding.py:217`, `primitives.py:979`.
5. **Raise/re-target the 48k rubric-gen truncation** (hyperparameter tables live at the end
   and get cut). `rubric_gen.py:131`.
6. **Auto-derive default surrogate/loss-term guard** (catch "agent dropped GRPO+OPSD, used
   vanilla CE") generically, not YAML-only. `paper_invariants.py:98`, `preflight_ast.py:997`.

---

## Honesty cluster (closes the sophisticated fab holes) — T5/T7

1. **Build the labeled honest/fab eval corpus** (~10–20 runs) — the META-unblocker for every
   default-flip. `data/grader_calibration.json` (today: one un-labeled run).
2. **Value-grounding in the evidence gate**: route magnitude-claiming leaves through
   `claim_grounding.check_claims_grounded` (the underused asset) vs token-presence only —
   closes "on-disk but wrong value." `leaf_scorer.py:695`, `claim_grounding.py:302`.
3. **Ground `provenance.json` before expanding determinism** (else determinism launders lies):
   cross-check claimed epochs/steps vs the metrics-history length. `deterministic_leaf_checker.py:362`.
4. **Stop scoring ungraded/batch-error leaves as 0.0** in the denominator (systematic harsh-on-
   good bias). `leaf_scorer.py:1908`.
5. **Build the rubric-gen annotator** (`check_kind`/`assertion`) so the *unfed*
   `deterministic_leaf_checker` actually routes (~half of leaves, cheaper + more reliable).
6. **Criterion-gated default-flip of the external validator** for borderline-verdict runs
   (every veto is a harness-side machine-check, can't hallucinate). `external_validator.py:64`.

---

## Convergence cluster (S-effort, attacks churn/plateau) — T4

1. **Feed score-history + tried-hypotheses into `propose_improvements`** (the trajectory
   exists at `primitives.py:7674` but isn't passed) → stop re-proposing failures.
2. **Make the plateau detector GATE re-implementation**, not just advise — wire it into
   `ForcedIterationPolicy` (`forced_iteration.py:404`).
3. **Champion-restore-on-decline**: ship the peak `code/` snapshot when the trajectory
   declines (`best_attempt.py:233`, `primitives.py:7714`).
4. **Persist `record_candidate_outcome`** and feed it into proposals (today near-no-op).
5. **BES on the improvement loop** graded against REAL post-experiment metrics + multi-regrade
   σ-gated SELECT (moves the SELECT margin out of grader noise — the documented BES weakness).
   `bes_rlm.py:167,251`, `select_stability.py`, `candidates.py:203`.
6. **Class-level recipe/lesson promotion** (not exact `arxiv_id`) → first genuine cross-paper
   learning; later point the miners at the Run/Test Bank. `recipe_library.py:111`, `lesson_distiller.py:60`.

---

## The single highest-leverage reading
- **Fastest ROI:** the **T1/T6 "flip the dark switches"** cluster — real $ + reliability, S-effort,
  A/B-validated, shippable now (execution flips don't need the corpus).
- **Highest score ceiling:** the **T2/T3 fidelity** cluster — auto-derive invariants + ground the
  rubric + recover tables — this is what lets *arbitrary* papers (not just SDAR) get faithfully graded.
- **Highest trust:** the **T5/T7 honesty** cluster — build the corpus, value-ground the gate — required
  before any honest-scoring default flip.
- **Best convergence:** the **T4** cluster — make the existing advisory machinery *binding*.

Recommended sequence: **T1/T6 quick wins first** (immediate $ + safe), then **T7 corpus**
(unblocks the rest), then **T2/T3 fidelity** (the score-ceiling raiser), with **T4** woven in.
