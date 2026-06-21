# MEGA-PROMPT — Unify the harness, solidify the history, and drive SDAR to the highest reproducible score

> **What this is.** A single self-contained brief for an autonomous engineering
> session (you, an agent with the superpowers skills) to (1) finish consolidating
> a fragmented git history, (2) unify and solidify the harness machinery — **BES,
> the actor/critic evidence loop, the external-validator agent, the evidence
> gates, grader fidelity** — into one coherent, TDD-covered system, and (3) use
> that system to reproduce the **SDAR** paper at the **highest rubric score
> achievable**, honestly. The North Star is the SDAR score; everything else is in
> service of making that score *real* and *repeatable*.
>
> **Non-negotiables (read first):**
> - **Zero paid GPU / zero live-API spend in tests or "to verify."** Every test is
>   hermetic (the suite is socket-hermetic via pytest-socket). Real-infra smokes
>   (RunPod/Azure/GCP pods) are **operator-gated** and explicitly labelled as
>   costing money. You may *prepare* and *document* them; you may not *launch* them.
> - **Default-OFF + fail-soft for every new/changed flag.** Unset ⇒ byte-for-byte
>   identical behaviour to today. There is a `tests/rlm/test_default_off_contract.py`
>   that enforces this — keep it green.
> - **Merges only on shared branches** (`main`, others' PR branches). Never
>   force-push them. You may force-push your *own* feature branches.
> - **TDD always** (superpowers:test-driven-development): red → green → refactor,
>   one behaviour per test, frequent commits.
> - **Honesty gate.** Never let a fabricated or ungrounded result raise a score.
>   The whole point of the evidence machinery below is that a high score must be
>   *earned on disk*. If you cannot ground a number, report it as unmet.

---

## 0. Orientation — the current state of the world (verify, don't trust)

The repo (`armaanamatya/openresearch`) is mid-consolidation. **Verify each of these
with git before acting; they were true at authoring time (2026-06-20):**

- **De-facto trunk:** `feat/bes-conversion-correctness` (PR #107, based on
  `feat/azure-bicep-canonical-aoai-hardening` = #109). All active work stacks here.
- **`main` had diverged** (2-ahead / 138-behind, not an ancestor) because #106
  (grader-fidelity) and #108 (azure-gcp) were integrated straight to `main`,
  bypassing the trunk. **PR #115 (`consolidate/main-trunk`)** reconciles this:
  trunk declared canonical, `main` folded in via a merge (no force-push), 55
  conflicts resolved trunk-wins, CI-fix for the Bicep `validate` job included.
  **Confirm #115's state and whether it merged.**
- **#110 integration in progress:** branch `feat/grounded-harness-integration` =
  trunk + a **squash-merge** of `feat/grounded-self-improvement-harness-reliability`
  (PR #110). 76 net-new files landed clean; 43 seam conflicts resolved
  trunk-canonical + #110 additive hooks; ruff clean; **the full-suite regression
  gate is the last check before commit + PR.** Confirm the suite result, commit
  as clean logical commits, open the PR to the trunk.
- **Workstreams A/B/C** (already clean PRs on the trunk): #111 Foundry provider
  unification, #112 GKE first-class backend, #114 hallucination diagnosis + Fix-1
  guard. Keep them coherent with the unification below.
- **A private "other repo"** with its own history problems is in scope but **not
  yet identified** — the operator owes you its URL + what's wrong. Surface this;
  do not silently drop it.

**First action:** run the orientation block and reconcile reality with the above.
```bash
git fetch --all --prune
git log --oneline --graph --decorate origin/main origin/feat/bes-conversion-correctness origin/feat/grounded-harness-integration | head -40
gh pr list --state open
gh pr view 115 --json mergeable,mergeStateStatus,statusCheckRollup
```

---

## 1. The North Star — SDAR (arXiv 2605.15155)

**Self-Distilled Agentic Reinforcement Learning.** This is the canonical "hard"
paper: a surrogate cannot pass its rubric. Source of truth in-repo:
`backend/agents/prompts/paper_hints.py` (the `2605.15155` `PaperHint`) and the
SDAR runbooks under `docs/runbooks/2026-0*-sdar-*`.

**Algorithmic invariants the rubric inspects (do not let these drift):**
| Invariant | Value / form |
|---|---|
| OPSD gate | `g_t = sigmoid(beta * delta_t)` — the sigmoid·β form is mandatory |
| Gate gradient | `stop_gradient` on the gate (gate weights advantage, doesn't backprop through it) |
| Self-distillation weight | `lambda = 0.1` |
| Gate temperature | `beta = 10` |
| Loss | `SDAR loss = GRPO loss + gated self-distillation loss` (BOTH required) |
| Models | Qwen3-1.7B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct (3 sizes) |
| Environments | ALFWorld + WebShop + Search-QA (3 distinct) |
| Baselines | GRPO, OPSD, Skill-SD, GRPO+OPSD, RLSD (5) |
| Seeds | 42, 43, 44 |
| Paper compute | 8× H800, 150 training steps |

**Cost-bounded default scope (what you actually run):** smallest-two models
(Qwen3-1.7B + Qwen2.5-3B, SKIP 7B), real HF weights (NO surrogate), real
ALFWorld + Search-QA + WebShop on a small representative slice (~32 tasks/env),
single 24–48 GB GPU. Reference invocation (operator-gated — costs money):
```
OPENRESEARCH_RUNPOD_CLOUD_TYPE=COMMUNITY \
OPENRESEARCH_BASELINE_EXTRA_GUIDANCE="SCOPE: reproduce SDAR using ONLY the two smallest model variants (Qwen3-1.7B-Instruct, Qwen2.5-3B-Instruct); SKIP 7B. Real pretrained HF weights, real ALFWorld+Search-QA+WebShop, ~32 tasks/env. Report both sizes." \
.venv/bin/python -m backend.cli reproduce 2605.15155 \
  --mode rlm --sandbox runpod --model claude-oauth \
  --vram-gb 38 --max-wall-clock 5400 --max-pod-seconds 5400 --max-usd 20
```

**Score ceiling is set by three things, in order:** (a) the implementation is
*real* (no stub — the NO-STUB block + runtime-compute detection enforce this);
(b) the grid actually *runs to completion and is graded on the complete
evidence* (the finalize-time freshness re-grade exists for exactly this); (c)
the algorithmic invariants above are present (the `PaperHint` regex invariants
catch the gate/λ/β). **Your job is to make all three robust, then push the
frontier of (a) via the unified candidate/critic loop.**

---

## 2. The unification — one coherent evidence-first harness

Today the machinery is powerful but **fragmented across parallel lineages**, and
#110 deliberately lands a *second* design for some of it (e.g. `evidence_audit.py`
vs `evidence_gate.py`; `build_validator_client` vs `build_grader_client`). The
goal of this phase is **ONE system, no duplicate designs, each layer flag-gated
and TDD-covered.** Resolve the duplication; do not ship two parallel evidence
stacks silently.

### 2.1 The actor / critic / external-validator loop (the core)
Unify these into a single, legible pipeline with clear seams:

- **Actor** = the RLM root (`backend/agents/rlm/run.py` + the 17 primitives).
  Writes code, runs experiments, calls `verify_against_rubric`.
- **Critic (deterministic, in-loop)** = the **EvidenceAudit** critic
  (`evidence_audit.py`, `OPENRESEARCH_EVIDENCE_AUDIT`): a run-level evidence
  snapshot from on-disk state + the lifecycle ledger; provides the single
  `result_is_fabricated` veto and `run_level_clean` predicate. **Reconcile it
  with the existing `evidence_gate.py` (A7) and `champion_artifact.py` so there is
  ONE evidence-truth source, not two.** Decide: does `evidence_audit` subsume
  `evidence_gate`, wrap it, or feed it? Write the ADR in the PR body.
- **External validator (independent, adversarial)** = `external_validator.py`
  (`OPENRESEARCH_EXTERNAL_VALIDATOR`, panel size `OPENRESEARCH_VALIDATOR_PANEL_N`):
  a cross-family panel running machine checks (provenance / not-all-constant /
  GPU-plausible / rerun-agrees / report-claims-grounded) + an LLM suspicion parse,
  on **every finalize path**. Transport (`build_validator_client`) is
  **fail-CLOSED** by design — keep it so. Reconcile its transport plumbing with
  `grader_transport.build_grader_client` so they share config without clobbering.
- **Pre-GPU code-review gate** = `code_review_gate.py`
  (`OPENRESEARCH_CODE_REVIEW_GATE`, requires `EXTERNAL_VALIDATOR=1`): a
  cross-family reviewer reads the training code *before* GPU dispatch and blocks
  only on `will_produce_fake/wrong_metrics` with a real file:line. Fail-open.
- **Report-claim gate** = `report_claim_gate.py` + `claim_grounding.py`
  (`OPENRESEARCH_REPORT_CLAIM_GATE`): deterministic backstop in
  `write_final_report_rlm` that caps the verdict when the narrative makes
  ungrounded result claims. `claim_grounding.py` is the shared engine — integrate
  it as the single dependency for the in-loop refusal, the validator predicate,
  and this gate.

**Unification deliverable:** a one-page ADR (`docs/superpowers/specs/`) — "the
evidence-first actor/critic architecture" — that draws the final box diagram
(actor → in-loop critic → finalize critic → external validator → report gate),
names every flag, states which is the single source of evidence truth, and lists
what was deduplicated. Then make the code match the diagram.

### 2.2 BES — solidify, don't expand
BES (`OPENRESEARCH_BES_ENABLED`, candidates-per-cluster) competes N isolated
implementations and SELECTs by **deterministic evidence**, never the LLM grade
(`bes_rlm.py`, the `_dispatch_competing_candidates` RDR path). Per the
2026-06-17 conversion-correctness spec, the proven failure mode is **conversion/
archival**, not generation — so: keep the champion bundle *coherent* (champion's
own leaves + `sample_count`), keep the conversion-provenance guard
(`conversion_guard.py` + `report.repair_projection_from_disk`), keep the archival
gate (`archive_completeness.py` → `ab_compare --require-stamped`). **Wire BES
SELECT to the unified critic in 2.1** so the candidate that wins is the one with
the strongest *grounded* evidence, and stamp the SELECT-stability instrumentation
(`select_stability.py`). Do not resurrect the cut evolutionary redesign.

### 2.3 Grader fidelity — the denominator under everything
The leaf grade is a non-deterministic LLM call. The σ-gate already cleared at
temperature=0 (σ=0.0067 @ samples=1). Keep `OPENRESEARCH_GRADER_SAMPLES` default
1; recommend samples=3 for fidelity-critical SDAR runs. Keep the median-of-N,
the independent grader transport (`grader_transport`), the evidence-fingerprint
floor (median-not-max), and the deterministic-leaf route for annotated leaves.
**The unified critic must not double-count with the grader** — the grader scores;
the critic vetoes ungrounded credit. Keep that separation crisp.

### 2.4 TDD + history hygiene
- Every behaviour change ships with a failing-first test. The `default_off`
  contract test is the floor, not the ceiling — also test the *enabled* behaviour
  of each gate with synthetic on-disk fixtures (zero GPU, zero network).
- **Clean history:** integrate via squash-merge or net-diff cherry-pick — never
  drag a 1000+-commit parallel branch's history onto the trunk. One feature
  cluster ⇒ one reviewable PR.
- **Gate every integration on the FULL trunk suite** (CI-parity), because the
  real risk is regressing the trunk's *enabled* P0 code, which a default-off test
  cannot catch. Locally: `OPENRESEARCH_MIN_DISK_GB=0 .venv/bin/python -m pytest tests/ -q`.

---

## 3. Execution methodology

Use the superpowers workflow exactly:
1. **superpowers:brainstorming** if any sub-goal is under-specified — converge the
   spec before designing.
2. **superpowers:writing-plans** → a bite-sized, TDD, file-path-exact plan per
   sub-project under `docs/superpowers/plans/`. One plan per independent subsystem
   (history consolidation; evidence-architecture unification; BES SELECT-to-critic
   wiring; SDAR run-readiness). Each plan must stand alone and produce working,
   tested software.
3. **superpowers:subagent-driven-development** to execute — fresh subagent per
   task, two-stage review (spec compliance, then code quality) after each.
   Dispatch parallel agents only for *independent* problem domains
   (superpowers:dispatching-parallel-agents); never let two agents edit the same
   working tree.
4. **superpowers:using-git-worktrees** for any parallel isolated execution
   (symlink `.venv` so `.venv/bin/python` works under `pythonpath=["."]`).
5. **superpowers:finishing-a-development-branch** to close each out.

**Validation policy (repo rule):** ≥3 paired A/B runs (`scripts/ab_compare.py`,
`--require-stamped`) **before flipping any default**, plus the grader-σ check.
Until then, every new capability ships **default-OFF**.

---

## 4. Phased plan (sequence; each phase is a clean PR or set of PRs)

**Phase 0 — Land the in-flight work (you are here).**
- Confirm #115 CI green; ensure the owner sees the supersession note (it moots
  #107/#109's path to main — their call).
- Finish `feat/grounded-harness-integration`: confirm full-suite green, commit the
  squash as clean logical commits, open the PR to the trunk with the
  duplicate-design ADR callout from 2.1.
- Get the **private repo** URL from the operator; scope its history fix separately.

**Phase 1 — Evidence-architecture unification (2.1).** ADR + dedupe
`evidence_audit`/`evidence_gate`/`champion_artifact`; reconcile validator/grader
transports; single `claim_grounding` engine. All flag-gated, TDD, full-suite green.

**Phase 2 — BES SELECT-to-critic wiring (2.2)** + SELECT-stability stamping.

**Phase 3 — SDAR run-readiness (no GPU spend).** Harden the path the SDAR run
exercises, entirely with hermetic tests + dry-runs: cells manifest + capacity gate
+ OOM ladder, the `PaperHint` invariants (gate/λ/β regexes), the NO-STUB +
runtime-compute detection, finalize-time freshness re-grade, the
attempt-isolation archival. Produce a single **operator run-card** (one runbook)
that the operator executes to launch the real GPU run, with the exact command,
the expected SSE milestones, the kill-criteria, and the post-run grading check.

**Phase 4 — Operator-gated SDAR runs + A/B.** The operator launches; you babysit
via the monitoring loops (`docs/runbooks/2026-05-29-monitoring-loops.md`),
triage failure classes, and feed fixes back as Phase-1/2/3 PRs. Use the unified
critic + external validator to *prove* the score is grounded. Iterate toward the
highest *honest* rubric score; record each arm with `experiment_arm` stamps.

---

## 5. Acceptance criteria

- **One** evidence-truth architecture (ADR + code match), no duplicate parallel
  stacks; every gate flag-gated and tested both off (byte-identical) and on
  (synthetic-fixture behaviour).
- History consolidated: `main` coherent (via #115), each feature a clean
  squash/cherry-pick PR, no multi-thousand-commit history dragged onto the trunk,
  full suite green on every PR.
- BES SELECT chooses the most *grounded* candidate; conversion/archival
  correctness preserved; SELECT-stability stamped.
- SDAR path is hermetically tested and dry-run-clean; a single operator run-card
  exists; the invariants (gate/λ/β/stop-grad/GRPO+OPSD/real-weights) are enforced
  by the rubric + `PaperHint`.
- The SDAR rubric score is driven up **only through grounded evidence** — the
  external validator + evidence critic veto anything fabricated; the reported
  score equals the best *substantiated* artifact.
- Zero paid GPU / live-API spend incurred by you; all real runs operator-launched
  and labelled.

---

## 6. Guardrails recap (do not violate)

Default-OFF + fail-soft · unset == byte-for-byte today · merges-only on shared
branches · TDD red-first · full-suite CI-parity gate · zero test/verify GPU or API
spend · ≥3 paired A/B + grader-σ before any default flip · keep `CLAUDE.md` and
`system_overview.md` updated when you add a primitive / SSE event / sandbox /
flag / fail-mode · honesty over score — an ungrounded number is an unmet leaf.

> **The single sentence:** make the SDAR score *real and repeatable* by unifying
> the actor/critic/validator/BES/grader machinery into one evidence-first system,
> on a consolidated history, with TDD and zero unauthorized spend — then push the
> frontier of a genuinely-grounded reproduction as far as it will honestly go.
