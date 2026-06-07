# BES Integration Brief — prompt for working inside the reproduction codebase

> **Drop this file at the repo root** (or `docs/bes/`) alongside the three reference docs
> (`01_TRANSCRIPT_AV.md`, `02_ARCHITECTURE_SPEC.md`, `03_BES_PROPOSAL.md`). It is written to be
> handed to an AI coding agent **and** read by humans. It is deliberately **codebase-agnostic**:
> the actual module names, frameworks, and storage are unknown to the author of this brief and
> **must be discovered by recon in the target repo** (§2) and confirmed with the maintainer.

---

## 0. Role & how to use this prompt

**You are** a principal-level engineer integrating **BES (Bidirectional Evolutionary Search)**
into an existing system that **autonomously reproduces ML research papers**.

**Operating rules (non-negotiable):**
1. **Recon before code.** Read the three reference docs (§1), then map them onto the *real* code
   (§2). Do **not** assume file names, frameworks, schemas, or providers — find them.
2. **Plan, then confirm.** Produce a concrete integration plan that maps the abstract BES design
   onto the actual modules, and **get maintainer sign-off before writing non-trivial code.**
3. **Opt-in only.** All BES behavior ships behind a **feature flag, default OFF.** With the flag
   off, the existing pipeline must behave **bit-for-bit as today.**
4. **Small, reviewable steps**; tests alongside each step; no drive-by refactors.
5. **Stop and ask** when the docs and the code disagree, or when a decision in §7 is unresolved.

**Read order:** this brief → `03_BES_PROPOSAL.md` (the idea) → `02_ARCHITECTURE_SPEC.md` (the
system) → `01_TRANSCRIPT_AV.md` (source of truth / nuance) → then the codebase.

---

## 1. What the three reference docs are

These were reconstructed from a whiteboard walkthrough (video `IMG_5104.MOV`) by the team. They
describe the **intended** architecture and the BES proposal — they are the design intent, **not**
a description of the current code.

| Doc | What it is | Use it for |
|---|---|---|
| **`01_TRANSCRIPT_AV.md`** | Full audio+visual transcript of the walkthrough: timestamped spoken transcript (grouped by topic), a complete whiteboard inventory (every box/arrow/note + ASCII diagrams), and a synchronized "said ↔ shown" table. Includes a flagged "uncertain readings" list. | Ground truth for *what was actually said/drawn*; resolve ambiguity here. Quotes are verbatim. |
| **`02_ARCHITECTURE_SPEC.md`** | Engineering spec of the **current** system + **proposed** new infra: components (RLM orchestrator, comprehension, baseline builder, experiment runner, configurable GPU backend, PaperBench rubric grader, report), data/control flow, rubric & scoring (6 criteria, dynamic leaf nodes, pass ≥ 0.6), known issues, and the proposed BES enhancements. Items marked **(PROPOSED)/(TBD)** are not built yet. | The map of the system. Confirm each component against real code during recon. |
| **`03_BES_PROPOSAL.md`** | Standalone proposal for BES: motivation, the mechanism (goal tree → forward search/evolve → back pass) with diagram, five integration ideas, trade-offs, and the open decisions for reviewers. | The spec for what you're building. §4–§5 below operationalize it. |

**Canonical facts to carry forward (from the docs):**
- Pipeline today: `paper → Orchestrator (RLM) → comprehension (ingest/plan/understand via sub-agents) → baseline (Docker env + training script) → run experiments (configurable GPU backend) → grade vs PaperBench rubric → report`.
- Rubric: **6 criteria**, static weights, **dynamic leaf nodes (2–5/category)** defined per-paper by the orchestrator; **pass threshold = overall score ≥ 0.6**; failures are raised **only** on not meeting the min rubric threshold.
- Pain points BES targets: **runs constantly failing**, **excessive movement between phases**, **no tests on the training script**, expensive run-and-fail backend loop, sub-agent **cost** (subscription burn).
- Compute backend is **configurable / swappable** (RunPod / local A5000-class cluster / Azure) — BES must stay backend-agnostic; the on-board GPU cost table is illustrative only.

---

## 2. Recon — locate the integration points (fill these in for THIS repo)

Before designing, find and record (file paths + the function/class that owns each responsibility).
Treat every row as **unknown until verified in code.**

| # | Integration point to locate | What you need to know |
|---|---|---|
| R1 | **Orchestrator / RLM entry** | Where a paper-repro run is driven; how the single context window + "sub-agent variables" are managed; model/provider. |
| R2 | **Sub-agent spawn mechanism** | API calls vs. Claude Code processes vs. a framework; is there a shared Agent/Tool interface to reuse? |
| R3 | **Baseline builder** | Where the Docker env is created and the **training script** is generated; is output files-on-disk, a template, or freeform? (This is Milestone 1's hook.) |
| R4 | **Experiment runner / job abstraction** | How a run is dispatched to compute; sync vs. queued; the backend wrapper(s). |
| R5 | **Rubric grader** | How the rubric tree + dynamic leaf nodes are represented and scored; deterministic vs LLM-judge; where 0.6 gate lives. |
| R6 | **Run state & persistence** | The data model for a run; where state/artifacts/logs live; resumability/idempotency. |
| R7 | **Config / feature-flag system** | How flags/config are read (env, config file, settings module) — Milestone 1 needs one. |
| R8 | **Cost/budget controls** | Any existing caps on sub-agent fan-out, tokens, or $ per paper. |
| R9 | **Metrics & tracing** | What's measured today (pass-rate, $/paper, wall-clock, # failed runs) and how it's logged. |

**Deliverable of recon:** a short "integration map" (R1–R9 → real paths) appended to your plan,
plus any contradictions with the docs flagged for the maintainer.

---

## 3. BES in one screen (operational recap)

Full detail in `03_BES_PROPOSAL.md` §3. Core loop:

```
input ─► GOAL TREE  ◄── back pass: decompose into SUB-GOALS (+ scoring criteria)
            │              (complete → return ; else → loop)
            ▼
       FORWARD SEARCH: generate candidates → SELECT valid → EVOLVE (combine/splice partials)
            │
            ▼
       BACK PASS: update candidate pool · mark off sub-goals
            │   stuck → refine goal tree ; complete → return
            └► loop
```

**Integration philosophy:** BES is a **search/orchestration strategy**, not a model. Build it as a
**self-contained engine with a narrow interface** the existing orchestrator can call when the flag
is on — do **not** entangle it with `isinstance`-style branching across the pipeline. The engine
should be agnostic to: the LLM provider, the compute backend (R4), and the concrete "candidate"
representation (injected, not hard-coded).

---

## 4. Milestone 1 — Baseline BES + TDD harness (build this first, behind a flag)

**Goal:** replace the current *one-shot training-script* baseline with a **BES-driven, test-first**
baseline so errors are caught **before** expensive runs — **only when the flag is on.**

**Target behavior:**
1. At the baseline stage (R3), if `BES_ENABLED` (name per R7) is on, route baseline construction
   through the BES engine instead of the one-shot generator.
2. **Back pass / decompose:** turn the paper's baseline goal into a **goal tree** of sub-goals with
   per-sub-goal **scoring criteria**. (Reuse the orchestrator's understanding output where possible.)
3. **TDD gate:** for each sub-goal, derive **tests first** (see §7-D for what "test" means here) and
   require them to pass **before** the candidate is accepted — i.e., gate *before* committing GPU
   compute (R4).
4. **Forward search:** generate multiple candidate implementations, **select** those that pass their
   tests/criteria, **evolve** (combine/splice partial successes) into new candidates.
5. **Back pass / update:** update the candidate pool, mark off satisfied sub-goals; if **stuck**,
   refine the goal tree; if **complete**, emit the baseline artifact the existing runner expects.
6. Hand the resulting baseline to the **unchanged** downstream runner + rubric grader.

**Design constraints:**
- **Interface boundary:** define a single entry like `build_baseline_bes(paper_ctx, budget, backend) -> BaselineArtifact`, returning the **same artifact type** the current baseline produces (discover it in R3) so the downstream pipeline is untouched.
- **Data model:** `GoalTree`, `SubGoal{criteria}`, `Candidate{repr, scores}`, `CandidatePool` — keep `Candidate.repr` **abstract/injected** (R3 tells you if it's files/diff/patch).
- **Budget:** accept an explicit budget (max iterations, max candidates, token/$ cap per R8) and **fail closed** when exceeded, with an actionable error.
- **Determinism:** thread a seed through candidate generation/selection; record config + code SHA + the BES decision trace in run artifacts (R6).

**Feature-flag contract:**
- Default **OFF**. Flag off ⇒ existing baseline path runs unchanged (add a regression test asserting parity).
- Flag on ⇒ BES path; on unrecoverable BES failure, **fall back** to the legacy baseline (configurable) and log why — never crash the pipeline silently.

---

## 5. Future milestones (expansion — sketch only; do not build until M1 lands)

Each is independently flaggable. See `03_BES_PROPOSAL.md` §4 and `02_ARCHITECTURE_SPEC.md` §8.

- **M2 — Validation agent:** an agent **independent of the RLM orchestrator** (separate context/account) that vets the plan/goal tree for hallucination before compute is spent. Blocking vs advisory = §7.
- **M3 — BES rubric leaf nodes:** generate the rubric's dynamic leaf nodes (R5) as BES sub-goals via sub-agents. Watch **cost** (R8).
- **M4 — Forward-pass pooling on failures:** pool accumulated run failures, batch them to the training script for a refined fix. Trade-off: slower feedback.
- **M5 — Evolution / innovation:** use the evolve step to improve on the paper's method, not just reproduce it.

---

## 6. Guardrails & non-negotiables

- **Never break the flag-off path.** Regression test it.
- **No hard-coded** providers, GPU types, shapes, or paths in the BES engine — inject/config them.
- **Cost is a first-class constraint.** Every fan-out (sub-agents, candidates) is bounded by an explicit budget; surface spend in the run trace.
- **Fail fast & closed** at boundaries with actionable messages (`f"BES budget exceeded: {n} candidates > cap {cap}"`). No `except: pass`, no silent fallback that hides drift.
- **Tests with every step:** unit (goal-tree/candidate/pool logic), integration (the `build_baseline_bes` contract), regression (flag-off parity).
- **Surgical edits:** match existing style; don't refactor unrelated code; mention dead code, don't delete it.
- **Honor the docs' uncertain-readings list** (`01_TRANSCRIPT_AV.md` bottom) — confirm those points with the maintainer rather than treating them as fact.

---

## 7. Open design decisions (resolve during recon / with the maintainer)

These were intentionally left open; pin them down before/while implementing M1:

- **A. BES engine placement & build-vs-adapt** — standalone module vs. extend orchestrator; build from the proposal vs. port the BES paper's reference impl.
- **B. Candidate representation** — what a "candidate solution" concretely is (training-script files / diff / patch / plan); this defines how "combine + evolve / splice" is implemented (text vs AST vs file-level).
- **C. Scoring source** — does BES reuse the PaperBench rubric (R5) to score candidates, or a lighter sub-goal-level score? Deterministic vs LLM-judge.
- **D. What "test" means in the TDD gate** — static checks (imports/shapes), component unit tests, a fast smoke run on tiny data, or a mix; who authors them (deterministic generator / agent / derived from sub-goals).
- **E. Termination thresholds** — concrete "complete" and "stuck" conditions (max iters, no-improvement rounds, budget caps).
- **F. Backend & budget** — confirm the compute-target abstraction (R4) BES hands off to, and the exact caps (R8).

> **SDAR** (referenced by Marcus) is used in the proposal only as an *example* of sub-goals
> ("coding logic", "architecture"); it is not a dependency.

---

## 8. Definition of done (Milestone 1)

- [ ] Recon map (R1–R9 → real paths) produced and reviewed.
- [ ] `BES_ENABLED` flag added; **flag-off parity** proven by a regression test.
- [ ] BES engine (goal tree / forward search / evolve / back pass) implemented behind the flag with the TDD gate firing **before** compute.
- [ ] `build_baseline_bes(...)` returns the **same artifact type** the legacy baseline returns; downstream pipeline unchanged.
- [ ] Budget caps enforced (fail-closed) + BES decision trace in run artifacts.
- [ ] Unit + integration + regression tests green; one end-to-end smoke run on a known paper with the flag **on**.
- [ ] **Measured against current harness** on a small paper set: pass-rate, $/paper, wall-clock, # failed runs (R9) — reported so the team can decide whether to widen rollout.

---

*Provenance: this brief and the three reference docs were generated from `IMG_5104.MOV`
(team whiteboard walkthrough, 2026-06-05). Treat the docs as design intent; treat the code as truth.*
