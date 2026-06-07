# Architecture Spec — Automated Research-Paper Reproduction System

**Status:** Draft v0.1 — reconstructed from the whiteboard walkthrough in `IMG_5104.MOV` (2026-06-05).
**Scope:** Documents the **current** system and the **proposed** "new infrastructure." Items the
team flagged as not-yet-built are marked **(PROPOSED)** or **(TBD)**. Source quotes/visuals: see
`01_TRANSCRIPT_AV.md`.

---

## 1. Purpose

Given a research paper as input, the system autonomously **reproduces** it: comprehends the
paper, builds a runnable baseline, runs the experiments on GPU, and grades the result against a
**PaperBench**-derived rubric — producing a report when the reproduction clears a quality bar.
The objective is a reliable, mostly-hands-off pipeline from *paper → reproduced result*.

---

## 2. System overview

```
 paper ─► Orchestrator (RLM) ─► Comprehension ─► Baseline ─► Run Experiments ─► Rubric grading ─► Report
                │                  (ingest,        (Docker env +    (GPU:            (PaperBench,        │
                │                   plan,           training         RunPod/Server)   6 criteria,        ▼
                └── sub-agents ◄──  understand)     script)                           pass ≥ 0.6)   improvement (TBD)
                    (sub-agent
                     variables)
```

---

## 3. Components (current)

### 3.1 Orchestrator (RLM)
- The control plane for the whole pipeline. Implemented as an **RLM** — a model with **one
  context window** that spawns **sub-agents** whose outputs come back as **"sub-agent variables"**
  in that context.
- **Why RLM:** explicitly chosen to mitigate **context rot / hallucination** over a long
  multi-phase job. (Consistent with the "Recursive Language Model" pattern; the team notes it
  "serves to fix" context rot but remains a concern.)
- Owns paper-specific decisions, including defining the dynamic rubric leaf nodes (§5).

### 3.2 Comprehension phase
- **Ingest:** load the input paper.
- **Plan → Understand:** the orchestrator plans, then dispatches **sub-agents** to understand
  the paper; each returns a sub-agent variable into the orchestrator's context.

### 3.3 Baseline builder
- **1. Create environment** — provisions an isolated env **inside Docker**.
- **2. Implement details** — generates the **training script** that realizes the paper's method.
- *Current behavior:* the baseline is produced largely as a **one-shot training script** that
  downstream stages accept as-is (flagged as a weakness — see §7).

### 3.4 Experiment runner
- Executes the baseline/training script. **This is where most wall-clock time goes** — "the
  agents are just writing a lot of code."
- Internal loop observed on the board: **Run → Pass → Verify → Run Tests**, looping against the
  rubric.

### 3.5 Compute backend (configurable)
- Experiments run on **GPU through a configurable compute layer**. The board says "RunPod or a
  server," but **"RunPod" is a stand-in for whatever compute is on hand** — e.g. a managed cloud
  (RunPod), a **local GPU cluster** (e.g. A5000s), or **Azure cloud GPUs** — selected per the
  compute available at the time. **Treat the backend as a swappable target, not a fixed vendor or
  GPU.**
- **Data Loader → Training script** with **three training stages: S1, S2, S3 (~1 hr each)**.
- The board also has a back-of-envelope **GPU cost comparison** (H100 / RTX 6000 / B200). It is
  **illustrative only — not a committed hardware choice** — so it is recorded in the transcript
  (`01_TRANSCRIPT_AV.md` §B6) rather than treated as a design decision here.

### 3.6 Rubric grader (PaperBench)
See §5.

### 3.7 Report
- On a passing run (overall rubric score **≥ 0.6**), the system emits a **report**.
- A downstream **improvement** stage is desired but **TBD**.

---

## 4. Data & control flow (current)

1. **Input:** paper → **ingest** (comprehension).
2. **Plan/understand** via sub-agents → sub-agent variables in orchestrator context.
3. **Baseline:** create Docker env → implement training script.
4. **Run experiments** on GPU (RunPod/server), Data Loader feeding training stages S1–S3.
5. **Grade** the run against the PaperBench rubric.
6. **Branch:**
   - **Pass** (score ≥ 0.6) → **Report** → (improvement, TBD).
   - **Fail** (below min threshold) → error; re-run / move between phases (the costly loop, §7).

---

## 5. Rubric & scoring

- **Origin:** rubric is **from PaperBench**; experiments "run against this."
- **Six main criteria** (static set; **static weights**):
  1. Method-to-Code
  2. Data + Preprocessing
  3. Experiment reproducibility
  4. Artifact completeness
  5. Metric eval
  6. Results match
- **Static vs. dynamic:** the **criteria and their weights are static**; the **leaf nodes** that
  actually produce the score are **dynamic** — **2–5 leaf nodes per category**, **defined by the
  orchestrator per paper**.
- **Modifier:** "Static **with paper-specific tailored guidelines.**"
- **Pass threshold:** overall score **≥ 0.6**.
- **Failure semantics:** an **ERROR is raised only when a run does not meet the minimum rubric
  threshold** (i.e., grading drives the fail signal).
- **(PROPOSED)** Generate the dynamic leaf nodes via **BES sub-goals** (sub-agents create the
  sub-goals) — see §8.4.

---

## 6. Infrastructure & cost

- **Compute:** GPU via a **configurable backend** (RunPod / local A5000-class cluster / Azure
  cloud GPUs — whatever compute is on hand; §3.5). Specific GPU/vendor is **not** fixed; the
  on-board cost table is just one illustrative comparison point, not a decision.
- **Agent/LLM cost:** sub-agents currently run on **cloud OAuth / Claude Code account
  subscriptions**, and the team is **burning through usage**. Cost scales poorly with more
  sub-agents — a primary constraint on any design that fans out (§8).
- **Latency:** "errors take hours"; the run/fail loop dominates wall-clock.

---

## 7. Known issues (current)

| # | Issue | Notes |
|---|---|---|
| 1 | **Compute cost** | GPU + heavy agent usage; subscription usage being exhausted. |
| 2 | **Hallucination / context rot** | RLM helps but remains a concern. |
| 3 | **Runs constantly failing** | The dominant pain point. |
| 4 | **Excessive movement between phases** | Lots of back-and-forth, especially around baseline ↔ run. |
| 5 | **No testing on the training script** | Failures surface only at run time, after expensive code-gen. |
| 6 | **Expensive backend loop** | Constantly running and failing is computationally expensive. |

---

## 8. Proposed new infrastructure

The proposal centers on a **new harness built around BES (Bidirectional Evolutionary Search) +
TDD**, to catch errors *before* they happen rather than discovering them via repeated failed
runs. Full detail in `03_BES_PROPOSAL.md`; summary of architectural deltas below.

### 8.1 BES + TDD harness at the baseline stage **(PROPOSED)**
- Replace the **one-shot training-script** baseline with a **BES-driven, test-first** process:
  decompose the goal (back pass → sub-goals), search candidate implementations (forward search),
  select/evolve valid ones, and **gate with tests (TDD)** before running expensive experiments.
- **Trade-off:** more expensive **upfront**, but reduces the run/fail back-and-forth → **cheaper
  and more reliable long-run**.
- Targets issues #3, #4, #5, #6.

### 8.2 Validation agent **(PROPOSED)**
- A **separate** agent, **independent of the RLM orchestrator**, that **double-checks the plan**
  before execution to catch hallucination / bad plans. Independence is the point — it should not
  share the orchestrator's context. Targets issue #2.

### 8.3 Forward-pass + pooling on run failures **(PROPOSED)**
- Instead of fixing failures one-by-one, **pool/accumulate run failures**, then **batch** them
  back to the training script for a refined fix.
- **Trade-off:** must wait for failures to accumulate → potentially **slower** per-fix, but fewer
  total iterations.

### 8.4 BES inside the rubric **(PROPOSED)**
- Use BES to **generate the rubric's dynamic leaf nodes** as sub-goals (via sub-agents).
- **Trade-off:** **cost** — more sub-agents → more spend.

### 8.5 Forward-pass for evolution / innovation **(PROPOSED)**
- Use the BES **evolve** step not just to repair but to **iterate on / improve the paper's
  method** ("innovation"), beyond plain reproduction.

---

## 9. Open questions / concerns (team-stated)

- **Cost** — especially if the design fans out into many sub-agents (subscription usage already
  strained).
- **Speed** — primarily the **forward pass** (and the batch-and-wait pooling in §8.3).
- **Effectiveness** — "we won't really know until we test."
- **The ask to reviewers:** *Do you agree or disagree with adding BES, and where should we go
  from here?*

---

## 10. Glossary

- **RLM** — orchestrator model: one context window + sub-agent variables; used to limit context
  rot (consistent with "Recursive Language Model").
- **BES** — **Bidirectional Evolutionary Search**; an external **search** paper being repurposed
  as harness infrastructure (see `03_BES_PROPOSAL.md`).
- **TDD** — Test-Driven Development; write tests first to prevent errors pre-run.
- **PaperBench** — source of the reproduction rubric.
- **SDAR** — a paper (referenced by Marcus) used as an example for BES sub-goals (e.g., "coding
  logic", "architecture").
- **Sub-agent variables** — outputs of sub-agents surfaced back into the orchestrator's single
  context window.
- **Leaf nodes** — the dynamic, per-paper scoring units under each static rubric category (2–5
  each), defined by the orchestrator.
