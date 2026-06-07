# Proposal — BES (Bidirectional Evolutionary Search) Harness

**Status:** Draft v0.1 for review.
**Source:** Last ~2 minutes of the whiteboard walkthrough `IMG_5104.MOV` (≈02:25–05:52),
plus the BES zone of the board. Companion docs: `01_TRANSCRIPT_AV.md`, `02_ARCHITECTURE_SPEC.md`.
**One-line ask:** *Do we agree to adopt BES (+ TDD) as a new harness for the paper-reproduction
pipeline — and where should we start?*

---

## 1. Motivation — why change anything

Today's pipeline reproduces papers by having the orchestrator write a baseline and then
**run experiments until they pass the rubric**. In practice that backend loop is the problem:

- **Runs are constantly failing**, and there's **a lot of movement between phases** (baseline ↔ run).
- It's **computationally expensive** — "errors take hours," and we're "constantly running and failing."
- There's **no testing on the training script**, so errors are only discovered *after* expensive code generation and a GPU run.
- Framed against how current LLMs search the solution space:
  - **Sparse signal** — pass/fail at the end gives little gradient to learn from.
  - **Narrow exploration** — one line of attack at a time; no systematic branching.

The thesis: **stop discovering errors by running, and start preventing them before running** — by
searching the solution space more deliberately and gating with tests.

---

## 2. What BES is

**BES = Bidirectional Evolutionary Search**, taken from a **search** paper. It is designed for
search problems; we **change the architecture to work for our use case** (reproducing papers).
"Bidirectional" = it works **both** top-down (decompose a goal into sub-goals — the *back pass*)
**and** bottom-up (generate, select, and evolve candidate solutions — the *forward search*),
looping between the two. "Evolutionary" = candidate solutions can be **combined/spliced and
evolved** into better candidates.

> Abik has done the research on this and believes it's the way to go.

---

## 3. How BES works (mechanism)

```
 input
   │
   ▼
┌──────────────┐     ◄── back pass ── creates SUB-GOALS
│  GOAL TREE   │           e.g. (SDAR paper, per Marcus):
│  • complete  │              1. coding logic
│      → return│              2. architecture
│  • else      │           (PROPOSED: sub-agents generate the sub-goals)
│      → loop  │
│  • gives the │
│    SCORING   │
│    CRITERIA  │
└──────┬───────┘
       ▼
┌────────────────────────────────────┐
│  FORWARD SEARCH                     │
│   • generate candidates [1][2][3]   │   ← multiple solutions in parallel
│   • SELECT the valid ones           │   ← scored vs. the criteria
│   • EVOLVE: "combine + evolve"      │   ← splice partial solutions:
│       (cand A's first half +        │      e.g. A correct on part 1,
│        cand B's second half → new)  │           B correct on part 2 → new candidate
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐         ┌─────────────────────┐
│  BACK PASS                          │  stuck  │ refine the GOAL TREE │
│   • update the CANDIDATE POOL       │────────►│ (re-decompose)       │
│   • mark off completed SUB-GOALS    │         └─────────────────────┘
└──────────────┬─────────────────────┘
               │  complete → return
               └───────────────► (loop continues until goals are met)
```

**Cycle in words:**
1. **Back pass (decompose):** turn the input into a **goal tree** of **sub-goals**, with a
   **scoring criteria** for each.
2. **Forward search (build):** generate multiple candidate solutions, **select** the valid ones,
   and **evolve** them by **combining/splicing** partial successes into new candidates.
3. **Back pass (update):** **update the candidate pool** and **mark off** satisfied sub-goals.
4. **Loop / refine / return:** if **stuck**, **refine the goal tree**; if **complete**, **return**.

---

## 4. How we'd apply BES to the pipeline

### 4.1 Primary: a new harness at the **baseline stage** + **TDD**
- The baseline stage has "a lot of movement between here and there"; that's where BES goes first.
- **New harness = BES + TDD:** decompose into sub-goals, search/evolve candidate implementations,
  and **write tests first** so errors are caught **before** the expensive run.
- **Economics:** **more expensive upfront**, but it **reduces the constant run-and-fail back-and-
  forth**, so it should be **cheaper and more reliable in the long run.**

### 4.2 Add a **validation agent** (independent of the orchestrator)
- A separate agent — **completely separate from the RLM orchestrator** — that **double-checks the
  plan** for soundness and hallucination before we commit compute to it.

### 4.3 Use BES **inside the rubric** (leaf nodes)
- Generate the rubric's **dynamic leaf nodes** as BES **sub-goals** (via sub-agents), creating
  sub-goals for each leaf goal.
- **Caveat:** **cost** — more sub-agents means more spend.

### 4.4 Forward-pass + **pooling** on run failures
- Rather than fixing failures individually, **pool/accumulate failures**, then **batch** them back
  to the training script for a refined fix.
- **Caveat:** we must **wait for failures to accumulate** → potentially **slower** per fix.

### 4.5 Forward-pass for **evolution / innovation**
- Use the **evolve** capability not only to repair but to **iterate on and improve** the paper's
  method — i.e., **innovation / re-iterating the paper**, beyond plain reproduction.

---

## 5. Expected benefits

- **Fewer failed GPU runs** (errors caught pre-run by TDD + validation) → lower compute spend
  over time.
- **Less phase thrashing** — structured goal tree replaces ad-hoc back-and-forth.
- **Denser search signal** — per-sub-goal scoring instead of one terminal pass/fail.
- **Broader exploration** — multiple candidates + evolutionary splicing instead of a single line
  of attack.
- **A path to innovation**, not just reproduction (via the forward/evolve pass).

---

## 6. Risks & open concerns (team-stated)

| Concern | Detail | Mitigation to explore |
|---|---|---|
| **Cost** | Many sub-agents → more spend; already burning through **Claude Code subscription** usage. | Cap sub-agent fan-out; cheapest GPU (RTX 6000, $11.40/run); reuse candidates across sub-goals. |
| **Speed** | Forward pass is slow; **pooling** adds wait-for-accumulation latency. | Tune batch size / pooling window; parallelize forward search. |
| **Effectiveness** | "We won't really know until we test." | Run a scoped A/B vs. current harness on a fixed paper set; measure pass-rate, $/paper, wall-clock. |
| **Upfront cost** | BES + TDD is more expensive before the first run. | Justify against the long-run reduction in failed runs. |

---

## 7. The decision we're asking for

1. **Adopt BES + TDD as the new harness** at the baseline stage? (agree / disagree)
2. Which of the secondary ideas to pursue, and in what order:
   - validation agent (§4.2),
   - BES-generated rubric leaf nodes (§4.3),
   - forward-pass pooling on failures (§4.4),
   - forward-pass for evolution/innovation (§4.5)?
3. **Where should we start** — and what would make this convincing enough to commit compute to?

---

## 8. References / terms
- **BES** — Bidirectional Evolutionary Search (external search paper; being repurposed).
- **TDD** — Test-Driven Development (write tests first; gate before running).
- **SDAR** — paper referenced by **Marcus**; used as the worked example for sub-goals
  ("coding logic", "architecture"). *(Acronym not expanded on the board/audio.)*
- **RLM orchestrator**, **PaperBench rubric (pass ≥ 0.6)**, **candidate pool**, **goal tree** —
  see `02_ARCHITECTURE_SPEC.md`.

> **Uncertain readings** carried over from the recording: "BES" spelled on the board as
> "Bidirectional Ev. Search" (audio confirms "Evolutionary"); SDAR sub-goal examples and the
> "combine + evolve" box are hand-drawn and lightly paraphrased. See the flagged list in
> `01_TRANSCRIPT_AV.md`.
