<!-- doc-meta: status=proposed; last-verified=2026-07-13 -->
# Reproduction capability — closing the gap to "almost all ML papers" — design

> **Status: PROPOSED.** Written 2026-07-13 after the twelve-subsystem audit and the
> Phase-0 remediation. This is the "why we still can't reproduce most papers, and what
> it would actually take" spec. It is deliberately about **capability**, not plumbing —
> the plumbing is now in decent shape and is not the bottleneck.
>
> Companion docs: `docs/history/specs/2026-07-07-deterministic-any-paper-execute-mode-design.md`
> (the execute-mode substrate this builds on), `learn.md` (the reliability rules),
> root `CLAUDE.md` (the invariants).

## 1. Problem

**In 14 months, zero modern training-heavy papers have been reproduced end-to-end
through the harness.** Not one. The record:

| Paper | Class | Result |
|---|---|---|
| Adam, All-CNN, ResNet, VAE | tiny, pre-2016, self-contained, 1 GPU | reproduced — **locally** |
| OmniZip | modern, **inference-only** | reproduced (thinnest evidence trail of any run) |
| SDAR | modern, agentic RL, 3 models × 3 envs | **partial 0.363; headline claim NOT reproduced** |
| 6 × recent GKE runs | modern, cloud | **all died before training started** |

The one genuine SDAR training signal in the repo — `0.456` — came from a **human
hand-running the authors' code**, bypassing the agent entirely. The agent's own
from-scratch SDAR implementation scored **0.0**.

This is not a plumbing failure. The audits confirm the harness is *honest*: the
campaign scored itself `0.000 / EXHAUSTED` rather than ship a fake, and the evidence
gate has never been caught passing a fabrication. The team's own audit says it
plainly: *"the harness is flawless and honest… the reproduction bottleneck is
per-attempt agent quality."*

**The bottleneck is that an LLM cannot author a faithful implementation of a modern
paper from the paper alone.** It can re-derive Adam. It cannot re-derive GRPO with
sigmoid-gated OPSD across ALFWorld/WebShop/Search-QA and land the numbers.

Every guard we have added for a year has been a guard against *the agent's own bad
output*. That is treating the symptom.

## 2. Goal / non-goals

**Goal.** A paper is reproducible by this system only if **all ten** of the following
hold. Today, seven of them do not. This spec is the plan to close them.

| # | Requirement | Today | Owner phase |
|---|---|---|---|
| R1 | Read the paper faithfully (tables, equations, appendices) | ❌ no table structure; LaTeX source never fetched | P1 |
| R2 | Know what to check, deterministically | ❌ `rubric_gen` emits no `check_kind` ⇒ **every leaf is an LLM opinion** | P1 (in flight) |
| R3 | Get the authors' code | ❌ repo found only if its URL is literally printed in the PDF | P1 |
| R4 | Get the data (incl. gated/private/large) | ⚠️ `HF_TOKEN` now reaches pods; gated/Kaggle/private unbuilt | P2 |
| R5 | Build a working environment | ⚠️ works often enough | — |
| R6 | Run at the paper's scale | ❌ no multi-node (`num_machines:1` hardcoded); no mid-training resume | P3 |
| R7 | Evaluate the way the paper did | ❌ no lm-eval/HELM adapter — most modern LLM papers report on standard benchmarks | P2 |
| R8 | Compare honestly | ✅ fixed 2026-07-13 (was circular: target floored to the agent's own prior best) | done |
| R9 | Give up cheaply when hopeless | ❌ no scope gate; cut-losses comparator unwired | P1 (in flight) |
| R10 | Measure the aggregate rate | ❌ **no reproduction-rate harness — the KPI is uncomputable** | P1 |

**Non-goals.**
- Making the agent a better from-scratch implementer. That is a model-capability bet with
  a 14-month record of zero wins. We stop making it. (It remains the fallback for papers
  with no published code — see R3 — but it is not the product.)
- Reproducing papers with no code AND no data AND no standard benchmark. Those get a fast,
  cheap, honest **INFEASIBLE** (see R9) rather than a burned budget.

## 3. Architecture — the thesis

> **Stop asking the model to re-derive the science. Ask it to *acquire, run, and verify*
> the science, and make the harness the thing that is rigorous.**

The model is good at: finding the repo, reading its README, wiring an entrypoint, mapping
the authors' output format onto our metrics schema, diagnosing a stack trace.
The model is bad at: reimplementing a novel training algorithm correctly from prose.

So the pipeline becomes **acquire → execute → verify**, with the LLM confined to the
acquire/adapt joints and the deterministic layer owning the verdict:

```
paper ──▶ [R1 ingest: LaTeX+tables]
             │
             ├──▶ [R2 rubric with deterministic check_kind]  ─────────┐
             │                                                        │
             ├──▶ [R3 repo acquisition: search, not regex] ──┐        │
             │                                               ▼        │
             ├──▶ [R4 data acquisition: gated/HF/Kaggle] ──▶ EXECUTE  │
             │                                               (authors'│
             ├──▶ [R7 eval harness: lm-eval/HELM adapter] ──▶ code)   │
             │                                               │        │
             └──▶ [R9 scope gate: cheap INFEASIBLE] ─X       ▼        ▼
                                                        deterministic verdict
                                                               │
                                                               ▼
                                                        [R10 rate harness]
```

**R9 is the economic keystone and is currently missing entirely.** For a
needle-in-haystack triage product, the cheapest correct answer — *"we cannot reproduce
this, don't spend"* — is a first-class product output, not a failure. Today the system
cannot produce it: a theory paper with zero experiments yields `raw_score=0.0` with
`insufficient_coverage=False`, i.e. it is scored as a *bad reproduction* rather than a
*non-applicable paper*.

## 4. Contracts

### R1 — ingestion fidelity
- Fetch `arxiv.org/e-print/<id>` (LaTeX source) **in addition to** PDF/ar5iv HTML. It is the
  only lossless source and is currently never requested. When present, tables/equations/
  algorithms come out exactly.
- Add a first-class `Table` type to `parser/model.py` (there is none — a 5-column results
  table currently becomes an unbound run of digits). PDF fallback: `page.find_tables()`.
- Ground every extracted claim: `paper_text.find(span)` must succeed or the claim is
  rejected. Today the "blinded re-extraction" anti-hallucination pass re-reads the LLM's
  *own* quoted span and never checks it exists in the paper.

### R2 — checkable rubric  *(landed 2026-07-13, behind `OPENRESEARCH_DETERMINISTIC_LEAVES`)*
- `rubric_gen` emits `check_kind` + `assertion` for hyperparameter / artifact-existence /
  numeric-result leaves, which routes them to the pure-Python `deterministic_leaf_checker`.
- **Bias: emit NO annotation rather than a guessed one.** A hallucinated assertion
  deterministically fails a *correct* reproduction — a false negative, the expensive error.
  Four one-directional refusal gates (structure / grounding / vocabulary / specificity) can
  only ever decline to annotate; a declined leaf is LLM-graded exactly as before.
- Judgment leaves ("is the method faithfully described") stay LLM-graded. That is correct —
  VAE scores ~4% deterministic *because it is genuinely all method-fidelity*, not because the
  system failed. The honest ceiling was never 100%.
- **Paper-declared coefficients (β, λ, gate thresholds, loss weights) are now deterministic too.**
  These *look* interpretive but are mechanical — `β=10` is a number you used or didn't — so
  moving them onto the evidence layer puts **method fidelity itself**, not just bookkeeping
  hyperparameters, under deterministic check. `provenance.json` carries a first-class
  `coefficients` block; a leaf addresses `coefficients.<name>`; the implementer is instructed
  (deterministically, from `rubric_gen.coefficient_fields`) to emit exactly the set the grader
  will check. Located-and-wrong (`β=1.0` vs paper's `10`) → deterministic 0.0 (the surrogate
  catch). Guards that keep it false-negative-safe: `on_missing→LLM` (absent ⇒ LLM, never
  auto-zero), a **CONTESTED gate** (when the authors' code and the paper text disagree — SDAR's
  scripts use β=5/λ=0.01 vs text β=10/λ=0.1 — the symbol drops to LLM so an execute-mode run
  faithful to the code is never zeroed for matching it), a ROLE gate, and **no range-checking**
  (so `alpha=0.0` ablations stay valid — the 2026-07-07 incident is regression-pinned).
- **Measured coverage** on the 10 real `best_runs/` rubrics (235 leaves): **~7%** on today's
  packed historical leaves → **~28% aggregate** once the atomic-split prompt (one constant per
  leaf, which is what PaperBench leaves are supposed to be) is on (All-CNN 35%, ResNet 43%,
  SDAR 6–22%). The rest is genuine method-fidelity that stays LLM-graded. The number to move is
  *leaf atomicity*, not the checker.

### R3 — repo acquisition
- Replace regex-URL-in-paper-text with real resolution: search by title + authors + venue
  (Papers-with-Code, GitHub search, the arXiv abstract page's code links).
- Rank candidates by (author match × star count × recency × does-it-contain-the-method-name).
- Record provenance: which repo, which commit, how it was found. A wrong repo is worse than
  no repo.
- The root model is currently **denied web/code search** (`WebFetch`/`WebSearch` sit in
  `_ROOT_DISALLOWED_TOOLS`). Keep the root denied — do this deterministically in the harness,
  outside the prompt-injectable REPL.

### R4 — data acquisition
- First-class `acquire_dataset(spec)` primitive with a credential-brokered path for gated HF,
  Kaggle, and direct-URL corpora. Today the generated `train.py` just calls `load_dataset()`
  and hopes.
- A gated dataset we cannot access must produce a **verified `Exclusion`** (an honest,
  disclosed gap that is removed from numerator *and* denominator), not an uncontrolled
  in-pod training error after paying for the pod.

### R7 — standard evaluation
- Adapter: run `lm-eval-harness` / HELM / `bigcode-eval` and map their output into our
  `metrics.json` schema. Today these exist **only as skill markdown** — no runner, no adapter.
- This unlocks the largest single class of modern papers (anything reporting MMLU / HumanEval /
  GSM8K / HELM), which are today unscoreable **regardless of whether we reproduce them**.

### R9 — scope gate + cut-losses
- **Scope classifier, pre-spend.** Classify: `empirical-ML-with-code` / `empirical-ML-no-code` /
  `theory` / `survey` / `human-eval` / `out-of-domain`. Only the first two proceed
  unconditionally; the rest need explicit operator opt-in or return INFEASIBLE at ~$0.
- **Feasibility estimate, pre-spend.** Compute required GPU-hours from the paper's own stated
  compute. If it exceeds the tier budget → offer a scaled-down rung, or INFEASIBLE. Never
  silently attempt a paper we cannot afford.
- **Cut-losses, mid-flight.** Wire `doomed_run_comparator` (built; deliberately connected to
  nothing) into the campaign AWAIT stage. Conservative thresholds — killing a healthy-but-slow
  run is a false negative.

### R10 — the rate harness  *(the instrument)*
- A runner that takes a **paper set**, runs each through the tiered pipeline, and computes an
  aggregate reproduction rate with a standard error.
- Honest roll-up: report `compute_adjusted_score` and **leaf coverage**, never the
  excluded-leaf headline. (`best_runs/README.md` currently states: *"Failed-or-skipped leaves
  are excluded from the roll-up rather than being scored as zero"* — which is how ResNet ships
  `verdict=reproduced` at a 0.368 compute-adjusted score.)
- Replace the **synthetic** vendored PaperBench rubrics (`third_party/paperbench/README.md:19`)
  with the real upstream artifacts, or the numbers aren't comparable to published baselines.
- `cli_paperbench run` is currently an explicit placeholder writing `mean_score: None,
  n_attempts: 0`, and `mean_standard_error()` is wired to nothing.

**Without R10 there is no way to know whether any of R1–R9 worked.** It is the instrument the
company steers by and it does not exist. It should land first.

## 5. Security precondition (blocking for public intake)

The root REPL is `environment="local"` — root-written Python `exec`s **in the orchestrator's
host process** — and retains `__import__`/`open`. Credentials are copied into that process's
`os.environ`. The paper is attacker-influenceable. Therefore **a prompt-injected PDF can
exfiltrate every API key and own the host.**

`docs/design/rlm-pivot-brief.md` §7 documents this honestly; it was an acceptable risk when
the input was a developer's own arXiv IDs. **It is not acceptable for a product that accepts
user uploads**, which is the entire deepinvent surface.

- **Mitigation (landed 2026-07-13, defense-in-depth only):** a credential vault scrubs the 27
  managed secrets out of the child's `os.environ` at the REPL boundary and hands them over an
  inherited pipe (never `env=`, so they never enter the child's `/proc/self/environ`). Proven
  with the real exploit: naive harvest of `os.environ` and of the child's `/proc/self/environ`
  both go from 3/3 keys to 0. This **raises the cost of the attack and closes the common/naive
  vectors. It does not close the class.**
- **Why it does not close the class (verified empirically, 2026-07-13).** Once an attacker has
  arbitrary `exec` + `__import__` in the process, the residuals are decisive and cannot be
  patched from inside that process:
  - **`/proc/<ppid>/environ`** — the parent (uvicorn) still holds every key in its exec
    snapshot. Tested on this host (`kernel.yama.ptrace_scope=1`, the *protective* setting): a
    child process **could still read its parent's `/proc/<ppid>/environ`**. `ptrace_scope`
    governs `ptrace()` memory attachment, not `/proc/<pid>/environ` reads for a same-UID child —
    so the sysctl the mitigation reasoning leaned on does not actually block this vector even at
    its strict value. In production, uvicorn is where the credentials live, so this is a live
    exfiltration path.
  - **`__import__("os").system(...)`** — arbitrary host RCE, wholly unaddressed.
  - **TOCTOU** on the vault's expose window; **`open(".env")`** in dev.
- **The only real fix — required before any public/untrusted intake:** run each reproduction in
  a disposable sandbox (container / microVM / gVisor) where the child has no `/proc` visibility
  into the orchestrator and no host filesystem. This converges with the in-cluster-Job
  durability work — the driver should be a Job anyway, since a killed driver is currently a
  durability failure (`learn.md`). **Until then, the operational control is the real one: do not
  expose public PDF upload.** Trusted arXiv IDs and operator-supplied papers match the threat
  model the system was designed against; arbitrary user uploads do not.
- **Engineering note on the treadmill.** The vault work was worth doing — it closes the vectors a
  casual injection would reach, and it is the right hygiene regardless. But every layer added
  this session surfaced the next residual, which is the signature of defending a process that has
  already granted the attacker code execution. Do not mistake "we added another mitigation" for
  "we closed the hole." The hole is the `exec`-on-host design; only isolation closes it.

## 6. Phasing

| Phase | Ships | Unlocks |
|---|---|---|
| **P1 — Instrument + honesty** | R10 rate harness · R2 deterministic leaves · R9 scope gate + cut-losses · R1 LaTeX/tables | We can finally **measure**, we stop paying for hopeless papers, and the grade stops being an LLM opinion |
| **P2 — Reach** | R3 repo search · R4 data acquisition · R7 eval-harness adapter | The large modern-paper classes become *attemptable at all* |
| **P3 — Scale** | R6 multi-node (Indexed Job/JobSet + torchrun c10d + Kueue gang-scheduling) · mid-training checkpoint/resume · sandboxed driver | The "deep" tier; papers beyond one box; survives preemption |

**The gating milestone, before any of this:** reproduce **one modern paper end-to-end on a
GCP GPU**. It has never happened. Until it does, every number in this spec is a projection.

## 7. Evidence red line (unchanged)

Nothing here weakens it. The fitness signal remains the deterministic evidence layer, never
an LLM grade. R2 *strengthens* it (fewer leaves are LLM-judged). R9's INFEASIBLE and R4's
`Exclusion` are honest, disclosed gaps — removed from numerator **and** denominator, never
scored as zeros, never scored as passes.

## 8. Open questions

1. **Paper set for R10.** Which corpus defines "almost all"? PaperBench's set, a deepinvent-
   curated patent-relevant set, or a random arXiv sample? This choice *defines the KPI* and
   should be made deliberately, not inherited.
2. **Execute-mode fidelity.** Running the authors' code proves *the claim reproduces*. It does
   **not** prove *the paper is reproducible from its text* — which is arguably the more
   patent-relevant question (can a skilled practitioner build this from the disclosure?). The
   two-axis verdict already models this; decide which axis the product sells.
3. **Multi-node ROI.** R6 is a genuine infrastructure project. Is the paper mix actually
   multi-node-bound, or does honest descoping (R9) cover the tail more cheaply?
