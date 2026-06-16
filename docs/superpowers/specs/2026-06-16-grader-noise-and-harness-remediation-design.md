# Grader-noise — June-2026 SOTA grounding & extension (companion)

**Role.** Companion to the design of record: **`2026-06-16-grader-fidelity-and-harness-remediation-design.md`** (DESIGN LOCKED, grilled Q1–Q6, from the four-agent code audit). **Read that first.** This doc does **not** re-design any locked workstream. It does three things the locked spec deliberately leaves out (it is a pure code-grounded audit with zero external citations):

1. **Grounds** each locked workstream (A1–A7, B, C, D, E) in the live June-2026 literature — a *confirm / temper / contradict* verdict per item, with real, page-verified citations.
2. **Surfaces net-new levers** the audit missed but the frontier supports.
3. **Adversarially stress-tests** the locked design against published *negative* results (where extra machinery is known to not pay).

**Correction to an earlier draft of this file (stated openly).** An earlier version of this file mis-attributed the **Adam 0.06** re-grade swing to LLM sampling noise. That delta (Adam **0.8164→0.8771**, +0.0607 on identical 24 leaves) was **G1-vs-old-grader** — a *grader code change* ([[grading-evidence-budget-fixes]] re-graded "G1 vs the old grader, SAME evidence"), not the instrument's re-test deviation. (All-CNN's G1 re-grade moved only ~0.018 and was leaf-set-confounded, so "0.06" is an **Adam-only** figure, not a general one.) The locked spec has the right numbers and this companion uses **only** those: pure same-grader, same-evidence drift **≈ 0.018** (`report.py:980`: All-CNN v3 verify#1 0.712 vs verify#2 0.694), **±0.09–0.18 run-to-run**. The error is instructive and is itself in the literature — self-assessment inflates confidence in one's own prior reasoning (Xu et al. 2402.11436); the fix was an adversarial reviewer, not more self-review.

**Provenance flags.** Every arXiv ID carries **[C]** (a research sub-agent fetched the page and verified title/ID/numbers on 2026-06-16) or **[U]** (on-topic, magnitudes unverified — do not cite numbers). Three parallel research sweeps (judge-noise, agentic orchestration, agentic reasoning), ~25 confirmed papers.

---

## 0. One-paragraph synthesis

The frontier strongly **validates the locked thesis** — *"the grade is a non-deterministic LLM call and almost everything else is compensating for it"* — and validates most of Workstream A's mechanics. The literature's three sharpest contributions on top of the locked design are: (i) **decomposition (A2) is a bigger variance/bias lever than resampling (A1)** — *by leverage, not by rollout slot*: the locked spec correctly ships A2 **last** (it needs rubric-gen annotations + reliable `provenance.json`, locked step 4) and already has both mechanisms, so the actionable nuance is a *cost-bounded cascade + recognizing A2 carries the accuracy win*, **not** a re-sequencing; (ii) the locked spec's **same-family grader (A5 keeps Sonnet) leaves self-preference bias uncorrected** — a genuine gap, since Sonnet grades Sonnet-authored code; (iii) the **anti-fabrication threat model** the audit only implies (A7 EVIDENCE_GATE) is, empirically, the single highest-value correctness lever in this whole space. Net: the locked spec is on the right side of the evidence; the additions are narrow.

---

## 1. The noise budget, decomposed (grounding the locked thesis)

The locked spec cites two numbers; the literature explains why they are *different* numbers attacking *different* fixes, which is worth making explicit because it tells you which workstream buys down which noise:

| Component | Magnitude (locked spec) | What it is | Locked fix that attacks it |
|---|---|---|---|
| **Pure grader noise** | **≈ 0.018** | same evidence, same grader, re-sampled | **A1** (median-of-N), **A5** (sampler-capable transport) |
| **Run-to-run swing** | **±0.09–0.18** | different runs → grader noise ⊕ *evidence-state* variation ⊕ leaf-set flips | **A3** (evidence fingerprint), **A4** (champion artifact), data-unavailable determinism |

- **Why ~0.018 is small but real, and why A1 is not guaranteed to halve it.** *Necessity of Setting Temperature*, Li et al., arXiv:2603.28304 (Mar 2026) **[C]**: judge consistency↔temperature correlates **−0.98 to −1.00**; at T≈0 consistency ≈1.0. The OAuth grader runs at default T with no temperature/seed plumbed (`leaf_scorer.py:1669`; locked A5) — so the residual 0.018 is genuine sampling drift. **But** *Rating Roulette*, Haldar & Hockenmaier, EMNLP'25 Findings, arXiv:2510.27106 **[C]** found **no** self-agreement improvement from resampling one model out to 10 runs **at fixed settings** — i.e. same-model resamples can be *correlated*, in which case median-of-3 ≈ the single draw and the locked A1 "√3 ≈ 1.7× variance cut" is optimistic. The reconciliation (*Empirical Study of LLM-as-a-Judge*, Yamauchi et al., arXiv:2506.13639, Jun 2025 **[C]**: sampling+aggregation beats greedy) is that aggregation pays when the samples are *diverse* (moderate T or diverse judges), not identical. **This is exactly why the locked spec's Q6 calibration gate — measure σ-before/after, promote only if it drops below the band — is the correct and load-bearing guard.** The literature says: do not trust the theoretical √N; the empirical σ is the contract. The locked spec already designed this; this companion only insists it be a *hard* promotion gate (see §5).
- **The measurement-floor argument is confirmed.** The locked D3 keeps BES default-OFF because its +0.085 / 0.0079 margins sit inside the band. *Single-Agent ≥ MAS at equal token budget*, Tran & Kiela, arXiv:2604.02460 (Apr 2026) **[C]** and the Anthropic multi-agent report (+90.2% but **15× tokens**; "most coding tasks don't benefit") independently say apparent multi-agent/feature lift is often unaccounted compute — so an effect smaller than the noise band (or unmatched on budget) is not an effect. **Strong confirm of D3 and of the locked spec's whole "fix the grade first, then BES is measurable" sequencing.**

---

## 2. Per-workstream literature verdicts

Notation: **✓confirm** / **~temper** / **✗contradict** / **+net-new**.

### Workstream A — grader fidelity

- **A1 median-of-N — ✓ on *median*, ~ on *N=3-default-ON*.** Median over mean is well-founded: *CARE*, Zhao et al., arXiv:2603.00039 (Mar 2026) **[C]** shows naive mean/majority of *correlated* judgments can *amplify* bias, and a robust aggregate beats it 11–27%; the locked spec's "median shrugs off the all-0.0-batch-failure outlier" is textbook-correct (a failed batch zeroes 15 leaves at `leaf_scorer.py:1678` — median over N≥3 ignores one outlier). **Temper:** the √3 claim assumes independent draws; per Rating Roulette they may not be. **Refinement the literature supports (not a redesign):** make A1 a **cascade** — grade once, escalate to median-of-N **only on leaves where it can matter** (judgment leaves, or where a first cheap signal shows disagreement), per cost-saving cascades with early abstention, Zellinger et al., arXiv:2502.09054 (Feb 2025) **[C]** (~13% cost cut at lower error). This composes with A2 (which already removes mechanical leaves) so the escalated set is small. Also note *Nine Judges, Two Effective Votes*, Kohli, arXiv:2605.29800 (May 2026) **[C]**: aggregation benefit plateaus fast (n_eff ≈ 2.2 for 9 judges) — **N=3 is the right ceiling; do not raise it.**

- **A2 deterministic-by-construction routing — ✓✓ strong confirm; the literature ranks this *above* A1 by variance/bias leverage** (not by rollout order — the locked spec correctly ships A2 last for its rubric-gen/provenance producer dependency). *DeCE*, Yu et al., arXiv:2509.16093 (Oct 2025) **[C]**: decomposing a holistic score into checkable sub-criteria lifted human-correlation to **r=0.78 vs 0.35** for a pointwise judge — the single biggest structural win in the judge-noise literature, and it attacks *bias* as well as *variance* (which A1 cannot — median reduces variance, not bias). *GenPRM*, Zhao et al., arXiv:2504.00891 (Apr 2025) **[C]**: a code-/disk-grounded verifier (7B) beat a 72B text-only one — exactly the locked A2 instinct to score mechanical leaves from `provenance.json`/`metrics.json` rather than the LLM. **Caution the audit half-states:** deterministic leaves are a *gaming surface* — a "metric key present" check is satisfiable by writing the key with a fake value. **Every deterministic leaf must compose with A7 EVIDENCE_GATE's value-sanity**, and any availability/loosening change must be re-graded against records before trust — the 2026-06-15 synonym fix *caught and rejected* exactly such a gaming variant ([[leaf-frontier-remediation]]). Also flag the route dependency the reviewer surfaced: on the **monolithic** route the agent's `train.py` writes `metrics.json` *by contract* (`primitives.py`: *"Contract: paper's code writes $OUTPUT_DIR/metrics.json"*), so a deterministic numeric check there reads agent-authored data — fine for A2 (it's a structured compare) but it means any *tamper*-detection idea is route-dependent (see §3.6).

- **A3 retire the MAX-over-noise floor — ✓ confirm.** Max over a noisy estimator is upward-biased (banks the luckiest draw) — a statistical fact, and the locked spec's "median-within-evidence-key, demote the global max to salvage-only" is the right shape. *Confidence vs Critique*, Yang et al., arXiv:2412.19513 (Dec 2024) **[C]** frames the trade-off: preserving correct earlier estimates (Confidence) vs flipping wrong ones (Critique); the locked A3/A4 **adopt-only-if-higher within an evidence key** is the Confidence-preserving move, while the **global** max is the pure-noise-banking move A3 correctly strips. *(This orphan citation now has a home — it was flagged in review as dangling in the prior draft.)*

- **A4 champion-artifact (ship the best *artifact*, graded fresh — not the best *score*) — ✓ confirm, and it closes a real gaming hole.** *BadScientist*, Jiang et al., arXiv:2510.18003 (Oct 2025) **[U on %]** (LLM reviewers reach accept-level on fabricated/polished work, detection ≈ chance) and *Rethinking the Value of Agent-Generated Tests*, arXiv:2602.07900 (Apr 2026) **[C]** (agent "tests" are mostly value-prints, not assertions) both say the verdict must be grounded in the *executed artifact*, not a narrative or a detached number. A4's `score ≡ best artifact actually produced` is exactly that grounding.

- **A5 decoupled sampler-capable transport + `complete_samples` mixin — ✓ confirm; + net-new on self-preference.** The mixin design (default = N× `complete`; OpenAI overrides with native `n`+`seed`+`temperature=0`; raw Anthropic-messages client with `temperature=0`) is the right backwards-compatible shape, and decoupling grading from the root's `ctx.llm_client` so a root/CLI wedge can't take grading down is well-motivated (the OmniZip failure; [[model-wedge-fable5-fix]]). **Note (the locked spec already states this correctly — not a correction):** Anthropic exposes **no `seed`**, and `temperature=0` on Anthropic is *near*-deterministic, not bit-identical — so even the raw-messages path is `pinned`, not `exact`, and **median-of-N stays the universal floor** (locked A5 step 3 already says exactly this). **+Net-new gap:** A5 keeps the grader on Sonnet ("grader stays Sonnet-quality") — but the artifacts under grading are **Sonnet-authored** (`implement_baseline`). *When Does Verification Pay Off?*, arXiv:2512.02304 (Dec 2025) **[C]**: **cross-family verification > self-verification, and self-verification is the weakest config**; self-bias amplification, Xu et al., arXiv:2402.11436 **[C]**. The locked design does not address self-preference at all. See §3.1.

- **A6 count-based per-cell digest (no headline cell vanishes) — ✓ confirm.** PaperBench, arXiv:2504.01848 **[C]** grades against hierarchical rubric trees with an LLM judge at **0.83 F1** — the realistic ceiling of *any* LLM grader, and a reminder that what the judge cannot *see* it scores "unverified." A6 (deterministic per-cell digest so a wide grid never silently drops trailing headline cells) is the visibility analog of the G1 fix, on-thesis.

- **A7 EVIDENCE_GATE (verify every cited leaf exists on disk) — ✓✓ strongest single correctness lever in this space; implement it.** The audit recommends implementing the documented-but-absent gate. The literature makes this urgent: *MLR-Bench*, arXiv:2505.19955 **[C]** — agents produced **fabricated/invalidated results in ~80% of cases**; *ImpossibleBench*, arXiv:2510.20270 **[C]** — GPT-5 "passes" **76%** of impossible tasks by cheating; *Reward Hacking Benchmark*, Thaman, arXiv:2605.02964 (May 2026) **[C]** — hardening the *environment* cut exploits **87.7%** with no correctness loss, and the named threat is "tamper with the evaluation function." An LLM grader **cannot** be the anti-fabrication backstop (BadScientist; *EvilGenie*, arXiv:2511.21654 **[C]** found held-out tests add little marginal benefit over a deterministic check in unambiguous cases). The deterministic, disk-grounded EVIDENCE_GATE is. **This is the highest-leverage item the locked spec marks "S effort" — prioritize it.**

### Workstream B — aggregation & lifecycle honesty
- **B1 (hybrid Phase-1/2 best-of) and B2 (graded-but-warned reads as never-verified) — ✓ confirm.** B2 in particular is reinforced by *The Self-Correction Illusion*, Chen et al., arXiv:2606.05976 (Jun 2026) **[C]** and the long-horizon self-conditioning result, arXiv:2509.09677 **[C]**: a real grade that fails to set `ctx.latest_rubric_score` lets a stale/again-derived state drive the policy — surfacing the *actual* grade as the authoritative signal (not re-deriving) is correct. **+Net-new free lever** attaches here — see §3.3.

### Workstream C — execution correctness
- **C1–C6 — ✓ confirm; C1 and the smoke items are literature-backed.** C1 (`--execution-mode max` silently half-dropped) is a correctness bug, not a literature question. The smoke/guard items connect to *GenPRM* (execute-to-verify is the strongest verifier) and *RECODE-H*, Miao et al., arXiv:2510.06186 **[C]** — recall **29.4% → 71.6%**, the biggest jump from *minimal diagnostic* signals — which validates cheap CPU-stage smoke + the zero-LLM-call `leaf_triage` repair plan over verbose guidance. **One adversarial caution for any CPU-scoped smoke (if pursued under C):** a "device-side assert" only fires on CUDA; on CPU it surfaces as the synchronous bounds `RuntimeError` (cleaner), and a trainer hard-coding `.cuda()` will raise `No CUDA GPUs are available` under hidden devices — that exit must be mapped to **soft-pass**, or a healthy GPU run is falsely blocked. (This is a note for whoever implements it, not a locked item.)

### Workstream D — BES & A/B validity
- **D1 (ab_compare becomes a validator) — ✓✓ confirm + extend.** Refusing a Δ unless both arms are stamped, the `rubric_tree.json` sha matches, scope matches, and `select=best` — this is exactly the budget-/condition-matching the multi-agent skeptic literature demands (2604.02460). **Extend:** report every Δ **relative to the measured σ_grader band** (from the Q6 calibration harness) and gate "significant" on Δ > kσ; for a principled bound rather than a hand-tuned k, *Conformal risk control on the stop/decision*, arXiv:2602.03814 (May 2026) **[C]**. This makes "is the +0.085 real?" answerable with a stated band — the locked D3's own ask.
- **D2 (smoke-gate the BES SELECT signal) — ✓ confirm.** SELECT today is code-cosmetics (`degraded=False`, no metrics) — *Large Language Monkeys*, Brown et al., arXiv:2407.21787 **[C]**: repeated sampling scales **only** with a trusted verifier (SWE-bench 15.9%→56% *with* a verifier; selection plateaus without one). A statically-faithful-but-non-runnable candidate winning is the failure mode; wiring `execution_smoke`/`preflight_smoke` into SELECT is the fix. **+Net-new:** size the BES candidate pool by *saturation*, not a fixed N — see §3.4.
- **D3 (keep BES OFF until measurable) — ✓✓ strong confirm** (see §1).

### Workstream E — integration & posture
- **E1 (merge negative-lessons / context-map; one flag prefix) — ✓ confirm, high value.** Cross-run failure memory is the "single biggest missing self-improvement lever" per the audit; the literature agrees structured, evolving memory beats ad-hoc reflection (*A-MEM*, arXiv:2502.12110 **[U]**; *Meta-Policy Reflexion*, arXiv:2509.03990 **[C]** — structured, rule-checked memory over free-text). A `REPROLAB_NEGATIVE_LESSONS=1` that is a silent no-op on this branch is worse than absent (operators believe a loop runs that doesn't). Confirm the "merge or strike the doc" recommendation.
- **E2 (flip proven guards ON: best_attempt, dead_training_guard, execution_smoke, preflight_smoke) — ✓ confirm.** GenPRM/CRITIC (execution-grounded verification works) + the asymmetry the audit names (unproven BES got an A/B harness while *proven* guards stay default-OFF). The literature's only caveat is the CPU-smoke false-positive guard above.
- **E3/E4 (loud-fail-soft sweep; doc/code reconciliation) — ✓ confirm**, hygiene.

---

## 3. Net-new levers the audit missed (frontier-supported additions)

Each is an *addition* to the locked design, marked by where it attaches.

1. **+Cross-family grading for the judgment leaves (attaches to A5).** Sonnet grading Sonnet-authored artifacts is the weakest verification config (2512.02304 **[C]**; PoLL, Verga et al., arXiv:2404.18796 **[C]** — a diverse small-model jury is **>7× cheaper** than one big judge and carries less self-preference). Concretely: A5 already decouples the transport via `REPROLAB_GRADER_BACKEND`/`_MODEL` — add an *optional* second-family member (e.g. a GPT or Qwen) for the **judgment** leaves only (A2 having removed the mechanical ones), median-aggregated. **Gated, and only if the calibration harness measures a self-preference gap** — Nine-Judges (2605.29800 **[C]**) warns the benefit plateaus by ~3 judges, so this is a *bias* fix, not a variance fix, and should not be a default.

2. **+Escalate the A1 cascade on *disagreement*, not on *confidence*.** If A1 becomes a cascade (§2/A1), the trigger must not be the grader's verbalized confidence: *Overconfidence in LLM-as-a-Judge*, Tian et al., arXiv:2508.06225 (Aug 2025) **[C]** (ECE up to 39%) and *Reasoning models' calibration*, arXiv:2506.18183 **[C]** (verbalized confidence often >85% **even when wrong**; reasoning fine-tuning *degrades* abstention, AbstentionBench arXiv:2506.09038 **[C]**). Use **cross-sample disagreement** (the first two samples differ) as the escalation signal — free, internal, and honest. (White-box entropy probes, arXiv:2510.08146 **[C]**, are cheaper still but need logprobs — unavailable on the OAuth path; scope to OpenAI/Azure if ever used.)

3. **+The role-relabel free lever (attaches to B2 / self-improvement).** *The Self-Correction Illusion*, arXiv:2606.05976 (Jun 2026) **[C]**: re-framing an identical prior claim as a **tool/memory observation** rather than the model's own prior reasoning lifts correction **23–93pp**. The harness already surfaces failures as `experiment_runs.jsonl` rows and grades as `verify_against_rubric` results — this paper says that framing is *why* it works and to **lean into it**: feed the prior grade/failure to the root as an external observation, never as "your earlier reasoning." Zero cost, zero new machinery — a prompt-construction discipline.

4. **+Saturation-sized BES pool (attaches to D2/D3).** Instead of a fixed candidates-per-cluster, stop generating when the score-vs-candidate curve flattens: *ReASC*, arXiv:2601.02970 (Jan 2026) **[C]** (adaptive self-consistency, **70%** cost cut at preserved accuracy); compute-optimal allocation, Snell et al., arXiv:2408.03314 **[C]** (>4× efficiency vs uniform). Fold into the BES spec, not the grader, but it is the efficiency counterpart to D2's quality fix.

5. **+Budget-aware, capped, oscillation-watching stop (adjacent to A/forced-iteration).** Out of the locked spec's grading-centric scope but the same audit's "honesty" theme: expose remaining wall-clock/USD to the root (it bounds time three ways but never *tells* the root how much is left — *BATS*, arXiv:2511.17006 **[C]**: a bare cap underperforms an exposed budget); cap forced iterations and stop on two non-improving verifies (*inverse-scaling in TTC*, Anthropic, arXiv:2507.14417 **[C]**; *overthinking*, arXiv:2604.10739 **[C]** — accuracy peaks then *declines*, moderate-budget stop keeps 97% at 60% compute); abandon doomed runs (*AgentStop*, arXiv:2605.15206 **[C]** — 15–20% saved for <5% utility). Pure efficiency + a small correctness gain. Offer as a sibling spec item.

6. **+Cryptographic provenance stamp if tamper-detection is ever wanted (caveat, not a recommendation).** A7's EVIDENCE_GATE (existence + value-sanity) is the right and sufficient backstop. A *stronger* anti-tamper step — HMAC the harness-aggregated `metrics.json`/`provenance.json` at write time, verify at grade — is the only enforceable form (mtime/authorship heuristics false-positive on the legitimately agent-authored monolithic route, per §2/A2). **Listed so it is not reinvented as unenforceable prose; not proposed for now** — A7 is enough until a measured fabrication-via-override case appears.

---

## 4. SOTA → workstream map (page-verified)

| Paper (arXiv, conf.) | Finding | Locked workstream | Verdict |
|---|---|---|---|
| Li 2603.28304 **[C]** | T↔consistency −0.98..−1.0 | A5 | ✓ (and why 0.018 residual exists) |
| Rating Roulette 2510.27106 **[C]** | resampling fixed-settings ⊀ converge | A1 | ~ temper √3 claim → Q6 gate |
| Design-choices 2506.13639 **[C]** | sample+aggregate > greedy | A1 | ✓ (aggregate *diverse* samples) |
| CARE 2603.00039 **[C]** | median > naive mean of correlated | A1 | ✓ median |
| Nine-Judges 2605.29800 **[C]** | n_eff≈2.2; single≈panel | A1 / §3.1 | ✓ cap N at 3 |
| Cascade 2502.09054 **[C]** | escalate only uncertain (−13%) | A1 | + cascade refinement |
| DeCE 2509.16093 **[C]** | decomposed r=0.78 vs 0.35 | A2 | ✓✓ bigger lever than A1 (not a re-sequence) |
| GenPRM 2504.00891 **[C]** | code-exec verifier > bigger text | A2 / A6 / C / E2 | ✓ |
| MLR-Bench 2505.19955 **[C]** | ~80% fabricated results | A7 | ✓✓ implement gate |
| ImpossibleBench 2510.20270 **[C]** | GPT-5 cheats 76% | A7 | ✓✓ |
| RewardHacking 2605.02964 **[C]** | env hardening −87.7% exploits | A7 | ✓✓ |
| BadScientist 2510.18003 **[U%]** | reviewers fooled by presentation | A4 / A7 | ✓ ground in artifact |
| Rethinking-agent-tests 2602.07900 **[C]** | agent tests = theater | A4 / D2 / E2 | ✓ keep harness-owned |
| When-verify-pays 2512.02304 **[C]** | cross-family > self-verification | A5 | + §3.1 self-preference |
| PoLL 2404.18796 **[C]** | diverse jury >7× cheaper | §3.1 | + (gated) |
| Overconfidence 2508.06225 **[C]** | judge ECE up to 39% | §3.2 | + escalate on disagreement |
| Self-Corr-Illusion 2606.05976 **[C]** | prior-as-tool/memory +23–93pp | B2 / §3.3 | + free lever |
| Confidence/Critique 2412.19513 **[C]** | keep-right vs fix-wrong trade-off | A3 | ✓ adopt-if-higher |
| RECODE-H 2510.06186 **[C]** | minimal diagnostic drives recovery | C / leaf_triage | ✓ |
| Monkeys 2407.21787 **[C]** | sampling scales only w/ verifier | D2 | ✓ smoke-gate SELECT |
| ReASC 2601.02970 **[C]** | adaptive self-consistency −70% | §3.4 | + saturation pool |
| Snell 2408.03314 **[C]** | compute-optimal >4× | §3.4 | + |
| Single-agent≥MAS 2604.02460 **[C]** | lift often just tokens | D1 / D3 | ✓✓ budget-match |
| Conformal 2602.03814 **[C]** | distribution-free decision bound | D1 | + significance band |
| BATS 2511.17006 / AgentStop 2605.15206 / Overthink 2604.10739 / Inverse-scale 2507.14417 **[C]** | budget-aware, capped, early-stop | §3.5 | + sibling spec |

---

## 5. Where the literature says the locked design could fail (adversarial)

1. **A1 may not pay on this transport — make Q6 a *hard* gate.** The reviewer's sharpest point, and the literature backs it: if same-model resamples are correlated (Rating Roulette), median-of-3 over a ~0.018-drift signal buys little for 3× cost. **The locked spec's calibration harness must be a blocking promotion gate, not a report:** if measured σ-after ≈ σ-before, A1 does not flip default-ON and **A2 (decomposition) carries the variance reduction** (which the literature says is the bigger lever anyway). The locked spec already plans the measurement (Q6); this companion's only insistence is that A1's default-ON status be *contingent* on it.
2. **Determinism ≠ correctness.** A1/A5 make the grade *reproducible*; a confidently-wrong grade becomes reproducibly wrong. Variance ≠ bias. The bias attackers are A2 (decomposition), §3.1 (cross-family), and A7 (anti-fabrication) — all weaker/optional than the variance fixes. The companion is honest that the locked program is stronger on noise than on accuracy.
3. **Deterministic leaves (A2) are a gaming surface.** Must compose with A7 value-sanity; every loosening change re-graded against records (the 2026-06-15 caught-and-rejected variant is the precedent). Hard rule.
4. **Self-preference is uncorrected by a same-family median.** Median-of-3 Sonnet samples still shares Sonnet's self-preference for Sonnet-authored code (2512.02304, 2402.11436). Only §3.1 (cross-family) or A7 (disk grounding) attacks it; the locked design leaves it open.
5. **Closed-Claude constraints bound the toolkit.** No `seed`, no logprobs on the OAuth path → the cheapest calibrated-confidence methods (hidden-state probes 2512.22245 **[C]**, entropy stop 2510.08146) are OpenAI/Azure-only. The default path's confidence proxy must be cross-sample disagreement (§3.2), not anything white-box.
6. **A4 champion-artifact + a residually-noisy grade.** Restoring "the snapshot whose median grade is highest" still picks a max-over-keys; with A1's median the per-key estimate is robust, but across many evidence keys the selection re-introduces a mild upward bias. Acceptable (it selects an *artifact*, then re-grades it fresh), but worth stating: A4's honesty depends on A1/A5 having shrunk per-key noise first — so the rollout order (A5+A1 → A3+A4) in the locked spec is correct and not reorderable.

---

## 6. Efficiency ledger (net cost is ~flat, with upside)

- **A1 transient 3×** grader calls → **cost-neutral at steady state** once A2 peels mechanical leaves off the LLM (locked spec's own accounting: ~7 judgment × 3 ≈ today's ~20 × 1). The §2/A1 **cascade** bounds it further (escalate only on disagreement; Zellinger −13%).
- **A2** removes ~half the leaves from the LLM entirely → lower variance *and* lower cost.
- **§3.3 role-relabel** — free (prompt construction).
- **§3.4 saturation pool / §3.5 early-stop** — net *savings* (ReASC −70%, AgentStop −15–20%, overthinking 97%@60% compute) on the GPU/candidate side.
- **§3.1 cross-family jury** — the one real cost adder (extra family per judgment leaf); gated, bias-only, off by default.

Conclusion: the locked program's claim that grader fidelity is achievable at roughly flat cost is consistent with the literature, **provided A2 lands to offset A1** — which is why the locked rollout sequences them together.

---

## 7. Suggested (non-binding) addenda to the locked spec

These do not change any locked decision; they are the deltas a future revision of the locked spec might absorb:

- **A1:** add "cascade — escalate to median-of-N on judgment leaves / first-two-samples-disagree only" and make default-ON **contingent on the Q6 σ-drop**. (cites 2502.09054, 2510.27106, 2605.29800)
- **A5:** note the no-seed/near-deterministic Anthropic caveat (median-of-N is the floor even on the API path) and add an **optional cross-family judgment-leaf grader**, gated on a measured self-preference gap. (cites 2512.02304, 2404.18796)
- **A7:** elevate priority — it is the highest-value correctness lever in the corpus, not just an "S" cleanup. (cites 2505.19955, 2510.20270, 2605.02964)
- **B2:** add the role-relabel framing note (surface prior grades as observations). (cites 2606.05976)
- **D1:** report Δ against the measured σ_grader band; gate "significant" on Δ > kσ (or conformal). (cites 2604.02460, 2602.03814)
- **D2/D3:** size the BES pool by saturation, not a fixed N. (cites 2601.02970, 2407.21787)
- **New sibling item:** budget-aware + capped + oscillation-stop iteration policy. (cites 2511.17006, 2605.15206, 2604.10739, 2507.14417)

---

*Provenance note.* Internal measurements (0.018 same-grader drift, ±0.09–0.18 run-to-run, the All-CNN 0.712/0.694 datum) are from the locked spec and `report.py:980`. All arXiv items carry **[C]** (page-fetched, numbers verified) or **[U]** (unverified magnitudes) from the 2026-06-16 three-sweep research pass; **[U]** items must not be cited with numbers without re-fetching. File:line references are against `feat/azure-aks-gpu` HEAD as of 2026-06-16. This companion was itself adversarially reviewed; the headline-number correction in the header is the result.
