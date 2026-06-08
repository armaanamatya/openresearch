# Semantic Invariant Verification — Fidelity Scoring Redesign (2026-06-01)

**Status:** design (proposed). Build target: a worktree, behind the existing scorer interface (drop-in). Does NOT change a running reproduction.
**Audience:** any worktree working on the reproduction harness — this doc is committed so it is reachable from any branch via `git show <ref>:docs/superpowers/specs/2026-06-01-semantic-invariant-verification-design.md`.
**Motivating observation:** the 2026-06-01 SDAR live run (`prj_09047604e591d969`) reached rubric 0.176 with real metrics (SDAR 0.114 > GRPO 0.065 — the paper's ordering), then the RLM root *itself* proposed `path_1: "Fix SDAR Gate Formula & Lambda Naming"` because three leaves (`sigmoid_gate_on_advantage`, `lambda_self_distill_weight_0p1`, `real_qwen_weights_not_surrogate`) were **soft-gated despite the math being correct** — they just did not match hand-coded regex. The agent is burning an iteration to rename variables to satisfy a string matcher. That is Goodhart, and it does not generalize to other papers.

## 1. Verdict — localized brittle layer, not poor design overall

The fidelity-scoring architecture is **sound and general** in its primary path and degrades honestly:

- **Primary (semantic, general):** `leaf_scorer.score_reproduction` has an LLM grade each rubric leaf 0.0–1.0 against gathered code+metrics evidence (`leaf_scorer.py` ~997–1054). Robust to naming.
- **Auto-generation (general):** papers with no `docs/papers/<id>.yaml` get an LLM-generated rubric from the paper text (`rubric_gen.generate_rubric_tree`), with concrete "method fidelity" leaves. No per-paper code required.
- **Contract validation (deterministic, general):** `rubric_contract.validate` diffs metrics/artifacts against `paper_targets`. Numeric, not code-syntactic.

The smell is **one secondary layer**: the **regex invariant gate**.

## 2. The brittle layer (exact mechanism)

- Patterns are hand-coded per paper as `InvariantSpec` objects in `backend/agents/prompts/paper_hints.py` (only SDAR `2605.15155` has them today).
- `leaf_scorer.run_invariant_checks` (~687–817) compiles each `must_match` / `must_not_match` pattern and `re.search`es it **line-by-line** over the agent's source.
- `_apply_invariant_gate` (~820–842): a `must_not_match` hit → **hard cap 0.0**; a `must_match` *miss* → **soft cap 0.5** (`INVARIANT_SOFT_CAP`).
- The SDAR yaml states the design explicitly: *"The leaf scorer reads code; these constants MUST appear literally."*

Representative patterns (`paper_hints.py`):
- `sigmoid_gate_on_advantage`: `r"(?:torch\.)?sigmoid\s*\(\s*(?:self\.)?beta\s*\*"`
- `beta_gate_sharpness_10`: `r"\bbeta\s*[:=]\s*10(?:\.0)?\b"`
- `lambda_self_distill_weight_0p1`: `r"(?:lambda|opsd_weight|...)\s*[:=]\s*0\.1\b"`
- `real_qwen_weights_not_surrogate` (must_match): `r"from_pretrained\s*\(\s*['\"]Qwen/Qwen"`; (must_not) `r"class\s+TinyLM\b"`, `r"#\s*surrogate\s+model"`

## 3. Root cause — three coupled defects

1. **Syntactic test for a semantic property.** `sigmoid(0.1 * beta * delta)` (operand order), `BETA = 10` (caps), `beta = cfg.beta` (config-loaded), or a gate decomposed across functions are all *correct* yet **miss → false-negative cap**. False-negatives are the worst failure mode for a grader.
2. **Goodhart / teaching-to-the-test.** Rewards string-matching, not science — the agent renames `LAMBDA`→`lambda` instead of improving the reproduction.
3. **Per-paper hand-coding → no generality.** "Works on all papers" cannot depend on a human authoring regexes per paper.

It also **conflates two concerns** that want different tools:
- *Anti-surrogate* — "is this a real Qwen, not a TinyLM stub?" (a **runtime** fact).
- *Algorithm-fidelity* — "is the gate really σ(β·Δ)?" (a **semantic** property of code).

## 4. Design — semantic, paper-derived, runtime-corroborated

One modular mechanism, separating the two concerns and matching each to the right tool.

### 4.1 Algorithm-fidelity → `SemanticInvariantVerifier` (naming-agnostic, general)
- **Invariants are DATA, extracted per-paper** — not hand-coded regex:
  - from the yaml `algorithm_invariants` block when present (already semantic prose, e.g. `gate_formula: "g_t = sigmoid(beta * Delta_t) ..."`), and
  - otherwise extracted by the rubric generator from the paper text (it already pulls "g_t=σ(β·Δ_t), stop-grad, β=10, λ=0.1"). → works for ANY paper.
  - Schema: `Invariant{ id, description (semantic prose), kind: formula|constant|structural|anti_surrogate, runtime_signal?: str }`.
- **Verify each invariant with an LLM judge** given the *full relevant function/file* (not 40 KB of truncated snippets) + the invariant's semantic description:
  *"Is this property implemented? Cite the exact line(s) that implement it, or answer 'absent'."*
  Recognizes `g = torch.sigmoid(self.b*(lp_s-lp_t))` regardless of naming / operand order / decomposition.
- **Regex demoted to a positive FAST-PATH:** a regex *hit* cheaply confirms "present" (skip the LLM call). A regex *miss* **never penalizes** — it escalates to the semantic judge. This one change removes the false-negatives while keeping the cheap common-case path.

### 4.2 Anti-surrogate → runtime evidence, not source regex
- "Real Qwen, not a stub" is a **runtime** fact. Assert it from the artifacts the run already produces: the model id / param-count in the load log, plus the training-health floors the harness ALREADY has (`insufficient_training` wall-clock floor, optimizer-step floors, `model_load_failures`). A 2-second `TinyLM` smoke fails those regardless of source strings.
- `class TinyLM` / `# surrogate` regex becomes a **weak hint**, not the authority.

### 4.3 Calibrated, anti-gaming gate
- Soft-cap only when an invariant is **semantically confirmed absent**; hard-floor only when a surrogate is **confirmed at runtime**.
- Defend the inverse gaming (math in comments, not in the run): **cross-check code-claims against runtime signals** — e.g. the gate invariant must be present in code AND corroborated by the `gate_active_ratio` metric the run emits. `code ∧ runtime` = the anti-gaming guarantee, without brittleness.

## 5. Components / change list (modular)
1. `schemas.Invariant` (new) — semantic invariant record (id, description, kind, optional runtime_signal).
2. `InvariantExtractor` — yaml `algorithm_invariants` → `Invariant[]`; for no-yaml papers, a rubric-gen sub-step extracts them from paper text. General, paper-derived.
3. `SemanticInvariantVerifier` — `(invariant, code_dir, runtime_evidence) -> Verdict{status: present|partial|absent|surrogate, cited_lines[], confidence, rationale}`. LLM judge over the full relevant code + a runtime cross-check.
4. `FidelityGate` (replaces `_apply_invariant_gate`) — calibrated: absent-core → soft cap; runtime-confirmed surrogate → hard floor. A regex MISS no longer caps.
5. `run_invariant_checks` — demoted to the regex FAST-PATH that only emits POSITIVE "present" signals (a miss is silent).
6. Wire `score_reproduction` to call the verifier for fidelity leaves; keep the public scorer signature (drop-in).

## 6. Generality (the explicit requirement)
- No per-paper code. Invariants are extracted (yaml or LLM) per paper. The SAME verifier scores SDAR and an arbitrary arXiv paper.
- Anti-surrogate floor is runtime-derived, so it applies to every paper, not just ones with a curated regex set.

## 7. Migration / backward-compat
- Keep `InvariantSpec` regex specs; reinterpret a *hit* as a cheap confirmation, a *miss* as "escalate, do not penalize". Zero regressions for the papers that already pass the regex.
- Behind the existing `score_reproduction(...)` interface → drop-in.

## 8. Immediate de-risk (1-line, independent of the full build)
- In `_apply_invariant_gate`, stop applying the **soft** cap on a `must_match` miss (keep the hard cap on a confirmed `must_not_match` surrogate hit). This removes the false-negative on correct-but-differently-named code immediately; the full semantic verifier then restores a *true* absence signal.

## 9. Testing
- Unit: `SemanticInvariantVerifier` returns `present` for naming-variant gate impls (operand order, caps, config-loaded, decomposed) that the regex MISSES; `absent` for a genuinely missing gate; `surrogate` for a TinyLM run (via runtime signal).
- Generality: score SDAR (yaml invariants) AND one no-yaml arXiv paper through the same path; assert no per-paper code on the no-yaml path.
- Regression: the existing `leaf_scorer` + `rubric_contract` tests stay green; a regex MISS no longer caps a correct reproduction.

## 10. Out of scope
- The RLM-root / primitives / cell-runner architecture (sound; see the 2026-05-31 OOM/GPU remediation).
- Numeric `paper_targets` result-matching (already semantic/deterministic and general).
