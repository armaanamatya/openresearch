# Phase 1 — Coverage Completion (2026-06-07)

**Status:** 🟡 PROPOSED. Agent-side (guidance-driven), no harness changes. Depends on Phase 0 merge.
**Goal:** Lift the heaviest under-scored leaves by *doing the work* (not excluding it) — the missing baselines, real eval templates, and provenance/curves. Targets the honest climb to ~0.48–0.52.

---

## 1. Key grounding — a "baseline" is agent-side code, not a harness axis

**Verified (Agent 3):** the matrix backend is axis-agnostic. A baseline is **just a string** the agent's `code/train.py` maps to two flags (`train.py:487-488`):
```python
opsd_enabled = baseline in ("sdar", "grpo_opsd")
gate_type    = "sigmoid" if baseline == "sdar" else "ones"
```
`BASELINES` is an agent-side list (`train.py:1174`); `train_one_run(baseline=...)` (`train.py:465`) dispatches. So **adding baselines is NOT a backend change** — it is (a) adding cells with the new `baseline` string to `cells.json`, and (b) extending the agent's `train.py` dispatch. The lever is the existing **`REPROLAB_BASELINE_EXTRA_GUIDANCE`** env var (instructs the RLM root to emit them), not a code patch.

## 2. Component A — the three missing baselines

Paper has 5 (GRPO, OPSD, Skill-SD, GRPO+OPSD, RLSD); the run produced 3 (GRPO, GRPO+OPSD, SDAR). Leaf `39e798b7` ("Skill-SD, RLSD, standalone OPSD absent") scored **0.15** and sits in the **heaviest area** (Method, eff. weight ≈ 0.057).

| Baseline | Recipe | Cost | Grounding |
|---|---|---|---|
| **standalone OPSD** | OPSD loss only, no GRPO RL term → `opsd_enabled=True, grpo_weight=0` | **Near-free** — OPSD machinery already exists | `train.py:18-24`, `:487` |
| **Skill-SD** | Self-distillation *with* a populated `skill_context` prompt slot | Med — slot exists but is always empty; needs a SkillBank feed | `build_prompt(question, skill_context)` `train.py:241-249` |
| **RLSD** | RL + self-distillation variant — another `(opsd_enabled, gate_type, schedule)` combination | Med | `train_one_run` dispatch `train.py:487` |

**Approach:** drive via `REPROLAB_BASELINE_EXTRA_GUIDANCE` with the exact loss recipes; start with **standalone OPSD** (the flag flip) to validate the cell→leaf plumbing before the Skill-SD SkillBank work. Note: the SkillBank/skill-retrieval subsystem (leaf `d2c1a0a8`, the 4 strategies UCB/KM/Full/Random) is **not** shipped by `full-scope-envs` and is a separate, higher-effort build — defer it.

## 3. Component B — real eval templates (rides the Phase 0 merge)

Leaf `5b717e4f` ("no ALFWorld/WebShop prompt templates; only Search-QA closed-book") scored **0.10**. The Phase 0 merge brings real multi-turn Search-QA `search()`/`answer()` templates (`search_qa_env.py`) and the env templates; activating Search-QA cells against the E5 index lifts this to ~0.55. **Effort: low** (config the cells to use the merged envs). Same merge lifts the Data leaf `9b6f6fef` (0.30→0.85, real E5 retrieval vs closed-book).

## 4. Component C — provenance + metric curves (cheap polish)

- **Provenance** (leaf `08e3c5c5`): emit an explicit link to the ZJU-REAL/SDAR reference repo in the run artifacts / report.
- **Curves** (leaf `ddd545`, scored 0.7 — "metrics logged as scalars; curves.json mentioned but not shown"): have the agent's `train.py` actually write `curves.json` (per-step `gate_mean`, `gap`, `opsd_loss`, `reward`), not just terminal scalars.
- **Effort: XS.** Low weight (Artifact area 0.05) but trivial and lifts two leaves toward 0.9.

## 5. Ranked plan (addition → leaf → effort), from Agent 3

Effective weight = area_weight × within-area leaf weight.

| Rank | Addition | Leaf (area) | Eff.W | Now→Est | Effort |
|---|---|---|---|---|---|
| 1 | E5 retriever + real Search-R1 (Phase 0 merge) | `9b6f6fef` (Data) | 0.042 | 0.30→0.85 | Low |
| 2 | Real eval templates (Phase 0 merge) | `5b717e4f` (Eval) | 0.046 | 0.10→0.55 | Low |
| 3 | **3 baselines** (Skill-SD/RLSD/OPSD) | `39e798b7` (Method) | 0.057 | 0.15→0.75 | Med |
| 4 | ALFWorld real **data** leaf | `e1acc62e` (Data) | 0.042 | 0.00→0.40* | Med |
| 5 | ALFWorld batch config | `1b7c8f27` (Exec) | 0.036 | 0.00→0.50 | Low |
| 6 | SkillBank + 4 retrieval strategies | `d2c1a0a8` (Data) | 0.028 | 0.00→0.30 | High |
| 7 | WebShop env + data | `e0515c40` (Data) | 0.028 | 0.00→0.40 | High (server) |

\* rows 4–5 (ALFWorld) are subject to the **sequencing trap** — the *config/data* leaves pass on "real env present + correct config" regardless of reward, but only if the env is brought in-scope *with* Phase 4's learning fix or as a verified exclusion. Bringing ALFWorld in-scope while it scores 0 reward converts excluded leaves into counted zeros. **Run rows 1–3 first** (Search-QA + baselines, no env-activation risk).

## 6. Testing

- Cell-emission test: with `REPROLAB_BASELINE_EXTRA_GUIDANCE` set for the 3 baselines, the generated `cells.json` contains `baseline ∈ {opsd, skill_sd, rlsd}` cells and `aggregate_cell_metrics` nests them at `per_model[m][env][baseline]`.
- Leaf-attribution test: a run with all 5 baselines lifts leaf `39e798b7` above the repair threshold.
- Provenance/curves: assert `curves.json` written and the report carries the repo link.

## 7. Definition of done

- [ ] standalone OPSD baseline emitted + scored (plumbing validated) before Skill-SD/RLSD.
- [ ] Search-QA cells run against the merged E5 retriever; Data + Eval leaves lift.
- [ ] `curves.json` + provenance link emitted.
- [ ] ALFWorld/WebShop env activation **not** done here (deferred to Phase 4) — no score regression from the sequencing trap.

## 8. Expected effect

Combined with Phase 0, Agent 3's grounded estimate is **~0.48–0.52 with zero new hardware** — the Data area climbs 0.09→~0.50, the heaviest leaf 0.15→0.75, eval templates 0.10→0.55. Clearing 0.6 then depends on Phase 4 (an env that actually learns).
