# Phase 4 — The 0.6 Frontier & Honesty Guard (2026-06-07)

**Status:** 🟡 PROPOSED. The hard, high-value work: make a multi-turn env actually *learn* (the only path past 0.6) and close the score-inflation hole that the env-coverage work opens.
**Goal:** (A) Unblock ALFWorld learning so result-match leaves can climb; (B) bring the environment exclusion axis to parity with the model axis so the climb is *credible*, not gamed.

> **Codex review (2026-06-07) — applied.** "Learning" must be proven by **terminal task success**, not shaped reward: A2 reward-shaping can make `grpo_loss != 0` and reward `> 0` *without* improving real success. Require a **seeded held-out evaluation showing terminal success-rate improvement over the untrained policy**, reporting shaped vs terminal rewards **separately** (§5). Component B (env-axis guard) is the **hard dependency** for Phase 0's finalize re-score — ship it first.

---

## 1. Why this is the frontier

Phases 0–1 reach ~0.48–0.52 honestly. Clearing **0.6 requires at least one multi-turn env actually learning** — and `full-scope-envs` makes ALFWorld *real* but *not learnable* (verified, Agent 3): sparse `float(won)` reward (`alfworld_env.py:458`,`:244-246`), no warm-start, no shaping. Cold-start 7B + tiny rollout G → all rewards 0 → GRPO advantage `(rᵢ−mean)/std` = 0 → `grpo_loss=0.0000` exactly → no gradient. The SDAR self-distillation machinery is live (gate climbs) but auxiliary loss cannot substitute for a real reward.

## 2. Component A — make ALFWorld learn (B4)

Four software fixes; **three are pure code**, one wants VRAM. Recommended order optimizes for fast iteration first.

| # | Fix | What | Where | VRAM? |
|---|---|---|---|---|
| **A1** | **Construct env once per cell** | The env reloads `AlfredTWEnv` ~82× (once per episode). Build once, `reset()` in place. The host-shared data cache already exists. | Agent `train.py` rollout loop + `EnvCacheManager.acquire_alfworld` (`env_cache.py:11`,`:249`) | **Pure code** (~2–4× faster steps) |
| **A2** | **Reward shaping / partial credit** | Replace pure 0/1 terminal with sub-goal progress credit (reach target object, open right receptacle). | `ALFWorldEnv.step`/`_finish` (`alfworld_env.py:452-487`) — emit dense intermediate `StepResult.reward` (the field already exists, `sdar_env_base.py:102-110`, currently 0 until terminal) | **Pure code** |
| **A3** | **SFT warm-start** | Before RL, fine-tune on a few hundred ALFWorld expert trajectories (the paper warm-starts) so the policy occasionally succeeds → non-zero advantage. | New pre-RL SFT phase in agent `train.py` + expert-trajectory loader (GiGPO/ALFWorld dataset) | **Pure code** (LoRA SFT on 1.7B/3B fits 24 GB) |
| **A4** | **Larger group size G** | Bigger G → more rollouts/step → higher chance ≥1 succeeds early. Currently `GROUP_SIZE=4` (`config.json:7`); paper = 8. | `cells.json`/`config.json` `GROUP_SIZE`; `capacity_gate` already budgets VRAM | **Wants VRAM** — each extra rollout multiplies KV-cache/activations |

**Order:** A1 (kills the 82× reload tax → fast iteration) → A2 (highest-leverage single change to get *any* gradient on small models within 24 GB) → A3 (paper-faithful) → A4 (only as hardware allows).

**Hardware vs code split (matches the 2026-06-02 meeting doc):** do **not** buy hardware to "fix ALFWorld" — A1–A3 are code. 24→80 GB only buys A4 (bigger G), un-sharding the 7B cells (currently split across 2 A5000s, halving the fleet), and running the full 3×3×5×multi-seed matrix in hours not days.

**Sequencing trap reminder (README §6):** only flip ALFWorld in-scope once A1–A3 give it a real reward — otherwise the verified-exclusion gate counts its leaves as 0 instead of excluding them, *lowering* the score.

## 3. Component B — environment-axis exclusion guard (close the laundering hole)

**The hole (verified CONFIRMED, Agent 2):** the **model** axis is guarded — every skipped model is split by `if key in op_skip:` (`leaf_scorer.py:445-486`, `op_skip` from `ScopeSpec.skip_models` via `_operator_skip_set:412`) into excluded (operator-sanctioned) vs kept (the `b5e36dd` anti-laundering fix). The **environment** axis is **unconditional** (`leaf_scorer.py:494-497`): every self-declared `environments_skipped` entry is excluded, with **no `op_skip` gate** and **no `skip_environments` field** (`ScopeSpec` only has `skip_models`, `schemas.py:880`). The call site confirms the asymmetry — `verify_against_rubric` threads `operator_skip_models` (`primitives.py:4590-4592`) but has no environment equivalent.

**Consequence:** an agent can write `environments_skipped: ["alfworld"]` for an env that merely *crashed or didn't learn* and the scorer drops every ALFWorld leaf from the denominator — exactly the inflation lever `b5e36dd` closed for models, still open for environments. This is why the corrected 0.431 (Phase 0) mixes principled (WebShop, pre-declared, server-heavy) and non-principled (ALFWorld, crashed) exclusion.

**The change (PROPOSED) — pick one, the boundary principle is the same:**
- **(B-a) Operator allow-list:** add `ScopeSpec.skip_environments`; gate `failed_envs` (`leaf_scorer.py:494`) through an `op_skip_envs` set exactly as models are gated at `:452`; thread `operator_skip_environments` at the `verify_against_rubric` call site (`primitives.py:4590`).
- **(B-b) Corroborated never-attempted:** honor an env exclusion only if the agent *never attempted* it and recorded a structured `data_load_failures`-style signal (mirroring the `scope.gaps` corroboration rule, `leaf_scorer.py:386-394`). An env that crashed mid-run stays a repairable failure in the denominator.

**Boundary principle:** *the operator (or a corroborated never-attempted signal) sanctions a scope reduction; the agent's self-declaration alone must not.* "Declared out of scope before trying" (WebShop) is transparent and excludable; "declared out of scope after it crashed" (ALFWorld) is laundering and stays counted.

**This gates Phase 0 Component B:** the finalize-time re-score must run *through* this guard, or it becomes an automated laundering path.

## 4. Component C — result-match quality

Once ALFWorld learns (A) and Search-QA runs against the real E5 retriever (Phase 1), the result-match leaves (`b78dce6e`, `272c4c09`, the +7.0% claim) can climb. Search-QA (dense reward) is achievable with tuning; ALFWorld is gated on A1–A3. This is the genuine-quality row and the last to move — it needs real numbers, not accounting.

## 5. Testing

- A1–A3: non-zero shaped reward + `grpo_loss != 0` are **necessary but not sufficient** (shaping can satisfy both without real progress). The real bar: a **seeded held-out evaluation showing terminal success-rate improvement over the untrained policy**, with shaped and terminal rewards reported **separately**.
- B: a self-declared `environments_skipped` for an *attempted-then-crashed* env stays in the denominator; an operator-sanctioned (or corroborated never-attempted) skip is excluded. Regression: the `prj_09047604e591d969` fixture's ALFWorld exclusion is **rejected** unless operator-sanctioned.
- C: result-match leaf climbs only with real metrics, not exclusion.

## 6. Definition of done

- [ ] ALFWorld produces a non-zero learning signal on ≥1 small model (A1–A3) before any in-scope activation.
- [ ] Env-axis guard at parity with the model axis; agent-only env exclusion no longer inflates the score.
- [ ] Phase 0 finalize re-score runs through the guard (no laundering path).
- [ ] Result-match leaves move on real numbers; the reported score is fully principled (no mixed exclusion).

## 7. Expected effect

The path past **0.6** — a multi-turn env that actually learns — and a score that is **credible to a skeptic**: every exclusion operator-sanctioned or corroborated, no agent-declared laundering. This is where "reproduce the paper" and "count it honestly" finally coincide.
