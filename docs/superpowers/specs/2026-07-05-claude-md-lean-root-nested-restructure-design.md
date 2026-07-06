<!-- doc-meta: status=current; last-verified=2026-07-05 -->
# Design — CLAUDE.md into a lean root + nested per-subtree docs (+ product-vision framing)

> **Date:** 2026-07-05 · **Status:** Current · design approved by operator.
> **Goal:** cut the ~31k-token root `CLAUDE.md` to a lean ~5–7k-token index, move
> subsystem detail into on-demand nested `CLAUDE.md` files, add a concise product-vision
> block, and lose **no** load-bearing rule. Docs-only change; harness behavior unchanged.

## 1. Problem

Root `CLAUDE.md` is **315 lines / ~120 KB / ~31k tokens loaded every session**. It has:
- **Duplicated sections** — `### Feature flags …` (×2), `### One-GPU-per-cell … (spec 2026-05-31)` (×2),
  plus a **contradictory RunPod-image default** (one block says `cuda-runtime` is the default, another says `cuda-devel`).
- **Incident-narrative bloat** — long "why we did this" prose that the repo's *own* doc policy
  (`docs/policies/documentation.md`) says must live in specs/runbooks/memory, "keep only the resulting rule here."
- **No nested `CLAUDE.md`** and **no `@imports`** — everything is paid for on every session.

## 2. Approach — lean hub + on-demand nested spokes (no eager `@imports`)

Nested `CLAUDE.md` files load **only when Claude works in that subtree**, so their detail costs
zero context until relevant. That is the entire win.

- **Chosen:** nested files, **no `@imports`**. Root stays tiny; detail loads on-demand.
- *Rejected — `@import` everything into root:* imports load **eagerly** at session start → re-bloats every session.
- *Rejected — also fan content into `docs/`:* the nested files + existing tier-1 specs already cover it; more surfaces = more drift.

## 3. Target shape

### 3.1 Root `CLAUDE.md` (~5–7k tokens) — six sections
1. **What OpenResearch is & where it's going** *(new)* — the autonomous ML-paper-reproduction engine behind
   **deepinvent.ai**; autonomous reproduction *today* → an **experiment-ideation** research layer *next*.
2. **Quickstart** — only the daily commands (backend run, tests, frontend, `cli reproduce`); drop exhaustive flag dumps.
3. **How it works (30-second map)** — RLM root writes Python calling the **19 primitives**; file-backed run state; SSE lifecycle; the "where to look first" pointer table.
4. **Rules that always apply** — load-bearing invariants (evidence-gate/fail-closed, forced-iteration, safe-builtins, SDK isolation, commit/doc policy, auth gotchas), one line each → pointer to the owning nested file/spec.
5. **Doc map** — the 4 nested files + tier-1 specs + `system_overview.md`.
6. **Fidelity anchors + "keep this lean"** — the `19` count, RunPod `SECURE` default, context-mode routing, maintenance rule.

### 3.2 Four nested `CLAUDE.md` (detail relocated from root)
| Path | Holds |
|---|---|
| `backend/agents/rlm/CLAUDE.md` | the 19 primitives; the **feature-flag catalog** (the biggest bloat source, as terse rules); prompts; forced-iteration/campaign; evidence/grader/validator gates; the two RLM auth surfaces; model registry + per-role/Foundry selection |
| `backend/services/runtime/CLAUDE.md` | sandboxes; one-GPU-per-cell `run_matrix`; GPU capacity/catalog; cloud backends (RunPod/GKE/Azure); execution-reliability (streaming/stall/finalize-on-timeout) |
| `frontend/CLAUDE.md` | Next.js 16 conventions; one-image-two-processes; SSE run lifecycle + event types; server-side proxy routes; lab/leaderboard surfaces |
| `tests/CLAUDE.md` | socket-hermetic suite; pytest config; single/parallel runs; the CLAUDE.md fidelity guards |

## 4. Invariants — nothing lost, fidelity preserved
- Every load-bearing **rule** is *relocated + condensed*, never deleted. Incident **narratives** already have cited specs/runbooks — root/nested keep only the rule + citation.
- Fix the 2 duplicated sections and the contradictory RunPod-image default while splitting.
- **Extend `tests/test_claude_md_fidelity.py`** to read the **root + nested** `CLAUDE.md` set (union) for its env-var / count / citation checks — so the guard covers the whole doc system, not just root. The 7 documented env vars, the `19` count, and the RunPod `SECURE`-default line must each appear *somewhere* in the set (and still be read in `backend/` for env vars).

## 5. Verification
- `tests/test_claude_md_fidelity.py` green (extended form).
- Root token-count check < 8k.
- Grep-proof that every relocated rule landed in some nested file (no orphaned rule).
- Docs-only ⇒ backend/frontend build + the rest of the suite otherwise unaffected.

## 6. Delivery
- One milestone commit: *"Restructure CLAUDE.md into a lean root + nested per-subtree docs (+ product-vision framing)."*
- No co-author trailer; author = local config. Not pushed unless the operator asks.
- Update `docs/policies/documentation.md` / `current-docs.txt` only if the freshness checker requires the nested files be registered.
