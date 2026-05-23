## 2026-05-23 — Reliability + production-path sprint

_Updated: 2026-05-23._

### Headline

15 commits across UI hardening, backend stability, Azure provider support, chat steering, and RLM behavior quality; repo is production-ready for single-user demo and staged for Vercel + Azure migration.

### Commits landed (newest first)

| SHA | Headline | Lines |
| --- | --- | --- |
| `b290449` | A 30-minute run was crashing at second-39 because paper_claims came back as a list | +116 / -1 |
| `2b7bcf1` | The graph is the workspace now — side rails are drag-resizable and viewport-aware | +655 / -21 |
| `ec76d91` | A second opinion at decision time: recommend_next_tool gives the root a Reflexion-lite advisor | +154 / -3 |
| `ad460ff` | Azure OpenAI joins the root-model lineup — the path to Azure deployment opens | +387 / -10 |
| `4e3bd38` | Two stability levers so SDK-deadlock runs stop pretending to be alive | +707 / -4 |
| `8c3371e` | The root was ignoring rlm_query — primitives now ask for it themselves on big slices | +229 / -17 |
| `9d7f8e9` | fix(lab+harness): counter strip layout + wrap_primitive arg coercion + system prompt rlm_query nudge | +157 / -4 |
| `c6511be` | feat(sidebar): aggregate counter strip + enriched subrlm/baseline node detail | +343 / -8 |
| `ad32198` | feat(sidebar): add Upload nav item above Lab/Library — one-shot ?new=1 trigger | +72 / -1 |
| `0cc2e77` | fix(lab): SSR-safe elapsed clock — initialize nowMs=null to avoid hydration mismatch | +13 / -3 |
| `1316b8a` | fix(rubric): align rubric_area key contract + defensive empty-name handling | +95 / -14 |
| `e4c30a0` | fix(lab): live status flip on primitive_call + real-time elapsed clock + RDR polling stops on empty 200 | +90 / -8 |
| `f664659` | docs+chore: runbook + CLAUDE.md + system_overview for chat steering / sidebar / F7 / backend-hang remedy | +134 / -11 |
| `becfac8` | fix(landing): bench cells render cleanly without overlap | +16 / -8 |
| `a07c336` | feat(rlm+ui): real-time chat steering + collapsible right sidebar with kind-specific node detail | +2034 / -55 |

### What works end-to-end

| Capability | Status | Where verified |
| --- | --- | --- |
| OAuth root + local GPU sandbox | ✅ | live arXiv run; watchdog/heartbeat flags surface accurately |
| API-key root parity | ✅ | `tests/rlm/test_chat_steering_auth_parity.py` (6/6) |
| Azure OpenAI provider | ✅ | `tests/rlm/test_build_llm_client.py::TestAzureOpenAI` (4/4) |
| SDK aclose deadlock recovery | ⚠️ | flagged via watchdog + heartbeat; not yet auto-recovered (out of scope this session) |
| paper_claims as list-of-dicts | ✅ | `tests/rlm/test_paper_claims_coercion.py` (7/7) |
| Failed-run lab UI + Rerun button | ✅ | shipped this session (see commits) |
| Chat steering POST → primitive round-trip | ✅ | curl verified; in-band UI |
| Sub-RLM spawning under hint | ✅ | live run: 5 sub-RLMs with focused decomposition |
| Resizable workspace | ✅ | shipped this session |

### Open / deferred

- W4 tool router landed but ROI untested in the wild — needs 3–5 runs to see if root actually calls `recommend_next_tool`.
- aclose deadlock root cause is upstream SDK; we observe + flag but don't auto-recover. Future track: subprocess-level restart-on-degraded.
- Cost ledger reports $0 for OAuth runs (SDK doesn't surface tokens). Documented in CLAUDE.md.
- Vercel/Azure production migration: provider abstraction ready (W2c); state migration (Postgres + Blob) + multi-tenant auth not started.
- Frontend vitest blocked on Node 21.7.3 (needs ≥22.12). Tests pass under Node 22; CI / coworker workflow notes in CLAUDE.md.

### Test surface

Backend: 681 passed, 1 xfailed (`tests/rlm/ tests/rdr/ tests/routes/ tests/agents/ tests/services/events/`). Frontend: tsc clean across all changes; vitest blocked on Node 21 (works under Node 22).

---

# RLM Phase 5/6 — Progress

_Updated: 2026-05-22 — debug-and-harden session._

## Objective

Debug-and-harden pass: fix the issue catalogue (I1–I13 + I4), drive PaperBench
papers end-to-end (plus one harder recent paper), and complete GitHub issue #62.

## Status

The session's seven commits (`2630a77` P0, `4e7b4a4` catalogue I5–I13,
`52625d6` run_experiment Bug A/B/C + I3, `d656c7d` I4 + idempotency deflake,
`c22feb7` I3 revert, `652f842` rdr design spec + impl prompt, `bfe9e3d` run-3
outcome) are **squashed into one commit on `origin/main`**, whose `Closes #62`
keyword closed issue #62. The `merge` branch keeps the un-squashed history.

Test suite green (I3's 4 tests removed with the revert): 1252 passed, 3 skipped.

## run_experiment Bug A/B/C — fixed and verified

Run 2's `run_experiment` failed in 6 s. Three compounding bugs in
`backend/agents/rlm/primitives.py`, all fixed and **verified live by run 2b**:

- **Bug A** — `_execute_in_sandbox` logged stdout only; a failed command's
  stderr traceback was discarded. Fix: `_combine_command_output` joins both.
- **Bug B** — the experiment ran the image `detect_environment` built before
  any code existed (missing deps). Fix: `run_experiment` rebuilds from
  `ctx.project_dir/Dockerfile` via `build_environment`.
- **Bug C** — the sandbox ran `network_disabled`. Fix: `_execute_in_sandbox`
  enables network for the experiment container (user-approved).

## I3 — reverted (root-prompt change backfired)

The `_PAPER_GROUNDING` section anchored the `qwen3-coder-featherless` root on
the understanding phase — run 3 looped on `understand_section` for 21
iterations and never reproduced. Reverted; the known-good prompt is restored.
The `ftrl` acronym-collision I3 targeted is unaddressed — a robust fix needs
more than a prompt nudge on this root model. See `learn.md` 2026-05-22.

## I4 — fixed

Workspace `paper_text` now loads from the parser's full-text blob (located via
the `ParsingCompleted` event), not a lossy chunk-reassembly. Guard:
`test_paper_text_equals_parser_full_text`.

## Runs

See `runlog.md`.

- **Run 1** — sequential-neural-score-estimation: leaf **0.366**, partial.
- **Run 2b** — mechanistic-understanding (`pb_..._1779457326`): leaf **0.079**,
  failed — `run_experiment` succeeded (Fix A/B/C verified live); weak baseline.
- **Run 3** — GoRL (arXiv 2512.02581): first attempt looped under I3; re-run
  (I3 reverted) progressed through the full pipeline but crashed on the
  Featherless Qwen3-Coder 49 152-token context cap. `failed`, leaf 0.0.

## rdr harness — built (2026-05-22)

The `rlm_rubric_orchestration` branch carries the full rdr harness — six
modules under `backend/agents/rdr/` (`models`, `decomposer`, `context_engineer`,
`agent`, `controller`, `run`), the `--mode rdr` CLI wiring, the
`scripts/rdr_paperbench.py` launcher, and 112 rdr tests including a full
offline end-to-end on the real `sequential-neural-score-estimation` bundle (27
clusters). The deterministic controller reproduces the paper cluster-by-cluster
against the official rubric and repairs weak clusters in a capped loop; no LLM
in the control path. Provider/model is dynamic — Claude OAuth (Sonnet) locally
or Azure OpenAI — via the existing `collect_agent_text` runtime resolution.

Full test suite green: 1362 passed, 3 skipped (the 3 are pre-existing
optional-dep skips: chromadb, tesseract). Built across six phase commits
squashed into one milestone commit on `rlm_rubric_orchestration`.

## Remaining

Real live end-to-end run (Claude OAuth + local GPU) on a PaperBench bundle —
verify the leaf score beats the rlm baseline (≈0.37) on
`sequential-neural-score-estimation`. Production wiring into the UI / SSE
bridge if the live run validates. See the design spec §10 success criteria.
