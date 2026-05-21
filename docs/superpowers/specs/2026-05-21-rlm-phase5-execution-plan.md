# RLM Phase 5 — End-to-End Execution Plan (#62)

> Executable companion to `2026-05-21-rlm-phase5-e2e-handoff.md`. Read the
> handoff for full background; this doc is the **plan of record** for Phase 5,
> refined by the 2026-05-21 recon + design grill. Updated as #59 lands and as
> decisions resolve.

## 1. Decisions locked (2026-05-21 grill)

- **D1 — Stay independent of #59.** Do not edit `primitives.py` / `binding.py`
  / `dashboard_emitter.py` or any #59-owned file. Aayush is actively on #59;
  its Tasks 10–12 are *strictly sequential* edits to `primitives.py` (#59 plan
  "Build order") — a concurrent edit clobbers. We pull #59's Tasks 10–14 as
  they land and re-merge.
- **D2 — Integration branch `feat/rlm-phase5-e2e`**, cut from
  `feat/rlm-phase3-orchestrator` (#60, complete). Phase 5 work + the #59↔#60
  merge live here; PR → `rlm-pivot`. Keeps #60's own PR isolated.
- **D3 — All four root models are supported targets** (GPT-5, Qwen3-Coder,
  Kimi K2.5, Claude); iteration/testing runs on the cheapest; the demo runs on
  a paper-validated model (GPT-5 or, if preferred, Claude).
- **D4 — Sandbox = Docker local** for now (user enabling Docker Desktop WSL
  integration); RunPod stays supported; a **Brev** backend is added later.
- **D5 — Qwen/Kimi root models: DEFERRED.** Phase 5 ships on **GPT-5 + Claude**
  (both keyed in `.env`, runnable today). Wiring Qwen3-Coder/Kimi (a Featherless
  or OpenRouter route + `api_key` plumbing) is a deferred follow-up — see §4.

## 2. Recon findings (2026-05-21 — corrects/updates the handoff)

1. **#59 is at Task 9/14, 7/9 primitives real.** Real: `understand_section`,
   `extract_hyperparameters`, `detect_environment`, `build_environment`,
   `plan_reproduction`, `implement_baseline`, `run_experiment`. Remaining #59
   work = **Tasks 10–14**: `verify_against_rubric` (T10), `propose_improvements`
   (T11), remove `set_final` (T12), `PRIMITIVE_DESCRIPTIONS` + registry/`__init__`
   (T13), integration test (T14).
2. **Ownership resolved.** The `NotImplementedError("Phase 3 (#60) …")` strings
   in #59's `primitives.py` are *stale #58 skeleton text*. #59's own plan
   (`phase2-implementation-plan.md`) owns all nine primitives — T10/T11 included.
   #60 is correctly complete and does not implement them.
3. **The deliverable gate is narrow.** A real run with a real score needs
   **#59 T13** (so `build_custom_tools` binds *real* primitives instead of #60's
   loud stub fallback) **+ T10** (so the rubric score is real). T11/T12/T14 are
   improvement-loop / cleanup / test — not blockers for a baseline-only score.
4. **The #59↔#60 merge is textually clean but needs a *semantic* review.**
   `git merge-tree` exits 0, but both branches rewrote `system_prompt.py` (#59
   for its loop; #60 for the real-`rlm` `.format()` template that `run.py`
   consumes). Resolution: **take #60's `system_prompt.py`**. #58's hand-built
   `root_loop.py` / `repl_host.py` / `sub_call.py` are dead under brief §3 (the
   engine is the `rlms` library) — leave them for Phase 6, do not delete here
   (surgical-change rule).
5. **No OAuth/subscription path for the root model.** `rlms` 0.1.1's clients
   (`openai`, `anthropic`, `litellm`, …) authenticate via **API keys** only —
   there is no Claude-subscription / OAuth path. `ANTHROPIC_API_KEY` is in
   `.env`, so the `claude` root works via the API directly; a `claude`-CLI shim
   is rejected (brittle, slow, muddies the demo's architecture story).
6. **Qwen/Kimi are not runnable as configured.** #60's `models.py` routes
   `qwen3-coder` / `kimi-k2.5` through the **`openrouter`** backend, but `.env`
   has **`FEATHERLESS_API_KEY`**, no `OPENROUTER_API_KEY`. The registry's
   `backend_kwargs` also omits an explicit `api_key`, so non-OpenAI backends get
   no key plumbed at all. Today **only `gpt-5` and `claude` actually run.** §4 + WS-C.
7. **`ftrl` is a real PaperBench paper** — Wolczyk et al., *"Fine-tuning RL
   Models is Secretly a Forgetting Mitigation Problem"*, ICML 2024 Spotlight
   (rubric: 233 nodes / 178 leaves). Only its `paper.md`/`addendum.md`/
   `rubric.json` are synthetic placeholders.
8. **Docker is unavailable** in this WSL2 distro (user enabling). `rlm` imports
   fine; Python 3.12.3; network to GitHub works.

## 3. Workstreams

Each step is `do → verify`. WS-A/B/C/D run this session (#59-independent);
WS-E is gated; WS-F runs throughout.

### WS-A — Integration branch + the #59↔#60 merge
1. Cut `feat/rlm-phase5-e2e` from `feat/rlm-phase3-orchestrator`.
2. `git config rerere.enabled true` — record the `system_prompt.py` resolution
   once; every later #59 re-merge replays it automatically.
3. Merge `origin/feat/rlm-phase2-foundation`. Resolve `system_prompt.py` →
   #60's version; `__init__.py` → union both export lists.
   *Verify:* `.venv/bin/python -m pytest tests/rlm/ -q` green; `import
   backend.agents.rlm` clean; no `<<<<<<<` markers.
4. Re-merge after every #59 task lands (D1).

### WS-B — Vendor the real PaperBench bundles
1. Recon `openai/preparedness` properly (the recursive-tree API call 404'd —
   resolve the branch SHA via `git ls-remote`, then the trees API; or sparse
   clone). Locate the upstream `ftrl` bundle + the full paper catalogue.
2. Vendor real `paper.md` / `addendum.md` / `rubric.json` into
   `third_party/paperbench/ftrl/`, overwriting the placeholders; keep `config.yaml`.
3. Shortlist 2–3 genuinely easy / CPU-runnable papers (small rubric, no GPU
   training) — **bring the shortlist to the user to pick** (demo content).
4. Vendor the chosen extra paper(s) — one Sonnet sub-agent per bundle dir
   (disjoint paths, parallel-safe).
   *Verify:* `reprolab paperbench summary --paper-id ftrl` → `node_count` /
   `leaf_count` match `config.yaml` (233 / 178) and PaperBench Table 7.

### WS-C — Root-model wiring — DEFERRED (D5)
Qwen3-Coder/Kimi wiring (provider route + `api_key` plumbing) is deferred.
Phase 5 runs on GPT-5 + Claude, which the OpenAI/Anthropic SDKs key directly
from `.env` — no `models.py` change needed. Revisit post-Phase-5.

### WS-D — Stub-path end-to-end smoke (proves #60's machinery)
1. With `REPROLAB_RLM_STUB_PRIMITIVES` set, `run_pipeline_rlm` on a tiny mock
   paper, cheapest root model.
   *Verify:* Algorithm-1 loop iterates; `repl_iteration` SSE events stream; the
   SQLite checkpoint records each iteration; `final_report.{json,md}` is written;
   **corpus-leak invariant** — the mock paper text appears in no event / store
   row / snapshot.
2. Cost: cents-scale; flagged, not significant.

### WS-E — Real end-to-end run (GATED: #59 T13+T10, Docker up)
1. When #59 reaches T13: re-merge; confirm `_resolve_custom_tools` logs
   `"real (#59 binding)"`, not the stub fallback.
2. `python -m backend.cli reproduce third_party/paperbench/ftrl --mode rlm
   --provider <…> --sandbox docker`. Fix failures one class at a time.
3. Repeat for the second paper.
   *Done:* ≥2 `runs/<id>/final_report.{json,md}` with real rubric scores.

### WS-F — Docs + PRs (throughout)
- `CHANGELOG.md` + `learn.md` entries (append-only); `system_overview.md` /
  `CLAUDE.md` kept in sync.
- #60 PR (`feat/rlm-phase3-orchestrator` → `rlm-pivot`) — body in handoff §10;
  user creates it (no `gh` auth).
- Phase 5 PR (`feat/rlm-phase5-e2e` → `rlm-pivot`) when WS-E completes.

## 4. Resolved — Qwen/Kimi provider routing (DEFERRED)

Recon finding #6: Qwen3-Coder & Kimi need a keyed provider + an `api_key`
plumbing fix in `models.py`. **Decision (user, 2026-05-21): defer.** Phase 5
ships on GPT-5 + Claude — both already runnable. Qwen/Kimi become a clean
follow-up (add a Featherless route, or an OpenRouter key, then plumb `api_key`).
D3 ("all four targeted") is satisfied as a roadmap item, not a Phase 5 blocker.

**Forward-compat guarantee (user, 2026-05-21).** `models.py`'s registry already
carries the `kimi-k2.5` and `qwen3-coder` `RootModel` entries — enabling them
later is purely additive (a provider route + `api_key`), no redesign. Phase 5
keeps all model selection registry-driven via `resolve_root_model`, so Kimi/Qwen
stay drop-in the moment a key is added.

## 5. Sequencing & gating

- **Now, parallel:** WS-A (merge) ∥ WS-B (vendor) — disjoint files. WS-D after
  WS-A (runs on GPT-5/Claude — no WS-C needed). WS-C deferred (D5).
- **Gated:** WS-E needs #59 T13+T10 *and* Docker. Until then WS-D is as far as
  we run.
- **Recurring:** `git fetch origin` at the top of every step; re-merge #59
  whenever `origin/feat/rlm-phase2-foundation` advances (D1, rerere from WS-A).

## 6. Risks

- **R1** — #59 slips; WS-E can't complete this session. Mitigation: WS-A–D are
  #59-independent and de-risk the merge + machinery; WS-E runs the moment #59
  lands T13.
- **R2** — per-primitive deadlines (Codex H2): `rlm`'s `max_timeout` doesn't
  interrupt a wedged primitive. #59's `build_environment`/`run_experiment`
  should bound their own execution — *verify* when #59's code lands; if absent,
  coordinate the fix with Aayush (do not edit his files unilaterally — D1).
- **R3** — a real paper surfaces primitive / root-model-prompt bugs. Mitigation:
  WS-D shakes out the machinery first; WS-E iterates one failure class at a time.
- **R4** — vendored paper-text licensing. PaperBench papers are openly released
  with the benchmark; vendor under `third_party/` with attribution (existing
  pattern) and record the source commit.

## 7. Demo-polish candidates (elegant, non-breaking wins)

`[small]` = done inline · `[ask]` = needs approval · `[flag]` = note only.

- `[deferred]` Featherless route + `api_key` plumbing — deferred with Qwen/Kimi (D5).
- `[small]` `git rerere` for clean #59 re-merges (WS-A).
- `[small]` explicit `api_key` plumbing in `models.py` `backend_kwargs` (a
  latent #60 gap — non-OpenAI backends currently get no key).
- `[ask]` cheap `claude`/Sonnet test-root entry.
- `[flag]` delete dead `root_loop.py`/`repl_host.py`/`sub_call.py` — Phase 6 / #63.

## 8. Doc-update contract

`CHANGELOG.md` entry per architectural change; `learn.md` Rule/How/Why per
lesson; both append-only. `system_overview.md` / `CLAUDE.md` kept in sync. This
plan is updated as #59 lands and decisions resolve.
