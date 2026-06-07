# `bes` branch — doc-alignment audit (2026-06-07)

**Scope:** all changes the `bes` branch added vs `main` — `git diff main...bes` (merge-base `1522579` → `bes`): **140 files, +16,640 / −246**.
**Method:** 10-dimension multi-agent audit (forward: doc→code; reverse: new code→docs), every medium/high finding adversarially re-verified against the on-disk files. 39 agents. No finding was refuted; several HIGHs were down-graded HIGH→MEDIUM on verification.
**Doc baseline:** the on-disk `bes` `CLAUDE.md` (uses the `REPROLAB_` prefix; title reads "OpenResearch / ReproLab").

---

## TL;DR — the one thing to understand first

`bes` is a divergent **SDAR/GPU feature line** whose `CLAUDE.md` describes a *superset* of the code that's actually on the branch — partly **`main`-lineage features** (SDK isolation, `cuda-devel`) and partly **merge-base doc-debt** that is absent on *both* `bes` and `main` (PEEK-lite, signal handlers, the dockerfile-guard symbol, two runbooks). Two consequences dominate every finding:

1. **The docs over-describe the code.** A large class of documented features (PEEK-lite context-map, MUSE-lite negative-lessons, claude-agent-sdk isolation, CLI signal handling, the Dockerfile shape guard, two runbooks) **do not exist on `bes`** — they live on `main`'s lineage or nowhere. On `bes` these read as "doc claims X, code absent."
2. **The code under-describes itself.** The genuinely *new* code the `bes` push added (GPU reservation, replay UI + route, `serve_local_llm`/`watch_run`/`reserve_gpus`, best-of-run rubric floor, latest-metrics scoring, always-on FSDP sharding, `brev` backend, `codex_repair` primitive, vLLM executor tier, three new SSE events) is **largely undocumented** — violating CLAUDE.md's own "Maintaining this doc" contract ("when you add a primitive, an SSE event type, a sandbox … update both").

So the alignment story is *not* "the docs are slightly stale." It's "this branch's docs and code describe two different systems." Below, findings are bucketed by **what to actually do**.

---

## Bucket A — Genuine within-`bes` doc drifts (fix on this branch regardless of merge)

These are real and actionable independent of `main`. Code and doc both exist on `bes` but disagree.

| ID | Sev (verified) | Finding | Evidence |
|----|------|---------|----------|
| **ACC-2** | **HIGH** (confirmed) | CLAUDE.md says the gpt-5-mini endpoint route "uses `OPENAI_API_KEY` automatically." False — `_resolve_endpoint()` reads `REPROLAB_ACCELERATOR_API_KEY` (default `"local"`) and passes it straight through, so pointing at `api.openai.com` with defaults sends `Bearer local` → 401. | `accelerator.py:376,387‑392,551‑555`; `run.py:1436‑1444`; CLAUDE.md:81 |
| **SBX-1** | **HIGH** (confirmed) | RunPod default image documented as `cuda-runtime`; code uses `cuda-devel`. (`main` already corrected its doc in `062c59d`; `bes` did not.) | CLAUDE.md:146 (`cuda-runtime`) vs code default `cuda-devel` |
| **ACC-1** | MED (adj. from HIGH) | `REPROLAB_SUBRLM_OPENAI_TIMEOUT_S=120` documented as the route's timeout knob, but **no code reads it** (client hardcodes `timeout=300.0`). | grep: name only in CLAUDE.md; `openai_client.py:71`; `accelerator.py:549‑555` |
| **ACC-3** | MED (confirmed) | `REPROLAB_ACCELERATOR_API_KEY` (the var operators must set for any non-Azure endpoint) is read by all providers but documented nowhere. | `accelerator.py:214,275,376`; CLAUDE.md:81 lists only 4 accel vars |
| **PRIM-1** | LOW | The curated "**12** domain primitives" list has drifted from the actual tool set assembled by `build_custom_tools()`. (Not a clean "12→16": the registry includes non-root entries, and `resolve_gpu_requirements` *is* documented under dynamic-GPU. The one clearly-undocumented *root* primitive is `codex_repair` — see PRIM-2.) | `binding.py:725`; CLAUDE.md:114‑115,173 |
| **SSE-1** | MED (confirmed) | "All events route through `sanitize_iteration` — the single egress chokepoint" is false; some events bypass it. | CLAUDE.md:124 vs `sse_bridge.py` emit sites |
| **SBX-2** | MED (adj.) | `REPROLAB_RUNPOD_CLOUD_TYPE` documented default `COMMUNITY`; every code site defaults `SECURE` (≈2× cost). | `config.py:183`; CLAUDE.md:77 |
| **AUTH-1** | MED (confirmed) | `qwen3-coder`/`kimi-k2.5` are documented root models but `OPENROUTER_API_KEY` (their credential) is missing from the doc's model→credential mapping. | CLAUDE.md:57 vs :63‑67 |

---

## Bucket B — New `bes` code that is undocumented ("update both docs" violations)

The reverse audit's highest-value set. Each is real surface added by the push with no doc home.

| ID | Sev (verified) | New surface (undocumented) | Evidence |
|----|------|----------------------------|----------|
| **NEW-1** | MED (confirmed) | **GPU-reservation subsystem** — `gpu_reservation.py` (`GpuReservationManager`), `scripts/reserve_gpus.py`, `scripts/reserve_and_run_sdar.py`. Operator-facing; zero CLAUDE.md mention. | `gpu_reservation.py:60,202`; scripts |
| **SCR-1** | MED (confirmed) | Three new operator scripts: `serve_local_llm.py`, `watch_run.py`, `reserve_gpus.py`. CLAUDE.md's multi-GPU section names only `batch_reproduce.py` + `local_gpu_allocator.py`. | CLAUDE.md:148‑152 |
| **SCR-2** | MED (confirmed) | `REPROLAB_EXECUTOR` vLLM executor tier + accelerator `local/auto/azure` modes — undocumented (doc shows accelerator as `endpoint`/`off` only). | grep `REPROLAB_EXECUTOR` in docs = 0 |
| **SCR-4** | MED (confirmed) | Backend route **`GET /runs/{id}/replay-events`** absent from CLAUDE.md's route list. | route def vs CLAUDE.md:120‑124 |
| **SSE-2** | MED (confirmed) | Undocumented SSE event **`experiment_completed`**. | emit site vs CLAUDE.md:123 |
| **SSE-3** | MED (confirmed) | Undocumented terminal SSE events **`run_fatal`, `run_interrupted`**. | emit sites vs CLAUDE.md:123 |
| **NEW-4 / BEH-4** | MED (confirmed) | **best-of-run rubric floor** + **latest-metrics scoring** shipped in `report.py`/`leaf_scorer.py`; CLAUDE.md's `final_report` description never mentions either. | `report.py:543‑576,711‑728` |
| **SBX-3** | MED (confirmed) | New **`brev` sandbox backend** exists; doc lists only `local/docker/runpod` (and `--sandbox` choices omit it). | `brev` backend on disk; CLAUDE.md:52,142 |
| **PRIM-2** | MED (confirmed) | **`codex_repair`** primitive (+8 `REPROLAB_CODEX_*` env vars in `.env.example`) absent from CLAUDE.md. | `primitives.py`; `.env.example:56‑64` |
| **BEH-1** | MED (adj.) | **Always-on multi-GPU FSDP/accelerate-launch sharding** (`_resolve_distributed_launch` fires whenever >1 GPU; only opt-OUT `REPROLAB_DISABLE_TORCHRUN_WRAP`). Docs still frame multi-GPU as count-only leasing. | `primitives.py:3156‑3158,2667` |
| **BEH-3** | MED (adj.) | **Model-level graceful degradation** (failed/skipped *models* now excluded from rubric, not just datasets); the fidelity spec still says exclusion "does not extend to" this. | `leaf_scorer.py:_detect_data_unavailable_leaves` |
| low | low | Allocator own-holder discount (`own_pids`); `REPROLAB_MIN_TRAIN_WALL_S`, `REPROLAB_DISABLE_TORCHRUN_WRAP`; `batch_reproduce --accelerator local|auto`; `heartbeat`/`recommend_next_tool`/`resolve_gpu_requirements` primitives; replay backend route in lifecycle section. | per-finding |

**Spec status drift (NEW-3 / BEH-2, MED confirmed):** `docs/superpowers/specs/2026-05-30-rubric-scoring-fidelity-design.md:4` reads "**Status:** … implementation pending," yet Component A (metrics-completeness gate) and part of D (events) **shipped** on `bes` (Component B remains unimplemented). The spec is also **uncited** from CLAUDE.md's reference-docs list. Update the spec's status header and cite it.

---

## Bucket C — Docs describe code that is absent on `bes` (resolves on merge; or shared doc-debt)

These are the "doc says X, code missing" cluster. They are **not** new drift caused by the push — they come from `bes`'s `CLAUDE.md` being a main-lineage copy. Split by where the code actually lives:

**C1 — On `main`, missing on `bes` (a real gap, but merge-time — the `bes` push never touched these files):**
- **STAB-1 (MED, confirmed) — claude-agent-sdk isolation.** CLAUDE.md:160 mandates `setting_sources=[]` + explicit `mcp_servers` + non-plan `permission_mode` at three sites as a *security boundary* (BUG-NEW-038). Defensible facts on `bes`: `claude_runtime.py:82` *does* construct `ClaudeAgentOptions` with `permission_mode` (`:84`) but **omits `setting_sources=[]`**, and `_resolve_mcp_servers()` (`:90,165`) *actively loads* MCP servers — the opposite of the documented `mcp_servers={}`. `main`'s `claude_runtime.py` has `setting_sources` (2 hits); `bes` has 0. → The documented isolation mitigation is not implemented on `bes`, so the inner model can pick up the dev's `settings.json` + MCP servers. **Out of push scope** (the push didn't modify this file) — close it at merge, not as a `bes`-push fix.

**C2 — Absent on BOTH `bes` and `main` (pre-existing doc-debt; out of strict push scope, flagged for completeness):**
- **FLAG-1** — PEEK-lite context-map (`REPROLAB_CONTEXT_MAP`, `context_map.py`, `read_context_map()`): `context_map.py` exists on **neither** branch.
- **FLAG-2** — MUSE-lite negative-lessons (`REPROLAB_NEGATIVE_LESSONS`, `lesson_distiller.py`, `mine_lessons`, `_negative_lessons_block`): all **absent on `bes`**.
- **STAB-3** — CLI signal handling (`_install_termination_handlers`, `killed` status, `killReason`): absent in `cli.py` on both.
- **STAB-2** — `RunStatus` Literal must include `killed`+`interrupted`: both absent in `live_runs.py` on both (code still *writes* `interrupted`).
- **STAB-4** — `_validate_dockerfile_shape`: **not defined in `primitives.py` on either branch** (a different LLM-repair mechanism exists instead).
- **SCR-7** — CLAUDE.md cites `docs/runbooks/running-the-project.md` (×2) and `2026-05-29-monitoring-loops.md`; **both files are absent** on both branches → broken doc references.

> Note: because these are present in `main`'s CLAUDE.md too, fixing them is a repo-wide doc task, not a `bes`-only one — but they will mislead anyone reading `bes`'s docs today.

---

## Bucket D — Merge/integration risk + repo hygiene (my own deterministic findings)

**D1 — Rename divergence (merge risk, NOT within-branch drift).** `main` completed `ReproLab→OpenResearch` (`OPENRESEARCH_*`); `bes` uses `REPROLAB_*` (282 hits, 0 `OPENRESEARCH_`) and its CLAUDE.md title hedges "OpenResearch / ReproLab." `main` also added a docs-freshness system + `docs/policies/documentation.md` that `bes` lacks. **Merging `bes`→`main` will conflict heavily on env-var names and docs.** Decide the rename direction *before* merging; within `bes`, code and docs are self-consistent, so this is purely an integration concern.

**D2 — 6.1 MB binary committed.** `paper-repro-bes-docs.zip` (the literal tip commit "adding docs") is a 6.1 MB zip of BES proposal/architecture/transcript markdown + an `out/` dir of A/V transcription scripts and a `montage.png`. Bloats history permanently; should be unpacked into `docs/` (markdown) or kept out of git (release asset / LFS).

**D3 — Run artifacts committed beyond the whitelist.** `runs/prj_09047604e591d969/` (17 files, 0.3 MB) includes `train_run.log`, `train_run2.log`, `cost_ledger.jsonl`, `dashboard_events.jsonl`, `code/metrics.json`, `code/commands.log`, `reports/*` — files **outside** the `.gitignore` whitelist (which intends to track only `final_report.{json,md}`, `demo_status.json`, `batch_child.log`, `experiment_runs.jsonl`). Likely an accidental `git add`; remove from tracking.

---

## Recommended order of operations

1. **Fix Bucket A now** — these mislead anyone using `bes` today (ACC-2 and SBX-1 are HIGH; both are one-line doc/code reconciliations).
2. **Close STAB-1 (C1)** — port the SDK-isolation hardening to `bes` (security boundary) or remove the invariant from `bes` docs until merge.
3. **Document Bucket B** — add the new GPU-reservation/replay/scripts/scoring/sandbox/primitive/SSE surface to CLAUDE.md (honor the "update both" rule); fix the fidelity-spec status header.
4. **Repo hygiene (D2/D3)** — drop the zip + stray run artifacts from tracking.
5. **Plan the rename merge (D1)** before integrating `bes`; treat Bucket C2 as a repo-wide doc-debt cleanup.
