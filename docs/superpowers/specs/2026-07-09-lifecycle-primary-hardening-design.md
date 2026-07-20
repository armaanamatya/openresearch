# Lifecycle-primary hardening → default-flip → validation — design

> **Doc status:** Draft · spec tier · authored 2026-07-09. Grounded by a deep read-only recon
> of `backend/agents/rlm/lifecycle_driver.py`, its wiring in `run.py`, and the shared
> primitive/telemetry machinery in `binding.py` + `sse_bridge.py`, on
> `chore/cleanup-sweep-2026-07-07`. Baseline: 168 lifecycle tests green (unit-level).
> Policy: `docs/policies/documentation.md`.
> **Companion spec:** `2026-07-09-cloud-posture-gcp-azure-primary-design.md` (Spec A) lands
> first so the A/B validation here runs on a trustworthy GCP/Azure cloud.

## 1. Problem

The project is committing to the **deterministic lifecycle harness** as *the* root loop and
retiring the flaky free-roaming `rlm.completion()` path. The motivation is all four of:
reliability (completion stalls/degenerate-loops, hostage to SDK/OAuth flakiness), determinism
& debuggability (auditable stage graph, resumable runs), a structured **substrate for the
future experiment-ideation layer**, and hard **cost bounding** (fixed stages, capped
repairs/climbs).

The harness is **~80% built and unit-tested, but blind at the integration seams.**
`lifecycle_driver.drive_lifecycle_chain` (`:183-496`) already runs
`understand → detect → plan → implement → run → verify` with bounded repair (stagnation-
fingerprint early-exit) and `run_lifecycle_primary` (`:499-674`) adds a bounded improvement
climb (`propose_improvements → implement → re-run → re-verify`, best-of tracking). It is wired
behind `OPENRESEARCH_LIFECYCLE_PRIMARY` (`run.py:4469-4517`), gated by an inputs-ready check
(`_primary_inputs_ready`, `:1194`), and synthesizes an `RLMChatCompletion`-shaped result for
finalize (`_synth_result_from_summary`, `:1217`). Cost/experiment/lifecycle ledgers come free
because the driver calls the already-wrapped primitives.

**But four load-bearing seams are missing** — the reason it cannot be default today:

1. **No `rubric_score` SSE events.** The completion path emits these via
   `binding.wrap_primitive` (`:897-1035`); the driver emits only `lifecycle_drive_step`
   (`:307-310`). Dashboard/leaderboard go blind mid-run.
2. **`ctx.latest_rubric_score/target/iteration` never set.** The completion path sets them
   after every verify (`binding.py:919-921`); the driver only writes `summary["rubric_score"]`
   (`:493-494`). Finalize's suspicious-partial check and any hybrid handoff read stale/None.
3. **No iteration counter.** The driver runs once, not in a loop; there is no monotonic
   counter. Consequences: rubric events have nothing to key on, and finalize stamps
   `iterations=0` (`run.py:4614` reads `rlm_logger.iteration_count`, which is 0 when the root
   never ran) even though six primitives executed.
4. **No per-step checkpointing.** The completion path checkpoints after every primitive
   (`binding.py:1061-1067`); the driver has no checkpointer, so a mid-run crash has no resume
   point inside a stage. And there is **no OFF-byte-identical / ON-end-to-end test pair.**

Three lower-severity gaps: improve-phase has no per-hypothesis observability; the canonical
`evidence_bundle.json` is not written by the driver; campaign-loop integration with a
lifecycle-synthesized result is unverified.

> **Post-extraction correction (2026-07-09).** A verbatim read of `binding.py` resolved the
> open question in §7 and narrows the blocker list. The lifecycle driver calls the
> **already-wrapped** primitives, so `binding.wrap_primitive` — on a successful
> `verify_against_rubric` — **already** sets `ctx.latest_rubric_score/target/iteration`
> (`binding.py:919-921`) **and** emits the `rubric_score` event via `ctx.emit`
> (`binding.py:1015-1035`), *even under lifecycle-primary*. So **seam 1 (events) and seam 2
> (ctx state) are largely already satisfied** — they simply carry `iteration=0`. The true
> root cause collapses to a single keystone plus two follow-ons:
> - **Keystone — iteration counter.** `ctx.current_iteration` is advanced only by the
>   root-loop `OpenResearchRLMLogger`, which never runs in lifecycle mode; it stays `0`. That
>   is why the (real) events and ctx state are all stamped `0`. **Fix:** have the lifecycle
>   driver own and advance `ctx.current_iteration` per step (this makes seams 1+2 correct *for
>   free*, since binding reads it).
> - **Finalize source.** `run.py:4614` passes `iterations=rlm_logger.iteration_count` — 0 in
>   lifecycle mode. **Fix:** under `_primary_active`, source the iteration count from the
>   driver summary instead of the (unused) root logger.
> - **Checkpoint shape.** `IterationCheckpointer.record(clean)` expects a sanitized *root
>   iteration* (`response`, `code_blocks`, `sub_calls`) — a shape lifecycle stages don't
>   produce. **Fix:** a lightweight stage-checkpoint (stage, iteration, code_path, latest
>   score) rather than reusing the root-iteration record.
>
> **Consequence for the design:** the `RunRecorder` shrinks — the risky
> `binding.wrap_primitive` refactor (§3.1) is **no longer needed for parity**, because binding
> already does the emit/ctx work. The recorder becomes a small lifecycle-side object that owns
> the iteration counter, advances `ctx.current_iteration` before each step, and writes the
> stage checkpoint. This is **lower risk and less code** than the original centerpiece. §3.1's
> "binding-refactor" phase is dropped; the "lifecycle-first" phase is the whole of it.

## 2. Goal / non-goals

**Goal:** make lifecycle-primary produce a downstream contract **byte-for-byte
indistinguishable** from the completion path (same events, same `ctx` state, same on-disk
artifacts, same finalize result shape), add resumability and full auditability, wire a clean
seam for the future ideation layer, prove parity with paired A/B runs on GCP/Azure, then flip
the default — keeping completion behind the flag as a reversible escape hatch until parity is
proven, retiring it (and its forced-iteration policy / degenerate-loop detector / RLM logger)
in a later, separate change.

**Non-goals (this spec):** deleting the completion path or its machinery now (a later spec
once parity is proven); building the ideation *feature* (this spec only lands the seam);
changing the 19 primitives, the evidence/verdict layer, the grader, or the cost model; the
cloud-posture work (Spec A).

## 3. Design

### 3.1 Centerpiece — a path-agnostic `RunRecorder`

The four blocker seams are all things `binding.wrap_primitive` already does for the completion
path. Rather than reimplement them inside the driver (duplication that drifts), **extract the
recording responsibilities into one small `RunRecorder`, created once per run, that both paths
call.** Lifecycle gets parity *by construction*, and retiring completion later becomes
"delete the caller," not "untangle telemetry."

```
RunRecorder(ctx, emit, checkpointer)
  ├─ iteration: monotonic int, harness-owned          # keystone — fixes gap 3
  ├─ next_iteration()               -> bumps + returns the counter
  ├─ record_primitive(name, result) -> ledger/checkpoint bookkeeping (gap 4)
  ├─ record_rubric(verify_result)   -> build_rubric_score_event + emit (gap 1)
  │                                     AND set ctx.latest_rubric_{score,target,iteration} (gap 2)
  └─ checkpoint(stage, code_path, score)  -> atomic rlm_state write (gap 4, resume)

completion path:  binding.wrap_primitive delegates its emit/ctx/checkpoint bookkeeping
                  to RunRecorder  (refactor; output asserted byte-identical)
lifecycle path:   drive_lifecycle_chain / run_lifecycle_primary call RunRecorder after
                  each successful step and after each verify
```

**Iteration semantics:** the counter increments **per driver step** (each of
understand/detect/plan/implement/run/verify bumps it; each bounded repair bumps it; each climb
round bumps it). This yields a non-zero, monotonic count that mirrors "root turns" closely
enough for finalize and observability, and is what the rubric event / `ctx.latest_rubric_
iteration` / final-report `iterations` all read.

**Risk-managed sequencing (implementation plan will enforce):** introduce `RunRecorder` and
make the *lifecycle* path use it first (new surface, low risk), with a test asserting its
emitted event stream + `ctx` state match a golden capture from a completion run on the same
fixture. Only then refactor `binding.wrap_primitive` to delegate, guarded by the existing 168
lifecycle + completion event-stream tests plus a new "completion output byte-identical" test.
If the binding refactor proves too risky, the fallback is driver-local recording that
reproduces the same emissions (Approach B) — more duplication, same external contract.

### 3.2 Improve-phase observability (gap 5)

In `run_lifecycle_primary`'s climb loop (`:577-674`), emit a per-iteration event naming the
chosen hypothesis and whether it improved the score (reuse `lifecycle_drive_step` with a
`phase: "improve"` + `hypothesis` payload, routed through `RunRecorder`). The dashboard can
then render the climb, not just a single opaque `run_lifecycle_primary` call.

### 3.3 Evidence bundle + campaign integration (gaps 6, 7)

- After a successful `run_experiment`, write `rlm_state/evidence_bundle.json` (via the
  existing `evidence_bundle` builder) when `OPENRESEARCH_CANONICAL_EVIDENCE_BUNDLE` is on, so
  the leaf scorer resolves the same "which metrics back this score" receipt under both paths.
- Add a test that the **campaign attempt loop** (`reproduction_campaign.py`) correctly reads a
  lifecycle-synthesized `result_obj` (verdict/score/target), so repeat-until-reproduced works
  under lifecycle-primary.

### 3.4 Resumability

The driver's `start_stage` machinery (`need_baseline … can_finalize`) + the persisted
`rlm_state/reproduction_plan.json` (`:557-569`) already sketch resume. The `RunRecorder`
checkpoint (§3.1) completes it: on restart, read the last checkpoint to pick the `start_stage`
and rehydrate `code_path` / latest score, so a crash mid-`run_experiment` resumes at
`need_experiment` rather than from scratch. This directly serves the determinism/reliability
motivation.

### 3.5 Ideation seam (substrate, not feature)

Add a **documented extension point after the backbone+climb**: an optional
`propose_novel_experiment` hook (default-OFF, `OPENRESEARCH_IDEATION_PRIMARY` reserved),
distinct from `propose_improvements` (which climbs the *reproduction* score). It receives the
verified evidence state and, when enabled later, proposes and runs a *new* experiment through
the same primitives + evidence gate. This spec only lands the **seam and its no-op default** so
the North-Star ideation layer is an extension, not a rewrite.

### 3.6 Downstream contract (what both paths MUST produce)

Enumerated so the parity test is concrete:

- `result_obj`: `RLMChatCompletion`-shaped — `verdict` ∈ {reproduced, partial, failed},
  `reproduction_summary`, `baseline_metrics` (dict), `rubric` {overall_score, target_score,
  meets_target}; or `None` only in the legacy no-evidence case.
- `iterations` (int) — non-zero, from the `RunRecorder` counter (not 0).
- On-disk: `experiment_runs.jsonl`, `cost_ledger.jsonl`, `code/`, `metrics.json`,
  `final_report.{json,md}`, and (flagged) `evidence_bundle.json`.
- `ctx.latest_rubric_score/target/iteration` populated whenever verify ran.
- SSE: `rubric_score` events per verify + `lifecycle_drive_step` per stage/climb.

## 4. Decisions

- **D1 — shared `RunRecorder`, lifecycle-first then binding-refactor** (vs driver-local
  telemetry). Cleaner end state, sets up completion retirement; sequenced to protect the
  working completion path. *Recommended.*
- **D2 — iteration = per-driver-step counter.** Simple, monotonic, non-zero, close enough to
  "root turns" for finalize/observability. *Recommended.*
- **D3 — completion stays behind the flag as a reversible fallback until parity is proven;
  retirement is a separate later spec.** Honors the repo's default-flip discipline. *Locked by
  the user ("fix then run experiments").*
- **D4 — ideation is a seam + no-op default now, not a feature.** *Recommended.*

## 5. Testing strategy

Adds the missing OFF/ON pair plus seam coverage, all socket-hermetic:

- **OFF byte-identical** — with `OPENRESEARCH_LIFECYCLE_PRIMARY` unset, a fixture run is
  byte-identical to today's completion output (the existing 168 tests must stay green).
- **ON end-to-end** — a fixture paper driven through lifecycle-primary emits `rubric_score`
  events, sets `ctx.latest_rubric_*`, stamps non-zero `iterations`, writes the same on-disk
  artifacts, and produces a finalize `result_obj` of the right verdict.
- **Parity** — the lifecycle event stream + `ctx` state matches a golden capture from a
  completion run on the same fixture (the `RunRecorder` contract test).
- **Resume** — a run crashed mid-`run_experiment` resumes at `need_experiment` from the
  checkpoint, not from scratch.
- **Climb observability** — each climb iteration emits a hypothesis-named event.
- **Campaign** — the attempt loop reads a lifecycle-synthesized result correctly.
- **Ideation seam** — default-OFF is a no-op; the hook is reachable when the reserved flag is
  set (no behavior yet).

## 6. Rollout / validation

1. Land the four blocker fixes + observability/evidence/campaign gaps behind the existing
   flag (still default-OFF), all tests green.
2. **Prove parity with ≥3 paired A/B papers on GCP/Azure** (completion vs lifecycle-primary
   on the same paper/seed), first target **Cutout (arXiv 1708.04552)** — it reuses the
   already-proven WRN-28-10 + CIFAR pipeline, so a score delta isolates the *harness*, not the
   implementation. Gate on the repo's **grader-σ** rule (delta within grader noise = parity).
3. **Flip the default** to lifecycle-primary once parity holds; completion remains reachable
   via `OPENRESEARCH_LIFECYCLE_PRIMARY=0` for one release cycle.
4. **Retirement spec** (separate) deletes completion + forced-iteration policy +
   degenerate-loop detector + RLM logger once the fallback is unused.

## 7. Risks / open questions

- **Does `binding.wrap_primitive` already emit `rubric_score` under lifecycle-primary?** The
  recon was uncertain (the emit may fire with `iteration=0`). Resolve by reading the emit
  keying at impl time; the `RunRecorder` makes the outcome deterministic regardless.
- **Binding-refactor blast radius.** Mitigated by lifecycle-first sequencing + the
  byte-identical completion test + the Approach-B fallback.
- **Iteration-count semantics** for a resumed run (continue vs reset) — default: continue from
  the checkpointed counter; assert in the resume test.
- **Cost of A/B validation** — 3 paired GCP runs bill A100 time; bounded by Spec A's mid-cell
  GPU-$ cap + `--max-run-gpu-usd`.
