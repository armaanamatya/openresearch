# OpenResearch / ReproLab — Project-State Audit — 2026-05-22

> Read-only audit of the whole repo: every open PR, every pivot issue, the
> branches, the backend + frontend code, the test suites, and the live UI.
> No feature code was changed and no PR was touched. Where this report and the
> kickoff handoff conflict, the repo wins — and it conflicts a lot: the handoff
> is several days stale.

## Scope, method, and the one thing to know

**Method.** Three parallel read-only investigation agents (git/PRs/issues; backend
code; frontend code), the backend + frontend test suites run for real, the RLM lab
exercised in a browser, Playwright e2e, and a CLI offline smoke. Every number below
has command output behind it.

**Commit state this audit ran against.** Local `main` = `ec8a70c`. `origin/main` =
`2e1ce37` — **local main is one commit behind origin** and has 6 uncommitted
working-tree changes. `2e1ce37` is by `lolout1`, titled *"RLM Phase 5
debug-and-harden + rubric-driven harness design"*, and is not in the local tree.
This report distinguishes "local main" from "origin/main" everywhere it matters.

**The one thing to know.** Two contradictory things are true at once:

1. The RLM pivot is *almost* code-complete — Phases 1-3 merged, Phase 4/6 sitting in
   open PRs — and one real, careful fix (C1) landed on `origin/main`.
2. The same commit that closed Phase 5 (#62) **deleted the honesty backstop** that
   stopped the system inflating reproduction scores, and ships a design doc that
   declares the entire RLM architecture "low-quality" and proposes a successor.

The pivot's stated reason #1 is *honesty* (`rlm-pivot-brief.md` §1). The audit's
single hardest finding is an honesty regression. Start at section A.

---

## Headline findings

- **C1 is genuinely fixed** at `origin/main`; **C2 is silently broken.** `2e1ce37`
  deleted `score = min(score, 0.35)  # honesty backstop` (`primitives.py:432`) and
  replaced it with nothing. The post-run scorer keeps all three C2 defects. The
  commit's self-reported `0.366` reproduction score sits a hair above the cap that
  commit just removed. (§A1)
- **A second pivot is already designed.** `2e1ce37` adds `rdr` ("rubric-driven
  reproduction harness") design docs that call the just-built RLM orchestrator
  low-quality and propose replacing its control loop. Zero `rdr` code exists. The
  open PRs (#70/#71/#72) are finishing an architecture the architect has moved off.
  (§G)
- **The 14-stage pipeline the pivot exists to delete is still the default.**
  `--mode sdk` (the CLI default) and `--mode offline` both route through
  `orchestrator.py`'s `PipelineStage` machine. Only `--mode rlm` reaches the RLM
  orchestrator. The most recent run on disk is an old-pipeline run that **failed**. (§F)
- **Two branches do the same cutover twice.** `feat/rlm-lab-go-live` (PR #71) and
  `feat/rlm-phase6-cleanup` (PR #72) both delete the 14-stage UI on divergent
  history, 97 overlapping files. Neither branch is shippable alone — each has
  what the other is missing. (§B, matrix below)
- **Phase 5 / #62 is closed but unverifiable here.** No RLM run artifacts and no
  `final_report.*` exist in this clone; `runs/` is gitignored; the claimed runs
  were produced on another contributor's machine. (§D)
- **Tests are healthy where they run.** Backend 1163 passed / 2 pre-existing fails.
  Frontend green except **PR #71 fails lint** (the fix is sitting uncommitted on
  `main`). No CI exists on the repo — every "tests pass" claim in a PR body is an
  unverified local run. (§I)

---

## A. Broken / dishonest — fix before anything ships

### A1. The honesty backstop was deleted, not fixed — `2e1ce37` (CRITICAL)

`docs/design/phase3-5-review-findings.md` (an independent review at local main)
named two Criticals: **C1** (`run_experiment` returns `metrics={}` → every score
capped at 0.35) and **C2** (the leaf scorer overwrites the honest rubric block with
an uncapped, partly-fabricated one). `2e1ce37`'s commit message claims to fix both.
Verified against the diff:

**C1 — genuinely fixed.** `2e1ce37` rewrites `_execute_in_sandbox`
(`backend/agents/rlm/primitives.py`) to read `metrics.json` back from the code mount
(`code_path/metrics.json`, then `code_path/outputs/metrics.json`), and adds an
instruction to the baseline agent (`baseline_implementation.py`) to *write* that
file. The loop is closed end-to-end. Blemishes: the write instruction is duplicated
verbatim in the prompt (copy-paste stutter), and it says "code root" while the
pre-existing orchestrator contract (`_sandbox_contract.py:69`) says
`$OUTPUT_DIR/metrics.json` — the codebase now has two metrics-path contracts. Not
breaks. **C1 verdict: fixed.**

**C2 — NOT fixed. The commit message overclaims.** `2e1ce37` made the in-loop and
post-run scoring paths *consistent* — both now call `leaf_scorer.score_reproduction`.
But it routed both through a scorer that still carries every C2 defect, and it
**deleted the honesty cap**. Verified by grepping the diff:

```
2e1ce37 diff, backend/agents/rlm/primitives.py — every 0.35 line is a REMOVAL:
  -369:  backstop is enforced mechanically: every area score is capped at 0.35 when
  -421:  degraded = (not results.get("success")) or (not results.get("metrics"))
  -431:  if degraded:
  -432:      score = min(score, 0.35)  # honesty backstop
post-2e1ce37 primitives.py: zero occurrences of 0.35 / degraded / min(score)
```

There is **no `+` line adding a cap back** anywhere in the diff. The replacement
`score_reproduction` (`backend/evals/paperbench/leaf_scorer.py`) has no degraded
detection and no cap. The three C2 sub-defects survive into `origin/main`:

- **C2a** — `_gather_evidence` (`leaf_scorer.py:96-99`) still reads `report["metrics"]`
  and `report["paper_title"]`; `RLMFinalReport` (`report.py:39-66`) has
  `baseline_metrics` and `paper`. Every RLM run is graded against evidence with
  **zero metrics and no paper identity**.
- **C2b** — no degraded 0.35 cap in `score_reproduction`. The honest ceiling was
  *removed*, not relocated (grep above).
- **C2c** — `amend_final_report` still writes `"meets_target": False` hardcoded
  (`leaf_scorer.py:309`, verified present at `origin/main`).

**Why this matters.** The 0.35 cap existed so a run that measured nothing could not
be stamped with a real-looking score. With it gone, a lenient LLM grader on
metric-free evidence can land anywhere. `2e1ce37` self-reports
`sequential-neural-score-estimation` at **0.366** — barely above the 0.35 floor it
just deleted. From the diff alone that cannot be told apart from honest small
partial-credit; it is a flag, not a proof — but it is exactly the number a removed
floor produces. A commit whose first section header is "Pipeline correctness and
honesty" deleted a line whose own comment said `# honesty backstop`.

**Fix:** before any score is presented as a real reproduction result —
(1) restore a degraded-run cap inside `score_reproduction` (detect a metric-less
run and clamp); (2) `_gather_evidence` must read `baseline_metrics`/`paper`;
(3) `amend_final_report` must compute `meets_target` or write `null`, never `False`.
Pin all three with a test that round-trips a real `RLMFinalReport` through the scorer
— `test_leaf_scorer.py` never does this today, which is why the drift went unnoticed.

### A2. The exploration tree — the brief's "centerpiece" — opens broken on the Phase 6 branch (HIGH)

Exercised live at `http://127.0.0.1:3001/lab?rlmFixture=1` (the dev server running
from the `feat/rlm-phase6-cleanup` worktree). All six UI regions render, the console
is clean (0 errors), the fixture data is correct — **but the exploration tree opens
un-framed**: the trunk nodes (`Paper`, `Comprehension`) sit off-screen to the left,
and the candidate nodes are clipped under the report rail
(`audit-2026-05-22-screenshots/rlm-lab-fixture-full.png`).

The `NodeDetailPopup` auto-opens for the frontier node — and renders **off-screen**.
Measured directly in the page:

```
[role=dialog] rect: { x: -163, y: 1106, w: 420 }   viewport: 1440 x 900
fullyVisible: false
```

The popup is 1106px down and 163px off the left edge — invisible. A user landing on
the lab sees a clipped tree and an open detail dialog they cannot see.

**Root cause:** `feat/rlm-phase6-cleanup` branched off `main` *independently* of
`feat/rlm-phase4-frontend`, and so never got commit `a279773`
*"fix(rlm-ui): frame the exploration tree on canvas mount"*. phase6-cleanup's
`exploration-canvas.tsx` lacks the ~17-line on-mount framing `useEffect` that
`feat/rlm-lab-go-live` has. PR #71's UI is framed; **PR #72's UI is not.** The
centerpiece of the redesign is broken on load on the branch meant to be the final
cutover.

**Fix:** cherry-pick the framing `useEffect` into phase6-cleanup's
`exploration-canvas.tsx` (or rebase phase6-cleanup onto phase4-frontend — see §B).

### A3. PR #71 fails `npm run lint`; the fix is sitting uncommitted on `main` (MEDIUM)

`npm run lint` on the `feat/rlm-lab-go-live` worktree exits 1:

```
src/components/library/library-filters.tsx
  39:5  error  Calling setState synchronously within an effect can trigger
               cascading renders   react-hooks/set-state-in-effect
✖ 1 problem (1 error, 0 warnings)
```

`main` lints clean, `feat/rlm-phase6-cleanup` lints clean — only go-live fails. The
exact fix for this error is one of the **6 uncommitted changes on local `main`**:
`library-filters.tsx` is rewritten to the "adjust state during render" pattern.
phase6-cleanup committed that fix (`6e3d0f2 ... fix library-filters lint error`);
go-live never got it. So the fix exists twice (uncommitted on main, committed on
phase6) and is missing from the one branch that fails. Commit the main working-tree
fix, or cherry-pick `6e3d0f2`'s `library-filters` hunk onto go-live.

---

## B. In-flight — the open PRs and the branch tangle

### Branch-state matrix — neither cutover branch is shippable alone

Both `feat/rlm-lab-go-live` (#71) and `feat/rlm-phase6-cleanup` (#72) perform a
"delete the 14-stage UI, RLM-only lab" cutover. They are on divergent history
(neither is an ancestor of the other) and each has what the other lacks:

| | `go-live` (PR #71) | `phase6-cleanup` (PR #72) |
|---|---|---|
| Exploration tree framing on load | ✅ has `a279773` | ❌ opens off-screen (§A2) |
| 3 tree SSE events emitted by backend | ❌ not emitted | ✅ `d3c6cc6` emits all 3 |
| `npm run lint` | ❌ 1 error (§A3) | ✅ clean |
| `MAX_DASHBOARD_EVENTS` | 20000 ✅ | **200 ❌** — truncates a real RLM run |
| Non-RLM run fallback notice | ✅ honest notice | ❌ dropped |
| Deletes old 14-stage **backend** | ❌ frontend only | ✅ orchestrator + stage agents |
| `tsc --noEmit` | ✅ | ✅ |
| `vitest` | ✅ 103 passed | ✅ 84 passed |

`phase6-cleanup`'s frontend `use-run.ts` caps the event buffer at **200** — a real
RLM run emits well over that, so a live run would silently lose its tail. go-live's
own code comments flag exactly this and keep 20000. Whoever reconciles these
branches must consciously keep go-live's buffer size and phase6's backend events.

### Open / draft PRs

**No CI is configured on this repo.** `gh pr checks` returns "no checks" for all
four PRs. Every test-count in a PR body is an unverified local run.

| PR | State | Base | Head | +/− / files | Mergeable | Reviews |
|---|---|---|---|---|---|---|
| **#70** Phase 4 — RLM lab frontend | OPEN | `main` | `feat/rlm-phase4-frontend` | +8383 / −50 / 52 | UNKNOWN (uncomputed) | none |
| **#71** Go-Live — RLM-only cutover | OPEN | `feat/rlm-phase4-frontend` | `feat/rlm-lab-go-live` | +239 / −6820 / 55 | clean | none |
| **#72** Phase 6 — cutover (#63) | **DRAFT** | `main` | `feat/rlm-phase6-cleanup` | +10668 / −23484 / 195 | UNKNOWN | none |
| #73 Phase 6 — cleanup (#63) | CLOSED | `feat/rlm-phase4-frontend` | `feat/rlm-phase6-cleanup` | — | — | reviewer notes only |

- **#70** is a coexistence branch: it adds the RLM lab *beside* the 14-stage UI
  (`runMode === "rlm"` switch). It is the shared unblocker — #71 is stacked on it,
  #72 was cut from it. Mergeable state is uncomputed by GitHub (large diff).
- **#71** is stacked on #70 (GitHub auto-retargets to `main` once #70 merges).
- **#72** ⊇ **#71**: the two branches' `diff main..` sets overlap on **97 files** —
  the whole `lab/rlm/` subtree, `lab-shell.tsx`, `use-run.ts`, every deleted
  14-stage component. #72 does everything #71 does, plus the backend cutover, the
  3 SSE events, and the doc rewrites. **Merging both is wasted review and a
  guaranteed 97-file conflict.** #72 should supersede #71.
- **#73 / #72 are the same branch and tip commit**; #73 was created 3 min after
  #72 with the wrong base (`feat/rlm-phase4-frontend` instead of `main`) and the
  author closed it as a duplicate. Clean dedup, no substance lost. #73 carries the
  only review notes in the cluster; #70/#71/#72 have **zero reviews**.

### Local-vs-remote drift to fix

- Local `main` is **1 commit behind** `origin/main` (`2e1ce37`) and
  fast-forwardable. `git pull` it.
- Local `feat/rlm-phase6-cleanup` is **1 commit ahead** of origin
  (`af2a487 test(runtime): use sys.executable ... for the artifact-env exec test`
  is unpushed). PR #72 does not see it. Push it.
- Stale, fully-merged, deletable branches: `origin/merge` (= `main`), `rlm-pivot`,
  `origin/feat/rlm-phase3-orchestrator`.

---

## C. Done — verified

| Phase / Issue | Evidence |
|---|---|
| **Phase 1 (#58)** stage→primitive mapping | `docs/rlm-pivot-mapping.md`; `rlm-pivot` merged |
| **Phase 2 (#59)** stage agents → primitives | `backend/agents/rlm/primitives.py` (9 primitives), `binding.py`, `context.py`; PR #68 merged; `tests/rlm/` green |
| **Phase 3 (#60)** RLM orchestrator | `backend/agents/rlm/{run,system_prompt,sse_bridge,checkpoint,models,report,rubric_gen}.py` all present on main; PR #67 merged |
| **C1** real metric extraction | Fixed at `origin/main` (`2e1ce37`) — see §A1 |
| RLM lab frontend (#61 build) | 16-component `frontend/src/components/lab/rlm/` module, pure-reducer, well-engineered — see §H |

Phase 3 is genuinely merged and real — the kickoff handoff's claim that Phase 3 is
"in progress, owned by a cofounder" is stale. The corpus sanitizer
(`sse_bridge.py`), the watchdog/timeout layering, and the RLM-fidelity offload
(paper never enters the root context) are sound work.

---

## D. Done — claimed but unverifiable here (Phase 5 / #62)

Issue **#62 (Phase 5) is CLOSED**, closed by `2e1ce37` which asserts all four
done-conditions met: DC1 two PaperBench papers with `final_report.md`; DC2 real
scores (`sequential-neural-score-estimation 0.366`, `mechanistic-understanding
0.079`); DC3 candidate lists differ (4 vs 3); DC4 `final_report.{json,md}` +
`repl_state.pickle` + `iterations/` per run dir.

**None of it is verifiable from this clone:**

- `runs/` is gitignored (`.gitignore`); `git ls-files runs/` is empty.
- `find runs/ -name 'final_report.*' / -name 'repl_state.pickle' / -name 'iterations'`
  → **nothing**. There are **zero RLM runs and zero final reports on disk.**
- All 16 `runs/prj_*` dirs are old-pipeline runs (`pipeline_state.json` +
  `reproduction_contract.json`), ingest stubs, or smoke stubs.
- The closing commit's author (`lolout1` / `appradhann@gmail.com`) is a different
  contributor on a different machine; the claimed `0.366`/`0.079` runs would have
  lived only there.

This is not evidence #62 is unmet — it is evidence it **cannot be confirmed here**,
and given §A1 (the cap was deleted), the `0.366` figure should not be trusted as a
real reproduction score until §A1 is fixed and a run is reproduced and inspected.
Verdict: **closed, claimed, unverified — and resting on a scorer with a deleted
honesty backstop.**

The most recent run actually on disk, `prj_ec07993a71c00894` (May 22), is an
**old 14-stage `sdk`-mode run that failed** — `demo_status.json` status `"failed"`,
13 KB `runner.stderr.log`, verdict the unmodified placeholder
`"pending_pipeline_result"`.

---

## E. Missing — specified but not built

- **`root_reasoning` event / Region 5 root reasoning** — issue #61 lists it; the
  frontend has no type, no handler, no rendering for it on any branch.
- **Region 3b live-iteration panel** — #61 spec'd an always-on panel; what exists
  is click-gated: the current iteration's code/stdout is reachable only by clicking
  the frontier node (`node-detail-popup.tsx`). No standing panel.
- **`variable_update` event** — #61 lists it; the implementation collapsed it into
  `repl_iteration.code_blocks[].vars`. Defensible, but it is a spec deviation.
- **The 3 tree events on the merge target** — `candidate_proposed`,
  `candidate_outcome`, `rubric_score` are emitted only on `feat/rlm-phase6-cleanup`
  (`d3c6cc6`). They are **not** emitted on `main`, `origin/main`, or the go-live
  backend. The RLM lab's tree/rubric regions populate from the fixture; against a
  real run on any branch except phase6-cleanup they render empty.
- **Phase 6 (#63) itself** — open; the work is the unmerged PR #72.
- **`rdr`** — designed, zero code (§G).

---

## F. Dead code that is still live

The pivot's entire premise is replacing the 14-stage `PipelineStage` machine. It is
**not dead — it is the default path**:

- `backend/agents/orchestrator.py` (~113 KB, `PipelineStage` enum), `topology.py`
  (hardcoded 5 improvement paths), `pipeline.py` (~27 KB), and 7 stage agents
  (`paper_understanding.py`, `environment_detective.py`, `baseline_implementation.py`,
  `experiment_runner.py`, `improvement.py`, `verification.py`,
  `dependency_verifier.py`) are all present on `main` **and** `origin/main`.
- `pipeline.py:28` imports `PipelineStage`/`PipelineState`; `run_pipeline_sdk`
  builds `ReproLabOrchestrator(...)`; `run_pipeline_offline` drives ~13
  `state.advance_stage(PipelineStage.*)` calls.
- `backend/cli.py` dispatches `--mode offline` → `run_pipeline_offline`,
  `--mode sdk` → `run_pipeline_sdk` (both the old machine), `--mode rlm` →
  `run_pipeline_rlm`. **`--mode sdk` is the CLI default.** The UI subprocess
  (`live_runs.py`) dispatches the same three.
- Only `--mode rlm` reaches `backend/agents/rlm/run.py`.

`2e1ce37` did not touch any of it. PR #72's body says `orchestrator.py` is
"Deleted" — that describes #72's *diff*, not trunk. On `origin/main` the 14-stage
machine is alive and is what an out-of-the-box CLI/UI run executes. Deleting it is
Phase 6's job and is unmerged.

`TODO/FIXME/XXX/HACK` across `backend/`: only 4 hits, all
`# TODO(#17 / runtime)` in `registry.py` — low-risk. Note the genuinely
load-bearing deferral (`metrics={}` "Phase 5 (#62)") was a plain comment, not a
tagged marker — a `TODO` grep never would have caught it.

---

## G. The second pivot — `rdr`

`2e1ce37` commits two design docs —
`docs/superpowers/specs/2026-05-22-rubric-driven-harness-{design,implementation}.md`
— for a **Rubric-Driven Reproduction Harness (`rdr`)**, described in the commit as
*"the pipeline's rubric-driven successor."* The design opens by stating the RLM
harness **"produces low-quality reproductions"** (citing this session's runs:
0.366 partial / 0.079 honest-fail / 0.0 looped-forever) and four RLM failure modes.

`rdr` replaces the free-form RLM root loop with a **deterministic Python controller**
(no LLM in the control flow): decompose the PaperBench `rubric.json` into ordered
work-clusters, assemble a token-budgeted context per cluster, run one Claude (Opus)
agent per cluster, and a capped score→repair loop. New `backend/agents/rdr/`
modules; reuses the 9 primitives, sandbox, leaf scorer, SSE bridge.

- **In code, `rdr` is additive** — the design specifies a new opt-in `--mode rdr`,
  `rlm`/`sdk`/`offline` "stay untouched."
- **In narrative, `rdr` is the successor** — its motivation is that RLM is
  low-quality; its success criterion is beating RLM's ≈0.37 leaf score.
- **Status: design-only. Zero `rdr` code.** `origin/rlm_rubric_orchestration` (the
  named implementation branch) is byte-identical to `origin/main`.

This is the audit's strategic flag: **the open PRs #70/#71/#72 are finishing — and
the Phase 6 PR is *deleting* the fallbacks for — an architecture the architect has
already written off.** `rlm-pivot-brief.md` is still the canonical RLM plan and is
not reconciled with `rdr`. The `rdr` implementation spec also references a third
machine path (`/home/abheekp/openresearch`) and is internally inconsistent about
which branch to commit to — it is not a clean handoff. Whether to merge #72 at all
given `rdr` is a strategic call this audit does not make — but it must be made
before #72 lands, because #72 deletes the 14-stage machine.

---

## H. UI assessment — region by region

Exercised in a browser at `:3001` (the `feat/rlm-phase6-cleanup` dev server).
Screenshots: `docs/design/audit-2026-05-22-screenshots/`.

**`/lab?rlmFixture=1` — the RLM lab, fixture replay** (`rlm-lab-fixture-full.png`,
`rlm-lab-history-expanded.png`). The fixture is a hand-authored 65-event "Attention
is all you need" run; `?rlmFixture=1` → `replayFixture("instant")`. Console: **0
errors, 0 warnings.**

| Region (#61) | State | Notes |
|---|---|---|
| 1 Run header | ✅ good | title, `prj_fixture`, `completed`, iteration 14, $1.84 |
| 2 Rubric score bar | ✅ good | `0.53`, `baseline 0.22 → 0.53`, `+0.31 vs target 0.70`, climb sparkline |
| 3a REPL-state rail | ✅ good | 18/18 vars with type+size, 8 primitives, collapsible |
| 3b Live-iteration panel | ⚠ partial | not an always-on panel — click-gated via node detail (§E) |
| 4 Exploration tree | ❌ **broken on load** | full node set in DOM, but opens un-framed/clipped; node popup off-screen (§A2) |
| 5 Rubric timeline + root reasoning | ⚠ partial | timeline present (sparkline); **root reasoning absent** |
| 6 Primitive-call history | ✅ good | collapsible, 24 calls, reverse-chron, shows the `build_environment` error→retry at iter 5→6 |

**Honesty rule — passes (code + render).** No fabricated values anywhere. Empty
states are honest: `"—"` not `"0.00"` for unscored; the cost tile is *omitted* when
cost is null, not shown as `$0.00`; a `≤ 0.35` score carries a hedged
*"may be a degraded run with no measured metrics"* note; node detail shows stdout
*metadata* (length, traceback flag) never raw stdout. Visual language matches the
brief — flat surfaces, hairline borders, sentence case, no marketing aesthetic, no
"superintelligence" language.

One structural caveat: the lab only looks *alive* because of the fixture. Because
the 3 tree events aren't emitted except on phase6-cleanup's backend (§E), and no
backend was running during the audit, a *real* run renders an empty tree and a
`"—"` rubric — which is honest, but means the populated lab a reviewer sees is
fixture data, correctly labelled *"fixture replay."*

**`/lab` (no fixture)** (`lab-plain-empty.png`) — clean upload view: "Upload PDF /
Drop a paper here… ReproLab will reproduce, verify, and report — independently",
arXiv-link input, Model selector (Sonnet). Honest, no fake run.

**`/library`** (`library-page.png`) — "Runs" table with All/Running/Completed/Failed
filters; honest empty state *"No runs to show…"*. Empty because no backend (`:8000`)
was running to serve `runs/` — expected, not a bug.

**`main`'s `/lab`** is still the **old 14-stage lab** (`progress-strip.tsx`,
`gate-chips.tsx`, `lab-canvas.tsx`, topology canvas) — no `rlm/` directory, zero
`rlm` imports. The RLM lab does not exist on `main`.

---

## I. Test results — every suite, real numbers

### Backend — `pytest tests/` on local `main` (`ec8a70c`)

```
1163 passed, 2 failed, 4 skipped — 6.93s
```

- **2 failed** — `tests/test_issue17_runtime.py::test_local_process_backend_executes_commands_with_artifact_env`
  and `tests/test_issue26_experiment_runner.py::...test_local_process_mode_runs_without_dockerfile`.
  Both are the **documented pre-existing** local-process-backend failures (the
  kickoff names them); a fix is in flight (`af2a487` on phase6-cleanup, §B). Not
  regressions.
- **4 skipped** — `bs4`, `pytesseract`, `chromadb` are not installed in this venv,
  so the HTML/OCR ingestion-tier tests and a semantic-layer test skip silently
  (review finding M8: those deps are in `requirements.txt` but the tests skip
  rather than fail, so the "best-source" cascade is an untested no-op in this env).
- Matches the review-doc baseline exactly. The suite is healthy on `main`.

### Frontend — `tsc` / `lint` / `vitest`, three worktrees

| Worktree | `tsc --noEmit` | `npm run lint` | `vitest run` |
|---|---|---|---|
| `main` (+ uncommitted mods) | ✅ exit 0 | ✅ exit 0 | ✅ 60 passed / 11 files |
| `feat/rlm-lab-go-live` (#71) | ✅ exit 0 | ❌ **exit 1** — `library-filters.tsx:39` (§A3) | ✅ 103 passed / 25 files |
| `feat/rlm-phase6-cleanup` (#72) | ✅ exit 0 | ✅ exit 0 | ✅ 84 passed / 21 files |

`main` lints clean **only because of the uncommitted working-tree fixes** — committed
`main` (`ec8a70c`) would carry the same `library-filters` lint error go-live has.

### Frontend e2e — `playwright test` on `feat/rlm-phase6-cleanup`

```
4 tests — 3 passed, 1 failed (7.1s)
  ✓ Cmd-K command palette + fuzzy search
  ✓ ? shortcut overlay
  ✓ RLM lab renders the exploration tree from the fixture   (rlm-lab.spec.ts)
  ✘ library status filter narrows the table   (lab-smoke-interactive.spec.ts:81)
```

The 1 failure is not a flake (it reproduces deterministically) and not a confirmed
code bug: the test expects a seeded run row (`prj_diffusion_smoke`) in the library
table, which requires the backend (`:8000`) running and serving `runs/`. No backend
ran during the audit → empty table → fail. It is a test-suite robustness gap — an
e2e test with an unstated backend dependency. The **RLM-lab fixture e2e passed.**

### CLI — `python -m backend.cli reproduce 2512.24601 --mode offline`

```
[ingest 1/6 … 6/6]  ✅  project prj_ac41983c..., paper fetched + parsed, workspace built
Starting agent pipeline (offline mode)...
Execution profile: efficient; sandbox: runpod
Sandbox preflight failed: ... REPROLAB_RUNPOD_API_KEY ... is not set.
exit 2
```

Ingest works end-to-end. The pipeline then **aborts at the sandbox preflight** — the
default sandbox resolved to `runpod`, which needs an API key. The documented offline
smoke command is **not runnable out-of-the-box**: it needs `--sandbox local`/`docker`
or a RunPod key. `--mode offline` is documented as "deterministic, no LLM" but still
requires a sandbox. (A full offline run was not re-attempted with `--sandbox local`.)

---

## J. Environment & audit caveats

- **No CI on the repo.** All "X passed" in PR bodies are unverified local runs.
- **Local `main` is 1 behind `origin/main`**; `feat/rlm-phase6-cleanup` is 1 ahead
  of origin (unpushed `af2a487`). §B.
- **6 uncommitted changes on `main`** — `next-env.d.ts` (Next codegen drift) plus
  React-correctness fixes to `lab-shell.tsx`, `progress-strip.tsx`,
  `telemetry-strip.tsx`, `library-filters.tsx`, `lib/pipeline/layout.ts`. Sound, but
  4 of 6 edit files that go-live/phase6 delete outright — that work will vanish at
  the cutover. Only the `library-filters.tsx` fix is durable (and is the §A3 fix).
  Commit it or lose it.
- `bs4`/`pytesseract`/`chromadb` absent from this venv → 4 ingestion/semantic tests
  skip silently.
- The `0.366`/`0.079` Phase 5 scores and the rdr-spec machine path (`abheekp`)
  point at other contributors' machines — not reproducible from this clone.
- Stray untracked dir `.claire/` at repo root (contains only an empty `worktrees/`)
  — looks like a typo of `.claude/`; delete it.
- Audit screenshots saved to `docs/design/audit-2026-05-22-screenshots/` (untracked).
- **Side-effect of the audit:** the CLI smoke (§I) created `runs/prj_ac41983c934a3432/`
  — an ingest-stub run (gitignored). No other writes outside this report and the
  screenshots directory. The audit otherwise made no changes to feature code, no
  pushes, no PR/issue touches.

---

## K. Prioritized next steps

1. **Fix C2 / restore the honesty backstop** (§A1) — *no number can be shown as a
   real reproduction score until this is done; the pivot's stated reason #1 is
   honesty.* Restore the degraded cap in `score_reproduction`, fix the C2a key
   reads and C2c `meets_target`, and add a real-`RLMFinalReport` round-trip test.
2. **Decide RLM vs `rdr` before merging #72** (§G) — *#72 deletes the 14-stage
   fallback; do not cut over to an architecture the `rdr` design calls
   low-quality without an explicit decision.* If RLM stays: reconcile
   `rlm-pivot-brief.md` with `rdr`. If `rdr` wins: #72's deletions may be premature.
3. **Resolve the #71/#72 redundancy** (§B) — *97-file overlap; merging both is
   wasted work and a guaranteed conflict.* Pick the path (merge #70 → rebased-#72;
   close #71 as superseded — or land #71 and re-scope #72 to backend-only). Either
   way, keep go-live's `MAX_DASHBOARD_EVENTS=20000` and phase6's 3 SSE events.
4. **Fix the exploration-tree framing on phase6-cleanup** (§A2) — *the brief's
   centerpiece is broken on load on the Phase 6 branch.* Cherry-pick `a279773`'s
   framing `useEffect`, or rebase phase6-cleanup onto phase4-frontend.
5. **Fix PR #71's lint failure** (§A3) — commit the `library-filters.tsx`
   working-tree fix on `main`, or cherry-pick it onto go-live.
6. **Independently verify a Phase 5 run** (§D) — *#62 is closed on unverifiable
   claims.* After step 1, run one real reproduction, inspect `final_report.md` and
   the rubric block, confirm the score is honest. Until then treat #62 as claimed,
   not done.
7. **Housekeeping** — `git pull` local main; push `af2a487`; tick #64's phase
   checkboxes; fix #63/#64 done-conditions that cite never-existent files
   (`root_loop.py`, `repl_host.py`, `sub_call.py`); delete merged branches
   (`origin/merge`, `rlm-pivot`, `origin/feat/rlm-phase3-orchestrator`) and `.claire/`;
   add a CI workflow so PR test claims are verified, not asserted.
8. **Make the offline smoke runnable** (§I) — default `--sandbox` to `local` for
   `--mode offline`, or document that the offline command needs `--sandbox local`.

---

## Appendix — evidence index

- Review baseline: `docs/design/phase3-5-review-findings.md` (21 sub-findings; 20
  verbatim-present at local `main`, M8 partial).
- `2e1ce37` C1/C2 diff verified via `git show 2e1ce37 -- backend/agents/rlm/primitives.py
  backend/evals/paperbench/leaf_scorer.py`.
- Branch ancestry via `git merge-base --is-ancestor`; PR metadata via
  `gh pr view/checks/diff`.
- Backend `pytest` log: `/tmp/pytest_main_audit.log`. Frontend logs:
  `/tmp/fe_{main,golive,phase6}_{tsc,lint,vitest}.log`. e2e: `/tmp/e2e_phase6.log`.
  CLI: `/tmp/cli_offline.log`.
- UI: `docs/design/audit-2026-05-22-screenshots/{rlm-lab-fixture-full,
  rlm-lab-history-expanded,rlm-lab-node-detail-failed,lab-plain-empty,library-page}.png`.
