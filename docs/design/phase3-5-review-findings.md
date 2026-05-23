# RLM Pivot Phase 3–5 — Independent Review Findings

> Reviewer: a fresh review pass over everything merged onto `main` since the
> Phase 2 review (`b14ea89..HEAD`, ~60 commits / ~24K lines). Scope: the Phase 3
> RLM orchestrator (`run.py`, `system_prompt.py`, `sse_bridge.py`,
> `checkpoint.py`), the new `models.py` / `report.py` / `rubric_gen.py` /
> `stub_primitives.py` / `leaf_scorer.py`, the `primitives.py` rewrite, and the
> HTML > PDF > OCR ingestion cascade. Claims verified from primary source.
> Test baseline: `pytest tests/` → **1163 passed, 2 failed, 4 skipped**; the 2
> failures (`test_issue17_runtime`, `test_issue26_experiment_runner`) are the
> same pre-existing local-process-backend failures, unrelated to this delta.

## Verdict

**Do not ship as "the pivot is done" — ship as honest in-progress.**

The orchestrator is real, careful work: the corpus sanitizer (`sse_bridge.py`)
is a genuine single-chokepoint design, the watchdog/timeout layering is sound,
the Phase 2 primitive fixes survived the merge, and the suite is green. But the
**deliverable the kickoff brief defines — "a real PaperBench rubric score on
disk" — is structurally unreachable in the current code**, and the one module
that produces the "authoritative" score does so by overwriting the honest
in-loop score with an uncapped one. Those are the two Critical findings; both
are honesty-rule violations, which the project itself names as reason #1. The
code runs and completes runs — it just cannot yet produce a *true* score, and
in one path it produces a *false* one. Fix the two Criticals before any run is
presented as a real reproduction.

---

## Critical

### C1 — `run_experiment` still hard-returns `metrics={}`; every rubric score is capped at 0.35. The pivot's deliverable is unreachable.
`backend/agents/rlm/primitives.py:472` — `_execute_in_sandbox` returns
`"metrics": {}` with the comment *"real metric extraction from artifacts is
Phase 5 (#62)"*. This is **review finding I7, carried forward and still not
done** — and the Phase 4-6 kickoff brief (`phase4-6-execution-prompt.md` §7.2)
names it explicitly: *"Wire real metric extraction — the I7 unblock. NAMED,
FIRST-CLASS, do not defer."*

Trace the consequence:
- `verify_against_rubric` (`primitives.py:600`):
  `degraded = (not results.get("success")) or (not results.get("metrics"))`.
  `not {}` is always `True` → **every run is `degraded`** → every area score is
  `min(score, 0.35)` (`:611`).
- The root prompt (`run.py:82-85`) correctly forbids inventing metrics: *"Every
  metric in your final report must come from a real `run_experiment` result."*
  Since `run_experiment` can never return one, the only honest report is a
  metric-less `partial` verdict — exactly what happens.
- Net: **no run can score above 0.35**, and the verify→improve loop has no real
  signal to optimize. "Produce `final_report.md` with a real rubric score" is
  not achievable with this code.

*Why it matters:* it is THE deliverable of the pivot (brief §1, kickoff §10).
Shipping the orchestrator without it means the system completes runs that all
look the same and all score ≤0.35.

*Fix:* port `experiment_runner.run_with_runtime`'s metric extraction
(`backend/agents/experiment_runner.py:303-321` — `metrics_path = baseline_dir /
"metrics.json"; metrics = json.loads(...)`) into `_execute_in_sandbox`: after
the commands run, read the `metrics.json` the paper's own code wrote in the
container's artifact dir and return it as `metrics`. Pin it with a test. Until
then, the UI and reports must not present any score as a real reproduction
score (see also the honesty findings below).

### C2 — The leaf scorer overwrites the honest in-loop rubric block with an uncapped, partly-fabricated one.
`backend/evals/paperbench/leaf_scorer.py` — `amend_final_report` (`:291`,
`:304`) **overwrites the entire `rubric` block** in `final_report.json`. It is
invoked only by `scripts/score_run.py:103` — a *manual post-run step*, not the
orchestrator — so a plain `run.py` run keeps its honest in-loop rubric. But
`score_run.py` is the documented path to "the authoritative PaperBench score"
and is what a Phase 6 demo / a reviewer comparing against PaperBench baselines
would run; when it runs, three defects compound:

1. **C2a — reads keys RLM reports never contain** (`leaf_scorer.py:96-99`).
   `_gather_evidence` reads `report["metrics"]` and `report["paper_title"]`,
   but `RLMFinalReport` (`report.py:39-66`) has `baseline_metrics` and `paper`
   (a dict) — no `metrics`, no `paper_title`. So for every RLM run the grader
   scores against evidence with **zero metrics and no paper identity**.
2. **C2b — no degraded 0.35 cap.** `verify_against_rubric` caps degraded runs
   at 0.35; `score_reproduction` has no equivalent. Because it overwrites the
   in-loop block, a metric-less run that produced nothing can be stamped with
   an uncapped `overall_score` — the honest 0.35 ceiling is silently discarded.
3. **C2c — `meets_target` is hardcoded `False`** (`leaf_scorer.py:309`). The
   function gets no target; a legitimate high score still renders `✘ below
   target`. The score block is fabricated, not computed — in both directions.

*Why it matters:* the leaf scorer is the score *of record*, written to the
artifact a reviewer reads. C2b lets it inflate; C2c lets it deflate. Either is
the exact honesty failure `verify_against_rubric`'s backstop exists to prevent.
The tests miss all three because `test_leaf_scorer.py` never feeds a real
`RLMFinalReport`-shaped `final_report.json` through `score_reproduction`.

*Fix:* read `baseline_metrics`/`paper` (not `metrics`/`paper_title`); apply the
degraded cap by detecting a metric-less run; accept a `target_score` and
compute `meets_target` (or write `None`, never a fabricated `False`). Add a
test that round-trips a `write_final_report_rlm` output through the scorer.

---

## Important

### I1 — `ctx.deadline_utc` is never set; the per-primitive deadline layer is dead code.
`run.py`'s docstring (`:14-17`) claims time is bounded *three* ways: `rlm`'s
`max_timeout`, **"#59's per-primitive deadlines (the real bound on a hung
primitive)"**, and the process watchdog. But `RunContext.deadline_utc`
(`context.py:45`) defaults to `None`, and `run_pipeline_rlm` constructs
`RunContext(...)` (`run.py:520-533`) **without passing `deadline_utc`** — and
nothing assigns it afterward. So `ctx.remaining_s()` always returns `None`, and
`_timeout_for(ctx, cap_s)` (`primitives.py:38-42`) always returns the static
`cap_s`. The per-primitive deadline never tightens to the run budget: a
`run_experiment` will wait its full `7200 s` static cap even when the run has
60 s of wall-clock left. Three bounds are really two.

*Why it matters:* this is a cross-module wiring gap — `primitives.py` and
`context.py` implement M-DEADLINE correctly and `run.py` documents it as
load-bearing, but the one line that arms it is missing. Each module is
internally consistent; the integration is broken — exactly the class of bug a
per-module review misses.

*Fix:* in `run_pipeline_rlm`, after `wall_clock_s` is resolved, pass
`deadline_utc=datetime.now(timezone.utc) + timedelta(seconds=wall_clock_s)` into
the `RunContext(...)` constructor. Add a test asserting `ctx.remaining_s()` is
not `None` for a run with a budget.

### I2 — `sanitize_iteration` redacts stdout/stderr prefixes but NOT the root model's `response` — up to 4000 chars of corpus can leak per iteration.
`sse_bridge.py:139-141` — `response` is truncated to `_RESPONSE_MAX_CHARS`
(4000) but is **not** passed through `redact_corpus`, while `_stream_metadata`
(`:165-166`) *does* redact stdout/stderr prefixes with the same sentinels. The
root model reads paper slices via REPL code (`print(context['paper_text'][:N])`)
and can quote what it saw in its next natural-language `response`; that
`response` goes verbatim into every `repl_iteration` event. `_finalize`
redacts `reproduction_summary` (`run.py:687-690`) — so the inconsistency is
visible: two of three egress points are redacted, the highest-volume one is not.

*Fix:* `response = redact_corpus(response[:_RESPONSE_MAX_CHARS], _sentinels)` in
`sanitize_iteration`.

### I3 — `propose_improvements` is not fail-soft on `_extract_json`, unlike its sibling LLM primitives.
`primitives.py:663-664` — `raw = ctx.llm_client.complete(...)` then
`_extract_json(raw)` with **no try/except**. `plan_reproduction` (`:340-348`)
and `verify_against_rubric` (`:587-592`) both wrap the identical
`complete`+`_extract_json` pair in a fail-soft `except` returning an error dict;
`propose_improvements` does not, despite a docstring claiming "malformed items
are dropped fail-soft". A truncated or JSON-less LLM response raises
`ValueError` out of the primitive. The `wrap_primitive` wrapper catches and
re-raises it, so the REPL absorbs it as a code-block error rather than crashing
the run — but the behavior diverges from the two siblings and the docstring.

*Fix:* wrap the `complete`/`_extract_json` call in the same fail-soft `except`
the other two LLM primitives use.

### I4 — The OCR-skip heuristic can strand the ingestion cascade on garbage text.
`backend/services/ingestion/parser/resolving_parser.py:141-146` — OCR is
skipped when HTML or PDF produced ≥`_MIN_USEFUL_CHARS` (200) characters. But a
200–999-char result of figure-noise soup scores `0.0` from
`score_text_quality` (the `<1000`-char rule) yet still has `len ≥ 200`, so OCR
— the tier that exists to rescue exactly this case — never runs, and `_choose`'s
`non_empty` fallback then ships the garbage. The length gate is the wrong
signal: gate OCR-skip on *quality* (`html_score`/`pdf_score` above a low
floor), not raw length. No test covers this branch.

### I5 — The HTML fetch is unbounded — memory-exhaustion DoS on attacker-controlled arXiv HTML.
`backend/services/ingestion/intake/fetchers/arxiv.py:78` — the new HTML fetch
does a single unbounded `response.read()`. `download_pdf` enforces a 100 MB cap
via chunked reads; the HTML path has a *minimum* size check but **no maximum**.
A malicious or buggy arXiv-HTML endpoint returning a multi-GB body OOM-kills
ingestion. `html_parser.py` then runs `BeautifulSoup` + a recursive `_walk`
(`:175`) on it with no size or depth bound — a deeply-nested document raises
`RecursionError`, which is not a `ParseError` and escapes the cascade's
`except ParseError`. *Fix:* add `_HTML_MAX_BYTES` chunked-read cap mirroring
`download_pdf`; cap input size before `BeautifulSoup`; convert `_walk` to an
explicit stack or re-raise non-`ParseError` as `ParseError`.

### I6 — `parsed_full_text.txt` is not produced on a failed parse; the RLM context silently degrades — reintroducing the bug commit `1b69fe7` set out to kill.
`backend/services/ingestion/parser/service.py:170-174` writes the
`parsed_full_text.txt` blob only after a *successful* parse; `cli.py:113-121`
reads it for RLM mode and, on missing/empty content, silently falls through to
the lossy workspace variable. A failed parse — or a re-run into a directory
holding a *stale* blob from a previous paper — feeds wrong/garbage context to
the RLM with no error surfaced. *Fix:* on parse failure delete any stale blob
or write a sentinel; log a WARNING when the blob is missing/empty in RLM mode.

### I7 — `stub_primitives.py` return shapes do not match the real primitives, contradicting its own docstring.
`stub_primitives.py:8` claims each stub *"returns deterministically-shaped data
matching the §5 return column."* Verified false for every structured primitive:
`_propose_improvements` returns `{tag, description, target_rubric_area,
estimated_uplift}` vs the real `ImprovementHypothesis` `{path_id, hypothesis,
rationale, expected_outcome, ...}` — **zero key overlap**; `_detect_environment`
omits `dockerfile` (which the real `build_environment` requires);
`_extract_hyperparameters` uses `epochs` vs the real `epochs_or_steps`; etc.
`_resolve_custom_tools` is all-or-nothing so this does not crash a single run,
but a stub run trains the root's REPL code against keys the real chain never
produces, and the docstring is a false guarantee. *Fix:* build each stub return
from the real schema (`Schema(...).model_dump()` with placeholder values), or
correct the docstring to state the stubs are loop-exercising only. Add a test
asserting each stub's key set is a subset of the real schema's fields.

### I8 — A degraded (stub) run is not honestly observable in its artifacts.
A stub run returns `ok=True`, `success=True`, `overall_score=0.5`, etc. The
only signal it was degraded is one `logger.info` line (`run.py:537`). The
persisted `final_report.json` / `demo_status.json` are structurally
indistinguishable from a real reproduction. *Fix:* thread `tools_label` from
`_resolve_custom_tools` into the persisted status as an explicit
`degraded: true` / `primitive_provider: "stub"` field; have `_finalize` refuse
a non-`failed` verdict when the provider is a stub.

### I9 — `checkpoint.py` advertises "resume" but resume is unimplemented and a restart actively crashes.
`checkpoint.py:1` docstring says *"run-state checkpoint / resume"*. Nothing in
`backend/` ever reads `iterations.jsonl` / `RLMRunIteration` back. `__init__`
(`:112`) hardcodes `self._version = 0`; on a process restart with the same
`project_id` the `rlm-run:<project_id>` aggregate is already at version N, so
the first `record()` passes `expected_version=0` against version N →
`ConcurrencyError`, which the docstring says is intentionally uncaught. A
restart doesn't resume — it crashes on the first checkpoint. *Fix:* implement
resume (load the aggregate, set `_version` to the current event count), or
rename the module to "iteration event log" and drop the word "resume".

---

## Minor

- **M1 — `report.py:368`**: `graded` defaults to `leaf_count`, so a rubric dict
  with no `graded` key claims full `N/N` coverage when coverage is unknown.
  Default to `0` or render "coverage unknown".
- **M2 — `leaf_scorer.py:304-310`**: `amend_final_report` drops the in-loop
  `areas` list (per-area justifications/weak_points); the markdown areas table
  then renders empty for every leaf-scored run. Carry `leaf_scores` into the
  rubric block.
- **M3 — `rubric_gen.py:146-154`** (and the identical pattern in
  `leaf_scorer._parse_batch_response`): first-`{`-to-last-`}` slice spans
  unrelated prose braces and burns a retry. Low impact (retries absorb it); a
  fenced-code-block strip is more robust. Note `primitives._extract_json`
  already does this correctly — reuse it.
- **M4 — primitive cost-ledger rows are still zero-usage** (`binding.py:41-49`,
  carried-forward M4). `_build_llm_client` (`run.py:114-130`) documents that the
  `LlmClient` protocol carries no token usage, so primitive-internal LLM cost is
  invisible; only `rlm`'s root/sub usage is counted. Any `cost_usd` shown in the
  UI under-reports. Label it honestly or omit it.
- **M5 — `arxiv.py:80-86`**: `response.info()` is read after the `with` block
  closed the response; works by header caching but is fragile and the
  `try/except: pass` masks it. Move inside the `with`.
- **M6 — `ocr_parser.py:42-43`**: the `version` property shells out to the
  tesseract binary on every access, and it's on the `ParsingCompleted` hot
  path. Cache it.
- **M7 — `checkpoint.py:152-154`**: the `iterations.jsonl` append is
  `flush()` without `os.fsync`, and runs after the event-store append — a crash
  between the two leaves the JSONL trailing by a (possibly torn) line.
  Forensic-only today since nothing reads it back; worth a comment or `fsync`.
- **M8 — ingestion test suites `importorskip` `bs4`/`pytesseract`**
  (`test_ingestion_html_parser.py:12`, `test_ingestion_ocr_parser.py:27`): in an
  environment missing those deps the HTML/OCR tiers are silently dead and the
  suite still reports green — the "best-source" feature becomes a silent no-op.
  Make `beautifulsoup4` a hard dependency and fail (not skip) in CI if missing.

---

## Cross-component data-flow assessment

The chain that *was* sound in Phase 2 is still sound — the `primitives.py`
rewrite preserved every Phase 2 fix (Dockerfile-keyed image tag, hoisted
`SandboxRuntimeError` import, `_cap_logs`, schema-named `plan_reproduction`
keys, truncation-aware `_extract_json`, clamped `k`). New hand-offs:

| Hand-off | Verdict |
|---|---|
| ingestion cascade → `parsed_full_text.txt` → `cli` → `workspace_claim_map` → `run._build_context` → `context_dict` | ⚠️ works on the happy path; **I6** — a failed/stale parse degrades silently. |
| `run._build_context` → `rlm.RLM.completion(context_dict)` (paper offloaded as `context`) | ✅ RLM fidelity invariant 1 holds — only `_context_metadata` (type/length) reaches the system prompt. |
| `RLMIteration` → `sanitize_iteration` → `repl_iteration` event / checkpoint | ⚠️ corpus chokepoint is real and careful, but **I2** — `response` egress is unredacted. |
| `run_experiment` `{success, metrics, logs}` → `verify_against_rubric` | ❌ **C1** — `metrics` is always `{}` → permanent `degraded` → 0.35 cap. |
| in-loop `verify_against_rubric` rubric → `final_report.json` → `leaf_scorer.amend_final_report` | ❌ **C2** — the leaf scorer overwrites the honest block with an uncapped/fabricated one, reading wrong keys. |
| `primitives._timeout_for(ctx, …)` ← `ctx.deadline_utc` ← `run.py` | ❌ **I1** — `run.py` never sets `deadline_utc`; the deadline layer is dead. |
| `implement_baseline` → `commands.json` → `run_experiment` | ✅ both derive `runs_root/project_id/code` independently; the Phase 2 fix holds. Note `implement_baseline` now returns `str | dict` (error dict on timeout) — the root must guard, and `PRIMITIVE_DESCRIPTIONS` still advertises `-> str` only (minor doc gap). |

## `rlm` library contract — re-confirmed

`run.py` constructs `RLM(environment="local", custom_tools=..., custom_sub_tools={},
custom_system_prompt=..., logger=ReproLabRLMLogger, on_subcall_*=...)`. Verified
against installed `rlm` 0.1.1: `environment="local"` is correct and mandatory
(I6 carried-forward — `DockerREPL` drops `custom_tools`); `logger.log(iteration)`
is the real per-iteration hook (`ReproLabRLMLogger.log` overrides it and
deliberately skips `super().log()` to avoid capturing the corpus); the
`system_prompt.py` brace-escaping for `rlm`'s `.format()` templating is correct
(literal braces in JSON/REPL examples are doubled, exactly one
`{custom_tools_section}` placeholder restored, count asserted). No pickle of run
data anywhere — the checkpointer persists only the sanitized JSON projection.

## What the AI pipeline missed

- **The deliverable was deferred, repeatedly, in a comment.** `metrics={}` has
  carried `"Phase 5 (#62)"` since Phase 2. The kickoff brief escalated it to
  "NAMED, FIRST-CLASS, do not defer" — and it is still deferred. Each session
  hardened the *plumbing* around `run_experiment` (timeouts, persistence,
  `asyncio.shield`) while the one line that makes a run *mean* something stayed
  a stub. A pipeline optimizing per-task correctness will polish the reachable
  and leave the load-bearing gap because no single task owned it.
- **"Authoritative" was asserted, not verified end-to-end.** `score_run.py`
  calls the leaf scorer the authoritative score; no test ever fed it a real
  `RLMFinalReport`. The schema the scorer reads (`metrics`, `paper_title`) and
  the schema `report.py` writes (`baseline_metrics`, `paper`) drifted apart
  because they were built by different tasks and only the fixture — written to
  match the scorer — was ever tested.
- **Module-internal consistency masked an integration gap.** `context.py` and
  `primitives.py` implement M-DEADLINE perfectly; `run.py` documents it as one
  of three time bounds; the constructor call that arms it is simply absent. No
  per-module review catches a missing argument that both sides are "ready" for.
- **Honesty was enforced in one place and overwritten in another.** The 0.35
  degraded cap is correct in `verify_against_rubric` — and then
  `amend_final_report` replaces the whole block. Two tasks, two notions of "the
  score", and the later writer wins silently.

## Empirical verification

- `pytest tests/` → 1163 passed, 2 failed (pre-existing, unrelated), 4 skipped.
- `pytest tests/rlm/` green; new suites `test_run.py`, `test_sse_bridge.py`,
  `test_checkpoint.py`, `test_report.py`, `test_rubric_gen.py`,
  `test_leaf_scorer.py`, `test_system_prompt.py`, `test_stub_primitives.py`
  all pass — but C1/C2/I1/I2/I7 sit precisely in the gaps between those
  per-module suites (see above).
- `rlm` 0.1.1 source re-read at `.venv/lib/python3.14/site-packages/rlm/` for
  the orchestrator's API use.
