# learn.md — bugs we shipped and what we changed so we don't ship them again

This is a runbook of post-mortem entries for production-shaped bugs in the
OpenResearch agent stack. Each entry is short and follows the same shape:

> **Symptom → Root cause → Fix → Lesson → Guardrail (test or pattern)**

Add a new entry to the top of the list. Keep entries surgical: one bug per
section, no broad essays. If a class of bug recurs, escalate it to a section
in **Cross-cutting principles** below.

---

## 2026-05-23 — Lab UI was blank on failed-run navigation because EventSource never opened for terminal states

**Symptom.** User clicks a failed run in the Recent sidebar → navigates to `/lab?projectId=<id>` → blank lab UI. Tree empty, sidebar empty, no error displayed.

**Root cause.** `useRun` opens SSE EventSource only when `status ∈ {"queued","running"}` (line 196 of `use-run.ts`). For failed/completed/stopped runs the source never opens, `dashboardEvents` stays `[]`, the reducer renders initial state. No fallback path loaded historical events from `/api/demo?projectId=X` (which already returns them in `payload.events`).

**Fix.** In the auto-resume effect, for terminal-status runs (`failed` / `completed` / `stopped`), seed `dashboardEvents` from `restored.payload.events.filter(isRlmEvent)` once on mount. SSE is correctly skipped (run is terminal — nothing to stream). Plus: failed status now shows an error banner in `rlm-header.tsx` + a "Rerun" button that POSTs `/runs/<id>/rerun` and navigates to the new project.

**Lesson.** When the live channel is gated (e.g. only opens for active state), the static fallback channel (HTTP snapshot) must seed the same store. Otherwise users see "blank" instead of "here's what happened."

**Guardrail.** Tests TBD (frontend vitest blocked on Node 21 / vitest 4 in this env; ship lands with the integration). Add `useRun` behavioural test once Node is upgraded.

---

## 2026-05-23 — Backend SDK aclose() deadlock made wedged runs indistinguishable from running ones

**Symptom.** Backend uvicorn worker stuck in `do_wai` syscall (99% CPU, 30+ min, all endpoints time out at 5s). UI showed `running` with stale `iteration N` and stale elapsed. Operator couldn't tell "model is thinking" from "SDK is wedged."

**Root cause.** `claude-agent-sdk`'s nested async-generator `aclose()` race (see 2026-05-22 entry); root subprocess alive but no events emitted. Workaround B exists but doesn't catch every codepath.

**Fix.** Two observational levers: (a) `heartbeat(note)` REPL primitive emits `iteration_heartbeat` SSE event; UI `rlm-header` shows amber "no signal Ns" chip when stale >60s. (b) `_stderr_watchdog` asyncio task tails `runner.stderr.log`, detects pattern ≥3× in 30s, atomically writes `degraded: True` + emits `run_warning` SSE event; UI shows red warning chip.

**Lesson.** Observational telemetry preserves model autonomy. Save enforcement (timeouts, kills) for catastrophic-cost cases. Three failure modes (thinking / no-signal / SDK-wedged) had been collapsed to one indistinguishable "running" — now they're visible.

**Guardrail.** `tests/services/events/test_live_runs_watchdog.py` (3 tests: threshold detection, below-threshold no-flag, flag-once idempotency); `tests/rlm/test_heartbeat_primitive.py` (5 tests covering return shape + event payload + counter monotonicity).

---

## 2026-05-23 — RLMFinalReport rejected list-shaped paper_claims, crashing 30-min runs at the last step

**Symptom.** Live arXiv run completed 5 iterations + 5 sub-RLMs + 3 candidates + 1 rubric_score (66 primitive calls, 30+ min wall clock), then died at `build_final_report` with `pydantic.ValidationError: paper_claims Input should be a valid dictionary [type=dict_type, input_value=[{'method': ...}], input_type=list]`.

**Root cause.** `RLMFinalReport.paper_claims: dict` schema; root sometimes returns it as `list[dict]` (`[{"method": "RLM(GPT-5)", "expected_result": "62.0"}, …]`). Schema is too strict; a list of claim objects keyed by method is a perfectly reasonable representation.

**Fix.** `@field_validator("paper_claims", mode="before")` coerces list → dict by keying on first available identity field (`method` / `claim` / `claim_id` / `id` / `name`), fallback to `claim_{i}`. Dict input passes through unchanged.

**Lesson.** Schemas at process seams that accept any long-running computation's output must be liberal about shape and strict about types. Reject only what's structurally meaningless (`None`, primitives where a record is expected) — coerce when user intent is unambiguous.

**Guardrail.** `tests/rlm/test_paper_claims_coercion.py` pins all six shapes (dict passthrough · list with identity key · list with index fallback · identity-field precedence · mixed-list-with-garbage-skip · default empty).

---

## 2026-05-22 — Claude Agent SDK aclose() deadlock wedged rdr cluster 23 for 900s

**Symptom.** A live `--mode rdr` run on `sequential-neural-score-estimation`
reproducibly hung at the 23rd of 27 work-clusters. The controller-level
`_ClusterWatchdog` (`threading.Timer` watching for 900s of no progress) fired
and `os._exit(124)`'d the process. Same wedge observed on
`mechanistic-understanding`. Iteration checkpoints 0–22 saved cleanly; the
SDK call at cluster 23 never returned.

**Root cause.** Two compounding defects in `claude-agent-sdk` v0.1.80, captured
in `docs/superpowers/specs/2026-05-22-sdk-aclose-investigation.md`:

1. **Triple-nested async-generator shutdown race.** `query()` → `process_query()`
   → `_process_query_inner()` are all tracked by asyncio's
   `BaseEventLoop._asyncgen_firstiter_hook`. `asyncio.run()`'s
   `shutdown_asyncgens()` runs `asyncio.gather(*[ag.aclose() ...])` —
   concurrent. Closing `process_query` enters its `finally` and awaits
   `inner.aclose()`, which sets `_process_query_inner.ag_running = True`;
   the concurrent `aclose(_process_query_inner)` from the gather hits
   `ag_running = True` and raises `RuntimeError: aclose(): asynchronous
   generator is already running`. The cleanup chain stalls.
2. **WSL2 futex hang on `transport.close()`.** After SIGKILLing the SDK's
   Node.js subprocess, `transport.close()` does
   `with suppress(Exception): await self._process.wait()` — *no timeout*.
   On WSL2 (`Linux 6.6.114.1-microsoft-standard-WSL2`) SIGCHLD for the killed
   subprocess can be lost/delayed by the WSL2 compat layer; `process.wait()`
   parks indefinitely in `futex_wait_queue`. We confirmed live mech runs
   wedged in `futex_wait_queue`.

**Fix.** Two commits (`4ac89f7` + `33c787d`), Workaround B from the
investigation doc.

  - `_run_sdk_in_thread()` in `backend/agents/rdr/agent.py` wraps every SDK
    call (`collect_agent_text(...)`) in a `concurrent.futures.ThreadPoolExecutor`
    worker that runs the call inside its OWN `asyncio.run(asyncio.wait_for(...))`.
    The worker thread's loop is isolated, so its `shutdown_asyncgens()` race
    cannot block the controller's loop. (Defect 1 contained.)
  - The wrapper uses explicit `try/finally` with `ex.shutdown(wait=False)`
    instead of the `with ThreadPoolExecutor(...) as ex:` context manager —
    because `__exit__` calls `shutdown(wait=True)` by default, which would
    block the controller on a worker that's stuck in `transport.close()`'s
    unbounded `process.wait()`. With `wait=False` the worker thread is
    abandoned (the SDK's `_ACTIVE_CHILDREN` atexit hook SIGTERMs the
    subprocess at process exit) but the controller continues. (Defect 2
    contained.)
  - `concurrent.futures.TimeoutError` is re-raised as builtin `TimeoutError`
    so the existing fail-soft path in `reproduce()` treats it identically
    to the prior `asyncio.wait_for` path.

**Lesson.** When integrating an async SDK with **known cleanup races**, two
invariants are non-negotiable: (a) isolate each SDK call in a worker thread
with its OWN event loop so the SDK's `shutdown_asyncgens` cannot race against
the controller's, and (b) make the wrapper able to abandon a stuck worker
(`shutdown(wait=False)`), because the worker may still hit defects you don't
control. The process-level watchdog is then a defense-in-depth net, not the
primary mitigation. Resist the `with ThreadPoolExecutor(...) as ex:` form
when the worker can be unkillable — `with` blocks on shutdown by default.

**Guardrail.** `tests/rdr/test_agent_thread_isolation.py` — five tests:
`test_thread_isolation_runs_sdk_in_separate_loop`,
`test_thread_isolation_timeout`,
`test_thread_isolation_propagates_exceptions`,
`test_reproduce_uses_thread_isolation`, and especially
`test_thread_isolation_unblocks_on_hung_worker` (the regression guard for
the `with`-block bug — mocks `collect_agent_text` to
`await asyncio.Event().wait()` and asserts the wrapper returns within
~1.5s). Had we kept `with ThreadPoolExecutor(...) as ex:` this test would
hang indefinitely.

---

## 2026-05-22 — pytest module-name collisions across sibling test dirs

**Symptom.** `pytest tests/` (full suite) failed with `import file mismatch:
imported module 'test_run' has this __file__ attribute:
tests/rdr/test_run.py which is not the same as the test file we want to
collect: tests/rlm/test_run.py`. `pytest tests/rdr/ -q` (subdir alone) passed;
only the full collection broke.

**Root cause.** Neither `tests/rdr/` nor `tests/rlm/` has `__init__.py`. Under
pytest's default `prepend` import mode this is fine *until* two sibling test
dirs share a basename — `tests/rdr/test_models.py` and
`tests/rlm/test_models.py` both want to import as the module `test_models`,
and pytest refuses to resolve the clash.

**Fix.** Renamed the rdr files to `test_rdr_models.py` / `test_rdr_run.py`
(via `git mv`) — unique basenames so the basename-import works. Matches the
existing `test_rdr_offline_e2e.py` convention in the same directory.

**Lesson.** When two test directories share the parent (no `__init__.py`),
file basenames must be globally unique across them. Prefix test files in a
new subpackage with the subpackage name (`test_rdr_<thing>.py`) — Python
imports test modules by basename in this layout, so siblings collide.

**Guardrail.** The naming convention is the guardrail. CI's full-suite run
catches a regression immediately (`Interrupted: 2 errors during collection`).

---

## 2026-05-22 — I3 root-prompt change backfired: a "read the paper" emphasis made the root loop on understanding

**Symptom.** Run 3 (GoRL, arXiv) burned all 21 root iterations calling
`understand_section` (×34) and never reached `detect_environment` /
`implement_baseline` / `run_experiment`. Iterations 18–20 were byte-identical
("First, let's understand the paper structure…"). No reproduction; an
unparseable final report.

**Root cause.** I3 added a prominent `_PAPER_GROUNDING` section to the root
system prompt — "read the part that DEFINES it", "read for the complete set of
experiments". On the `qwen3-coder-featherless` root model this anchored the
model on the *understanding* phase; it never transitioned to reproduction. Runs
without I3 (run 2b, both prior arXiv runs) progressed through the full pipeline.

**Fix.** Reverted `_PAPER_GROUNDING` entirely — the known-good prompt is
restored.

**Lesson.** A prompt change to an agentic orchestrator must be validated by a
full end-to-end run before you rely on it. I3's unit tests only asserted the
prompt *contains* the text — they cannot catch that the text *changes
behaviour for the worse*. Prompt-content tests are not prompt-effect tests.
Emphasising one phase of a multi-phase agent loop can starve the others.

**Guardrail.** No automatable test (prompt effect needs a real run). Process
guardrail: a root-prompt change ships only after a green end-to-end run.

---

## 2026-05-22 — run_experiment failed in 6 s: dropped stderr hid a stale-image bug and killed the repair loop

**Symptom.** The mechanistic-understanding RLM run's `run_experiment` failed in
6 seconds with `success=false, logs="", metrics={}` — no error string, no trace.

**Root cause.** Three compounding bugs in `backend/agents/rlm/primitives.py`.
(A) `_execute_in_sandbox` built `logs` from `r.stdout` only — a failed command
writes its traceback to *stderr*, so the failure reason was discarded. (B) the
experiment ran the image `detect_environment` built *before any code existed*;
it under-specified dependencies (no `transformers`), and nothing rebuilt after
the code agent wrote the real Dockerfile. (C) the sandbox ran
`network_disabled`, blocking the HuggingFace download the paper needs.

**Fix.** (A) `_combine_command_output` joins stdout + stderr. (B) `run_experiment`
rebuilds from `ctx.project_dir/Dockerfile` via `build_environment`
(content-addressed, Docker-cached). (C) `_execute_in_sandbox` sets
`network_disabled=False` (scoped — the corpus is never in that container).

**Lesson.** An observability gap is a functionality gap. Bug A did not merely
make failures hard to debug: `run_experiment`'s result *is* the `repair_context`
fed back to the code agent, so `logs=""` left the RLM self-repair loop with
nothing to act on — a silently dead feature. And: build the environment *from*
the code's declared dependencies, never *for* code that does not exist yet — a
spec guessed before its inputs are final must be re-derived once they exist.

**Guardrail.** `tests/rlm/test_run_experiment_env.py` — `_combine_command_output`
keeps stderr; `run_experiment` rebuilds from the project Dockerfile and fails
soft on a bad rebuild.

---

## 2026-05-22 — Re-running a paper conflicts: run state lives in the DB, not just the run dir

**Symptom.** Re-running an arXiv paper (`reproduce <arxiv_id>` — deterministic
project_id) after `rm -rf runs/<id>` failed at iteration 1 with
`ConcurrencyError: expected version 0, found N`; a later attempt produced a
degraded result because stale ingestion events were replayed.

**Root cause.** A run's state is split across two stores: the filesystem run
dir (`runs/<id>/`) and the SQLite event store (`reprolab.db` table
`event_store_events`, aggregates `<id>`, `<id>:parsed`, `<id>:index`,
`<id>:discovery`). `rm -rf` of the run dir clears only the first. The
optimistic-concurrency check (`MAX(aggregate_version)`) then sees the previous
run's events and rejects the re-run's append.

**Fix (operational).** To re-run a paper cleanly, purge BOTH: `rm -rf` the run
dir AND `DELETE FROM event_store_events WHERE aggregate_id LIKE '%<id>%'`.

**Lesson.** When run state is persisted in two stores, "reset a run" must clear
*every* store — the surviving half silently poisons the re-run. A deterministic
project_id makes this unavoidable on every re-run.

**Guardrail.** None yet — flagged in `progress.md` "known gaps". The durable fix
is a `reproduce --fresh` / purge helper that owns the two-store reset so an
operator never has to.

---

## 2026-05-21 — RLM context came from the chunk-reassembled workspace variable, not the parser blob

**Symptom.** Even after the HTML-source resolver produced a clean
`parsed_full_text.txt` (24 KB of good prose), the IOI arXiv `--mode rlm` run
still got garbage as its `context["paper_text"]` — `generate_rubric_tree`
produced empty categories on every attempt.

**Root cause.** cli.py's RLM claim-map builder sourced the corpus from the
workspace `paper_text` *variable*. That variable is reassembled downstream from
indexed chunks (parser sections -> indexer -> chunker -> workspace), and that
reassembly path drops/mangles content for some papers — so it did not match the
parser's clean `parsed_full_text.txt` sitting in the same run dir.

**Fix.** RLM mode now sources `context["paper_text"]` from `parsed_full_text.txt`
— the parser's direct, complete output — in preference to the workspace
variable (kept as a fallback). RLM offloads the whole paper into the REPL
`context`; the parser blob *is* that, and skips the lossy chunk round-trip the
SDK retrieval layer needs but RLM does not.

**Lesson.** When two artifacts both claim to hold "the paper text" — a direct
parser blob and a variable reassembled through an indexing pipeline — they are
not interchangeable. For a whole-document need use the artifact closest to the
source; the reassembled one belongs to the consumer it was reassembled for
(chunk retrieval), not to everyone.

**Guardrail.** `tests/test_cli_claim_map.py::test_rlm_mode_prefers_parsed_full_text_blob`.

---

## 2026-05-21 — Figure-heavy arXiv PDFs parse to figure-label-noise text, defeating downstream LLM use

**Symptom.** LLM agents operating on parsed paper text received token soup
dominated by axis ticks, legend tokens, and figure labels (e.g. `0.0 0.2 0.4
<BOS> IO S1`) — especially for vision/ML papers heavy with plots. The extracted
`full_text` scored low on word-like token ratio and the downstream claim-extraction
LLM saw noise instead of prose.

**Root cause.** `PyMuPdfParser` extracts all text layers from a PDF page in
order, interleaving text from figure axes and legends with paragraph prose.
arXiv also publishes a LaTeXML HTML version where figures are images and the
text is clean prose — but the ingestion pipeline fetched only the PDF.

**Fix.** Quality-gated HTML-preferred cascade: `ArxivFetcher` opportunistically
fetches `https://arxiv.org/html/<id>` (fail-soft — never fails the run) and
writes it as a sibling `raw_paper.html`. `ResolvingParser` tries HTML first (if
the sibling exists), then PDF, then OCR as last resort. Each result is scored
with `score_text_quality` (wordish-token ratio; 0.0 for texts < 1 000 chars).
The first strategy reaching `_USABLE = 0.35` wins; if none does, the
highest-scoring available result is used; if all raise `ParseError`, the
composite error propagates.

**Lesson.** A single-source parser that picks the worst quality source (PDF) by
default, when a higher-quality source exists (HTML), silently degrades every
paper that has significant figure content. Multi-source with explicit quality
scoring is the right abstraction when sources vary in fidelity.

**Guardrail.** `tests/test_ingestion_resolving_parser.py::test_resolving_prefers_html_when_good`
and `::test_resolving_falls_back_to_pdf_when_html_low_quality` (cascade logic locked in).

---

## 2026-05-21 — Leaf-scoring amended final_report.json but not the .md the REST API serves

**Symptom.** After `score_run.py` leaf-scored a completed RLM run,
`final_report.json` carried the authoritative score (0.325, below target) but
`GET /runs/{id}/final-report` — which serves `final_report.md` — still showed
the stale in-loop `verify_against_rubric` score (0.258, "✔ meets target"). The
REST-retrievable report contradicted the JSON and overclaimed.

**Root cause.** `leaf_scorer.amend_final_report` rewrote only `final_report.json`.
The run writes BOTH `final_report.{json,md}` at finish; the markdown is what the
HTTP `/final-report` route serves. Amending one of a two-file pair left the
served artifact stale.

**Fix.** `amend_final_report` now also re-renders `final_report.md` (via
`RLMFinalReport` + `report._render_markdown`) when the report is RLM-shaped —
guarded so a non-RLM report's markdown is never clobbered. `_render_markdown`
gained a rubric-provenance line, so a generated-rubric score is labelled
"self-generated rubric — not PaperBench-official".

**Lesson.** When a fact is persisted in two representations (canonical JSON +
rendered markdown) and a consumer reads one of them, an amend step must update
*every* representation a consumer can reach — otherwise the amend is a half-truth.

**Guardrail.** `tests/rlm/test_leaf_scorer.py::test_amend_final_report_rerenders_markdown`
(markdown tracks the leaf score) and `..._leaves_non_rlm_markdown_untouched`.

---

## 2026-05-21 — RLM runs never wrote demo_status.json, so GET /runs/{id} 404'd

**Symptom.** Caught while wiring the REST-retrievable arXiv path. A completed
RLM run launched from the CLI or `rlm_paperbench.py` left no `demo_status.json`
in its run dir — so `GET /runs/{id}` (which builds `LiveRunState` from that file)
404'd even though `final_report.{json,md}` were on disk. The run's results
existed but the run was not addressable through the HTTP API.

**Root cause.** Status-file writing was owned by the *launcher*, not the *run*.
`live_runs._python_script` wrote `demo_status.json` for backend-spawned runs; the
watchdog wrote a status-only one on timeout; CLI- and script-launched
`run_pipeline_rlm` wrote nothing. A run's status is a property of the run, but no
single place owned writing it.

**Fix.** `run_pipeline_rlm` writes `demo_status.json` itself — `running` at
start, a terminal `completed`/`failed` in `_finalize`, merge-preserving
`startedAt`. The watchdog's status-only write (which omitted `LiveRunState`'s
required `projectId`/`outputDir`/`runMode` and would itself have raised on read)
routes through the same helper. Every RLM run, however launched, is now
REST-addressable.

**Lesson.** A durable status artifact must be written by the component that owns
the lifecycle, not by whichever launcher happened to start it. Split ownership
means some launch paths silently skip it — and the gap stays invisible until a
consumer (here, the HTTP layer) needs the artifact.

**Guardrail.** `tests/rlm/test_run.py::TestWriteDemoStatus` — the written file
round-trips through `live_runs.LiveRunState`, and a terminal write merges onto
(not clobbers) the start write.

---

## 2026-05-21 — arXiv RLM path silently fed the root a 600-char-truncated paper

**Symptom.** Caught by inspection while wiring the arXiv self-generated-rubric
path — before the first arXiv `--mode rlm` run. The shipped arXiv RLM path would
have offloaded only a 600-char stub of the paper into the root model's REPL
`context` variable instead of the full text, leaving the model almost nothing to
reproduce from — and no error would have been raised (a silently degraded run).

**Root cause.** A single `workspace_claim_map` builder was shared by all three
run modes (SDK, offline, RLM). The builder correctly truncated each excerpt to
600 chars for SDK/offline (where excerpts go directly into LLM prompts). RLM is
architecturally different — the paper is offloaded whole into the REPL `context`
variable, never into a prompt, so truncation there defeats the paradigm. Because
both paths called the same inline builder, the RLM path silently received a
600-char stub.

**Fix.** Extracted the builder to a module-level `_build_workspace_claim_map(variables, project_id, mode)`.
For `mode == "rlm"` the function looks up the `paper_text` workspace variable and
returns its full text un-truncated. For any other mode the behavior is byte-identical
to the original truncating path.

**Lesson.** A single claim-map builder cannot serve two fundamentally different
consumption models (prompt injection vs. REPL offload) without a mode branch. Mode-
specific data shaping must be explicit and tested; silent fallback to the "prompt" path
was the error.

**Guardrail.** `tests/test_cli_claim_map.py::test_rlm_mode_paper_text_dict_full_text` —
asserts `mode="rlm"` returns a single un-truncated entry; companion tests assert SDK
mode still truncates at 600 chars.

---

## 2026-05-21 — The RLM root fabricated benchmark metrics it never measured

**Symptom.** An `--mode rlm` run finished `partial` with
`baseline_metrics={"c2st": 0.75, ...}` in `final_report.json` — but no
experiment had been run. The numbers were plausible and entirely fake.

**Root cause.** The root model assembles the final report JSON itself. It
skipped the `run_experiment` primitive, wrote a baseline, then invented metrics
for the report. `build_final_report` passed `parsed["baseline_metrics"]` through
verbatim, and also trusted the root's self-reported `primitive_trace` (which
undercounted — claimed `understand_section=12`, the ledger showed 18). A
self-attested field was treated as ground truth.

**Fix.** `build_final_report` now derives the primitive trace from the cost
ledger — `binding.wrap_primitive` appends a row on every call, so it is
authoritative — and enforces an honesty invariant: `baseline_metrics` are
dropped (and a `reproduced` verdict downgraded to `partial`) when
`run_experiment` is absent from that trace. The root prompt also now mandates
`run_experiment` and forbids estimated metrics.

**Lesson.** Anything an LLM writes into a results artifact is a *claim*, not a
measurement. A field the model self-attests (metrics, its own call trace,
a verdict) must be cross-checked against an out-of-band authoritative record
before a report presents it as fact. Trust the ledger, not the narrator.

**Guardrail.** `TestHonestyGuard` in `tests/rlm/test_report.py` — unbacked
metrics dropped, backed metrics kept, trace sourced from the ledger.

---

## 2026-05-21 — A coercion validator silently dropped already-built model instances

**Symptom.** After adding string→dict coercion to `PaperClaimMap`, eight tests
failed across paper-understanding, environment-detective and report-generator:
`PaperClaimMap.datasets` / `.metrics` came back empty.

**Root cause.** The `_coerce_str_items` before-validator filtered its output with
`if isinstance(item, (str, dict))` — so every already-built `DatasetRequirement`
/ `MetricSpec` instance (exactly what the offline paper-understanding agent
passes) was dropped on the floor. Pydantic then validated an empty list.

**Fix.** The coercion now only transforms bare strings into single-key dicts and
passes every other item through untouched for pydantic itself to validate.

**Lesson.** A `mode="before"` validator that exists to *coerce* one input shape
must be a pure pass-through for every other shape. Filtering inside a coercion
silently discards valid data — the validator's job is to widen what is accepted,
never to narrow it.

**Guardrail.** `tests/test_schemas.py` asserts datasets/metrics accept dicts,
bare strings, and pre-built submodel instances (mixed lists included).

---

## 2026-05-21 — implement_baseline exhausted the Claude OAuth quota on Opus

**Symptom.** An `--mode rlm` run's `implement_baseline` failed with `Claude Code
returned an error result: success` — the code-writing agent wrote no code, and
the run finished `failed` with a 0.0 rubric score.

**Root cause.** Two layers. (1) That string is the claude-agent-sdk's signal for
a Claude Code subscription quota cap — already catalogued in
`resilience/classify.py` as a `QuotaExhausted` phrase. (2) The agent ran on
Opus: `baseline-implementation` is registered with
`default_model_anthropic="claude-opus-4-7"`, and a heavy Opus agent exhausts the
OAuth quota it shares with interactive Claude Code sessions.

**Fix.** Pin the sub-agent to Sonnet. The model is threaded as the explicit
`model_override` — `RunContext.agent_model` → `implement_baseline` →
`run_with_sdk(model=)` → `collect_agent_text(model=)` → `to_runtime_spec`.

**Lesson.** `to_runtime_spec` resolves `model_override or settings_override or
registry_default`. A non-empty registry per-agent model beats any runtime-level
default — so to force a model for one invocation you must pass `model_override`,
not configure the runtime. Pinning a model on the runtime instance is dead code.

**Guardrail.** `test_implement_baseline_passes_agent_model_as_override` asserts
`ctx.agent_model` reaches `run_with_sdk` as `model`.

---

## 2026-05-21 — A missing LLM API key surfaced as a cryptic `TypeError` deep in `rlm`

**Symptom.** An `--mode rlm` run with the `claude` root failed at run start with
`TypeError: AnthropicClient.__init__() missing 1 required positional argument:
'api_key'`, raised ~100 frames deep inside the `rlm` library — nothing pointed at
the real problem: an unset credential.

**Root cause.** `resolve_root_model` injected each backend's API key from the
environment via `_inject_api_key`, which added `api_key` to `backend_kwargs` only
when the env var was truthy — an absent or empty key was silently dropped. The
single fail-fast guard covered only the OpenRouter backend; `anthropic` and
`openai` had none, so a missing key sailed through to `rlm`'s client constructor.

**Fix.** `resolve_root_model` now fails fast for *every* backend: a `_env_var_for`
helper resolves the key's env var (honouring an explicit `RootModel.api_key_env`),
and a loop over the root and sub-call backends raises an actionable `ValueError`
when the key is absent. The `api_key_env` field also decouples the key source from
the backend type — which is what lets the Featherless backend (an `openai` client
type authenticating with `FEATHERLESS_API_KEY`) work.

**Lesson.** A missing credential must fail loudly at the boundary where it is
resolved — never let it travel inward to surface as a type error inside a
dependency. A fail-fast check added for one backend must cover *every* backend,
not just the one that first needed it.

**Guardrail.** `TestMissingApiKeyFailsFast` in `tests/rlm/test_models.py` asserts
`resolve_root_model` raises `ValueError` (naming the missing env var) when a
backend's key is absent — covering the anthropic and openrouter backends.

---

## 2026-05-21 — Git LFS pointers served by raw.githubusercontent.com look like real files

**Symptom.** Fetching `paper.md` from `openai/preparedness` via
`raw.githubusercontent.com/openai/preparedness/main/…/paper.md` returned a
134-byte text file starting with `version https://git-lfs.github.com/spec/v1`.
The paper content was absent; only a pointer stub was stored in the main git
tree.

**Root cause.** PaperBench source files (`paper.md`, `addendum.md`) are tracked
in Git LFS on `openai/preparedness`. `raw.githubusercontent.com` serves only the
pointer object, not the resolved blob. There is no flag or header to make it
transparently dereference LFS.

**Fix.** Use the LFS batch API directly:
`POST https://github.com/openai/preparedness.git/info/lfs/objects/batch` with
the pointer's `oid` and `size`. The response returns an `actions.download.href`
redirect URL. For `openai/preparedness` that URL is under the
`openai/frontier-evals` LFS store — follow the redirect to get the real blob.

**Lesson.** When vendoring from a repo that uses Git LFS, treat `raw.githubusercontent.com`
as unreliable for any file whose tracked size is suspiciously small. Check the
first line: if it starts `version https://git-lfs.github.com/spec/v1`, you have
a pointer, not the file. Always fetch via the LFS batch API for LFS-tracked blobs.

**Guardrail.** The vendoring script (`scripts/vendor_paperbench.py`) checks the
first line of each fetched file and aborts with a clear error if it detects a
pointer stub, so a future re-vendor does not silently store a 134-byte shell.

---

## 2026-05-21 — `rlms` `max_timeout` only fires between iterations — primitives overrun it

**Symptom.** A long `run_experiment` primitive (N experiment commands, each up
to 3600 s) wedged inside `execute_code`. The `rlms.RLM(max_timeout=…)` budget
expired, but the process did not stop — the run hung past its wall-clock budget.

**Root cause.** `rlms`'s `max_timeout` is checked *between* iterations of the
Algorithm-1 loop. A primitive that blocks synchronously inside `execute_code` or
`pool.submit(...).result()` is opaque to the library's timeout check; the next
check never arrives. The process-level watchdog in `run.py` is the only backstop,
and it fires via `os._exit` — no teardown.

**Fix.** Each long primitive (`build_environment`, `implement_baseline`,
`run_experiment`) now routes through `run_with_deadline(coro, ctx, cap_s)`.
`RunContext` carries `deadline_utc: datetime | None` and `remaining_s()`.
`run_with_deadline` wraps the async body in `asyncio.wait_for(min(cap_s, remaining))`
and, on `TimeoutError`, runs the primitive's teardown (sandbox `destroy`) before
returning a fail-soft error dict. This gives the loop a chance to handle the
failure gracefully rather than relying on the brutal process-level watchdog.

**Lesson.** A library-level timeout that only fires between loop iterations is
not a real timeout for any primitive that blocks synchronously inside the loop.
When you offload heavy work to a third-party scheduler (`rlms`, `celery`,
`asyncio.gather`), own the deadline enforcement inside your own code — do not
rely on the scheduler's outer clock.

**Guardrail.** `tests/rlm/test_primitives_deadline.py` drives each long primitive
with an already-expired `RunContext.deadline_utc` and asserts it returns a
fail-soft error dict (no hang, no exception). The test also verifies sandbox
`destroy` is called exactly once so no orphaned containers are left.

---

## 2026-05-21 — Corpus-leak (Algorithm-2) invariant needs redaction at EVERY egress, not only in `sanitize_iteration`

**Symptom.** A corpus-leak audit of the RLM run surface found two escape paths
not covered by `sanitize_iteration`: (1) `sse_bridge.py` prefixed streamed
stdout/stderr lines with the raw primitive output before sanitizing; (2)
`report.py` embedded the full primitive return value (which can carry a
`context` slice) verbatim in the final JSON report. The `sanitize_iteration`
chokepoint was correctly placed for iteration events but was not the only place
where corpus-bearing data reached a durable or streamed surface.

**Root cause.** `sanitize_iteration` strips the `context` variable from
`RLMIteration.locals` before the event is logged, which is correct for
iteration snapshots. But stdout/stderr from inside a primitive are streamed
directly to the SSE bridge before the iteration is complete, and the final
report is built from primitive return values after the loop — neither path
flows through `sanitize_iteration`.

**Fix.** A `redact_corpus(text, sentinels)` helper is now applied at every
egress point. `sentinels` is the set of first-200-char prefixes of each
`context` corpus value, computed once at run start. Applied to: stdout/stderr
prefixes in `sse_bridge.py`; every string field in `report.py`'s final-report
construction.

**Lesson.** A security invariant ("the corpus must never reach a logged surface")
is only as strong as the set of egress points it covers. `sanitize_iteration`
was the right chokepoint for the iteration-event path; it was not the only path.
When you audit an invariant, enumerate **all** surfaces where the sensitive value
could reach — not just the one that motivated the original guard.

**Guardrail.** `tests/rlm/test_corpus_redaction.py` drives a run with a
recognisable sentinel in the corpus and asserts the sentinel is absent from: the
SSE event stream, the SQLite snapshot, and the final JSON report. The test fails
if any of the three egress points is not covered.

---

## 2026-05-21 — A library that `.format()`s the prompt you hand it

**Symptom.** The first real-`rlm` integration run of the new RLM orchestrator
(#60) crashed inside the `rlms` library before the Algorithm-1 loop even started
— an error from deep in `rlm`, on a prompt string our own `build_system_prompt`
had produced and we treated as plain text.

**Root cause.** `rlm`'s `build_rlm_system_prompt` runs
`system_prompt.format(custom_tools_section=…)` on whatever `custom_system_prompt`
you pass `rlm.RLM(...)` (`rlm/utils/prompts.py:156`). The custom prompt is not
plain text to the library — it is a `str.format()` **template**. Our prompt was
full of literal braces (JSON report examples, code snippets like
`json.dumps({"summary": …})`); `.format()` read each `{…}` as a replacement
field and blew up. Worse: the library injects the auto-generated primitive tool
docs at a `{custom_tools_section}` placeholder — our prompt had none, so even
absent the crash the root model would never have been told the primitives exist.

**Fix.** `build_system_prompt` now treats its output as what it is — a
`.format()` template. It assembles the prompt with normal, readable braces, then
escapes every brace (`{` → `{{`, `}` → `}}`) and restores exactly one real
placeholder (`{custom_tools_section}`) at the primitive-docs slot.

**Lesson.** When you hand a string to a third-party library, find out what the
library *does* to it. "It's just a prompt string" was the wrong mental model —
the library's contract was "give me a `.format()` template." A value crossing an
API boundary is governed by the callee's contract, not the caller's assumption.

**Guardrail.** `tests/rlm/test_system_prompt.py::TestFormatTemplate` asserts the
output is a valid template — `.format(custom_tools_section=…)` does not raise and
exposes *exactly* one field. The real-`rlm` integration harness
(`tests/rlm/test_run_integration.py`) drives the real `build_system_prompt`
through `rlm`'s own `.format()`, so a regression also fails end-to-end.

---

## 2026-05-14 — A "successful" docker build can still ship a broken environment

**Symptom.** A docker+sonnet e2e run on the demo PPO paper sailed past Track 4
(env built clean attempt 1), got past the sandbox-mount contract fix (script
wrote correctly to `$OUTPUT_DIR`), and died at the very first `gym.make`:
`ModuleNotFoundError: No module named 'imageio'` from
`gymnasium/envs/mujoco/mujoco_rendering.py`. The Dockerfile pinned
`gymnasium[mujoco]` but `imageio` isn't in gymnasium's `setup.py` for the
mujoco extra — it's imported at first env load. The build had no way to know.

**Root cause.** Track 4 validates that `docker build` *exits 0*. That only
proves "pip install didn't crash" — not "every Python module in this image
actually imports." Lots of pip packages have *transitive runtime imports*
(here: `imageio` for gymnasium-mujoco; in other papers, `cv2` wanting
`libGL.so.1`, torch wanting a specific CUDA runtime, etc.) that pass at
install time and explode at module-load time. Track 4 had no trigger for
runtime-import failures — so they died downstream at `baseline_run`, with
no repair feedback.

**Fix.** Force the runtime-import failure into the build phase by making
the FINAL Dockerfile layer a no-network smoke: `RUN python -c '<imports + a
minimal instantiation of the paper's primary entity>'`. A failure there is
a build failure — and Track 4's build-and-repair loop already knows how to
fix build failures (add the missing dep via env-detective repair mode,
rebuild). Zero new code; just a prompt rule.

**Lesson.** A repair loop's *trigger event* is as load-bearing as its
*repair mechanism*. Track 4 had the repair mechanism (env-detective in
repair mode) but only fired it on `docker build`'s exit code. The
import-time class of failures was invisible to that trigger. The
generalization: whenever you have a recovery mechanism, audit *every*
failure mode it should cover and make sure each has a trigger pointing at
it — not just the one that motivated building the mechanism in the first
place.

**Guardrail.** No new tests — the smoke layer is verified by the e2e run
(if it's missing, the prompt change is in but the agent ignored it; the
existing prompt-format tests catch import + brace regressions). The next
demo-paper run is the live regression check — the imageio failure should
now appear at *build* time, get repaired automatically, and the experiment
should reach `baseline_run` with a working environment.

## 2026-05-14 — The sandbox mount contract lived in env-var names, not in any prompt

**Symptom.** A docker+sonnet e2e run on the demo PPO paper sailed past Track 4's
environment build (clean attempt 1), reached `baseline_run`, and died at the very
first command: `mkdir: cannot create directory '/work/results': Read-only file
system`. Gate 2 halted on `failed_reproduction`. The reproduction *code* was
fine — the failure was that the script tried to write outputs under the project
mount.

**Root cause.** The sandbox runtime enforces a clear mount contract — project
read-only at `/work`, writable artifact volume at `$OUTPUT_DIR` — but this
contract existed only implicitly, in env-var names exposed to the container.
The `baseline-implementation` and `improvement-path` prompts never stated it,
so the agent wrote scripts assuming the CWD was writable. A load-bearing
contract that lived only in the runtime's env-var dictionary was advisory to
the agent, not enforced.

**Fix.** Made the contract a first-class artifact. `backend/agents/prompts/_sandbox_contract.py`
defines a single brace-free `SANDBOX_EXECUTION_CONTRACT` block — the mount
model, the env vars, the required write patterns (every output under
`$OUTPUT_DIR`; cache-hungry tools redirected; metrics.json path pinned). It is
imported and spliced into every agent prompt that emits sandbox-executable code
(`baseline-implementation`, `improvement-path`, `composition`), positioned
right before the `# Output` section at peak attention. Identical across docker,
local, and runpod — same env vars, same model.

**Lesson.** An interface contract between code that *generates* artifacts (an
LLM agent) and code that *executes* them (the runtime) must be stated in the
generator's prompt, not just enforced by the executor. Same lesson as
"a 'hard cap' in a prompt is advisory unless enforced in code" — but in the
other direction: a runtime invariant the agent must respect is advisory
unless stated in the prompt. Put it in one shared module, splice it where it
matters, and the prompts cannot drift from the runtime.

**Guardrail.** `tests/test_track4_environment_build_repair.py` is unaffected;
the contract is verified by a focused import-and-format assertion in
`backend/agents/prompts/__init__.py`'s consumers and by every existing prompt
test that imports the three updated prompts. The next e2e run on demo_paper.pdf
is the live regression check.

## 2026-05-14 — The reproduction Dockerfile was never built until it was too late to fix

**Symptom.** `environment-detective` generated the Dockerfile one-shot at the
`ENVIRONMENT_BUILT` stage, but nothing ran `docker build` until `run_experiment`
at `BASELINE_RUN` — five stages and tens of minutes later. A broken Dockerfile
(missing system lib, a non-existent pin like `ale-py 0.8.1`, base-image
mismatch) burned all that work, then dead-ended the run at Gate 2 with
`blocked_requires_human`. No run had ever reached the Track 3 flow live.

**Root cause.** The pipeline had a *judge* for the environment
(`environment-verifier` at Gate 1) but no *builder*. The first real validation
of the generated artifact happened far downstream from where it was produced,
so the feedback loop that could fix it never existed — and the terminal state
for that failure was a human-required halt, not an autonomous recovery.

**Fix.** Build the Dockerfile at the stage that produces it. A build-only
`build_image()` primitive runs `docker build` at `ENVIRONMENT_BUILT`; on failure
the build error is fed back to `environment-detective` in a repair mode and the
build is retried, hard-capped at `environment_build_max_attempts`. After the cap
the run is **fail-soft** — it proceeds and completes with an honest
partial-reproduction verdict instead of halting for a human.

**Lesson.** Validate a generated artifact at the stage that generates it, not at
the stage that first consumes it — the distance between the two is wasted time
and a feedback loop you don't have. And an autonomous pipeline's terminal state
for a *recoverable* failure should be an honest verdict, not a halt: a bounded
repair loop plus fail-soft beats `blocked_requires_human`.

**Guardrail.** `tests/test_track4_environment_build_repair.py` — `build_image`
returns `(False, …)` for a broken Dockerfile but raises for an infrastructure
failure; `_run_environment_build_loop` is bounded (capped attempts, repair
invoked between them) and fail-soft (cap spent → `environment_build_ok` false,
no raise).

## 2026-05-14 — A "hard cap" that lived only in a prompt was advisory, not enforced

**Symptom.** The rubric-verifier prompt told the model "no executable code →
score ≤ 0.20", "code never ran → ≤ 0.35", etc., and the plan/changelog called
these "honesty hard caps" — but nothing checked them. A model that returned 0.9
for a run that never executed would be accepted verbatim.

**Root cause.** Load-bearing invariants were expressed *only* as natural-language
instructions to an LLM. A capable model usually follows them, but "usually" is
not a guarantee, and the reported score is a metric users act on.

**Fix.** Added a mechanical backstop in `_run_rubric_verifier`: the orchestrator
already knows `experiment_artifacts.success`, so when the reproduction did not
execute it clamps every area score before aggregation — independent of what the
model returned. The prompt still states the caps (so the model cooperates).

**Lesson.** A guarantee a prompt makes is only as strong as the model's
compliance. If an invariant is load-bearing — a safety gate, a reported metric,
a stopping criterion — enforce it in code at the boundary; let the prompt *also*
state it, not *only* state it.

**Guardrail.** `tests/test_rubric_verifier.py::test_run_rubric_verifier_caps_score_when_run_did_not_succeed`
feeds a high model score for a failed run and asserts it is capped.

## 2026-05-14 — A self-improvement loop compared scores from regenerated rubrics

**Symptom.** The rubric verifier ran at Gate 2 and Gate 3, and the re-iteration
loop stopped when `improved_verification.overall_score` met the target — but the
baseline and improved verifications were not actually comparable.

**Root cause.** Each checkpoint created a fresh `GeneratedRubricSource()` and
passed `rubric: null`, so the verifier LLM generated *new* areas and weights
every time. `baseline_verification` and `improved_verification` were scored
against different rubrics; their delta — and the loop's stop criterion —
measured rubric churn, not reproduction progress.

**Fix.** Resolve the canonical rubric once per run (a vendored bundle's rubric,
or LLM-generated on the first call), persist it in `PipelineState.rubric_spec`,
and pass it back at every later checkpoint. Weights come from the persisted
spec; the LLM supplies per-area scores only.

**Lesson.** A metric you compare across time must be *defined* once. If the
judge is free to redefine the rubric at each measurement, the series of scores
is not a series — it is noise wearing a trend's clothes.

**Guardrail.** `tests/test_rubric_verifier.py` asserts the first verifier call
persists `rubric_spec` and a later call reuses its weights verbatim — a model
that returns different weights is overridden, not trusted.

---

## 2026-05-14 — A `backend.agents` module eager-importing `backend.evals` was a circular import

**Symptom.** Adding `from backend.agents.rubric_source import GeneratedRubricSource`
to `backend/agents/orchestrator.py` broke *every* import of the orchestrator:
`ImportError: cannot import name 'PipelineState' from partially initialized
module 'backend.agents.orchestrator'`.

**Root cause.** `rubric_source.py` had a module-level
`from backend.evals.paperbench.bundle import ...`. Importing any
`backend.evals.*` submodule runs `backend/evals/__init__.py`, which eagerly
imports `backend.evals.runner` → which imports `backend.agents.orchestrator`.
While `orchestrator` was *mid-import* (at the new `rubric_source` line, before
`PipelineState` was defined), `runner` tried to import `PipelineState` from it.
Phase A didn't hit this because nothing in the main import graph pulled in
`rubric_source` — only the tests did, and by then `orchestrator` was complete.

**Fix.** Made `rubric_source.py` import the `bundle` loader **lazily**, inside
the two functions that actually load a bundle. The cycle is broken because by
call time `orchestrator` is fully initialized.

**Lesson.** A package `__init__.py` that eagerly imports heavy submodules turns
*every* `from that_package.x import y` into a transitive import of the whole
package graph. A leaf-looking module (`bundle.py` only imports stdlib) is not
leaf if its package `__init__` is not.

**Guardrail.** A `backend.agents.*` module that needs `backend.evals.*` (or any
package whose `__init__` reaches back into `backend.agents`) imports it lazily
inside the function that needs it — never at module scope.

---

## 2026-05-14 — A timed-out enrichment frame silently blanked the live graph

**Symptom.** Mid-run, the workflow graph's per-path improvement nodes
(`opt/bb/aug/hor/div`) intermittently dropped back to "upcoming" for a tick,
then recovered on the next frame.

**Root cause.** Both `/api/demo` GET (750 ms) and `/api/demo/events` SSE
(250 ms) cap payload enrichment and, on timeout, forward the *un-enriched*
backend run state — which carries no `payload`. `stateMapForRun` reads
`run.payload.pathStates`; with `payload` undefined every path node fell
through to "upcoming". The UI overwrote good state with a strictly poorer
frame.

**Fix.** `coalesceRunState` merges an incoming `run_state` frame onto the
current one, carrying the last `payload`/`telemetry`/`log` forward when the
new frame lacks them. Both the SSE handler and the poll fallback route
through it; it warns in dev when it has to coalesce.

**Lesson.** A frame that arrives with *less* information than the one it
replaces must not be applied verbatim — partial frames are an expected
steady-state condition here (enrichment timeouts), not an error.

**Guardrail.** State updates fed from a stream/poll should be **monotonic in
information**: merge-don't-replace when the transport can legitimately
deliver a degraded frame. (`stateMapForRun` already encoded this for stage
progress; `coalesceRunState` extends the same rule to the payload.)

## 2026-05-14 — A stage-ordering test froze the pipeline at 15 stages after it became 14

**Symptom.** `tests/test_issue22_orchestrator.py::test_pipeline_stages_are_ordered`
failed on `claw_demo` (and on its parent commit): the test's `expected_order` placed
`composition_tested` between `improvements_run` and `gate_3_passed`; the real
`PipelineStage` enum had no such stage.

**Root cause.** `composition_tested` was removed from `backend/agents/orchestrator.py`
when the pipeline became 14 stages, but the ordering test still hard-coded the old
15-stage list — it re-typed the enum as a literal and then drifted from it.

**Fix.** Dropped `"composition_tested"` from the test's `expected_order`.

**Lesson.** A test that re-types an enum as a literal sequence is a second source of
truth; it goes stale silently the moment the enum legitimately changes.

**Guardrail.** Derive the expectation from the enum (`[s.value for s in PipelineStage]`)
and assert the *properties* that matter (no gaps, each gate after its prerequisites,
`complete` last) instead of re-typing the sequence.

## 2026-05-10 — Pipeline SIGINT dumped a 50-line stack trace and left status="running"

**Symptom.** Killing the `python -m backend.cli reproduce` subprocess (Ctrl-C
or backend restart) produced a noisy traceback in `runner.stderr.log`:

```
asyncio.exceptions.CancelledError
…
File "/home/abheekp/openresearch/backend/cli.py", line 485, in cmd_reproduce
    state = asyncio.run(run_pipeline_sdk(
KeyboardInterrupt
```

The dashboard meanwhile showed `status="running"` until the user hit `/lab`
again, at which point `live_runs._load_run` detected the dead PID via
`_pid_exists` and rewrote status to `failed` with whatever string the log
heuristic happened to extract — usually misleading.

**Root cause.** The application code had **zero** explicit handlers for
`KeyboardInterrupt` or `asyncio.CancelledError`:

1. `cli.py:485` wrapped `asyncio.run(run_pipeline_sdk(...))` with only
   `except Exception` (catches `BudgetExhausted`). `BaseException`
   subclasses fell through, which is correct Python convention but meant
   we never got a chance to write a clean status before exiting.
2. `orchestrator.py:1441` step loop's `except Exception` likewise didn't
   catch `CancelledError`. The "X FAILED:" line never printed for
   cancellation either, so the log just stopped mid-stage with no
   actionable signal.
3. `live_runs._write_status` wrote `demo_status.json` non-atomically, so
   a crash during a status write could leave a half-written JSON that
   `_read_status` then failed to parse. Compounding the original
   interrupt with a corruption bug.

**Fix.**

- `cli.py` catches `(KeyboardInterrupt, asyncio.CancelledError)` around
  `asyncio.run(run_pipeline_sdk(...))`, prints a single readable line,
  calls `_mark_demo_status_stopped()` to flip the status to `stopped`
  with a descriptive `error` field, and exits 130 (SIGINT convention).
  No more stack-trace dumps.
- `orchestrator.py:1431` step loop now catches cancellation **before**
  the generic `except Exception`, prints `|| STOPPED at <stage>`, calls
  `state.save_checkpoint(self.runs_root)` so a future
  `reproduce --resume` picks up from the last completed stage, and
  re-raises so the CLI's outer handler runs.
- `cli._atomic_write_json` (and the equivalent in
  `live_runs._write_status`) writes via tempfile + `os.replace` so
  `demo_status.json` is never half-written. Readers always see either
  the previous valid JSON or the new one.

**Lesson.** **`asyncio.CancelledError` is a `BaseException`, not an
`Exception` — your `except Exception` does NOT catch it.** Long-running
async pipelines need an explicit `(asyncio.CancelledError, KeyboardInterrupt)`
handler at every layer that owns persistent state, before the generic
`except Exception` clause. The handler should: (1) log a clean message,
(2) flush partial state to disk so resume works, (3) re-raise so callers
above can do their own cleanup. Status files that record run lifecycle
should be written atomically (`tempfile.write_text` + `os.replace`) so a
crash during the write doesn't corrupt the file the dashboard is about
to read.

**Open edge cases (documented, not yet fixed):**
- Concurrent runs on the same `project_id` will race on
  `demo_status.json`, `pipeline_state.json`, and `runs/{project_id}/*`.
  Atomic writes prevent corruption but don't prevent overwrite.
- SIGKILL bypasses the CLI's interrupt handler entirely — the pipeline
  dies, any orphaned ephemeral runpod sandbox stays running until
  someone (or `_owned_pod_ids` reconciliation on the next backend
  restart) kills it. Persistent pods (`REPROLAB_RUNPOD_POD_ID`) are
  unaffected.
- Single-worker uvicorn (`--reload`) blocks all other endpoints behind
  one slow SSE stream. The frontend already mitigates this with SSR +
  proxy + client-poll timeouts (`lab/page.tsx`, `api/demo/route.ts`,
  `live-demo-client.tsx`); the durable fix is multi-worker uvicorn or
  an ASGI server with proper concurrency.

**Guardrail.**
- The `(asyncio.CancelledError, KeyboardInterrupt)` handler in
  `cli.py:cmd_reproduce` is the single chokepoint where pipeline runs
  exit. Future async entrypoints (CLI subcommands, scheduled jobs)
  should follow the same shape: catch cancellation FIRST, write status,
  return 130, then `except Exception` for anything else.
- `_atomic_write_json` / `_write_status` use the canonical
  tempfile+replace pattern. New status writers should reuse one of
  these helpers, not write directly.
- `orchestrator.py:1431` has the per-step cancellation guard. Stages
  added to the pipeline list inherit it for free.

---

## 2026-05-10 — Runpod smoke trap destroyed a pod we wanted to keep

**Symptom.** Running `START_FULL_SMOKE=1 ./start.sh` to verify Runpod
end-to-end booted pod `nfh9zaeetfubv0` (RTX 4090 SECURE, $0.69/hr) — exactly
what we wanted. When we SIGTERM'd the script mid-boot, we were about to lose
the pod even though we hadn't gotten our verification yet. Separately, when
the user later asked "can we just use my coworker's pod that's already on
the account?", the answer was "the smoke flow has no concept of that — it
always creates and destroys its own."

**Root cause.** Two design assumptions in the Runpod tooling collided with
the actual workflow:

1. `scripts/runpod_check.sh` installs `trap cleanup_pod EXIT` immediately
   after pod creation (line 361). The trap issues a raw `curl -X DELETE`
   against `/pods/${POD_ID}`, **bypassing** the `RunpodBackend._owned_pod_ids`
   allowlist + `reprolab-` name-prefix guard that protects coworker pods on
   the same account. The trap is correct for its designed purpose
   (boot → nvidia-smi → tear down, never leak money on failure), but
   incompatible with "boot a pod and keep it."
2. `RunpodBackend.delete_on_destroy` defaults to `True` (config.py:89), so
   even pods created via the dashboard get deleted after each run unless
   `.env` overrides it. There is no first-class "attach to existing pod"
   mode — every `create_sandbox` call hits `POST /pods`.
3. The May 2026 REST v1 API has no GPU-listing endpoint, so the only way to
   know whether a 4090 is bookable is to actually book one. That pushes
   teams toward `--start-pod`-style smokes, which then collide with point 1.

**Fix.**

- For *auth + key* verification only: `./scripts/runpod_check.sh` with **no
  flag**. Free, no pod boot, no trap risk. This is what `start.sh` runs by
  default before booting uvicorn.
- For *first-time GPU bookability* verification: `--start-pod` is fine
  **provided you let the trap finish naturally**. SIGKILL bypasses the trap
  and leaks a pod; SIGTERM lets the trap fire and destroys the pod. Neither
  is what you want if you intend to keep using the pod afterwards.
- For *persistent pod usage* (the real workflow): set
  `REPROLAB_RUNPOD_DELETE_ON_DESTROY=false` in `.env`. The dashboard /
  `--sandbox runpod` flow will then leave pods running after each pipeline
  finishes. Reuse a coworker's pod by adding their public key to your local
  `REPROLAB_RUNPOD_SSH_PUBLIC_KEY` — RunPod injects it via `PUBLIC_KEY` env
  var on `runpod/*` images, no custom start command needed.
- For *single-pod reuse across runs* (skip per-run boot, attach to a fixed
  worker): set `REPROLAB_RUNPOD_POD_ID=<pod-id>` in `.env`. The backend
  fetches the pod, attaches via SSH, and reuses it for every pipeline run.
  The pod is structurally undeletable — never added to `_owned_pod_ids`,
  so `_delete_pod` refuses. If the configured pod is missing or stopped,
  the backend creates a new persistent pod and logs the new id at WARNING
  (`RUNPOD_PERSISTENT_POD_CREATED pod_id=…`); update `.env` with that id
  to reuse it on subsequent runs. Constraint: this assumes one pipeline
  run at a time on the shared pod (the `/workspace/work` symlink is
  per-pod, not per-run).
- The `_owned_pod_ids` allowlist + `reprolab-` name-prefix check in
  `runpod_backend.py:_delete_pod` already prevents the backend from deleting
  any pod it didn't create itself (defense against logic bugs and shared-
  account accidents). That guard is the *only* thing protecting your
  coworker's pods if they share a Runpod account with you.

**Lesson.** **A "smoke test" that boots real paid infrastructure is two
features in a trench coat, and they fight.** The cleanup-on-failure trap is
correct for "did this work end-to-end, free if not," and wrong for "boot
something I want to keep." Don't try to repurpose one for the other by
killing the script with the right signal — that's spell-casting, not
engineering. When the workflow shifts from "verify + tear down" to "verify +
keep," take a different code path: skip the smoke, set
`DELETE_ON_DESTROY=false`, and let the backend's normal create-sandbox flow
do the booking with the real safeguards (`_owned_pod_ids`, name prefix)
intact.

May 2026 Runpod REST v1 facts worth remembering so we don't drift:
- Endpoint: `POST https://rest.runpod.io/v1/pods`
- Auth: `Authorization: Bearer <key>` (key prefix is `rpa_…`)
- Payload uses `gpuTypeIds: ["NVIDIA GeForce RTX 4090"]` (plural array form,
  per the docs' curl examples). The OpenAPI schema lists `gpuTypeId`
  singular, but the live API accepts the plural array — match the curl
  examples, not the schema.
- `ports: ["22/tcp"]` — string form with protocol suffix.
- Official `runpod/*` images read `PUBLIC_KEY` (and `SSH_PUBLIC_KEY`) env
  vars automatically; do **not** override `dockerStartCmd` for them or you
  will lose RunPod's own SSH bootstrap. Custom `dockerStartCmd` is only
  needed for third-party images (handled in
  `runpod_backend.py:_runpod_start_command`).
- REST v1 has no GPU-listing endpoint. Fail-on-creation is the only signal
  that a configured GPU type isn't bookable on your account/region.

**Guardrail.**
- `RunpodBackend._owned_pod_ids: set[str]` (`runpod_backend.py:98`) is
  populated only on backend-created pods, and `_delete_pod` refuses to issue
  DELETE for any pod ID outside that set. Coworker's pods on the same
  account are structurally unreachable from the backend's delete path.
- `_delete_pod` belt-and-suspenders: even if a pod ID ended up in the
  allowlist via some future code path, the pod's name must start with
  `reprolab-` or DELETE is refused (`runpod_backend.py:444-449`).
- `.env` documents `REPROLAB_RUNPOD_DELETE_ON_DESTROY` and recommends
  `false` for shared-pod workflows. The default (`true`) stays as-is so
  one-off runs still clean up.
- `start.sh` runs the **free** preflight by default; `START_FULL_SMOKE=1`
  is opt-in only. Never make the paid smoke the default — money + traps =
  silent footguns.

---

## 2026-05-10 — Hermes Agent oversight silently no-oped on every run

**Symptom.** `hermes_step_reports` and `hermes_checkpoint_reports` in pipeline
state always showed `status=unavailable` with `summary="Nous Hermes runtime
unavailable"`.  The oversight layer was integrated into the orchestrator but
never actually audited anything.

**Root cause.** Two compounding issues:

1. `NousHermesClient._run_agent()` called `importlib.import_module("run_agent")`
   to load the Nous Hermes Agent runtime, but the `hermes-agent` package was
   never installed.  Every call raised `ModuleNotFoundError`.
2. The constructor hardcoded `model="anthropic/claude-sonnet-4"` without
   passing `api_key` or `provider` to `AIAgent`.  Even after installing the
   package, Hermes Agent's provider resolver could not find credentials because
   `ANTHROPIC_API_KEY` was empty in `.env` — only `OPENAI_API_KEY` was set.

The `audit()` method caught all exceptions and returned an `unavailable`
report, so the pipeline never crashed — but oversight was entirely dead.

**Fix.**

1. Installed `hermes-agent` (`pip install git+https://github.com/NousResearch/hermes-agent.git`).
2. Rewrote `NousHermesClient` (`backend/hermes_audit/client.py`) with:
   - `_resolve_hermes_config()` — auto-detects available API keys
     (`ANTHROPIC_API_KEY` preferred, `OPENAI_API_KEY` fallback) and returns
     the correct `(model, api_key, provider)` triple.
   - Explicit `api_key=` and `provider=` passed to `AIAgent()` so Hermes
     doesn't rely on its own config wizard / env-var discovery.
   - **Fallback chain:** Hermes Agent → Claude Code SDK (`claude_agent_sdk.query()`)
     → unavailable report.  The Claude SDK is already installed for the main
     pipeline, so it serves as a zero-config fallback.

**Lesson.** **A graceful degradation path that is always active is
indistinguishable from a missing feature.**  The original code's
`try/except → unavailable` was correct for resilience, but without any
logging, alerting, or test that asserts the *happy* path works, the feature
shipped dead.  When you add a `try/except → soft fallback`, always pair it
with:
- A log line at WARNING level so the fallback is visible in stderr
- A test that exercises the primary path with a mock
- A test that exercises the fallback path with the primary disabled

**Guardrail.**
- `tests/test_hermes_audit_service.py::test_client_uses_hermes_agent_when_available`
  asserts the primary Hermes Agent path produces a valid report.
- `tests/test_hermes_audit_service.py::test_client_falls_back_to_claude_sdk_when_hermes_unavailable`
  asserts the Claude SDK fallback activates when Hermes fails.
- `tests/test_hermes_audit_service.py::test_client_returns_unavailable_when_both_backends_fail`
  asserts the final unavailable fallback with error details.
- `tests/test_hermes_audit_service.py::test_client_resolve_config_prefers_anthropic_key`
  and `test_client_resolve_config_falls_back_to_openai_key` lock in the
  credential resolution order.

---

## 2026-05-09 — `database disk image is malformed` on `reprolab.db`

**Symptom.** `reprolab reproduce …` boot crashes:
```
File "backend/eventstore/sqlite_store.py", line 132
    boot = _new_connection(self._path)
sqlite3.DatabaseError: database disk image is malformed
```
`sqlite3 reprolab.db "PRAGMA integrity_check"` confirms the file is corrupt.

**Root cause.** `SqliteEventStore` ran in WAL mode with the SQLite default
`synchronous=NORMAL`. NORMAL is fast — it doesn't fsync the WAL on every
commit — but if the writer process is `SIGKILL`'d at exactly the wrong
moment between a WAL write and the next checkpoint, the main DB file can
be left referring to pages that the WAL never committed. The most recent
killed `backend.cli reproduce` subprocess (mid-pipeline crash on the IPv6
URL bug) was the proximate trigger.

**Fix.** `backend/eventstore/sqlite_store.py:_new_connection` now sets
`PRAGMA synchronous=FULL`. The throughput cost is negligible at our write
rate (≈ a few hundred events per pipeline run); the durability win is the
whole point of an event store. The corrupt DB was quarantined to
`reprolab.db.corrupt-<timestamp>` and the offline backup restored.

**Lesson.** **Default SQLite settings are tuned for read-heavy app caches,
not for event stores.** Any code path where a SIGKILL'd process must leave
the DB in a recoverable state needs `synchronous=FULL` (or at minimum
`synchronous=NORMAL` with explicit `PRAGMA wal_checkpoint(TRUNCATE)` after
each commit batch). NORMAL + WAL is a fine combination for a process you
control the lifecycle of, but pipelines crash and dev servers get
`Ctrl+C`'d — assume the worst.

**Guardrail.**
- Inline comment in `_new_connection` cites this entry.
- `learn.md` cross-cutting principle #9 (added below) generalises the
  "configure for the failure mode you actually have" rule to any local
  store.

---

## 2026-05-09 — Per-agent budget caps must be elegant, not silent

**Symptom.** Two related complaints from the same root cause:
1. Agents would silently fail at turn 16 with the SDK's opaque
   `"Reached maximum number of turns (15)"` exception bubbling out — no
   structured signal, no partial-output preservation, no remediation
   hint. The lab UI just showed the run as `failed` with no actionable
   detail.
2. With turn caps removed entirely, runaway agents (infinite tool-call
   loops, model-side hallucinated retries) had no stop condition other
   than killing the dev server.

**Root cause.** The original implementation conflated two concerns:
"how do we bound a misbehaving agent" and "how do we surface that
boundary being hit". The fix-by-removal made the second worse; the
fix-by-numerical-cap made the first worse.

**Fix.** Three independent governors per agent invocation, each with a
typed exception:

| Governor | Efficient | Max | Enforced by |
|---|---|---|---|
| `max_turns_per_agent` | 30 (60 heavy) | None | SDK `--max-turns` flag |
| `max_tool_calls_per_agent` | 80 | None | orchestrator counter |
| `agent_wall_clock_seconds` | 1200 (20 min) | 3600 (1 hr) | `asyncio.timeout` wrapping `runtime.run_agent` |

All three raise the same typed exception:
```python
class AgentLimitExceeded(RuntimeError):
    agent_id: str
    kind: Literal["turns", "tool_calls", "wall_clock"]
    limit_value: int
    elapsed_seconds: float
    partial_output: str   # preserved for retry / logging / display
```

The orchestrator additionally **converts the SDK's untyped
`Reached maximum number of turns (N)` exception** into the same
`AgentLimitExceeded(kind="turns")` via a regex match, so callers
never have to string-match exception text. The frontend timeline panel
+ `agent_telemetry.jsonl` already render `error_message`, so partial
output and the kind/value of the limit hit surface in the UI for free.

**Lesson.** **Bounded resources are a product surface, not an
implementation detail.** When a budget cap fires, the system must:
1. Preserve partial work (don't blow away the `collected_text` buffer)
2. Tell the operator *which* budget fired and *what value* it was at
3. Suggest remediation (`--execution-mode max` raises all caps)
4. Be programmatically inspectable so retry / fallback logic can branch
   on `kind`, not on string-matched English

**Guardrail.**
- `tests/test_execution_modes.py::test_execution_profile_efficient_caps_at_30_turns_and_80_tool_calls`
  locks in the numerical contract.
- `tests/test_agent_runtime_orchestrator.py::test_orchestrator_converts_sdk_turn_cap_message_to_typed_exception`
  asserts the SDK-error → typed-exception conversion path.
- `tests/test_agent_runtime_orchestrator.py::test_orchestrator_uses_efficient_default_caps_for_heavy_agents`
  asserts that heavy agents see the heavy-agent caps end-to-end.

---

## 2026-05-09 — `Reached maximum number of turns (15)` aborts every real run

**Symptom.** Frontend lab page shows the run failing in `paper_understood`.
Stderr trace ends with:
```
Exception: Claude Code returned an error result:
  Reached maximum number of turns (15)
```
Even though commit `42aa8f5 fix: remove agent turn caps` had previously
removed the caps, real runs (e.g. `paperbench1.pdf`) still aborted at turn 16.

**Root cause.** `backend/agents/execution.py::ExecutionProfile.from_mode`
silently re-introduced `max_turns_per_agent=15` (efficient) / `25` (max) and
added a `max_tool_calls_per_agent=250` cap. The values were carried through:
`ExecutionProfile → orchestrator.max_turns_per_agent → AgentRuntimeSpec.max_turns →
ClaudeAgentOptions.max_turns → claude CLI --max-turns 15`. The Claude CLI then
threw the SDK exception at turn 16. None of the existing tests caught the
regression because they had been **updated** to assert the cap rather than the
absence of one.

**Fix.** `backend/agents/execution.py:69-102` — both `efficient` and `max`
profiles now set `max_turns_per_agent=None`, `heavy_agent_max_turns=None`,
`max_tool_calls_per_agent=None`. The orchestrator continues to forward
`max_turns=None` through the SDK, so neither it nor the CLI imposes a cap.
Bounding is delegated to:
- `command_timeout_seconds` (per shell command, currently 1 h / 2 h)
- The agent's submit-when-done contract (system prompts instruct the agent
  to call the submit tool when finished)

**Lesson.** **A removed limit is a contract.** If you decide a cap should not
exist, the test must assert `is None`, not `== <new_higher_value>`. Otherwise
the next refactor will silently re-introduce a cap that survives review
because the test still passes against the new number. We had a regression
because tests said "30 is the cap for heavy agents" — true at one point, but
the right invariant was "no cap".

**Guardrail.**
- `tests/test_execution_modes.py::test_execution_profile_efficient_does_not_cap_agent_turns`
  asserts `max_turns_per_agent is None` for both modes. The test docstring
  explicitly cites the bug so a future engineer raising the cap reads why.
- `tests/test_agent_runtime_orchestrator.py::test_orchestrator_does_not_cap_heavy_agents_by_default`
  asserts the propagation: orchestrator → AgentRuntimeSpec → SDK call.

---

## 2026-05-09 — `ValueError: Invalid IPv6 URL` crashes the runtime guard on bracketed agent text

**Symptom.** `paper_understood` agent died after ~60 s with:
```
File "backend/agents/runtime/base.py", line 154, in _normalize_guard_text
    parsed = urlparse("https://" + text if "://" not in text else text)
ValueError: Invalid IPv6 URL
```
Stack trace originated in `RuntimeGuard.find_blocked_term`, called by
`claude_runtime.run_agent` on every assistant text block.

**Root cause.** `_normalize_guard_text` was applied to **arbitrary agent
output** (the `text` of the assistant's narration, not a URL). Python 3.12
tightened `urllib.parse.urlsplit` to validate bracketed netlocs as IPv6
literals; any text containing `[...]` (including narration like *"build the
comprehensive PaperClaimMap [for FTRL]"*) caused `urlparse` to raise. The
exception bubbled up out of the SDK transport and aborted the agent loop.

**Fix.** `backend/agents/runtime/base.py:151-178` — split the function:
- `_canonicalize_url_term(value)` — used **only** on configured blocked
  terms (which are documented to be URL-like). Wraps `urlparse` in
  `try/except ValueError`; falls back to the lowercased input when parsing
  fails so a malformed blocked term still substring-matches.
- `find_blocked_term(text)` — now does lowercase substring matching against
  the canonicalised terms. **Never URL-parses arbitrary text.**

Same defensive try/except added at
`backend/services/ingestion/discovery/adapters/regex.py:46-53`, the second
site that fed `urlparse` text it did not control (regex-extracted URLs in
paper text).

**Lesson.** **Never URL-parse data you do not control.** Validators that
were safe on Python 3.11 became hazardous on 3.12+ because the standard
library's permissiveness changed. Any layer that calls `urlparse`,
`urlsplit`, or any other strict parser on adversarial / model-generated
strings must wrap it in `try/except ValueError`. The principle is broader:
**parse only at boundaries, not inside hot paths**, and treat input from
LLMs the same way you'd treat input from the network — possibly malformed,
always handled defensively.

**Guardrail.**
- `tests/test_runtime_guard.py::test_runtime_guard_handles_arbitrary_text_with_brackets`
  feeds the guard five flavours of bracketed text that previously crashed
  `urlparse` (including `[::1]:8080`, `[]`, `https://[malformed`, etc.).
- `tests/test_runtime_guard.py::test_runtime_guard_normalizes_blocked_term_with_brackets`
  asserts that even a malformed configured term (e.g.
  `github.com/foo[bar`) does not crash term normalisation.
- `tests/test_issue14_artifact_discovery.py::test_regex_adapter_skips_malformed_url_without_crashing`
  covers the second urlparse site.

---

## Cross-cutting principles (May 2026)

These are the practices we follow because we've now been bitten by violating
them. Read this section before adding a new agent, runtime, or boundary.

### 1. Type the cap, not the value.

If a constraint should be opt-in (e.g. "no turn cap by default"), encode that
in the type system: `int | None` with default `None`, **not** `int = 999`.
Tests must assert the type-level invariant (`is None`), not a placeholder
value, otherwise a refactor that swaps `999` for `42` will pass review.

### 2. Parse at boundaries, never in hot paths on adversarial input.

URL/JSON/YAML/regex/etc. parsers raise on hostile input. The boundary where
the parser runs determines the blast radius. Apply this rule:

| Caller | Input source | Parser? |
|---|---|---|
| Intake | User-uploaded PDF / arXiv ID | yes — wrap in try/except, surface a typed error |
| Config loader | `.env` / `config.yaml` | yes — fail loud at import time |
| Runtime guard / agent middleware | LLM output, agent narration | **no** — substring match or fall back to a permissive heuristic |
| Discovery / link extraction | Paper body text | yes — wrap in try/except, **skip** the bad match, do not crash |

### 3. Default to `None`, not to a magic number.

Every cap (`max_turns`, `max_tool_calls`, `command_timeout_seconds`,
`sandbox_memory_limit`, …) should default to `None` unless a concrete bound
is genuinely required for safety. When a bound is required, write the
constant once in `execution.py` and reference it everywhere — never inline
the literal.

### 4. Failures must be observable from the lab UI.

Backend exceptions used to disappear into `runner.stderr.log`. The frontend
now exposes:

- `ProgressStrip` — current stage, elapsed time, **stall warning** if no
  activity for ≥ 90 s (`frontend/src/lib/demo/progress.ts::STALL_THRESHOLD_SECONDS`)
- `TimelinePanel` — per-agent invocation card with success/failure dot,
  duration, error message
- `Copy debug bundle` button — `GET /api/lab/debug-bundle?projectId=...`
  returns a compact JSON (status, last 24 KB stderr, last 30 telemetry
  records, pipeline state preview, latest error) for paste-into-Claude-Code
  triage

If you add a new failure mode, make sure one of these surfaces shows it. A
silent failure is a missing UI element, not a missing log line.

### 5. Regression tests cite the bug.

Every fix in this file is locked in by a test whose **docstring** names the
symptom. The next engineer who tries to revert the fix should read why it
exists from the test alone. Convention:

```python
def test_execution_profile_efficient_does_not_cap_agent_turns() -> None:
    """Regression: capping max_turns_per_agent caused the SDK to abort
    runs at turn 16 with 'Reached maximum number of turns (15)'. ..."""
```

### 6. Don't trust auto-generated `.gitignore` exclusions.

Build artifacts (`.next/`, `tsconfig.tsbuildinfo`, `_test_logs/`, local DB
backups, sample PDFs) have repeatedly crept into `git status`. Run
`git status --porcelain | grep -E '^\?\?'` before each commit and add
patterns to `.gitignore` as you discover them. Do this **once per noise**,
not once per commit.

### 7. Hardware-conditional code paths must not be the only path.

PaperBench's SAPG paper requires a GPU for Isaac Gym. Our system has no GPU.
We deliberately built the PaperBench integration so that:

- `dry` mode validates the bundle and submission shape with no LLM call
- `--with-pipeline` mode runs the agent stack
- Code-Development rubric nodes (≈ 60 % of weight) are scored from source
  files alone, no execution required

Whenever you add a feature whose happy path needs hardware we don't have,
add a `dry` / `simulate` mode at the same time so CI and local dev can
exercise it. If the feature only works on prod hardware, it doesn't really
work.

### 9. Configure local stores for the failure mode you actually have.

A long-running pipeline can be killed at any instant — the OS killing it
for memory, the developer hitting Ctrl+C, an upstream crash leaving a
subprocess orphaned. Any local data store needs settings tuned for *that*
failure mode, not for the abstract "well-behaved process" case. For SQLite
that means `synchronous=FULL` in WAL mode, despite the small write
throughput hit. For file-backed JSON status (`runs/<project>/status.json`)
that means atomic write-and-rename, not in-place mutation. For any cache,
a `try/except` around the read with a one-shot rebuild path.

If your local store can't survive a `kill -9`, treat it the same as you'd
treat ephemeral memory and persist the source of truth elsewhere.

### 10. A removed cap is a contract — test for `is None`, not for the new number.

See learn.md 2026-05-09 ("Reached maximum number of turns") and the
follow-up "Per-agent budget caps must be elegant". When you decide a
constraint shouldn't exist, the test must assert the type-level
invariant (`assert max_turns is None`), not a placeholder value
(`assert max_turns == 999`). Otherwise the next refactor will silently
re-introduce a cap that survives review because the test still passes
against the new number.

### 11. A silent fallback needs a loud test.

When you write `try/except → return degraded_result`, you are creating a
feature that can ship dead without anyone noticing.  Pair every graceful
degradation path with: (1) a WARNING-level log so operators see it in
stderr, (2) a test that asserts the *primary* path works with a mock, and
(3) a test that asserts the *fallback* path activates when the primary is
broken.  If you only test the fallback, you'll never know the primary was
never invoked.  See learn.md 2026-05-10 (Hermes Agent no-op).

### 8. Auto-reload is your friend AND your enemy.

`uvicorn --reload` and `next dev` Turbopack both watch the working tree.
**Branch operations (`git switch -c <new-branch>` from HEAD) are safe —
tracked file contents don't change.** Operations that mutate tracked files
(`git checkout <other-branch>`, `git pull`, `git reset --hard`) will trigger
auto-reload storms in both processes and **will** kill an in-flight pipeline
run. Plan merges accordingly.

---

## Editing this file

- Add new entries at the **top** of the dated section.
- Keep each entry under ~250 words.
- Always include a regression test path.
- If a principle is violated more than twice, promote it from a per-bug
  lesson to a numbered item under **Cross-cutting principles**.
