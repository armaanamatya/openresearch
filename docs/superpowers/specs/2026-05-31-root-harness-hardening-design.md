# Root Harness Hardening + ml-intern Pattern Integration — Design

Date: 2026-05-31
Status: **Design locked via grill** (ready for `/iterate` implementation)
Branch: `harden/root-harness` (worktree `/home/sww35/openresearch-harden`, off `5.30.26_sdar`@`9ba5dec`)
Audience: Claude Code / Codex / human principal engineers

---

## 0. Thesis, scope, non-goals

**Thesis (unchanged from the originating spec):** do **not** replace OpenResearch with Claude Code, OMA, or `huggingface/ml-intern`. Harden OpenResearch as the source-of-truth harness, then integrate the best external *patterns* behind narrow contracts. Agents are workers; external harnesses are adapters; **the benchmark harness is the authority.**

**Scope of THIS pass (locked):** a *hardening sweep* — every change **tightens or revives existing logic**; nothing pulls in an external system as a dependency. Concretely:
- Gap A — provider tool-contract + SDK-isolation parity
- Gap B — experiment provenance manifest + bind final metrics to artifacts + narration-proof scoring
- Budget/watchdog centralization
- Invariant-guard tests + boot validator
- **Benchmark-integrity bug fix** (`RuntimeGuard` blacklist activation — surfaced during the ml-intern review)
- A curated set of **ml-intern-derived improvements** (borrowed *patterns*, with attribution — `huggingface/ml-intern` is Apache-2.0)

**Non-goals (explicitly deferred):**
- Building an HF execution backend (HF Jobs / HF Space). Contract locked in Appendix A; **no HF code this pass.**
- The SFT-from-traces flywheel (Appendix C — roadmap, local-only when built).
- Replacing any agent loop, scorer, or sandbox.

**Rollout discipline (invariant 8):** every phase ships its regression tests *before* the next phase becomes default-on. Additive/refactor items are default-on after tests; the three deterministic behavioral changes (hermetic isolation, metric-projection, blacklist enforcement) are default-on with a `REPROLAB_*` escape hatch; the one LLM-behavioral change (citation-clamp) ships **observe-first** then flips after one SDAR run. See §10.

---

## 1. Verified current-state baseline (recon, 2026-05-31)

All claims below were confirmed by reading the working tree. File:line anchors are load-bearing.

**Provider runtime (Gap A):**
- `AgentRuntimeSpec` is a frozen dataclass; `tools: tuple[ToolSpec,...]` is the agent's allowlist (`base.py:96-109`). `ToolSpec.input_schema` exists but is **populated nowhere** (`registry.py:64-71` builds name+description only).
- **Claude root does NOT pass `allowed_tools`** to `ClaudeAgentOptions` (`claude_runtime.py:82-91`); **OpenAI root DOES** enforce `spec.tools` (`openai_runtime.py:159`, `_build_tools`). Real drift.
- `setting_sources` appears in **zero** `.py` files. `mcp_servers` is omitted-when-empty (`claude_runtime.py:90`), not explicit `{}`. `permission_mode="bypassPermissions"` is the dataclass default (`base.py:107`), never overridden.
- Inversion: Claude restricts *sub-agents* (`_tools_for_sub_agent`, `claude_runtime.py:199-212`) but **not the root**; OpenAI restricts both.
- Installed SDK `claude-agent-sdk==0.2.87` supports `allowed_tools`, `setting_sources`, `mcp_servers`, **`strict_mcp_config`**, **`max_budget_usd`**, `can_use_tool`, `disallowed_tools`. The fix is fully feasible.
- `sdk_isolation.py` is the **aclose-retry** resilience wrapper — NOT hermetic config isolation. Do not conflate.

**Provenance / scoring (Gap B):**
- `run_id = f"{project_id}-{uuid4}"` is minted at `primitives.py:3736` then **discarded** — not in the return dict, not in the persisted record.
- `_persist_experiment_result` (`primitives.py:3099`) appends `{timestamp, **result, model_id, eval_env}` to `experiment_runs.jsonl`. **Present:** metrics, success, logs (collapsed blob), `artifact_dir` (path only), `resource_limits.sandbox_mode`, exit_code, wall_time. **Missing:** structured command (`CommandResult.command` exists at `interface.py:83` but is collapsed), `env_id` (available at call site, not persisted), seed, `run_id`, artifact hashes.
- **RLM `final_report.baseline_metrics` is model-injected** (`report.py:695`), gated only by a binary "did `run_experiment` ever run" check (`report.py:697`). **RDR projects from the artifact** (`controller.py:1184`). So **invariant 2 holds for RDR, fails for the default RLM/hybrid path.**
- `services/runtime/artifacts.py` defines `CommandLogEntry` / `append_command_log` / `write_provenance` — the **right shape**, but **dead** (only re-exported; `provenance.json` seeded `{}` and never filled).
- `verify_against_rubric` (`primitives.py:4284`) → `score_reproduction` (`leaf_scorer.py:845`). Two channels: (a) the deterministic invariant gate (`run_invariant_checks`, `leaf_scorer.py:687`) reads code only, narration-proof, but fires only when `ctx.paper_hint_invariants` is non-empty; (b) the LLM grader's `_gather_evidence` (`leaf_scorer.py:238`) **includes `reproduction_summary` (model narration)**. So invariant 7 holds only where invariants are declared.

**Budget / runtime:**
- There are **four** backends, not three: `LocalProcessBackend`, `LocalDockerBackend`, `RunpodBackend`, **`BrevBackend`** (already an external-CLI+SSH backend — proves the adapter seam).
- `RunBudget` (`budget.py`) is a **passive** value object — caps + `check_*` predicates, but enforces nothing. **Only `RunpodBackend.exec` calls `check_pod_seconds`/`check_run_gpu_usd`** (`runpod_backend.py:306,315`). `BrevBackend` doesn't accept `run_budget` at all → uncapped. GPU budget is enforced **in-arrears** (after an exec interval).
- Process watchdog `_arm_watchdog` (`run.py:824`, hard `os._exit`) is **RLM-only**; RDR's only hard stop is per-cluster `asyncio.wait_for`; legacy `_ClusterWatchdog` (`controller.py:64`) is dead.
- Backend dispatch: `_backend_for_sandbox_mode` (`primitives.py:1907`). ABC: `RuntimeBackend` (`interface.py:115`) with `create_sandbox/exec/copy_out/copy_in/destroy` + optional `probe_alive/soft_recover`.

**Benchmark-integrity (surfaced during ml-intern review):**
- `RuntimeGuard.find_blocked_term`/`raise_if_blocked` is implemented + tested (`base.py:72,81`; `test_runtime_guard.py`). PaperBench bundles ship `blacklist.txt`, parsed by `bundle.blacklist_entries()` (`evals/paperbench/bundle.py:62`).
- **But `cli.py:1240` computes `blacklist_terms` and never uses it (dead); `to_runtime_spec` (`registry.py:59-76`) never sets `guard=` → every agent gets an empty `RuntimeGuard()`.** The `baseline-implementation` agent runs on the host with a network-enabled `Bash` (the prompt itself fetches `raw.githubusercontent.com/...`), so **the paper's own repo is reachable and unblocked** — exactly what PaperBench forbids. This undermines scoring truth.

**MCP / telemetry:**
- MCP default-off in code (`_resolve_mcp_servers`, `claude_runtime.py:165` — only `apify-arxiv` SSE, only when `apify_api_token` set), but this box's `.env` has a live `APIFY_API_TOKEN` → effectively on here. **Zero tests** for `_resolve_mcp_servers`.
- No default-on trace/telemetry egress (`telemetry.py` writes local JSONL; `report_to="none"` in `rl_scaffold.py:267`). Invariant 5 satisfied **by absence** — no guard test.

---

## 2. The eight invariants — current status

| # | Invariant | Status today | Closed by |
|---|-----------|--------------|-----------|
| 1 | Only `run_experiment` creates scored facts | RDR ✓ / RLM mostly ✓ (RLM report can restate) | Gap B projection (P3) |
| 2 | Every final metric traces to a persisted artifact via a manifest | **RLM ✗** / RDR ✓ | Gap B manifest + projection (P2,P3) |
| 3 | No provider runtime silently widens tool access | Claude root unrestricted ✗ | Gap A (P1) + MCP test (P5) |
| 4 | No headless external CLI with auto-approval inside the harness | ✓ (own sub-agents are sandboxed; HF CLI never invoked) | Held; HF contract (App A) + permission test (P1) |
| 5 | No default-on trace upload for PaperBench/private artifacts | ✓ by absence | Telemetry guard test (P5) |
| 6 | No foreign local tool imported without path/network/timeout/artifact review | ✓ (Brev reviewed; HF deferred) | HF contract (App A) |
| 7 | No scoring leaf satisfied by narration alone | partial (only where invariants declared) | Validated-citation clamp (P3) |
| 8 | Every phase ships regression tests before next is default-on | process | Rollout discipline (§10) |
| — | **(implicit) Benchmark integrity — paper's own repo unreachable** | **✗ blacklist not wired** | RuntimeGuard activation (P1, #7) |

---

## 3. Gap A — provider tool / SDK parity (DECISION: authoritative + hermetic + tested)

**Change (`claude_runtime.py`):**
1. Add a shared `_tools_for_agent(agent, extensions)` helper; refactor `_tools_for_sub_agent` to call it. Pass `allowed_tools=_tools_for_agent(agent, mcp_tool_extensions)` to the **root** `ClaudeAgentOptions` — MCP-merged, so an MCP-enabled root can call its MCP tools. Omit `allowed_tools` only when the list is empty (preserve "all defaults") — but see the registry guard below.
2. `setting_sources=[]` — hermetic: no ambient `CLAUDE.md`/`.claude/settings.json`/discovered-MCP leakage.
3. `mcp_servers=mcp_servers` **always** (explicit `{}`) + `strict_mcp_config=True`.
4. **Keep** `permission_mode="bypassPermissions"` (headless runs have no approver; real controls are `allowed_tools` + sandbox + `RuntimeGuard`). Invariant 4 targets a future external-CLI shell-out, not the harness's own sub-agents.

**Guards/tests:**
- Fail-closed registry guard: no registered agent may declare empty tools (so empty never silently diverges across providers). Add to a registry validation + test.
- Cross-provider **parity test**: Claude `allowed_tools` == OpenAI tool names == registry declaration.
- `permission_mode` regression test (pins `bypassPermissions`; catches a future accidental change → invariant 4).
- `setting_sources=[]` / `strict_mcp_config=True` assertion test.

**Escape hatch:** `REPROLAB_SDK_HERMETIC` (default true) disables `setting_sources=[]`/`strict_mcp_config` for local debugging.

**Footnote (do not block on):** `ToolSpec.input_schema` is dead. Leave it (reserved for future typed/MCP tools per Appendix A); document, do not wire this pass.

---

## 4. Benchmark-integrity — RuntimeGuard blacklist activation (#7) (DECISION: ADOPT, do first)

This is a **bug fix**, sequenced in P1 because it shares the runtime/registry surface with Gap A and is foundational to scoring truth.

**Change:**
- `to_runtime_spec` (`registry.py:38-76`): accept + set `guard=RuntimeGuard(blocked_terms=...)`. Threaded uniformly so **ALL** registry agents (baseline-implementation, improvement-path, rubric-verifier, improvement-orchestrator) get the guard — no agent legitimately needs the paper's own repo.
- Thread blocklist through `RunContext` (`context.py` — add `blocked_terms` field) → `invoke.py:42` (`collect_agent_text` `blocked_terms=` param) → `to_runtime_spec` → every agent spec. Callers: `baseline_implementation.py:2160`, `rdr/agent.py:155`.
- `cli.py:1240`: stop discarding `blacklist_terms`; feed it into `ctx.blocked_terms`.

**Blocklist sources — CURATED only (grill-resolved 2026-05-31; supersedes the regex-derivation below):** three precise sources, unioned:
  1. `bundle.blacklist_entries()` (`evals/paperbench/bundle.py:62`) — paperbench-bundle runs.
  2. cli `--blacklist` (`cli.py:1240`) — explicit override.
  3. **NEW: arXiv-id-keyed `paper_hints` blocked-resources list** (`prompts/paper_hints.py`) — protects the canonical SDAR *arXiv* run (`reproduce --paper-hint 2605.15155`), which loads neither the ftrl bundle nor `--blacklist`. SDAR entry: `2605.15155 → github.com/BartekCupial/finetuning-RL-as-CL` (mirrors `third_party/paperbench/ftrl/blacklist.txt`, a single curated line).
  - **DROPPED: regex auto-derivation from the discovery adapter** (`regex.py`). It sweeps up *all* cited `github:owner/repo` including legitimate framework deps (`huggingface/trl`) → would break the reproduction AND fail this section's own "trl stays allowed" test. Curated lists are precise; auto-derivation is not. A precise author-repo classifier could revisit this later.
  - **Visibility:** emit a `run_warning` when a benchmark/paper-hint run resolves an **empty** active blocklist, so the integrity gap is never silent.
- The URL canonicalizer (`base.py:188`) already matches `…/x/y.git` ≡ `github.com/x/y`.

**Enforcement is DETECTIVE in P1 (grill-resolved):** the Claude guard check (`claude_runtime.py:101-107`) is post-hoc — it inspects `tool_input` *after* the model emits the call, but under `bypassPermissions` the CLI has already run the local Bash, so a blocked fetch touches host disk *before* the `RuntimeGuardViolation` raises and fails the run. Score-truth is preserved (a detected cheat yields no passing score), but the bytes transiently exist. True PREVENTION needs the SDK `can_use_tool` callback, which (verified in 0.2.87, `client.py:161`) requires **streaming-input mode + a non-`bypassPermissions` mode** — a larger rework. **Scheduled as an explicit follow-up phase (P1.5), not merely logged.** OpenAI is already preventive (we own its Bash tool; guard checked before `subprocess.run`).

**Tests:** the paper's own repo (`github.com/BartekCupial/finetuning-RL-as-CL`, the SDAR blacklist entry) raises `RuntimeGuardViolation` from the authoring agent; a non-blacklisted framework repo (`huggingface/trl`) does not; the SDAR arXiv run (`2605.15155`) resolves a non-empty guard via `paper_hints`.

**Escape hatch:** `REPROLAB_BENCHMARK_GUARD` (default true). Off only for non-benchmark exploratory runs.

---

## 5. Gap B — provenance, projection, narration-proof scoring

### 5a. Manifest (DECISION: enrich in-place + revive `commands.log`)
- **Stamp at the chokepoint** `_persist_experiment_result` (`primitives.py:3099`). Stop discarding `run_id` (`:3736`). Add to the `experiment_runs.jsonl` record: `experiment_run_id`, `env_id`, `sandbox_backend` (promote from `resource_limits`), structured command list, `metrics_sha256` (hash of `metrics.json`). Best-effort (record `null`+reason): image **digest** (opaque on RunPod → record name/tag), `seed` (the `--seed` value if set), per-artifact hashes.
- **Revive `services/runtime/artifacts.py`**: every backend `exec` appends a `CommandLogEntry` to `commands.log` (so `CommandResult.command` stops collapsing into the blob).
- **Back-link** `experiment_run_id` + `metrics_sha256` into `final_report` (completes: metric → experiment record → `metrics.json` hash → `commands.log` → backend).
- Hard-required (fail-closed): `experiment_run_id`, `env_id`, `sandbox_backend`, structured command, `metrics_sha256` (on success), `exit_code`, `success`, `wall_time_s`.

### 5b. Metric binding (DECISION: project from artifact)
- `final_report.baseline_metrics` is **projected from the canonical experiment record's `metrics.json`** — the root no longer types metric values. The root **selects** the canonical `experiment_run_id` (deterministic fallback: the record matching the final verified state, else latest successful).
- Model-derived/summary numbers move to a clearly **non-authoritative** narrative field (never fed to the leaderboard/scorer as ground truth). Makes RLM behave like RDR (`controller.py:1184`).
- **Escape hatch:** `REPROLAB_METRIC_PROVENANCE` (default true).

### 5c. Invariant 7 — validated citation (DECISION: require validated citation, observe-first)
- The LLM grader must cite a concrete `file:line` OR `metric-key` for any leaf scored > 0; the citation is **validated against the gathered evidence** (file present in code listing / key present in `metrics.json`). Uncited / unvalidatable positive → clamp 0. Label `reproduction_summary` as untrusted narration.
- **Observe-first:** ships logging the *would-clamp* deltas (enforce-OFF) and flips to enforce after one SDAR run confirms no spurious score loss (`REPROLAB_RUBRIC_REQUIRE_CITATION`).
- Test (deterministic even for an LLM path): mock a citation-less positive → assert clamp.

---

## 6. Budget / watchdog / conformance (DECISION: ABC template-method)

- **`RuntimeBackend.exec` becomes a concrete template method** wrapping an abstract `_exec_impl()`. It tallies cumulative cost via a per-backend `cost_rate_usd_per_hour()` hook (0 for local) and calls the `check_*` predicates + wall-clock around every exec. RunPod's logic **moves up** (not rewritten); Brev + future HF inherit the cap. `run_budget`/`gpu_plan` become base state.
- **Upfront projected-cost gate (#5):** before `create_sandbox`/first exec, compute `projected_usd = (max_pod_seconds/3600) * rate` and call `check_run_gpu_usd` — closes the in-arrears overshoot. Slots into the template-method's call site.
- **Watchdog unify:** lift `_arm_watchdog` to a shared harness step both RLM and the RDR entry call; delete dead `_ClusterWatchdog`. Defer a first-class `cancel()` API.
- **Conformance test:** parametrized over all four backends — asserts each honors the contract (budget enforcement, manifest emission, path/network boundaries). Guards the existing backends through the refactor; later targets the HF backend.

---

## 7. Guard tests + boot validator (DECISION: tests + boot validator)
- MCP default-off test: `_resolve_mcp_servers()` returns empty when token unset; correct SSE shape + tool-extension merge when set (invariant 3).
- Telemetry-stays-local negative test: fail if any HTTP/upload egress occurs during a run (invariant 5).
- **BUG-LR-014** boot validator: warn (don't block) when a shell credential shadows `.env` (`OPENAI_API_KEY`/`ANTHROPIC_API_KEY`/`REPROLAB_RUNPOD_API_KEY`/`APIFY_API_TOKEN`).

---

## 8. ml-intern-derived borrows (patterns only; Apache-2.0, attributed)

ml-intern is **not** better than OpenResearch overall — OpenResearch leads on the hard problems (26-class failure classifier, multi-signal watchdog, fcntl GPU leasing, OOM escalation, owned-pod allowlist, quality-scored HTML>PDF>OCR cascade, structural SSE egress projection). These are the **narrow, real** wins, all tightening existing logic.

| # | Borrow | ml-intern source | OpenResearch target | Phase | Default |
|---|--------|------------------|---------------------|-------|---------|
| 1 | **Doom-loop detector** (in-flight; sig = code **+** stdout/result hash to not kill polling) | `agent/core/doom_loop.py:104` | gap; hook `sse_bridge.py:353`; inject via REPL stderr (`safe_repl_traceback_patch.py:52` seam). Closes **BUG-LR-015** | P5 | on + `REPROLAB_DOOM_LOOP` hatch |
| 2 | **ar5iv HTML fallback** | `papers_tool.py:721` | `arxiv.py:_fetch_html` (add `ar5iv.labs.arxiv.org/html/<id>` 2nd source, reuse existing validation) | P0 | on |
| 3 | **Sweeper ownership guard** | `sweep_orphan_sandboxes.py:71` | `pod_sweeper.py:192-260` (add `name_prefix="reprolab-"` filter — the backend already enforces it at `runpod_backend.py:1089`) | P4 | on |
| 4 | **Sweeper preserves in-flight** | `sweep_orphan_sandboxes.py:130` | `pod_sweep_scheduler.py:70` — feed `preserve_pod_ids` from the live-run registry (plumbed but never fed) | P4 | on |
| 5 | **Upfront projected-cost gate** | `cost_estimation.py:219`+`agent_loop.py:338` | template-method exec (§6) | P4 | on |
| 6 | **Chat secret-scrubber** | `redact.py:18` | `respond_to_user` (`primitives.py:4848`) + `post_message` (`messages.py:73`) — regex-scrub tokens at ingress+egress (complements, not replaces, `sse_bridge`) | P5 | on |
| 7 | **RuntimeGuard blacklist activation** | (ml-intern `research_tool.py:108` `blocked_domains` inspired the check) | §4 | P1 | on + `REPROLAB_BENCHMARK_GUARD` hatch |
| 8 | **GitHub reference-impl discovery** (`find_examples`+`read_file`, org-allowlist + guard-gated) | `github_find_examples.py:267`, `github_read_file.py:67` | new optional tool on `baseline-implementation`; **requires #7**; needs `GITHUB_TOKEN`+`thefuzz` | P6 | **off** (`REPROLAB_GITHUB_EXAMPLES`) |
| 9 | **Run-complete notifications** | `messaging/gateway.py:24`, `slack.py:126`, `session.py:248` | hook `build_run_complete_event` (`sse_bridge.py:477`); summary-only ⇒ invariant-5-safe | P6 | off (`REPROLAB_NOTIFY_*`) |
| 10 | Validate-saved-model on resume | `session_resume.py:224` | `resume_run` (`live_runs.py:583`) | P4 | on |
| 11 | Leaderboard percentiles | `build_kpis.py:131` `_percentile` + zero-exclusion | `leaderboard.py:150` (cost p50/p95 per paper/model) | P5 | on |
| M1 | Root-model rate-limit backoff | `agent_loop.py:411` (2-tier, total>60s) | `claude_oauth_client.py:309`, OpenAI client | P4 | on |
| M2 | arXiv `not_a_pdf` retry (arxiv-only) | `papers_tool.py:74` | `remote_pdf.py:205` (retry `not_a_pdf` only when `fetched_via=="arxiv"`) | P0 | on |
| M3 | Live price-catalog + static fallback | `cost_estimation.py:194` | `gpu_catalog.py`/`pricing.py` (`{**static, **live}`) | P4 | **off** (no URL ⇒ static = today) |
| M4 | head+tail `exec.log` truncation + pointer | `sandbox_client.py:125` + `local_tools.py:71` (host path) | **all backends** (scope widened 2026-05-31 audit): same tail-only `[-32000:]` verified at `runpod_backend.py:1323` AND `local_docker.py:351`; `local_process.py` has **no cap at all** (add one). Factor a shared `truncate_head_tail(text, max_chars, head_ratio≈0.25, spill_path)` → `services/runtime/` (alongside revived `artifacts.py`); spill pointer reuses `commands.log`. | P4 | on |

**Provenance gate (2026-05-31 borrow sweep).** `tanbiralam/claude-code` and `lolout1/NOT_CLAUDECODE` were evaluated as a requested pattern source and **rejected wholesale**: a clone-and-`diff` proved both are the *same artifact* — the **leaked, unlicensed proprietary source of Anthropic's Claude Code CLI** (byte-identical `src/`; `package.json` `name:"claude-code"` + `@anthropic-ai/*` internal deps; WebFetch guard calls `api.anthropic.com/api/web/domain_info`; `NOT_CLAUDECODE` re-badges it and strips the leak disclosure). Borrowing from it would IP-contaminate OpenResearch's clean-room — a benchmark-integrity liability. **Action: quarantine from the `harden` branch.** The one relevant capability it showcases (a parse-tree shell-command-safety classifier that would harden the §4 `RuntimeGuard` beyond substring matching — zsh `=cmd` expansion, process substitution, `&&`-split escapes) is to be built **clean-room from public refs** (`tree-sitter-bash`/`bashlex` + public PaperBench blacklist semantics), never from this leak. Tracked as a candidate hardening of #7, NOT a borrow.

**Rejected (recorded in Appendix B):** `finish_reason=length` repair, dangling-tool-call repair, malformed-args guard, named-section lookup, figcaption (we're better), S2 enrichment, `_owns_space` (we're better), reconnect-during-stream (→ HF contract note), `redact`-as-`sse_bridge`-replacement, ml-intern `approval_policy`, effort-probe/model-switching, prompt-caching (already done), telemetry kind-tagging, context-compaction, `plan_tool`, research-tool, session-resume mechanism, mongo doc-size guard, KPI scheduler.

---

## 9. Phase plan (each phase ships its tests)

- **P0 — Paper fidelity** (synergistic with committed `ec3fbc3`): ar5iv fallback (#2), arXiv `not_a_pdf` retry (M2).
- **P1 — Provider-runtime hardening:** Gap A parity + hermetic (§3) **+ RuntimeGuard blacklist activation (#7, §4, DETECTIVE)** — same registry/runtime surface. Tests: parity, permission_mode, hermetic, fail-closed empty-tools, blacklist-blocks-paper-repo, SDAR-arxiv-nonempty-guard.
- **P1.5 — Preventive guard (scheduled follow-up; grill-resolved 2026-05-31):** rework `claude_runtime` to streaming-input mode + a permission model where `can_use_tool` fires (deny-on-blocked instead of relying on `bypassPermissions`), so a blocked fetch is *prevented*, not just detected-after-the-fact. Bigger blast radius (every tool routes through the callback) ⇒ its own phase after P1's detective guard + tests land. Test: `can_use_tool` denies a Bash command referencing a blocked term before execution.
- **P2 — Manifest (§5a):** revive `artifacts.py` + enrich `experiment_runs.jsonl` + persist `run_id` + back-link. Additive. Tests: manifest fields present, `commands.log` written, back-link round-trip.
- **P3 — Scoring truth:** metric-projection (§5b) + invariant-7 citation observe-first (§5c). Tests: projection-from-artifact, model-injection-rejected, citation-clamp (mocked).
- **P4 — Budget/runtime (§6):** template-method + watchdog unify + conformance test + #5,#3,#4,#10,M1,M3,M4. Tests: Brev/local budget inheritance, upfront gate, sweeper ownership+preserve, watchdog-on-RDR, conformance (all backends).
- **P5 — Loop & egress safety + guards:** doom-loop (#1), chat scrubber (#6), MCP default-off test, telemetry-egress test, boot validator (BUG-LR-014), leaderboard percentiles (#11).
- **P6 — Capability (after P1's #7):** GitHub reference-impl tool (#8, default-OFF, allowlist+guard-gated), run-complete notifications (#9, off-by-default). Tests: allowlist+guard reject blacklisted/non-allowlisted repos; notification fires on completion with summary-only payload.

**Appendices ship with the doc:** A (HF contract), B (rejection log), C (SFT roadmap), D (native gaps).

---

## 10. Rollout / flag table

| Change | Class | Default | Flag |
|--------|-------|---------|------|
| Gap A tools/hermetic | tighten | on after tests | `REPROLAB_SDK_HERMETIC` (hermetic only) |
| RuntimeGuard blacklist (#7) | behavioral (deterministic) | on after tests | `REPROLAB_BENCHMARK_GUARD` |
| Manifest (§5a) | additive | on after tests | — |
| Metric-projection (§5b) | behavioral (deterministic) | on after tests | `REPROLAB_METRIC_PROVENANCE` |
| Citation-clamp (§5c) | behavioral (LLM) | **observe-first → flip post-SDAR** | `REPROLAB_RUBRIC_REQUIRE_CITATION` |
| Budget template-method + #5/#3/#4/M1/M4 (§6) | refactor/additive | on after tests | — |
| M3 price-catalog | additive | off (no URL ⇒ static) | `RUNPOD_PRICE_CATALOG_URL` |
| Doom-loop (#1) | additive (in-flight inject) | on after tests | `REPROLAB_DOOM_LOOP` |
| Chat scrubber (#6) | additive | on after tests | — |
| Guard tests / boot validator | additive (warn) | on | — |
| GitHub tool (#8) | net-new capability | **off** | `REPROLAB_GITHUB_EXAMPLES` (+`GITHUB_TOKEN`) |
| Notifications (#9) | net-new capability | off (empty config) | `REPROLAB_NOTIFY_*` |

---

## Appendix A — HF / ml-intern integration contract (DEFERRED; no code this pass)

`huggingface/ml-intern` is an **autonomous agent + HF-native toolset** (300-iter `agent_loop.py`; tools: papers/datasets/Hub/Jobs/Spaces), **with no built-in scoring** — confirming it is a *worker*, and OpenResearch's `verify_against_rubric` stays the authority. Two invariants it specifically trips: its CLI runs **headless with auto-approval** (invariant 4 — never shell to it) and it **auto-uploads traces to HF datasets** (invariant 5 — must be disabled).

**Locked contract for any future HF integration:**
1. **Execution** (HF Jobs / HF Space) = a 5th `RuntimeBackend` on the **unchanged** 5-method ABC. The async/queued impedance is the *adapter's* problem (collapse commands into one job script at first `exec`; **reconnect-with-terminal-state-check** since liveness is a re-fetchable server-side job status, not a live socket — per `jobs_tool.py:435`). It routes through `run_experiment` + the manifest + the centralized budget (`cost_rate_usd_per_hour()` hook) + the scorer. Must pass the §6 conformance test.
2. **Tools** (papers/datasets/Hub) = future domain primitives behind the harness tool contract; **read-only by construction** (no `push_to_hub`/upload reachable under `bypassPermissions`).
3. **Agent** (ml-intern as a builder) = behind the `AgentRuntime` contract, scored by *our* harness.
4. **Trace upload off by default**, behind an explicit flag, guarded by the §7 telemetry test.

---

## Appendix B — Honest rejection log

Recorded so we don't re-litigate. Each was rejected because OpenResearch is equal-or-better, the failure mode doesn't exist on our architecture, or it's net-new surface for no gain: `finish_reason=length` tool-call repair (REPL not tool-JSON), dangling-tool-call repair (REPL), malformed-args guard (subsumed by doom-loop + forced_iteration), named-section lookup (fights offload-corpus design), figcaption preservation (we re-inject captions, `html_parser.py:92`), Semantic-Scholar enrichment (net-new external surface, out of scope), `_owns_space` (our `_owned_pod_ids` allowlist is stronger), reconnect-during-stream for RunPod (large retrofit; → HF contract note instead), `redact` as `sse_bridge` replacement (sse_bridge is structurally stronger; only the additive chat scrubber taken), ml-intern `approval_policy` (our `ApprovalService` is richer — see App D), effort-probe/model-switching (intentionally not exposed), prompt-caching (already in `_oauth_backend_patch.py`), telemetry kind-tagging (equivalent attribution exists), context-compaction (RLM offloads the corpus), `plan_tool` (rubric-gated termination beats a self-reported todo), research-tool (our `rlm_query` is the better-integrated version), session-resume mechanism (architecture mismatch; our checkpoint+archive is deliberate), mongo doc-size guard (our `repl_snapshot.py` tombstone guard is stronger), KPI scheduler (overkill vs request-time leaderboard).

---

## Appendix C — SFT-from-traces flywheel (ROADMAP, not this sweep)

A north-star: turn completed reproduction runs into tagged fine-tuning data to improve the agents over time. ml-intern's `sft/tagger.py:181` (pure `trajectory→tags`) + `scripts/build_sft.py:96` (`_reshape_to_sft`) are the references. **Constraints for OpenResearch:** the upload half is invariant-5-forbidden → **local-only**; an RLM REPL+code-block trajectory is a poorer SFT substrate than tool-call threads (reshape is real work); needs a consumer first. OpenResearch already has richer raw material to feed a tagger (`failure_classifier` 26 classes, `_authoritative_primitive_trace.by_primitive`, cost ledger). Defer until there's a fleet/KPI consumer.

---

## Appendix D — Native gaps surfaced (not from ml-intern)

- **`ApprovalService` is unwired from the RLM run path.** `backend/services/approval/service.py` is a full SQLite-backed approval system with typed thresholds (`dataset_download`, `long_run`, `gpu_spend`, `external_upload`, …) but `backend/agents/` references it **zero** times — only routes/tests use it. Wiring it into `run_experiment`/`build_environment`/`propose_improvements` is the right fix (adopting ml-intern's weaker inline gate would regress). Tracked as a separate follow-up.

---

## Attribution

Patterns in §8 are adapted (not copied wholesale) from `huggingface/ml-intern` (Apache-2.0). Each adopted module should carry a short `# Pattern adapted from huggingface/ml-intern <path> (Apache-2.0)` note at its definition site.
