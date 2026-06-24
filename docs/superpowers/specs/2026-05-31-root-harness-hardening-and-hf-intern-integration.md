# OpenResearch Root Harness Hardening + HF Intern Integration Spec

Date: 2026-05-31
Status: Design locked for implementation planning
Audience: Claude Code, Codex, and human principal engineers
Scope: `openresearch` / ReproLab harness quality, not a wholesale agent rewrite

## 0. Executive thesis

The correct move is not to replace OpenResearch with Claude Code, OMA, or
`huggingface/ml-intern`. The correct move is to harden OpenResearch as the
source-of-truth harness, then integrate the best external systems behind narrow
contracts.

The harness must own:

- experiment execution truth: only `run_experiment` can create benchmark facts
- scoring truth: only `verify_against_rubric` / PaperBench scorer can grade
- budget truth: cost, wall-clock, GPU, and cancellation policy live centrally
- provenance truth: every final metric has a manifest path back to code, command,
  environment, artifact, and backend
- tool truth: every provider sees the same effective tool contract

Agents are workers. External harnesses are adapters. The benchmark harness is the
authority.

## 1. Inputs and current code anchors

Current OpenResearch facts:

- Provider-neutral agent contract: `backend/agents/runtime/base.py`
- Claude SDK adapter: `backend/agents/runtime/claude_runtime.py`
- OpenAI adapter: `backend/agents/runtime/openai_runtime.py`
- Agent registry: `backend/agents/registry.py`
- RLM root construction: `backend/agents/rlm/run.py`
- RLM primitive binding: `backend/agents/rlm/binding.py`
- Authoritative execution primitive: `backend/agents/rlm/primitives.py::run_experiment`
- Authoritative scorer primitive: `backend/agents/rlm/primitives.py::verify_against_rubric`
- Sandbox backend interface: `backend/services/runtime/interface.py`
- Runtime app service: `backend/services/runtime/service.py`
- RDR deterministic controller: `backend/agents/rdr/controller.py`

External source analysis:

- `huggingface/ml-intern` is valuable as HF-native ML tooling: datasets, papers,
  Hub, HF Jobs, HF Space sandbox, traces. It is not an eval harness replacement.
- `open-multi-agent` is valuable as orchestration inspiration, not as a scoring or
  execution authority.
- The leaked Claude Code repo must not be used. Only official Claude Code / Agent
  SDK behavior and docs are acceptable inputs.

May 2026 Claude Code operating guidance applied here:

- Use explore -> plan -> implement -> verify/commit for multi-file risky work.
- Give the agent deterministic checks it can run.
- Use project settings, permissions, hooks, and subagents to constrain autonomy.
- Use subagents for context isolation and specialized review, with restricted
  tools.
- Use MCP only through explicit config and allowed tools.

Official references:

- https://code.claude.com/docs/en/best-practices
- https://code.claude.com/docs/en/settings
- https://code.claude.com/docs/en/agent-sdk/permissions
- https://code.claude.com/docs/en/agent-sdk/subagents
- https://code.claude.com/docs/en/hooks
- https://code.claude.com/docs/en/agent-sdk/mcp

## 2. Non-negotiable invariants

1. No external agent loop may bypass `run_experiment` for scored runs.
2. No final report may cite a metric unless the metric is in a persisted run
   artifact and linked from the run manifest.
3. No provider-specific runtime may silently widen tool access.
4. No headless external CLI may run with auto-approval inside the harness.
5. No trace upload is default-on for PaperBench or private artifacts.
6. No local shell/write tool from another project is imported without a path,
   network, timeout, and artifact boundary review.
7. No scoring leaf may be marked satisfied by narration alone.
8. Every phase below must ship with regression tests before the next phase becomes
   default-on.

## 3. Primary current gaps

### Gap A: provider tool-contract drift

`AgentRuntimeSpec.tools` exists, and the OpenAI runtime builds root tools from it.
The Claude runtime currently builds subagent tools via `_tools_for_sub_agent(...)`
but does not appear to pass the root agent's allowed tools to
`ClaudeAgentOptions`. This makes cross-provider comparisons suspect.

Also check SDK isolation drift. `CLAUDE.md` says Claude SDK call sites must pass
`setting_sources=[]`, explicit `mcp_servers`, and non-plan permission mode. The
current `claude_runtime.py` passes `mcp_servers` only when non-empty and does not
visibly pass `setting_sources=[]`. That must be resolved by tests, not by trust.

### Gap B: experiment provenance is not first-class enough

`experiment_runs.jsonl`, `dashboard_events.jsonl`, cost ledger, artifacts, and
final report exist, but the harness needs one canonical run manifest tying:

- code snapshot
- command list
- sandbox backend
- image / hardware
- env vars hash
- metrics file path and hash
- logs path and hash
- dataset and model identifiers
- score payload
- cost rows

This is the difference between "the agent produced numbers" and "the harness can
prove where every number came from."

### Gap C: HF integration has high value but dangerous defaults

`ml-intern` has excellent HF ecosystem tooling, but its headless mode auto-approves
and its local shell tools are too permissive for benchmark use. Integrate its ideas
and selected handlers behind OpenResearch contracts. Do not call its headless loop
from RLM/RDR.

### Gap D: scorer evidence should be more adversarial

The scorer should behave like a skeptical evaluator. Every passed leaf should have
evidence pointers to files, logs, or metrics. Borderline leaves should be
recheckable. Unsupported claims should be visible.

### Gap E: harness regressions are not packaged as a benchmark of the harness

There are many tests, but we need a small named harness regression suite that
tests the harness itself as a product:

- successful minimal reproduction
- failed experiment then repair
- bad metrics shape
- missing dataset
- provider runtime mismatch
- scorer rejects unsupported claim
- interrupted run resume
- orphaned remote job cleanup

## 4. Target architecture

The target is a contract-centered harness:

```text
RDR / RLM controller
  -> primitive registry
    -> run_experiment(...)
      -> ExecutionBackend adapter
        -> local_process | docker | runpod | hf_jobs | hf_space_sandbox
      -> RunManifest writer
      -> metrics/log/artifact validators
    -> verify_against_rubric(...)
      -> EvidenceContract validator
      -> scorer
      -> rubric_breakdown persistence
  -> final_report builder
    -> manifest and evidence reconciliation
```

External systems enter only here:

- Claude Agent SDK: provider runtime for code/research agents.
- OpenAI Agents SDK: provider runtime for code/research agents.
- HF Intern: read-only HF research tools and HF compute backends.
- OMA: optional future source of scheduling ideas only.

## 5. Phase plan

### Phase 0 - Baseline, guardrails, and no-op inventory

Goal: make the current behavior measurable before changing it.

Implementation:

- Add `docs/runbooks/harness-hardening-validation.md`.
- Record current branch, merge commit, dirty status, and the existing default run
  command for SDAR or a tiny smoke bundle.
- Add a `scripts/harness_smoke.sh` or `scripts/harness_smoke.py` that runs only
  cheap local tests by default.
- Add a `pytest` marker `harness_contract` for the new regression suite.

Acceptance:

- `python -m pytest tests/test_agent_runtime_claude_adapter.py tests/test_agent_runtime_openai_adapter.py`
  still passes before behavioral changes.
- `python -m pytest -m harness_contract` works even if initially empty or xfailed.

Rollback:

- Documentation and test marker only; no runtime behavior.

### Phase 1 - Provider runtime contract hardening

Goal: one `AgentRuntimeSpec` means one effective tool contract across providers.

Implementation:

- Add `backend/agents/runtime/contracts.py` with:
  - `effective_tool_names(spec) -> tuple[str, ...]`
  - `runtime_contract_snapshot(spec) -> dict`
  - validation for duplicate names, empty tool names, and inherited subagent tools
- Update `ClaudeAgentRuntime.run_agent(...)` to pass explicit root allowed tools
  to `ClaudeAgentOptions` after confirming the installed SDK parameter name via
  introspection. Expected modern name is `allowed_tools`.
- Always pass isolation settings:
  - `setting_sources=[]`
  - `mcp_servers={}` even when empty, if supported by installed SDK
  - non-plan `permission_mode`
- Preserve subagent tool restrictions; no subagent should inherit all tools unless
  that is explicit in the registry.
- Emit/write a per-agent `tool_contract_snapshot` into run artifacts whenever an
  agent is invoked by RLM/RDR.

Tests:

- `tests/test_agent_runtime_claude_adapter.py::test_claude_runtime_applies_root_tools`
- `tests/test_agent_runtime_claude_adapter.py::test_claude_runtime_passes_sdk_isolation_options`
- `tests/test_agent_runtime_openai_adapter.py::test_openai_runtime_contract_snapshot_matches_claude`
- registry test: every `AgentSpec` with tools has the same effective names through
  both runtimes

Acceptance:

- Claude and OpenAI test doubles receive equivalent root tool sets.
- A missing/unsupported SDK parameter is handled by a compatibility helper and
  covered by tests.
- No behavior silently widens permissions.

Rollback:

- Feature flag `REPROLAB_STRICT_AGENT_TOOL_CONTRACT=0` can restore old permissive
  behavior for emergency local use only. Default should become strict after tests.

### Phase 2 - Canonical run manifest and provenance contract

Goal: every scored fact has a stable evidence path.

Implementation:

- Add `backend/agents/rlm/manifest.py` with Pydantic models:
  - `RunManifest`
  - `ExperimentAttemptManifest`
  - `ArtifactRecord`
  - `MetricRecord`
  - `ScoreRecord`
- Write `runs/<project_id>/run_manifest.json` atomically.
- Append/update on:
  - run start
  - environment build
  - every `run_experiment` attempt
  - every `verify_against_rubric`
  - final report write
- Include at minimum:
  - `schema_version`
  - `project_id`, `mode`, `paper_id`, `arxiv_id`
  - git HEAD, dirty flag, diff hash
  - provider/root/subagent model labels
  - sandbox backend and backend config summary
  - command list and timeouts
  - image/env id, hardware/GPU, dataset/model identifiers
  - metrics path/hash, logs path/hash, artifact path/hash
  - cost ledger row references
  - scorer output path/hash
- Add `manifest_path` and `attempt_id` to each `run_experiment` result dict.
- Final report builder must reconcile `final_report.rubric` with the latest
  manifest score record.

Tests:

- `tests/rlm/test_run_manifest.py`
- `tests/rlm/test_run_experiment_manifest_wiring.py`
- `tests/rlm/test_final_report_manifest_reconciliation.py`

Acceptance:

- A single test run creates `run_manifest.json`.
- Deleting or corrupting `metrics.json` after run completion causes the manifest
  validation test to fail.
- Final report references manifest IDs, not just free-form primitive traces.

Rollback:

- Manifest writing is additive. If broken, final scoring still works, but CI fails
  until fixed.

### Phase 3 - HF Intern read-only tool provider

Goal: import the useful HF discovery tools without importing an unsafe agent loop.

Implementation:

- Add optional integration package:
  - `backend/integrations/hf_intern/__init__.py`
  - `backend/integrations/hf_intern/tools.py`
  - `backend/integrations/hf_intern/schemas.py`
- Expose only read-only tools first:
  - `hf_inspect_dataset`
  - `hf_search_papers`
  - `hf_fetch_paper_metadata`
  - `hf_search_docs`
  - `hf_fetch_docs`
  - optionally `hf_repo_file_read`
- Do not expose:
  - local `bash`
  - local `write` / `edit`
  - sandbox create/delete
  - HF Jobs submit/cancel
  - dataset/repo upload/delete
  - trace upload
- Add `REPROLAB_HF_INTERN_TOOLS=1` default-off.
- Merge tools into RLM `build_custom_tools(ctx)` only when enabled and dependency
  checks pass.
- For RDR/Claude subagents, expose through `AgentRuntimeSpec.tools` only after
  Phase 1 contract tests pass.

Tests:

- unit tests with mocked HF APIs
- no-token behavior: tools return clear unavailable results, not crashes
- read-only guarantee: no write/delete/job tool appears in tool registry
- prompt injection guard: tool output is treated as evidence candidates, not
  benchmark facts

Acceptance:

- `hf_inspect_dataset` can validate schema/splits/sample rows for a public dataset
  in a mocked test.
- Enabling HF tools changes only the tool registry and prompt guidance, not scoring.

Rollback:

- Set `REPROLAB_HF_INTERN_TOOLS=0`.

### Phase 4 - HF Jobs execution backend under `run_experiment`

Goal: add HF Jobs as a compute backend without letting an agent own execution.

Implementation:

- Add `backend/services/runtime/hf_jobs_backend.py`.
- Implement the existing `RuntimeBackend` interface when feasible. If HF Jobs does
  not map cleanly to long-lived sandbox semantics, add a narrower
  `ExperimentBackend` adapter under `run_experiment`, but keep the public primitive
  unchanged.
- Add sandbox mode / backend enum value `hf-jobs`.
- Map `run_experiment(code_path, env_id)` to:
  - package code and command files
  - submit a single HF Job with explicit timeout and hardware
  - stream logs into the run directory
  - fetch artifacts back to `runs/<id>/artifacts/`
  - require `metrics.json`
  - cancel on budget/wall-clock/deadline
  - record HF job id in `run_manifest.json`
- Use HF Intern's cost estimation as inspiration, but centralize policy in
  OpenResearch `RunBudget`.
- Scheduled jobs are out of scope.

Tests:

- mocked `HfApi.run_job`
- timeout/cancel path
- missing metrics path
- log streaming path
- manifest field coverage
- budget cap refusal before submit

Acceptance:

- `run_experiment` behavior is the same shape for local, docker, runpod, and
  hf-jobs.
- HF job ids are visible in manifest and final diagnostics.
- A failed HF job is repairable or fatal according to the existing failure taxonomy.

Rollback:

- `--sandbox hf-jobs` only; no default change until at least three paired benchmark
  runs match local/runpod behavior within expected variance.

### Phase 5 - Scorer evidence hardening

Goal: make rubric scores auditable and adversarial.

Implementation:

- Extend scorer output with `evidence_refs` per leaf:
  - file path
  - line range if available
  - metric JSON path
  - log excerpt hash or bounded excerpt
  - manifest attempt id
- Persist `runs/<id>/rubric_breakdown.json` if not already fully covered by
  `rubric_evaluation.json`.
- Add a scorer post-pass:
  - leaves with score above threshold but no evidence refs are downgraded or
    marked `unsupported_evidence`
  - final report lists unsupported or weakly supported claims
- Add optional second-judge consistency only for borderline leaves, not all leaves.

Tests:

- claim with metric evidence passes
- claim with narration only fails or is capped
- borderline leaf invokes consistency check when enabled
- final report includes unsupported evidence summary

Acceptance:

- Every nonzero leaf score has at least one evidence reference.
- Missing evidence is visible in UI/events and final JSON.

Rollback:

- Env flag `REPROLAB_STRICT_EVIDENCE_REFS=0` for emergency comparison runs. Default
  strict after regression suite passes.

### Phase 6 - Harness regression suite

Goal: make harness quality measurable as a first-class product.

Implementation:

- Add `tests/harness_contract/`.
- Add small fixtures under `tests/fixtures/harness_contract/`.
- Add marker in `pyproject.toml`: `harness_contract`.
- Define named scenarios:
  - `minimal_success`
  - `experiment_failure_then_repair`
  - `bad_metrics_shape`
  - `metricless_success_is_degraded`
  - `provider_tool_contract_parity`
  - `interrupted_resume`
  - `orphan_remote_cleanup`
  - `scorer_rejects_unsupported_claim`

Acceptance:

- `python -m pytest -m harness_contract` completes locally without paid compute.
- Paid/network tests are marked separately and skipped by default.
- CI can run the cheap subset on every PR.

Rollback:

- None. Tests can be xfailed only with issue references and expiration dates.

### Phase 7 - RDR scheduler and artifact merge hardening

Goal: improve the deterministic harness spine before adopting any generic
multi-agent scheduler.

Implementation:

- Add cluster dependency graph metadata to `WorkCluster`.
- Add cluster confidence and artifact completeness scoring.
- Add automatic requeue rules:
  - failed cluster
  - low evidence coverage
  - missing command/artifact
  - contradiction with another cluster
- Add merge conflict detector for generated artifacts.
- Add `rdr_cluster_manifest.json` and per-cluster artifact hashes.
- Keep control flow deterministic; no LLM decides scheduling policy.

Tests:

- dependency ordering
- low-evidence requeue
- artifact conflict detection
- deterministic result for same cluster graph

Acceptance:

- RDR can explain why each cluster ran, skipped, retried, or repaired.
- RLM Phase 2 receives weak leaves plus artifact/evidence context, not a vague
  "please improve" prompt.

Rollback:

- Gate behind `REPROLAB_RDR_SCHEDULER_V2=1` until paired runs improve or match.

### Phase 8 - Claude Code operating envelope for implementation agents

Goal: make Claude Code effective without letting it bypass harness safety.

Implementation recommendations:

- Use Plan Mode for Phase 1-5 implementation planning.
- Use normal mode only after a specific phase plan is accepted.
- Use narrow prompts with file anchors and exact tests.
- Add project `.claude/settings.json` only after review; do not commit personal
  credentials or local settings.
- Recommended shared settings:
  - deny reading `.env`, `.env.*`, secrets, local run artifacts with private data
  - allow targeted test commands
  - ask before paid compute, network job submission, destructive git, or deletes
  - disable bypass permissions in managed/team settings where possible
- Use subagents:
  - `runtime-contract-reviewer`: read-only, checks provider parity
  - `security-adversary`: read-only, checks sandbox and data exfiltration risks
  - `test-author`: read/write tests only
  - `implementation-agent`: scoped to one phase
- Use hooks:
  - PreToolUse block writes to `.env`, `runs/**` fixtures unless explicitly allowed
  - PreToolUse block `git reset --hard`, broad deletes, direct `ml-intern` headless
  - Stop hook runs the phase's targeted tests and blocks success claims when tests fail

Acceptance:

- Claude Code implementation sessions can be resumed from the spec alone.
- Each phase prompt contains a deterministic verification command.
- No phase prompt asks the agent to inspect or use leaked Claude Code source.

## 6. Phase dependency graph

```text
Phase 0
  -> Phase 1
    -> Phase 3
    -> Phase 8
  -> Phase 2
    -> Phase 4
    -> Phase 5
  -> Phase 6
Phase 7 can start after Phase 2, but should not become default before Phase 6.
```

Critical path:

1. Phase 1: provider parity
2. Phase 2: manifest
3. Phase 6: regression suite
4. Phase 3/4: HF tools and HF Jobs
5. Phase 5/7: scorer and scheduler upgrades

## 7. Anti-patterns to reject during review

- "Just call `ml-intern` headless from RLM."
- "Let the model submit HF Jobs directly."
- "Use a generic multi-agent scheduler as the scorer."
- "Use Claude Code leaked source as implementation reference."
- "Treat trace viewer compatibility as proof of reproducibility."
- "Let local bash/write tools from external repos operate in the benchmark repo."
- "Patch the prompt only" when the invariant belongs in code.
- "Rely on final report prose" instead of manifest-linked evidence.

## 8. Implementation prompt for Claude Code

Use this prompt as the root handoff. Keep the phase number narrow.

```text
You are implementing the OpenResearch/ReproLab root harness hardening spec:
docs/superpowers/specs/2026-05-31-root-harness-hardening-and-hf-intern-integration.md

Work only on PHASE <N>: <phase name>.

Operating rules:
- Explore first in plan mode. Read the spec, CLAUDE.md, and the exact files listed
  for this phase before editing.
- Preserve the core invariant: agents are workers; run_experiment, manifests,
  budgets, and verify_against_rubric are the source of truth.
- Do not inspect, clone, use, or copy leaked Claude Code source. Use only official
  Claude Code / Agent SDK docs and installed package signatures.
- Do not call ml-intern headless. If integrating HF Intern, import or reimplement
  selected safe handlers behind OpenResearch contracts only.
- Keep changes phase-scoped. No unrelated refactors.
- Add tests before or with the implementation.
- Run the exact verification commands listed below and fix failures.
- If an SDK signature differs from the spec, add a small compatibility helper and
  test both paths with fakes.
- Do not widen tool permissions. If a provider cannot enforce the tool contract,
  fail closed with a clear ProviderFeatureUnsupported/ProviderConfigurationError.

Phase files to inspect:
- <list from the phase>

Required tests:
- <phase-specific tests>

Verification command:
- <command>

Deliverable:
- Code changes
- Updated/added tests
- Short summary of behavior changes
- Any remaining risk or skipped paid/network tests
```

## 9. Phase-specific Claude Code prompts

### Prompt for Phase 1

```text
Implement Phase 1: Provider runtime contract hardening.

Read:
- backend/agents/runtime/base.py
- backend/agents/runtime/claude_runtime.py
- backend/agents/runtime/openai_runtime.py
- backend/agents/registry.py
- tests/test_agent_runtime_claude_adapter.py
- tests/test_agent_runtime_openai_adapter.py
- CLAUDE.md section "claude-agent-sdk isolation"

Tasks:
1. Add runtime contract snapshot helpers.
2. Ensure Claude root agents receive explicit allowed tools matching
   AgentRuntimeSpec.tools.
3. Ensure Claude SDK calls are isolated from local/user settings where the installed
   SDK supports it: setting_sources=[], explicit mcp_servers, non-plan
   permission_mode.
4. Preserve existing MCP subagent extension behavior.
5. Add parity tests and SDK-isolation tests with fake SDK classes.

Verification:
python -m pytest tests/test_agent_runtime_claude_adapter.py tests/test_agent_runtime_openai_adapter.py tests/test_agent_runtime_factory.py
```

### Prompt for Phase 2

```text
Implement Phase 2: Canonical run manifest and provenance contract.

Read:
- backend/agents/rlm/context.py
- backend/agents/rlm/run.py
- backend/agents/rlm/primitives.py
- backend/agents/rlm/report.py
- backend/services/runtime/interface.py
- tests/rlm/test_run_experiment.py
- tests/rlm/test_final_report_populated_from_run.py

Tasks:
1. Add backend/agents/rlm/manifest.py with schema-versioned Pydantic models.
2. Atomically write runs/<project_id>/run_manifest.json.
3. Add attempt ids and manifest_path to run_experiment results.
4. Record metrics/log/artifact hashes.
5. Reconcile final report with the latest score/attempt manifest.

Verification:
python -m pytest tests/rlm/test_run_experiment.py tests/rlm/test_final_report_populated_from_run.py tests/rlm/test_run_artifact_contract.py tests/rlm/test_run_manifest.py
```

### Prompt for Phase 3

```text
Implement Phase 3: HF Intern read-only tool provider.

Read:
- backend/agents/rlm/binding.py
- backend/agents/rlm/primitives.py
- backend/agents/rlm/system_prompt.py
- C:/tmp/ml-intern/agent/tools/dataset_tools.py
- C:/tmp/ml-intern/agent/tools/research_tool.py
- C:/tmp/ml-intern/agent/core/tools.py

Tasks:
1. Add backend/integrations/hf_intern read-only tool wrappers.
2. Default-off gate: REPROLAB_HF_INTERN_TOOLS=1.
3. Expose only read-only tools. No local bash/write/edit, no uploads, no jobs.
4. Merge into RLM custom_tools only when enabled.
5. Add tests proving no unsafe tool names are exposed.

Verification:
python -m pytest tests/rlm/test_binding.py tests/rlm/test_system_prompt.py tests/integrations/test_hf_intern_tools.py
```

### Prompt for Phase 4

```text
Implement Phase 4: HF Jobs execution backend under run_experiment.

Read:
- backend/services/runtime/interface.py
- backend/services/runtime/service.py
- backend/services/runtime/runpod_backend.py
- backend/services/runtime/local_process.py
- backend/agents/rlm/primitives.py run_experiment path
- C:/tmp/ml-intern/agent/tools/jobs_tool.py
- C:/tmp/ml-intern/agent/core/cost_estimation.py

Tasks:
1. Add hf-jobs backend or narrower ExperimentBackend if RuntimeBackend semantics do
   not fit.
2. Keep run_experiment public contract unchanged.
3. Submit one explicit job, stream logs, fetch artifacts, require metrics.json.
4. Enforce RunBudget before submit and during polling.
5. Record job id and artifacts in run_manifest.json.
6. Keep scheduled jobs out of scope.

Verification:
python -m pytest tests/rlm/test_run_experiment.py tests/runtime/test_hf_jobs_backend.py tests/rlm/test_run_manifest.py
```

### Prompt for Phase 5

```text
Implement Phase 5: scorer evidence hardening.

Read:
- backend/agents/rlm/primitives.py verify_against_rubric
- backend/evals/paperbench/leaf_scorer.py
- backend/agents/rlm/report.py
- tests/evals/
- tests/rlm/test_verify_against_rubric.py

Tasks:
1. Add evidence_refs per nonzero leaf.
2. Persist rubric_breakdown.json or extend rubric_evaluation.json.
3. Cap or mark unsupported nonzero leaves.
4. Add final report unsupported-evidence summary.
5. Add optional borderline second-judge check behind a flag.

Verification:
python -m pytest tests/evals tests/rlm/test_verify_against_rubric.py tests/rlm/test_final_report_populated_from_run.py
```

## 10. Definition of done

The full program is done when:

- root and subagent tool contracts are provider-parity tested
- every experiment attempt has a manifest entry
- every final metric can be traced to an artifact hash
- HF read-only tools are useful but cannot mutate state
- HF Jobs can run only through `run_experiment`
- scorer nonzero leaves carry evidence references
- `python -m pytest -m harness_contract` is a stable cheap gate
- one paid/network validation run demonstrates hf-jobs or runpod behavior without
  changing the public report contract

## 11. Principal-engineer review checklist

Before merging any phase:

- Is the invariant enforced in code, not just prompt text?
- Does the test fail before the fix?
- Does the phase widen tool/network/write permissions?
- Is rollback a flag or mode, not a revert?
- Can a future reviewer reconstruct what happened from files in `runs/<id>/`?
- Does the change improve benchmark trustworthiness, not just agent convenience?
- Is the integration behind an OpenResearch contract?
- Are paid compute and trace upload impossible by accident?

If the answer to any of these is no, do not merge.

