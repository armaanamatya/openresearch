# GitHub-repo-first reproduction — design

> **Doc status:** Draft · spec tier · authored 2026-06-21.
> Brainstormed interactively; grounded by five read-only recon passes over the
> live code (citations are as-of-2026-06-21 and must be re-confirmed at
> implementation time). Policy: [`docs/policies/documentation.md`](../../policies/documentation.md).

## 1. Summary

When a paper links an official code repository, OpenResearch should **discover that
repo, clone it, and use it as the reproduction starting point** — preferring the
authors' code over a from-scratch reimplementation "whenever possible" — and then
**measure two distinct things**:

1. **Execution** — *did the authors' code run?* (it executed end-to-end and emitted
   real, provenanced metrics), and
2. **Replication** — *did it reproduce the paper?* (those metrics match the paper's
   claimed numbers).

This is not a greenfield feature. The codebase was **scaffolded for exactly this and
left unfinished** (GitHub issue #62). The work is: connect a dead wire, finish the
stubbed context slot, add one deterministic clone step, add one execution signal, and
add a repo-URL input — all behind a default-OFF flag so the system is byte-identical
when the flag is unset.

### Reasoning-style reference
The motivating artifact (`image.webp`) is an agent trace that orients on a project,
reads the paper, recognizes the linked `https://github.com/ZJU-REAL/SDAR` repo, and
"clones the source repo to inspect." We adopt the **behavior** (autonomous repo
discovery → clone → use, narrated transparently in the SSE trace) but **not** the
surrounding vocabulary in that trace (an `orx` skill, "experiment tree," "baseline
branch," "machine-readable report"). Those are explicitly **out of scope** (§11).

## 2. Current state (grounded)

| Capability | State today | Citation (as-of recon) |
|---|---|---|
| Extract GitHub links from a paper | **Works** — regex adapter emits `DiscoveredArtifact(kind="repository", locator="github:owner/repo")` | `backend/services/ingestion/discovery/adapters/regex.py:86` |
| Discovery → reproduction path | **DEAD END** — stored as `ArtifactDiscovered` events, materialized into a `discovered_artifacts` workspace var that the RLM pipeline never reads | `discovery/service.py:104`, `context/workspace/service.py:481`, **not** referenced in `backend/agents/rlm/` |
| Repo content in root context | **Reserved but empty** — `repo_files = None  # Not yet populated (#62)` | `backend/agents/rlm/run.py` `_build_context()` (~`:574`, dict ~`:606-618`) |
| `artifact_index` carrying repo metadata | **Always `{}`** in real runs; no construction site; not in system prompt | `primitives.py:1979` (`plan.get("artifact_index")`), `baseline_implementation.py:2732` (`artifact_index or {}`) |
| "Adapt Existing Repository" (Mode 1) | **Aspirational prompt text**, references a non-existent "Artifact Discovery Agent"; **no actual `git clone`** anywhere in the core pipeline | `backend/agents/prompts/baseline_implementation.py:14-26` |
| Generated code location | `runs/<id>/code/` (canonical project dir consumed by cell-runner, env_pin, evidence gate) | `primitives.py:2298` |
| `detect_environment` | Deterministic; infers env from `paper_claim_map`; does **not** read any repo | `primitives.py:1017` |
| Per-paper blacklist | `blocked_resources` in `paper_hints.py` + CLI `--blacklist` | `paper_hints.py:145-151` |
| Repo-URL input (UI / API / CLI) | **None** — run-start takes arXiv/PDF/preset only | `app.py` (`/runs`, `/runs/upload`, `/runs/arxiv`), `live_runs.py:175-214`, `cli.py`, `frontend/src/components/lab/upload-view.tsx:280-315` |
| Two-axis verdict | `implementation_verdict` (FIDELITY) + `replication_verdict`; **no execution axis** | `reproducibility_verdict.py:78-79`, `two_axis_report.py:249-271` |
| Execution ("it ran") signal | Implicit only — `experiment_runs.jsonl` `success=true`+metrics + evidence gate; **no first-class field** | `report.py:1344` (`_has_experiment_evidence`) |
| Multi-cloud code transport | Azure/GCP ship **only `code/`** to the GPU Job (blob prefix `runs/<id>/code/`); a sibling `repo/` is not traversed. RunPod naive-rglobs `project_root` (a `repo/` under it would ride — harmless, capped) | `k8s_job_backend.py:815-832`, `azure_blob.py:49`, `gcs_blob.py:55`, `runpod_backend.py:843-857` |
| Orchestrator git + egress | Orchestrator image installs `git` + TLS certs, runs in-cluster | `docker/orchestrator/Dockerfile:50-56` |

**The gap, in one sentence:** discovery produces a repo URL that nothing consumes;
there is no clone; there is no execution field; there is no repo-URL input. Every one
of these converges on a single function — `_build_context()` — for resolve+clone+expose.

## 3. Goals / non-goals

**Goals**
- Prefer the authors' repo when one is available (discovered or user-provided).
- Clone deterministically, robustly, and seamlessly across **all** sandboxes
  (`local`, `docker`, `runpod`, Azure AKS, GCP GKE) and the GPU execution path.
- Adapt the repo into the runnable `code/`, keeping a pristine `repo/` for provenance.
- Measure and report **execution** ("ran") and **replication** ("reproduced") as two
  honest, separately-stamped axes, with repo provenance (URL + commit SHA).
- Byte-identical behavior when the master flag is unset.

**Non-goals (this spec)**
- The `orx` model (experiment tree, baseline branch, machine-readable report).
- Private-repo authentication / secret handling (public repos only, v1).
- Papers-with-Code (or any external) lookup beyond the paper's own text.
- git-LFS weight pulls (weights continue to come from the agent's existing HF path).
- Any change to the GPU execution / cell-runner / capacity machinery.

## 4. Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Scope | Narrow — finish #62 inside the existing flow; reuse discovery, `repo_files`, `artifact_index`, Mode 1, two-axis. |
| Integrity / scoring | **Mode-selected + stamped.** Default *adapt*; clean-room via blacklist (hard-exclude → scratch) or `REPRODUCTION_MODE=reference` (read-not-run); provenance stamped. |
| Measurement | Two measured axes: **execution** (new field) + **replication** (existing verdict), scope-aware. |
| Run signal | **Adapt-then-measure** — one run on the minimally-adapted code; record the `repo/ ↔ code/` adaptation delta for honesty. |
| Trigger | **Hybrid** — deterministic harness clone at run setup + a thin `inspect_repository` primitive (the 18th) for agent-driven deep-reads, narrated in the SSE trace. |
| Layout | Clone pristine into `runs/<id>/repo/`; adapt-mode seeds `code/` *from* `repo/`. |
| Repo source | **Both** — user-provided URL wins over auto-discovered. |
| Auth | Public repos only (v1); clean failure → from-scratch fallback. |
| Rollout | Flag-gated **default-OFF**; validate on SDAR (the repo in `image.webp`); then flip default-ON. |

## 5. Design

### 5.1 Data flow

```
ingest ─▶ RepoResolver(user_url, discovered_artifacts, blacklist, mode)
            └─▶ RepoSpec{url|None, source:user|discovered, mode:adapt|reference|none, reason}
        ─▶ RepoProvisioner.clone(RepoSpec) on the ORCHESTRATOR HOST
            └─▶ runs/<id>/repo/ (pristine, commit SHA pinned) + RepoManifest
        ─▶ _build_context(): repo_files = manifest   (root sees it from iteration 1)
                              persist RepoSpec → rlm_state/repo_spec.json
detect_environment ─▶ merges repo/'s real deps (requirements/setup/pyproject/env.yml)
implement_baseline ─▶ adapt-mode: seed code/⟵repo/ (once, code/ empty), sub-agent adapts
                       reference-mode: sub-agent reads repo/, writes code/ from scratch
                       (artifact_index repo metadata injected deterministically here)
run_experiment ─▶ existing cell-runner / GPU path — only code/ ships to the backend
finalize ─▶ final_report.reproduction{mode, repo_url, commit_sha, execution{ran,...},
                                       adaptation{files_changed}}  (cheap, always on a repo run)
            reproducibility.replication_verdict  (existing two-axis, under its own flag)
```

### 5.2 New module: `backend/services/ingestion/repo/`

A single, focused package. Three units, each independently testable.

**`resolver.py` — `RepoResolver` (pure, no IO).**
`resolve(user_url: str | None, discovered: list[DiscoveredArtifact], blacklist: set[str], mode_override: str | None) -> RepoSpec`
- Priority: `user_url` > highest-confidence `discovered` repository artifact > none.
- Normalizes `github:owner/repo` and full URLs to a canonical `https://github.com/owner/repo`.
- Blacklist: a resolved URL on the blacklist is **dropped** (treated as not-found),
  preserving the existing "blocked = do not use" semantics; if no other repo resolves the
  run proceeds **scratch**. Blacklisting a paper's official repo is therefore the
  hard-exclude clean-room path.
- `mode` ∈ {`adapt`, `reference`}, default `adapt`. `OPENRESEARCH_REPRODUCTION_MODE=reference`
  forces global clean-room (clone + read the repo, but reimplement in `code/` and do **not**
  run the authors' code) for benchmark-faithful runs.
- Returns `RepoSpec{url|None, source, mode, reason}` — `reason` is a human string for the SSE/event.

**`provisioner.py` — `RepoProvisioner` (IO; sandbox-agnostic — always runs on the orchestrator host).**
`clone(spec: RepoSpec, dest: Path) -> RepoManifest`
- `git clone --depth 1 --no-tags` (shallow); `GIT_LFS_SKIP_SMUDGE=1` unless
  `OPENRESEARCH_REPO_CLONE_LFS` is set; `--config` to disable interactive prompts.
- Caps: `OPENRESEARCH_REPO_CLONE_TIMEOUT_S` (default 300), `OPENRESEARCH_REPO_CLONE_MAX_MB`
  (default 2048 — post-clone size check; over → discard + fail-soft).
- Records `commit_sha` (`git rev-parse HEAD`).
- **Fail-soft**: any failure (network/egress blocked, private/auth, 404, oversize,
  timeout) → return `None`, emit a `repo_clone_failed` `run_warning`, and the run
  proceeds from-scratch (Mode 2). A blocked clone never aborts a run.

**`manifest.py` — `build_manifest(repo_dir) -> RepoManifest` (pure).**
- `RepoManifest{path, commit_sha, file_tree (capped: ≤~200 files, depth ≤ 4),
  key_files (README*, requirements*.txt, setup.py, pyproject.toml, environment*.yml,
  detected train/main entrypoints — names + short excerpts), size_mb, lfs_skipped}`.
- **Constant-size guarantee:** the manifest, not the raw tree, is what enters the root's
  context (RLM Algorithm 1 invariant). The root navigates deeper via `inspect_repository`
  / `rlm_query`. Hard ceiling on manifest bytes mirrors the context-map cap pattern.

### 5.3 Wiring into `_build_context()` (`run.py:~574`) — the keystone

This single function gains (all guarded by the master flag — §5.10):
1. Load `discovered_artifacts` (the currently-dead var) via the discovery service.
2. `spec = RepoResolver.resolve(user_repo_url, discovered, blacklist, mode_override)`.
3. If `spec.url`: `manifest = RepoProvisioner.clone(spec, runs/<id>/repo/)`.
4. `context["repo_files"] = manifest.as_context()` (replaces the `None` stub — finishes #62).
5. Persist `spec` + `manifest.commit_sha` → `rlm_state/repo_spec.json` (the deterministic
   source of truth that `implement_baseline` and the report writer read — **not** the
   root-assembled plan, which is untrusted).

This simultaneously **connects the dead discovery wire** and **finishes the #62 slot** in
the place both already belong.

### 5.4 Use: adapt vs reference

**`implement_baseline` (`primitives.py:1979` / `baseline_implementation.py`).**
- Read the resolved `repo_spec.json` (deterministic; ignore whatever the root put in
  `plan["artifact_index"]`, then *merge* the trusted repo metadata into `artifact_index`
  so the sub-agent context carries `{repo_url, commit_sha, path, mode}`).
- **adapt-mode**, first call only (code/ empty): deterministically copy `repo/` → `code/`
  (tree copy excluding `.git/`), then run the sub-agent with a rewritten Mode-1 prompt:
  *"The authors' reference implementation is already in your working directory; adapt it
  to run in this environment and at this scope — do not rewrite from scratch."* (Removes
  the dead "Artifact Discovery Agent" reference and the unreliable "clone or copy"
  instruction — the harness already did the clone+seed.)
- **reference-mode**: leave `code/` empty; the sub-agent reads `repo/` (read-only) for
  exact details and writes its own `code/` (Mode 2 + a "reference available at repo/" note).
- **Re-entrant calls (repairs)**: never re-seed — `code/` already holds the adapted work.

**`detect_environment` (`primitives.py:1017`).** When `repo/` exists (flag on), merge the
repo's declared dependencies (`requirements*.txt`, `setup.py`/`pyproject.toml`,
`environment*.yml`) into the inferred `EnvironmentSpec`, repo-declared deps taking
priority as ground truth. Accuracy win over inferring from prose, and it strengthens the
dynamic-GPU SKU resolution (repo may pin torch/CUDA). Byte-identical when no `repo/`.

### 5.5 New primitive: `inspect_repository` (the 18th)

Thin read tool over `runs/<id>/repo/` on the orchestrator host:
- `inspect_repository(path="", grep=None, max_bytes=...)` → bounded file contents /
  subtree listing; or `inspect_repository(reclone_url=...)` to re-point to a different
  repo (root override → re-resolve + re-clone).
- Flag OFF → returns `{"status": "disabled"}` (mirrors `read_context_map`'s no-op pattern
  so the registry count is stable and off-state is inert).
- Emits a `primitive_call` + the narration events in §5.7.
- **Registry contract:** bump `PRIMITIVE_REGISTRY` to 18 and update
  `tests/rlm/test_registry.py::EXPECTED` (CLAUDE.md invariant).

### 5.6 Measurement: execution + replication + provenance

Per the grounded two-axis findings, **execution is not a first-class field** — we add it.

**Always (cheap, deterministic) when a repo run finalizes** — attach at the report writer
(`report.py::write_final_report_rlm`, the single chokepoint feeding all three finalize
paths), independent of the `OPENRESEARCH_TWO_AXIS_VERDICT` gate:

```jsonc
final_report.reproduction = {
  "mode": "adapt" | "reference" | "scratch",
  "repo_url": "https://github.com/ZJU-REAL/SDAR",
  "commit_sha": "abc1234…",
  "provider": "github",
  "execution": {                 // ← the new "did it run" axis
    "ran": true,                 // = _has_experiment_evidence(): success=true + non-empty metrics
    "status": "success" | "partial" | "failed",
    "metrics_produced": true
  },
  "adaptation": { "files_changed": 7, "files_added": 1, "files_removed": 0 }  // repo/ ↔ code/ diff
}
```
`execution.ran` is sourced from the **evidence layer** (`_has_experiment_evidence`,
`report.py:1344`) — so it is evidence-gated and cannot be forged by a green-looking report.

**Replication ("reproduced the paper")** reuses the existing machinery unchanged:
`reproducibility.replication_verdict` (`two_axis_report.py:295` attach), computed by the
repro-spec extractor + verdict engine against the paper's claimed effect ± margin from
`rlm_state/repro_spec.json`. It remains gated by `OPENRESEARCH_TWO_AXIS_VERDICT`. The
existing fidelity axis (`implementation_verdict`) rides along as a bonus.

**Operator guidance (documented, not enforced):** for the full "ran + reproduced" picture
on a repo run, enable both `OPENRESEARCH_USE_AUTHOR_REPO=1` and
`OPENRESEARCH_TWO_AXIS_VERDICT=1`. The "ran" axis is available with only the former.

### 5.7 Inputs (repo URL) + SSE narration

- **API:** optional `repo_url: str | None` on `StartRunRequest` + `StartArxivRunRequest`
  (`live_runs.py:175-214`) and the `/runs/upload` form (`app.py`).
- **CLI:** `--repo-url <url>` (`cli.py`), threaded into the run context.
- **UI:** optional "Official code repository (optional)" text input on the
  paper-understanding screen (`upload-view.tsx`). **Separable phase** — backend works via
  CLI/API without it.
- **SSE events** (for the narrated trace from `image.webp`): `repo_resolved`
  (url + source + mode + reason) and `repo_cloned` (commit_sha + size + key_files), routed
  through `sse_bridge` (corpus-free control events). System-prompt guidance tells the root
  to consult `repo_files` and narrate repo discovery/clone/inspection (instruction omitted
  when the flag is off).

### 5.8 Multi-cloud + GPU (the seamless part)

The provisioner runs **only on the orchestrator host**, which already has `git` + TLS
certs and runs in-cluster for cloud deployments. `repo/` is an **orchestrator-host-only**
artifact (manifest + provenance/diff); in adapt-mode the runnable `code/` is seeded from
it. Therefore **only `code/` ever crosses to a GPU backend — exactly as today**:

| Backend | What ships | `repo/` behavior | Change needed |
|---|---|---|---|
| local / docker | code/ runs in place | host-local | none |
| Azure AKS / GCP GKE | only `code/` → blob prefix `runs/<id>/code/` | sibling, not traversed → excluded by architecture | **none on the blob path** |
| RunPod | naive rglob of `project_root` | could ride (harmless, capped) | add `repo` to the transport exclusion set (tidy; keeps it host-only) |

Concretely: add `"repo"` to the shared blob exclusion `_EXCLUDED_DIR_PARTS`
(`azure_blob.py:49`, `gcs_blob.py:55`) and to the RunPod uploader walk
(`runpod_backend.py:843-857`) so `repo/` is host-only on every backend. **The GPU
execution / cell-runner / capacity path is untouched** — code that came from a repo is
indistinguishable from agent-written code to the execution layer.

**Infra precondition:** the orchestrator pod needs outbound egress to `github.com`. If a
cluster egress policy blocks it, the clone fails-soft → from-scratch (a loud
`repo_clone_failed` warning, never a crash). Documented in the operator notes.

### 5.9 Modes summary

| Mode | When | `code/` | `repo/` | Scored as |
|---|---|---|---|---|
| **adapt** (default) | repo resolved, not blacklisted | seeded from `repo/`, then adapted | pristine reference | execution + (replication under two-axis) |
| **reference** | `OPENRESEARCH_REPRODUCTION_MODE=reference` (global clean-room) | agent's from-scratch impl | read-only reference | normal from-scratch + provenance note |
| **scratch** | no repo / official repo blacklisted / clone failed / flag off | agent's from-scratch impl | — | unchanged (today's behavior) |

### 5.10 Flags / config (all default-OFF; unset ⇒ byte-identical)

| Flag | Default | Effect |
|---|---|---|
| `OPENRESEARCH_USE_AUTHOR_REPO` | **off** | Master. Off ⇒ no resolve/clone, `repo_files` stays `None`, `inspect_repository` returns `disabled`, `detect_environment`/`implement_baseline` unchanged, no `reproduction` stamp. |
| `OPENRESEARCH_REPRODUCTION_MODE` | `adapt` | `reference` forces clean-room globally (benchmark runs). |
| `OPENRESEARCH_REPO_CLONE_TIMEOUT_S` | `300` | Clone wall-clock cap. |
| `OPENRESEARCH_REPO_CLONE_MAX_MB` | `2048` | Post-clone size cap; over → fail-soft. |
| `OPENRESEARCH_REPO_CLONE_LFS` | `off` | Off ⇒ `GIT_LFS_SKIP_SMUDGE=1`. |

`config.py` gains these fields; `OPENRESEARCH_*` canonical (the `REPROLAB_*` bridge applies).

## 6. Robustness & failure modes

- **Clone fails** (egress/auth/404/oversize/timeout) → fail-soft to from-scratch + warning.
- **Empty/garbage repo** (no recognizable entrypoint) → manifest still built; adapt-mode
  seeds `code/`; the sub-agent + existing repair loop handle it; evidence gate is the floor.
- **Huge repo** → shallow + size cap; LFS skipped → weights via the agent's HF path as today.
- **Security:** cloned code only ever executes inside the sandbox that already runs all
  experiment code — **no new trust surface**. `repo/` on the orchestrator is read-only
  reference (never executed there).
- **Determinism:** commit SHA pinned at clone and stamped in the report.

## 7. Testing plan

- **Unit (pure):** `RepoResolver` priority/blacklist/normalization; `manifest` cap/shape.
- **Integration (local git fixture, no network):** `RepoProvisioner` clone+manifest;
  fail-soft on bad URL; size/timeout caps.
- **Wiring:** `_build_context` populates `repo_files` + `repo_spec.json`; `implement_baseline`
  seeds `code/` from `repo/` once and injects `artifact_index`; `detect_environment` merges
  repo deps; report attaches `reproduction.execution` from a synthetic `experiment_runs.jsonl`.
- **Off-state regression:** master flag off ⇒ existing forced-iteration / report / role-model
  suites unchanged; `repo_files is None`; registry count test (`tests/rlm/test_registry.py`).
- **Registry:** primitive count 17 → 18.
- **Multi-cloud (cheap):** assert `repo/` excluded from the blob upload set and the RunPod walk.
- **E2E (operator-run, GPU):** SDAR adapt run on a GPU backend → `reproduction.execution.ran=true`
  + `replication_verdict` populated (validation gate before the default-ON flip).

## 8. File touch-list

**New** — `backend/services/ingestion/repo/{resolver,provisioner,manifest}.py` + tests
under `tests/services/ingestion/repo/`.

**Changed**
- `backend/agents/rlm/run.py` — `_build_context()`: resolve+clone+expose; thread `repo_url`.
- `backend/agents/rlm/primitives.py` — `detect_environment` (merge repo deps),
  `implement_baseline` (seed `code/`, inject `artifact_index`), register `inspect_repository`.
- `backend/agents/baseline_implementation.py` + `backend/agents/prompts/baseline_implementation.py`
  — rewrite Mode 1 (code already seeded; drop the dead "Artifact Discovery Agent").
- `backend/agents/rlm/report.py` (+ `two_axis_report.py` only if `execution` is nested under
  `reproducibility`) — attach `final_report.reproduction` (mode/url/sha/execution/adaptation).
- `backend/app.py`, `backend/services/events/live_runs.py` — `repo_url` on the run-start models.
- `backend/cli.py` — `--repo-url`.
- `backend/agents/rlm/sse_bridge.py` — `repo_resolved` / `repo_cloned` events.
- `backend/agents/rlm/system_prompt.py` — repo-aware guidance (flag-gated).
- `backend/services/runtime/{azure_blob.py,gcs_blob.py,runpod_backend.py}` — exclude `repo/`.
- `backend/config.py` — new flags.
- `frontend/src/components/lab/upload-view.tsx` (+ proxy/types) — optional repo-URL field.
- **Docs:** `CLAUDE.md` (primitive count 17→18, new SSE events, new flags, feature-flag
  block) + `system_overview.md` (data-flow drift) per the doc-update contract.

## 9. Rollout

Ship default-OFF. Validate on **SDAR** (`github.com/ZJU-REAL/SDAR`) on a GPU backend:
confirm `execution.ran=true`, a populated `replication_verdict`, and a sane adaptation
delta. Then flip `OPENRESEARCH_USE_AUTHOR_REPO` default-ON (per the ≥3-paired-run
discipline used elsewhere in the repo). **Operator may overrule and ship default-ON now**
if they accept the unvalidated risk — this is the one decision left open to the operator.

## 10. Open items to confirm at implementation time

1. Exact `_build_context()` line + the discovery-service handle available there
   (`run.py:~574`).
2. Whether `execution`/`reproduction` is cleanest as a top-level `final_report` key or
   nested under the existing `reproducibility` block (prefer top-level so it survives with
   the two-axis flag off; confirm the leaderboard/UI read path).
3. The precise RunPod `project_root` definition (`runs/<id>/` vs `runs/<id>/code/`) — only
   affects whether the RunPod exclusion is a real change or a no-op safety net.

## 11. Out of scope (explicit)

The `orx` model (experiment tree, baseline node/branch, machine-readable report);
private-repo auth/secrets; Papers-with-Code; git-LFS weight pulls; any GPU-execution /
cell-runner / capacity change.

## 12. Related

- Issue #62 (the `repo_files` slot this finishes).
- Two-axis verdict: `two_axis_report.py`, `reproducibility_verdict.py`,
  `repro_spec_extractor.py`; `OPENRESEARCH_TWO_AXIS_VERDICT`.
- Discovery: `backend/services/ingestion/discovery/`.
- Blacklist: `paper_hints.py`, `--blacklist`.
- Baseline test paper: `docs/runbooks/2026-05-23-sdar-baseline-handoff.md`.
