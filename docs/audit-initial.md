# OpenResearch — Phase 0–2 reproducibility audit (initial report)

> **Doc status:** dated audit snapshot (2026-06-09) · branch `bes` @ `1b468a5` · method:
> 11-agent parallel audit (repo map, branch/history, env vars, README/commands, backend
> tests, frontend checks, Docker/infra, CI, security/secrets, hygiene, ops capabilities) +
> orchestrator-run real `docker build`. Every finding below cites verified evidence
> (file:line or command output). Remediation tracked in
> `docs/audits/2026-06-09-reproducibility-audit.md` (final report, written after fixes).
> Prior related audits: `docs/audits/2026-06-07-bes-doc-alignment-audit.md` (REMEDIATED —
> doc↔code drift; not re-litigated here), `docs/audits/2026-05-31-backend-core-opportunity-backlog.md`.

---

## 1. Repo map (verified)

Two-process app. **FastAPI backend** (`backend/app.py::create_app`, factory pattern, :8000)
+ **Next.js 16 frontend** (:3000 dev / `$PORT` in docker) that reaches the backend
**server-side only** via proxy routes under `frontend/src/app/api/*` (fallback
`http://127.0.0.1:8000`; no CORS layer; `frontend/src/proxy.ts` implements the demo-gate
cookie wall). Reproduction runs are long-lived subprocesses writing file-backed state to
`runs/<id>/` (`demo_status.json`, `rlm_state/`, `dashboard_events.jsonl`,
`final_report.{json,md}`, `cost_ledger.jsonl`, `experiment_runs.jsonl`, `code/`) — layout
verified against a real run dir. SQLite event store: default `sqlite:///openresearch.db`
with a deliberate legacy fallback to `reprolab.db` (`backend/config.py:336-338`; only the
legacy file exists locally, 12.5 MB, ignored/untracked).

| Path | What it is |
|---|---|
| `backend/` | FastAPI app, CLI (`python -m backend.cli`), RLM/RDR agents, sandbox backends, event store. `backend/requirements.txt` is the dependency source of truth (root `requirements.txt` is a 1-line forwarder; `pyproject.toml` deps are stale — missing `rlms`, `python-dotenv`, `Pillow`, `python-multipart`). |
| `frontend/` | Next.js 16 (engines `>=20.19 <21 \|\| >=22.12`), vitest (291 tests), Playwright e2e (7 specs, self-hosting on :3001). |
| `tests/` | 3,616 pytest tests / 343 files; collection clean. |
| `scripts/` | 24 operational scripts (batch/paperbench harnesses, runpod preflight, docs checker, dev launcher). `scripts/loops/` is **missing** despite being documented (see §4). |
| `tools/` | 5 dev helpers; `tools/seed-fake-run.sh` is a divergent legacy duplicate of `scripts/seed-fake-run.sh` (pre-RLM run shape). |
| `docs/` | Policy-governed (`docs/policies/documentation.md` + `scripts/docs_freshness_check.py`, CI-enforced). Three docs escape the manifest (§6). |
| `runs/`, `best_runs/`, `data/`, `logs/`, `findings/` | Run outputs (gitignored w/ whitelist), vendored exemplar runs, runtime-mutated calibration seed, untracked diagnostics, **undocumented** tracked ad-hoc run logs. |
| `Dockerfile` + `docker/entrypoint.sh` + `docker-compose.yml` | 3-stage build; tini PID 1 running uvicorn + `next start`; compose mounts docker.sock + `./runs`. Railway deploys the same image (`railway.json`). **No Kubernetes manifests exist on this branch** (exhaustive `kind:` search: zero); AKS/Terraform/Helm IaC landed on `origin/main` (PR #99), not `bes`. |
| `.github/workflows/` | One workflow: `docs-freshness.yml`. **No test/lint/build CI of any kind.** |
| `Makefile` | `help` / `docs-check` / `test` only. |
| `HANDOFF.md`, `CHANGELOG.md` | Stale: HANDOFF describes branch `harden/root-harness` on a different machine (2026-05-31); CHANGELOG self-declares stale at 2026-05-23. |

Entry points: `backend/app.py::create_app` (uvicorn `--factory`), `backend/cli.py`
(ingest / inspect / regenerate-report / reproduce / paperbench), `start.sh` (backend-only
preflight launcher), `scripts/dev.sh` (unified dev), `docker/entrypoint.sh`. TODO/FIXME
debt: 4 genuine markers, all in `backend/agents/rlm/` (debt is tracked in specs/runbooks
instead — by policy).

## 2. Branch analysis

- `bes` vs **local** `main` (a05d60f): 83 ahead / 0 behind; +14,570/−2,256 over 83 files.
  Themes: ~30 SDAR execution fixes (torchrun/FSDP, dataset caps, forced-iteration), ~12
  accelerator/local-vLLM, ~6 docker-free local-GPU sandbox + batch harness, STAB-1..4
  stability fixes, 2 flag-gated features (PEEK-lite, MUSE-lite), docs/hygiene. Intrinsic
  risk vs main: **low** (features default-off, tests added).
- **Premise break:** during the audit a fetch moved `origin/main` to `f9b14af` — now
  **17 commits / 470 files / +90k lines ahead of bes** (PR #99: full Azure AKS GPU backend
  with Terraform+Helm; PR #100: execution-reliability redesign). Both lines edited the same
  hot files (`primitives.py`, `run.py`, `cli.py`, `accelerator.py`, `CLAUDE.md`). `bes` is
  **no longer a superset**; merge-back will conflict. Decision needed: merge origin/main
  into bes, or re-land bes-unique work on main. **Not done in this audit** (large,
  conflict-laden, product decision on trunk).
- **Orphaned valuable work** (in *neither* bes nor main):
  - `origin/feat/rlm-wedge-hardening` 0a0084b..b63e16a (5 commits, 38 files): the
    **evidence-gate forge-row fix** (closes the HIGH-severity integrity hole recorded as
    OPEN in project memory) + the **RunPod local-docker-build short-circuit** (fixes the
    exact "wasted local build under runpod" rough edge CLAUDE.md flags). Port to whichever
    branch becomes trunk.
  - `origin/pipeline-validation-mech-understanding` 178c08c: `scripts/loops/*` (documented
    by CLAUDE.md but missing here — restored as part of this remediation), 314d813
    (BUG-NEW-043 child-RLM traceback surfacing), 6182eac (PaperBench demo_status fix).
  - `origin/feat/gepa-integration`: complete GEPA prompt-optimization subsystem (product
    decision; ignore the 5 earlier gepa branches).
- Disposable: ~17 of 25 surveyed refs are fully contained in bes or main and safe to delete
  (watchdog-probe-recover, rlm-stability, feat/parallel-rdr-cluster-dispatch,
  feat/efficiency-tier1-2, abheek_recent_runs_panel, integrate/harden-into-sdar, …).

## 3. Reproduction failures (clean-machine walkthrough)

Verified by actually running installs/tests/builds. **B** = blocker, **H** = high.

1. **[B] `npm ci` fails on macOS arm64 — the documented dev platform.**
   `frontend/package.json:16` pins `@rolldown/binding-linux-x64-gnu` (linux/x64-only
   native binary) as a direct dependency; npm hard-fails `EBADPLATFORM`. Nothing depends on
   it. Same failure reproduced in the real `docker build` on Apple Silicon
   (`FROM node:20-bookworm-slim` → linux/arm64 → same error), so **Docker build is broken
   too**. The lockfile already carries every platform's binding as rolldown's own optional
   deps. Fix: remove the pin, refresh lockfile.
2. **[H] Documented install sequence is unsatisfiable.** `backend/requirements-dev.txt`
   pins `pytest>=8,<9` while `rlms==0.1.1` (in `backend/requirements.txt`) requires
   `pytest>=9.0.2` → `ResolutionImpossible` on a clean machine. The working venv has
   pytest 9.0.3 and 3,608 tests pass under it. Fix: `pytest>=9.0.2,<10`.
3. **[H] Test suite is not disk-hermetic.** 31 tests fail on any host with <15 GB free:
   `run_experiment`'s production disk-floor preflight (`primitives.py`, default
   `OPENRESEARCH_DISK_FLOOR_GB=15`) probes the **real host filesystem** inside unit tests
   with mocked sandboxes. With the floor disabled: 3,608 passed / 11 skipped / 1 xfailed /
   0 failed. Fix: conftest fixture pinning the floor for sandbox-mocked tests.
4. **[H] One test is 98 % of suite wall time.** `tests/rlm/test_run.py::
   test_stub_run_is_honestly_observable_in_artifacts` runs 862 s of an 881 s suite (next
   slowest: 10 s). Makes local iteration and CI miserable. Two other tests leak
   600-s-sleeping daemon threads.
5. **[H] Clean-machine run trap: every run defaults to the `runpod` sandbox.**
   `config.py:167` + `.env.example:72` default `OPENRESEARCH_DEFAULT_SANDBOX=runpod`, which
   requires RunPod creds + SSH key + a local Docker daemon — but README prerequisites
   mention none of them, and `./start.sh` hard-fails its preflight without `.env`. README
   also never tells you `start.sh` boots the backend only.
6. Suite is not network-hermetic: a worker re-established real `api.openai.com`
   connections during pytest because `.env` is auto-loaded at import time
   (`rlm/clients/__init__.py: load_dotenv()`), defeating `env -u OPENAI_API_KEY`. Culprit
   test not yet isolated (CI without `.env` is naturally hermetic; flagged as debt).
7. Doc quickstarts conflict: 4 half-overlapping quickstarts (README / CLAUDE.md /
   running-the-project.md / setup-guide.md); the runbook's omits venv creation entirely.
8. Environmental (host, not repo): the local venv's pip is broken (Homebrew python@3.14
   pyexpat dlopen failure on macOS 26) and the venv has drifted from pins
   (claude-agent-sdk 0.2.82 < pin, Pillow 12 > cap, bs4/pytesseract missing → 3 ingestion
   test files silently skip).

## 4. Infra findings

**Docker/compose (verified by code-reading + targeted repros + one real build):**
- **[H] Compose SQLite URL broken two ways.** `sqlite:///app/runs/openresearch.db` is a
  *relative* URL (both extractors strip `sqlite:///` naively) → resolves to
  `/app/app/runs/` — outside the mounted volume, in a dir the image never creates
  (reproduced: `sqlite3.OperationalError`). And even if fixed, `docker/entrypoint.sh`
  sources `/app/.env` **after** compose `environment:` is applied, so the documented
  `cp .env.example .env` silently overrides the DB URL back to an unpersisted in-container
  path. The "persisted SQLite event store" comment is false either way.
- **[M] Entrypoint crash-teardown is dead code** under `set -euo pipefail`: errexit fires
  on the failing `wait -n`, so the surviving sibling gets namespace-SIGKILL instead of
  SIGTERM and the diagnostic line never prints (demonstrated; notable given the repo's
  documented history of SIGKILL corrupting the event store).
- **[M] Compose publishes the backend on host `0.0.0.0:8000`**, bypassing the frontend
  unlock gate (read endpoints deliberately ungated; run-start ungated when
  `OPENRESEARCH_DEMO_SECRET` is empty) — combined with the docker.sock mount this is a
  LAN-reachable pivot. Fix: bind `127.0.0.1:8000:8000`.
- **[M] `start.sh` `docker info` preflight documented in CLAUDE.md + runbook does not
  exist on bes** (commit 5acdf4b is not an ancestor; lives on 3 other branches).
- Low: image runs as root (no `USER`); `curl | bash` nodesource install + tag-only base
  pins (non-reproducible provenance); full `docker.io` engine installed where only the
  socket-speaking SDK is needed; `.dockerignore` `.next/` doesn't match `frontend/.next`
  (15 MB context bloat); `start.sh` COPY'd into an image where it cannot work; compose
  references nonexistent `docs/setup-guide.md`.

**CI (Phase 8):**
- **[H] No CI runs any test.** The suite is provably CI-ready (keyless, dockerless, green
  with the disk-floor pinned; frontend lint/tsc/vitest/build all green and fast).
- **[H] The only existing gate (docs-freshness) is RED on `main`** since 2026-06-08
  (f9b14af removed `best_runs/adam/code/paper.pdf`; README on main still references it) —
  and its `paths:` filter is the exact mechanism that let tree-side breakage merge green.
- Hygiene nits: tag-pinned actions, no `permissions:`, no `concurrency`, no
  `workflow_dispatch`, no dependabot.

**Env vars (Phase 2.2):** ~175 distinct vars; `.env` untracked/ignored; zero stale entries
in `.env.example`, but:
- **[H] `.env.example` pins the known-broken `-runtime-` RunPod image** and documents it
  as the default — `cp .env.example .env` silently re-introduces the exact regression the
  code default (`-devel-`, commit 88c45b0) was reverted to fix.
- **[H] `OPENRESEARCH_DYNAMIC_GPU` (the CLAUDE.md-documented name, also what
  `--dynamic-gpu/--no-dynamic-gpu` writes) is read by nothing** — only
  `OPENRESEARCH_DYNAMIC_GPU_ENABLED` is honored (empirically verified). A cost-control
  flag is a silent no-op.
- **[M]** Missing from `.env.example`: the entire root-model credential surface
  (`OPENRESEARCH_RLM_ROOT_MODEL`, `OPENROUTER_API_KEY`, `FEATHERLESS_API_KEY`) + ~30
  operator-facing flags CLAUDE.md tells people to set. Stale "RLM root has NO OAuth path"
  comment (claude-oauth exists). Phantom var `OPENRESEARCH_EVIDENCE_GATE` documented as
  "the backstop" — no code reads it (real gates: `OPENRESEARCH_METRIC_PROVENANCE`,
  `OPENRESEARCH_METRICS_COMPLETENESS_CHECK`).
- BUG-LR-014 partially remediated: warn-only shell-shadows-.env validator exists on the
  CLI reproduce path (tests pass); server boot has none.

**Security (Phase 9): clean on secrets.** No real keys in tracked files or full git
history (every pattern hit is a fixture/placeholder/redaction-regex source; only
`.env.example` was ever committed). Demo gate (`hmac.compare_digest`), BYO-credentials
redaction, REPL safe-builtins boundary (eval/exec/compile/input stay blocked) all verified
as documented. Real items: compose :8000 exposure (above); LLM-generated shell strings run
host-side with `shell=True` behind a substring blocklist (by design — documented as a trust
boundary, not fixed); `eval`-based tilde expansion of `OPENRESEARCH_RUNPOD_SSH_KEY_PATH`
at 3 sites (injection-shaped, operator-controlled input); unlock cookie is an unsalted
SHA-256 of the secret (replayable until rotation — acceptable for a demo gate).

**Hygiene:** 25 files (6.5 MB: 5 MB montage.png, 11 JPEGs, an internal meeting transcript)
under `paper-repro-bes-docs/` are tracked **despite being gitignored** (the D2 remediation
untracked only the zip); `findings/` is an undocumented top-level dir of tracked raw run
logs; `data/calibration.json` is tracked but runtime-mutated (dirty-tree noise after first
preserved run); `HANDOFF.md` stale; `start_backend.sh` redundant; `frontend/next-env.d.ts`
churns between dev/build variants; freshness manifest misses `running-the-project.md` +
`harness-breakdown.md`.

**Ops capabilities (the 10 requested):** 6 solid (repeatable Playwright tasks, periodic
background processes, atomic state/log updates, 4-layer dead-run detection, browser status
surfaces incl. SSE + replay, clean run-dir persistence) / 4 partial:
- **[M]** CLI-launched runs never stamp a `pid` into `demo_status.json` → the orphan sweep
  skips them *by design*; a SIGKILLed CLI run shows `running` forever (API-spawned runs
  are fine).
- **[M]** `scripts/loops/{kill_and_restart.sh,lab_watch_loop.sh}` referenced by CLAUDE.md
  and the monitoring runbook **do not exist on bes** (only on 178c08c).
- **[M]** RLM-mode "Resume from last checkpoint? [Y/n]" is a no-op that still archives the
  checkpoints away (`args.resume` consumed only by the RDR path; `repl_state.pickle` has
  writers but zero readers).
- **[M]** No `runs/` retention/GC despite the `.preserved` marker contract; growth is
  unbounded. `periodic_liveness_sweep` is implemented but wired nowhere. Monitor scripts
  don't treat `killed`/`interrupted` as terminal. No relaunchable per-run config snapshot
  (`--seed` accepted, documented non-load-bearing).

## 5. Prioritized fix list (implementation order)

**P0 — reproduction blockers (fix now):**
1. Remove `@rolldown/binding-linux-x64-gnu` pin; refresh lockfile; verify `npm ci` +
   `docker build`.
2. `requirements-dev.txt` (+ pyproject dev extra) → `pytest>=9.0.2,<10`.
3. Compose DB URL → `sqlite:////app/runs/openresearch.db`; entrypoint: compose env wins
   over `.env`; fix dead teardown (`wait -n … || EXIT_CODE=$?`).
4. Test hermeticity: pin `OPENRESEARCH_DISK_FLOOR_GB` for sandbox-mocked tests; fix the
   862-s stall test + 600-s daemon-sleep fakes.
5. README/start.sh clean-machine path: Docker prerequisite, `--sandbox local` first-run
   guidance, `.venv` check + actionable preflight errors, "backend-only" wording.

**P1 — env/config:** `.env.example` reconciliation (RunPod image contradiction, root-model
creds, operator flags, OAuth note, minimal 3-line local dev header);
`OPENRESEARCH_DYNAMIC_GPU` alias fix (config + cli + CLAUDE.md + test); phantom
`OPENRESEARCH_EVIDENCE_GATE` doc fix; implement the documented `docker info` preflight in
`start.sh`.

**P2 — Docker/local-run:** compose port binding `127.0.0.1:8000`; `.dockerignore`
(`**/.next/`, bes-docs, logs, findings); drop dead `start.sh` COPY; fix compose doc path;
tilde-expansion hardening (3 sites); Makefile canonical targets
(`check`/`test-backend`/`test-frontend`/`smoke`/`docker-build`/`dev`).

**P3 — tests/smoke:** repo-hygiene invariant (`git ls-files -i -c` empty); smoke targets
(factory boot, CLI --help, compose config); keep all 3,608 green.

**P4 — CI:** new `ci.yml` (backend pytest -n auto keyless + frontend lint/tsc/vitest/build
+ compose validate; paths-filtered docker build); harden `docs-freshness.yml`
(workflow_dispatch, permissions, drop PR paths filter). Note: main's red docs-freshness
needs a one-line fix **on main**.

**P5 — hygiene:** untrack `paper-repro-bes-docs/`, `findings/*.log`,
`data/calibration.json`; archive `HANDOFF.md`; delete `start_backend.sh` +
`tools/seed-fake-run.sh`; manifest gaps; `next-env.d.ts` decision; title typos
("OpenResearch / OpenResearch").

**P6 — ops gaps (minimal versions):** pid stamping for CLI runs (+test); wire
`periodic_liveness_sweep` into app lifespan; terminal-state sets in `watch_run.py` /
`lab_screenshot_tail.mjs`; restore `scripts/loops/*` from 178c08c; `scripts/prune_runs.py`
honoring `.preserved` (dry-run default); honest RLM resume prompt; gate `create_app` debug
print/marker behind an env flag; `run_config.json` launch snapshot.

**Deferred (documented, deliberate):**
- Merging `origin/main` (17 commits, +90k lines, AKS/exec-reliability) into `bes` — trunk
  decision + conflict resolution beyond audit scope; **do this before further bes feature
  work.**
- Porting the evidence-gate forge fix + RunPod build short-circuit from
  `feat/rlm-wedge-hardening` — port to whichever branch becomes trunk.
- GEPA subsystem (product decision); non-root Docker user + nodesource/node-copy rework
  (behavior-changing image surgery); network-hermeticity socket guard (needs culprit-test
  bisection); RLM resume read-path (medium feature); k8s manifests (none exist on bes —
  AKS IaC lives on main).
