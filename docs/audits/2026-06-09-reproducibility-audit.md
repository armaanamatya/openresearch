# Reproducibility audit & remediation — 2026-06-09 (branch `bes`)

> Dated audit report. Initial findings (repo map, branch analysis, full
> prioritized list): [`docs/audit-initial.md`](../audit-initial.md). Method:
> 11-agent parallel audit (~1.25M tokens, 715 tool calls) + orchestrator-run
> real builds/boots. Remediation: 12 commits, `dd2a9bf..d6ad50f`.
> Predecessors: `2026-06-07-bes-doc-alignment-audit.md` (doc↔code drift,
> REMEDIATED), `2026-05-31-backend-core-opportunity-backlog.md`.

## 1. Executive summary

**What was broken (verified, not guessed):**

1. **`npm ci` failed on macOS arm64 and the Docker build failed on Apple
   Silicon** — `frontend/package.json` pinned `@rolldown/binding-linux-x64-gnu`
   (an x64-only native binary) as a direct dependency (`EBADPLATFORM`). The
   documented dev platform could not install the frontend, and `docker build`
   died in stage 2.
2. **The documented install sequence was unsolvable** — `requirements-dev.txt`
   pinned `pytest>=8,<9` while `rlms==0.1.1` requires `pytest>=9.0.2`
   (`ResolutionImpossible` on every clean machine).
3. **`docker compose up` was broken three ways**: the SQLite URL was a
   relative 3-slash form resolving to the never-created `/app/app/runs/`
   (outside the "persistence" volume); the entrypoint `source`d `.env` *after*
   compose `environment:`, so a copied `.env.example` silently overrode the
   DB URL; and `source`-ing crashed the container outright on
   python-dotenv-valid unquoted values with spaces (`GeForce: command not
   found`, exit 127). Plus the crash-teardown after `wait -n` was dead code
   under `set -e` (surviving child got SIGKILL — the kill mode that has
   corrupted the event store before).
4. **The test suite was hermetic to neither disk nor network**: 31 tests
   failed on any host with <15 GB free (run_experiment's production disk-floor
   preflight probed the real filesystem inside sandbox-mocked unit tests), and
   one stub-mode unit test made a **real paid OpenAI call** (rubric
   generation), which — with the quota-dead key that `load_dotenv()` injects
   from `.env` at import time — turned into an 862-second 429-retry stall:
   98 % of total suite wall time.
5. **No CI ran any test** (the only workflow was docs-freshness, and its
   paths filter is the exact mechanism that let `main` go red on 2026-06-08).
6. **Operational gaps**: CLI-launched runs were invisible to the orphan sweep
   (no pid stamped — SIGKILLed runs showed `running` forever); the periodic
   liveness sweeper existed but was wired nowhere; monitors didn't recognize
   `killed`/`interrupted` as terminal; the documented
   `scripts/loops/{kill_and_restart,lab_watch_loop}.sh` didn't exist on this
   branch; no `runs/` GC existed despite the `.preserved` contract demanding
   one; the RLM "Resume from last checkpoint? [Y/n]" prompt was a no-op that
   archived the checkpoints regardless of the answer.
7. **Config/docs traps**: `.env.example` pinned the RunPod image the code
   default was deliberately reverted away from; `OPENRESEARCH_DYNAMIC_GPU`
   (the documented name, and what `--no-dynamic-gpu` writes) was read by
   nothing; `OPENRESEARCH_EVIDENCE_GATE` was a phantom; the root-model
   credential surface was absent from `.env.example`; `start.sh`'s documented
   docker preflight didn't exist and its unconditional `export` shadowed the
   `.env` sandbox choice; clean-machine README never mentioned that the
   default sandbox needs Docker + RunPod credentials.
8. **Hygiene**: 6.5 MB tracked-but-gitignored files (incl. a 5 MB PNG and an
   internal meeting transcript); tracked runtime-mutated files; a stale
   foreign-machine HANDOFF.md at the root; duplicate/legacy launcher scripts.

**Security posture: clean.** No real secrets in tracked files or anywhere in
git history (all pattern hits are fixtures/placeholders/redaction-regex
source). Demo gate, BYO-credential redaction, and the REPL safe-builtins
boundary verified as documented. No rotation needed.

**All of the above is fixed and verified** (see §2/§4), except the explicitly
deferred items in §6.

## 2. Reproduction status (after fixes)

| Surface | Status | Evidence |
|---|---|---|
| Backend install (clean machine) | **PASS** | `pip install --dry-run -r backend/requirements.txt -r backend/requirements-dev.txt` resolves in a clean `python:3.12` container (pytest 9.0.3 + rlms 0.1.1) |
| Backend tests | **PASS** | `pytest tests/ -n auto`, provider keys stripped: **3609 passed / 11 skipped / 1 xfailed / 0 failed in ~34 s** (was: 31 env-failures + 880 s, on this 4.8 GB-free host) |
| Frontend install | **PASS** | `npm ci` on darwin/arm64: 525 packages, 7 s (was `EBADPLATFORM`) |
| Frontend lint / types / tests / build | **PASS** | eslint clean · `tsc --noEmit` clean · vitest 291/291 · `next build` exit 0 |
| Docker build | **PASS** | `docker build .` exit 0 on Apple Silicon (was: stage-2 npm failure) |
| docker compose up | **PASS** | container `healthy`; backend `/health` `{"status":"ok"}`; frontend HTTP 200; leaderboard serving; in-container DB URL = compose-set absolute path despite mounted `.env` (smoke on remapped ports 13000/18000) |
| `make smoke` | **PASS** | app factory OK · CLI OK · compose OK |
| docs-check | **PASS** | 10 current-state docs enforced (was 9 + 2 escapees) |
| CI | **ADDED, not yet exercised** | `ci.yml` (backend + frontend + compose-validate) — first run happens on push/PR; jobs mirror locally-verified commands exactly |
| Kubernetes | **N/A on this branch** | zero manifests on `bes` (exhaustive search); AKS Terraform/Helm landed on `origin/main` (PR #99) |
| Playwright e2e | **NOT RUN** | needs a live backend + prod build; config + 7 specs verified by reading; browser-install step now documented |

## 3. Files changed (12 commits, grouped)

- **Reproduction blockers**: `frontend/package.json` + lockfile (rolldown pin);
  `backend/requirements-dev.txt`, `backend/requirements.txt` header,
  `pyproject.toml` (pytest pin, dependency mirror, phantom `src/`).
- **Docker/compose**: `docker-compose.yml` (absolute DB URL, loopback :8000,
  doc path), `docker/entrypoint.sh` (data-parse `.env` with
  container-env-wins precedence; live teardown), `Dockerfile` (dead `start.sh`
  COPY), `.dockerignore` (`**/.next/`, logs, findings, bes-docs, best_runs).
- **Test health**: `tests/conftest.py` (disk-floor autouse),
  `backend/agents/rlm/run.py` (stub mode skips paid rubric generation),
  `tests/rlm/test_run.py` (key-scrub autouse, pid tests),
  `tests/rlm/test_primitive_wall_clock.py` (Event-released fakes),
  `tests/test_repo_hygiene.py` (+ ignored-tracked invariant).
- **Config**: `backend/config.py` (`OPENRESEARCH_DYNAMIC_GPU` alias),
  `.env.example` (RunPod image, OAuth note, root-model creds, operator flags,
  minimal local-dev header, quoted GPU type).
- **Launcher**: `start.sh` (docker preflight, `.env`-aware sandbox default,
  `.venv` check, no `eval`), `scripts/runpod_check.sh` (no `eval`).
- **Ops**: `backend/app.py` (periodic liveness sweep wired into lifespan;
  debug print/markers behind `OPENRESEARCH_DEBUG_RUNS_ROOT=1`),
  `backend/agents/rlm/run.py` (pid stamp), `backend/cli.py` (honest
  interrupted-run notice replacing the no-op resume prompt) +
  `tests/cli/test_resume_offer.py`, `scripts/watch_run.py` +
  `scripts/lab_screenshot_tail.mjs` (terminal sets),
  `scripts/loops/*` + `frontend/e2e/lab-watch.spec.ts` (restored from
  178c08c), `scripts/prune_runs.py` + `tests/scripts/test_prune_runs.py`
  (the `.preserved`-honoring GC, dry-run default).
- **CI**: `.github/workflows/ci.yml` (new), `docs-freshness.yml` (hardened,
  paths filter dropped).
- **Hygiene**: untracked `paper-repro-bes-docs/` (25 files / 6.5 MB),
  `findings/*.log`, `data/calibration.json`, `frontend/next-env.d.ts`;
  deleted `start_backend.sh`, `tools/seed-fake-run.sh`; archived `HANDOFF.md`
  → `docs/archive/2026-05-31-root-harness-hardening-handoff.md`; `Makefile`
  (setup/check/test/smoke/docker-build/dev/clean).
- **Docs**: `CLAUDE.md`, `README.md`, `system_overview.md`,
  `docs/runbooks/running-the-project.md`, `docs/policies/current-docs.txt`,
  `docs/runbooks/2026-06-01-harness-breakdown.md` (moved+bannered),
  `backend/agents/rlm/context_map.py` docstring; `docs/audit-initial.md` +
  this report.

## 4. Key verification commands (exact, with results)

```text
docker build -t openresearch:audit .                      → exit 0 (was: EBADPLATFORM in frontend stage)
docker run --rm python:3.12-slim pip install --dry-run \
  -r requirements.txt -r requirements-dev.txt             → resolves; pytest-9.0.3 + rlms-0.1.1
env -u OPENAI_API_KEY … pytest tests/ -q -n auto          → 3609 passed, 11 skipped, 1 xfailed in 34.14s
  (pre-fix: 31 failed [disk_exhausted] / 880.88s of which one test = 862.32s;
   stall mechanism captured via -o faulthandler_timeout: openai _sleep_for_retry
   ← rubric_gen.generate_rubric_tree ← run_pipeline_rlm ← stub-mode unit test)
cd frontend && npm ci                                     → 525 packages in 7s
npm run lint / npx tsc --noEmit / npm test / npm run build → all clean; vitest 291/291
docker compose -f docker-compose.yml -f smoke-override up → healthy; /health ok; UI 200;
  in-container OPENRESEARCH_DATABASE_URL=sqlite:////app/runs/openresearch.db (compose wins over .env)
OPENRESEARCH_DEFAULT_SANDBOX=docker PATH=/usr/bin:/bin ./start.sh → actionable preflight exit 1
python scripts/prune_runs.py (dry-run, real runs/)        → keeps all .preserved dirs, deletes 0
make smoke / make docs-check                              → OK / OK (10 docs)
```

## 5. Remaining manual steps (unavoidable)

- **Local venv repair (this machine only):** the Homebrew python@3.14 in
  `.venv` has a broken pyexpat (`pip check/freeze/install` crash;
  `python3 -m venv` fails at ensurepip). `brew reinstall python@3.14 expat`
  or rebuild the venv with `uv venv`. The venv has also drifted from pins
  (claude-agent-sdk 0.2.82 < pin ≥0.2.87, Pillow 12 > cap,
  beautifulsoup4/pytesseract missing → 3 ingestion test files silently skip).
- **One-line fix on `main`:** its README references the deleted
  `best_runs/adam/code/paper.pdf` → docs-freshness red since 2026-06-08.
- Optional local cleanup: `mv reprolab.db openresearch.db` (retires the
  config fallback), `rm -rf backend/agents/{gepa,diagnostics}` (pycache
  ghosts), `rm -rf logs/_no_runs_root`.

## 6. Recommended next steps (prioritized)

1. **Trunk decision (blocks everything else):** `origin/main` is now 17
   commits / +90k lines ahead of `bes` (PR #99 AKS GPU backend + Terraform/
   Helm; PR #100 exec-reliability redesign) and both lines edited
   `primitives.py`/`run.py`/`cli.py`/`accelerator.py`/CLAUDE.md. Either merge
   `origin/main` into `bes` (expect real conflicts) or re-land bes-unique
   work on main. Do not start new feature work on `bes` first.
2. **Port the orphaned evidence-gate forge-row fix** —
   `origin/feat/rlm-wedge-hardening` 0a0084b..b63e16a (5 commits) closes the
   HIGH-severity `experiment_runs.jsonl` forge hole (recorded OPEN in project
   memory) *and* the RunPod local-build short-circuit. Port to whichever
   branch wins #1.
3. Cherry-pick 314d813 (BUG-NEW-043 child-RLM tracebacks) + 6182eac
   (PaperBench demo_status) from `pipeline-validation-mech-understanding`.
4. Branch GC: ~17 of 25 surveyed refs are fully contained in bes/main
   (list in `docs/audit-initial.md` §2) — delete them.
5. Decide GEPA's fate (`origin/feat/gepa-integration` holds the complete
   subsystem; the 5 earlier gepa branches are stages of the same line).
6. Run-reproducibility follow-ups: persist a `run_config.json` launch
   snapshot per run; implement (or formally drop) the RLM
   `repl_state.pickle` resume read-path; make `--seed` load-bearing.
7. Image polish (deliberate deferrals): non-root `USER`, digest-pinned bases,
   replace the `curl | bash` nodesource install (copy node from the builder
   stage), swap `docker.io` for `docker-ce-cli` or drop it.

## 7. Risks / debt (blunt)

- **The suite's network hermeticity is fixed at the known leak, not proven
  globally.** `load_dotenv()` at import time still injects `.env` credentials
  into every test process; I closed the one path that demonstrably dialed out
  (stub-mode rubric generation) and scrubbed keys for `tests/rlm/test_run.py`,
  but a future test can still reach a real API on a developer machine. A
  socket-blocking autouse fixture (pytest-socket) with explicit opt-in markers
  is the durable fix. CI is immune (no `.env`, no keys).
- **`bes` is no longer a superset of `main`.** Until the trunk decision,
  every commit here (including this audit's 12) deepens the divergence; the
  merge will conflict in the hot files either way.
- **CI is untested in anger** until the first push/PR. The commands are
  byte-identical to locally-verified ones, but runner variance (disk, npm
  registry, pip resolution drift on unpinned transitive deps) can still bite.
- **The evidence-gate forge hole remains open on this branch** (fix exists
  only on `feat/rlm-wedge-hardening`). A motivated REPL payload can still
  forge `experiment_runs.jsonl` success rows.
- **LLM sub-agent Bash runs host-side with `shell=True`** behind a substring
  blocklist — by design, but it means an untrusted paper is a host-level
  trust decision. Documented, not fixed.
- **`runs/` GC is a tool, not a policy:** `prune_runs.py` exists but nothing
  schedules it; growth is bounded only by operator habit.
- Compose-level resource limits and a frontend healthcheck are still absent
  (single healthcheck probes the backend only; the entrypoint watchdog now
  actually tears down, which softens this).

---

## Addendum — remediation round 2 (same day): trunk reunification + deferred items

All items deferred in §6 (except branch GC and GEPA, see below) were executed:

1. **`origin/main` merged into `bes`** (`2697ebb`) — bes is again a strict
   superset (AKS GPU backend + Terraform/Helm, exec-reliability redesign,
   cell-runner/env_pin line all aboard). Conflicts: CLAUDE.md (both lines'
   sections kept) and 4 `runs/` artifacts (bes's whitelist policy won;
   `tokens_total.json` added to the whitelist test to match main's gitignore).
   **Main's incoming code carried four clean-machine bugs of the same class
   this audit hunts** — all fixed on top:
   - 6 new test files hardcoded `/home/sww35/...` as the import root
     (collection errored everywhere but the author's box) → repo-relative.
   - a test shelled GNU `timeout` (absent on macOS: the busy loop never ran)
     → pure-bash `$SECONDS` spin.
   - the CPU-liveness stall probe was psutil-first with a `/proc` fallback,
     but **psutil was never declared** — on macOS the signal silently
     vanished and quiet CPU-busy local jobs were stall-killed → declared.
   - a credentials test failed on any Mac with a real `claude login`
     (unmocked Keychain probe) → probe faked.
   Also: OCR test fixture could never clear the parser's 200-char floor;
   README references to main's deleted `best_runs` PDF fixed both here and
   **on `main` via PR #101** (the sole CI gate had been red since 06-08).
2. **Evidence-gate forge fix ported** (`59d1d36`) — the FM-004 write-time
   gate + ledger cross-check existed on NEITHER bes nor main; the forge hole
   recorded OPEN in project memory since 2026-05-30 is now CLOSED on the
   trunk (`OPENRESEARCH_EVIDENCE_GATE=0` opts out; forge + replay tests in).
   Five legacy tests shipped evidence-less success verdicts and were adapted
   per the wedge branch's own patterns.
3. **Wedge hardening ported** (`b92bf6f`, `05ec187`) — runpod
   `build_environment` short-circuit (RunPod runs no longer need a local
   Docker daemon; prerequisite docs updated everywhere), hallucinated
   `runpod/` FROM-tag normalization, BUG-NEW-046 (FINAL_VAR refused when
   `run_experiment` never ran), sleep-robust wall-clock watchdog (Timer's
   monotonic clock pauses during macOS sleep) combined with main's hard
   ceiling, FM-001 transport retry, the Qwen3-1.7B id fix (two trunk tests
   pinned the wrong id), ISSUE-1 `--project-id` mismatch guard.
4. **BUG-NEW-043 ported** (`31aeb3d`) — child-RLM tracebacks surfaced +
   recursion-limit bump. (The branch's BUG-NEW-033 `rlm_query_misuse_patch`
   was NOT ported — that module never reached the trunk; still open.)
5. **PaperBench demo_status fix ported** (`e9c6e3f`, code only — the same
   commit's docs purge was rejected) + launching pid stamped.
6. **`run_config.json` launch snapshot** (`5049775`) — every run persists
   secrets-excluded resolved launch parameters; tested incl. a
   planted-credential leak check.
7. **Dockerfile**: node now copied from the frontend builder stage (no more
   `curl|bash` nodesource; serving Node == building Node; gnupg dropped).
8. **Local venv rebuilt** on uv-managed CPython 3.12 (Homebrew 3.14's pyexpat
   had broken pip itself); all pins satisfied, bs4/pytesseract/azure deps in,
   previously-skipped ingestion tests now run; the week-old :8000 dev server
   restarted onto it.

**Final state:** full suite **4,471 passed / 9 skipped / 1 xfailed / 0
failed (~33 s)** on the merged trunk; frontend lint/types/vitest/build green;
`docker build` + `docker compose up` healthy (builder-stage node v20.20.2
serving); docs-check OK; PR #101 open against main (docs-check green on it).
**CI exercised in anger:** the first `ci.yml` run on a truly keyless
ubuntu runner caught 5 more tests that passed locally only because `.env`
leaks credentials at import (host-dependent root-model resolution via the
macOS Keychain; presence-checking SDK constructors) — fixed with planted
fake keys and verified with `.env` physically removed. Second run:
**backend ✓ frontend ✓ compose-validate ✓** (run 27233993367, ~76 s test
step). Pushed: `origin/bes` @ `2302ddb`.

**Still open after round 2:** remote-branch GC (deleting others' branches
needs owner sign-off — list in `docs/audit-initial.md` §2); GEPA subsystem
product decision; BUG-NEW-033 misuse patch; RLM checkpoint-resume read path;
socket-level test-network hermeticity; non-root container user.

---

## Addendum — Codex continuation recovery and final validation

Continuation started from a clean `bes` checkout at `4d15886` (`origin/bes`).
The interrupted Makefile edit was already committed and syntactically valid;
there were no staged or unstaged repo changes. A pre-existing backend process
was already listening on `:8000` (`.venv/bin/uvicorn ... --port 8000`), so the
compose runtime smoke used temporary alternate host ports and left that process
alone.

Additional documentation hardening in this continuation:

- Added the missing canonical docs requested by the audit brief:
  `docs/reproduction.md`, `docs/architecture.md`, `docs/infra.md`, and
  `docs/troubleshooting.md`.
- Added those docs to `docs/policies/current-docs.txt` so `docs-check` enforces
  freshness markers and internal links.
- Refreshed README setup/test pointers and fixed stale setup/deployment command
  drift (`npm ci`, combined backend requirements install, four-slash container
  SQLite URL).
- Removed two unused eslint-disable comments in `frontend/e2e/lab-watch.spec.ts`.

### Ops capability audit

| Capability | Classification | Files / functions | How to test | Current blocker / limitation |
|---|---|---|---|---|
| Running Playwright/browser tasks repeatedly | Implemented but unverified in this continuation | `frontend/e2e/lab-watch.spec.ts`, `scripts/loops/lab_watch_loop.sh` | Start frontend/backend, then run `LAB_BASE_URL=http://localhost:3000 LAB_WATCH_MAX_CYCLES=1 scripts/loops/lab_watch_loop.sh` | Browser binaries and live UI needed; not part of `make check` |
| Running a background process every few minutes | Implemented | `scripts/loops/lab_watch_loop.sh`; app lifespan hook for `periodic_liveness_sweep` in `backend/app.py` | Set `LAB_WATCH_INTERVAL_S=300`; inspect `/tmp/lab-watch-loop.log`; unit/integration coverage via backend tests | Shell loop is operator-managed, not a durable scheduler |
| Updating background state/logs | Implemented and verified by tests | `demo_status.json` writers in `backend/cli.py` and `backend/services/events/live_runs.py`; `dashboard_events.jsonl`; `code/.exec_live.log` | Run a local reproduction and tail `runs/<id>/demo_status.json` / `dashboard_events.jsonl` | Full scheduled artifact workflow still manual |
| Generating a document/output artifact on a schedule | Partially implemented | `final_report.{json,md}` writers in RLM/RDR paths; `scripts/loops/*` can poll/watch | Complete a run and inspect `runs/<id>/final_report.md` | No first-class scheduler that periodically creates reports without an active run |
| Detecting dead/stuck runs | Implemented and verified by tests | `backend/services/events/run_liveness.py`, local exec heartbeat/stall code in `backend/agents/rlm/primitives.py` | Run pytest liveness/watchdog tests; kill a child run and query run status | SIGKILL is still unhandleable at process level; reconciliation is best-effort |
| Killing/restarting unhealthy processes | Partially implemented | `scripts/loops/kill_and_restart.sh`, CLI signal handling in `backend/cli.py`, entrypoint child teardown | Use a disposable run and invoke `scripts/loops/kill_and_restart.sh <id> <n> <pdf>` | Script is SDAR/retry-sprint oriented, not a generic supervisor |
| Viewing local output/status from browser or local link | Implemented and compose-verified | `/lab`, `/leaderboard`, `/library`, `frontend/src/app/api/demo/*`, backend run routes | Compose smoke: `curl -I http://127.0.0.1:13000/lab` returned 200 | Browser visual Playwright pass not run in this continuation |
| Persisting outputs in a clean directory structure | Implemented and documented | `runs/<project_id>/`, `run_config.json`, `demo_status.json`, ledgers, `code/`, reports | Start a run and inspect `runs/<id>/`; unit tests cover run_config secrets exclusion | `runs/` can still grow unbounded without operator cleanup |
| Cleaning old runs safely | Implemented but unscheduled | `scripts/prune_runs.py`, `tests/scripts/test_prune_runs.py` | `python scripts/prune_runs.py --dry-run`; pytest prune tests | No automatic retention policy; `.preserved` policy relies on operator use |
| Reproducing a run from logs/config | Partially implemented | `run_config.json`, `dashboard_events.jsonl`, `experiment_runs.jsonl`, `cost_ledger.jsonl` | Inspect `runs/<id>/run_config.json` and rerun equivalent CLI command | RLM checkpoint resume read path and load-bearing `--seed` remain open |

### Validation matrix

| Area | Command | Result | Evidence / notes | Remaining issue |
|---|---|---|---|---|
| Git state | `git status --short`; `git branch --show-current`; `git log --oneline --decorate -n 8` | PASS | clean at recovery; branch `bes`; HEAD `4d15886` before continuation edits | Dirty until this addendum/docs commit is created |
| Makefile syntax/help | `make help`; `make smoke` | PASS | help printed canonical targets; smoke printed `app factory OK`, `CLI OK`, `compose OK` | none |
| Backend clean dependency install | `uv venv --python 3.12 --seed /private/tmp/.../venv`; `pip install -r backend/requirements.txt -r backend/requirements-dev.txt` | PASS | resolved and installed successfully in throwaway Python 3.12.12 venv | system `python3` 3.14 still has broken ensurepip on this host |
| Frontend install resolution | `cd frontend && npm ci --dry-run --no-audit --no-fund` | PASS | dry run completed; platform bindings remained optional | dry-run only, per requested command |
| Backend tests | `.venv/bin/python -m pytest tests/ -q -n auto`; `make check` | PASS | `4471 passed, 9 skipped, 1 xfailed`; rerun inside `make check`: same pass count in 32.26s | 20 warnings; optional `faiss`, `chromadb`, `torch` tests skipped |
| Frontend lint/types/tests | `npm run lint`; `npx tsc --noEmit`; `npm test`; `make check` | PASS | eslint clean after stale-disable cleanup; tsc clean; vitest `291 passed` | Playwright e2e not run |
| Docker build | `docker build -t openresearch:audit .` | PASS | build completed using cached layers; image tagged `openresearch:audit` | root runtime image and docker socket risk remain |
| Compose config | `docker compose config`; `docker compose config -q` | PASS | backend bound to `127.0.0.1:8000`; DB URL `sqlite:////app/runs/openresearch.db` | none |
| Compose runtime health | `docker compose -p openresearch_audit -f docker-compose.yml -f /private/tmp/openresearch-compose-audit.yml up -d --no-build`; `curl /health`; `curl -I /lab`; `docker compose ... down` | PASS | container healthy; `/health` returned `{"status":"ok","version":"0.1.0"}`; `/lab` returned HTTP 200; project torn down | used alternate host ports because a pre-existing backend occupied `8000` |
| Shell syntax | `bash -n start.sh`; `bash -n docker/entrypoint.sh` | PASS | no syntax errors | none |
| Start-script failure modes | throwaway copies of `start.sh` with `OPENRESEARCH_DEFAULT_SANDBOX=local`, `docker`, and `runpod` | PASS | local skipped Docker; docker warned when Docker CLI absent from PATH; missing `.venv/bin/uvicorn` printed install command | runpod temp-copy skipped missing preflight script; real repo has it |
| Docs freshness | `make docs-check`; `make check` | PASS | `14 current-state docs, 3 tracked PDFs`; OK | none |
| Env/config validation | `docker compose config`; `make smoke`; settings imported during app factory boot | PASS | compose env wins for DB URL; app factory booted | `.env` can still leak credentials into tests without stronger socket blocking |
| Local run / health check | compose health check and pre-existing local backend process | PASS | compose `/health` OK; existing local uvicorn on `:8000` was not killed | no real paper reproduction run launched in this continuation |

---

## Addendum — Round 3: adversarial review of the audit's own changes

On "recheck and continue," two multi-agent review passes were run over the
session's own work: a fact-check of the four new tier-2 docs (every testable
claim verified against code, findings adversarially re-verified) and a
five-group adversarial code review of all session commits (RLM ports, merge
conflict resolutions, docker/entrypoint/start.sh, CI + test adaptations, ops
scripts), with every finding independently re-verified against the current
tree before counting. Result: 2 confirmed doc inaccuracies + 11 confirmed
code findings (3 medium, 8 low). All 13 are fixed and test-covered.

### Confirmed findings → fixes

| # | Sev | Finding | Fix (commit) |
|---|-----|---------|--------------|
| 1 | MED | Evidence-gate forge cross-check defeated across a warm retry: `ctx.cost_ledger` is seeded from the root-writable `cost_ledger.jsonl` at run start, so a forged ledger row from a crashed prior attempt satisfied the "unforgeable in-memory count" (empirically reproduced) | `RunCostLedger.session_call_count` — only in-process appends count; seeding stays for budget continuity (`87f9622`) |
| 2 | MED | finalize-on-timeout partials never pass the gate: a `partial_timeout` row (harness-loaded metrics) was forced to `failed` with a factually wrong note — the fatal-path `partial` verdict was dead code for the exact scenario the 2026-06-08 redesign salvages | Second-tier `evidence_cap`: cap at `partial`, gated on ≥1 in-process ledger call; note text corrected (`87f9622`) |
| 3 | MED | Containerized sweeper falsely interrupted live host-launched runs: compose bind-mounts `./runs`, host pids don't exist in the container's pid namespace, and nothing refreshed `updatedAt` during a run | `pidHost` stamped by all 3 writers + sweeper skips mismatches; EPERM now reads alive; 30s cost daemon refreshes `updatedAt` (true heartbeat) (`05aeed5`) |
| 4 | MED | Both .env shell parsers kept inline ` # comments` in values; the corrupted export outranks pydantic's correct parse → hard `ValidationError` boot crash on `.env.example`'s own suggested header line; container restart loop | Entrypoint delegates parsing to python-dotenv itself (`docker/load_env.sh`); start.sh gets a dotenv-faithful bash lib (`319cd6e`) |
| 5 | LOW | `run_config.json` persisted value-borne secrets (credentialed `OPENRESEARCH_DATABASE_URL` DSN, bootstrap command) | Exact-name denylist + URL-userinfo redaction (`19a0106`) |
| 6 | LOW | `_normalize_runpod_from_line` landed dead-on-arrival (its runpod gate sits after the runpod short-circuit return); CLAUDE.md promised a protection that never ran | Unconditional after the shape guard (self-gating); docs corrected (`19a0106`) |
| 7 | LOW | Parsers dropped dotenv-valid `export KEY=` / `KEY = v` lines; duplicate keys were first-wins (dotenv: last-wins) — split-brain flags between Docker and local | Same parser rewrite (`319cd6e`) |
| 8 | LOW | CRLF .env corrupted values (trailing `\r`, kept literal quotes) | CR strip + dotenv delegation (`319cd6e`) |
| 9 | LOW | `setdefault("pid")` inherited a DEAD prior attempt's pid on reused project dirs → live run falsely swept | Unconditional overwrite (run.py executes inside the run process); old test premise inverted (`05aeed5`) |
| 10 | LOW | Stub-mode rubric-gen skip left the arXiv rubric wiring covered by zero tests | Two mocked pipeline tests (generate→context+persist; None→rubric-less) (`19a0106`) |
| 11 | LOW | Doc fixes: `reproduction.md` Playwright claim (self-starting webServer on :3001, `reuseExistingServer:false`), `infra.md` "runs/ is gitignored" (whitelist idiom tracks 7 artifact types) | Corrections applied (`f54c437`) |

Also: deduplicated a doubled `_ensure_local_data_root` merge artifact in
`run.py`; merged PR #101 (main's docs-freshness CI confirmed green on the
merge commit, run 27252417055).

### Verification (round 3)

| Check | Result |
|---|---|
| Backend suite (`pytest tests/ -n auto`) | **4485 passed**, 9 skipped, 1 xfailed, 38.1s (+14 new regression tests) |
| Frontend (`lint` / `tsc --noEmit` / `vitest`) | clean / clean / 291 passed |
| `make docs-check` | OK (14 current-state docs) |
| Shell syntax (`bash -n` × 4 files) | OK |
| `docker build` (new `load_env.sh` COPY) | exit 0 |
| In-container boot proof | image booted with a mounted `.env` containing the previously boot-crashing line `OPENRESEARCH_DEFAULT_SANDBOX=local   # …` → `/health` 200 (`Settings()`' Literal field would have raised on any corrupted value) |

---

## Addendum — Round 4 (2026-06-10): "do all" — every parked item resolved

### Ports and security hardening

- **BUG-NEW-033 ported** (the last orphaned fix): `rlm_query_misuse_patch.py`
  auto-recovers the `(slice, question)` misuse of `rlm_query`/`llm_query`
  (the two-arg form routed the question as a CLI model name and shipped the
  CLI error string as `paper_claims`); system prompt now teaches the composed
  single-prompt form. Tests added (the source branch had none).
- **In-run forge residual closed** (per-row ledger provenance):
  `CostLedgerEntry.outcome` ("ok"/"failed"/"raised") is stamped by
  `binding.wrap_primitive` on every exit path including the contract-guard
  rejection; the evidence gate now requires ≥1 success-compatible in-process
  `run_experiment` call to back a success row. A real-but-failed call + a
  forged row no longer passes. Remaining (narrower) residual re-documented:
  one REAL success can still shelter a forged sibling row.
- **Socket-level test hermeticity**: pytest-socket addopts
  (`--disable-socket --allow-unix-socket --allow-hosts=127.0.0.1,::1`) — the
  862s-stall class is structurally impossible; canary tests prove the guard
  is armed.
- **Non-root container**: both servers run as uid 10001; docker socket via
  compose `group_add` (`OPENRESEARCH_DOCKER_GID`); SSH-key injection moved to
  `$HOME/.ssh`. Verified in-container: `/health` 200 as `app`,
  `docker.from_env().ping()` OK, `runs/` writable.

### Trunk events (the hard part of the day)

origin/main moved +149 (the azure-fork merge: score-integrity, steering
injection, attempt isolation, All-CNN showcase) — re-merged into bes. The
incoming code had regressed the `REPROLAB_→OPENRESEARCH_` rename across ~20
files (main has no test CI; the regression broke the local suite's disk-floor
fixture). Fixed structurally: all env-name literals normalized (153 files),
`Settings.env_prefix` flipped to the canonical `OPENRESEARCH_` (bridge keeps
legacy exports working), credential fields gained canonical AliasChoices, the
AKS in-Job entrypoint renamed with a dual-spelling injection shim for pinned
ACR images. A symbol-level two-parent audit caught a ~361-line `primitives.py`
region dropped by conflict resolution (restored; `plan_reproduction` taken
from main to keep its new compute_scope warnings).

Mid-session the parallel track pushed its OWN merge of main into bes (plus
Azure Bicep, RunPod 404-idempotent teardown, UI panels, and a new uv-lock+ruff
CI lint job) — the two reunifications were reconciled commit-by-commit
(legacy credential aliases kept; their cleaner test fixes adopted; their
`cap_overall_budget` and dead-code removals taken; my provenance threading
and image-compat shim kept). Two CI-only regressions caught and fixed by the
new gates: a stale `uv.lock`, and ruff `--fix` stripping noqa-less re-exports.

### Branch GC (verified, executed)

22 remote branches deleted: 15 fully-merged ancestors of the final trunk,
3 whose every fix is ported and test-covered (`feat/rlm-wedge-hardening`,
`pipeline-validation-mech-understanding`,
`feat/integrate-perf-accelerator-into-stability`), 2 stale single-commit
cleanups superseded by this audit, and 2 GEPA siblings (tagged
`archive/gepa-*` first). Kept: `azure` (active local worktree),
`run-archives` (only copy of archived run data), `localqwen` (uncertain
supersession), `feat/gepa-integration` (below).

### GEPA decision: ADOPT-LATER (evidence-gated)

Multi-agent assessment of `origin/feat/gepa-integration` (~910 LOC
prompt-optimization subsystem, 36/37 hermetic tests, well-engineered
fail-soft): a direct merge is infeasible (101 conflicted files from
duplicated lineage; 2 semantic conflicts: its anytime-scoring duplicates the
finalize-on-timeout mechanism, its BUG-NEW-050 fallback weakens the evidence
gate). The only field evidence shows ZERO accepted candidates (pre-timeout-fix)
and no post-fix run. Decision per the repo's own validate-don't-build rule:
**entry gate** = one $0 claude-oauth A/B showing post-fix GEPA accepts
candidates and moves a rubric score on ≥1 paper — if not run by the next
audit cycle, downgrade to drop. Re-land = hand re-application of the
GEPA-only payload (~8 hook edits), never a git merge; required fixes at
re-land: OPENRESEARCH_GEPA_* names, socket-gate the live-API test, route
`gepa_*` SSE through sanitization, make off-mode fully inert.

### Checkpoint-resume: resolved as DOCUMENTED-BLOCKED

`rlms 0.1.1`'s `RLM.completion(prompt, root_prompt)` takes no message
history and spawns a fresh environment per call — true mid-run REPL resume
requires upstream library support. The practical resume value already ships
at the three layers where time/money go: the primitive warm-retry cache
(`primitive_cache.jsonl`), cell-grid resume (`OPENRESEARCH_RESUME_CELLS` +
skipped-cell aggregation), and prior-attempt evidence injection into the
implementer prompt (main, 2026-06-09). `checkpoint.py` stays emit-only by
design (its sanitized log cannot reconstruct REPL state).

### Flagged for owner awareness

Main's two-axis verdict feature (`OPENRESEARCH_TWO_AXIS_VERDICT`, default
OFF) projects the legacy verdict from root-writable `rlm_state/` artifacts
(`fidelity_certificate.json`, `repro_spec.json`) AFTER the evidence gate —
when that flag is on, a forging root could bypass the gate by writing a green
certificate. Acceptable while default-off; gate-order needs revisiting before
the flag ever defaults on.

### Round-5 continuation: last verification gaps closed

- `make check` (the documented aggregate gate): green end-to-end.
- Full `docker compose up --build` smoke on the final configuration:
  healthcheck **healthy**, `/health` + `/lab` 200, **uid 10001** confirmed in
  the running compose stack (an earlier smoke had silently used a stale
  `openresearch/app:latest` tag — rebuilt and re-verified).
- **Playwright e2e executed for the first time in this audit**: 18 passed,
  1 skipped, **3 failed — all pre-existing UI-side regressions** (Playwright
  is not in CI, so these were invisible):
  1. `rlm-lab.spec.ts` — the page renders **two** `node-detail-sidebar`
     elements (identical aria-labels, both collapsed) → strict-mode locator
     violation. Looks like an accidental double render from the recent
     CollapsiblePanel work. Genuine UI bug.
  2. `lab-smoke-interactive.spec.ts` (library status filter) — seeded
     `prj_diffusion_smoke` row never appears in the library table.
  3. `reports.spec.ts` — "worker reports" section text not found for a
     seeded run.
  Left for the UI track (UI components are deliberately out of this audit's
  lane); none affect the backend pipeline, API, or the verified compose
  deployment. Consider adding a Playwright job to CI once green.
