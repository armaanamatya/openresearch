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
