<!-- doc-meta: status=current; last-verified=2026-08-01 -->
# tests/CLAUDE.md

> Loaded when working in the test suite. Root context: ../CLAUDE.md.

## Running tests

```bash
pip install -r backend/requirements.txt
pip install -r backend/requirements-dev.txt   # adds pytest + parallel runners

.venv/bin/python -m pytest tests/                              # all
.venv/bin/python -m pytest tests/ -n auto                      # parallel (needs requirements-dev)
.venv/bin/python -m pytest tests/path/to/test_x.py::test_name  # single test
.venv/bin/python -m pytest tests/ --reruns 2                   # rerun flaky tests
```

Locked/CI-matching env: `uv venv --python 3.12 && uv sync --frozen` (Python 3.12, matches the Docker image; the local dev venv may run newer Python — floor is 3.11). Lint: `uvx ruff@0.15.16 check .` (E4/E7/E9/F defaults; `pyproject.toml` carries per-file ignores, incl. `tests/*` → `F841`/`E741`/`E702`/`E402` for capture-as-side-effect assignments and post-`importorskip()` imports).

Pytest config lives in `pyproject.toml` under `[tool.pytest.ini_options]`: `testpaths = ["tests"]`, `pythonpath = ["."]`.

## Conventions

- **Socket-hermetic — no live network, ever.** `pyproject.toml`'s `addopts` sets `--disable-socket --allow-unix-socket --allow-hosts=127.0.0.1,::1`: every test runs with real network sockets blocked, full stop. Only loopback (TestClient/uvicorn-style tests) and unix sockets (docker SDK doubles) stay open. A test that would otherwise reach an external API must inject a fake transport/client or monkeypatch the credential so the code path never dials out — don't rely on "no API key present" as your isolation. A leaked `.env` credential can't turn a test into a live call; it fails fast with `SocketConnectBlockedError` instead of hanging or spending money.
- **Env-hermetic too (2026-07-13) — no ambient `.env`/host-credential bleed, ever.** `Settings` (`backend/config.py`) declares `SettingsConfigDict(env_file=".env")`, and pydantic-settings re-reads that file from disk on EVERY `Settings()` construction regardless of `os.environ` — `monkeypatch.delenv` cannot touch a value that never travelled through the process env. `tests/conftest.py` closes this test-side only (production is untouched, byte-identical): at import time it scrubs every leaking env namespace/name (`OPENRESEARCH_`/`REPROLAB_`/`ANTHROPIC_`/`OPENAI_`/`AZURE_`/`GCP_`/`RUNPOD_` prefixes + a fixed set of unprefixed credentials) and sets `Settings.model_config["env_file"] = None`; the autouse `_isolate_environment` fixture re-asserts both per test and resets the `Settings` cache, so nothing leaks test-to-test. **Rule:** a test asserting a Settings-backed DEFAULT must never rely on ambient env or the developer's real `.env` — inject explicitly via `monkeypatch.setenv`/`delenv` (it runs in the test body, after every fixture, so it always wins). `dotenv_disk_reads_enabled` is the documented, narrow opt-back-in fixture for the few tests whose SUBJECT is the dotenv-read path itself. `tests/test_env_hermeticity.py` is the guard that fails loudly the moment hermeticity regresses (e.g. it diffs a plain `Settings()` against `Settings(_env_file=None)` field-by-field). WHY: this one root cause was behind 18 suite failures, and one of them printed a real Azure Foundry API key into a pytest assertion diff.
- **Tests live under `tests/`, mirroring the `backend/` package** — `tests/routes/` ~ `backend/routes/`, `tests/services/events/` ~ `backend/services/events/`, `tests/agents/rlm/` ~ `backend/agents/rlm/`, `tests/cli/` ~ `backend/cli.py`, etc. Shared fixtures live in `tests/conftest.py`. One wrinkle: RLM-harness-level tests are split across `tests/agents/rlm/` (per-module unit tests) and a parallel `tests/rlm/` (harness/integration-level: registry, run, custom-tools integration) — a few filenames exist in both trees, so when tracing a rule to its regression test, check both before concluding it's missing.
- **Every new flag or invariant ships a hermetic OFF+ON test pair.** This is the convention behind the entire feature-flag catalog in the root doc: default-OFF must be proven byte-identical to the prior baseline, and default-ON must be proven to actually change behavior — both without a network call or a GPU. A flag landed without both halves is unfinished, not just under-tested. The same applies to stability/correctness bugfixes (REPL safe-builtins, forced-iteration, evidence-gate, dockerfile-shape guard, run-status enum, CLI signal handling, etc. — the rules live in the rlm-scoped nested doc, `../backend/agents/rlm/CLAUDE.md`): each ships a named regression test alongside the fix, in the same change, not as a follow-up. Don't delete one of these tests to make a refactor pass — fix the refactor.

## CLAUDE.md fidelity guard

`tests/test_claude_md_fidelity.py` keeps the doc set honest against the code — it fails the suite the moment a documented claim goes stale:
- `test_documented_env_var_is_read_in_code` — every var in its `_DOCUMENTED_ENV_VARS` list must appear in the doc text AND be `git grep`-able somewhere under `backend/`.
- `test_custom_tools_count_matches_doc` — imports `PRIMITIVE_REGISTRY` from `backend.agents.rlm.primitives`, asserts `len(...) == 21`, and asserts the literal `"21"` appears in the doc.
- `test_all_doc_citations_resolve` — every `docs/`-rooted markdown path cited anywhere in the doc set must exist on disk.

The guard reads the root file **plus every nested `CLAUDE.md`** (`_read_claude_docs()`
concatenates the set — landed 2026-07-05), so a documented env var, the primitive count, or a
`docs/*.md` citation may live in root OR in any nested file; the guard only needs to find it
somewhere in the set. If you add a doc-fidelity claim to a nested file, check the guard's
actual file list in `_read_claude_docs()` first. The file also carries three cloud/auth
posture guards over `_POSTURE_DOCS` (GKE must be "NOT USED" — never "parked"; any OAuth
mention needs a forbidden marker; known stale OAuth-recommendation phrases are banned).

When you add a primitive, change `PRIMITIVE_REGISTRY`'s size, or touch a `_DOCUMENTED_ENV_VARS` entry, update the doc(s) and this test together, in the same change.
