<!-- doc-meta: status=current; last-verified=2026-06-09 -->
# Troubleshooting

## Frontend Native Dependency Failure

Symptom:

```text
EBADPLATFORM Unsupported platform for @rolldown/binding-linux-x64-gnu
```

Cause: a platform-specific Rolldown binding was installed as a direct
dependency. Current `frontend/package.json` should not list any direct
`@rolldown/binding-*` dependency; the lockfile keeps platform bindings as
optional transitive packages.

Fix:

```bash
cd frontend
npm ci
```

If this still fails, inspect `frontend/package.json` for direct native binding
pins and regenerate `package-lock.json` on the target platform.

## Docker Daemon Not Reachable

`--sandbox docker` and `--sandbox auto` require a local Docker daemon for
`build_environment`. `--sandbox local` and `--sandbox runpod` do not.

Check:

```bash
docker info
```

Fix: start Docker Desktop/OrbStack, or run with `--sandbox local` or
`--sandbox runpod`.

## Missing `.venv` or Uvicorn

Symptom from `./start.sh`:

```text
.venv/bin/uvicorn not found
```

Fix:

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt
```

If host Python cannot create a venv because `ensurepip` is broken, use a known
Python 3.12 provider such as:

```bash
uv venv --python 3.12 --seed .venv
.venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt
```

## Pytest Dependency Conflict

Symptom:

```text
ResolutionImpossible: rlms==0.1.1 requires pytest>=9.0.2
```

Fix: use the current `backend/requirements-dev.txt`, which pins
`pytest>=9.0.2,<10`. Stale branches with `pytest>=8,<9` are unsatisfiable.

## API Keys and Claude OAuth

The root RLM model and Claude sub-agents are separate auth surfaces.

- `--model gpt-5`: needs `OPENAI_API_KEY`
- `--model claude`: needs a credited `ANTHROPIC_API_KEY`
- `--model claude-oauth`: uses `claude login`
- `--model qwen3-coder` or `kimi-k2.5`: needs `OPENROUTER_API_KEY`
- `--model qwen3-coder-featherless`: needs `FEATHERLESS_API_KEY`

Do not set an empty-credit Anthropic API key and expect OAuth fallback. The SDK
tries the key first and fails. For local Claude subscription use, leave
`ANTHROPIC_API_KEY=` empty and run `claude login`.

## RunPod SSH

RunPod runs need an API key and SSH key:

```bash
OPENRESEARCH_RUNPOD_API_KEY=...
OPENRESEARCH_RUNPOD_SSH_KEY_PATH=~/.ssh/id_ed25519
```

`start.sh` derives `OPENRESEARCH_RUNPOD_SSH_PUBLIC_KEY` from the private key
when possible. If pod SSH fails, verify the public key is registered in RunPod,
the private key path exists, and the key is readable by the process.

## SQLite Path

Use four slashes for an absolute path inside containers:

```bash
OPENRESEARCH_DATABASE_URL=sqlite:////app/runs/openresearch.db
```

`sqlite:///app/runs/openresearch.db` is relative to `/app`, so it resolves as
`/app/app/runs/openresearch.db` and can break persistence.

## Disk-Floor Preflight

Production `run_experiment` checks free disk using
`OPENRESEARCH_DISK_FLOOR_GB` (default 15). Tests disable it through
`tests/conftest.py` so mocked sandboxes do not fail based on host free space.

For a local run on a constrained machine:

```bash
OPENRESEARCH_DISK_FLOOR_GB=0 .venv/bin/python -m backend.cli reproduce demo_paper.pdf --sandbox local
```

Only disable it when you understand the risk: long ML runs can fill disks.

## Port Conflicts

Defaults:

- backend: `8000`
- frontend: `3000`
- compose backend: `127.0.0.1:8000`
- compose frontend: `3000`

Find conflicts:

```bash
lsof -iTCP:8000 -sTCP:LISTEN
lsof -iTCP:3000 -sTCP:LISTEN
```

Workarounds:

```bash
.venv/bin/uvicorn backend.app:create_app --factory --reload --port 8010
cd frontend && OPENRESEARCH_BACKEND_URL=http://127.0.0.1:8010 npm run dev -- --port 3010
```

For compose smoke testing while local dev servers are running, use a temporary
override file that maps container `8000/3000` to alternate host ports.

