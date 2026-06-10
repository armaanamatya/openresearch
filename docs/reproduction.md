<!-- doc-meta: status=current; last-verified=2026-06-09 -->
# Reproduction

This is the boring fresh-clone path. It is intentionally narrower than the
historical runbooks: one local setup, one local run path, one check path.

## Requirements

| Tool | Expected |
|---|---|
| Python | 3.11+; Python 3.12 is the verified clean-install target and Docker runtime |
| Node.js | >=20.19 <21, or >=22.12 |
| npm | 10+ |
| Docker | Required for `--sandbox docker`, `--sandbox auto`, Docker image builds, and compose |
| LLM auth | One root auth path: `OPENAI_API_KEY`, credited `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `FEATHERLESS_API_KEY`, or `claude login` with `--model claude-oauth` |
| RunPod | Required only for `--sandbox runpod`: API key plus SSH key |

The default local-dev sandbox in `.env.example` is `local`, which does not need
Docker or RunPod. The code default remains `runpod` when no environment file or
shell override is present.

## Fresh Clone

```bash
git clone <repo-url> openresearch
cd openresearch

python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt

cd frontend
npm ci
cd ..

cp .env.example .env
```

Edit `.env` only for the auth path and sandbox you actually use. For no-cloud
local smoke work, keep:

```bash
OPENRESEARCH_DEFAULT_SANDBOX=local
ANTHROPIC_API_KEY=
```

Then use either an API key for the root model or:

```bash
claude login
.venv/bin/python -m backend.cli reproduce demo_paper.pdf --sandbox local --model claude-oauth
```

## Run Locally

Terminal 1, backend:

```bash
.venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000
```

Terminal 2, frontend:

```bash
cd frontend
OPENRESEARCH_BACKEND_URL=http://127.0.0.1:8000 npm run dev
```

Open `http://localhost:3000/lab`.

Backend-only launcher:

```bash
./start.sh
```

`start.sh` reads `OPENRESEARCH_DEFAULT_SANDBOX` from the shell, then `.env`,
then falls back to `runpod`. It does not start the frontend.

## Smoke Checks

```bash
make smoke
make docs-check
bash -n start.sh
bash -n docker/entrypoint.sh
```

Expected smoke output includes:

```text
app factory OK
CLI OK
compose OK
```

## Test Commands

```bash
.venv/bin/python -m pytest tests/ -q -n auto

cd frontend
npm run lint
npx tsc --noEmit
npm test
```

Playwright browser checks are separate because they need browser binaries and a
running app:

```bash
cd frontend
npx playwright install chromium
npx playwright test
```

## Docker

```bash
docker build -t openresearch:audit .
docker compose config
docker compose up -d
curl -fsS http://127.0.0.1:8000/health
curl -fsS -I http://127.0.0.1:3000/lab
docker compose logs --tail=120 app
docker compose down
```

Compose mounts `./runs` at `/app/runs` and sets
`OPENRESEARCH_DATABASE_URL=sqlite:////app/runs/openresearch.db`. The four
slashes are required for an absolute in-container path.

## Expected Outputs

Each run writes to `runs/<project_id>/`:

- `demo_status.json`
- `run_config.json`
- `dashboard_events.jsonl`
- `rlm_state/`
- `cost_ledger.jsonl`
- `experiment_runs.jsonl`
- `code/`
- `final_report.json` and `final_report.md` when complete

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `No module named pip` while creating a venv | Host Python/ensurepip is broken | Use a known Python 3.12 (`uv venv --python 3.12 --seed .venv`) or repair the host Python |
| `ResolutionImpossible` involving pytest | Stale branch or stale lock of dev deps | Use `backend/requirements-dev.txt` with `pytest>=9.0.2,<10` |
| `EBADPLATFORM` for `@rolldown/binding-linux-x64-gnu` | Stale frontend dependency pin | Remove direct platform binding pins and run `npm ci` from the current lockfile |
| `Docker daemon not reachable` | `docker` or `auto` sandbox without a daemon | Start Docker/OrbStack or use `--sandbox local`/`runpod` |
| RunPod SSH failures | Missing key path or public key not registered | Set `OPENRESEARCH_RUNPOD_SSH_KEY_PATH` and verify the key in RunPod |
| Empty/partial SDK output | LLM credential issue, often no-credit Anthropic API key | Leave `ANTHROPIC_API_KEY` empty for Claude OAuth, or provide a credited API key |
| SQLite opens under `/app/app/runs` | Three-slash SQLite URL | Use `sqlite:////app/runs/openresearch.db` in containers |
| Tests fail with `disk_exhausted` | Production disk-floor preflight | Tests disable this by default; set `OPENRESEARCH_DISK_FLOOR_GB=0` for local mocked runs |

