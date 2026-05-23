# ReproLab

ReproLab reproduces research papers end-to-end, runs the resulting code, scores it against a PaperBench-style rubric, and serves the run in a live lab UI.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
cd frontend && npm ci && cd ..
export REPROLAB_BACKEND_URL=http://127.0.0.1:8000
.venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000
cd frontend && npm run dev
python -m backend.cli reproduce ftrl --mode rlm --max-usd 0.50 --sandbox local
python -m backend.cli ingest 2512.24601
```

- Modes: `--mode rlm` is the default hybrid path, `--mode rdr` is the pure rubric-driven controller, and `--mode rlm-pure` is the pre-hybrid escape hatch.
- UI: `/` landing, `/lab` live run view, `/leaderboard` persistent ranking, `/library` run browser.
- Docs: [system_overview.md](system_overview.md), [CLAUDE.md](CLAUDE.md), [setup guide](docs/guides/setup-guide.md), [deployment guide](docs/guides/deployment.md), [e2e testing](docs/runbooks/e2e-testing.md).
