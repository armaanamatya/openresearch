<!-- doc-meta: status=current; last-verified=2026-07-20 -->
# OpenResearch onboarding

Use this page as the single starting point for the repository.

1. Read the public overview in [README.md](README.md).
2. Set up and run locally through [docs/operations.md](docs/operations.md).
3. Use the supported development commands in the [Makefile](Makefile):
   `make setup`, `make smoke`, `make check`, `make dev`, `make dev-backend`, and
   `make dev-frontend`.
4. Read [docs/architecture.md](docs/architecture.md), then
   [docs/engineering-guide.md](docs/engineering-guide.md).

The authority order is **code → architecture → CLAUDE.md → README.md**.

## Day-to-day rules

- Keep generated run output in `runs/`; commit only the small, reviewed
  summaries allowed by `.gitignore`.
- Treat `best_runs/` as curated evidence, not a general output directory.
- Prefer `make` targets over copying one-off shell commands from historical
  handoffs. `scripts/dev.sh` remains available only when its isolated,
  timestamped run-log layout is specifically needed.
- Before changing a current-state document, verify it against the code and run
  `make docs-check`.
