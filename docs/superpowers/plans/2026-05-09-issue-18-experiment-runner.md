
## Execution Notes

- Aligned artifact utilities to the shared `src/reprolab/artifacts/` package shape from issue `#19` instead of keeping a second artifact module under `src/reprolab/experiments/`.
- Aligned sandbox imports and models to the shared `src/reprolab/sandbox/` package shape from issue `#17` (`models.py`, `runtime_backend.py`, `local_docker_backend.py`).
- Full verification ran as `python -m pytest -v --rootdir .` because default pytest root discovery resolves to the shared repo root when invoked from a git worktree.
- Blockers:
  - Shared sandbox package shape from `#17` is aligned locally, but this issue did not exercise real Docker execution.
  - Missing concrete orchestrator/task persistence path from `#10`.
