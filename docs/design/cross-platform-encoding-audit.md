# Cross-Platform Encoding Audit

**Generated:** 2026-05-17 by background sub-agent (Explore, read-only)
**Triggering bug:** `UnicodeEncodeError: 'charmap' codec can't encode character '→'` at `backend/hermes_audit/storage.py:30`
**Companion doc:** `docs/design/cross-platform-encoding-fix.md` (strategy)

---

## Executive summary

**43 HIGH hazards + 1 MEDIUM (safe to ignore) across 8 backend modules.**

Every HIGH hazard follows the same shape: `Path.read_text()` or `Path.write_text()` without `encoding="utf-8"`. On Windows the platform default is **cp1252**, so any data with `→`, `—`, Greek letters, math symbols, or smart quotes crashes. macOS and most Linux distros run UTF-8 by default — the bug ships green on those and explodes on Windows.

### Top modules by hazard count

| Rank | Module | Count |
|---:|---|---:|
| 1 | `backend/agents/orchestrator.py` | 11 |
| 2 | `backend/agents/experiment_runner.py` | 8 |
| 3 | `backend/agents/baseline_implementation.py` | 8 |
| 4 | `backend/agents/environment_detective.py` | 5 |
| 5 | `backend/agents/verification.py` | 3 |
| — | `backend/agents/paper_understanding.py` | 3 |
| — | `backend/agents/report_generator.py` | 2 |
| — | `backend/agents/improvement.py` | 1 |
| | **TOTAL** | **41** |

(The 43-vs-41 delta is because two `orchestrator.py` sites cover both a read and a write at the same logical operation — counted as two sites but one fix.)

---

## HIGH hazards — file:line table

Pattern in every row: missing `encoding="utf-8"` on a text-mode `Path` I/O.

### `backend/agents/orchestrator.py` (11)

| Lines | Operation |
|---|---|
| 292, 303 | checkpoint state JSON |
| 806, 812, 816 | research artifact reads (claim map, plans) |
| 1126, 1128 | environment spec / Dockerfile writes |
| 1151 | environment spec read |
| 2110, 2113, 2116 | final report payload reads / writes |

### `backend/agents/experiment_runner.py` (8)

| Lines | Operation |
|---|---|
| 76, 135 | run.log / commands.log write |
| 249, 262, 263 | metrics.json / provenance.json writes |
| 304 | stdout/stderr capture write |
| 475, 479 | metrics.json read |

### `backend/agents/baseline_implementation.py` (8)

| Lines | Operation |
|---|---|
| 330, 333 | config.json + assumptions.json write |
| 382, 383, 386, 392 | result.json + Dockerfile + manifest write |
| 411 | baseline_result.json write |
| 462 | baseline_result.json read |

### `backend/agents/environment_detective.py` (5)

| Lines | Operation |
|---|---|
| 95, 96 | Dockerfile + environment_spec.json write |
| 139 | re-read environment_spec.json |
| 146, 147 | re-write modified spec |

### `backend/agents/verification.py` (3)

| Lines | Operation |
|---|---|
| 95 | Dockerfile read |
| 349 | run.log read |
| 372 | provenance.json read |

### `backend/agents/report_generator.py` (2)

| Lines | Operation |
|---|---|
| 1031 | final report JSON write |
| 1034 | final report Markdown write |

### `backend/agents/paper_understanding.py` (3)

| Lines | Operation |
|---|---|
| 76 | claim_map.json write |
| 116, 122 | claim_map.json reads |

### `backend/agents/improvement.py` (1)

| Line | Operation |
|---|---|
| 148 | metrics.json write |

---

## MEDIUM hazards (1) — safe to defer

| File | Line | Why it's OK |
|---|---:|---|
| `backend/services/events/live_runs.py` | 1453 | `subprocess.run(["taskkill", ...], capture_output=True)`. Windows-only; output is ASCII PIDs. No `text=True`, so `capture_output` returns bytes — encoding never applies. Safe to ignore unless the call grows. |

---

## Already-fixed reference sites (use as the pattern to copy)

| File | Lines | What they got right |
|---|---|---|
| `backend/hermes_audit/storage.py` | 30, 39, 45 | `encoding="utf-8"` on every text I/O (fixed during this audit) |
| `backend/agents/dependency_verifier.py` | 255 | `encoding="utf-8", errors="replace"` for resilient reads |
| `backend/agents/resilience/cost.py` | all | uniformly utf-8 |

Plus 11+ binary-mode (`"rb"` / `"wb"`) sites across the tree — those are correctly *not* flagged because binary mode doesn't apply encoding at all.

---

## Repo-wide standardization (recommended)

Of the 43 hazards, **26 are JSON I/O**. Introducing one helper consolidates the encoding decision:

```python
# backend/utils/io.py  (new file)
from pathlib import Path
import json
from typing import Any

def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Always utf-8, ensure_ascii=False — readable JSON that survives Windows."""
    path.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
```

Then migrate the 26 JSON sites to use the helper. Reviewers get one knob (`utils/io.py`) to audit, not 26.

The remaining 17 sites are mixed (Dockerfiles, log files, Markdown reports) — explicit `encoding="utf-8"` at each site, no helper needed.

---

## The PYTHONUTF8 escape hatch — investigation outcome

The launcher (`scripts/dev.ps1` and `scripts/dev.sh`) sets `PYTHONUTF8=1`. The failing stack trace, however, shows **system Python**:

```
C:\Users\Armaan\AppData\Local\Programs\Python\Python311\Lib\encodings\cp1252.py
```

— not the venv at `.venv\Scripts\python.exe`. Combined with `File "<string>", line 105, in <module>` at the top of the trace, this is a `python -c "<inline code>"` invocation that escaped our env-var configuration. **The sub-agent did not pinpoint the exact dispatch site** — needs a follow-up grep for `subprocess.run([..., "-c"`, `python -c`, or similar in the agent execution path.

This is a *separate* hardening track from the encoding fix: even after every text I/O gets `encoding="utf-8"`, we want every subprocess that spawns Python to be invoked with `env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}` explicitly. Belt + suspenders.

---

## Validation plan (run after the fix PR)

- [ ] On Windows with `PYTHONUTF8` **unset**, pipeline runs without `UnicodeEncodeError` through all 9 stages.
- [ ] On Linux with `LANG=C` (forcing ASCII default), same expectation. CI smoke if available.
- [ ] grep across `backend/` for `\.write_text\(` / `\.read_text\(` with no `encoding=` returns ~zero hits.
- [ ] `dev.ps1` and `dev.sh` both produce identical logs across a smoke run — encoding parity check.

---

## Suggested PR layout

One PR, three commits:

1. **`backend/utils/io.py`** — new `write_json` / `read_json` helpers (~20 LOC).
2. **JSON-site migration** — the 26 JSON sites switch to the helpers. Mechanical.
3. **Non-JSON sites** — the 17 Dockerfile / log / Markdown sites get explicit `encoding="utf-8"`. Mechanical.

Estimated: < 1 hour of mechanical work + reviewer-friendly diffs.
