# Cross-Platform Encoding — Fix Strategy

**Status:** Active · 2026-05-17
**Audit (in progress):** `docs/design/cross-platform-encoding-audit.md` (written by sub-agent)
**Triggering bug:** `UnicodeEncodeError: 'charmap' codec can't encode character '→'` at `backend/hermes_audit/storage.py:30` during the `environment_built` stage.

---

## 1. Root cause

Python's default text-mode I/O uses `locale.getpreferredencoding(False)`. That value differs per OS:

| Platform | Default | Risk |
|---|---|---|
| Windows (any locale) | **cp1252** (charmap) | crashes on `→`, `—`, `α`, smart quotes, anything outside Latin-1 |
| macOS (Catalina+) | `utf-8` | safe by default |
| Linux with `LANG=en_US.UTF-8` | `utf-8` | safe |
| Linux with `LANG=C` / minimal containers | **ASCII** | same crash as Windows |
| Python with `PYTHONUTF8=1` (3.7+) | `utf-8` | safe — but **only for the process that has the env var** |

So a file write that works on a Mac developer's machine, on CI's Linux runners, AND on a Windows user with `PYTHONUTF8=1` will still crash on a Windows user whose subprocess spawn path doesn't propagate the env var. That is what just happened to us.

## 2. Why `PYTHONUTF8=1` alone is not the fix

`scripts/dev.ps1` and `scripts/dev.sh` both set `PYTHONUTF8=1`. The stack trace of the actual crash shows:

```
File "C:\Users\Armaan\AppData\Local\Programs\Python\Python311\Lib\encodings\cp1252.py"
```

— that's the **system** Python (`AppData\Local\Programs\Python\Python311`), not our venv (`.venv\Scripts\python.exe`). Somewhere in the pipeline, a subprocess is spawning system Python (probably via `subprocess.run([..., "-c", "<inline code>"])`, as evidenced by `File "<string>", line 105` at the top of the stack), and that subprocess didn't inherit `PYTHONUTF8` — or hit a system Python that didn't honor it.

Two takeaways:
- Env-var-based defaults are fragile across the subprocess boundary.
- The correct fix is to make every file write explicit, so the result is the same regardless of how Python was invoked.

## 3. Fix strategy

### 3a. Defensive coding rule

Every text-mode I/O site in the backend gains an explicit `encoding=`:

```python
# BEFORE
path.write_text(report.model_dump_json(indent=2))

# AFTER
path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
```

For reads where the file might come from an external/unknown source (paper text extracted from a PDF, third-party JSON):

```python
path.read_text(encoding="utf-8", errors="replace")
```

`errors="replace"` swaps invalid bytes for `U+FFFD` rather than crashing. Use sparingly — only when the data is presentational, not when it's structured (JSON, code).

### 3b. Consider helper functions where it pays off

If the audit finds >10 sites doing the same "write a Pydantic model as JSON to a path" pattern, introduce:

```python
# backend/utils/io.py
def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.write_text(json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8")

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
```

This centralizes the encoding decision and gives reviewers one knob to audit instead of N.

`ensure_ascii=False` is the right default — it produces smaller, more readable JSON; UTF-8 handles the non-ASCII chars natively.

### 3c. Subprocess hygiene

For any `subprocess.run(...)` / `subprocess.Popen(...)` that uses `text=True` or `capture_output=True`:

```python
# BEFORE
subprocess.run(cmd, capture_output=True, text=True)

# AFTER
subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
```

Also pass an explicit `env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}` if the subprocess is itself invoking Python. Belt + suspenders: explicit encoding in *our* code, plus push UTF-8 mode into the child.

The audit will surface where these subprocesses live. The user's stack trace `File "<string>", line 105` suggests at least one `python -c "<inline>"` pattern — that one needs both fixes.

## 4. PR sequencing

```
Tier 1 launcher (in flight)
  └── Encoding cleanup PR (this work)
        └── Tier 2a root logger
              └── Tier 2b per-agent transcripts
                    └── Tier 2c verify subcommand
```

Encoding cleanup goes between Tier 1 and Tier 2 because Tier 2 introduces *new* file-I/O sites — they should be born with explicit encoding rather than retrofitted later.

The cleanup PR itself splits into two commits for review hygiene:

1. **Mechanical fix** — add `encoding="utf-8"` everywhere the audit flags HIGH. Pure additive, no logic change.
2. **Helpers + subprocess fixes** — introduce `backend/utils/io.py`, migrate call sites that benefit. Optional / can be a follow-up.

## 5. Validation

Before merge:

- [ ] Run a full pipeline locally on Windows with **`PYTHONUTF8` unset** in the calling shell, confirm it doesn't crash. (This is the regression test — if it survives without the env var, we know the explicit encoding is doing the work.)
- [ ] CI smoke on Linux with `LANG=C` (forcing ASCII default). Same expectation.
- [ ] grep for `\.write_text\(` and `\bopen\(` with no `encoding=` after the patch — should be ~zero in `backend/`.

## 6. Out of scope

- Frontend (TypeScript) — Node's `fs.writeFile` defaults to UTF-8, not relevant.
- Generated reproduction code that the pipeline writes inside `prj_*/code/` — that's user-bound; we already pass `PYTHONUTF8=1` to that subprocess via `dev.ps1`/`dev.sh`, and the docker sandbox uses Linux defaults anyway.
- Database / SQLite encoding — already UTF-8 by config.

---

## Open question for the user

Should I land the audit findings as one mega-PR (all HIGH sites at once) or split by module (`backend/agents/`, `backend/services/`, `backend/hermes_audit/`, ...)? Recommend: **one PR with multiple commits**, each commit scoped to a module. Easier to review than a single 200-line diff; cheaper to revert than 8 separate PRs.
