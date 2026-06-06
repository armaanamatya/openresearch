# OpenResearch TUI — Terminal Runs Monitor — Design

**Status:** proposed (2026-05-25)
**Spec target:** `openresearch tui` — a read-only terminal dashboard for observing run state, sandbox/GPU usage, log tails, and rubric scores across local + RunPod runs.

---

## Why

The lab's web UI is the primary surface, but a TUI fills three real gaps:

1. **Headless workflow.** When you're SSH'd into a server, watching `tail -f` + `nvidia-smi` + `runs/*/demo_status.json` in three panes is what you do today. The TUI consolidates that.
2. **Backend-independent observability.** The web UI requires the FastAPI process to be up. A file-based TUI keeps reading even when the backend is restarting, crashed, or has not been started at all (e.g., during post-mortem of a finished run).
3. **Multi-run situational awareness.** Today's only way to see "how are my 2 runs doing" is to open 2 browser tabs or run 2 separate `tail -f` sessions. The TUI gives one ground-truth view.

User-requested attributes: **robust, succinct, elegant.** That eliminates "big multi-screen Textual app with 12 widgets" — the design below is one screen, six panes, no modals.

---

## Architecture

```
                ┌─────────────────────────────────┐
                │  openresearch tui  (Textual app)    │
                ├─────────────────────────────────┤
                │  RunList     (left, polls       │
                │              runs/ dir)         │
                │  RunDetail   (right, shows      │
                │              selected run)      │
                │   ├─ Overview                   │
                │   ├─ Logs (tail)                │
                │   ├─ Events (tail)              │
                │   ├─ Costs / GPU                │
                │   └─ Rubric                     │
                └─────────────────────────────────┘
                       │           │          │
                       ▼           ▼          ▼
                  runs/*       nvidia-smi  RunPod REST
                  (file watch) (subproc)   (/v1/pods)
```

### Tech stack
- **Textual** (PyPI: `textual`, MIT, well-maintained). Modern declarative TUI framework. Async event loop integrates naturally with the file-watch + HTTP-poll workflow. Reactive widgets re-render only when their underlying observed data changes.
- **watchfiles** (already in `requirements.txt` from the runtime stack) for inotify-based file change detection. Falls back to polling on systems without inotify.
- **httpx** (already in stack) for RunPod REST calls.

### Data sources (read-only)
| Pane | Source | Refresh strategy |
|---|---|---|
| Run list | `runs/*/demo_status.json` | inotify on `runs/`; 5 s polling fallback |
| Overview | `runs/<id>/demo_status.json` + `final_report.json` if present | inotify on the file |
| Logs | `runs/<id>/code/outputs/*/exec.log` | `tail -f` style, follow last file (newest mtime) |
| Events | `runs/<id>/dashboard_events.jsonl` | inotify, parse new lines only |
| Costs / GPU | `runs/<id>/cost_ledger.jsonl` + `nvidia-smi` subprocess (local) + RunPod REST `/v1/pods/{id}` (remote) | Ledger: inotify. GPU: 5 s polling. Pod: 30 s polling. |
| Rubric | `dashboard_events.jsonl::rubric_score` events + `final_report.json::rubric` when terminal | Same as events |

### Why file-based, not SSE-based
- **Backend independence.** A TUI that crashes when uvicorn is down is bad UX. File-based works always.
- **No new endpoints.** The web UI's `/runs/<id>/events` SSE stream is fine for the browser, but reusing it requires authentication negotiation, demo-secret handling, and SSE-client work that buys nothing over watching the same JSONL file the backend writes.
- **Replay is free.** Opening the TUI on a finished run replays everything from the file — no need for the backend to remember.

(A `--live` flag could opt into SSE for sub-second updates if a user actually needs that. v1 doesn't ship this.)

---

## Layout (one screen, six panes)

```
┌───────────────────────────────────────────────────────────────────────────────┐
│  openresearch TUI                                                  q quit  ? help │
├──────────────┬────────────────────────────────────────────────────────────────┤
│ Active (2)   │ ▸ VAE  1312.6114                          rubric: pending     │
│ ● VAE        │ ──────────────────────────────────────────────────────────   │
│ ● Adam       │ Status:   running        Sandbox: local docker (RTX 2060)    │
│              │ Started:  21:00 UTC      GPU:     30% util, 939/6144 MiB     │
│ Recent (3)   │ Cost:     $0.00 (OAuth)  Iter:    1 (run_experiment active)  │
│ ○ Dropout    │ Wall:     1h 43m         Pod:     —                          │
│ ○ Adam(old)  │                                                              │
│ ○ Dropout(o) │ ─── Logs ────────────────────────────────────────────────── │
│              │ [mnist_nz20] epoch=8/100  train_elbo=-128.62  test_elbo=-127 │
│              │ [mnist_nz20] epoch=9/100  train_elbo=-127.81  test_elbo=-126 │
│              │ [mnist_nz20] epoch=10/100 train_elbo=-127.04  test_elbo=-126│
│              │                                                              │
│              │ ─── Rubric (latest) ──────────────────────────────────────── │
│              │ Method fidelity      ▓▓▓▓▓▓▓▓▓░  0.90                        │
│              │ Result match         (pending — run_experiment in progress)  │
│              │ Data fidelity        ▓▓▓▓▓▓░░░░  0.60                        │
│              │                                                              │
│              │ [1] Overview  [2] Logs  [3] Events  [4] Costs  [5] Rubric  │
└──────────────┴────────────────────────────────────────────────────────────────┘
```

### Pane responsibilities

**RunList (left, 14 cols wide)**
- Two sections: `Active` (status=running) above `Recent` (last 8 by completed_at desc).
- Filled circle `●` for selected, hollow `○` otherwise.
- Up/down arrows or `j`/`k` to navigate; Enter selects.
- Red dot for failed runs; green check for completed.

**RunDetail header (right, top 5 lines)**
- Paper title + arXiv id + rubric headline (or "pending" when no rubric_score event yet).
- Status / sandbox / GPU summary / cost / iter / wall-clock — one key:value per line, two columns.
- Sandbox row varies:
  - local docker: shows GPU name + util/mem from nvidia-smi
  - RunPod: shows pod id + $/hr + GPU type from `/v1/pods/{id}`
  - local: shows hostname

**Tabbed body** (right, below header, switched via `1`–`5`)
- **[1] Overview** — paperMeta JSON pretty-printed; scope (requested / ran / gaps); models used; primitive call counts.
- **[2] Logs** — tail of the latest `exec.log` under `code/outputs/*/exec.log`. Auto-scrolls. ANSI colors preserved.
- **[3] Events** — latest 50 lines of `dashboard_events.jsonl`, summarized: `<ts> <event_type> <primitive?> <status?>`. Color-coded by event type.
- **[4] Costs / GPU** — cost ledger sum by primitive (table), live GPU util sparkline (local or RunPod-derived), pod info (when applicable).
- **[5] Rubric** — area bars (filled bar at score, paper-target hairline), per-area scores, when `compute_scope.is_clipped`: shows both raw + compute_adjusted (once that ships).

### Keybindings
```
q              quit
j / down       next run in list
k / up         prev run in list
1 2 3 4 5      switch tabs in detail pane
g              jump to top of current pane
G              jump to bottom of current pane
/              filter run list (substring on paper title or project_id)
r              force refresh (rare — most updates are inotify-driven)
?              show help overlay
```

---

## CLI surface

```
openresearch tui [PROJECT_ID]
  --runs-root PATH       Override runs/ location (default from settings/env)
  --refresh N            Polling interval seconds (default 5)
  --no-color             Disable ANSI colors
```

- No arg: open multi-run dashboard.
- `PROJECT_ID` arg: open the dashboard with that run pre-selected and visible (skip the manual navigation step).

Entry point: add a new Click command to `backend/cli.py` next to `reproduce` / `ingest`:

```python
@cli.command(name="tui")
@click.argument("project_id", required=False)
@click.option("--runs-root", default=None, type=click.Path(file_okay=False))
@click.option("--refresh", default=5, type=int)
@click.option("--no-color", is_flag=True)
def cmd_tui(project_id, runs_root, refresh, no_color):
    """Launch the terminal runs monitor."""
    from backend.cli_tui.app import OpenResearchTUI
    OpenResearchTUI(runs_root=runs_root, project_id=project_id, refresh_s=refresh,
                color=not no_color).run()
```

---

## File structure (new)

```
backend/cli_tui/
├── __init__.py
├── app.py            # OpenResearchTUI Textual app + main bindings
├── run_list.py       # RunList widget — left sidebar
├── run_detail.py     # RunDetail widget — right pane + tabs
├── data/
│   ├── runs_index.py        # discovers runs/*, exposes a reactive list
│   ├── run_state.py         # loads + caches one run's demo_status + final_report
│   ├── log_tailer.py        # async file tailer for exec.log + events.jsonl
│   ├── gpu_probe.py         # nvidia-smi subprocess wrapper (local) + RunPod REST (remote)
│   └── runpod_probe.py      # /v1/pods/{id} polling
└── styles.tcss        # Textual CSS for the layout (one file, ~80 lines)
```

Tests:
```
tests/cli_tui/
├── test_runs_index.py       # discovers runs, classifies active vs recent
├── test_run_state.py        # parses demo_status + final_report shapes
├── test_log_tailer.py       # file tail with rotation handling
├── test_gpu_probe.py        # mock nvidia-smi output parsing
└── test_runpod_probe.py     # mock /v1/pods/{id} response
```

UI widgets are not unit-tested at the pixel level (Textual's snapshot tooling is heavy for the value). The data layer has full unit coverage; the widget layer is exercised by a single integration test that boots the app against a fixture `runs/` and asserts no exceptions.

---

## Edge cases & robustness

| Case | Behavior |
|---|---|
| `runs/` empty | RunList shows "No runs yet. Start one with `openresearch reproduce <paper>`." |
| `runs/<id>/demo_status.json` malformed | Show the run with status="error reading status"; don't crash. |
| `exec.log` rotated (deleted + recreated) | log_tailer reopens; UI shows a "—— log rotated ——" separator. |
| `runs/<id>` deleted while selected | Selected run shows "[deleted]" banner; auto-select first active or None. |
| RunPod API returns 401 | Pod pane shows "RunPod auth missing — check OPENRESEARCH_RUNPOD_API_KEY". TUI keeps running. |
| nvidia-smi missing | GPU pane shows "no NVIDIA GPU detected". |
| Backend writing demo_status mid-read | Atomic-write contract is the backend's existing invariant (already true); we read once per inotify event, retry on JSONDecodeError. |
| Very long log lines | Wrap at terminal width; don't truncate (it's a log viewer). |
| 100+ runs in `runs/` | Limit RunList active to 50 (sort by updatedAt desc); Recent capped at 8 visible, more via `j` past the end (lazy load). |
| Terminal resize | Textual handles natively; layout flexes (CSS-like). |

---

## What's intentionally OUT of scope

- **Launching runs from inside the TUI.** Read-only viewer keeps the surface tight. New runs use CLI or web UI.
- **Killing runs from inside the TUI.** Same reason.
- **Editing rubric, replaying events, manual checkpoint restore.** Web UI handles those when needed.
- **Comparing two runs side-by-side.** Single-run detail only. Comparing belongs in the leaderboard view (web UI).
- **Pod-side nvidia-smi push.** RunPod GPU util isn't surfaced by the REST API; getting live util requires the pod-side heartbeat daemon to also write nvidia-smi snapshots and us to scp them back. Worthwhile follow-up but adds SSH dependency to the TUI — not v1.

---

## Backward compat / risk

- Pure additive: new `backend/cli_tui/` package + new CLI command. No existing code path changes.
- Adds **textual** dep. PyPI package, ~5 MB, vendor-friendly license (MIT), well-maintained.
- Adds **watchfiles** dep (or reuses one if already present — to confirm during impl).

---

## Phasing (1 PR, 5 commits)

| Phase | Output | Tests |
|---|---|---|
| 1 | Data layer (`runs_index`, `run_state`, `log_tailer`, `gpu_probe`, `runpod_probe`) — all pure-Python, no Textual yet | 5 test files, full unit coverage |
| 2 | Textual app skeleton + RunList widget — boots, lists runs, navigation works | smoke test that app starts |
| 3 | RunDetail header + Overview tab | snapshot of header rendering |
| 4 | Logs / Events / Costs / Rubric tabs | data flow into widgets |
| 5 | CLI integration (`openresearch tui`), keybindings, help overlay, docs in README | end-to-end smoke against fixture runs |

Codex adversarial review after phase 5: file race conditions, nvidia-smi parsing robustness, RunPod auth error paths.

---

## Open questions

1. **`textual` dependency acceptable?** It's the modern Python TUI standard (Posit, Anthropic, Pomerium use it). Alternative is Rich Live + Layout which is simpler but doesn't handle interactive navigation as cleanly. Recommend Textual.
2. **Auto-select first active run on start?** Default yes. If you'd rather start on the most-recent-completed (faster to inspect last-night's run), say so.
3. **Default refresh cadence**: 5 s feels right for a passive observer. Aggressive observers can pass `--refresh 1`. inotify-driven updates are independent of this and always fire immediately.
4. **GPU util sparkline**: how many seconds of history (default 60) — flag-tunable in v2.
