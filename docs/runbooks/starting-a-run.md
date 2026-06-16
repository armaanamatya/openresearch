# Starting a run

Copy-paste guide to launch a OpenResearch reproduction **and** confirm the whole
workflow (ingest → understand → plan → implement → execute → verify → report)
actually ran end-to-end.

> This is the operational "do this" doc. For the *why* — the full sandbox×prerequisite
> matrix, what each pipeline stage does, and the architecture — see
> [running-the-project.md](running-the-project.md).

## Engine note (read once): OrbStack and Docker Desktop are equivalent

This repo never checks for "Docker Desktop" specifically. The code talks to the
**Docker socket** via `docker.from_env()` + ping. Both Docker Desktop and
OrbStack expose that socket identically, so a single check — `docker info` — is
the source of truth for **either** engine. One teammate on Docker Desktop and
one on OrbStack are fully interchangeable. Anywhere this doc says "engine up,"
it means `docker info` returns 0; it does **not** mean a particular product.

> The default sandbox is `runpod`. Since `875995c`, `build_environment`
> short-circuits to a no-op for **both `local` and `runpod`** — only `docker`
> and `auto`/unknown do a **local** `docker build`. So the engine must be up
> only for `--sandbox docker`/`auto` runs; `local` and `runpod` need no daemon.

Convention used below: credentials live in **`.env`** (or OAuth via
`claude login`); your **shell stays empty** of `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` / `OPENRESEARCH_RUNPOD_API_KEY`. Every launch is prefixed with
`env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY` so the `.env` values win (a stale
shell export silently shadows `.env` — BUG-LR-014). This is why the
shell-empty guard below and your credential setup do **not** conflict.

---

## Pre-run readiness checklist

Each item is a one-line **CHECK** (exits 0 when ready) plus the **FIX** if it
fails. Run from the repo root (`/Volumes/CS_Stuff/openresearch`).

1. **Docker engine up (OrbStack OR Docker Desktop)** — needed for `docker` and
   `runpod`; skip only for `--sandbox local`.
   - CHECK: `docker info >/dev/null 2>&1`
   - FIX: start your engine — OrbStack: `open -a OrbStack`; Docker Desktop:
     `open -a Docker` (macOS) / `systemctl start docker` (Linux). Re-verify with
     `docker info`. Do **not** look for a "Docker Desktop" string; the socket is
     what matters and both engines answer it.

2. **Python venv + deps importable** (FastAPI app constructs, RLM engine +
   provider SDKs + Docker SDK present).
   - CHECK: `.venv/bin/python -c "from backend.app import create_app; create_app()" >/dev/null 2>&1`
   - FIX: `python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt`
     (note: `rlms` pulls `pytest>=9`; install `requirements.txt` alone — see
     CLAUDE.md "Local multi-GPU sandbox" gotcha).

3. **Docker Python SDK importable** (a live daemon is not enough; the `docker`
   package must be in the venv or builds raise `backend_unavailable`).
   - CHECK: `.venv/bin/python -c "import docker; docker.__version__" >/dev/null 2>&1`
   - FIX: `.venv/bin/pip install -r backend/requirements.txt`

4. **One root-model credential** (the RLM root loop). Preferred: OAuth, which
   needs **no key** and also covers sub-agents.
   - CHECK (OAuth): `claude --print ping >/dev/null 2>&1`
   - CHECK (alt, API key in `.env`): `grep -Eq '^(OPENAI_API_KEY|ANTHROPIC_API_KEY|FEATHERLESS_API_KEY)=.+' .env`
   - FIX: run `claude login` once (free, subscription). OR put exactly one key in
     **`.env`** (not your shell): `OPENAI_API_KEY=sk-…` for `--model gpt-5`
     (~$1/run), `FEATHERLESS_API_KEY=…` for `--model qwen3-coder-featherless`
     (cheapest), or a **funded** `ANTHROPIC_API_KEY` for `--model claude`.

5. **Claude sub-agent OAuth** (`implement_baseline`, `verify_against_rubric`, …
   via claude-agent-sdk). OAuth is free on subscription.
   - CHECK: `claude --print ping >/dev/null 2>&1`
   - FIX: `claude login` once. On macOS this stores creds in the Keychain
     (probed via `security find-generic-password -s "Claude Code-credentials"`),
     not `~/.claude/.credentials.json` — both paths are handled.
   - WARNING: a **no-credit** `ANTHROPIC_API_KEY` does **not** fall back to OAuth
     — the SDK tries the key, hits `400 credit balance too low`, and the run dies
     at the first Sonnet call with `cost_usd=0.0`. `claude --print ping` proves
     only the *subscription*, never the *API key's balance*. For safe local dev,
     leave `ANTHROPIC_API_KEY` empty.

6. **RunPod credentials — ONLY for `--sandbox runpod`.** Three gates, all
   required: engine up (item 1), an API key, and an SSH key (`create_sandbox`
   raises `backend_unavailable` if the SSH key is missing).
   - CHECK (API key): `grep -Eq '^OPENRESEARCH_RUNPOD_API_KEY=.+' .env`
   - CHECK (SSH key): `test -f "${OPENRESEARCH_RUNPOD_SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"`
   - FIX: get a key at <https://console.runpod.io/account/api-keys>, set
     `OPENRESEARCH_RUNPOD_API_KEY=…` in `.env`. Create a key if absent:
     `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ''` (or point
     `OPENRESEARCH_RUNPOD_SSH_KEY_PATH` at an existing one). Preflight:
     `scripts/runpod_check.sh`.

7. **Shell-shadows-.env guard** — your shell must NOT export the three keys, or
   a stale value silently shadows `.env`.
   - CHECK: `[ -z "$OPENAI_API_KEY$ANTHROPIC_API_KEY$OPENRESEARCH_RUNPOD_API_KEY" ]`
   - FIX: `unset OPENAI_API_KEY ANTHROPIC_API_KEY OPENRESEARCH_RUNPOD_API_KEY` for
     the session, OR always launch with the
     `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY` prefix (used below) so `.env`
     wins regardless.

---

## One-shot readiness command

Paste this block from the repo root. It prints a single **READY** /
**NOT-READY** verdict. The RunPod lines are commented out — uncomment them only
when launching `--sandbox runpod`.

```bash
# Wrapped in a subshell ( ... ) so a NOT-READY `exit` ends the check, NOT your terminal.
( cd /Volumes/CS_Stuff/openresearch || exit 1
  docker info >/dev/null 2>&1 \
    || { echo "NOT-READY: docker engine down — start OrbStack or Docker Desktop, then 'docker info' (skip only for --sandbox local)"; exit 1; }
  .venv/bin/python -c "from backend.app import create_app; create_app()" >/dev/null 2>&1 \
    || { echo "NOT-READY: venv/deps — run: .venv/bin/pip install -r backend/requirements.txt"; exit 1; }
  .venv/bin/python -c "import docker" >/dev/null 2>&1 \
    || { echo "NOT-READY: docker python SDK missing — .venv/bin/pip install -r backend/requirements.txt"; exit 1; }
  { claude --print ping >/dev/null 2>&1 \
      || grep -Eq '^(OPENAI_API_KEY|ANTHROPIC_API_KEY|FEATHERLESS_API_KEY)=.+' .env; } \
    || { echo "NOT-READY: no credential — run 'claude login' (free) or put one key in .env"; exit 1; }
  claude --print ping >/dev/null 2>&1 \
    || echo "WARN: 'claude login' not active — sub-agents will rely on a FUNDED ANTHROPIC_API_KEY (no-credit key dies at first Sonnet call)"
  [ -z "$OPENAI_API_KEY$ANTHROPIC_API_KEY$OPENRESEARCH_RUNPOD_API_KEY" ] \
    || echo "WARN: shell exports a key — launch with 'env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY' so .env wins (BUG-LR-014)"

  # --- uncomment for --sandbox runpod only ---
  # grep -Eq '^OPENRESEARCH_RUNPOD_API_KEY=.+' .env \
  #   || { echo "NOT-READY: OPENRESEARCH_RUNPOD_API_KEY missing in .env"; exit 1; }
  # test -f "${OPENRESEARCH_RUNPOD_SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}" \
  #   || { echo "NOT-READY: RunPod SSH key missing — ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ''"; exit 1; }

  echo "READY"
)
```

---

## Launch commands

Baseline test paper: **SDAR — arXiv `2605.15155`**. "Smallest-two scope" is not a
CLI flag — it is passed as guidance via `OPENRESEARCH_BASELINE_EXTRA_GUIDANCE` (pin
Qwen3-1.7B + Qwen2.5-3B), optionally reinforced with `--paper-hint 2605.15155`.
Every command uses the `env -u …` prefix so `.env` (or OAuth) wins over any
stale shell export, and `--model claude-oauth` (free, covers both auth surfaces).
Swap in `--model gpt-5` (key in `.env`, ~$1/run) if you prefer the OpenAI root.

### `--sandbox local` — free, no Docker, no RunPod (host subprocess; needs local GPU for GPU papers)

```bash
cd /Volumes/CS_Stuff/openresearch && \
OPENRESEARCH_BASELINE_EXTRA_GUIDANCE="Reproduce only the two smallest variants: Qwen3-1.7B and Qwen2.5-3B. Single GPU." \
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  .venv/bin/python -m backend.cli reproduce 2605.15155 \
    --mode rlm --sandbox local --model claude-oauth \
    --paper-hint 2605.15155 \
    --max-usd 5 --max-wall-clock 7200
```

### `--sandbox docker` — CPU, needs the engine up (OrbStack OR Docker Desktop)

```bash
cd /Volumes/CS_Stuff/openresearch && \
OPENRESEARCH_BASELINE_EXTRA_GUIDANCE="Reproduce only the two smallest variants: Qwen3-1.7B and Qwen2.5-3B. Single GPU." \
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  .venv/bin/python -m backend.cli reproduce 2605.15155 \
    --mode rlm --sandbox docker --model claude-oauth \
    --paper-hint 2605.15155 \
    --preflight-sanity \
    --max-usd 5 --max-wall-clock 7200
```

### `--sandbox runpod` — GPU, needs RunPod creds (API key + SSH key); no local Docker engine (`875995c`)

```bash
cd /Volumes/CS_Stuff/openresearch && \
OPENRESEARCH_BASELINE_EXTRA_GUIDANCE="Reproduce only the two smallest variants: Qwen3-1.7B and Qwen2.5-3B. Single GPU." \
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u OPENRESEARCH_RUNPOD_API_KEY \
  .venv/bin/python -m backend.cli reproduce 2605.15155 \
    --mode rlm --sandbox runpod --model claude-oauth \
    --paper-hint 2605.15155 \
    --preflight-sanity \
    --max-usd 8 --max-wall-clock 10800
```

Notes:
- The engine must be up only for `docker` and `auto`/unknown (local `docker build`
  in `build_environment`); `local` **and** `runpod` short-circuit it (`875995c`).
- `--preflight-sanity` runs a short sandbox smoke test before the real run for
  `local`/`docker` (skip is automatic for runpod transient-capacity reasons).
- Pin a specific id with `--project-id <id>` if you want to know the run dir up
  front; otherwise the CLI prints `runs/<project_id>/` at startup.
- Default RunPod image is `cuda-devel` (`runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`,
  `runpod_backend.py::DEFAULT_RUNPOD_IMAGE`). The lighter `cuda-runtime` was reverted (`88c45b0`):
  it lacks dev headers, so `bitsandbytes`/`flash-attn`/`deepspeed` installs failed silently. Override
  with `OPENRESEARCH_RUNPOD_IMAGE=...-runtime-...` only if all deps are pre-built wheels.

---

## Confirm the whole workflow ran

Run artifacts live **either** flat at the run root **or** nested under
`attempts/<latest>/`, depending on the run. Resolve whichever actually holds the
final report first:

```bash
RUN=runs/<project_id>                       # printed at launch
ATT="$RUN"                                  # most runs are flat
if [ -d "$RUN/attempts" ]; then             # some runs nest per-attempt artifacts
  _latest="$RUN/attempts/$(ls -1 "$RUN/attempts" | sort | tail -1)"
  [ -f "$_latest/final_report.json" ] && ATT="$_latest"
fi
echo "artifact dir: $ATT"
```

**1. The proof-of-real-run pair (forged-evidence cross-check).** A genuine
EXECUTE phase leaves a `success=true` + non-empty `metrics` row in
`experiment_runs.jsonl` **AND** ≥1 `run_experiment` row in the (unforgeable)
cost ledger. A metrics row with **zero** ledger calls means the row was hand
-written into the JSONL and the report will be downgraded to `failed`.

```bash
EV=$([ -f "$ATT/experiment_runs.jsonl" ] && jq -s 'map(select(.success==true and (.metrics|length>0)))|length' "$ATT/experiment_runs.jsonl" || echo 0)
CALLS=$([ -f "$ATT/cost_ledger.jsonl" ] && grep -c '"agent_id": "run_experiment"' "$ATT/cost_ledger.jsonl" || echo 0)
echo "success+metrics rows=$EV   run_experiment ledger calls=$CALLS"
[ "$EV" -gt 0 ] && [ "$CALLS" -ge 1 ] && echo "REAL RUN" \
  || echo "NOT A REAL EXECUTE (forged, or no experiment ran — e.g. the run died before run_experiment)"
```

**2. Final-report verdict + that the metrics are backed by a real call.**

```bash
jq '{verdict,
     baseline_metrics_n: (.baseline_metrics | length),
     run_experiment_in_trace: (.primitive_trace.by_primitive.run_experiment // 0),
     overall_score: .rubric.overall_score,
     scope_ran: (.scope.ran | length)}' "$ATT/final_report.json"
# verdict must be reproduced|partial|failed; for a non-failed verdict,
# run_experiment_in_trace must be >= 1.
```

**3. Cost-ledger has `run_experiment` spend** (already counted above as
`$CALLS`); inspect the rows if you want the per-call detail:

```bash
grep '"agent_id": "run_experiment"' "$ATT/cost_ledger.jsonl" | head
```

**4. Replay postmortem** (`scripts/replay_run.py` is present in this repo). Pass
the **run root**, not the attempt dir; it flags evidence/verdict mismatch,
dangling `sub_rlm_spawned` without `complete`, and event-gap stalls.

```bash
.venv/bin/python scripts/replay_run.py "$RUN"
# Healthy: no evidence_verdict_mismatch, dangling_subcalls == 0,
#          max_event_gap_seconds < 120.
```

**5. Preserved marker** (written only when `verdict != failed`):

```bash
test -f "$ATT/.preserved" && echo "preserved (non-failed)" || echo "no marker (failed run)"
```

---

## Failure-mode → fix

| Symptom (where it surfaces) | Class | Discriminator (run this) | Fix |
|---|---|---|---|
| `SandboxRuntimeError(backend_unavailable)` at **build_environment**; "Docker daemon is not reachable" | **Engine / Docker down** | `docker info` exits non-zero | Start OrbStack **or** Docker Desktop; re-verify `docker info`. Engine-agnostic — do not look for "Docker Desktop" specifically. Only reachable under `--sandbox docker`/`auto` (since `875995c`, `runpod` short-circuits build_environment — a runpod run can't hit this). |
| Same error but `docker info` **works** | **Docker SDK missing** | `.venv/bin/python -c "import docker"` fails | `.venv/bin/pip install -r backend/requirements.txt` |
| `implement_baseline` returns no `code_path`, empty files, `success` None/empty, **no error_code** | **claude-agent-sdk AUTH wedge ("success-with-no-text")** — subprocess ran, wrote nothing; NOT Docker | `claude --print ping` (OAuth) AND `test -w "$RUN/code"` (writable). If ping fails or `ANTHROPIC_API_KEY` is a no-credit key → auth. | `claude login` (or `claude login --reset`). Leave `ANTHROPIC_API_KEY` empty so OAuth is used; a no-credit key never falls back. Check quota if subscription-exhausted. |
| Run dies at first Sonnet call: `cost_usd=0.0`, `400 credit balance too low` | **AUTH — no-credit API key** | `ANTHROPIC_API_KEY` is set but unfunded | Empty the key + `claude login`, or fund the Anthropic **API** account. |
| `RunPod API key is missing` / `401` at **pod creation** (before any pod work) | **RunPod auth** | `grep -Eq '^OPENRESEARCH_RUNPOD_API_KEY=.+' .env` fails, or shell shadows it | Set `OPENRESEARCH_RUNPOD_API_KEY` in `.env`; prefix launch with `env -u OPENRESEARCH_RUNPOD_API_KEY`. |
| `SSH private key not found` at pod creation | **RunPod SSH** | `test -f "${OPENRESEARCH_RUNPOD_SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"` fails | `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ''` or set `OPENRESEARCH_RUNPOD_SSH_KEY_PATH`. |
| Run routes to wrong provider / unexpected `401` while `.env` looks right | **Shell shadows .env (BUG-LR-014)** | `env \| grep -E 'OPENAI_API_KEY\|ANTHROPIC_API_KEY\|OPENRESEARCH_RUNPOD_API_KEY'` shows an export | Launch with `env -u …` prefix, or `unset` the vars for the session. |
| `no root model could be resolved` at startup | **No root credential** | none of OPENAI/ANTHROPIC/FEATHERLESS in `.env` and `claude --print ping` fails | Set one key in `.env` or `claude login`. |
| `parsed_full_text.txt missing — parser likely failed` | **Ingest/parse** | `test -f "$RUN/parsed_full_text.txt" && [ $(stat -f%z "$RUN/parsed_full_text.txt") -gt 1024 ]` fails | Re-run `.venv/bin/python -m backend.cli ingest <pdf>`; image-only PDFs need OCR; check `parsing_failed` events. |
| `dockerfile parse error … unknown instruction: You've` | **Sub-agent prose-stomp (BUG-NEW-042)** | first non-blank Dockerfile line is not `FROM`/`ARG`/`# syntax=` | Repairable — the shape guard restores the snapshot and the root retries `implement_baseline`. |

---

## Known gaps

Things that can still pass quietly / mislead — check these when a run "succeeds"
but feels wrong:

- **Lossy paper text proceeds by default.** `OPENRESEARCH_ALLOW_LOSSY_PAPER_TEXT`
  defaults **true**: if `parsed_full_text.txt` is <1 KB the run degrades to the
  workspace fallback instead of failing. Set it `false` to hard-fail on bad
  ingest. Always sanity-check `stat -f%z "$RUN/parsed_full_text.txt"`.
- **No-credit `ANTHROPIC_API_KEY` does NOT fall back to OAuth.** It dies at the
  first Sonnet call. `claude --print ping` passing proves only the subscription,
  never the API key's balance. Safest: keep the key empty.
- **Shell shadowing warns but does not block.** The CLI prints a warning yet
  runs; the `env -u …` prefix is the real guard.
- **Engine NOT required for `runpod` (since `875995c`).** `build_environment`
  short-circuits to a no-op under `runpod` (the pod boots its own image over SSH),
  so a down Docker daemon no longer breaks RunPod runs — the local daemon is needed
  only for `--sandbox docker`/`auto`. (Was a rough edge; resolved in the merge.)
- **RunPod default image is `cuda-devel`.** The lighter `cuda-runtime` was reverted
  (`88c45b0`) — it lacks dev headers, so `bitsandbytes`/`flash-attn`/`deepspeed`
  installs failed silently. Override to `-runtime-` via `OPENRESEARCH_RUNPOD_IMAGE` only
  if all deps are pre-built wheels.
- **`local` sandbox depends entirely on the host venv/GPU.** No Docker isolation
  — missing paper deps or CUDA on the host fail the run; install paper deps into
  `.venv` (the batch scheduler does this per-run automatically).
- **`--max-usd` / `--max-wall-clock` bound the whole run, not each primitive.**
  `implement_baseline` has its own independent 4-hour per-call ceiling; a wedged
  sub-agent can burn time/quota up to that even under a tight run budget.
- **Forged-evidence gate is content+ledger only.** It catches hand-written JSONL
  rows, but it cannot tell a *correct* reproduction from a poor one — that is the
  rubric's job; read `rubric.overall_score` and `scope.gaps`, not just `verdict`.
