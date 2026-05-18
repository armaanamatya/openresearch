# Railway YOLO Launch Runbook — openresearch

**Date written:** 2026-05-18
**Branch this targets:** `replix_merge`
**Railway project / service:** `hearty-luck` / `openresearch`
**Live URL:** https://openresearch-production.up.railway.app
**Access model:** public, no demo-secret gate (yolo — cost exposure accepted)

This runbook is the delta + firefighting reference for this launch. The base
spec is `docs/deployment.md` — read it first. This file does NOT re-derive the
architecture; it captures what's different, the operational state observed on
2026-05-18, and what to do when something breaks.

---

## Snapshot of live state (2026-05-18)

| Item | State | Notes |
|---|---|---|
| URL | openresearch-production.up.railway.app | 200 on `/lab` |
| Frontend | Next.js | renders cleanly, zero console errors/warnings |
| Backend | FastAPI on internal `:8000` | reachable via `/api/pipeline/topology` proxy |
| Volume | mounted at `/app/runs` | empty, "No recent runs" in sidebar |
| Sandbox | local-process (`REPROLAB_FORCE_SANDBOX=local`) | offline e2e (`backend.cli reproduce --mode offline --sandbox local`) passed exit 0, reached `gate_2_passed` |
| Demo-secret gate | disabled (`REPROLAB_DEMO_SECRET=""`) | yolo |
| Deployed code | stale — pre-`replix_merge` commits | Railway branch tracking is broken; fix below |
| Target port | 8080 | divergent from `deployment.md`'s 3000 recommendation; works because Railway sets `PORT=8080` and the entrypoint binds Next.js to `${PORT:-3000}` |
| Replicas | 1 | mandated by single-volume constraint — don't bump |

---

## Procedure: reconnect branch + redeploy

The deployed image lags GitHub HEAD because Railway can't see the configured
branch ("GitHub Repo not found" on the Source panel). The Source Repo
connection itself works — only the branch lookup is broken.

1. **Confirm the fix is on GitHub.** `git log origin/replix_merge -1` should
   show `170be53 fix(web): default rawError to "" so production Next build typechecks`
   at HEAD. (Built on top of `3c60b13` merge.)
2. Railway Dashboard → service `openresearch` → **Settings → Source**.
3. Click the pencil/edit icon next to **Source Repo** OR find the branch
   selector under "Branch connected to production". Pick **`replix_merge`**.
   Save.
4. Railway auto-triggers a new build on save. Watch the **Deploys** tab — the
   build that previously failed at `[frontend 7/7]` on the rawError TS error
   will now complete (the fix is on `replix_merge` HEAD).
5. After deploy succeeds, run the verification probes below.

---

## Env vars (paste-ready)

KEY=VALUE only, no comments, no section headers — Railway's env editor
parses bare-line format.

```
ANTHROPIC_API_KEY=<your-key>
REPROLAB_DATABASE_URL=sqlite:////app/runs/reprolab.db
REPROLAB_BACKEND_URL=http://127.0.0.1:8000
REPROLAB_FORCE_SANDBOX=local
REPROLAB_PROVIDER_FALLBACK_DISABLED=true
REPROLAB_DEMO_SECRET=
PORT=8080
```

Deltas vs `docs/deployment.md`'s recommended set:

- `REPROLAB_DEMO_SECRET=` (empty) — disables the gate. **Set to any random
  string and redeploy** to flip the kill switch (see Cost Monitoring).
- `REPROLAB_PROVIDER_FALLBACK_DISABLED=true` — avoids the OPENAI 401 trap that
  surfaces a misleading auth error when Anthropic blips and the chain tries to
  fall back to an invalid OpenAI key. From commit `70441c4`.
- `PORT=8080` — matches your existing Railway target-port. The
  `deployment.md` recommends 3000; both work as long as the env var matches
  the target port.

---

## Verification probes (after redeploy)

```bash
BASE=https://openresearch-production.up.railway.app

# Railway healthcheck (frontend fallback — see Gotcha #1)
curl -sS $BASE/health
# expect: {"status":"ok"}

# REAL liveness — proxies through Next.js to FastAPI
curl -sS $BASE/api/pipeline/topology | head -c 200
# expect: JSON beginning {"nodes":[{"id":"src",...

# Models list (also a real proxy probe)
curl -sS $BASE/api/models
# expect: [{"id":"sonnet",...},{"id":"opus",...}]

# UI loads
curl -sIS $BASE/lab | head -1
# expect: HTTP/2 200
```

If `/api/pipeline/topology` returns 500 or HTML, the backend is dead even
though `/health` is green — that's Gotcha #1.

---

## Cost monitoring (yolo guard-rails)

The codebase has a per-run `--max-usd` cap. There is **no global daily cap**,
no per-IP rate limit, no concurrent-run throttle. A full 14-stage run with
parallel improvement paths costs ~$15–30 in Anthropic spend (measured: prior
RLM run was $5.78 at gate_2_passed, partial).

Passive guard-rails — set in dashboards, not code:

1. **Anthropic Console → Spend alerts** — $50 daily warning, $200 daily
   critical. Email-only; doesn't stop spend, but you'll know.
2. **Railway → Service → Settings → Replica Limits** — cap **memory at 2 GB**
   (current default 8 GB). OOM-kills runaway concurrent runs before they hose
   the container.
3. **Kill switch** — when spend spikes, set in Railway env vars:
   ```
   REPROLAB_DEMO_SECRET=<random>
   ```
   Save & redeploy. Existing `_enforce_demo_gate` (backend) + `proxy.ts`
   (frontend) immediately lock the site. To re-open, clear the var.
4. **(Optional) Frontend disclaimer** — a one-line `<small>` on the home
   page: *"Each reproduction run uses ~$5–30 of LLM budget and 15–75 min.
   Please be considerate."* Discourages spam, no new code paths.

---

## Known gotchas

### 1. `/health` is a frontend-only fallback

`frontend/src/app/health/route.ts:20` returns `{"status":"ok"}` even when the
backend probe fails. Railway's green check means Next.js is alive, NOT
FastAPI. Use `/api/pipeline/topology` as the real liveness signal.

### 2. TS errors slip past `next dev`

Next dev's hot reload doesn't run the full prod build. Always run from
`frontend/`:
```bash
npx tsc --noEmit
```
before push. The 2026-05-18 incident was a `null` fallback in FailurePanel's
`rawError` prop — fixed in `170be53`.

### 3. Local-process sandbox runs experiment code IN the Railway container

`REPROLAB_FORCE_SANDBOX=local` means agent-generated commands `exec` in the
same process tree as the orchestrator. **Not an isolation boundary.**
Acceptable for the demo because runs are bounded by cost cap and Railway
memory limit; if you ever care about isolation, flip to RunPod (see
deployment.md §"RunPod (remote GPU)").

### 4. SSE under 75-min runs is untested on Railway edge

Heartbeat exists (`live_runs.py:365`) and `X-Accel-Buffering: no` is set, so
it *should* survive Railway's edge proxy. Unverified at real-run scale. If
the UI appears to hang mid-pipeline, check whether SSE was dropped before the
run finished — backend logs will show the run still progressing.

### 5. `numReplicas: 1` is mandatory

Railway blocks two deployments mounting one volume. Redeploys are a brief
outage; that's the trade. To horizontally scale you'd need Postgres + object
storage for `runs/`. Out of scope for v1.

### 6. Stale image after broken branch tracking

If `git push` doesn't trigger a Railway build, the branch selector is wrong.
See Procedure section above. Don't trust the "You're on the latest version"
banner — that refers to the upstream-template sync, not the deployed branch.

---

## Deferred from `docs/deployment.md`'s pre-prod checklist

Explicitly NOT blocking this launch:

- **Schema consolidation** — the `reprolab.db.corrupt-*` files cited in the
  doc no longer exist; doc is stale on this. Defer.
- **Secrets manager** (Vault/Doppler/AWS SM) — Railway env vars are fine for
  one API key.
- **Log rotation on `runs/`** — accept unbounded growth; redeploy &
  re-volume before it hurts.
- **Worker queue** (Celery/ARQ) — subprocess model + 2 GB cap is enough.
- **RunPod credentials** — `sandbox=local` is the v1 path; RunPod is the v2
  upgrade for GPU papers.
- **Node version pinning** — Dockerfile already pins `node:20-bookworm-slim`.

---

## Failure-mode quick-reference

| Symptom | First check | Fix |
|---|---|---|
| Anthropic spend spike | `runs/` for recent project ids; cost_ledger.jsonl per project | Flip kill switch (REPROLAB_DEMO_SECRET=anything → redeploy) |
| Container OOMs | Railway logs for "exit 137" / "killed" | Lower memory cap; reduce concurrent runs |
| SSE stalls mid-run | `/api/pipeline/topology` to confirm backend up | Refresh; if backend is fine, edge dropped SSE — heartbeat should auto-recover |
| Build fails on push | Railway Deploys → latest → Build logs | 99% of the time it's a TS error; run `npx tsc --noEmit` locally |
| `/health` green but UI broken | Hit `/api/pipeline/topology` | If 500/HTML, backend died; check Railway runtime logs |
| Deploy stuck on old image | "Deploys" tab shows last build before recent push | Branch tracking lost; reconnect per Procedure section |
| OPENAI 401 mid-run | `cost_ledger.jsonl` for provider= openai entries | Confirm `REPROLAB_PROVIDER_FALLBACK_DISABLED=true` is set |
| Visitor reports stuck stage | Open the SSE stream directly: `curl -sN $BASE/api/demo/events/$PROJECT_ID` | If events are flowing, it's a UI/coalesce issue; if not, edge dropped or backend hung |

---

## Pointers

- Base spec: `docs/deployment.md`
- 14-stage architecture: `system_overview.md`
- Architectural lessons + post-mortems: `learn.md`
- Cost ledger format: `backend/agents/resilience/cost.py`
- Per-run budget enforcement: `backend/agents/resilience/budget.py`
- Demo-secret gate: `backend/app.py:_enforce_demo_gate` + `frontend/src/proxy.ts`
- Sandbox override at request boundary: `backend/services/events/live_runs.py:apply_sandbox_override`
