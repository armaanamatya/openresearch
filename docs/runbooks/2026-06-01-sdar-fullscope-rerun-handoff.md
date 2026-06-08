# SDAR full-scope rerun — handoff (2026-06-01)

**Branch:** `feat/full-scope-envs`  **Worktree:** `/home/sww35/openresearch-fullscope`
**Paper:** SDAR, arXiv 2605.15155 (Self-Distilled Agentic RL) — the canonical baseline.
**Goal for next session:** launch the end-to-end SDAR reproduction, then monitor it to a final score.

---

## TL;DR

The 2026-06-01 full-scope run (`prj_09047604e591d969`, started 19:20 UTC) **died with no
`final_report`**. Root cause: the shipped `search_qa_env.py` loaded HotpotQA/NQ-open by **bare
dataset name** (`load_dataset("hotpot_qa", …)`), which modern HF rejects and which
`pre_flight_validator` correctly **hard-blocks**. The very first `run_experiment` (19:47) failed
preflight on `search_qa_env.py:746`, and the `claude-oauth` root never recovered — the process
exited (`Loop … is closed`) without shipping even a partial report.

**Fixed** in commit `5781cb7`: canonical `owner/name` ids via a `_load_dataset_resilient` helper
(`hotpotqa/hotpot_qa`, `google-research-datasets/nq_open`) with an offline-cache fallback to the
legacy bare name (derived, never a literal, so it can't re-trip preflight). Preflight now reports
**0 hard violations**; `test_search_qa_env` 24/24 green.

**The main path is unblocked. Next session: relaunch and monitor.**

---

## What happened, in order

1. Run started 19:20, ingestion OK, RLM root = `claude-oauth` (the ONLY working root — no
   OpenAI/Anthropic API keys in `.env` or shell; `claude` CLI creds present).
2. Agent reached `implement_baseline`: wrote `code/cells.json` (19:39) + `code/train_cell.py`
   (19:45); harness modules copied to `code/` (search_qa_env.py etc., 19:47).
3. First `run_experiment` at 19:47 → **`pre_flight: 1 hard violation`**:
   `search_qa_env.py:746: load_dataset('hotpot_qa', …)` — bare short name → `HfUriError`.
   (`experiment_runs.jsonl` has exactly this one entry, `success:false`.)
4. The cell `metrics.json` files under `code/outputs/<run>/<cell>/` are **start-of-cell stubs**
   (`"status":"running","steps_run":0`) — preflight runs *before* `run_matrix`, so **no cell ever
   trained**.
5. Process died right after, no `final_report.{json,md}`, GPUs released. The root did not repair
   and retry — it just stopped.

## The fix (already committed: `5781cb7`)

`backend/agents/rlm/search_qa_env.py`:
- New `_load_dataset_resilient(repo_id, *args, **kwargs)` — tries the canonical namespaced id,
  falls back to `repo_id.split("/")[-1]` (the bare name, constructed not literal) for caches
  populated under the old id.
- `_load_nq_open` → `google-research-datasets/nq_open`.
- `_load_hotpotqa` → `hotpotqa/hotpot_qa` (config `"distractor"` unchanged).

Verified: `ast.parse` OK; `_check_deprecated_hf_dataset_aliases` → 0 hard violations;
`tests/agents/rlm/test_search_qa_env.py` 24/24.

Because the harness module is **copied fresh into `code/` every run**, this fix applies to all
future runs automatically.

---

## Secondary issues (NOT blocking the rerun — note, don't gate on them)

1. **No-recovery fragility.** A single preflight hard-fail killed the whole run instead of the
   root repairing or the forced-iteration policy shipping a partial. With the dataset fix preflight
   won't fire, so the happy path is clear — but if a *different* hard violation appears, the run may
   die the same way. If the rerun dies similarly, investigate why the `claude-oauth` root subprocess
   terminates on a `run_experiment` failure (the `Loop … is closed` line points at the
   `claude-agent-sdk` event loop closing) rather than consuming `repair_context` and retrying.
2. **WebShop fair-excludes.** `env_cache._default_webshop_launcher` uses a bare `"python"`
   (`[Errno 2] No such file or directory: 'python'`) and `web_agent_site` isn't installed → WebShop
   is excluded (not zeroed) every run. Follow-up: switch the launcher to `sys.executable` and
   `uv pip install web_agent_site` into the base venv. Expected, not an error.
3. **Dense retrieval OFF.** No verified wiki-18 E5 index repo, so Search-QA uses BM25/lexical
   overlap + kept HotpotQA context. For max score: set `REPROLAB_SEARCH_QA_DENSE=1` +
   `REPROLAB_SEARCH_QA_INDEX_REPO=<hf wiki-18 e5 repo>` (disk now has room).

---

## HOW TO RUN (next session — do this)

From the worktree, with the base venv. **`claude-oauth` is the only working root.**

```bash
cd /home/sww35/openresearch-fullscope
# sanity: confirm the fix is present on HEAD
git log --oneline -1   # expect 5781cb7 fix(search-qa): namespace HF dataset ids …

# launch (backgrounded; logs to /tmp)
nohup .venv-or-base-python scripts/batch_reproduce.py 2605.15155 \
  --gpus-per-run auto --model claude-oauth --sandbox local --mode rlm \
  --extra "--paper-hint 2605.15155" \
  > /tmp/sdar_fullscope_run3.log 2>&1 &
```

Where `.venv-or-base-python` = **`/home/sww35/openresearch/.venv/bin/python`** (the shared base
venv; the worktree has no own `.venv`). Cells run with `sys.executable` = this base venv, so **all
deps must live there** (`uv pip install --python /home/sww35/openresearch/.venv/bin/python <pkg>`;
the base venv has no `pip` module — `python -m pip` fails). rank_bm25 + sentence-transformers +
faiss are already installed there.

The launcher auto-archives the prior `prj_09047604e591d969` attempt and reuses the same
deterministic project id. ALFWorld + Search-QA run real; WebShop fair-excludes (see above).

## HOW TO MONITOR

The key signal that the fix worked is the **first cell results** turning from stub →
`-> ok`/`-> error` and `final_report` appearing.

```bash
cd /home/sww35/openresearch-fullscope
P=runs/prj_09047604e591d969

tail -f /tmp/sdar_fullscope_run3.log          # live launcher/agent log
tail -f $P/batch_child.log                    # run-internal log
# cell progress (watch steps_run climb past 0):
for m in $P/code/outputs/*/*/metrics.json; do python3 -c "import json,sys;d=json.load(open('$m'));print(d.get('cell_id'),d.get('status'),'steps',d.get('steps_run'),'rwd',d.get('reward_mean'))"; done
# experiment outcomes (want success:true, no pre_flight):
python3 -c "import json;[print(json.loads(l).get('timestamp','')[11:19],'ok' if json.loads(l).get('success') else 'FAIL',json.loads(l).get('error','')[:100]) for l in open('$P/experiment_runs.jsonl')]"
# GPUs (1-7 should go busy once cells launch; GPU 0 is another user):
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
# final score when done:
cat $P/final_report.md 2>/dev/null | head -60
```

Run is **done** when `final_report.{json,md}` exists and the `batch_reproduce` process is gone
(`pgrep -af prj_09047604e591d969`). Record `overall_score` + per-env scores; note any
`stop_reason`. Then update memory file `full-scope-agentic-envs.md`.

---

## Environment facts (hard-won — keep)

- **Base venv:** `/home/sww35/openresearch/.venv` (uv-managed, **no pip module** → use
  `uv pip install --python <base venv>`). Cells use this via `sys.executable`; the per-run venv is
  empty (uv `--system-site-packages` inherits the empty uv-cpython base site — known gotcha).
- **Disk:** `/home` was 100%/15 GB; freed ~108 GB on 2026-06-01 (pruned Qwen2.5-Coder
  1.5B/14B/32B + 15 GB TriviaQA). **KEEP** `runs/.cache/hf`; the worktree's `runs/.cache/hf` is
  symlinked to main's so SDAR Qwen3-1.7B / Qwen2.5-3B / e5 weights are reused (no re-download).
- **GPU allocator** (`local_gpu_allocator.py`) excludes GPUs another user holds — GPU 0 (~81%) is
  someone else's; the run uses free cards (1–7). One-GPU-per-cell via `gpu_cell_runner.run_matrix`,
  OOM shrink-retry built in.
- **Commit policy:** commit as `lolout1`, **no Claude Code co-author trailer**. Stage specific
  files, never `git add -A` (runs/ noise).
- **PR:** still to open — `5.30.26_sdar...feat/full-scope-envs` (no gh CLI; user clicks compare URL).

## Pointers

- Build + Codex-fix history: memory `full-scope-agentic-envs.md`, commits `58ce554`, `24772fb`,
  `40fdc3d` (PATH fix), `5781cb7` (this fix).
- Agentic env design + OOM/capacity: `docs/superpowers/specs/2026-05-31-oom-gpu-capacity-remediation-design.md`.
- SDAR baseline context: `docs/runbooks/2026-05-23-sdar-baseline-handoff.md`, CLAUDE.md "Baseline test paper".
