# SDAR-on-GCP with the Actor–Critic Evidence Layer — New-Session Handoff

> **Purpose:** everything a fresh session needs to run **SDAR (arXiv 2605.15155) end-to-end on GCP**
> with the new **actor–critic / EvidenceAudit** harness turned on, and to judge whether the run is
> honest. Authored 2026-06-20.
> **Branch:** `feat/grounded-self-improvement-harness-reliability` (pushed to **both** `origin`/openresearch
> and `deepinvent`; tip `4c636e42`).
> **Read alongside (do not duplicate):** `2026-06-20-grounded-self-improvement-operator-checklist.md`
> (canonical funded validator panels), `2026-06-20-validation-coverage-and-capped-rerun-handoff.md`
> (the 6h-timeout trap + capped-rerun), `2026-06-19-sdar-gcp-e2e-guardrails-handoff.md` +
> `2026-06-16-sdar-on-gcp-a100-vm.md` (GCP cluster/VM mechanics), `2026-05-23-sdar-baseline-handoff.md`
> (the SDAR run + debug cycle).

---

## 1. Goal & success definition

Run SDAR smallest-two scope on GCP and confirm the harness ships an **honest, evidence-backed** report:
- **If training is real** → real non-zero metrics + `code/provenance.json`; the critic passes; verdict
  reflects the measured score.
- **If a cell fabricates** (the SDAR-v6 failure: real 8-GPU training but all-0.0 metrics with *real* keys)
  → the critic **vetoes** it to `fabrication_suspected` and the run repairs or fails *honestly* — it must
  NOT ship a green report over fake metrics.

**The run is a success when:** `final_report.json` has (a) a verdict consistent with the measured score,
(b) `meets_target` populated (not null), (c) `validation` stamped by the grok panel (not `unavailable`),
(d) the run reached `_finalize` (NOT killed at the wall-clock before the validator ran — see §6 the trap).

---

## 2. What changed since the last SDAR-GCP run (the new layer)

The actor–critic evidence layer (this branch) sits *on top of* the prior grounded-self-improvement work.
The fitness signal is the **deterministic evidence layer, never the LLM grade**. New, all **flag-gated,
default-OFF (byte-identical when off)** — so you must turn them ON for this run:

| Flag | Set to | Why |
|---|---|---|
| `OPENRESEARCH_EVIDENCE_AUDIT` | `1` | Master switch for the unified `EvidenceAudit` critic. ON ⇒ the run_experiment veto + verdict gate require real evidence (non-zero/non-stub metrics, provenance). Subsumes `ZERO_METRICS_GUARD`/`STUB_METRICS_GUARD`/antifab. **This is the one that catches SDAR-v6.** |
| `OPENRESEARCH_EXTERNAL_VALIDATOR` | `1` | Turns on the grok adversarial panel (now runs on EVERY finalize path, incl. abort/hard-stop). |
| `OPENRESEARCH_VALIDATOR_BACKEND` | `azure-foundry` | Independent cross-family validator transport (grok). FAIL-CLOSED: a misconfigured validator raises rather than silently judging with the executor's own model. |
| `OPENRESEARCH_VALIDATOR_MODEL` | `grok-4.3` | The deployed Foundry model id (matches `AZURE_FOUNDRY_DEPLOYMENT`). |
| `OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S` | `1800` | **The 6h-trap fix.** Bounds a single run_experiment so the loop reaches `_finalize` (and the validator) instead of being SIGTERM-killed at the global wall-clock with the validator skipped. |
| `OPENRESEARCH_LEAF_EVIDENCE_GATE` | `1` *(optional)* | Per-leaf veto: a result-claiming leaf with no on-disk cell evidence is zeroed. (Renamed from the `EVIDENCE_GATE` split-default; the verdict-level `OPENRESEARCH_EVIDENCE_GATE` stays default-ON.) |

Credential preflight (new, default-ON): the CLI now validates the root key before ingest. Opt out with
`OPENRESEARCH_SKIP_CRED_PREFLIGHT=1` only in a known-good env.

---

## 3. Model roles & the keyless-root caveat

Three independently-billed surfaces (pick to keep the validator cross-family from the executor):

- **Root** (drives the loop): `gpt-5` (recommended, reliable; needs `OPENAI_API_KEY`) or `gpt-chat-latest`
  via Foundry (keyless-of-Anthropic; the 2026-06-20 validation run used this). **Do NOT use `claude-oauth`
  as the root** — it risks the degenerate-refusal loop (it reads the paper then `FINAL_VAR`s without
  implementing). gpt-5/gpt-chat are the validated roots.
- **Executor** (`implement_baseline` — writes the real code): `--models executor=sonnet` (Sonnet via OAuth
  or `ANTHROPIC_API_KEY`). gpt-4o/grok are NOT SDAR-validated executors and tend to stub — keep Sonnet/gpt-5.
- **Validator** (grok panel): `azure-foundry` / `grok-4.3` (needs `AZURE_FOUNDRY_ENDPOINT` +
  `AZURE_FOUNDRY_DEPLOYMENT` + `AZURE_FOUNDRY_API_KEY`). Cross-family vs the Sonnet executor ⇒ `separation=independent`.

The canonical funded two-transport panels are in `2026-06-20-grounded-self-improvement-operator-checklist.md` —
use those exact creds.

---

## 4. The run-spec (ship the flags as one file)

Instead of a long env whitelist, write a JSON run-spec and pass `--run-spec`. Example
`runs/_specs/sdar_gcp_actor_critic.json`:

```json
{
  "OPENRESEARCH_EVIDENCE_AUDIT": "1",
  "OPENRESEARCH_EXTERNAL_VALIDATOR": "1",
  "OPENRESEARCH_VALIDATOR_BACKEND": "azure-foundry",
  "OPENRESEARCH_VALIDATOR_MODEL": "grok-4.3",
  "OPENRESEARCH_VALIDATOR_PANEL_N": "1",
  "OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S": "1800",
  "OPENRESEARCH_LEAF_EVIDENCE_GATE": "1",
  "OPENRESEARCH_REPAIR_MAX_ITERATIONS": "4",
  "OPENRESEARCH_DEFAULT_SANDBOX": "gcp"
}
```

`--run-spec` loads these into the env BEFORE flag resolution; explicit CLI flags still win
(`backend/cli.py::_load_run_spec`). Keep scope pinning (smallest-two: Qwen3-1.7B + Qwen2.5-3B) via the
SDAR `PAPER_HINTS` default + `OPENRESEARCH_BASELINE_EXTRA_GUIDANCE` / `--scope-spec` as in the baseline handoff.

---

## 5. The GCP run procedure (self-contained)

The whole run is driven by `scripts/gcp_sdar_preflight.sh` (the established, idempotent path) — do NOT
hand-roll a VM. GCP defaults (override via env): `PROJECT=deepinvent-ext-ut`, `ZONE=us-central1-c`,
`INSTANCE=sdar-a100-8g`, GPU machine `a2-highgpu-8g` (8×A100), CPU machine `e2-standard-16`,
`REMOTE_DIR=/home/abheekp/openresearch`, spot required, `MIN_GPUS=8`. `prepare` runs on the **cheap CPU**
machine (no GPU billing); `launch` flips to the **8×A100** and starts the reproduce detached.

```bash
# from the repo root (locally — the script drives the VM over gcloud SSH)
scripts/gcp_sdar_preflight.sh status     # VM/instance state
scripts/gcp_sdar_preflight.sh prepare    # CPU machine: sync repo, install SDAR deps, warm model/dataset
                                         # caches, provision ALFWorld/WebShop/Search-QA  -> must reach [GREEN]
# >>> inject the actor-critic flags now (see below), BEFORE launch <<<
scripts/gcp_sdar_preflight.sh launch     # flips to 8×A100, verifies env/GPU, execs sdar_gcp_run.sh detached
scripts/gcp_sdar_preflight.sh monitor    # tail progress
scripts/cancel_gcp_sdar_run.sh --stop-vm # when done (or: gcp_sdar_preflight.sh stop)
```

**Inject the new flags:** `sdar_gcp_run.sh` sources `runs/.cache/sdar_gcp.env` (written by `prepare`) and
the harness honors `os.environ`/`.env`. After a GREEN `prepare`, append §4's flags to that env file **on the
VM** (the script runs there), before `launch`:

```bash
# on the VM (e.g. via: scripts/gcp_sdar_preflight.sh sync then SSH, or add to the prepare step):
cat >> runs/.cache/sdar_gcp.env <<'EOF'
OPENRESEARCH_EVIDENCE_AUDIT=1
OPENRESEARCH_EXTERNAL_VALIDATOR=1
OPENRESEARCH_VALIDATOR_BACKEND=azure-foundry
OPENRESEARCH_VALIDATOR_MODEL=grok-4.3
OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S=1800
OPENRESEARCH_LEAF_EVIDENCE_GATE=1
EOF
```

**Model roles on GCP (what the script actually does):** it defaults to **pure-Foundry, OAuth-free** —
root + executor + grader + verifier all = `AZURE_FOUNDRY_DEPLOYMENT` (default `gpt-chat-latest`, a
reasoning-class model already verified to drive the REPL loop). To make the validator a *real independent*
check, give it a DIFFERENT deployment than root/executor:
- **Easiest (weak separation):** keep root/exec = `gpt-chat-latest`, set the validator to a different
  Foundry deployment — `OPENRESEARCH_VALIDATOR_BACKEND=azure-foundry`, `OPENRESEARCH_VALIDATOR_MODEL=grok-4.3`
  (cross-deployment ⇒ `separation=weak`, the machine-checked veto still stands).
- **Strongest (independent):** run the **executor on Sonnet** (`CLAUDE_CODE_OAUTH_TOKEN` in `.env` +
  `OPENRESEARCH_SDAR_MODELS=executor=sonnet,grader=foundry,verifier=foundry`) and the validator on grok-Foundry
  → cross-FAMILY ⇒ `separation=independent`.
Switch the root model with `AZURE_FOUNDRY_DEPLOYMENT=...` (e.g. `grok-4.3`); per-role with
`OPENRESEARCH_SDAR_MODELS=...`. `OPENRESEARCH_SDAR_ROOT=foundry` is the neutral root alias.

**Scope:** the script defaults to the **full 3×3 matrix** (Qwen3-1.7B + Qwen2.5-3B + Qwen2.5-7B × envs;
the 7B shards over 2 cards). For a cheap de-risk smoke FIRST, stage smallest-two guidance at
`runs/.cache/sdar_scope_guidance.txt` before `launch` (the script auto-loads it; an explicit
`OPENRESEARCH_BASELINE_EXTRA_GUIDANCE` still wins). The default guidance already includes the full
anti-fabrication + SDAR-method spec (lambda_SDAR=0.01, beta=5.0, OPSD/GRPO, the three gates, 150 real steps).

---

## 6. Known gotchas (read before launching)

1. **The 6h timeout trap (the #1 prior failure).** Without `OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S=1800`,
   a long cell runs to the global wall-clock, SIGTERM fires the hard-stop, and historically the validator
   was skipped → unvalidated report. Two fixes are now in: (a) the validator runs on the hard-stop path too
   (this branch), and (b) the per-experiment timeout lets the loop reach `_finalize` normally. Set it.
2. **Root model:** gpt-5 / gpt-chat-latest only. `claude-oauth` as root = degenerate-loop risk.
3. **Executor stubs:** keep `executor=sonnet`. A non-Sonnet/gpt-5 executor tends to emit a stub
   (`{total_length, chunk_count}`) — which the stub guard (now under `EVIDENCE_AUDIT`) will veto, wasting the run.
4. **Cred preflight** aborts fast on a dead key — good. If it false-blocks a valid OAuth setup, set
   `OPENRESEARCH_SKIP_CRED_PREFLIGHT=1`.
5. **Cells route:** SDAR uses `code/cells.json` + `code/train_cell.py` (one GPU per cell). If a repair drops
   `cells.json`, it auto-restores from `rlm_state/last_cells.json`. Watch for `cells_manifest_dropped` warnings.
6. **GPU/quota:** SDAR smallest-two fits one 24–48 GB GPU per cell; full scope needs more (see
   `2026-05-31-sdar-full-reproduction-resource-map.md`).

---

## 7. How to read the result (the new report fields)

In `final_report.json`:
- `reproducibility.verdict` / `verdict` — must be consistent with `rubric.overall_score` (no more
  reproduced-at-0.0; the write-chokepoint caps it).
- `meets_target` — now populated (`true`/`false`), not null.
- `validation` — the grok panel verdict (`status: clean | vetoed | unavailable`, `veto_set`, `separation`).
  `unavailable` means the panel didn't run (investigate — likely the trap or a validator cred issue).
- A vetoed run shows `failure_class=fabrication_suspected` in `experiment_runs.jsonl` + a
  `run_warning` with `code=fabrication_suspected`. **This is the system working** (catching a fake), not a bug.
- `runs/<id>/code/provenance.json` present ⇒ real training was recorded; absent + all-zero metrics ⇒ the
  zero-metrics veto should have fired.

Tail live: `tail -f runs/<id>/code/.exec_live.log` and `runs/<id>/dashboard_events.jsonl`.

---

## 8. Code state & open items (so the new session isn't surprised)

- **Done + pushed (12 commits, 6916 tests pass, 0 regressions):** the EvidenceAudit critic + run_experiment
  veto + grok-validator-on-every-path + scoring fidelity + recipe evidence-only guards + credential preflight
  + orphan-sweep salvage + detect_env FROM-base validator (wired) + BES-select-by-evidence + verdict-gate
  consults the audit + CLAUDE.md drift fixes.
- **DEFERRED (high-risk, quality-first — not done):** `run_experiment` dispatcher split, `ForcedIterationPolicy`
  rule-chain refactor, real `rerun_agrees` (needs re-exec infra), BES `EvidenceAudit` hard pre-filter.
- **Merge to `main`:** NOT done. `main` is a divergent month-long line (took grader-fidelity #106 +
  azure-gcp #108 independently); landing this branch is a careful ~27-commit 3-way-merge ("scoped replay"
  onto `origin/main`), tracked on the parked branch `feat/actor-critic-evidence-critic-on-main`. The work is
  safe on both feature-branch remotes regardless. Do the SDAR validation run from the feature branch.
- **Validation precondition not yet run here:** the spec's per-default-flip A/B (≥3 paired SDAR runs) +
  the THRESHOLD≈16 ~$2 pre-GPU smoke — see the operator checklist before flipping any default ON.
