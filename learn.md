# learn.md — cross-cutting reliability rules (active)

> **Active, append-only.** This log holds cross-cutting reliability RULES in
> **Rule / How / Why** shape (per the `/iterate` discipline) — not per-bug
> postmortems. Append only: improve formatting or wording in place, never
> delete a rule. One rule per section. **Newest rule on top.**
>
> Incident *narratives* (symptom → root cause → fix, with the specific
> `prj_*` run and commit) live in per-bug memory files and
> `docs/superpowers/specs/` — this file cites them but doesn't retell them.
> The pre-2026-06 postmortem log (`Symptom → Root cause → Fix → Lesson →
> Guardrail` shape) is frozen and archived at
> [`docs/archive/learn.md`](docs/archive/learn.md).

---

## Cloud posture: GCP/Azure primary, RunPod legacy (2026-07-09)

**Rule:** `--sandbox auto` resolves to docker/local ONLY and never a paid remote backend; gcp/azure/runpod are explicit. Foundry LLM rows price via `FOUNDRY_ALIASES` (no more $0). An explicit `--vram-gb` is used verbatim (no 1.25x headroom). `gcp_gpu_skus` mismatch fails loud at the GCP preflight (default stays `["gcp_a100_80x8"]`, Terraform-synced). A cell that would breach `--max-run-gpu-usd` mid-flight is killed (`gpu_budget_exceeded`).

**How:** `execution.resolve_sandbox_mode`, `pricing.FOUNDRY_ALIASES`, `GpuRequirements.vram_is_explicit`, `gpu_resolver.validate_configured_skus` + `gke_job_backend.validate_gcp_skus_against_cluster`, `k8s_job_cell_runner._watch_job` GPU-$ heartbeat.

**Why:** RunPod is legacy; GCP/Azure are the supported clouds. The old auto->runpod default, the $0 Foundry ledger, and uncapped mid-cell burn made the primary clouds untrustworthy for overnight campaigns.

---

## 2026-07-07 — A hyperparameter guard must key on the variable's ROLE, not an ambiguous name

**Rule.** A preflight/sanity guard that range-checks a hyperparameter (learning rate, dropout, …) must scope to names that unambiguously denote that role — never a Greek-letter/coefficient name (`alpha`, `beta`, `lambda`, `tau`, `eta`) shared with RL/regularization coefficients, where values outside the "sane LR range" (`0.0` ablation, `>1.0` weight) are legitimate.

**How.** `_check_absurd_learning_rate` listed `alpha` in `LR_NAMES`, so UCPO's faithful `alpha=0.0` sharpening ablation tripped the "outside sane LR range [1e-7,1.0]" hard block. Fix: drop `alpha`; keep the unambiguous LR names (`lr`/`learning_rate`/`base_lr`/…) and both bounds. More generally: when you widen a guard's scope (e.g. `OPENRESEARCH_PREFLIGHT_UNION_SCOPE`), re-audit its name/value heuristics against RL configs — a supervised-training assumption (loss>0, lr∈[1e-7,1], variance>0) false-fails legitimate RL / degenerate-slice values.

**Why.** `prj_618` (UCPO) blocked for a full run + repair loop on `alpha=0.0` — the executor even documented the false-positive in-code. Same family as the RL-smoke false-fail (`[[project_sdar_gcp_rl_smoke_fix]]`) and the SDAR file-scoping false-block.

---

## 2026-07-07 — `$0` in `cost_ledger.jsonl` / `demo_status.json` does NOT mean $0 spent

**Rule.** Never trust the cost ledger alone to conclude a run spent nothing.

**How.** Real LLM spend lives in `tokens_total.json`; Foundry-routed models
(`claude-opus-4-8`, `claude-sonnet-5`) are unpriced in the ledger, and GPU
node-idle time is invisible to it entirely. Verify actual spend via
`kubectl get nodes` (catches stray/idle A100s) **plus** `tokens_total.json`
— never the ledger in isolation.

**Why.** All 6 runs in the 2026-07-07 triage
(`docs/runbooks/2026-07-07-all-runs-triage-and-hardening-handoff.md`) showed
`$0` in the ledger while `tokens_total.json` showed real spend (e.g. 137K
output tokens).

---

## 2026-07-07 — A faithful implementation can fail preflight because the guard reads code MID-WRITE (harvest race)

**Rule.** Never conclude "surrogate/toy" from a preflight block — inspect the
FINAL code snapshot and file mtimes before ruling on faithfulness.

**How.** `implement_baseline` may report `ok` ~140s in while the writer keeps
landing files 12–14 minutes later; the guard then validates a stale
snapshot. Guardrail: `OPENRESEARCH_IMPL_ABANDON_GUARD` returns
`implement_timeout_abandoned` and refuses to cache `ok` on an aclose-stall
give-up.

**Why.** `prj_23f04429cd3beaf7` false-blocked a genuinely faithful SDAR
implementation this way — real `from_pretrained` via the model registry,
real GRPO, `grep nn.Linear|nn.Embedding` = 0.

---

## 2026-07-07 — The monolithic exec path (`k8s_job_backend.exec`) is NOT GKE-ready

**Rule.** On GKE, training must route through the cell-matrix
(`code/cells.json` + `train_cell.py`), or the `OPENRESEARCH_GKE_SYNTH_CELL`
synthesis that generates them — never the monolithic exec path.

**How.** The monolithic exec path runs `commands.json` via `sh -c`,
overrides the base-image GCS-download ENTRYPOINT, and never restages the
uploaded code into the pod; its smoke bootstrap also emits a
host-absolute `cd <laptop path>`.

**Why.** `prj_618445173e9ae4f2` and `prj_c912f5df415f410c` both died with
`sh: cd: can't cd to .../code`.

---

## 2026-07-07 — Reliability fixes ship default-OFF and protect NOTHING until enabled in the run-spec

**Rule.** Landing a fix behind a default-OFF flag is not the same as
shipping the fix — turning it ON in the run-spec is a required deployment
step, not a nicety.

**How.** Repo flag discipline requires new flags default-OFF and
byte-identical-off. A landed fix (`OPENRESEARCH_PREFLIGHT_UNION_SCOPE`,
`OPENRESEARCH_IMPL_ABANDON_GUARD`, `OPENRESEARCH_HARDEXIT_CLEANUP`) does
nothing until it's turned on in `configs/*run_spec.json`.

**Why.** Every run in the 2026-07-07 triage failed with the relevant fix
present in the codebase but OFF in the run-spec.

---

## 2026-07-07 — Cosmetic template defaults masquerade as agent failures; verify against artifacts, not the manifest

**Rule.** Judge "toy vs faithful" from the real `plan_reproduction` output
and the code itself — never from `demo_status.json` / `reprolab_manifest.json`.

**How.** The `ppo-cartpole-v1` / `mean_reward=475` "benchmark" block is a
hardcoded demo stub (`backend/services/events/live_runs.py`), stamped on
EVERY run and never overwritten when a run fails pre-scoring. It looks
exactly like a toy-task substitution but isn't.

**Why.** This stub produced a phantom "critical (6) UCPO→CartPole"
substitution finding and briefly misled triage; both the SDAR and UCPO
implementations were in fact faithful.

---

## 2026-07-07 — A killed driver is a durability failure, not a science failure

**Rule.** Launch long GPU runs from a durable host (or `nohup`, relying on
the primitive cache) — don't re-derive the science after a driver dies.

**How.** Run drivers die from host-suspend, OOM, or operator SIGTERM (this
WSL2 laptop host is not durable). A run left `orphaned`/`interrupted` with
`$0` is resumable via `rlm_state/primitive_cache.jsonl`.

**Why.** `prj_c912f5df415f410c` was orphaned by host-suspend/OOM;
`prj_618445173e9ae4f2` and `prj_13f7eef55bd0b55c` by SIGTERM.

---

## 2026-07-07 — Evidence-not-grade when auditing an implementation for "surrogate/toy"

**Rule.** Before ruling an implementation "fake," check the FINAL code
snapshot against deterministic markers — never a grade or a first
impression.

**How.** Check `grep nn.Linear|nn.Embedding` = 0 and a real
`AutoModelForCausalLM.from_pretrained(...)` (literal, or a grounded
variable-arg call resolved through the model registry) in the final
snapshot. Preflight scope must cover the training-file union, not just
`train.py` (`OPENRESEARCH_PREFLIGHT_UNION_SCOPE`).

**Why.** Two separate SDAR "surrogate" verdicts were overturned as faithful
once the real files were read.
