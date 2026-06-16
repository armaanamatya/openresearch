# Azure / Kubernetes GPU Backend — Implementation Handoff Prompt

> **Paste-ready kickoff for a fresh agent.** This is the *prompt*; the **source of truth** is the
> locked design + standup runbook (linked in §2). It is written to be handed to an AI coding agent
> **and** read by a human. The design is **locked** — your job is to *implement it elegantly and
> robustly*, not to re-open it. When the design and the live code disagree, the **code is truth** —
> re-confirm the change-map and flag the drift, don't silently improvise.

---

## 0. Role & operating rules (non-negotiable)

**You are** a principal-level infra + backend engineer adding an **Azure AKS (Kubernetes) GPU
execution backend** to a system that autonomously reproduces ML papers. The deliverable is a
**client artifact for DeepInvent** (their Azure tenant): it must be **scalable, cost-predictable,
and clean enough to hand over and review**.

1. **Read before you build.** §2 docs first, then re-map them onto the *real* tree (§3). The
   change-map line numbers were verified 2026-06-03 — **they have since shifted** (a large BES +
   harden merge landed). Re-confirm every `file:line` against `HEAD` before editing.
2. **Never break `local` / `runpod` / `docker`.** Byte-for-byte. A non-Azure run must take a code
   path with **zero new branches on its critical line**. This is the acceptance bar — prove it with
   a regression smoke at the final gate.
3. **Iterate against gates, don't sprint.** This work came in via `/iterate`: every phase (§4) has a
   concrete pass/fail **success-check**. Stop at the gate; never advance on "should work."
4. **Quota is the critical path, not the code.** A fresh subscription has **0** A100 quota; the
   request is a hours→days support round-trip. **File it FIRST**, in parallel with coding. If it is
   still pending, validate the wiring on the **CPU fake-GPU stub** and keep the real-GPU gate
   **OPEN** (reported blocked-on-quota, never green).
5. **Production posture is part of the deliverable.** Workload Identity (AAD federation), **zero
   static secrets in-cluster**; Terraform modular with remote state; scale-to-zero GPU pool.
6. **Surgical edits only.** New files per design §4a, edits per §4b. No drive-by refactors; match
   surrounding style; mention dead code, don't delete it.

**Commit convention:** author as `lolout1`, **no co-author trailer**. One reviewable commit per
distinct change (a Terraform module, the runner drop-in, the wiring edits, each gate's evidence).

---

## 1. The one-paragraph what

Fill the **Azure-shaped hole the codebase already left** — `SandboxMode`,
`_backend_for_sandbox_mode`, `gpu_capacity._describe_azure` (a `NotImplementedError` stub today) —
with a **Job-native AKS GPU backend** selected by `--sandbox azure`. The orchestrator (RLM root +
primitives) keeps running **locally, unchanged**; the only thing that moves is *where the training
matrix executes*. The clean seam is **`k8s_job_cell_runner.run_matrix(...)`** — a **drop-in for
`gpu_cell_runner.run_matrix`** with the **identical signature and return contract** (`{cell_id →
result_dict}`, same `ok` / `oom_failed` / other status vocabulary), so **every downstream consumer
(`aggregate_cell_metrics` → leaf scorer → `final_report`) is untouched**. One per-cell K8s Job,
GPU node pool autoscales 0→N, Azure Blob is the code/artifact bus, Azure Files (RWX) is the shared
`HF_HOME` + pip cache, OOM shrink-retry happens **in-Job**.

---

## 2. Source of truth — read in this order

| # | Doc | Use it for |
|---|---|---|
| 1 | `docs/superpowers/specs/2026-06-03-azure-aks-gpu-backend-design.md` | **The design.** Decision table (§1), end-to-end architecture (§2), 3-layer split (§3), the file change-map (§4), the runner contract (§5), Terraform tree (§6), auth flow (§7), budget mapping (§8), phased plan + gates (§11). |
| 2 | `docs/runbooks/2026-06-03-azure-aks-gpu-backend-handoff.md` | **The how.** Prereqs, the quota commands (§1), per-phase standup commands, the CPU-stub fallback (§6), the troubleshooting matrix (§7), cost notes (§8). |
| 3 | `docs/superpowers/specs/2026-05-31-oom-gpu-capacity-remediation-design.md` | The **cell-matrix model** this extends: `describe_capacity`, `run_matrix`, `aggregate_cell_metrics`, the cell status vocabulary, the in-cell OOM shrink ladder you mirror in-Job. |
| 4 | `CLAUDE.md` | §Sandboxes, §One-GPU-per-cell execution, §Dynamic GPU selection; the **shell-vs-`.env` precedence** pitfall (prefix the CLI with `env -u OPENAI_API_KEY …`). |

---

## 3. Recon — re-confirm the change-map against HEAD (it moved)

The design's §4b lists exact `file:line` edits verified 2026-06-03. A large merge has since shifted
them. Before Phase 1b, **locate each by symbol, not line number**, and record the real location:

| Seam (design §4b) | Find by | Confirm |
|---|---|---|
| `SandboxMode` enum | `class SandboxMode` in `backend/agents/execution.py` | add `azure = "azure"` + `ensure/resolve` handling (mirror `runpod`) |
| `--sandbox` choices | the `--sandbox` arg in `backend/cli.py` | `"azure"` is **not** in the choices today |
| `default_sandbox` / `force_sandbox` Literals + `azure_*` settings block | `backend/config.py` | widen Literals; add `azure_*` block mirroring `runpod_*` |
| `_backend_for_sandbox_mode` | symbol in `backend/agents/rlm/primitives.py` | add `azure → AksJobBackend(...)` |
| `_execute_cell_matrix` runner-select | symbol in `backend/agents/rlm/primitives.py` | `azure → k8s_job_cell_runner.run_matrix`, else `gpu_cell_runner.run_matrix`, **identical args** |
| `_describe_azure` | symbol in `backend/services/runtime/gpu_capacity.py` | replace `NotImplementedError` with the settings-driven descriptor (§4b) |
| `build_environment` azure short-circuit | the `build_environment` primitive | no-op success under `azure` (mirror the `local` short-circuit) |

**Deliverable of recon:** a short "change-map → real paths @ HEAD" note appended to the design doc,
plus any contradiction flagged.

---

## 4. Build order — the `/iterate` spine (stop at each gate)

| Phase | Deliverable | **Success-check gate** |
|---|---|---|
| **0** | (a) file the A100 quota request; (b) `az login` + remote-state storage account | `terraform init` succeeds against remote state **AND** quota ticket filed (record # in the runbook) |
| **1a** | Terraform L1 (`infra/azure/`) + Helm L2 + base image → ACR | **hello-GPU Job**: pool scales 0→1, `nvidia-smi` runs, writes a marker to Blob via workload identity, pool scales back to 0 |
| **1b** | `k8s_job_cell_runner.run_matrix` drop-in + the 7 wiring edits (§3) + `azure_*` settings | **one** SDAR cell end-to-end: code from Blob, weights cached on Files, `metrics.json` lands in `runs/<id>/code/outputs/<id>/<cell>/` and `aggregate_cell_metrics` folds it |
| **1c** | full matrix + budget enforcement + **in-Job** OOM shrink + Pending-timeout→`capacity_exhausted` | **smallest-two SDAR matrix** completes on Azure, scored into `final_report.json`, under budget — **AND** `--sandbox local` + `--sandbox runpod` smoke runs still pass unchanged |

**Cap:** if quota is not granted in the engagement window, validate 1a/1b on a **CPU node + fake-GPU
stub** (runbook §6) to prove Blob bus + Files cache + Job watch + artifact download + aggregation —
but the **real-GPU gate stays open and is reported as blocked-on-quota, not success.**

---

## 5. The runner contract (the seam that keeps it clean)

`k8s_job_cell_runner.run_matrix` must match `gpu_cell_runner.run_matrix` exactly:

```python
def run_matrix(cells, cell_script, *, output_root,
               gpus=None, max_parallel=None, max_oom_retries=2,
               per_cell_timeout_s=None) -> dict[str, dict]:  # {cell_id → CellResult.to_dict}
```

- `gpus` → **ignored** (K8s scheduler places). `max_parallel` → orchestrator concurrency cap +
  namespace `ResourceQuota`.
- `per_cell_timeout_s` → Job `activeDeadlineSeconds`. `max_oom_retries` → `OPENRESEARCH_CELL_MAX_OOM_RETRIES`
  consumed by the **in-Job wrapper's** shrink ladder (the orchestrator does **not** resubmit on OOM).
- **Status mapping:** Job Succeeded → `ok`; wrapper exits `oom_shrink_exhausted` → `oom_failed`; Job
  failed/timeout/**stuck-Pending past timeout** → other (`_execute_cell_matrix` already counts `n_err`).
- Per-cell `metrics.json` is pulled from Blob into `output_root/<cell_id>/metrics.json`.

If the return shape and status vocabulary match, **nothing downstream changes** — that is the whole
design.

---

## 6. Guardrails & definition of done

- [ ] `local` / `runpod` / `docker` **byte-for-byte unchanged** — proven by a regression smoke at the 1c gate.
- [ ] `k8s_job_cell_runner.run_matrix` returns the **identical shape + status vocabulary**; no downstream edits.
- [ ] **No new SSE event types** — Job-log tailing emits the existing `repl_iteration` / `primitive_call` / `run_warning`.
- [ ] **Zero static secrets in-cluster** — Workload Identity only; federated-credential subject is `system:serviceaccount:<ns>:<sa>`.
- [ ] Budget caps (`max_run_gpu_usd`, `max_pod_seconds`) enforced **before** each Job submit; per-cell `activeDeadlineSeconds` + Job `ttlSecondsAfterFinished` set.
- [ ] Terraform is modular, remote-stated, and `plan`-clean; transient Jobs are **orchestrator-managed, not in the IaC** (3-layer split, design §3).
- [ ] Each phase gate passed (or the real-GPU gate explicitly reported blocked-on-quota).
- [ ] Commits as `lolout1`, no co-author trailer; one reviewable commit per distinct change.

---

## 7. First actions (in order)

1. **Quota** — run the runbook §1 commands; file the `Standard NCADSA100v4 Family vCPUs` increase; record the ticket #.
2. **Recon** — re-confirm the §3 change-map against `HEAD`; append the real paths to the design doc.
3. **Phase 0 bootstrap** — `az login`, create the remote-state storage account, `terraform init`.
4. Proceed phase-by-phase (§4), stopping at each success-check.

---

*Provenance: locked design `2026-06-03-azure-aks-gpu-backend-design.md` (grilled, 9 decisions) +
its standup runbook. This prompt is the implementation kickoff; treat the two source docs as the
contract and the live code as truth.*
