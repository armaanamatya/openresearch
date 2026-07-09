# Cloud posture realignment — GCP/Azure primary, RunPod legacy — design

> **Doc status:** Draft · spec tier · authored 2026-07-09. Grounded by four read-only recon
> passes over `chore/cleanup-sweep-2026-07-07` (RLM core, runtime/cloud, product surface,
> repo health) plus targeted reads of `backend/agents/execution.py` and
> `backend/agents/resilience/pricing.py`. Baseline: 309 in-scope tests green
> (runtime/pricing/config/evals/fidelity). Policy: `docs/policies/documentation.md`.
> **Companion spec:** `2026-07-09-lifecycle-primary-hardening-design.md` (Spec B) — this is
> Spec A and lands first, so Spec B's A/B validation runs on a trustworthy cloud.

## 1. Problem

The project is committing to **GCP (GKE) + Azure (AKS)** as the first-class execution clouds
and demoting **RunPod** to a supported-but-legacy backend. Today the code and docs still say
the opposite in three load-bearing places, and three cost-visibility holes make the primary
clouds untrustworthy for the overnight campaigns and the upcoming lifecycle-primary A/B runs:

1. **Default says RunPod.** `DEFAULT_SANDBOX_MODE = SandboxMode.runpod` (`execution.py:61`).
   The quickstart, `runtime/CLAUDE.md`, and the flag registry all frame RunPod as the
   headline sandbox.
2. **Foundry LLM spend prices at $0.** The ledger records the role aliases
   `opus-foundry`/`sonnet-foundry` (see `pricing.py:44-54` comment), which don't resolve
   against `PRICING` (whose keys are bare `claude-opus-4-8` / `claude-sonnet-5`), so
   `estimate_cost_usd` returns `None` → every Foundry-routed row books `$0`. The primary LLM
   path is cost-blind.
3. **GPU-$ cap leaks mid-cell.** `RunBudget.check_run_gpu_usd` fires only at `run_experiment`
   return, so a cell wedged on a ~30 KB/s dataset mirror (`cave.cs.toronto.edu`) holds an
   A100 idle for the full timeout — ~$6–16 per paper of untracked, uncapped burn.
4. **Two GKE footguns live in runbook lore, not in code.** `gcp_gpu_skus` defaults to
   `["gcp_a100_80x8"]` (`config.py:485-494`) — a phantom 8× SKU that yields
   `GpuResolutionError` on single/2×/4× clusters; and `--vram-gb N` gets a 1.25× headroom
   multiplier (`gpu_resolver.py:204`) even when the operator set it explicitly, so
   `--vram-gb 80` on an 80 GB fleet demands 100 GB → no SKU ever matches.

## 2. Goal / non-goals

**Goal:** make GCP + Azure the default, trustworthy execution path — correct default
sandbox selection, truthful LLM cost, a GPU-$ cap that actually bounds a stuck cell, and the
two footguns promoted from lore to tested guards — with every doc surface reflecting the new
posture. RunPod stays fully functional, just no longer the default or the headline.

**Non-goals (this spec):** deleting or refactoring any RunPod code path; the lifecycle-primary
rewrite (Spec B); wiring `cloud_failover.py` or the in-cluster GCP orchestrator image (both
written-but-unwired, deferred); changing the evidence/verdict layer; RunPod liveness-based
stall detection (a separate, larger runtime change).

## 3. Design

### 3.1 Sandbox default & selection

Change the default and make `auto` a **local-dev-only** resolver that never silently selects a
paid remote backend. (Correction from recon: today `resolve_sandbox_mode` returns
`DEFAULT_SANDBOX_MODE` — i.e. `runpod` — for `auto` even on a normal dev machine, so `auto`
currently reaches out to a paid cloud by default. That is the surprise we're removing.)

```
DEFAULT_SANDBOX_MODE:  SandboxMode.runpod  ─▶  SandboxMode.auto

auto resolution (execution.py resolve_sandbox_mode):
   • OPENRESEARCH_FORCE_SANDBOX still wins (unchanged)
   • WSL without reachable docker      → local   (unchanged)
   • docker daemon reachable           → docker
   • otherwise                         → local
   ── auto NEVER selects a paid remote backend ──
   ── gcp / azure / runpod are opt-in only (--sandbox {gcp,azure,runpod} / env) ──
```

**Posture, not auto-magic.** "GCP/Azure primary" means they are the documented, cost-correct
clouds for campaigns and are selected **explicitly** (`--sandbox gcp` / `--sandbox azure`);
production/campaign commands already pass this. `auto` is for local dev. This avoids probing
cloud creds on every run and avoids surprise A100 spend from an unqualified `auto`.

**Invariant tested (see §5):** `auto` resolves to docker (daemon up) or local (else), and
**never** to `runpod`/`gcp`/`azure`. `runpod`, `brev`, `simulate` remain valid `SandboxMode`
members and fully functional when chosen explicitly.

### 3.2 Foundry LLM pricing (cost truth)

Add alias resolution so the ledger's Foundry role ids price at their real Claude rates.
`_resolve_pricing` (`pricing.py:78`) already strips a `provider.` prefix; extend it with an
explicit **role-alias map** applied before lookup:

```
FOUNDRY_ALIASES = {
    "opus-foundry":   "claude-opus-4-8",     # 15 / 75 per 1M
    "sonnet-foundry": "claude-sonnet-5",     #  3 / 15 per 1M
    # add any further Foundry-routed roles as they appear in role_models
}
```

Resolution becomes: exact → alias map → provider-prefix strip → None. The alias table lives
next to `PRICING` and is covered by a test that a `opus-foundry` usage row estimates non-zero.
(Rates mirror the non-Foundry siblings per the existing `pricing.py:44-54` note; Foundry
billing is separately reconciled via Azure Cost Management, but the ledger is no longer a
false zero.)

### 3.3 GPU-$ heartbeat cap (mid-cell bounding)

Today the `RunBudget` GPU-$ cap only fires at `run_experiment` return, so a wedged cell burns
the full deadline before anyone checks the cost. Add a **mid-cell budget heartbeat** to the
existing cell poll loop:

- **Periodic budget check** — the poll loop (already ticking every `watch_poll_interval_s`)
  computes accrued GPU-$ = elapsed-wall-clock × SKU $/hr × gpu_count and terminates the job
  when it would meet/exceed the run's `max_run_gpu_usd`, instead of waiting for
  `run_experiment` to return. It reuses the context-var getters the runner already uses
  (`_get_run_budget()`, `_get_gpu_plan()`) — no signature threading.

Termination is **fail-loud**, logged as a distinct terminal reason `gpu_budget_exceeded`
(non-retryable, recorded to the failure capsule). **No new env var and OFF-preserving:** the
check is a no-op when `max_run_gpu_usd` is unset or the SKU rate is 0, so runs that don't set a
cap are unchanged. Per-cell *runtime* bounding already exists via `per_cell_timeout_s` →
`job_deadline`; this spec adds only the missing *cost* bound.

### 3.4 GKE footgun guards (lore → tested code)

- **`gcp_gpu_skus` mismatch** — add a startup validation in the GCP preflight
  (`ensure_gcp_available` / gpu resolver init) that, when the configured SKUs cannot match any
  cluster `reprolab/sku` node label, raises a **clear, actionable** error naming the
  configured vs available SKUs and the exact `OPENRESEARCH_GCP_GPU_SKUS='[...]'` fix — instead
  of a bare `GpuResolutionError` on the first GPU call. **Do NOT change the shipped default**:
  `["gcp_a100_80x8"]` is deliberately synced to the Terraform default in
  `infra/gcp/variables.tf` and guarded by `tests/config/test_gcp_sku_pool_invariant.py`
  (`_EXPECTED_DEFAULT = ["gcp_a100_80x8"]`). The operator whose live cluster differs (e.g. the
  single-A100 `deepinvent-ext-ut`) sets `OPENRESEARCH_GCP_GPU_SKUS` to match; the new
  validation makes that requirement loud and self-explaining instead of a silent per-call
  failure. **Decision D2** (see §4).
- **VRAM-override double-multiply** — in `gpu_resolver` (anchor `:204`), skip the 1.25×
  headroom multiplier when the VRAM target came from an explicit `--vram-gb` override; apply
  headroom only to the LLM-estimated value. Operator intent is used verbatim.

### 3.5 RunPod demotion (no deletion)

- No code paths removed; RunPod stays supported and tested.
- Removed from default selection (§3.1).
- **Info-level log** when `--sandbox runpod` is explicitly chosen: "RunPod is a legacy
  backend; GCP/Azure are the supported clouds." No behavior change, cheap signal.
  **Decision D3** (see §4).

### 3.6 Docs sweep

- **Root `CLAUDE.md`** — reorder the sandbox list to lead with `gcp`/`azure`; add a one-liner
  that RunPod is legacy. The fidelity anchor "RunPod cloud-type default = SECURE" is about
  cloud-*type*, not primacy → `test_claude_md_fidelity.py` stays green (verify explicitly).
- **`backend/services/runtime/CLAUDE.md`** — restructure so GCP + Azure are the primary
  sections and RunPod is a clearly-marked "Legacy" section.
- **`docs/reference/flags.md`** — regenerate via `scripts/gen_flag_registry.py` after the
  config changes (a freshness test guards it).
- **`learn.md`** — one rule promoting the two footgun fixes from lore to guarded behavior,
  plus the mid-cell GPU-$ cap.
- **New runbook** `docs/runbooks/2026-07-09-cloud-posture-gcp-azure-primary.md` — the
  decision + the three cost-visibility fixes + operator launch defaults.

## 4. Decisions

- **D1 — default → `auto` = docker/local only; clouds are explicit.** `auto` never selects a
  paid remote backend (removes today's silent `auto`→`runpod`). GCP/Azure primacy is a
  documentation + explicit-selection posture, not auto-detection. *(Revised from the original
  "gcp→azure preferring auto" after recon showed auto currently returns runpod and that
  auto-probing cloud creds risks surprise spend.)*
- **D2 — GKE SKU footgun: validate-and-fail-loud, default UNCHANGED.** The `["gcp_a100_80x8"]`
  default is intentionally synced to the Terraform default and guarded by
  `test_gcp_sku_pool_invariant.py`; changing it would break the config↔TF invariant. The fix
  is a loud, self-explaining validation error that tells the operator to set
  `OPENRESEARCH_GCP_GPU_SKUS` to match their cluster. *(Revised from the original
  change-the-default proposal after reading the invariant test.)*
- **D3 — RunPod demotion is an info log, not silent and not a hard warning.** *Recommended.*

## 5. Testing strategy

Before/after (OFF/ON) pairs, all socket-hermetic:

- **Default sandbox** — `auto` resolves to `docker` when the daemon is reachable and `local`
  otherwise (WSL-without-docker → `local`); `auto` never resolves to `runpod`/`gcp`/`azure`;
  `OPENRESEARCH_FORCE_SANDBOX` still overrides. Explicit `--sandbox {gcp,azure,runpod}` is
  unaffected.
- **Foundry pricing** — `opus-foundry` / `sonnet-foundry` usage rows estimate non-zero and
  match their sibling rates; unknown ids still return `None`.
- **GPU-$ heartbeat** — the pure accrual helpers (`_accrued_gpu_usd`, `_over_gpu_budget`) are
  unit-tested; a cell whose accrued wall-clock × $/hr × gpu_count meets the cap terminates with
  `gpu_budget_exceeded`; no cap set (or SKU rate 0) → no-op, behavior unchanged.
- **VRAM override** — explicit `--vram-gb` passes through unscaled; LLM-estimated value still
  gets 1.25×.
- **`gcp_gpu_skus` validation** — mismatch raises the actionable error naming configured vs
  available and the fix string.
- **Fidelity** — `test_claude_md_fidelity.py` stays green after the doc edits.

## 6. Rollout

Land Spec A on its own branch, reviewed independently. The mid-cell cap and SKU-default
changes ship OFF-preserving where a default flip would change existing behavior; the
default-sandbox flip is the one intentional behavior change and is gated by the byte-identical
local-dev test. No paired-A/B gate needed here (this is infra correctness, not a
grader-affecting flag) — but the first real GCP run after landing is the Cutout (arXiv
1708.04552) validation, which doubles as an end-to-end smoke of the new posture.

## 7. Risks / open questions

- **Default-sandbox blast radius.** Anything that assumed `auto`→runpod (CI, scripts, saved
  operator commands) shifts. Mitigation: the byte-identical local-dev test + an explicit grep
  for `DEFAULT_SANDBOX_MODE` / `runpod` assumptions across scripts and configs during impl.
- **Foundry rate drift.** The alias rates mirror public Claude pricing; if Foundry negotiated
  rates differ, the ledger is an estimate (as it always was) — Azure Cost Management stays the
  source of truth. Documented, not silently implied.
- **Exact current `auto` semantics** must be read at impl time (`execution.py` resolver) to
  guarantee the byte-identical invariant — the spec asserts the requirement, the plan verifies
  the current branch.
