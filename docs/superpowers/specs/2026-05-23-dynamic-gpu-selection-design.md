# Dynamic GPU Selection from Paper Hardware Clues

**Status:** Approved-for-execution (2026-05-23)
**Author:** lolout1 (planning + design via Opus; Sonnet executes; Codex reviews final)
**Scope:** RLM orchestrator + RunPod sandbox backend

## Why

Today, RunPod sandbox runs **always** provision a fixed GPU SKU (default RTX 4090) defined by `REPROLAB_RUNPOD_GPU_TYPE`. The paper's stated hardware ("trained on 8× A100 80GB") is collected into `PaperClaimMap.hardware_clues` by `understand_section` and then **dropped on the floor** — no code consumes it. Papers that legitimately need ≥40GB VRAM fail at experiment time on RTX 4090's 24GB; papers that comfortably fit on RTX 4090 are penalty-free.

We want to **read the paper's hardware claim, estimate whole-workload VRAM (training + inference + eval), pick the cheapest RunPod SKU that satisfies that VRAM with a configurable safety margin, and provision the pod accordingly** — with a hard `$/hr` cap, a `force_single_gpu` invariant on by default, and an auto-escalation path when the experiment OOMs anyway.

Scope is **deliberately RunPod-only**. The resolver still emits informational events when sandbox is local or docker — for parity in the UI — but only RunPod consumes the plan authoritatively.

## Locked decisions

| # | Decision | Source |
|---|---|---|
| D1 | Policy: VRAM-based match with explicit inference headroom — NOT GPU-model fidelity | Q1 (option C, user adjusted) |
| D2 | Headroom: LLM estimates whole-workload `estimated_vram_gb` × multiplier (default **1.25**) + tier-up to next RunPod SKU | Q2 (option A) |
| D3 | Catalog: static vendored `gpu_catalog.py` + fallback ladder; no live RunPod API in hot path | Q3 (option A) |
| D4 | Extraction stage: new plan-time primitive `resolve_gpu_requirements`, RLM-native, SSE-emitted | Q4 (option A) |
| D5 | Flags: two — `REPROLAB_DYNAMIC_GPU` and `REPROLAB_FORCE_SINGLE_GPU`. Defaults: both ON | Q5 (option B) |
| D6 | Cost ceiling: `REPROLAB_MAX_GPU_USD_PER_HOUR` (default **10.0**); plus `REPROLAB_MAX_RUN_GPU_USD` (default **10.0**) on `RunBudget` | Q6 (option A) + user budget direction |
| D7 | OOM policy: auto-escalate up catalog ladder; max **2** escalations; gated by `$/hr` cap; transparent to the LLM | Q7 (option A) |
| D8 | No-clues fallback: RTX 4090 (24 GB), with `gpu_fallback` SSE warning so substitution is visible | Q8 (option A) |
| D9 | Existing `GpuMode` enum is sandbox-agnostic/Docker-focused; leave it alone. `$/hr` cap is the natural count bound when `force_single_gpu=off` | Q9 (option A) |
| D10 | OAuth orthogonality confirmed (recon + guard test): host-side LLM auth never touches the pod | Recon |

## Invariants

I1. **`gpu_count = 1`** whenever `REPROLAB_FORCE_SINGLE_GPU=on` (default). No code path may exceed this.
I2. **Catalog never returns a SKU exceeding `MAX_GPU_USD_PER_HOUR × gpu_count`.** Verified by `find_ladder` unit test.
I3. **The resolver is a pure function** of (claim_map, env_spec, settings, run_budget). No I/O. No network. No side effects except the SSE event emission, which happens in the primitive wrapper, not the resolver.
I4. **OAuth auth never enters the pod** — pod credentials path is empty by construction. Verified by `test_oauth_runpod_orthogonality.py`.
I5. **Resume safety**: `GpuPlan` is checkpointed to `runs/<id>/rlm_state/gpu_plan.json` atomically after `resolve_gpu_requirements` completes. Resumed runs reuse the plan; they do NOT re-resolve. This avoids cost drift across restarts.

## Architecture

### Module layout

```
backend/services/runtime/
├── gpu_catalog.py              [NEW]  vendored SKU table + find_ladder()
└── gpu_resolver.py             [NEW]  pure logic: GpuPlan from inputs

backend/agents/rlm/
├── primitives.py               [MOD]  + resolve_gpu_requirements; OOM detect + escalate in run_experiment
└── system_prompt.py            [MOD]  prompt RLM root to call resolve_gpu_requirements after understand_section pass

backend/agents/
└── schemas.py                  [MOD]  + GpuPlan, GpuRequirements

backend/agents/resilience/
└── budget.py                   [MOD]  + RunBudget.max_run_gpu_usd field + check_run_gpu_usd()

backend/services/runtime/
└── runpod_backend.py           [MOD]  __init__ accepts Optional[GpuPlan]; gpu_type/gpu_count/volume_gb sourced from plan when present

backend/services/events/
└── live_runs.py                [MOD]  emit gpu_resolved | gpu_escalated | gpu_fallback through sse_bridge

backend/config.py               [MOD]  + 7 new Settings fields

backend/cli.py                  [MOD]  + 4 new CLI flags (additive)

frontend/src/components/lab/rlm/
└── node-detail-sidebar.tsx     [MOD]  render GpuPlan badge on the work-cluster node when present
```

### Data types

```python
# backend/agents/schemas.py

@dataclass(frozen=True)
class GpuRequirements:
    """LLM-extracted requirements from paper text."""
    estimated_vram_gb: int | None         # None when LLM cannot estimate
    paper_gpu_string: str | None          # raw extracted string, e.g. "8x A100 80GB"
    paper_gpu_count: int | None           # parsed count, None if unknown
    reasoning: str                         # short, surfaced in SSE event
    confidence: float                      # 0.0–1.0; <0.4 triggers fallback

@dataclass(frozen=True)
class GpuPlan:
    """Resolved provisioning plan, consumed by RunpodBackend."""
    runpod_id: str                        # e.g. "NVIDIA A100 80GB PCIe"
    short_name: str                       # e.g. "a100_80"
    vram_gb: int                          # SKU's VRAM (>= required)
    gpu_count: int                        # 1 when force_single_gpu=on
    cloud_type: str                       # "COMMUNITY" | "SECURE"
    sku_usd_per_hr: float                 # per-GPU rate (catalog vendored)
    total_usd_per_hr: float               # sku_usd_per_hr * gpu_count
    container_disk_gb: int                # derived: max(50, vram_gb)
    volume_gb: int                        # derived: max(20, vram_gb // 4)
    source: str                           # "paper" | "fallback" | "manual"
    requirements: GpuRequirements          # for audit / display
    ladder_remaining: tuple[str, ...]     # short_names available for OOM escalation
    resolved_at: str                      # ISO timestamp
```

### Resolver contract

```python
# backend/services/runtime/gpu_resolver.py

def resolve(
    requirements: GpuRequirements,
    *,
    dynamic_gpu_enabled: bool,
    force_single_gpu: bool,
    max_gpu_usd_per_hour: float | None,
    headroom_multiplier: float,
    fallback_vram_gb: int,
    manual_vram_override: int | None = None,
) -> GpuPlan:
    """Pure function. No I/O. Returns a GpuPlan or raises GpuResolutionError.

    Raises:
        GpuResolutionError: when no SKU in the catalog satisfies (vram_gb >= needed)
                            AND (sku.approx_usd_per_hr * gpu_count <= cap).
                            Error message names the cheapest SKU that would have satisfied VRAM
                            ignoring the cap, so the user knows how to raise the cap.
    """
```

### Catalog (initial vendored content)

```python
# backend/services/runtime/gpu_catalog.py

CATALOG: tuple[GpuSku, ...] = (
    # sorted by (vram_gb ASC, approx_usd_per_hr ASC) for human readability;
    # find_ladder() re-sorts the filtered result by price ASC for selection
    GpuSku("NVIDIA GeForce RTX 4090",         "rtx4090",   24, "COMMUNITY", 0.34,
           aliases=("rtx 4090", "geforce 4090", "rtx4090")),
    GpuSku("NVIDIA RTX A5000",                "a5000",     24, "COMMUNITY", 0.36,
           aliases=("a5000", "rtx a5000")),
    GpuSku("NVIDIA A100 40GB PCIe",           "a100_40",   40, "COMMUNITY", 1.19,
           aliases=("a100 40", "a100 40gb", "a100-40")),
    GpuSku("NVIDIA RTX A6000",                "a6000",     48, "COMMUNITY", 0.49,
           aliases=("a6000", "rtx a6000")),
    GpuSku("NVIDIA L40S",                     "l40s",      48, "COMMUNITY", 0.86,
           aliases=("l40s",)),
    GpuSku("NVIDIA A100 80GB PCIe",           "a100_80",   80, "COMMUNITY", 1.89,
           aliases=("a100 80", "a100 80gb", "a100-80", "a100")),
    GpuSku("NVIDIA H100 80GB HBM3",           "h100_80",   80, "COMMUNITY", 4.39,
           aliases=("h100", "h100 80", "h100 80gb")),
    GpuSku("NVIDIA H200",                     "h200",     141, "SECURE",    7.99,
           aliases=("h200",)),
)

def find_ladder(
    min_vram_gb: int,
    max_per_gpu_usd_per_hr: float | None,
    cloud_types: tuple[str, ...] = ("COMMUNITY",),
) -> list[GpuSku]:
    """Returns SKUs with vram_gb >= min AND price <= cap AND cloud_type in cloud_types,
    sorted by ascending price. `cloud_types` defaults to `("COMMUNITY",)` to match the
    repo's default `REPROLAB_RUNPOD_CLOUD_TYPE=COMMUNITY`. To unlock SECURE-only SKUs
    like H200, the resolver passes `("COMMUNITY", "SECURE")` when
    `Settings.runpod_cloud_type == "SECURE"`."""
```

**Catalog price caveat (operational note, restated):** Prices are approximate snapshots,
vendored, refreshed quarterly. RunPod COMMUNITY pricing fluctuates ±20%. The resolver's
*ranking* between SKUs is what matters; absolute numbers may drift.

### Primitive contract

The primitive is a **pure resolver** — the *caller* (the RLM root) supplies its
LLM-derived estimate. The primitive does NOT internally call an LLM. This is intentional:
the root already has full context (paper corpus, env_spec, dataset description) and is
the right place to reason about training + inference + eval workload VRAM. Keeping the
primitive pure makes it deterministic and trivially testable.

```python
# backend/agents/rlm/primitives.py

def resolve_gpu_requirements(
    requirements: GpuRequirements | dict[str, Any],
    *,
    ctx: RunContext,
) -> GpuPlan:
    """Plan-time GPU resolver. RLM root supplies `requirements` based on its reasoning
    over the accumulated claim_map.hardware_clues + env_spec. Primitive maps to a GpuPlan
    via the catalog and emits a `gpu_resolved` SSE event.

    Args accept either the typed dataclass OR a loose dict (coerced to GpuRequirements);
    the dict path makes REPL code from the RLM root easier to write.

    Flow:
    1. If `--vram-gb N` CLI override is set on ctx, force estimated_vram_gb=N
       BEFORE applying the multiplier — i.e., the override is a floor on the LLM's
       judgment, then headroom is still added on top.
    2. If `run_state.gpu_plan` already exists → return cached plan (idempotency).
    3. Coerce `requirements` to `GpuRequirements`; raise `ValueError` on malformed input.
    4. Call `gpu_resolver.resolve(requirements, settings, run_budget)`.
    5. Atomically write `runs/<id>/rlm_state/gpu_plan.json` and emit `gpu_resolved` event.
    6. Return GpuPlan.
    """
```

### System prompt addition

A 3-sentence paragraph in `system_prompt.py` instructing the root:

> *After your initial pass of `understand_section` covers the abstract + method + experiments sections, construct a `GpuRequirements` from the accumulated `hardware_clues`. Estimate `estimated_vram_gb` for the WHOLE workload — not just training. Include inference, evaluation, and any auxiliary models the paper loads. Then call `resolve_gpu_requirements(requirements)` ONCE; subsequent `run_experiment` calls will reuse the cached plan automatically.*

The `--vram-gb` CLI override (if present) is honored as a **floor on the LLM's
estimate**: the multiplier still applies. To bypass headroom entirely, the user can pass
`--dynamic-gpu-headroom 1.0`.

### OOM detection + escalation

In `run_experiment`, after the pod's command returns:

```python
def _detect_cuda_oom(exit_code: int, stderr_tail: str) -> bool:
    if exit_code == 137:
        return True
    return any(
        marker in stderr_tail
        for marker in (
            "CUDA out of memory",
            "RuntimeError: CUDA error: out of memory",
            "torch.cuda.OutOfMemoryError",
            "cuBLAS error: CUBLAS_STATUS_ALLOC_FAILED",
        )
    )
```

On detection, if `run_state.gpu_escalations < max_escalations` and `ladder_remaining` is non-empty:
1. Tear down current pod (existing `destroy()` path).
2. Pop next rung from `ladder_remaining`, build new `GpuPlan`, atomically update run state.
3. Emit `gpu_escalated` SSE event.
4. Recreate pod with new plan; re-run experiment from scratch.
5. Increment `run_state.gpu_escalations`.

When ladder exhausted or cap exceeded: emit `gpu_fallback` (terminal), raise `GpuEscalationExhausted` with cumulative pod-cost summary so the user sees what was spent.

### Backend wiring

`RunpodBackend.__init__` gains optional `gpu_plan: GpuPlan | None`. When provided:
- `self.gpu_type = gpu_plan.runpod_id`
- `self.gpu_count = gpu_plan.gpu_count`
- `self.cloud_type = gpu_plan.cloud_type`
- `self.container_disk_gb = max(self.container_disk_gb, gpu_plan.container_disk_gb)`
- `self.volume_gb = max(self.volume_gb, gpu_plan.volume_gb)`

When `gpu_plan is None`: back-compat path — reads `Settings.runpod_*` as today.

`_backend_for_sandbox_mode` is updated to thread `run_state.gpu_plan` (if present) when constructing `RunpodBackend`. Other backends ignore the plan.

### Config surface (final)

| Env var | Settings field | Default | CLI flag |
|---|---|---|---|
| `REPROLAB_DYNAMIC_GPU` | `dynamic_gpu_enabled: bool` | `True` | `--dynamic-gpu / --no-dynamic-gpu` |
| `REPROLAB_FORCE_SINGLE_GPU` | `force_single_gpu: bool` | `True` | `--force-single-gpu / --no-force-single-gpu` |
| `REPROLAB_MAX_GPU_USD_PER_HOUR` | `max_gpu_usd_per_hour: float \| None` | `10.0` | `--max-gpu-usd-per-hour` |
| `REPROLAB_MAX_RUN_GPU_USD` | `max_run_gpu_usd: float \| None` | `10.0` | `--max-run-gpu-usd` |
| `REPROLAB_DYNAMIC_GPU_HEADROOM` | `dynamic_gpu_headroom: float` | `1.25` | `--dynamic-gpu-headroom` |
| `REPROLAB_DYNAMIC_GPU_FALLBACK_VRAM_GB` | `dynamic_gpu_fallback_vram_gb: int` | `24` | (env only) |
| `REPROLAB_DYNAMIC_GPU_MAX_ESCALATIONS` | `dynamic_gpu_max_escalations: int` | `2` | (env only) |
| (none) | (none) | (none) | `--vram-gb N` (manual override; bypasses LLM estimate) |

Empty string OR `0` on `MAX_GPU_USD_PER_HOUR` and `MAX_RUN_GPU_USD` means "no cap."

### SSE event types (added)

| Event | Payload | When |
|---|---|---|
| `gpu_resolved` | `{ plan: GpuPlan, source: "paper"|"fallback"|"manual" }` | After resolver returns |
| `gpu_escalated` | `{ from_sku, to_sku, escalation_index, reason: "cuda_oom"|"runpod_capacity" }` | After successful escalation |
| `gpu_fallback` | `{ reason, cumulative_pod_usd, raised: Optional[error] }` | When ladder exhausted or LLM cannot estimate |

All routed through `sse_bridge.sanitize_iteration` — same chokepoint as existing events.

## Error / fallback matrix

| Situation | Behavior | SSE event |
|---|---|---|
| LLM estimate confidence < 0.4 | Use `fallback_vram_gb` (24); count=1 | `gpu_fallback` (non-terminal) + `gpu_resolved` |
| Required VRAM > largest catalog SKU's VRAM | Fail fast: "Paper needs ≥X GB, largest catalog SKU is Y" | `gpu_fallback` (terminal) |
| Required SKU exceeds `MAX_GPU_USD_PER_HOUR × count` | Fail fast; error names cheapest in-cap SKU | `gpu_fallback` (terminal) |
| RunPod capacity error on pod create | Drop down ladder; retry with next-cheapest viable SKU | `gpu_escalated` (reason: runpod_capacity) |
| Pod hits CUDA OOM at experiment time | Tear down; pop ladder rung; recreate; re-run | `gpu_escalated` (reason: cuda_oom) |
| OOM with empty ladder OR escalations exhausted | Fail with cumulative-cost report | `gpu_fallback` (terminal) |
| `REPROLAB_DYNAMIC_GPU=off` + sandbox=runpod | Skip resolver; use legacy `Settings.runpod_gpu_type` | (no events) |
| Sandbox != runpod | Resolver runs for *informational* telemetry only; plan not consumed | `gpu_resolved` (source: informational) |
| `--vram-gb N` manual override | Skip LLM; treat N as estimate; multiplier still applied | `gpu_resolved` (source: manual) |
| `REPROLAB_FORCE_SANDBOX=docker` overrides `--sandbox runpod` | Resolver inert; legacy Docker path | (no events) |

## Testing

| File | Coverage |
|---|---|
| `tests/services/runtime/test_gpu_catalog.py` | Catalog sorted invariant; `find_ladder(min_vram, cap)` returns correct subset; aliases lookup |
| `tests/services/runtime/test_gpu_resolver.py` | Multiplier; tier-up; cap enforcement; force_single_gpu=on → count=1; force_single_gpu=off → count = floor(cap/per_gpu); confidence threshold → fallback; manual override bypasses LLM-derived estimate; pure-function determinism (same input → same plan) |
| `tests/services/runtime/test_gpu_resolver_parsing.py` | Regression set of 20 paper-hardware phrases → expected (VRAM, count, gpu_string) (vendored test data) |
| `tests/rlm/test_resolve_gpu_requirements.py` | Primitive emits `gpu_resolved` event; idempotent; cached in run_state; low-confidence → `gpu_fallback`; manual override path |
| `tests/rlm/test_runpod_oom_escalation.py` | exit-code-137 → escalation; stderr-substring match → escalation; max_escalations=2 honored; ladder-exhausted → `GpuEscalationExhausted` with cumulative cost |
| `tests/services/runtime/test_runpod_backend_gpu_plan.py` | `RunpodBackend` reads plan when provided; back-compat path with `gpu_plan=None`; volume_gb / container_disk_gb derivations |
| `tests/agents/runtime/test_oauth_runpod_orthogonality.py` | Guard: with `--model claude-oauth` + `--sandbox runpod`, no Anthropic API env var is ever read; pod env injection is empty for ANTHROPIC_API_KEY |
| `tests/cli/test_dynamic_gpu_flags.py` | CLI flags override env; `--vram-gb` bypasses LLM call |
| `tests/integration/test_dynamic_gpu_e2e.py` | Mocked-LLM, mocked-RunPod end-to-end: paper containing "trained on 8x A100 80GB" + `--max-gpu-usd-per-hour 1.5` → expected SKU is RTX A6000, count=1, source="paper" |

All new tests are **deterministic** (mocked LLM, mocked RunPod). No tests hit live RunPod; the existing `START_FULL_SMOKE=1` path remains the only live-pod check.

## Rollout phases (implementation order)

1. **Catalog + resolver pure logic** (`gpu_catalog.py`, `gpu_resolver.py`, schemas, tests). No production code paths touched.
2. **`RunBudget.max_run_gpu_usd`** field + `check_run_gpu_usd()` enforcement in `RunpodBackend.exec`.
3. **Primitive** `resolve_gpu_requirements` + system-prompt update + run-state checkpoint.
4. **Backend wiring**: `RunpodBackend.__init__(gpu_plan=None)` + `_backend_for_sandbox_mode` threading.
5. **OOM detection + escalation** in `run_experiment`.
6. **Settings + CLI flags** (config.py + cli.py).
7. **SSE event emission** through `sse_bridge`.
8. **UI**: render plan badge in `node-detail-sidebar.tsx`.
9. **Docs**: `CLAUDE.md` + `system_overview.md` updated with "Dynamic GPU selection" section.
10. **Codex review pass** on the final diff per user direction.

## Acceptance gates

A. **Determinism**: `pytest tests/services/runtime/test_gpu_resolver.py tests/services/runtime/test_gpu_catalog.py -q` green; resolver is provably pure (no I/O imports in `gpu_resolver.py`).
B. **Back-compat**: `pytest tests/services/runtime/test_runpod_delete_guardrails.py tests/rlm/test_runpod_wiring.py -q` still green with no changes to those tests; existing runs with `dynamic_gpu_enabled=False` behave identically to today.
C. **OAuth orthogonality**: `pytest tests/agents/runtime/test_oauth_runpod_orthogonality.py -q` green.
D. **Type check**: `npx tsc --noEmit` from `frontend/` passes; `python -m pytest tests/ -q` overall passes.
E. **No surface bloat**: `git diff --stat` shows additions concentrated in the new files + ≤6 modified files (config.py, cli.py, schemas.py, primitives.py, runpod_backend.py, sse_bridge.py).

## Out of scope (intentionally deferred)

- **Ingest-time pre-resolution** caching `paper.gpu_requirements` in paperMeta. Can be added later as a non-breaking layer on top.
- **Live RunPod API price refresh**. Catalog stays static; quarterly manual refresh.
- **`GpuMode` enum unification**. Existing enum stays sandbox-agnostic.
- **Multi-region awareness** (e.g., prefer EU pods). Out of scope.
- **Per-paper persistent pods** (re-use a warm pod across reruns). Out of scope; existing `REPROLAB_RUNPOD_POD_ID` already covers single-pod attachment.
- **Spot/interruptible SKUs**. Out of scope.

## Operational notes

- Catalog prices are **approximate, vendored, refreshed quarterly**. The resolver's cost decisions are correct in ranking; absolute prices may drift ±20%.
- The `MAX_RUN_GPU_USD` cap on `RunBudget` is enforced lazily — checked at the start of each `exec()` call in `RunpodBackend`, same surface as `check_pod_seconds`. Overage by 1 command's worth of pod-seconds is possible; sub-second precision is not a goal.
- When sandbox is **not** RunPod, the resolver still runs (LLM call still happens) to populate `paperMeta.gpu_requirements` for UI/telemetry. This is intentional — same UI affordance regardless of sandbox. The LLM cost is one extra completion per run; negligible.
