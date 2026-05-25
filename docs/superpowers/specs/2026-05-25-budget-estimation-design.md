# Budget Estimation — Design

**Status**: Locked 2026-05-25, ready for implementation.
**Owner**: Opus (spec + review) / Sonnet (executes) / Codex (adversarial review)
**Predecessor**: builds on the existing `resolve_gpu_requirements`, `build_workspace`, `cost_ledger.jsonl`, and `.preserved` marker scaffolding.

---

## Goal

Before a paper reproduction run starts, the user must see — and approve — an accurate, calibrated estimate of:

1. **GPU cost** on RunPod for the resolved best-fit SKU × estimated wall-clock hours.
2. **API cost** across every supported LLM provider, broken down per-model so the user can pick the cheapest viable root + sub-agent pair.
3. **Time** to finish the run (best-case + 90th percentile).
4. **A compressed-recipe alternative** that swaps paper-faithful schedules (e.g. 3000-epoch SGD) for modern fast equivalents (e.g. 200-epoch Adam), with the cost/time delta surfaced explicitly.

The estimate is pre-Begin and blocking: the user cannot launch a run without seeing the panel.

## Non-goals

- We do not aim for ±5% precision. Per-run cost is governed by adaptive RLM iterations the root model chooses; perfect estimation would require simulating the agent itself. Target precision: ±30% at launch, tightening to ±10% over time as `.preserved` runs accumulate calibration data.
- We do not auto-pick the cheapest provider for the user. We surface the trade and let them choose.
- We do not estimate cost for the experimental `rdr` or `rlm-pure` modes initially — `rlm` only in v1.

## Architecture

```
backend/services/pricing/
├── catalog.py        # MODEL_PRICING + GPU_PRICING tables; last_audited_utc per entry
├── estimator.py      # Public estimate_paper_budget(...) driver
├── calibration.py    # Reads .preserved runs + cost_ledger.jsonl; computes priors
├── cache.py          # Keyed on (paper_sha, recipe_mode, catalog_version)
└── schemas.py        # PaperBudgetEstimate Pydantic model
```

```
backend/routes/
└── estimate.py       # POST /paper/estimate
```

```
frontend/src/components/lab/budget/
├── budget-panel.tsx  # The blocking modal
└── budget-panel.css
```

### `catalog.py`

Two top-level dicts, both with `last_audited_utc` and `schema_version`:

```python
MODEL_PRICING: dict[str, ModelPriceEntry] = {
    "anthropic.claude-opus-4-7":     ModelPriceEntry(usd_per_1m_input=15.0,  usd_per_1m_output=75.0,  …),
    "anthropic.claude-sonnet-4-6":   ModelPriceEntry(usd_per_1m_input=3.0,   usd_per_1m_output=15.0,  …),
    "anthropic.claude-haiku-4-5":    ModelPriceEntry(usd_per_1m_input=0.80,  usd_per_1m_output=4.0,   …),
    "anthropic.claude-oauth":        ModelPriceEntry(usd_per_1m_input=0.0,   usd_per_1m_output=0.0,   subscription_note="…"),
    "openai.gpt-5":                  ModelPriceEntry(usd_per_1m_input=1.25,  usd_per_1m_output=10.0,  …),
    "openai.gpt-5-mini":             ModelPriceEntry(usd_per_1m_input=0.25,  usd_per_1m_output=2.0,   …),
    "openai.gpt-5-nano":             ModelPriceEntry(usd_per_1m_input=0.05,  usd_per_1m_output=0.40,  …),
    "google.gemini-2-5-pro":         ModelPriceEntry(usd_per_1m_input=1.25,  usd_per_1m_output=10.0,  …),
    "google.gemini-2-5-flash":       ModelPriceEntry(usd_per_1m_input=0.075, usd_per_1m_output=0.30,  …),
    "featherless.qwen3-coder-480b":  ModelPriceEntry(usd_per_1m_input=0.0,   usd_per_1m_output=0.0,   subscription_usd_per_month=9.0, subscription_note="…"),
    "moonshot.kimi-k2-5":            ModelPriceEntry(usd_per_1m_input=0.15,  usd_per_1m_output=2.50,  …),
}

GPU_PRICING: dict[str, GpuPriceEntry] = {
    # Mirror gpu_catalog.py SKU IDs so the resolver and the estimator share keys.
    "rtx_4090_24gb_community": GpuPriceEntry(usd_per_hour=0.34, …),
    "rtx_4090_24gb_secure":    GpuPriceEntry(usd_per_hour=0.69, …),
    "rtx_a6000_48gb_secure":   GpuPriceEntry(usd_per_hour=0.79, …),
    "a100_pcie_80gb":          GpuPriceEntry(usd_per_hour=1.89, …),
    "a100_sxm_80gb":           GpuPriceEntry(usd_per_hour=1.89, …),
    "h100_pcie_80gb":          GpuPriceEntry(usd_per_hour=2.69, …),
    "h100_sxm_80gb":           GpuPriceEntry(usd_per_hour=3.29, …),
    "h200_141gb":              GpuPriceEntry(usd_per_hour=3.99, …),
}

CATALOG_SCHEMA_VERSION = 1   # bump on every audit pass; invalidates cache
```

The actual prices are the operator's responsibility per quarter. The schema version exists so a stale estimate gets discarded when the table is bumped.

**Audit cadence**: quarterly. Each audit pass writes `docs/runbooks/pricing-audit-YYYY-QN.md` documenting every changed `MODEL_PRICING` or `GPU_PRICING` entry, bumps `CATALOG_SCHEMA_VERSION`, and commits both. The runbook is the audit trail; reviewers can `git blame` any pricing entry back to a runbook.

### `estimator.py`

Single public function:

```python
async def estimate_paper_budget(
    source: PaperSource,            # PdfPathSource | ArxivIdSource | ArxivUrlSource
    *,
    recipe_mode: Literal["strict", "compressed"] = "strict",
    target_root_model: str | None = None,   # if None, estimate for all supported roots
) -> PaperBudgetEstimate:
    ...
```

Steps:
1. **Resolve paper identity**: SHA256 of the PDF bytes (or, for arXiv, fetch then hash). Cache lookup → return immediately on hit.
2. **Ingestion**: reuse the existing `build_workspace` to produce `claim_map` (paper metadata, datasets, training_recipe, model_architecture). This is the same code path a real run takes — no second implementation.
3. **GPU resolution**: call `resolve_gpu_requirements(claim_map, …)` → `(GpuPlan, vram_estimate_gb)`. The estimator only uses `GpuPlan.primary_sku_id` for v1 (no ladder fan-out).
4. **Experiment-count + epoch estimate**: one Sonnet call with the paper text and a tight system prompt:
   > "Return JSON `{experiment_count, total_epochs_across_all_experiments, avg_epoch_seconds_on_target_gpu, confidence: 'high'|'medium'|'low'}`. Use the paper's reported training cost when stated; otherwise extrapolate from the architecture, dataset size, and batch size."

   Cap input at ~30k tokens (slice the paper), one-shot, JSON-mode. Cost: ~$0.02.
5. **Wall-clock estimate**:
   ```
   estimated_seconds = experiment_count × total_epochs × avg_epoch_seconds × overhead_multiplier
   overhead_multiplier = 1.5  # accounts for sandbox boot, eval, checkpoint I/O
   ```
   Bound check: if the recipe is compressed, multiply by `compression_ratio` (e.g., 0.15 for paper-faithful → claim-match).
6. **API token estimate**: from `calibration.py`, look up per-primitive averages keyed on `(paper_category, recipe_mode)` where category is one of `vision_cls`, `nlp_seq`, `rl_policy`, `generative`, etc. Multiply by predicted primitive call counts.
7. **Per-provider cost calc**: for each `(root_model, sub_agent_model)` pair in the supported set, compute `input_tokens × usd_per_1m_input + output_tokens × usd_per_1m_output + subscription_share_for_this_run`.
8. **Both recipes**: re-run steps 5+6+7 with `recipe_mode='compressed'` if v1 only generated strict.
9. **Return** the populated `PaperBudgetEstimate`.

### `calibration.py`

```python
def recompute_calibration(runs_root: Path) -> CalibrationData:
    """Aggregate .preserved runs into per-primitive token averages.

    Reads runs/<id>/.preserved + runs/<id>/cost_ledger.jsonl across every
    preserved run. Computes:
      - per_primitive_avg_input_tokens[primitive_name]
      - per_primitive_avg_output_tokens[primitive_name]
      - primitive_call_count_per_recipe[recipe_mode]
      - bucketed_by_paper_category[category][primitive_name] = avg

    Persists to data/calibration.json with schema_version.
    """
```

`recompute_calibration` runs after each successful run (hooked into `write_final_report_rlm` post-`.preserved`-stamp). Fast — file I/O over ~30 preserved runs is sub-second.

The estimator queries calibration data via `get_primitive_priors(paper_category, recipe_mode)`. When n < 5 for a category, it falls back to the static defaults.

### Cache

`backend/services/pricing/cache.py` — files under `runs_root/_estimates/`:

```
runs/_estimates/
└── {sha8_paper}_{recipe_mode}_{catalog_version}.json
```

`get_cached(sha, mode)` returns the cached estimate if its `catalog_schema_version == CATALOG_SCHEMA_VERSION` and `calibration_schema_version == CALIBRATION_SCHEMA_VERSION`.

### HTTP API: `POST /paper/estimate`

**Request body**:
```json
{
  "source_kind": "pdf_path" | "arxiv_url" | "arxiv_id",
  "source": "<value>",
  "recipe_mode": "strict" | "compressed"   // optional, defaults to "both"
}
```

Or multipart upload (PDF body, same shape as `/runs/upload`).

**Response (200)**:
```json
{
  "paper": {"id": "1412.6980", "title": "Adam", "sha256": "abc…"},
  "gpu": {
    "sku_id": "rtx_4090_24gb_community",
    "label": "RTX 4090 24GB (RunPod COMMUNITY)",
    "usd_per_hour": 0.34,
    "estimated_hours": {"p50": 4.5, "p90": 6.2},
    "usd_total": {"p50": 1.53, "p90": 2.11}
  },
  "api": [
    {
      "provider": "anthropic",
      "model_id": "claude-opus-4-7",
      "input_tokens": 250000,
      "output_tokens": 80000,
      "usd": 9.75,
      "is_subscription": false,
      "subscription_note": null
    },
    {
      "provider": "anthropic",
      "model_id": "claude-oauth",
      "input_tokens": 250000,
      "output_tokens": 80000,
      "usd": 0.0,
      "is_subscription": true,
      "subscription_note": "Covered by your Claude Code subscription — rate limits apply."
    },
    {…openai.gpt-5…},
    {…google.gemini-2-5-pro…},
    {…featherless.qwen3-coder-480b…},
    {…moonshot.kimi-k2-5…}
  ],
  "recipes": {
    "strict": {
      "label": "Strict reproduction",
      "description": "Paper's recipe verbatim: 3000-epoch SGD with linear decay from lr=10.",
      "gpu_usd": 1.53,
      "api_usd_best": 0.85,       // cheapest provider total
      "api_usd_worst": 9.75,      // priciest provider total
      "wall_clock_hours_p50": 4.5,
      "fidelity_label": "high",
      "declared_reductions": []
    },
    "compressed": {
      "label": "Claim-match (minimize-compute)",
      "description": "Modern fast equivalent: 200-epoch Adam at lr=0.001.",
      "gpu_usd": 0.30,
      "api_usd_best": 0.20,
      "api_usd_worst": 2.40,
      "wall_clock_hours_p50": 0.9,
      "fidelity_label": "claim-match",
      "declared_reductions": [
        "Replaced 3000-epoch SGD+linear-decay with 200-epoch Adam (matches reported accuracy in known follow-up work).",
        "Reduced batch sweep from {32, 64, 128, 256} to {64}."
      ]
    }
  },
  "calibration_metadata": {
    "based_on_n_preserved_runs": 12,
    "precision_window_pct": 25,
    "catalog_schema_version": 1,
    "calibration_schema_version": 1,
    "estimated_at_utc": "2026-05-25T17:00:00Z"
  }
}
```

### Frontend: `BudgetPanel`

Inserted between the upload zone and the Begin button. Lifecycle:

1. File dropped / arXiv URL pasted → upload-view sets `pendingEstimate = true`.
2. `useEffect` POSTs to `/api/demo/estimate`, sets `estimate` state on response.
3. While loading, the Begin button shows a spinner and is disabled.
4. Once estimate lands, `BudgetPanel` renders:
   - **Two recipe cards** (strict / compressed) side-by-side, each showing `gpu_usd`, `api_usd_range`, `wall_clock_hours_p50`, `declared_reductions` (compressed only).
   - **Provider breakdown table** below, expandable per-row to show input/output token breakdown.
   - User selects a recipe → sets `selectedRecipe` (default: strict).
   - User picks a provider → updates the displayed total.
5. Begin button enabled, and clicking it forwards `selectedRecipe + selectedProvider + estimate.id` to the actual `/runs/upload` body — backend uses these as defaults for `--minimize-compute` and `--max-usd`.

UI sketch:

```
┌────────────────────────────────────────────────────────────────────┐
│  Estimated budget for "Adam: A Method for Stochastic Optimization" │
│  ▸ GPU: RTX 4090 24GB COMMUNITY ($0.34/hr × ~4.5h = $1.53)         │
│  ▸ Confidence: ±25% (calibrated from 12 prior runs)                │
│                                                                    │
│  ┌──────────────────────┐ ┌──────────────────────┐                 │
│  │ Strict reproduction  │ │ Claim-match (faster) │                 │
│  │ $2.38 │ ~4.5h        │ │ $0.50 │ ~0.9h        │                 │
│  │ Paper's recipe       │ │ Modern fast subs.    │                 │
│  │ verbatim             │ │ 2 reductions noted   │                 │
│  └──────────────────────┘ └──────────────────────┘                 │
│                                                                    │
│  ▸ API provider breakdown (click to expand) ▼                      │
│    ◯ Claude Opus 4.7        — $9.75 (per-token)                    │
│    ◯ Claude OAuth subscript — $0.00 (sub. limits apply)            │
│    ● OpenAI GPT-5           — $0.85 (per-token, cheapest)          │
│    ◯ Gemini 2.5 Flash       — $0.20                                │
│    ◯ Qwen3-Coder Featherless — $0.09 ($9/mo amortized × this run)   │
│                                                                    │
│  [ Begin ▷ ]                                                       │
└────────────────────────────────────────────────────────────────────┘
```

### Coupling the estimate to the actual run

Per the user's lock on Q8: the panel estimate becomes the budget cap.

When Begin is clicked:
- Frontend posts to `/runs/upload` (or `/runs/arxiv`) with extra body fields:
  - `estimate_id: <sha8_paper>_<recipe_mode>_<catalog_version>` (acts as cache key)
  - `--max-usd` defaulted to `1.5 × api_usd_best`
  - `--max-pod-seconds` defaulted to `1.5 × estimated_hours × 3600`
- Backend reads the cached estimate, forwards budget defaults to `cmd_reproduce` via the existing `_REPRODUCE_DEFAULTS` merge path.
- User can still override in the Advanced panel (existing `--max-usd` input).

### `.preserved` calibration feedback

Every time a run finishes with a `.preserved` marker:
1. `write_final_report_rlm` already writes the marker (shipped).
2. We hook a post-write call: `calibration.recompute_calibration(runs_root)`.
3. `calibration.json` is updated.
4. Next estimate uses the fresh priors.

This is what closes the precision loop: every successful run makes the next estimate tighter.

## Pre-flight invariants (codex-blockable)

These must all be true after implementation:

1. `MODEL_PRICING` entries cover every provider listed in `ROOT_MODELS` from `backend/agents/rlm/models.py`. No orphan model.
2. `GPU_PRICING` keys are exactly the SKU IDs from `gpu_catalog.py`. No drift.
3. `last_audited_utc` is within 90 days for every entry — failing assertion logs a stale-pricing warning at startup but doesn't crash.
4. `CATALOG_SCHEMA_VERSION` bump invalidates every cached estimate (verified by unit test).
5. `OAuth` provider entries have `usd_per_1m_input == 0.0` and a non-empty `subscription_note`.
6. Featherless / Moonshot subscription pricing surfaces `subscription_usd_per_month` and an explicit per-run amortization basis (`assumed_runs_per_month`).
7. The `/paper/estimate` endpoint never starts a real run — guarded by an assertion that the response is built without spawning a subprocess.
8. The estimate cache key includes both `CATALOG_SCHEMA_VERSION` and `CALIBRATION_SCHEMA_VERSION`. Either bump invalidates.
9. `BudgetPanel` cannot be bypassed: Begin button MUST observe `estimateReady === true` before allowing submission.
10. Failing estimate (network / LLM error) does NOT block the user from launching — surface the error, show a "Skip estimate and start anyway" fallback button. Production resilience.

## Adversarial test cases (for Codex review)

Codex will be asked: "Find ways this estimate could be wrong by >50% in production. List concrete failure modes."

Expected findings to seed:
- Paper with no reported training cost AND a non-standard architecture → LLM hallucinates wall-clock; should drop to "low confidence, ±100%"
- Paper with multiple GPU configurations (ablation) → estimator picks the most expensive one
- Sandbox boot overhead on RunPod cold-start can be 5-10 min, easy 50%+ skew on a 30-min run
- LLM Sonnet sub-agent retry-loops on prompt failures (e.g. ImportError) burn 3× the calibrated token average
- OAuth subscription rate-limits cause the run to stall mid-`implement_baseline`; cost shows $0 but the run effectively fails
- Pricing audit drift > 90 days produces a stale-warning but still computes — user sees wrong numbers
- Cache key includes `catalog_version` but not the calibration table hash, so a calibration update doesn't invalidate stale estimates
- Schema-version mismatch silently breaks the cache hit path

## Open items deferred to v1.1

- Multi-GPU SKU support (currently single-GPU only)
- `rdr` / `rlm-pure` mode estimates
- Per-experiment ladder showing each individual experiment's predicted cost
- Slack/webhook notification when actual cost exceeds estimate by >50%

## Implementation order (TDD)

1. `catalog.py` + `schemas.py` with a small unit test for entry shape.
2. `cache.py` with round-trip + invalidation tests.
3. `calibration.py` reading `.preserved` runs (test fixture).
4. `estimator.py` happy path (mocked LLM call) → end-to-end test.
5. `routes/estimate.py` + FastAPI binding → integration test.
6. Frontend `BudgetPanel` component (vitest renders).
7. `useRun` integration: panel state, Begin-button gating, body merge.
8. Codex adversarial review pass.
9. Manual smoke test: upload Adam PDF, see panel, click Begin, verify the actual run honors the budget cap.
10. Ship.

## Out of scope, named explicitly

- The estimator does not predict the rubric SCORE. We measure cost, not quality.
- The estimator does not predict failure probability. A run can fail and consume part of its budget; the user accepts that as a known property of agent runs.
- We do not compare against external pricing APIs (e.g. anthropic.com/pricing JSON) — there isn't one, and scraping is brittle.
