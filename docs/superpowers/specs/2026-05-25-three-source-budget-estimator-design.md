# Three-Source Budget Estimator — Design

**Status:** proposed (2026-05-25)
**Supersedes:** parts of `2026-05-25-budget-estimation-design.md` (the LLM-only `_extract_workload` leg + zero-floor precision math).
**Locked invariants from predecessor (kept):** pre-Begin blocking panel, dual cache schema (`{sha8}_{recipe_mode}_{catalog_v}_{calibration_v}`), `.preserved` marker as GC protection, secret redaction at `_extract_text_from_pdf`, 2.0× overhead multiplier configurable via `REPROLAB_ESTIMATE_OVERHEAD_MULTIPLIER`.

---

## Why this refactor

Today's estimator has three known failure modes:

1. **Silent zero contamination.** OAuth's `claude-agent-sdk` does not return token-usage in responses. Every `cost_ledger.jsonl` row written by an OAuth run has `input_tokens: 0` / `output_tokens: 0`. `calibration.py:recompute_calibration` averages those zeros instead of skipping them, and `get_primitive_priors` honors a zero-valued entry over the static default (because `dict.get(key, default)` returns the stored 0, not the default). Net effect with our 30 preserved runs: calibration overrides defaults with all-zero priors → estimated API time/cost trends to ~$0.
2. **GPU leg is a single LLM opinion.** `_extract_workload` calls Sonnet once on the redacted paper text and trusts its `hours_per_experiment` / `num_experiments` numbers. No grounding against the 31 preserved runs we already have. Today's VAE estimate said `confidence=0.4` and "16 GB safe default" — that's an admission of weakness, not a calibrated answer.
3. **Precision window is decorative.** `precision_window = max(10, 100 − calibration_n × 5)` collapses uncertainty into a sample-count function. It cannot widen when k-NN neighbors disagree, and cannot shrink when k-NN neighbors agree closely.

The user requested: *more precise, robust, elegant, depending on compute and paper.* That phrasing aligns with a per-paper feature vector + k-NN over past runs + an explicit uncertainty model.

---

## Architecture

Three **independent** estimators, each returning a `PointEstimate(mean, sigma, source, n_samples, detail)`:

```
                    ┌──────────────────────────────┐
PaperFeatures ──────►  HeuristicEstimator           ├──┐
                    └──────────────────────────────┘  │
                    ┌──────────────────────────────┐  │  inverse-variance
PaperFeatures ──────►  KnnEstimator (preserved      ├──┼─►  combine
preserved_runs ─────►  runs filtered by category)   │  │
                    └──────────────────────────────┘  │
                    ┌──────────────────────────────┐  │
PaperFeatures ──────►  LlmEstimator (_extract_      ├──┘
paper_text ─────────►  workload, kept as-is)        │
                    └──────────────────────────────┘
                                                       ▼
                                          PaperBudgetEstimate(mean, ci_lo, ci_hi, …)
```

The three estimators are deliberately independent — they consume the same `PaperFeatures` and produce comparable `(time_s, cost_usd)` tuples in the same units. The combiner is a pure function of three `PointEstimate`s with no knowledge of how each was produced. This is the elegance lever: each estimator can be replaced or extended without touching the combiner.

### Combiner: inverse-variance weighting

For each output dimension (`api_time_s`, `gpu_hours`, `api_cost_usd`, `gpu_cost_usd`):

```python
weights   = [1 / max(σᵢ², ε) for σᵢ in sigmas]
μ_final   = Σ (μᵢ · wᵢ) / Σ wᵢ
σ_final   = 1 / √Σ wᵢ
```

Floor `ε = 1e-9` so a degenerate σ=0 estimator can't dominate by infinity. Sources at high σ are downweighted automatically, so the cold-start behavior (k-NN absent, k-NN σ=∞) reduces to a weighted average of heuristic+LLM — exactly the no-data fallback we want.

### Confidence interval

Report `[μ_final − 1.96·σ_final, μ_final + 1.96·σ_final]` as the 95% CI. UI shows the mean as the headline, the ±half-width as the precision number. When σ_final's relative width exceeds 50%, the UI badges the estimate as `low_confidence`.

---

## Component design

### 1. `paper_features.py` (new)

Single canonical paper-feature extractor used by all three estimators and by retrodiction logging. Pure function on `(paper_text, llm_client)`:

```python
@dataclass(frozen=True)
class PaperFeatures:
    sha8: str                          # for cache + identity
    category: str                      # "vae" | "optimizer" | "cnn" | "rl" | "nlp" | "other"
    model_size_class: str              # "tiny" (<1M) | "small" (1M-100M) | "medium" (100M-1B) | "large" (1B+) | "unknown"
    estimated_vram_gb: int             # carry-over from _extract_workload
    num_experiments: int               # carry-over
    datasets: tuple[str, ...]          # ("MNIST",) etc.
    gpu_hints: tuple[str, ...]         # ("RTX 4090",) from _extract_gpu_clues regex
    feature_vector: tuple[float, ...]  # numeric encoding for k-NN distance (10 dims)

def extract_features(paper_text: str, llm_client) -> PaperFeatures: ...
```

Implementation:
- Run existing `_extract_gpu_clues` regex first (cheap)
- Run existing `_extract_workload` LLM call (one round-trip) but reshape its output
- Categorize by keyword presence (regex over abstract + intro): `dropout|variational|autoencoder` → `vae`, `optimizer|sgd|adam` → `optimizer`, `convolutional|imagenet|cifar` → `cnn`, etc. Multi-match → highest-priority single category by a fixed table.
- Cache the result alongside the budget estimate keyed by sha8.

### 2. `estimators/heuristic.py` (new)

Closed-form, always available, conservative σ. Per dimension:

```
api_time_s   = Σ_primitives (call_count × avg_tokens) / throughput_tok_per_s
gpu_hours    = features.num_experiments × hours_per_experiment(category, model_size_class)
api_cost_usd = Σ_primitives (call_count × avg_tokens) × $price_per_token
gpu_cost_usd = gpu_hours × catalog.rate(features.estimated_vram_gb)
```

Uses `_DEFAULT_PRIORS` and `_DEFAULT_PRIMITIVE_CALL_COUNTS` (already exist). `hours_per_experiment` is a 5×4 table (category × model_size_class) seeded from public reproduction times (e.g., MNIST CNN tiny = 0.1h, ImageNet CNN large = 50h).

σ = 50% of μ in the no-data regime. This is the cold-start safety net.

### 3. `estimators/knn.py` (new)

For each preserved run, load `runs/<id>/timing.json` (new file — see §timing capture below) and `final_report.json::paper_features`. Build a flat list of `(features, timing)` tuples at module init, refreshed each call (small N — under 100 runs).

K-NN logic:
- Filter preserved runs to same `category`. If fewer than 3, expand to category superset; if still under 3, return `unavailable` (σ=∞ → no weight in combiner).
- Distance = Euclidean over `feature_vector` (normalized).
- Pick k=5 nearest neighbors. Use weighted median (closer neighbors weighted higher) for μ; sample std-dev among neighbors for σ.

When the empirical data improves (more preserved runs), σ shrinks organically — no manual tuning of `precision_window`.

### 4. `estimators/llm.py` (refactor of today's `_extract_workload`)

Keep `_extract_workload` but extract it into its own class implementing the `Estimator` protocol. σ derived from the LLM's stated `confidence` field: `σ = (1 − confidence) × μ`. So confidence=0.8 → σ=20% of μ.

This is the same LLM call we have today — just wrapped in the same interface. No behavior change in isolation; the difference is the combiner.

### 5. `ensemble.py` (new)

```python
def combine(estimates: list[PointEstimate]) -> PaperBudgetEstimate:
    # Per dimension: inverse-variance weighted mean + combined σ
    # Returns the existing PaperBudgetEstimate shape (back-compat with UI)
```

Trivial logic, but isolated for testability. Edge cases: all σ=∞ (return None — caller falls back to skip-estimate UX), one σ=0 (cap to ε), negative μ (clamp to 0).

### 6. `timing.py` (new, capture side)

Stamped at the same place as `.preserved` (in `write_final_report_rlm`):

```json
{
  "wall_clock_s": 1234,
  "primitive_wall_clock_s": {"understand_section": 12.3, ...},
  "primitive_call_counts": {"understand_section": 3, ...},
  "gpu_hours": 0.34,        // total seconds containers had GPU device requests / 3600
  "gpu_type": "rtx2060",
  "gpu_count": 1,
  "iterations": 4,
  "rubric_score": 0.42,
  "paper_features": {...}   // copy of PaperFeatures used at estimate time, or recomputed at terminal
}
```

Primitive wall-clock comes from existing `worker_reports.jsonl` `started_at`/`finished_at` aggregation. GPU hours: walk `experiment_runs.jsonl` start/end timestamps; multiply count of `device_requests`-present container runs.

### 7. Calibration fixes (must do, included in this PR)

Same scope as the surgical option I mentioned — these are non-negotiable for the ensemble to work:

- `calibration.py:recompute_calibration`: skip ledger rows where `(input_tokens + output_tokens) == 0` (treat as missing, not as data point).
- `calibration.py:get_primitive_priors`: treat zero-valued calibration entry as missing → fall back to default. Implemented as `entry.get(k) or defaults[k]` (since 0 is falsy).

Both are one-line changes with unit tests proving the no-OAuth-data path falls back cleanly.

---

## Data flow at estimate time

1. Frontend uploads PDF → backend `POST /paper/estimate` (existing route)
2. `_extract_text_from_pdf` + redact (existing)
3. `extract_features(text, llm)` → `PaperFeatures` (one LLM call)
4. Each of the three estimators runs (heuristic is sync; k-NN reads disk; LLM is the same call as step 3 if we cache it — yes, share the workload call)
5. `combine([heuristic, knn, llm])` → `PaperBudgetEstimate`
6. Cache by `{sha8}_{recipe_mode}_{catalog_v}_{calibration_v}` (unchanged)
7. Frontend renders with new `breakdown` section listing each source's contribution

Estimate latency: still one LLM round-trip + a few ms of math + a few ms of disk I/O. Net change: same wall-clock, more grounded answer.

---

## UI changes (frontend/components/lab/budget/)

`BudgetPanel.tsx` gains a collapsible "How this estimate was made" detail:

```
Estimate: $X (±$Y, 95% CI)
└── ▾ Sources
    ├── k-NN over 7 similar past runs:  $X₁ (±$Y₁)   weight 0.62
    ├── LLM analysis (confidence 0.7):  $X₂ (±$Y₂)   weight 0.28
    └── Heuristic baseline:             $X₃ (±$Y₃)   weight 0.10
```

When k-NN is unavailable: row is hidden, weights renormalize between heuristic and LLM. UI badge `low_confidence` shown when k-NN row is missing OR when σ_final / μ_final > 0.5.

Recipe cards (strict/compressed) unchanged.

---

## Backward compatibility

- `PaperBudgetEstimate` schema gains optional `breakdown: list[SourceBreakdown]` (default `[]` for old cached estimates). Old caches still readable.
- `CATALOG_SCHEMA_VERSION` stays at 1.
- `CALIBRATION_SCHEMA_VERSION` bumps 1 → 2 (skip-zero behavior is semantically different). Old `data/calibration.json` is invalidated on first load; next preserved run triggers recompute. No user-visible disruption.
- `_extract_workload` exported signature unchanged — gpu resolver path keeps working.

---

## Testing

Each estimator tested in isolation:
- `test_heuristic.py`: 4 fixtures (tiny VAE, large CNN, RL paper, unknown category). Expects μ within hand-calculated bounds; σ ≥ 50% of μ.
- `test_knn.py`: synthetic preserved-runs directory. Cold-start (N=2 same-category) returns unavailable; warm (N=5+) returns matching neighbors; sigma shrinks as N grows.
- `test_llm.py`: stub LLM client; assert confidence field maps to σ correctly.
- `test_ensemble.py`: combine math. Edge cases: σ=∞ (should be no-op), σ=0 (should be capped, not div-by-zero), all sources agree (σ_final shrinks), sources disagree (σ_final widens).
- `test_paper_features.py`: category classification on hand-labelled samples covering each category.
- `test_calibration_skip_zero.py`: synthetic ledger with mix of zero and non-zero rows; assert zeros are skipped.

Integration:
- `test_estimate_endpoint.py`: end-to-end POST /paper/estimate with stubbed LLM, against a synthetic preserved-runs directory. Validates the breakdown structure and that the final μ is between min and max of source μs.
- Retrodiction: a small CLI `python -m backend.services.pricing.retrodict` that walks each preserved run, computes an estimate as if at upload time (excluding that run from k-NN), and reports estimated-vs-actual delta. Output is informational, not part of CI yet, but available for tracking accuracy over time.

---

## Implementation phases

| Phase | Output | Tests | Risk |
|---|---|---|---|
| 1 | `paper_features.py` + `timing.py` capture wired to `.preserved` write | feature classification + timing.json schema | low — purely additive |
| 2 | `estimators/heuristic.py`, `estimators/knn.py`, `estimators/llm.py` (refactor existing `_extract_workload` into class form) + `ensemble.py` + the combiner unit tests | 6 test files | medium — replaces existing logic, regression risk on first deploy |
| 3 | Calibration zero-skip + version bump v1→v2 | calibration test | low — surgical |
| 4 | UI: collapsible breakdown + low_confidence badge | frontend snapshot + e2e | low |
| 5 | Codex adversarial review + bug fixes | — | medium — surfaces real issues |

Each phase is a separate commit. The whole PR ships together (no half-finished implementation per the user's standing rule).

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Category misclassification → wrong k-NN cohort → biased estimate | Conservative classifier (defaults to `other`); k-NN unavailable in `other` until ≥10 samples; superset fallback |
| Few preserved runs in early days | k-NN reports `unavailable`, combiner downweights automatically; heuristic+LLM still produce a reasonable answer |
| Outlier preserved runs poison the median | Weighted median (not mean) for k-NN; outlier detection via IQR threshold (drop neighbors >2× IQR from cluster centroid) |
| LLM disagrees strongly with k-NN | Combiner's job — high σ_LLM (if confidence is honest) means it gets downweighted; if LLM is consistently optimistic, retrodiction surfaces the bias and we can add a per-source calibration factor in a follow-up |
| Timing.json missing on old preserved runs | Backfill script: walk preserved runs, recompute `paper_features` from cached estimate (when present) and `wall_clock_s` from `final_report.json::started_at/completed_at`; runs without an existing estimate get features recomputed on the fly |
| Schema change v2 invalidates existing calibration.json | One run cycle to repopulate; falls back to defaults cleanly during the transition |

---

## Out of scope (intentional)

- Bayesian prior updating beyond the inverse-variance combiner (mathematically appealing but overkill at N<100)
- Online learning / RL of source weights (premature optimization)
- Multi-paper batch estimation
- Cost forecasting under different recipe modes other than `strict` / `compressed` (these stay hardcoded; could grow into a recipe DSL later)

---

## Open question

None — the user approved the three-source ensemble option explicitly. Phasing above is my recommendation; if you want a different split (e.g., phases 1+3 first, then 2+4+5 as a follow-up PR), say so before I start the implementation plan.
