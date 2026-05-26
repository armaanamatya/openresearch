"use client";

import { useEffect, useState } from "react";

import "./budget-panel.css";

// ---------------------------------------------------------------------------
// Types — mirror backend.services.pricing.schemas
// ---------------------------------------------------------------------------

export interface ApiCostBreakdown {
  provider: string;
  model_id: string;
  input_tokens: number;
  output_tokens: number;
  usd: number;
  is_subscription: boolean;
  subscription_note: string | null;
}

export interface RecipeEstimate {
  label: string;
  description: string;
  gpu_usd: number;
  api_usd_best: number;
  api_usd_worst: number;
  wall_clock_hours_p50: number;
  fidelity_label: "high" | "claim-match";
  declared_reductions: string[];
}

export interface GpuEstimate {
  sku_id: string;
  label: string;
  usd_per_hour: number;
  estimated_hours: { p50: number; p90: number };
  usd_total: { p50: number; p90: number };
  // PR-ε.6: ensemble sigma fields (optional — absent on old cached estimates)
  estimated_hours_sigma?: number;
  low_confidence?: boolean;
}

export interface CalibrationMetadata {
  based_on_n_preserved_runs: number;
  precision_window_pct: number;
  catalog_schema_version: number;
  calibration_schema_version: number;
  estimated_at_utc: string;
}

export interface SourceBreakdown {
  source: "heuristic" | "knn" | "llm" | string;
  mean: number;
  sigma: number;       // Infinity when unavailable
  weight: number;      // normalised inverse-variance weight (0 when unavailable)
  n_samples: number;
}

export interface PaperBudgetEstimate {
  paper: { id?: string; title?: string; sha256?: string };
  gpu: GpuEstimate;
  api: ApiCostBreakdown[];
  recipes: Record<string, RecipeEstimate>;
  calibration_metadata: CalibrationMetadata;
  estimate_id: string;
  // PR-ε.6: ensemble breakdown (optional — absent on old cached estimates)
  estimate_breakdown?: SourceBreakdown[];
}

export type RecipeMode = "strict" | "compressed";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtUsd(v: number): string {
  if (v === 0) return "$0.00";
  if (v < 0.01) return `$${v.toFixed(4)}`;
  if (v < 10) return `$${v.toFixed(2)}`;
  return `$${v.toFixed(2)}`;
}

function fmtHours(h: number): string {
  if (h < 1) return `${Math.round(h * 60)} min`;
  return `${h.toFixed(1)}h`;
}

function fmtTokens(n: number): string {
  if (n < 1000) return n.toString();
  if (n < 1_000_000) return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}k`;
  return `${(n / 1_000_000).toFixed(1)}M`;
}

// ---------------------------------------------------------------------------
// BudgetPanel — the blocking modal
// ---------------------------------------------------------------------------

export interface BudgetPanelProps {
  estimate: PaperBudgetEstimate | null;
  loading: boolean;
  error: string | null;
  selectedRecipe: RecipeMode;
  selectedProvider: string | null;
  onSelectRecipe: (mode: RecipeMode) => void;
  onSelectProvider: (modelId: string) => void;
  onSkip: () => void;
  onConfirm: () => void;
  busy: boolean;
}

function fmtSigma(sigma: number, mean: number): string {
  if (!isFinite(sigma)) return "unavailable";
  if (mean === 0) return "±0h";
  const pct = Math.round((sigma / mean) * 100);
  return `±${fmtHours(sigma)} (${pct}%)`;
}

function SourceLabel({ source }: { source: string }) {
  switch (source) {
    case "heuristic": return <span title="Category × model-size lookup table">Heuristic baseline</span>;
    case "knn":       return <span title="k nearest preserved past runs">k-NN (past runs)</span>;
    case "llm":       return <span title="LLM analysis of paper text">LLM analysis</span>;
    default:          return <span>{source}</span>;
  }
}

export function BudgetPanel({
  estimate,
  loading,
  error,
  selectedRecipe,
  selectedProvider,
  onSelectRecipe,
  onSelectProvider,
  onSkip,
  onConfirm,
  busy
}: BudgetPanelProps) {
  const [breakdownOpen, setBreakdownOpen] = useState(false);
  const [howOpen, setHowOpen] = useState(false);

  if (loading) {
    return (
      <div className="budget-panel budget-panel--loading">
        <div className="budget-panel-spinner" aria-label="Loading estimate" />
        <p className="budget-panel-loading-text">
          Estimating budget &amp; runtime…
        </p>
        <p className="budget-panel-loading-sub">
          Analyzing paper, resolving GPU SKU, computing per-provider API cost.
        </p>
      </div>
    );
  }

  if (error || !estimate) {
    return (
      <div className="budget-panel budget-panel--error">
        <h3 className="budget-panel-title">Estimate unavailable</h3>
        <p className="budget-panel-error-msg">
          {error ?? "The budget estimate could not be computed."}
        </p>
        <p className="budget-panel-error-hint">
          You can still launch the run without an upfront estimate — costs will
          be tracked in the cost ledger as the run progresses.
        </p>
        <div className="budget-panel-actions">
          <button
            type="button"
            className="budget-panel-skip-btn"
            disabled={busy}
            onClick={onSkip}
          >
            Skip estimate and start anyway
          </button>
        </div>
      </div>
    );
  }

  const recipeKeys = Object.keys(estimate.recipes) as RecipeMode[];
  const activeRecipe = estimate.recipes[selectedRecipe];

  return (
    <div className="budget-panel">
      <header className="budget-panel-header">
        <div>
          <h3 className="budget-panel-title">Estimated budget</h3>
          <p className="budget-panel-sub mono">
            {estimate.paper.title ?? estimate.paper.id ?? "Unknown paper"}
          </p>
        </div>
        <div className="budget-panel-confidence">
          <span className="budget-panel-confidence-label">Confidence</span>
          <span className="budget-panel-confidence-value">
            ±{estimate.calibration_metadata.precision_window_pct}%
          </span>
          <span className="budget-panel-confidence-meta">
            from {estimate.calibration_metadata.based_on_n_preserved_runs} prior runs
          </span>
        </div>
      </header>

      <section className="budget-panel-gpu">
        <span className="budget-panel-gpu-label">GPU</span>
        <span className="budget-panel-gpu-sku">{estimate.gpu.label}</span>
        <span className="budget-panel-gpu-rate">
          {fmtUsd(estimate.gpu.usd_per_hour)}/hr
        </span>
        <span className="budget-panel-gpu-hours">
          × ~{fmtHours(estimate.gpu.estimated_hours.p50)} =
        </span>
        <span className="budget-panel-gpu-total">
          {fmtUsd(estimate.gpu.usd_total.p50)}
        </span>
      </section>

      {/* API token aggregate — sums across all candidate providers so the
          operator sees how much LLM throughput the estimate is built on. */}
      {(() => {
        const totalIn = estimate.api.reduce((s, r) => s + r.input_tokens, 0);
        const totalOut = estimate.api.reduce((s, r) => s + r.output_tokens, 0);
        const cheapest = estimate.api.reduce(
          (best, r) => (r.usd < best.usd ? r : best),
          estimate.api[0],
        );
        return (
          <section className="budget-panel-gpu" aria-label="API token summary">
            <span className="budget-panel-gpu-label">API</span>
            <span className="budget-panel-gpu-sku mono">
              ~{fmtTokens(totalIn)} in&nbsp;/&nbsp;{fmtTokens(totalOut)} out
            </span>
            <span className="budget-panel-gpu-rate">
              across {estimate.api.length} provider{estimate.api.length === 1 ? "" : "s"}
            </span>
            <span className="budget-panel-gpu-hours">
              cheapest: <span className="mono">{cheapest.provider}.{cheapest.model_id}</span> =
            </span>
            <span className="budget-panel-gpu-total">{fmtUsd(cheapest.usd)}</span>
          </section>
        );
      })()}

      <section className="budget-panel-recipes">
        {recipeKeys.map((key) => {
          const r = estimate.recipes[key];
          const isSelected = key === selectedRecipe;
          return (
            <button
              type="button"
              key={key}
              className={`budget-panel-recipe-card${isSelected ? " selected" : ""}`}
              onClick={() => onSelectRecipe(key)}
              aria-pressed={isSelected}
            >
              <div className="budget-panel-recipe-head">
                <span className="budget-panel-recipe-label">{r.label}</span>
                <span className="budget-panel-recipe-fidelity">
                  {r.fidelity_label}
                </span>
              </div>
              <div className="budget-panel-recipe-cost">
                {fmtUsd(r.gpu_usd + r.api_usd_best)}
                <span className="budget-panel-recipe-cost-sub">
                  &nbsp;to {fmtUsd(r.gpu_usd + r.api_usd_worst)}
                </span>
              </div>
              <div className="budget-panel-recipe-time">
                ~{fmtHours(r.wall_clock_hours_p50)}
              </div>
              <p className="budget-panel-recipe-desc">{r.description}</p>
              {r.declared_reductions.length > 0 && (
                <ul className="budget-panel-recipe-reductions">
                  {r.declared_reductions.map((red, i) => (
                    <li key={i}>{red}</li>
                  ))}
                </ul>
              )}
            </button>
          );
        })}
      </section>

      <section className="budget-panel-breakdown">
        <button
          type="button"
          className="budget-panel-breakdown-toggle"
          onClick={() => setBreakdownOpen((v) => !v)}
          aria-expanded={breakdownOpen}
        >
          {breakdownOpen ? "▾" : "▸"} API provider breakdown
        </button>
        {breakdownOpen && (
          <div className="budget-panel-breakdown-table" role="radiogroup" aria-label="Pick API provider">
            {estimate.api.map((row) => {
              const fullId = `${row.provider}.${row.model_id}`;
              const isSelected = fullId === selectedProvider;
              return (
                <label
                  key={fullId}
                  className={`budget-panel-breakdown-row${isSelected ? " selected" : ""}`}
                >
                  <input
                    type="radio"
                    name="budget-provider"
                    value={fullId}
                    checked={isSelected}
                    onChange={() => onSelectProvider(fullId)}
                  />
                  <span className="budget-panel-breakdown-provider mono">{fullId}</span>
                  <span className="budget-panel-breakdown-tokens mono">
                    {fmtTokens(row.input_tokens)}&nbsp;in&nbsp;/&nbsp;
                    {fmtTokens(row.output_tokens)}&nbsp;out
                  </span>
                  <span className="budget-panel-breakdown-usd">
                    {fmtUsd(row.usd)}
                    {row.is_subscription && (
                      <span className="budget-panel-breakdown-tag">subscription</span>
                    )}
                  </span>
                  {row.subscription_note && (
                    <span className="budget-panel-breakdown-note">
                      {row.subscription_note}
                    </span>
                  )}
                </label>
              );
            })}
          </div>
        )}
      </section>

      {/* PR-ε.8: "How this estimate was made" collapsible breakdown */}
      {estimate.estimate_breakdown && estimate.estimate_breakdown.length > 0 && (() => {
        const breakdown = estimate.estimate_breakdown!;
        // Low-confidence badge: sigma/mean > 50% on available sources
        const availSources = breakdown.filter(s => isFinite(s.sigma) && s.mean > 0);
        const gpuSigma = estimate.gpu.estimated_hours_sigma ?? null;
        const gpuMean = estimate.gpu.estimated_hours.p50;
        const lowConf = estimate.gpu.low_confidence ?? (
          gpuSigma !== null && gpuMean > 0 && gpuSigma / gpuMean > 0.5
        );
        return (
          <section className="budget-panel-breakdown">
            <button
              type="button"
              className="budget-panel-breakdown-toggle"
              onClick={() => setHowOpen(v => !v)}
              aria-expanded={howOpen}
            >
              {howOpen ? "▾" : "▸"} How this estimate was made
              {lowConf && (
                <span className="budget-panel-confidence-badge budget-panel-confidence-badge--low" title="High uncertainty — σ/μ > 50%">
                  low confidence
                </span>
              )}
            </button>
            {howOpen && (
              <div className="budget-panel-how-table">
                {breakdown.map(src => {
                  const unavailable = !isFinite(src.sigma);
                  return (
                    <div
                      key={src.source}
                      className={`budget-panel-how-row${unavailable ? " budget-panel-how-row--unavailable" : ""}`}
                    >
                      <span className="budget-panel-how-source">
                        <SourceLabel source={src.source} />
                        {src.n_samples > 0 && (
                          <span className="budget-panel-how-samples"> ({src.n_samples} runs)</span>
                        )}
                      </span>
                      {unavailable ? (
                        <span className="budget-panel-how-unavail">no similar past runs yet</span>
                      ) : (
                        <>
                          <span className="budget-panel-how-mean mono">{fmtHours(src.mean)}</span>
                          <span className="budget-panel-how-sigma mono">{fmtSigma(src.sigma, src.mean)}</span>
                          <span className="budget-panel-how-weight">weight {(src.weight * 100).toFixed(0)}%</span>
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        );
      })()}

      <div className="budget-panel-actions">
        <button
          type="button"
          className="budget-panel-confirm-btn"
          disabled={busy}
          onClick={onConfirm}
        >
          Begin run · {fmtUsd(activeRecipe.gpu_usd + activeRecipe.api_usd_best)}
          &nbsp;→&nbsp;{fmtUsd(activeRecipe.gpu_usd + activeRecipe.api_usd_worst)}
        </button>
      </div>
    </div>
  );
}
