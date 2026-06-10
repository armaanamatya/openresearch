"use client";

/**
 * TwoAxisVerdictCard — U17, two-axis reproducibility verdict UI.
 *
 * Renders when final_report.json carries schema_version >= 2 and
 * implementation_verdict is present. Legacy runs (schema_version < 2 or
 * implementation_verdict absent) must NOT render this card — the caller is
 * responsible for the guard, but the component also guards internally.
 *
 * Axes:
 *   FIDELITY (implementation_verdict): faithful | partial | broken
 *   REPLICATION (replication_verdict): replicated | partially-replicated |
 *                                       contradicted | inconclusive
 *
 * Key design decision (Part A decision 6):
 *   "contradicted on a faithful run" = legitimate negative finding,
 *   NOT an error. It must read that way explicitly.
 *
 * Developer details (rationale[] + per_claim reasons) are collapsed by
 * default behind a <details> disclosure so the user-facing surface stays
 * clean (operator requirement: developer internals private/collapsed).
 *
 * Spec: docs/runbooks/2026-06-08-agent-codegen-tdd-hardening-handoff.md
 *        Part A decision 6 + A.1 U17.
 */

import styles from "./two-axis-verdict-card.module.css";

// ─── Types ────────────────────────────────────────────────────────────────────

export type FidelityVerdict = "faithful" | "partial" | "broken";

export type ReplicationVerdict =
  | "replicated"
  | "partially-replicated"
  | "contradicted"
  | "inconclusive";

export interface PerClaimResult {
  claim_id: string;
  status: string;
  credit: number;
  reason: string;
  /** Whether this claim was in scope / eligible for evaluation. */
  eligible: boolean;
  /** Measured mean effect, if available. */
  measured_mean?: number | null;
  /** CI lower bound. */
  ci_low?: number | null;
  /** CI upper bound. */
  ci_high?: number | null;
  /** Claimed effect size from the paper, if extracted. */
  claimed_value?: number | null;
}

export interface Reproducibility {
  replication_credit: number;
  fidelity_score: number;
  /** Legacy single-axis verdict — present for back-compat. */
  legacy_verdict?: string | null;
  per_claim?: PerClaimResult[];
  rationale?: string[];
}

/** The two-axis report fields from final_report.json (schema_version >= 2). */
export interface TwoAxisReport {
  schema_version: number;
  implementation_verdict: FidelityVerdict;
  replication_verdict: ReplicationVerdict;
  reproducibility: Reproducibility;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Returns true when the report carries two-axis fields (schema_version >= 2). */
export function isTwoAxisReport(
  report: Partial<TwoAxisReport> | null | undefined,
): report is TwoAxisReport {
  if (!report) return false;
  // When schema_version is explicitly present, it must be >= 2.
  // This gates off schema_version=1 legacy reports even if they happen to
  // carry implementation_verdict.
  if (typeof report.schema_version === "number") {
    return report.schema_version >= 2;
  }
  // Graceful fallback for early beta payloads that have implementation_verdict
  // but no schema_version tag yet.
  return typeof report.implementation_verdict === "string";
}

/**
 * Plain-language one-liner that explains the verdict combination to a user.
 * Key rule: faithful + contradicted = POSITIVE finding ("interesting negative
 * result"), not a failure.
 */
export function verdictOneLiner(
  fidelity: FidelityVerdict,
  replication: ReplicationVerdict,
): string {
  if (fidelity === "faithful") {
    switch (replication) {
      case "replicated":
        return "Faithfully reproduced — the paper's headline result replicated.";
      case "partially-replicated":
        return "Faithfully reproduced — the paper's result partially replicated (right direction, magnitude off).";
      case "contradicted":
        return "Faithfully reproduced — the paper's headline result did NOT replicate (contradicted). This is a legitimate negative finding.";
      case "inconclusive":
        return "Faithfully reproduced — replication is inconclusive (insufficient seeds or unmet eligibility requirements).";
    }
  }
  if (fidelity === "partial") {
    switch (replication) {
      case "replicated":
        return "Partially reproduced — the paper's result replicated, but implementation fidelity is incomplete.";
      case "partially-replicated":
        return "Partially reproduced — the paper's result partially replicated.";
      case "contradicted":
        return "Partially reproduced — contradicted result; fidelity issues may affect the conclusion.";
      case "inconclusive":
        return "Partially reproduced — replication is inconclusive (partial implementation limits the verdict).";
    }
  }
  // broken
  switch (replication) {
    case "replicated":
      return "Implementation broken — reported replication is unreliable due to code failures.";
    case "partially-replicated":
      return "Implementation broken — reported partial replication is unreliable.";
    case "contradicted":
      return "Implementation broken — contradicted result is inconclusive given code failures.";
    case "inconclusive":
      return "Implementation broken — no reliable verdict possible.";
  }
}

/**
 * Count eligible, non-null per_claim entries that have both measured_mean and
 * claimed_value — these are the claims we can plot in the graph.
 */
export function plottableClaims(
  perClaim: PerClaimResult[] | undefined,
): PerClaimResult[] {
  if (!perClaim) return [];
  return perClaim.filter(
    (c) =>
      c.eligible &&
      c.measured_mean != null &&
      c.claimed_value != null,
  );
}

/**
 * Derive seed count from per_claim: count unique claim_id entries.
 * The spec says "infer from per_claim if present."
 */
export function inferSeedCount(perClaim: PerClaimResult[] | undefined): number | null {
  if (!perClaim || perClaim.length === 0) return null;
  return perClaim.filter((c) => c.eligible).length;
}

// ─── Badge sub-components ──────────────────────────────────────────────────────

function FidelityBadge({ verdict }: { verdict: FidelityVerdict }) {
  const cls =
    verdict === "faithful"
      ? styles.badgeGreen
      : verdict === "partial"
      ? styles.badgeAmber
      : styles.badgeRed;

  const label =
    verdict === "faithful" ? "faithful" : verdict === "partial" ? "partial" : "broken";

  return (
    <span
      className={`${styles.badge} ${cls}`}
      data-testid="fidelity-badge"
      data-verdict={verdict}
    >
      {label}
    </span>
  );
}

function ReplicationBadge({ verdict }: { verdict: ReplicationVerdict }) {
  const cls =
    verdict === "replicated"
      ? styles.badgeGreen
      : verdict === "partially-replicated"
      ? styles.badgeAmber
      : verdict === "contradicted"
      ? styles.badgeContradicted
      : styles.badgeMuted;

  const labelMap: Record<ReplicationVerdict, string> = {
    replicated: "replicated",
    "partially-replicated": "partially replicated",
    contradicted: "contradicted",
    inconclusive: "inconclusive",
  };

  return (
    <span
      className={`${styles.badge} ${cls}`}
      data-testid="replication-badge"
      data-verdict={verdict}
    >
      {labelMap[verdict]}
    </span>
  );
}

// ─── Measured-vs-claimed graph ────────────────────────────────────────────────

const GRAPH_W = 260;
const BAR_H = 12;
const LABEL_W = 80;
const TRACK_W = GRAPH_W - LABEL_W - 8;
const PAD_T = 8;
const ROW_STEP = BAR_H + 10;

/**
 * MeasuredVsClaimedGraph — simple SVG dot+CI-interval plot.
 *
 * For each eligible claim with both measured_mean and claimed_value:
 *   - A grey track (background bar)
 *   - A claimed value marker (◇)
 *   - A measured dot with CI bars
 *   - A label (truncated claim_id)
 *
 * All values are normalised to [min, max] across the dataset so different
 * scales can coexist. Falls back to a "no plottable claims" note when empty.
 */
function MeasuredVsClaimedGraph({ claims }: { claims: PerClaimResult[] }) {
  if (claims.length === 0) {
    return (
      <p className={styles.noPlot} data-testid="no-plot-notice">
        no plottable claims
      </p>
    );
  }

  // Compute domain: union of measured_mean, ci_low, ci_high, claimed_value.
  const allValues = claims.flatMap((c) =>
    [c.measured_mean, c.ci_low, c.ci_high, c.claimed_value].filter(
      (v): v is number => v != null,
    ),
  );
  const domainMin = Math.min(...allValues);
  const domainMax = Math.max(...allValues);
  const domainRange = domainMax - domainMin || 1;

  function toX(value: number): number {
    return LABEL_W + ((value - domainMin) / domainRange) * TRACK_W;
  }

  const svgH = PAD_T + claims.length * ROW_STEP + 16;

  return (
    <svg
      width={GRAPH_W}
      height={svgH}
      viewBox={`0 0 ${GRAPH_W} ${svgH}`}
      style={{ display: "block", overflow: "visible" }}
      role="img"
      aria-label="Measured vs claimed values per eligible claim"
      data-testid="measured-vs-claimed-graph"
    >
      {/* X-axis label row */}
      <text
        x={LABEL_W}
        y={PAD_T - 2}
        style={{
          fontSize: 7,
          fontFamily: "var(--font-mono)",
          fill: "var(--muted)",
        }}
      >
        min
      </text>
      <text
        x={LABEL_W + TRACK_W}
        y={PAD_T - 2}
        textAnchor="end"
        style={{
          fontSize: 7,
          fontFamily: "var(--font-mono)",
          fill: "var(--muted)",
        }}
      >
        max
      </text>
      <text
        x={LABEL_W + TRACK_W / 2}
        y={PAD_T - 2}
        textAnchor="middle"
        style={{
          fontSize: 7,
          fontFamily: "var(--font-mono)",
          fill: "var(--muted)",
        }}
      >
        value
      </text>

      {claims.map((claim, i) => {
        const rowY = PAD_T + i * ROW_STEP;
        const midY = rowY + BAR_H / 2;
        const measuredX = toX(claim.measured_mean!);
        const claimedX = toX(claim.claimed_value!);
        const ciLowX = claim.ci_low != null ? toX(claim.ci_low) : measuredX;
        const ciHighX = claim.ci_high != null ? toX(claim.ci_high) : measuredX;

        const labelText =
          claim.claim_id.length > 12
            ? claim.claim_id.slice(0, 11) + "…"
            : claim.claim_id;

        return (
          <g key={claim.claim_id} data-testid={`claim-row-${claim.claim_id}`}>
            {/* Label — SVG <text> does not accept title attr; truncate for display */}
            <text
              x={LABEL_W - 4}
              y={midY + 3.5}
              textAnchor="end"
              style={{
                fontSize: 8,
                fontFamily: "var(--font-mono)",
                fill: "var(--ink-2)",
              }}
            >
              <title>{claim.claim_id}</title>
              {labelText}
            </text>

            {/* Track background */}
            <rect
              x={LABEL_W}
              y={rowY}
              width={TRACK_W}
              height={BAR_H}
              rx={2}
              fill="var(--canvas-bg)"
              stroke="var(--line)"
              strokeWidth={0.5}
            />

            {/* CI interval bar */}
            {claim.ci_low != null && claim.ci_high != null && (
              <rect
                x={ciLowX}
                y={midY - 2}
                width={Math.max(1, ciHighX - ciLowX)}
                height={4}
                fill="color-mix(in srgb, var(--accent) 35%, transparent)"
                data-testid={`ci-bar-${claim.claim_id}`}
              />
            )}

            {/* Claimed value diamond marker */}
            <polygon
              points={`${claimedX},${rowY + 1} ${claimedX + 4},${midY} ${claimedX},${rowY + BAR_H - 1} ${claimedX - 4},${midY}`}
              fill="var(--muted-2, #6b7280)"
              opacity={0.8}
              data-testid={`claimed-marker-${claim.claim_id}`}
            />

            {/* Measured value dot */}
            <circle
              cx={measuredX}
              cy={midY}
              r={3}
              fill="var(--accent)"
              data-testid={`measured-dot-${claim.claim_id}`}
            />
          </g>
        );
      })}

      {/* Legend */}
      <g transform={`translate(${LABEL_W}, ${PAD_T + claims.length * ROW_STEP + 4})`}>
        <circle cx={4} cy={4} r={3} fill="var(--accent)" />
        <text
          x={10}
          y={7.5}
          style={{
            fontSize: 7,
            fontFamily: "var(--font-mono)",
            fill: "var(--accent)",
          }}
        >
          measured
        </text>
        <polygon
          points="46,0 50,4 46,8 42,4"
          fill="var(--muted-2, #6b7280)"
          opacity={0.8}
        />
        <text
          x={54}
          y={7.5}
          style={{
            fontSize: 7,
            fontFamily: "var(--font-mono)",
            fill: "var(--muted)",
          }}
        >
          claimed
        </text>
      </g>
    </svg>
  );
}

// ─── Developer Details disclosure ─────────────────────────────────────────────

function DeveloperDetails({
  rationale,
  perClaim,
}: {
  rationale: string[] | undefined;
  perClaim: PerClaimResult[] | undefined;
}) {
  const hasContent =
    (rationale && rationale.length > 0) ||
    (perClaim && perClaim.length > 0);

  if (!hasContent) return null;

  return (
    <details className={styles.devDetails} data-testid="developer-details">
      <summary className={styles.devSummary}>Developer details</summary>
      <div className={styles.devBody}>
        {rationale && rationale.length > 0 && (
          <div className={styles.devSection}>
            <span className={styles.devLabel}>rationale</span>
            <ul className={styles.rationaleList}>
              {rationale.map((line, i) => (
                <li key={i} className={styles.rationaleLine}>
                  {line}
                </li>
              ))}
            </ul>
          </div>
        )}

        {perClaim && perClaim.length > 0 && (
          <div className={styles.devSection}>
            <span className={styles.devLabel}>per-claim reasons</span>
            <ul className={styles.claimList}>
              {perClaim.map((c) => (
                <li key={c.claim_id} className={styles.claimItem}>
                  <span className={styles.claimId}>{c.claim_id}</span>
                  <span className={styles.claimStatus}>{c.status}</span>
                  <span className={styles.claimCredit}>
                    credit {c.credit.toFixed(2)}
                  </span>
                  {c.reason && (
                    <span className={styles.claimReason}>{c.reason}</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </details>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────

export interface TwoAxisVerdictCardProps {
  report: TwoAxisReport;
}

/**
 * TwoAxisVerdictCard — renders when final_report.json carries schema_version >= 2.
 *
 * Layout:
 *
 *   ┌──────────────────────────────────────────────────┐
 *   │  FIDELITY  [faithful 🟢]  REPLICATION  [contradicted]  │
 *   │  0.75 replication credit · 2 seeds               │
 *   │  "Faithfully reproduced — but the paper's        │
 *   │   headline result did NOT replicate (contradicted)│
 *   │   This is a legitimate negative finding."        │
 *   ├──────────────────────────────────────────────────┤
 *   │  measured-vs-claimed graph (per eligible claim)  │
 *   ├──────────────────────────────────────────────────┤
 *   │  ▸ Developer details  (collapsed)                │
 *   └──────────────────────────────────────────────────┘
 */
export function TwoAxisVerdictCard({ report }: TwoAxisVerdictCardProps) {
  // Internal guard: never render for legacy reports.
  if (!isTwoAxisReport(report)) return null;

  const { implementation_verdict, replication_verdict, reproducibility } = report;
  const { replication_credit, per_claim, rationale } = reproducibility;

  const oneLiner = verdictOneLiner(implementation_verdict, replication_verdict);
  const seedCount = inferSeedCount(per_claim);
  const plotClaims = plottableClaims(per_claim);

  return (
    <section
      className={styles.card}
      data-testid="two-axis-verdict-card"
      aria-label="Two-axis reproducibility verdict"
    >
      {/* Axis badges row */}
      <div className={styles.axisRow}>
        <div className={styles.axisGroup}>
          <span className={styles.axisLabel}>fidelity</span>
          <FidelityBadge verdict={implementation_verdict} />
        </div>
        <span className={styles.axisSep} aria-hidden="true">⟂</span>
        <div className={styles.axisGroup}>
          <span className={styles.axisLabel}>replication</span>
          <ReplicationBadge verdict={replication_verdict} />
        </div>
      </div>

      {/* Credit + seed count row */}
      <div className={styles.metaRow}>
        <span
          className={styles.creditValue}
          data-testid="replication-credit"
          title="Replication credit: 1.0 = fully replicated, 0.75 = partially replicated, low = contradicted"
        >
          {replication_credit.toFixed(2)} credit
        </span>
        {seedCount !== null && (
          <span className={styles.seedCount} data-testid="seed-count">
            {seedCount} {seedCount === 1 ? "claim" : "claims"} evaluated
          </span>
        )}
      </div>

      {/* One-liner plain-language description */}
      <p
        className={
          replication_verdict === "contradicted" && implementation_verdict === "faithful"
            ? styles.oneLineNegativeFinding
            : styles.oneLine
        }
        data-testid="verdict-one-liner"
      >
        {oneLiner}
      </p>

      {/* Measured-vs-claimed graph */}
      {per_claim && per_claim.length > 0 && (
        <div className={styles.graphSection}>
          <span className={styles.graphLabel}>measured vs claimed</span>
          <MeasuredVsClaimedGraph claims={plotClaims} />
        </div>
      )}

      {/* Developer details — collapsed */}
      <DeveloperDetails rationale={rationale} perClaim={per_claim} />
    </section>
  );
}
