"use client";

import type { RlmRunState } from "../../../hooks/use-rlm-run";
import { useCountUp } from "./use-count-up";
import styles from "./rubric-strip.module.css";

const CHART_W = 280;
const CHART_H = 96;
const PAD_L = 28; // left padding for y-axis labels
const PAD_R = 48; // right padding for line labels
const PAD_T = 8;
const PAD_B = 16; // bottom padding for x-axis

function TrajectoryChart({
  series,
  baseline,
  target,
}: {
  series: Array<{ iteration: number; score: number }>;
  baseline: number | null;
  target: number | null;
}) {
  const innerW = CHART_W - PAD_L - PAD_R;
  const innerH = CHART_H - PAD_T - PAD_B;

  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  const xFor = (i: number) =>
    series.length > 1 ? PAD_L + (i / (series.length - 1)) * innerW : PAD_L + innerW / 2;
  const yFor = (score: number) => PAD_T + (1 - clamp(score)) * innerH;

  const points = series
    .map((p, i) => `${xFor(i).toFixed(1)},${yFor(p.score).toFixed(1)}`)
    .join(" ");

  const last = series[series.length - 1];
  const lastX = xFor(series.length - 1);
  const lastY = yFor(last.score);

  return (
    <svg
      width={CHART_W}
      height={CHART_H}
      viewBox={`0 0 ${CHART_W} ${CHART_H}`}
      style={{ display: "block", overflow: "visible" }}
    >
      {/* Y-axis labels */}
      {[0, 0.5, 1].map((v) => (
        <text
          key={v}
          x={PAD_L - 4}
          y={yFor(v) + 3.5}
          textAnchor="end"
          style={{
            fontSize: 8,
            fontFamily: "var(--font-mono)",
            fill: "var(--muted)",
          }}
        >
          {v.toFixed(1)}
        </text>
      ))}

      {/* Hairline grid at 0.5 */}
      <line
        x1={PAD_L}
        y1={yFor(0.5)}
        x2={PAD_L + innerW}
        y2={yFor(0.5)}
        stroke="var(--line)"
        strokeWidth={0.5}
        strokeDasharray="3 3"
      />

      {/* Baseline reference — dashed gray */}
      {baseline !== null && (
        <>
          <line
            x1={PAD_L}
            y1={yFor(baseline)}
            x2={PAD_L + innerW}
            y2={yFor(baseline)}
            stroke="var(--muted-2)"
            strokeWidth={1}
            strokeDasharray="4 3"
          />
          <text
            x={PAD_L + innerW + 4}
            y={yFor(baseline) + 3.5}
            style={{ fontSize: 8, fontFamily: "var(--font-mono)", fill: "var(--muted)" }}
          >
            baseline
          </text>
        </>
      )}

      {/* Target (paper) reference — dashed white */}
      {target !== null && (
        <>
          <line
            x1={PAD_L}
            y1={yFor(target)}
            x2={PAD_L + innerW}
            y2={yFor(target)}
            stroke="var(--ink-2)"
            strokeWidth={1}
            strokeDasharray="4 3"
            opacity={0.5}
          />
          <text
            x={PAD_L + innerW + 4}
            y={yFor(target) + 3.5}
            style={{ fontSize: 8, fontFamily: "var(--font-mono)", fill: "var(--ink-2)", opacity: 0.6 }}
          >
            target
          </text>
        </>
      )}

      {/* RLM score polyline — orange */}
      {series.length > 1 && (
        <polyline
          points={points}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={1.5}
          strokeLinejoin="round"
        />
      )}

      {/* End dot + label */}
      <circle cx={lastX} cy={lastY} r={2.5} fill="var(--accent)" />
      <text
        x={PAD_L + innerW + 4}
        y={lastY + 3.5}
        style={{ fontSize: 8, fontFamily: "var(--font-mono)", fill: "var(--accent)", fontWeight: 600 }}
      >
        {last.score.toFixed(3)}
      </text>

      {/* X-axis: step labels */}
      <text
        x={PAD_L}
        y={CHART_H - 2}
        style={{ fontSize: 8, fontFamily: "var(--font-mono)", fill: "var(--muted)" }}
      >
        step 0
      </text>
      {series.length > 1 && (
        <text
          x={PAD_L + innerW}
          y={CHART_H - 2}
          textAnchor="end"
          style={{ fontSize: 8, fontFamily: "var(--font-mono)", fill: "var(--muted)" }}
        >
          {series.length - 1}
        </text>
      )}

      {/* Legend */}
      <g transform={`translate(${PAD_L}, ${PAD_T - 2})`}>
        <line x1={0} y1={0} x2={10} y2={0} stroke="var(--muted-2)" strokeWidth={1} strokeDasharray="3 2" />
        <text x={13} y={3.5} style={{ fontSize: 8, fontFamily: "var(--font-mono)", fill: "var(--muted)" }}>baseline</text>
        <line x1={52} y1={0} x2={62} y2={0} stroke="var(--accent)" strokeWidth={1.5} />
        <text x={65} y={3.5} style={{ fontSize: 8, fontFamily: "var(--font-mono)", fill: "var(--accent)" }}>RLM</text>
        {target !== null && (
          <>
            <line x1={87} y1={0} x2={97} y2={0} stroke="var(--ink-2)" strokeWidth={1} strokeDasharray="3 2" opacity={0.5} />
            <text x={100} y={3.5} style={{ fontSize: 8, fontFamily: "var(--font-mono)", fill: "var(--ink-2)", opacity: 0.6 }}>paper</text>
          </>
        )}
      </g>
    </svg>
  );
}

interface RubricStripProps {
  rubric: RlmRunState["rubric"];
}

const STATUS_RANK: Record<"pass" | "partial" | "fail", number> = {
  fail: 0,
  partial: 1,
  pass: 2,
};

function justFlipped(
  area: { area: string; status: "pass" | "partial" | "fail" },
  previous: Array<{ area: string; status: "pass" | "partial" | "fail" }>,
): boolean {
  const prior = previous.find((p) => p.area === area.area);
  if (!prior) return false;
  return STATUS_RANK[area.status] > STATUS_RANK[prior.status];
}

function statusGlyph(status: "pass" | "partial" | "fail"): string {
  return status === "pass" ? "✓" : status === "partial" ? "◐" : "✗";
}

/**
 * RubricStrip — band 2 of the RLM lab UI.
 *
 * Renders: animated current score (count-up tween), baseline→fill→target
 * progress bar, line-chart sparkline, per-area status chip row with
 * fail→pass flip highlights, and a climb annotation that names the
 * attributable candidate on jumps of ≥ +0.05.
 *
 * Honesty rule: when rubric.current is null, the placeholder "—" is rendered
 * instead of a fabricated number.
 *
 * Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.1, §4.3.
 */
export function RubricStrip({ rubric }: RubricStripProps) {
  const { current, baseline, target, series, areas, previousAreas, attributableCandidate } = rubric;

  const isScored = current !== null;
  const tweenedCurrent = useCountUp(isScored ? current! : 0, 400);

  const fmtScore = (n: number) => n.toFixed(2);
  const fillPct = isScored ? `${Math.min(1, Math.max(0, tweenedCurrent)) * 100}%` : "0%";

  const hasClimb = isScored && baseline !== null;
  const delta = hasClimb ? current! - baseline! : null;

  // Attribution tail: only when the last step's delta meets the threshold AND
  // a candidate is live. Spec §4.1.
  const ATTRIBUTION_THRESHOLD = 0.05;
  let attributionLabel: string | null = null;
  if (series.length >= 2 && attributableCandidate) {
    const stepDelta = series[series.length - 1].score - series[series.length - 2].score;
    if (stepDelta >= ATTRIBUTION_THRESHOLD) {
      attributionLabel = `from candidate ${attributableCandidate.title}`;
    }
  }

  return (
    <div
      className={styles.strip}
      role="region"
      aria-label="Rubric score"
      title="Rubric score updates when verify_against_rubric or report generation emits scored areas."
    >
      {/* Current score — large prominent number, count-up animated. */}
      <div className={styles.scoreBlock}>
        <span className={isScored ? styles.scoreValue : styles.scorePlaceholder}>
          {isScored ? fmtScore(tweenedCurrent) : "—"}
        </span>
        <span className={styles.scoreLabel}>rubric score</span>
      </div>

      {!isScored && (
        <div className={styles.emptyHint}>
          Rubric scores appear after baseline implementation, experiment execution, and verification.
        </div>
      )}

      {/* Progress bar: baseline marker → fill → target marker. */}
      <div className={styles.barBlock}>
        <div className={styles.barTrack} aria-hidden="true">
          <div className={styles.barFill} style={{ width: fillPct }} />
          {baseline !== null && (
            <div
              className={styles.baselineMarker}
              style={{ left: `${baseline * 100}%` }}
              title={`baseline ${fmtScore(baseline)}`}
            />
          )}
          {target !== null && (
            <div
              className={styles.targetMarker}
              style={{ left: `${target * 100}%` }}
              title={`target ${fmtScore(target)}`}
            />
          )}
        </div>
      </div>

      {/* Trajectory chart — multi-line: RLM (orange), baseline (gray dashed), target (white dashed). */}
      {series.length > 0 && (
        <div className={styles.sparkBlock} aria-hidden="true">
          <TrajectoryChart series={series} baseline={baseline} target={target} />
        </div>
      )}

      {/* Per-area status chip row with fail→pass flip highlights. */}
      {areas.length > 0 && (
        <ul className={styles.areaChipRow} aria-live="polite" aria-label="Rubric areas">
          {areas.map((a, i) => {
            const flipped = justFlipped(a, previousAreas);
            const statusClass =
              a.status === "pass"
                ? styles.chipPass
                : a.status === "partial"
                ? styles.chipPartial
                : styles.chipFail;
            const cls = [styles.areaChip, statusClass, flipped ? styles.justFlipped : ""]
              .filter(Boolean)
              .join(" ");
            return (
              <li
                key={a.area || `__area_${i}`}
                className={cls}
                data-area-chip
                data-area={a.area}
                data-just-flipped={flipped ? "true" : "false"}
                title={`${a.area || "—"}: ${a.status} (${a.score.toFixed(2)})`}
              >
                <span className={styles.chipGlyph} aria-hidden="true">
                  {statusGlyph(a.status)}
                </span>
                {a.area && <span className={styles.chipName}>{a.area}</span>}
              </li>
            );
          })}
        </ul>
      )}

      {/* Climb annotation — only when a climb exists; the candidate attribution
          tail rides on the same row when the last step met the threshold. */}
      {hasClimb && (
        <div className={styles.climbBlock}>
          <span className={styles.climbAnnotation}>
            {`baseline ${fmtScore(baseline!)} → ${fmtScore(current!)}`}
          </span>
          {delta !== null && target !== null && (
            <span className={styles.deltaAnnotation}>
              {`${delta >= 0 ? "+" : ""}${fmtScore(delta)} vs target ${fmtScore(target)}`}
            </span>
          )}
          {attributionLabel && (
            <span className={styles.attributionAnnotation}>{attributionLabel}</span>
          )}
        </div>
      )}
    </div>
  );
}
