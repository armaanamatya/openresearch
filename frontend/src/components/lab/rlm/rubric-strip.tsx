"use client";

import type { RlmRunState } from "../../../hooks/use-rlm-run";
import { useCountUp } from "./use-count-up";
import { Sparkline } from "./sparkline";
import styles from "./rubric-strip.module.css";

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

      {/* Line-chart sparkline (one polyline across the series). */}
      {series.length > 0 && (
        <div className={styles.sparkBlock} aria-hidden="true">
          <Sparkline series={series} width={120} height={28} />
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
