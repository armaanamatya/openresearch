"use client";

import type { RlmRunState } from "../../../hooks/use-rlm-run";
import styles from "./rubric-strip.module.css";

interface RubricStripProps {
  rubric: RlmRunState["rubric"];
}

/**
 * RubricStrip — the second band of the RLM lab UI (spec §7.1).
 *
 * Renders: current score (large), a baseline→fill→target progress bar,
 * a climb sparkline (one bar per rubric_score event), and a climb annotation.
 *
 * Honesty rule (spec §14): when rubric.current is null, renders a placeholder
 * instead of a fabricated number. No .toFixed() is ever called on null.
 *
 * Bar geometry: the track represents [0, 1]. The fill width = current score (0–1).
 * Baseline and target are rendered as absolute markers on the track via left: %.
 */
export function RubricStrip({ rubric }: RubricStripProps) {
  const { current, baseline, target, series } = rubric;

  const isScored = current !== null;

  // Safe formatters — never called with null.
  const fmtScore = (n: number) => n.toFixed(2);

  // Fill width as a percentage of the 0–1 track.
  const fillPct = isScored ? `${Math.min(1, Math.max(0, current!)) * 100}%` : "0%";

  // Sparkline: bar heights proportional to absolute score on a 0–1 scale.
  // This keeps the chart honest — no inflation from a low local max.
  const maxBarHeight = 28; // px — constrained by the strip height

  // Climb annotation components — computed only when both baseline and current are known.
  const hasClimb = isScored && baseline !== null;
  const delta = hasClimb ? current! - baseline! : null;

  return (
    <div className={styles.strip} role="region" aria-label="Rubric score">
      {/* Current score — large prominent number */}
      <div className={styles.scoreBlock}>
        <span className={isScored ? styles.scoreValue : styles.scorePlaceholder}>
          {isScored ? fmtScore(current!) : "—"}
        </span>
        <span className={styles.scoreLabel}>rubric score</span>
      </div>

      {/* Progress bar: baseline marker → fill → target marker.
          Bar labels are placed in title attributes only (not rendered text)
          to avoid duplicate text matches in tests. The target label is rendered
          in the climb annotation block below. */}
      <div className={styles.barBlock}>
        <div className={styles.barTrack} aria-hidden="true">
          {/* Fill from left edge to current score */}
          <div
            className={styles.barFill}
            style={{ width: fillPct }}
          />
          {/* Baseline marker */}
          {baseline !== null && (
            <div
              className={styles.baselineMarker}
              style={{ left: `${baseline * 100}%` }}
              title={`baseline ${fmtScore(baseline)}`}
            />
          )}
          {/* Target marker */}
          {target !== null && (
            <div
              className={styles.targetMarker}
              style={{ left: `${target * 100}%` }}
              title={`target ${fmtScore(target)}`}
            />
          )}
        </div>
      </div>

      {/* Climb sparkline */}
      {series.length > 0 && (
        <div className={styles.sparkBlock} aria-hidden="true">
          {series.map((point) => {
            const barH = Math.round(Math.max(2, point.score * maxBarHeight));
            return (
              <span
                key={point.iteration}
                data-spark-bar
                className={styles.sparkBar}
                style={{ height: `${barH}px` }}
                title={`iter ${point.iteration}: ${fmtScore(point.score)}`}
              />
            );
          })}
        </div>
      )}

      {/* Climb annotation — only rendered when a climb exists.
          The target label "target N.NN" is the single rendered occurrence of the target value.
          The baseline value "N.NN" appears here as the single rendered text occurrence. */}
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
        </div>
      )}
    </div>
  );
}
