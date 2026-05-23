"use client";

import styles from "./sparkline.module.css";

interface SparklineProps {
  series: Array<{ iteration: number; score: number }>;
  width?: number;
  height?: number;
}

/**
 * Sparkline — SVG line chart for the rubric climb.
 *
 * Y axis is [0, 1] (rubric scores); X axis is series index (0..N-1).
 * Empty series renders nothing. One point renders a small circle.
 *
 * Spec: docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.1.
 */
export function Sparkline({ series, width = 120, height = 28 }: SparklineProps) {
  if (series.length === 0) return null;

  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  const yFor = (score: number) => height - clamp(score) * height;

  if (series.length === 1) {
    const cy = yFor(series[0].score);
    return (
      <svg
        className={styles.svg}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        aria-hidden="true"
      >
        <circle className={styles.dot} cx={width / 2} cy={cy} r={2} />
      </svg>
    );
  }

  const xStep = width / (series.length - 1);
  const points = series
    .map((p, i) => `${(i * xStep).toFixed(2)},${yFor(p.score).toFixed(2)}`)
    .join(" ");

  return (
    <svg
      className={styles.svg}
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      aria-hidden="true"
    >
      <polyline className={styles.line} points={points} />
    </svg>
  );
}
